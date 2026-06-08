package convert

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// ggmlOracle locates a built ggml (library + headers) to use as the
// bit-identical reference for the Go quantizers. It prefers explicit env vars
// so it can point at the llama.cpp fork that actually builds llama-quantize:
//
//	OLLAMA_GGML_LIB_DIR  dir containing libggml-base.so
//	OLLAMA_GGML_SRC_DIR  dir containing ggml-quants.h / ggml-common.h
//	OLLAMA_GGML_INC_DIR  dir containing ggml.h
//
// It falls back to the in-tree vendored copy, and reports ok=false (skip) when
// no ggml build is available.
func ggmlOracle() (libDir, srcDir, incDir string, ok bool) {
	libDir = cmpOr(os.Getenv("OLLAMA_GGML_LIB_DIR"), filepath.Join("..", "build", "lib", "ollama"))
	srcDir = cmpOr(os.Getenv("OLLAMA_GGML_SRC_DIR"), filepath.Join("..", "ml", "backend", "ggml", "ggml", "src"))
	incDir = cmpOr(os.Getenv("OLLAMA_GGML_INC_DIR"), filepath.Join("..", "ml", "backend", "ggml", "ggml", "include"))
	if _, err := os.Stat(filepath.Join(libDir, "libggml-base.so")); err != nil {
		return "", "", "", false
	}
	return libDir, srcDir, incDir, true
}

func cmpOr(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// TestQuantizeMatchesGGML proves the Go quantizers are byte-for-byte identical
// to ggml's reference kernels (quantize_row_*_ref) for a fixed input. This is
// what makes the fused single-pass quantizer a safe drop-in for shelling out to
// llama-quantize. It builds a tiny C harness against the ggml oracle; if none
// is configured it skips (CI has no vendored ggml after the CGO removal).
func TestQuantizeMatchesGGML(t *testing.T) {
	libDir, srcDir, incDir, ok := ggmlOracle()
	if !ok {
		t.Skip("ggml not available; set OLLAMA_GGML_LIB_DIR/OLLAMA_GGML_SRC_DIR/OLLAMA_GGML_INC_DIR (e.g. to a llama.cpp fork build) to run")
	}

	const N = 1 << 20 // multiple of 256 (QK_K) and 32 (QK8_0)
	src := makeF32Data(N)

	cases := []struct {
		name  string
		ref   string
		block string
		qk    int
		goOut []byte
	}{
		{"Q8_0", "quantize_row_q8_0_ref", "block_q8_0", qk80, quantizeQ8_0(src)},
		{"Q4_K", "quantize_row_q4_K_ref", "block_q4_K", qkK, quantizeQ4_K(src)},
		{"Q6_K", "quantize_row_q6_K_ref", "block_q6_K", qkK, quantizeQ6_K(src)},
	}

	dir := t.TempDir()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			harness := filepath.Join(dir, tc.name)
			cCode := fmt.Sprintf(`#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ggml.h"
#include "ggml-quants.h"
int main(void) {
    const int N = %d;
    float *data = malloc(N * sizeof(float));
    for (int i = 0; i < N; i++)
        data[i] = (float)(i %% 2000 - 1000) * 0.001f;
    int nb = N / %d;
    size_t bsz = sizeof(%s);
    void *out = calloc(nb, bsz);
    %s(data, out, N);
    fwrite(out, bsz, nb, stdout);
    free(data); free(out);
    return 0;
}`, N, tc.qk, tc.block, tc.ref)

			cSrc := filepath.Join(dir, tc.name+".c")
			if err := os.WriteFile(cSrc, []byte(cCode), 0o644); err != nil {
				t.Fatal(err)
			}

			compileArgs := []string{
				"-O2", "-o", harness, cSrc,
				"-I" + incDir, "-I" + srcDir,
				"-L" + libDir, "-lggml-base",
				"-lm", "-lpthread",
				"-Wl,-rpath," + libDir,
			}
			if out, err := exec.Command("gcc", compileArgs...).CombinedOutput(); err != nil {
				t.Fatalf("compile failed: %v\n%s", err, out)
			}

			cOutput, err := exec.Command(harness).Output()
			if err != nil {
				t.Fatalf("harness failed: %v", err)
			}

			goHash := sha256.Sum256(tc.goOut)
			cHash := sha256.Sum256(cOutput)
			t.Logf("Go   %s: %d bytes, SHA-256 %s", tc.name, len(tc.goOut), hex.EncodeToString(goHash[:]))
			t.Logf("GGML %s: %d bytes, SHA-256 %s", tc.name, len(cOutput), hex.EncodeToString(cHash[:]))

			if len(tc.goOut) != len(cOutput) {
				t.Fatalf("length mismatch: Go=%d GGML=%d", len(tc.goOut), len(cOutput))
			}
			if goHash != cHash {
				bsz := len(cOutput) / (N / tc.qk)
				for i := range cOutput {
					if tc.goOut[i] != cOutput[i] {
						t.Logf("first mismatch at byte %d (block %d, offset %d): Go=0x%02x GGML=0x%02x",
							i, i/bsz, i%bsz, tc.goOut[i], cOutput[i])
						break
					}
				}
				t.Fatalf("HASH MISMATCH: Go %s does not match ggml's %s", tc.name, tc.ref)
			}
		})
	}
}
