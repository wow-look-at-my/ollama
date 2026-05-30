package convert

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestQuantizeQ8_0_MatchesC(t *testing.T) {
	const N = 1 << 20

	src := make([]float32, N)
	for i := range src {
		src[i] = float32(i%2000-1000) * 0.001
	}

	goOut := quantizeQ8_0(src)
	goHash := sha256.Sum256(goOut)

	cRef := filepath.Join(t.TempDir(), "q8_reference")
	cSrc := filepath.Join(t.TempDir(), "q8_reference.c")

	cCode := `#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#define QK8_0 32
static inline uint16_t fp32_to_fp16(float f) {
    uint32_t b; __builtin_memcpy(&b, &f, 4);
    uint32_t sign = (b >> 16) & 0x8000;
    int32_t exp = ((b >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (b >> 13) & 0x3FF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (exp << 10) | mant;
}
typedef struct { uint16_t d; int8_t qs[QK8_0]; } block_q8_0;
void quantize_row_q8_0_ref(const float *x, block_q8_0 *y, int64_t k) {
    const int nb = k / QK8_0;
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            float v = fabsf(x[i*QK8_0 + j]);
            if (v > amax) amax = v;
        }
        float d = amax / ((1 << 7) - 1);
        float id = d ? 1.0f/d : 0.0f;
        y[i].d = fp32_to_fp16(d);
        for (int j = 0; j < QK8_0; ++j)
            y[i].qs[j] = roundf(x[i*QK8_0 + j] * id);
    }
}
int main() {
    int N = 1 << 20;
    float *data = malloc(N * sizeof(float));
    for (int i = 0; i < N; i++)
        data[i] = (float)(i % 2000 - 1000) * 0.001f;
    int nb = N / QK8_0;
    block_q8_0 *out = calloc(nb, sizeof(block_q8_0));
    quantize_row_q8_0_ref(data, out, N);
    fwrite(out, sizeof(block_q8_0), nb, stdout);
    free(data); free(out);
    return 0;
}
`
	if err := os.WriteFile(cSrc, []byte(cCode), 0644); err != nil {
		t.Fatal(err)
	}

	if out, err := exec.Command("gcc", "-O2", "-o", cRef, cSrc, "-lm").CombinedOutput(); err != nil {
		t.Skipf("gcc not available or compile failed: %v\n%s", err, out)
	}

	cOutput, err := exec.Command(cRef).Output()
	if err != nil {
		t.Fatalf("C reference failed: %v", err)
	}

	cHash := sha256.Sum256(cOutput)

	t.Logf("Go output: %d bytes, SHA-256: %s", len(goOut), hex.EncodeToString(goHash[:]))
	t.Logf("C  output: %d bytes, SHA-256: %s", len(cOutput), hex.EncodeToString(cHash[:]))

	if goHash != cHash {
		t.Fatalf("HASH MISMATCH: Go quantization does not match C reference")
	}
}
