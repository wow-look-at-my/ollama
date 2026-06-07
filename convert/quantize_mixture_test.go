package convert

import (
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
)

type zeroWriterTo struct{ n int64 }

func (z zeroWriterTo) WriteTo(w io.Writer) (int64, error) {
	buf := make([]byte, 32<<10)
	var written int64
	for written < z.n {
		chunk := int64(len(buf))
		if rem := z.n - written; rem < chunk {
			chunk = rem
		}
		m, err := w.Write(buf[:chunk])
		written += int64(m)
		if err != nil {
			return written, err
		}
	}
	return written, nil
}

// TestFusedMixtureMatchesLlamaQuantize builds a synthetic dense model, quantizes
// it with the real llama-quantize binary, and asserts that planQuantization
// chose the same per-tensor type for every tensor. This validates the ported
// k-quant mixture (including the weight_name_comparer ordering that drives the
// i_attention_wv / i_ffn_down counters) against ground truth. gemma4 is a dense,
// non-MoE, non-Falcon arch, so it exercises the same selection path as llama.
//
// Set OLLAMA_LLAMA_QUANTIZE to the llama-quantize binary (and ensure its ggml
// libraries are on the loader path) to run; otherwise it skips.
func TestFusedMixtureMatchesLlamaQuantize(t *testing.T) {
	exe := os.Getenv("OLLAMA_LLAMA_QUANTIZE")
	if exe == "" {
		t.Skip("set OLLAMA_LLAMA_QUANTIZE to a llama-quantize binary to run")
	}

	// tied: no output.weight (token embeddings reused as output, like gemma) so
	// token_embd is the one bumped to Q6_K. untied: separate output.weight.
	for _, tc := range []struct {
		name string
		tied bool
	}{{"untied", false}, {"tied", true}} {
		t.Run(tc.name, func(t *testing.T) {
			testFusedMixture(t, exe, tc.tied)
		})
	}
}

func testFusedMixture(t *testing.T, exe string, tied bool) {
	const (
		nLayer = 16
		nEmbd  = 512
		nFF    = 1024
		nVocab = 512
		nHead  = 8
	)
	type spec struct {
		name     string
		ne0, ne1 int
	}
	specs := []spec{
		{"token_embd.weight", nEmbd, nVocab},
		{"output_norm.weight", nEmbd, 0},
	}
	if !tied {
		specs = append(specs, spec{"output.weight", nEmbd, nVocab})
	}
	for i := 0; i < nLayer; i++ {
		p := "blk." + itoa(i) + "."
		specs = append(specs,
			spec{p + "attn_q.weight", nEmbd, nEmbd},
			spec{p + "attn_k.weight", nEmbd, nEmbd},
			spec{p + "attn_v.weight", nEmbd, nEmbd},
			spec{p + "attn_output.weight", nEmbd, nEmbd},
			spec{p + "ffn_gate.weight", nEmbd, nFF},
			spec{p + "ffn_up.weight", nEmbd, nFF},
			spec{p + "ffn_down.weight", nFF, nEmbd},
			spec{p + "attn_norm.weight", nEmbd, 0},
			spec{p + "ffn_norm.weight", nEmbd, 0},
		)
	}

	// GGUF tensors use ne-order shapes; the plan tensors use the converter's
	// pre-reversal shape (ne0 last), which is what planQuantization reads.
	gguf := make([]*ggml.Tensor, 0, len(specs))
	plan := make([]*ggml.Tensor, 0, len(specs))
	for _, s := range specs {
		elems := int64(s.ne0)
		var ggufShape, planShape []uint64
		if s.ne1 == 0 {
			ggufShape = []uint64{uint64(s.ne0)}
			planShape = []uint64{uint64(s.ne0)}
		} else {
			elems *= int64(s.ne1)
			ggufShape = []uint64{uint64(s.ne0), uint64(s.ne1)}
			planShape = []uint64{uint64(s.ne1), uint64(s.ne0)}
		}
		gguf = append(gguf, &ggml.Tensor{Name: s.name, Kind: 0 /*F32*/, Shape: ggufShape, WriterTo: zeroWriterTo{n: elems * 4}})
		plan = append(plan, &ggml.Tensor{Name: s.name, Kind: 0, Shape: planShape})
	}

	kv := ggml.KV{
		"general.architecture":                   "llama",
		"general.name":                            "synthetic",
		"general.file_type":                       uint32(0),
		"llama.block_count":                       uint32(nLayer),
		"llama.context_length":                    uint32(2048),
		"llama.embedding_length":                  uint32(nEmbd),
		"llama.feed_forward_length":               uint32(nFF),
		"llama.attention.head_count":              uint32(nHead),
		"llama.attention.head_count_kv":           uint32(nHead),
		"llama.attention.layer_norm_rms_epsilon":  float32(1e-5),
		"llama.rope.dimension_count":              uint32(nEmbd / nHead),
		"tokenizer.ggml.model":                    "llama",
		"tokenizer.ggml.tokens":                   make([]string, nVocab),
	}

	dir := t.TempDir()
	inPath := filepath.Join(dir, "synth.gguf")
	outPath := filepath.Join(dir, "synth.q4km.gguf")

	in, err := os.Create(inPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := ggml.WriteGGUF(in, kv, gguf); err != nil {
		in.Close()
		t.Fatalf("WriteGGUF: %v", err)
	}
	in.Close()

	cmd := exec.Command(exe, "--allow-requantize", inPath, outPath, "Q4_K_M")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("llama-quantize failed: %v\n%s", err, out)
	}

	outFile, err := os.Open(outPath)
	if err != nil {
		t.Fatal(err)
	}
	defer outFile.Close()
	decoded, err := ggml.Decode(outFile, -1)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]uint32{}
	for _, ti := range decoded.Tensors().Items() {
		got[ti.Name] = ti.Kind
	}

	planned, ok := planQuantization(plan, ggml.FileTypeQ4_K_M, nLayer)
	if !ok {
		t.Fatal("planQuantization reported the model unsupported")
	}

	for _, s := range specs {
		want := uint32(0) // F32 for tensors we leave unquantized (norms)
		if k, inPlan := planned[s.name]; inPlan {
			want = k
		}
		if got[s.name] != want {
			t.Errorf("%s: plan=%s llama-quantize=%s", s.name, ggml.TensorType(want), ggml.TensorType(got[s.name]))
		}
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [20]byte
	pos := len(b)
	for i > 0 {
		pos--
		b[pos] = byte('0' + i%10)
		i /= 10
	}
	return string(b[pos:])
}
