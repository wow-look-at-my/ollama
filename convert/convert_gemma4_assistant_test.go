package convert

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
)

// writeGemma4AssistantFixture writes a synthetic assistant checkpoint: a
// config.json plus a model.safetensors whose tensors are F32 and zero-filled
// (the converter only cares about names/shapes/dtypes, not data values).
func writeGemma4AssistantFixture(t *testing.T, dir, configJSON string, shapes map[string][]int) {
	t.Helper()

	names := make([]string, 0, len(shapes))
	for name := range shapes {
		names = append(names, name)
	}
	slices.Sort(names)

	td := make(map[string]*tensorData, len(shapes))
	offset := 0
	for _, name := range names {
		n := 1
		for _, d := range shapes[name] {
			n *= d
		}
		size := n * 4 // F32
		td[name] = &tensorData{Offsets: []int{offset, offset + size}, Type: "F32", Shape: shapes[name]}
		offset += size
	}

	header, err := json.Marshal(td)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, int64(len(header))); err != nil {
		t.Fatal(err)
	}
	buf.Write(header)
	buf.Write(make([]byte, offset)) // zero-filled tensor data region

	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatal(err)
	}
}

// gemma4TestBaseKV builds a synthetic converted-target KV (arch gemma4, an
// embedding_length = the assistant's n_embd_backbone, and a 16-token tokenizer)
// and round-trips it through WriteGGUF/Decode so the accessors see the decoded
// array types — exactly how production obtains baseKV from the base GGUF.
func gemma4TestBaseKV(t *testing.T) ggml.KV {
	t.Helper()
	tokens := make([]string, 16)
	for i := range tokens {
		tokens[i] = fmt.Sprintf("tok%d", i)
	}
	plain := ggml.KV{
		"general.architecture":         "gemma4",
		"general.quantization_version": uint32(2),
		"gemma4.embedding_length":      uint32(8),
		"gemma4.vocab_size":            uint32(16),
		"tokenizer.ggml.model":         "llama",
		"tokenizer.ggml.pre":           "gemma4",
		"tokenizer.ggml.tokens":        tokens,
		"tokenizer.ggml.scores":        make([]float32, 16),
		"tokenizer.ggml.token_type":    make([]int32, 16),
		"tokenizer.ggml.bos_token_id":  uint32(2),
		"tokenizer.ggml.eos_token_id":  uint32(1),
	}

	f, err := os.CreateTemp(t.TempDir(), "base")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := ggml.WriteGGUF(f, plain, nil); err != nil {
		t.Fatal(err)
	}
	return decodeGGUF(t, f).KV()
}

// gemma4AssistantLayerShapes adds the per-layer HF tensors for an assistant whose
// layers are typed by layerTypes. Like Gemma 4, sliding-attention layers use a
// smaller head_dim (4) than full-attention layers (global_head_dim 8), so the
// attention tensors differ in size by layer type. n_embd=8, n_head=2, n_ff=16.
// Each layer also carries a layer_scalar (-> layer_output_scale.weight), as the
// real Gemma4AssistantForCausalLM checkpoint does.
func gemma4AssistantLayerShapes(shapes map[string][]int, layerTypes []string) {
	const (
		nEmbd         = 8
		nHead         = 2
		headDim       = 4
		globalHeadDim = 8
		nFF           = 16
	)
	for i, lt := range layerTypes {
		hd := headDim
		if lt != "sliding_attention" {
			hd = globalHeadDim
		}
		qOut := nHead * hd
		p := fmt.Sprintf("model.layers.%d.", i)
		shapes[p+"input_layernorm.weight"] = []int{nEmbd}
		shapes[p+"self_attn.q_proj.weight"] = []int{qOut, nEmbd}
		shapes[p+"self_attn.q_norm.weight"] = []int{hd}
		shapes[p+"self_attn.o_proj.weight"] = []int{nEmbd, qOut}
		shapes[p+"post_attention_layernorm.weight"] = []int{nEmbd}
		shapes[p+"pre_feedforward_layernorm.weight"] = []int{nEmbd}
		shapes[p+"mlp.gate_proj.weight"] = []int{nFF, nEmbd}
		shapes[p+"mlp.up_proj.weight"] = []int{nFF, nEmbd}
		shapes[p+"mlp.down_proj.weight"] = []int{nEmbd, nFF}
		shapes[p+"post_feedforward_layernorm.weight"] = []int{nEmbd}
		shapes[p+"layer_scalar"] = []int{1}
	}
}

func decodeGGUF(t *testing.T, f *os.File) *ggml.GGML {
	t.Helper()
	r, err := os.Open(f.Name())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	decoded, err := ggml.Decode(r, -1)
	if err != nil {
		t.Fatal(err)
	}
	return decoded
}

func TestConvertGemma4MTPDraft(t *testing.T) {
	dir := t.TempDir()

	// Mirrors google/gemma-4-31B-it-assistant: distinct local/global head dims
	// (head_dim 4 vs global_head_dim 8) and distinct KV head counts
	// (num_key_value_heads 2 sliding vs num_global_key_value_heads 1 global).
	config := `{
  "architectures": ["Gemma4AssistantForCausalLM"],
  "use_ordered_embeddings": false,
  "requires_target_arch": "gemma4",
  "text_config": {
    "num_hidden_layers": 2,
    "hidden_size": 8,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "head_dim": 4,
    "global_head_dim": 8,
    "intermediate_size": 16,
    "rms_norm_eps": 1e-6,
    "sliding_window": 4,
    "num_kv_shared_layers": 2,
    "attention_k_eq_v": true,
    "layer_types": ["sliding_attention", "full_attention"],
    "rope_parameters": {
      "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
      "sliding_attention": {"rope_theta": 10000.0}
    }
  }
}`

	shapes := map[string][]int{
		"pre_projection.weight":     {8, 16},
		"post_projection.weight":    {8, 8},
		"model.embed_tokens.weight": {16, 8},
		"model.norm.weight":         {8},
	}
	gemma4AssistantLayerShapes(shapes, []string{"sliding_attention", "full_attention"})
	writeGemma4AssistantFixture(t, dir, config, shapes)

	out, err := os.CreateTemp(t.TempDir(), "assistant")
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()

	if err := ConvertGemma4MTPDraft(os.DirFS(dir), out, gemma4TestBaseKV(t)); err != nil {
		t.Fatal(err)
	}

	decoded := decodeGGUF(t, out)
	kv := decoded.KV()

	if got := kv.Architecture(); got != "gemma4_assistant" {
		t.Errorf("architecture = %q, want gemma4_assistant", got)
	}
	if got := kv.Uint("n_embd_backbone"); got != 8 {
		t.Errorf("n_embd_backbone = %d, want 8", got)
	}
	if got := kv.Uint("block_count"); got != 2 {
		t.Errorf("block_count = %d, want 2", got)
	}
	if got := kv.Uint("embedding_length"); got != 8 {
		t.Errorf("embedding_length = %d, want 8", got)
	}
	if got := kv.Uint("vocab_size"); got != 16 {
		t.Errorf("vocab_size = %d, want 16", got)
	}
	if got := kv.Bools("attention.sliding_window_pattern"); !slices.Equal(got, []bool{true, false}) {
		t.Errorf("sliding_window_pattern = %v, want [true false]", got)
	}
	if got := len(kv.Strings("tokenizer.ggml.tokens")); got != 16 {
		t.Errorf("tokenizer tokens = %d, want 16", got)
	}
	if v := kv["gemma4.embedding_length"]; v != nil {
		t.Errorf("stale gemma4.* key present: gemma4.embedding_length = %v", v)
	}

	// Head dims: global (full-attention) vs SWA (sliding) differ.
	if got := kv.Uint("attention.key_length"); got != 8 {
		t.Errorf("key_length = %d, want 8 (global_head_dim)", got)
	}
	if got := kv.Uint("attention.key_length_swa"); got != 4 {
		t.Errorf("key_length_swa = %d, want 4 (head_dim)", got)
	}
	if got := kv.Uint("rope.dimension_count"); got != 8 {
		t.Errorf("rope.dimension_count = %d, want 8 (global_head_dim)", got)
	}
	// rope.dimension_count_swa must be emitted: the loader defaults n_rot_swa to
	// n_rot_full (the global head dim), which would rope too many dims on the
	// narrower SWA heads. See llama-model.cpp / gemma4-assistant.cpp.
	if got := kv.Uint("rope.dimension_count_swa"); got != 4 {
		t.Errorf("rope.dimension_count_swa = %d, want 4 (head_dim)", got)
	}
	// Per-layer KV head count: sliding layer = num_key_value_heads (2), full
	// (global) layer = num_global_key_value_heads (1). Gemma 4 differs by type.
	if got := kv.Ints("attention.head_count_kv"); !slices.Equal(got, []int32{2, 1}) {
		t.Errorf("head_count_kv = %v, want [2 1]", got)
	}

	byName := map[string][]uint64{}
	for _, tns := range decoded.Tensors().Items() {
		byName[tns.Name] = tns.Shape
	}

	// GGUF ne order = reversed HF order.
	shapeChecks := map[string][]uint64{
		"token_embd.weight":               {8, 16},
		"mtp.pre_projection.weight":       {16, 8},
		"mtp.post_projection.weight":      {8, 8},
		"output_norm.weight":              {8},
		"blk.0.attn_q.weight":             {8, 8},  // sliding: n_head*head_dim = 2*4
		"blk.0.attn_q_norm.weight":        {4},     // head_dim
		"blk.1.attn_q.weight":             {8, 16}, // full: n_head*global_head_dim = 2*8
		"blk.1.attn_q_norm.weight":        {8},     // global_head_dim
		"blk.0.ffn_gate.weight":           {8, 16},
		"blk.0.ffn_down.weight":           {16, 8},
		"blk.0.layer_output_scale.weight": {1},
		"rope_freqs.weight":               {4}, // global_head_dim/2
	}
	for name, want := range shapeChecks {
		got, ok := byName[name]
		if !ok {
			t.Errorf("missing tensor %q", name)
			continue
		}
		if !slices.Equal(got, want) {
			t.Errorf("tensor %q shape = %v, want %v", name, got, want)
		}
	}

	required := []string{
		"token_embd.weight", "mtp.pre_projection.weight",
		"mtp.post_projection.weight", "output_norm.weight", "rope_freqs.weight",
	}
	for i := range 2 {
		for _, s := range []string{
			"attn_norm", "attn_q", "attn_output", "attn_q_norm", "post_attention_norm",
			"ffn_norm", "ffn_gate", "ffn_up", "ffn_down", "post_ffw_norm",
			"layer_output_scale",
		} {
			required = append(required, fmt.Sprintf("blk.%d.%s.weight", i, s))
		}
	}
	for _, name := range required {
		if _, ok := byName[name]; !ok {
			t.Errorf("missing required tensor %q", name)
		}
	}

	for name := range byName {
		if strings.HasPrefix(name, "blk.") && strings.HasSuffix(name, "rope_freqs.weight") {
			t.Errorf("unexpected per-layer rope_freqs tensor %q (want a single global rope_freqs.weight)", name)
		}
		if strings.Contains(name, "attn_k") || strings.Contains(name, "attn_v") {
			t.Errorf("unexpected attention k/v tensor %q (drafter is Q-only)", name)
		}
	}

	// 4 global + 1 rope_freqs + 2 layers x 11 (incl. layer_output_scale) = 27
	if got := len(byName); got != 27 {
		t.Errorf("tensor count = %d, want 27", got)
	}
}

func TestConvertGemma4MTPDraftRejectsNonGemma4Base(t *testing.T) {
	dir := t.TempDir()
	writeGemma4AssistantFixture(t, dir,
		`{"architectures":["Gemma4AssistantForCausalLM"],"text_config":{"num_hidden_layers":1,"hidden_size":8,"global_head_dim":4,"head_dim":4,"num_attention_heads":2}}`,
		map[string][]int{"model.norm.weight": {8}})

	out, err := os.CreateTemp(t.TempDir(), "assistant")
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()

	baseKV := gemma4TestBaseKV(t)
	baseKV["general.architecture"] = "llama"

	err = ConvertGemma4MTPDraft(os.DirFS(dir), out, baseKV)
	if err == nil || !strings.Contains(err.Error(), "gemma4") {
		t.Errorf("expected error mentioning gemma4, got %v", err)
	}
}

func TestConvertGemma4MTPDraftOrderedEmbeddings(t *testing.T) {
	dir := t.TempDir()

	config := `{
  "architectures": ["Gemma4AssistantForCausalLM"],
  "use_ordered_embeddings": true,
  "num_centroids": 4,
  "centroid_intermediate_top_k": 2,
  "text_config": {
    "num_hidden_layers": 2,
    "hidden_size": 8,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 4,
    "global_head_dim": 8,
    "intermediate_size": 16,
    "rms_norm_eps": 1e-6,
    "sliding_window": 4,
    "layer_types": ["sliding_attention", "full_attention"],
    "rope_parameters": {
      "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
      "sliding_attention": {"rope_theta": 10000.0}
    }
  }
}`

	shapes := map[string][]int{
		"pre_projection.weight":                  {8, 16},
		"post_projection.weight":                 {8, 8},
		"model.embed_tokens.weight":              {16, 8},
		"model.norm.weight":                      {8},
		"masked_embedding.centroids.weight":      {4, 8},
		"masked_embedding.token_ordering.weight": {16},
	}
	gemma4AssistantLayerShapes(shapes, []string{"sliding_attention", "full_attention"})
	writeGemma4AssistantFixture(t, dir, config, shapes)

	out, err := os.CreateTemp(t.TempDir(), "assistant")
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()

	if err := ConvertGemma4MTPDraft(os.DirFS(dir), out, gemma4TestBaseKV(t)); err != nil {
		t.Fatal(err)
	}

	decoded := decodeGGUF(t, out)
	kv := decoded.KV()

	if !kv.Bool("use_ordered_embeddings") {
		t.Errorf("use_ordered_embeddings = false, want true")
	}
	if got := kv.Uint("n_centroids"); got != 4 {
		t.Errorf("n_centroids = %d, want 4", got)
	}
	if got := kv.Uint("centroid_top_k"); got != 2 {
		t.Errorf("centroid_top_k = %d, want 2", got)
	}

	byName := map[string][]uint64{}
	for _, tns := range decoded.Tensors().Items() {
		byName[tns.Name] = tns.Shape
	}
	if got, ok := byName["mtp.centroids.weight"]; !ok {
		t.Errorf("missing mtp.centroids.weight")
	} else if !slices.Equal(got, []uint64{8, 4}) {
		t.Errorf("mtp.centroids.weight shape = %v, want [8 4]", got)
	}
	if _, ok := byName["mtp.token_ordering.weight"]; !ok {
		t.Errorf("missing mtp.token_ordering.weight")
	}
}
