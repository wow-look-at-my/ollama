package convert

import (
	"errors"
	"sort"
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

// ggml tensor type ids (block kinds) the fused quantizer can emit.
const (
	kindQ4_K uint32 = 12
	kindQ5_K uint32 = 13
	kindQ6_K uint32 = 14
)

// ErrFusedUnsupported signals that the fused single-pass quantizer cannot
// reproduce llama-quantize's output for this model/type (e.g. a tensor would
// need a quant kind we don't implement, or has an incompatible shape). Callers
// fall back to shelling out to llama-quantize.
var ErrFusedUnsupported = errors.New("convert: fused quantization unsupported for this model/type")

// tensorCategory mirrors llama.cpp's tensor_category: broad buckets used by the
// k-quant "mixture" to decide each tensor's precision.
type tensorCategory int

const (
	catOther tensorCategory = iota
	catTokenEmbd
	catOutput
	catAttnQ
	catAttnK
	catAttnV
	catAttnQKV
	catAttnKVB
	catAttnOutput
	catFFNUp
	catFFNGate
	catFFNDown
)

// categorize ports llama.cpp's tensor_get_category (src/llama-quant.cpp).
func categorize(name string) tensorCategory {
	switch {
	case name == "output.weight":
		return catOutput
	case name == "token_embd.weight" || name == "per_layer_token_embd.weight":
		return catTokenEmbd
	case strings.Contains(name, "attn_qkv.weight"):
		return catAttnQKV
	case strings.Contains(name, "attn_kv_b.weight"):
		return catAttnKVB
	case strings.Contains(name, "attn_v.weight"):
		return catAttnV
	case strings.Contains(name, "attn_k.weight"):
		return catAttnK
	case strings.Contains(name, "attn_q.weight"):
		return catAttnQ
	case strings.Contains(name, "attn_output.weight"):
		return catAttnOutput
	case strings.Contains(name, "ffn_up"):
		return catFFNUp
	case strings.Contains(name, "ffn_gate"):
		return catFFNGate
	case strings.Contains(name, "ffn_down"):
		return catFFNDown
	default:
		return catOther
	}
}

func categoryIsAttnV(c tensorCategory) bool {
	return c == catAttnV || c == catAttnQKV || c == catAttnKVB
}

// allowsQuantization ports the relevant subset of llama.cpp's
// tensor_allows_quantization: only 2D+ "*.weight" tensors are quantized, and a
// handful of small/structural tensors are always left alone. We deliberately
// reject (via the caller's fallback) anything outside the cases we can mirror
// exactly rather than risk a divergent file.
func allowsQuantization(name string, dims int) bool {
	if dims < 2 {
		return false
	}
	if !strings.HasSuffix(name, "weight") {
		return false
	}
	if strings.Contains(name, "_norm.weight") {
		return false
	}
	// excluded structural/small tensors (kept at source precision by ggml)
	for _, ex := range []string{
		"ffn_gate_inp.weight",
		"altup", "laurel", "per_layer_model_proj",
		"ssm_conv1d", "shortconv.conv.weight",
		"time_mix_",
		"attn_rel_b.weight",
		".position_embd", "sam.pos_embd", "sam.neck.", "sam.net_",
		".rel_pos", ".patch_embd", ".patch_merger",
	} {
		if strings.Contains(name, ex) {
			return false
		}
	}
	return true
}

// useMoreBits ports llama.cpp's use_more_bits: the layers that get bumped to a
// higher-precision quant in the *_K_M mixtures.
func useMoreBits(iLayer, nLayers int) bool {
	return iLayer < nLayers/8 || iLayer >= 7*nLayers/8 || (iLayer-nLayers/8)%3 == 2
}

// parseLayer mirrors sscanf(name, "blk.%d.", &n): the layer index, or -1.
func parseLayer(name string) int {
	const p = "blk."
	if !strings.HasPrefix(name, p) {
		return -1
	}
	rest := name[len(p):]
	i := 0
	for i < len(rest) && rest[i] >= '0' && rest[i] <= '9' {
		i++
	}
	if i == 0 || i >= len(rest) || rest[i] != '.' {
		return -1
	}
	n, err := strconv.Atoi(rest[:i])
	if err != nil {
		return -1
	}
	return n
}

// weightNameLess ports llama-model-loader.h's weight_name_comparer, which is
// the order llama-quantize iterates tensors and therefore the order in which
// the i_attention_wv / i_ffn_down counters advance.
func weightNameLess(a, b string) bool {
	al, bl := parseLayer(a), parseLayer(b)
	if al != bl {
		return al < bl
	}
	return a < b
}

// mixtureState mirrors the counters in llama.cpp's quantize_state_impl that the
// per-tensor type selection reads and advances.
type mixtureState struct {
	nAttentionWV      int
	iAttentionWV      int
	nFFNDown          int
	iFFNDown          int
	hasTiedEmbeddings bool
}

// quantType ports the subset of llama_tensor_get_type_impl covering the ftypes
// ollama exposes for create (Q8_0, Q4_K_S, Q4_K_M) on dense, non-MoE,
// non-Falcon models. It returns the chosen ggml type and whether the fused
// quantizer can emit it; counters advance exactly as llama.cpp's do.
func (st *mixtureState) quantType(ftype ggml.FileType, cat tensorCategory) (uint32, bool) {
	var def uint32
	switch ftype {
	case ggml.FileTypeQ8_0:
		def = kindQ8_0
	case ggml.FileTypeQ4_K_S, ggml.FileTypeQ4_K_M:
		def = kindQ4_K
	default:
		return 0, false
	}
	newType := def

	switch {
	case cat == catOutput || (st.hasTiedEmbeddings && cat == catTokenEmbd):
		// output / tied token embeddings: Q6_K for the K-quants, unchanged for Q8_0
		if ftype != ggml.FileTypeQ8_0 {
			newType = kindQ6_K
		}
	case cat == catTokenEmbd:
		// untied token embeddings: no override for these ftypes
	case categoryIsAttnV(cat):
		if ftype == ggml.FileTypeQ4_K_M && useMoreBits(st.iAttentionWV, st.nAttentionWV) {
			newType = kindQ6_K
		} else if ftype == ggml.FileTypeQ4_K_S && st.iAttentionWV < 4 {
			newType = kindQ5_K
		}
		st.iAttentionWV++
	case cat == catFFNDown:
		if ftype == ggml.FileTypeQ4_K_M {
			if useMoreBits(st.iFFNDown, st.nFFNDown) {
				newType = kindQ6_K
			}
		} else if ftype == ggml.FileTypeQ4_K_S && st.iFFNDown < st.nFFNDown/8 {
			newType = kindQ5_K
		}
		st.iFFNDown++
	case cat == catAttnQKV:
		if ftype == ggml.FileTypeQ4_K_M {
			newType = kindQ5_K
		}
	}

	// Q5_K is part of the mixture but we don't implement its kernel yet; signal
	// the caller to fall back to llama-quantize rather than emit a wrong file.
	if newType == kindQ5_K {
		return newType, false
	}
	return newType, true
}

// planQuantization computes the per-tensor target kind for every quantizable
// tensor, matching llama-quantize. It returns ok=false when the fused path
// can't faithfully reproduce the result (unsupported kind or incompatible
// shape), so the caller can fall back.
func planQuantization(ts []*ggml.Tensor, ftype ggml.FileType, nLayer int) (map[string]uint32, bool) {
	type entry struct {
		name string
		cat  tensorCategory
		ne0  uint64
		ok   bool
	}

	st := &mixtureState{hasTiedEmbeddings: true, nFFNDown: nLayer}
	entries := make([]entry, 0, len(ts))
	for _, t := range ts {
		cat := categorize(t.Name)
		if categoryIsAttnV(cat) {
			st.nAttentionWV++
		}
		if cat == catOutput {
			st.hasTiedEmbeddings = false
		}
		var ne0 uint64
		if len(t.Shape) > 0 {
			ne0 = t.Shape[len(t.Shape)-1] // ggml ne[0] (writeFile reverses Shape)
		}
		entries = append(entries, entry{t.Name, cat, ne0, allowsQuantization(t.Name, len(t.Shape))})
	}

	// counters advance in weights_map order, not file order
	sort.SliceStable(entries, func(i, j int) bool { return weightNameLess(entries[i].name, entries[j].name) })

	plan := make(map[string]uint32)
	for _, e := range entries {
		if !e.ok {
			continue
		}
		kind, ok := st.quantType(ftype, e.cat)
		if !ok {
			return nil, false
		}
		blk := uint64(qkK)
		if kind == kindQ8_0 {
			blk = uint64(qk80)
		}
		// the fused kernels quantize the flat row-major data, which only matches
		// ggml's per-row quantization when each row aligns to a block boundary
		if e.ne0 == 0 || e.ne0%blk != 0 {
			return nil, false
		}
		plan[e.name] = kind
	}
	return plan, true
}

// applyFusedQuantization rewrites the gathered model tensors in place so each
// one quantizes to its llama-quantize-matching target kind during writeFile,
// and stamps the quantized file_type into the KV. It returns ErrFusedUnsupported
// when the fused path can't reproduce llama-quantize's output (unsupported
// kind/shape, or a tensor without a plain safetensor source), so callers fall
// back to shelling out to llama-quantize.
func applyFusedQuantization(tensors []*ggml.Tensor, kv KV, fileType ggml.FileType) error {
	nLayer := int(kv.Uint(kv.Architecture() + ".block_count"))
	plan, ok := planQuantization(tensors, fileType, nLayer)
	if !ok {
		return ErrFusedUnsupported
	}

	for _, gt := range tensors {
		kind, quant := plan[gt.Name]
		if !quant {
			continue
		}
		st, ok := gt.WriterTo.(safetensor)
		if !ok {
			// e.g. an FP8-scaled tensor with a custom WriterTo; let llama-quantize handle it
			return ErrFusedUnsupported
		}
		gt.WriterTo = fusedQuantizer{src: st, kind: kind}
		gt.Kind = kind
	}

	kv["general.file_type"] = uint32(fileType)
	kv["general.quantization_version"] = uint32(2)
	return nil
}
