package convert

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"os"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type gemma4Model struct {
	gemmaModel
	Architecture string
	TextModel    struct {
		HiddenSize              uint32   `json:"hidden_size"`
		NumHiddenLayers         uint32   `json:"num_hidden_layers"`
		IntermediateSize        uint32   `json:"intermediate_size"`
		NumAttentionHeads       uint32   `json:"num_attention_heads"`
		NumKeyValueHeads        uint32   `json:"num_key_value_heads"`
		HeadDim                 uint32   `json:"head_dim"`
		GlobalHeadDim           uint32   `json:"global_head_dim"`
		VocabSize               uint32   `json:"vocab_size"`
		RMSNormEps              float32  `json:"rms_norm_eps"`
		MaxPositionEmbeddings   uint32   `json:"max_position_embeddings"`
		SlidingWindow           uint32   `json:"sliding_window"`
		SlidingWindowPattern    *int32   `json:"_sliding_window_pattern"`
		LayerTypes              []string `json:"layer_types"`
		FinalLogitSoftcapping   float32  `json:"final_logit_softcapping"`
		EnableMoeBlock          bool     `json:"enable_moe_block"`
		NumExperts              *uint32  `json:"num_experts"`
		TopKExperts             *uint32  `json:"top_k_experts"`
		ExpertIntermediateSize  *uint32  `json:"moe_intermediate_size"`
		HiddenSizePerLayerInput *uint32  `json:"hidden_size_per_layer_input"`
		NumKVSharedLayers       uint32   `json:"num_kv_shared_layers"`
		AttentionKEqV           bool     `json:"attention_k_eq_v"`
		NumGlobalKeyValueHeads  *uint32  `json:"num_global_key_value_heads"`
		QueryPreAttnScalar      *uint32  `json:"query_pre_attn_scalar"`
		UseDoubleWideMLP        bool     `json:"use_double_wide_mlp"`
		RopeParameters          map[string]*struct {
			RopeTheta           float32  `json:"rope_theta"`
			PartialRotaryFactor *float32 `json:"partial_rotary_factor"`
		} `json:"rope_parameters"`
	} `json:"text_config"`

	VisionModel struct {
		HiddenSize        uint32  `json:"hidden_size"`
		NumHiddenLayers   uint32  `json:"num_hidden_layers"`
		NumAttentionHeads uint32  `json:"num_attention_heads"`
		IntermediateSize  uint32  `json:"intermediate_size"`
		PatchSize         uint32  `json:"patch_size"`
		NumChannels       uint32  `json:"num_channels"`
		PoolingKernelSize uint32  `json:"pooling_kernel_size"`
		LayerNormEps      float32 `json:"layer_norm_eps"`
	} `json:"vision_config"`

	AudioModel *struct {
		HiddenSize        uint32  `json:"hidden_size"`
		OutputProjDims    uint32  `json:"output_proj_dims"`
		NumHiddenLayers   uint32  `json:"num_hidden_layers"`
		NumAttentionHeads uint32  `json:"num_attention_heads"`
		ConvKernelSize    uint32  `json:"conv_kernel_size"`
		RMSNormEps        float32 `json:"rms_norm_eps"`
	} `json:"audio_config"`
}

func (p *gemma4Model) KV(t *Tokenizer) KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "gemma4"
	kv["tokenizer.ggml.model"] = "llama"
	kv["tokenizer.ggml.pre"] = "gemma4"

	tc := p.TextModel

	kv["gemma4.block_count"] = tc.NumHiddenLayers
	kv["gemma4.embedding_length"] = tc.HiddenSize

	// Per-layer FFN width: when use_double_wide_mlp is set, KV-shared layers get 2x FFN width.
	if tc.UseDoubleWideMLP && tc.NumKVSharedLayers > 0 {
		firstShared := int(tc.NumHiddenLayers) - int(tc.NumKVSharedLayers)
		ffnWidths := make([]int32, tc.NumHiddenLayers)
		for i := range ffnWidths {
			if i >= firstShared {
				ffnWidths[i] = int32(tc.IntermediateSize * 2)
			} else {
				ffnWidths[i] = int32(tc.IntermediateSize)
			}
		}
		kv["gemma4.feed_forward_length"] = ffnWidths
	} else {
		kv["gemma4.feed_forward_length"] = tc.IntermediateSize
	}
	kv["gemma4.context_length"] = tc.MaxPositionEmbeddings
	kv["gemma4.attention.head_count"] = tc.NumAttentionHeads
	// Per-layer KV head count array: SWA layers use NumKeyValueHeads, global layers use NumGlobalKeyValueHeads
	if tc.NumGlobalKeyValueHeads != nil && *tc.NumGlobalKeyValueHeads != tc.NumKeyValueHeads && len(tc.LayerTypes) > 0 {
		kvHeads := make([]int32, len(tc.LayerTypes))
		for i, lt := range tc.LayerTypes {
			if lt == "sliding_attention" {
				kvHeads[i] = int32(tc.NumKeyValueHeads)
			} else {
				kvHeads[i] = int32(*tc.NumGlobalKeyValueHeads)
			}
		}
		kv["gemma4.attention.head_count_kv"] = kvHeads
	} else {
		kv["gemma4.attention.head_count_kv"] = tc.NumKeyValueHeads
	}
	// key_length = global head dim, key_length_swa = local (SWA) head dim
	kv["gemma4.attention.key_length"] = tc.GlobalHeadDim
	kv["gemma4.attention.value_length"] = tc.GlobalHeadDim
	kv["gemma4.attention.key_length_swa"] = tc.HeadDim
	kv["gemma4.attention.value_length_swa"] = tc.HeadDim
	kv["gemma4.attention.layer_norm_rms_epsilon"] = tc.RMSNormEps
	kv["gemma4.attention.sliding_window"] = tc.SlidingWindow

	// Sliding window pattern from layer_types
	if len(tc.LayerTypes) > 0 {
		kv["gemma4.attention.sliding_window_pattern"] = slices.Collect(func(yield func(bool) bool) {
			for _, lt := range tc.LayerTypes {
				if !yield(lt == "sliding_attention") {
					break
				}
			}
		})
	}

	kv["gemma4.attention.shared_kv_layers"] = tc.NumKVSharedLayers

	// RoPE: dimension_count is the full global head dim (freq_factors handle partial rotation)
	if rp, ok := tc.RopeParameters["full_attention"]; ok && rp != nil {
		kv["gemma4.rope.freq_base"] = rp.RopeTheta
		kv["gemma4.rope.dimension_count"] = tc.GlobalHeadDim
	}
	if rp, ok := tc.RopeParameters["sliding_attention"]; ok && rp != nil {
		kv["gemma4.rope.freq_base_swa"] = rp.RopeTheta
		kv["gemma4.rope.dimension_count_swa"] = tc.HeadDim
	}

	if tc.FinalLogitSoftcapping > 0 {
		kv["gemma4.final_logit_softcapping"] = tc.FinalLogitSoftcapping
	}

	// MoE
	if tc.EnableMoeBlock && tc.NumExperts != nil {
		kv["gemma4.expert_count"] = *tc.NumExperts
		if tc.TopKExperts != nil {
			kv["gemma4.expert_used_count"] = *tc.TopKExperts
		}
		if tc.ExpertIntermediateSize != nil {
			kv["gemma4.expert_feed_forward_length"] = *tc.ExpertIntermediateSize
		}
	}

	// PLE — always emit, even when 0
	pleSize := uint32(0)
	if tc.HiddenSizePerLayerInput != nil {
		pleSize = *tc.HiddenSizePerLayerInput
	}
	kv["gemma4.embedding_length_per_layer_input"] = pleSize

	// Vision model KV metadata
	vc := p.VisionModel
	if vc.NumHiddenLayers > 0 {
		kv["gemma4.vision.block_count"] = vc.NumHiddenLayers
		kv["gemma4.vision.embedding_length"] = vc.HiddenSize
		kv["gemma4.vision.attention.head_count"] = vc.NumAttentionHeads
		kv["gemma4.vision.feed_forward_length"] = vc.IntermediateSize
		kv["gemma4.vision.patch_size"] = vc.PatchSize
		numCh := vc.NumChannels
		if numCh == 0 {
			numCh = 3
		}
		kv["gemma4.vision.num_channels"] = numCh
		nMerge := vc.PoolingKernelSize
		if nMerge == 0 {
			nMerge = 3
		}
		kv["gemma4.vision.projector.scale_factor"] = nMerge
		eps := vc.LayerNormEps
		if eps == 0 {
			eps = 1e-6
		}
		kv["gemma4.vision.attention.layer_norm_epsilon"] = eps
	}

	// Audio model KV metadata
	if p.AudioModel != nil && p.AudioModel.NumHiddenLayers > 0 {
		ac := p.AudioModel
		kv["gemma4.audio.block_count"] = ac.NumHiddenLayers
		kv["gemma4.audio.embedding_length"] = ac.HiddenSize
		kv["gemma4.audio.feed_forward_length"] = ac.HiddenSize * 4
		kv["gemma4.audio.attention.head_count"] = ac.NumAttentionHeads
		eps := ac.RMSNormEps
		if eps == 0 {
			eps = 1e-6
		}
		kv["gemma4.audio.attention.layer_norm_epsilon"] = eps
		if ac.ConvKernelSize > 0 {
			kv["gemma4.audio.conv_kernel_size"] = ac.ConvKernelSize
		}
	}

	return kv
}

func (p *gemma4Model) Tensors(ts []Tensor) []*ggml.Tensor {
	// First pass: collect vision clamp scalar values into a packed tensor.
	// Layout: per vision layer (0..N-1), 7 linears (q,k,v,out,gate,up,down) × 4 values (inMin,inMax,outMin,outMax).
	// Then 4 values for the projector (mm.input_projection).
	clampSuffixes := []string{".input_min", ".input_max", ".output_min", ".output_max"}
	clampMap := make(map[string]float32)
	for _, t := range ts {
		name := t.Name()
		for _, sfx := range clampSuffixes {
			if strings.HasSuffix(name, sfx) && (strings.Contains(name, "vision_tower") || strings.Contains(name, "embed_vision")) {
				var buf bytes.Buffer
				t.WriteTo(&buf)
				data := buf.Bytes()
				if len(data) >= 4 {
					clampMap[name] = math.Float32frombits(uint32(data[0]) | uint32(data[1])<<8 | uint32(data[2])<<16 | uint32(data[3])<<24)
				}
			}
		}
	}

	var out []*ggml.Tensor
	for _, t := range ts {
		name := t.Name()

		// Skip embedding_post_projection_norm — used as weightless RMS norm in inference
		if strings.Contains(name, "embedding_post_projection_norm") {
			continue
		}

		// Vision tensor renaming: match published mmproj GGUF names
		if strings.HasPrefix(name, "v.blk.") {
			name = strings.Replace(name, ".attn_norm.", ".ln1.", 1)
			name = strings.Replace(name, ".ffn_norm.", ".ln2.", 1)
			name = strings.Replace(name, ".attn_output.", ".attn_out.", 1)
			name = strings.Replace(name, ".post_attention_norm.", ".attn_post_norm.", 1)
			name = strings.Replace(name, ".post_ffw_norm.", ".ffn_post_norm.", 1)
			name = strings.Replace(name, ".layer_output_scale.", ".out_scale.", 1)
		}

		// per_dim_scale: apply softplus to weight data and add .weight suffix.
		if strings.HasPrefix(name, "a.blk.") && strings.HasSuffix(name, "per_dim_scale") {
			name = name + ".weight"
			t.SetRepacker(softplusRepacker)
		}

		// Depthwise conv1d: squeeze middle dimension [C, 1, K] → [C, K].
		if strings.HasPrefix(name, "a.blk.") && strings.Contains(name, "conv_dw") && strings.HasSuffix(name, ".weight") {
			t.SetRepacker(squeezeMiddleDim)
		}

		shape := t.Shape()

		// Convert scalar tensors (input_min/max, output_min/max) to 1D
		if len(shape) == 0 {
			shape = []uint64{1}
		}

		// Depthwise conv1d shape: safetensors [C, 1, K] → GGUF ne[K, C].
		// Shape array here maps to GGUF ne[] directly, but safetensors reader
		// stores shape in PyTorch order [C, 1, K] which the GGUF writer inverts.
		// Published GGUF has ne[0]=K, ne[1]=C → shape array must be [K, C].
		if strings.HasPrefix(name, "a.blk.") && strings.Contains(name, "conv_dw") && strings.HasSuffix(name, ".weight") && len(shape) == 3 {
			shape = []uint64{shape[0], shape[2]}
		}

		// MoE expert weights: no transpose needed. Safetensors stores [experts, out, in]
		// which the framework reverses to GGUF ne=[in, out, experts], matching ggml_mul_mat_id.
		// (transposeExperts was incorrectly swapping dims — removed)

		// Audio conv weights are forced to F32 via tensorBase.Kind() in reader.go
		// (im2col doesn't support BF16). No kindOverride needed — the Kind() method
		// controls both the GGUF header type AND the WriteTo data encoding path.
		var kindOverride *uint32

		// Vision patch embedding: reshape from [n_embd, ksize_sq_c] to [n_embd, 3, patch_size, patch_size]
		// Must be stored as F16 (not BF16) because the Conv2D im2col kernel requires F16/F32.
		if strings.Contains(name, "v.patch_embd.weight") && len(shape) == 2 {
			nEmbd := shape[0]
			patchSize := uint64(p.VisionModel.PatchSize)
			if patchSize == 0 {
				patchSize = 16
			}
			numCh := uint64(p.VisionModel.NumChannels)
			if numCh == 0 {
				numCh = 3
			}
			t.SetRepacker(p.reshapePatchEmbed)
			shape = []uint64{nEmbd, numCh, patchSize, patchSize}
			f16Kind := uint32(1) // tensorKindFP16
			kindOverride = &f16Kind
		}

		// Vision position embedding: keep 3D [2, maxPos, nEmbd] — matching published mmproj format.
		// The framework reverses shape to GGUF ne=[nEmbd, maxPos, 2]. No data repacking needed.

		kind := t.Kind()
		if kindOverride != nil {
			kind = *kindOverride
		}
		out = append(out, &ggml.Tensor{
			Name:     name,
			Kind:     kind,
			Shape:    shape,
			WriterTo: t,
		})
	}

	// Generate a single global rope_freqs.weight for proportional RoPE on global attention layers.
	// This matches the published GGUF format: one global tensor shared by all layers.
	// Global layers use partial_rotary_factor (0.25) — only rotate that fraction of dims.
	// Dimensions beyond the rotated portion get freq_factor=1e30 (effectively no rotation).
	tc := p.TextModel
	if tc.GlobalHeadDim > 0 {
		globalFreqsSize := tc.GlobalHeadDim / 2 // freq_factors are per dimension pair

		// Compute number of rotated pairs for global layers
		partialRotaryFactor := float32(0.25) // default
		if rp, ok := tc.RopeParameters["full_attention"]; ok && rp != nil && rp.PartialRotaryFactor != nil {
			partialRotaryFactor = *rp.PartialRotaryFactor
		}
		nRotFull := int(float32(tc.GlobalHeadDim) * partialRotaryFactor / 2)

		freqs := make(ropeFactor, globalFreqsSize)
		for j := range freqs {
			if j < nRotFull {
				freqs[j] = 1.0
			} else {
				freqs[j] = 1e30 // effectively disable rotation
			}
		}
		out = append(out, &ggml.Tensor{
			Name:     "rope_freqs.weight",
			Kind:     0, // F32
			Shape:    []uint64{uint64(len(freqs))},
			WriterTo: freqs,
		})
	}

	// Emit packed vision clamp data as a single F32 tensor.
	// Layout: numLayers × 7 linears (q,k,v,out,gate,up,down) × 4 floats (inMin,inMax,outMin,outMax)
	// then 4 floats for the projector. Total = (numLayers*7 + 1) * 4 floats.
	if len(clampMap) > 0 {
		numLayers := int(p.VisionModel.NumHiddenLayers)
		linearNames := []string{"attn_q", "attn_k", "attn_v", "attn_out", "ffn_gate", "ffn_up", "ffn_down"}
		suffixes := []string{".input_min", ".input_max", ".output_min", ".output_max"}

		totalFloats := (numLayers*len(linearNames) + 1) * 4 // +1 for projector
		clampData := make([]float32, totalFloats)

		for layer := range numLayers {
			for li, ln := range linearNames {
				for si, sfx := range suffixes {
					sfxMap := map[string]string{"attn_q": "q_proj", "attn_k": "k_proj", "attn_v": "v_proj", "attn_out": "o_proj", "ffn_gate": "gate_proj", "ffn_up": "up_proj", "ffn_down": "down_proj"}
					for origName, val := range clampMap {
						if strings.Contains(origName, fmt.Sprintf("layers.%d.", layer)) && strings.HasSuffix(origName, sfx) && strings.Contains(origName, sfxMap[ln]) {
							idx := (layer*len(linearNames)+li)*4 + si
							clampData[idx] = val
							break
						}
					}
				}
			}
		}
		// Projector clamp values
		projIdx := numLayers * len(linearNames) * 4
		for si, sfx := range suffixes {
			for origName, val := range clampMap {
				if strings.Contains(origName, "input_projection") && strings.HasSuffix(origName, sfx) {
					clampData[projIdx+si] = val
					break
				}
			}
		}

		var buf bytes.Buffer
		binary.Write(&buf, binary.LittleEndian, clampData)
		out = append(out, &ggml.Tensor{
			Name:     "v.clamp_data",
			Kind:     0, // F32
			Shape:    []uint64{uint64(totalFloats)},
			WriterTo: &buf,
		})
	}

	return out
}

// reshapePatchEmbed reshapes the vision patch embedding from HF layout [n_embd, ksize*ksize*channels]
// to GGUF layout [n_embd, channels, patch_size, patch_size].
func (*gemma4Model) reshapePatchEmbed(_ string, data []float32, shape []uint64) ([]float32, error) {
	if len(shape) != 2 {
		return data, nil
	}
	nEmbd := int(shape[0])
	ksqC := int(shape[1])
	nChannels := 3
	patchSize := int(math.Sqrt(float64(ksqC / nChannels)))

	// HF layout: [n_embd, patch_size * patch_size * channels] (row-major)
	// Need: [n_embd, channels, patch_size, patch_size]
	result := make([]float32, len(data))
	for e := range nEmbd {
		for c := range nChannels {
			for h := range patchSize {
				for w := range patchSize {
					srcIdx := e*ksqC + h*patchSize*nChannels + w*nChannels + c
					dstIdx := e*nChannels*patchSize*patchSize + c*patchSize*patchSize + h*patchSize + w
					result[dstIdx] = data[srcIdx]
				}
			}
		}
	}
	shape[0] = uint64(nEmbd)
	shape[1] = uint64(nChannels * patchSize * patchSize)
	return result, nil
}

// softplusRepacker applies softplus (ln(1 + exp(x))) to tensor data.
// Used for per_dim_scale tensors which the published GGUF stores pre-activated.
func softplusRepacker(_ string, data []float32, shape []uint64) ([]float32, error) {
	result := make([]float32, len(data))
	for i, x := range data {
		result[i] = float32(math.Log(1 + math.Exp(float64(x))))
	}
	return result, nil
}

// squeezeMiddleDim squeezes the middle dimension from [C, 1, K] → [C, K] for depthwise conv1d weights.
// Data layout stays the same since the middle dim is 1 — just a shape change.
func squeezeMiddleDim(_ string, data []float32, _ []uint64) ([]float32, error) {
	return data, nil
}

func (p *gemma4Model) Replacements() []string {
	return []string{
		// ClippableLinear wraps nn.Linear — strip .linear. from weight path
		".linear.weight", ".weight",
		".linear.bias", ".bias",

		// Audio SSCP (Sub-Sample Convolution Projection)
		"model.audio_tower.subsample_conv_projection.conv_0.conv", "a.conv1d.0",
		"model.audio_tower.subsample_conv_projection.conv_0.norm", "a.conv1d.0.norm",
		"model.audio_tower.subsample_conv_projection.conv_1.conv", "a.conv1d.1",
		"model.audio_tower.subsample_conv_projection.conv_1.norm", "a.conv1d.1.norm",
		"model.audio_tower.subsample_conv_projection.layer0.conv", "a.conv1d.0",
		"model.audio_tower.subsample_conv_projection.layer0.norm", "a.conv1d.0.norm",
		"model.audio_tower.subsample_conv_projection.layer1.conv", "a.conv1d.1",
		"model.audio_tower.subsample_conv_projection.layer1.norm", "a.conv1d.1.norm",
		"model.audio_tower.subsample_conv_projection.input_proj_linear", "a.pre_encode.out",

		// Audio conformer blocks
		"model.audio_tower.conformer", "a.blk",
		"model.audio_tower.layers", "a.blk",

		// Audio conformer attention
		"attention.attn.relative_position_embedding.pos_proj", "linear_pos",
		"self_attn.relative_k_proj", "linear_pos",
		"attention.attn.per_dim_key_scale", "per_dim_k_scale",
		"attention.attn.per_dim_scale", "per_dim_scale",
		"self_attn.per_dim_scale", "per_dim_scale",
		"attention.attn.q_proj", "attn_q",
		"attention.attn.k_proj", "attn_k",
		"attention.attn.v_proj", "attn_v",
		"attention.pre_attn_norm", "ln1",
		"attention.post_norm", "ln2",
		"attention.post", "attn_out",
		"self_attn.post", "attn_out",
		"norm_pre_attn", "ln1",
		"norm_post_attn", "ln2",

		// Audio conformer feedforward
		"ffw_layer_start.pre_layer_norm", "ffn_norm",
		"ffw_layer_start.post_layer_norm", "ffn_post_norm",
		"ffw_layer_start.ffw_layer_1", "ffn_up",
		"ffw_layer_start.ffw_layer_2", "ffn_down",
		"ffw_layer_end.pre_layer_norm", "ffn_norm_1",
		"ffw_layer_end.post_layer_norm", "ffn_post_norm_1",
		"ffw_layer_end.ffw_layer_1", "ffn_up_1",
		"ffw_layer_end.ffw_layer_2", "ffn_down_1",
		"feed_forward1.pre_layer_norm", "ffn_norm",
		"feed_forward1.post_layer_norm", "ffn_post_norm",
		"feed_forward1.ffw_layer_1", "ffn_up",
		"feed_forward1.ffw_layer_2", "ffn_down",
		"feed_forward2.pre_layer_norm", "ffn_norm_1",
		"feed_forward2.post_layer_norm", "ffn_post_norm_1",
		"feed_forward2.ffw_layer_1", "ffn_up_1",
		"feed_forward2.ffw_layer_2", "ffn_down_1",

		// Audio conformer lightweight conv1d
		"lconv1d.depthwise_conv1d", "conv_dw",
		"lconv1d.pre_layer_norm", "conv_norm",
		"lconv1d.conv_norm", "norm_conv",
		"lconv1d.linear_start", "conv_pw1",
		"lconv1d.linear_end", "conv_pw2",

		// Audio block final norm
		"norm_out", "layer_pre_norm",

		// Audio embedder and output projection
		"model.embed_audio.embedding_projection", "mm.a.input_projection",
		"model.audio_tower.output_proj", "mm.a.fc",

		// Vision encoder
		"model.vision_tower.encoder.layers", "v.blk",
		"model.vision_tower.patch_embedder.input_proj", "v.patch_embd",
		"model.vision_tower.patch_embedder.position_embedding_table", "v.position_embd.weight",
		"model.vision_tower.std_bias", "v.std_bias",
		"model.vision_tower.std_scale", "v.std_scale",

		// Vision multimodal projector
		"model.embed_vision.embedding_projection", "mm.input_projection",

		// Text model
		"model.language_model.embed_tokens_per_layer", "per_layer_token_embd",
		"model.language_model.embed_tokens", "token_embd",
		"model.language_model.per_layer_model_projection", "per_layer_model_proj",
		"model.language_model.per_layer_projection_norm", "per_layer_proj_norm",
		"model.language_model.norm", "output_norm",
		"model.language_model.layers", "blk",

		// Shared attention replacements (work for both text and vision tensors)
		"input_layernorm", "attn_norm",
		"self_attn.q_proj", "attn_q",
		"self_attn.q_norm", "attn_q_norm",
		"self_attn.k_proj", "attn_k",
		"self_attn.k_norm", "attn_k_norm",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",
		"mlp.gate_proj", "ffn_gate",
		"mlp.down_proj", "ffn_down",
		"mlp.up_proj", "ffn_up",

		// Post norms
		"post_attention_layernorm", "post_attention_norm",
		"pre_feedforward_layernorm_2", "pre_ffw_norm_2",
		"pre_feedforward_layernorm", "ffn_norm",
		"post_feedforward_layernorm_1", "post_ffw_norm_1",
		"post_feedforward_layernorm_2", "post_ffw_norm_2",
		"post_feedforward_layernorm", "post_ffw_norm",

		// PLE
		"per_layer_input_gate", "inp_gate",
		"per_layer_projection", "proj",
		"post_per_layer_input_norm", "post_norm",

		// MoE
		"router.proj", "ffn_gate_inp",
		"router.scale", "ffn_gate_inp.scale",
		"router.per_expert_scale.weight", "ffn_down_exps.scale",
		"router.per_expert_scale", "ffn_down_exps.scale",
		"experts.gate_up_proj.weight", "ffn_gate_up_exps.weight",
		"experts.gate_up_proj", "ffn_gate_up_exps.weight",
		"experts.down_proj.weight", "ffn_down_exps.weight",
		"experts.down_proj", "ffn_down_exps.weight",
		"moe.gate_proj", "ffn_gate_exps.weight",
		"moe.up_proj", "ffn_up_exps.weight",
		"moe.gate_up_proj.weight", "ffn_gate_up_exps.weight",
		"moe.gate_up_proj", "ffn_gate_up_exps.weight",
		"moe.down_proj", "ffn_down_exps.weight",
		"moe.per_expert_scale.weight", "ffn_down_exps.scale",
		"moe.per_expert_scale", "ffn_down_exps.scale",

		// Layer scalar
		"layer_scalar", "layer_output_scale.weight",

		// Draft (MTP assistant) model tensors
		"draft.pre_projection", "draft.pre_projection",
		"draft.post_projection", "draft.post_projection",
		"draft.model.embed_tokens", "draft.token_embd",
		"draft.model.norm", "draft.output_norm",
		"draft.model.layers", "draft.blk",
		"draft.masked_embedding.centroids", "draft.centroids",
		"draft.masked_embedding.token_ordering", "draft.token_ordering",
	}
}

type gemma4AssistantConfig struct {
	UseOrderedEmbeddings     bool   `json:"use_ordered_embeddings"`
	NumCentroids             uint32 `json:"num_centroids"`
	CentroidIntermediateTopK uint32 `json:"centroid_intermediate_top_k"`
	RequiresTargetArch       string `json:"requires_target_arch"`
	TextConfig               struct {
		NumHiddenLayers        uint32   `json:"num_hidden_layers"`
		HiddenSize             uint32   `json:"hidden_size"`
		MaxPositionEmbeddings  uint32   `json:"max_position_embeddings"`
		NumAttentionHeads      uint32   `json:"num_attention_heads"`
		NumKeyValueHeads       uint32   `json:"num_key_value_heads"`
		NumGlobalKeyValueHeads *uint32  `json:"num_global_key_value_heads"`
		HeadDim                uint32   `json:"head_dim"`
		GlobalHeadDim          uint32   `json:"global_head_dim"`
		IntermediateSize       uint32   `json:"intermediate_size"`
		RMSNormEps             float32  `json:"rms_norm_eps"`
		SlidingWindow          uint32   `json:"sliding_window"`
		NumKVSharedLayers      uint32   `json:"num_kv_shared_layers"`
		AttentionKEqV          bool     `json:"attention_k_eq_v"`
		FinalLogitSoftcapping  float32  `json:"final_logit_softcapping"`
		LayerTypes             []string `json:"layer_types"`
		RopeParameters         map[string]*struct {
			RopeTheta           float32  `json:"rope_theta"`
			PartialRotaryFactor *float32 `json:"partial_rotary_factor"`
		} `json:"rope_parameters"`
	} `json:"text_config"`
}

func (p *gemma4Model) DraftKV(draftFsys fs.FS) (KV, error) {
	bts, err := fs.ReadFile(draftFsys, "config.json")
	if err != nil {
		return nil, fmt.Errorf("read draft config.json: %w", err)
	}
	bts = sanitizeNonFiniteJSON(bts)

	var cfg gemma4AssistantConfig
	if err := json.Unmarshal(bts, &cfg); err != nil {
		return nil, fmt.Errorf("parse draft config.json: %w", err)
	}

	tc := cfg.TextConfig
	kv := make(KV)

	kv["draft.block_count"] = tc.NumHiddenLayers
	kv["draft.embedding_length"] = tc.HiddenSize

	if tc.NumAttentionHeads > 0 {
		kv["draft.attention.head_count"] = tc.NumAttentionHeads
	}
	if tc.HeadDim > 0 {
		kv["draft.attention.key_length_swa"] = tc.HeadDim
		kv["draft.attention.value_length_swa"] = tc.HeadDim
	}
	if tc.GlobalHeadDim > 0 {
		kv["draft.attention.key_length"] = tc.GlobalHeadDim
		kv["draft.attention.value_length"] = tc.GlobalHeadDim
	}
	if tc.RMSNormEps > 0 {
		kv["draft.attention.layer_norm_rms_epsilon"] = tc.RMSNormEps
	}
	if tc.IntermediateSize > 0 {
		kv["draft.feed_forward_length"] = tc.IntermediateSize
	}

	if len(tc.LayerTypes) > 0 {
		kv["draft.attention.sliding_window_pattern"] = slices.Collect(func(yield func(bool) bool) {
			for _, lt := range tc.LayerTypes {
				if !yield(lt == "sliding_attention") {
					break
				}
			}
		})
	}

	if rp, ok := tc.RopeParameters["full_attention"]; ok && rp != nil {
		kv["draft.rope.freq_base"] = rp.RopeTheta
		if rp.PartialRotaryFactor != nil {
			rotDims := uint32(float32(tc.GlobalHeadDim) * *rp.PartialRotaryFactor)
			kv["draft.rope.dimension_count"] = rotDims
		}
	}
	if rp, ok := tc.RopeParameters["sliding_attention"]; ok && rp != nil {
		kv["draft.rope.freq_base_swa"] = rp.RopeTheta
	}

	kv["draft.use_ordered_embeddings"] = cfg.UseOrderedEmbeddings
	if cfg.NumCentroids > 0 {
		kv["draft.num_centroids"] = cfg.NumCentroids
	}
	if cfg.CentroidIntermediateTopK > 0 {
		kv["draft.centroid_intermediate_top_k"] = cfg.CentroidIntermediateTopK
	}

	return kv, nil
}

// gemma4AssistantModel converts a Gemma 4 MTP "assistant" (drafter) checkpoint
// into a standalone GGUF with general.architecture = "gemma4_assistant". Unlike
// the qwen NextN drafter, the Gemma 4 assistant is a separate model that the
// runtime loads via --mtp-head and attaches to the target: it cross-attends the
// target's KV cache (so attention is Q-only), uses the target's last-layer
// activations via a pre/post projection, and has a centroid-routed LM head.
type gemma4AssistantModel struct {
	cfg           gemma4AssistantConfig
	nEmbdBackbone uint32 // target model n_embd; pre_projection maps from 2*this
	vocabSize     uint32
}

func (*gemma4AssistantModel) Replacements() []string {
	return []string{
		// ClippableLinear wraps nn.Linear — strip .linear. from the weight path
		".linear.weight", ".weight",
		".linear.bias", ".bias",

		// MTP head: target-activation projections, embeddings, centroid LM head
		"pre_projection", "mtp.pre_projection",
		"post_projection", "mtp.post_projection",
		"masked_embedding.centroids", "mtp.centroids",
		"masked_embedding.token_ordering", "mtp.token_ordering",
		"model.embed_tokens", "token_embd",
		"model.norm", "output_norm",
		"model.layers", "blk",

		// Attention is Q-only — the drafter cross-attends the target's KV cache,
		// so there are no k_proj/v_proj tensors.
		"input_layernorm", "attn_norm",
		"self_attn.q_proj", "attn_q",
		"self_attn.q_norm", "attn_q_norm",
		"self_attn.o_proj", "attn_output",

		// MLP
		"mlp.gate_proj", "ffn_gate",
		"mlp.down_proj", "ffn_down",
		"mlp.up_proj", "ffn_up",

		// Norms
		"post_attention_layernorm", "post_attention_norm",
		"pre_feedforward_layernorm", "ffn_norm",
		"post_feedforward_layernorm", "post_ffw_norm",

		// Per-layer output scalar
		"layer_scalar", "layer_output_scale.weight",
	}
}

func (m *gemma4AssistantModel) Tensors(ts []Tensor) []*ggml.Tensor {
	out := make([]*ggml.Tensor, 0, len(ts)+1)
	for _, t := range ts {
		shape := t.Shape()
		// layer_output_scale is a scalar in the checkpoint; emit it as 1-D.
		if len(shape) == 0 {
			shape = []uint64{1}
		}
		out = append(out, &ggml.Tensor{
			Name:     t.Name(),
			Kind:     t.Kind(),
			Shape:    shape,
			WriterTo: t,
		})
	}

	// Generate the single global rope_freqs.weight for proportional RoPE on the
	// full-attention (non-SWA) layers, exactly like the base gemma4 converter.
	// The gemma4_assistant arch creates blk.%d.rope_freqs per non-SWA layer, but
	// the tensor-name template "rope_freqs" has no %d, so every layer resolves to
	// the same name "rope_freqs.weight"; the first non-SWA layer loads it and the
	// rest reuse it (TENSOR_DUPLICATED). So the file needs exactly one copy.
	tc := m.cfg.TextConfig
	if tc.GlobalHeadDim > 0 {
		globalFreqsSize := tc.GlobalHeadDim / 2

		partialRotaryFactor := float32(0.25)
		if rp, ok := tc.RopeParameters["full_attention"]; ok && rp != nil && rp.PartialRotaryFactor != nil {
			partialRotaryFactor = *rp.PartialRotaryFactor
		}
		nRotFull := int(float32(tc.GlobalHeadDim) * partialRotaryFactor / 2)

		freqs := make(ropeFactor, globalFreqsSize)
		for j := range freqs {
			if j < nRotFull {
				freqs[j] = 1.0
			} else {
				freqs[j] = 1e30 // effectively disable rotation past the rotated portion
			}
		}
		out = append(out, &ggml.Tensor{
			Name:     "rope_freqs.weight",
			Kind:     0, // F32
			Shape:    []uint64{uint64(len(freqs))},
			WriterTo: freqs,
		})
	}

	return out
}

func (m *gemma4AssistantModel) KV(baseKV ggml.KV) ggml.KV {
	tc := m.cfg.TextConfig
	kv := ggml.KV{}

	// Copy the tokenizer (and quantization provenance) verbatim from the base
	// model. The runtime compares the assistant's vocab text against the target's
	// (llama_mtp_vocab_matches), so the standalone assistant GGUF must carry the
	// full tokenizer. A fresh KV (rather than maps.Clone) avoids dragging in stale
	// gemma4.* hparams that don't apply to the assistant.
	for k, v := range baseKV {
		if strings.HasPrefix(k, "tokenizer.") {
			kv[k] = v
		}
	}
	if v, ok := baseKV["general.quantization_version"]; ok {
		kv["general.quantization_version"] = v
	}
	kv["general.architecture"] = "gemma4_assistant"
	kv["general.file_type"] = uint32(1) // F16; re-quantized by createModel if requested

	const p = "gemma4_assistant."
	kv[p+"vocab_size"] = m.vocabSize
	kv[p+"block_count"] = tc.NumHiddenLayers
	kv[p+"embedding_length"] = tc.HiddenSize
	kv[p+"context_length"] = tc.MaxPositionEmbeddings
	if tc.IntermediateSize > 0 {
		kv[p+"feed_forward_length"] = tc.IntermediateSize
	}
	kv[p+"attention.head_count"] = tc.NumAttentionHeads
	// Per-layer KV head count: SWA layers use num_key_value_heads, global
	// (full-attention) layers use num_global_key_value_heads. Gemma 4 uses a
	// different KV head count for the two attention types, so emit a per-layer
	// array (mirrors the base gemma4 target converter) rather than a scalar.
	if tc.NumGlobalKeyValueHeads != nil && *tc.NumGlobalKeyValueHeads != tc.NumKeyValueHeads && len(tc.LayerTypes) > 0 {
		kvHeads := make([]int32, len(tc.LayerTypes))
		for i, lt := range tc.LayerTypes {
			if lt == "sliding_attention" {
				kvHeads[i] = int32(tc.NumKeyValueHeads)
			} else {
				kvHeads[i] = int32(*tc.NumGlobalKeyValueHeads)
			}
		}
		kv[p+"attention.head_count_kv"] = kvHeads
	} else {
		kv[p+"attention.head_count_kv"] = tc.NumKeyValueHeads
	}
	kv[p+"attention.key_length"] = tc.GlobalHeadDim
	kv[p+"attention.value_length"] = tc.GlobalHeadDim
	kv[p+"attention.key_length_swa"] = tc.HeadDim
	kv[p+"attention.value_length_swa"] = tc.HeadDim
	kv[p+"attention.layer_norm_rms_epsilon"] = tc.RMSNormEps
	// sliding_window is read as a required key by the gemma4_assistant loader
	// (src/models/gemma4-assistant.cpp), so always emit it.
	kv[p+"attention.sliding_window"] = tc.SlidingWindow
	if tc.NumKVSharedLayers > 0 {
		kv[p+"attention.shared_kv_layers"] = tc.NumKVSharedLayers
	}
	kv[p+"attention.k_eq_v"] = tc.AttentionKEqV

	// Per-layer sliding-window pattern (len == n_layer); required by the loader.
	if len(tc.LayerTypes) > 0 {
		pattern := make([]bool, len(tc.LayerTypes))
		for i, lt := range tc.LayerTypes {
			pattern[i] = lt == "sliding_attention"
		}
		kv[p+"attention.sliding_window_pattern"] = pattern
	}

	if rp, ok := tc.RopeParameters["full_attention"]; ok && rp != nil {
		kv[p+"rope.freq_base"] = rp.RopeTheta
		kv[p+"rope.dimension_count"] = tc.GlobalHeadDim
	}
	if rp, ok := tc.RopeParameters["sliding_attention"]; ok && rp != nil {
		kv[p+"rope.freq_base_swa"] = rp.RopeTheta
		// n_rot for SWA layers defaults to n_rot_full (the global head dim) in the
		// loader (llama-model.cpp); the SWA q heads are head_dim-wide, so without
		// this the drafter would rope global_head_dim dims on head_dim-wide heads.
		kv[p+"rope.dimension_count_swa"] = tc.HeadDim
	}

	if tc.FinalLogitSoftcapping > 0 {
		kv[p+"final_logit_softcapping"] = tc.FinalLogitSoftcapping
	}

	// MTP assistant-specific metadata.
	kv[p+"n_embd_backbone"] = m.nEmbdBackbone
	kv[p+"use_ordered_embeddings"] = m.cfg.UseOrderedEmbeddings
	if m.cfg.NumCentroids > 0 {
		kv[p+"n_centroids"] = m.cfg.NumCentroids
	}
	if m.cfg.CentroidIntermediateTopK > 0 {
		kv[p+"centroid_top_k"] = m.cfg.CentroidIntermediateTopK
	}
	if m.cfg.RequiresTargetArch != "" {
		kv[p+"requires_target_arch"] = m.cfg.RequiresTargetArch
	}

	return kv
}

func (m *gemma4AssistantModel) validateTensors(tensors []*ggml.Tensor) error {
	present := make(map[string]bool, len(tensors))
	for _, t := range tensors {
		present[t.Name] = true
	}

	required := []string{
		"token_embd.weight",
		"mtp.pre_projection.weight",
		"mtp.post_projection.weight",
		"output_norm.weight",
	}
	for i := range int(m.cfg.TextConfig.NumHiddenLayers) {
		for _, suffix := range []string{
			"attn_norm.weight", "attn_q.weight", "attn_output.weight",
			"attn_q_norm.weight", "post_attention_norm.weight",
			"ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight",
			"ffn_down.weight", "post_ffw_norm.weight",
		} {
			required = append(required, fmt.Sprintf("blk.%d.%s", i, suffix))
		}
	}
	if m.cfg.UseOrderedEmbeddings {
		required = append(required, "mtp.centroids.weight")
	}

	var missing []string
	for _, name := range required {
		if !present[name] {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("gemma4 assistant draft is missing required tensors: %v", missing)
	}
	return nil
}

// ConvertGemma4MTPDraft converts a Gemma 4 MTP assistant (drafter) safetensors
// checkpoint into a standalone GGUF with general.architecture = "gemma4_assistant".
// baseKV is the already-converted target model's KV; it supplies the tokenizer
// (the runtime requires the assistant's vocab to match the target's) and
// n_embd_backbone (the target's embedding length). The output contains only the
// assistant's own tensors — it is loaded separately via --mtp-head, not embedded
// into the target GGUF.
func ConvertGemma4MTPDraft(fsys fs.FS, f *os.File, baseKV ggml.KV) error {
	if arch := baseKV.Architecture(); arch != "gemma4" {
		return fmt.Errorf("gemma4 MTP draft requires a gemma4 base model, got %q", arch)
	}

	nEmbdBackbone := baseKV.Uint("embedding_length")
	if nEmbdBackbone == 0 {
		return fmt.Errorf("gemma4 MTP draft requires a base model with embedding_length")
	}

	tokens := baseKV.Strings("tokenizer.ggml.tokens")
	if len(tokens) == 0 {
		return fmt.Errorf("gemma4 MTP draft requires a base model with a tokenizer (tokenizer.ggml.tokens)")
	}

	bts, err := fs.ReadFile(fsys, "config.json")
	if err != nil {
		return fmt.Errorf("read draft config.json: %w", err)
	}
	bts = sanitizeNonFiniteJSON(bts)

	var cfg gemma4AssistantConfig
	if err := json.Unmarshal(bts, &cfg); err != nil {
		return fmt.Errorf("parse draft config.json: %w", err)
	}
	if cfg.TextConfig.NumHiddenLayers == 0 {
		return fmt.Errorf("gemma4 MTP draft config is missing text_config.num_hidden_layers")
	}

	m := &gemma4AssistantModel{
		cfg:           cfg,
		nEmbdBackbone: nEmbdBackbone,
		vocabSize:     uint32(len(tokens)),
	}

	ts, cleanup, err := parseTensors(fsys, strings.NewReplacer(m.Replacements()...))
	if cleanup != nil {
		defer cleanup()
	}
	if err != nil {
		return err
	}
	if err := ensureUniqueTensorNames(ts); err != nil {
		return err
	}

	tensors := m.Tensors(ts)
	if len(tensors) == 0 {
		return fmt.Errorf("gemma4 MTP draft safetensors produced no GGUF tensors")
	}
	for _, t := range tensors {
		t.Shape = slices.Clone(t.Shape)
		slices.Reverse(t.Shape)
	}

	if err := m.validateTensors(tensors); err != nil {
		return err
	}

	kv := m.KV(baseKV)

	var parameters uint64
	for _, t := range tensors {
		parameters += t.Elements()
	}
	kv["general.parameter_count"] = parameters

	return ggml.WriteGGUF(f, kv, tensors)
}
