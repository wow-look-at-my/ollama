package server

import (
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/types/model"
	xserver "github.com/ollama/ollama/x/server"
)

// cloneFamilies returns a copy of families that is never nil.
//
// api.ModelDetails.Families marshals to JSON, and a nil slice marshals to
// `null`. Stock ollama always emits a populated array, so clients (and our own
// simple-llm-ui) expect `"families": [...]` or at worst `[]` — never `null`.
// The common `append([]string(nil), s...)` idiom returns nil when s is empty,
// which is exactly how the safetensors path used to leak `"families": null`, so
// every place that copies the slice for serialization routes through here.
func cloneFamilies(families []string) []string {
	out := make([]string, len(families))
	copy(out, families)
	return out
}

// finalizeModelDetails fills in family/parameter metadata that the safetensors
// create path leaves empty in the model config blob, and guarantees a non-nil
// Families slice.
//
// GGUF models get model_family/model_families/model_type written into their
// config blob at create time (derived from GGUF KV metadata in createModel).
// The experimental safetensors create path (x/create) only records the format,
// so Family/Families/ParameterSize come back empty here. Rather than emit a
// broken details block, derive them from the model's HuggingFace config.json the
// same way the GGUF path derives them from GGUF KV metadata, so /api/tags,
// /api/show and /api/ps never report a safetensors model with an empty family or
// a null families array.
//
// For non-safetensors models this only mirrors a known Family into Families (if
// needed) and applies the nil-slice guard — it performs no extra I/O.
func finalizeModelDetails(name model.Name, details *api.ModelDetails) {
	// A known family with an empty list: mirror it. Cheap, and lets us skip the
	// config.json read below when the slice was the only thing missing.
	if len(details.Families) == 0 && details.Family != "" {
		details.Families = []string{details.Family}
	}

	if details.Format == "safetensors" &&
		(details.Family == "" || len(details.Families) == 0 || details.ParameterSize == "") {
		if info, err := xserver.GetSafetensorsLLMInfo(name); err == nil {
			if arch, ok := info["general.architecture"].(string); ok && arch != "" {
				if details.Family == "" {
					details.Family = arch
				}
				if len(details.Families) == 0 {
					details.Families = []string{arch}
				}
			}
			if details.ParameterSize == "" {
				if paramCount, ok := info["general.parameter_count"].(int64); ok && paramCount > 0 {
					details.ParameterSize = format.HumanNumber(uint64(paramCount))
				}
			}
		}
	}

	// Safety net: Families must never be nil (it marshals to JSON null). This
	// catches models where the architecture genuinely can't be derived.
	if details.Families == nil {
		details.Families = []string{}
	}
}
