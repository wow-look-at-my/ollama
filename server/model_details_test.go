package server

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestCloneFamilies(t *testing.T) {
	// The whole point of this helper is that the result never marshals to null.
	cases := map[string][]string{
		"nil":       nil,
		"empty":     {},
		"populated": {"gemma4"},
		"multiple":  {"gemma4", "clip"},
	}

	for name, in := range cases {
		t.Run(name, func(t *testing.T) {
			got := cloneFamilies(in)
			if got == nil {
				t.Fatalf("cloneFamilies(%v) returned nil; must always be non-nil", in)
			}
			if len(got) != len(in) {
				t.Fatalf("cloneFamilies(%v) = %v, want len %d", in, got, len(in))
			}

			b, err := json.Marshal(got)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if string(b) == "null" {
				t.Fatalf("cloneFamilies(%v) marshalled to null", in)
			}
		})
	}

	// A clone must not alias the input.
	in := []string{"gemma4"}
	got := cloneFamilies(in)
	got[0] = "mutated"
	if in[0] != "gemma4" {
		t.Fatalf("cloneFamilies aliased its input: %v", in)
	}
}

func TestFinalizeModelDetailsNonSafetensors(t *testing.T) {
	t.Run("gguf is left untouched", func(t *testing.T) {
		details := api.ModelDetails{
			Format:        "gguf",
			Family:        "gemma4",
			Families:      []string{"gemma4"},
			ParameterSize: "31B",
		}
		finalizeModelDetails(model.ParseName("whatever"), &details)

		if details.Family != "gemma4" || details.ParameterSize != "31B" {
			t.Fatalf("gguf details mutated: %+v", details)
		}
		if len(details.Families) != 1 || details.Families[0] != "gemma4" {
			t.Fatalf("gguf families mutated: %+v", details.Families)
		}
	})

	t.Run("mirrors family into empty families list", func(t *testing.T) {
		details := api.ModelDetails{Format: "gguf", Family: "llama"}
		finalizeModelDetails(model.ParseName("whatever"), &details)

		if len(details.Families) != 1 || details.Families[0] != "llama" {
			t.Fatalf("families = %v, want [llama]", details.Families)
		}
	})

	t.Run("nil families becomes non-nil empty slice", func(t *testing.T) {
		details := api.ModelDetails{Format: ""}
		finalizeModelDetails(model.ParseName("whatever"), &details)

		if details.Families == nil {
			t.Fatal("families is nil; must be a non-nil empty slice")
		}
		if len(details.Families) != 0 {
			t.Fatalf("families = %v, want []", details.Families)
		}
	})
}

// writeSafetensorsModel writes a minimal safetensors model manifest (a ConfigV2
// blob plus a config.json layer and one tensor layer) to the model store, the
// same shape the experimental safetensors create path produces. The ConfigV2
// blob deliberately leaves family/parameter metadata empty (as that path does),
// which is exactly the case finalizeModelDetails has to recover from.
func writeSafetensorsModel(t *testing.T, nameStr, arch string) model.Name {
	t.Helper()
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	cfg := model.ConfigV2{
		ModelFormat:  "safetensors",
		Capabilities: []string{"completion"},
		Architecture: "amd64",
		OS:           "linux",
	}
	cfgJSON, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal config: %v", err)
	}
	configLayer, err := manifest.NewLayer(bytes.NewReader(cfgJSON), "application/vnd.docker.container.image.v1+json")
	if err != nil {
		t.Fatalf("config layer: %v", err)
	}

	// HuggingFace config.json — GetSafetensorsLLMInfo derives the architecture
	// from model_type.
	hfJSON, err := json.Marshal(map[string]any{
		"model_type":  arch,
		"hidden_size": 2560,
	})
	if err != nil {
		t.Fatalf("marshal config.json: %v", err)
	}
	configJSONLayer, err := manifest.NewLayer(bytes.NewReader(hfJSON), "application/vnd.ollama.image.json")
	if err != nil {
		t.Fatalf("config.json layer: %v", err)
	}
	configJSONLayer.Name = "config.json"

	// One tensor so the parameter count is non-zero.
	header := map[string]any{
		"model.embed_tokens.weight": map[string]any{
			"dtype":        "BF16",
			"shape":        []int64{16, 8},
			"data_offsets": []int64{0, 256},
		},
	}
	headerJSON, _ := json.Marshal(header)
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, uint64(len(headerJSON))); err != nil {
		t.Fatalf("write header size: %v", err)
	}
	buf.Write(headerJSON)
	tensorLayer, err := manifest.NewLayer(&buf, manifest.MediaTypeImageTensor)
	if err != nil {
		t.Fatalf("tensor layer: %v", err)
	}
	tensorLayer.Name = "model.embed_tokens.weight"

	name := model.ParseName(nameStr)
	if err := manifest.WriteManifest(name, configLayer, []manifest.Layer{configJSONLayer, tensorLayer}); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	return name
}

func TestFinalizeModelDetailsSafetensors(t *testing.T) {
	name := writeSafetensorsModel(t, "gemma4-st", "gemma4")

	// Start from what the config blob actually contains for a safetensors model:
	// the format, but empty family/families/parameter size.
	details := api.ModelDetails{Format: "safetensors"}
	finalizeModelDetails(name, &details)

	if details.Family != "gemma4" {
		t.Errorf("Family = %q, want gemma4", details.Family)
	}
	if len(details.Families) != 1 || details.Families[0] != "gemma4" {
		t.Errorf("Families = %v, want [gemma4]", details.Families)
	}
	if details.ParameterSize == "" {
		t.Errorf("ParameterSize is empty; want it derived from the tensors")
	}
}

// TestBuildModelListSummarySafetensors exercises the actual /api/tags code path
// end to end: build the summary, render the list response, and confirm the JSON
// never contains "families":null for a safetensors model.
func TestBuildModelListSummarySafetensors(t *testing.T) {
	name := writeSafetensorsModel(t, "gemma4-st-list", "gemma4")

	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		t.Fatalf("parse manifest: %v", err)
	}

	summary, err := buildModelListSummary(name, mf)
	if err != nil {
		t.Fatalf("buildModelListSummary: %v", err)
	}

	resp := summary.ListModelResponse()
	if resp.Details.Family != "gemma4" {
		t.Errorf("Family = %q, want gemma4", resp.Details.Family)
	}
	if len(resp.Details.Families) != 1 || resp.Details.Families[0] != "gemma4" {
		t.Errorf("Families = %v, want [gemma4]", resp.Details.Families)
	}
	if resp.Details.ParameterSize == "" {
		t.Error("ParameterSize is empty; want it derived from config.json/tensors")
	}

	b, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(b), `"families":null`) {
		t.Fatalf("/api/tags response contains families:null: %s", b)
	}
}

// TestGetModelInfoSafetensors drives the actual /api/show code path
// (GetModelInfo) for a safetensors model and confirms the details block is no
// longer broken: family/families/parameter_size populated, families never null.
func TestGetModelInfoSafetensors(t *testing.T) {
	name := writeSafetensorsModel(t, "gemma4-st-show", "gemma4")

	resp, err := GetModelInfo(api.ShowRequest{Model: name.String()})
	if err != nil {
		t.Fatalf("GetModelInfo: %v", err)
	}

	if resp.Details.Format != "safetensors" {
		t.Errorf("Format = %q, want safetensors", resp.Details.Format)
	}
	if resp.Details.Family != "gemma4" {
		t.Errorf("Family = %q, want gemma4", resp.Details.Family)
	}
	if len(resp.Details.Families) != 1 || resp.Details.Families[0] != "gemma4" {
		t.Errorf("Families = %v, want [gemma4]", resp.Details.Families)
	}
	if resp.Details.ParameterSize == "" {
		t.Error("ParameterSize is empty; want it derived from config.json/tensors")
	}

	b, err := json.Marshal(resp.Details)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(b), `"families":null`) {
		t.Fatalf("/api/show details contains families:null: %s", b)
	}
}

// TestListModelResponseNeverNullFamilies guards the serialization site directly:
// even a summary built with a nil Families slice must not marshal to null.
func TestListModelResponseNeverNullFamilies(t *testing.T) {
	summary := modelListSummary{
		Details: api.ModelDetails{Format: "safetensors", Families: nil},
	}
	resp := summary.ListModelResponse()
	if resp.Details.Families == nil {
		t.Fatal("ListModelResponse produced nil Families")
	}

	b, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(b), `"families":null`) {
		t.Fatalf("families serialized to null: %s", b)
	}
}
