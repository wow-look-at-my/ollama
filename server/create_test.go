package server

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
)

func TestConvertFromSafetensors(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	// Helper function to create a new layer and return its digest
	makeTemp := func(content string) string {
		l, err := manifest.NewLayer(strings.NewReader(content), "application/octet-stream")
		if err != nil {
			t.Fatalf("Failed to create layer: %v", err)
		}
		return l.Digest
	}

	// Create a safetensors compatible file with empty JSON content
	var buf bytes.Buffer
	headerSize := int64(len("{}"))
	binary.Write(&buf, binary.LittleEndian, headerSize)
	buf.WriteString("{}")

	model := makeTemp(buf.String())
	config := makeTemp(`{
		"architectures": ["LlamaForCausalLM"], 
		"vocab_size": 32000
	}`)
	tokenizer := makeTemp(`{
		"version": "1.0",
		"truncation": null,
		"padding": null,
		"added_tokens": [
			{
				"id": 0,
				"content": "<|endoftext|>",
				"single_word": false,
				"lstrip": false,
				"rstrip": false,
				"normalized": false,
				"special": true
			}
		]
	}`)

	tests := []struct {
		name     string
		filePath string
		wantErr  error
	}{
		// Invalid
		{
			name:     "InvalidRelativePathShallow",
			filePath: filepath.Join("..", "file.safetensors"),
			wantErr:  errFilePath,
		},
		{
			name:     "InvalidRelativePathDeep",
			filePath: filepath.Join("..", "..", "..", "..", "..", "..", "data", "file.txt"),
			wantErr:  errFilePath,
		},
		{
			name:     "InvalidNestedPath",
			filePath: filepath.Join("dir", "..", "..", "..", "..", "..", "other.safetensors"),
			wantErr:  errFilePath,
		},
		{
			name:     "AbsolutePathOutsideRoot",
			filePath: filepath.Join(os.TempDir(), "model.safetensors"),
			wantErr:  errFilePath, // Should fail since it's outside tmpDir
		},
		{
			name:     "ValidRelativePath",
			filePath: "model.safetensors",
			wantErr:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create the minimum required file map for convertFromSafetensors
			files := map[string]string{
				tt.filePath:      model,
				"config.json":    config,
				"tokenizer.json": tokenizer,
			}

			_, err := convertFromSafetensors(files, nil, false, "", true, "", false, func(resp api.ProgressResponse) {})

			if (tt.wantErr == nil && err != nil) ||
				(tt.wantErr != nil && err == nil) ||
				(tt.wantErr != nil && !errors.Is(err, tt.wantErr)) {
				t.Errorf("convertFromSafetensors() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestSHA256FileMmap(t *testing.T) {
	dir := t.TempDir()
	// cover empty, small, and sizes around the 8 MiB hashing chunk boundary
	for _, size := range []int{0, 1, 4096, 8<<20 - 1, 8 << 20, 8<<20 + 12345} {
		data := make([]byte, size)
		for i := range data {
			data[i] = byte(i*31 + 7)
		}
		p := filepath.Join(dir, fmt.Sprintf("f-%d", size))
		if err := os.WriteFile(p, data, 0o644); err != nil {
			t.Fatal(err)
		}

		got, err := sha256FileMmap(p)
		if err != nil {
			t.Fatalf("size %d: %v", size, err)
		}
		want := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
		if got != want {
			t.Errorf("size %d: got %s, want %s", size, got, want)
		}
	}
}

// TestConvertFromSafetensorsSourceMode exercises the local create source mode:
// inputs are read straight from a source directory (not the blob store), hashed
// concurrently with conversion, and staged into the blob store afterwards.
func TestConvertFromSafetensorsSourceMode(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	srcDir := t.TempDir()
	writeSrc := func(name, content string) string {
		p := filepath.Join(srcDir, name)
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
		return p
	}

	// minimal empty safetensors (8-byte little-endian header length + "{}")
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, int64(len("{}")))
	buf.WriteString("{}")
	model := buf.String()

	files := map[string]string{
		"model.safetensors": writeSrc("model.safetensors", model),
		"config.json":       writeSrc("config.json", `{"architectures": ["LlamaForCausalLM"], "vocab_size": 32000}`),
		"tokenizer.json": writeSrc("tokenizer.json", `{
			"version": "1.0", "truncation": null, "padding": null,
			"added_tokens": [{"id": 0, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}]
		}`),
	}

	if _, err := convertFromSafetensors(files, nil, false, "", true, "", true, func(api.ProgressResponse) {}); err != nil {
		t.Fatalf("source-mode convert: %v", err)
	}

	// every input should have been hashed and staged into the blob store under
	// its true digest, leaving the source files untouched.
	for name, src := range files {
		data, err := os.ReadFile(src)
		if err != nil {
			t.Fatal(err)
		}
		digest := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
		blob, err := manifest.BlobsPath(digest)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := os.Stat(blob); err != nil {
			t.Errorf("%s: input not staged at %s: %v", name, digest, err)
		}
	}
}

func TestRemoteURL(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
		hasError bool
	}{
		{
			name:     "absolute path",
			input:    "/foo/bar",
			expected: "http://localhost:11434/foo/bar",
			hasError: false,
		},
		{
			name:     "absolute path with cleanup",
			input:    "/foo/../bar",
			expected: "http://localhost:11434/bar",
			hasError: false,
		},
		{
			name:     "root path",
			input:    "/",
			expected: "http://localhost:11434/",
			hasError: false,
		},
		{
			name:     "host without scheme",
			input:    "example.com",
			expected: "http://example.com:11434",
			hasError: false,
		},
		{
			name:     "host with port",
			input:    "example.com:8080",
			expected: "http://example.com:8080",
			hasError: false,
		},
		{
			name:     "full URL",
			input:    "https://example.com:8080/path",
			expected: "https://example.com:8080/path",
			hasError: false,
		},
		{
			name:     "full URL with path cleanup",
			input:    "https://example.com:8080/path/../other",
			expected: "https://example.com:8080/other",
			hasError: false,
		},
		{
			name:     "ollama.com special case",
			input:    "ollama.com",
			expected: "https://ollama.com:443",
			hasError: false,
		},
		{
			name:     "http ollama.com special case",
			input:    "http://ollama.com",
			expected: "https://ollama.com:443",
			hasError: false,
		},
		{
			name:     "URL with only host",
			input:    "http://example.com",
			expected: "http://example.com:11434",
			hasError: false,
		},
		{
			name:     "URL with root path cleaned",
			input:    "http://example.com/",
			expected: "http://example.com:11434",
			hasError: false,
		},
		{
			name:     "invalid URL",
			input:    "http://[::1]:namedport", // invalid port
			expected: "",
			hasError: true,
		},
		{
			name:     "empty string",
			input:    "",
			expected: "http://localhost:11434",
			hasError: false,
		},
		{
			name:     "host with scheme but no port",
			input:    "http://localhost",
			expected: "http://localhost:11434",
			hasError: false,
		},
		{
			name:     "complex path cleanup",
			input:    "/a/b/../../c/./d",
			expected: "http://localhost:11434/c/d",
			hasError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := remoteURL(tt.input)

			if tt.hasError {
				if err == nil {
					t.Errorf("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestRemoteURL_Idempotent(t *testing.T) {
	// Test that applying remoteURL twice gives the same result as applying it once
	testInputs := []string{
		"/foo/bar",
		"example.com",
		"https://example.com:8080/path",
		"ollama.com",
		"http://localhost:11434",
	}

	for _, input := range testInputs {
		t.Run(input, func(t *testing.T) {
			firstResult, err := remoteURL(input)
			if err != nil {
				t.Fatalf("first call failed: %v", err)
			}

			secondResult, err := remoteURL(firstResult)
			if err != nil {
				t.Fatalf("second call failed: %v", err)
			}

			if firstResult != secondResult {
				t.Errorf("function is not idempotent: first=%q, second=%q", firstResult, secondResult)
			}
		})
	}
}

func TestSetTemplate(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	t.Run("valid template", func(t *testing.T) {
		layers, err := setTemplate(nil, "{{ .Prompt }}")
		if err != nil {
			t.Fatalf("setTemplate returned error for valid template: %v", err)
		}

		if len(layers) != 1 {
			t.Fatalf("expected 1 layer, got %d", len(layers))
		}

		if got, want := layers[0].MediaType, "application/vnd.ollama.image.template"; got != want {
			t.Fatalf("unexpected media type: got %q, want %q", got, want)
		}
	})

	t.Run("invalid template", func(t *testing.T) {
		_, err := setTemplate(nil, "{{ if .Prompt }}")
		if err == nil {
			t.Fatal("expected error for invalid template, got nil")
		}

		if !errors.Is(err, errBadTemplate) {
			t.Fatalf("expected errBadTemplate, got %v", err)
		}
	})
}
