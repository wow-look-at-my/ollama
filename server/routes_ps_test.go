package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

// psServerWith builds a Server whose scheduler has the given runners loaded.
func psServerWith(runners map[string]*runnerRef) *Server {
	return &Server{sched: &Scheduler{loaded: runners}}
}

func callPs(t *testing.T, s *Server, query string) api.ProcessResponse {
	t.Helper()
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)
	ctx.Request = httptest.NewRequest(http.MethodGet, "/api/ps"+query, nil)

	s.PsHandler(ctx)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}
	var resp api.ProcessResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode /api/ps: %v (body %s)", err, w.Body.String())
	}
	return resp
}

func residentRunner() *runnerRef {
	return &runnerRef{
		model:           &Model{ShortName: "resident:latest"},
		llama:           &mockLlm{totalSize: 5 << 30, vramSize: 5 << 30, contextLength: 4096, loadProgress: 1},
		sessionDuration: time.Minute,
		expiresAt:       time.Now().Add(time.Minute),
	}
}

func loadingRunner() *runnerRef {
	r := &runnerRef{
		model:           &Model{ShortName: "loading:latest"},
		llama:           &mockLlm{totalSize: 5 << 30, loadProgress: 0.42, contextLength: 4096},
		sessionDuration: time.Minute,
	}
	r.loading.Store(true)
	return r
}

// findByLoading returns the single entry whose Loading flag matches want.
func findByLoading(t *testing.T, models []api.ProcessModelResponse, want bool) api.ProcessModelResponse {
	t.Helper()
	var out []api.ProcessModelResponse
	for _, m := range models {
		if m.Loading == want {
			out = append(out, m)
		}
	}
	if len(out) != 1 {
		t.Fatalf("expected exactly one model with loading=%v, got %d (%+v)", want, len(out), models)
	}
	return out[0]
}

func TestPsHandlerExcludesLoadingByDefault(t *testing.T) {
	s := psServerWith(map[string]*runnerRef{
		"resident": residentRunner(),
		"loading":  loadingRunner(),
	})

	// Default /api/ps preserves the resident-only contract: the loading model
	// is omitted, and the resident entry carries neither loading nor progress.
	resp := callPs(t, s, "")
	if len(resp.Models) != 1 {
		t.Fatalf("default /api/ps returned %d models, want 1 (resident only): %+v", len(resp.Models), resp.Models)
	}
	m := resp.Models[0]
	if m.Loading {
		t.Fatalf("resident model reported loading=true")
	}
	if m.Progress != 0 {
		t.Fatalf("resident model reported progress=%v, want 0", m.Progress)
	}
}

func TestPsHandlerIncludeLoadingReportsProgress(t *testing.T) {
	s := psServerWith(map[string]*runnerRef{
		"resident": residentRunner(),
		"loading":  loadingRunner(),
	})

	resp := callPs(t, s, "?include=loading")
	if len(resp.Models) != 2 {
		t.Fatalf("/api/ps?include=loading returned %d models, want 2: %+v", len(resp.Models), resp.Models)
	}

	loading := findByLoading(t, resp.Models, true)
	if loading.Progress < 0.41 || loading.Progress > 0.43 {
		t.Fatalf("loading model progress = %v, want ~0.42", loading.Progress)
	}

	resident := findByLoading(t, resp.Models, false)
	if resident.Progress != 0 {
		t.Fatalf("resident model progress = %v, want 0 (omitted)", resident.Progress)
	}
}

func TestPsHandlerNoLoadingFieldWhenAllResident(t *testing.T) {
	s := psServerWith(map[string]*runnerRef{"resident": residentRunner()})

	// Even with the flag set, a fleet of fully-resident models reports no
	// loading entries and the field stays absent in the serialized JSON.
	w := httptest.NewRecorder()
	gin.SetMode(gin.TestMode)
	ctx, _ := gin.CreateTestContext(w)
	ctx.Request = httptest.NewRequest(http.MethodGet, "/api/ps?include=loading", nil)
	s.PsHandler(ctx)

	if got := w.Body.String(); contains(got, "\"loading\"") || contains(got, "\"progress\"") {
		t.Fatalf("resident-only /api/ps leaked loading/progress fields: %s", got)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
