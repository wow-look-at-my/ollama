package llm

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"sync/atomic"
	"testing"
)

// fakeHealthServer serves /health with a body that the test can swap between
// requests, standing in for llama-server's health endpoint.
func fakeHealthServer(t *testing.T) (*llamaServerRunner, *atomic.Value) {
	t.Helper()
	var body atomic.Value
	body.Store(`{"status":"loading model"}`)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body.Load().(string))
	}))
	t.Cleanup(srv.Close)

	port := srv.Listener.Addr().(*net.TCPAddr).Port
	// getServerStatus inspects s.cmd.ProcessState (nil here -> still running).
	s := &llamaServerRunner{cmd: &exec.Cmd{}, port: port}
	return s, &body
}

func TestGetServerStatusParsesLoadProgress(t *testing.T) {
	s, body := fakeHealthServer(t)

	// A mid-load health body carries the fraction; it is cached on the runner.
	body.Store(`{"status":"loading model","progress":0.42}`)
	status, err := s.getServerStatus(context.Background())
	if err != nil {
		t.Fatalf("getServerStatus: %v", err)
	}
	if status != ServerStatusLoadingModel {
		t.Fatalf("status = %v, want %v", status, ServerStatusLoadingModel)
	}
	if got := s.LoadProgress(); got < 0.41 || got > 0.43 {
		t.Fatalf("LoadProgress = %v, want ~0.42", got)
	}

	// Once ready, progress reads as a full 1 even though "ok" carries no fraction.
	body.Store(`{"status":"ok"}`)
	status, err = s.getServerStatus(context.Background())
	if err != nil {
		t.Fatalf("getServerStatus: %v", err)
	}
	if status != ServerStatusReady {
		t.Fatalf("status = %v, want %v", status, ServerStatusReady)
	}
	if got := s.LoadProgress(); got != 1 {
		t.Fatalf("LoadProgress = %v, want 1", got)
	}
}

func TestGetServerStatusProgressAbsentIsZero(t *testing.T) {
	// An unmodified llama-server reports the loading status without a progress
	// field; the Go side must read it as 0 and not error.
	s, body := fakeHealthServer(t)
	body.Store(`{"status":"loading model"}`)

	status, err := s.getServerStatus(context.Background())
	if err != nil {
		t.Fatalf("getServerStatus: %v", err)
	}
	if status != ServerStatusLoadingModel {
		t.Fatalf("status = %v, want %v", status, ServerStatusLoadingModel)
	}
	if got := s.LoadProgress(); got != 0 {
		t.Fatalf("LoadProgress = %v, want 0", got)
	}
}

func TestSetLoadProgressClamps(t *testing.T) {
	s := &llamaServerRunner{}
	s.setLoadProgress(-0.5)
	if got := s.LoadProgress(); got != 0 {
		t.Fatalf("LoadProgress = %v, want 0", got)
	}
	s.setLoadProgress(2)
	if got := s.LoadProgress(); got != 1 {
		t.Fatalf("LoadProgress = %v, want 1", got)
	}
}
