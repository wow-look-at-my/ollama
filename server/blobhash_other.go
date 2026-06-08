//go:build !unix

package server

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
)

// sha256FileMmap computes the "sha256:<hex>" digest of a file. On non-unix
// platforms it falls back to a streaming read (no mmap). It still runs
// concurrently with conversion, so the hash overlaps the quantize pass.
func sha256FileMmap(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return fmt.Sprintf("sha256:%x", h.Sum(nil)), nil
}
