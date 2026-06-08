package server

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
)

// sha256File computes the "sha256:<hex>" digest of a file with a streaming read.
//
// It is called from goroutines that run concurrently with conversion, which
// memory-maps the same files for quantization. On Linux the OS page cache backs
// both the read here and the conversion's mmap, so the bytes are read from disk
// once and the hash overlaps the quantize pass. (The bytes are still processed
// twice on the CPU - hashing and quantizing are distinct passes over the data -
// but there is no redundant second memory mapping.)
func sha256File(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.CopyBuffer(h, f, make([]byte, 1<<20)); err != nil {
		return "", err
	}
	return fmt.Sprintf("sha256:%x", h.Sum(nil)), nil
}
