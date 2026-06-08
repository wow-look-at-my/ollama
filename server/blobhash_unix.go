//go:build unix

package server

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"syscall"
)

// sha256FileMmap computes the "sha256:<hex>" digest of a file by memory-mapping
// it read-only and hashing the mapped bytes. Memory mapping lets this run
// concurrently with conversion (which mmaps the same files) while sharing the OS
// page cache, so the bytes are read from disk once. Falls back to a streaming
// read if the file cannot be mapped (e.g. zero length or an unmappable fs).
func sha256FileMmap(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return "", err
	}

	h := sha256.New()
	if size := fi.Size(); size > 0 {
		data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_PRIVATE)
		if err != nil {
			if _, err := io.Copy(h, f); err != nil {
				return "", err
			}
			return fmt.Sprintf("sha256:%x", h.Sum(nil)), nil
		}
		defer syscall.Munmap(data)

		// Hash in chunks so pages fault in progressively rather than all at once.
		const chunk = 8 << 20
		for off := 0; off < len(data); off += chunk {
			end := min(off+chunk, len(data))
			h.Write(data[off:end])
		}
	}

	return fmt.Sprintf("sha256:%x", h.Sum(nil)), nil
}
