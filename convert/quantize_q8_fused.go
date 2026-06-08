package convert

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"

	"github.com/ollama/ollama/fs/ggml"
)

const kindQ8_0 = 8

// fusedQuantizer decodes a safetensor to float32 (via mmap when available) and
// quantizes it to its target ggml kind in one pass, writing the block-encoded
// bytes straight into the GGUF.
type fusedQuantizer struct {
	src  safetensor
	kind uint32
}

func (q fusedQuantizer) WriteTo(w io.Writer) (int64, error) {
	var elemSize int
	switch q.src.dtype {
	case "F16", "BF16":
		elemSize = 2
	case "F32":
		elemSize = 4
	default:
		return 0, io.ErrUnexpectedEOF
	}
	nElems := int(q.src.size) / elemSize

	f32s := make([]float32, nElems)

	if q.src.mmap != nil && len(q.src.mmap.data) > 0 && q.src.offset+q.src.size <= int64(len(q.src.mmap.data)) {
		data := q.src.mmap.data[q.src.offset : q.src.offset+q.src.size]
		decodeTensorToF32(q.src.dtype, data, f32s)
	} else {
		f, err := q.src.fs.Open(q.src.path)
		if err != nil {
			return 0, err
		}
		defer f.Close()

		r, err := func() (io.Reader, error) {
			if readerAt, ok := f.(io.ReaderAt); ok {
				return io.NewSectionReader(readerAt, q.src.offset, q.src.size), nil
			} else if seeker, ok := f.(io.Seeker); ok {
				_, err := seeker.Seek(q.src.offset, io.SeekStart)
				return f, err
			} else {
				_, err := io.CopyN(io.Discard, f, q.src.offset)
				return f, err
			}
		}()
		if err != nil {
			return 0, err
		}

		br := bufio.NewReaderSize(r, 128<<10)
		switch q.src.dtype {
		case "F16":
			u16s := make([]uint16, nElems)
			if err := binary.Read(br, binary.LittleEndian, u16s); err != nil {
				return 0, err
			}
			convertF16ToF32(f32s, u16s)
		case "BF16":
			u16s := make([]uint16, nElems)
			if err := binary.Read(br, binary.LittleEndian, u16s); err != nil {
				return 0, err
			}
			convertBF16ToF32(f32s, u16s)
		case "F32":
			if err := binary.Read(br, binary.LittleEndian, f32s); err != nil {
				return 0, err
			}
		}
	}

	var out []byte
	switch q.kind {
	case kindQ8_0:
		out = quantizeQ8_0(f32s)
	case kindQ4_K:
		out = quantizeQ4_K(f32s)
	case kindQ6_K:
		out = quantizeQ6_K(f32s)
	default:
		return 0, fmt.Errorf("fused quantizer: unsupported kind %d", q.kind)
	}
	n, err := w.Write(out)
	return int64(n), err
}

func decodeTensorToF32(dtype string, data []byte, dst []float32) {
	switch dtype {
	case "F16":
		u16s := make([]uint16, len(dst))
		for i := range u16s {
			u16s[i] = binary.LittleEndian.Uint16(data[i*2:])
		}
		convertF16ToF32(dst, u16s)
	case "BF16":
		u16s := make([]uint16, len(dst))
		for i := range u16s {
			u16s[i] = binary.LittleEndian.Uint16(data[i*2:])
		}
		convertBF16ToF32(dst, u16s)
	case "F32":
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
	}
}

// ConvertModelQ8_0 is retained for callers/tests that want a direct Q8_0
// conversion; it routes through the general fused quantizer.
func ConvertModelQ8_0(fsys fs.FS, f *os.File) error {
	return ConvertModelQuantized(fsys, f, ggml.FileTypeQ8_0, nil)
}
