//go:build ggmlbench

package ggmlbench

import "testing"

// benchElems and makeF32Data mirror the parent convert package's benchmark
// input exactly so the Go and C kernels quantize identical data.
const benchElems = 1 << 20

func makeF32Data(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(i%2000-1000) * 0.001
	}
	return out
}

func benchmarkGGMLQuantize(b *testing.B, qk, blockBytes int, fn func([]float32, []byte)) {
	n := (benchElems / qk) * qk
	src := makeF32Data(n)
	dst := make([]byte, n/qk*blockBytes)
	b.SetBytes(int64(n * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		fn(src, dst)
	}
}

func BenchmarkGGMLQuantizeQ8_0(b *testing.B) { benchmarkGGMLQuantize(b, 32, 34, ggmlQuantizeQ8_0) }
func BenchmarkGGMLQuantizeQ4_K(b *testing.B) { benchmarkGGMLQuantize(b, 256, 144, ggmlQuantizeQ4_K) }
func BenchmarkGGMLQuantizeQ6_K(b *testing.B) { benchmarkGGMLQuantize(b, 256, 210, ggmlQuantizeQ6_K) }
