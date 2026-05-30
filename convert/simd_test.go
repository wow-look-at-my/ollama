package convert

import (
	"math"
	"testing"

	"github.com/d4l3k/go-bfloat16"
	"github.com/x448/float16"
)

const benchSize = 1 << 20 // 1M elements

func makeF16(n int) []uint16 {
	out := make([]uint16, n)
	for i := range out {
		out[i] = float16.Fromfloat32(float32(i%1000) * 0.01).Bits()
	}
	return out
}

func makeBF16(n int) []uint16 {
	out := make([]uint16, n)
	for i := range out {
		out[i] = uint16(math.Float32bits(float32(i%1000)*0.01) >> 16)
	}
	return out
}

func makeF32(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(i%1000) * 0.01
	}
	return out
}

// --- F16 → F32 ---

func BenchmarkF16ToF32_Old(b *testing.B) {
	src := makeF16(benchSize)
	dst := make([]float32, benchSize)
	b.SetBytes(int64(benchSize * 2))
	b.ResetTimer()
	for range b.N {
		for i, v := range src {
			dst[i] = float16.Frombits(v).Float32()
		}
	}
	_ = dst
}

func BenchmarkF16ToF32_SIMD(b *testing.B) {
	src := makeF16(benchSize)
	dst := make([]float32, benchSize)
	b.SetBytes(int64(benchSize * 2))
	b.ResetTimer()
	for range b.N {
		convertF16ToF32(dst, src)
	}
	_ = dst
}

// --- F32 → F16 ---

func BenchmarkF32ToF16_Old(b *testing.B) {
	src := makeF32(benchSize)
	dst := make([]uint16, benchSize)
	b.SetBytes(int64(benchSize * 4))
	b.ResetTimer()
	for range b.N {
		for i, v := range src {
			dst[i] = float16.Fromfloat32(v).Bits()
		}
	}
	_ = dst
}

func BenchmarkF32ToF16_SIMD(b *testing.B) {
	src := makeF32(benchSize)
	dst := make([]uint16, benchSize)
	b.SetBytes(int64(benchSize * 4))
	b.ResetTimer()
	for range b.N {
		convertF32ToF16(dst, src)
	}
	_ = dst
}

// --- BF16 → F32 ---

func BenchmarkBF16ToF32_Old(b *testing.B) {
	raw := makeBF16(benchSize)
	u8s := make([]byte, benchSize*2)
	for i, v := range raw {
		u8s[i*2] = byte(v)
		u8s[i*2+1] = byte(v >> 8)
	}
	b.SetBytes(int64(benchSize * 2))
	b.ResetTimer()
	for range b.N {
		_ = bfloat16.DecodeFloat32(u8s)
	}
}

func BenchmarkBF16ToF32_SIMD(b *testing.B) {
	src := makeBF16(benchSize)
	dst := make([]float32, benchSize)
	b.SetBytes(int64(benchSize * 2))
	b.ResetTimer()
	for range b.N {
		convertBF16ToF32(dst, src)
	}
	_ = dst
}

// --- F32 → BF16 ---

func BenchmarkF32ToBF16_Old(b *testing.B) {
	src := makeF32(benchSize)
	b.SetBytes(int64(benchSize * 4))
	b.ResetTimer()
	for range b.N {
		_ = bfloat16.EncodeFloat32(src)
	}
}

func BenchmarkF32ToBF16_SIMD(b *testing.B) {
	src := makeF32(benchSize)
	dst := make([]uint16, benchSize)
	b.SetBytes(int64(benchSize * 4))
	b.ResetTimer()
	for range b.N {
		convertF32ToBF16(dst, src)
	}
	_ = dst
}
