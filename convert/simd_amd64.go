//go:build amd64

package convert

import (
	"math"

	"github.com/x448/float16"
	"golang.org/x/sys/cpu"
)

var useAVX2 = cpu.X86.HasAVX2

//go:noescape
func f16ToF32AVX(dst *float32, src *uint16, chunks int)

//go:noescape
func f32ToF16AVX(dst *uint16, src *float32, chunks int)

//go:noescape
func bf16ToF32AVX(dst *float32, src *uint16, chunks int)

//go:noescape
func f32ToBF16AVX(dst *uint16, src *float32, chunks int)

//go:noescape
func qxProductsAVX(x *float32, n int, iscale float32, nmaxm1, negnmax int32, prodlx, prodl2 *float32)

// qxSums computes the make_qx_quants reduction sums for one group. The AVX2 path
// vectorizes only the element-wise products and sums them sequentially, so the
// result is bit-identical to the scalar kernel (and thus to ggml). The fixed
// buffer caps the AVX2 path at 16 elements, which covers q6_K's group size;
// other sizes fall through to the scalar loop.
func qxSums(x []float32, iscale float32, nmax int) (sumlx, suml2 float32) {
	n := len(x)
	i := 0
	if useAVX2 && n >= 8 && n <= 16 {
		var plx, pl2 [16]float32
		bulk := (n / 8) * 8
		qxProductsAVX(&x[0], bulk, iscale, int32(nmax-1), int32(-nmax), &plx[0], &pl2[0])
		for ; i < bulk; i++ {
			sumlx += plx[i]
			suml2 += pl2[i]
		}
	}
	for ; i < n; i++ {
		l := max(-nmax, min(nmax-1, nearestInt(iscale*x[i])))
		w := x[i] * x[i]
		fl := float32(l)
		sumlx += w * x[i] * fl
		suml2 += w * fl * fl
	}
	return sumlx, suml2
}

func convertF16ToF32(dst []float32, src []uint16) {
	n := len(src)
	if n == 0 {
		return
	}
	bulk := 0
	if useAVX2 && n >= 8 {
		bulk = (n / 8) * 8
		f16ToF32AVX(&dst[0], &src[0], n/8)
	}
	for i := bulk; i < n; i++ {
		dst[i] = float16.Frombits(src[i]).Float32()
	}
}

func convertF32ToF16(dst []uint16, src []float32) {
	n := len(src)
	if n == 0 {
		return
	}
	bulk := 0
	if useAVX2 && n >= 8 {
		bulk = (n / 8) * 8
		f32ToF16AVX(&dst[0], &src[0], n/8)
	}
	for i := bulk; i < n; i++ {
		dst[i] = float16.Fromfloat32(src[i]).Bits()
	}
}

func convertBF16ToF32(dst []float32, src []uint16) {
	n := len(src)
	if n == 0 {
		return
	}
	bulk := 0
	if useAVX2 && n >= 8 {
		bulk = (n / 8) * 8
		bf16ToF32AVX(&dst[0], &src[0], n/8)
	}
	for i := bulk; i < n; i++ {
		dst[i] = math.Float32frombits(uint32(src[i]) << 16)
	}
}

func convertF32ToBF16(dst []uint16, src []float32) {
	n := len(src)
	if n == 0 {
		return
	}
	bulk := 0
	if useAVX2 && n >= 8 {
		bulk = (n / 8) * 8
		f32ToBF16AVX(&dst[0], &src[0], n/8)
	}
	for i := bulk; i < n; i++ {
		dst[i] = uint16(math.Float32bits(src[i]) >> 16)
	}
}
