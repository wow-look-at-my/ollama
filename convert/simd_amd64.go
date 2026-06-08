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

//go:noescape
func qkxProductsAVX(x, w *float32, n int, iscale, mn float32, nmax int32, pL, pL2, pXL *float32, lout *int32)

// qkxSums computes the make_qkx2_quants weighted reduction sums for one group
// and fills Laux. Like qxSums, the AVX2 path vectorizes only the element-wise
// products and sums sequentially, so the result is bit-identical to the scalar
// kernel. The buffers cap the AVX2 path at 32 elements (q4_K's group size).
func qkxSums(x, weights []float32, iscale, mn float32, nmax int, Laux []uint8) (sumL, sumL2, sumXL float32) {
	n := len(x)
	i := 0
	if useAVX2 && n >= 8 && n <= 32 {
		var pL, pL2, pXL [32]float32
		var lout [32]int32
		bulk := (n / 8) * 8
		qkxProductsAVX(&x[0], &weights[0], bulk, iscale, mn, int32(nmax), &pL[0], &pL2[0], &pXL[0], &lout[0])
		for ; i < bulk; i++ {
			sumL += pL[i]
			sumL2 += pL2[i]
			sumXL += pXL[i]
			Laux[i] = uint8(lout[i])
		}
	}
	for ; i < n; i++ {
		l := nearestInt(iscale * (x[i] - mn))
		if l < 0 {
			l = 0
		}
		if l > nmax {
			l = nmax
		}
		Laux[i] = uint8(l)
		fl := float32(l)
		w := weights[i]
		sumL += w * fl
		sumL2 += w * fl * fl
		sumXL += w * fl * x[i]
	}
	return sumL, sumL2, sumXL
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
