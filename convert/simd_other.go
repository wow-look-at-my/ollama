//go:build !amd64

package convert

import (
	"math"

	"github.com/x448/float16"
)

func convertF16ToF32(dst []float32, src []uint16) {
	for i, v := range src {
		dst[i] = float16.Frombits(v).Float32()
	}
}

func convertF32ToF16(dst []uint16, src []float32) {
	for i, v := range src {
		dst[i] = float16.Fromfloat32(v).Bits()
	}
}

func convertBF16ToF32(dst []float32, src []uint16) {
	for i, v := range src {
		dst[i] = math.Float32frombits(uint32(v) << 16)
	}
}

func convertF32ToBF16(dst []uint16, src []float32) {
	for i, v := range src {
		dst[i] = uint16(math.Float32bits(v) >> 16)
	}
}

// qxSums computes the make_qx_quants reduction sums for one group (scalar).
func qxSums(x []float32, iscale float32, nmax int) (sumlx, suml2 float32) {
	for i := range x {
		l := max(-nmax, min(nmax-1, nearestInt(iscale*x[i])))
		w := x[i] * x[i]
		fl := float32(l)
		sumlx += w * x[i] * fl
		suml2 += w * fl * fl
	}
	return sumlx, suml2
}

// qkxSums computes the make_qkx2_quants weighted reduction sums for one group
// and fills Laux (scalar).
func qkxSums(x, weights []float32, iscale, mn float32, nmax int, Laux []uint8) (sumL, sumL2, sumXL float32) {
	for i := range x {
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
