package convert

import (
	"encoding/binary"
	"math"

	"github.com/x448/float16"
)

const (
	qk80     = 32
	blockQ80 = 2 + qk80 // 34 bytes: FP16 scale + 32 x int8
	qkK      = 256
	kScaleSz = 12
	blockQ4K = 2 + 2 + kScaleSz + qkK/2   // 144 bytes: 2xFP16 + 12 scales + 128 quants
	blockQ6K = qkK/2 + qkK/4 + qkK/16 + 2 // 210 bytes: 128 ql + 64 qh + 16 int8 scales + FP16 d
)

// groupMaxEps mirrors ggml's GROUP_MAX_EPS (1e-15f): blocks whose largest
// magnitude falls below it are emitted as all-zero.
const groupMaxEps float32 = 1e-15

// nearestInt mirrors ggml's nearest_int: round-half-to-even via the 1.5*2^23
// magic-number trick. The K-quant kernels depend on this exact rounding;
// math.Round (half away from zero) would diverge on ties and break the
// bit-for-bit match with llama-quantize.
func nearestInt(fval float32) int {
	val := fval + 12582912.0 // 1.5 * 2^23 forces the integer into the low mantissa bits
	return int(int32(math.Float32bits(val)&0x007fffff) - 0x00400000)
}

func quantizeQ8_0(src []float32) []byte {
	nBlocks := len(src) / qk80
	out := make([]byte, nBlocks*blockQ80)
	quantizeQ8_0Blocks(out, src, nBlocks)
	return out
}

func quantizeQ8_0Blocks(dst []byte, src []float32, nBlocks int) {
	for i := range nBlocks {
		block := src[i*qk80 : (i+1)*qk80]
		off := i * blockQ80

		var amax float32
		for _, v := range block {
			if a := float32Abs(v); a > amax {
				amax = a
			}
		}

		d := amax / 127.0
		var id float32
		if d != 0 {
			id = 1.0 / d
		}

		binary.LittleEndian.PutUint16(dst[off:], ggmlFP32ToFP16(d))

		for j, v := range block {
			dst[off+2+j] = byte(int8(roundf(v * id)))
		}
	}
}

func quantizeQ4_K(src []float32) []byte {
	nBlocks := len(src) / qkK
	buf := make([]byte, nBlocks*blockQ4K)

	var L [qkK]uint8
	var Laux [32]uint8
	var weights [32]float32
	var mins [8]float32
	var scales [8]float32

	for i := range nBlocks {
		x := src[i*qkK : (i+1)*qkK]
		off := i * blockQ4K

		var maxScale, maxMin float32
		for j := range 8 {
			var sumX2 float32
			for l := range 32 {
				sumX2 += x[32*j+l] * x[32*j+l]
			}
			avX := float32(math.Sqrt(float64(sumX2 / 32)))
			for l := range 32 {
				weights[l] = avX + float32Abs(x[32*j+l])
			}
			scales[j] = makeQKX2Quants(32, 15, x[32*j:32*j+32], weights[:], L[32*j:32*j+32], &mins[j], Laux[:], -1, 0.1, 20)
			if scales[j] > maxScale {
				maxScale = scales[j]
			}
			if mins[j] > maxMin {
				maxMin = mins[j]
			}
		}

		invScale := float32(0)
		if maxScale > 0 {
			invScale = 63.0 / maxScale
		}
		invMin := float32(0)
		if maxMin > 0 {
			invMin = 63.0 / maxMin
		}

		var packedScales [kScaleSz]byte
		for j := range 8 {
			ls := min(63, nearestInt(invScale*scales[j]))
			lm := min(63, nearestInt(invMin*mins[j]))
			if j < 4 {
				packedScales[j] = uint8(ls)
				packedScales[j+4] = uint8(lm)
			} else {
				packedScales[j+4] = uint8(ls&0xF) | uint8((lm&0xF)<<4)
				packedScales[j-4] |= uint8((ls >> 4) << 6)
				packedScales[j] |= uint8((lm >> 4) << 6)
			}
		}

		binary.LittleEndian.PutUint16(buf[off:], ggmlFP32ToFP16(maxScale/63.0))
		binary.LittleEndian.PutUint16(buf[off+2:], ggmlFP32ToFP16(maxMin/63.0))
		copy(buf[off+4:off+4+kScaleSz], packedScales[:])

		for j := range 8 {
			var sc, m uint8
			getScaleMinK4(j, packedScales[:], &sc, &m)
			d := float16.Frombits(binary.LittleEndian.Uint16(buf[off:])).Float32() * float32(sc)
			if d == 0 {
				continue
			}
			dm := float16.Frombits(binary.LittleEndian.Uint16(buf[off+2:])).Float32() * float32(m)
			for ii := range 32 {
				l := nearestInt((x[32*j+ii] + dm) / d)
				if l < 0 {
					l = 0
				}
				if l > 15 {
					l = 15
				}
				L[32*j+ii] = uint8(l)
			}
		}

		q := buf[off+4+kScaleSz:]
		for j := 0; j < qkK; j += 64 {
			for l := range 32 {
				q[l] = L[j+l] | (L[j+l+32] << 4)
			}
			q = q[32:]
		}
	}

	return buf
}

// quantizeQ6_K mirrors ggml's quantize_row_q6_K_ref. Each 256-element
// super-block is split into 16 sub-blocks of 16: each sub-block scale comes
// from makeQXQuants, the super-block scale is the largest-magnitude sub-scale,
// and quants are 6-bit (4 low bits packed into ql, 2 high bits into qh).
func quantizeQ6_K(src []float32) []byte {
	nBlocks := len(src) / qkK
	buf := make([]byte, nBlocks*blockQ6K)

	var L [qkK]uint8
	var scales [qkK / 16]float32

	for i := range nBlocks {
		x := src[i*qkK : (i+1)*qkK]
		off := i * blockQ6K
		qlOff := off
		qhOff := off + qkK/2
		scOff := off + qkK/2 + qkK/4
		dOff := off + qkK/2 + qkK/4 + qkK/16

		var maxScale, maxAbsScale float32
		for ib := range qkK / 16 {
			scale := makeQXQuants(16, 32, x[16*ib:16*ib+16])
			scales[ib] = scale
			if a := float32Abs(scale); a > maxAbsScale {
				maxAbsScale = a
				maxScale = scale
			}
		}

		if maxAbsScale < groupMaxEps {
			// all-zero block: buf is already zeroed and ggmlFP32ToFP16(0) == 0
			continue
		}

		iscale := -128.0 / maxScale
		binary.LittleEndian.PutUint16(buf[dOff:], ggmlFP32ToFP16(1.0/iscale))
		for ib := range qkK / 16 {
			buf[scOff+ib] = byte(int8(min(127, nearestInt(iscale*scales[ib]))))
		}

		d := float16.Frombits(binary.LittleEndian.Uint16(buf[dOff:])).Float32()
		for j := range qkK / 16 {
			dj := d * float32(int8(buf[scOff+j]))
			if dj == 0 {
				continue
			}
			for ii := range 16 {
				l := max(-32, min(31, nearestInt(x[16*j+ii]/dj)))
				L[16*j+ii] = uint8(l + 32)
			}
		}

		for j := 0; j < qkK; j += 128 {
			g := j / 128
			ql := buf[qlOff+g*64:]
			qh := buf[qhOff+g*32:]
			for l := range 32 {
				q1 := L[j+l+0] & 0xF
				q2 := L[j+l+32] & 0xF
				q3 := L[j+l+64] & 0xF
				q4 := L[j+l+96] & 0xF
				ql[l+0] = q1 | (q3 << 4)
				ql[l+32] = q2 | (q4 << 4)
				qh[l] = (L[j+l] >> 4) | ((L[j+l+32] >> 4) << 2) | ((L[j+l+64] >> 4) << 4) | ((L[j+l+96] >> 4) << 6)
			}
		}
	}

	return buf
}

// makeQXQuants mirrors ggml's make_qx_quants for rmse_type == 1 with no
// importance weights (the configuration q6_K uses): the per-element weight is
// x*x. It returns the block scale used to reconstruct values.
func makeQXQuants(n, nmax int, x []float32) float32 {
	var maxv, amax float32
	for i := range n {
		if ax := float32Abs(x[i]); ax > amax {
			amax = ax
			maxv = x[i]
		}
	}
	if amax < groupMaxEps {
		return 0
	}
	iscale := float32(-nmax) / maxv
	sumlx, suml2 := qxSums(x[:n], iscale, nmax)
	var scale float32
	if suml2 != 0 {
		scale = sumlx / suml2
	}
	best := scale * sumlx
	for is := -9; is <= 9; is++ {
		if is == 0 {
			continue
		}
		iscale = -(float32(nmax) + 0.1*float32(is)) / maxv
		sumlx, suml2 = qxSums(x[:n], iscale, nmax)
		if suml2 > 0 && sumlx*sumlx > best*suml2 {
			scale = sumlx / suml2
			best = scale * sumlx
		}
	}
	return scale
}

func getScaleMinK4(j int, q []uint8, d, m *uint8) {
	if j < 4 {
		*d = q[j] & 63
		*m = q[j+4] & 63
	} else {
		*d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
		*m = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
	}
}

func makeQKX2Quants(n, nmax int, x, weights []float32, L []uint8, theMin *float32, Laux []uint8, rmin, rdelta float32, nstep int) float32 {
	xmin := x[0]
	xmax := x[0]
	sumW := weights[0]
	sumX := sumW * x[0]
	for i := 1; i < n; i++ {
		if x[i] < xmin {
			xmin = x[i]
		}
		if x[i] > xmax {
			xmax = x[i]
		}
		w := weights[i]
		sumW += w
		sumX += w * x[i]
	}
	if xmin > 0 {
		xmin = 0
	}
	if xmax == xmin {
		for i := range n {
			L[i] = 0
		}
		*theMin = -xmin
		return 0
	}

	iscale := float32(nmax) / (xmax - xmin)
	scale := 1.0 / iscale
	var bestError float32
	for i := range n {
		l := nearestInt(iscale * (x[i] - xmin))
		if l < 0 {
			l = 0
		}
		if l > nmax {
			l = nmax
		}
		L[i] = uint8(l)
		diff := scale*float32(L[i]) + xmin - x[i]
		bestError += weights[i] * diff * diff
	}

	mn := xmin
	for is := range nstep + 1 {
		iscale = (rmin + rdelta*float32(is) + float32(nmax)) / (xmax - mn)
		var sumL, sumL2, sumXL float32
		for i := range n {
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
		D := sumW*sumL2 - sumL*sumL
		if D > 0 {
			thisScale := (sumW*sumXL - sumX*sumL) / D
			thisMin := (sumL2*sumX - sumL*sumXL) / D
			if thisMin > 0 {
				thisMin = 0
				thisScale = sumXL / sumL2
			}
			var curError float32
			for i := range n {
				diff := thisScale*float32(Laux[i]) + thisMin - x[i]
				curError += weights[i] * diff * diff
			}
			if curError < bestError {
				copy(L[:n], Laux[:n])
				bestError = curError
				scale = thisScale
				mn = thisMin
			}
		}
	}
	*theMin = -mn
	return scale
}

func float32Abs(f float32) float32 {
	return math.Float32frombits(math.Float32bits(f) &^ (1 << 31))
}

func roundf(f float32) float32 {
	return float32(math.Round(float64(f)))
}

func ggmlFP32ToFP16(f float32) uint16 {
	scaleToInf := math.Float32frombits(0x77800000)
	scaleToZero := math.Float32frombits(0x08800000)
	base := (float32Abs(f) * scaleToInf) * scaleToZero

	w := math.Float32bits(f)
	shl1W := w + w
	sign := w & 0x80000000
	bias := shl1W & 0xFF000000
	if bias < 0x71000000 {
		bias = 0x71000000
	}

	base = math.Float32frombits((bias>>1)+0x07800000) + base
	bits := math.Float32bits(base)
	expBits := (bits >> 13) & 0x00007C00
	mantBits := bits & 0x00000FFF
	nonsign := expBits + mantBits
	if shl1W > 0xFF000000 {
		return uint16(sign>>16) | 0x7E00
	}
	return uint16(sign>>16) | uint16(nonsign)
}
