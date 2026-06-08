#include "textflag.h"

// func f16ToF32AVX(dst *float32, src *uint16, chunks int)
TEXT ·f16ToF32AVX(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ chunks+16(FP), CX
	TESTQ CX, CX
	JZ done_f16_to_f32

loop_f16_to_f32:
	VMOVDQU (SI), X0
	VCVTPH2PS X0, Y0
	VMOVUPS Y0, (DI)
	ADDQ $16, SI
	ADDQ $32, DI
	DECQ CX
	JNZ loop_f16_to_f32

done_f16_to_f32:
	VZEROUPPER
	RET

// func f32ToF16AVX(dst *uint16, src *float32, chunks int)
TEXT ·f32ToF16AVX(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ chunks+16(FP), CX
	TESTQ CX, CX
	JZ done_f32_to_f16

loop_f32_to_f16:
	VMOVUPS (SI), Y0
	VCVTPS2PH $4, Y0, X0
	VMOVDQU X0, (DI)
	ADDQ $32, SI
	ADDQ $16, DI
	DECQ CX
	JNZ loop_f32_to_f16

done_f32_to_f16:
	VZEROUPPER
	RET

// func bf16ToF32AVX(dst *float32, src *uint16, chunks int)
TEXT ·bf16ToF32AVX(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ chunks+16(FP), CX
	TESTQ CX, CX
	JZ done_bf16_to_f32

loop_bf16_to_f32:
	VPMOVZXWD (SI), Y0
	VPSLLD $16, Y0, Y0
	VMOVUPS Y0, (DI)
	ADDQ $16, SI
	ADDQ $32, DI
	DECQ CX
	JNZ loop_bf16_to_f32

done_bf16_to_f32:
	VZEROUPPER
	RET

// func f32ToBF16AVX(dst *uint16, src *float32, chunks int)
TEXT ·f32ToBF16AVX(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ chunks+16(FP), CX
	TESTQ CX, CX
	JZ done_f32_to_bf16

loop_f32_to_bf16:
	VMOVDQU (SI), X0
	VPSRLD $16, X0, X0
	VMOVDQU 16(SI), X1
	VPSRLD $16, X1, X1
	VPACKUSDW X1, X0, X0
	VMOVDQU X0, (DI)
	ADDQ $32, SI
	ADDQ $16, DI
	DECQ CX
	JNZ loop_f32_to_bf16

done_f32_to_bf16:
	VZEROUPPER
	RET

// Round-to-nearest-even via the 1.5*2^23 magic number, matching ggml's
// nearest_int (and the Go nearestInt): l = (bits(x+12582912.0) & 0x7fffff) - 0x400000.
DATA qxmagic<>+0(SB)/4, $0x4b400000 // 12582912.0
GLOBL qxmagic<>(SB), RODATA|NOPTR, $4
DATA qxmask<>+0(SB)/4, $0x007fffff
GLOBL qxmask<>(SB), RODATA|NOPTR, $4
DATA qxsub<>+0(SB)/4, $0x00400000
GLOBL qxsub<>(SB), RODATA|NOPTR, $4

// func qxProductsAVX(x *float32, n int, iscale float32, nmaxm1, negnmax int32, prodlx, prodl2 *float32)
//
// For each of n (a multiple of 8) elements computes, bit-identically to the
// scalar make_qx_quants inner loop:
//
//	l  = clamp(nearest_int(iscale*x), -nmax, nmax-1)
//	fl = float(l); w = x*x
//	prodlx = ((x*x)*x)*fl   (= w*x*fl)
//	prodl2 = ((x*x)*fl)*fl  (= w*fl*fl)
//
// Only the element-wise work is vectorized (plain VMULPS, no FMA); the caller
// sums prodlx/prodl2 sequentially so the reduction order matches the scalar
// kernel exactly.
TEXT ·qxProductsAVX(SB), NOSPLIT, $0-48
	MOVQ x+0(FP), SI
	MOVQ n+8(FP), CX
	MOVQ prodlx+32(FP), DI
	MOVQ prodl2+40(FP), DX

	VBROADCASTSS iscale+16(FP), Y10
	MOVL nmaxm1+20(FP), AX
	VMOVD AX, X11
	VPBROADCASTD X11, Y11
	MOVL negnmax+24(FP), AX
	VMOVD AX, X12
	VPBROADCASTD X12, Y12
	VBROADCASTSS qxmagic<>(SB), Y13
	VPBROADCASTD qxmask<>(SB), Y14
	VPBROADCASTD qxsub<>(SB), Y15

	SHRQ $3, CX
	TESTQ CX, CX
	JZ done_qx

loop_qx:
	VMOVUPS (SI), Y0     // x
	VMULPS Y10, Y0, Y1   // p = x*iscale
	VADDPS Y13, Y1, Y1   // val = p + magic
	VPAND Y14, Y1, Y1    // ival = bits & 0x7fffff
	VPSUBD Y15, Y1, Y1   // l = ival - 0x400000
	VPMINSD Y11, Y1, Y1  // l = min(l, nmax-1)
	VPMAXSD Y12, Y1, Y1  // l = max(l, -nmax)
	VCVTDQ2PS Y1, Y1     // fl = float(l)
	VMULPS Y0, Y0, Y2    // xx = x*x
	VMULPS Y0, Y2, Y3    // wx = xx*x
	VMULPS Y1, Y3, Y3    // prodlx = wx*fl
	VMOVUPS Y3, (DI)
	VMULPS Y1, Y2, Y4    // wfl = xx*fl
	VMULPS Y1, Y4, Y4    // prodl2 = wfl*fl
	VMOVUPS Y4, (DX)
	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $32, DX
	DECQ CX
	JNZ loop_qx

done_qx:
	VZEROUPPER
	RET

// func qkxProductsAVX(x, w *float32, n int, iscale, mn float32, nmax int32, pL, pL2, pXL *float32, lout *int32)
//
// For each of n (a multiple of 8) elements computes, bit-identically to the
// scalar make_qkx2_quants inner loop:
//
//	l  = clamp(nearest_int(iscale*(x-mn)), 0, nmax)   (stored to lout)
//	fl = float(l)
//	pL  = w*fl ; pL2 = (w*fl)*fl ; pXL = (w*fl)*x
//
// As with qxProductsAVX only the element-wise work is vectorized (plain VMULPS,
// no FMA); the caller sums pL/pL2/pXL sequentially so the reduction order
// matches the scalar kernel.
TEXT ·qkxProductsAVX(SB), NOSPLIT, $0-72
	MOVQ x+0(FP), SI
	MOVQ w+8(FP), BX
	MOVQ n+16(FP), CX
	MOVQ pL+40(FP), DI
	MOVQ pL2+48(FP), DX
	MOVQ pXL+56(FP), R8
	MOVQ lout+64(FP), R9

	VBROADCASTSS iscale+24(FP), Y10
	VBROADCASTSS mn+28(FP), Y11
	MOVL nmax+32(FP), AX
	VMOVD AX, X12
	VPBROADCASTD X12, Y12
	VBROADCASTSS qxmagic<>(SB), Y13
	VPBROADCASTD qxmask<>(SB), Y14
	VPBROADCASTD qxsub<>(SB), Y15
	VPXOR Y9, Y9, Y9

	SHRQ $3, CX
	TESTQ CX, CX
	JZ done_qkx

loop_qkx:
	VMOVUPS (SI), Y0     // x
	VMOVUPS (BX), Y1     // w
	VSUBPS Y11, Y0, Y2   // t = x - mn
	VMULPS Y10, Y2, Y2   // p = t * iscale
	VADDPS Y13, Y2, Y2   // val = p + magic
	VPAND Y14, Y2, Y2    // ival = bits & 0x7fffff
	VPSUBD Y15, Y2, Y2   // l = ival - 0x400000
	VPMAXSD Y9, Y2, Y2   // l = max(l, 0)
	VPMINSD Y12, Y2, Y2  // l = min(l, nmax)
	VMOVDQU Y2, (R9)     // lout = l
	VCVTDQ2PS Y2, Y2     // fl = float(l)
	VMULPS Y1, Y2, Y3    // pL = w*fl
	VMOVUPS Y3, (DI)
	VMULPS Y2, Y3, Y4    // pL2 = pL*fl
	VMOVUPS Y4, (DX)
	VMULPS Y0, Y3, Y4    // pXL = pL*x
	VMOVUPS Y4, (R8)
	ADDQ $32, SI
	ADDQ $32, BX
	ADDQ $32, DI
	ADDQ $32, DX
	ADDQ $32, R8
	ADDQ $32, R9
	DECQ CX
	JNZ loop_qkx

done_qkx:
	VZEROUPPER
	RET
