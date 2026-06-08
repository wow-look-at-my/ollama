//go:build ggmlbench

package ggmlbench

// cgo bindings to ggml's reference quantize kernels. The void* shims keep cgo
// from having to model the block_q* structs.

/*
#include <stdint.h>
#include "ggml.h"
#include "ggml-quants.h"

static void bench_q8_0(const float *x, void *y, int64_t k) { quantize_row_q8_0_ref(x, (block_q8_0 *)y, k); }
static void bench_q4_K(const float *x, void *y, int64_t k) { quantize_row_q4_K_ref(x, (block_q4_K *)y, k); }
static void bench_q6_K(const float *x, void *y, int64_t k) { quantize_row_q6_K_ref(x, (block_q6_K *)y, k); }
*/
import "C"

import "unsafe"

func ggmlQuantizeQ8_0(src []float32, dst []byte) {
	C.bench_q8_0((*C.float)(unsafe.Pointer(&src[0])), unsafe.Pointer(&dst[0]), C.int64_t(len(src)))
}

func ggmlQuantizeQ4_K(src []float32, dst []byte) {
	C.bench_q4_K((*C.float)(unsafe.Pointer(&src[0])), unsafe.Pointer(&dst[0]), C.int64_t(len(src)))
}

func ggmlQuantizeQ6_K(src []float32, dst []byte) {
	C.bench_q6_K((*C.float)(unsafe.Pointer(&src[0])), unsafe.Pointer(&dst[0]), C.int64_t(len(src)))
}
