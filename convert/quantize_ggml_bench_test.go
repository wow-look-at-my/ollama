//go:build ggmlbench

package convert

// Benchmarks of ggml's reference quantize kernels (quantize_row_*_ref) - the
// exact functions the Go kernels in quantize.go are byte-for-byte identical to
// (see TestQuantizeMatchesGGML) - so BenchmarkQuantize* (Go) and
// BenchmarkGGMLQuantize* (C) can be compared head-to-head.
//
// Built only under the "ggmlbench" tag, with CGO and a ggml build (e.g. the
// llama.cpp fork that produces llama-quantize):
//
//	CGO_ENABLED=1 \
//	CGO_CFLAGS="-O3 -I<ggml/include> -I<ggml/src>" \
//	CGO_LDFLAGS="-L<lib> -lggml-base -lm -Wl,-rpath,<lib>" \
//	go test ./convert/ -tags ggmlbench -run '^$' -bench Quantize -benchmem
//
// where <lib> holds libggml-base.so. The void* shims below keep cgo from having
// to model the block_q* structs.

/*
#include <stdint.h>
#include "ggml.h"
#include "ggml-quants.h"

static void bench_q8_0(const float *x, void *y, int64_t k) { quantize_row_q8_0_ref(x, (block_q8_0 *)y, k); }
static void bench_q4_K(const float *x, void *y, int64_t k) { quantize_row_q4_K_ref(x, (block_q4_K *)y, k); }
static void bench_q6_K(const float *x, void *y, int64_t k) { quantize_row_q6_K_ref(x, (block_q6_K *)y, k); }
*/
import "C"

import (
	"testing"
	"unsafe"
)

func benchmarkGGMLQuantize(b *testing.B, qk, blockBytes int, fn func(*C.float, unsafe.Pointer, C.int64_t)) {
	n := (benchElems / qk) * qk
	src := makeF32Data(n)
	out := make([]byte, n/qk*blockBytes)
	b.SetBytes(int64(n * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		fn((*C.float)(unsafe.Pointer(&src[0])), unsafe.Pointer(&out[0]), C.int64_t(n))
	}
}

func BenchmarkGGMLQuantizeQ8_0(b *testing.B) {
	benchmarkGGMLQuantize(b, qk80, blockQ80, func(x *C.float, y unsafe.Pointer, k C.int64_t) { C.bench_q8_0(x, y, k) })
}

func BenchmarkGGMLQuantizeQ4_K(b *testing.B) {
	benchmarkGGMLQuantize(b, qkK, blockQ4K, func(x *C.float, y unsafe.Pointer, k C.int64_t) { C.bench_q4_K(x, y, k) })
}

func BenchmarkGGMLQuantizeQ6_K(b *testing.B) {
	benchmarkGGMLQuantize(b, qkK, blockQ6K, func(x *C.float, y unsafe.Pointer, k C.int64_t) { C.bench_q6_K(x, y, k) })
}
