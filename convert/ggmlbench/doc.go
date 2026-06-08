// Package ggmlbench benchmarks ggml's reference quantize kernels
// (quantize_row_*_ref) so they can be compared head-to-head against the pure-Go
// kernels benchmarked in the parent convert package (BenchmarkQuantize*). The Go
// kernels are byte-for-byte identical to these C ones (see
// convert.TestQuantizeMatchesGGML), so the only difference the benchmarks
// measure is implementation speed.
//
// The benchmarks live here, in a separate cgo package, because the convert
// package contains Go assembly (simd_amd64.s) which cannot coexist with cgo.
// They are built only under the "ggmlbench" build tag with CGO and a ggml build
// (e.g. the llama.cpp fork that produces llama-quantize):
//
//	CGO_ENABLED=1 \
//	CGO_CFLAGS="-O3 -I<ggml/include> -I<ggml/src>" \
//	CGO_LDFLAGS="-L<lib> -lggml-base -lm -Wl,-rpath,<lib>" \
//	go test ./convert/ggmlbench/ -tags ggmlbench -run '^$' -bench Quantize -benchmem
//
// where <lib> holds libggml-base.so. Without the tag this package is empty, so
// normal builds and CI are unaffected.
package ggmlbench
