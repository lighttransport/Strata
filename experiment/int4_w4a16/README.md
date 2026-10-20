# Signed INT4 W4A16 on RDNA4

Standalone signed INT4 weight tests with genuine BF16 and FP16 activations.
Production engine source and defaults are unchanged. The runner is adapted
from the adjacent NVFP4 W4A16 experiment, whose measured sources are preserved.

Weights use two's-complement nibbles `0..7,-8..-1`, low nibble first, in
row-major `[N,K]`. This differs from E0M3 sign-magnitude storage; E0M3 must be
recoded first. Activations are row-major `[M,K]`, with their full BF16/FP16
mantissa precision. Output is FP32 `[M,N]` and K is zero padded to 64.

Two scale cases are benchmarked: unit local/global scales, and nonnegative
finite E4M3 scales per K32 group with a global weight scale of 0.75 applied
in the output epilogue. Unit cases use the same kernel with scale values of
one; they are not a specialized scale-free kernel. All 16 integer codes times
all 127 finite nonnegative E4M3 scale codes are exactly representable in both
BF16 and FP16, checked exhaustively. Arbitrary FP16/FP32 scales, affine zero
points, and particular quantized model formats are outside this test.

Packed weights are decoded into BF16/FP16 while loading WMMA inputs. This uses
F16/BF16 WMMA with FP32 accumulation, not native integer WMMA: directly
converting arbitrary 16-bit activations to INT4/INT8 would lose information.
K32 describes the scale group and tiled LDS panel; each floating WMMA has K16.

Six modes compare packed and predecoded weights with matching 16x16, 16x64,
and 64x64 output tiles. The simple 16x16 mode uses one wave; both LDS tiled
modes use four. Predecoded weights occupy four times the packed nibble payload,
excluding scales. Their preparation is excluded from GEMM time and measured
separately as `expand_weights`.

## Build and validation

```sh
make -C experiment/int4_w4a16 -j2 all
make -C experiment/int4_w4a16 test isa
make -C experiment/int4_w4a16 bench
```

Requires HIP with gfx1201 builtins. Build products stay in ignored `build/`.
The executable accepts `--test` or `--bench --output FILE`.

The 38 correctness cases check codebooks (including -8), all finite local scale
codes, basis vectors, cancellation, zeros, normal activation exponent ranges,
K tails, odd sizes, and multiple tiles. GPU weight expansion must equal host
expansion. All six GEMM modes must agree numerically with predecoded output,
meet an FP32 summation-error bound against scalar GPU FP64, and preserve output
canaries. CPU FP64 cross-checks every small-case output and 64 samples per
benchmark shape. Every output is checked again after each timed round.

Timing uses five warmups per GEMM mode, five rounds of 20 launches, rotating
mode order, and HIP events. Allocations, transfers, and validation are excluded;
CPU submission gaps may contribute. Buffers remain resident without cache
flushing. M=1 uses WMMA and is not a specialized GEMV baseline.

See [RESULTS.md](RESULTS.md) for measured results. These are synthetic arithmetic
tests, not model-quality or end-to-end inference tests. No tuned library,
GPU memory sanitizer, exhaustive activation subnormal/NaN/Inf test, or
cross-vendor parity check was run. Only standalone HIP sources were added;
CUDA/SYCL sources were not changed or built.
