# FP4 / NVFP4 on RDNA4 WMMA

Standalone W4A4 exploration on the RX 9070 XT: keep E2M1 operands packed in
4-bit storage, then use INT8, FP8, split INT4, or BF16 WMMA. This experiment
changes no Strata engine source, API, model format, or default path. Its runner
is adapted from the adjacent E0M3 experiment; that experiment and its frozen
results remain unchanged.

## Exact representations

The FP4 E2M1 magnitudes are `0, 0.5, 1, 1.5, 2, 3, 4, 6`, with a sign bit and
two zeros. [NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)
uses E4M3 scales per 16 elements plus a global FP32 scale per tensor.

| WMMA input | Exact mapping | Scale handling |
|---|---|---|
| INT8 | Multiply E2M1 by 2: `0,1,2,3,4,6,8,12`, with sign | Divide each integer dot by 4, then apply the K16 scale product |
| FP8 E4M3 | Every unscaled E2M1 value is exact | Apply the scale product to each K16 subtotal |
| INT4 | One signed INT4 value cannot represent the whole codebook with a common scale | Use two signed INT4 components and two K32 WMMA calls per logical K16 dot |
| BF16 | E2M1 times its local E4M3 scale is exact | Fold local scales into BF16 inputs, then accumulate across K in WMMA |

The INT4 obstacle is range: representing 0.5 as a nonzero integer requires a
step no larger than 0.5, but 6 then needs an integer of at least 12. Requantizing
to INT4 would change the input values. This experiment instead decomposes each
`x2 = 2*x` into `lo = trunc(x2/2)` and `hi = x2-lo`, both in [-6,6]. Two K32
calls compute `dot([Alo,Ahi],[Blo,Blo]) + dot([Alo,Ahi],[Bhi,Bhi])`.
The extra K positions represent component expansion within a K16 scale group;
they do not combine independently scaled groups.

Do not fold arbitrary NVFP4 local scales into FP8. For example,
`1.5 * 1.125 = 1.6875` is not exactly representable in E4M3, and `6 * 448`
exceeds its range. Only 1482 of the 2032 E2M1-code/finite-scale combinations fit
FP8 exactly in the exhaustive CPU check. All 2032 fit BF16 exactly. The global
FP32 scale product stays in the output epilogue, not in the BF16 operands.

The same E2M1 codebook reasoning applies to MXFP4, but its K32 E8M0 scales are
different. This runner tests NVFP4-style local scales; it does not validate
MXFP4 scale extremes, MXFP4 loaders, or a complete model.

## Build and reproduce

```sh
make -C experiment/nvfp4_rdna4 cpu
make -C experiment/nvfp4_rdna4 -j2 all
make -C experiment/nvfp4_rdna4 test isa
make -C experiment/nvfp4_rdna4 bench
```

Requires a C++17 compiler and Linux HIP with gfx1201 builtins. CPU checks require
no GPU. The executable accepts `--test` (default) or `--bench --output FILE`.
No downloads, weights, CUDA, or GEMM library are needed. Build products and the
default timing CSV stay in ignored `build/`.

Operands are row-major `A[M,K]`, `B[N,K]`, low nibble first, with K padded to a
multiple of 64 using zero codes. Scales use linear `[row,K/16]` indexing; finite
codes 0 through 126 include zero, subnormals, and maximum 448. This is not the
swizzled NVIDIA scale-storage layout. Output is row-major FP32. Test cases use
global scale product 0.9375 (0.75 times 1.25) for varying local scales and 1 for
unit local scales. Quantization itself is outside this experiment.

## Kernels and measurement

- `int8_fused`, `fp8_fused`: unpack in registers, execute a K16 WMMA, then scale
  the subtotal in FP32. Neither changes the input values.
- `int4_split_k32`: exact component decomposition described above.
- `bf16_fused`: decode and scale inputs directly for a one-wave 16x16 tile.
- `bf16_predecoded`, `fp8_preexpanded`: expanded inputs are prepared outside
  GEMM timing. FP8 still needs per-K16 output scaling. Its two input-expansion
  launches are timed separately as `expand_fp8_both`.
- `bf16_tiled_fused`, `fp8_tiled_fused`, `int8_tiled_fused`: matching 64x64 output
  tiles with four waves and K32 shared-memory panels. Each input tile is decoded
  once. BF16 absorbs local scales into inputs; INT8 and FP8 retain scaled K16
  subtotals. `bf16_tiled_predecoded` is the same BF16 topology on expanded inputs.
- `fp8_unit`, `int8_unit`: controls that accumulate across all K inside WMMA.
  They run only on unit local scales and are not substitutes for scaled NVFP4.

Five warmups precede five paired rounds of 20 launches with rotating mode order.
HIP events measure launch sequences, so CPU submission gaps may contribute.
Allocation, host transfers, preparation, and validation are outside GEMM timing.
Buffers remain resident and are reused without cache eviction. TOPS uses logical
`2*M*N*K`, not the extra work of the split-INT4 expansion. The miner remains
running; clocks, power limits, and miner settings are unchanged.

All 23 correctness cases check every GPU output: codebooks, every finite scale
pair, K16/K32 lane placement, zeros, extrema, cancellation, odd sizes, padding,
and multiple output tiles. INT8, FP8, and split INT4 must match a grouped FP32
reference exactly. BF16 follows its own WMMA accumulation order. All paths must
also satisfy an FP32 summation-error bound against FP64; output canaries check
trailing stores. A scalar GPU reference is cross-checked on the CPU for every
small-case output and 64 outputs per benchmark shape. Every GPU output is
validated again after each timed round.

The `isa` target inspects compiler-generated assembly for the expected
[gfx12 WMMA builtins](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html).
This is not a GPU memory-sanitizer run or a linked-executable disassembly.
Only the standalone HIP and CPU targets are built; production CUDA/HIP/SYCL
sources remain untouched.

See [RESULTS.md](RESULTS.md) for the measured comparison and its limits. These
are W4A4 synthetic arithmetic tests, not W4A16, model-quality, or end-to-end
inference measurements. Separate tests with full-precision BF16 and FP16
activations are documented in [W4A16.md](W4A16.md).
