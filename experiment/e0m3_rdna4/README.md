# E0M3 on RX 9070 XT INT4 WMMA

Standalone representability, correctness, and timing experiment. It is not part of
the Strata build and changes no engine API, model format, or default path.
The numeric contract follows the gemm reference's `cuda/int4/int4_numeric.h` and
`cuda/int4/README.md`; the small adapted numeric helpers retain its MIT license.

## Representability

E0M3 is sign-magnitude, with values -7 through 7 and two zeros. Signed INT4 is
two's complement, with values -8 through 7. Recoding preserves every numeric
value; it canonicalizes negative zero. Reinterpreting the bits does not work:

```text
E0M3 nibble: 0 1 2 3 4 5 6 7 8 9 A B C D E F
INT4 nibble: 0 1 2 3 4 5 6 7 0 F E D C B A 9
```

Both operands use row-major logical shapes `A[M,K]` and `B[N,K]`. The earlier K
element occupies the low nibble. K is padded to a multiple of 64 with zero codes.
Each row has one UE4M3 scale byte per 16 K elements, in linear order. Accepted
scale codes are 0 through 126, including zero, subnormals, and maximum 448.
Code 56 (`0x38`) is scale 1. Output is row-major FP32 `C[M,N]`, alpha 1, beta 0.

For each K16 block, compute an exact INT32 dot with signed INT4 WMMA, then
`C = fmaf(float(dot), scale_A * scale_B, C)`. The dot is within [-784,784].
Every decoded input is exact in BF16 and FP32; each scale product, scaled input
product, and scaled K16 subtotal is exact in FP32. The subtotal needs at most
18 significant binary digits (784 times two four-bit scale significands).
Accumulation across differently scaled blocks can still round. This establishes
value/product representability, not bitwise NVIDIA floating-point MMA parity.

The K32 control is enabled only for unit scales. A single K32 subtotal cannot
preserve two independently scaled K16 contributions in general.

## Build and run

Requires an RX 9070 XT (`gfx1201`), Linux HIP toolchain, C++17 compiler, and make.
The CPU tests need only a C++17 compiler. No weights, downloads, CUDA, rocBLAS,
or external gemm checkout are needed.

```sh
make -C experiment/e0m3_rdna4 cpu
make -C experiment/e0m3_rdna4 -j2 all
make -C experiment/e0m3_rdna4 test isa
make -C experiment/e0m3_rdna4 bench
```

`HIPCC` and `CXX` can select compiler commands. The executable accepts `--test`
(the default), or `--bench --output FILE`. Benchmark mode always runs correctness
first. Generated executables, assembly, logs, and default CSV are under ignored
`build/`. The ISA target verifies the K16/K32 IU4 and BF16 instructions in
compiler-generated assembly; it is not a disassembly of the linked executable.

## Correctness and timing protocol

- CPU: all 65,536 16-bit patterns repeated in a packed 32-bit word; all 2,032
  code/finite-scale inputs; all integer-value/finite-scale product combinations.
- GPU: all 256 code pairs, every finite scale pair, basis matrices for K16/K32
  lane placement, both zeros, extrema, cancellation, varying scales, padding,
  irregular dimensions, and multiple K blocks. A raw-bit negative control must fail.
- The independent scalar GPU reference computes blockwise FP32 and FP64 answers
  without WMMA or packed-word recoding. All its outputs in small tests, and 64
  evenly spaced outputs per benchmark shape, are also checked on the CPU.
- Every output of both K16 INT4 paths must equal the grouped FP32 reference.
  Unit-scale K32 results must also match exactly. All paths, including BF16,
  must satisfy an FP32 summation error bound against FP64 based on the sum of
  absolute products. Output canaries check trailing stores. This is not a GPU
  memory-sanitizer run.
- Timings use five warmup launches per mode, then five paired rounds of 20
  launches with rotating mode order. HIP events measure the launch sequence;
  CPU submission gaps can contribute. Validation runs after each timed round.
  Reports retain median, minimum, maximum, and every round, with logical
  `2*M*N*K` operations for TOPS.

`fused_k16` recodes in registers; `preconverted_k16` uses preconverted packed
INT4 buffers. `recode_both` measures the two conversion launches separately.
`bf16_k16` uses inputs decoded exactly before timing. `fused_k32_unit` recodes
in registers and accumulates unscaled INT32 across K. Preparation, allocation,
transfers, reference calculation, and validation are outside GEMM timing.

These are simple one-wave 16x16 kernels using global loads, without LDS tiling,
cross-tile reuse, or specialized decode kernels. Buffers stay resident and are
reused without cache eviction. The BF16 control is a matching simple WMMA
implementation, not a tuned library baseline. K32 unit timings do not predict
the throughput of per-K16 scaled E0M3 GEMM.

## Measured results

See [RESULTS.md](RESULTS.md) and the compact evidence in `results/20261019/`.
The miner remains running throughout; no clocks, power limits, or miner settings
are changed. These results do not measure an idle GPU, model accuracy, a complete
Strata inference path, or NVIDIA parity. CUDA and SYCL builds are not applicable
to these isolated HIP/CPU sources; production backend files are untouched.
