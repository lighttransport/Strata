# RX 9070 XT E0M3 results, 2026-10-19

E0M3 values are exactly representable by recoded signed INT4. The standalone
implementation passed CPU and RX 9070 XT checks. Its simple scaled K16 kernels
were slower than the simple BF16 WMMA control in this miner-running sweep. The
unit-scale K32 control was faster at 1024 cubed, but omits per-K16 scaling.

## Machine and protocol

RX 9070 XT, gfx1201, 64 physical CUs (HIP reports 32 multiprocessors), wave32,
16 GB VRAM; Threadripper 1950X, 16 cores / 32 threads, OS MemTotal 131869116 KiB.
System RAM speed was not measured. Inputs and outputs stayed GPU-resident during
timing; this is not a RAM, PCIe-transfer, or streamed-weight benchmark.

HIP 7.16.26385, AMD clang 24.0.0git, kernel 7.0.0-38-generic, amdgpu
7.1.9.31600000. Build flags: `-O3 -std=c++17 --offload-arch=gfx1201
-Wall -Wextra -Wpedantic`. No fast-math flag. Source hashes, compiler version, and
the final executable hash are recorded in the [manifest](results/20261019/manifest.json).

The miner stayed running; settings were unchanged. Before/after snapshots both
showed 100% GPU utilization, 214 W package power, 86 C junction temperature, and
6,162,857,984 bytes of VRAM used. SCLK was 1903/1886 MHz and MCLK 875/875 MHz.
These are snapshots, not average clocks or continuous load measurements.

Five warmups per mode, five paired rounds of 20 launches, rotating mode order,
HIP event timing. Allocation, conversion/decoding, transfers, and correctness
checks are excluded from GEMM timing. Recoding both operands is measured
separately. Host submission gaps may contribute to these short timings. All
outputs were validated after each timed round. No cache eviction was performed.

## Median time and range

All times below are milliseconds: median [minimum, maximum]. Each cell has five
rounds from the same run. The broad ranges limit conclusions about small differences.

| M,N,K | Scales | Fused INT4 K16 | Preconverted INT4 K16 | BF16 K16 |
|---|---|---:|---:|---:|
| 1,2048,4096 | varying | 0.2722 [0.2603, 0.6124] | 0.2553 [0.2164, 0.4265] | 0.1212 [0.1194, 0.1281] |
| 1,4096,2048 | varying | 0.1653 [0.1617, 0.6028] | 0.1545 [0.1490, 0.6191] | 0.0928 [0.0905, 0.2609] |
| 16,2048,4096 | varying | 0.7728 [0.7463, 0.8108] | 0.7503 [0.7146, 0.8079] | 0.1226 [0.1188, 0.5944] |
| 16,4096,2048 | varying | 0.3987 [0.3801, 0.5789] | 0.3830 [0.3718, 0.5142] | 0.0936 [0.0919, 0.6156] |
| 64,2048,4096 | varying | 0.9473 [0.8578, 0.9819] | 0.8851 [0.8526, 0.9414] | 0.6088 [0.2478, 0.6144] |
| 64,4096,2048 | varying | 0.8007 [0.7628, 0.8107] | 0.7663 [0.7592, 0.8264] | 0.6112 [0.2232, 0.6256] |
| 512,2048,4096 | varying | 3.2106 [3.1321, 3.2993] | 3.2630 [3.1422, 3.4615] | 0.9304 [0.8740, 1.3004] |
| 512,4096,2048 | varying | 2.9646 [2.9354, 3.9181] | 3.3064 [2.9889, 4.0688] | 1.0718 [1.0666, 1.3542] |
| 1024,1024,1024 | varying | 0.4650 [0.4511, 1.2827] | 0.4682 [0.4579, 0.5174] | 0.2115 [0.2042, 0.2127] |
| 1024,1024,1024 | unit | 0.4703 [0.4575, 1.2344] | 0.4525 [0.4371, 1.2923] | 0.2124 [0.1970, 0.8911] |

Unit-scale K32 INT4 at 1024 cubed: **0.1762 [0.1672, 0.1819] ms**,
**12.19 TOPS** using logical `2*M*N*K` operations.
It accumulates across K in INT32 and applies no scales. It is not a replacement
for the independently scaled K16 path.

## Separate recoding cost

Two GPU conversion launches, one per operand; no allocation or transfer included.
Preconverted GEMM times exclude this cost. A model integration would need to
distinguish reusable weight conversion from per-invocation activation conversion.

| M,N,K | Scales | Recode both, ms |
|---|---|---:|
| 1,2048,4096 | varying | 0.3989 [0.3727, 1.2444] |
| 1,4096,2048 | varying | 0.3982 [0.3851, 1.2492] |
| 16,2048,4096 | varying | 0.3778 [0.1908, 1.2263] |
| 16,4096,2048 | varying | 0.4048 [0.3859, 1.2427] |
| 64,2048,4096 | varying | 1.0445 [0.3852, 1.2252] |
| 64,4096,2048 | varying | 0.3701 [0.2478, 0.8194] |
| 512,2048,4096 | varying | 0.3728 [0.3108, 0.4227] |
| 512,4096,2048 | varying | 0.3472 [0.2568, 1.0687] |
| 1024,1024,1024 | varying | 0.1513 [0.1299, 1.1287] |
| 1024,1024,1024 | unit | 0.1509 [0.1402, 0.7945] |

## Correctness and evidence

- 22 GPU test cases passed, including all 256 code pairs and all 127 x 127
  finite scale pairs, lane-placement basis tests, zeros, cancellation, and tails.
- The raw E0M3-as-INT4 negative control failed 155/256 outputs, as expected.
- Both INT4 K16 paths matched the grouped FP32 reference at every checked output.
  The unit K32 control also matched exactly. Every benchmark output was checked.
- BF16 and INT4 passed the FP32 summation error bound against FP64. Across the
  varying-scale benchmark cases, maximum absolute FP64 errors were 118.0023003
  for INT4 K16 and 374.4547157 for BF16. The synthetic scale set includes 448;
  these are accumulation-rounding measurements, not quantization/model errors.
  All unit-scale benchmark errors were zero.
- CPU tests cover 65,536 packed 16-bit patterns repeated in 32-bit words, 2,032
  exact BF16 inputs, and all signed-value/finite-scale product combinations.
- Generated assembly contains signed `v_wmma_i32_16x16x16_iu4`,
  `v_wmma_i32_16x16x32_iu4`, and `v_wmma_f32_16x16x16_bf16`.
- The final source rebuild produced the identical measured executable hash.

[Per-round CSV](results/20261019/timings.csv),
[complete successful run output](results/20261019/validation-and-timings.txt), and
[hardware, protocol, and hashes](results/20261019/manifest.json) preserve the
compact evidence. Build binaries and full compiler assembly remain ignored.

No idle-GPU run, GPU memory sanitizer, model-quality test, production integration,
or cross-vendor bitwise comparison was performed. The BF16 and INT4 kernels are
untuned one-wave baselines, so these timings do not establish peak WMMA throughput.
Production CUDA/HIP/SYCL sources and Strata defaults were not changed.
