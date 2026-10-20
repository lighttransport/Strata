# NVFP4 arithmetic on RX 9070 XT, 2026-10-19

Keep **tiled FP8 and tiled BF16** as candidates for packed W4A4 operands. In this
miner-running experiment, tiled FP8 was faster for the 512x2048x4096 projection;
FP8 and BF16 were close for 512x4096x2048. Do not choose one universal dispatch
from these noisy timings. Exact split INT4 worked, but the tested implementation
was much slower. INT8 was generally behind tiled FP8 in the larger cases.

All these packed-input paths preserve the E2M1 values. FP8 keeps scales separate
per K16; BF16 folds the local scale into each input exactly. Expanded BF16
controls can be faster, but consume four times the payload storage and exclude
preparation cost. These are kernel comparisons, not whole-model speedups.

## Same-run matched-tile comparison

Milliseconds: median [minimum, maximum], five paired rounds of 20 launches.
INT8, FP8, and BF16 below all use 64x64 output tiles, four waves, and K32
shared-memory panels. Packed paths include unpacking and local scaling in GEMM.
Large range overlap precludes claims about small differences.

| M,N,K | Scales | Packed -> INT8 | Packed -> FP8 | Packed -> BF16 | Predecoded BF16 |
|---|---|---:|---:|---:|---:|
| 1,2048,4096 | varying | 0.5474 [0.5344, 0.5538] | 0.3545 [0.3246, 0.3608] | 0.6300 [0.6124, 0.6555] | 0.1734 [0.1539, 0.2784] |
| 1,4096,2048 | varying | 0.3548 [0.3250, 0.7289] | 0.2480 [0.2454, 0.5893] | 0.3753 [0.3704, 0.5357] | 0.1225 [0.1213, 0.5738] |
| 16,2048,4096 | varying | 0.5734 [0.5531, 0.5921] | 0.4552 [0.3482, 0.5744] | 0.6462 [0.6317, 0.6624] | 0.2125 [0.1895, 0.5735] |
| 16,4096,2048 | varying | 0.3675 [0.3507, 0.5963] | 0.2551 [0.2496, 0.5930] | 0.3909 [0.3766, 0.5355] | 0.1289 [0.1272, 0.3041] |
| 64,2048,4096 | varying | 0.5818 [0.5654, 0.5938] | 0.5797 [0.3670, 0.5967] | 0.6499 [0.6291, 0.6783] | 0.5502 [0.2034, 0.5791] |
| 64,4096,2048 | varying | 0.4403 [0.3579, 0.5958] | 0.5803 [0.2580, 0.6005] | 0.4301 [0.3976, 0.5133] | 0.5758 [0.1327, 0.6023] |
| 512,2048,4096 | varying | 0.9641 [0.9456, 1.0358] | 0.6459 [0.6314, 0.6652] | 0.9774 [0.8883, 1.0206] | 0.5999 [0.5287, 0.6337] |
| 512,4096,2048 | varying | 0.7938 [0.7517, 0.8324] | 0.7370 [0.5240, 0.8692] | 0.7199 [0.6998, 0.9160] | 0.4396 [0.4194, 1.1291] |
| 1024,1024,1024 | varying | 0.3482 [0.3460, 0.6077] | 0.3007 [0.2508, 0.6244] | 0.3147 [0.2983, 0.5528] | 0.1600 [0.1500, 0.5978] |
| 1024,1024,1024 | unit | 0.3449 [0.3415, 0.6282] | 0.2476 [0.2473, 0.6116] | 0.3085 [0.3004, 0.3129] | 0.1586 [0.1562, 0.1635] |

At 512x2048x4096, packed tiled FP8 reached **13.30 TOPS** (0.6459 ms),
about **1.51x** the packed tiled BF16 throughput in this run. At
512x4096x2048, packed BF16 measured 0.7199 ms and FP8 0.7370 ms; their ranges
overlap, so this does not establish a reliable winner for that shape.

## Untiled controls at 1024 cubed

These one-wave 16x16 kernels expose the cost of decoding and per-K16 scaling.
The INT4 path uses two K32 instructions for each logical K16 block. Its result
is exact, but these timings do not support choosing it for NVFP4.

| Kernel | Varying-scale time, ms |
|---|---:|
| int8_fused | 0.6654 [0.5766, 1.5045] |
| fp8_fused | 0.6344 [0.5435, 1.3508] |
| int4_split_k32 | 2.4504 [2.3972, 2.7187] |
| bf16_fused | 1.1121 [1.0901, 1.7936] |
| bf16_predecoded | 0.2057 [0.1948, 1.2082] |
| fp8_preexpanded | 0.5237 [0.4969, 0.7608] |

Unit-scale controls at 1024 cubed:

| Kernel | Time, ms |
|---|---:|
| fp8_unit | 0.2204 [0.2194, 1.0806] |
| int8_unit | 0.2932 [0.2899, 0.2998] |

These controls accumulate across all K without block scaling. They do not
implement general NVFP4 K16 scales. Their unit-scale FP64 errors were zero.

## Preparation cost

Expanding both packed inputs to raw FP8 uses two GPU launches. The values remain
exact, but scales still need applying during GEMM. Expansion times below are
excluded from `fp8_preexpanded` GEMM timing and are not transfer timings.

| M,N,K | Scales | Expand both to FP8, ms |
|---|---|---:|
| 1,2048,4096 | varying | 0.5727 [0.3656, 0.6288] |
| 1,4096,2048 | varying | 0.7733 [0.5967, 1.2017] |
| 16,2048,4096 | varying | 0.7692 [0.7188, 0.7856] |
| 16,4096,2048 | varying | 0.7099 [0.4639, 0.7970] |
| 64,2048,4096 | varying | 1.1529 [0.5925, 1.1845] |
| 64,4096,2048 | varying | 0.7406 [0.4742, 0.7814] |
| 512,2048,4096 | varying | 0.8365 [0.7037, 1.1368] |
| 512,4096,2048 | varying | 0.8134 [0.3337, 1.1660] |
| 1024,1024,1024 | varying | 0.2944 [0.2262, 1.2451] |
| 1024,1024,1024 | unit | 0.2664 [0.0482, 0.6818] |

## Validation and machine

- 23 GPU cases passed: all 256 code pairs, all 127x127 finite scale pairs,
  basis matrices, zero/sign cases, cancellation, padding, and multi-tile tails.
- Every output was checked after every timed GEMM round. INT8, FP8, and split
  INT4 matched the grouped FP32 reference numerically exactly. BF16 and all
  other paths passed the FP32 summation-error bound against the FP64 reference.
- CPU checks verified every code, split product, and code/finite-scale value.
  All 2032 scaled values fit BF16 exactly; only 1482 fit FP8 exactly. This
  exhaustive codebook result is not a model-data error estimate.
- Generated assembly includes the intended signed INT8 and INT4 K32, FP8, and
  BF16 WMMA instructions. CPU and gfx1201 builds passed without compiler warnings.

RX 9070 XT, gfx1201, 64 physical CUs (HIP reports 32 multiprocessors), wave32,
16 GB VRAM. Host: Threadripper 1950X, 16 cores / 32 threads, OS MemTotal
131869116 KiB; system RAM speed was not measured. HIP 7.16.26385, AMD clang
24.0.0git, kernel 7.0.0-38-generic, amdgpu 7.1.9.31600000. Flags:
`-O3 -std=c++17 --offload-arch=gfx1201 -Wall -Wextra -Wpedantic`.

The miner stayed running. Snapshots bracketing the screening and tiled sweeps
showed 100% GPU use, SCLK 1936/1929 MHz, MCLK 875/875 MHz, power 209/210 W,
junction temperature 86/86 C, and 6,162,857,984 bytes VRAM used before/after.
No settings were changed. These are snapshots, not continuous telemetry.

Five warmups, five paired rounds, 20 launches per round, rotating mode order.
All data remained resident, with no cache flush. HIP-event launch-sequence
measurements can include CPU submission gaps. Input preparation, allocation,
host transfers, and validation are outside GEMM timing. Global scale products
were 0.9375 for varying-scale cases and 1 for unit-scale cases. The fixed local
scale set includes zero, subnormals, and maximum 448; see the runner for exact
data generation. FP64 absolute errors and all hashes are in the manifest.

## Evidence and limits

[Final per-round CSV](results/20261019/timings.csv),
[complete final run output](results/20261019/validation-and-timings.txt), and
[hardware, protocol, source/binary hashes, and errors](results/20261019/manifest.json).
The earlier untiled screening is preserved separately under
`results/20261019/screening/`; the comparisons above use only the final same-run
data. Intermediate exploratory output remains in ignored `build/`.

No W4A16 workload, MXFP4 E8M0 scale path, model quality, production engine,
idle-GPU baseline, GPU memory sanitizer, or tuned GEMM library was tested.
These results support further work on packed FP8/BF16 paths, not a production
default change or a claim about peak hardware throughput. The small-M cases
use WMMA rather than a dedicated GEMV kernel.
