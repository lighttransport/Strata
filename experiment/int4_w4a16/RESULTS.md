# INT4 W4A16 results

Measured 2026-10-19 on RX 9070 XT (gfx1201) with the miner running and settings
unchanged. All 38 correctness cases passed across six modes. All measured
outputs equaled the same-run predecoded baseline numerically and passed the
FP64 error bound. Both activation types passed the exhaustive 2032-combination
weight/scale representability check. The HIP build was warning-free; generated
assembly contains F16 and BF16 WMMA.

These tests use signed two's-complement INT4 [-8,7], not E2M1 FP4 or E0M3 storage.
They use floating WMMA after exact weight expansion, preserving the activations.
Unit scales and E4M3 K32 scales both work. This does not validate arbitrary model
scale formats or establish a native INT4 WMMA path for 16-bit activations.

The table compares matching 64x64 output tiles. Entries are median milliseconds
(minimum–maximum) over five rounds of 20 launches after five warmups. Large ranges
show miner interference; close medians are not a reliable performance ranking.

| Scale | Activation | M×N×K | Packed INT4 | Predecoded 16-bit |
|---|---|---|---:|---:|
| unit | BF16 | 512×2048×4096 | 1.046 (1.012–1.169) | 1.096 (0.990–1.164) |
| unit | BF16 | 512×4096×2048 | 0.702 (0.678–1.285) | 0.718 (0.689–1.430) |
| unit | BF16 | 1024×1024×1024 | 0.238 (0.232–0.583) | 0.250 (0.238–0.599) |
| unit | FP16 | 512×2048×4096 | 1.038 (0.979–1.241) | 1.148 (0.982–1.178) |
| unit | FP16 | 512×4096×2048 | 0.683 (0.661–1.537) | 0.700 (0.678–1.303) |
| unit | FP16 | 1024×1024×1024 | 0.239 (0.225–0.577) | 0.241 (0.228–0.597) |
| e4m3_k32 | BF16 | 512×2048×4096 | 1.058 (1.021–1.132) | 0.995 (0.937–1.135) |
| e4m3_k32 | BF16 | 512×4096×2048 | 0.703 (0.671–1.556) | 0.711 (0.683–1.250) |
| e4m3_k32 | BF16 | 1024×1024×1024 | 0.237 (0.229–0.242) | 0.242 (0.234–0.606) |
| e4m3_k32 | FP16 | 512×2048×4096 | 1.053 (0.889–1.116) | 1.048 (0.968–1.170) |
| e4m3_k32 | FP16 | 512×4096×2048 | 0.699 (0.677–1.556) | 0.713 (0.701–1.463) |
| e4m3_k32 | FP16 | 1024×1024×1024 | 0.248 (0.233–0.599) | 0.257 (0.243–0.607) |

The K32-scaled packed kernel reached approximately 12.2 logical TOPS at
512×4096×2048. Its median was close to the predecoded baseline in this run.
That is promising for avoiding persistent expanded weights, but there is no
idle-GPU or tuned library comparison. Do not infer a speedup over NVFP4 from
separate runs under varying miner load. M=1,16,64 and all other tile results
are retained in the CSV; these kernels do not replace a tuned GEMV baseline.

Hardware: RX 9070 XT, Threadripper 1950X, 32 logical CPU threads,
131869116 KiB OS MemTotal; RAM speed unmeasured. Software: HIP 7.16.26385,
clang 24, Linux 7.0.0-38-generic, `-O3 -std=c++17 --offload-arch=gfx1201`.
[Before](results/20261019/before.txt) and [after](results/20261019/after.txt)
GPU snapshots record clocks, utilization, power, temperature, and VRAM use.

Resident buffers are reused without cache eviction. Predecoded weight preparation
is excluded from GEMM timing and recorded separately as `expand_weights`.
No production source/default changes, model inference, GPU memory sanitizer,
CUDA/SYCL build, or cross-vendor bitwise validation was performed.

[Build and validation details](README.md),
[1260 timing samples](results/20261019/timings.csv),
[validation log](results/20261019/validation-and-timings.txt), and
[manifest with source/binary hashes](results/20261019/manifest.json).
