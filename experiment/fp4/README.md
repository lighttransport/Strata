# MXFP4 decode and GEMM for MiMo-V2.6-Flash on RTX 5060 Ti + Threadripper 1950X

A standalone benchmark and profiler (not part of the Strata build) that answers one question before engine
work starts: what is the fastest correct way to multiply MiMo-V2.6-Flash's native MXFP4 expert weights on this
machine's GPU and CPU, for decode (1 token), MTP/batch verify (2-16 tokens) and prefill (512-8192 tokens)?
Every kernel runs on MiMo-V2.6-Flash expert shapes, with seeded dummy weights or the real checkpoint
(`/mnt/nvme01/models/m26f`), and is checked against an fp64 reference.

Measured on 2026-10-11: RTX 5060 Ti 16 GB (GB206, sm_120, 36 SMs, 128-bit GDDR7, driver/CUDA 13.2), AMD
Threadripper 1950X (16 cores, Zen 1, AVX2 without VNNI, 2 NUMA nodes), Linux 6.8. Raw numbers are in
`results/20261011-1822/` (CSV, logs, `machine.txt`, `summary.md` made by `scripts/summarize.py`). The prefill and MoE numbers come from the full pass. The decode and small-batch numbers come from the final rerun (`gpu_decode_v4*.csv`), made after the 256-element K chunk was enabled for small tiles.

## Results in one page

**GPU, RTX 5060 Ti** (read bandwidth measured 423 GB/s of 448 spec; PCIe link Gen3 x8, 7.2 GB/s
host-to-device measured; full tables in `results/20261011-1822/summary.md`):

| question | answer (measured) |
|---|---|
| Matrix units | int8 mma 152 TOPS, `kind::mxf4` 303 TFLOPS, `kind::mxf8f6f4` 152, plain fp8 / fp16 with fp32 accumulate only 76 / 38 (half rate on GeForce), fp16 with fp16 accumulate 76 |
| Batch-1 decode, 8 experts gate+up (35.7 MB) | `g1_dp4a` 170 us = **418 GB/s (99% of measured peak)**, relL2 0.7-1% (q8 activations) |
| same with exact fp16 activations | `g2_hfma2` / `g2_hmul_f32` 172-175 us, 408-415 GB/s, relL2 **2e-4** |
| Fused MoE layer (x -> 8 experts -> y, 2 launches) | **281 us = 380 GB/s (90%)**; 47 layers in one CUDA graph: 13.2 ms, i.e. **75.6 tok/s for the expert part if all experts were in VRAM** |
| MTP / batch verify, N tokens on the same 8 experts | tensor-core kernels stay flat: `k2_f16fold_mma` (fp16 activations, 2e-4) 188 / 188 / 190 / 220 us for N = 1 / 4 / 8 / 16; GEMV `g1_dp4a` 170 / 176 / 282 / 377 us |
| Prefill grouped GEMM, 256 experts, T = 2048 (64 tokens per expert on average) | `kind::mxf4` 74 TFLOPS (11.3 ms per layer, but FP4 activations: 15% error); `kind::mxf8f6f4` 45-49 TF (17.7 ms, 3%); MSD 45-50 TF (18 ms, 2%); int8 mma 30 TF (27 ms, 1%); fp16 21-23 TF (37 ms, 2e-4) |
| Prefill, T = 8192 | `kind::mxf4` 108 TF (31 ms per layer), `mxf8f6f4` 65 TF (51 ms), int8 40 TF (83 ms), fp16 25 TF (134 ms) |

**CPU, Threadripper 1950X** (read bandwidth 88 GB/s at 8 threads, 80-84 at 16-32):

| question | answer (measured) |
|---|---|
| Best GEMV kernel | `c3_q8_off_r1` (offset table, no sign ops): 76.8 GB/s at 16 threads, ggml-style `c2_q8_sign` 74.9; the fp32 path tops out at 50 GB/s (compute bound) |
| Full MoE layer | 1.50 ms at 16 threads (71 GB/s): **14.2 tok/s for the expert part** if every expert is read from RAM |
| Per core | 6.9 GB/s single-thread: Zen 1 is compute bound per core, so it takes 12+ cores to saturate memory |
| MTP verify | compute bound: N = 2 / 4 / 8 cost 1.3x / 2.1x / 4.2x of N = 1 at 32 threads |
| Threads | SMT does not help: perf shows 71% backend-stall cycles at 16 threads and IPC halving at 32 |

**Accuracy** (relL2 against fp64 on real layer-1/47 experts, activations N(0,1) with 8 outliers of x20):
- The weight decode of every kernel is exact. Against an fp64 reference that uses the kernel's own quantized
  activations, every GPU tensor-core kernel matches to <= 3e-6 (`--mode selftest`, 60/60 EXACT). The fp32
  CPU path matches to 2e-7.
- The remaining error is the activation format only:

  | activation format | relL2 |
  |---|---:|
  | fp16 | 2e-4 |
  | q8 per 32 | 0.7-1% |
  | e4m3 per token | 1.4-2.4% |
  | MXFP8 | 2.3-3.5% |
  | MSD 2 x MXFP4 | 1.7-2.5% |
  | MXFP4 | 15-16% |

  MXFP4 activations are not usable for this model without further work. `kind::mxf8f6f4` (FP4 weights x
  FP8 activations) is the usable native path.

## What to build in the `mimo_v2` engine path (recommendation)

1. **Decode, GPU-resident experts:** use `g1_dp4a`-style GEMV, or the fp16-activation `g2` for the same speed
   and 50x lower activation error. Fuse each MoE layer into the two launches of `g3`. Both sit at the
   bandwidth wall, so FP4 tensor cores bring nothing here.
2. **MTP / speculative verify (2-16 tokens):** use the tensor-core grouped GEMM with BN = 8/16 and fp16
   activations (`k2_f16fold_mma`). It needs no FP4 hardware, verifies 8 tokens for the price of one, and is
   exact up to fp16 rounding. The int8 / FP8 / native kinds are the same speed, so they offer no reason to
   lose accuracy.
3. **CPU experts:** use the AVX2 offset-table kernel with q8 activations at 12-16 threads, one thread per
   core. The CPU, not the GPU, sets decode speed here. The full MXFP4 expert set is 162 GB, so most experts
   live in 125 GB of RAM at 76 GB/s, about 14 tok/s for the expert part. Verify on the CPU is compute bound
   (N = 4 costs 2.1x), which caps MTP gains for CPU-resident experts.
4. **Prefill:** experts do not fit in VRAM, and PCIe Gen3 x8 moves 7.2 GB/s. One layer's 256 experts
   (3.4 GB) take about 470 ms to stream, 4-40x longer than any GEMM here at T <= 8192. So:
   - while streaming, take the most accurate kernel (`k2_f16fold_mma`, 37 ms at T = 2048, 134 ms at 8192)
     and overlap transfer with compute;
   - use the largest prefill chunk memory allows, since cost per layer is nearly fixed by the transfer;
   - `kind::mxf8f6f4` (1.5-3x faster than the software paths, 3% error) only pays off for expert sets that
     stay resident in VRAM (e.g. a REAP-pruned model).
5. **The two papers:**
   - MSD (2605.13915) works as described: two `kind::mxf4` passes cut MXFP4-activation error from 16% to
     2%, at the speed of `mxf8f6f4`, which has similar error. On this GPU it brings nothing over
     `mxf8f6f4`.
   - XOR-Trellis (2610.00432) is a different (2.25-bit) format and was not benchmarked.

How the GEMM got here (T = 2048, gate+up, `kind::mxf4`):

| version | TFLOPS |
|---|---:|
| v1, synchronous loads | 28 |
| v2, 3-stage cp.async | 39 |
| v3, grid order for L2 locality (rows fastest) | 52 |
| v4, 256-element K chunks | 74 |

Fragment reuse (32x32 warp tiles, ldmatrix) changed little, because the GEMM is bound by weight bandwidth.
At T = 2048 a weight byte feeds about 128 MACs, below the ridge point of about 450 MAC/byte for FP4.


## K-quant experts: Strata q23 pack and ggml-org Q2_K mix (added 2026-10-11)

`common/kquant.h` ports ggml's Q2_K/Q3_K blocks, reference quantizers (no imatrix: the uncalibrated path of
`strata-mimo-q2-pack`) and q8_K activations. Every expert is requantized from MXFP4 (exact f32 dequant first);
on real layer-1 experts that costs a weight NMSE of 0.10-0.12 for Q2_K gate/up and 0.021-0.024 for Q3_K down.

- CPU (`cpu/kq_cpu.h`): AVX2 ports of `ggml_vec_dot_q2_K_q8_K` / `q3_K`; full MoE layers for the q23 mix
  (Q2_K gate/up + Q3_K down, 72.9 MB per layer) and ggml-org's Q2_K GGUF mix (Q2_K gate/up + MXFP4 down,
  79.7 MB). Selftest: exact against fp64 on the q8_K-quantized input (<= 5e-7); the per-256 q8_K activation
  scale itself costs 2-4% with the outlier activations used here. `bench_cpu` runs them by default
  (`--no-kquant` / `--kquant-only`). Measured on the 1950X (2026-10-12, machine idle, 116 GB free; real
  layer-1 experts, `results/20261012-kquant-cpu/`), median us per call, same session as the MXFP4 layer:

  | kernel | 8 threads | 16 threads | GB/s at 16 |
  |---|---:|---:|---:|
  | Q2_K gate+up, 8 experts (44.0 MB) | 1437 | 1173 | 37.6 |
  | Q3_K down, 8 experts (28.8 MB) | 752 | 743 | 38.8 |
  | MoE layer, q23 mix (72.9 MB) | 2890 | 1969 | 37.0 |
  | MoE layer, Q2_K + MXFP4 down (79.7 MB) | 2468 | 2094 | 38.1 |
  | MoE layer, MXFP4 c2 (106.9 MB), for comparison | 2388 | 1540 | 69.5 |

  The K-quant kernels are compute bound on Zen1: 37-41 GB/s at 8-16 threads on dummy and real weights, while
  the read reference reaches 84 GB/s and the MXFP4 layer 69.5 GB/s in the same session. So on this CPU the
  q23 layer, 32% fewer bytes, is 28% slower than MXFP4 (1.97 vs 1.54 ms; expert part at 47 layers about
  11 vs 14 tok/s). The q23 pack is what fits in 128 GB, but making it pay off on the CPU needs a faster Q2_K
  dot (the ggml AVX2 port unpacks 4-bit scales and mins per 16 and does two madd chains per block). 32-thread
  (SMT) timings varied 3x between runs (436-2079 us for Q3_K down) and are not used. A first run with 120 GB
  of page cache and 1 GB free per node measured 42-54 GB/s read bandwidth instead of 80-85 GB/s; it is kept
  in `results/20261012-kquant-cpu/busy-pagecache/` and not used.
- GPU (`gpu/kq_gpu.cuh`): dp4a GEMVs on the native 84/110-byte blocks with int8 activations per 32 (plus
  per-16 sums for the Q2_K mins), R rows per lane group so loads overlap; fused 2-launch MoE layers.
  Exact against fp64 on their quantized input (6/6 in `--mode selftest`). Measured (`--mode kquant`,
  real layer-1 experts, `results/20261011-1822/gpu_kquant_real.csv`; taken while a CPU-heavy conversion ran):

  | kernel | us per call | GB/s |
  |---|---:|---:|
  | Q2_K gate+up, 8 experts (44.1 MB), 1 / 2 / 4 rows per lane group | 173 / 111 / 110 | 254 / 395 / 400 |
  | Q3_K down, 8 experts (28.8 MB), 1 / 2 / 4 rows | 103 / 85 / 79 | 281 / 338 / 364 |
  | fused MoE layer, q23 mix | 253 (275 per layer in a 47-layer graph) | 288 |
  | fused MoE layer, Q2_K + MXFP4 down | 232 (264 in the graph) | 343 |

  With one row per lane group the K-quant GEMVs were latency bound (60% of bandwidth); four rows reach 95%.
  The q23 layer is 32% fewer bytes than MXFP4 but only about 10% faster per layer: the fused gate/up kernel
  re-quantizes x in every CTA and the Q3_K down kernel stays below 90% of bandwidth.

## What the weights look like

MiMo-V2.6-Flash-RL: 48 layers (layer 0 dense, 47 MoE), hidden 4096, 256 routed experts, top-8, no shared
expert, expert intermediate 2048. Per expert, three MXFP4 matrices, exactly as stored in the safetensors:

| tensor | dtype | shape | meaning |
|---|---|---|---|
| `experts.N.{gate,up}_proj.weight` | U8 | [2048, 2048] | 2048 x 4096 E2M1 codes, element 2i in the low nibble |
| `experts.N.{gate,up}_proj.weight_scale` | U8 | [2048, 128] | E8M0 exponent per 32 inputs |
| `experts.N.down_proj.weight` | U8 | [4096, 1024] | 4096 x 2048 codes |
| `experts.N.down_proj.weight_scale` | U8 | [4096, 64] | |

One matrix is 4.46 MB, one expert 13.37 MB, the 8 routed experts of a layer 107 MB, so decoding one token
reads **5.03 GB of expert weights** (47 layers). Real exponents span 118..122 (2^-9 .. 2^-5); zero is always
stored as -0 (code 8, 15% of all codes in layer 1). Expert e of every layer lives in
`model_pp0_ep{e/4}_shard0.safetensors`. Attention (`qkv_proj`) and the dense layer are FP8 e4m3 with 128x128
block scales and `o_proj` is BF16; those are not part of this study.

Nibble order: element 2i in the low nibble is the OCP/compressed-tensors convention that llama.cpp's
converter assumes for this layout (`conversion/base.py: repack_mxfp4_blocks`). It does not affect any speed
number here, and every kernel is self-consistent with the fp64 reference built from the same convention.
Proving it against the original BF16 weights is not possible from this checkpoint alone.

**Capacity:** the full checkpoint is about 166 GB, more than this machine's 125 GB RAM + 16 GB VRAM. An
end-to-end run here needs a REAP-pruned or requantized expert set, or NVMe streaming
(`docs/MIMO_V26_TR16_B550.md`).

## Research summary

**The GPU has FP4 and INT8 tensor cores.** The starting assumption was "no hardware FP4/INT4, FP8 matrix
cores, no INT8 matrix cores". The PTX ISA and the measurement below say otherwise: sm_120 runs
`mma.sync ... s8` (sm_80+), `e4m3` (sm_89+) and, with the arch-specific target `sm_120a`, the block-scaled
`kind::mxf4` (m16n8k64, two E8M0 scales per row) and `kind::mxf8f6f4` instructions. There is no tcgen05/TMEM
(that is datacenter Blackwell only). llama.cpp already uses `kind::mxf4` (`ggml-cuda/mma.cuh`).

**The two papers.**
- arXiv 2610.00432 (XOR-Trellis, Sep 2026) is a ~2.25-bit trellis code with E2M1 palettes, not MXFP4. It has
  no CPU/GPU kernel and no measured speed. Its "4-entry palette, permuted by state bits" decode would map to
  PRMT/pshufb, which matters only for a later 2-bit repack, so it is not benchmarked here.
- arXiv 2605.13915 (Multi-Scale Dequant, May 2026) keeps weights packed and splits the *activation* into two
  low-bit parts, so a native low-bit GEMM runs twice: y = W·x1 + W·x2. It is pointless for bandwidth-bound
  decode and a candidate for prefill. Implemented here as `k5_msd_mxf4x2`: MXFP4 main part + MXFP4 residual,
  two `kind::mxf4` passes.

**Known techniques** (llama.cpp CUDA `vec_dot_mxfp4_q8_1` and AVX2 `ggml_vec_dot_mxfp4_q8_0`, vLLM Marlin
W4A16, Triton `matmul_ogs`):
- E2M1 x 2 = {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12} is an exact int8. A 16-entry byte table looked up with PRMT
  (GPU) or pshufb (AVX2) turns 8 codes into int8 for dp4a / vpmaddubsw / int8 mma, with the E8M0 scale x 0.5
  applied per block.
- E8M0 is a power of two, so the scale is an exponent, never a rounding.

**What this study adds** (all verified bit-exact against fp64, see Accuracy):
- **No weight repack.** Every kernel reads the checkpoint bytes as stored. Instead the activation is permuted
  once per token into "evens then odds" order inside each 32-block. One 32-bit load of packed codes then
  splits into the low-nibble half and the high-nibble half, and both halves line up with consecutive
  activation bytes (pshufb on AVX2, dp4a / int8 / fp8 / fp4 mma fragments on CUDA).
- **fp16 decode by table.** Every E2M1 value is an fp16 with a zero low byte, so one PRMT table lookup gives
  the fp16 high bytes and one `prmt & 0xff00ff00` builds a `half2`. With the scale (2^-9..2^-5) folded in by
  `hmul2`, weights become exact fp16. The fp16 tensor-core path and the fp16-activation GEMV need no
  activation quantization.
- **AVX2 offset table.** The table holds (value + 12) >= 0, so `vpmaddubsw` takes the weight as its unsigned
  operand directly; the 12·sum(x) correction is a per-token constant per block. This drops the two
  `vpsignb` of the ggml kernel.
- **fp8 scale folding** (`k1_f8fold`): E2M1 · 2^(e - rowmax) is still an exact e4m3 while a row's exponent
  spread is small (exponents 118..122 in the real layer-1 experts checked; the real-weight selftest is exact). The fp8 mma then accumulates over all of K with a single per-row
  scale at the end.

## Kernels

| id | device | weights decoded to | activations | instruction |
|---|---|---|---|---|
| c1_f32 | CPU | int8 -> fp32 | fp32 | FMA |
| c2_q8_sign | CPU | int8 (ggml) | q8 per 32 | vpsignb + vpmaddubsw |
| c3_q8_off_rR / c3h / c4_colsN | CPU | int8 + 12 | q8 per 32 | vpmaddubsw (R rows, hadd tree, N tokens) |
| g1_dp4a | GPU | int8 (PRMT) | q8 per 32 | dp4a |
| g2_hfma2 / g2_hmul_f32 | GPU | fp16 (PRMT high byte) | fp16 | hfma2 / hmul2 + fp32 add |
| g3_moe_fused | GPU | int8 | q8 (made in-kernel) | dp4a, gate/up+SwiGLU+q8(h), then down+combine: 2 launches |
| k0_i8_mma | GPU | int8 | q8 per 32 | mma s8, float epilogue per block |
| k1_f8fold_mma | GPU | e4m3 with folded exponent | e4m3 per token | mma e4m3 |
| k2_f16fold_mma | GPU | fp16 with folded scale | fp16 | mma f16 |
| k3_mxf8f6f4 | GPU (sm_120a) | e2m1 (8-bit container, bits 5:2) | MXFP8 e4m3 | kind::mxf8f6f4 scale_vec::1X |
| k4_mxf4 | GPU (sm_120a) | raw checkpoint bytes | MXFP4 | kind::mxf4 scale_vec::2X m16n8k64 |
| k5_msd_mxf4x2 | GPU (sm_120a) | raw checkpoint bytes | MXFP4 + MXFP4 residual | kind::mxf4 twice |

The tensor-core kernels share one grouped-GEMM main loop (`gpu/mma_native.cu`):
- 8 warps cover 128 weight rows times BN tokens of one expert tile.
- A 3-stage cp.async pipeline feeds K chunks of 128 elements (256 for `kind::mxf4`).
- B fragments come from `ldmatrix.x4`.
- `blockIdx.x` walks the row tiles of one (expert, token tile), so that tile's activations and the expert's
  weights stay in L2. With the opposite order every wave swept all activations, more than the 32 MB L2, and
  the GEMM was 1.5-2x slower.

## Running

```
cmake -S experiment/fp4 -B experiment/fp4/build -G Ninja -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build experiment/fp4/build
experiment/fp4/build/bench_cpu --selftest                  # fp64 checks, dummy weights
experiment/fp4/build/bench_gpu --mode selftest --weights real  # fp64 + exactness checks, real layer-1 experts
experiment/fp4/build/bench_gpu --mode peak                 # tensor-core peak per instruction
experiment/fp4/scripts/run_all.sh                          # everything below, CSV into results/<date>/
experiment/fp4/scripts/perf_cpu.sh                         # perf stat per CPU kernel
experiment/fp4/scripts/ncu.sh                              # Nsight Compute (needs counter access, see script)
```

`bench_gpu` options: `--mode selftest|peak|decode|moe|prefill|all`, `--weights dummy|real`, `--layer L`,
`--experts P` (pool size, default 256 = one full layer, 3.4 GB of VRAM), `--tokens`, `--ncols`, `--kinds`,
`--csv`. `bench_cpu`: `--threads`, `--ncols`, `--sets` (expert sets rotated so nothing stays in cache),
`--weights real --layer L`, `--moe-only`, `--csv`.

## Not tested

- Nsight Compute counters: this account gets `ERR_NVGPUCTRPERM`. `scripts/ncu.sh` says how to enable them.
- Batched decode across *different* experts per token. The N-token numbers assume all N tokens use the same
  8 experts, which is the upper bound on weight reuse; real MTP drafts share only some experts.
- Activation quantization time on the GPU outside the fused MoE layer. The GEMM benchmarks quantize on the
  host and time only the GEMM.
- PCIe streaming of experts from host RAM, attention, the FP8 trunk and the LM head.
