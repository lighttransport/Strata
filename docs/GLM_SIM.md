# GLM-5.3-Flash performance simulator (`tools/sim`)

`tools/sim/glm_sim.py` estimates decode and prefill speed of GLM-5.3-Flash on a CPU + GPU machine from a
hardware description, a weight pack and the engine's knobs, without running the engine. It models Strata's
current decode pipeline (`src/program/glm_decode.cpp`) and is calibrated on the measurements in this
directory, so it can be used to compare placements, packs, MTP depths and even machines while the real
machine is busy or absent. It is an estimate: on the rows it was calibrated on it is within 5 % on average,
and on the other recorded runs within 10-25 % (see "Validation").

Run it with the system `python3` (numpy is needed only for routing-trace replay):

```sh
python3 tools/sim/glm_sim.py estimate --hw tr16 --pack q23 --mtp 2 --decode-cache-mib 4800 --acceptance prime
python3 tools/sim/glm_sim.py sweep --hw tr16 --affinity 0 0.05 0.1 --decode-cache-mib 3000 4800 6500
python3 tools/sim/glm_sim.py validate
python3 tools/sim/glm_sim.py plan --hw tr16 --target-decode 30 --lossless
python3 tools/sim/glm_sim.py estimate --hw tr16 --set pcie.h2d_gbps=25 --set memory.dram_gbps=180
python3 tools/sim/glm_sim.py hw --hw tr16 > my-pc.json     # edit, then --hw my-pc.json
python3 -m unittest tools.test_sim
```

## Hardware input

A hardware config (`tools/sim/hw.py`) has: CPU cores, clock, NUMA nodes, SIMD width (FP32 FMA peak is derived
as 16 flop/cycle/core for AVX2, 32 for AVX-512, or given with `cpu.gflops`); DRAM bandwidth the expert kernel
can stream (`memory.dram_gbps`) and per-core bandwidth; GPU VRAM, bandwidth, GEMV efficiency, routed-expert
kernel rate, launch cost, VRAM held by the desktop; PCIe generation and lanes or a measured H2D rate; disk read
rate; and, for remote expert tensor parallelism, a link (round trip, bandwidth) and a worker node (expert-row
bandwidth, RAM). Presets:

| Preset | Machine | Source of the numbers |
|---|---|---|
| `tr16` | Threadripper 1950X, 2 NUMA nodes, 125 GiB DDR4, RTX 5060 Ti 16 GB, PCIe Gen3 x8 | 72 GB/s plain 15-thread stream with the q23 kernel, 7.18 GB/s H2D (nsys), 380-393 GB/s dense GEMVs, 235 GB/s tier kernel (GLM_Q2_DECODE_REDESIGN.md, GLM_SINGLE_OPTIMIZATION.md) |
| `b550` | Ryzen 9 3950X, 1 node, 60 GiB, RX 9070 XT | GLM_B550_REAP50.md, GLM_REMOTE_TP_COMM.md (26 GB/s expert rows) |
| `xeon_v100` | 2 x Xeon Gold 6240, 2 x V100 32 GB | GLM53_V100.md; not validated |
| `tr16+b550-ib`, `tr16+b550-1gbe` | tr16 decoder with b550 as remote worker | GLM_REMOTE_TP_COMM.md (20.9 us / 401 us round trips) |

Only `tr16` is validated; the other presets exist to explore.

## What is modeled

**Model and packs** (`model.py`): 45 layers, 3 dense, 42 MoE x 288 experts, top-8, one shared expert, 11 MLA
and 34 KDA mixers, one MTP block. Per-layer expert formats of the shipped UD-Q2_K_XL (`q2_orig`, 99.03 GB),
the `q23` pack (Q2_K gate/up + Q3_K down, blk.11/12/44 kept, 111.19 GB), `q22`, REAP-50 in q23 (144 experts,
55.09 GB) and Q4_K_M, a hypothetical Q8_0 and EXL3 (bytes only). The byte census of q2_orig, q23 and REAP-50
q23 equals the GGUF headers exactly. Fixed weights: 6.45 GB streamed by the GPU per token.

**Decode step** (`decode.py`), width `nt` tokens (1 ordinary, depth+1 for an MTP verify) over `rows` sequences:

```
step = dense layers + sum over MoE layers + head/tail
unsplit layer = max(CPU experts, GPU shared + resident experts) + GPU mixer/router chain + mailbox handoff
split verify  = max(cpu_A, gpu_B) + max(cpu_B, gpu_A)      groups A = nt/2 tokens, B = the rest
CPU experts   = max(bytes / bandwidth, MACs / quantized-dot rate) + per-layer overhead
```

CPU expert bytes are the distinct experts of the window (measured union ratios 1.82x / 2.45x / 3.27x for
2 / 3 / 4 consecutive tokens; independent rows use the uniform formula) minus the share served by the GPU
tier. The compute side uses a per-core quantized-dot rate: 63 % of FP32 FMA peak for Q2_K/Q3_K (17 GMAC/s per
Zen 1 core, from 6 GB/s out of L2), 30 % for IQ2_XS/IQ3_XXS, which is why the original pack runs at 38 GB/s
and is compute-bound while q23 reaches 65-72 GB/s. GPU: dense GEMVs at `bandwidth x gemv_efficiency` with a
small per-column factor, 44 launches per layer, resident experts at the tier rate, a fixed per-layer cost for
router, mHC and mailbox kernels. MTP rounds add `depth` draft steps (draft MLA, 8 CPU draft experts, LM head),
the verify step and one resync step per accepted token; tokens per round come from per-position acceptance
profiles (`prime`, `json`, `csv`, `coding`, `mixed`) fitted to the measured accepted/proposed counts.

**GPU tier** (`routing.py`): share of routed bytes served by a tier holding a fraction `f` of the 12,096 expert
slots: `h = 1.16 f^0.545`, fitted to the static prior fill (18.1 / 20.5 / 25.5 % at 400 / 500 / 750 slots).
Policies: `static` (ranked by the prompt's routes only, half of that), `static_prior`, `adaptive` (x1.08).
Route affinity `x` raises the resident share by `(1 - h)(1 - exp(-x / 0.22))`, which reproduces 28.6 / 37.9 /
52.8 / 62 % at 0.02 / 0.05 / 0.10 / 0.15. With `--trace build-q2-v3/trace-prime.routes` the union ratios and
hit rates are instead replayed through `tools/glm_residency_sim.py`.

**VRAM** (`vram.py`): dense copies (6,154 MiB with Q8 decode), MTP dense, verify history per depth, state
(145.6 MiB + 0.0228 MiB per context token), staging ring, prefetch buffers, prefill scratch, reserve, desktop.
The leftover is the tier; the v5 run (budget 14,336, reserve 512, request 5,632) reproduces 4,900 MiB / 559
slots against the measured 4,908 MiB / 565.

**Placements**: `cpu` (default), `gpu_stream` (experts streamed over PCIe per token, `decode_experts=gpu`),
`ram_tier` for hosts whose RAM does not hold the pack (`exact` fetches misses from disk, `frozen` restricts
routing to the resident set, `hybrid` adds a margin), remote tensor parallelism (`--remote-share` of each
expert's rows on the worker; per layer `max(local, remote + round trip)`; the worker's share also lives in the
worker's RAM), `--gpus 2` (a second tier), `--batch M` independent sequences.

**Algorithms from the literature** (docs/GLM_DECODE_RESEARCH.md): `--tier-policy lru` (recency cache, fitted to
an LRU replay of the routing traces), `--pcie-prefetch 1` (next-layer routes predicted one layer ahead and
uploaded during the CPU pass; lossless), `--expert-deferral 1` (routed output added one layer late so the GPU
chain never waits; lossy), `--spec-tail-topk N` (fewer experts for draft positions >= 2; lossy).

**Architecture knobs for what-ifs**: `--pcie-share` streams a share of each layer's non-resident experts over
PCIe for the GPU to compute alongside the CPU pass; `--tier-compress` stores resident experts in a denser format
(more slots, lossy); `--param name=value` overrides any calibrated parameter (for example
`gpu_kernels_per_layer=10` for graphs, `cpu_quant_scale=2` for a faster dot kernel); `--set gpu.tier_gbps=380`
for a better resident-expert kernel. `estimate` prints the CPU-expert-bound and GPU-bound ceilings of the
configuration. docs/GLM_DECODE_ARCHITECTURE_PLAN.md uses these to rank the paths to 50 tok/s.

**Prefill** (`prefill.py`): every chunk streams the touched 16-expert groups over PCIe (all of them above about
64 tokens, 111 GB for q23); `chunk time = max(PCIe + disk + layer sync, GPU fixed + per-token work)`. The GPU
side is fitted (4 s + 2.0 ms/token on the 5060 Ti): 8K chunks are GPU-bound (~20 s), 1K chunks are
PCIe-bound (15.5 s). `--prefill-legacy` reproduces the pre-optimization state (FP32 MLA, no dequant-once).

## Validation

`validate` runs 32 recorded configurations (`tools/sim/data/measured_tr16.json`, each with its source). All are
within their tolerance (15 % for quiet runs, up to 40-60 % for the noisy, batch and disk-bound rows). Mean
absolute error over the 15 calibration rows is 4.8 %. Selected rows:

| Case | Measured tok/s | Predicted | Error |
|---|---:|---:|---:|
| q2_orig ordinary, no tier | 9.74 | 9.83 | +1 % |
| q23 ordinary, static tier 3.5 GiB | 13.1 | 13.6 | +4 % |
| q23 MTP2 split, adaptive 4.8 GiB, prime | 21.0 | 21.8 | +4 % |
| q23 MTP2 split, static 4.8 GiB, prime | 18.0 | 19.9 | +10 % |
| q23 MTP2, affinity 0.02 / 0.05 / 0.10 / 0.15 | 24.5 / 25.4 / 27.8 / 28.3 | 23.2 / 24.3 / 25.9 / 27.3 | -5 / -5 / -7 / -4 % |
| q23 v5 selected (affinity 0.10), 1K/1K | 22.6 | 23.7 | +5 % |
| q23 MTP1 lossless (v11) | 19.5 | 20.1 | +3 % |
| REAP-50 q23 MTP1, Q8 dense | 23.3 | 21.0 | -10 % |
| 64 GB exact / frozen / hybrid | 2.4 / 23 / 15.1 | 2.9 / 19.4 / 19.0 | approximate |
| remote TP over IB, 75 % local | 17.1 | 15.6 | -8 % |
| batch of 8 sequences, MTP2 | 23.4 | 32.4 | +38 % (acceptance of the 8 prompts unknown) |
| prefill q23 1K / 8K | 66 / 395 | 64 / 402 | -3 / +2 % |
| prefill REAP-50 1K / 8K | 103 / 407 | 124 / 402 | +21 / -1 % |

The affinity series also reproduces the measured turn to a GPU-bound round at 0.10, and the ordinary-decode
rows the measured 24 ms GPU and 72 ms CPU per step. The free parameters are in `kernels.Params`; `fit`
re-tunes them (coordinate grid search) after a model change, but the shipped defaults are the calibrated set.

## Speculation methods

`--speculation mtp` is GLM's own draft block: `depth` sequential draft steps per round (each about 3.5 ms:
draft MLA, 8 CPU draft experts, LM head), a verify step of `depth + 1` tokens, and one resync step per accepted
token. `--speculation dflash` is a block-diffusion drafter in the style of DFlash / DFlash2 (z-lab): one drafter
forward per round proposes a block of `--draft-block` tokens (7 drafts + anchor at the default 8), conditioned on
the target hidden states the verify step already produced; no recurrent state, so no resync. Its weights live on
the GPU (`--draft-model-mib`, default 1,536) and come out of the expert tier. Acceptance profiles `dflash_chat` /
`dflash_code` / `dflash_math` reproduce the published accepted lengths 4.10 / 4.39 / 5.46 of DFlash2 at block 8.
The engine caps a step at 8 tokens (`cpu::MAXT`); `--max-verify-width 16` lifts the cap in the model to explore.

Union bytes of wide windows come from the routing traces in `build-q2-v3` (2 / 3 / 4 / 6 / 8 / 12 / 16 tokens
touch 1.78 / 2.46 / 3.10 / 4.3 / 5.3 / 7.1 / 8.7x one token's experts; fitted as `1.05 w^0.763`).

| q22 pack, adaptive tier, prime/code text | tokens per round | round ms | CPU GB per token | tok/s |
|---|---:|---:|---:|---:|
| none | 1.00 | 66 | 2.4 | 15.3 |
| mtp depth 1 / 2 / 3 | 1.97 / 2.89 / 3.72 | 88 / 122 / 157 | 2.1 | 22.5 / 23.7 / 23.7 |
| dflash block 8, 1.5 GB drafter | 5.42 | 271 | 2.8 | 20.0 |
| dflash block 8, 512 MiB drafter | 5.42 | 257 | 2.7 | 21.1 |
| dflash block 16 (cap lifted), code / math | 7.46 / 10.14 | 567 | 3.7 / 2.7 | 13.2 / 17.9 |

Why a block drafter does not help here: with CPU-resident experts the verify cost is the distinct expert bytes of
the window, and that grows almost as fast as the accepted tokens (5.3x bytes for 5.4 tokens at block 8, versus
2.8x bytes for 2.9 tokens at MTP depth 2), so CPU bytes per delivered token stay near 1x. A block drafter only
pays off when expert reads are not the bottleneck: experts on the GPU (hit rate above about 60 %, or a second
card), or a CPU with several times the bandwidth. It also costs tier VRAM. The same arithmetic says MTP depth 3
gains nothing over depth 2 on this machine, which matches the measured D02 result.

## Paths to 30 tok/s on this machine (single GPU, 124 GiB)

Cumulative levers on the prime fixture; "lossless" means the greedy output equals the model's own:

| Step | Change | tok/s | CPU GB/token | Lossless |
|---|---|---:|---:|---|
| A | q23, MTP2, adaptive 4.0 GiB tier (today) | 22.0 | 2.36 | yes |
| B | q22 pack (Q2_K down; held-out ppl 4.867 vs 4.825) | 23.7 | 2.13 | yes, as a different quantization |
| C | B + MTP3 or dflash | 23.7 / 21.1 | 2.14 / 2.67 | yes |
| E | B + MTP3 + Q4_K dense copies (+840 MiB tier) | 24.3 | 2.07 | no (dense precision) |
| F | E + reserve 256 MiB, prefill scratch 512 MiB | 24.8 | 2.02 | no |
| G | F + gate-threshold expert skip 10 % | 26.8 | 1.82 | no (measured: 28 % of units skipped cost ~9 % bytes) |
| H | G + route affinity 0.05 | 31.3 | 1.45 | no (KL 0.143, top-1 87.5 %) |
| J | G + route affinity 0.10 | 34.8 | 1.2 | no (KL 0.165, top-1 86.9 %); GPU-bound |

Lossless configurations top out at 23.7 tok/s (`plan --lossless`): the round would have to be 1.26x faster,
i.e. CPU expert bytes below 1.7 GB per token or 91 GB/s of sustained expert bandwidth, which 15 Zen 1 cores do
not reach. Every lossy lever buys bytes: affinity is the largest single one (0.05 gives about -25 %). With two
cards (`--gpus 2`) lossless q22 MTP3 predicts 32 tok/s and q23 29 tok/s, because the second tier removes about
45 % of the CPU bytes.

## B550 + RX 9070 XT (preset `b550`, not validated)

Ryzen 9 3950X, one NUMA node, 60 GiB RAM, 40 GB/s expert bandwidth assumed (26 GB/s was measured while the
box ran other jobs), 16 GB VRAM at 640 GB/s, PCIe Gen4 x16. Only REAP-50 q23 fits the RAM; q22/q23 need the
RAM tier. `plan --hw b550 --target-decode 30`:

| Pack | Speculation | Affinity | Skip | Dense | Placement | Decode tok/s | Prefill 1K tok/s | Lossless |
|---|---|---:|---:|---|---|---:|---:|---|
| q2_orig | MTP2 | 0 | 0 | q8 | RAM tier, exact | 5.0 | 39 | yes (disk-bound) |
| REAP-50 q23 | MTP3 | 0.10 | 0.1 | q4 | cpu | 24.1 | 308 | no |
| REAP-50 q23 | MTP3 | 0.10 | 0.1 | q8 | cpu | 23.5 | 308 | no |
| REAP-50 q23 | MTP1 | 0.10 | 0.1 | q4 | cpu | 22.9 | 308 | no |
| q22 | MTP3 | 0.10 | 0.1 | q4 | RAM tier, frozen | 21.8 | 37 | no |

Lossless decode is disk-bound at 5 tok/s because no lossless pack fits 60 GiB. The best lossy configuration
reaches 24 tok/s with 120 ms of CPU experts against 91 ms of GPU work per round, so the 9070 XT is close to a
second limit; 30 tok/s would need CPU expert bytes under 0.9 GB per token or about 50 GB/s of sustained expert
bandwidth. If the 3950X's quiet rate is the measured 26 GB/s rather than 40, all decode figures drop by about
a third.

DFlash2 block drafting on this box (REAP-50 q23, 512 MiB drafter; MTP rows use the `mixed` profile):

| Method | Tokens per round | Round ms | Decode tok/s |
|---|---:|---:|---:|
| MTP depth 2 | 2.64 | 188 | 14.1 |
| DFlash block 4, chat / code / math | 3.2 / 3.4 / 3.6 | 227 | 14.2 / 14.8 / 16.0 |
| DFlash block 6 | 4.3 / 4.5 / 5.1 | 318 | 13.4 / 14.2 / 16.1 |
| DFlash block 8 (DFlash2 default) | 5.0 / 5.4 / 6.4 | 423 | 11.8 / 12.8 / 15.1 |

Block drafting is a wash against MTP here and gets worse with the block size, for the same reason as on tr16
but stronger: at 35-40 GB/s the verify step's distinct expert bytes (5.3x one token at block 8) dominate the
round. Math-style text with 6.4 accepted tokens is the one case where blocks of 4-6 beat MTP, by about 2 tok/s.
With q22 in a frozen RAM set, DFlash8 gives 12.0 vs MTP2's 13.4 tok/s (14.4 vs 15.5 with affinity 0.05).

## What the model says about this machine

- Single-stream decode is bound by the CPU expert pass: 2.4 GB of expert bytes per token at 63-65 GB/s, with
  the GPU chain hidden under it by split verify. Doubling DRAM bandwidth alone gives only +12 %, because 15
  Zen 1 cores saturate near 75 GB/s on the Q2_K/Q3_K dot (the roofline turns compute-bound).
- Lossless 30 tok/s is out of reach with one card (`plan --lossless` tops out at 23.7 with q22, MTP2/3);
  affinity 0.10 with MTP3 on q22 or REAP-50 reaches 32-35 but becomes GPU-bound (dense GEMVs twice per split
  round plus resident experts). See "Paths to 30 tok/s".
- Prefill at 1K prompts is PCIe-bound (15.5 s per chunk at 7.2 GB/s); at 8K chunks it is GPU-bound, so a faster
  PCIe link would raise short-prompt prefill 2.5x but leave 8K prefill unchanged.

## Limits

- Routing is synthetic unless a trace is given; the tier curve is fitted on q23 traces of coding fixtures.
- The 64 GB modes, batch decode and remote TP have single noisy measurements each; treat them as +-30 %.
- Context-length effects on attention are small in the model (MLA uses a top-2048 indexer); very long contexts
  are not validated.
- Nothing here measures the machine: all inputs come from `docs/` and `build-q2-*` traces.
