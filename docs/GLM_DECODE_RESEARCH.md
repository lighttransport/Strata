# GLM-5.3-Flash on 128 GB RAM + 16 GB VRAM: what others do, and what the simulator says about it

Survey date: 2026-10-08. Question: does anyone run a ~100 GB GLM-5.3-Flash faster than Strata's 20-24 tok/s on
a 128 GB + 16 GB class machine (or on less: 64 GB + 8 GB), and which published algorithms transfer? Every
algorithm judged transferable was added to `tools/sim` as a knob and evaluated on the tr16 model
(docs/GLM_SIM.md). Predictions below are simulator outputs, not measurements.

## 1. Reported GLM-5.3-Flash speeds elsewhere

| Setup | Engine / quant | Decode tok/s | Source |
|---|---|---:|---|
| Strix Halo 128 GB, DGX Spark (unified ~256 GB/s) | llama.cpp mainline, 1-3 bit | 15-18 | unsloth docs |
| Strix Halo 128 GB | EXL3 | 26-30 | llamaperf |
| 2x RTX 5090 + host RAM | llama.cpp IQ4_XS | 22 | llamaperf |
| 4x RTX 3090 | 3-bit | ~28 | llamaperf |
| RTX 3090 + host RAM | Strata UD-Q4_K_XL | 19.8 | llamaperf |
| 2x DGX Spark | vLLM NVFP4 | ~40 | llamaperf |
| M3 Ultra 256 GB | Q4 | 37 | llamaperf |
| 2x CMP 170HX 64 GB (all in VRAM) | EXL3 3.05 bpw | 96 | llamaperf |
| 1x B200 (all in VRAM) | llama.cpp, MTP n=2 | 58.6 -> 86.5 | unsloth docs |
| **tr16: 1950X + 5060 Ti 16 GB, 124 GiB** | **Strata q23, MTP2** | **20-24** | docs/GLM_SINGLE_OPTIMIZATION.md |

No CPU-offload setup in the survey beats 24 tok/s with a single consumer GPU; the faster ones hold the whole
model in GPU or unified memory. Unsloth's quant table also settles the 64 GB question: even UD-IQ1_S is 93 GB
(top-1 agreement 70.9 %), so no quantization of the full model fits 64 GB; only pruning (REAP-50, 55 GB) or an
expert store outside RAM does.

## 2. Algorithms found, and how they fared in the simulator

Base for the lossless rows: q22 pack, MTP depth 3, split verify, adaptive 3.9 GiB tier, prime fixture
(23.7 tok/s). Base for the lossy rows: the 50 tok/s stack of docs/GLM_DECODE_ARCHITECTURE_PLAN.md (M4, 50.7).

| Idea | Where from | Simulator knob | Lossless base | Lossy stack | Verdict |
|---|---|---|---:|---:|---|
| GPU expert cache with LRU/recency policy | llama.cpp PR 29887 (72-89 % hits at 10-20 % coverage on Qwen3.8-Flash-Next), RFC 24528, FATE/llama-moe-cache (99.5 % on Qwen3-30B) | `--tier-policy lru` | 23.7 -> 24.5 | 50.7 -> 52.9 | GLM routes are less skewed: ideal LRU on our traces hits 27 % at today's size, 41 % at 10 %, 58 % at 20 %, only 3-5 points above the frequency policy. Worth having, not a breakthrough |
| Predict next-layer routes, prefetch those experts over PCIe | Pre-gated MoE, FATE (97 % accuracy), SpecPrefetch, SeqMoE (97 % hits at 45 % residency), DALI | `--pcie-prefetch 1` | 23.7 -> 25.3 | 50.7 -> 54.6 | Lossless and additive: PCIe becomes extra expert bandwidth (7.2 GB/s here, +6 %; a Gen4 x16 link would give +18 %). Needs a cross-layer gate predictor |
| Expert deferral: add part of the routed output one layer late so the GPU never waits | KTransformers SOSP'25 (CPU utilisation <75 % -> ~100 %, up to 1.45x, accuracy drop <= 0.5 %) | `--expert-deferral 1` | 23.7 -> 23.8 (split verify already overlaps); ordinary decode 15.3 -> 23.9; batch 4 ordinary 24.3 -> 31.0 | +0.4 | The same gain Strata gets from split verify, but for any window shape and without duplicate dense reads. Lossy |
| Verifier expert-set sizing by draft position | AcceptMoE (73-77 % less H2D traffic, -0.27 pt accuracy) | `--spec-tail-topk 4` | 23.7 -> 28.3 (MTP3), 29.5 with MTP5 | 56.9 -> 64.2 | Positions >= 2 of the window route to 4 experts: window bytes and MACs fall 20-25 % for a 2 % acceptance loss (the AcceptMoE penalty). The largest single addition; lossy, needs the quality gates |
| Cost-aware draft selection (reuse already-active experts) | EcoSpec (up to 1.62x on GPU-resident MoE) | not modelled | | | Would lower the window union below 2.46x for 3 tokens; no numbers to calibrate on |
| Dynamic expert skipping (30-50 % of routed experts) | ACE (50 % skip on Qwen3.6-35B-A3B with small loss), SERE, EAT, MoBiLE (half experts for unimportant tokens, 1.6-1.7x) | `--expert-skip` (existing) | | | Literature supports 20-30 % as realistic; the plan assumes 10-20 % |
| Block-diffusion drafter | DFlash2 (accepted length 4.1-5.5 at block 8) | `--speculation dflash` (existing) | 23.7 -> 21.1 | 43 -> 50.7 | Pays only when experts stop being the bottleneck (see the plan) |
| Self-speculation with one GPU-resident draft expert per layer | DraftExpert (84-87 % acceptance, 1.45x on DeepSeek-V2-Lite) | not modelled | | | A drafter that reuses the target's attention; costlier per draft than DFlash, no extra model |
| Experts in flash with direct GPU paths | "Beyond Capacity" (8 NVMe, 28 GB/s), MoE-Infinity | `--set disk.fetch_gbps` with `--placement ram_tier` | | | For 64 GB hosts: 6 / 12 / 25 GB/s of expert fetch give 10.7 / 13.4 / 15.3 tok/s lossless (q22, MTP2, 8 GB VRAM), versus 3.6 at 1 GB/s and 18.7 for REAP-50 in RAM |
| Mixed-precision cold experts, on-the-fly decompression | HOBBIT, FloE (9.3x per-expert compression, 4-8 % degradation) | `--tier-compress` (resident side) | | | The CPU side is bandwidth-bound, so a denser CPU format only helps if its kernel stays memory-bound (the IQ formats do not: 37-42 GB/s) |
| AMX CPU kernels | KTransformers, CoX-MoE, TriMoE | `--param cpu_quant_scale` | no change below 8 rows | | tr16 has no AMX and is memory-bound; matters for batched decode on Xeons |
| Training routers for cache locality | "Cacheable by Design?" (negative result: miss reduction and quality tightly coupled; inference-time cache-aware rerouting ~80 % miss reduction at <= 3.4 % ppl) | `--affinity` (existing) | | | Confirms Strata's inference-time affinity is the right side of that trade |

Combined, the lossless additions move the single-GPU figure from 23.7 to 26.1 tok/s (LRU + prefetch); adding the
verifier tapering gives 31-33 at a small quality cost; on the lossy 50 tok/s stack the three together give 64,
and 75 with two rows. Lossless 30 tok/s single-stream stays out of reach with one card: the CPU still streams
1.9 GB per token at 63 GB/s.

## 3. For less hardware

- **64 GB RAM + 8 GB VRAM**: 8 GB holds the dense weights only, so there is no expert tier. Lossless decode is
  bound by the expert store: NVMe at 1 GB/s gives 3.6 tok/s; a 4-8 drive array at 12-25 GB/s (the "Beyond
  Capacity" design) gives 13-15 tok/s. REAP-50 q23 in RAM gives 18.7 (20 with deferral + prefetch) and is lossy.
- **64 GB RAM + 16 GB VRAM**: see docs/GLM_SIM.md "B550"; lossy REAP-50 reaches 24 on a 3950X.
- Quantizing below 2 bits does not free RAM: UD-IQ1_M is 97.6 GB.

## 4. Trace facts used above (build-q2-v3 routing traces, q23 pack)

- Adjacent-token expert reuse: 23 % of a token's routes were used by the previous token in the same layer
  (chance 2.8 %, i.e. 8x chance; Qwen3-30B is reported at 2x chance).
- Distinct experts per window of 2 / 3 / 4 / 8 / 16 tokens: 1.78 / 2.46 / 3.10 / 5.3 / 8.7x one token.
- Ideal per-layer LRU hit rate vs tier coverage: 3.3 % -> 20.8 %, 4.7 % -> 26.9 %, 10 % -> 41.2 %,
  13.4 % -> 47.7 %, 20 % -> 57.7 %, 31 % -> 69.9 %. A recency cache with only 16 promotions per token and no
  frequency gate hits under 10 %, which is why Strata's decayed-frequency admission is right for a slow PCIe link.

## Sources

- Unsloth, GLM-5.3-Flash: how to run locally. https://unsloth.ai/docs/models/glm-5.3-flash
- llamaperf, GLM-5.3 local performance by GPU. https://llamaperf.com/model/glm-5-3
- llama.cpp PR 29887, GPU cache for MoE experts kept in host memory. https://github.com/ggml-org/llama.cpp/pull/29887
- llama.cpp discussion 24528, MoE expert cache RFC. https://github.com/ggml-org/llama.cpp/discussions/24528
- llama-moe-cache (FATE in llama.cpp). https://github.com/ongunm/llama-moe-cache
- KTransformers, SOSP 2025. https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/
- Pre-gated MoE / FATE cross-layer gate. https://arxiv.org/abs/2502.12224
- SpecPrefetch. https://arxiv.org/abs/2607.24787
- SeqMoE. https://arxiv.org/abs/2609.12978
- DALI. https://arxiv.org/abs/2602.03495
- AcceptMoE. https://arxiv.org/abs/2608.02989
- EcoSpec. https://arxiv.org/abs/2607.12696
- DraftExpert. https://arxiv.org/abs/2607.24434
- ACE expert skipping. https://arxiv.org/abs/2609.05228
- MoBiLE. https://arxiv.org/abs/2510.12357
- SERE (ICLR 2026). https://arxiv.org/pdf/2602.07616
- Cacheable by Design? https://arxiv.org/abs/2608.18261
- Beyond Capacity (flash with direct GPU paths). https://arxiv.org/abs/2608.14333
- FloE. https://arxiv.org/abs/2505.05950
- Cloud-grade SLOs for local MoE inference. https://arxiv.org/abs/2606.10493
- DFlash / DFlash2. https://arxiv.org/abs/2602.06036 , https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2
