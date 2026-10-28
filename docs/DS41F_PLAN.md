# DeepSeek V4.1 Flash in Strata: quantization per machine, forecasts, and the run plan

Written 2026-10-18. Everything not marked "measured" is a simulator output with a stated calibration and a printed
hypothesis list. Nothing was run on the A64FX cluster or the GPU node, and no weights were downloaded for this document.

The question: run DeepSeek V4.1 Flash on (1) TR16 (128 GB RAM + RTX 5060 Ti 16 GB), (2) TR16 + B550 over
InfiniBand (192 GB RAM, two 16 GB GPUs), (3) the Xeon box (181 GB RAM + 2x Tesla V100 32 GB), (4) A64FX nodes
(32 GiB HBM2 each, `/local` SSD at about 80 MB/s per node). Which quantization on each, what speed, and what has
to be built. The user's premise holds: the Engram n-gram tables are read sparsely (48 rows of 264 B per token) and
stay on SSD in their released FP8 form on every machine; only the routed experts and the trunk are quantized.

## 1. The model, in bytes

Sources: the gemm repository's native A64FX runner (`a64fx/ds41f/ds41f_model.h`, `a64fx/doc/ds41f.md:2516-2521`), its
NumPy reference (`reference_numpy.py`), the Hugging Face card, and the public GGUF conversions.

| Item | Value |
|---|---|
| Layers | 40, all MoE; causal encoder-decoder (20 + 20); 16B parameters active per token in decode, 8B in prefill |
| Hidden / vocab / context | 5120 / 129,280 / 1,048,576 |
| Attention | CSA2: 64 heads x 512, q_lora 1280, kv_lora 512, o_lora 1024; KV sources at layers 2, 8, 14, 20 reused by the other layers; indexer sources at 2, 8, 14, 20, 24, 28, 32, 36; top-512 selected rows + 128-token window; compressed KV in FP4; **890 B of KV per token, 0.93 GB at 1M tokens** |
| MoE | 384 routed experts, top-6 with bias, 1 shared expert, intermediate 2304; 35.4M parameters per expert |
| Routed experts | **543.6B params = 288.78 GB in the released MXFP4** (two E2M1 nibbles per byte, one E8M0 scale per 32 inputs; 18.8 MB per expert); **4.51 GB per decoded token** at MXFP4, 2.4 GB at 2.25 bpw, 1.66 GB at IQ1_S |
| Trunk | attention, shared experts (1.42 GB), router, mHC (4 streams), Engram projections, norms: **7.20 GB FP8 read per token**; embed BF16 1.32 GB (lookup); head BF16 1.32 GB |
| Engram | two hashed n-gram tables at layers 1 and 14: 384,006,168 + 384,016,682 rows x 264 B (256 FP8 + 8 E8M0) = **202,758,032,400 B**; 3 n-gram orders x 8 heads = 24 rows per layer per token, **48 random 264 B reads per token** (12.7 KB); the rows are decoded to BF16, projected (6144 -> 4 keys + 1 value of 5120) and gated into the residual streams |
| DSpark drafter | 7.93 GB, 3 stages, draft block 5; on 12 A64FX nodes speculative decoding was slower than plain (20.6 vs 24.6 tok/s) |
| Checkpoint | 510.3 GB (FP8 dense + E8M0 block scales, MXFP4 experts, FP8 Engram); 96,085 tensors in 48 shards |

Public quantization evidence on these experts (V4-Flash, same MXFP4 format; Unsloth's table): the MXFP4 repack is
bit-exact; the only sub-4-bit results are antirez's **IQ2_XXS at 86.7 GB: PPL 6.08 vs 4.53, KLD 0.41, top-1 78 %**
and a mixed Q4K/Q2 at 97.6 GB: PPL 5.82. The only V4.1 Q2 GGUF (IQ2_XXS gate/up + Q2_K down, 152.9 GB of experts)
has no quality measurement. **Every sub-4-bit candidate below therefore needs the quality gate of section 5
before it ships**; the numbers in the tables are speed, not quality.

### The 1017 refresh (checked 2026-10-18)

DeepSeek-V4.1-Flash-1017 is a weights-only update: its `config.json` is identical to V4.1-Flash's (hidden 5120, 40 layers,
384 experts top-6 with `sqrtsoftplus`/`noaux_tc` and scale 1.5, intermediate 2304, sliding window 128,
`compress_ratios` [0,0] + [2]x18 + [1]x20 + [0,0,0] for 40 layers + 3 MTP, KV sources [2,8,14,20], indexer sources
[2,8,14,20,24,28,32,36] with 32 heads x 128 and top-512 over 2048 candidate blocks of 8 from layer 20, hc_mult 4 with 20
Sinkhorn iterations, Engram layers [1,14] of 384,006,168 / 384,016,682 rows, 3 next-token-prediction layers, DSpark
block 5, FP8 weight blocks [32,32] with UE8M0 scales, `expert_dtype: fp4`), and `model.safetensors.index.json` reports
the same 510,286,023,000 bytes in 48 shards. The card calls it a minor update (reasoning, coding, agentic and visual
gains, shorter outputs) and ships no Jinja chat template. Everything in this document applies unchanged; a pack made
from 1017 carries a different source fingerprint. The simulator records these facts (`Geometry.mtp_layers`,
`compress_ratios`, `candidate_blocks`, `release`).

## 2. What the simulator says

The model is `tools/sim/ds41f.py` (CLI `tools/glm_ds41f_sim.py`). Per layer: GPU trunk GEMV (Strata's measured
rates: 5060 Ti 380 GB/s, V100 720 GB/s, 44 launches), the CPU routed experts as `max(bytes / stream, MACs / kernel
rate)` with the measured per-format kernel rates (IQ1_S 92 GMAC/s on 15 Zen 1 cores, Q2_K 255, MXFP4 41.4 GB/s;
IQ2_XXS 120 and IQ1_M 83 are hypotheses), the GPU expert tier with GLM's fitted hit curve, the shared expert and
resident experts overlapping the CPU pass, the mailbox handoff; per step the Engram page reads (48 x 4 KiB at the
measured 83 us QD1 latency, queue depth 16, 40 % row-cache hits, prefetched under the previous layer) and the head.
Two-node: the B550 takes the share of expert rows that balances the two streams within its 56 GiB, one request and
reply per layer over the measured 15.7 us / 3.26 GB/s link. A64FX: the stage structure of `tools/sim/a64fx.py` with
this model's bytes, two profiles (`current` fitted to the gemm runner's 40.7 ms split: attention 17.3, experts 9.7,
collectives ~6, Engram 0.6, other 5; `redesign` with the GLM-plan constants). Prefill: Strata's streamed-expert
model (every touched expert uploaded per chunk over PCIe, GEMM on the GPU) on the GPU machines; the per-position
cost of the current runner on A64FX (its prefill is unbatched, 33.5 tok/s measured) and an int8 GEMM roofline for
the redesign.

### Validation

```
record                                                                     measured predicted   error  ok
V4.1-Flash 12 A64FX nodes, best decode (INT8, BF16 reductions, split mHC),    24.56     24.39   -0.7%  yes
V4.1-Flash 12 A64FX nodes, DSpark speculative (emitted tokens)                20.60     21.22    3.0%  yes  (cross-check)
V4.1-Flash 12 A64FX nodes, prefill (unbatched per position)                   33.50     33.20   -0.9%  yes
V4-Flash on TR16 + RX 9070 XT (hetero/ds4f), bit-exact defaults, 1024-toke     5.02      9.08   80.8%  yes  (cross-check)
V4-Flash on TR16 + RX 9070 XT, prefill 1024 tokens (W4A8 + GPU tier B2)       14.22     46.93  230.0%  yes  (cross-check)
5 of 5 rows within tolerance
```

The two fitted rows are the gemm runner's best decode and its prefill on 12 nodes. The DSpark row (not fitted) lands
within 3 %. The two V4-Flash rows on TR16 + RX 9070 XT are cross-checks of a different runner (`hetero/ds4f`) whose
own analysis puts it at less than half its bandwidth ceiling (12 tok/s); the model predicts that ceiling, not that
runner, so they are reported with a wide tolerance and not fitted.

### Tables

Simulator outputs, not measurements. Every table's hypotheses are listed below it.

### Pack sizes

| pack | bpw | experts | GB | GB/token (6 active x 40) | lossless | formats |
|---|---:|---:|---:|---:|---:|---:|
| mxfp4 | 4.25 | 384 | 288.8 | 4.51 | yes | released MXFP4 experts, bit-exact repack |
| iq3xxs | 3.06 | 384 | 208.1 | 3.25 | no | IQ3_XXS gate/up/down (3.06 bpw) |
| q23 | 2.90 | 384 | 196.8 | 3.07 | no | Q2_K gate/up + Q3_K down, Strata's GLM pack recipe |
| iq2xxs_q2k | 2.25 | 384 | 152.9 | 2.39 | no | IQ2_XXS gate/up + Q2_K down (the public MixedQ2 GGUF, 2.25 bpw) |
| iq2s_q2k | 2.58 | 384 | 175.5 | 2.74 | no | IQ2_S gate/up + Q2_K down |
| tetra | 2.00 | 384 | 135.9 | 2.12 | no | llvq Tetra, 2.06 bpw (Strata native) |
| iq2xxs | 2.06 | 384 | 140.1 | 2.19 | no | IQ2_XXS gate/up/down (2.06 bpw) |
| iq1m | 1.75 | 384 | 118.9 | 1.86 | no | IQ1_M gate/up/down (1.75 bpw) |
| iq1s | 1.56 | 384 | 106.2 | 1.66 | no | IQ1_S gate/up/down (1.56 bpw) |
| iq2xxs_q2k_reap288 | 2.25 | 288 | 114.7 | 2.39 | no | IQ2_XXS gate/up + Q2_K down (the public MixedQ2 GGUF, 2.25 bpw), REAP-pruned to 288 experts |
| iq2xxs_q2k_reap256 | 2.25 | 256 | 101.9 | 2.39 | no | IQ2_XXS gate/up + Q2_K down (the public MixedQ2 GGUF, 2.25 bpw), REAP-pruned to 256 experts |

Engram: 202.8 GB on SSD, 48 rows x 264 B per token. Trunk FP8 7.20 GB per token (Q4_K 4.05, Q8_0 7.65); head BF16 1.32 GB; KV 890 B/token.

### tr16

hw.py tr16: 72 GB/s plain stream, 65.6 in-model; 5060 Ti 380 GB/s GEMV, 235 GB/s tier, PCIe 3 x8 7.18 GB/s; x4 NVMe work mount 3.68 GB/s, 4 KiB QD1 83 us

| pack | bpw | GB | trunk | fits | RAM GiB / tier GiB | tier hit | decode 8K plain / DSpark best / sweep | decode 32K | CPU bound | prefill 8192 | step ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| iq1s | 1.56 | 106 | q4 | yes | 104 / 8 | 34 % | 11.9 / 22.3 / 16.3 | 11.9 | compute | 273 | 84 = CPU 66 + GPU 15 + Engram 0.0 + head 2 |
| iq1m | 1.75 | 119 | q4 | yes | 116 / 8 | 32 % | 10.8 / 20.1 / 14.7 | 10.8 | compute | 244 | 93 = CPU 75 + GPU 15 + Engram 0.0 + head 2 |
| iq1m_iq2xxs | 1.85 | 126 | q4 | yes | 123 / 8 | 31 % | 10.7 / 19.9 / 14.6 | 10.7 | compute | 231 | 94 = CPU 76 + GPU 15 + Engram 0.0 + head 2 |
| iq2xxs_q2k_reap256 | 2.25 | 102 | q4 | yes | 100 / 8 | 28 % | 13.5 / 25.4 / 18.6 | 13.5 | compute | 284 | 74 = CPU 56 + GPU 15 + Engram 0.0 + head 2 |
| iq2xxs_q2k_reap288 | 2.25 | 115 | q4 | yes | 112 / 8 | 26 % | 13.3 / 25.0 / 18.3 | 13.3 | compute | 253 | 75 = CPU 57 + GPU 15 + Engram 0.0 + head 2 |
| tetra_reap320 | 2.00 | 113 | q4 | yes | 111 / 8 | 26 % | 13.3 / 25.0 / 18.3 | 13.3 | compute | 256 | 75 = CPU 57 + GPU 15 + Engram 0.0 + head 2 |
| iq2xxs_q2k | 2.25 | 153 | q4 | **no** | 148 / 8 | 27 % | 13.4 / 25.2 / 18.5 | 13.4 | compute | 190 | 75 = CPU 56 + GPU 15 + Engram 0.0 + head 2 |

### tr16+b550

hw.py tr16: 72 GB/s plain stream, 65.6 in-model; 5060 Ti 380 GB/s GEMV, 235 GB/s tier, PCIe 3 x8 7.18 GB/s; x4 NVMe work mount 3.68 GB/s, 4 KiB QD1 83 us

| pack | bpw | GB | trunk | fits | RAM GiB / tier GiB | tier hit | decode 8K plain / DSpark best / sweep | decode 32K | CPU bound | prefill 8192 | step ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| iq2xxs_q2k | 2.25 | 153 | q4 | yes | 107 / 8 | 27 % | 16.7 / 31.7 / 23.2 | 16.7 | compute | 190 | 60 = CPU 42 + GPU 15 + Engram 0.0 + head 2 |
| tetra | 2.00 | 136 | q4 | yes | 96 / 8 | 29 % | 17.0 / 32.4 / 23.7 | 17.0 | compute | 214 | 59 = CPU 41 + GPU 15 + Engram 0.0 + head 2 |
| iq2xxs | 2.06 | 140 | q4 | yes | 99 / 8 | 29 % | 16.9 / 32.2 / 23.6 | 16.9 | compute | 208 | 59 = CPU 41 + GPU 15 + Engram 0.0 + head 2 |
| iq1s | 1.56 | 106 | q4 | yes | 76 / 8 | 34 % | 15.0 / 28.3 / 20.8 | 15.0 | compute | 273 | 67 = CPU 49 + GPU 15 + Engram 0.0 + head 2 |
| q23 | 2.90 | 197 | q4 | **no** | 134 / 8 | 23 % | 19.4 / 27.6 / 20.2 | 19.4 | memory | 148 | 52 = CPU 34 + GPU 15 + Engram 0.0 + head 2 (remote wait 3) |

### xeon_v100

hw.py xeon_v100 (AVX-512, 150 GB/s, V100 0.8 x 900 GB/s GEMV, tier 400 with the current kernel, PCIe 3 x16 12 GB/s); RAM 181 GiB per the user (preset said 160); disk unmeasured 2 GB/s

| pack | bpw | GB | trunk | fits | RAM GiB / tier GiB | tier hit | decode 8K plain / DSpark best / sweep | decode 32K | CPU bound | prefill 8192 | step ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| iq2xxs_q2k | 2.25 | 153 | q8 | yes | 148 / 52 | 74 % | 33.5 / 67.6 / 49.5 | 33.4 | compute | 570 | 30 = CPU 15 + GPU 13 + Engram 0.0 + head 1 |
| iq2s_q2k | 2.58 | 176 | q8 | yes | 169 / 52 | 70 % | 37.2 / 70.1 / 51.4 | 37.2 | compute | 570 | 27 = CPU 12 + GPU 13 + Engram 0.0 + head 1 |
| tetra | 2.00 | 136 | q8 | yes | 132 / 52 | 78 % | 35.3 / 71.8 / 52.6 | 35.3 | compute | 570 | 28 = CPU 13 + GPU 13 + Engram 0.0 + head 1 |
| q23 | 2.90 | 197 | q8 | **no** | 188 / 52 | 66 % | 37.2 / 64.1 / 47.0 | 37.2 | memory | 546 | 27 = CPU 12 + GPU 13 + Engram 0.0 + head 1 |
| iq3xxs | 3.06 | 208 | q8 | **no** | 199 / 52 | 64 % | 30.2 / 60.3 / 44.2 | 30.2 | compute | 514 | 33 = CPU 18 + GPU 13 + Engram 0.0 + head 1 |
| iq1s | 1.56 | 106 | q8 | yes | 104 / 52 | 87 % | 37.6 / 77.2 / 56.6 | 37.6 | compute | 570 | 27 = CPU 12 + GPU 13 + Engram 0.0 + head 1 |

### A64FX nodes

| nodes | pack | profile | GiB/node | fits | decode plain / DSpark best | prefill | step ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6 | iq2xxs_q2k | current | 27.7 | yes | 21.6 / 25.0 | 29 | 46.3 = experts 13.6 + trunk 3.4 + attention 17.3 + collectives 3.9 + Engram 0.95 + other 5.0 |
| 6 | iq2xxs_q2k | redesign | 27.7 | yes | 68.8 / 87.6 | 321 | 14.5 = experts 1.6 + trunk 2.4 + attention 2.4 + collectives 3.1 + Engram 0.95 + other 2.8 |
| 6 | tetra | current | 25.1 | yes | 22.3 / 25.9 | 30 | 44.8 = experts 12.1 + trunk 3.4 + attention 17.3 + collectives 3.9 + Engram 0.95 + other 5.0 |
| 6 | tetra | redesign | 25.1 | yes | 70.3 / 89.7 | 321 | 14.2 = experts 1.4 + trunk 2.4 + attention 2.4 + collectives 3.0 + Engram 0.95 + other 2.8 |
| 6 | iq1m | current | 22.4 | yes | 23.1 / 26.9 | 31 | 43.3 = experts 10.6 + trunk 3.4 + attention 17.3 + collectives 3.9 + Engram 0.95 + other 5.0 |
| 6 | iq1m | redesign | 22.4 | yes | 71.8 / 92.0 | 321 | 13.9 = experts 1.2 + trunk 2.4 + attention 2.4 + collectives 2.8 + Engram 0.95 + other 2.8 |
| 8 | iq2xxs_q2k | current | 21.2 | yes | 24.4 / 28.2 | 33 | 40.9 = experts 10.2 + trunk 2.6 + attention 17.3 + collectives 3.2 + Engram 0.74 + other 5.0 |
| 8 | iq2xxs_q2k | redesign | 21.2 | yes | 78.2 / 101.3 | 414 | 12.8 = experts 1.2 + trunk 1.8 + attention 2.4 + collectives 2.7 + Engram 0.74 + other 2.8 |
| 8 | q23 | current | 26.3 | yes | 22.8 / 26.2 | 31 | 43.9 = experts 13.1 + trunk 2.6 + attention 17.3 + collectives 3.2 + Engram 0.74 + other 5.0 |
| 8 | q23 | redesign | 26.3 | yes | 74.4 / 95.5 | 414 | 13.4 = experts 1.5 + trunk 1.8 + attention 2.4 + collectives 3.0 + Engram 0.74 + other 2.8 |
| 8 | iq3xxs | current | 27.6 | yes | 22.4 / 25.8 | 30 | 44.6 = experts 13.9 + trunk 2.6 + attention 17.3 + collectives 3.2 + Engram 0.74 + other 5.0 |
| 8 | iq3xxs | redesign | 27.6 | yes | 73.4 / 94.1 | 414 | 13.6 = experts 1.6 + trunk 1.8 + attention 2.4 + collectives 3.1 + Engram 0.74 + other 2.8 |
| 12 | mxfp4 | current | 25.2 | yes | 23.8 / 28.6 | 32 | 42.0 = experts 9.6 + trunk 1.7 + attention 17.3 + collectives 6.0 + Engram 0.56 + other 5.0 |
| 12 | mxfp4 | redesign | 25.2 | yes | 69.8 / 93.9 | 585 | 14.3 = experts 1.5 + trunk 1.2 + attention 2.4 + collectives 4.8 + Engram 0.56 + other 2.8 |
| 12 | iq3xxs | current | 18.9 | yes | 24.0 / 28.9 | 33 | 41.6 = experts 9.3 + trunk 1.7 + attention 17.3 + collectives 6.0 + Engram 0.56 + other 5.0 |
| 12 | iq3xxs | redesign | 18.9 | yes | 75.0 / 102.7 | 585 | 13.3 = experts 1.1 + trunk 1.2 + attention 2.4 + collectives 4.2 + Engram 0.56 + other 2.8 |
| 12 | iq2xxs_q2k | current | 14.6 | yes | 25.5 / 30.9 | 35 | 39.2 = experts 6.8 + trunk 1.7 + attention 17.3 + collectives 6.0 + Engram 0.56 + other 5.0 |
| 12 | iq2xxs_q2k | redesign | 14.6 | yes | 79.1 / 109.7 | 585 | 12.6 = experts 0.8 + trunk 1.2 + attention 2.4 + collectives 3.8 + Engram 0.56 + other 2.8 |

Hypotheses: 8B-active prefill (encoder half) not exploited: full 40-layer expert traffic charged; GPU tier hit curve taken from GLM (1.16 f^0.545); IQ1_M CPU rate 0.9 x IQ1_S (no Strata kernel; ggml's is pdep-bound); IQ2_S rate 200 GMAC/s from the GLM IQ2_S CPU kernel's 17.5 vs 19.6 tok/s (Q2_K) on TR16; IQ3_XXS rate from GLM's measured 3.2 GB/s per core; Q2-class SVE expert rate never measured on A64FX; a64fx current profile constants (see A64FX_CURRENT / A64FX_REDESIGN); a64fx redesign profile constants (see A64FX_CURRENT / A64FX_REDESIGN); llvq Tetra CPU rate (AVX2 decode) assumed 120 GMAC/s on 15 Zen 1 cores; per-layer mixer fixed cost 40 us + 0.12 us per 1K context per source layer (GLM-fitted); prefill mixer cost 36 us per token per layer and 21.4 us per token per layer of MMQ GEMM scaled by TFLOPS (GLM-fitted on the 5060 Ti)

A DRAM-speed IQ1 kernel on TR16 (what-if, `--set cpu_rate_override_gbps=60`):

```
tr16: iq1s (1.56 bpw, 106.2 GB, 384 experts), trunk q4, head q4, context 8192, width 1
  RAM 104.1 of 124 GiB (experts 98.9); VRAM trunk 3.8 + head 0.35 + KV 0.01 + tier 8.3 of 16 GiB; SSD 292 GiB (Engram 189); fits
  decode 24.3 tok/s: step 41.2 ms = CPU experts 23.1 (memory-bound, 1.66 GB/token, tier serves 34 %) + exposed GPU 15.4 + Engram 0.00 (of 0.22) + head 2.0
  prefill 273 tok/s at 8192 tokens (3.66 ms/pos; pcie-bound: upload 27.1 s, GEMM 7.0 s, mixers 14.8 s, Engram 0.44 s)
  hypotheses: GPU tier hit curve taken from GLM (1.16 f^0.545); per-layer mixer fixed cost 40 us + 0.12 us per 1K context per source layer (GLM-fitted); CPU expert kernel assumed to stream at 60 GB/s; prefill mixer cost 36 us per token per layer and 21.4 us per token per layer of MMQ GEMM scaled by TFLOPS (GLM-fitted on the 5060 Ti); 8B-active prefill (encoder half) not exploited: full 40-layer expert traffic charged
```

## 3. Recommendation per machine

| Machine | Expert pack | Trunk / head | Decode (plain / DSpark at 0.977 acceptance) | Prefill 8K | What limits it |
|---|---|---|---:|---:|---|
| **TR16, 128 GB + 16 GB** | **IQ1_S, all 384 experts (106 GB)**, the user's choice; IQ1_M (119 GB) if the gate prefers it and the page cache can be given up; REAP-256 at 2.25 bpw (102 GB) is the comparison the gate must beat | Q4_K / Q4_K on the GPU (4.1 + 0.37 GB), ~8 GB expert tier | **12 / 22 tok/s** with today's IQ1_S kernel; **~18-20 / 35** if the kernel reached DRAM speed | ~270 | the IQ1_S CPU kernel is compute-bound at 18 GB/s (92 GMAC/s, 1.66 GB of experts per token = 66 ms); the GPU tier serves a third of the bytes; the kernel, not the bit width, is the lever |
| **TR16 + B550 (IB)** | **2.25 bpw, all 384 experts (153 GB)**: 2.25 bpw is the smallest full-expert pack that fits two nodes; IQ1_S also fits and is faster today | Q4_K on the TR16 GPU | **17 / 32** (IQ1_S 15 / 28) | ~190-270 | the B550's 26 GB/s carries 28 % of the expert bytes; the two streams balance at ~80 ms; the value of the second node is capacity (full experts at 2.25 bpw), not speed |
| **Xeon 181 GB + 2x V100** | **2.25 bpw, all 384 experts (153 GB) in RAM**; IQ2_S/Q2_K (175 GB) also fits; Q23 (197 GB) and IQ3 (208 GB) do not | **Q8_0 / Q8_0 split over the two cards** (7.7 + 0.7 GB), ~52 GB of expert tier serving ~74 % of the bytes | **34 / 68** | ~570 | the best machine of the three: 150 GB/s DRAM and two 720 GB/s cards; a lossless Q8 trunk; still compute-bound on the IQ2_XXS codebook kernel (36 AVX-512 cores) |
| **A64FX 12 nodes** | **native MXFP4 (lossless), 24.6 GB per node** | FP8 as released | 24.6 measured today (gemm); **~63 / 90** (plain / DSpark) with the GLM-plan redesign (in-model mHC 42 us/site) | 33 measured; ~590 redesign | no quantization decision at all; the runner's synchronization is the whole gap (see docs/GLM_A64FX_PLAN.md) |
| **A64FX 8 nodes** | 2.25 bpw (19 GB) or IQ3_XXS (26 GB) per node; MXFP4 (36 GB) does not fit | FP8 | ~24 today / **~56** redesign (2.25 bpw; ~61 with the Q2_K/Q3_K pack, which fits) | ~410 redesign | 8 = 2x2x2 torus, 3 tree steps per allreduce |
| **A64FX 6 nodes** | 2.25 bpw (25.5 GB per node, 2 GB of margin); IQ1_M or Tetra for room | FP8 | ~22 today / **~49** redesign (measured IQ2_XXS kernel, in-model mHC) | ~320 redesign | the only way onto 6 nodes is a 2-bit expert pack |

Engram on every machine: 203 GB of SSD, read-only, in the released shards (or repacked once into one contiguous
264 B-per-row file per table for O_DIRECT reads); 48 page reads per token cost 0.3-0.5 ms on TR16's NVMe and are
hidden under the previous layer with prefetch, 0.4 ms per token on 12 A64FX nodes (4 rows per node, measured 1.2 ms
exposed in the gemm runner); the 6 GB of E8M0 scales can be cached in RAM/HBM (`--engram-scale-cache`). An 8 MiB
row cache hits 30-50 % (gemm); a 1 GB cache is cheap and worth measuring. Not quantized, not needed.

## 4. Staging: where the bytes go (sizes only; the decision is deferred)

No copy of the checkpoint exists on TR16, B550 or the Xeon box (2026-10-18: TR16 196 + 142 GB free, B550 13 GB
free). The 510 GB release is staged on the A64FX nodes' `/local` (28 min at 300 MB/s aggregate). Per machine:

| Machine | Engram | Expert pack | Trunk + head | DSpark (optional) | SSD total |
|---|---:|---:|---:|---:|---:|
| TR16 | 203 GB | IQ1_S 106 / IQ1_M 119 | 4.5 GB | 8 GB | **315-335 GB** |
| TR16 + B550 | 203 (TR16) | 153 split ~110 / 43 | 4.5 | 8 | 325 (TR16) + 45 (B550) |
| Xeon | 203 | 153 | 8.4 | 8 | **372 GB** |
| A64FX node (12 / 8 / 6) | 17 / 25 / 34 | 24 / 19 / 25.5 | 0.8 / 1.2 / 1.6 | 0.7 | 43 / 46 / 62 GB on `/local` |

Pack building needs the 289 GB MXFP4 source once; it runs on the A64FX cluster against the checkpoint on the shared
filesystem (the tool is plain C++17; 12-node PJM batch jobs, docs/DS41F_QUANT_JOB.md "On the A64FX cluster") or on the Xeon
box after a 510 GB download. Packs are 102-197 GB each.

## 5. Quality gate

- Reference: teacher-forced logits of the exact FP8/MXFP4 model on Strata's 16 x 8,192-token eval corpus (the GLM
  gate's shape, docs/GLM_FP8_REFERENCE.md), produced by the gemm `ds41f_run` exact path on 12 A64FX nodes (6.9 tok/s
  teacher-forced; 131k positions is a one-allocation job) in the `eval16-fp8.logits` format `tools/glm_tr16_quality.py`
  reads. The same job dumps the per-layer expert input statistics the IQ1_S/IQ1_M/IQ2 quantizers need as an imatrix
  (llama-imatrix GGUF layout: `blk.L.ffn_{gate,up,down}_exps.weight.in_sum2` [cols, experts] and `.counts`).
- Metrics: KL to the reference, top-1 agreement, PPL, per candidate, with the GLM gate (KL <= 0.12, top-1 within
  5 pp) as the bar and antirez's IQ2_XXS (KLD 0.41) as the do-not-ship marker; HumanEval 164 on the shipping
  candidate once a runner exists.
- Order: IQ1_S full and IQ1_M full (TR16, primary), REAP-256 at 2.25 bpw (TR16 fallback), 2.25 bpw full (Xeon,
  two-node), IQ3_XXS (8 A64FX nodes). The public llama.cpp branch (PR #28696) is an independent check once it loads
  V4.1; today it does not.

## 6. The pack tool (`strata-ds41f-pack`, `tools/ds41f_pack.cpp`)

```
strata-ds41f-pack experts MODEL_DIR experts-iq1s.gguf --gate=iq1_s --up=iq1_s --down=iq1_s --imatrix=ds41f-imatrix.gguf [--keep=reap256.txt] [--layers=0-39] [--threads=32]
strata-ds41f-pack experts MODEL_DIR experts-mixedq2.gguf --gate=iq2_xxs --up=iq2_xxs --down=q2_k --uncalibrated
strata-ds41f-pack trunk   MODEL_DIR trunk-q4k.gguf --trunk=q4_k --head=q4_k         # Xeon: --trunk=q8_0 --head=q8_0
strata-ds41f-pack verify  MODEL_DIR experts-iq1s.gguf
strata-ds41f-pack dequant MODEL_DIR 7 123 down out.f32                              # exact MXFP4 -> f32 of one projection
```

It reads the release safetensors directly (header scan, `pread` per expert), dequantizes MXFP4 exactly (E2M1 x
2^(e-127), low nibble first) and FP8 dense tensors with their 32-row x N-column E8M0 blocks, requantizes with ggml's
quantizers (IQ1_S/IQ1_M/IQ2_*/IQ3_XXS need `--imatrix`; K-quants and Q8_0 may run `--uncalibrated`), and writes the
Strata expert sidecar (GGUF v3, `blk.L.ffn_{gate,up,down}_exps.weight`, `strata.expert_pack.architecture = deepseek4`,
source fingerprint over every shard, FNV hash per tensor, `blk.L.ffn_exps_keep` for a REAP-pruned layer, a
`.quality.json` with the normalized MSE per projection) or the trunk GGUF (safetensors names kept, Engram tables and
routed experts skipped). Tensor names are spelled once in C11: `include/strata/host/ds41f_names.h`.
Test without a model: `python3 tools/test_ds41f_pack.py --converter build-ds41f/strata-ds41f-pack` (synthetic two-shard
checkpoint; exact-dequant check against NumPy, K-quant pack with a keep list, IQ1_S pack with a synthetic imatrix,
trunk pack, corruption detection). Build: a CPU-only tree, `cmake --build build-ds41f --target strata-ds41f-pack`.

## 6b. Quantizing on the xeon+tesla GPU node

The resumable, streamable pipeline (chunk directories, `--deadline`, self-requeueing Slurm job, resumable fetch that skips
the Engram shards, rsync transfer to TR16) is in [DS41F_QUANT_JOB.md](DS41F_QUANT_JOB.md). Measured quantizer cost on
TR16: IQ1_S 2.83 M weights per core-second (53 core-hours for the model), IQ2_XXS/Q2_K 2.38 M (63), Q2_K/Q3_K 4.94 M (30).
Order: Q2_K/Q3_K full as the bring-up pack (no imatrix), then IQ1_S, IQ2_XXS/Q2_K and IQ1_M after the A64FX imatrix job.

## 7. Kernels per machine (measured where possible, 2026-10-18)

**TR16 / B550 CPU, IQ1_S.** The engine's IQ1_S kernel (`Fmt32<19>`, muse worktree only, unmerged) runs at 92 GMAC/s on
15 TR16 threads = 18 GB/s, compute-bound. The hypothesis was that its codebook arithmetic (three `vpaddb`, magnitude and
sign per token) was the cost, so `src/kernels/cpu/iq1s_w1a8_avx2.hpp` stores the grid as `g+1` bytes and computes the
exact integer dot `8 (sum(u x) - B) + sign B` with one `vpmaddubsw` + `vpmaddwd` per 32 weights and a per-block
correction from precomputed activation sums (the `hetero/ds4f` W4A8 identity). `iq1s_w1a8_bench` (parity against a
double reference, rel 9e-8, and against the engine's decode in the same harness) measured on 8 threads **while another
job held the CPU**, 2304 x 5120 experts streamed from DRAM:

| kernel | width 1 | width 2 | width 4 |
|---|---:|---:|---:|
| engine decode (Fmt32<19>) | 51.2 GMAC/s, 10.0 GB/s | 87.8 | 127.2 |
| W1A8 (+ load-pair grid build) | 57.2 GMAC/s, 11.2 GB/s | 94.8 | 132.2 |

So the arithmetic was not the bottleneck: **both kernels sit at the same decode floor, about 6.5-7 GMAC/s per Zen 1 core
at width 1, set by the four random 8-byte grid loads and the index extraction per 32 weights.** On 15 cores that is
~100 GMAC/s at one token and ~250 at four. Consequences for TR16: plain decode with IQ1_S experts stays near 12 tok/s;
DSpark windows of 3-4 tokens amortize the decode and are the real lever (the simulator's 22 tok/s row); a 1-bit format
with a cheaper decode would need ternary values stored directly (2 bits per weight, 2.0+ bpw, 136+ GB, which does not fit
TR16). Re-run `build-ds41f/iq1s_w1a8_bench --bench --threads 15` on an idle CPU before quoting absolute numbers.

**TR16 / B550 GPU.** The dp4a IQ1_S tier kernel (worktree) needs measuring on 2304-wide experts; the `iq2xxs_q2k` pack
needs Q2_K/IQ2_XXS added to the GPU down-format list (`iq_kernels.cu` `STRATA_D_FMTS`). Shape caps to lift for hidden
5120: `kMaxBlocks` 16 -> 20 in `q23_avx2.cpp` and `glm_q23.cu`, `kMaxBlocks` 64 -> 80 in `nvfp4_*.cpp`, `kNativeFF`
2048 -> 4096 (worktree has it). CSA2 attention: port `glm_index_score/select` + `glm_mla` (query 64 x 512 against the
128-row FP32 window + 512 rows gathered from the FP4 compressed cache; indexer 32 x 128 FP4 keys over the 2048 candidate
blocks of layer 20 with `qsa_select.hpp`'s radix top-k; KV written once per source layer, reused by 36). mHC via
`glm_mhc_*` (hc_mult 4, 20 iterations). Engram: an `EngramTable` sibling of `PleTable` over `PleReader` (264 B rows,
two tables, 24 rows each, the V4.1 hash from gemm `ds41f_engram.c` / `make_token_map.py`).

**Xeon + 2x V100.** Cascade Lake has AVX-512 VNNI and non-microcoded gathers: the W1A8 scheme with `vpdpbusd` and a
`vpgatherqq` grid fetch is the variant to try there (not measured). GPU: dp4a IQ tier and MMQ prefill as GLM on sm70,
Q8_0 trunk over the two cards with the measured layer pipeline (Q23 20.3 -> 34.8 tok/s on GLM), FP16 WMMA prefill GEMMs.
Engram from the node's storage needs its random-read latency measured first (`xeon_tesla` machine in the simulator
assumes 2 ms per page: a 4 GB RAM row cache and issue-at-token-id with 13 layers of slack for the layer-14 table).

**A64FX.** 12 nodes: gemm's MXFP4 SDOT path (FP4 -> exact integer grid via `svtbl_s8`, int8 group-32 activations, two
`svdot_s32`, fused W1/W3 pair ~380 GMAC/s per node; quality cosine 0.988 is an activation-quantization question for the
gate) inside the Strata A64FX port. 6/8 nodes: SVE Q2_K and IQ kernels. Measured 2026-10-18 on one node: Strata's SDOT
kernels (`tools/a64fx_probe/q23_sdot_sve.c`, `iq_sdot_sve.c`, parity with ggml within 2.4e-7) stream per node, on the
2304 x 5120 gate/up shape, Q2_K 180-192 GB/s, IQ2_XXS 78-85 and IQ1_S 60 (ggml 95 / 11 / 8), and Q2_K down (5120 x 2304)
133-136; built with clang 21 -mcpu=a64fx the kernels are 25-45 % faster than with FCC at 2-4 tokens (the simulator uses
that build, job 52440185).
The IQ kernels are 7x ggml but still 2-3x slower per byte than Q2_K, so in the simulator the 6-node IQ packs drop from
the old 250 GB/s hypothesis (~70 tok/s) to 51-54 tok/s redesigned (data: `tools/sim/data/measured_a64fx_kernels.json`;
ggml's 11/8 GB/s are why the `current` A64FX rows for IQ packs are slow). The IQ kernels fetch eight grid entries per
512-bit vector with one 64-bit gather and fold IQ1_S's delta into the weights as 8g +- 1. Gather-free variants were
measured too (job 52439675): decoding IQ2_XXS's grid in registers from eight 64-byte `svtbl` tables costs about as much as
the gathers at one token (72 vs 80 GB/s) and wins only at 2-4 tokens (+4 to +30 %, used for the MTP width curve); a load-time
repack with resolved sign bytes (+12 % memory) adds 4-6 % at one token; expanding the codes doubles the rate (IQ2_XXS 125,
IQ1_S 120 GB/s) but costs 61-64 % more memory, which leaves IQ1_S at Q2_K's size and speed with worse quality. IQ1_S has
no compact gather-free form (64 tables). So the gather kernels stay for one-token decode. Engram as gemm does (owned rows per rank, prefetch thread,
scale cache); CSA2 per gemm's `ds41f_attention.c`.

## 8. Runner (phase 2, design)

Strata has no DeepSeek-V4 support (the engine accepts `glm5next` and `qwen4exp` only). The port follows the GLM
A64FX plan's shape (docs/GLM_A64FX_PLAN.md section 4) with a second architecture family `deepseek4`:

1. GPU trunk (CUDA/HIP today; the Vulkan runner's split as the template): CSA2 attention with the top-512 indexer
   and 128-token window, KV reuse from the four source layers, the compressor on source layers, mHC with 4 streams
   and 20 Sinkhorn iterations, router (sqrt-softplus scores + bias, top-6, scale 1.5), shared expert, head; Q4_K or
   Q8_0 copies from the trunk GGUF.
2. CPU expert pool as today for Q2/IQ formats (the IQ1_S `Fmt32<19>` kernel exists; IQ1_M needs one; the
   `NVFP4` AVX2/AVX-512 kernels extend to MXFP4 for the A64FX/Xeon lossless case); the GPU tier serves IQ1_S/IQ2.
3. **Engram through `PleReader`** (`include/strata/ngram/ple_reader.hpp`, O_DIRECT 4 KiB page reads, issue at token id,
   collect before the layer) generalized to 264 B rows with E8M0 scales and the V4.1 hash: NFKC-lowercase token
   compression to 99,092 ids, three n-gram orders with per-layer multipliers, 8 primes per order (`ds41f_engram.c`,
   `make_token_map.py`), then the `engram.wkv` projection and the signed-sqrt sigmoid gate.
4. DSpark drafter as the speculation path (`speculation=external` in the sim; verify bytes grow with the window's
   union ratio, so acceptance below ~0.9 per position does not pay).
5. Two nodes over `expert_tp.hpp` / `ucomm`; A64FX through the GLM port (uTofu `ucomm` backend, 4-CMG pool).

## 8b. The runner on TR16 + RX 9070 XT (measured 2026-10-21)

The DS4.1F runner is `src/program/ds41_decode.cpp`. `strata-glm-decode` takes a GGUF whose architecture is
`deepseek41`, the schema of llama.cpp PR #30817, for example Unsloth's UD-IQ1_S.

Placement:
- The trunk runs on the GPU from the GGUF's own Q6_K/Q5_K/Q8_0/BF16/F32 tensors.
- The routed experts are mapped from the GGUF and run on the CPU lane (IQ1_S `Fmt32<19>`), or on the GPU from an
  adaptive VRAM tier: one arena of about 1100 experts, 2 promotions per token.
- In prompts of 512 tokens or more, the experts are streamed to the GPU through llama.cpp's MMQ.
- Engram rows are read from the mapped table.

Against llama.cpp on the same GGUF (out-of-sample wikitext, 1063 tokens), mean KL is 0.05 and perplexity 5.55
vs 5.56.

Measured:

| What | tok/s |
|---|---:|
| Decode, 128 tokens after a 1K prompt, GPU tier hit 0.61 | 12.5 |
| Prefill, 1063 tokens, chunks of 256, CPU lane + GPU tier | 27.3 |
| Prefill, 1063 tokens, a 1024-token pass with streamed MMQ experts | 42.1 |

`python3 tools/glm_ds41f_sim.py roofline` (`ds41f.RunnerHw`, fitted to the three rows above) gives these decode
estimates, with exact routing, in ms per token:

| Config | Trunk | Experts | Other | Hit | tok/s |
|---|---:|---:|---:|---:|---:|
| now | 30.6 | 40.1 | 7.5 | 0.61 | 12.8 |
| misses split between the CPU lane and PCIe uploads | 30.6 | 22.4 | 7.5 | 0.61 | 16.5 |
| next-layer prefetch (+0.09 hit) | 30.6 | 30.9 | 7.5 | 0.70 | 14.5 |
| target: trunk GEMV at 480 GB/s, prefetch, split, fewer syncs | 17.6 | 17.3 | 2.5 | 0.70 | 26.8 |
| target without the split | 17.6 | 30.9 | 2.5 | 0.70 | 19.6 |

And these prefill estimates, in tok/s:

| Config | 1K | 2K | 4K | 8K |
|---|---:|---:|---:|---:|
| now: CPU lane, chunks of 256 | 29 | 28 | 27 | 26 |
| now: streamed, chunks of 1024 | 42 | 41 | 39 | 36 |
| target: one layer-major pass per prompt | 127 | 247 | 303 | 251 |
| target, enc-skip (layers past 20 on the last 2560 tokens) | 127 | 247 | 354 | 345 |
| same, with 15 GB of experts not page-cached (SSD at 1.7 GB/s) | 61 | 120 | 232 | 345 |

The target assumes:
- the copies of one layer overlap the GPU work of another;
- host copies reach 13.8 GB/s, the PCIe 3.0 x16 DMA rate; the shared mapping copies at 21 GB/s;
- the dense half runs at 2.3 ms per token (batched attention, indexer and compressor; GPU top-k);
- MMQ experts take 0.4 ms per token.

Every expert must stay in RAM. The 115 GB of experts fit in 128 GB only if Engram reads do not fill the page cache,
and on 2026-10-21 about 15 % of the expert pages were not cached.

Comparison with Xolotl (a Strata fork; RTX 4090, 90 GB RAM, about 152 GiB of experts mostly on NVMe):
- **Prefill.** Its exact 244-317 tok/s comes from the same layer-major pass (8K tokens), enc-skip, next-layer prestage
  and background Engram reads. The table's target is that design on this machine's PCIe 3.0.
- **Decode.** Its 26-27 tok/s relies on a route bias toward VRAM-resident experts (lossy: code perplexity
  2.38 -> 2.87) and DSpark. With exact routing it runs at about 3.5 tok/s, because its experts live on NVMe.
- **Here,** every expert is in RAM, so exact routing already gives 12.5 tok/s, and the decode target above is about
  20-27 tok/s without changing the routing.

## 8c. Two nodes: TR16 + B550 over InfiniBand (measured 2026-10-21)

**Split.** Every routed expert is split by FFN rows, the GLM remote tensor-parallel scheme
(`include/strata/net/expert_tp.hpp`):
- TR16 keeps rows [0, fl) of gate/up and the first fl down columns.
- A `strata-glm-tp-worker` on the B550 holds rows [fl, 2304) in its RAM.
- The experts are linear after the per-row clamped SwiGLU, so the two router-weighted partial sums add.
- Cuts step by 1/9 (2304 = 9 IQ1_S blocks). The default share is 2/9 (fl = 1792): 23.85 GiB on the B550, which fits
  its 32 GB `/dev/shm`.

**Running it.**
- `strata-glm-decode MODEL PROMPT N --remote-tp=ADDR:PORT [--remote-tp-share=2/9]`, or
  `tools/ds41_remote_tp.sh WORKER_HOST WORKER_BIN ADDR:PORT -- strata-glm-decode ...`. The wrapper checks locked
  memory on both nodes, a leftover worker, and the worker host's free RAM.
- The worker (setup magic `STD1`) computes with the DS4.1 expert lane: Strata's IQ1_S rows, not ggml's pdep path.

**Weights.**
- At load, TR16 reads the worker's rows with `pread` (never through its mapping) and sends them over IB: 23.85 GiB in
  34 s.
- It then drops those gate/up rows from its own page cache, about 17 GB.
- The worker keeps them in `/dev/shm/strata-tp-<fingerprint>.weights`. A later run with the same model and share maps
  that file and nothing is sent (0.02 s).

**Decode.**
- After the router, TR16 sends each layer's routes and z (f32, 20 KiB per token). It then runs its GPU tier slices and
  CPU lane on its own rows, and adds the reply.
- The reply wait left over is 0.02-0.14 ms per layer.
- The tier slots hold TR16's slices, so 9/7 as many experts fit in the same VRAM.

**Streamed prompt.**
- The VRAM slots still hold whole experts, so MMQ is unchanged.
- TR16's rows come from the mapping. Down tensors are pinned (`STRATA_DS41_PIN_GB`) and DMAed; gate/up rows go
  through the stager.
- A fetcher thread pulls the worker's rows of each item over IB (16 buffers ahead), then they are uploaded to VRAM
  scratch and placed by `ds41_copy_rows`. The 100-byte down-column pieces are too narrow for the copy engine.
- Prompt logits are bit-identical to a single-node run.

| measured (TR16 load average 2-8) | TR16 alone | TR16 + B550 (2/9) |
|---|---|---|
| decode after a 1063-token prompt, 128 tokens | 8.3 tok/s (tier hit 0.61) | 9.3 tok/s (0.645) |
| decode after a 600-token prompt, tier from empty | 9.1 tok/s | 10.3 tok/s |
| 600-token prompt (CPU lane, chunks of 256) | 18.9 tok/s | 25.3 tok/s |
| 2048-token prompt (streamed) | 117-132 tok/s | 118-124 tok/s |
| 4096-token prompt (streamed) | 224 tok/s | 186 tok/s |
| wiki2 1063 tokens, all logits, KL vs llama.cpp | 0.0508 | 0.0508 |
| generated tokens vs single node (600-token prompt, tier off, 32 tokens) | - | 32 of 32 equal; last-token KL 7e-5 |

**Prompt and decode trade-offs.**
- The streamed prompt is no faster: TR16 still uploads all 107 GB to its GPU. At 4K it is 17 % slower, since the
  worker's 24 GB arrive over IB (8.2 s) and every gate/up row goes through the stager.
- Pinning the gate/up rows per expert (`STRATA_DS41_PIN_GATE_UP=1`, about 12k registrations) makes every later HIP
  host copy slow: decode fell to 3 tok/s. Down tensors are pinned whole instead.

**GPU worker (phase 2).** `strata-ds41-worker` speaks the same protocol and stores the same rows, and uses the B550's
RTX 5060 Ti (CUDA build, sm_120). It reports `TpReady.ok = 2`.
- **Decode.** A VRAM tier of its slices: about 4,100 slices at 3/9, 11.7 GiB. It is refilled from the routing scores
  after each prompt and adapts by up to 4 promotions per request. Misses go to the CPU lane.
  - Hit rate 0.90-0.96; 0.4 ms per request.
- **Prompt passes** (64 tokens or more). TR16 sends each layer's routes and z (f32); the worker answers with its
  partial sums.
  - The worker uploads the layer's blobs from its registered RAM, double-buffered with the next layer, and runs MMQ.
    That is about 115 ms per layer at 4K.
  - TR16 computes only its gate/up rows, as two MMQ products, with h zero-padded to the full down rows, so its down
    tensors stay whole and pinned. It no longer fetches rows.
  - The producer uploads only the staged parts: uploading whole slots had pushed 121 GB per 4K pass over PCIe.
- **Share 3/9.** 35.8 GB, held in the worker's anonymous memory, because `/dev/shm` was capped at 32 GB (52 GB since).
  - The rows are sent again on every run, in 67 s.
  - 4/9 (47 GB) was killed at 42 GB of rows on the 62 GB B550 (by systemd-oomd, not the kernel), so the worker now
    keeps a 16 GiB reserve.
  - Share 5/9 (59.6 GiB) fits with `--rows=hugetlb` on a 2 MiB hugetlb pool, `kho=off`, swap off and systemd-oomd
    masked: docs/MULTINODE_WORKER_MEMORY.md (measured 2026-10-25 with a probe, not yet with a decoder).

| measured, GPU worker at 3/9 (TR16 load average 2-5) | TR16 + B550 |
|---|---|
| 4096-token prompt | 214 tok/s (single node 224 idle; 5.5 when the experts no longer fit TR16's page cache) |
| 2048-token prompt | 138 tok/s |
| decode after a 1063-token prompt, 128 tokens | 9.4 tok/s (TR16 tier hit 0.68; worker 0.90) |
| decode after the 4096-token prompt | 7.8 tok/s |
| wiki2 1063 tokens, all logits, vs llama.cpp | mean KL 0.044, top-1 92.9 %, PPL 5.534 vs 5.556 |
| 2048-token prompt logits vs single node | same argmax, KL 0.013 (split summation order) |

**Gains and limits.** The gains are prompt speed with TR16 holding about 30 GB less of the experts, which also keeps
it fast when a browser takes its RAM. Decode is bound by TR16's own CPU lane and trunk, since the B550 cannot hold a
share above 3/9.

**Share.** The simulator (`glm_ds41f_sim.py roofline`) gives the idle decode as:
- 12.8 tok/s alone
- 15.2 at 2/9
- 13.5 at 3/9
- 11.6 at 4/9

Above 2/9 the B550's CPU lane is the bound. Larger shares pay only with the B550's GPU (RTX 5060 Ti) holding a tier
of its slices, which is the next step. That GPU would also take its share of the streamed prompt, instead of
fetching the rows back.

## 8d. The runner on the x64 GPU node: 2x Tesla V100 PCIe 32 GB, 2x Xeon Gold 6240, 181 GiB (measured 2026-10-21)

**Packs.** The cluster packs come from the llama.cpp PR #28696 converter (Q8_0 staging, imatrix). The runner reads
them as well as Unsloth's #30817 files:
- **Source layers:** kv/index sources are found from the compressor and indexer tensors.
- **Engram:** one table per Engram layer (`blk.L.engram_embd`). `engram_wkv` serves as `engram_kv`, and the gate
  is the product `engram_q ⊙ engram_k`, as llama.cpp's `build_engram` uses it.

**Fit.** Q23 (KL 0.214, the best measured) has 183 GiB of experts and 195 GiB of Engram (Q8_0). Only the experts
fit, with 55 GiB of them in the two GPUs.

Engram on the shared Lustre filesystem is then the limit:
- random 4 KiB reads reach ~120/s with O_DIRECT and ~1,550/s buffered with 256 threads
- each token needs 48 rows

Q23 reached 9.2 tok/s decode and 16.9 tok/s for a 600-token prompt (`STRATA_DS41_ENGRAM_DIRECT=0`, 256 threads,
`STRATA_DS41_ENGRAM_ASYNC`). Engram was still most of the time.

**IQ2_XXSE4** (IQ2_XXS gate/up, Q2_K down, Engram Q4_K; KL 0.289): 142.4 GiB of experts and 103 GiB of Engram.
- With `STRATA_DS41_ENGRAM_RAM_GIB=80`, 78 % of each table's rows are in memory, beside 87 GiB of experts in RAM
  and 55 GiB in VRAM. Peak RSS is 168-172 GiB.
- The pack's files are striped over one Lustre OST each, which made a load take 16 min.
- A copy striped over 8 OSTs (`lfs setstripe -c 8 -S 4M`, 2 min to copy) loads in 260 s and serves the remaining
  random rows fast.

**Residency** (`STRATA_DS41_RAM_COPY=1`):
- The most-routed experts go to GPU0 (25.3 GiB) and to GPU1 (30.0 GiB). GPU1 runs its routes itself: MMVQ for
  decode, MMQ for prompt passes.
- The rest are read into anonymous memory, because Lustre keeps little page cache.

| UD/cluster pack on 2x V100 | load | prompt | decode |
|---|---|---|---|
| Q23, Engram O_DIRECT from Lustre | 277 s | 2.7 tok/s (600) | 3.0 tok/s |
| Q23, buffered Engram, 256 threads, async | 4.5 min | 16.9 tok/s (600), 23.9 (2048) | 9.2 tok/s (6.3 after 2K) |
| IQ2_XXSE4, 80 GiB Engram in RAM, 1-OST files | 948 s | 25.8 tok/s (600) | 11.3 tok/s |
| IQ2_XXSE4, the same from the 8-OST copy | 260 s | 37.3 tok/s (1008), 131 tok/s (4096) | 10.2 tok/s at 1K, 8.1 at 4K |

**Quality vs llama.cpp on the same pack** (1,008 wikitext tokens, all logits, `tools/ds41f_llama_dump.cpp` built
against llama.cpp-ds41 on a CPU node):
- **Q23:** mean KL 0.036, top-1 95.2 %, PPL 3.371 vs 3.369.
- **IQ2_XXSE4:** PPL 3.446 vs 3.363 at first (+2.5 %).
  - With every expert on the CPU lane: 3.390 (+0.8 %). The gap was in the GPU tiers.
  - The IQ2_XXS MMVQ dot (as in llama.cpp's CUDA code) scales each 32-weight group with an integer division by 8,
    which truncates toward zero. ggml-cpu and MMQ scale in floats.
  - With `native_mmvq_set_iq2xxs_exact` (on in the DS4.1 runner, `STRATA_DS41_IQ2XXS_EXACT`; off by default for the
    existing models): PPL 3.406 (+1.3 %), mean KL 0.024, top-1 94.5 %.
  - What remains is the GPU's Q8_1 activation rounding vs the CPU's Q8_K.

**Decode tuning on IQ2_XXSE4** (1,008-token prompt + 128 tokens; the final profile is now decode alone):

| change | decode tok/s | per token: Engram / CPU experts / GPU to router |
|---|---|---|
| baseline (78 % of each Engram table in RAM) | 10.0 | 48.9 / 21.9 / 14.4 ms |
| Engram RAM filled table by table: layer 1's table whole, layer 14's 55 % (read in the background) | 11.8 | 32.7 / 22.7 / 14.1 |
| + in-memory rows served inline (threads only for file reads) + an 8K-token routing profile from this pack | 13.6 | 20.2 / 21.4 / 17.4 |
| + 36 CPU lane threads (of 36 cores) | 14.2 | 18.7 / 20.2 / 17.4 |

- `STRATA_GLM_IQ512_CLAMP=1` (AVX-512 IQ rows) was no faster here and moved KL from 0.024 to 0.030.
- What remains of Engram is the second table's file reads, which are latency bound (~50 ms per row).
- A pack with a Q3_K Engram, IQ2_XXSE3 (~79 GiB, both tables in memory), is being quantized for the gate.

## 8e. Decode on 2x V100: speculative decoding with DSpark (measured 2026-10-22)

All of this is opt-in (`STRATA_DS41_*`), on IQ2_XXSE4, in a 160-168 GiB / 64-CPU allocation (52-60 GiB of Engram in
RAM). Prompts: a 600-token wikitext prompt and a 600 / 4096-token C++ prompt; 64 greedy tokens.

**Plain decode** went from 10.3 to about 18 tok/s at 600 tokens of context:
- `ENGRAM_PINNED=1`: the Engram rows' host buffer pinned. The pageable upload had made the host wait for the whole
  GPU stream at layers 1 and 14 (the "Engram" 34 ms per token was that wait, not the filesystem).
- `GPU_TOPK_MIN=1`: the indexer's top-k on the GPU for one-token passes too. The per-token attention reads the
  selected count from the device (`nsel_dev`); the first version took the batched kernel instead and corrupted the
  output.
- One-token output projection without the group pack copies; lane spin 256; 32 CPU lane threads (of the
  allocation's 32 cores: 36 threads stall the spin barriers).

**DSpark** (`DSPARK=<gguf>`): the 3-layer DSV4.1 drafter of `llama.cpp`'s `dflash` arch, on GPU0 (8 GB):
- MXFP4 experts and BF16 trunk through few-token GEMVs with the activations in LDS (`ds41_mxfp4_gemv_mt`,
  `ds41_bf16_gemv_mt`); its keys from the target's stream means at layers 37-39 (fc, norm, attn_kv).
- The target verifies [anchor, drafts] in one pass; `rewind()` drops the rejected positions (token history, the
  ratio-2 compressor's open group).
- `DSPARK_N=5`, `DSPARK_PMIN=0.4` (the confidence head truncates the draft).
- llama.cpp on a CPU node accepts 74 % (code) / 54 % (prose) of 4 drafts per step on this pack.

Verify passes needed their own work:
- `DECODE_MMQ=1`: one MMQ product per layer for GPU0's resident experts, and MMQ on GPU1.
- `ATTN_MULTI=1`: one launch for the pass's tokens.
- `OUT_REGROUP=1`, `SHARE_Q8=1`.
- `LANE_ASYNC=1`: the CPU lane's workers start while the main thread enqueues the GPU work.
- `ENGRAM_PREFETCH=1`: the second Engram table's file rows of each drafted token are read into the page cache as the
  draft decides it. Verify passes read 5x the rows; this cut their stall from 53 to ~12 ms.

**Residency is domain specific.** The routing profile picks the GPU-resident experts. A wikitext profile gives a
0.62-0.67 GPU hit rate on wikitext but 0.135-0.18 on code. Two fixes:
- a mixed profile (wikitext 8K + code 4K routes): 0.56 on both;
- `REBALANCE=N`: after the prompt, swap up to N of the prompt's most-routed RAM experts with the least-routed resident
  ones (5-12 s once). With it, the 4K code prompt goes from 0.135 to 0.53.

| IQ2_XXSE4, 2x V100 | context | tokens per step | decode tok/s |
|---|---|---|---|
| plain (wikitext) | 600 | 1 | ~18 |
| DSpark (wikitext, wikitext profile) | 600 | 2.7-2.9 | 20-22 |
| DSpark (code, mixed profile) | 600 | 4.3 | 21.8 |
| plain (code, wikitext profile) | 4K | 1 | 7.1 |
| plain (code, + rebalance) | 4K | 1 | 9.1 |
| DSpark (code, + rebalance) | 4K | 3.7 | 19.8-20.4 |
| DSpark (wikitext) | 4K | 2.0 | 14.0 |
| DSpark, mixed profile + rebalance: wikitext / code | 600 | 2.6 / 3.8 | 18.6 / 20.9 |
| DSpark, mixed profile + rebalance: code / wikitext | 4K | 3.4 / 2.1 | 19.9 / 13.7 |

With a 4096-token prompt GPU0 keeps only ~11 GiB of experts: the prompt's chunk buffers and the drafter take the
rest. `STRATA_DS41_PROMPT_BATCH=2048` leaves it 13 GiB: the 4K code prompt then decodes at 21.8 tok/s (GPU hit rate
0.61), and the prompt runs at 259 instead of 371 tok/s.

Later in the day, on a 140 GiB allocation (36 GiB of Engram in RAM, so ~50 ms of Engram reads per step):
- `hc_coeff` runs on a thread per comb element, with every sum in the serial order (bit-identical tokens). It had
  been one thread per token, ~17 us twice per layer.
- `ATTN_SPLIT=4` splits each token's attention output rows over 4 blocks plus a combine. One-token attention goes
  from 5.8 to 2.9 ms per token, a verify pass's from 8.3 to 5.6 ms.
- `DECODE_GROUPED=1` (`native_expert_grouped` over the routed GPU experts of a short pass, on both GPUs) is no
  faster than the MMQ path.
- `GROW=1` (with `DECODE_GROUPED`) shrinks the chunk buffers after the prompt and copies the prompt's most-routed RAM
  experts into the freed VRAM: +675 experts (6.3 GiB) after a 4K prompt, GPU hit rate 0.68 on the 4K code prompt.
- All together: 19.0 tok/s (wikitext) and 18.3 (code) at 600 tokens.
- `DSPARK_Q8=1` quantizes the drafter's BF16 matrices (all but the grouped output projection) to Q8_0 at load, read
  by native MMVQ. A draft goes from 17.5 to 15.0 ms; acceptance is unchanged.
- Prefetching the first Engram table's rows too was slower: the reads compete with the second table's for Lustre
  IOPS.

**Full node** (181 GiB, 72 CPUs, 36 lane threads), every option above on (with `ATTN_SPLIT=4`, `DSPARK_Q8=1`,
mixed profile and rebalance), 600-token prompts, 64 tokens:

| pack | Engram in RAM | prompt | tokens per step | decode tok/s |
|---|---|---|---|---|
| IQ2_XXSE4 | 76 GiB (74 %) | wikitext | 2.78 | 25.3 (DSpark) / 21.9 (plain) |
| IQ2_XXSE4 | 76 GiB (74 %) | code | 3.76 | 21.2 (0.9 s of Engram reads) |
| IQ2_XXSE3 | 79 GiB (all) | wikitext | 2.56 | 30.7 |
| IQ2_XXSE3 | 79 GiB (all) | code | 3.71 | 34.6 |

With Engram entirely in RAM (IQ2_XXSE3, KL 0.314 vs 0.289) a verify step falls to ~85 ms and decode reaches 31-35
tok/s. IQ2_XXSE4's remaining gap is its second table's file reads.

One-token passes now send the peer GPU's routes through its GEMV chains even with `DECODE_MMQ=1`. Its MMQ product
spans all of its ~200 resident experts, and plain decode spent more time waiting on it (9.3 ms per token) than on
the CPU lane (4.8 ms).

CUDA graphs would not pay. `STRATA_DS41_GRAPH_PROBE=1` (CUDA only) times one plain layer's dense sequence for one
token (38 kernels): 425 us launched directly, 392 us as a graph. The time is in the small kernels themselves.

### 8f. A decode model for the V100 runner (`ds41f.V100Hw`, `python3 tools/glm_ds41f_sim.py v100`)

A step is plain (one pass of 1 token) or DSpark (draft, then one verify pass of T = 1 + drafted tokens). A pass
costs `A + B T + C x (CPU experts) + host + Engram stall + index scoring`:
- **CPU experts:** 40 x T x 6 x (1 - GPU tier hit). The tier hit is the resident fraction ^ 0.40 (47.4 GiB with the
  drafter on GPU0 gives 0.644; 55.3 GiB without it gives 0.684).
- **A, B, C** = 32.9 ms, 6.1 ms per token, 0.06 ms per CPU expert. The lane alone costs 0.12; half of it overlaps
  the GPU. They are fit on the points without Engram reads.
- **Engram:** the second table's misses at 800 reads/s, less what the draft and layers 0-13 hide. The first table's
  misses (Engram budget under 51.5 GiB) are not hidden.
- **Long context:** each token's 8 index layers read 6.5 x position keys (512 B FP32, 256 FP16, 136 Q8) at
  800 GB/s or score them at 12 TFLOPS (4x with FP16 tensor cores), whichever is longer, plus a top-k at 10 G
  elements/s. The KV rows come off the GPU expert tier.

Calibration (`validate`, 18 of 18 rows):
- **Fitted:** the four full-node points with little or no Engram reading, within 4.3 %.
- **Cross-checks** (Lustre varies by day, tolerance 25 %): the code run with a 52 ms-per-pass stall +24 %; the
  140 GiB node -18 % and -24 %, where that day's Lustre served more reads than 800/s; the 168 GiB node -5 %.

| lever (DSpark, wikitext acceptance) | IQ2_XXSE4, 76 GiB Engram | IQ2_XXSE3, Engram in RAM |
|---|---|---|
| now (measured 22.3-25.3 / 30.7) | 26.3 | 30.7 |
| Engram whole in RAM (IQ2_XXSE3 or IQ2_XXSE43 pack, or Lustre at 1,550 reads/s) | 30.7 | - |
| dense kernels -25 % | 28.7 | 36.2 |
| CPU lane 2x faster | 27.9 | 34.2 |
| acceptance 0.68 -> 0.75 | 28.0 | 32.7 |
| draft time -50 % | 26.3 (the draft hides Engram reads) | 33.5 |
| +6.3-7.2 GiB of resident experts (GROW, drafter experts off GPU0) | 26.6 | 31.3-31.4 |
| Engram in RAM + lane 2x + dense -25 % | 41.1 | 41.1 |

Reading the table:
- On IQ2_XXSE4 the Engram reads come first: IQ2_XXSE43 (KL 0.307) removes them as well as IQ2_XXSE3 does.
- Once Engram is in RAM, the dense kernels, the CPU lane and the draft are the levers. All three together reach ~41.

**Batched sessions** (one pass carries every session's tokens; forecast before the implementation, measured in 8g):

| sessions | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| plain, aggregate / per session (tok/s) | 21.9 / 21.9 | 29.2 / 14.6 | 34.9 / 8.7 | 38.7 / 4.8 |
| DSpark, aggregate / per session (tok/s) | 26.3 / 26.3 | 29.4 / 14.7 | 31.3 / 7.8 | 32.3 / 4.0 |

Batching plain decode raises throughput more than speculation does: verify passes of several sessions read too
many Engram rows and CPU experts.

**Long context** (IQ2_XXSE4 DSpark, tok/s):

| context | FP32 KV | FP16 KV | Q8 KV | Q8 + FP16 tensor-core scoring |
|---|---|---|---|---|
| 128K | 25.9 | 25.9 | 25.9 | 25.9 |
| 512K | 24.7 | 24.8 | 24.8 | 25.7 |
| 1M | 23.3 | 23.4 | 23.4 | 25.2 |

The indexer's verify-pass cost at 1M is 19 ms (FP32) / 6.6 ms (Q8 + tensor cores). On IQ2_XXSE4 part of it is
offset because a longer pass hides more Engram reads; with Engram in RAM the full cost shows.

Where a verify step goes (600 tokens, about 2.7 tokens per step, ~120 ms):
- draft 16 ms
- CPU lane 30-40 ms
- GPU experts and peer merge ~15 ms
- small dense kernels ~28 ms
- Engram ~12 ms

The CPU lane runs at ~0.12 ms per IQ2_XXS expert on 32 threads (`LANE_BENCH=1`), and verify passes route ~7 experts
per layer to it.

Tried and slower: streaming CPU experts to the GPUs at decode (`DECODE_STREAM`; PCIe at 0.33 ms per expert), routing
the drafter on the GPU (`DSPARK_GPU_ROUTE`), the AVX-512 clamped IQ rows, IQ1SXE4 (its IQ1_S CPU rows are slower per
expert, and its Engram files sit on one OST).

**Exactness.** `VERIFY_EXACT=1` makes a verify pass round as one-token passes do: GEMVs per column, the
single-token BF16 kernel, one CPU-lane group per route, no MMQ.
- With it and the same residency (`DSPARK_OFF=1` loads the drafter but decodes plainly), the speculative output at
  4K context is bit-identical to greedy decoding: 24 tokens, every top-2 logit margin (`MARGINS=1`).
- Against a run without the drafter, tokens can differ at near ties. The drafter's 8 GB moves experts from GPU0
  to the CPU lane, whose Q8_K activations round differently from the GPU's Q8_1: up to ~0.5 in a logit gap.
- The default (batched) verify numerics also differ only by rounding.

## 8g. Batched sessions on 2x V100 (measured 2026-10-23)

`STRATA_DS41_SESSIONS=S` decodes S independent sequences together: each pass carries one token per session.
- **Prompts.** Each session's prompt runs through the single-sequence path. `park(s)` then copies that state into
  per-session buffers: the window ring (window + 8 slots), the compressed caches, the ratio-2 compressor stash and
  the token history (for the Engram hash). It then clears the single-sequence state for the next prompt.
- **Per-session kernels.** Attention, the compressor, index scoring and top-k each take per-token positions and
  per-token window/cache offsets, so each step is still one launch for all sessions. FP32, FP16 and Q8 KV
  (`STRATA_DS41_KV`) all work; with FP16 and Q8 each token's selected rows are gathered to floats from its own cache.
- **Shared code.** The dense layers, the MoE (GPU tier, peer GPU, CPU lane) and Engram see an ordinary pass of S
  tokens. The CPU lane already groups a layer's routes by expert, so each CPU expert is read once per pass.
- **Options.** `STRATA_DS41_SESSION_LEN` is each session's prompt length; session s takes the slice starting at
  s x that length. `STRATA_DS41_SESSION_CHECK=n` re-runs the first n sessions alone and compares their tokens.

**Results.** Full node, 36 lane threads, mixed profile, rebalance, Engram 76 GiB (IQ2_XXSE4) or 70 GiB
(IQ2_XXSE3). 256-token prompts from the wikitext prompt file, plain greedy decoding, 24 tokens per session.
Aggregate tok/s counts every session's tokens:

| sessions | pack / variant | pass (ms) | aggregate tok/s | per session |
|---|---|---|---|---|
| 1 | IQ2_XXSE4 (Engram 64 GiB) | 61 | 16.3 | 16.3 |
| 8 | IQ2_XXSE4 | 174 | 45.9 | 5.7 |
| 16 | IQ2_XXSE4 | 297 | 53.9 | 3.4 |
| 16 | IQ2_XXSE4, `DECODE_GROUPED=1` | 284 | 56.4 | 3.5 |
| 16 | IQ2_XXSE3 (Engram all in RAM) | 262 | **61.2** | 3.8 |
| 32 | IQ2_XXSE4 | 612 | 52.3 | 1.6 |

Single-session plain decode on the same node is 21.9-22.3 tok/s, so 16 sessions reach about 2.5-2.8x.

What bounds a 16-session pass (IQ2_XXSE4, ~290 ms):
- **MoE, ~170-180 ms.** The CPU lane takes ~70 ms per pass. That is about 30 distinct CPU experts per layer at
  ~9 MB each, read at memory bandwidth. The host then waits another ~40 ms for the peer GPU's experts.
- **Dense layers, ~95 ms.** The breakdown is Engram and pre-mix 20-31 ms (10 with Engram in RAM), q 20, attention
  15, attention output 27, the rest ~10. Below `GEMM_MIN` (32 tokens), MMVQ handles at most 8 columns per launch,
  so the dense weights are re-read every 8 tokens.
- **32 sessions cross `GEMM_MIN`.** The dense projections then dequantize to FP16 for cuBLAS (attention output
  111 ms, q 69 ms per pass), so 32 sessions are no faster than 16. Moving the GEMM path down to 16 tokens without
  int8 MMQ falls to 33.8 tok/s at 16 sessions.

Fixed on the way: `resident_mmq` took at most 64 routes per pass (T x top-k). Past 10 sessions the GPU-resident
experts fell back to per-token GEMV chains: 16 sessions ran at 39.0 tok/s, 410 ms per pass. It now takes 512.

**Exactness.** Batched passes round like multi-token passes (MMVQ over several columns, MMQ experts), not like
one-token passes. `SESSION_CHECK=4` (16 tokens each) shows:
- FP32 KV: 3 of 4 sessions identical; the fourth matches its first 8 tokens.
- FP16 and Q8 KV: 2 of 4 identical; the others diverge after 5-13 tokens.

Once one token differs, the rest of that session differs. Near-tie rounding fits this pattern, but it has not been
confirmed against logit margins.

**Next.**
1. Dense projections through int8 MMQ for 16+ token passes (`STRATA_DENSE_MMQ=1` with `GEMM_MIN=16`): each weight
   is read once per pass.
2. The peer GPU's multi-token expert path: why it finishes ~40 ms after the CPU lane.
3. Engram fully in RAM.

If the dense part falls to ~50 ms and the peer stops trailing the lane, a 32-session pass becomes bound by the CPU
lane (~150 ms). That is about 120-150 tok/s aggregate.

## 8h. Long context on 2x V100: 27K to 256K (measured 2026-10-23/24)

**Setup.** The compressed KV caches are stored as FP16 or Q8_0 rows (`STRATA_DS41_KV=f16|q8`). The indexer's score
buffer is tiled (`STRATA_DS41_SCORE_MIB`), so a 2048-token prompt pass at 256K no longer needs a T x context
buffer. A Q8_0 row is 1.7 KB per token, about 0.44 GB at 256K.

**Prompts.** The needle prompts are wikitext with a code word planted at 10 % of the length and asked for at the
end. They are 26.7K, 106K and 255K tokens; the 255K one is built from wikitext-103 train. The continuation prompts are
plain wikitext-103 text with no question.

**First runs.** These ran plain decoding at a 60 GiB Engram budget, without DSpark or rebalance. Every run found the
needle:

| prompt | KV | prefill tok/s | decode tok/s |
|---|---|---|---|
| 26.7K needle | FP32 / FP16 / Q8 | 257 / 227 / 253 | 14.6 / 13.3 / 14.9 |
| 106K needle | Q8 / FP16 | 207 / 148 | 9.9 / 9.6 |

At 106K a token cost ~101 ms against ~45 at 600 tokens:
- **GPU expert hit 0.33 (0.68 at 600).** No rebalance ran after the prompt, so the CPU lane took 29 ms.
- **Engram misses, 20 ms.** Only 60 GiB of the tables was in RAM.
- **Index scoring and top-k, 13.4 ms** (`kv_index`). This is the only part that grows with the context.

Attention stayed at ~3.5 ms.

**Full decode configuration.** IQ2_XXSE4 with DSpark, rebalance, 64 GiB of ranked Engram rows, Q8 KV, the streamed
prompt (`PROMPT_BATCH=2048`) and 32 tokens:

| prompt | index kernels | prefill tok/s | decode tok/s | kv_index ms/pass | GPU hit |
|---|---|---|---|---|---|
| 106K needle | warp per row | 202 | 15.5 | 23.4 | 0.28 |
| 106K needle | `INDEX_FAST` | 242 | 16.7 | 15.9 | 0.28 |
| 255K needle | `INDEX_FAST` | 215 | 17.4 | 34.2 | 0.29 |
| 27K needle | `INDEX_FAST` + `TOPK_CHUNKED` | 179 | 18.8 | 19.7 | 0.28 |
| 106K continuation | `INDEX_FAST` | 244 | 13.2 | 13.1 | 0.50 |
| 106K continuation | `INDEX_FAST` + `TOPK_CHUNKED` | 242 | 15.3 | 8.5 | 0.50 |

Every needle run found the code word. `INDEX_FAST` gave the same tokens as the warp-per-row kernel.

**Index kernels.**
- **`STRATA_DS41_INDEX_FAST=1`** scores with a thread per cache row. The token's queries sit in LDS and every head's
  dot product stays in registers. The default kernel spends a warp per row and reduces across the warp once per
  head. The math at 106K is ~690K rows x 32 heads x 128 per token, about 0.4 ms of FP32.
- **`STRATA_DS41_TOPK_CHUNKED=1`** is the top-k compaction with a contiguous run of keys per thread and two block
  scans. The default kernel ran two scans per 256 keys, ~13K barriers per token and index layer at 106K.
- **Chunked top-k exactness is not confirmed.** `TOPK_CHECK` reported 17 of 128 checked selections differing from
  the host's selection. That needs the same check on the default kernel before chunked top-k can be called exact.

**Where the time goes at long context.**
- **The CPU lane.** It took 68-82 ms per verify pass in the needle runs. Their GPU hit (0.28) is low even after
  rebalance, because the needle runs decode the planted code word and a repeated question, whose routes the prompt
  did not rank. The continuation runs reach 0.50.
- **GPU 0 holds 11.8-12.2 GiB of experts after a long streamed prompt (17.8 at 600).** The prompt's chunk buffers,
  stream slots and score buffer take that VRAM when the residency is placed.
- **The index work grows with context** (8.5-34 ms per pass).
- **The draft.** Plain continuation drafts poorly (1.5 tokens/step against 3.2-3.6 on the needle task).

**For 256K, in order:**
1. Place the decode residency after the prompt, or refill GPU 0 to its 600-token size after it. `GROW` covers only
   GPU 0's freed VRAM and does not move the RAM copy.
2. Confirm or fix the chunked top-k's exactness. Then bring the selection's radix histogram down too (per-warp bins).
3. Rank the routes of the generated tokens, not only the prompt's, for the rebalance. Either rebalance again after
   the first decode steps, or let the tier promote on misses.

## 8i. ISTA-DASLab GSQ-RCO GGUFs on 2x V100 (measured 2026-10-25)

`ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF`: per-tensor quantization types chosen by RCO, standard GGUF
scalar quants (not EXL3). Builds IQ1_M 120.7 GB, IQ2_XXS 138.2 GB, IQ2_XS 155.2 GB, plus one shared Engram file
(IQ4_NL, 110.6 GB). The IQ2_XXS build's routed experts: IQ1_M 59.5 GB, IQ2_XXS 49.1 GB, IQ2_XS 23.6 GB (132.1 GB,
mean expert 8.60 MB against 9.95 for IQ2_XXSE4); gate and up share a type per layer, down may differ. Dense
tensors Q5_K / Q8_0 / BF16, 6.2 GB.

What Strata needed: the separate Engram file (`<arch>.engram.file`), a third metadata schema (Engram hash tables
as integer tensors, `attention.candidate_*` keys), and an IQ1_M MMQ kernel for prompt passes, mapped losslessly onto
Q6_K's tile (`third_party/ggml/patches/apply_mmq_iq1m.py`, within 5e-5 of the exact reference).

Quality: wikitext-2 test, 16 x 2048 tokens, KL against the Q8_0 reference over the second half of each chunk
(llama.cpp's `--kl-divergence` definition; `tools/ds41f_kl_base.py`):

| Pack | Runner | Mean KL | Top-1 | PPL (Q8_0: 2.667) |
|---|---|---|---|---|
| Q23 | llama.cpp | 0.214 | 85.7% | 3.210 |
| IQ2_XXSE4 | llama.cpp | 0.2888 | 83.02% | 3.356 |
| IQ2_XXSE4 | Strata | 0.2901 | 83.02% | 3.361 |
| GSQ-RCO IQ2_XXS | Strata | 0.3785 | 80.43% | 3.540 |

Strata reproduces llama.cpp on IQ2_XXSE4 (chunk 1: KL 0.1889 against 0.1897, PPL 1.851 against 1.852), so the GSQ
number is the pack's own. On this test the GSQ IQ2_XXS build is below IQ2_XXSE4; 49 % of its backbone parameters are
IQ1_M. Its model card reports reasoning benchmarks, which this test does not measure.

Speed (IQ2_XXS build): load 261 s; plain decode wiki600 21.3 tok/s; DSpark code600 25.0 tok/s (4.27 tokens per
step) at Engram 64 GiB (IQ2_XXSE4: 30.1). At Engram 80 GiB the DSpark run did not complete. 2048-token prompt
passes need `STRATA_DS41_STREAM_MIN=1000000` (the streaming buffers otherwise exceed VRAM after residency).

## 8j. IQ2_XXSE4 decode at a node-safe memory budget (measured 2026-10-25/26)

Memory: the decoder holds about 104 GiB plus the Engram RAM (RAM expert copy 94.8 GiB, the rest ~9 GiB). Engram
76 GiB peaked at 180.5 GiB and several runs took the node down (Slurm NODE_FAIL during the load), so the runs here
use Engram RAM 56 GiB (peak RSS 158-167 GiB, 13-21 GiB free on the node). The Engram stall (rows read from Lustre)
then varies by run: the same baseline gave 18.8 and 26.7 tok/s.

Expert kernels (STRATA_EXPERT_V3=1, 69ea6521): IQ2_XXS gate/up and Q2_K down on the S27 multi-row grouped kernels.
Layer 20, per layer: GPU 0 8 routes 0.305 ms (MMQ 0.649), 36 routes 0.547 (0.794); GPU 1 likewise. Per decode pass:
GPU 0 experts 34 -> 25 ms, GPU 1 busy 30 -> 16 ms; verify time per step without the Engram stall 97 -> 86 ms (code600,
two runs each). The CPU lane (26-29 ms) is then the limit. Quality, 1008-token reference in 5-token passes vs
llama.cpp IQ2_XXSE4: MMQ KL 0.0229, S27 grouped 0.0259, GEMV chains 0.0198 (all equivalent). On code4k its outputs
lowered draft acceptance (3.0 vs 3.7 tokens per step), so it is not the default.

Engram table types at the same 56 GiB (ranked rows from one wiki + code profile; the profile is row ids and counts,
so it serves every table type), baseline config (GROW, DECODE_MMQ, DSpark), decode tok/s:

| Pack | Engram tables | KL vs Q8_0 | code600 (128 tok) | wiki600 | code4k |
|---|---|---|---|---|---|
| IQ2_XXSE4 | Q4_K, Q4_K | 0.289 | 24.2 | 26.0 | 24.7 |
| IQ2_XXSE43 | Q4_K, Q3_K | 0.307 | 29.8 | 27.4 | 23.9 |
| IQ2_XXSE3 | Q3_K, Q3_K | 0.314 | 31.0 | 30.0 | 26.1 |

E3 is +28 % on code600 (Engram stall 1.69 -> 0.78 s per 128 tokens), +15 % on wiki600, +6 % on code4k, for +0.025
KL. On code4k E43 / E3 generate different text and accept fewer drafts (3.15-3.2 vs 3.71 tokens per step).
Engram read options on E3 code600 changed nothing beyond noise: 256 read threads 31.0, prefetch threads 128 / 32
29.8 / 31.7 (default 64: 31.0).

The S27 expert kernels on IQ2_XXSE3 (same session, Engram RAM 56 GiB, peak RSS 155 GiB):

| Run | Baseline (MMQ) tok/s | S27 tok/s | Tokens per step (base / S27) | Verify ms per step without Engram (base / S27) |
|---|---|---|---|---|
| wiki600, run 1 | 29.96 | 32.89 | 2.56 / 2.63 | 69.2 / 64.3 |
| wiki600, run 2 | 29.89 | 31.91 | 2.56 / 2.63 | 69.2 / 66.8 |
| code600, 128 tokens | 33.22 | 34.86 | 4.27 / 4.00 | 94.6 / 83.6 |

wiki600 +8 % (both repeats), code600 +5 %: the per-step compute falls 4-12 %, but on code S27's slightly different
rounding changes the text and lowers draft acceptance, as on E4 code4k. Best configuration at a node-safe memory
budget so far: IQ2_XXSE3 + S27 (STRATA_EXPERT_V3=1 STRATA_DS41_DECODE_GROUPED=1 STRATA_DS41_GROWN_GROUPED=1) with
GROW and DSpark: wiki600 32-33, code600 34.9 tok/s.

Code acceptance with S27, a second session (E3, 128 tokens; greedy, so a repeat gives the same tokens):

| Prompt | Baseline tok/s | S27 tok/s | Tokens per step (base / S27) |
|---|---|---|---|
| code600 | 34.92 | 35.98 | 4.27 / 4.00 |
| code4k | 31.08 | 33.22 | 3.66 / 3.53 |

The lower acceptance with S27 is systematic (code600 4.27 -> 4.00 in both sessions; code4k -3.5 %), smaller than its
per-step saving: S27 is ahead on every prompt measured (code600 +3 %, code4k +7 %, wiki600 +8 %). Best at Engram RAM
56 GiB: IQ2_XXSE3 + S27, wiki600 32-33, code600 36.0, code4k 33.2 tok/s.

## 8k. TR16 (RX 9070 XT 16 GB, 128 GB RAM): UD-IQ1_S decode and prefill levers (measured 2026-10-27)

Unsloth UD-IQ1_S, Engram on the local NVMe (O_DIRECT), 15 lane threads, `numactl --interleave=all`, code600 prompt,
64-128 generated tokens. Run-to-run noise on the shared desktop is about +-1 tok/s. All switches below are opt-in.

**Decode** (tok/s; "same tokens" = identical greedy output to the row above it):

| Configuration | Decode | Notes |
|---|---|---|
| defer3 (previous best) | 10.9-11.7 | lossy, KL 0.078 vs the exact run |
| + DECODE_GROUPED, GPU_RESERVE_MIB=1536, EXPERT_PROFILE | 13.2-13.5 | same tokens |
| + FUSE=1 SHARE_Q8=1 | +6-8 % | same tokens |
| + IQ1S_FAST=1 (lean IQ1_S lane rows) | +5-7 % | lane bench +12 % |
| + GPU_ROUTE=1 (top-k and GPU-held experts on the GPU), GROW off | 15.9-16.0 | same tokens, KL unchanged |
| defer-all + GPU_ROUTE | 16.3 | lossy, KL ~0.2 |
| defer-all + CPU_TOPK=4 + GPU_ROUTE | 18.7 | lossy |

- Q6_V2=1 (warp-per-row Q6_K GEMV): attention-output phase 10.1 -> 6.6 ms per token, same tokens.
- No gain on TR16 (kept opt-in): LANE_BANDS (NUMA band layout in place by move_pages, PR #4016's idea: 12.8 vs 13.2;
  prompt 21 -> 14 tok/s), threads 16/24/31, LANE_ASYNC, SYNC_SPIN, huge pages for the lane (a huge-page copy of the
  experts reads at the same 15-17 GB/s as the 4 KiB page-cache pages).
- MISS_READ=N (queued expert reads): with 30 % of the experts paged out, 7.2 -> 7.7 tok/s at the drive's 1.6 GB/s;
  no effect when every expert is cached.
- Lossy KL (1008 tokens in 5-token passes, against the exact run): DEFER_RANK=3 0.078, =1 0.146, DEFER_ALL 0.200,
  CPU_TOPK=4 0.145, =3 0.240, GPU_SUBST=0.8 0.076, =0.6 0.118.
- The limit: the CPU lane, ~50 ms a token for ~0.59 GB of IQ1_S experts (12 GB/s in decode against 20-26 GB/s in the
  lane bench), plus ~25 ms of GPU work. An exact run tops out near 18 tok/s; 30 needs deferral (lossy) and a faster lane.
- RAM_COPY on TR16 triggered systemd-oomd (86.6 GB of anonymous copies plus reclaim): it killed the session's scope.
  Runs go in their own `systemd-run --user --scope`.

**Prefill** (tok/s; the warm column is the second of two identical prompts, so nothing pays its first-touch cost):

| Prompt | Default | PIN_GB=100 STREAM_MIN=256 PREFILL_HYBRID=0 KV=f16 INDEX_GEMM_MIN=64 | Long-prompt set (2026-10-27) |
|---|---|---|---|
| 600 | 16.8 (chunked, CPU experts) | 52-62 | 83 gated / 61 ungated |
| 4K | 201 | 249 | 331 (12.4 s) |
| 8K | 148 | ~330 warm | 329 (24.90 s: 12.34 + 12.56) |

Long-prompt set: the 249/330 switches with `PREFILL_HYBRID` left unset (ungated prefetch), `STRATA_DS41_STAGE_SLOTS=8`,
`STRATA_DS41_DESKTOP_MIB=1024`, `STRATA_DS41_ENGRAM_THREADS=256`. Encoded, 8K went from 26.6 s (308 tok/s) to 24.9 s
(329 tok/s) warm; the first prompt pays 16.96 s for the first streamed pass. Those numbers use a 256 MiB stream ring
and leave only ~2.2 GB of VRAM. With the 2 GiB desktop reserve that the decoder now enforces (below) the ring is
128 MiB and 8K/16K run at 306/307 tok/s.

- A streamed pass streams every expert over PCIe 3.0 x16. PIN_GB=100 pins 37 of 40 layers in place for direct DMA
  (no extra RAM).
- **The expert DMA is the limit, not the GPU.** A 4096-token pass moves 116.1 GB (all 15360 expert slots less the
  resident tier, which is lent back to the prompt buffers): `STRATA_DS41_PREFILL_TIMING=1` reads the copy stream at
  12.47-12.56 GB/s, i.e. 9.3 s busy per pass. The card negotiates 32 GT/s to its switch, but the Threadripper 1950X
  root port (`00:03.1`) is 8 GT/s: 15.75 GB/s raw, and a pinned H2D microbench reaches 12.2-13.0 GB/s. An 8K prompt is
  two 4096 passes, so it pays that 9.3 s twice; nothing in the trunk can recover it.
- Phase split of a warm 4096 pass (`PREFILL_TIMING`): 12.4 s = 1.0 s of GPU wait for the slots + 4.1 s attention +
  ~4.8 s expert products (2.4 s with the DMA idle) + 1.9 s dense/trunk + ~1.0 s Engram. Every GPU phase runs 25-70 %
  slower while the DMA is in flight (attention 3.1 -> 4.1 s, dense 1.3 -> 2.2 s); it is contention, not clocks
  (sclk held 3.1-3.3 GHz, 100-260 W through the pass).
- `STRATA_DS41_EXPERT_DEBUG=1` (copies kept, products skipped) times the expert segment at 8.0 s; `=2` (products on
  stale slots) at 2.4 s. The products hide under the transfers; the transfers do not hide under the products.
- `PREFILL_HYBRID=0` makes the producer wait for each layer's routing before it copies, which idles the copy engine
  between layers: 4K 13.2 s gated vs 12.4 s ungated, 8K 25.3 vs 24.9. Keep it only for short prompts, where skipping
  unused experts outweighs the stall: 600 tokens 7.2 s gated vs 9.8 s ungated (the pass reads 44 GB of 108 there).
- Engram reader threads (256) do cut the row read (1.6 -> 0.4-1.7 s a pass) but it overlaps the DMA, so the pass time
  barely moves.
- `PREFILL_HYBRID>=1` (low-count experts on the CPU lane during the stream) froze the machine once (no kernel message):
  do not use until understood.
- After a streamed prompt the GPU tier now also takes the prompt's routing (TIER_PROMPT_WEIGHT, default 10): tier hit
  0.10 -> 0.59, decode after a 600-token prompt 6.1 -> 10.3 tok/s (as after a chunked prompt).
- **`PROMPT_BATCH=8192` is the one lever left and it still does not fit.** `cudaMalloc` fails at load: the 8192 buffers
  are roughly 4-6 GB above the 4096 set (q_ and att_ 1.07 GB each, kv_sel_ 1.34 GB, engram_kv_ 0.84 GB, h_/h2_/flat_
  +1.0 GB, xh_ +0.3 GB), against ~3.1 GB free after dense weights and the 4096 buffers. One 8192 pass would move the
  116 GB once instead of twice (9.3 s) and, by the split above, put 8K near 16 s (~510 tok/s). The way there is a
  layer-major pass whose trunk runs each layer over two 4096-token halves while the FFN runs over all 8192 rows: the
  trunk buffers stay at 4096, only the residual/FFN arrays grow (+1.3 GB). Not attempted yet.
- Diagnostics added for this (opt-in, byte-identical when off: the 8K run's tokens match the pre-change binary exactly):
  `STRATA_DS41_PREFILL_TIMING=1` (per-pass phase split and `chunk_dma GB= copy_s= GB/s= gpu_wait=`),
  `STRATA_DS41_EXPERT_DEBUG=1|2`.

**The 2 GiB desktop reserve (2026-10-27).** The decoder now leaves `STRATA_DS41_DESKTOP_MIB` (default 2048) MiB of VRAM to the
compositor: every device allocation refuses (with the requested size, the free size and the reserve) if it would dip
below it, the GPU-resident budgets (tier, grow, slots) are clamped to it, and the stream-slot size halves itself down
to 32 MiB when the reserve leaves no room for two slots. On this desktop a decoder that grew past the reserve took the
GNOME session down once. The measured price of the reserve is the stream ring: 8 x 128 MiB instead of 8 x 256 MiB
(the 256 MiB ring no longer fits beside it), and a 4096 pass goes 12.45 -> 13.2 s.

| 8K / 16K, warm, with the reserve | pass 1 | pass 2 | pass 3 | pass 4 | tok/s |
|---|---|---|---|---|---|
| 8K (two 4096 passes) | 13.25 | 13.49 | | | 306 |
| 16K (four 4096 passes) | 13.13 | 13.38 | 13.45 | 13.43 | 307 |

- The 8192-token single pass (which would halve the DMA) **does not fit under the reserve**. With the scratch tiles
  below it needs ~7.4 GB of pass buffers + 6.3 GB of dense weights + 2.0 GB of reserve + ~0.6 GB of MMQ stream-k
  scratch + slots: ~14.5 GB of the 16.3 GB card, i.e. ~0.6 GB over. It ran to `cudaMalloc` at 2.0 GB free.
- New scratch tiles, all opt-in and byte-identical off (`ENGRAM_TILE=n` the Engram projection, `MIX_TILE=n` the
  hyper-connection mixes, `SHEXP_TILE=n` the shared expert, `OUTA_TILE=n` the attention output projection and its FP16
  conversion, `SCORE_MIB=n` the indexer's score tile): together they cut a 8192-token pass's buffers by ~1.6 GB
  (Engram 0.8, mixes 0.6, shared expert 0.2, xh_ 0.47) without touching the kernels. `STRATA_DS41_MMQ_ROWS` caps the
  MMQ row buffers, and an expert whose rows exceed the cap now fails loudly instead of overrunning them (an 8192-token
  prompt routed 5245 tokens to one expert).
- **500 tok/s at 8K is not reachable on this card with these kernels.** 500 needs 16.4 s for 8192 tokens; a single
  8192 pass would have ~17 s of compute alone (attention 6.3 s, expert MMQ 6.6 s, dense 2.6 s, Engram 2 s) because
  every GPU phase runs ~35 % slower while the DMA is in flight, and the 9.3 s of DMA hides under that. The way to 500
  is ~1.4x faster attention and expert kernels (both run at ~9 TFLOPS today), or the layer-major pass (trunk over two
  4096 halves, FFN over 8192 rows) which keeps the fast attention and halves the DMA but leaves the same compute.
**PREFILL_HALVES (2026-10-27): one expert stream per prompt, trunk per tile.** `STRATA_DS41_PREFILL_HALVES=1
STRATA_DS41_PREFILL_TILE=4096` runs each layer's trunk over 4096-token tiles (the attention-local buffers hold the tile,
so `q_`/`att_`/`oa_`/`xh_`/the ring are tile-sized) and its FFN once over the whole pass: an 8K prompt becomes one
8192-token pass, so the experts cross PCIe once (116 GB, 9.3 s) instead of twice. The residual and FFN arrays stay at
the pass width, and the trunk's tiles are swapped in and out of them (h, x, coefficients; ~60 MB of copies per pass).

| 8K / 16K, warm, 2 GiB reserve | halves (row truncation, KL 0.0007) | halves, no truncation | two passes, default |
|---|---|---|---|
| 8K | 20.1 s, **408 tok/s** | 25.1 s, 327 | 26.8 s, 306 |
| 16K | 41.5 s (19.9 + 21.7), **394 tok/s** | 51.6 s, 318 | 53.4 s, 307 |

- Quality: the last position's logits match the two-pass path to **KL 0.0008** (the chunking noise floor is 0.0006-0.001,
  so this is exact within measurement), top-1 identical. `--dump-logits` both and `cmp_f32`; the comparison runs with
  enc-skip as shipped.
- **The row truncation inside the merged pass needed one invariant fixed.** `sel_`/`sel_count_`/`sel_cnt_` (the indexer's
  per-token selection, written at the indexer-source layer and read by later layers that reuse it) were indexed relative
  to each call. `narrow()` keeps that consistent because it shifts the buffers to the pass's current first token; a
  per-layer row range does not, and layers with different kept ranges read each other's selections. Indexing the
  selection by the pass's absolute row instead (`sel_base_`, T rows instead of a tile's) took the same truncation from
  KL 1.06 to KL 0.03, and a 1024-row boundary slack (so the first truncated tile's windows are complete) from 0.03 to
  **0.0007**. The final row range per layer is `end - need - 1024`, 256-aligned, with `need = window + (layers-1-l)*(window-1)`.
- The MMQ grouping was also tried with hot experts split into their own products (a 4K pass routed 5245 tokens to one
  expert, and the launch's grid is max_rows x experts): it cost 2.5 s, so the plain row-cap grouping stays.
- With the truncation, the pass is no longer DMA bound: 20.1 s at 8K of which 9.5 s is DMA and the compute is
  attention ~5.5 s + output projection 1.5 s + FFN (router, shared expert, 8192-row expert MMQ) 11.0 s.
- **The FP16 KV cache is now read by the attention kernel directly** (`ds41_attention_batched(..., comp_half)`: the
  compressed rows are loaded as `uint16_t` and converted in the tile loader). Before, each 256-token launch gathered its
  512 selected rows per token to a float buffer (~500 MB per launch); the same values are now converted at read time,
  so the result is unchanged (KL 0.0008 both ways) and an 8K pass went 20.8 -> 20.1 s.
- 500 tok/s at 8K needs ~1.4x faster attention and expert kernels (both run at ~10 TFLOPS of a ~50 TFLOP FP32 /
  ~97 TOPS INT8 card). The tensor-core attention kernel (`attn_wmma_kernel`, Volta+) is CUDA-only - it is not compiled
  on HIP - so the HIP attention is the CUDA-core tiled kernel. `STRATA_DS41_ATTN_V1=1` (the older batched kernel) is not
  a path: with FP16 caches it faults.

**The gfx12 WMMA attention port (designed, not written).** The reference is `~/work/gemm/pixal3d/rdna4` on this card:
gfx12 has the WMMA builtin `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(A, B, C)` (wave32; lane l -> half = l/16,
id = l%16; A[i] = A[id][half*8+i], B[i] = B[half*8+i][id], C[i] = C[half*8+i][id]), and its FA2 port measures
**37 TF/s** for an f16 flash attention at head_dim 128 (19.5 % of the 190 TF/s microbench peak) with an LDS XOR swizzle
(`d ^ ((row&7)<<4)`) as its main lever. Two findings bound the port: (a) 16-bit LDS runs out past head_dim 128 (`smK +
smV + smP > 64 KB`), so a d512 kernel must stream the k dimension in 64-dim chunks; (b) synchronous WMMA on RDNA4 caps
around 30-40 TF/s - the 80-90 % figures need async MMA and a 5090-sized register file - so plan for ~3x, not ~10x.
Mapping onto `attn_tiled_kernel`'s block (one token, 16 heads, 64-row tiles, 8 waves): QK is the FA2 pattern exactly
(m = the 16 heads, n = the 64 rows, k = dim; the row list is the window ring + the indexer's selected FP16 cache rows,
which must be converted to FP16 as they are staged); PV needs P as [head][row] in LDS and V transposed to [dim][row],
with the 16 heads x 512 dims output split over the 8 waves (64 dims each = 4 float8 = 32 floats per lane, the same
register budget the current kernel keeps). The k = 512 reduction for QK has to be split over the waves too, with an LDS
partial sum (8 x 1024 floats = 32 KB, or float LDS atomics into 4 KB) and one sync per tile. The LDS budget is the binding constraint: this card reports **64 KB of shared memory per block**
(128 KB per CU, two blocks), so a whole 64-row x 512-dim FP16 K tile (64 KB) leaves no room for Q. Use **32-row
tiles**: Q FP16 16 KB + K 32 rows x 512 dims FP16 32 KB (each wave reads its own 64-dim slice of the *same* tile, so the
k split needs no extra LDS) + 8 waves of [16 heads x 32 rows] FP32 partials 16 KB (or 4 waves, 8 KB) + P 1 KB + scores
2 KB = 59 KB. The same buffer becomes V transposed to [dim][row] for the PV phase after the QK of a tile, so 20 tiles
per 640-row token.
**First cut built and measured (2026-10-27): `attn_wmma_gfx12_kernel`, `STRATA_DS41_ATTN_WMMA=1`.** One block per
(token, 16-head group), 256 threads (8 waves), 32-row tiles: Q 16 KB + K 32 KB + scores 2 KB + P 1 KB = 60 KB of LDS.
QK is 8 waves x 4 of the 32 k-steps summed with LDS atomics; PV is 8 waves x a 64-dim output slice with V staged
transposed. It is correct - the 8K prompt's logits match the tiled kernel to KL 0.0033 (the FP16 Q/K/V conversion),
top-1 identical - but **slower: the 8K pass 23.6 s against 20.1 s** (attention ~9 s against ~5.6 s). The cause is the
tile: at 32 rows the two LDS staging passes (K, then V transposed) and the four syncs per tile outweigh the MMA work,
and each staging element only feeds 16 MACs. A per-element staging was replaced by 8-element vectors (23.6 -> 21.0 s)
but that version changed the values (KL 6, top-1 0%) and its layout turned out to be identical on inspection - an
unresolved bug; it is not in the tree. Next: 64-row tiles with the k dimension streamed in two 256-dim chunks (Q 16 KB +
K/V 32 KB + scores 4 KB + P 2 KB = 54 KB) - the same staged bytes per row but twice the MACs per element and half the
syncs - and the vectorized staging debugged.

**64-row scalar staging experiment (2026-10-27).** Streamed K and V in two 256-dimension chunks, with 64-row
softmax and eight-wave atomic QK reduction. On the RX 9070 XT, the second 8K pass took **26.08 s (314 tok/s)**;
`PREFILL_TIMING` measured attention 11.36 s and experts 11.24 s. Logits against the tiled path had mean KL 0.0028,
p99 0.003 and top-1 agreement 100%. The first pass took 54.32 s with unusually high Q and attention times; it is not a
warm timing. This scalar-staging implementation was reverted: it loses to the recorded 32-row WMMA (23.6 s) and tiled
(20.1 s) runs. Those comparisons are individual runs, not repeated measurements. A later 64-row vector-staged version measured 23.48 s
warm, mean KL 0.0032; replacing atomics with partial sums measured 19.29 s, mean KL 0.0028. Both were slower than the
32-row approach below. The scalar run used two GPU and six host 64 MiB stream slots and the default desktop reserve.

The subsequent `STRATA_DS41_GPU_BENCH=1` expert comparison exited with SIGSEGV after the device-to-device test, before
printing the expert timings. The tier had not been initialized in the layout the benchmark expected; the focused
prefill mode below now uses a loaded tier expert.

**8K prefill above 500 tok/s (2026-10-27).** On the RX 9070 XT / Threadripper 1950X, 128 GiB RAM, with the
UD-IQ1_S model and warm page cache, the opt-in `STRATA_DS41_ATTN_WMMA=2` uses 32-row gfx12 WMMA attention: aligned
8-value staging, four-way QK partial sums instead of LDS atomics, and a swizzled K tile reused for V. Set
`STRATA_DS41_MMQ_ROWS=256` for this run: at an 8192-token pass this sets a 4096-row expert-product cap. An expert with
more rows is split into consecutive products; all its routes remain in the sum. The smaller buffers let the pass use
**three 128 MiB GPU stream slots** within the 2 GiB desktop reserve, against two 64 MiB slots at the default scratch cap.
The other flags were `DECODE_GROUPED=1`, `GROW=0`, `GPU_RESERVE_MIB=1536`, `FUSE=1`, `SHARE_Q8=1`,
the route profile, `ENGRAM_DIRECT=1`, `STREAM_MIN=256`, `ENGRAM_THREADS=256`, and
`PREFILL_HALVES=1`, `PREFILL_TILE=4096`, `PROMPT_BATCH=8192`, `ENGRAM_TILE=256`,
`SHEXP_TILE=2048`, `OUTA_TILE=4096`, `SCORE_MIB=16`, `STAGE_MIB=128`, `STAGE_SLOTS=6`, `PIN_GB=100`, `KV=f16`, and
`INDEX_GEMM_MIN=64` (all with the `STRATA_DS41_` prefix), plus `STRATA_IQ1S_FAST=1` and `STRATA_Q6_V2=1`.
The normal truncation slack stayed at 1024.

Three identical 8K prompts in one process took **18.41 s for the first pass, then 15.28 and 15.22 s warm (536 and 538 tok/s)**.
After a benchmark-only buffer fix and rebuild, a repeat took 18.01 s, then **15.64 and 14.66 s (524 and 559 tok/s)**.
Both runs used `PREFILL_TIMING=1`; the same-day earlier tiled baseline was 20.08 s warm (408 tok/s), with the larger default scratch
cap. On the last warm pass, attention was 1.83 s, output projection 1.49 s, and experts 9.71 s including 2.66 s of GPU
copy waits; 116.1 GB crossed the copy stream in 9.28 s. These are two processes with two warm repetitions each, not a broader statistical run. For the two dumped next-token logits, mean KL versus the earlier tiled reference was 0.0021 and top-1
agreement 100%; the split and unsplit optimized runs agreed to the comparator's printed precision. This gate covers those
two positions, not a corpus-wide quality evaluation.

The default-path 8K token output matched the saved two-prompt baseline byte-for-byte before the benchmark-only fix; this
checks the generated token line, not every intermediate or logit. The final repeat HIP decoder SHA-256 was
`573e8df88281c3e3988fd7e417d2c281d968a8887d9ac5ee3718f4f330d90113`; the attention and decoder source SHA-256s were
`64dd366621ff85fa290c8426265f8f37ab1c15a85118f34afe620ce6d738e06d` and
`9abb58aeb9a5e834d14a63ac67b43467e2726c928e9d766bb12e59e5300a7530`. RAM speed was not recorded in this run.

At 16K, the same configuration had two 128 MiB GPU slots; the warm prompt took **33.31 s (492 tok/s)** across two 8K
chunks, against **38.22 s (429 tok/s)** for a same-day tiled run with the same split-expert settings. The two dumped
next-token logits had mean KL 0.0010 and 100% top-1 agreement against that tiled run. An attempt to lower truncation
slack to 768 failed the 8K quality gate (mean KL 0.0204); 512 also failed (0.0668). The 1024 default is unchanged.

The first isolated resident-expert comparison was invalid: its output buffer was too small for the down projection.
After fixing it, one 20-repetition run with `GPU_BENCH=prefill` measured IQ1_S gate/up and Q2_K down dequantization at
0.034 and 0.020 ms respectively. At M=128, MMQ including activation quantization took 0.151 / 0.085 ms, while the
already-dequantized FP16 hipBLAS GEMM took 0.102 / 0.042 ms. These timings omit routing and per-expert launch costs and
do not by themselves justify changing the streamed path. The earlier general GPU benchmark crashed when invoked without
its contiguous resident layout; it now rejects that situation, and the `prefill` mode uses a loaded tier expert.

**Measured issue rate at these shapes** (`tools/hip/wmma_attn_probe.hip`, one block per token, 16 heads x 64-row tiles
x 512 dims, 8 waves with a 64-dim k slice each and an LDS partial-sum reduction, QK + PV): **26-45 TF/s** (44.5 at
steady state, 8192 blocks) against ~10 TFLOPS for the CUDA-core kernel - a 3-4x headroom on the attention. The probe
keeps its inputs in L1 and has no softmax/P conversion, so the real kernel will land below it.
Expected: the attention is ~5.6 s of the 20.1 s pass; at that rate it becomes ~1.5-2 s, i.e. 16-16.5 s and
500-510 tok/s at 8K. It converts Q/K/V to FP16 (FP32 accumulate), so it needs its own opt-in switch and the KL gate
against the current logits.
- Tiling the attention itself was tried and rejected: `q_`/`att_`/the output projection per 256-2048 tokens takes a
  4096 pass from 3.6 s to 6.1-6.3 s of attention+output (the Tensor-core GEMMs at M = 256-2048 run at a third of the
  M = 4096 rate), so it saves the 3.5 GB but costs 2.5 s a pass - more than the second DMA pass it would buy. The
  window ring does not need it: writing the KV rows per attention batch (instead of per pass) shrank the ring from
  681 MB to 31 MB at 8192 with no measurable cost.



## 9. Next steps (in order)

1. GPU node: probe storage (sequential MB/s, random 4 KiB latency) and the download path; fetch the non-Engram shards;
   run the Q2_K/Q3_K bring-up job (DS41F_QUANT_JOB.md). Record the Xeon's quantizer rate and the GPU node -> TR16 rate.
2. A64FX, one 12-node allocation: exact reference logits on the eval corpus and the imatrix dump (`ds41f_run`).
3. GPU node: IQ1_S full and IQ2_XXS/Q2_K full packs; the gate on each.
4. TR16: `iq1s_w1a8_bench --threads 15` on an idle CPU; the Engram page-read cost with a 1 GB row cache on the x4 NVMe;
   decide whether DSpark acceptance on this model justifies the width-4 path.
5. Clear SSD space (TR16 needs ~320 GB) and transfer; phase 2: the `deepseek4` runner.

Reproduce: `python3 tools/glm_ds41f_sim.py validate | report | packs --machine tr16 | a64fx --nodes 6 8 12`;
`python3 -m unittest tools.test_sim_ds41f`; `python3 tools/test_ds41f_pack.py`; `build-ds41f/iq1s_w1a8_bench [--bench]`.

**Route-gated streamed prefill follow-up (2026-10-27).** With `STRATA_DS41_PREFILL_HYBRID=0`, the producer waits for each layer's router and avoids copying experts not selected in that layer. On this 8K prompt it skipped 82.52 GB of transfers (corrected byte accounting; 89.48 GB were copied), but the warm prompts took 14.34 and 14.19 s (571 and 577 tok/s) with timing enabled. A second run without detailed GPU timing took 15.57 and 14.21 s; this variation does not establish a repeatable speedup over the ungated 524-559 tok/s runs. The two-position logit check was mean KL 0.0021, p99 0.002, top-1 100%. `PREFILL_HYBRID=0` is safe; values >=1 remain prohibited on this desktop.

Two follow-up sweeps did not improve the result: `OUTA_TILE=8192` retained ~1.55 s output projection and warm passes of 15.58 and 15.62 s (same quality gate); `STAGE_MIB=160 STAGE_SLOTS=4` fit only two GPU slots and took 16.08 and 14.27 s warm. Keep `OUTA_TILE=4096` and 128 MiB stage buffers. With route gating on, the present bottleneck remains streamed expert compute/transfer (about 9 s of expert products and 12.6 s copy-stream event time on the fastest warm gated pass); copy event intervals also include idle and host routing waits, so they are not active-DMA measurements. These are short repeated-in-process trials, not a stable multi-run benchmark. The 600 tok/s goal remains unverified.

**Expert row-cap follow-up (2026-10-27).** Raising the cap to 8192 removed more expert splits, but scratch allocation reduced the stream ring from three 128 MiB slots to two 64 MiB slots. With safe route gating, warm 8K passes took 14.55 and 14.71 s; KL remained 0.0021 / p99 0.002 / top-1 100%. This is slower than the best 4096-cap gated pass (14.19 s), so retain `MMQ_ROWS=256` (4096 effective cap) for the measured setup.

**Expert-kernel and smaller-slot probes (2026-10-27).** The focused resident IQ1_S benchmark on the RX 9070 XT compared MMQ (including activation quantization) with dequantization to FP16 plus cuBLAS GEMM. At M=64, gate/up was 0.080 ms MMQ versus 0.092 ms dequant+GEMM; down was 0.045 versus 0.047 ms. At M=128, dequant+GEMM sums were 0.136 ms gate/up and 0.063 ms down, versus 0.150 and 0.073 ms MMQ. At M=256, MMQ was faster: 0.220 versus 0.297 ms gate/up and 0.106 versus 0.086 ms down. These are sums of separate weight-dequant and GEMM event timings with synthetic inputs; they omit the FP16 activation conversion, grouped launch behavior and prompt routing, and are not a quality test. The small M=128 advantage does not justify replacing the grouped MMQ path.

Reducing staging from 128 to 96 MiB increased the ring from three to four GPU slots, but the ungated 8K warm passes took 15.84 and 15.60 s (517 and 525 tok/s), with mean KL 0.0021 / p99 0.002 / top-1 100%. The copy stream carried 116.1 GB in 9.64-9.68 s (about 12.0 GB/s) and GPU waits were about 2.14 s. Keep 128 MiB slots. The 600 tok/s target remains unverified.

Combining route gating with 96 MiB slots kept four GPU slots and measured 14.36/14.33 s warm (571/572 tok/s), mean KL 0.0021 / p99 0.002 / top-1 100%. It did not beat the best 128 MiB gated result. The measured 8K result remains above 500 tok/s; 600 tok/s is still unproven.

**Route-packing and projection-tile probes (2026-10-27).** A counted contiguous route-pack implementation preserved the generated token line and the two-position logit result (mean KL 0.0021, top-1 100%) but did not improve host enqueue time or warm throughput: two runs measured 15.78/15.76 s and 15.85/15.67 s. The change was reverted; keep the existing per-expert route packing. `SHEXP_TILE=4096` reduced shared-expert chunk count but used enough scratch to cut the ring from three 128 MiB slots to two; warm passes were 17.51/16.34 s (KL 0.0016). `ENGRAM_TILE=512` likewise fit only two slots and measured 16.41/16.17 s (KL 0.0021), with no change to the ~0.53 s per-pass Engram time. Retain `SHEXP_TILE=2048` and `ENGRAM_TILE=256` for this configuration. Neither tile experiment supports a performance gain.

**Latest projection and index-buffer sweeps (2026-10-27).** The larger shared-expert tile and Engram tile did not help: 4096 shared-expert rows reduced the stream ring to two 128 MiB slots and the warm passes were 17.51/16.34 s (KL 0.0016); 512 Engram rows also left two slots and measured 16.41/16.17 s (KL 0.0021), with Engram time unchanged. Keep 2048 / 256.

Increasing `SCORE_MIB` from 16 to 64 cut the measured 8K index phase from about 0.44 to 0.25-0.27 s, but 64 MiB of score storage reduced the ring to two 128 MiB slots. Pairing it with 96 MiB staging restored three slots but warm passes were 15.80 s. `SCORE_MIB=32` kept three 128 MiB slots and index time fell to 0.34 s, but warm runs were 15.53 and 16.83 s; no throughput gain was demonstrated. Keep the proven 16 MiB setting until an index change can retain overlap and show repeated end-to-end improvement. The two-position quality checks were KL 0.0020-0.0022 with 100% top-1.

**Profile and six-slot transfer follow-up (2026-10-27).** Replacing the 1008-token route profile with a captured 8193-token profile for the same prompt raised the reported tier profile hit from 0.639 to 0.676, but streamed 116.1 GB as before and warm passes were 16.47/15.71 s; no prefill gain was demonstrated. `STAGE_MIB=64` fit six GPU slots and reduced GPU wait from roughly 2.5 to 1.6 s, but doubled per-pass stream items (910 to 1862), reduced copy throughput to 11.8 GB/s, and warm passes were 16.39/16.46 s. Keep the 1008-token profile and 128 MiB slots for the current measured setup. Two-position logits checks remained KL 0.0021 and top-1 100%.

**Pinned-host and full-trunk-tile probes (2026-10-27).** Increasing the pinned expert budget from 100 to 105 GiB pinned 39/40 layers and reduced host-stage copy time from ~0.83 to ~0.30 s, but 8K warm passes were 16.06 and 17.42 s; keep 100 GiB. A single 8192-row trunk tile without extra scratch reduction was safely refused by the allocator (free memory was below the reserve plus slot margin). Combining it with `MIX_TILE=1024` ran with only two 32 MiB GPU slots: attention fell to ~1.52 s and output projection to ~0.75 s, but expert time rose to ~11.3 s, with warm passes 16.37/16.16 s and KL 0.0022. Keep the two 4096-row trunk tiles. The reserve stayed intact.

**MMQ grouping follow-up (2026-10-27).** With safe route gating and the same 8K setup, baseline warm passes were 14.74 and 14.04 s. `STRATA_DS41_MMQ_COMPACT=1` measured 13.83 and 13.95 s; the warm pair without detailed GPU event timing was 15.75 and 13.85 s. The logits were byte-identical to the non-compact run and passed the tiled reference check (mean KL 0.0021, p99 0.002, top-1 100%). The baseline's own spread and the no-timing spread mean compact grouping has not shown a repeatable end-to-end gain. Reducing the MMQ effective row cap from 4096 to 2048 measured 14.69 and 13.89 s, also without a repeatable gain; logits passed the same check. Keep compact grouping opt-in and the measured 4096 cap. Fastest measured result in this follow-up was 592 tok/s; 600 tok/s requires about 13.65 s per 8K pass, and higher targets remain unverified. These are short in-process trials on the RX 9070 XT with the 2 GiB reserve, 128 MiB stream slots, and warm page cache, not a stable multi-run result.

**Scatter and staging follow-up (2026-10-27).** A parallel atomic scatter was tested on the RX 9070 XT under safe route gating, then removed: warm passes were 14.16/14.12 s versus 13.63/13.71 s for the ordered scatter, with the same 2 GiB reserve and three 128 MiB GPU slots. The candidate logits passed the tiled check (mean KL 0.0023, p99 0.002, top-1 100%) but differed slightly from ordered accumulation, and it did not improve speed. The 8-thread host staging baseline was compared with 16 threads (13.82/13.87 s) and 4 threads (13.69/13.73 s); neither showed a repeatable improvement. Retain 8 threads. A single 8-thread baseline pass measured 13.63 s (601 tok/s), the second 13.71 s (598 tok/s), and the pair averaged just below 600. A full 8193-token route profile raised the tier's reported profile hit from 0.639 to 0.676, but streamed bytes stayed 89.48 GB; warm times were 13.65 and 17.13 s. It did not show an end-to-end improvement. The reliable measured target remains 500+; 600+ is not yet repeatable.

**Ordered grouped-scatter probe (2026-10-27).** Reordered each streamed down-product's output by token, then accumulated each token's routes sequentially in their original order. The logits were byte-identical to the ordered-scatter baseline and passed the tiled reference check (mean KL 0.0021, p99 0.002, top-1 100%), but both implementations were slower: stable bucketing measured 14.19/14.01 s warm versus the recent ordered-scatter pair 13.63/13.71 s. GPU expert time was about 9.0 s versus 8.65 s. Comparison-sort preprocessing was also slower and noisier (14.10/16.98 s); replacing the sort with linear counting improved host work but not end-to-end time. The opt-in kernel and metadata path were removed. Keep the ordered scatter.

Next kernel candidate: a gfx12 WMMA gate/up implementation for streamed IQ1_S experts. In the current pass the routed expert phase is about 8.65 s, and the earlier rocprof trace identified IQ1_S MMQ as the largest kernel family. Validate any prototype against the existing MMQ output and repeated 8K timings with the same reserve; leave MMQ as default/fallback until it demonstrates an end-to-end gain.

**Paired MMQ compact follow-up (2026-10-27).** Repeated safe-gated 8K passes with a 2 GiB VRAM reserve, 128 MiB staging slots, and the same prompt/model. The warm pass was 13.083 s with normal grouping and 12.763 s with `STRATA_DS41_MMQ_COMPACT=1`; repeating in reverse order gave 12.669 s normal and 12.648 s compact. The four timings average 12.876 s normal / 12.706 s compact (636 / 645 tok/s); this small difference is not enough to identify compact grouping as the cause. All four clear 600 tok/s in these paired warm passes. The fastest was 647.8 tok/s, just below 650 (12.60 s); 700 and 750 remain unverified. Normal and compact logits were byte-identical, and the two-position tiled check remained mean KL 0.0021, p99 0.002, top-1 100%. This is a two-pair in-process result, not a long-run distribution. Keep compact grouping opt-in until more repeats show a consistent gain.

**RDNA4 IQ1_S MMQ tile probes (2026-10-27).** On the same card, 8K prompt, and 2 GiB reserve, changing the large IQ1_S tile from 256x128 to 128x64 gave a single warmed pass of 14.93 s (549 tok/s), versus the recent 12.65-13.08 s baseline passes; logits were byte-identical. The resident gate/up microbench at M=128 improved from the earlier 0.150 ms to 0.123 ms, but the full path did not. Its other GPU phases and copy throughput were also slower in that pass, so clock/transfer variability limits the causal inference. Separate occupancy-only variants (occupancy 4 or 1, retaining 256x128) measured M=256 gate/up at 0.320 / 0.322 ms, versus the earlier 0.220 ms baseline; M=128 was unchanged around 0.151 ms. These were microbench screens, not full-prompt measurements. Restoring the original config rebuilt successfully; the restored baseline's next warm pass was 12.96 s (632 tok/s) with logits identical to the previous baseline. No MMQ config change is retained. The next substantial gain likely needs a new IQ1_S kernel path, since simple tile/occupancy tuning did not produce one.

**IQ1_S codebook-loader probe (2026-10-27).** In a temporary HIP build, replaced each MMQ loader's IQ1_S grid lookup plus shifts/masks and two scalar stores with one lookup into a pre-split 16 KiB pair table and one aligned vector store. The resident gate/up microbench was noisy but generally close to the original (M=64 0.074 ms, M=128 0.144 ms, M=256 0.210 ms). The safe-gated 8K warm pass was 13.47 s, slower than the restored baseline's 12.96 s; the GPU expert sum was 8.53 vs 8.14 s. Logits remained byte-identical, so this was a performance-only loss. Restored the original loader byte-for-byte and rebuilt HIP. The direct table doubled codebook traffic and did not improve the end-to-end path; a useful new expert kernel will need to avoid more than unpack instructions alone.

**IQ1_S stream-K probe (2026-10-27).** Enabled GGML stream-K in the non-fallback gfx12 IQ1_S configs to split each product's K reduction among more CTAs. The resident microbench was mixed: M=128 improved (~0.124 vs ~0.150 ms) while M=16/32/64 slowed; M=256 was flat. In the safe-gated 8K run the warm pass was 13.20 s (620 tok/s) versus 12.96 s (632 tok/s) baseline, and the expert kernel sum was 8.39 vs 8.14 s. Logits had mean KL 0.0002, p99 0.000, top-1 100%. Crucially, the MMQ scratch pool bypassed the decoder's reserve guard: sampled peak use was 16.65 GB of 17.10 GB, leaving only 0.45 GB free instead of the required 2 GiB. This stream-K configuration must not be used. Restored the original MMQ config byte-for-byte, rebuilt HIP, and confirmed idle VRAM use returned to 1.57 GB. A future stream-K attempt would first need reserve-aware scratch allocation and a faster full-pass result.

**Guarded DSpark expert cache (2026-10-27).** `STRATA_DS41_DSPARK_CACHE_EXPERTS=1` is an opt-in GPU cache (45 slots by default; `STRATA_DS41_DSPARK_CACHE_SLOTS` can lower it) for the streamed draft's MXFP4 experts. It requires unpinned batch staging and adds about 538 MiB of GPU allocation for this draft; its allocation uses the physical-VRAM reserve guard. On RX 9070 XT / Threadripper 1950X, with UD-IQ1_S, a 2 GiB target expert tier, 128 input and 32 generated code tokens, two cached runs measured 9.71/9.78 tok/s versus 8.80 without the cache; same-tier greedy measured 9.78. All output IDs matched. On a 256-token prefix, cache/no-cache/greedy single runs were 9.50/9.67/9.20 tok/s with matching IDs; that comparison did not establish a cache gain. Cache self-tests checked host copies and device samples across hits and misses. The smallest sampled free physical VRAM in these screens was 5.50 GB, with a 3 GiB watchdog floor; no new GPU fault or OOM was found in the subsequent kernel log. These short screens do not establish the 30 tok/s target or universal memory safety. Keep the cache opt-in and monitor both physical VRAM and host memory pressure.

**Low-weight CPU-route trim (2026-10-27).** `STRATA_DS41_CPU_TRIM` is an opt-in fraction from 0 to 0.25: on multi-token decode/verify passes, skip CPU-only expert picks from the third pick onward when their original normalized weight is below that fraction. Kept weights are not renormalized; GPU-held routes retain their weights. A zero setting was byte-identical to the fixed-tier unmodified logits over 128 positions. With the guarded draft cache and a 2 GiB target expert tier on RX 9070 XT / Threadripper 1950X, one 128-input/32-output screen measured 9.03, 10.52 and 11.90 tok/s at thresholds 0.08, 0.10 and 0.12; untrimmed cached runs were 9.71/9.78. Against fixed-tier teacher-forced logits for those same 128 input positions, top-1 agreement was 94.5%, 89.8% and 82.8%, respectively; mean KL was 0.0408, 0.1425 and 0.2772. Generated IDs matched the untrimmed run only for the 0.10 screen. This is a single short prompt, not a general quality evaluation, and the speed numbers do not establish 30 tok/s. Do not enable by default.

The 24-slot setting passed cache byte checks and matched the 45-slot output in short tests. On the same 128-input/32-output screen, it measured 9.29 tok/s with 46 hits in 182 accesses, versus 9.71/9.78 with 45 slots. Its reported draft GPU allocation was 1,148 MiB versus 1,524 MiB, saving about 377 MiB; the smallest sampled free physical VRAM was 5.90 GB. Capacity must be at least the draft block size times its top-k, and is checked before model allocation.

**Wider draft and fallback screens (2026-10-27).** `STRATA_DS41_DSPARK_EXTEND=1` allows up to seven draft positions with the existing eight-position verifier; cache capacity is checked against the wider draft batch before target allocation. The draft was trained with a five-position block, so the extension is opt-in. On RX 9070 XT / Threadripper 1950X with the guarded cache and a 3,203 MiB target tier, 256 input / 32 generated code tokens measured 10.51/10.54 tok/s versus 9.47/9.83 with the five-position window, with matching IDs and 10.03 same-tier greedy. On a wiki prefix, seven positions were slower: 4.31 versus 4.57, with matching IDs and low acceptance. `STRATA_DS41_DSPARK_AUTO=1` temporarily uses single-token target steps after low draft acceptance and periodically probes the draft again; on that wiki screen it measured 7.96 versus 11.94 greedy and matched the 32 IDs, while on the code screen it skipped no steps and measured 10.28. Minimum sampled free physical VRAM for the wider-window screens was above 4.27 GB under a 3 GiB watchdog floor. These short screens do not establish broad output equivalence or the 30 tok/s goal.

With seven-position drafting, the cache, auto fallback and `CPU_TRIM=0.08`, two 256-input/64-output code screens measured 13.82/13.84 tok/s, versus 11.67 untrimmed speculation and 10.36 same-tier greedy; all 64 IDs matched on this prompt. Fixed-tier 256-position quality checks gave trim 0.08 mean KL 0.0358, top-1 93.8%, perplexity 4.969 versus 4.920 on code, and KL 0.0587, top-1 95.3%, perplexity 2.757 versus 2.654 on wiki. These are two short samples, not a general quality claim. Free physical VRAM sampled above 4.27 GB with a 3 GiB watchdog floor, and no new GPU fault or OOM appeared after the runs. The 30 tok/s goal remains unverified.

### Guarded 8K prefill follow-up (2026-10-28)

The AMD `Dev` allocator now checks physical VRAM before every device allocation and refuses any allocation that would leave less than 2,560 MiB (2 GiB desktop reserve plus 512 MiB headroom). The gfx12 WMMA attention validates row geometry, cache-selection indices and buffer spans before loading; an opt-in device flag reports rejected rows after stream synchronization. `ds41_attention_multi` checks its full scratch capacity, and the optional two-stage top-k scratch now has overflow and VRAM-reserve checks.

`tools/ds41f_prefill_guard.sh` runs in a systemd scope with `MemoryHigh=64G`, `MemoryMax=72G`, swap disabled and `OOMPolicy=kill`. It refuses startup below 8 GiB free physical VRAM, 96 GiB `MemAvailable`, or 10% full memory PSI over 60 seconds. It stops below 2.5 GiB free VRAM or 32 GiB `MemAvailable`, near 56 GiB scope use, above 50% full PSI over 10 seconds, on a new AMD gfxhub fault or kernel OOM/reset, or after the run timeout. It disables direct expert pinning and bulk page population. The first guard version compared events by boot timestamp; the current cursor-based monitor is described below.

HIP and CUDA decoder builds passed. A single guarded 8K screen split into two 4K batches was stopped by the PSI guard at 51.72%; it reached 22.35 GiB peak scope memory, had at least 109.1 GiB host memory available and at least 4.63 GiB free VRAM. The timestamp-based kernel monitor saw no new AMD fault or OOM event. This is an interrupted run, not a throughput result. The prior paired warm result remains 645 tok/s by configuration average, with 647.8 tok/s the fastest individual pass; this guard update establishes no new prefill speed or quality result. The historical TCP permission fault remains unidentified.

The HIP decoder SHA-256 is `ca8b83fa711545255824700f1ed52725b779eb8c0f33afc9c6b0ca4dec072de2`; the CUDA decoder is `41b89ffa6f15f345c2af276cc7c20ce21e354d538691905d95929fcc16f07d58`. The guard script is `51c8111d55500c5ac24c4e354b2edb5892b0b95f77b0d4ab5dafb3b61e23efc1`; `ds41.cu` is `aacb60e9b392b3736e986dd9b25fc95f7c45afa57533189e8615342ebfb43457`, and `ds41_decode.cpp` is `2e6c012e1f120a9e56c8429de54cf5536de65d2a6795dd9bf6d86c04b82337d5`.

**Decode GPU-fault guard (2026-10-28).** The kernel journal retained repeated AMD `gfxhub` page faults attributed to `strata-glm-decode`; the latest was Oct 27 at 13:32 local time and reported `PERMISSION_FAULTS:0x3`. This is a GPU read permission fault; the log does not prove an out-of-bounds access. Two decoder segfaults were also recorded later that afternoon. The root cause is not established. The decode runner now refuses to start if the last 24 hours of kernel logs contain an AMD page/protection fault, reset, OOM or segfault, and it checks again after scope setup. The decoder waits behind a launch gate until systemd confirms `MemoryHigh=48G`, `MemoryMax=64G`, swap disabled, valid cgroup counters, and healthy live RAM/VRAM/PSI. It stops near 44 GiB scope use, watches cgroup OOM counters and new kernel faults, enables the existing gfx12 WMMA attention row/selection bounds check, keeps 2 GiB for the desktop plus 512 MiB physical headroom, and disables bulk page population and host expert pinning. The current log contains matching recent decoder faults, so no new decode run is permitted yet. The WMMA check covers its row/selection metadata and is not a general GPU memory sanitizer.

**Prefill OOM/OOB guard hardening (2026-10-28).** The guarded runner now uses a kernel-journal cursor and checks cgroup memory events from scope startup through completion. It bounds prompt size, context, batch, output count and expert tier, and refuses extra arguments that could override those limits. Engine checks now cap MMQ expert rows, validate streamed route and staging spans before dispatch, and apply the physical VRAM reserve to the expert arena. MMQ validates product geometry and scratch arithmetic; the prefill identity-table planner includes the documented 128-row MMQ read tail. HIP and CUDA decoder builds passed, and shell syntax and whitespace checks passed. Several attempts with an 8K batch were refused by the physical VRAM reserve before valid prefill. A bounds-checked 8K run in two 4K chunks stopped at 55.30% full memory PSI (50% limit), with 38.2 GiB peak scope memory, 109.7 GiB minimum `MemAvailable` and 3.44 GiB minimum free VRAM; the runner reported no new kernel fault or OOM. This is a safe interruption, not a speed or complete OOB-validation result. The prior qualified result remains 645 tok/s average and 647.8 tok/s fastest, with 650+ unverified. The historical AMD permission fault remains unexplained.

The prefill runner now scans 24 hours of kernel logs and refuses to start after an AMD page/protection fault, OOM, reset or decoder segfault. It starts a waiting shell in the cgroup and releases the decoder only after the memory limits, event counters, live RAM/VRAM and PSI checks pass; MMQ device-bounds validation now defaults on for this runner. The present log matches the quarantine, so no new prefill was launched. This validation is diagnostic overhead and does not establish a new throughput result. The WMMA and MMQ checks cover specific metadata and dispatch bounds; they are not a general GPU memory sanitizer.

### Decode cache follow-up (2026-10-28)

The streamed DSpark cache remains opt-in and defaults to 45 slots. Its bounded setting now allows up to 100 slots; cache metadata and the per-pass lock bitmap are sized dynamically, and the device allocation still goes through the physical-VRAM reserve check. Only 45 and 90 slots have been measured; 100 slots is a new, unmeasured capacity point. Mapped expert pinning remains rejected on AMD because that path has coincided with permission faults and reclaim pressure.

On RX 9070 XT / Threadripper 1950X, a warmed 600-token code prefix and 128 generated tokens, with the target expert tier fixed at 2,560 MiB, `DEFER_ALL=1`, `CPU_TOPK=2`, grouped GPU routing and five-position drafts measured 23.03 tok/s at 45 cache slots and 27.32 tok/s at 90 slots. Draft time fell from 1.594 s to 0.795 s; cache hits/misses changed from 347/234 to 483/98. The generated token IDs were identical. The 90-slot draft allocation was 2,268 MiB, about 807 MiB above the 45-slot allocation; minimum free physical VRAM was 2.81 GiB, above the 2.5 GiB desktop-plus-headroom floor. This is one matched pair, not a repeated speed result. The route configuration is highly lossy and the generated text repeats token cycles; a prior fixed-tier check for CPU top-two measured mean KL 0.826, top-1 71.5%, and perplexity 5.537 versus 2.791. Do not enable this route by default. The 30 tok/s target remains unmet and 40 tok/s was not reached.

A scheduled seven-position screen with the same cache and route settings was stopped before inference when free VRAM fell to 2.07 GiB. Lowering the requested target tier to 1,792 MiB let it complete at 23.46 tok/s for 128 outputs, with 3.23 GiB minimum free VRAM; the wider verify passes took longer and the target tier hit rate fell. A 64-output five-position screen at the 2,560 MiB tier was also stopped before generation when free VRAM briefly reached 2.49 GiB. At a 2,048 MiB target tier it completed at 19.40 tok/s and kept 2.72 GiB free. All guarded runs reported no new AMD page fault, kernel OOM or cgroup memory event. These screens show the reserve guard stopping safely when concurrent or scratch allocations reduce its margin; they do not qualify 30+ tok/s. HIP and CUDA decoder builds passed after the cache change, along with `bash -n` and `git diff --check`.

**Next decode screen (2026-10-28).** The best 90-slot run spent 3.84 s verifying and 0.80 s drafting for 128 output tokens; it had 98 cache misses in 581 accesses. The opt-in cache ceiling is expanded to 100 slots, adding an estimated 180 MiB from the measured 45-to-90 allocation slope; this point is not measured. The cache simulator now models 16/24/45/90/100 slots, including while the real cache is active, with zero device-memory cost; first compare simulated 90-slot hits with the real count. HIP and CUDA decoder targets compile. Do not run while the 24-hour kernel-fault quarantine finds the Oct 27 AMD faults and decoder segfaults. Once it clears, screen `STRATA_DS41_DSPARK_SCHED=1` at 90 slots, use the simulated 100-slot hit rate to decide whether to allocate the additional staging buffer, then measure the selected candidate under the same prompt and 2.5 GiB free-VRAM guard. The 30 tok/s target remains open.

**Prefill guard tightening (2026-10-28).** The guarded prefill runner now parses bounded decimal settings without Bash octal or overflow behavior, rejects negative prompt IDs, and refuses an explicit request to disable MMQ validation. Its systemd scope is capped at 56 GiB `MemoryHigh`, 64 GiB `MemoryMax`, and no swap; the live watchdog stops at 48 GiB scope use, 32 GiB `MemAvailable`, 2.5 GiB free physical VRAM, or 20% memory PSI full avg10. Prefill entry checks the token pointer, count, vocabulary range and context span before dispatch. The slow MMQ diagnostic checks product geometry, monotonic row bounds, active row IDs against an explicit destination capacity, and caps diagnostic metadata at one million rows; prefill, DS41 decode, worker and parity call sites provide that capacity. The MMQ identity table retains its 128-entry readable tail. The guarded wrapper always enables MMQ bounds validation after rejecting an explicit opt-out. HIP and CUDA builds passed for the decoder, worker and MMQ parity/benchmark callers. Shell syntax and tracked diff whitespace checks passed. The local guard probe refused before launch because the user systemd manager is unavailable; the independent 24-hour kernel scan still matches the Oct 27 AMD permission faults and decoder segfaults. No GPU inference was run. The checks cover these dispatch and metadata bounds, not every GPU memory access; the faulting kernel remains unknown. The previous prefill result remains 645 tok/s average, with 647.8 tok/s fastest; no new 650+ measurement is qualified.

**Decode guard and screen audit (2026-10-28).** Rechecked the 600-token code-prefix decode logs. The unpinned five-position draft with 90 cache slots, a 2,560 MiB target tier, `DEFER_ALL=1` and `CPU_TOPK=2` remains the fastest 128-output result at 27.32 tok/s: 103/122 draft proposals accepted, 0.795 s drafting, 3.837 s verification, 483 cache hits / 98 misses, and 2.81 GiB minimum free physical VRAM. An 80-slot top-two screen measured 25.05 tok/s with NUMA interleave and 24.74 without it; a 75-slot top-one screen measured 21.78 tok/s with 65.1% acceptance. Seven-position drafting measured 23.46 tok/s at a smaller target tier. Retain 90 slots and five positions for the next control; compare the still-unmeasured 100-slot cache with the same prompt, cache simulation and adaptive schedule under the 2.5 GiB free-VRAM floor.

A 64-output batch with 6.72 GiB of mapped draft experts measured 29.13 tok/s, but its 128-output repeat measured 20.04 tok/s. AMD mapped expert pinning is rejected, so this is not a safe or repeatable candidate. The decode runner now bounds decimal arguments and prompt size, count, token representation and context span; it rejects extra decoder options that could override those bounds and forces MMQ metadata validation. The recent 24-hour kernel scan still matches AMD faults, and the user systemd manager is unavailable. No new decode run was launched. 30 tok/s remains unmet and 40 tok/s remains unverified.

**Decode and prefill OOM/OOB launch gates (2026-10-28).** Both runners now validate prompt IDs against the GGUF tokenizer count and confirm the input embedding table has the same vocabulary width, using a bounded metadata scan before starting the GPU process. The prelaunch physical-VRAM minimum is the larger of 8 GiB and the requested expert tier plus the 2.5 GiB reserve and 1 GiB startup margin. Before release, each scope must confirm no swap, `memory.oom.group=1`, and zero `high`, `max`, `oom`, `oom_kill` and `oom_group_kill` counters; `MemoryHigh` is 48 GiB for decode and 56 GiB for prefill, both with 64 GiB `MemoryMax`. Runtime monitoring includes group OOM events and additional kernel OOM/allocation-failure patterns. The MMQ row-ID and gfx12 WMMA metadata checks cover their known dispatch paths; this is not a general GPU memory sanitizer. The Oct 27 page-fault cluster and decoder segfaults remain inside the 24-hour quarantine. Separately, the required user systemd manager is unavailable. No GPU run was launched. Bash syntax, Python AST parsing and whitespace checks passed.

**Decode speed follow-up (2026-10-28).** The best current lossy 128-output screen remains 27.32 tok/s; its profile reports 3.84 s of target verification and 0.80 s of drafting. The guarded runner now accepts strict 0/1 opt-ins for `FUSE`, `SHARE_Q8`, `IQ1S_FAST` and `Q6_V2`, which it previously dropped when constructing the clean decoder environment. Earlier same-machine exact decode screens measured `FUSE+SHARE_Q8` at +6–8% and `IQ1S_FAST` at +5–7% with unchanged token IDs, but those numbers do not qualify these options on the current lossy configuration. Next compare the opt-ins and `STRATA_DS41_DSPARK_SCHED=1` on the same 600-token/128-output prompt, then run the 90/100-slot cache simulator as an unscored GPU diagnostic. The simulator uses no additional VRAM but consumes CPU time during real drafting, so keep it out of throughput A/Bs. Retain the 2.5 GiB free-VRAM floor. The Oct 27 fault/segfault events remain within 24 hours and the user systemd manager is unavailable, so no GPU run was launched. 30+ tok/s remains unachieved; 40+ tok/s remains a stretch.

**WMMA bounds fail-fast (2026-10-28).** Guarded DS41 runs now copy and check the gfx12 WMMA attention bounds-error flag after each batched attention layer, before later layers can consume output from a rejected batch. This is a targeted metadata check; it cannot detect every GPU out-of-bounds access or establish the cause of the Oct 27 page faults. No GPU run is permitted while those faults and decoder segfaults remain in the 24-hour quarantine or the user systemd manager is unavailable.

HIP `strata-glm-decode` compiled and linked after this change. The CUDA build was stopped for the planned hardware reboot at step 187/192; a dry run shows five decoder compile/link steps remain. Static shell syntax, Python AST and whitespace checks passed. This is compile validation only; no GPU inference or runtime OOB check was performed.

**Cross-boot fault quarantine (2026-10-28).** After reboot, the current-boot kernel log was clean but the retained previous boot still contained the Oct 27 page faults and decoder segfaults. Both launchers now use `journalctl --boot=all` for their 24-hour preflight scan so a reboot cannot hide a recent GPU fault. The all-boots scan confirms the latest event is the decoder segfault at Oct 27 14:31:49 JST, so the quarantine remains active until after Oct 28 14:31:49 JST if no newer fault appears. In this shell, `systemctl --user` returns `Operation not permitted`; that means the user bus cannot be accessed here, not that the manager is confirmed stopped. CUDA and HIP `strata-glm-decode` targets now both compile and link. Shell syntax, Python AST parsing and whitespace checks pass. No GPU inference was run; 27.32 tok/s remains the best recorded decode result and the 30 tok/s goal remains unmet.

**Guarded scope access (2026-10-28).** A read-only `systemctl --user show-environment` query succeeded with the approved elevated shell, confirming the user manager is available when the sandbox permits its bus connection. The normal workspace shell still receives `Operation not permitted`; future guarded launch commands must use the approved execution context. At 05:43 JST, host memory showed 115 GiB available, memory PSI was 0%, and free physical VRAM was about 14.9 GiB. These are point-in-time readings, not authorization to run: the 24-hour all-boots fault scan still matches the Oct 27 events. No inference was run.

**Scope OOM journal quarantine (2026-10-28).** A wider journal scan found a DS41 decode systemd scope ending with `Failed with result 'oom-kill'` at Oct 27 19:35:48 JST, later than the retained AMD page faults and decoder segfaults. The 24-hour quarantine therefore extends until after 19:35:48 JST Oct 28, provided a fresh scan finds no later fault or OOM. Both decode and prefill guards now check all-boots kernel records plus the full system journal for DS41 scope OOM and `systemd-oomd` kills. Preflight uses `journalctl --grep` so it returns only matching OOM messages. The runtime cursor covers both and advances after every poll so it reads only new entries. Shell syntax and whitespace checks passed, and the filter matched the retained OOM event; no inference was launched.

**Decode route-sync timing (2026-10-28).** With `STRATA_DS41_DECODE_TIMING=1`, the decoder now reports host wall time spent waiting on `route_ev_` in milliseconds per speculative verification pass. This isolates the synchronous GPU-route ID handoff identified in the source review; timing unset leaves output unchanged. HIP and CUDA decoder targets compile. The metric has not been measured on the RX 9070 XT because the recent scope OOM is still inside the 24-hour quarantine.

**Decode host-buffer reuse and post-reboot checkpoint (2026-10-28 06:20 JST).** The serial MoE path reuses route-ID/weight, CPU-mask, and grouped-GPU scratch across layers and verification passes. In GPU-route mode it skips filling grouped-GPU arrays because those resident products already ran into `g_sum_`; CPU route masks and tier counters remain active. Routing math and bounds checks are unchanged. HIP and CUDA `strata-glm-decode` builds pass; runtime impact is unmeasured. After the 05:32 JST reboot, the all-boots 24-hour scan at 06:17 still finds the Oct 27 AMD page faults, decoder segfaults, and the later scope `oom-kill` at 19:35:48 JST. No newer matching event is present. Inference remains gated until after 19:35:48 JST, followed by a fresh clear scan. At 06:17, `MemAvailable` was 115 GiB, memory PSI full avg10 0%, and free physical VRAM 14.99 GiB; these are transient. The best result remains 27.32 tok/s. After the gate clears, use the guarded 90-slot control, test scheduling and kernel switches one at a time, then collect the separate `route_sync` profile. Keep at least 2.5 GiB physically free; 30 tok/s remains unmet and 40 tok/s remains a stretch.

**Draft-head optimization candidate (2026-10-28).** The CPU DSpark Markov-head path copies a full vocabulary row to host and synchronizes once per drafted position. The existing opt-in `STRATA_DS41_DSPARK_DEVICE_HEAD=1` keeps Markov projection and argmax on GPU and transfers only draft token IDs (plus confidence values when scheduling is enabled); the guarded runner already passes it through, and the added Markov table uses normal checked device allocation. After the 90-slot matched control, compare this option alone on the same prompt/output count, including token IDs/quality, draft-head time and total throughput. There is no result yet on this lossy configuration, so leave it opt-in unless the guarded A/B is faster and acceptable.

**Decode gate recheck (2026-10-28 06:22 JST).** The all-boots 24-hour scan still finds the Oct 27 AMD faults, decoder segfaults and scope OOM; no newer match is present. Live point-in-time readings were 114.7 GiB `MemAvailable`, PSI full avg10 0%, and about 15.86 GiB free physical VRAM. `systemctl --user show-environment` succeeds in the approved elevated context after reboot; the normal sandbox cannot access the user bus. The quarantine remains the only launch blocker at this check. No inference was run.

**Device draft-head OOB guard (2026-10-28 06:29 JST; capacity and non-finite checks rebuilt 06:32).** The opt-in GPU Markov head checks the token index against the validated vocabulary before row pointer arithmetic. Invalid IDs produce a zero row and set a device error flag. The argmax kernel flags any non-finite logit; if no usable index remains (such as an all-NaN or all-`-inf` row), it writes token 0 so later draft positions stay in range. After stream synchronization the host rejects a flagged draft and range-checks copied token IDs. The Markov-table and one-int flag capacities are checked by pre-run scratch validation; allocations use the checked device allocator. HIP and CUDA decoder targets build; whitespace and shell syntax checks pass. This protects this specific draft-head path and does not establish general kernel memory safety. The 06:32 all-boots scan still finds the Oct 27 scope OOM, so no GPU run was launched; after 19:35:48 JST, repeat the scan and keep the 2.5 GiB free-VRAM floor.

**Device-head prefix reuse (2026-10-28 06:35 JST).** Engram prefetch in the GPU draft-head path now uses DSpark's bounded `known` token-prefix vector rather than constructing and copying a temporary vector for each accepted draft token. It supplies the same tokens and prefetch range; output capacity is reserved for B tokens and prefix capacity for B+1. HIP and CUDA decoder targets build; no runtime speed result exists yet. The 06:35 all-boots scan still finds the Oct 27 scope OOM, so leave inference gated until after 19:35:48 JST and a fresh clean scan. Keep the 2.5 GiB free-VRAM floor.

## 8l. Two nodes, whole-expert ownership over IB (STD2, 2026-10-29)

The row split of 8c is not bitwise the single node's (split summation order) and cannot run beside the canonical
tier (it needs `fl_ == ff`). STD2 (`include/strata/net/expert_tp.hpp`, opt-in) gives the B550 whole experts instead:
- `STRATA_DS41_REMOTE_OWN=ADDR:PORT STRATA_DS41_REMOTE_OWN_MIB=N` (decoder): the most-used experts per byte of the
  profile (`STRATA_DS41_REMOTE_OWN_PROFILE`, else `STRATA_DS41_CANON_PROFILE`) after the canonical tier's, up to N MiB,
  are sent at setup (pread, 40 GiB in 73-97 s over IB) and dropped from TR16's page cache. Per layer, the owned routes
  and the activations go out right after the router; the reply is every owned route's raw output, merged with the
  canonical tier's in the lane's route order (`finish_with_external`). Owned experts are excluded from canonical
  streaming. Prompt passes go to the worker too (CPU lane there).
- `strata-ds41-worker` (CUDA, built on the B550 because of its glibc): STD2 keeps the owned blobs (anonymous,
  pinned, hugetlb or `--weights-dir=/dev/shm` by fingerprint) and runs passes of up to 8 tokens' tier-held routes with
  the canonical kernels (`ds41_canon_dot` / `ds41_canon_swiglu`, CUDA build), the rest on its expert lane
  (`ExpertLane::run_raw`). Its tier adapts (scores, promotions, a refill after prompt passes); placement never changes
  a value. `STRATA_DS41_WORKER_TIER_SLOTS` caps it, `--gpu-reserve-mib` leaves VRAM to others.
- B550's glibc 2.39 `expf` matches TR16's 2.43 on all 2^32 inputs (both differ from the port in the same two).

Verified with the miners running on the B550 (no timing): c_osd 1,000 teacher-forced positions byte-identical to the
all-CPU exact reference with 2 GiB (CPU only), 4 GiB (200-slot GPU tier, hit 0.78) and 40 GiB owned (5,687 experts,
722-slot tier on the 5060 Ti beside a 5.7 GB miner, B550 memory PSI 0); 128-token DSpark decodes with 8 and 40 GiB
owned generate the same IDs as the single-node exact greedy run. Launcher: `experiment/ds41f_tr16_decode/two_node.sh`
(worker in a `systemd-run --user --scope` with MemoryMax / no swap; `MODE=own`).

Deferred until the B550 is idle: timing of owned-set size and tier, hugetlb rows to ~59 GiB, TR16 page-cache relief.

**Measured (2026-10-29, B550 miners stopped; DSpark best.env, 128 tokens, all IDs equal to exact greedy):**

| | c_osd (4.57 tok/step) | c_llama (3.28 tok/step) | TR16 wait on the worker per call |
|---|---|---|---|
| TR16 alone | 14.02 | 7.62 | - |
| + B550 owns 8 GiB (1,144 experts) | 14.32 | 10.39 | 3.9-4.5 ms |
| + 24 GiB (3,415) | 14.81 | 11.06 | 8.1 ms |
| + 40 GiB (5,688) | 15.42 | 12.41 | 11.9 ms |

Worker: 15 lane threads, 1,620-slot tier (13.8 GiB) on the 5060 Ti; transfer 8/24/40 GiB in 12/43/79 s (kept in
/dev/shm for the second prompt). The worker's overall tier hit was only 0.10-0.17 (prompt passes, run on its CPU lane,
are counted too) and TR16 waits ~12 ms per call at 40 GiB: the B550 side is now the bound. Next: fill the worker tier
from the profile at setup, run its prompt passes on the GPU, and split the owned set by measured per-node throughput.

**Worker improvements (2026-10-29, B550 idle; all IDs equal to exact greedy; c_osd 1,000-position capture with 40 GiB
owned byte-identical):**
- Tier pre-fill: STD2 setup carries the owned experts' profile counts (`TpSetup.pad = 1`); the worker fills its GPU
  tier from them before `TpReady`. Worker tier hit 0.10-0.13 -> 0.62-0.89.
- Prompt chunks (up to 64 tokens) on the worker GPU: tier-held routes with the canonical kernels, misses on its lane.
  Prompt 17-20 -> 26-30 tok/s at 40 GiB owned.
- `STRATA_DS41_REMOTE_OWN_LOAD=f` (owned set by load share instead of the hottest band): slower at every f
  (c_osd 12.0/12.6/12.8, c_llama 9.2/9.6/9.7 for f = 0.3/0.45/0.6): with its tier the B550 is the faster node per
  byte, so it should own the hot band (the default, load share 0.94 at 40 GiB).
- `STRATA_DS41_WORKER_GPU_MIN_NT=9` (decode passes on the worker CPU only): slower (12.6 / 9.2 vs 15.1 / 11.3).
- Best two-node decode stays ~15 (c_osd) / 11-12 (c_llama) tok/s at 40 GiB band (single node 14.0 / 7.6). 48 GiB in
  /dev/shm does not fit its 52 GB tmpfs with a 4 GiB reserve (needs --rows=anon/hugetlb).

**Larger shares and the decode split (2026-10-29, B550 idle; all IDs equal to exact greedy).** Rows in the worker's
anonymous memory (`--rows=anon --reserve-gib=3`, worker scope MemoryMax=58G; /dev/shm caps at 48 GiB):

| B550 owns | TR16 canonical streaming | c_osd | c_llama |
|---|---|---|---|
| 40 GiB | 4 experts/layer | 14.94 | 10.87 |
| 48 GiB | 4 | 15.97 | 11.84 |
| 52 GiB | 4 | 15.86 | 12.25 |
| 52 GiB | 2 | 16.60 | 12.59 |
| 52 GiB | 0 | **17.61** | **14.52** (prompt 36.5 tok/s) |

Decode-only split of a 6-wide verify pass (c_osd, 48 GiB, streaming 4, ~222 ms): per layer TR16's host spent ~1.8 ms
issuing the streamed experts' pageable copies and ~1.2 ms waiting for its GPU chain (~120 ms a pass, the largest
part), ~1.3 ms waiting for the worker (whose decode request takes 2.9 ms: 2.0 ms CPU lane, 57-59 % of its routes on its
GPU) and ~0.26 ms merging. With the B550 taking the hot band TR16's lane has room, so streaming no longer pays; at
streaming 0 the worker reply wait rises to 1.9-2.2 ms per layer and the B550 becomes the bound. Two-node best:
`STRATA_DS41_REMOTE_OWN_MIB=53248 STRATA_DS41_CANON_STREAM=0` with experiment/ds41f_tr16_decode/best.env.

**Two-node optimization loop (2026-10-29, `experiment/ds41f_tr16_decode/trial.sh`, c_osd 128 tokens, all IDs exact).**
Kept (exact):
- Worker per-layer tier pools sized to each layer's blob (1,619 -> 2,003 slots in 14.2 GiB), STD1 buffers not
  allocated in STD2 mode: 18.2 -> 19.0 tok/s.
- Incremental tier refill (residents still wanted keep their slots): the after-prompt refill had re-uploaded 14.2 GiB
  (~1 s) inside the first decode request: 19.0 -> 22.8 tok/s (best single run; repeats 20.2-22.8 as the session went on).
- Canonical dot kernel: 4-way integer dots (`STRATA_DP4A`; IQ1_S as 8 sum g x - 8 sum x + delta sum x, IQ2_XXS signs
  folded per byte), activations staged in LDS, independent loads first: 149 -> ~70 us per call (single-token passes).
- `STRATA_DS41_REMOTE_OWN_SKIP_MIB` (owned band independent of the canonical tier's final size),
  `STRATA_DS41_MULTI_ROWS`, `ExpertLane::finish_with_external2` (one copy per source), TR16 14 lane threads.
Measured and dropped: worker `--stream=1..6` (its DMA + kernel slower than its lane), load-share ownership, 16-40 GiB
owned (TR16's lane becomes the bound), DSpark N=4 / N=7, device head + GPU argmax, 75-90 draft-cache slots (canonical
tier shrinks), multi-entry and row-staged and thread-per-block canonical kernels (all slower than v9), dense GEMV 4 rows.
Lossy routing fails the C++ agreement gate even at tiny settings (c_osd: window affinity 0.01 95.4 %, route affinity
0.005 95.9 %) and gave no speed (stacked affinity + trim: 21.6 tok/s).
Where a 6-wide verify pass goes (~160-180 ms): trunk GPU ~45 ms; per layer the FFN waits ~1.5 ms for the worker
(whose CPU lane at DDR4-2133 and GPU chain are both ~1.4-1.9 ms), ~0.55 ms for TR16's canonical chain, ~0.4 ms merge;
drafting ~38 ms a step. 30 tok/s needs ~150 ms a step.

**Kernel work (2026-10-29).** `experiment/ds41f_tr16_decode/kbench` times the canonical dot against a candidate,
bitwise, on verify-like groups (10 groups x 2 entries; IQ1_S and IQ2_XXS; HIP and CUDA builds). The RTX 5060 Ti ran the
single-entry kernel ~2.5x slower than the RX 9070 XT (1.40 vs 0.56 ms per layer-sized call pair). The multi-entry
kernel (`ds41_canon_dot_multi_kernel<8, 2, 4>`: weights decoded once for 2 entries, each wave walks 4 rows per staged
activation) is 30-40 % faster on the 5060 Ti in every multi-entry case and bitwise equal; default on CUDA
(`STRATA_DS41_CANON_MULTI`, off by default on HIP where it is mixed). Two-node c_osd: 22.78 / 22.98 tok/s (exact), the
worker's GPU now finishing before its CPU lane (0.14 ms wait). Worker `--stream=1/2` stays slower (20.8 / 20.6): a
7-9 MB expert's DMA (~0.6 ms) costs more than it saves. Thread-per-block and row-staged variants were slower.

**Remaining exact levers (2026-10-29).** DSpark expert cache 75 slots (the canonical tier shrinks to ~340 experts,
verification does not slow): draft 1.07 -> 0.66 s per 128 tokens, c_osd 24.8 / 24.7 tok/s; 90 slots equal, 100 worse.
Worker 16 lane threads: 24.9 (24 threads: 22.8). B550 owns 44 GiB: worse. Current two-node best configuration:
`experiment/ds41f_tr16_decode/best2n.env` + `trial.sh` defaults (TR16 14 lane threads, worker `--threads=16
--weights-dir=/dev/shm --reserve-gib=3 --gpu-reserve-mib=1024`, B550 owns 48 GiB, skip 3,328 MiB, draft cache 75):
best 24.9 tok/s, repeats 21.3-24.9 exact (spread from desktop load on TR16 and other sessions' builds on the B550;
trial.sh now waits for quiet nodes). 30 tok/s was not reached: per 6-wide verify pass ~155-170 ms (trunk ~45 ms GPU,
per layer ~1.4 ms waiting for the worker whose CPU lane at ~27 GB/s is the bound, ~0.5 ms TR16 GPU, ~0.4 ms merge),
plus ~24 ms drafting per step.
