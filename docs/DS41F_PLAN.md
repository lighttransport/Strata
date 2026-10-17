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
| **A64FX 12 nodes** | **native MXFP4 (lossless), 24.6 GB per node** | FP8 as released | 24.6 measured today (gemm); **~70 / 90** with the GLM-plan redesign | 33 measured; ~590 redesign | no quantization decision at all; the runner's synchronization is the whole gap (see docs/GLM_A64FX_PLAN.md) |
| **A64FX 8 nodes** | 2.25 bpw (19 GB) or IQ3_XXS (26 GB) per node; MXFP4 (36 GB) does not fit | FP8 | ~24 today / **~78** redesign | ~410 redesign | 8 = 2x2x2 torus, 3 tree steps per allreduce |
| **A64FX 6 nodes** | 2.25 bpw (25.5 GB per node, 2 GB of margin); IQ1_M or Tetra for room | FP8 | ~22 today / **~69** redesign | ~320 redesign | the only way onto 6 nodes is a 2-bit expert pack |

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

Pack building needs the 289 GB MXFP4 source once; it can run on the A64FX cluster against the staged shards (the tool is plain
C++20 and cross-compiles) or on the Xeon box after a 510 GB download. Packs are 102-153 GB each; build two at a time.

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
gate) inside the Strata A64FX port. 6/8 nodes: SVE Q2_K (ggml has it: 97 GB/s = 296 GMAC/s per node, measured 2026-10-18,
`tools/sim/data/measured_a64fx_kernels.json`; ggml's IQ2_XXS/IQ1_S manage only 11/8 GB/s per node, which is why the `current`
A64FX rows for IQ packs are slow) and an SVE IQ2_XXS/IQ1_S kernel with the same
integer scheme (`svtbl` over the `g+1` grid, `svdot_s32`); Engram as gemm does (owned rows per rank, prefetch thread,
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
