# MiMo-V2.6 Pro (1T) and Flash (309B) on TR16 and B550: plan and simulator study (2026-10-10)

Goal: MiMo-V2.6-Flash on one node at two sizes (about 120 GB for TR16's 128 GB,
about 63 GB for B550's 64 GB) and MiMo-V2.6-Pro across both nodes (192 GB).
Nothing here is measured on these machines: speeds are simulator predictions
(`tools/sim/glm_sim.py --model mimo_flash|mimo_pro`, TR16 calibration, about
±40% for a new architecture) and quality numbers are quoted from community
GGUF releases measured against the native MXFP4 weights.

## Architecture

From the `XiaomiMiMo/MiMo-V2.6-{Pro,Flash}-RL` configs (MIT license); the
vision and audio encoders (about 0.7B each) are not counted:

| | MiMo-V2.6-Flash | MiMo-V2.6-Pro |
|---|---|---|
| Model type | `mimo_v2` | `mimo_v2` |
| Layers | 48 (1 dense, 47 MoE), 3 MTP layers | 70 (1 dense, 69 MoE), no MTP in the config (separate MTP and DFlash drafter GGUFs exist) |
| Hidden / expert rows | 4096 / 2048 | 6144 / 2048 |
| Routed experts | 256, top-8, no shared expert | 384, top-8, no shared expert |
| Attention | GQA 64 q / 4 kv heads (8 on sliding layers), qk 192, v 128; 39 of 48 layers use a 128-token sliding window with attention-sink bias | GQA 128 q / 8 kv heads, qk 192, v 128; 60 of 70 layers sliding-window |
| Routed parameters | 302.8B (25.17M per expert) | 1,000B (37.75M per expert) |
| Total / active | 309B / 15B | 1.02T / 42B |
| Native expert format | MXFP4 (4.25 bits per weight) | MXFP4 |

Strata does not implement `mimo_v2`. It is the simplest of the three new
architectures: standard GQA attention (no MLA, no KDA, no mHC), sliding windows
that keep the KV cache tiny (10 full-attention layers in Pro, 9 in Flash),
sigmoid routing with `noaux_tc` bias. Work needed: a `mimo_v2` model path
(GQA with sink bias and a 128-token window, fused QKV layout, partial rotary
0.334), MTP for Flash (3 draft layers), and generalizing the expert pool, the
GPU tier, the pack tool (hidden 4096 already matches Flash's geometry for MiMo
Flash; 6144 for Pro) and the remote worker.

## Quality evidence (community releases)

[AesSedai](https://huggingface.co/AesSedai/MiMo-V2.6-Flash-MOPD-GGUF) measured
size-targeted quantizations (FFN quantized, trunk Q8_0/Q6_K) against the native
MXFP4 model (wikitext perplexity and KL):

| Model | Bits per weight | Size | Perplexity change | KL |
|---|---:|---:|---:|---:|
| Flash | 3.5 | 126 GiB | +2.0% | 0.134 |
| Flash | 3.0 | 108 GiB | +5.5% | 0.173 |
| Flash | 2.5 | 90 GiB | +10.5% | 0.227 |
| Flash | 2.0 | 72 GiB | +27.2% | 0.366 |
| Pro | 3.0 | 357 GiB | +9.8% | 0.189 |
| Pro | 2.5 | 298 GiB | +19.6% | 0.262 |
| Pro | 2.0 | 238 GiB | +43.7% | 0.414 |

A REAP-50 build of Flash (128 of 256 experts, HOPE second-order saliency) exists
as [patrickbdevaney/MiMo-V2.6-Flash-REAP50-GGUF](https://huggingface.co/patrickbdevaney/MiMo-V2.6-Flash-REAP50-GGUF)
(Q2_K 66.2 GB, MXFP4 92.4 GB) without published quality numbers. No REAP build
of Pro exists. For calibration of expectations, GLM-5.3's REAP-50 cost +76%
perplexity at 4-bit and REAP-25 +15% (`docs/GLM53_FULL_TR16_B550.md`).

## Sizes (simulator packs; fixed weights estimated at 5.5 bits)

| Pack | Experts | Routed GB | GB per token | Fixed weights (Q4 GPU copy) |
|---|---:|---:|---:|---:|
| `mimof_q23` | 256 | 109.6 | 3.43 | 2.9 GiB |
| `mimof_q22` | 256 | 99.4 | 3.10 | 2.9 GiB |
| `mimof_reap25_q23` | 192 | 82.2 | 3.43 | |
| `mimof_reap50_q23` | 128 | 54.8 | 3.43 | |
| `mimof_reap50_q22` | 128 | 49.7 | 3.10 | |
| `mimop_reap50_q22` | 192 | 164.1 | 6.84 | 11.0 GiB |
| `mimop_reap50_iq2` | 192 | 160.2 | 6.67 | |
| `mimop_reap62_q23` | 144 | 135.8 | 7.54 | |
| `mimop_reap62_q22` | 144 | 123.1 | 6.84 | |
| `mimop_q23` (full) | 384 | 362.0 | 7.54 | does not fit |

Q2_K/Q3_K experts are about 2.9 bits per weight, so the full Flash at Q23 is the
"3.0 BPW" class (+5.5% perplexity in the table above); Q22 is about 2.6 bits
(+8 to +10%).

## MiMo-V2.6-Flash, one node

| Node | Pack | Total weights | Decode (MTP1) | Prefill | Expected quality |
|---|---|---:|---:|---:|---|
| TR16, 128 GB | `mimof_q23` | about 114 GB | **20.0** tok/s | 415 at 8K (single pass, streaming) | ≈ +5.5% perplexity (3.0 bpw class) |
| TR16 | `mimof_q22` | about 104 GB | 21.9 | | ≈ +8–10% |
| B550, 64 GB | `mimof_reap50_q23` | about 59 GB | **9.0** | 217 at 1K | REAP-50 loss unknown (likely large) |
| B550 | `mimof_reap50_q22` | about 54 GB | 10.1 | 226 | larger |
| TR16 with the 63 GB pack | `mimof_reap50_q23` | 59 GB | 21.2 | | as above |

MTP depth 3 does not help in the model (19.2 vs 20.0) because the union of
experts over four tokens costs more bytes than the extra accepted drafts save;
acceptance for MiMo's MTP layers is unknown and uses the GLM-Flash profile.
The TR16 numbers are close to GLM-5.3-Flash's (3.43 versus 3.09 GB per token),
and the fixed weights are small (2.9 GiB at Q4), so the 16 GB GPU keeps a 6 GiB
resident expert tier. **MiMo-V2.6-Flash at Q2_K/Q3_K on TR16 is the most
promising new target: about the GLM-Flash speed with a stronger model, if the
+5.5% class loss holds.** On B550 the 63 GB class requires REAP-50; the
alternative of the full model at about 1.6 bits per weight would be worse than
the 2.0-bpw row (+27%).

## MiMo-V2.6-Pro, two nodes (TR16 + B550)

`python3 tools/sim/glm_sim.py --model mimo_pro multinode --node tr16 --node b550 --link ib --split both --params tools/sim/data/params_tr16_20261009.json --pack <pack> --dense-format q4 --speculation none --decode-cache-mib 2000`

| Pack | Fit (TR16 / B550 GiB) | Best split | Decode IB | Decode 1 GbE |
|---|---|---|---:|---:|
| `mimop_reap50_q22` (192 experts) | 114.6 / 38.2 | rows 0.25 | 8.1 | — |
| `mimop_reap50_iq2` | 97.0 / 52.2 | experts 0.35 (not implemented) | 6.2 (compute-bound IQ decode) | — |
| `mimop_reap62_q23` (144 experts) | 94.8 / 31.6 | rows 0.25 | 7.7 | 6.0 |
| `mimop_reap62_q22` | 86.0 / 28.7 | rows 0.25 | 8.1 | — |

Prefill (single 8K pass, streaming): 286 tokens/s predicted for REAP-62 Q23.
Pro's fixed weights are large (18.7B attention parameters, 11 GiB at Q4 on the
16 GB GPU, about 12 GB read per token), so the GPU tier is empty and the step is
partly GPU-bound (`cpu+gpu`). **Pro needs pruning to half or fewer of its
experts and Q2-class weights to fit 192 GB, and still predicts only about
8 tokens/s; at that pruning depth its quality advantage over a well-quantized
MiMo-Flash on one node is doubtful.** No speculative decoding was assumed for Pro
(the official config has no MTP; ggml-org ships separate MTP and DFlash drafter
files, which would add 30–60% if they port).

## Recommendation and plan

1. Implement `mimo_v2` in Strata against MiMo-V2.6-Flash first (hidden 4096,
   2048-row experts, the same expert-pool geometry as GLM-Flash; GQA with sliding
   windows instead of MLA/KDA). Bring-up on TR16 with parity against
   `transformers` on short prompts.
2. Calibrate and convert MiMo-Flash experts from the native MXFP4 to Q2_K/Q3_K
   with the generalized `strata-glm-q2-pack` (about 110 GB); measure decode and
   prefill on held-out 2K/8K prompts, corpus perplexity/KL against MXFP4, and
   HumanEval. Target: 20 tokens/s at ≤ +6% perplexity.
3. For B550: convert the existing REAP-50 HOPE checkpoint (or prune ourselves
   with the calibration counts) to Q2_K/Q3_K, measure quality before investing
   in tuning.
4. MiMo-Pro only after 1–3 and only if a REAP of Pro with acceptable quality
   appears; the plan otherwise mirrors GLM-5.3 (row split 75/25 over IB,
   `docs/GLM53_FULL_TR16_B550.md`).

## Simulator changes

`tools/sim/model.py` registers `mimo_flash` and `mimo_pro` (`MODELS`,
`_register`) with config-derived geometry, estimated fixed-weight tables
(GQA attention size, KV cache only on full-attention layers) and pack families
(full, REAP-25/50 for Flash; full, REAP-25/50/62 for Pro; Q23/Q22/IQ2/Q4).
Select with `glm_sim.py --model mimo_flash …`. Kernel parameters, tier hit curves
and MTP acceptance are the GLM-Flash TR16 calibration; KL estimates printed by
`plan` are not meaningful for these models. Tests: `tools/test_sim_multinode.py`
(`MimoModelTest`).
