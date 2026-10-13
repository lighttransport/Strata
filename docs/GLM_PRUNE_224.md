# GLM-5.3-Flash pruned to 224 of 288 routed experts (TR16, 2026-10-11)

Goal: a smaller model file that keeps decode and prefill speed. Q2_K is the smallest expert format that decodes at
memory speed on the 1950X (67 GB/s; IQ formats 40 GB/s), so going below the Q22 pack means removing experts, not
bits. The REAP-50 base from Hugging Face (144 experts) cost +29.8% perplexity; this prunes far less, with our own
calibration.

## Method

1. **Saliency calibration.** The usual Q2 calibration run (`STRATA_GLM_CALIBRATION_DIR`, 32,768 code and prose
   targets from `tools/glm_q2_corpus.py`, UD-Q2_K_XL base, 57 minutes) now also records the router gate weight of
   every routed token (`gates.json`, `include/strata/artifact/expert_calibration.hpp`), next to the existing
   per-expert input moments and visit counts.
2. **Ranking** (`tools/glm_reap_rank.py`). REAP (arXiv 2510.13999) scores an expert by the mean over its routed
   tokens of gate weight times output norm. The output norm is estimated from the calibration moments:
   E‖W_down h‖² ≈ Σ_k ‖W_down[:,k]‖² E[h_k²] (diagonal approximation, column norms dequantized from the model),
   saliency = E[g] · sqrt(E‖W_down h‖²). The 64 lowest of each layer are dropped (224 is the nearest multiple of 16
   the loader accepts). The MTP layer (blk.45) is not calibrated; its plan drops the 64 experts with the highest
   router (load-balancing) bias, which can only change draft acceptance, not output.
3. **Pruning** (`strata-glm-prune-experts`, `tools/glm_prune_experts.cpp`) slices every routed tensor with an expert
   dimension (gate/up/down experts, `ffn_gate_inp`, `exp_probs_b`) of an assembled file, rewrites
   `glm5next.expert_count`, recomputes the expert-pack checksums and records `strata.prune.source_experts` and
   `strata.prune.plan_hash`; the result is reopened through the loader. No requantization.

Dropped experts carried 16.0% of routed traffic per layer on average (10.3-22.2%) but only 6.3% of traffic times
saliency (at most 13.4%).

## Results

Held-out: 64 sequences, 32,768 targets (`build-q2-redesign/corpus/evaluation.ids`), teacher-forced against the
original UD-Q2_K_XL logits (perplexity 8.57 on this set).

| Model | File | Perplexity | KL | Top-1 |
|---|---:|---:|---:|---:|
| **Q23, 224 experts** | **95.5 GB** | **8.966 (+4.6% vs original)** | **0.191** | **82.1%** |
| Q22, 224 experts | 88.1 GB | 9.049 (+5.6%) | 0.216 | 80.8% |
| REAP-50 Q23, 144 experts (third-party prune) | 63.2 GB | 11.12 (+29.8%) | 0.587 | 69.8% |

The unpruned Q23 run (120.9 GB) was stopped by the run guard's memory limit after 58 of the 64 sequences (peak
RSS 116.9 GB on this 125 GB host). On those same 58 sequences, the cost of pruning is:

| Model (first 58 sequences) | Perplexity | KL | Top-1 |
|---|---:|---:|---:|
| Q23, 288 experts | 8.678 | 0.159 | 83.6% |
| Q23, 224 experts | 8.804 (+1.5%) | 0.191 (+0.033) | 82.3% (−1.3 pp) |
| Q22, 224 experts | 8.885 (+2.4%) | 0.217 (+0.058) | 81.1% (−2.4 pp) |

Speed on a quiet machine (2026-10-12, TR16; qualified profile: release decoder, affinity 0.08, MTP depth 1,
512 tokens x 3, held-out code prompts; staged prefetch off in every arm because it currently requires exactly
288 experts). The run guard's quiet-machine rejection was on, except where noted:

| Model | Decode, 8K (tok/s) | Prefill 8K, single pass (tok/s) | Decode, 2K (tok/s) | Prefill 2K (tok/s) |
|---|---|---:|---|---:|
| Q23, 288 experts | 22.4 / 22.0 / 21.8 (see below) | 295 | 22.0 / 22.9 / 20.8 | 107 |
| Q23, 224 experts | 21.8 / 22.8 / 23.0 | 341 (+16%) | 21.9 / 22.2 / 21.1 | 131 (+22%) |
| Q22, 224 experts | 23.0 / 23.2 / 23.0 | 351 (+19%) | 22.2 / 22.6 / 23.6 | 141 (+31%) |

Pruning leaves decode where it was (all three within their trial spread, 21-23 tok/s) and speeds up prefill by
16-31%, about what 21% fewer expert bytes streamed over PCIe per chunk predicts. The simulator (`tools/sim`,
packs `reap224_q23` / `reap224_q22`) predicted no decode change (−2%, within its error): top-8 routing reads the
same bytes per token.

The 288-expert file could not be timed with the guard on. In four attempts its run was stopped at the start of
decode, either by kernel memory compaction (`kcompactd` at 98% of a core, triggered by the decoder's own
huge-page arenas with 108 GB resident) or by 46-68 MB of process swap. Its 8K decode above was therefore taken
with interference recorded and a 512 MB swap allowance: compaction ran at 264-297 s and 228 MB went to swap
(prefill measured before that). Its 2K run happened to see neither. The pruned files leave 25-33 GB more memory
free and were not affected. An earlier screen of the same three files while another agent's builds were running
measured 13-20 tok/s decode, in the same order.

## Tuning the 224-expert Q23 file (2026-10-12)

Target: 35 tok/s decode and 600 tok/s prefill on TR16 within the 35-track gate (vs unchanged 288-expert Q23
on the 16-sequence corpus: perplexity at most +6%, KL at most +0.08, top-1 drop at most 3 pp; HumanEval at
least 152/164). Neither target was reached. The best profile that passes the gate is
`configs/experimental/glm53f-q23-reap224-tr16-tuned.json`:

| | 2K held-out | 8K held-out |
|---|---:|---:|
| Decode (tok/s, 512 tokens x 3, medians) | 29.0 | 31.6 |
| Prefill (tok/s, single pass) | 168 | 496 |
| Before tuning (qualified profile, prefetch off) | 21.9 / 131 | 22.8 / 341 |

Quality: corpus perplexity +4.90%, KL +0.0636, top-1 −1.90 pp (routing and Q4 dense; soft acceptance does not
change teacher-forced metrics); HumanEval 152/164 at the floor (unchanged Q23: 154). There is no quality margin
left for further lossy steps.

What each step did (TR16, quiet machine except where an upload was running and recorded):

| Step | Kind | Effect |
|---|---|---|
| Staged prefetch for pruned files (`STRATA_GLM_REAP_PREFETCH=1`, now any count below 288; 14 groups) | lossless | 8K prefill 340 → 494, 2K 131 → 168 |
| Streaming prefetch (`STRATA_GLM_PREFETCH_STREAM=1`, ported) | lossless | none here: all 14 groups already fit the pool |
| MTP depth 3 + soft acceptance (margin 4, top-k 4, 3 per window) + repeat guard 4 (ported) | lossy, HumanEval only | 2K decode 22.3 → 25.9 (depth 3 exact alone: 17.6) |
| Affinity 0.10 + Q4 dense | lossy | 25.9 → 27.3 |
| `STRATA_GLM_MTP_SYNC_STATE_ONLY=1` (ported) | lossless | 27.3 → 29.2 (resync 1.0 s → 0.19 s per 512 tokens) |
| `STRATA_GLM_MTP_DRAFT_TOP_K=4` (new: CPU draft layer runs its 4 highest-weight routes) | lossless (drafts only) | +2–3% |
| Accept top-k 6 + cost-aware lookup | lossy, HumanEval only | 8K 29.9 → 31.6 |

Tried without gain: ordered KDA + fused mHC save, KDA chunk 256/1024, row parts 8, column tiles (prefill within
±2%); fused mHC/MLA kernels/head blocks, fused routing, norm-Q8/lookup-once (decode within noise); MTP depth 4
(−14%); a larger GPU budget (tier +90 MiB, no gain); `CUBLAS_COMPUTE_32F_FAST_16F` for the FP16 projections
(no change; cuBLAS keeps FP32 accumulation). Deferral does not apply to MTP verify windows. Higher affinity
buys little on this file: 0.12 and 0.14 (both passing the corpus gate, the latter at its edge) decoded 28.8 and
29.0 at 8K without Q4 dense, and the mix2 profile failed the gate (perplexity +6.99%).

Why the targets were not reached:
- Decode is bound by host memory bandwidth. The CPU reads about 1.56 GB of non-resident expert weights per
  token at about 69 GB/s (55% of routes already hit the 6.75 GB GPU tier); an earlier step trace of this
  profile family put 81 of 94.5 ms per verify round in the CPU expert wait. 35 tok/s needs about 1.30 GB per
  token, which the routing levers reach only beyond the quality budget.
  The Q22 file (Q2_K down projections, 9% fewer expert bytes) with the same profile reached 33.1 (2K) / 32.3
  (8K) tok/s at affinity 0.08 and passed the corpus gate at its edge (perplexity +5.33%, KL +0.0779, top-1
  −2.61 pp; affinity 0.10 failed on KL), but scored 139/164 on HumanEval, 13 below the floor, so it is rejected.
  The CPU side is saturated for this access pattern: 16 threads instead of 15 measured 31.6 tok/s at 8K
  (expert pass 67.7 GB/s, control 31.6 at 68.7 GB/s) and NUMA-local expert placement 30.0.
  Streaming cold experts to the GPU during decode does not help either: with a concurrent 7.2 GB/s pinned
  host-to-device copy (a stand-in for that traffic), the same 8K decode fell from 31.6 to about 22 tok/s; the
  CPU expert pass only lost 8% (70.5 to 64.8 GB/s), the rest is the decoder's own PCIe traffic queuing behind it.
- Prefill is now bound by GPU compute, not PCIe (88.5 GB of experts stream in 12.3 s; an 8K pass takes 16.5 s).
  An Nsight Systems profile of the 8K pass: expert MMQ 5.2 s, FP16 dense projections 3.4 s (about 77% of the
  FP32-accumulate tensor peak), MLA attention about 2.6 s, KDA 1.3 s. 600 tok/s needs about 3 s less GPU time,
  which means new kernels (FP8 or FP16-accumulate projections, fused attention, faster Q2_K MMQ), not settings.
  Measured follow-ups: FP16-accumulate cuBLAS for the FP16 projections (`CUBLAS_COMPUTE_16F`, widened in place)
  raised 8K prefill from 485 to 521 tok/s in the same session, but the greedy continuation after the 8K prompt
  diverged at token 13 and the corpus evaluation does not exercise batch prefill, so it was not kept. llama.cpp's
  MMQ on this card runs Q2_K at 30.8 TFLOPS, Q3_K 37.1, Q8_0 50.7 (4096 x 512 x 14336): transcoding expert
  groups to Q8_0 on the GPU would save about 2 s of matmul but cost about 1 s of writes, so together with the
  FP16 projections the estimate stays near 550-570 tok/s.


## Kernel work, 2026-10-13 (plan in `docs/GLM_PRUNE_224_PLAN.md`)

Profile `configs/experimental/glm53f-q23-reap224-tr16-tuned.json`, TR16, 512 tokens x 3 (prefill: 64 tokens x 3):

| | Prefill | Decode |
|---|---:|---:|
| 8K held-out prompt | 615 tok/s | 32.3 tok/s |
| 16K prompt (8K held-out + evaluation text, two 8K passes) | 604 tok/s | 28.9 tok/s |
| 2K held-out prompt | 169 tok/s | 30.6 tok/s |
| Before this work (8K / 2K) | 496 / 168 | 31.6 / 29.0 |

Quality: unchanged from the gated profile for prompts under 4,096 tokens (the 2K output is bit-identical to the
profile that scored HumanEval 152/164 and passed the corpus gate). From 4,096 tokens the lossy prefill paths below
apply; their prefill-path check (`--eval-prefill`, corpus through batch prefill, against the original logits)
measured KL 0.1726 vs 0.1711 unchanged (+0.0015), perplexity 4.934 vs 4.927. Applied to every prompt they cost
HumanEval 6 passes (146/164), which is why they are gated by width (`STRATA_PREFILL_LOSSY_MIN_TOKENS`).

Prefill, 8K, in order (each opt-in): FP16 accumulation for the dense projections (`STRATA_PREFILL_F16_ACCUM`,
485 -> 521); Q2_K/Q3_K expert groups transcoded to Q8_0 on the GPU for MMQ's faster path
(`STRATA_PREFILL_Q8_TRANSCODE`, from 4,096 tokens; matmul 5.2 -> 3.5 s, transcode 1.0 s); a vectorized FP16 MLA
gather (lossless, +2.5%); streaming prefetch (+2.8% now); FP16 tensor absorb products (`STRATA_GLM_MLA_ABSORB_F16`,
+3%); softmax writing FP16 directly (lossless); warp routing and a single-pass KDA output norm (lossless, +1.9%);
FP16 scores/values chain (`STRATA_GLM_MLA_F16_OUT`, +1.5%); KDA column tiles (+1.7%); coalesced dense dequant
(`STRATA_GLM_DEQUANT_COALESCED`, lossless); one FP16 input cast per share region (lossless); FP16 indexer scoring
(`STRATA_GLM_MLA_INDEX_F16`, +1%). An 8K pass is now within about 10% of the PCIe streaming limit (12.3 s of
expert transfer). A single 16K pass does not fit the 16 GB card beside the dense weights.

Decode: the CPU expert pass is the limit at about 70 GB/s (NUMA-local rows 72.5 GB/s in
`strata-glm-q2-kernel-bench`, one core alone 10.9 GB/s, so memory-, not compute-bound; a plain read reaches
84-88 GB/s). Added and measured: the custom AVX2 Q2_K/Q3_K kernel (`STRATA_Q23_AVX2`, +3% at 8K); a bit-exact
Q2_K table-lookup kernel (`STRATA_Q2K_LUT`, no gain: not multiply-bound); window expert sharing within verify groups
(`STRATA_GLM_ROUTE_WINDOW_MARGIN`; 0.10: 33.7 / 33.8 tok/s at 8K / 2K but corpus KL +0.098, fails); route min-share
pruning (no gain); draft without routed experts (`STRATA_GLM_MTP_DRAFT_TOP_K=0`, acceptance collapses, 25 tok/s);
continuation margins 1-3 (slower); owned NUMA arenas (no change vs mapped). DIMMs already run at their rated
3200 MT/s. Under the 35-track gate decode stays at 30.6 (2K) / 32.3 (8K); window sharing, the one lever above 33.5 at
2K, costs more KL than the gate allows. Also measured: Q3 mixers and a 7.2 GB tier (resident share stays near 54%, 30.0-31.1
tok/s), and a runtime 192-expert prune of this file (router bias -inf for the 32 lowest-saliency experts per layer of
a REAP-192 plan; resident share 59%, decode 30.7 / 30.3 tok/s, no gain).

## Reproduce

```
tools/glm_prune_pipeline.sh     # assemble Q22, calibrate, rank, prune Q23 and Q22, evaluate (about 3.5 hours)
```
