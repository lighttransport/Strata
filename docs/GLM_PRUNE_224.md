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

## Reproduce

```
tools/glm_prune_pipeline.sh     # assemble Q22, calibrate, rank, prune Q23 and Q22, evaluate (about 3.5 hours)
```
