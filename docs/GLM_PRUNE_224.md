# GLM-5.3-Flash pruned to 224 of 288 routed experts (TR16, 2026-10-11)

**For output quality, prefer full Q23 with all 288 experts when RAM permits.**
The 120.9 GB full model retains the experts removed by the 95.5 GB REAP-224 file.
Both are available in the [Hugging Face weight repository](https://huggingface.co/lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts).
Use unchanged routing and strict MTP acceptance for quality priority; the
[full-model guide](GLM_RELEASE_Q2Q3.md) records the quality/speed choices.

**Current qualification, 2026-10-13:** the tuned REAP-224 release candidate completed
HumanEval with 150/164 passes, below the 152-pass floor. It is not release-qualified.
The 152/164 result below describes an earlier build/profile. See the
[completed validation report](GLM_RELEASE_20261013.md) for the latest results.

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
least 152/164). Neither target was reached. The profile that passed the gate in this historical run was
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

## Decode-35 experiments (2026-10-13)

These sections preserve the measurements from the optimization work. During release preparation the
unsuccessful coarse schedule, new merge modes, route-first ordering, fused KDA verification, uneven splits,
device draft state, selective draft extension and MMQ tile hint were removed. Their flags below describe
the frozen experimental binaries, not the release candidate. The original patch and new files are retained
locally under `build-release-glm-20261013-prep/`. Replay capture, readback tracing and test coverage remain.
Current release validation is recorded in [GLM_RELEASE_20261013.md](GLM_RELEASE_20261013.md).

No candidate passed the performance gates; the tuned profile above stays unchanged. A fresh baseline on TR16
(1950X, 128 GB DDR4-3200, RTX 5060 Ti 16 GB at 150 W, PCIe 3.0 x8) measured **29.87 tok/s at 2K and 31.55 at 8K**.
Each median uses three fresh processes, each with one 512-token warmup and one timed continuation. All runs
passed the quiet/zero-swap guard, and the timed token hashes matched across processes at each width.

The coarse scheduler replayed the first 1,024 CPU service calls from separate 2K and 8K training continuations.
Every expert output and routed sum matched the phased reference bitwise. Six timed passes per arm, with legacy
controls before and after, gave these packed-weight rates (GB/s):

| Schedule | 2K capture | 8K capture |
|---|---:|---:|
| Legacy, pooled control median | 69.27 | 68.99 |
| Coarse, automatic spans | 66.27 | 65.84 |
| Coarse, 1 span | 60.21 | 58.46 |
| Coarse, 2 spans | 64.07 | 64.82 |
| Coarse, 4 spans | 68.61 | 68.57 |
| Coarse, 8 spans | 69.83 | 69.73 |

The best gains, 0.81% and 1.08%, miss the 5% replay gate, so no live coarse-scheduler profile was promoted.
The causal window-policy byte screen saved 0.20% at 2K and increased bytes by 1.00% at 8K, including initial
uploads; it also misses its 5% gate. Existing tier options were screened separately (one fresh process per
cell, 512-token warmup plus 512 timed tokens, training prompts; these are screening results, not qualification):

| Tier policy | 2K tok/s | 8K tok/s |
|---|---:|---:|
| Control | 31.66 | 29.66 |
| Recent prompt routes | 30.16 | 31.92 |
| Learn before affinity | 31.37 | 30.12 |

Step tracing put hidden/logit readbacks, including waits, allocation and logit checks, at 0.43 ms/token (2K)
and 0.58 (8K). An 8K CUDA trace estimated 0.49 ms/token of potentially exposed synchronous D2H activity after
excluding overlapping GPU kernel/graph execution. This still includes device transfer time and draft
readbacks whose results are immediately needed; it does not demonstrate the required 0.5 ms/token of removable
stall. The asynchronous-readback prototype was therefore deferred. The new window-scoring policy was also
deferred. No new corpus, prefill-path or HumanEval qualification is claimed for these rejected experiments.

Builds: CUDA 13.2.86/GCC 13.3 and HIP gfx1201/Clang 23 decoder and replay benchmark; oneAPI 2026.1 shared CPU
library and flow test. CUDA host tests, the HIP build's shared CPU tests, the SYCL build's 564-case bitwise flow
test and the 5,952-case Q2 lookup test passed. A shuffle-immediate portability fix was needed for oneAPI.
AMD/Intel GPU execution and a full SYCL GPU-engine build were not tested.
Compact measurements and hashes are in [the evidence record](fixtures/glm_tr16_decode35_20261013/summary.json);
local raw records and frozen binaries are under `build-tr16-decode35-20261013/`. Interrupted runs are excluded.

The frozen decode-35 experiments kept the REAP-224 weights, acceptance settings and quality gate above. Their coarse CPU
schedule was opt-in: `STRATA_NATIVE_FLOW_SCHEDULE=coarse`; unset or `legacy` keeps the existing schedule.
`STRATA_NATIVE_FLOW_PARTS=1|2|4|8` fixes the spans per expert and NUMA side. Without it, the pool chooses enough
spans to occupy the workers. Gate/up spans keep fused hidden quantization on 256-row boundaries. Workers prefer
their local side and previous expert, and claim down rows only after that expert's gate/up work is ready.
Small groups without enough parallel work and unsupported layouts use the existing path. Arithmetic, routed
reduction order and completion callbacks are unchanged. `CPU_FLOW_COARSE` reports how many layers used it.

`STRATA_GLM_EXPERT_REPLAY=FILE` captures the first 1,024 CPU service calls, including activations, selected IDs,
original route weights and the CPU/resident mask. It preserves repeated positions after speculative rollback;
there are no weight payloads. Capture is diagnostic only and is rejected by the throughput runner. Replay uses
the same model and compares every expert output and routed sum bitwise against the phased CPU implementation:

```bash
STRATA_Q23_AVX2=1 \
  build-glm-release/strata-glm-q2-kernel-bench MODEL --replay CAPTURE 6 15
```

Keep the capture's model/config hashes with it. Replay measures the CPU service alone; it cannot predict
end-to-end decode speed. The removed scheduler was screened with automatic and fixed span counts, requiring
at least 5% higher packed GB/s before live qualification. `STEP_TRACE` also reports hidden-state readback time; that time includes
any GPU wait and is only an upper bound on removable readback overhead.

`tools/glm_tier_window_replay.py MODEL ROUTING_EVENTS --output RESULT` ranks experts using a training prefix
and scores a later suffix. It compares individual route counts with one count per expert per service group,
charging unique CPU reads and initial replacement uploads. Its 5% byte-reduction gate is a fixed-route screen,
not a latency or quality prediction. With affinity enabled, changing residency can change routes.

`tools/glm_tr16_decode35.py` compares saved configs at both widths, one warmup and one timed continuation per
fresh process. It reverses config order on alternating repetitions, requires zero swap and a quiet machine,
records token hashes, and checks that the decoder, prompts and model identity stay fixed. For example:

```bash
python tools/glm_tr16_decode35.py --config CONTROL --config CANDIDATE \
  --prompt-2k PROMPT_2K --prompt-8k PROMPT_8K --output FRESH_DIRECTORY --trials 3
```

The speed gate requires a median of at least 35 tok/s at both widths. It does not qualify quality: a winning
config still needs an independent repeat, the corpus and prefill-path checks, and all 164 HumanEval tasks on
the same build/config. Failed or interrupted runs do not count.

## Architectural continuation (2026-10-13, quality checks deferred)

This historical iteration targeted 35 tok/s decode and 700 tok/s at an 8,192-token prefill, with a two-hour
work limit. The user requested implementation and speed screens before rerunning quality gates. The switches
described below were subsequently removed; the tuned profile stayed unchanged. Raw records and frozen builds are in
`build-tr16-arch-20261013/`.

- `STRATA_GLM_VERIFY_FUSE_KDA=1`: with split verification, join windows of at least four tokens at KDA MoE
  layers after the first MoE. Both token groups share projections and CPU expert reads, then separate again
  for the split layers. Single-sequence execution only; shared split projections retain their existing path.
- `STRATA_GLM_VERIFY_SPLIT_FIRST=1|2|3`: choose the first group's width for single-sequence verification
  windows of two to four tokens, clamped to leave at least one token in the second group. Zero keeps the
  balanced split. Wider lookup windows and multi-sequence batches retain their existing split.
- `STRATA_GLM_MOE_MERGE_GROUPS=ready`: merge only when the second publication is already available and the
  groups share a cold expert. `early-device` waits for both publications and lets the CPU finish group A
  early while the GPU adds its own resident expert rows. These modes require canonical single-sequence
  service without RAM routing, unbiased tier learning, route deferral or step tracing. Existing `1` and
  `early` modes retain their strict-combine requirement.
- `STRATA_GLM_MTP_DEVICE_HIDDEN=1`: retain the recursive draft hidden row on the GPU between proposals,
  avoiding its host round trip. This applies to single-sequence MTP; the first proposal still uses the
  verified target hidden row.
- `STRATA_GLM_MTP_EXTENSION_MARGIN=M`: with configured draft depth above three, keep the first three
  proposals and extend only through draft logits whose top-two gap is at least M. A low third-token gap
  prevents the next draft call; later low-confidence proposals are discarded before target verification.
  Zero disables the extension policy. It changes proposal windows and needs coding-quality checks with
  the configured acceptance policy.
- `STRATA_GLM_PREFILL_GROUP_BUDGET=N`: a **lossy** long-prefill experiment. Keep N whole 16-expert groups per
  layer, ranked by summed calibration-prior frequency. Discarded groups are excluded from routing and H2D
  prefetch. Zero disables it; 14 retains every group of REAP-224. It requires a prior, single-GPU GGUF and no
  RAM tier, applies from `STRATA_PREFILL_LOSSY_MIN_TOKENS` (4,096 by default), and excludes the draft layer.
  Decode can still route to all 224 experts, but its prompt state has changed.
- `STRATA_GLM_PREFILL_TOP_K=N`: a **lossy** product budget for long prefill on one GPU without a RAM tier.
  Use N routed experts per prompt token, up to the model's top-k; zero keeps the model value. It uses the
  same minimum prefill width as the group budget and excludes draft and decode execution. It can combine
  with the group budget to reduce both routed products and weight transfers.
- `STRATA_PREFILL_MMQ_COLUMNS=32|64|128`: an MMQ column-tile selection hint for grouped Q8 products with at
  least 1,024 routed rows. It changes the tile estimate while preserving the full launch bounds; the actual
  tile also depends on the backend. Zero keeps the original choice. Floating-point summation can differ.

The CUDA and HIP builds pass. CUDA MMQ tests cover the default choice and all three hints, including empty
experts, uneven long batches, row permutations and partial tiles. The fused KDA path passed bitwise graph,
mixed-width, retained-prefix and rollback checks against sequential logits; the early-device merge passed
the same checks, as did the 3+1 split. These are functional checks, not corpus, long-prefill or HumanEval
qualification. AMD GPU execution was not tested. The preceding shared-CPU changes were also built with
oneAPI; these decoder and MMQ changes apply to CUDA and HIP.

The speed screens use the same TR16 hardware and held-out prompts as the decode-35 experiment above. Each
cell is one fresh process with a 512-token warmup and 512 timed tokens; these are not repeated speed gates.
The controls used the tuned profile, with every new switch off. The frozen builds, config differences,
token hashes and guards are recorded in [the architecture evidence](fixtures/glm_tr16_arch_20261013/summary.json).

| Decode screen | 2K tok/s | 8K tok/s |
|---|---:|---:|
| Initial control | 30.10 | 31.61 |
| Final build, all new switches off | 30.06 | 31.80 |
| Merge only when B is ready | 30.49 | 31.65 |
| Unsplit verification | 26.26 | — |
| Early-device merge | 25.91 | — |
| Fused KDA verification | 25.82 | — |
| Device draft hidden | 30.08 | — |
| First group 1 / first group 3 | 28.86 / 28.91 | — |
| Selective extension, margin 2 | 31.07 | 29.66 |

The ready-only mode found no already-published second group in either screen. Early-device merging cut
timed CPU expert bytes from 778.2 to 726.3 GB, but waiting for both groups outweighed the saved reads. The
fused and uneven splits also lost overlap. Device hidden storage matched the control's tokens and speed
(30.08 versus 30.05 tok/s on the same build). The extension screen used draft depth 5 and maximum 5 soft
accepts per window, with the existing margin 4, top-k 6 and repeat guard 4. Its 8K regression rejects it as a
general decode improvement; its output changed and its coding quality is unqualified. No decode candidate
reached 35 tok/s at both widths.

With every new switch off, the final build matched both the 512-token warmup and the 512-token timed output
from the frozen baseline at each prompt width. This checks 2,048 emitted tokens across the two widths; it is
not a new corpus or HumanEval run.

Prefill screens use two 8,192-token passes per fresh process and 64-token decode continuations. The latter
are too short to qualify decode. Keeping 12 groups alone reached 643.98 tok/s; 11 groups reached 654.98,
reducing transferred expert weights from 86.49 to 67.96 GB per pass. Combining 12 groups with top-k 6 reached
679.27 tok/s; 11 groups with top-k 7 reached 678.11. Smaller MMQ column hints did not help: with 11 groups,
64 columns fell to 620.89 tok/s and 32 columns to 552.99.

Ten groups with top-k 6 crossed 700 in the first screen: 713.91 and 713.04 tok/s. It transfers 61.78 GB per
pass and retains 73.29–84.15% of the calibration-prior mass across the 42 MoE layers. This is a lossy speed
screen, not a qualified 700 tok/s profile. A fresh-process repeat with three prefills and 512-token decode
continuations measured **689.05 tok/s** median prefill (686.15–693.66), below 700. The original screen used
64-token continuations and two prefills; both protocols and results are retained. Three timed 512-token
continuations after one warmup measured **34.57 tok/s** median decode (34.30–34.85), using the altered prompt
state. That number does not establish a decode-kernel gain or a qualified decode profile.

The final nine-group/top-k-6 screen used three prefills in one fresh process, then one 512-token decode
warmup and three timed continuations. Prefill reached **718.39 tok/s median** (717.80–719.38), versus
611.84 for the final control's two passes, a 17.4% gain. Decode measured **33.95 tok/s median**
(33.30–34.94). Expert transfers fell to 55.60 GB per prefill; retained prior mass was 66.39–79.57% across
layers. All runs passed the quiet/zero-swap guard and the 512 MiB GPU-free reserve. This is the first
three-pass screen above 700 in this iteration, not independent-process or quality qualification. Longer
prompts were not measured. The tuned profile stayed unchanged. The frozen experimental candidate used
`STRATA_GLM_PREFILL_GROUP_BUDGET=9` and `STRATA_GLM_PREFILL_TOP_K=6`; both options were later removed after
the release rejection screen recorded in [GLM_RELEASE_20261013.md](GLM_RELEASE_20261013.md).

The two-hour iteration ended with the 35 tok/s decode target unmet. Corpus, long-prefill quality and
HumanEval gates were deferred at that point. Sixteen Python measurement-tool tests and the whitespace check passed.

The CPU replay kept 15 workers: four timed rounds gave 69.73 GB/s, versus 40.42 with 11 workers. Seven workers
were rejected because the captured NUMA service needs workers on both nodes. A separate activation-reuse
probe over 1,024 captured services per width found no avoided cold jobs at relative-RMS thresholds through
0.2, and at most 1.41% at 0.5. That job-count proxy did not justify implementing an approximate output cache.


## Staged expert admission (one-hour extension, 2026-10-13)

The user extended the architectural iteration by one hour, still deferring quality gates. The new opt-in
`STRATA_GLM_MOE_MERGE_GROUPS=staged` starts a prefix of group A's cold experts, then checks whether group B
has published its routes. If B is ready and shares a still-pending cold expert, the remaining jobs share
weight reads and publish each group's completion independently. Otherwise A finishes normally. Neither
readiness check waits for B. `STRATA_GLM_MERGE_FRONT_EXPERTS=0..4` selects the prefix size (default 1);
zero checks after A's route preparation without dispatching a prefix. At least one A job remains pending.

Completed rows retain their original route positions and weights. The prefix dispatch uses zero reduction
weights, and the final dispatch reduces all rows in the original canonical order. Each dispatch joins its
workers before the service thread changes the job list. The experiment requires single-sequence canonical
layer flow and excludes RAM/unbiased tier routing, partial deferral, remote TP, prepared/exact expert caches,
and observer/replay diagnostics. Both strict host reduction and device resident-row addition are supported.

CUDA and HIP builds pass. GPU checks used a fixed tier and a 256-token prefix. Device combine/front 1
and strict combine/front 4 passed verify
widths 1–4, mixed-width graph capture/replay, retained prefixes, history reallocation and rollback with
bitwise-equal logits. The device check merged 15 of 10,140 candidate pairs (B was ready after the prefix in
41 checks); the strict/front-4 check merged 486 pairs (B ready in 2,282 checks). Neither found B ready before
the prefix. These diagnostic counts are not throughput measurements.

The 2K staged screens used one 512-token warmup and one timed 512-token continuation per fresh process.
Front sizes 1, 2 and 4 measured 29.30, 29.48 and 29.64 tok/s, respectively, versus the preceding build's
30.06 control. Their timed token hashes matched the frozen baseline. The four-expert prefix reduced CPU
expert bytes from 778.21 to 770.44 GB (1.0%), while layer-flow time rose from 11.14 to 11.58 seconds. The
zero-expert prefix measured 29.08 tok/s with no admitted pairs and different generated tokens; it is not a
bitwise-output control, and the cause of its different sequence was not isolated. These results do not
support enabling staged merging in the tuned profile.

A second opt-in experiment, `STRATA_GLM_VERIFY_ROUTE_FIRST=1`, publishes both token groups' routes before
running either group's resident/shared experts. It retains separate buffers and resident executors, advances
mixer state in the same token order, and uses a separate graph key. Single-sequence split verification is
required; split projections and expert deferral are excluded. Its purpose is to make B's routes available
earlier without waiting on the CPU. Fused full-window KDA layers keep their existing execution order.
Route-first checks passed with both device combine/front 2 and strict combine/front 4, covering the same
graph replay, retained-prefix and rollback cases. They merged 99 and 756 pairs, respectively. All 22 GLM
CTest checks passed: 15 host/CPU checks in the sandbox, then the seven GPU checks with CUDA access after
the sandbox attempt reported no visible GPU. Sixteen Python measurement-tool tests also passed. HIP was
compile-tested only; shared CPU source is unchanged from the preceding SYCL validation.

The route-first/front-2 timing screen measured **29.27 tok/s at 2K**, with the same timed token hash as the
frozen baseline. B was ready after the prefix in 784 of 12,285 checks, and 488 pairs merged. CPU expert
bytes were 776.62 GB, only 0.2% below the preceding control. The screen passed the quiet/zero-swap/VRAM
guards but did not establish a speed gain. These new scheduling variants were screened at 2K only; the
8K default-output check does not measure them. A further route-first/early-device timing arm was skipped to
reserve the end of the authorized hour for final default-output checks.

The final default controls on the route-first-capable binary measured **29.89 tok/s at 2K** and
**31.55 tok/s at 8K**. Both 512-token warmups and both timed 512-token sequences matched the frozen baseline
(2,048 emitted tokens total). The final 8K control's two prefills measured 603.95 and 603.18 tok/s. An extra
nine-group/top-k-6 repeat was skipped because another model load would exceed the remaining window;
the preceding three-pass 718.39 tok/s result remains a speed screen with quality deferred.

The extension ended with the 35 tok/s decode target unmet. Both new scheduling features were opt-in and
were subsequently removed during release preparation. The tuned profile stayed unchanged. All completed
screens passed the quiet, zero-swap and 512 MiB GPU-free
guards. No new corpus or HumanEval gates were run, no commits were created and nothing was pushed.

Timing results and resource guards are recorded in
[the compact evidence](fixtures/glm_tr16_incremental_20261013/summary.json). The frozen binaries, source
patches, configs and raw logs are retained locally under `build-tr16-incremental-20261013/`.


## Reproduce

```
tools/glm_prune_pipeline.sh     # assemble Q22, calibrate, rank, prune Q23 and Q22, evaluate (about 3.5 hours)
```
