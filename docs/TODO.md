# GLM-5.3-Flash: open work (as of 2026-10-16)

The measured state these items start from is in [GLM53F_RELEASE.md](GLM53F_RELEASE.md); the two-GPU history is in
[GLM_TR16_RUNNER_PLAN.md](GLM_TR16_RUNNER_PLAN.md). Items are grouped by area, most important first within each.
"Gate" means the agreed quality gate: corpus `tr16-35` (perplexity within +6%, KL within +0.08, top-1 within 3 pp of
the unpruned Q23 model) plus HumanEval at or above 152/164, plus no loops at 8K.

## 1. Quality qualification

Tracked in [lighttransport/Strata#1](https://github.com/lighttransport/Strata/issues/1).

- **Find a two-GPU decode configuration that passes HumanEval.** The current one (affinity 0.14, soft acceptance
  6/8) passes the corpus gate but scored 146/164. Candidates in order: affinity 0.08 (passed the corpus gate:
  KL 0.197), then the strict acceptance of the single-GPU profile (margin 4, top-k 6). Each needs the corpus run
  (~10 min) and a full HumanEval (~2 h). Record the speed cost.
- **Requalify the single-GPU REAP-224 profile on the current binaries.** It scored 152/164 once and 150/164 on a
  rerun; with the score at the floor, run it twice more to know its variance before calling it qualified.
- **Code-generation spot checks for every demo and profile.** The RX 9070 XT demo answer failed the C++ oracle
  (rejects inputs longer than 20 characters despite "leading zeroes are allowed"); the two-GPU answer passed all
  400,532 cases. Make the oracle run part of recording (`docs/fixtures/glm53_cpp_quality/test_generated.cpp`).
- **HumanEval variance.** Single runs move by 2-3 tasks between identical settings (152 vs 150). Decide whether the
  floor applies to one run or to a median of runs.

## 2. Quantization: smaller weights at the same quality

Tracked in [lighttransport/Strata#2](https://github.com/lighttransport/Strata/issues/2).

Research note with sources and a ranked plan (GSQ refinement in the existing Q2_K/Q3_K format first): [GLM_QUANT_RESEARCH.md](GLM_QUANT_RESEARCH.md).

- **GSQ + RCO for the GLM experts.** ISTA-DASLab's GSQ-RCO compressed GGUFs are what upstream Strata ships for its
  Qwen model (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S sizes, [MODELS.md](MODELS.md)). Applying the same recipe to GLM-5.3-Flash's
  routed experts could give a smaller file at the Q23 quality, or better quality at the REAP-50 size. Work items:
  obtain or reproduce the quantizer for this architecture; calibrate on GLM (the coverage priors and the
  calibration corpus already exist); add CPU expert kernels for the resulting formats (the GLM CPU pool and the
  canonical Q2_K/Q3_K arithmetic do not cover IQ2/IQ3 codebooks today) and GPU kernels for the resident tier;
  then gate it against Q23.
- **Requantize the kept experts after REAP pruning.** REAP-224 and REAP-50 slice the Q23 file without
  requantization. A second quantization pass calibrated on the pruned model (`strata-glm-q2-pack` with the
  pruned routing statistics) may recover some of the 4.9% perplexity REAP-224 costs.
- **Mixed precision by expert importance.** The REAP saliency scores and the coverage priors rank experts; keep
  the hot ones at Q3/Q4 and push cold ones to Q2 or lower. The worker cache and the CPU kernels already handle
  mixed Q2_K/Q3_K tensors per projection; per-expert type mixing needs a pack format change.
- **EXL3 codebooks** ([GLM_EXL3.md](GLM_EXL3.md)): the native EXL3 path loads existing EXL3 models only. Measure its
  size/quality point against Q23 on the gate before investing in kernels.
- **8 GB-card model.** An RTX 3070 (8 GB) with 64 GB reaches 10-20 tok/s on REAP-50 with presets gated only on
  matched perplexity and a 30-task HumanEval subset. A smaller expert format is what would let 32 GB machines run
  at all (today 0.3 tok/s).

## 3. Multi-GPU optimization

Tracked in [lighttransport/Strata#3](https://github.com/lighttransport/Strata/issues/3).

Measured limits on TR16 (RTX 5060 Ti + RX 9070 XT), from the steady-state profile in plan section 15:

- **Dense weights read twice per verify step** (once per verify group): 23 ms of an 88 ms step at 8K. Sharing the
  read serializes the groups and lost 5 tok/s. Open options: a three-group or wider single-group verify that
  keeps the CPU lane busy another way, or moving part of the dense set to the second card (tensor-parallel
  projections across vendors, with a host-bounce reduction).
- **Tier promotions** (~12 x 9.1 MB uploads per step): capping in-flight promotions gave +3 tok/s at 8K. Next: score
  promotions by expected hits per byte, and use anonymous pinned copies of hot experts so CUDA can DMA directly
  (it refuses file-backed pages).
- **One process for both targets.** Decode is fastest with the CUDA card as primary, prefill with the HIP card as
  primary. Switch roles between prefill and decode (hand the KV/KDA state to the other card after prefill, ~100 MB
  at 8K) so one server gets 870 prefill and 40+ decode.
- **Prefill 900+.** 870-875 tok/s at 8K; the mixers are half the prefill time now (MLA ~200 ms, KDA ~75 ms per layer).
  Remaining ideas: overlap the worker's activation upload with routing, split the worker's partial return by token
  halves, run part of the MLA heads on the second card.
- **Worker request path**: ~0.27 ms per request; a shared-memory flag instead of the socket and one sync per request.
- **Layer pipelines for two-socket, same-vendor servers** (merged from zen5: NUMA-aware layer pipelines, dual-CUDA
  workers): timing and quality on the V100 server are pending.
- **More than two GPUs**: the expert-parallel worker is one process; generalize to N workers with an expert
  partition by measured speed per card (`tools/glm_mixed_fraction_calibrate.py` does this for two).

## 4. Single-GPU and CPU work

Tracked in [lighttransport/Strata#4](https://github.com/lighttransport/Strata/issues/4).

- **CPU expert lane**: whole-expert span access (`STRATA_NATIVE_FLOW_SCHEDULE=span`) forecast +1 tok/s, unmeasured.
- **MLA decode cache kernel**: small at 8K (about 2% of the step) but grows with context; measure at 32K before
  building.
- **Draft cost**: the output head is 0.85 ms per draft step (154,880 rows); a draft-vocabulary subset needs a
  frequency-ranked subset that covers >99% of generated tokens to avoid losing acceptance.

## 5. Vulkan runner

Tracked in [lighttransport/Strata#5](https://github.com/lighttransport/Strata/issues/5).

- Fast engine numerical gate (KL 0.20 against strict, limit 0.01); the Q8 emulation reduced it to 0.12.
- 2048-token prefill windows hit the amdgpu 2-second scheduler timeout on the RX 570; split the dense prefix into
  per-layer submissions.
- Complete a full HumanEval (the last run stopped at 107 of 119 completed tasks).
- Integrate with setup and the API server once a profile passes the gate.

## 6. Release and docs

- Push `glm53f` (nothing is pushed; the earlier audit found credentials in old community snapshots that must stay
  excluded, see [GLM_RELEASE_20261013.md](GLM_RELEASE_20261013.md)).
- Update the Hugging Face model card (`release/README.md`) with the REAP-224 rows and the two-GPU results.
- Move superseded GLM docs (listed as historical in the release notes) into `docs/history/`.
- The setup installer and the server do not yet offer the GLM two-GPU configuration; add it once one passes the gate.
