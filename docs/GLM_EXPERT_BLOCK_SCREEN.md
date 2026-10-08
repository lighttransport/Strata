# GLM expert-block quality screen

The tested selection rule failed the quality gate. Its mixed layer profile raised
code perplexity by 4.04% and prose by 10.02% on 8,192 tuning targets, against the
additional 1% allowance. Implementation stopped before the tiled weight format and
predictor training, following the redesign's stopping rule. The 35 tokens/s target
has not been reached. [Measured results](glm53_flash_expert_block_screen_measurement.json)
record the source identity, corpus hashes, profile, controls and memory samples.

This is the first gate for the single-stream redesign: select 16-channel blocks
inside the original routed experts before implementing a new weight layout or
trained predictor. The diagnostic computes every gate/up row, ranks blocks by
`sum(hidden[channel]^2 * down_column_squared_norm[channel])`, zeros omitted
channels, then runs ordinary hidden quantization and the full down projection.
It does **not** save those weight reads and must not be used as a speed benchmark.
The ranking omits channel covariance and downstream amplification; it is an
optimistic test of this selection rule, not a mathematical accuracy bound.

The full-width engine remains the default. The new CLI flags require ordinary
`--eval-corpus`, canonical Q2_K/Q3_K experts, zero routing affinity, no gate or
route skipping, and no expert cache, MTP, remote TP or GPU expert execution.
Original-format exception layers retain all channels.

## Measurements on October 7, 2026 (UTC)

Threadripper 1950X, 16 physical cores, two NUMA nodes, 128 GB DDR4 and RTX 5060 Ti
16 GB. The screen uses 15 workers plus the host, context 8192, four teacher-forced rows per
call, Q8 dense storage, the canonical expert pack and no routed-expert GPU tier.
NUMA placement is lazy; these results establish no throughput rate.

The smoke sample has 256 code and 256 prose targets from the tuning partition of
the frozen engineering corpus. It is not the 32K held-out corpus or HumanEval.
The unbiased Q23 baseline scored perplexity 2.98494 on code and 10.69069 on prose.
The all-block control matched all 79,298,560 reference logits bit for bit.

| Retained blocks per eligible expert | Retained channels | Code perplexity change | Prose perplexity change |
| ---: | ---: | ---: | ---: |
| 32 / 128 | 25% | +34.22% | +77.44% |
| 48 / 128 | 37.5% | +33.85% | +39.82% |
| 64 / 128 | 50% | +16.86% | +26.18% |
| 80 / 128 | 62.5% | +11.26% | +17.04% |
| 96 / 128 | 75% | +2.53% | +13.00% |
| 112 / 128 | 87.5% | +2.28% | +3.50% |

Every screened uniform pruning level exceeds the additional 1% perplexity
allowance in both categories. Baseline, control and sweep completed with zero
process swap. The subsequent sweep completed 78 cases: each of 39 eligible layers
individually retained 32 or 64 blocks. The allocator selected a mixed profile for
17 layers; other layers remained full width.

| Mixed-profile sample | Code perplexity change | Prose perplexity change | Ideal union-byte reduction |
| --- | ---: | ---: | ---: |
| 512 targets, used for layer selection | -4.62% | +0.78% | 23.27% |
| 8,192 tuning targets | +4.04% | +10.02% | 23.20% |

The small sample's improvement did not generalize. The byte reductions are
estimates from the selected block unions, not physical reads eliminated by this
diagnostic. Full gate/up and down execution still ran. The 8K full-mask control
matched all 1,268,776,960 logits bit for bit. Its baseline, control and candidate
had zero swap and at least 5,450 MiB sampled GPU free memory. No throughput is
qualified by these runs. The held-out 32K corpus, predictor fitting, HumanEval
and tiled-kernel work were not run after the tuning gate failed. This rejects the
tested heuristic/profile, not every possible method of partial expert execution.

Raw records are under `build-q2-v6/screen/`: the baseline, full-mask control,
uniform sweep, per-layer sweep, source/corpus manifests and memory guards.
The diagnostic decoder was built in isolation with rebuilt CPU code and unchanged
CUDA archives; `build-q2-v6/build-manifest.json` records their hashes and commands.

## Reproduction

Build `strata-glm-decode`, `strata-glm-block-norms`, and
`expert_block_screen_test` with native expert support and the existing GLM CUDA
settings. Use the Python environment that supplies Strata's server dependencies.
Only one GPU experiment should run at a time.

```sh
build-exl3-venv/bin/python tools/glm_block_screen.py prepare MODEL Q23_PACK WORK \
  --norms-tool build-glm/strata-glm-block-norms
build-exl3-venv/bin/python tools/glm_block_screen.py run WORK --split smoke \
  --decoder build-glm/strata-glm-decode
build-exl3-venv/bin/python tools/glm_block_screen.py layers WORK --budgets 32 64 \
  --decoder build-glm/strata-glm-decode
```

`WORK` must be new. Preparation verifies the frozen corpus hashes and separates
24,576 fitting targets, 8,192 tuning targets and 32,768 evaluation targets. Fitting
and tuning are disjoint sequences; they can contain material from the same source
file. The existing calibration/evaluation split is disjoint by source content.
Code and prose remain equally represented. Norm generation is CPU-only and
streams one projection at a time; the norm file includes geometry, source identity
and a payload checksum. The identity uses the model's existing header/file-generation
fingerprint, not a cryptographic hash of every weight.

`--split tune` and `--split evaluation` run the larger corpora. They save baseline
logits, require bitwise full-mask parity, report category-specific perplexity,
KL and top-token agreement, and sweep the six pruning levels. They remain
diagnostic; HumanEval, coding oracles and clean delivered-token measurements are
separate release gates. New GPU runs also enforce a sampled 512 MiB free-memory
reserve through `glm_q2_run_guard.py`.

The layer allocator charges nonnegative code/prose NLL changes separately and
uses 80% of the 1% perplexity budget. Its additive estimates cannot capture layer
interactions. It writes `adaptive-oracle.json`, which requires a combined model
evaluation before any promotion.

To evaluate its combined profile against a previously saved baseline:

```sh
build-exl3-venv/bin/python tools/glm_block_screen.py profile WORK \
  WORK/adaptive-oracle.json --split smoke --decoder build-glm/strata-glm-decode
```

The tuning measurement above used the same baseline/control commands with
`tune.ids`, then `profile --split tune`. It did not repeat the six uniform
pruning levels on the larger corpus.

## Calibration capture and checks

After a profile passes the feasibility gate, `tools/glm_block_screen.py capture
WORK` records the full-width fitting run. Per-layer binary streams contain
sequence/position IDs, original expert IDs and routing weights, normalized expert
inputs, and pre-mask contribution energies for all 128 blocks. Energies are
unweighted by routing for predictor fitting. Route weights are retained separately.
The collector computes hidden activations transiently and saves block energies,
avoiding a second complete set of hidden vectors on disk. `read_capture()` streams
the records and rejects invalid shapes, values and truncation.
The 512-target capture-control matched every baseline logit. Streaming validation
matched all 172,032 energy records to their input routes across 42 layers.

`expert_block_screen_test` covers full-mask identity, deterministic ties, invalid
inputs, norm corruption, canonical widths 1–4 and the CPU pool transform boundary.
`tools/test_glm_block_screen.py` covers corpus partitioning, separate category
gates, incomplete results, sensitivity allocation, capture decoding and GPU probes.
The existing native pool, dataflow, Q23 and canonical arithmetic checks remain
applicable. Default execution does not enable the transform.
