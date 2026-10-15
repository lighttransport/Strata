# Resume: two-GPU GLM-5.3-Flash REAP-224 runner on TR16 (target 50+ decode, 900+ prefill at 8K)

Written 2026-10-14 22:25 JST. Repo `/mnt/nvme02/work/Strata`, branch `glm53f`, HEAD `eed6ccd9`. **Nothing committed.**
Read `AGENTS.md`, `docs/GLM_TR16_RUNNER_PLAN.md` (simulator, calibration, probes, ranked lever path) and
the plan file `~/.claude/plans/plan-multi-gpu-5060ti-9070xt-functional-pnueli.md` first.

## Targets and rules
- User target (latest): **50+ tok/s decode, 900+ tok/s prefill at 8K**, multi-GPU (RX 9070 XT primary on HIP
  + RTX 5060 Ti worker on CUDA, or the reverse), REAP-224 Q23
  (`/mnt/nvme01/models/glm53f/prune/GLM-5.3-Flash-Q23-REAP224-assembled.gguf`, prior
  `/mnt/nvme01/models/glm53f/prune/coverage-keep224.json`).
- Quality gate unchanged: HumanEval >= 152/164 and 35-track corpus gate. Lossless levers need equal token streams;
  anything else runs the gate. The user said to skip the quiet re-run and go straight to optimized runs.
- New features opt-in, default byte-identical; build CUDA (`build-glm-release`) and HIP (`build-hip-rocm101-tr16`,
  `LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib`) for every touched file. No commits or pushes without permission.
- Shared machine: desktop holds ~2.4 GB of the 5060 Ti, so CUDA-primary runs need `decode_cache_mib` <= 10240.
  Use `build-exl3-venv/bin/python` for bench tools. `pgrep -f` patterns in scripts match their own command line:
  anchor them or wait by PID.

## Simulator (done, uncommitted)
`tools/sim/glm_runner.py`, `tools/sim/glm_traces.py`, CLI `tools/glm_runner_sim.py`
(calibrate / validate / decode / prefill / sweep / manifest / probe-plan / records-from-bench), data in
`tools/sim/data/{hw_tr16_runner,glm_kernels_tr16,glm_runner_params_tr16,glm_runner_records_tr16,glm_levers}.json`,
tests `tools/test_glm_runner_sim.py` (`PYTHONPATH=tools python3 -m unittest tools.test_glm_runner_sim tools.test_glm_heterogeneous tools.test_sim`, all pass).
Probes: `tools/glm_tr16_runner_probes.py`, `tools/glm_mixed_worker_loopback.py`, `tools/glm_runner_profile_report.py`,
`tools/glm_transfer_probe.cu` (new bounce/register modes). Receipts: `docs/benchmarks/tr16_runner_sim_20261014/`.
Key findings: decode is bound by the primary GPU chain (~1.0 ms per slot: dense projections ~0.45, residents ~0.10,
gaps ~0.31, ~60 launches); both vendors can HostRegister one shared memfd; lossless sweep predicts ~49.5 decode with
the interleaved layer split, ~42 without; prefill ~900-930 needs HIP mixer x1.5 and expert FFN x2-3.

## Engine changes made this session (uncommitted, NOT YET BUILT OR TESTED)
Backups of the originals: `/mnt/nvme01/tmp/claude-1000/-mnt-nvme02-work-Strata/6585116e-1342-4708-81b1-7f8916240f9d/scratchpad/`
(`glm_decode.cpp.before-side`, `glm_mixed.h`, `glm_gpu_worker.cpp`).
1. `STRATA_GLM_RESIDENT_STREAM=1` (`src/program/glm_decode.cpp`): resident expert rows of each verify group run on a
   side stream outside the captured mixer segment (`enqueue_moe_residents_side`, events `resident_fork/done/consumed`),
   joined before that group's wait-add. Refuses strict combine and deferral (needs `STRATA_GLM_CANON_COMBINE=device`).
   Graph key gets `+128*resident_side_stream`.
2. Protocol v5 (`include/strata/host/glm_mixed.h`): `STRATA_MIXED_VERSION 5`, `STRATA_MIXED_OUTPUT_ROWS 8192`
   (one chunk/ack per layer for partial prefill), new `activation_half, activation_pad` fields.
   **This changes the default protocol and buffer size** (~+120 MiB memfd): before keeping it, either make it opt-in or
   confirm with the user; `glm_mixed_transport_test` and `tools/glm_mixed_worker_loopback.py` offsets must be updated
   (loopback tool hard-codes v4 layout).
3. `STRATA_GLM_MIXED_REGISTER=1` (decoder + `src/program/glm_gpu_worker.cpp`): `cudaHostRegister` of the shared mapping;
   activations DMA straight into it, worker uploads from it, replies DMA from it.
4. `STRATA_GLM_MIXED_EXCHANGE=f16` (with REGISTER): primary converts prefill activations to FP16 (`k::glm_f16`), worker
   gathers with `glm_gather_expert_f16_from_half`. Decode requests set `activation_half=0`.

## Screen results so far (hip-cuda unless noted, 8K holdout-code, 512 tokens, single timed trial, +/-10% noise)
Configs in `build-glm-release/runner-sim/screen-20261014/*.json` (copies of
`docs/benchmarks/tr16_mixed_20261014/numerical_profile.json` + knobs), runner `run.sh` (`CASES=cuda-hip` for the reverse).

| variant | decode tok/s | note |
|---|---:|---|
| s0 base | 27.1 | prefill 535 |
| s1 SPLIT_PROJECTIONS | 27.7 | between waits doubled, CPU service down |
| s2 BATCHED_RESIDENT | 32.6 | best so far; token stream differs from base (check gate) |
| s3 MLA_GRAPHS | 30.6 | |
| s4 split+mla | 24.4 | |
| s5 batched+mla | 31.2 | |
| s6 batched+mla+split | 27.2 | |
| s0 base, cuda-hip (cache 10240) | 32.5 | CUDA primary not worse than HIP primary here |
| s5, cuda-hip | running at handoff | |

## In flight at handoff
`build-glm-release/runner-sim/chain2-20261014.sh` (PID 584375, log `chain2-20261014.log`): finishes the s5 cuda-hip screen,
then builds `strata-glm-decode strata-glm-gpu-worker glm_mixed_test glm_mixed_transport_test` for CUDA and HIP and runs
the host tests. Check the log for build errors first.

## Remaining tasks, in order
1. Fix any build errors from changes 1-4; decide v5 default vs opt-in (keep defaults byte-identical); update
   `src/host/glm_mixed_transport_test.cpp`, `tools/glm_mixed_worker_loopback.py` to v5; run `ctest -L '^glm$'` on both builds.
2. Screen (3 trials each, both orientations) the prepared configs `s7-s5-residentstream`, `s8-s5-register`,
   `s9-s5-register-f16`, `s10-s5-residentstream-register-f16` (and `-cudaprimary` copies) with the new binaries.
   Check token-stream equality against s5 for the lossless claims (resident stream and register must be identical;
   f16 exchange changes prefill only: compare prefill logits hashes).
3. Run the next sweep levers in the engine: head ordering (group A publish before B's dense layers, enqueue order at
   `glm_decode.cpp` ~6815), fused publish (`STRATA_GLM_FUSE_ROUTE=1` exists, only width 1: extend to width 2),
   sampling/lookup trim, CPU span access, MLA KV kernel, GPU draft bank (`--mtp-experts=gpu` refused with
   `--decode-cache-mib`). Then the interleaved layer split (topology B) if the worker topology stays below 50.
4. Prefill to 900+: primary fraction 0.6-0.75 (`STRATA_GLM_MIXED_PRIMARY_FRACTION`), F16 exchange + one partial per
   layer, resident skip, HIP mixer sub-batch / hipBLASLt, expert FFN efficiency (rocprof decomposition first).
5. After each lever: record measured vs predicted (`tools/glm_runner_sim.py validate --add-record`), recalibrate.
6. Best speed candidate: corpus gate (`tools/glm_tr16_quality.py`) and HumanEval (`tools/glm_q2_humaneval.py`,
   pass `--prefill-experts f16-batched`, mixed via `tools/glm_mixed_decode.py`), 512x3 at 2K and 8K.
7. Update `docs/GLM_TR16_RUNNER_PLAN.md`, this file, and the memory note `tr16-runner-simulator`.

## Prompt to resume
> Resume the TR16 two-GPU GLM-5.3-Flash REAP-224 work from `resume-multi-gpu.md` (target 50+ decode, 900+ prefill at
> 8K, HumanEval >= 152/164 gate). Read AGENTS.md and docs/GLM_TR16_RUNNER_PLAN.md. No commits or pushes. First check
> `build-glm-release/runner-sim/chain2-20261014.log` for build/test results of the uncommitted engine changes
> (RESIDENT_STREAM, MIXED_REGISTER, MIXED_EXCHANGE=f16, protocol v5), fix them keeping defaults byte-identical, then
> screen s7-s10 in both orientations with 3 trials and token-stream checks, and continue down the remaining-task list.


## Follow-up in progress, 2026-10-15 (Codex)

See section 9 of `docs/GLM_TR16_RUNNER_PLAN.md` for implemented knobs and validation limits. CUDA/HIP
builds and all 22/21 GLM tests pass; 140 Python tests pass. No commits or pushes.

The default transport is restored to v4; v5 is opt-in and queried by both binaries. Resident work now
uses its own graph, with reader events and reset/teardown cleanup. Added opt-in batched fused routing,
worker GPU dispatch, fused SwiGLU/F16, runtime-matched HIP single products and GPU draft/cache compatibility.
The resumed quality gate and prefill coverage are explicit in the quality driver.

The identical frozen HIP v1 binary is not repeatable on the plain canonical 512-token prompt sample:
its two runs differ in tokens and prefill logits (4.08% relative L2). CUDA tokens and prefill/final logits
match the frozen baseline on the same sample. Worker rows and ordered partials match before/after on
both vendors. Do not claim HIP whole-model losslessness from these checks.

The old task-owned screen controller was stopped only after its active benchmark finished. Revised
screens run via `tools/glm_tr16_optimize.py`, with frozen binaries and three timed trials, under
`build-glm-release/runner-sim/opt-codex-20261015-screen/` (driver log `/tmp/strata-v4-screen.log`). Start with
that log/results. The installed HIP library is 100401, not the bundled gfx1201 table's 100500; a 62-shape
accuracy-checked table is in `build-glm-release/runner-sim/opt-codex-20261014/hipblaslt-tuned.txt`.

Next: inspect the revised s7-s10 results, choose the measured parent/orientation, then screen fused-route,
dispatch, SwiGLU, hiplt and draft variants individually and together, with fractions 0.5/0.6/0.67/0.75.
Use new output roots and do not overlap timed GPU work with compilation, profiling or CPU-heavy tests.
The targets remain 50/900 at 8K on code and chat separately, with HumanEval >=152 and the resumed corpus
gate. No speed candidate is quality-qualified yet. The layer split is still conditional on measured
worker results and a validated forecast; do not report it as implemented.


The first revised screen root was rejected by the swap guard (launcher VmSwap); no speed data from those
attempts is usable. It was stopped and retained. The restarted screen is under a task-only systemd user
scope with `MemorySwapMax=0`, at `build-glm-release/runner-sim/opt-codex-20261015-noswap/`, driver log
`/tmp/strata-v4-noswap-screen.log`. Global swap was not changed. The screen tool now requires an inherited
no-swap cgroup for future launches.


## Session 2026-10-15 (Claude): decode screen and decision point

Section 10 of `docs/GLM_TR16_RUNNER_PLAN.md` has the table. Short version: the Codex HIP-primary collapse
(14-18 tok/s) is the resident side stream (and registration on HIP), not the code baseline; the current
binaries give 28.9 on the parent. Best measured single trials: no adaptive refresh + 4 tasks/thread or
prefetch 256/1024 at 32.6-33.2 tok/s (no adapt needs the quality gate). Rejected: resident stream, device
dispatch + register, group merge, thread-count changes. Worker registration fails on the CUDA worker
("invalid argument", likely the memfd size not page-aligned for cudaHostRegister). `--mtp-experts=gpu` is
still refused with `--decode-cache-mib` despite `MTP_GPU_CACHE`.

Forecast: full lossless stack 44.4 with the layer split, 40.5 without. **Decision needed from the user**
before building the layer split (multi-week, ~+4 tok/s forecast): (a) lock in and qualify the measured
settings (~33), (b) fund the layer split anyway, or (c) allow a lossy budget (routing affinity, residency,
acceptance) to go beyond ~44.

Next steps that do not depend on that decision: repeat the best combination (no adapt + tasks 4 +
prefetch 256) with 3 trials in both orientations; fix CUDA-worker registration (page-align the mapping);
fix the GPU draft-bank refusal; run the corpus gate on no-adapt.


## Layer split, 2026-10-15 night (Claude)

Implemented and working (coherent output), not faster. Section 11 of `docs/GLM_TR16_RUNNER_PLAN.md`.
Code: `STRATA_GLM_LAYER_RANGE`, stage protocol (`include/strata/host/glm_stage.h`, `src/host/glm_stage.c`,
`src/host/glm_stage_test.c` registered as `glm_stage_test`), `include/strata/core/glm_stage.hpp`, decoder
hooks (`stage_exchange`, `stage_forward`, `serve_stage`, `--stage-protocol-size`), launcher
`tools/glm_stage_run.py`, runner `build-glm-release/runner-sim/split-20261015/run.py` (logs there).
Builds pass on CUDA and HIP; `ctest -L glm` passes on both builds after these edits.
Result: 12.5-17.6 tok/s against the 36.0 tok/s mixed control; per round primary ~60 ms + stage ~45 ms
sequential. Analysis says cross-stage overlap cannot beat the sequential round. Not recommended to continue.
Best measured config: mixed runner, `STRATA_GLM_MIXED_ADAPT_CACHE=0` (+ tasks/thread 4 or prefetch 256),
36.0 tok/s quiet; needs the corpus gate and HumanEval.

## Fast-iteration push, 2026-10-15 early morning (Claude)

Screens: `build-glm-release/runner-sim/screen-20261015/` (`run.sh`; env `PROMPT`, `TOKENS`, `TRIALS`, `CASES`,
`COMBINE`), prompt `build-glm-release/runner-sim/prompt-1k.ids` (first 1000 + last 24 tokens of the 2K
holdout code prompt, no trailing newline), 256 tokens, one warm-up + 3 timed trials. Quality deferred by the
user; stream health (distinct 4-grams, loops) checked per run.

Path (means of 3 timed trials, decode tok/s):
- no adaptive refresh (`MIXED_ADAPT_CACHE=0`), hip-cuda: 35.8
- + `ROUTE_AFFINITY=0.30`: 38.8
- + GPU draft bank (`STRATA_GLM_MTP_GPU_CACHE=1`, `mtp_experts: gpu`; code fixes: main's decode-cache refusal and
  the early MTP preparation now accept it): 41.9
- orientation cuda-hip (CUDA primary, decode cache 10240 capped by VRAM at 426 slots): 44.0
- + `STRATA_GLM_SPLIT_JOINT_DENSE=1` (new: dense layers 0-2 once per window; same tokens): 45.1
- + `MTP_ACCEPT_TOP_K=8 MTP_ACCEPT_MARGIN=6` (lossy): **48.1** (47.9 / 49.6 / 46.9)
No effect: `STRATA_GLM_MIXED_SPIN_US` (new socket busy-poll), primary tier size, worker cache in hip-cuda,
depth 4, wider acceptance (margin 8, top-k 12, max soft 4).
- + `STRATA_GLM_TIER_LOOKUP_BATCH=1` (new: resident lookup uploaded once per verify step; same tokens): 48.5
- + worker `STRATA_GLM_MIXED_SCAN_ONCE=1` (new: skip the full-cache eviction scan when the primary mask is
  unchanged; prologue 0.06 -> 0.015 ms/request) + `STRATA_GLM_MIXED_FUSED_RETURN=1` (existing): **49.0** at 1K/256
  (48.7 / 50.5 / 47.7); at **8K/512: 46.1** (44.2 / 46.3 / 47.7), prefill 8K 510 tok/s. The 8K warm-up trial
  looped (distinct 4-grams 0.45, repeat guard blocked 0): lossy settings, quality gate must check loops.
Best config file: `build-glm-release/runner-sim/screen-20261015/p4-both.json`, case cuda-hip (CUDA primary, HIP
worker), decode cache 10240 (tier capped by VRAM at 426 slots). Lossy parts: affinity 0.30, soft acceptance
top-k 8 margin 6, no adaptive refresh, fused return rounding. Diagnostics: `STRATA_GLM_MIXED_PROFILE_DECODE=1`
(worker phase totals); in-run nsys wrapper `build-glm-release/runner-sim/nsys-mixed/primary.sh`.
Per step now: the primary GPU spends ~0.42 ms/slot in kernels and ~0.55 ms waiting for CPU+worker; worker
body ~0.27 ms/request. No effect: thread count 13-15, pool spin, task granularity, worker cache size.
Next ideas: draft overlap (draft the next window on the HIP card during verify), worker body (one sync, smaller
copies), layer-3 bubble, quality gate on p4-both (corpus + HumanEval + loop check at 8K).

## Draft overlap check, 2026-10-15 afternoon (Claude)

Not built: measured to lose. The next window's draft needs the verify's final hidden state and bonus token, so
an overlapped draft has to start from the MTP's own guesses ("ahead" drafting). Depth-7 probe on p4-both, 1K/256
(`screen-20261015/d7-probe.json`, `out-d7-probe`): a round fully accepts and the MTP guesses the bonus in 22/62
(warm-up) to 32/49 (timed) rounds; the ahead window then accepts 1.38 drafts on average against 2.65 for a fresh
window. Saving the ~8 ms draft in those rounds costs ~1.3 tokens each: net loss.
Draft cost (in-run nsys, CUDA primary): output head 0.85 ms/step (q4_k, 154880 rows), draft experts gu+down
0.68 ms/step, 3 steps/round. A draft vocabulary subset saves at most ~2 ms/round but token ids < 65536 cover only
~95% of prompt tokens, so acceptance drops; MTP_DRAFT_TOP_K is ignored on the GPU draft bank (~1 ms/round at best).
Remaining lever for 50 at 8K is the CPU+worker wait (~55% of each primary slot).
