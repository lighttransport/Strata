# Resume: GLM-5.3-Flash decode optimization on the B550

Written 2026-10-10 at the end of a session. Goal: 30+ tok/s decode on the B550 (stretch 40+), with some quality
loss allowed as long as generated code stays coherent. Work **only on the B550**; tr16 is used by another session
(git and file edits on tr16 are fine, no builds or runs there). Never `p4 submit`; commit only when the user asks.

## Where it stands

Machine: Ryzen 9 3950X (16 cores), 62 GB DDR4 **at 2133 MT/s** (JEDEC fallback; DIMMs are CMK32GX4M2A2666C16, XMP
off), RX 9070 XT 16 GB (gfx1201), ROCm 7.14. Model: REAP-50 Q23
`models/glm53f-reap/GLM-5.3-Flash-REAP50-Q23-assembled.gguf`. Repo on B550: `~/work/Strata`
(branch state = tr16's glm53f files, copied with scp), build `cmake --build build-hip --target strata-glm-decode -j 24`.

Decode, 128 greedy tokens after a 2048-token prefill, confirmed in two alternating rounds (±0.05 tok/s):

| Configuration | tok/s | CPU / GPU wait / head+tail ms | KL vs aff 0 | HumanEval 90 |
|---|---:|---|---:|---:|
| start of goal (aff 0.08) | 17.2 | 32.5 / 19.4 / 5.5 | 0.154 | |
| aff 0.08 + lossless switches | 19.5 | 30.8 / 16.3 / 3.9 | 0.154 | 81 |
| layer profile `prof-mix2.json` (0.08 / 0.14 / 0.18) | 23.3 | 21.5 / 17.4 / 3.8 | 0.187 | 83 |
| mix2 + defer 1 route, cap 0.16 | 25.0 | 22.6 / 13.4 / 3.8 | 0.194 | 83 |
| mix2 + defer 1 route | 26.8 | 23.4 / 10.1 / 3.7 | 0.237 | 82 |
| mix2 + defer 2 routes | 28.2 | 21.1 / 10.3 / 3.8 | 0.286 | 54/60 |

Projected at DDR4-2666 (CPU phase × 0.8): mix2 ≈ 26, cap 0.16 ≈ 28, defer 1 ≈ 31, defer 2 ≈ 32 tok/s.
Full table, readings and dead ends: `docs/fixtures/glm_b550_levers_20261009/levers2/README.md`.

Lossless switches (all opt-in): `STRATA_GLM_HC_FUSED=1 STRATA_GLM_TIER_DIRECT=1 STRATA_GLM_MLA_KERNELS=1
STRATA_GLM_HEADS_BLOCKS=1` (+ `STRATA_GLM_FUSE_ROUTE=1`, within noise). Lossy: `STRATA_GLM_ROUTE_AFFINITY_PROFILE=<json>`,
`STRATA_GLM_DEFER_EXPERTS=1 STRATA_GLM_DEFER_ROUTES=n`, `STRATA_GLM_DEFER_MAX_SHARE=x`, `STRATA_GLM_DEFER_MIN_LAYER=l`.

## Code state

- Committed on glm53f (tr16): `e5232ca3` — fused mHC read, direct tier DMA, MLA decode kernels, heads GEMV, fused
  router+publish, expert deferral (full / check / partial via group-B mailbox slots), `STRATA_GLM_EVAL_WIDTH`.
- **Uncommitted** (tr16 and B550 identical): `src/program/glm_decode.cpp` — `STRATA_GLM_DEFER_MAX_SHARE`,
  `STRATA_GLM_DEFER_MIN_LAYER`, STEP_TRACE tail split (`tail_sync_ms`, `tail_plan_ms`, `tail_copy_ms`); and
  `docs/fixtures/glm_b550_levers_20261009/levers2/README.md`. Commit when the user asks.
- Not mine, leave alone: `include/strata/kernels/glm_prefill.hpp` (`std::size_t` fix by the other session),
  `tools/sim/*`, `docs/GLM_SIM.md`, `tools/glm_q2_humaneval.py`, `tools/glm_q2_run_guard.py`,
  `tools/glm_low_memory_bench.py`, `tools/test_sim.py`.
- B550-only copy: `tools/humaneval_b550.py` (= glm_q2_humaneval.py + `<|user|>` stop 154827, context 4096,
  prefill 2048, `--offset`). Not in git.

## Harness (b550:~/work/kexp/)

| Script | Use |
|---|---|
| `steptrace.sh <tag> [K=V ...]` | speed: per-token cpu / between / head / tail (env from the step-1 config; `AFF`, `THREADS`, `EXE`) |
| `evalkl.sh <tag> ref [K=V ...]` | teacher-forced KL / top-1 / PPL vs `datasets/eval4-aff0.logits` (add `STRATA_GLM_EVAL_WIDTH=1`) |
| `humaneval.sh <tag> [K=V ...]` | pass@1 (`LIMIT`, `OFFSET`; 30 tasks ≈ 7 min); `hesum.py` merges runs |
| `codecheck.sh <tag>` + `codegate.py` | coding-bench fixtures (`TOKENS`, `FIXTURES`); csv needs `TOKENS=1024` |
| `trace_decode.sh` + `chain.py` / `ktrace.py` / `tail.py` | rocprofv3 kernel trace and per-token breakdowns |
| `queue*.sh` | the serial run queues of this session (examples of chaining) |

Always run one measurement at a time on the B550 (scripts wait for load < 3). Decode with mmq prefill is
deterministic per prompt, so repeated coding-bench trials are replicas; use HumanEval for pass rates.

## Remaining items

1. **DDR4-2666 (user BIOS action).** When XMP/DOCP is on: `tools/membw/membw --threads 16 --mib 2048 --repeats 5
   --unroll 8` (expect ~37 GB/s vs 30), then two rounds of the five finalists above with `steptrace.sh`. This is
   the step expected to cross 30 tok/s (defer 1 / defer 2).
2. **Pick the default lossy configuration** with the user: mix2 + defer 1 cap 0.16 (no measurable quality cost) vs
   mix2 + defer 1/2 (faster; the long csv generation fails under every deferral setting). A larger long-form check
   would settle it: e.g. HumanEval with `--tokens 2048` on harder tasks, or the long_cpp coding-bench fixture
   (needs context 16384, check VRAM).
3. **Quality gate before any default changes**: the tr16 `ctest -L glm_quality` gate cannot run while tr16 is busy;
   on the B550 use `evalkl.sh` + HumanEval. Nothing is default-on yet.
4. **Commit** the uncommitted items above when asked.

## Further optimization opportunities (B550)

Per-token budget for mix2 + defer 1 at 2133: CPU 23.4, GPU wait 10.1, head 2.2, tail 1.5 ms.

- **Deferral × affinity conflict.** Heavy affinity leaves only high-weight cold routes, so deferring them costs
  KL. Ideas: defer by layer group only where the aff-0.16 scan showed low sensitivity (layers 24-46 KL/hit ≈ half of
  3-23); choose the deferred route by expected loss (weight × layer sensitivity) instead of lowest weight; adapt n
  per layer from the number of cold routes (n=0 when only 1 cold route).
- **Profile tuning.** Only four profiles were tried (mix, mix2, mix3, mix4). A per-layer fit from the group scan
  (KL ∝ a^1.5, hits ∝ a^0.43) or a finer scan (per 3-4 layers) could buy more hits at the same KL.
- **GPU wait (10 ms with deferral).** Remaining serial chain per MoE layer: wait/add → hc_write → mixer →
  hc_read → norm → router+publish. Candidates: fuse `conv` + `rms_norm_rows` in KDA layers (~100 launches/token),
  fold `native_quantize_q8_1` into the producing kernels (~270 launches, ~1.3 ms of gaps), `kda_chunk` (0.87 ms,
  ~160 GB/s on its state). Dense GEMVs are at bandwidth (480-660 GB/s of 640); do not revisit.
- **Head/tail (3.7 ms).** Tail now instrumented: run one `steptrace.sh` and read `tail_sync_ms / tail_plan_ms /
  tail_copy_ms` from the STEP_TRACE line. Candidates: overlap `tier_submit/tier_plan` with the next step's head,
  read back argmax instead of full logits (155k floats) when sampling is greedy.
- **CPU phase (21-23 ms, DRAM-bound).** Only bytes or bandwidth move it: more VRAM hits (profiles), XMP. GPU PCIe
  streaming of cold experts, MTP, thread count, tier knobs and cold-route pruning are measured dead ends.
- **Startup.** `STRATA_GLM_TIER_DIRECT` registers 52 GB of mapping (~9 s); could run in a background thread during
  prefill.

## Resume prompt

```
Continue GLM-5.3-Flash decode optimization on the B550 (ssh b550, repo ~/work/Strata, build-hip). Read
./resume-b550.md and docs/fixtures/glm_b550_levers_20261009/levers2/README.md first.
Work only on the B550 (tr16 is busy with another session; git/file edits on tr16 are fine). Target 30+ tok/s
decode, quality judged by coding coherence (HumanEval via ~/work/kexp/humaneval.sh) with KL recorded alongside
(evalkl.sh). Current best: 28.2 tok/s (mix2 profile + STRATA_GLM_DEFER_ROUTES=2) at DDR4-2133; 25.0 tok/s with no
measurable quality cost (defer 1, cap 0.16). First check whether XMP (DDR4-2666) is enabled
(`udevadm info /sys/devices/virtual/dmi/id | grep CONFIGURED_SPEED`); if so, re-measure membw and the five
finalists. Then pursue the "Further optimization opportunities" list, one opt-in switch at a time, A/B with
steptrace.sh in two alternating rounds, and gate lossy changes with HumanEval. Do not commit unless asked.
```

## Latest checkpoint: 2026-10-10, software-only round complete

User target: **35+ decode tok/s with coherent code**, **no DDR clock change yet**. Work is local on b550.
All four DIMMs report 2133 MT/s. No work/build/run on tr16; no commit or p4 submit. Original source GGUF stays
unchanged. All new code is uncommitted, and the inherited edits listed above are preserved.

**Best coding-coherent result: Q22 fit4, 33.5708 / 33.6053 tok/s**, two alternating rounds. Same-day Q23
mix2 + defer 2 reference repeats 28.3086 / 28.3065 (one preceding unexplained 26.1032 outlier, same tokens/cache).
Target **35+ with coding coherence is still not achieved**. Q22 fit5 + KDA row parts 4 measured 35.0717 but
failed the matched coding gate: 23/30 passes, six malformed outputs, two token-cap hits. Do not select it.

### Selected experimental recipe

`configs/experimental/glm-b550-q22-fit4-2133.json` (does not change a default):
- Original model `models/glm53f-reap/GLM-5.3-Flash-REAP50-Q23-assembled.gguf` plus
  `models/glm53f-reap/REAP50-Q22-down-only-experimental.gguf` (16,647,198,848 bytes, 42 tensors,
  uncalibrated Q3_K -> Q2_K down-only sidecar, original gate/up untouched).
- CANON=1, native CPU path, 12 threads, GPU budget 15360, cache request 12288 MiB, ctx 4096, mmq prefill 2048.
- fit4 affinity per groups 3–9 / 10–16 / 17–23 / 24–30 / 31–37 / 38–46 = .12/.14/.20/.30/.36/.42.
- HC_FUSED / TIER_DIRECT / MLA_KERNELS / HEADS_BLOCKS / FUSE_ROUTE / NORM_Q8_FUSED / TIER_LOOKUP_ONCE = 1,
  DEFER_EXPERTS=1 / DEFER_ROUTES=2; no share cap, no dense Q3, no MTP, default KDA row parts 1.

Q22 fit4: HumanEval chat first30 **26/30**, valid syntax 30/30, no token-cap hits. Same-day Q23 control 29/30.
Three failures omit required typing imports; one includes an incorrect self-test. No post-hoc corrections to
reported scores. KL against the existing aff-0 Q23 reference = **0.4231298**, PPL 5.6525006 (4 x 512, width1).
Fast rejected variant KL .4627022, PPL 5.8289702. Full164 and long-form code remain untested.

### Code and validation added this round

- `STRATA_GLM_TIER_LOOKUP_ONCE=1`: stable one-token pipeline residency upload once, tiny speed gain.
- `STRATA_GLM_NORM_Q8_FUSED=1`: normalized floats and Q8_1 blocks together; persistent scratch allocated before
  graph capture and shared state reset at batch/incompatible norm boundaries. HIP 32 bitwise kernel cases pass.
- `STRATA_GLM_DENSE_Q4=q3mixers`: lossy Q3 mixer experiment, not selected.
- `include/strata/core/model.hpp`, `tools/glm_q2_pack.cpp`: explicit down-only Q22 sidecar support on validated
  Q23 assemblies. Gate/up changes and stacked sidecars rejected. Decoder permits the native assembled path
  on one NUMA node; original-source and packed CPU backends retain legacy constraints.
- Seven `tools/test_glm_assemble.py` tests pass; real `tools/test_glm_q2_pack.py` integrity/rejection checks pass.
  HIP canon_expert_test, glm_q8_test, glm_q23_parity (including Q2/Q2) and glm_mailbox_test pass.
- Full decoder/converter/q8-test targets build on HIP gfx1201 and CUDA sm89; CUDA compile only on this AMD PC.
- Old decoder / new options off / both lossless switches on produced byte-identical logits on 256 scored
  tokens (4 x64); hashes saved. No default numerical path altered.

### Records and dead ends

`docs/fixtures/glm_b550_levers_20261010/README.md`, `measurements.json`, `quality-summary.json`, raw logs,
exact scripts/profiles, `lookup-once.patch`, `q22-overlay.patch`. Refresh scripts: collect_measurements.py,
aggregate_quality.py. Harnesses accept EXPERT_PACK, GPU_BUDGET_MIB, and optional MTP flags. Check active user
units/processes before another measurement; there should be none after this round. Never edit a running script.

- Very strong Q23 fit7/fit8: 34.06 / 35.02 but 4/10 / 3/10 coding smoke, rejected.
- Strong capped Q22 + Q3 fit8: 34.9952 but 3/10 and seven cap hits, rejected.
- Q22 MTP recheck: 21.43 mix2, 23.02 fit4/depth1, 21.33 fit4/depth2, still a dead end.
- CANON=0 Q23: 20.58; Q22 generic native GPU rejects down type10. **Q22 requires CANON=1**.
- Budget16000/reserve256 and Q3 mixers top out 33.83; CPU threads/tasks stay ~33.6; MLA graphs 32.23;
  KDA columns32 33.62; rowparts4 on fit4 33.94 (not separately quality-qualified).

Next software opportunities: calibrated Q22 if the old calibration inputs can be copied from tr16 (files/read
only, no inference/build there), finer layer-affinity allocation while keeping early layers mild, exact-order
GPU producer/launch optimizations. Tail is only ~1.6 ms; further strong early affinity is not a useful shortcut.
DDR-2666 remains a later explicit user BIOS action; do not change it in software.

## Newer checkpoint: 2026-10-10, longer-context prefill work in progress

The previous decode changes, outgoing-history fixes and setup-test mock fixes
were committed and pushed to origin/glm53f through e4dbb18a. That push is complete.
The new prefill work below is uncommitted. DDR remains 2133 MT/s on all four DIMMs;
no tr16 inference/builds and no model-file edits.

User now prioritizes 8K+ prefill and approved faster FP16 only if quality checks pass.
Fixtures are `docs/fixtures/glm_b550_prefill_20261010/`.

- First 8K FP16 run: 285.353 tok/s versus matched first-prefill MMQ 89.0986.
  Final-prompt top1 agrees; KL(reference, FP16) 0.00015057. This is one position.
- 16K FP16 warm repeats: 322.795 / 322.406 prefill, 33.2524 / 33.9327 decode.
- New 32K candidate: 289.963 prefill, 32.0677 decode, RAM peak 48.836 GiB,
  minimum physical VRAM free 1371 MiB, complete/clean guard pass.
- Unrestricted FP16 short coding: 27/30 passes, three malformed, three cap hits.
  Matched 8K-context MMQ: 27/30, two malformed, one cap. Do not select unrestricted FP16.
- Added opt-in `STRATA_GLM_F16_PREFILL_MIN_TOKENS=4096`: short batches use MMQ,
  larger batches use FP16. All 30 short tasks then match the MMQ token hashes exactly.
  The inherited 8K MMQ syntax/cap failures remain; this does not establish full coding quality.
- Prefill-only KDA columns 32 / chunk 256 with KDA_PREPARE=1; decode keeps columns
  128 / chunks 64. Expanded prepared/unprepared kernel parity passes bitwise at
  columns 32/64/128 and chunks 64/256/2048. HIP and CUDA decoder/test builds pass.
  SYCL does not build these GLM GPU targets. New controls unset reproduce the old
  baseline's first 128 generated tokens exactly (default-parity.json).
- Optional CLI `--dump-prefill-logits=PATH`, benchmark `record_prefill_logits` and
  `profile_layers`; coding harness supports context/batch/mode/threads/background
  and explicit task subsets. Generated Python still executes inside bwrap.

Active sequential performance matrix: exec session 69165,
`python3 /tmp/b550-prefill-matrix.py`, progress `matrix.runner.txt` in fixtures.
It completed hybrid-32k-01 and is on base32k-01; then base16k-01,
hybrid-16k-01, f16-default-kda8k-01 and hybrid-8k-01. Do not run another GPU job
or compile during qualified timing. Poll using write_stdin and read compact summaries.

Next: finish matrix; run `/tmp/b550-prefill-recall.py` sequentially for 8K/12K/16K/32K
with 256 raw tokens and score_recall.py (only first assistant turn counts).
The 12K recall has 9673 input tokens and a 1481-token MMQ tail, testing the mode transition.
Run matched long coding on HumanEval indices 10,11,17,25 with frozen
background-14000.txt at ctx16384, baseline MMQ and candidate hybrid. run_quality.py
accepts --task-indices and --background; launch via 60 GiB systemd scope.
Background/prompt manifests contain immutable hashes. Add measured table, selection
decision, configs and checkpoint after quality checks; no new commit/push unless asked.
The original coherent 35+ decode target remains unmet.

### Later prefill checkpoint (supersedes active-session notes above)

All timing matrices above finished. Qualified first-prefill MMQ rates:
8K 89.0986, 16K 93.9893, 32K 94.5277 tok/s. base32k-01 and hybrid-8k-01
are excluded for external `hf` CPU interference. The other measured cases are clean.

Fastest measured settings: F16_BUCKETS=1, F16_BUCKET_STEP=64, prefill KDA
ROW_PARTS=4 / COLUMNS=128 / CHUNK=256 / KDA_COLUMN_TILES=1 / KDA_PREPARE=1,
F16_PREFILL_MIN_TOKENS=4096. Batch8192 at ctx8192: 453.489 prefill /33.4427 decode.
Batch16384 at ctx16384: 595.696 /33.4047. Batch16384 at ctx32768: 616.434 /32.1229.
Largest RAM peak49.452 GiB; minimum VRAM free1393 MiB at32K. All guards clean.

Unrestricted KDA row-part optimization still failed the short syntax gate:
he-best-short 27/30, three malformed, one cap vs MMQ27/30,two malformed,one cap.
Added `STRATA_GLM_PREFILL_KDA_MIN_TOKENS=4096`, gating ALL prefill geometry;
short batches inherit global columns/rowparts and64-token chunks. Prepared inputs
remain bitwise-tested. New HIP/CUDA builds pass (gated-build logs in /tmp).

Current active GPU job: gated best short30, exec session52207,
`he-gated-best-short` fixture. Completed tasks so far match the MMQ token hashes.
After it finishes, run `/tmp/b550-prefill-final-long-quality.py`: matched MMQ and
gated-best on HumanEval10,11,17,25 with14K code background,ctx16384. Then run
`/tmp/b550-prefill-final-recall.py` for8K/12K/16K/24K/32K,256raw output tokens.
Both helpers are prepared but NOT started. Add a final32K timing on the gated build
after the quality tests; avoid compilation or another GPU job during timing.

Five recall probes include9673 tokens with batch8192 (MMQ tail1481), and19423
tokens with batch16384 (MMQ tail3039). Exact prompts/hashes are frozen in manifests.
Latest candidate configs are q22-cCONTEXT-bBATCH-gated-best.config.json in fixtures.
Initial-logit diagnostics keep top1 at8K/16K/32K; KL values for fastest settings are
.0003749 / .0135344 / .000003566. Single-position diagnostics are not full quality.

Still required: finish quality and mixed-tail recall gates; select only passing
settings; save optional experimental presets and measured report; update checkpoint.
New work remains uncommitted. No DDR change, no tr16 work, no new push.

### Long quality result and fallback (latest)

Gated best short30 passed27/30 and all30 token hashes matched MMQ.
Matched14K-background long tasks10,11,17,25: MMQ3/4; gated row4 FP16 2/4.
New failure task25 means fastest616 tok/s candidate is REJECTED for selection.
Do not run the prepared gated-best recall helper as a selection gate.
Serial-KDA fallback now running: he-gated-serial-long-targets, config
q22-c16384-b16384-gated-serial.config.json (rows1,columns32,other gated settings).
Its first task10 failed, as baseline did; remaining tasks pending.
No presets selected, no commit/push. DDR remains2133.

Serial-KDA FP16 finished2/4 (new failure17), REJECTED too. Larger-batch MMQ
fallback running exec47420, he-mmq-gated-long-targets, same4 frozen longtasks;
config q22-c16384-b8192-mmq-gated.config.json. FP16 recall is no longer needed
for selection. If MMQ quality passes, check recall and final timing on that path.

MMQ gated long4 finished3/4 and ALL FOUR token hashes exact baseline.
Recall ladder now running exec91341 via /tmp/b550-prefill-mmq-recall.py,
recall-mmq-gated-CONTEXT-01, contexts8192/12288/16384/24576/32768, batch8192.
After recall, run final guarded code timings8K/16K/32K on these MMQ configs,
then optional presets/report. New FP16 candidates remain rejected.

MMQ recall8K/12K/24K passed strict4fact JSON+stop;16K correct values but
prose prefix caused strictformatfailure.32K hit systemdRuntimeMax15m30s,
no resultJSON; progress retained, kswapd0 activity/accountingchange, cause
unconfirmed. Do NOT select32K. Finaljob exec11827: /tmp/b550-prefill-mmq-finish.py
runs matched16K MMQcontrol then short30 then qualified8K/16Kcodeperf.
Perfhelper32K removed pending diagnosis/qualification.

Final MMQshort30 complete27/30 and all30 tokenhashes exact control.
Matched16K recallcontrol outputALL256tokenexact candidate, so bothstrictformat
failures shared. Final8K code timing clean118.546 prefill /29.6735 decode,
ALL128generatedtokens+cachefingerprint+294tierswaps exact earlierbase8k-02.
Need current-condition control before attributing decode timingdifference.
Final16Kattempt01 complete115.237/33.1407 but NOTclean (kswapd0 interference).
Exec11827 auto-retrying16K throughattempt03. After it exits, run prepared
/tmp/b550-prefill-mmq-controls.py for matched current8K/16K baseline timings.
32K remains excluded service timeout; no32Kpreset. No newpush/commit.

Current8Kcontrol clean87.4782 prefill/33.3769decode vs candidate118.546/29.6735:
+35.5%prefill but -11.1%decode initial; all128tokens/cachefingerprint/tierswaps
identical. Current16Kcontrol also accountingcollapsed (memorycurrent19GiB
vsRSS46GiB, GPUbusy3%), stoppedowncontrol, recordretained. So pressure affects
originalMMQ too. Own16Kcandidate retry stoppedafterdisqualifiedreclaim; no
more16Kretries. Final8Krepeat now exec57313, final-mmq-8192-02, timeout300.
Afterrepeat: optional8Kpreset with explicittradeoff, measuredREADME/summary,
finaldiffchecks/checkpoint. No16K/32Kselectedpreset pending clean qualification.

### Completed prefill optimization checkpoint (supersedes active sessions)

All own inference jobs exited. Final 8K repeat clean: prefill107.489 / decode33.3234,
versus same-day current control87.4782 /33.3769: +22.9% prefill, decode near control.
First8Kcandidate clean118.546/29.6735; retain variation, do not hide slowdecode.
Both8Kcandidate outputs all128tokens and decodecachefingerprints exact currentcontrol.
Selected optional config: configs/experimental/glm-b550-q22-fit4-prefill8k-2133.json.
MMQ experts, batch8192, KDAprepare1, prefillcols32/chunk256/min4096, scratch4096.
Benchmark-only STEP_TRACE and prefill logit dumps omitted from selected config.
Original fit4 and defaults unchanged. DDR2133, GPUautomatic, no tr16 work.

Finalshort30:27/30, two malformed, one cap; ALL30tokens exact MMQcontrol.
Long14Kbackground tasks10/11/17/25:3/4, ALL4tokens exact control. Knownfailure10.
FP16 row4 and serial candidates rejected2/4 longcoding (newfailure25/17).
Recall8K/12K/24K strict4facts+JSON+stop passed.16Kstrictformatfailed dueprose,
but matchedcontrol ALL256tokens identical and expectedvalues present. Do not
loosen scorer or report strict16Kpass.32KMMQrecall service timeout15m30s.
16Kcodeattempt01 excludedkswapd; retry02 stoppedalreadydisqualifiedreclaim.
Current16Kcontrol also accountingcollapsed/lowGPU, stoppedownjob, evidence retained.
Externalhf used~12GiBRAM; left alone. Sharedpressure possiblecause, notconfirmed.
No16K/32Kpreset selected; requalify on quiet system before recommendation.

HIP/CUDA builds pass; HIP expandedGPUparity passes; CUDAcompile-only; SYCL doesnot
build theseGLMGPUtargets.11harnessunit tests passed. Defaultunset128tokenparity
passed. Docs fixturesREADME and summary.json/final-comparison.json record results.
New changes remain UNCOMMITTED/UNPUSHED. Previous originglm53f e4dbb18a unchanged.
Preserve unrelated release/ and tools/hf_release_upload.py, plus other priorfiles.
35+ coherentdecode goal remains unmet; selected longerprefill recipe doesnot claim it.

### BF16 WMMA experiment in progress

User authorized BF16 WMMA and asked to use ~/Work/gemm/main. Actual readablepath
references/gemm, reference AGENTS read; no edits there. Ported RDNA4
128x128/32-K BF16 fragments from rdna4/vlm/bench_vlm_gemm.c and tuned layouts/
prefetch from llm/hip_llm_runner.c and vlm/generated/mm0_bf16_directa_pf2.hip.
Reference MIT license retained in src/prefill/wmma_gfx12.cu, sourcehashmanifest
in docs/fixtures/glm_b550_bf16_20261010/reference-manifest.json.
New opt-in --prefill-experts=bf16-batched, STRATA_GLM_BF16_WMMA=1 custom (absent:
library BF16). FP32acc/output, BF16activation and directBF16 quantweightstores.
SingleGPU only. DefaultMMQ/F16 modes unchanged; BF16 onlyprefill, noDDR/GPUclocks.
Added Gemm::bf16_batched, glm_BF16cast/gather, iq_dequant_bf16 and genericIQcoverage.
Server allowlist/humanharness includebf16-batched. Allworkuncommitted.

HIP/CUDA finalbuilds pass (tmp/bf16-prefill/build-*-final-all.txt). CPUtests5server,
11benchmark pass. FinalCTest glm_prefill_parity+glm_bf16_prefill_test bothpass.
GPUCPU/libraryoracles, batchedbounds, maskedgather, Q2_K/Q3_K+IQdirectRNE pass.
Microv1scalar 15.64/7.10TF vsBLAS89.97/89.05; v2vectorLDS32.43/32.99 vs80.14/97.25;
v3prefetch43.41/42.64 vs98.22/97.13. ShapesT128N4096K4096B2 andT256N4096K2048B2.
Rawlogs/snapshotsv1/v2 preserved in BF16fixtures. No GPUjob overlaps qualifiedperf.
LibraryBF16 matched14K codingtasks10,11,17,25 completed2/4 (10/11passed;17/25fail),
vsMMQ3/4. Rejectedqualitygate. CustomWMMA resident4case completed10fail then
engine300sec silence timeout on11 duringdecode afterprefill. Partialnotqualified.
HF~15GiBRAM, globalSwapfull, concurrentreclaim/buildactivity recorded; rootcause
notisolated. Processcleanedup. NoBF16preset selected yet.

Active helper tmp/bf16-prefill/run-fresh-quality.py (GPUjobsee currenttools session):
freshsingle WMMA17/25, eachserviceRuntime240 +RAM60G/Swap0. Theseprobesdifferent
residentprotocol, don'tclaimmatchedqualityparity. Afterfinish, run prepared
python3 tmp/bf16-prefill/run-perf.py -> BF16fixtures/perf.runner.txt. It runs MMQ8K,
BF16BLAS8K,WMMA8K,WMMA16K onceeach withguardtimeout300/450 and128outputs. Preserve
rejectedcases, don'tretryunderinterference. Then writeREADME/summary/default128
parity, finalchecks and checkpoint. No commit/push. Preserve release/ and
user tools/hf_release_upload.py. New scratch in tmp/bf16-prefill ignored locally.

BF16freshWMMA17+25 bothcompleted andfailedtests (85/102generatedtokens).
Resident WMMA partialfirst10fail,then300ssilence11. NeitherBF16 candidatepasses
quality; do NOT selectpreset. Micro+kernelparityvalid, notfullqualityqualification.
Activeperfhelper tmp/bf16-prefill/run-perf.py ->BF16fixtures/perf.runner.txt;
MMQ8K/BF16lib8K/WMMA8K/WMMA16K boundedfreshservices, noCPUbuildnow. Afterallfinish,
extractclean vsrejectedresults, verifydefault128 againstfrozencontrol, writeresult
summary/README andnewcheckpoint. FinalCTest bothGPUtests passed10.58sec.

Verified SDK correction: /opt/rocm/include/hip/hip_version.h reportsMAJOR7 MINOR14
PATCH60850; earlier7.1.4 label wasmiswritten. UseexactinstalledHIPversion from
/opt/rocm/share/hip/version forcurrentBF16report. Historicalmeasurements needtheir
ownversionevidence; do notguessoroverwritetheirversionmetadata.

### BF16 experiment completed (supersedes active-session notes)

All own inference/build jobs have exited. Opt-in --prefill-experts=bf16-batched;
STRATA_GLM_BF16_WMMA=1 selects gfx1200/gfx1201 custom, otherwise library BF16.
Only expert prefill changes, FP32 accumulation/output; singleGPU, no fallback
for custom unavailablehardware. New IQ BF16 path isolated as dequant_expert_bf16,
legacygeneric supportqueries/entrypoints remain unchanged. Existing modes/defaults
preserved. Model/sidecar/referencegemmrepo unedited. DDR2133 and GPUauto unchanged.

Final HIP and CUDA rebuilds passed: tmp/bf16-prefill/build-{hip,cuda}-isolated.txt.
Final HIP CTest glm_prefill_parity+glm_bf16_prefill_test pass,10.37sec total.
Server5 andbench11 tests pass. CUDA compile-only/noNVIDIA; SYCLnotarget/noSDK.
SDKcorrection: installedHIP7.14.60850 localpackaging, notprior7.1.4 label.

Qualified coldcodeprefix timings (ctx/input,PP8192 unlesscontrol2048), each128outputs:
MMQ8Kcontrol57.1006pref/33.3381decode; defaultALL128tokens exactfrozenoldreference.
BF16library8K389.076/33.4788; WMMA8K392.713/33.4540; WMMA16K426.786/33.4496;
WMMA32K445.692/32.1624. FinalhostAPIisolation WMMA8Krepeat405.242/33.5823 clean,
ALL128tokens exactearlierWMMA8K. Largest cgrouppeak49.104GiB, minGPUfree>=1310
forBF16cases. FasterMMQ8Koptimizedcontrol timedout330s withaccountingcollapse,
no finalizedresult; retainprogress/runner. Don'tclaimratioagainstits partialrun.
Microcustomv3~43TFLOPS vslibrary~97/98 onactualexpertshapes. v1/v2 snapshots/logs
preserved; actualportreferencehashes+MITlicense retained. Noestablishedcustom
advantageoverlibrary in fullmodel single-run 8K timings.

QUALITY: libraryBF16long14K frozen tasks10,11,17,25 scored2/4 vsMMQ3/4; newfails17/25.
ResidentWMMAfourtaskprotocol incomplete:10failed,11engine300s silence afterprefill.
Concurrenthf/build/reclaimactivity recorded; rootcauseunconfirmed. Freshsingle
WMMA17/25 completedbothfailedmissingimports; differentprocessprotocol,notmatched
qualityparity. NO BF16selectedpreset/defaultchange. CurrentMMQpreset remains;
needsquiet-memoryrequalification forrepeatabilityunderheavyexternalRAMuse.
35+ coherentdecode remainsunmet. BF16 report+summary:
docs/fixtures/glm_b550_bf16_20261010/README.md,summary.json,reference-manifest.json,
timing-build-manifest.json (earliertimingbinary/source),build-manifest.json(final),
final-wrapper-parity.json. Rawconfigs/results/qualityrecords allretained.

New code/docs remain UNCOMMITTED/UNPUSHED, origin glm53f still e4dbb18a. Preserve
unrelated release/,tools/hf_release_upload.py and other prioruntrackedfiles.
Only local scratch tmp/bf16-prefill (ignored with itslocal.gitignore).

Finalcomment-onlycleanup rebuiltHIP/CUDA successfully, logsbuild-*-complete.txt.
Repeatmeasurementbinary manifest archivedrepeat-timing-build-manifest.json;
currentbuild-manifest regenerated. Allownjobs exited; finalgitdiffcheck clean.
