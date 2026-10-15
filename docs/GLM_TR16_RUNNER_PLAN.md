# GLM-5.3-Flash REAP-224 Q23 on TR16 with two GPUs: the simulator-first path to 50 decode / 900 prefill tok/s

Status 2026-10-14 evening: simulator built, calibrated on the existing and the new instrumented runs, component probes
taken (section 8); no engine change yet.
Targets (user, 2026-10-14): 50+ tok/s single-stream decode and 900+ tok/s prefill at an 8K prompt on
REAP-224 Q23, with the coding gate unchanged (HumanEval >= 152/164 and the 35-track corpus gate). The
primary GPU orientation was left to the simulator. Nothing in this document is a quality claim.

Hardware: Threadripper 1950X (15 worker threads), 128 GB DDR4, RTX 5060 Ti 16 GB (CUDA, NUMA node 1,
PCIe 3 x8 = 7.2 GB/s), RX 9070 XT 16 GB (HIP gfx1201, NUMA node 0, PCIe 3 x16 = 14.3 GB/s). Measured
resources: `docs/benchmarks/tr16_mixed_20261014/resources.json`.

## 1. What the model is

`tools/sim/glm_runner.py` builds one verify step (and one 8K prefill) as a dependency graph on the
existing `tools/sim/heterogeneous.py` scheduler. Every kernel, transfer and host step is an event with a
measured or fitted cost; nothing is derived from FLOPS. The decode graph reproduces the runner's single
GPU stream: per MoE layer, group A's segment (wait-add of its previous layer, mixer, router/publish,
resident experts) then group B's, the CPU service thread serving slots in publication order, the worker
asked when the service thread reaches the slot and joined before the strict combine. Head (embedding and
dense layers per group), tail (norm, logits, readback, sampling), the three-step draft chain, the resync
and the remainder of the step are explicit. The STEP_TRACE scopes (head, cpu_service, between, tail) are
read off the schedule exactly as the runner defines them (`glm_decode.cpp` `StepTrace`), so calibration
compares scope by scope.

Files: `tools/sim/glm_runner.py` (model), `tools/sim/glm_traces.py` (log/trace readers),
`tools/glm_runner_sim.py` (CLI: calibrate, validate, decode, prefill, sweep, manifest, probe-plan),
`tools/sim/data/hw_tr16_runner.json` (resources, extends `hw_tr16_mixed.json`),
`tools/sim/data/glm_kernels_tr16.json` (constants with provenance; `free` lists the fitted ones),
`tools/sim/data/glm_runner_params_tr16.json` (fitted table), `tools/sim/data/glm_runner_records_tr16.json`
(measured rows), `tools/sim/data/glm_levers.json` (levers), tests `tools/test_glm_runner_sim.py`.

```bash
PYTHONPATH=tools python3 -m unittest tools.test_glm_runner_sim tools.test_glm_heterogeneous tools.test_sim
PYTHONPATH=tools python3 tools/glm_runner_sim.py calibrate --output tools/sim/data/glm_runner_params_tr16.json
PYTHONPATH=tools python3 tools/glm_runner_sim.py validate
PYTHONPATH=tools python3 tools/glm_runner_sim.py sweep --primary hip --worker cuda --target-decode 50 --target-prefill 1000
PYTHONPATH=tools python3 tools/glm_runner_sim.py probe-plan
```

## 2. Calibration (receipt: `docs/benchmarks/tr16_runner_sim_20261014/validation.json`)

Fit rows: the instrumented cuda-primary/hip-worker 8K holdout-code step trace
(`build-glm-release/mixed-redesign-trace-v2/`, STEP_TRACE=2, 149 verify steps, 3.44 accepted per step),
the HIP-alone 8K decode row, the three 8K prefill timelines (hip, cuda, cuda-hip) and the HIP 2K prefill
row. Held out: every other row of `docs/benchmarks/tr16_mixed_20261014/measurements.json`. Gate: holdouts
within 10% (decode) / 15% (prefill), traced scopes within 15%. Mean |log error| after the fit: 0.024.

| Row (REAP-224) | kind | split | measured | predicted | error |
|---|---|---|---:|---:|---:|
| cuda-primary + hip worker, 8K holdout code (trace) | decode | fit | 32.40 | 32.48 | +0.3% (scopes within 3%) |
| cuda alone, 8K holdout chat | decode | holdout | 31.14 | 31.12 | -0.1% |
| hip alone, 8K holdout chat | decode | fit | 35.01 | 35.05 | +0.1% |
| hip-primary + cuda worker, 8K holdout chat | decode | holdout | 33.72 | 34.60 | +2.6% |
| cuda alone, 2K holdout chat | decode | holdout | 31.75 | 29.79 | -6.2% |
| hip alone, 2K holdout chat | decode | holdout | 35.14 | 36.00 | +2.5% |
| hip-primary + cuda worker, 2K holdout chat | decode | holdout | 32.71 | 35.69 | +9.1% |
| hip alone 8K (timeline / measurement) | prefill | fit / holdout | 562.9 / 554.0 | 563.1 | +0.0% / +1.6% |
| cuda alone 8K (timeline / measurement) | prefill | fit / holdout | 421.6 / 392.5 | 422.7 | +0.3% / +7.7% |
| cuda-primary + hip worker 8K (timeline) | prefill | fit | 487.2 | 487.8 | +0.1% |
| hip-primary + cuda worker 8K | prefill | holdout | 523.0 | 528.1 | +1.0% |
| cuda alone 2K | prefill | holdout | 162.7 | 156.9 | -3.6% |
| hip alone 2K | prefill | fit | 240.4 | 279.0 | +16.1% (known miss) |

Outside the gate, reported only: the train-code decode rows (no acceptance counters; the same
configuration measured 33.7 on holdout chat and 31.1 on train code, a prompt effect the model does not
represent) and the hip-primary mixed 2K prefill row (measured 133.9, predicted 218: the mixed 2K path
carries an unmodelled per-layer cost of about 100 ms; 2K prefill is not on the target path).

Fitted constants of note (per slot of width 2 at 8K): CUDA primary segment about 0.80 ms (KDA) and
1.03 ms (MLA), HIP about 0.72 / 0.85 ms; CPU lane 65.6 GB/s effective; worker request 0.32 ms plus a
0.13 ms socket round trip; primary service-thread cost per worker request 0.05 ms; ownership refresh
7.0 ms every 75 requests; HIP prefill 4.3 ms fixed + 9.2 ms per 16-expert group at 8K (CUDA 4.8 + 12.0);
worker prefill acknowledgement 0.4 ms per 512-row chunk.

## 3. What the calibrated model says about today's runner

1. **Decode is bound by the primary GPU's per-slot chain, not by CPU bytes.** The traced step spends
   0.664 ms of host service and 0.258 ms of exposed wait per slot: a 0.92 ms period set by launch + mixer
   + router/publish + resident experts on the primary stream, while the CPU kernel itself takes 0.457 ms.
   This is why halving the CPU bytes with the worker (1.42 -> 0.62 GB per output) gained nothing and why
   the HIP card alone decodes 12% faster than the CUDA card alone. The per-slot nsys/rocprof
   decomposition (probe 1 below) must confirm the chain's composition before engine work.
2. **The fixed per-step work is 28.7 ms of a 106 ms step** (head 3.6, tail 4.5, draft chain 11.7,
   resync 1.5, other 7.4). At 3.44 accepted tokens per step, 50 tok/s is 68.8 ms per step.
3. **After the GPU chain is trimmed, the worker protocol binds**: its per-request path (socket round
   trip, 0.32 ms service, rows copy) is serialized per slot through the service thread.
4. **Prefill at 8K is compute-bound on the HIP card, not link-bound**: fitted 13.5 ms per 16-expert group
   (dequantize + GEMM) against about 5 ms at the measured 100 TFLOP/s GEMM microbenchmark, plus the
   5.94 s mixer chain. The expert streams are hidden behind the 4-slot staging ring except at 2K.

Standalone decode levers on the HIP-primary base (35.3 tok/s in the model): resident experts on a side
stream +12.0%, GPU draft bank +4.5%, fused publish/launch +4.4%, sampling/lookup trim +3.6%, head
ordering +2.5%, MLA cache kernel +1.6%; device combine -1.7%, direct dispatch +0.2%, asynchronous
ownership +0.1%, LFU partition -0.5%, three verify groups -11%, Maya-style layer pipeline with
NUMA-local 8-thread lanes -33% (the 42 GB/s lanes lengthen every CPU slot).

## 4. The ranked lossless path (receipts: `docs/benchmarks/tr16_runner_sim_20261014/sweep-*.json`)

Greedy forward selection over lossless levers, HIP primary (the CUDA-primary sweep selects
`hip_primary` first and then follows the same path). Each gain is conditional on the engine change
delivering the edited constant; the "required" column is what must be measured.

| # | Lever (engine knob) | decode | prefill | Required / check |
|---|---|---:|---:|---|
| 0 | base: HIP primary, CUDA worker, fraction 0.5 | 35.3 | 528 | - |
| 1 | resident experts on their own stream (overlapped with the next mixer) | 39.5 | 528 | nsys: resident kernels off the mixer stream; combine waits on both |
| 2 | F16 activation exchange, one partial per layer, no chunk acks (`STRATA_GLM_MIXED_EXCHANGE=f16`) | 39.5 | 600 | prefill logits hash equal (worker already gathers F16) |
| 3 | HIP expert FFN efficiency x2 (hipBLASLt / MMQ + SwiGLU epilogue, 1024-row sub-batches) | 39.5 | 673 | rocprof: group cost 13.5 -> <= 6.8 ms |
| 4 | HIP mixer chain x1.3 (sub-batch, hipBLASLt projections, WMMA attention) | 39.5 | 755 | rocprof mixer decomposition; logits hash |
| 5 | primary fraction 0.6 | 39.5 | 834 | prefill screen |
| 6 | layer-range split, two groups interleaved per stage, shared 15-thread lane (topology B) | 42.4 | 834 | host bounce latency; cross-vendor mixer rounding |
| 7 | fused router/publish/mhc (`STRATA_GLM_FUSE_ROUTE` and segment fusion) | 44.6 | 834 | nsys kernel count per slot |
| 8 | sampling/lookup/readback off the critical path (other 7.4 -> 4.0 ms) | 46.6 | 834 | SPECULATIVE remainder per step |
| 9 | head ordering (group A published before group B's dense layers) | 48.2 | 834 | STEP_TRACE head and layer-3 wait |
| 10 | skip streaming experts resident on either card | 48.2 | 861 | PREFILL_WORK transferred bytes |
| 11 | CPU expert access pattern 70 -> 82 GB/s (`STRATA_NATIVE_FLOW_SCHEDULE=span`) | 49.2 | 861 | kernel bench packed GB/s |
| 12 | MLA decode cache kernel x0.6 on the 8K term | 50.6 | 861 | nsys MLA slot at 2K vs 8K |
| 13 | GPU draft bank (`--mtp-experts=gpu`) | 50.7 | 861 | draft_ms per step; 5.8 slots per layer lost |
| 14 | primary fraction 0.67 (re-balance after 3-4) | 50.7 | 897 | prefill screen |

Without the layer split (worker topology only), the same levers plus a 2x faster worker request path
and asynchronous ownership reach **47.9 decode / 904 prefill**: the worker protocol's per-slot host cost
(request packing, reply wait) is the residual gap to 50. Allowing the lossy LFU partition adds 0.1 tok/s;
three verify groups lose 10%. The layer split's prediction is insensitive to the split layer (16..32:
50.4-50.9) and to the bounce latency (0.05-0.5 ms: 50.7-50.3); its pipelined (group after group) order
and NUMA-local lanes are both worse than the interleaved order with the shared lane.

**Prefill 1000 is not reached by the lever set as assumed (897-904).** Requirement grid at 8K
(HIP primary, F16 exchange, resident skip; mixer speedup x gemm speedup, per primary fraction):

| fraction | mixer x1.3 | x1.5 | x1.7 | x2.0 |
|---|---|---|---|---|
| 0.60 | 861 / 894 / 930 / 959 | 891 / 927 / 965 / 997 | 916 / 954 / 995 / 1028 | 946 / 986 / 1029 / 1065 |
| 0.67 | 897 / 927 / 959 / 993 | 932 / 964 / 998 / 1036 | 959 / 993 / 1030 / 1069 | 991 / 1028 / 1067 / 1110 |
| 0.75 | 935 / 961 / 988 / 1017 | 1005 / 1035 / 1067 / 1100 | 1064 / 1097 / 1133 / 1171 | 1105 / 1141 / 1179 / 1221 |

(cells: gemm speedup x2.0 / x2.4 / x3.0 / x4.0). 1000 needs the HIP mixer chain at least 1.5x faster
(5.94 -> 3.96 s) together with the expert FFN at least 2x faster and 0.75 of the groups on the HIP card,
or mixer 1.5x with FFN 3x at 0.67. Both are kernel-efficiency claims that only the rocprof decomposition
can confirm; the model shows the HIP card at 7.3 s of compute in a 9.1 s prefill, so the CUDA worker's
link (3.8 s of H2D) is not the limit at these splits.

## 5. Measurements to take before engine work (`tools/glm_runner_sim.py probe-plan`)

1. Per-slot GPU decode decomposition on both primaries (nsys / rocprof over 32 verify steps): kernel time
   of mixer, router/publish, residents and the wait-add gaps per slot. Confirms finding 3.1 and sets the
   constants for levers 1, 7 and 12.
2. Per-slot STEP_TRACE=2 traces for hip-cuda, cuda-hip, hip and cuda at 8K and 2K
   (`tools/glm_mixed_bench.py --capture-execution-trace`). Only a cuda-primary per-slot trace exists.
3. Worker round trip and request service histogram (`strata-glm-gpu-worker --loopback-bench`, a new
   diagnostic mode; `STRATA_GLM_MIXED_PROFILE_DECODE=1`).
4. NUMA-local lane bandwidth with 8 threads per node, alone and concurrently (prices topology B lanes).
5. Prefill per-kernel decomposition at 8K on both cards (rocprof / nsys) and a sub-batch screen.
6. Host bounce latency and shared-mapping registration (`cudaHostRegister` / `hipHostRegister` on one
   memfd) with `tools/glm_transfer_probe.cu` extended.
7. Draft step cost with CPU vs GPU draft experts.

Each probe writes a receipt under `build-glm-release/runner-sim/<date>/`; kept copies go to
`docs/benchmarks/tr16_runner_sim_<date>/`. Re-run `calibrate` and `validate` after each.

## 6. Engine order and qualification

Every lever is an opt-in knob with the default path byte-identical; CUDA and HIP are built for every
touched file; the quiet-run guard, 512 x 3 medians at 2K and 8K held-out, frozen before/after binaries
(`tools/glm_tr16_measure.py`) and the prediction-vs-measurement record per lever apply as in
`docs/GLM_TR16_MIXED_REDESIGN.md`. Lossless levers need bit-identical token streams and the last-prompt
logits hash; any lever that changes a stream runs the corpus gate and, before promotion, the full
HumanEval. The final profile is qualified on the same build and configuration as its speed numbers.

Decode, in sweep order: resident side stream (1), fused publish (7), sampling/lookup trim (8), head
ordering (9), CPU span access (11), MLA cache kernel (12), GPU draft bank (13); then the topology decision.
The layer split (6) is built only if, after 1 and 7-13 are measured and the model re-calibrated, the
worker-topology prediction stays below 50 while the split's prediction stays at or above 50 with the
measured bounce latency; otherwise the worker request path (fast request service, asynchronous
ownership, direct dispatch) is finished first.

Prefill: F16 exchange with one partial (2), HIP expert FFN efficiency (3), HIP mixer chain (4), fraction
re-balance (5, 14), resident skip (10). The required constants in section 4 are the acceptance criteria
for 3 and 4.

## 7. Limitations

- The decode model is calibrated on one per-slot trace (cuda primary). HIP constants are fitted to the
  HIP-alone 8K row only; the HIP-primary mixed rows validate within 2.6% (8K) and 9.1% (2K).
- Prompt-dependent acceptance is not modelled: the same configuration differs by up to 8% between the
  train-code and holdout-chat prompts.
- The topology-B graph assumes the two verify groups interleave per stage as they do today, bit-identical
  expert rows on both vendors (validated for q23) and unmodelled mixer rounding differences across vendors.
- Lever gains are conditional: the model prices what a knob would do if it delivers its constant; it
  does not know whether a kernel can be made that fast. Sections 4 and 5 name the measurement for each.
- The hip 2K prefill row misses by 16% and the mixed 2K prefill row by 63%; 2K prefill is reported, not
  targeted.
- No quality is predicted. HumanEval sits at the 152/164 floor on the current profile.

## 8. Probes taken 2026-10-14 (receipts: `docs/benchmarks/tr16_runner_sim_20261014/probes/`, driver `tools/glm_tr16_runner_probes.py`)

Correction first: the log reader originally paired the warm-up trial's STEP_TRACE with the timed trial's token
rate (each trial's SPECULATIVE line precedes its own STEP_TRACE). Trials are now grouped by their DECODE_TRIAL
marker; the timed cuda-primary trace is 160 steps at 98.8 ms (head 3.4, host service 55.5, exposed wait
20.8, tail 3.7, draft 8.8, resync 1.5, other 5.1 ms) with 3.20 accepted tokens per step. The per-slot
numbers in section 3 are unchanged; the fixed per-step work is 22.5 ms, not 28.7.

New instrumented traces (`tools/glm_mixed_bench.py --capture-execution-trace`, holdout-code prompt, 512
tokens, one timed trial each; the desktop was active on the 5060 Ti and the user on the machine):

| case | 8K decode | 8K prefill | 2K decode | 2K prefill |
|---|---:|---:|---:|---:|
| hip alone | 30.3 | 628 | 31.5 | 227 |
| hip primary + cuda worker | 27.5 / 30.7 (two runs) | 526 / 525 | 30.8 | 134 |
| cuda alone (tier 10240 MiB) | 28.8 | 429 | 31.3 | 157 |
| cuda primary + hip worker (tier 10240 MiB) | 31.1 | 499 | 29.9 | 187 |

Two identical hip-cuda 8K runs 20 minutes apart differ by 11%; the worker wait counters of today's
cuda-primary runs are 45% above the quiet original trace. Prompt dependence is acceptance: on hip alone
the holdout-chat prompt accepts 3.74 tokens per step and holdout-code 3.30 at the same 106-109 ms per
step. The 2026-10-14 matrix logs were therefore ingested with their own acceptance per round
(`records-from-bench`), which brings every 8K decode holdout within 5% except today's two cuda-primary
mixed runs (+17.5% / +24%: slower host service than the quiet trace). 2K prefill rows miss by 22-75%
(unmodelled per-layer cost in the mixed 2K path; hip-alone 2K +22-29%). `validate` reports
`calibrated: false` for these reasons; the 8K decode and 8K prefill rows that the targets are defined on
are within tolerance.

Component results:

- **Per-slot GPU decode decomposition** (`tools/glm_runner_profile_report.py`; rocprofv3 on HIP alone,
  nsys on CUDA alone, width 2, 8K context): kernel time per slot excluding the wait-add spin is
  0.68 ms on HIP (mixer 0.47, resident experts 0.10, router/publish 0.04, other 0.07) and 0.70 ms on
  CUDA (mixer 0.41, residents 0.11, publish 0.03, copies 0.10, other 0.05), with 59-65 launches and
  0.31-0.32 ms of inter-kernel gaps per slot. Both cards therefore spend about 1.0 ms of GPU chain per
  slot, as the model inferred; the dense projections (Q4_K mmvq / hipBLAS GEMMs run once per verify
  group) are the largest part, not the resident experts. This adds the `shared_window_projections` lever
  (one projection pass per window) and lowers the expected gain of the resident side stream.
- **Worker loopback** (`tools/glm_mixed_worker_loopback.py`, real protocol, no decoder): ownership no-op
  round trip 42 us (CUDA) / 40 us (HIP); decode partial requests at width 2: CUDA 105 us for 1 route to
  500 us for 16, HIP 157 to 396 us. Fitted: CUDA 0.04 ms fixed + 0.026 ms per route, HIP 0.10 + 0.016.
  The model's earlier 0.6 ms worker path was 2x too high; in-run requests (MIXED_WORKER elapsed
  0.41-0.47 ms p50) carry 7-8 routes plus contention.
- **Host bounce and shared mapping** (`tools/glm_transfer_probe.cu` bounce/register modes): one D2H plus
  one H2D of 128 KB takes 47 us on either card (1 MB: 303 us CUDA, 177 us HIP). Both vendors register and
  DMA into the same memfd mapping, alone and concurrently (6.95 and 13.9 GB/s): the zero-copy exchange
  for levers A1/P1 is feasible.
- **CPU lanes**: interleaved 15 threads 75.7 / 75.2 GB/s (width 1 / 2) on the synthetic flow bench
  against 65.6 GB/s effective in-run. Node-local 8-thread lanes could not be measured: the flow bench
  refuses a single-node membind (node-owned rows); 42 GB/s per node remains the plain-read estimate.
- **Draft step**: HIP alone, 128 tokens: draft chain 12.2 ms per step with CPU draft experts.
  `--mtp-experts=gpu` is refused together with `--decode-cache-mib`, so the GPU draft bank needs an
  engine change before it can be measured.
- **Ownership refresh** costs 13.4 ms per update with the CUDA worker (x8 link) against 7.0 ms with the
  HIP worker; the table now carries it per worker vendor.

Re-fitted sweep (lossless, 44 records, `sweep-*.json`): HIP primary base 28.3 decode / 526 prefill ->
49.5 / 899 with the interleaved layer split, 42.0 / 929 without it (worker topology plus a 2x faster
worker request path); CUDA primary base 34.9 / 485 -> 49.5 / 899 via `hip_primary`. The first decode
levers in both sweeps are now `shared_window_projections` (+15%) and `residents_side_stream` (+12%),
then the layer split (+15%), head ordering, CPU span access, fused publish, MLA cache kernel and the
sampling/lookup trim. Prefill levers and their order are unchanged; 1000 still needs section 4's
requirement grid. The base numbers moved with today's slower, non-quiet rows: the next calibration must
use quiet runs only (no desktop activity, nothing else on the CPU), which the bench driver does not yet
enforce for CPU load.

## 9. Implementation and validation, 2026-10-15

The current target is 50 decode / 900 prefill tok/s at 8K, separately on code and chat holdouts,
with one configuration and the existing 152/164 HumanEval floor. Section 4's 1000-token requirement grid
is historical modelling, not a measured result. No commits or pushes have been made.

The working tree now has these opt-in paths:

- Protocol v4 remains the default: 143,413,600 bytes and 512 output rows. `STRATA_GLM_MIXED_PROTOCOL=5`
  selects 269,275,496 bytes, 8192 output rows and the F16 activation flag. Both binaries expose
  `--protocol-layout`; layout and validation rules are in `strata_host`. The launcher checks the two
  v5 layouts before starting inference, and the loopback tool reads its offsets from the query.
- `STRATA_GLM_MIXED_REGISTER=1` registers the shared mapping. `STRATA_GLM_MIXED_EXCHANGE=f16` requires
  v5 and uses F16 activations when registration succeeds. Registered replies retain their lease until
  DMA finishes. Failed registration retains staging; no F16 flag is sent with an F32 activation.
- `STRATA_GLM_RESIDENT_STREAM=1` needs split verification, device combine and no deferral. Resident
  work replays its own graph outside the mixer graph. Streams, events and registered mappings have
  cleanup; reset, compaction and snapshots drain resident readers before reusing their buffers.
- `STRATA_GLM_FUSE_ROUTE_BATCH=1` combines routing and publication for widths 1..8, including unbiased
  learning routes. It preserves ties and route-weight arithmetic and refuses the coupled route-window margin.
- `STRATA_GLM_MIXED_DEVICE_DISPATCH=1` builds worker routes on the GPU and, when registered, transfers
  activations and replies directly. It uses an ordered device partial sum and refuses MIXED_FUSED_RETURN.
- `STRATA_GLM_PREFILL_FUSED_SWIGLU=1` combines the FP32 SwiGLU calculation and F16 conversion.
  `STRATA_GLM_HIP_BATCH_SINGLE=1` uses individual F16 products with an explicitly supplied hipBLASLt
  tuning table. `STRATA_GLM_MTP_GPU_CACHE=1` permits the GPU draft bank alongside the target decode cache;
  existing allocation and headroom checks still apply. These paths have no qualified whole-model gain yet.

The head already publishes group A before group B's dense head in the current decoder. No new head-order
change has been made. Projection sharing still needs evidence that it preserves overlap.

Validation so far:

- CUDA and HIP decoder/worker builds pass. All 22 CUDA and 21 HIP GLM-labelled tests pass. Batched
  router publication is checked against separate kernels, including ties, unbiased routes and graph replay.
  Fused SwiGLU/F16 is checked bitwise against the separate operations, with output guards.
- 140 simulator and tool unit tests pass. Validation can use `--context-filter 8192`; excluded 2K rows
  remain visible, and an 8K failure still fails the selected gate. The existing model is not yet requalified.
- Worker rows at widths 1, 2, 4 and 8 match the frozen worker on both vendors, with nonzero activations,
  changed clip/skip limits and exact ordered partial sums. Registered v5 and GPU dispatch pass these checks.
- A 512-token prompt / 32-token greedy default-path sample has equal CUDA tokens and prefill/final logits
  before and after, with new knobs unset. HIP does not establish bitwise preservation: two runs of the
  identical frozen v1 binary differ in tokens and have 4.08% relative L2 difference in prefill logits.
  The cause is unresolved. This is a plain 512-token prefill profile, not the 8K optimized speed profile.
- SYCL is not built: the local Intel SDK is absent. No SYCL execution claim is made.

Before resident graph replay was enabled, three timed 512-token trials on the 8K code holdout measured
14.20 decode / 509.28 prefill tok/s with HIP primary and 30.04 / 473.79 with CUDA primary. Registered
exchange alone on the HIP-primary parent measured 29.98 / 533.21. These were on the shared TR16 desktop,
are not quality-qualified and must not be treated as a gain over earlier single-trial results.

The installed gfx1201 hipBLASLt reports version 100401; the bundled gfx1201 100500 table does not match.
`tune_hipblaslt` produced a local, version-matched table for 62 expert and dense shapes, all passing its
accuracy and padding checks. The single F16 T=256,N=4096,K=4096 probe measured 0.0824 ms. This is a
component result, not full FFN speed or whole-model quality.

`tools/glm_tr16_optimize.py` freezes CUDA/HIP binaries and runs isolated variants, both orientations and
separate holdouts. It refuses existing output roots and retains failed attempts. Speed flags require at
least 512 generated tokens, three trials and an actual 8192-token prompt; they never qualify quality.
The revised resident/register/F16 screens are in flight under
`build-glm-release/runner-sim/opt-codex-20261015-screen/`. Earlier checks and tuning receipts are under
`build-glm-release/runner-sim/opt-codex-20261014/`.

The quality driver has an explicit `--quality-gate tr16-35` (PPL +6%, KL +0.08, top-1 -3 pp), configured
prefill batch and `--eval-prefill` coverage. Existing regression tolerances remain unchanged. Full corpus
and HumanEval qualification, 8K code/chat target measurements, the final fraction sweep and the conditional
cross-vendor layer split remain pending. Neither speed target has been demonstrated.


The first revised screen root was rejected by the swap guard (launcher VmSwap); no speed data from those
attempts is usable. It was stopped and retained. The restarted screen is under a task-only systemd user
scope with `MemorySwapMax=0`, at `build-glm-release/runner-sim/opt-codex-20261015-noswap/`, driver log
`/tmp/strata-v4-noswap-screen.log`. Global swap was not changed. The screen tool now requires an inherited
no-swap cgroup for future launches.

## 10. Decode screen and layer-split forecast, 2026-10-15

Screens: `build-glm-release/runner-sim/screen-20261015/` (runner `run.sh`, configs `*.json`), HIP primary
(RX 9070 XT) with CUDA worker, REAP-224, 8K holdout code, 512 tokens, one warm-up and one timed trial,
`STRATA_GLM_STEP_TRACE=2` unless noted, systemd scope with `MemorySwapMax=0`, desktop active. Single trials
vary by about +/-1.5 tok/s, so differences under 3 tok/s are not established.

| variant (on parent = profile + `BATCHED_RESIDENT=1`) | decode tok/s | note |
|---|---:|---|
| parent | 28.9 | host service 74 ms of a 114 ms step |
| `RESIDENT_STREAM=1` | 14.2 | exposed waits ~130 ms/step on HIP; reject |
| `FUSE_ROUTE_BATCH=1` | 29.2 | neutral |
| `MIXED_DEVICE_DISPATCH=1` + `MIXED_REGISTER=1` | 18.9 | worker registration fails (invalid argument); exposed waits grow |
| `MTP_GPU_CACHE=1` + `--mtp-experts=gpu` | refused | "decode-cache-mib requires ... CPU-draft MTP" |
| threads 14 / 13 / 12 | 29.0 / 26.4 / 28.9 | no core-contention effect |
| `MIXED_ADAPT_INTERVAL=512` | 30.6 | |
| `MIXED_ADAPT_CACHE=0` (no adaptive refresh) | 31.2 | worker + ownership waits halve; residency change, needs the gate |
| no adapt + `NATIVE_TASKS_PER_THREAD=4` | 32.6 | lossless kernel setting |
| no adapt + `Q23_PREFETCH=256` / `1024` | 33.0 / 33.2 | lossless; within noise of each other |
| strict combine (no trace) | 29.8 | |
| strict + `MOE_MERGE_GROUPS=1` (needs adapt and unbiased tier learning off) | 25.8 | control without merge 32.2; reject |

Per-step decomposition of the parent (STEP_TRACE / MIXED_CLIENT / CPU_EXPERT): host service 70-74 ms,
of which the CPU expert flow is about 53 ms (in-run ~47 GB/s against 75 GB/s synthetic, on ~30 MB per
slot), worker waits 8-17 ms and blocking ownership refreshes 6-11 ms; exposed GPU waits 14-16 ms; head,
tail, draft and the remainder about 23 ms. Per slot the CPU lane is busy and the GPU wait is near zero,
so decode is CPU-lane bound in the mixed runner; the primary's own GPU chain (~0.7 ms of kernels per
slot, 59 ms per step) binds next.

Calibrated forecast (`tools/glm_runner_sim.py`, today's shares; parent predicted 29.2, measured 28.9):
the interleaved layer split alone gives about 32 tok/s (+10%). The full lossless lever stack reaches
**44.4 with the layer split and 40.5 without** (`build-glm-release/runner-sim/sweeps-20261015/`); 900
prefill is reached at the end of the stack (899). Several simulator levers did not hold up when measured
(shared projections, resident side stream), while batched residents measured +20% against +3% predicted,
so lever-by-lever forecasts carry large error. Neither measurement nor forecast shows a lossless path to
50 decode tok/s on this machine; the layer split is a multi-week cross-process change worth about
4 tok/s in the forecast.

## 11. Layer split implemented (sequential first version), 2026-10-15

Opt-in, uncommitted. `STRATA_GLM_LAYER_RANGE=first:end` makes a decoder own only layers [first, end):
state, dense weights (preload filter), NUMA expert placement, tier sources and decode-cache candidates
are limited to those layers. The primary (`0:K`) keeps embedding, head and MTP and, after its last layer,
ships the residual streams of the verify window (both split-verify groups) to a stage process (`K:45`)
over a shared memfd (`include/strata/host/glm_stage.h`, validation in `src/host/glm_stage.c`, test
`glm_stage_test`; transport `include/strata/core/glm_stage.hpp`). The stage (`STRATA_GLM_STAGE_SERVE=1`)
serves BATCH/VERIFY with residual in and out, and mirrors restore_verified, verify-history enable and
invalidate, checkpoint/save/restore, snapshot/restore (as a device checkpoint), reset, decode-cache
preparation and freeze. Launcher: `tools/glm_stage_run.py` (HIP primary, CUDA stage, `--boundary`,
`--primary-cpus/--stage-cpus` taskset partitions, `--pool-spin-us`). `STRATA_GLM_STAGE_TRACE=1` prints
per-round primary time, stage round trip and stage batch time. Default runs are unchanged (the mixed
control on the rebuilt binaries measured 36.0 tok/s, 8K holdout code, quiet night run).

Measured (HIP primary layers 0-23, CUDA stage 24-44, REAP-224, 8K holdout code, 512 tokens):

| setup | decode tok/s | note |
|---|---:|---|
| shared cores, pool spin 1 ms | 17.6 | 20-28 ms scheduler stalls per round from pinned threads of both processes on one core set |
| shared cores, spin 20 ms | 15.0 | |
| disjoint core sets (4+4 physical cores per process, 7 threads), spin 20 ms | 14.6 / 12.5 / 15.4 | median verify round 111 ms; a few multi-second rounds (post-prefill NUMA page relocation) |

Per round the primary's half (layers 0-23, both groups interleaved, plus head) takes ~60 ms and the stage's
half ~45 ms; the exchange itself costs 0.1 ms. The halves run one after the other, so a verify round is
~105-111 ms against ~85-90 ms in the single-process mixed runner, and decode is lower even before stalls.

Cross-stage overlap does not recover this. Within each process the two verify groups already overlap GPU
and CPU work; pipelining groups across processes gives that up (a group alone pays GPU then CPU serially
per slot), and halving a group barely halves its time because the GPU cost per slot (~0.7 ms of kernels,
~60 launches) does not scale with tokens. Estimates with the measured halves: A-then-B pipelining ~110 ms,
4 one-token groups ~116 ms, both no better than the sequential 105 ms. The layer split therefore does not
offer a path to 50 tok/s on this machine; the best measured decode remains the mixed runner (36.0 tok/s
quiet, no adaptive refresh, quality gate pending).

## 12. Fast-iteration decode push, 2026-10-15 (1K prompt / 256 tokens, 3 timed trials; quality deferred)

| step (cumulative) | decode tok/s (mean) |
|---|---:|
| mixed, no adaptive refresh, HIP primary | 35.8 |
| + routing affinity 0.30 (lossy) | 38.8 |
| + GPU draft bank alongside the decode cache (fixed refusal and early MTP preparation) | 41.9 |
| CUDA primary + HIP worker | 44.0 |
| + `STRATA_GLM_SPLIT_JOINT_DENSE=1` (dense prefix once per window) | 45.1 |
| + soft acceptance top-k 8, margin 6 (lossy) | 48.1 |
| + `STRATA_GLM_TIER_LOOKUP_BATCH=1` | 48.5 |
| + worker `STRATA_GLM_MIXED_SCAN_ONCE=1` and `MIXED_FUSED_RETURN=1` | **49.0** (47.7-50.5) |
| same at 8K prompt / 512 tokens | **46.1** (44.2-47.7); 8K prefill 510 |

Not yet quality-qualified; the 8K warm-up of the last configuration looped (target-model loop under lossy
routing; the repeat guard did not trigger). The CUDA primary now waits on the CPU+worker lane for ~55% of
each slot; the worker body (~0.27 ms/request) and CPU per-slot overhead are the binding costs.

## 13. Prefill push on two GPUs, 2026-10-15 afternoon

Target 900 tok/s at 8K (holdout code). Screens: `build-glm-release/runner-sim/screen-20261015/` (`run2.sh` adds
`FRACTION`), HIP primary (RX 9070 XT) + CUDA worker, 8 generated tokens, two or three prefill trials per process.
Quality is not checked; the lossy switches are marked.

Diagnosis, in order found:

- HIP alone, 8K: 611 tok/s. Per-layer experts are 2.3 GB; on KDA layers the 14.3 GB/s link is exposed (FFN 190 ms
  against 97 ms after an MLA layer, where the long mixer hides the upload).
- Mixed (fraction 0.6, protocol 4): 506-520 tok/s. The worker waited about 58 ms per layer for its request, because
  the request was sent only after the primary's own staging joined, and then sat behind its own groups.
- The CUDA worker never registered the v5 mapping (`invalid argument`): the launcher backed the "memfd" with a
  disk `tempfile`, and CUDA refuses file-backed shared pages (HIP accepts them). Fixed: `tools/glm_mixed_run.py`
  uses `memfd_create` when `STRATA_GLM_MIXED_REGISTER=1`; registration also rounds to whole pages.
- rocprofv3 of the primary: expert F16 GEMMs at about 30 TFLOP/s effective, FP32 tiny-tile GEMMs in the MLA
  absorb/index products, `scatter_expert_weighted` atomics (about 9 G atomics per prefill at 7.5 G/s; the build
  already emits hardware `global_atomic_add_f32`), and expert uploads that arrive group by group with 11-14 ms of
  idle GPU before each: the host staging memcpy runs at about 9.7 GB/s (68 GB, 7.0 s per prefill), below the link.
  More or node-local staging threads did not raise it; node-0 memory traffic (memcpy write + DMA read) is the wall.

Path (prefill tok/s, later trials of a process; each row adds to the previous):

| change | tok/s |
|---|---:|
| mixed v4, fraction 0.6 | 506-520 |
| protocol 5 + registered mapping + F16 activations (registration fix) | 595 |
| `STRATA_GLM_PREFETCH_STREAM=1` (existing) | 596-604 |
| fraction 0.79 (11 of 14 groups on HIP) | 717-719 |
| `STRATA_GLM_MLA_ABSORB_F16=1` (existing, lossy) | 733 |
| `STRATA_GLM_MLA_TILE=256` (new; 64 default) + scratch cap 4096 MiB | 748-750 |
| `STRATA_GLM_PREFILL_ORDERED_SCATTER=1` (new: one launch per expert, plain float4 adds in stream order; the worker partial is added on the main stream afterwards) | 777 |
| `STRATA_GLM_PREFETCH_DIRECT=1` (new: DMA straight from the registered model pages, no staging) | 811 |
| fraction 0.86 | 826 |
| `STRATA_GLM_MLA_INDEX_F16=1` (existing, lossy) | 837 |
| `STRATA_GLM_MIXED_HALF_PARTIAL=1` (new, lossy: the worker returns its partial as FP16) | 852 |
| fraction 0.79 | 857-862 |
| `STRATA_GLM_PREFETCH_EARLY=1` (new: next layer's uploads start when this layer's MoE is queued) | **870-875** |

Best: `pm-best.json` with `FRACTION=0.79`, case hip-cuda: 9392 / 9408 / 9416 ms = **872 / 871 / 870 tok/s**.
Also new and neutral: `STRATA_GLM_MIXED_EARLY_SUBMIT=1` with `EARLY_ROUTE_SYNC=1` (worker inputs leave before the
shared expert), `STRATA_GLM_STAGE_NUMA_SPLIT=1` (staging threads on both nodes, each copying its node's half),
`STRATA_GLM_ROUTER_WHOLE=1` (one router product), HIP direct-upload registration without the CUDA-only attribute
check. No effect or worse: MMQ experts (578 vs 611 alone), 6/8 staging threads, the tuned hipBLASLt table with
`HIP_BATCH_SINGLE` (798 vs 826), fused SwiGLU, KDA row parts 8, bucket step 32/128, fraction 0.64-0.71 and 0.86-0.93
at the end. `MLA_F16_OUT` is CUDA-only.

Caveats:
- `PREFETCH_DIRECT` pins the uploaded layers' expert pages (page cache) for the process lifetime: up to the
  primary's share of the 86 GB of experts. Registration costs about 78 ms per 1.7 GB the first time.
- The first prefill of a process is sometimes 10-25% slower (one-time conversions and registration).
- This is the HIP-primary orientation. The best decode (46-49 tok/s) is CUDA-primary (cuda-hip); the HIP-primary
  decode with these settings is about 37 tok/s on 32-token samples. One configuration for both targets is open.
- Trace of the best config (synced, per KDA layer): mixer 78 ms, routing 20, own experts 65, worker after own 5-8.
  The mixers are now half of the prefill (4.7 s): MLA layers about 200 ms each, KDA about 75 ms.

Remaining 3% to 900: overlap the worker's activation upload with routing (protocol message before routing), split
the worker's partial return by token halves so its D2H overlaps compute, or move part of the MLA attention
(heads) to the CUDA card. Builds pass and `ctest -L glm` passes on both (23 CUDA, 22 HIP).

Fraction calibration for other GPU pairs: `tools/glm_mixed_fraction_calibrate.py --config C --case hip-cuda
--groups 14 --model M --prior P --prompt 8k.ids --output DIR [--objective prefill|decode]
[--primary-tok-s X --worker-tok-s Y]` hill-climbs over whole 16-expert groups (one fresh bench process per
candidate) and writes `DIR/fraction.json` with every measurement and the best `STRATA_GLM_MIXED_PRIMARY_FRACTION`.

## 14. Release phase: decode under the quality gate, 2026-10-15 evening

Rule for the release (user): no experimental code, no quality degradation beyond the agreed gate (HumanEval
>= 152/164, the 35-track corpus gate `tr16-35`, no loops at 8K). Measured first: with every lossy setting
removed from the 49 tok/s stack (strict acceptance, affinity 0, adaptive refresh on, no fused return) decode
is 20 tok/s at 1K, so a bit-exact release cannot reach 50; the release keeps the lossy settings that pass the
gate.

Corpus gate (`tools/glm_tr16_quality.py --quality-gate tr16-35`, CUDA primary + HIP worker, p4-both settings,
baseline = single CUDA unbiased strict on the same REAP-224 weights; receipts
`build-glm-release/release-decode/corpus-aff*/result.json`):

| routing affinity | KL (limit 0.253) | top-1 (limit 0.839) | perplexity (limit 5.213) | gate |
|---|---:|---:|---:|---|
| 0.30 (the 49 tok/s profile) | 0.310 | 0.817 | 5.488 | fail |
| 0.20 | 0.244 | 0.8386 | 5.245 | fail (top-1, perplexity, both just outside) |
| 0.14 | 0.210 | 0.853 | 5.111 | pass |
| 0.08 | 0.197 | 0.857 | 5.056 | pass |

Soft acceptance (top-k 8, margin 6, max soft 3), no adaptive refresh and the fused return stay in all rows.
The release value is affinity 0.14: it passes with margin and nothing is tuned to the edge of the gate.

Speed at affinity 0.14 (`screen-20261015/rel-aff0.14*.json`, cuda-hip, 3 timed trials): **42.0 tok/s at 1K/256**
(40.7 / 42.5 / 42.3) and **40.0 at 8K/512** (39.5 / 41.5 / 38.8), 8K prefill 465 in that orientation. The 8K
streams are clean (distinct 4-grams 0.97-0.98, no cycles, repeat guard idle). Against the targets: 50 is not
reached under the gate; the gate costs about 7 tok/s of the 49.

HumanEval 164 on the same binaries and settings: `build-glm-release/release-decode/humaneval-aff0.14.sh`
(result pending at the time of writing; see the end of this section).

Code removed for the release (defaults were already byte-identical; the removed paths measured no gain or were
dead ends, see sections 9-11): the layer split (`STRATA_GLM_LAYER_RANGE`, the stage protocol and process,
`tools/glm_stage_run.py`), `STRATA_GLM_RESIDENT_STREAM`, `STRATA_GLM_MIXED_DEVICE_DISPATCH`,
`STRATA_GLM_FUSE_ROUTE_BATCH` (the batched router kernel and its parity test stay),
`STRATA_GLM_PREFILL_FUSED_SWIGLU` (the kernel and its precision test stay), `STRATA_GLM_HIP_BATCH_SINGLE`,
`STRATA_GLM_MIXED_FIXED_GRAPH_GEOMETRY`, `STRATA_GLM_MIXED_SPIN_US`, `STRATA_GLM_ROUTER_WHOLE`,
`STRATA_GLM_STAGE_NUMA_SPLIT`. One serve-path fix: `STRATA_GLM_MTP_GPU_CACHE=1` now keeps the GPU draft bank
beside the decode cache in the server too, so HumanEval runs the configuration that was timed.

Release configurations: decode `screen-20261015/rel-aff0.14.json` (CUDA primary, HIP worker, decode cache
10240, GPU draft bank); prefill `screen-20261015/pm-best.json` with fraction 0.79 (HIP primary, 870 tok/s,
section 13). They are different orientations; one process cannot run both at these speeds.
