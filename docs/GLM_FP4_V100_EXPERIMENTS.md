# Opt-in V100 FP4 experiments

## Current round: ordered KDA prefill

Prepared inputs with four ordered KDA row partitions measured **32.10 decode
tok/s and 520.89 prefill tok/s** on two V100 PCIe 32 GB cards, a dual Xeon Gold
6240 host and a 181 GiB allocation. The same-binary control measured
31.87 / 454.22. Prefill improved **14.68%**; the small decode difference is not
claimed as an improvement with unlocked clocks. The 50 decode and 800–1000
prefill targets remain unmet.

| Configuration, same binary | Decode tok/s, median | Prefill tok/s, median |
| --- | ---: | ---: |
| GPU reduction control | 31.87 | 454.22 |
| Four ordered KDA row partitions | 31.89 | 490.58 |
| Ordered partitions and prepared inputs | 32.10 | 520.89 |

Each configuration used two complete warmups, a discarded prefill request and
three timed 8,192-input / 512-output requests. All timed requests had zero major
faults and zero swap, exited cleanly and kept the frozen fast profile's 512 token
IDs. All three configurations passed the same eight reused training tasks with
identical generated IDs. The newly frozen twenty holdouts remain unscored while
other candidates are investigated; this is an interim selection, with no new
held-out quality claim.

Use the opt-in
[`glm53f-fp4-dual-v100-kda.json`](../configs/experimental/glm53f-fp4-dual-v100-kda.json)
preset. It adds `STRATA_GLM_PREFILL_KDA_ROW_PARTS=4`,
`STRATA_GLM_KDA_ORDERED_ROWS=1` and `STRATA_GLM_KDA_PREPARE=1` to the GPU
reduction preset. The route-drop and MTP bounds below are unchanged. All prefill
routes and checkpoint weights are retained. The original reference, fast and
GPU reduction presets remain unchanged.

[Compact evidence](glm_fp4_v100_kda_measurement.json) records source and binary
hashes, all three timings, expert work, transfer counts and training checks. The
first timed request still creates 333 CPU scale-cache entries; the last two
create none. The reported median includes the slower first request.

The separate simulator profile
[`glm_fp4_v100_kda.json`](../tools/sim/data/glm_fp4_v100_kda.json) uses these
measurements without fitting timing constants. It reports **1.56×** more decode
throughput needed for 50, **1.54×** more prefill throughput for 800 and **1.92×**
for 1000. Its loose streaming and matrix-DMA bounds remain 173.4 and 1436.7 tok/s;
they omit arithmetic, reconstruction, scheduling and dependencies. Removing all
non-verification work would reach only 37.12 decode tok/s. The earlier activity
capture measured the control configuration and is not reused as this preset's
kernel-duration bound.

## Earlier GPU reduction qualification

GPU reduction alone measured **32.67 decode tok/s and 454.29 prefill tok/s** on
two V100 PCIe 32 GB cards with the 181 GiB allocation. The same-build control
measured 32.43 / 395.97: prefill improved **14.73%**, while the 0.74% decode
change is too small to claim a general improvement with unlocked clocks.
The requested 50 decode and 800–1000 prefill targets remain unmet.

| Configuration | Decode tok/s, median | Prefill tok/s, median |
| --- | ---: | ---: |
| Frozen fast binary, same-day rerun | 31.06 | 389.89 |
| Final build, original fast settings | 32.43 | 395.97 |
| Final build + GPU reduction, selected | 32.67 | 454.29 |
| Final build + input reuse + GPU reduction | 32.22 | 454.84 |

All three full qualification configurations used two complete warmups and three
timed 512-output requests. Every timed request had zero major faults and zero
observed swap. Every output matched the frozen fast profile's 512 IDs, SHA256
`6b12286983b9dfbb205cb8f1cbdb4b4140dc0f7d4a3366e91dd9962239ed72d8`.
The original approximately 5.26 tok/s reference is preserved separately.
Input reuse did not survive full qualification and is not enabled in the
selected profile. None of the rejected experiments changes a default.

Use the opt-in
[`glm53f-fp4-dual-v100-gpu-reduce.json`](../configs/experimental/glm53f-fp4-dual-v100-gpu-reduce.json)
profile with the sealed RAM-cache setup in [GLM_FP4_V100.md](GLM_FP4_V100.md).
It adds only `STRATA_GLM_FP4_GPU_REDUCE=1` to the existing fast settings.
Route dropping remains bounded at 0.5 original routing mass / seven routes,
with renormalization; MTP remains depth three, margin three, top eight, at most
three soft accepts and repeat guard four. Prefill retains every route, and the
checkpoint is unchanged. These lossy bounds are not output-error bounds.

The final decoder SHA256 is
`252e052c64c1c04757c42212eba946db0e6149ebef6626ecaa7201302a77b172`.
The 475-file engine source manifest matched the local reviewed source.
Fresh training passed 8/8 for the baseline and selected policy; the candidate
was frozen before the fresh held-out comparison. Both passed **17/20** fresh
holdouts, with identical generated token IDs for all twenty cases. Both failed
`popcount64`, `reverse_bits64` and `median_three`. There was no held-out tuning.
These small C++ coding checks do not establish general quality or perplexity.

## Experimental switches

These switches extend the native FP4 executor. They are off by default and do not
change the frozen reference or fast profiles. They target two V100 PCIe 32 GB
GPUs, a dual Xeon Gold 6240 host and a 181 GiB allocation. Performance and quality
must be checked on the complete model before selecting a combination.

| Switch | Values | Work changed |
| --- | --- | --- |
| `STRATA_GLM_FP4_CPU_INPUT_REUSE` | `1` | Quantize each input token once and gather its Q8 row for routed experts. |
| `STRATA_GLM_FP4_CPU_Q8_PREPARE` | `1` | Prepare Q8 magnitudes and sign masks once per activation block. |
| `STRATA_GLM_FP4_CPU_ROW_TILE` | `1` (default), `2`, `4` | Reuse activation loads across CPU output rows. |
| `STRATA_GLM_FP4_CPU_SCALE_QUARTETS` | `1` | Cache four decoded scale indices per Q8 block; charge the existing CPU scale-cache limit. |
| `STRATA_GLM_FP4_CPU_TEAM_SIZE` | `0` (default), `2`, `4`, `8` | Divide CPU workers into independent expert teams. |
| `STRATA_GLM_FP4_PREFILL_QUEUE` | `1` (default), `2`, `4` | Retain prefetched experts until their GPU consumption event completes. |
| `STRATA_GLM_FP4_DENSE_SPLIT` | `1` | Increase independent K partitions for dense matrices with 513–8192 rows and at least 4096 columns. |
| `STRATA_GLM_FP4_GPU_REDUCE` | `1` | Sum routed GPU contributions on the primary GPU; use peer copies when available. |
| `STRATA_GLM_FP4_PREFILL_BATCH_EXPERTS` | `1` (default), `2`, `4` | Group compatible prefill expert projections into batched cuBLAS calls. |

CPU tiling preserves each row's dot-product and reduction order. Prepared Q8
operands and quartet caches preserve the existing W4A8 representation; they do
not introduce another quantization step. Small teams retain expert reduction
order. The original Q8 kernel remains the default. Scalar and AVX2 fallbacks
remain available.

Prefill queues require device routing, asynchronous uploads and the GPU arena.
Admission bounds the extra working set to 256 MiB per GPU. Grouped prefill also
requires each expert to have more than eight rows, limits padding to 25%, and
falls back to individual expert execution when those conditions do not hold.
It retains every prefill route and scatters contributions in expert order.
Admission uses each group’s row count; two pairs or one quartet may remain
in flight within the same bound. Grouping may change cuBLAS arithmetic and therefore requires model-level quality
checks. Reconstruction and GEMM still share one compute stream; this experiment
does not overlap those two stages.

Dense splitting keeps the same partitioning across verification widths 1–8,
but changes the reduction tree relative to the original kernel. It therefore
needs numerical and model quality checks. The default retains the original
partitioning, including its existing small-matrix splits.

GPU reduction uses ordered additions. When peer access is unavailable, secondary
results pass through pinned host memory. Decode still needs CPU activation data
for nonresident experts. Unsupported route shapes use the existing host path.

`STRATA_GLM_FP4_PROFILE=1` adds `FP4_GPU_INTERVAL` diagnostics for reconstruction,
GEMM, skinny products, matrix copies and scale preparation. Event intervals can
include host enqueue gaps; intervals on separate streams overlap. A report is
flushed at a routed layer and can include preceding dense work. Do not sum these
intervals into request latency or interpret `input_wait_ms` as pure copy time.

`FP4_TRANSFER` reports cumulative submitted matrix H2D bytes, split into startup,
request preparation/prefill and decode. Counts include compressed scale metadata
and staging padding, but exclude activation transfers, D2H traffic and expanded
GPU allocations. Subtract counters bracketing one request before using them in
the simulator. `FP4_BATCH` reports actual grouped coverage and padding.

The standalone `strata-glm-fp4-cpu-bench` target rotates 64 MiB of synthetic FP4
codes and checks bitwise parity before timing. Its metadata stays hot, and its
timings exclude quantization and full-model scheduling. Use it to investigate
kernels, not to predict end-to-end tokens per second.

Fresh coding fixtures are in `tools/glm_fp4_optimization_quality.py`: eight
training cases and twenty held-out cases with executable oracles. Freeze the
fixtures before experimentation and freeze the selected policy before running
the held-out cases. These checks are not a general quality or perplexity result.

## Measurement protocol

The October 19 campaign uses the same frozen 8,192 input token IDs as the earlier
reference. Full qualification generates 512 IDs (511 timed decode advances),
with two complete warmup requests, one discarded one-token request, then three
timed requests in one process. The discarded request is printed as `TRIAL 0`;
it is not a decode measurement. Short screens use one complete warmup and one
128-output timed request. Their decode rates cannot be compared directly with
the 512-output qualification.

All qualified timed requests must have zero major page faults and zero observed
swap, return the requested number of IDs without an EOS token, and exit cleanly.
Executable/library prefaulting happens outside the timed requests. The original
fast binary's first timed request still constructed 333 CPU scale-cache entries;
the following two constructed none. Report all three timings and use the median,
without treating that first request as fully warmed metadata.

The node has two V100 PCIe 32 GB cards, two Xeon Gold 6240 CPUs, 32 model workers,
CUDA 12.9 and GCC 12.2.1. The job requests 181 GiB RAM and uses
`numactl --interleave=all`. Clocks were not locked and installed DIMM speed was
not measured. Huge pages were disabled; the mapped weights used 4 KiB pages.
No administrator page-policy change was attempted.

`tools/glm_fp4_cpu_stream.cpp` supplies a separate read-only bandwidth probe.
Build it with `g++ -O3 -std=c++20 -pthread`, then run it under the same NUMA policy.
On this node, 32 pinned workers reading 4 GiB measured a median **117.816 GB/s**
across five samples after two warmups. Each sample makes fifteen complete passes
and verifies its checksum. This is a bandwidth scenario for the conditional
roofline, not a measurement of FP4 unpack, dot-product or scale-cache throughput.

The compact evidence is in
[`glm_fp4_v100_optimization_measurement.json`](glm_fp4_v100_optimization_measurement.json).
Raw output remains separate; hashes identify the frozen reports and executables.
The previous reference measurements and profiles remain unchanged.

## Short screens

These are single 8,192-to-128 timings, useful for rejecting candidates, not the
final 512-output throughput claim. Build labels identify immutable executables
in the JSON evidence; controls and output hashes are retained there.

| Candidate | Decode tok/s | Prefill tok/s | Finding |
| --- | ---: | ---: | --- |
| v1 control | 22.14 | 390.83 | Original fast settings |
| CPU row tile 2 / 4 | 20.27 / 19.86 | 391.70 / 390.93 | Slower |
| Quartet scales, 512 MiB cache | 8.97 | 390.66 | Cache construction/eviction overwhelms isolated kernel gain |
| Owning scale copies | 20.40 | 390.80 | Slower than borrowed metadata |
| MTP depth 5 / 7 | 18.02 / 15.73 | 391.53 / 391.36 | More speculative work; changed outputs; not selected |
| v2 control | 21.30 | 396.46 | Matched GPU experiment control |
| GPU reduction | 21.33 | 443.59 | Prefill gain; unchanged short output |
| Queue 4 + GPU reduction | 21.18 | 431.25 | Slower than reduction alone |
| Refined grouped pairs + reduction | 19.56 | 430.19 | Slower; changed arithmetic/output |
| Dense split + reduction | 16.24 | 444.09 | More MTP rounds and slower decode |
| v4 input Q8 reuse | 22.39 | 395.92 | Advance to full qualification |
| Input reuse + prepared operands | 22.46 | 396.98 | Near input reuse alone |
| Revised CPU teams 4 / 8 | 20.74 / 21.88 | 396.46 / 396.06 | Not selected |

The 6,144 MiB quartet-cache screen incurred 148 timed major faults and is
excluded. Queue 2 incurred 43; initial grouped pairs incurred 21; refined
grouped quartets with reduction incurred 32. Their raw records are preserved,
but none qualifies as a throughput result. The clean initial quartet request
admitted only pairs under its workspace rule, as confirmed by the batch counters.

For the real 8,192-by-4,096 MXFP8 attention projection, the dense split kernel
also lost in isolation: 64.54→72.48 microseconds at width 1,
68.97→75.05 at width 4, and 71.50→77.52 at width 8. That probe rotates three
matrix copies and times 1,000 CUDA-graph launches after warmup. It is one kernel
measurement, not a repeated model benchmark. Numerical checks passed, but the
changed reduction tree can alter MTP proposals; the model screen increased from
29 to 37 rounds. Dense splitting remains an unselected opt-in experiment.

## Updated resource bounds and gaps

Run the separate, unfitted native profile:

```sh
python3 tools/glm_runner_sim.py native-fp4 \
  --profile tools/sim/data/glm_fp4_v100_optimization.json \
  --target-decode 50 --target-prefill 800 --output tmp/native-fp4-roof.json
```

Use `--target-prefill 1000` for the stretch target. The original simulator
profile and its default results remain unchanged. An explicit `--cpu-gbps`
override is a scenario, even when a measured read probe exists in the profile.

| Metric | Selected measurement | Target | Throughput factor still needed |
| --- | ---: | ---: | ---: |
| Decode | 32.67 tok/s | 50 | 1.53× |
| Prefill | 454.29 tok/s | 800 | 1.76× |
| Prefill stretch | 454.29 tok/s | 1000 | 2.20× |

The median decode request took 15.6427 seconds for 511 advances. It executed
27,590 CPU expert groups, 625 main-model target rows and 146 speculative rounds
(3.5 advances per round). Reading each group's original FP4 codes once amounts
to 347.16 GB. At the separate 117.816 GB/s read-probe rate, that gives a
**2.947-second / 173.4 tok/s conditional streaming roof**. It omits scale
metadata, unpack/dot instructions, activation traffic and scheduling. The
observed 18.8% of this loose roof does not identify a single removable cost.

Verification consumed 13.5103 seconds, or 86.4% of decode time. Removing all
non-verification work would reach only **37.82 tok/s**. With non-verification
work fixed, verification must become **1.67× faster** to reach 50 tok/s.
The deeper-MTP screens and CPU-kernel experiments did not deliver that gain.

The median prefill took 18.0325 seconds. Its actual submitted matrix H2D traffic
was **70,177,140,736 bytes on GPU 0** and **53,274,591,616 on GPU 1**. Using the
previous concurrent DMA probe's 12.3075 / 12.3355 GB/s links and 24.615 GB/s
aggregate gives a **5.702-second / 1,436.7 tok/s conditional roof**, limited by
the primary link. Measured throughput is 31.6% of that optimistic bound.
The old compressed-extent proxy remains separately visible in the simulator;
it is not substituted for these phase counters.

Reaching 800 requires reducing prefill by 7.79 seconds; reaching 1000 requires
9.84 seconds. At unchanged matrix traffic, the primary link would need average
rates of 6.85 and 8.57 GB/s respectively. Both are below the isolated DMA probe,
but neither proves the targets achievable: activation traffic, reconstruction,
attention/KDA/mHC work, launches and resource contention are omitted, and the
bound assumes optimistic overlap. The phase counter ends at `compact`, so
post-compact cache preparation remains outside its prefill-byte count.

Further work should measure kernel-active time and host enqueue gaps separately
before replacing the decode kernels or prefill schedule. A tiled prefill kernel
that consumes compressed weights directly, or overlap between reconstruction
and GEMM, remains an experiment to investigate. Neither was implemented or
measured here, and neither is assigned a speculative speedup. Simply increasing
queue depth or grouping cuBLAS calls did not improve the measured workload.

## Validation and limits

The final decoder's original greedy path, with all new switches off, produced
exactly the frozen reference's 128 token IDs on the 8,192-token input. This was
an output check only, without a full warmup. The frozen fast profile also kept
its complete 512-token output on the final build. Both original measurement
JSON files, both original presets, the original simulator profile and the
reference branch are unchanged.

CUDA SM70 and HIP decoder builds passed. The V100 ran the FP4 numerical tests,
including the new dense-split width checks; final CPU tests passed on VNNI,
forced AVX2 and scalar paths. The local GLM host suite accounted for twenty
passes and eight hardware/topology skips. Its socket transport test initially
failed inside the restricted sandbox and passed outside it. Benchmark/simulator
Python tests passed (36 tests). HIP GPU execution, single-GPU execution and the
non-peer transfer fallback were not exercised. The separate SYCL project does
not compile the changed engine sources.

A CPU-only build of every target still fails linking the unrelated
`strata-glm-exl3-check` and `strata-glm-exl3-dequant` tools because
`strata_env_is_one` is unresolved. Their source and link rules were unchanged;
the GLM test targets were built directly. This report does not claim that the
repository-wide all-target build passes.

Compute Sanitizer completed the selected GPU-reduction path at 128 input tokens
and eight outputs with **zero errors** and process exit zero. Its timings are
instrumented diagnostics only. The larger 2,048-token grouped-pair/reduction/
dense-split diagnostic was interrupted during its first prefill to stay within
the allocation; it has no completed clean result and is not counted as passed.

The additional fully metadata-warmed phase profile was skipped when only 322
seconds remained, below the queue's 330-second full-run guard. Earlier detailed
phase profiles remain available, explicitly labeled with their 333 first-timed
CPU scale-cache misses. They must not be presented as fully warmed metadata
costs. The selected request's aggregate MTP timers and exact matrix-transfer
counters used above come from the completed three-run qualification.
