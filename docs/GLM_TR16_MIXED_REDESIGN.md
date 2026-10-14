# GLM 5.3 Flash on TR16: mixed GPU redesign

## Scope and status — 2026-10-14

This work adds a resource simulator and measures the existing Q2/Q3 runner.
The proposed kernel changes below are design work. The Project Maya study also
adds an opt-in worker cache admission policy, tested on CUDA and HIP. Its measured
threshold experiment did not show a reliable gain. The coding quality gate remains **152/164 HumanEval**;
timing results alone cannot qualify a configuration.

Upstream `Niko1221/Strata/main` at
`296b59d2f73c8f04a805df29c2fa97655adace78` was merged into `glm53f` as
`62d48657968e4e36053ea51b6e070e0c1ea98db6`. Existing local GLM work was preserved.
CUDA and ROCm 10.1 decoder/worker builds passed. Both backends passed the three
GLM host tests, Q2/Q3 GPU parity and prefill parity. SYCL execution and compilation
are not qualified: this machine has no `icpx`/`dpcpp` SDK.

The merge retains GLM environment-switch semantics and opt-in HC Q8 behavior.
Upstream gfx1201 Q8 GEMV improvements reach shared kernels. Priority flags in
upstream `core/mtp.cpp` and `core/verify.cpp` do not automatically affect the
standalone GLM runner, which owns its streams. NVFP4 additions do not convert
this Q2/Q3 model or make its expert executor faster by themselves.

## Hardware and measured resources

TR16 has a Threadripper 1950X, 128 GB RAM (about 125 GiB usable), RTX 5060 Ti
16 GB on NUMA node 1, and RX 9070 XT 16 GB on NUMA node 0. Device generation is
not the host-link generation: the NVIDIA path is PCIe 3 x8; AMD is PCIe 3 x16.
The model volume has a PCIe 3 x1 link and the work volume a PCIe 3 x4 link.
Identify storage by mount/PCI address; Linux NVMe device numbers can change.
The earlier hardware record reports DDR4-3200 from the user; configured DIMM
speed was not independently reread for these measurements.

| Resource | Measured decimal GB/s | Scope |
|---|---:|---|
| RAM, 15 workers, interleaved | 63.79 | streaming read |
| NUMA-local RAM, 8 workers | 42.55 / 41.65 | nodes 0 / 1 |
| Remote RAM, 8 workers | 20.70 / 20.93 | cross-node read |
| NVIDIA H2D / D2H | 7.20 / 7.15 | pinned 64 MiB transfers |
| AMD H2D / D2H | 14.31 / 13.90 | pinned 64 MiB transfers |
| Concurrent NVIDIA + AMD H2D | 7.21 + 14.31 | independent transfers |
| Model NVMe | 0.934 | O_DIRECT, 8 MiB, queue depth 4, random |
| Work NVMe | 3.685 | O_DIRECT, 8 MiB, queue depth 4, random |
| Model / work NVMe, 4 KiB | 0.179 / 0.186 | random, queue depth 4 |
| Model / work NVMe, 64 KiB | 0.881 / 1.151 | random, queue depth 4 |
| Model / work NVMe, 1 MiB | 0.934 / 3.573 | random, queue depth 4 |

The GPU probe also executed and checked a simple arithmetic kernel on both GPUs.
Its copy rates (295/533 GB/s) are not expert-kernel throughput. Probe trials,
source hashes and exit status are in
[resources.json](benchmarks/tr16_mixed_20261014/resources.json).
The CPU Q2/Q3 flow probe measured 69.77, 69.91, 67.50 and 62.80 logical GB/s at
verification widths 1, 2, 4 and 8. These figures describe that microbenchmark,
not the complete model or additional bandwidth available alongside DMA.
The small-read receipt is [storage_small_reads.json](benchmarks/tr16_mixed_20261014/storage_small_reads.json).
Fitting mean latency at queue depth 1 and retaining the independent large-read
bandwidth predicts the held-out queue-depth-4 rates within 0.02–8.19% across
the six size/drive combinations. See
[storage_validation.json](benchmarks/tr16_mixed_20261014/storage_validation.json).
This qualifies that storage component over those samples, not whole-model
throughput or other queue depths. Each size/depth used one bounded 2 s sample.

### ROCm runtime requirement

Both the HIP decoder and a HIP worker launched by another process require the
ROCm 10.1 runtime libraries. Preserve any other necessary library directories:

```bash
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

A successful build does not establish that the runtime loader or GPU kernels
work. The probes and parity runs above were executed with this setting.

## What SparkGLM contributes

SparkGLM targets two DGX Sparks with unified memory and a high-speed interconnect.
Its NVFP4 deployment and memory layout are materially different from two discrete
16 GB cards. Its NVMe cache stores reusable inference state; that is not evidence
that streaming all active expert weights from this machine's disks will be fast.
See the [project](https://github.com/Enntity/sparkglm) and
[limitations](https://github.com/Enntity/sparkglm/blob/main/docs/LIMITATIONS.md).

The kernel study used Atlas
`dfb9c26a95c96faa9f94c44e23343b0f2b4f9db6`. A later check of the
[engine manifest](https://github.com/Enntity/sparkglm/blob/main/install/atlas-source.json)
found `6f27e8fa0afe65180a169e54361a77565787ae27`, adding a refusal-content fix;
the manifest says the decode path is unchanged and performance was not remeasured.
Atlas is AGPL-3.0-only. This simulator is independently implemented; no Atlas
kernel code was copied into Strata.

Useful design principles from the inspected kernels:

- Build an expert-major work list and reuse resident weight tiles across actual
  verification rows. Persistent scheduling can reduce launches and repeated loads.
- Fuse routing and adjacent MLA operations while preserving top-k tie behavior,
  reduction order and row invariance. Validate widths 1–32 and speculative rollback.
- Measure KDA/MLA prefill separately. Specialized KDA kernels have eligibility and
  recurrent-state layout constraints; an operator gain is not a whole-model gain.

Source references: [expert stream](https://github.com/Enntity/atlas/blob/dfb9c26a95c96faa9f94c44e23343b0f2b4f9db6/kernels/gb10/glm-5.3-flash/nvfp4/glm_moe_decode_stream.cuh),
[routing](https://github.com/Enntity/atlas/blob/dfb9c26a95c96faa9f94c44e23343b0f2b4f9db6/kernels/gb10/glm-5.3-flash/nvfp4/glm_moe_route.cu),
[MLA](https://github.com/Enntity/atlas/blob/dfb9c26a95c96faa9f94c44e23343b0f2b4f9db6/kernels/gb10/glm-5.3-flash/nvfp4/glm_mla_fuse.cu),
[KDA research](https://github.com/Enntity/atlas/blob/dfb9c26a95c96faa9f94c44e23343b0f2b4f9db6/research/flash-kda-prefill/README.md).
GB10 NVFP4 instructions, CUDA scheduling features and unified-memory assumptions
need separate CUDA Q2/Q3 and HIP implementations. Speculative acceptance must be
measured on our prompts; DFlash acceptance is not a free simulator multiplier.

## Project Maya follow-up

Studied [Project Maya](https://github.com/mw00/project-maya) at
`5c970a863ce2fdd87b6e23ffb4822f6d76522b1c` (v1.1.8). It is MIT licensed and
derives from Strata, with a different GLM engine and quantization choices.
Its published machine results are not calibration data for this Q2/Q3 runner;
zero-shot accuracy percentages do not establish our HumanEval floor.

The most relevant source is
[`glm_fast_path.cu`](https://github.com/mw00/project-maya/blob/5c970a863ce2fdd87b6e23ffb4822f6d76522b1c/src/core/glm_fast_path.cu):
it measures CPU expert service and host-to-device transfer separately, prices
their concurrent lanes, and can revisit an initially noisy CPU measurement.
An individually slower transfer may help when it overlaps several CPU experts.
Shared RAM bandwidth, resident GPU work and partial-result traffic can remove
that gain, so neither a per-expert comparison nor adding nominal bandwidths is
sufficient.

The simulator's `miss_lanes` manifest enumerates CPU/CUDA/HIP counts under those
shared resources. It requires explicit staging slots, adds their bytes to peak
memory, and accepts resident work as `fixed_events`. Staging precedes compute
within each GPU lane; no unimplemented streaming overlap is assumed. Kernel
costs remain inputs, not advertised FLOPS converted to a forecast.

Maya's [expert dispatch](https://github.com/mw00/project-maya/blob/5c970a863ce2fdd87b6e23ffb4822f6d76522b1c/src/kernels/cuda/glm_fast.cu)
uses aged route counts to avoid retaining experts used only briefly. The mixed
worker now has an independently implemented admission policy:

```bash
export STRATA_GLM_MIXED_ADAPT_CACHE=1
export STRATA_GLM_MIXED_ADAPT_MIN_ROUTES=6
export STRATA_GLM_MIXED_ADAPT_MIN_GAIN_PCT=10
```

Both new thresholds default to zero, preserving the existing selection path.
When enabled, they gate optional replacements by the observed decayed route
count and the relative gain in the existing prior-plus-route score. The policy
keeps the four-replacement limit, excludes primary owners, and fills empty or
primary-displaced slots even below the count threshold. With the existing
`ADAPT_AFTER_ROWS` option, ownership changes still occur after the old rows
finish. It changes residency, which can affect answers under affinity or mixed
CPU/GPU arithmetic; it is experimental until independently quality-qualified.
`STRATA_GLM_MIXED_ADAPT_METRICS=1` adds adaptive upload MiB to the worker's final
summary, separating decode cache updates from the much larger prefill uploads.

The selection rule lives in C11 `src/host/glm_mixed.c`. The optional
`strata_glm_mixed_policy` shared-library target exposes the same rule to
`tools/sim/cache_policy.py`; there is no Python copy of the admission rule.
The simulator's caller must provide the actual learning counts. Biased execution
routes cannot silently substitute for the runner's unbiased learning routes.

Maya's [calibration notes](https://github.com/mw00/project-maya/blob/5c970a863ce2fdd87b6e23ffb4822f6d76522b1c/docs/SETTINGS.md)
also motivate separating fresh prompts from repeated warmed prompts, tuning
storage read granularity on the actual disk, and measuring gains before keeping
a setting. Our three-trial matrix is explicitly a warmed repeated-prompt study.
It cannot by itself predict first-answer or ordinary multi-turn chat throughput.
The NVMe probe now accepts `--block-kib` for 4 KiB/64 KiB demand-read measurements;
the existing 8 MiB throughput must not price small random page faults without a
matching measurement. Read size, queue depth and cache misses remain explicit
simulator assumptions.

The [RDNA expert report](https://github.com/mw00/project-maya/blob/5c970a863ce2fdd87b6e23ffb4822f6d76522b1c/docs/AMD_EXPERT_KERNELS.md)
leaves gfx1201 on its prior dispatch. Its IQ2/IQ3 changes on gfx1100/gfx1151 are
not measured Q2_K/Q3_K improvements for RX 9070 XT. No model weights were
downloaded or converted, and no Maya kernel code was copied into this change.

## Recommended design order

1. **Resident hybrid execution.** Keep dense/recurrent state on AMD initially,
   allocate disjoint hot-expert caches on both GPUs, and keep RAM experts on the
   CPU when a transfer costs more than CPU execution. Use measured per-layer costs
   to set the split, reserving VRAM for KV, MTP snapshots, scratch and desktop use.
2. **GPU launch and synchronization work.** Fuse eligible routing/MLA operations,
   batch resident Q2/Q3 expert work, and overlap independent CPU and GPU branches.
   Return one reduced partial per GPU/layer. Preserve arithmetic unless an
   explicitly lossy variant independently clears the quality gate.
3. **Selective expert promotion.** Admit experts based on observed reuse and
   saved CPU service, with bounded staging buffers and eviction only after the
   last consumer completes. Reuse within MTP verification belongs to the actual
   route trace. Do not assume knowledge of future routes.
4. **Prefill kernels and partitioning.** Profile KDA, MLA and expert GEMM
   independently; reduce the measured mixer floor and tune uneven expert splits.
   Single-request layer pipelining still has dependencies and transfer overhead.
5. **NVMe as a cold tier.** Warm RAM deliberately; use bounded asynchronous reads
   only for RAM misses. Prefix/KV caching can improve repeated-prompt TTFT but
   must be reported separately from cold prefill and steady decode.

Existing upstream discussion includes [heterogeneous draft roles #490](https://github.com/Niko1221/Strata/issues/490),
its [PR #492](https://github.com/Niko1221/Strata/pull/492), and
[cross-vendor Vulkan #1214](https://github.com/Niko1221/Strata/issues/1214).
Draft placement must account for draft KV, prompt processing, restore ordering
and residency changes. On our retained mixed trace, drafting occupied 1.40 s of
15.80 s. Eliminating that entire phase would reach only about 35.5 decode tok/s
with everything else fixed, before any extra-device transfer or lost expert-cache
capacity. It is therefore a lower-priority route to the 50 tok/s target than
reducing the verification and host-service costs.

Moving a 9.1 MB expert to AMD takes about 0.64 ms at the measured H2D rate,
versus roughly 0.13 ms of CPU byte service in the width-1 flow microbenchmark.
About five reuses on AMD (ten on NVIDIA) are needed merely to amortize the
transfer in this simplified comparison, before GPU compute and coordination.

## Simulator

`tools/sim/heterogeneous.py` schedules explicit dependency graphs with shared
resource capacities, finite queues and max-min bandwidth allocation. It reports
elapsed time, per-resource service, a DAG/resource lower bound, and the scope of
any derived token rate. GPU-to-GPU paths explicitly pass through host memory.
Events may specify `minimum_ms` for independent I/O latency. Chunked storage
reads each occupy a queue slot and share the measured disk/RAM bandwidth; tail
overfetch is charged in full. The profile's 1 MiB chunks describe a proposed
O_DIRECT cold tier, while mmap demand reads require their measured size profile.

`tools/sim/data/hw_tr16_mixed.json` contains the measured resource rates.
Its cache partition is illustrative: replace cache sizes and provide all phase
peak allocations before using a deployment estimate. Architectural GPU VRAM
bandwidth is identified separately from measured kernel costs. Unknown launch
latency must be supplied; zero latency is an optimistic assumption.
The illustrative decode partition gives 10 GiB to the CUDA worker, 6 GiB to
the HIP primary and 95 GiB to RAM. Their combined expert capacity can hold the
full expert table if GPU and RAM inventories are disjoint. The memory limits
use the runner's 13/14 GiB operative GPU budgets; reported device availability
is retained separately. Dense weights, recurrent/KV state, MTP snapshots,
workspace and staging must still fit. Prefill may need a different cache budget
when it borrows/reclaims decode slots.

```bash
python3 tools/sim/glm_sim.py hw --hw tr16-mixed
python3 tools/sim/glm_sim.py heterogeneous --manifest execution.json --output estimate.json
PYTHONPATH=tools python3 -m unittest tools.test_glm_heterogeneous tools.test_glm_mixed_bench tools.test_sim tools.test_sim_multinode
```

The simulator uses Python's standard library. Build the optional C11 admission
binding with `cmake --build "$BUILD" --target strata_glm_mixed_policy` only when
comparing that policy through `sim.cache_policy.CacheAdmissionPolicy`.
For a runnable transfer example, create a manifest as follows, then pass it to
the `heterogeneous` command above:

```bash
PYTHONPATH=tools python3 - <<'PY'
import dataclasses, json
from pathlib import Path
from sim.heterogeneous import transfer_events
hardware = json.loads(Path('tools/sim/data/hw_tr16_mixed.json').read_text())
document = dict(schema='strata.heterogeneous.v1', hardware=hardware,
    allocations={}, output_tokens=1,
    provenance=dict(scope='Single 9.1 MB transfer component; not model throughput'),
    events=[dataclasses.asdict(e) for e in
            transfer_events('expert-upload', 9100000, 'ram', 'hip', hardware)])
Path('execution.json').write_text(json.dumps(document, indent=2) + '\n')
PY
```

This example predicts 0.636 ms for one AMD upload. Its reciprocal is a component
rate; it is not GLM decode tok/s. The illustrative cache budgets are checked,
but a real deployment manifest must include every other live allocation.

A `strata.heterogeneous.v1` event manifest contains `hardware`, `allocations`,
`output_tokens`, `provenance` and `events`. Each event gives `name`, `work`
(resource demand in bytes or milliseconds), `after`, and optional queue `slots`.
An event's `minimum_ms` is a duration floor, independent of bandwidth demands.
The route form supplies `groups`, layer/width `costs`, explicit initial cache
inventories and `accepted_tokens`. All execution groups, including rejected MTP
branches, must be included. JSON cache keys are `[layer, expert]` pairs.

Route replay reports **expert-group throughput**, not end-to-end model throughput.
Draft, dense prefix, sampling and other control costs must be supplied as events
for an end-to-end estimate. Groups larger than their cache require an explicit
staged DAG; the replay refuses unaccounted temporary allocations. The old
identical-GPU estimate path rejects `tr16-mixed`.

`tools/glm_route_cache_plan.py` produces a disjoint cache candidate from an
explicit training request and independently supplied layer/width kernel costs.
It ranks isolated CPU service saved per cache byte, respects both cache budgets,
and records source hashes. Evaluate its inventory on separate held-out requests;
the greedy ranking omits coordination and concurrency and is not an optimal
throughput claim. The output is simulator input, not a runner configuration.

```bash
python3 tools/glm_route_cache_plan.py --training-routes train.routes.jsonl \
  --request 1 --costs measured-kernel-costs.json \
  --hardware tools/sim/data/hw_tr16_mixed.json --allocations phase-peak-memory.json \
  --output cache-candidate.json
```

## Scoped performance bounds

A retained pre-merge diagnostic serviced 375,624,433,664 CPU expert bytes for
512 outputs: 0.734 GB/output. Holding those routes and acceptance fixed gives:

| Hypothetical execution | Transport-only ceiling at measured rates |
|---|---:|
| Upload all those bytes through both GPU links | about 29.3 decode tok/s |
| Read all those bytes from model NVMe | about 1.27 decode tok/s |
| Read all those bytes from work NVMe | about 5.02 decode tok/s |

These are bounds on that streaming workload, not bounds on resident hybrid
execution. At 50 decode tok/s the model disk permits at most 18.7 MB of misses
per output before accounting for compute. Full cold expert-table reads on that
disk take at least 119 s for full Q2/Q3 or 92.6 s for REAP224; at 8192 prompt
tokens that is at most about 68.8 or 88.5 prefill tok/s, respectively. Warm RAM
avoids this disk bound.
The two PCIe 3 GPU links have an encoding-only aggregate ceiling of 23.63 GB/s
(8 GT/s, 128/130 encoding, 24 lanes). That permits at most about 32.2 tok/s for
the same all-streamed byte workload before packet overhead and compute.
Measured-rate bounds and encoding-only bounds are separate simulator fields;
neither establishes a maximum for a resident-cache design with different work.

Historical aggregate-phase sensitivity gave about 591 prefill tok/s with the
measured coordination residual and unchanged kernels. Assuming twice-fast
mixers and experts gave about 954 tok/s. A separate decode sensitivity gave
52.7 tok/s only after assuming both host service and between-verify waits halved.
These are conditional engineering targets, not demonstrated maximum throughput.
The phase residual includes nonlinear work and model error; it is not a direct
communication measurement. See the preceding analysis in [MULTI_GPU.md](MULTI_GPU.md).

## Measurement and qualification protocol

The same-day controls use frozen before/after decoder and worker binaries.
The matrix covers full Q2/Q3 and REAP224, 2K/8K code/chat, NVIDIA alone, AMD alone,
and AMD-primary mixed execution at fraction 0.5. Each loaded process runs a warm
decode followed by three timed 512-token trials. Prefill is repeated three times.
The original control's separate-process trials are retained separately: process
loading dominated wall time, so the remaining matrix changed to in-process trials.
Those protocols are not pooled into one statistic.

The REAP224 configuration originates from the profile previously scoring
152/164. This does not qualify the merged binaries, the full pack or every GPU
split. Earlier 35.26 tok/s measurements with 149/164 and corpus failure remain
excluded from quality-qualified recommendations.
The portable [numerical profile](benchmarks/tr16_mixed_20261014/numerical_profile.json)
records affinity 0.14, MTP depth 3, soft acceptance margin 4 / top-k 6 / maximum
3, and the adaptive after-row cache settings. Set its model and prior paths to
the local files before running it. This template replaces the original absolute
model path; the measured configuration's original hash remains in the receipt.
For the full 288-expert pack, set `STRATA_GLM_REAP_PREFETCH=0`: the REAP-only
prefetch option rejects that geometry before inference.

`tools/glm_tr16_redesign_report.py` records medians/ranges, binary/config hashes,
old/new comparisons and held-out errors. A code-prompt rate is used to forecast
chat at the same shape. Tolerances are 10% decode and 15% prefill. Passing this
simple holdout validates only that workload forecast; it does not validate new
kernels, routing policies, pruning or MTP acceptance. Instrumented route/phase
runs are separate from timing trials. Incomplete, failed and loop-rejected runs
are excluded, with reasons retained. An incomplete matrix is never calibrated.

The REAP224 matrix finished. The full-pack matrix and fresh instrumented traces
were stopped for the requested wrap-up; their partial results remain in the local
build tree. No new quality-qualified maximum is established.

Compact results, source/config/binary hashes, exclusions and the wrap-up receipt
are in [measurements.json](benchmarks/tr16_mixed_20261014/measurements.json).
The full-pack run completed two processes. Its AMD process overlapped final
build/parity checks and is excluded from timing comparisons. The NVIDIA process
is a single partial-matrix observation, not a full-pack qualification.

### Completed timing controls

All rates below are on this TR16, with the same experimental numerical profile.
Three timed 512-token decode trials follow a warm-up in each loaded process.
They are throughput observations, not a new HumanEval qualification.

| REAP224, 8K training code, post-merge | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| NVIDIA alone | 395.00 | 31.13 |
| AMD alone | 552.37 | 31.43 |
| AMD-primary mixed, fraction 0.5 | 525.57 | 31.11 |

The AMD training-code row is excluded from qualification because its warm-up
looped, although the timed trials passed the loop check. On 8K held-out chat,
AMD alone measured 554.02 prefill and 35.01 decode tok/s. Mixed measured 523.01
and 33.72. These results do not support a 1.5–2× mixed-GPU gain.

A later paired NVIDIA control measured 392.88/30.94 before the upstream merge
and 391.50/30.82 after it: −0.35% prefill and −0.40% decode, with equal last-prompt
logit hashes. This pair showed no material regression. Earlier controls drifted
more; the later pair does not establish a general no-degradation guarantee.

For Maya-style admission on 8K held-out code, the old worker measured
519.37/31.72, the new default 513.57/30.09, and minimum routes 6 / gain 10%
517.97/30.68. Adaptive upload increased from 24,385.8 to 25,715 MiB with the
thresholds. Keep both thresholds at their default zero unless a new workload
demonstrates a benefit. Equal prefill hashes do not establish identical decode:
the existing lossy adaptive profile produced different generated streams.

A 10-of-14 AMD prefill split measured 562.09 tok/s at 8K and 168.03 at 2K.
These used only 32 generated tokens and are prefill screens. They have no coding
quality qualification and do not establish decode gains.

`STRATA_GLM_MIXED_PREFILL_PRIMARY_ONLY=1` is a separate opt-in path: run all
prefill expert groups on the primary GPU and keep the worker for decode. The
fraction still selects the split when this switch is absent. This supports
measuring whether the short-context worker overhead outweighs its expert work.
The simulator's aggregate adapter charges no join communication at fractions
zero or one; mixer and other primary work still remain serial. This is a phase
sensitivity assumption, not automatic placement or a default change.
The final 2K held-out-code smoke ran AMD alone and AMD-primary mixed with this
switch. Both completed; last-prompt logits matched byte for byte. The mixed
log reported 14 primary / zero worker prefill groups and zero worker prefill
rows, followed by worker decode requests. Only 32 tokens were generated, and
parity checks overlapped the single-GPU control: use this as a numerical/liveness
check, not a performance or coding-quality result.

### Final validation

CUDA (`sm_120`) and HIP (`gfx1201`, ROCm 10.1) decoder/worker builds passed.
All 128 Python checks, 22 CUDA GLM checks and 21 HIP GLM checks passed. The first
HIP suite invocation lacked its library path and some test targets; after
building those targets and supplying `LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib`,
the complete registered HIP GLM suite passed.
Default worker expert rows matched the frozen post-merge worker byte for byte
within each backend on layer 3, experts 0/1, rows 1/2/4/8, two repeats. This is
a component check, not whole-decode bit identity. The optional C11 simulator
binding loaded and applied the admission rule successfully.
Receipts and executable hashes are in
[validation.json](benchmarks/tr16_mixed_20261014/validation.json).
SYCL remains unbuilt because its SDK is absent. A new 152/164 HumanEval gate,
full-pack timing matrix and fresh instrumented route matrix remain uncompleted;
all new runner policies remain opt-in.

## Reproduce the component probes and sensitivity calculation

Run probes with inference and other substantial CPU/GPU work idle. The storage
probe reads existing files with O_DIRECT; it creates no weight copies and does
not drop the system page cache. Choose representative files on each volume.

```bash
mkdir -p /tmp/tr16-probes
nvcc -O3 tools/glm_transfer_probe.cu -o /tmp/tr16-probes/probe-cuda
/opt/rocm/core-10.1/bin/hipcc -O3 -DSTRATA_PROBE_HIP tools/glm_transfer_probe.cu -o /tmp/tr16-probes/probe-hip
g++ -O3 -march=native -fopenmp tools/glm_memory_bandwidth.cpp -o /tmp/tr16-probes/probe-ram
python3 tools/glm_tr16_resource_probe.py --binaries /tmp/tr16-probes \
  --model-file "$MODEL_FILE" --work-file "$WORK_VOLUME_FILE" \
  --output /tmp/tr16-component-results
python3 tools/glm_tr16_sensitivity.py \
  --prefill build-glm-release/mixed-validation/redesign-prefill-sim-v2.json \
  --decode-trace build-glm-release/mixed-validation/redesign-decode-trace-v2.json \
  --decode-benchmark build-glm-release/mixed-redesign-trace-v2/results.json \
  --output /tmp/tr16-sensitivity.json
python3 tools/glm_tr16_redesign_report.py \
  --runs build-glm-release/tr16-redesign-20261014 --output /tmp/tr16-report.json
```

The historical input files remain in the local build tree. Their identities and
calculated scenarios are retained in
[sensitivity.json](benchmarks/tr16_mixed_20261014/sensitivity.json). Reproducing a
historical calculation requires those exact inputs; fresh traces are a new
measurement. `resources.json` is a compact frozen probe receipt, not the output
of a later rerun.

For a new end-to-end matrix, freeze both builds into `cuda/` and `hip/`
subdirectories of separate before/after roots, then use:

```bash
python3 tools/glm_tr16_measure.py --before "$BEFORE_BINARIES" --after "$AFTER_BINARIES" \
  --reap224-config "$REAP_CONFIG" --full-config "$FULL_CONFIG" \
  --python build-exl3-venv/bin/python --output "$NEW_RESULTS_DIRECTORY"
```

The driver records input hashes and will resume only complete batches with
matching receipts when `--resume` is supplied. Preserve partial batches and use
a new directory when retrying. The configuration files must select the intended
weights, expert prior and numerical profile; this driver does not choose new
loss knobs. Apply an external memory limit if appropriate for the host.
