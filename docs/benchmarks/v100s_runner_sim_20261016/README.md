# GLM release profile on the v100s node

Measurements on 2026-10-16 JST (2026-10-15 UTC), from the integrated TR16 engine
`83f3cac9aab5c16c50d032b3eed26b380728b0da`. The machine is called **v100s node**.
Timing qualification and model quality qualification are separate.

## Configuration and protocol

Two Tesla V100 PCIe 32 GB GPUs, two Xeon Gold 6240 CPUs at a nominal 2.60 GHz,
36 physical / 72 logical CPUs, four NUMA domains (two per socket), and a 160 GiB
RAM allocation (190,991 MiB OS-visible in NUMA topology). RAM speed was unavailable.
Red Hat Enterprise Linux 8.10, GNU 12.2.1, NVIDIA driver 580.178.04, CUDA 12.9,
sm_70, Release build. Both cards report PCIe 3.0 x16 and 250 W power limits.
GPU 0 is the primary; GPU 1 is an expert worker. CPU masks reserve physical core
35 for the worker and cores 0–34 for the primary (34 pool workers plus host).
GPU 0 is attached to NUMA domain 0 and GPU 1 to domain 2.

The captured TR16 release uses **REAP-224 Q23, affinity 0.14, MTP depth 3**,
GPU draft experts with top-k 4, soft acceptance margin 6 / top-k 8 / max-soft 3,
repeat guard 4, lookup depth 7 / minimum match 4 / cost policy, canonical experts
and `device` combine, joint dense verification, batched residents, stable worker
ownership and prompt prior. The baseline primary/worker caches are
10,240/14,336 MiB. Each GPU has a 30,720 MiB experiment budget.

The [portable configuration](../../../configs/experimental/glm53f-q23-reap224-v100s-aff014.json)
keeps those numerical knobs. Platform adaptations are 34 CPU workers, CUDA on
both GPUs, four prefetch groups and a symmetric prefill split. This four-domain
machine cannot use the runner's two-domain CPU weight sharding: use mapped-lazy
weights, `PACK_FUSE_QUANT=0`, and `NUMA_AFTER_PREFILL=0`. Prefill expert cache is
zero, matching the release CLI benchmark. These measurement tools and configuration do not change engine defaults. This is a configuration comparison on one integrated build; no same-day pre-sync engine comparison was completed.

Each clean case ingests an 8,192-token prompt, with context capacity 12,288 and
prefill batch 8,192. It measures three prefills, creates a saved prefill state in
an additional untimed pass, performs one complete 512-token decode warmup, then
three timed 512-token decodes restored from that state. Adaptive cache state can
continue learning across repeats. Reported decode timing excludes prefill,
model load and initial ownership admission. Token zero is outside the timed loop:
throughput uses `(generated - 1) / milliseconds`, as the engine reports it.
This is one-shot CLI throughput, not server request latency or concurrent serving.

For RAM cases, one sealed, read-only shared memfd holds the GGUF, pinned by
[`ResidentModel`](../../../tools/glm_resident_model.py), under
`numactl --interleave=all`. Its verified pages are divided equally among all four
NUMA domains. Both GPU processes share these pages. Use `--warm-weights` in the
children; `--lock-weights` would create private COW pages. Initial staging and
SHA-256 readback take several minutes and are outside the timed loop.

## Measurements

The filesystem baseline, with the same release flags, measured **41.126 tok/s**
(median; 42.1515 / 41.1260 / 40.4045). Shared RAM measured **39.5533 tok/s**
for MTP-3 and **37.2144 tok/s** for MTP-2. Shared RAM removed storage faults and
accelerated prefill; these runs do not show faster steady decode from RAM staging.
The mode-specific simulator estimates **40.139 tok/s** for MTP-3 and
**36.863 tok/s** for MTP-2, conditional on the observed acceptance of 3.650 and
2.903 timed output tokens per speculative round. These are predictions on clean
512-token holdouts, fitted only to separate 256-token diagnostic traces.
All three held-out code repetitions are within 2.5% of measurement for MTP-3
and within 1.6% for MTP-2. The shared RAM 12/24 GiB cache case measured
40.6434 tok/s. The held-out chat prompt measured 41.6872 tok/s; its median predicted rate is
41.936 tok/s. Across all nine clean code/chat repetitions, maximum prediction
error is 2.7%. These holdouts use observed acceptance; acceptance on a new prompt
is not predicted.

| Prompt / backing | MTP | Primary / worker GiB | Timed tok/s | Median | Timing health |
| --- | ---: | ---: | --- | ---: | --- |
| code / filesystem | 3 | 10 / 14 | 42.1515, 41.1260, 40.4045 | 41.1260 | pass |
| code / shared RAM | 3 | 10 / 14 | 38.0851, 39.5533, 40.4159 | 39.5533 | pass |
| code / shared RAM | 2 | 10 / 14 | 37.6593, 37.2144, 36.6079 | 37.2144 | pass |
| code / shared RAM | 3 | 12 / 24 | 40.6434, 40.3946, 41.3183 | 40.6434 | pass |
| code / shared RAM | 3 | 10 / 24 | 51.2713, 42.6158, 41.7396 | 42.6158 | **fail: looping trial** |
| code / shared RAM | 2 | 10 / 24 | 36.5699, 36.1951, 38.3491 | 36.5699 | pass |
| chat / shared RAM | 3 | 10 / 14 | 41.6872, 40.5367, 42.0100 | 41.6872 | pass |

All cases in the table have one full warmup and three timed repetitions in one
process. Case order was not randomized and these are not independent process
relaunches. The small cache-size differences are not an established general
speedup. Cache variants were measured because a V100 cache-hit curve was absent;
the simulator does not extrapolate that curve. Full numeric records, warmup and
prefill timings, token-health checks and source/output hashes are in
[results.json](results.json).

## Measured simulator inputs

[component-measurements.json](component-measurements.json) preserves numeric
measurements and hashes of the original raw files, which remain frozen privately.

| Component | GPU 0 | GPU 1 |
| --- | ---: | ---: |
| 64 MiB pinned H2D, median GB/s | 12.315 | 12.292 |
| 64 MiB pinned D2H, median GB/s | 12.689 | 13.126 |
| Device memory probe, median GB/s | 576.056 | 576.372 |
| Worker ownership no-op p50, microseconds | 23.3 | 32.6 |

Transfer probes have one warmup and five timed 0.5-second trials. The device
memory probe is not an expert-kernel timing. Worker loopback timings include
socket handling, validation, GPU execution and copies, with a zero activation;
they are not isolated GPU times. The original loopback used default worker flags;
the later wide/shared probes use the release environment. With release flags,
the GPU 1 partial-return p50 for width 2 / 16 routes is 372.98 microseconds for
16 distinct experts, versus 247.18 for eight experts each reused by two tokens.
At width 4 / 32 routes it is 662.92 versus 287.53 microseconds. These probes
confirm why route count and unique-expert count must be kept separate; they do
not measure the primary mixer chain at width 4.

The real-expert probe rotates 64 model experts, beyond LLC, across the three
actual Q23 expert format pairs. At width 2 and eight canonical Q2_K/Q3_K experts,
17 pool workers plus host take 1.312 ms; 34 plus host take 0.803 ms. The latter
workload's pinned H2D alone takes about 6.016 ms, excluding GPU compute. All RAM
is interleaved: the 17-worker run is not a socket-local data result. The probe
source is [real-expert-probe.cpp](real-expert-probe.cpp).

The traced RAM MTP-3 decode has 81.252 GB/s effective CPU expert flow
(`CPU_EXPERT bytes / layer_flow_ms`). This includes expert arithmetic and is
not a plain RAM bandwidth measurement. Primary routing residency is traced;
worker ownership was not captured. CPU route share comes from measured CPU bytes,
primary share from weighted traced residency, and worker share is the remainder.
The simulator marks this approximation explicitly.

## Simulator limits and reproduction

The V100 profile uses measured V100 components and fitted effective primary GPU
chain costs. It does not inherit the TR16 RTX/Radeon fit or use advertised FLOPS.
Publish, launch, router and other primary work are folded into effective costs;
a zero subterm does not mean a physical operation was free. Worker RPC service
already contains its return copies and is charged once. `device` combine retains
the CPU wait for the worker partial. The hypothetical `device3` join is different.

The supported scope is warm 8K REAP-224 decode with this expert-worker topology.
MTP-3 uses the unsuffixed V100 profile; MTP-2 uses the `_mtp2` kernel, parameter
and record files. Each profile checks its verification grouping. The smaller
MTP-2 window changes head, tail and draft costs as well as group width, so the
two modes must not share one set of per-round constants.
Route shares must be supplied: no cache hit curve was measured. Acceptance is an
input, not a quality prediction. Full-288, other contexts, storage latency, prefill
and the native layer pipeline need separate timing profiles. Topology B in the
legacy simulator is not the native one-token-ahead MTP pipeline and is rejected
by the V100 profiles. MTP-2 initially missed by about 11%; that failed validation
is retained in [simulator-initial-validation.json](simulator-initial-validation.json).

Run the simulator without a model or GPU:

```bash
python3 tools/glm_runner_sim.py validate \
  --hardware tools/sim/data/hw_v100s_runner.json \
  --kernels tools/sim/data/glm_kernels_v100s.json \
  --params tools/sim/data/glm_runner_params_v100s.json \
  --records tools/sim/data/glm_runner_records_v100s.json

python3 tools/glm_runner_sim.py decode \
  --hardware tools/sim/data/hw_v100s_runner.json \
  --kernels tools/sim/data/glm_kernels_v100s.json \
  --params tools/sim/data/glm_runner_params_v100s.json \
  --primary gpu0 --worker gpu1 --cpu-lanes all34 --combine device \
  --groups 2,2 --draft-bank primary --experts-per-layer 224 --policy recorded \
  --primary-slots 28.047619 --worker-slots 33 \
  --shares cpu=0.22283040651416294,primary=0.44478127289117525,worker=0.33238832059466183 \
  --accepted 3.65
```

For MTP-2 validation, add `_mtp2` before `.json` in all three V100 kernel,
parameter and record filenames above. For MTP-2 decode, also use `--groups 1,2`,
`--accepted 2.903409090909091` and
`--shares cpu=0.1939730643116298,primary=0.4875436381736093,worker=0.3184832975147609`.

For a new fit, use `calibrate` with the same hardware, kernel and records files,
an explicit `--output`, and `--rounds 8`. Fit rows contain instrumented diagnostic
scopes; clean 512-token repetitions are held out. Validation is conditional on
observed acceptance, with routing shares frozen from the diagnostic. It does not
predict acceptance on a new prompt.

[REPRODUCE.md](REPRODUCE.md) contains model identities and the GPU benchmark
recipe. [build.json](build.json) records the engine/dependency revision, binary
hashes and the frozen source-manifest hash. All 487 engine/tool source, header
and CMake files were checked against the integrated engine commit; all match. Curated evidence contains no model weights or raw profiler captures.

## Conditional sensitivity toward 60 tok/s

The [sensitivity output](sensitivity.json) keeps route shares, acceptance and
control costs fixed. At 3.65 output tokens per round, it predicts 40.14 tok/s
for the baseline, 40.17 with 20% higher effective CPU expert flow, 45.98 with a
20% reduction in the fitted primary KDA/MLA chain terms, 53.80 with a 40%
reduction, and 58.65 with a 50% reduction. Making the worker RPC 20% faster
alone does not change the rounded baseline.

These are hypothetical parameter edits, not implemented speedups. The fitted
primary terms include exposed launch, router, publication and mixer costs;
the fit alone cannot distinguish kernel execution from submission overhead.
A profiler trace is needed for that distinction. Achieving 60 from the measured
39.55 tok/s at unchanged yield requires about 34% less total time per round.
The measurements do not establish such an improvement.

## Native integration repair

After collecting the release timings, the native GPU check exposed a regression
in the TR16 prototype cleanup: it had also removed ownership guards used by the
native contiguous stages. The head stage requested `blk.23.hc_attn_fn.weight`
outside its `[0,23)` range. Commit `bba75e56` restores stage bounds in execution,
verification order, prefetch, CPU service, source registration, NUMA preparation
and cache selection. Ordinary decoding owns the complete `[0,45)` range.

The rebuilt CUDA sm_70 binary passed the full-288 Q23 native checker on both
V100s: split 23, socket-local weights (domains 0/1 and 2/3), 16/15 CPU workers,
22 GiB cache per stage, adaptive cache window 8, 64 prompt tokens and 16 output
tokens, with every proposal forced to miss. Serial and overlapped token IDs,
committed state, route statistics and cache-policy history matched, and resident
cache payloads matched the GGUF bytes. See
[native-pipeline-check-fixed.json](native-pipeline-check-fixed.json). This check
does not compare every logit; the separate full-logit serial checker was not run.

The timing table and fitted profiles remain tied to `83f3cac9` and its preserved
binary. Full 512-token release throughput was not repeated after the native-only
ownership repair. The [legacy numerical-output control](legacy-output-parity-fixed.json) compared
one input token and four emitted token IDs between the preserved and repaired
binaries, with MTP/cache disabled and the required four-domain
`Q2_NUMA_WEIGHTS=0, PACK_FUSE_QUANT=0` memory fallback. The tokens matched.
This is a bounded smoke check; it does not establish
all-settings or all-backend output equivalence.

## Excluded attempts

The asymmetric 10/24 GiB MTP-3 case has an unqualified 42.6158 tok/s median:
one timed trial repeats a 13-token cycle five times, triggering the health check.
That looping trial reports 51.2713 tok/s and is not a valid speed target result.
The original case and all three timings remain in `results.json`; the case is
excluded as a whole rather than selecting its healthy repetitions.

Two larger-primary-cache attempts failed before collecting timings: automatic
sizing is unsupported with MTP, and 15,360 MiB exceeds the 12,288 MiB primary
limit. Their hashes and reasons are in `results.json`. Earlier RAM-staging
attempts failed with a private COW mapping (exit 137) and a sealed writable
file descriptor (`EPERM` on the older kernel). The measured implementation
uses a read-only descriptor and shared read-only mapping; seals, child reads,
budget/hash failure and release of the owner mapping are tested. An initial
real-expert probe incorrectly enabled canonical arithmetic for IQ formats;
it was corrected to match the engine, and only complete 75-row probes are used.
All failed raw attempts are preserved privately and excluded from timing claims.

An earlier repeated-GEN server experiment paid large cache-preparation costs on
each request. It is excluded from the CLI table and from the fitted warm model.
This report does not establish server latency from CLI timings.

## Validation and exclusions

CUDA sm_70 decode, GPU worker, pruner and host/protocol tests built and ran.
The synchronized source also built with HIP gfx1201, MMQ enabled and disabled;
eight host/protocol tests passed in each of those three builds. HIP GPU execution
was unavailable. SYCL was not built because `icpx` was unavailable. All 86 Python simulator,
shared-RAM and mixed-launcher checks pass; legacy TR16 prediction parity is checked
separately from its pre-existing validation misses.

Token health checks detect early EOS and consecutive repetition cycles. They do
not qualify soft acceptance, affinity, factual accuracy or coding quality.
Repeated trials are not token-identical. No quality qualification or 60 tok/s
achievement is claimed. The optional Nsight run failed before decoding: CUDA `--device-info` enumeration
segfaulted under profiler injection, while ordinary enumeration and all clean
benchmark runs succeeded. The raw profiler failure is retained privately; no
Nsight kernel costs are used in the model. Primary GPU costs remain fitted
aggregates. The native pipeline forced-miss checker is a correctness diagnostic,
not a clean throughput sample. Full-288 timing was outside this release-profile
report; its optional staging was stopped before a benchmark ran.

[validation.json](validation.json) records the exact checks, scopes and skipped qualifications.
