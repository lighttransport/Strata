# Q3 decode and prefill experiments on the v100s node

On the same physical GPU pair, affinity 0.08/MTP3 reaches **43.61 decode
tok/s and 562.05 prefill tok/s**. This is 5.7% faster decode and 5.6% faster
prefill than paired same-day old-build controls. Greedy reaches **22.97 /
563.20 tok/s**: decode is 0.6% slower, while prefill is 5.3% faster. Both
candidates reproduce their timed tokens across fresh processes on all three
prompts. The 50 decode / 700 prefill targets are not reached.

Both frozen candidates miss their HumanEval preservation floors: greedy
scores 153/164 against a floor of 154, and fast scores 150/164 against 153.
The fast profile also fails the original HumanEval release criteria. Keep
the previous greedy Q3 profile as the quality reference; these tuning
profiles remain explicit experiments.

This experiment retains the original 288 experts and all original dense
formats in GLM-5.3-Flash UD-Q3_K_XL. The merged file is 147,535,921,504
bytes (137.4 GiB), SHA-256
`7800e21bc527db7ae6eaea36a190711c3c500109c2e5f5904017f1d2d9089b5a`.
It uses the byte-preserving model and socket-row plan from the
[previous Q3 quality run](../v100s_q3_20261017/README.md).
There is no additional pruning or requantization. Decode exchange remains
FP32; the presence-based `STRATA_GLM_MLA_F16` switch is absent.

The v100s node has two Tesla V100 PCIe 32 GB GPUs, two Xeon Gold 6240 CPUs,
36 physical cores/72 logical CPUs, and four SNC NUMA nodes grouped into two
physical sockets. Allocations request both GPUs, 72 logical CPUs and 160 GiB
RAM for at most three hours each. RAM speed was unavailable. Each renewal
checks the machine class and GPU NUMA roots, verifies the entire shared
model's SHA-256, and samples ten weight panels for socket placement.

## Timing

| Mode | Build/profile | Decode tok/s | Prefill tok/s | Decode repeats across fresh processes |
| --- | --- | ---: | ---: | --- |
| Greedy, affinity 0 | Previous build/profile | 23.12 | 535.09 | No |
| Greedy, affinity 0 | Frozen candidate | 22.97 | 563.20 | Yes |
| Affinity 0.08, MTP3 | Previous build/profile | 41.25 | 532.47 | No |
| Affinity 0.08, MTP3 | Frozen candidate | 43.61 | 562.05 | Yes |

For each prompt/build/mode, the result is the median of six timed replays
from two fresh processes, followed by the geometric mean of the three
prompt medians. [final-timing.json](final-timing.json) retains prompt results,
process medians, trial ranges and hashes. The old controls finish normally
but their adaptive-cache output varies; their rates are observed controls,
not repeatable decode qualifications. These comparisons change both the
build and the explicit profile; they do not isolate a single kernel's gain.

Each process uses 8,192 input tokens and generates 512 tokens. One full
decode warmup precedes three timed state replays; an independent prefill
warmup precedes three timed prefills. Each timed prefill must reproduce the
warmup's full logits and route counts. Both modes report 511 tokens divided
by decode time, excluding the initial token selected from prefill logits.
Initialization, shared-model copying, checksum verification, cache admission
and warmups are outside timing.

Candidates are selected using only `train-code`. Profiles are frozen before
running `holdout-code`, `holdout-chat` or coding-quality generation. The final
matrix pairs the previous decoder/worker and frozen candidates on all three
prompts, twice in fresh processes: 24 process runs, 72 timed decode replays
and 72 timed prefills. Existing public
[prompt IDs](../../fixtures/glm_tr16_20261009/prompts/8k/) are used unchanged.
[timings.json](timings.json) records every completed and failed training or
final attempt, individual trial times, stream hashes and configuration
changes. Same-day old-build controls use the same hardware and timing
protocol as the candidates.

Adaptive cache learning can change residency between replays. Changing
streams are reported as complete observed timings and fail the decode
repeatability gate. No such result qualifies the 50 tok/s decode target.
Prefill repeatability and the 700 tok/s prefill target are recorded separately;
coding quality is a further check.

## Implementation and rejected experiments

The fused Volta gate/up kernel now honors the model's asymmetric SwiGLU
clamp when `STRATA_GLM_FUSED_CLAMP=1`. `STRATA_SM70_TABLE=1` selects this
kernel on Volta. The default selection and old unclamped mode are preserved.
The native expert microbenchmark reads the actual model geometry, expert
count and clamp, and supports scaled activations for a stress comparison.

A separate opt-in CPU translation unit supports clamped IQ rows on
AVX512F/BW/DQ/VL CPUs without VNNI/VBMI. It supports separately mapped gate
and up rows and uses exact parity-based sign expansion for IQ2/IQ3.
`STRATA_GLM_IQ512_CLAMP=1` enables it; the forced ISA cap and
`STRATA_NO_IQ512` disable it. The older unclamped AVX-512 unit retains its
VNNI/VBMI guard. The clamped CPU path changes the floating-point reduction
order and requires independent quality testing before use elsewhere.

On these Xeons the new CPU path and both gather variants lost full-run
throughput against default AVX2, despite passing correctness checks. They
are excluded from the frozen experimental profiles. Lower CPU worker
counts also regressed. Cache admission, inflight upload count, thread task
size, prefetch capacity, primary prefill share, worker placement, upload
stream, panel size and bucket size were tested without changing affinity or
MTP acceptance. Recorded scheduling variants, failed attempts and ineligible
measurements remain in the matrix evidence.

The selected profiles both use static cache admission, a primary prefill
share of 11/18, eight prefetch groups, 8,192-token batches and 512-row panels.
Greedy uses four CPU tasks per thread and the default GPU expert kernel;
fast uses twelve and the clamped Volta table kernel. Other affinity and
MTP parameters stay fixed. The selection and aggregation rule were frozen
at 2026-10-17 03:26:06 UTC, before held-out timing or quality generation.

Instrumented 128-token greedy diagnostics measured about 20–21 ms of CPU
service and 17.7 ms between GPU service points per decode step, plus the
head and tail. These counters are not a throughput qualification. The
worker's lifetime idle counter includes initialization and prefills; it is
not a utilization measurement for the timed decode region.
[diagnostics.json](diagnostics.json) preserves the separate warmup and timed
counters.

## Coding quality

| Frozen candidate | HumanEval | C++ | Optimization preservation |
| --- | ---: | ---: | --- |
| Greedy, affinity 0 | 153/164 | 24/24 | Fail: HumanEval floor 154 |
| Affinity 0.08, MTP3 | 150/164 | 21/24 | Fail: HumanEval floor 153 and two lost C++ passes |

Each mode reaches the 2,048-token cap on one HumanEval task and no C++
tasks. Greedy loses HumanEval/130 and /163 and gains /128 against the
previous greedy reference. Fast loses /39, /108, /129 and /134 and gains
/116 against the previous fast reference. All input token hashes match
the previous run.

The fast C++ failures are `checked_slice` (mandatory) and `mul_i64`
(held-out), both oracle failures, and `parse_hex` (held-out), which emits
two complete C++ blocks. The frozen extractor labels that last output an
incomplete-code-block failure; it is 575 tokens and stops normally. Its
unchanged total of 21 passes hides two lost previous passes and two gains:
`parse_i64` and `replace_all` now pass. No output was repaired or selectively
regraded. All four parser contract reviews pass; each original 4K parser
also passes all 400,532 oracle cases.

The original fast release gate fails both HumanEval criteria, the mandatory
C++ criterion and the held-out C++ criterion. Its original-parser manual
review passes. Neither new profile qualifies for promotion under all the
planned gates. The previous greedy Q3 profile remains the quality reference.

Both modes use chat format, low reasoning, temperature zero and one sample
per task, with at most 2,048 generated tokens. The fast mode fixes route
affinity at 0.08, MTP depth at 3, soft margin at 6, top-k at 8, at most three
soft accepts and repeat guard at 3. Greedy uses affinity zero and no
speculation. All 164 pinned HumanEval tasks and all 24 unchanged C++ tasks
are generated for each mode: 376 samples total.
The first 23 C++ inputs have 1,024 tokens; the original parser has 4,096.

Generated programs run locally under the existing Bubblewrap graders,
with isolated user, process, filesystem and network namespaces. The frozen
extractors and oracles are unchanged; failed, truncated or multi-block
outputs are not repaired or selectively regraded. Both `parse_u64` outputs
per mode receive a manual contract review, including `noexcept`, allocation,
ASCII handling, overflow and preserving the output on failure.
[quality.json](quality.json) records per-task outcomes and artifact hashes.

The optimization preservation check requires at least 154/164 HumanEval and
all 24 C++ passes for greedy; fast requires at least 153/164 and every one of
the previous fast profile's 21 C++ passes. The original fast release gate is
unchanged and reported separately. The earlier fast failures in `parse_i64`,
`parse_hex` and `replace_all` are not waived. No profile is retuned after
viewing held-out timing or quality results.

A transient scheduler-query failure ended the shared RAM owner during the
first quality attempt. Its controller subsequently cancelled the GPU step
after 54 greedy HumanEval samples. The resumed run retains those samples
and their original manifest; [generation.json](generation.json) checks the
saved prefix hashes and records the interrupted observer separately. The
operational fix caches one verified allocation deadline, with retries for
the initial query. Inference binaries, profiles, quality runners, extractors
and oracles stay frozen.

## Memory and validation

One sealed shared model is locked by its owner per allocation. Routed row
halves are placed on their owning sockets and interleaved within each
socket's SNC nodes. The primary and worker use explicit CPU masks; these
must be adapted to the actual cpuset and PCI/NUMA topology on other machines.
Both GPUs reserve 31,616 MiB pools (30.875 GiB each). MTP, attention and
staging reduce the bytes available for cached experts. Reservation is not
populated weight capacity. [memory.json](memory.json) records resident
placement proofs, sampled NVML usage and separately observed process swap.
Summed RSS double-counts shared pages; the launcher's setup-only PSS sample
is not a full-run host RAM peak. No deduplicated peak was measured.
Across setup, timing and coding checks, sampled NVML peaks are 32,350 MiB on
GPU 0 and 32,448 MiB on GPU 1; minimum sampled free memory is 145 and 47 MiB.
No process swap is observed in the RAM-owner, timing or coding samples.
The configured 12,288-token context limit was not saturated: timing covers
8,192 input plus 512 output tokens, and coding inputs are at most 4,096.

Default-output validation compares the old and new decoder with all STRATA
switches unset on the same sealed model: 64 input tokens, 16 output tokens,
and complete prefill and final 154,880-float logit arrays. The probe uses CPU
decode experts, zero decode-cache admission and single-GPU F16-batched
prefill. All compared bytes are identical; this probe does not cover every
cache layout or model.
The GPU clamp test compares mode 8 with mode 0, both with clamping enabled,
on both cards, four layers, token widths 1–8 and activation scales 1 and 64:
128 finite, byte-identical layer comparisons, including the wider lookup
verification windows. The old kernel's high-amplitude
negative control reproduces three differing layers.

Seven CPU checks execute successfully on the target Xeons, including the
portable sign-table test, flag values 0/1/2, fallback, forced AVX2 and gather.
Real-model CPU parity also passes. The initial synthetic test used an
output-relative tolerance that rejected four cancellation-sensitive valid
reductions. Its raw failure is preserved. The corrected independent oracle
uses ggml-dequantized weights and actual Q8_K activations, double accumulation
and propagated floating-point error intervals. Runtime kernels were not
changed for that test correction.

The remote build began with a fresh CUDA sm70 Release build. The final CPU
experiment rebuilds all CPU objects, reuses hashed unchanged CUDA/GGML/host
and program objects, and relinks the executables. A subsequent EOF-only
whitespace cleanup in the shared CPU include produces byte-identical CPU
objects in both local backend builds; the compiled and final source hashes
are recorded separately. Local affected targets compile for CUDA sm80 and
HIP gfx1201; the corrected GPU kernel also compiles
for HIP gfx906. Three C11 host tests and the portable sign test pass per
local backend. Five CPU AVX-512 execution tests skip locally because that CPU
lacks AVX512BW; the actual Xeon execution checks pass. The 60 focused Python
tests pass. SYCL compilers are unavailable; Windows and HIP GPU execution
are untested. [validation.json](validation.json) binds the exact source,
binaries, compiler identity, reused artifacts and validation scope.

The [greedy experiment](../../../configs/experimental/glm53f-ud-q3-k-xl-v100s-opt-greedy.json)
and [affinity 0.08/MTP3 experiment](../../../configs/experimental/glm53f-ud-q3-k-xl-v100s-opt-aff008-mtp3.json)
retain the frozen functional settings. Their metadata records the actual
scores and failed quality gates. The [previous greedy reference](../../../configs/experimental/glm53f-ud-q3-k-xl-v100s-greedy.json)
is unchanged. The candidates change no installer or server defaults.

Both final profile files pass actual API configuration loading with mocked
weights, engine and listener: positional arguments, chat stop handling and
environment forwarding match the measured settings. This is a configuration
check, not an HTTP or GPU serving smoke test. Engine source changes are in
commit `1cfc6fb4`; [validation.json](validation.json) distinguishes the exact
compiled source from the EOF-only cleanup and binds the final profile hashes.

After all 376 samples were saved, the owned GPU allocation was cancelled,
renewal disabled, both experiment controllers stopped and the stale resident
descriptor removed. The shutdown check verifies zero owned queued/running
jobs. Model files and frozen raw evidence are retained. The local grading
observer also exited after all four suite grades were saved.

Use the previous Q3 report's shared-resident-owner recipe with the selected
profile path. It clears inherited STRATA environment switches before
launching. Static cache requires the presence-based `STRATA_GLM_TIER_ADAPT`
switch to be absent; setting it to `0` does not disable it.

Connection settings, private filesystem paths, model weights, datasets,
binaries, raw logit dumps and generated programs are excluded from this report.
