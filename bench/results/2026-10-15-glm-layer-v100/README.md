# GLM 5.3f: contiguous layers and one-ahead MTP on two V100s

This experiment ports Maya's contiguous layer ownership and one-ahead MTP
schedule into Strata's existing GGUF decoder. It adds shared model mappings,
per-stage CPU pools, GPU-local socket placement, pinned residual transport,
checkpoint/replay, fixed cache slabs and confirmed-window cache adaptation.
The default remains `gpu_layout=legacy`. See
[configuration and source study](../../../docs/GLM_MULTI_GPU.md).

## Hardware and protocol

The measurements used two NVIDIA Tesla V100-PCIE 32 GB cards at their 250 W
power limit and two Intel Xeon Gold 6240 CPUs: 36 physical cores, 72 logical
CPUs and four NUMA domains across two sockets. The node reports
195,834,796 KiB of system RAM; the job reserved 160 GiB and 72 logical CPUs.
RAM speed was unavailable. GPU order was `0,1`; each stage's submitter and
16 workers occupied disjoint physical cores near its GPU's CPU socket.
The coordinator's core was reserved from the pools. Weights used one shared,
locked COW mapping with socket placement. CUDA was 12.9.86, GCC 12.2.1,
Release `sm_70`, NVIDIA driver 580.178.04. The dependency revision was
`3cf03257f219afbe7334045ff7c6a06ac68c627d`, with a clean dependency checkout.

All timing cases used the frozen C++ parser prompt at exactly 4,096 tokens,
512 generated tokens with EOS stops disabled, context 32,768, prefill panels of 4,096, 32 worker
threads, GPU budget 32,768 MiB per device, target cache 22,528 MiB per stage,
GPU draft experts, draft depth one and CUDA graphs. The measured cache was
fixed (`adapt=false`); adaptive cache correctness was tested separately.
Inherited `STRATA_*` variables were cleared and the complete environment was
recorded. The [opt-in recipe](../../../tools/glm_v100_layer.env) matches the
native profiles. Canonical Q23 adds `STRATA_GLM_CANON=1`.

Each process ran two full 512-token warmups, an excluded short request,
three timed 512-token requests and three validation requests. Startup and
warmup are excluded. Decode throughput is 511 target steps divided by decode
time, including draft, rejection/replay and terminal drain. Response time
includes prefill and decode, excluding model startup. A means serial layer
execution with MTP still enabled; B means overlapped MTP with the same draft,
cut, cache and arithmetic. The order was A1, B1, B2, A2, with fresh process
startup for each entry. Reported arm medians pool six timed requests each.
All 24 timed/validation token sequences are checked across the four processes.
Forced-length timing may continue beyond a natural EOS; the separate quality
suite uses real EOS/role stops and is not performance-comparable.
No compiler, profiler or other experiment ran alongside timed requests.

Placement screening used two explicit cuts per cohort, then selected the
higher measured decode median. This measures only placements that ran and
is specific to this workload. Native Q2 cuts 24 and 26 diverged at output
index 84; native Q23 cuts 22 and 23 also produced different sequences.
Cross-cut rates therefore compare different numerical trajectories. The
same-cut A/B controls must produce identical token IDs. No knobs were
selected using the later held-out coding tasks.

## Timing results

| Model and arithmetic | Source cohort | Cut | Serial tok/s | Pipeline tok/s | Decode change | Response time, serial → pipeline |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Q2, native IQ | v9, allocation A | 24 | 17.58 | 30.25 | +72.1% | 59.42 → 47.27 s |
| Q23, canonical K experts | v9, allocation A | 24 | 20.89 | 34.68 | +66.0% | 55.96 → 46.22 s |
| Q23, native IQ/K | v10, allocation B | 23 | 20.33 | 34.76 | +71.0% | 56.65 → 46.22 s |

All three ABBA cohorts produced identical token IDs across their 24 timed and
validation sequences. The 60 tok/s target has not been reached.

Native Q23 screening measured 31.83 tok/s at cut 22 and 34.80 at cut 23. Cut 22
failed the capped parser gate; cut 23 passed. At the selected cut, the pooled
pipeline median was 34.76 tok/s versus 20.33 serialized (+71.0%); response time
fell 18.4%. Every ABBA process passed all 400,532 parser cases. See
[measurements-native-v10.json](measurements-native-v10.json) and
[comparisons-native-v10.json](comparisons-native-v10.json).

Q2 cut screening measured 30.31 tok/s at 24 and 28.90 at 26. Canonical Q23
measured 34.69 at 24 and 32.93 at 26; those two canonical cuts emitted the
same 512-token sequence. At cut 24, Q2 accepted 455/511 MTP proposals and
canonical Q23 accepted 427/511 in the inspected request. The head processed
568 and 596 rows respectively, including replay, versus 511 target rows on
the tail. These row counts explain why balancing layer counts alone does
not balance stage time. Per-stage times, cache hits and every trial are in
[measurements-v9.json](measurements-v9.json); pooled A/B comparisons are in
[comparisons-v9.json](comparisons-v9.json).

In a native-Q23 cut-23 validation request, 471/511 proposals were accepted
(92.2%). The head processed 552 rows in 12.50 s, while the tail processed 511
rows in 12.46 s plus 1.11 s of draft work. Combined expert-cache hit fraction
was 51.8%. Cache preparation alone took 14.80 s of the reported prefill phase;
the current implementation prepares stage caches serially. These are one
request's diagnostics, not separately repeated microbenchmarks.

The separate interleave processes measured 29.99 tok/s for Q2 and 30.67 for
canonical Q23, versus socket-placement pipeline medians 30.25 and 34.68.
These are single-process placement comparisons, not repeated ABBA memory
experiments. The small Q2 difference does not establish a useful gain.

**The speedups above are against the matched serial layer control.** Layer
prefill was approximately 130–135 tok/s, while the earlier
[legacy V100 recipe](../2026-10-14-glm-v100/README.md) measured roughly
430–465 tok/s with different helper, panel, cache and context settings.
That earlier recipe is not a same-settings control. Current cache preparation
and the lossless layer prefill path need optimization; this report does not
establish a general full-request latency win over legacy.

An inspected canonical-Q23 cut-24 run used 29,564 MiB of tail CUDA memory
(28.9 GiB), leaving about 2.9 GiB free. The inspected native-Q23 cut-23 run
used 29,815.9 MiB (29.1 GiB), with 2,678.25 MiB (2.6 GiB) free. These include
CUDA context use, beyond the decoder's own allocation counters.

## Quality gates

A throughput result is not a coding-quality qualification. The fixed parser
oracle compiles the **final answer after `</think>`**, then exercises 400,532
independent cases. Reasoning text and successful compilation alone do not pass.

The frozen 512-token timing gate failed for native Q2 and canonical Q23 in
both serial and pipeline execution. Those failures are retained as recorded.
For Q2, the first natural stop is at token 431; forced generation then starts
a second answer whose last final-answer block is incomplete at token 512.
The later natural-stop Q2 quality run passed all 400,532 cases, and all 431
IDs exactly matched the earlier v9 timing prefix. Its correct code is preserved
in [q2-native-parser.cpp](q2-native-parser.cpp). Thus the forced-length failure
does not describe Q2's first natural completion.

Canonical Q23 completed C++ but failed valid near-`UINT64_MAX` inputs because
of an extra `(max-9)/10` guard; its failing generated code is preserved in
[q23-canonical-parser.cpp](q23-canonical-parser.cpp). A separate canonical
natural-stop quality suite was not run. Native Q23 cut 23 passed all 400,532
cases in screening and every ABBA process; its exact generated code is in
[q23-native-parser.cpp](q23-native-parser.cpp).

Native Q23 then scored 15/20 held-out contracts in both modes, with the five
failures reaching the cap. Its three training contracts also failed at the cap.
The unchanged 4K parser passed with a natural 290-token completion; all three
long-prefix coding contracts and both 8K/16K recall cases passed. All 29 serial
and pipeline token sequences and oracle results matched exactly. Q2 also scored 15/20 held-out contracts in both modes, with its five failures
reaching the cap. It passed 1/3 training contracts, the natural-stop fixed
parser, all three long-prefix coding contracts and both recall cases. All 29
Q2 serial/pipeline sequences and oracle results matched exactly.

| Native model, both modes | Training | Held-out | Fixed 4K parser | Long-prefix C++ | Recall 8K / 16K | Equal paired sequences |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| Q2 | 1/3 | 15/20 | Pass, 431 tokens | 3/3 | Pass / Pass | 29/29 |
| Q23 | 0/3 | 15/20 | Pass, 290 tokens | 3/3 | Pass / Pass | 29/29 |

The five held-out failures are different tasks for the two quantizations.
[quality-v10.json](quality-v10.json) records every task, cap and oracle outcome.
[quality-prefix-v10.json](quality-prefix-v10.json) proves the natural parser
completions match their timing prefixes. Q2 serial quality ran in allocation B
and pipeline quality in C, on the same node with matching CPU/GPU types and
binary identity checked before resuming; see
[quality-resume-v10.json](quality-resume-v10.json). These quality runs are not
an additional performance comparison.

The broader quality protocol contains three training contracts, twenty
held-out C++ contracts, the unchanged fixed 4K parser prompt with a 1,024-token
cap, three contracts following an unrelated 8K coding prefix, and exact JSON
recall at 8K and 16K. Serial and pipeline run the same 29 tasks with real
EOS/role stops. Missing/incomplete final answers fail; a complete final answer
can pass at the cap, which is recorded. These are correctness checks, not
performance measurements. There is no BF16 quality reference or perplexity
measurement in this experiment.

## Correctness and build scope

[validation-v9.json](validation-v9.json) records full-model GPU checks:

- Q2 and Q23 forced rejection with adaptive windows of eight: identical emitted
  tokens, committed recurrent/indexer state, committed route statistics and
  cache-policy fingerprints against serialized execution. Every resident
  expert payload was read back after admission and compaction and matched
  the original GGUF bytes. There were sixteen payload checks per adaptive case.
- Q23 split versus unsplit logits and committed state were bitwise equal.
  A separate forced-rejection case crossed the 8,192-token boundary from a
  prefix of 8,190 tokens and preserved token/state parity.
- Three- and four-stage Q2 diagnostics matched the full reference using logical
  stages on one physical GPU. These validate ownership and ordering; they
  are not measurements on three or four physical GPUs.
- Clean-HEAD and modified legacy paths produced byte-identical token records,
  prefill logits and final logits for Q2 and Q23. Each comparison generated
  sixteen tokens. This checks the default with the selected legacy settings;
  it is not an exhaustive comparison of every environment combination.
- Compute Sanitizer memcheck reported zero errors for a four-output Q2 run
  with forced misses, CPU draft, no target cache and graphs disabled. Adaptive
  payload/graph testing is separate and was not run under memcheck.

[validation-native-v10.json](validation-native-v10.json) adds native Q23
forced-rejection adaptive parity: 31 rejected proposals, 32 outputs, identical
committed state/policy/routes and sixteen matching payload readbacks.
[serving-frontier-v10.json](serving-frontier-v10.json) records an immediate
GEN/STOP cancellation with zero outputs, cancellation after eight outputs,
then a 32-token replay equal to baseline. The v9/v10 baseline IDs also match.

Six host tests and 39 Python tests passed. The production scheduler's fake
backend covered 240 decode traces and sixteen prefill schedules, including
accept/reject, cache transitions, count/EOS, cancellation, failures and slot
reuse around 3/4, 255/256 and 8191/8192. ASAN/UBSAN passed with LeakSanitizer
disabled because sandbox ptrace support was unavailable. Local CUDA 13.2
`sm_80` and HIP/ROCm 10.1 `gfx1201` builds passed; these were compile-only.
GPU execution used the remote CUDA `sm_70` build. No AMD GPU was available.
The common CPU pool also serves SYCL, but `icpx` was unavailable, so SYCL
compilation is unverified. Older flow/NUMA pool tests require exactly two
NUMA domains and skipped on these four-domain hosts; the new disjoint-pool
test passed. Windows placement and physical three-or-more-GPU execution are
unverified.

Earlier failed implementation attempts remain in the private raw evidence:
individual allocations first changed cache admission across request resets,
then ran out of headroom despite a fixed payload ceiling. The slab allocator
and stable admission ceiling corrected those failures. Those failed attempts
are not qualified measurements.

## Source and evidence

The base commit is `7a33c9c796337b9871f0b3639ece7d8281a5f391`.
[implementation-v9.patch](implementation-v9.patch) and
[source-v9.json](source-v9.json) freeze the entire tested v9 source delta,
including the preexisting optional device-greedy change. The v9 timing binary
SHA-256 is `a0eade4a7082550c2add397a5f9849dfbaae75b37c839baed168883b4b78a699`.

The final engine v10 adds only the
[serving terminal-state fix](serving-frontier-v10.patch): clear the active
request before publishing DONE/ERR so an immediately following STOP targets
the new request. Decoder arithmetic, scheduler and kernels are unchanged.
The v10 timing/quality binary SHA-256 is
`070fa623b07c28d4c8b9420a58a433d326bb189d6314c98c745d03e48d5b63ae`.
[source-final.json](source-final.json) records the final source hashes and
cohort identities; [artifact-manifest.json](artifact-manifest.json) identifies
each shareable file. The v9 and v10 timing cohorts are kept separate.
Documentation and the opt-in
shell recipe were finalized after compilation; the recipe reproduces the
explicit environment recorded by the measurement harness.

[models.json](models.json) records all five streamed model-file SHA-256 hashes
and sizes, verified against the earlier local transfer evidence. No publisher
checksum was available. Neither model weights, full raw logits nor binaries
are included here. Shareable JSON records omit machine-specific paths and use
allocation aliases; raw records and logs remain unchanged in the experiment
scratch area and are identified by SHA-256. Public placement copies omit node
identifiers and must not be used directly as validated runtime plans.

Maya source was studied at
[`4c9f1b1bcb4b40d202fd6c4ed41e5608c78f0f0e`](https://github.com/mw00/project-maya/tree/4c9f1b1bcb4b40d202fd6c4ed41e5608c78f0f0e).
The [source-study table](../../../docs/GLM_MULTI_GPU.md#maya-source-study)
explains which ideas were adapted and which were deferred. Its
[MIT notice](../../../docs/licenses/project-maya.txt) is retained. Maya's
published rates use different models/settings and are not Strata measurements.

## Reproduce

Use a fresh output directory and a Python environment with Strata's tokenizer
requirements. Build the decoder for the GPU being measured and run inside an
allocation with two GPUs, enough locked RAM and the stated CPU allowance:

```sh
python3 bench/results/2026-10-15-glm-layer-v100/reproduce.py \
  --decoder "$DECODER" --model "$Q2_MODEL" --profile q2 \
  --cuts 24 26 --output-dir "$RESULTS/q2"
python3 bench/results/2026-10-15-glm-layer-v100/reproduce.py \
  --decoder "$DECODER" --model "$Q23_MODEL" --profile q23-native \
  --cuts 22 23 --output-dir "$RESULTS/q23-native"
```

`--print-commands` displays the full timing commands and environment without
running them. The script selects a measured plan and runs ABBA; it retains
failed forced-length parser gates and exits 2 if any such gate fails. Q2 can
fail this gate after EOS while its natural-stop parser passes, as described
above. Do not equate a
completed timing record with passing quality. Use `q23-canonical` only to
reproduce that separately reported arithmetic cohort.

For each mode, use the selected explicit cut for the later quality suite:

```sh
python3 bench/results/2026-10-15-glm-layer-v100/reproduce_quality.py \
  --decoder "$DECODER" --model "$Q23_MODEL" --profile q23-native \
  --pipeline none --cut 23 --output "$RESULTS/q23-quality-serial"
python3 bench/results/2026-10-15-glm-layer-v100/reproduce_quality.py \
  --decoder "$DECODER" --model "$Q23_MODEL" --profile q23-native \
  --pipeline mtp --cut 23 --output "$RESULTS/q23-quality-pipeline"
```

Repeat with profile `q2`, its model and cut 24. Compare every task's token IDs
as well as its independent oracle result. The quality runner preserves failed
or capped answers; its process exit alone does not mean all tasks passed.
