# Native FP4 prefill on two V100 GPUs

Decode tuning is paused at **41.24 tok/s**. The opt-in
[grouped working profile](../configs/experimental/glm53f-fp4-dual-v100-grouped.json)
keeps MTP depth three, bounded decode routing and the grouped resident FP4 core.
Its prefill baseline is **519.34 tok/s**, measured with three full warmups and
three timed 8,192-input / 512-output requests on two Tesla V100 PCIe 32 GB GPUs,
a dual Xeon Gold 6240 and a 181 GiB allocation. Clocks were unlocked; RAM speed
was not measured. The original 5.26 tok/s reference remains available.

The active target is **800+ prefill tok/s**. At 8,192 input tokens that requires
reducing prefill from **15.77 s to 10.24 s**, a **1.54×** throughput gain. All
checkpoint weights and prefill routes are retained during these experiments.
Decode settings are fixed, and Fresh20 remain unscored.

## Streamed prefill (opt-in)

`STRATA_GLM_FP4_PREFILL_STREAM=1` uploads each layer's uncached routed experts through a ring of upload frames on
each GPU, while the previous layer's attention, router and shared expert run, instead of one expert at a time inside
the routed step. The frames are borrowed from the GPU's cached layer-45 draft experts and restored after layer 44.
Prompts of 4,096 tokens or more use it; each 8,192-token batch of a longer prompt streams again. It requires two
SM70 GPUs, MTP, the arena, registered RAM, GPU reduction, device routing and async uploads, as the grouped profile
has. The measured setting adds these to the grouped profile's environment
([streamed profile](../configs/experimental/glm53f-fp4-dual-v100-streamed.json)):

| Switch | Value | What it does |
| --- | --- | --- |
| `STRATA_GLM_FP4_PREFILL_STREAM` | `1` | The streamed prefill above |
| `STRATA_GLM_FP4_PREFILL_STREAM_FRAMES`, `..._FRAMES1` | `96`, `64` | Ring frames on GPU 0 and GPU 1 (8-136; 64 by default) |
| `STRATA_GLM_FP4_PREFILL_STREAM_HOLD` | `1` | Hold GPU 0's next-layer uploads until GPU 1's partial sum has crossed the shared link |
| `STRATA_GLM_FP4_PREFILL_STREAM_EARLY` | `1` | Send GPU 1 its FP16 activations before the router and shared expert |
| `STRATA_GLM_PREFILL_LEAN_HOST` | `1` | Read each token's embedding row once, on several threads, and copy the hidden rows for MTP on a helper thread |

Measured on two Tesla V100 PCIe 32 GB, a dual Xeon Gold 6240 and a 181 GiB allocation, clocks unlocked, RAM speed not
measured; each pair is the same binary and job, three full warmups and three timed 8,192-input / 512-output requests
per side, zero timed major faults:

| Pair (context) | Control prefill / decode tok/s | Streamed prefill / decode tok/s | 512-token outputs |
| --- | ---: | ---: | --- |
| source v532 (12,288) | 522.18 / 41.17 | **782.84 / 41.73** (+49.9%) | identical |
| source v538 (12,288) | 522.86 / 41.47 | 779.79 / 41.50 | identical |
| source v542 (12,288) | 519.69 / 41.39 | 781.47 / 41.34 | identical |
| source v553 (12,288) | 519.79 / 41.23 | 783.71 / 41.39 | identical |
| source v558, FP16 latent cache (12,288) | 524.06 / 41.61 | 791.02 / 41.99 | identical (both FP16) |
| source v570, 1M context, FP8 cache (see [long context](GLM_FP4_V100.md#long-context-opt-in)) | 515.66 / 38.04 | **807.31 / 38.10** | identical |

The reused coding panels passed 8/8 long-context and 8/8 natural-length cases on both sides of the v532, v538 and
v542 pairs. The long-context panel's answers differ from the control's from its third case on: a streamed prompt
leaves different experts cached on GPU 1 for the next prompt, and CPU and GPU expert arithmetic differ. The gate
used is the documented one (identical 512-token pair outputs and both panels passing); bit-identical multi-prompt
outputs were not reached. At a 12,288-token context the 800 tok/s target is not met (779.79-791.02); the 1M-context
pair, whose decode cache is smaller, measured 807.31.

Long prompts stream every 8,192-token batch. Single requests at a 65,536-token context with
`STRATA_GLM_ACTIVE_POOLS=1`, prefill tok/s for 16K / 32K / 64K prompts: control 505 / 495 / 489 (source v552),
streamed 635 / 583 / 559 (source v556; not a timed pair). The
gain shrinks with length because GPU 1 then uploads many experts on demand inside the routed step (a 1,048,000-token
prompt: 195,921 demand uploads on GPU 1 against 23,814 on GPU 0); that is the next limit.

A fresh diagnostic uses the settled profile on source 377: two full warmups,
a discarded request and one 8,192-input / eight-output request. Its first eight
output IDs match the settled reference, with zero timed major faults and swap.
Its instrumented prefill took **16.0435 s**; this is a phase diagnostic, not a
throughput qualification.

| Recorded event intervals | GPU 0, s | GPU 1, s |
| --- | ---: | ---: |
| Matrix H2D | 6.1483 | 4.6545 |
| Packing and compressed scales | 0.9184 | 0.6825 |
| FP16 operand reconstruction | 1.1666 | 1.0087 |
| GEMM | 3.4517 | 1.8662 |

These intervals overlap and can include host enqueue gaps. They are not GPU
activity unions and must not be added to estimate request latency. Routed
calls total 13.7011 s including 4.4934 s waiting on their input dependencies.
The explicit request markers exclude the later cached MTP-prefix preparation.
The [measurement record](glm_fp4_v100_dflash2_measurement.json) stores the raw
result/log hashes and the per-layer and fixed-projection intervals. The
[simulator profile](../tools/sim/data/glm_fp4_v100_target50.json) imports the
diagnostic without changing its selected reference or fitting capacities.

The first direct-operand component skips the packed FP4 code intermediate.
It passed **70 bitwise cases, 28 changed-input graph checks, CUDA memcheck and
racecheck**. Across the two expert matrix shapes, five alternating 32-call
samples measured **2.0084×** faster preparation. Its allocating model prototype
failed default parity: output first differed at token 32, and CPU expert-group
counts also changed. Full model timing was skipped. The component result does
not establish a usable model speedup or identify the cause of the mismatch.

A second component fuses compressed CSF32 scale decoding with canonical FP4 to
FP16 reconstruction. An independent C11 scale decoder checks its fixture.
It passed **40 bitwise cases, eight changed-input graph checks, invalid-scale
and invalid-slot checks, CUDA memcheck and racecheck**. Five cyclic 32-call
samples measured **3.1444×** faster preparation than the original three-step
path and **1.6938×** faster than separate scale decoding and direct reconstruction.
These component timings exclude matrix transfer and model scheduling.

The workspace revision borrows the existing 96 MiB operand workspace and its
two-slot retirement events. Both default-off and default-on model runs preserved
all 128 original reference tokens, with zero timed major faults. Its repeated
model comparison measured **518.80 / 524.34 prefill tok/s** and **41.44 / 41.42
decode tok/s** for control / candidate. Each side used three full warmups and
three timed 8,192-input / 512-output requests; all six streams matched the mild
reference, with zero timed major faults or swap. The **1.07%** prefill gain
missed the 3% gate, so coding and whole-model instrumentation were skipped and
the option remains unselected. A separate functional comparison then passed
all eight long-context and all eight natural-length coding tasks on both sides,
with four ordered KDA partitions throughout. This narrows the later combined
regression without establishing its cause or changing the failed speed gate.
CUDA and HIP builds and six host checks per backend passed; HIP GPU execution
was not tested.

An existing unordered eight-partition KDA prefill path measured **520.19 /
531.70 prefill tok/s**, a **2.21%** gain versus four ordered partitions. Each
side used three full warmups and three timed 8,192-input / eight-output
requests. All six eight-token prefixes matched, with zero timed major faults
or swap. This is a prefill-only screen; eight outputs do not establish decode
throughput or coding quality. It also missed the 3% gate. The combination subsequently measured **536.54 prefill / 42.01 decode tok/s**
with three full warmups and three timed 8,192-input / 512-output requests.
That is **3.42%** more prefill throughput than the earlier same-day 518.80
control on the identical binary. Two screens and a profiler capture intervened; clocks were
unlocked; this was not a fresh adjacent pair. All three outputs were identical
and passed length, EOS and repetition checks, with zero timed major faults or
swap. The long-context reused coding panel passed only **7/8**: `argmin_first` did
not produce exactly one complete C++ code block. Natural-length coding was
skipped, and the combination is rejected as a usable profile. A separate
whole-model 512-input / eight-output CUDA memory check passed with zero
reported errors. The first checker attempt timed out attaching; the retry used
the identical binary on disk with RAM executable relaunch disabled and explicit
process attachment. Instrumented timings are excluded. The 800 tok/s target
still needs **1.49×** the rejected candidate rate.

Two existing MLA options were screened on the public build, keeping the settled
decode settings and four ordered KDA partitions:

| MLA prefill option | Prefill tok/s | Gain over control |
| --- | ---: | ---: |
| Original 64-query tiles | 519.57 | — |
| 256-query tiles and compact FP16 scratch | 526.61 | 1.35% |
| FP16 score/value, index and softmax chain on 64-query tiles | 530.43 | 2.09% |

Each option used three full warmups and three timed 8,192-input / eight-output
requests. All nine timed prefixes matched, with zero timed major faults or swap.
These are prefill-only screens; eight outputs do not qualify decode throughput
or coding quality. Both options missed the 3% gate, so their full-length timing
and coding stages were skipped. Neither is selected. The FP16 chain changes
prefill precision. A separate combination measured **535.85 / 519.53 prefill
tok/s**, candidate / fresh control, in the same eight-output protocol. Its
3.14% gain qualified it for a fresh full-length comparison; the individual
failed gates remain failed.

The full comparison measured **518.70 / 524.91 prefill tok/s** and **41.76 /
39.57 decode tok/s**, control / candidate. Each side used three full warmups
and three timed 8,192-input / 512-output requests, with identical repetitions
within each side, zero timed major faults or swap, and clean uninstrumented
exits. Candidate outputs changed but passed length, EOS and repetition checks.
The **1.20% prefill gain** missed the 3% gate, and **5.25% slower decode** also
failed the decode guard. Coding and whole-model memory checks were skipped.
This combination is rejected; the eight-output screen did not predict its
full-length result. No new prefill preset is selected, and 800 tok/s remains
unmet.

A separate Nsight Systems 2025.1.3 capture of the settled baseline measured
these overlap-corrected GPU activity unions during one diagnostic 8,192-token
prefill. It followed two complete warmups and a discarded request; its first
eight output IDs matched the reference.

| Prefill activity union | GPU 0, s | GPU 1, s |
| --- | ---: | ---: |
| Kernels | 9.6728 | 3.4046 |
| Copies | 6.7831 | 5.2130 |
| All GPU activity | 14.5526 | 7.1993 |
| GPU idle within 16.283 s | 1.7304 | 9.0837 |

The phase boundary uses the profiler-start endpoint and engine prefill timer,
with some host logging/timer uncertainty. Copies and kernels overlap; their
unions must not be added. Benchmark residency monitoring observes the profiler
launcher, so this capture supplies no native residency qualification.

Keeping the recorded GPU durations while deleting all idle time would reach
only **562.9 prefill tok/s**. Keeping only the primary kernel durations would
reach **846.9 tok/s**. At 800 tok/s those kernels leave just **0.567 s** for
remaining costs. These are conditional scenarios for the trace, not hardware
rooflines or speed forecasts. The next scheduling work must improve copy/compute
overlap and distribute more work to the secondary GPU. Large raw traces remain
outside Git; their sizes and hashes are recorded with the compact activity data.

The CUDA operand APIs and synthetic probes are available in the repository;
the experimental host scheduling remains outside the active decoder path.
The kernels decode canonical FP4 codes directly and fuse CSF32 exception
handling with coalesced FP16 writes, preserving the original half roundings.
No model or checkpoint download is needed for these component checks:

```sh
cmake --build build-glm-v100 --target glm_fp4_raw_operands_test glm_fp4_fused_operands_test
build-glm-v100/glm_fp4_raw_operands_test --bench
build-glm-v100/glm_fp4_fused_operands_test --bench
```

The probes compare original preparation, direct reconstruction and fused
reconstruction on the same allocations. Their timings do not predict model
tok/s. A fresh build of the public source measured **3.1452×** original/fused
and **1.6936×** separate/fused preparation, with the same 40 bitwise and eight
graph cases and clean CUDA memory/race checks. CUDA and HIP rebuilds each
passed six host tests. The public CUDA decoder also reproduced all 128 original
default output tokens. HIP GPU execution was not tested; the separate SYCL
backend does not compile this CUDA path. The original native executor and all
selected presets retain their existing execution paths.

## Current simulator report

The simulator now exposes these observations in `current_prefill`, separate
from the historical `decode` and `prefill` resource-accounting blocks. The
[current analysis](glm_fp4_v100_prefill_analysis.json) keeps the settled
**41.24 decode / 519.34 prefill tok/s** profile, all seven prefill experiment
comparisons and their original gates, and the newer activity unions. A short
screen has no decode throughput in this summary. The separate fused-operand
coding follow-up does not turn its failed speed gate into a pass. No rejected
candidate is selected and no component ratio rescales model throughput.

```sh
python3 tools/glm_runner_sim.py native-fp4 \
  --profile tools/sim/data/glm_fp4_v100_target50.json \
  --target-decode 40 --target-prefill 800 \
  --output docs/glm_fp4_v100_prefill_analysis.json
```

| Prefill target | Budget for 8,192 tokens | Gain required from working profile | Time to remove |
| --- | ---: | ---: | ---: |
| 800 tok/s | 10.240 s | 1.54× | 5.534 s |
| 1000 tok/s | 8.192 s | 1.93× | 7.582 s |

The 800 target has only 0.567 s beyond the trace's retained primary-GPU
kernel union. The 1000 target requires at least **1.181×** faster primary
kernels even if every other cost were free. Neither calculation predicts an
implementation's throughput. Current grouped-profile work counters have not
been qualified for a new resource roofline; the simulator therefore retains
the historical resource blocks with their original workloads and does not
substitute current tok/s into them. See the
[roofline notes](GLM_FP4_V100_ROOFLINE.md#current-grouped-profile-and-prefill-gaps)
for the distinction.

## Further prefill optimization opportunities

The next work should change the measured critical path. The following order
is an experiment priority, not an expected speedup. Keep the working decode
profile and four ordered KDA partitions fixed while isolating each change.

| Priority | Evidence | First implementation to test | Required observation |
| --- | --- | --- | --- |
| 1. Overlap matrix transfer and computation | GPU 0 has 6.783 s of copy activity and 14.553 s of total activity. Deleting idle alone reaches only 562.9 tok/s. | Prepare the next expert matrix in bounded, preallocated upload/reconstruction slots; use explicit events between upload, reconstruction and GEMM streams. | A smaller primary all-activity union and lower full-request latency, without extra matrix transfers, faults, workspace growth or decode loss. |
| 2. Use GPU 1 for primary attention work | GPU 1 is idle for 9.084 s of the 16.283 s trace; GPU 0 has 9.673 s of kernels versus 3.405 s on GPU 1. | Partition independent MLA query tiles and their fixed projections between devices with explicit input, KV and result ownership. | Less primary kernel time after including peer copies and synchronization; preserved output and cache state. |
| 3. Integrate exact operand reconstruction into the schedule | Standalone fused preparation is 3.145× faster, but the workspace prototype improved full-model prefill only 1.07%. | Reuse bounded operand slots and retirement events, preserving the existing half roundings; profile exposed reconstruction time rather than extrapolating the component ratio. | A full-length gain above the gate on a fresh adjacent pair; both coding panels pass. |
| 4. Reduce ordered KDA and MLA launch/scratch costs | The prepared four-partition KDA kernel occupies about 0.998 s on GPU 0. Unordered rows failed coding in combination; the larger-tile/FP16-chain MLA combination failed full-length guards. | Batch launches and reuse scratch while keeping ordered recurrence; isolate MLA gather/cast fusion from score/value precision and decode cache layout. | Reduced relevant kernel/idle time, repeatable full-length outputs and unchanged settled decode throughput. |

For transfer overlap, avoid per-expert allocation and global synchronization
inside the hot loop. A slot must remain alive until its consuming GEMM has
finished; reusing a code/scale or FP16 buffer after only the upload event is
insufficient. Record bytes and event dependencies per slot, and distinguish
compressed source wire extents from expanded expert allocations. Earlier
queue, bank and pipeline variants already failed timing or quality checks;
another queue depth or double-buffer flag alone is not evidence of a gain.

Attention redistribution needs native ModelOpt integration. The current
[`dense()` implementation](../src/program/glm_fp4_native.inc) chooses
`contexts.front()`. The generic secondary MLA tile machinery in
[`glm_decode.cpp`](../src/program/glm_decode.cpp) is not initialized by the
native ModelOpt `configure_prefill()` branch, which returns after constructing
the native executor. Merely enabling a generic split option or changing expert
ownership does not move this attention work. A bridge must handle projected
inputs, KV/index state and returned query tiles, preserve per-query GEMM shapes
and reduction order where exactness is intended, and leave MTP history and
rollback valid. Measure peer-transfer cost and contention with expert work;
the observed idle interval is not a freely available speedup.

KDA launch improvements cannot by themselves remove the 5.534 s needed for
800 tok/s: the recorded prepared kernel is much smaller than that gap.
Keep the recurrence order while trying memory-aware tiling or graph reuse.
For MLA, the combined eight-output screen gained 3.14%, but its full run gained
only 1.20% in prefill and lost 5.25% in decode. Separate prefill scratch lifetime
from decode cache layout before revisiting those options. The FP16 chain also
changes arithmetic, so a mild-prefix match cannot substitute for coding checks.

Memory accounting is part of each experiment. The recorded 42-file resident
weight image is **177,463,841,024 bytes** (about 165.3 GiB), within the **181 GiB**
allocation. Keep that allocation and the two 32 GB device limits; budget host
runtime, pinned staging, KV/history, dense weights, decode expert admission and
all operand slots together. Keep the registered resident weights in place.
Reserve and reuse stable arenas rather than growing scratch unnoticed or
displacing the decode keep set. Failed allocating prototypes remain failures;
their component timings do not justify changing the active executor.

Each candidate should follow the existing staged protocol:

1. Build the affected backends and check operand bits, changed-input graph
   replay, bounds and memory ownership. Check the original 128 default tokens
   when a runtime path changes; keep new behavior opt-in.
2. Run a fresh same-build adjacent control/candidate pair on the two V100s:
   three complete warmups, one discarded request, then three timed requests
   with 8,192 inputs and 512 outputs on each side. Keep the 32-worker setting,
   resident weights, zero timed major faults/swap and clean uninstrumented
   exits. Record source, binary, environment and config hashes. Clocks were
   unlocked and RAM speed unmeasured in the results above; disclose either
   change. An eight-output screen can prioritize candidates but cannot
   qualify decode throughput or replace this pair.
3. Require at least 3% faster prefill, decode at least 40 tok/s and within 3%
   of the fresh control before advancing. Then run both reused eight-case
   long-context and eight-case natural-length coding panels on both sides.
   Report changed outputs, EOS, length and repetition checks separately from
   coding quality. Keep Fresh20 unscored until a policy is frozen.
4. Capture activity only after a candidate passes those gates. Compare phase
   boundaries, unions, transfers and workspace high-water marks. Keep profiler
   timings separate from qualifications. A 512-input / eight-output memory
   check covers that request only; it does not qualify long decode. Preserve
   failed attempts and raw evidence hashes as well as successful runs.

The present cleanup changes only formatting in the CUDA operand kernels and
probes; their C++ token sequences and benchmark order are unchanged. Historical
source and binary hashes in the frozen measurement records still identify the
measured builds. CUDA 12.9 SM70 compilation and linking passed; standalone
SM70 disassembly matched after normalizing translation-unit symbol names.
Five relevant host tests passed; the two operand GPU probes skipped because
the local driver was unavailable. HIP and SYCL do not compile the changed
CUDA files. The Python run completed 151 tests with one existing routing-trace
test skipped, and the generated analysis reproduced byte for byte. This
simulator/documentation update adds no new GPU throughput measurement and
promotes no prefill option.
