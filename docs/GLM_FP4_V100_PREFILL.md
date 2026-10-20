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
