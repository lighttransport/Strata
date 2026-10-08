# B550 Q3 expert-cache study — 2026-10-08

The requested next-stage study used the existing
`/mnt/disk01/models/glm53f-reap/GLM-5.3-Flash-REAP50-Q3_K_M.gguf` on Ryzen 9
3950X / RX 9070 XT. It collected real decode routes, replayed cache policies,
and measured read-only expert-sized SSD I/O. **The result supports an exact
exclusive-cache implementation experiment; it does not demonstrate a runtime
speedup yet.** No model tensors or routing scores were changed.

## Actual inference diagnostics

The model has 67.18 GiB of main expert tensors: 42 layers, 144 experts per layer,
top 8. It selects about 3.732 GiB of logical expert weights per decode step
before GPU cache hits. This is a smaller stored model but more active weight
bytes per token than the full Q23 recipe (103.55 GiB stored, about 2.876 GiB
selected). Do not extrapolate its tok/s directly to the full model.

Both traces used 60 GiB RAM, no model swap, a 6144 MiB GPU expert cache, 2048
prefill batch, 12 CPU workers, FP16 batched prefill and no speculation. GPU
power policy was automatic. Each generated 384 tokens, covering 383 timed
single-decode steps; each trace contains all 42 layers and all eight expert IDs
per step. No stop token occurred in either output. These are code-review and
document-handover prompts, not a broad workload sample.

| Case | Prompt tokens | Prefill tok/s | Decode tok/s | Process disk reads, whole run |
| --- | ---: | ---: | ---: | ---: |
| Code | 634 | 7.154 | 3.893 | 95.94 GiB |
| Document | 506 | 5.831 | 4.303 | 91.83 GiB |

These are **cold/paging diagnostics**, not warm-fit speed qualifications. Both
reached the 60 GiB cgroup cap. There were 4468 / 4217 memory-limit events,
504085 / 376768 major faults, and millions of file refaults, with no OOM or
model swap. The code run also recorded heavy `kswapd0` CPU usage. Although the
existing harness marked the document run `clean`, that flag does not assert
absence of page churn: these counters show it. Input/output tokens and raw
telemetry are preserved with the traces.

## Replay: original routes, exact expert fetches

Each workload uses its first 64 generated steps to rank GPU admissions. The
remaining 319 steps are evaluated without future route information. A mixed
case carries cache state from the entire code trace into the document trace,
with 702 evaluation steps after the initial 64-step training prefix. It models
a workload switch, not a single uninterrupted generated conversation.

The replay models fixed GPU admission and three CPU policies: static residency,
LRU, and an 80%-pinned / 20%-LRU hybrid. CPU and GPU entries are exclusive. A
miss reads the originally selected expert; there is no residency-biased routing.
Static/pinned entries include deterministic expert-ID tie breaks for unseen
experts. Those policies pay proactive initial loads; plain LRU starts with only
its training-prefix accesses. Their warmup costs therefore differ and must not
be treated as free policy gains. The records include initial residency and
proactive-load bytes. None of the reported steady-window bytes includes startup
loads, inference compute, PCIe traffic or GPU-cache promotion latency.

Selected candidate: **56 GiB CPU expert cache + 8 GiB GPU expert cache**, plain
LRU on the CPU tier. These are expert-storage budgets, not total process usage.
The eventual engine must also budget OS, pinned buffers, fixed weights, KV and
workspace; this is not proof that every context fits those cache sizes.

| Trace | Average SSD MiB/token | P95 SSD MiB/token | Expert miss fraction | CPU evictions |
| --- | ---: | ---: | ---: | ---: |
| Code | 45.68 | 159.25 | 1.195% | 0 |
| Document | 53.13 | 193.38 | 1.390% | 0 |
| Code → document | 25.80 | 113.75 | 0.675% | 125 |

The traces are short: isolated runs mostly reveal newly encountered experts,
not sustained eviction pressure. The mixed run is longer but remains limited.
Increasing CPU cache from 56 to 58 GiB reduced mixed-case average reads only
from 25.80 to 25.29 MiB/token. On this evidence, better residency management is
more promising than spending the last 2 GiB of host headroom. Longer traces,
more domains and repeated conversation changes remain necessary.

## Actual SSD reads

The probe selected 384 experts first encountered after the code training prefix,
shuffled deterministically, and read their three real GGUF projections. Each
pass read 4.270 GiB after 4 KiB alignment. `O_RDONLY | O_DIRECT` bypassed the host
page cache; `/proc/self/io` confirmed physical-read accounting equal to requested
aligned bytes. There was no buffered fallback, no file writes and no concurrent
inference. Two passes per worker count ran in forward/reverse order. These are
short read probes, not a guarantee of SSD performance under other transfers.

| Reader threads | Median GiB/s | Median expert service latency | P95 expert service latency |
| --- | ---: | ---: | ---: |
| 1 | 1.265 | 8.74 ms | 9.46 ms |
| 4 | 1.514 | 29.24 ms | 30.59 ms |
| 8 | 1.499 | 59.07 ms | 63.64 ms |
| 16 | 1.462 | 105.81 ms | 174.00 ms |

An expert service comprises its three reads. Latencies exclude time waiting in
the benchmark's application queue; increasing concurrent workers saturates the
drive rather than providing a throughput gain beyond four. **Start with at most
four background readers and prioritize demand misses.** A deep prefetch queue
can delay a required expert behind speculative reads.

At 1.514 GiB/s, the selected replay's average bytes correspond to about 17–34 ms
of bandwidth time per token; its P95 bytes correspond to about 74–125 ms. These
are I/O-only estimates. Actual layer dependencies, read latency, CPU expert math,
GPU work and transfers add costs, and overlap has not been measured. They are
not predicted tok/s. The existing page-fault-driven path is much slower than
an inference that only paid that idealized bandwidth cost.

## Tighter GPU-margin control

An additional control requested a 9728 MiB GPU expert cache, with a 512 MiB
free-VRAM guard and the same 60 GiB RAM cap. It was stopped after 422 seconds
before completing prefill. Concurrent `rsync` and `mv` processes were writing
storage; a five-second device sample showed about 51 MiB/s reads and 24.5 MiB/s
writes, and separate per-process samples confirmed active writers. Only the
study's decoder was stopped. Other work was left untouched.

That attempt supplies **no decode or prefill result and no evidence of a GPU
capacity failure**. Do not compare its startup duration with the baseline or
attribute it to the larger expert cache. The previous default Q23 service was
restarted afterward; warmup can be delayed while those transfers remain active.

## Recommended implementation experiment

Implement bounded, mostly exclusive RAM/GPU residency with unchanged top-8
routing. Fetch misses with reusable aligned buffers and bounded asynchronous
reads. Avoid retaining a permanent CPU copy of each GPU expert, protect hot
decode entries from prefill scans, and record cold bytes/token, per-layer stalls,
pressure/refault counters, queue depth and cache churn. Start from the 56 + 8 GiB
expert-budget candidate, adjusting total allocations to actual headroom.

The current engine's resident-biased router is not this design. Keep it disabled.
Validate routing and logits under matched arithmetic, then measure actual decode
and long-prefill throughput. The evidence justifies this Q3 implementation test;
it does not yet justify a 10+ tok/s promise or extending the conclusion to a
100+ GiB full model.

## Reproduction and evidence

[Collected evidence](fixtures/glm_q3_cache_20261008/README.md) includes configs,
frozen prompt IDs, output IDs, routes, tensor byte ranges, replay output and SSD
results. [Compact report](benchmarks/glm_b550_q3_cache_study_20261008.json).
The inference binary was unchanged; the harness only gained a `routing_trace`
config field exposing the engine's existing trace flag. Four replay tests and
all eleven memory-harness tests passed. TR16's GPU was not used.

## Exact-cache implementation follow-up

The opt-in [exact CPU-cache prototype](GLM_EXACT_CACHE_PROTOTYPE.md) now implements
bounded direct reads behind the original routing decisions. Its implementation,
correctness checks and runtime measurements are documented separately from this
simulation study.
