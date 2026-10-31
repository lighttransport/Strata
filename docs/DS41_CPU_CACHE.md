# Experimental DS41 CPU expert cache

These opt-in switches belong to the experimental DS41 path in
`strata-glm-decode`, not the production `strata-dsv41` engine. They cache unchanged
expert bytes from the original GGUF. They do not prune experts or alter routing.
With `STRATA_DS41_CPU_CACHE_MIB` unset or zero, the existing mapped path runs.

| Switch | Meaning |
| --- | --- |
| `STRATA_DS41_CPU_CACHE_MIB=N` | Fixed allocation budget in MiB, 1–45056. Linux direct reads; four reader threads. |
| `STRATA_DS41_CPU_CACHE_PROFILE=FILE` | Optional matching `STRATA_DS41_ROUTES` trace to preload at startup. |
| `STRATA_DS41_CPU_CACHE_POLICY=lru\|lfu` | LRU by default; optional decaying frequency policy. |
| `STRATA_DS41_CPU_CACHE_SPLIT=1` | Compute already cached routes while missing experts load, then merge all routes in their original order. Off by default. |

The cache requires local, unsplit CPU experts, `STRATA_DS41_POPULATE=0`,
`STRATA_DS41_RAM_COPY=0`, `STRATA_DS41_PIN_GB=0`,
`STRATA_DS41_PIN_EXPERTS=0` and `STRATA_DS41_GPU_EXPERT_MIB=0`.
The static canonical GPU cache can run alongside it. Mapped-page miss readers,
remote experts, deferred or lossy routes and other incompatible modes are rejected
at startup. This is an experiment, not an automatic installation choice.

Choose a budget that leaves room for dense tensors, prompt buffers and the OS.
Allocation accounting includes direct-read alignment and every live lease, even
after a lookup is removed. An active expert is never evicted to satisfy the budget;
insufficient space for active leases raises an error. A caller that abandons an
asynchronous read batch still has its readers drained before their memory is freed.

## Profiles and replacement

A route trace has one record per input position: `layers * top_k` native `int32_t`
expert IDs in layer order. Use the same model geometry and checkpoint. Empty,
truncated and out-of-range CPU preload profiles are rejected. For coding workloads,
train on separate generation tasks and keep evaluation prompts out of the profile.

Preloading ranks experts by trace count divided by their aligned allocation size.
Experts held by the canonical GPU cache are excluded. The selected experts load
coldest first so the hottest start newest in LRU. Preloaded entries remain evictable;
the cache does not reserve a permanent hot partition. Preload I/O adds startup time.

LFU tracks accesses, multiplying a layer's heat by 0.97 every 240 accesses to that
layer. With a profile, it starts with sixteen tokens' expected accesses as a small
prior. Without a profile, heat starts at zero. Equal scores use recency to choose
the victim. Leases protect in-use entries in either policy.

## Read and compute overlap

The split path starts all missing reads, computes the routes whose experts were
already ready, waits for remaining reads, then computes the remaining routes.
It keeps raw outputs until one final weighted merge in the original route order,
including any results from the canonical GPU tier. No incomplete read is consumed.

`DS41 cpu_cache_stats` reports unique-expert hits and misses, evictions, scheduled
read bytes, live allocation bytes and time blocked waiting for reads. Split counters
report calls, cached routes computed first, and their compute time; that time is
not a measurement of how long disk and CPU work actually overlapped. Decode resets
the counters after prefill. Compare startup and prefill separately from decode.

The CPU-only `exact_expert_cache_test` covers direct-read bytes, asynchronous
publication, leases, LRU/LFU eviction and read-failure recovery.
`ds41_cpu_cache_test` checks mapped versus discontiguous spans and split route
merges with repeats, negative IDs and a second external source. Full model runs
must additionally check output parity and task correctness.
