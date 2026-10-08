# GLM exact CPU expert-cache prototype

This opt-in Linux prototype implements the next experiment from
[the Q3 cache study](GLM_Q3_CACHE_STUDY.md). It uses the original router's
selected experts and original GGUF bytes. It does not bias routing toward
resident weights or change quantization or expert arithmetic.

## Storage and lifetime

`STRATA_GLM_EXACT_CACHE_MIB` enables a bounded CPU LRU cache after GPU cache
admission. Its budget counts all allocated projection storage, including
4 KiB alignment padding. Each expert has gate, up and down buffers read directly
from its GGUF offsets with `O_DIRECT`; there is no buffered fallback on error.
Four persistent reader threads share a demand-only queue. All current layer
misses finish before CPU expert computation starts, while independent GPU work
can proceed. There is no speculative read-ahead or overlap with CPU computation
of that same layer yet.

GPU-resident experts bypass the CPU cache. The prototype requires fixed GPU
residency and rejects adaptive/residency-biased routing, speculation, prepared
weights, expert packs and other incompatible paths. It drops mapped expert pages
at the prefill-to-decode transition. LRU eviction cannot free an entry while a
CPU job holds its lease. A request reset releases the cache before prefill;
this first version does not retain hot experts across requests.

The budget covers CPU expert storage only. Fixed host weights, runtime buffers,
KV, page tables and the OS need additional headroom. Direct I/O avoids adding
another permanent file-cache copy of CPU entries. Cache counters report hits,
misses, evictions, aligned read bytes, resident bytes and acquisition wait time.
`wait_ms` includes allocation and demand reads; it is not just SSD service time.
Counters are cumulative and emitted every 32 positions and at reset/destruction.

Prefill is unchanged. It still stages expert weights through the existing file
mapping, so this prototype does not promise better cold-prefill throughput.

## Build and correctness checks

Build the HIP target using the existing B550 configuration described in
[AMD HIP](AMD_HIP.md). The standalone CPU test requires Linux direct-I/O support
on `/tmp`:

```sh
c++ -O2 -std=c++20 -pthread -Iinclude tools/test_glm_exact_cache.cpp \
  -o /tmp/test_glm_exact_cache
/tmp/test_glm_exact_cache
cmake --build build-hip-glm-v11 --target strata-glm-decode -j4
```

The test checks unaligned source ranges against original bytes, hits, duplicate
IDs, bounded eviction, active leases and recovery after a short read. Address
and undefined-behavior sanitizers can be used with `-fsanitize=address,undefined`;
leak tracing is unavailable in the local sandbox, so that run used
`ASAN_OPTIONS=detect_leaks=0`.

For a CLI inference check, set `STRATA_GLM_EXACT_CACHE_CHECK=1` along with the
cache budget. After identical GPU admission, the decoder checkpoints its state,
runs eight fixed sequential inputs with mmap, restores the checkpoint, and
requires bitwise equality of every logit using the direct cache. It then restores
the state for generation. This diagnostic adds cold reads and duplicate route
trace positions; never count it as a performance run.

## Initial correctness result on B550

On Ryzen 9 3950X / RX 9070 XT, REAP50 Q3_K_M passed the eight-step
checkpoint comparison with **every logit bitwise equal**. All 336 layer-route
rows (eight steps × 42 MoE layers, eight experts per row) also matched exactly.
The check used a 52 GiB CPU budget, fixed 8 GiB GPU expert cache and 60 GiB
process cgroup. It had no OOM, swap or memory-limit events. This establishes
storage-path parity for that fixture, not broad model quality.

The final build also passed with a 128 MiB budget: 2325 evictions, 2336 misses,
all eight sequential logit vectors bitwise equal, and all 336 paired layer-route
rows identical. Active expert storage stayed at or below 125.25 MiB. This
forced-miss run is a correctness test, not a throughput setting.

## Performance on B550, 2026-10-08/09

Both arms used the same prototype binary, Q3_K_M file, 634-token C++ review
prompt, fixed 8192 MiB GPU expert budget (720 actual experts, 8190 MiB), 12 CPU
workers, FP16 batched prefill, context 4096 and 60 GiB cgroup with swap disabled.
Each process ran one untimed prefill and two timed prefills, then generated
384 tokens twice from the same checkpoint (383 timed decode steps per pass).
The exact arm used a 57344 MiB CPU cache. The control left that setting unset.

| Path | First decode pass, tok/s | Repeated warm pass, tok/s | Warm-pass median / P95 step |
| --- | ---: | ---: | ---: |
| Exact CPU cache | 3.839 | 6.630 | 151.4 / 171.7 ms |
| Existing mmap | 4.376 | 6.528 | 152.8 / 174.4 ms |

**This does not demonstrate a meaningful throughput win.** Cold filling was
slower, and warm speed was similar. Separate launches produced different
prefix-trained GPU admission fingerprints and different token streams, despite
identical configuration and prompt; this is not a strict matched-residency A/B.
The within-process parity check above does hold residency and model state fixed.
Both performance arms reproduced their own full 384-token output on repetition;
the exact arm also reproduced all 16,086 layer-route rows.

The exact cache retained 4715 CPU experts (52.43 GiB including alignment), with
no evictions. Its repeated pass needed no additional expert reads. In the last
50 seconds, total process reads were 41.6 MiB, versus 43.7 MiB for mmap—other
runtime reads remain. CPU expert work accounted for 49.23 of 57.77 warm decode
seconds. Once this workload fits, explicit caching does not remove that compute
cost; it did not achieve 10 tok/s.

The useful measured difference is memory control: the last 50 seconds peaked
at **53.69 GiB** with the exact cache, versus **60.00 GiB** for mmap. Both had
at least 2198 MiB VRAM free and no OOM or swap. Whole-run RAM peaks were 60 GiB;
there were 8644 / 11680 memory-limit events and `kswapd0` activity during prefill,
so neither whole run is a clean no-paging qualification. The cache protects
decode storage but does not repair the existing prefill scan.

Timed prefill rates were 7.225 and 11.869 tok/s for the exact arm, versus 7.722
and 15.383 for mmap. Exact caching is inactive during prefill; those differences
reflect separate runs with varying page residency and are not a prefill benefit.
These short-prompt scans do not meet the 200 tok/s prefill target.

The throughput runs used binary
`f57c3cd3f69023c3a63686f39186e17632d0e6d35c05979bb1c47a4a14810d02`.
Final review then tightened budget accounting for leases that survive an error;
this only changes cache ownership accounting, not weights or arithmetic. The final
binary, `6f4951b463803750e6f77c8dda51cf5eab368c9314b516567aac0bc7fd91958b`,
received a fresh forced-eviction parity run. The full throughput pair was not
repeated after that fix; its earlier header is preserved with the evidence.

Next experiments should target the measured CPU expert cost and avoid throwing
away useful prefill residency before cold cache fill. Huge-page cache storage,
larger GPU residency, and overlapping demand I/O with ready CPU jobs are plausible
experiments, not measured improvements. Cross-request retention and long-prefill
reuse remain separate work. Keep this feature opt-in.

## Scope

This is a demand-cache prototype, not a release-default setting. Longer context,
multiple requests, other models and devices require further validation. The full
100+ GiB model has not been tested with this implementation. The replay study's
56 GiB CPU + 8 GiB GPU recommendation was a storage simulation, not a runtime
capacity or throughput guarantee.

## Reproduce the measured runs

[Raw evidence](fixtures/glm_exact_cache_20261009/README.md) and
[compact results](benchmarks/glm_exact_cache_20261009.json) preserve the runs.
Configuration paths assume the existing B550 installation. Use a separate
prototype executable so the normal service can retain its known binary. Stop
inference first; do not run these alongside another GPU or SSD workload.

```sh
tools/glm_b550_60g.sh stop
mkdir -p build-hip-glm-v11/exact-cache-prototype
cmake --build build-hip-glm-v11 --target strata-glm-decode -j4
cp build-hip-glm-v11/strata-glm-decode \
  build-hip-glm-v11/exact-cache-prototype/strata-glm-decode
cp docs/fixtures/glm_exact_cache_20261009/*.json \
  build-hip-glm-v11/exact-cache-prototype/
.venv-glm-hip/bin/python tools/glm_low_memory_bench.py \
  build-hip-glm-v11/exact-cache-prototype/code.json \
  docs/fixtures/glm_q3_cache_20261008/code.ids \
  --output build-hip-glm-v11/exact-cache-prototype/code-repeat \
  --ram-gib 60 --tokens 384 --trials 2 --timeout 1500 --single \
  --gpu-capacity-mib 16304 --gpu-used-limit-mib 15792
# Repeat with baseline.json to disable only the exact CPU cache.
# check.json: --tokens 16 --trials 1, enables within-process parity.
# eviction-check.json: --tokens 2 --trials 1, forces 128 MiB LRU eviction.
```

Do not include `STRATA_GLM_TIER_ADAPT` in these configs: even the value `0`
enables that older presence-based switch. Exact caching deliberately rejects it.
The normal service config remains unchanged; this prototype is not its default.
