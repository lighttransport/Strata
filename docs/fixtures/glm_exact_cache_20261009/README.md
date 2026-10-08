# Exact-cache prototype evidence

See [implementation and results](../../GLM_EXACT_CACHE_PROTOTYPE.md).
B550 Ryzen 9 3950X / RX 9070 XT, REAP50 Q3_K_M, 2026-10-08/09 local time.
The prompt is the frozen [634-token code fixture](../glm_q3_cache_20261008/code.ids).

- `check`: 52 GiB CPU cache, eight-step within-process mmap/direct parity.
- `code`: 56 GiB CPU cache, two 384-token generations from the same checkpoint.
- `baseline`: exact CPU cache unset, otherwise the same benchmark configuration.
- `eviction-final`: final build, 128 MiB cache, forced eviction plus eight-step
  parity. It uses `eviction-check.json` and writes `eviction-check.routes.csv`.

All arms use a fixed 8192 MiB GPU expert budget and 60 GiB process cgroup.
Separate performance arms selected different actual GPU expert sets and generated
different text, so do not claim a strict matched-residency speedup. The parity
checks hold GPU residency fixed inside each process. Their route files contain
an eight-step mmap prefix followed by the same eight steps through the cache,
then ordinary generation. Duplicate positions are intentional in these checks.
The performance route files contain two repeated 383-step trajectories.

Each `*.result.json` contains the command, configuration, timings, memory and
I/O samples, interference records and cgroup counters. Logs include cache and
CPU compute counters. `*.stdout` is generated token IDs. Performance arms had
prefill page pressure and are not clean no-paging qualifications.

`exact_expert_cache.benchmarked.hpp` preserves the header used by the throughput
pair (binary hash `f57c3cd3f69023c3a63686f39186e17632d0e6d35c05979bb1c47a4a14810d02`).
The final header adds live-lease accounting across error recovery; the final
binary hash is `6f4951b463803750e6f77c8dda51cf5eab368c9314b516567aac0bc7fd91958b`.
Only the forced-eviction parity check was repeated on that final build.
To reconstruct the earlier build, use the preserved header in an isolated copy
of this working tree; do not overwrite unrelated active work.

Run `python3 summarize.py` in a writable copy of this directory to regenerate
`summary.json`. `manifest.json` records collected file hashes. No model weights
or binaries are included. The normal Q23 service configuration was not changed.
