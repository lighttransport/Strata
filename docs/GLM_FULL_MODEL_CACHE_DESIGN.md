# Full GLM weights on B550: cache feasibility

This is a design and capacity analysis, not an implemented full-model deployment
or a new throughput measurement. The user permits tighter memory margins if the
run avoids OOM and thrashing. The current B550 service was not interrupted.

## Measured capacity and existing evidence

B550 reports 67316260864 bytes of Linux RAM (62.693 GiB), and 17095983104 bytes
of VRAM (15.921875 GiB). A 63 GiB application RAM budget exceeds reported physical
RAM even before OS/kernel/driver allocations. The ordinary service's 60 GiB cap
already leaves only 2.69 GiB of reported physical memory outside the cgroup,
not a full 4 GiB. A global 15.6 GiB GPU usage target leaves about 330 MiB free;
it is not a 15.6 GiB engine-owned allocation allowance. Driver and desktop use
must be counted, along with fixed weights, attention state and workspaces.

The inspected full-model Q23 overlay has 103.552734375 GiB of main routed
expert tensors, across 42 layers with 288 experts and top-8 routing. This count
includes original tensors absent from the expert sidecar. The sidecar alone is
98.244 GiB of payload and is not the entire effective model. Main expert bytes
selected per token total about 2.8765 GiB before cache hits or speculative work.
This is the existing full Q23 recipe; another '100 GB' artifact needs its own
census. GB and GiB are not interchangeable.

Earlier full-model tests capped at 60 GiB on TR1950X measured about 2.30–2.36
single-sequence decode tok/s. They are historical CUDA results, not a B550
forecast. Residency-biased routing improved speed but failed the quality screen.
See [the earlier full-model constrained-memory report](GLM_64G_16G_SAFE.md).
The current `RamTier` / `STRATA_GLM_PREFER_RESIDENT_MIB` implementation biases
routing toward resident experts; it is not an exact disk-cache solution.

A static frequency/size ranking of the recorded calibration routes gives:

| Combined distinct CPU+GPU expert capacity | In-sample selection hit fraction | Estimated uncached expert GiB/token |
| --- | ---: | ---: |
| 60 GiB | 84.80% | 0.439 |
| 62 GiB | 86.08% | 0.402 |
| 64 GiB | 87.30% | 0.366 |
| 66 GiB | 88.48% | 0.332 |
| 70 GiB | 90.66% | 0.269 |
| 80 GiB | 95.11% | 0.141 |

Source: `build-q2-redesign/calibration/coverage.json`, GU/down counts checked
identical and counted once; tensor directory from the four Q2 base shards with
the Q23 expert overlay. Raw estimates are in
[the capacity record](benchmarks/glm_full_model_cache_capacity_20261008.json).
These are greedy static in-sample estimates, not an optimal-cache bound,
decode-only trace replay, dynamic-cache prediction or measured SSD bandwidth.
Temporal locality may improve dynamic caching; workload shifts may hurt it.

At 10 tok/s, 0.33–0.40 GiB/token would require 3.3–4.0 GiB/s of sustained SSD
reads merely for misses, before compute and request latency. Thus slight margin
relaxation alone is not a credible basis for promising 10+ tok/s. Decode needs
measured temporal hit rates and miss latency, not just a model-file size or
aggregate RAM+VRAM capacity calculation.

## Proposed exact cache design

1. Keep original top-8 routing and quantized tensors unchanged. On a miss, fetch
   the selected expert and wait if necessary. Prefetch predictions may only
   schedule reads; they must not replace experts or change router scores.
2. Make CPU and GPU expert tiers mostly exclusive. GPU-resident experts should
   not consume another permanent CPU copy. Track ownership explicitly and
   release eligible source file pages after upload. Keep transfer buffers and
   shared page-boundary effects in the accounting. A nominal 8 GiB GPU cache
   does not add 8 GiB of distinct capacity if its CPU backing stays resident.
3. Use an application-owned bounded RAM cache for whole expert projections,
   with asynchronous bulk reads into reusable buffers. Protect hot decode
   experts from prefill scans; bypass the hot tier for streaming reads. Limit
   outstanding reads and admission/eviction churn. Calibration and prompt
   routes can seed admission, while live hit rates drive adaptation.
4. Reorder long prefill to reuse expert weights across more prompt tokens.
   The present chunked path can revisit most experts every 2048-token chunk.
   Larger batches or layer/expert-major scheduling can amortize SSD reads;
   their activation/route/output buffers must have an explicit budget. This
   is more promising for prefill than decode, whose future routes depend on
   the just-computed hidden states. It is an engine redesign, not a flag that
   has already been tested.
5. Redistribute GPU memory by phase: prefill workspace first, then reclaim it
   for decode experts. Long contexts need more attention state, so cache
   capacity must shrink accordingly. Do not assign the entire VRAM budget
   to weights or assume the short-context budget also fits at 128K.

Allocation should follow observed headroom: start at the current RAM cap and
increase only when other resident memory permits; test GPU global usage toward
15.3–15.6 GiB with a small measured reserve. Keep model swap disabled. Watch
memory/IO PSI, major-fault and refault rates, disk bytes per generated token,
and latency tails. Shrink/bypass admission or stop on sustained thrashing,
without waiting for an OOM. Linux provides cgroup memory accounting and pressure
signals: [cgroup v2](https://docs.kernel.org/admin-guide/cgroup-v2.html) and
[PSI](https://docs.kernel.org/accounting/psi.html).

## Next validation

First replay real decode-route sequences through an exclusive CPU/GPU cache
simulator and measure cold bytes/token, admission churn and a miss-latency model.
The static estimate above cannot replace that replay. Then benchmark B550's
actual expert-sized SSD reads without perturbing another inference run. Only
implement/qualify the exact async tier if these establish a useful speed ceiling.
Acceptance should include unchanged routing and output/logit comparisons under
matched arithmetic, no sustained pressure stalls, and single-request prefill /
decode timing separated from startup.

The full model is not currently on B550: it has the processed REAP50 assembly
and REAP50 Q3_K_M. Its model SSD has only about 8.3 GiB free, so the approximately
100 GB full-model experiment needs additional space or a different storage
location. No existing model was deleted and no transfer was started.

## Follow-on Q3 study

The requested [REAP50 Q3 route-replay and SSD study](GLM_Q3_CACHE_STUDY.md) is complete. It supports an exclusive-cache implementation experiment for that smaller expert set, without establishing full-model throughput.
