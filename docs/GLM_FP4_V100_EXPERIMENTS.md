# Opt-in V100 FP4 experiments

An opt-in [DFlash2 integration](GLM_DFLASH2_V100.md) is under evaluation.
Its measurements do not replace the qualified MTP result below.

## Current round: ordered KDA prefill

Prepared inputs with four ordered KDA row partitions measured **32.10 decode
tok/s and 520.89 prefill tok/s** on two V100 PCIe 32 GB cards, a dual Xeon Gold
6240 host and a 181 GiB allocation. The same-binary control measured
31.87 / 454.22. Prefill improved **14.68%**; the small decode difference is not
claimed as an improvement with unlocked clocks. The 50 decode and 800–1000
prefill targets remain unmet.

| Configuration, same binary | Decode tok/s, median | Prefill tok/s, median |
| --- | ---: | ---: |
| GPU reduction control | 31.87 | 454.22 |
| Four ordered KDA row partitions | 31.89 | 490.58 |
| Ordered partitions and prepared inputs | 32.10 | 520.89 |

Each configuration used two complete warmups, a discarded one-step decode request and
three timed 8,192-input / 512-output requests. All timed requests had zero major
faults and zero swap, exited cleanly and kept the frozen fast profile's 512 token
IDs. All three configurations passed the same eight reused training tasks with
identical generated IDs. The newly frozen twenty holdouts remain unscored while
other candidates are investigated; this is an interim selection, with no new
held-out quality claim.

Use the opt-in
[`glm53f-fp4-dual-v100-kda.json`](../configs/experimental/glm53f-fp4-dual-v100-kda.json)
preset. It adds `STRATA_GLM_PREFILL_KDA_ROW_PARTS=4`,
`STRATA_GLM_KDA_ORDERED_ROWS=1` and `STRATA_GLM_KDA_PREPARE=1` to the GPU
reduction preset. The route-drop and MTP bounds below are unchanged. All prefill
routes and checkpoint weights are retained. The original reference, fast and
GPU reduction presets remain unchanged.

[Compact evidence](glm_fp4_v100_kda_measurement.json) records source and binary
hashes, all three timings, expert work, transfer counts and training checks. The
first timed request still creates 333 CPU scale-cache entries; the last two
create none. The reported median includes the slower first request.

The separate simulator profile
[`glm_fp4_v100_kda.json`](../tools/sim/data/glm_fp4_v100_kda.json) uses these
measurements without fitting timing constants. It reports **1.56×** more decode
throughput needed for 50, **1.54×** more prefill throughput for 800 and **1.92×**
for 1000. Its loose streaming and matrix-DMA bounds remain 173.4 and 1436.7 tok/s;
they omit arithmetic, reconstruction, scheduling and dependencies. Removing all
non-verification work would reach only 37.12 decode tok/s. The earlier activity
capture measured the control configuration and is not reused as this preset's
kernel-duration bound.

## Further optimization screens

The following same-binary comparisons used the same two V100 PCIe 32 GB cards,
dual Xeon Gold 6240 host and 181 GiB allocation. Each used one full warmup, a
discarded one-step decode request and one timed 8,192-input / 128-output request, with
zero timed major faults. They are short screens, not the 512-output qualification
above. Clocks were unlocked; RAM speed was not measured.

| Experiment | Matched control decode tok/s | Candidate decode tok/s |
| --- | ---: | ---: |
| Global cache admission, parity ownership | 21.65 | 17.02 |
| Global cache admission, flexible ownership | 21.65 | 15.36 |
| Global admission ranked by surviving routes | 21.67 | 15.88 |
| FP8 accumulator chains, two / four | 21.73 | 17.55 / 18.43 |
| GPU submission worker placement, primary / both | 22.20 | 22.04 / 22.22 |
| Adaptive cache, 32 / 64 verification rows | 22.64 | 18.67 / 19.35 |
| Vocabulary projection split across GPUs | 22.01 | 19.05 |
| Vocabulary split charged to scratch reserve | 22.73 | 22.90 |
| Vocabulary accumulators, two / four | 22.73 | 22.52 / 22.64 |
| Shared gate packing and K splitting | 21.67 | 17.37 |
| Compact verification history | 22.73 | 22.55 |
| Prefill expert bank | 22.73 | 17.76 |
| Aggregate arena capacity accounting | 22.21 | 17.70 |
| Expert bank with aggregate accounting | 22.21 | 17.41 |
| 64 distinct logical CPU workers | 22.16 | 18.23 |
| 64 workers with pinned GPU submission workers | 22.16 | 19.84 |
| First FP16 operand pipeline | 22.77 | 19.81 |
| Revised operand pipeline | 21.68 | 21.70 |
| Selective fused down projection | 22.24 | 22.03 |

None of these variants is selected. Real-matrix FP8 microbenchmarks found faster
individual products, but the global accumulator change lost throughput on the
model. The head copy also reduced secondary expert capacity. A separate reserve
experiment recovered capacity and identical output IDs, but its 0.74% decode
difference does not establish an improvement. The bank increased prefill from
520.85 to 538.07 tok/s, while reducing cached experts and decode throughput.
Its combined throughput tradeoff was rejected. All four original bank memory diagnostics
were interrupted because of instrumentation cost and the rejected tradeoff;
none establishes a clean bank memory check. Aggregate accounting also lost
decode throughput, and its compact-history combination exhausted a 512 MiB
allocation with only a 145.61 MiB largest free hole. It has no qualified timing.

Synthetic prefill tests compared reconstruction plus cuBLAS with fused WMMA at
twelve expert shapes, using six rotated samples each. The new 128-row split-K
kernel passed numerical, complete synthetic memcheck and racecheck runs, but
lost every shape and remains unused by the executor. The existing 32-row kernel
won only the down projection with 16, 32 or 64 token rows: medians were
63.50 / 64.50 / 80.26 microseconds, versus 79.58 / 83.66 / 102.09 for cuBLAS and
reconstruction. The selective model trial measured 520.11 prefill tok/s versus
520.81 for its control and supplied no model-level improvement.

A separate synthetic probe wrote complete WMMA fragments directly to global
memory and used a bounded shared tail for partial rows. It tested 32-, 64- and
256-row tiles, with one, two or four K partitions for the widest tile, at sixteen
expert shapes. Eight rotated samples per variant were repeated with default and
100% shared-memory carveout. Numerical boundary and large-shape checks,
memcheck and racecheck completed without errors or hazards. Every 256-row
variant was slower than reconstruction plus cuBLAS at every shape and carveout;
the fastest wide variant at each shape still took 1.45–2.43 times the baseline
time. Smaller tiles reproduced the earlier small down-projection wins. The new
probe stays outside the engine and adds no predicted speedup to the simulator.

The 64-worker runs retained identical output IDs and CPU expert-group counts
but lost decode throughput. Pinning the GPU submission workers recovered part
of that loss. The selected preset retains 32 workers. Reconstruction/GEMM
overlap and selective fused prefill have not improved model throughput. The
contiguous bank failed the quality check below. The selected preset's loss bounds
stay unchanged during these experiments.

A 0.55 route-budget screen incurred ten timed major faults and was rejected.
The retries used a fresh matched control, two full warmups and executable/library
preparation during the second warmup, completed before timing. Each then used
one discarded decode step and one timed 8,192-input / 128-output request, with
zero timed major faults and zero swap:

| Route mass budget | Decode tok/s | Prefill tok/s |
| --- | ---: | ---: |
| 0.50 control | 23.28 | 522.49 |
| 0.55 | 22.01 | 521.98 |
| 0.60 | 25.19 | 522.58 |
| 0.70 | 23.43 | 521.95 |

The 0.60 screen had 8.2% higher short decode throughput, but passed only **5/8**
reused training contracts versus the control's **8/8**. It failed `sum_small`,
`suffix_match` and `adjacent_changes`, so full 512-output qualification was
skipped and the policy was rejected. It reduced CPU expert groups from 7,071
to 4,760 in the timed request. Dropping more was not uniformly
faster: 0.70 used 3,033 groups but 33 speculative rounds, versus the control's
31 rounds. None changes the selected 0.50 budget. The checkpoint and all prefill
routes remain intact.

The first operand pipeline also lost prefill throughput, 494.61 versus 520.82
tok/s, and changed subsequent decode work. A separate opt-in revision places
reconstruction on a third stream and reuses the dense weight workspace. Its
prefill measured 513.95 versus 520.50 tok/s, with identical 128 output IDs and
CPU expert-group counts. It recovered the first pipeline's decode regression
but supplied no speedup. Both pipeline implementations and selective fused
prefill completed full-model 512-input / 8-output memory checks without reported
errors; instrumented timings are excluded.

A slab-backed bank measured 32.91 decode / 537.43 prefill tok/s in one short
screen, versus 22.08 / 520.65 for its control. These are not qualified throughput
numbers: the control created 443 CPU scale-cache entries during timing and the
slab created none, and output IDs differed. The subsequent matched three-repeat
8192-input / 512-output comparison measured **32.10 / 519.72** for the control
and **32.58 / 535.98** for the slab. Both had identical repeats, no EOS IDs,
zero timed major faults and swap, and clean shutdown. Prefill improved 3.13%;
the 1.51% decode difference is not claimed with unlocked clocks. The slab passed
only **7/8** long-context reused training contracts versus the control's **8/8**:
`argmin_first` did not produce exactly one complete C++ code block. It is
rejected and the KDA preset stays selected. The slab completed one whole
4096-input prefill and clean shutdown under memcheck, with all model kernels
instrumented and zero reported errors. Both GPUs used all 42 bank layers and
one 1960 MiB slab each. This establishes memory-check coverage for that request,
without a memory-check claim for long decode. The initial training harness stopped before
scoring because its frozen long input files were absent; its failed evidence is
retained. The retry used frozen neutral prefixes of 4096–4102 tokens for the
eight public training contracts and left the twenty fresh holdouts unscored.

An FP16 lane-cache probe used synthetic FP8 data at query, shared projection and
vocabulary shapes. Six rotated samples measured 37–48% less query-projection
kernel time and about 25% less vocabulary time. It preserved exact output bits
at widths 1–8, including ordinary and outlier activations. Production-kernel
checks also passed independent operand comparisons, memcheck and racecheck.
The opt-in engine trial below tests the extra resident-memory cost; these
synthetic timings supply no model throughput claim or simulator speedup.

The first engine screens used the same 8192-input / 128-output protocol:

| Fixed FP16 cache | Decode tok/s | Prefill tok/s |
| --- | ---: | ---: |
| Matched control | 22.68 | 521.11 |
| Vocabulary projection | 22.07 | 517.47 |
| Vocabulary, extra bytes charged to scratch | 22.39 | 521.24 |
| Shared projections, extra bytes charged to scratch | 19.64 | 521.07 |
| All selected fixed projections | 17.38 | 508.53 |

None qualified for further model validation. Caching all matching MXFP8
projections with scratch borrowing failed at startup: 139 matrices needed
4,419,747,840 extra bytes and could not retain the required 256 MiB scratch
minimum. That run supplies no throughput result. Bounded subsets also lost in
the matched short protocol, with zero timed major faults, swap and EOS IDs:

| Extra FP16 cache limit | Matrices cached | Decode tok/s | Prefill tok/s |
| --- | ---: | ---: | ---: |
| Matched control | 0 | 22.20 | 521.27 |
| 512 MiB | 16 | 19.20 | 520.07 |
| 1024 MiB | 33 | 18.54 | 520.87 |
| 1280 MiB | 41 | 18.56 | 521.59 |

Actual extra bytes stayed within each limit. These configurations are rejected;
full 512-output qualification and fresh held-out scoring were skipped. The
latest default-output check preserved all 128 original IDs with a clean exit.
The selected preset remains unchanged.

[Screen evidence](glm_fp4_v100_target50_screens.json) records binary and raw-log
hashes, output hashes, settings and matching controls. The separate
[simulator profile](../tools/sim/data/glm_fp4_v100_target50.json) imports screens
as observations with `fit: false` and retains the qualified KDA workload as its
performance anchor. These observations add no predicted speedups or quality
claims. The new twenty holdouts remain unscored.

Current CUDA SM70 and HIP decoder builds pass. Native FP4 execution is CUDA
only; the HIP build checks compatibility of the shared decoder and host code.
Eight relevant C11 tests, 26 benchmark tests and 134 simulator tests pass, with
one existing simulator fixture skip. Adaptive-cache memcheck completed with
155 actual promotions, and the reserve-adjusted head and packed shared-gate
checks also completed without reported memory errors. Their instrumented
timings are excluded. Completed default-output checks through the fixed FP16
cache implementation preserved all 128 original token IDs and exited
cleanly. Older output checks with residency rejections remain in the raw record
and supply no throughput claim. The selected KDA preset remains unchanged.

## Earlier GPU reduction qualification

GPU reduction alone measured **32.67 decode tok/s and 454.29 prefill tok/s** on
two V100 PCIe 32 GB cards with the 181 GiB allocation. The same-build control
measured 32.43 / 395.97: prefill improved **14.73%**, while the 0.74% decode
change is too small to claim a general improvement with unlocked clocks.
The requested 50 decode and 800–1000 prefill targets remain unmet.

| Configuration | Decode tok/s, median | Prefill tok/s, median |
| --- | ---: | ---: |
| Frozen fast binary, same-day rerun | 31.06 | 389.89 |
| Final build, original fast settings | 32.43 | 395.97 |
| Final build + GPU reduction, selected | 32.67 | 454.29 |
| Final build + input reuse + GPU reduction | 32.22 | 454.84 |

All three full qualification configurations used two complete warmups and three
timed 512-output requests. Every timed request had zero major faults and zero
observed swap. Every output matched the frozen fast profile's 512 IDs, SHA256
`6b12286983b9dfbb205cb8f1cbdb4b4140dc0f7d4a3366e91dd9962239ed72d8`.
The original approximately 5.26 tok/s reference is preserved separately.
Input reuse did not survive full qualification and is not enabled in the
selected profile. None of the rejected experiments changes a default.

Use the opt-in
[`glm53f-fp4-dual-v100-gpu-reduce.json`](../configs/experimental/glm53f-fp4-dual-v100-gpu-reduce.json)
profile with the sealed RAM-cache setup in [GLM_FP4_V100.md](GLM_FP4_V100.md).
It adds only `STRATA_GLM_FP4_GPU_REDUCE=1` to the existing fast settings.
Route dropping remains bounded at 0.5 original routing mass / seven routes,
with renormalization; MTP remains depth three, margin three, top eight, at most
three soft accepts and repeat guard four. Prefill retains every route, and the
checkpoint is unchanged. These lossy bounds are not output-error bounds.

The final decoder SHA256 is
`252e052c64c1c04757c42212eba946db0e6149ebef6626ecaa7201302a77b172`.
The 475-file engine source manifest matched the local reviewed source.
Fresh training passed 8/8 for the baseline and selected policy; the candidate
was frozen before the fresh held-out comparison. Both passed **17/20** fresh
holdouts, with identical generated token IDs for all twenty cases. Both failed
`popcount64`, `reverse_bits64` and `median_three`. There was no held-out tuning.
These small C++ coding checks do not establish general quality or perplexity.

## Experimental switches

These switches extend the native FP4 executor. They are off by default and do not
change the frozen reference or fast profiles. They target two V100 PCIe 32 GB
GPUs, a dual Xeon Gold 6240 host and a 181 GiB allocation. Performance and quality
must be checked on the complete model before selecting a combination.

| Switch | Values | Work changed |
| --- | --- | --- |
| `STRATA_GLM_FP4_CPU_INPUT_REUSE` | `1` | Quantize each input token once and gather its Q8 row for routed experts. |
| `STRATA_GLM_FP4_CPU_Q8_PREPARE` | `1` | Prepare Q8 magnitudes and sign masks once per activation block. |
| `STRATA_GLM_FP4_CPU_Q8_BIAS` | `1` | Requires CPU Q8. Offset integer FP4 weights by 12 and cancel that offset with a signed activation-sum correction. Keeps the existing Q8 arithmetic; adds two bytes per four activation values. Default off. |
| `STRATA_GLM_FP4_CPU_Q8_POLICY` | `1` | Requires CPU Q8, row tile 1, and global bias/preparation/local-scale options off. Select exact operand and scale kernels by verification width and expert projection shape; retain the original kernel at unmeasured widths. Emits cumulative CPU width counts. Default off. |
| `STRATA_GLM_FP4_CPU_ROW_TILE` | `1` (default), `2`, `4` | Reuse activation loads across CPU output rows. |
| `STRATA_GLM_FP4_CPU_SCALE_QUARTETS` | `1` | Cache four decoded scale indices per Q8 block; charge the existing CPU scale-cache limit. |
| `STRATA_GLM_FP4_CPU_SCALE_LOCAL` | `1` | Requires CPU Q8. Decode scale quartets into at most 256 temporary bytes per output row, using vector gathers for borrowed CSF32 codes. Adds no persistent scale cache. Default off. |
| `STRATA_GLM_FP4_CPU_SCALE_LOCAL_M1` | `1` | Use the same bounded scale decoder only for one-token CPU expert groups. Requires CPU Q8; default off. |
| `STRATA_GLM_FP4_CPU_TEAM_SIZE` | `0` (default), `2`, `4`, `8` | Divide CPU workers into independent expert teams. |
| `STRATA_GLM_FP4_PREFILL_QUEUE` | `1` (default), `2`, `4` | Retain prefetched experts until their GPU consumption event completes. |
| `STRATA_GLM_FP4_DENSE_SPLIT` | `1` | Increase independent K partitions for dense matrices with 513–8192 rows and at least 4096 columns. |
| `STRATA_GLM_FP4_GPU_REDUCE` | `1` | Sum routed GPU contributions on the primary GPU; use peer copies when available. |
| `STRATA_GLM_FP4_EXPERT_FUSION` | `1` | Combine narrow SM70 NVFP4 gate/up products, SwiGLU and FP16 conversion. Default off; other formats and prefill use their existing path. |
| `STRATA_GLM_KDA_ROW_PARTS` with `STRATA_GLM_KDA_ORDERED_ROWS=1` | `2`, `4` | Partition KDA rows in the original reduction order and save full speculative rollback snapshots directly. Default remains one partition; compact replay requires serial KDA. State/output/history, model rollback and CUDA sanitizer checks passed. The matched MTP pair measured 40.51 / 40.48 decode tok/s, supplying no model gain. See [GLM_DFLASH2_V100.md](GLM_DFLASH2_V100.md). |
| `STRATA_GLM_FP4_EARLY_READBACK` | `1` | Read narrow routed inputs before the shared FFN and wait on that copy's event, preserving the final sum order. Default off. The matched MTP pair measured 37.78 / 40.26 decode tok/s; both passed reused long and natural-length coding checks. See [GLM_DFLASH2_V100.md](GLM_DFLASH2_V100.md). |
| `STRATA_GLM_FP4_KDA_QKV_BATCH` | `1` | Batch three resident MXFP8 KDA projections in one narrow SM70 launch, with the original reduction. Default off; requires one FP8 accumulator and dense splitting off. |
| `STRATA_GLM_FP4_PREFILL_BATCH_EXPERTS` | `1` (default), `2`, `4` | Group compatible prefill expert projections into batched cuBLAS calls. |
| `STRATA_GLM_FP4_PREFILL_MMA` | `0` (default), `1`, `32`, `64` | Reconstruct FP4 tiles in shared memory and use FP16 WMMA with FP32 accumulation. Mode 1 uses tile 32 only for 4096-by-2048 matrices with 16–64 token rows. |
| `STRATA_GLM_FP4_PREFILL_OPERAND_PIPELINE` | `1` | Reconstruct the next expert on the upload stream while cuBLAS processes the current expert. |
| `STRATA_GLM_FP4_PREFILL_OPERAND_OVERLAP` | `1` | With the operand pipeline enabled, separate reconstruction from uploads and reuse the dense weight workspace. |
| `STRATA_GLM_FP4_GLOBAL_CACHE` | `0` (default), `1`, `2`, `3` | Rank decode admissions across layers; modes 2 and 3 can move selected experts between GPUs. |
| `STRATA_GLM_FP4_FP8_ACCUMULATORS` | `1` (default), `2`, `4` | Give narrow FP8 tensor-core products independent accumulation chains. |
| `STRATA_GLM_FP4_WORKER_CPU0`, `STRATA_GLM_FP4_WORKER_CPU1` | `-1` (default), allocated CPU ID | Pin a GPU submission worker on Linux while retaining its GPU memory locality policy. |
| `STRATA_GLM_FP4_CACHE_ADAPT_ROWS` | `0` (default), `32`, `64`, `128` | Reconsider each layer's parity-owned decode cache after this many verification rows. |
| `STRATA_GLM_FP4_CACHE_TOTAL_FREE` | `1` | Size the decode cache from all arena holes that can hold packed projections, retaining the existing per-device scratch reserves. |
| `STRATA_GLM_FP4_CPU_LOGICAL` | `1` | Repin native FP4 CPU workers to distinct allocated logical CPUs on Linux, reserving the first allocated CPU for the coordinator. |
| `STRATA_GLM_FP4_PREFILL_STREAM` and its `_FRAMES`, `_FRAMES1`, `_HOLD`, `_EARLY` | see [streamed prefill](GLM_FP4_V100_PREFILL.md#streamed-prefill-opt-in) | Stream each layer's uncached prefill experts through a ring of borrowed upload frames during the previous layer's attention. |
| `STRATA_GLM_PREFILL_LEAN_HOST` | `1` | Read each prefill token's embedding row once, on several threads; copy MTP's hidden rows on a helper thread. |
| `STRATA_GLM_FP4_EVICTION_STATS` | `1` | Report cache and physical evictions per GPU (`FP4_EVICTIONS`) and the stream counters. Diagnostic. |
| `STRATA_GLM_FP4_CACHE_SHRINK0_MIB` | `0` (default) through `16384` | Shrink GPU 0's expert cache by this many MiB, to separate the cache size from the other costs of a long context. Diagnostic. |
| `STRATA_GLM_FP4_KEEP_RESELECT_ROWS` | `0` (default) through `65536` | Choose each layer's GPU experts again once per request after this many decode rows ([measured slower](#keep-re-selection-during-decode)). Excludes the adaptive and global cache. |
| `STRATA_GLM_FP4_KEEP_RESELECT_SWAPS` | `0` (no cap, default) through `288` | With re-selection: at most this many swaps per GPU and layer, each by a margin of two decode routes. |
| `STRATA_GLM_FP4_KEEP_RESELECT_PREFETCH` | `0` (default) through `8` | With re-selection: upload this many newly kept experts per decode layer in the background (at most the upload slots), instead of when first routed. |
| `STRATA_GLM_FP4_HEAD_TP` | `1` | Split narrow MXFP8 vocabulary products across two peer-accessible GPUs. |
| `STRATA_GLM_FP4_HEAD_TP_BORROW` | `1` | Charge the secondary head copy to that GPU's original scratch reserve. Requires head partitioning and the original decode reserves. |
| `STRATA_GLM_FP4_FIXED_HALF` | Mask `0`–`7`, default `0` | Cache exact FP16 operands in lane order: bit 1 selects the vocabulary projection, bit 2 selects packed MXFP8 matrices with at most 8192 rows and at least 4096 columns, bit 4 selects 2048-by-4096 block-FP8 shared projections. |
| `STRATA_GLM_FP4_FIXED_HALF_MAX_MIB` | `0` (unlimited, default) through `8192` | Cap extra fixed FP16 resident bytes, following the existing fixed-matrix traversal; skipped matrices retain their original format. |
| `STRATA_GLM_FP4_FIXED_HALF_BORROW` | `1` | Charge the extra fixed FP16 bytes against the original scratch reserves, retaining at least 256 MiB. Requires the arena and original decode reserves. |
| `STRATA_GLM_FP4_HEAD_ACCUMULATORS` | `0` (inherit default), `1`, `2`, `4` | Override FP8 accumulation chains for vocabulary products only. |
| `STRATA_GLM_FP4_BLOCK_GATE_PACK` | `1` | Pack and split K for the 2048-by-4096 block-FP8 shared gate/up projections. |
| `STRATA_GLM_FP4_PREFILL_BANK` | `1` | Prefetch the next main layer's experts during fixed projections for passes with at least 4096 rows. |
| `STRATA_GLM_FP4_PREFILL_BANK_SLAB` | `1` | Allocate each enabled prefill bank in one 1960 MiB block, with non-owning projection views, to test later arena coalescing. |
| `STRATA_GLM_FP4_ROUTE_DROP_CAP` | `0.5` (default) through `0.75` | Explicitly extend the permitted original routing-mass budget for a separate lossy experiment. Does not change `ROUTE_DROP` by itself. |
| `STRATA_GLM_FP4_ROUTE_AFFINITY` | `0` (off, default) through `0.5` | Add a selection-score bonus to experts admitted to GPU decode memory. Original sigmoid scores still determine retained routing weights. Prefill and the MTP draft layer receive no bonus. Requires fixed decode admission. |
| `STRATA_GLM_FP4_ROUTE_AFFINITY_SECONDARY` | `1` | With positive native affinity, prefer admissions on the second configured GPU only. Requires two GPUs. The default bonus covers both GPUs. |
| `STRATA_GLM_VERIFY_GRAPH_MAX_TOKENS` | `4` (default) through `8` | Set the maximum window captured by the opt-in verification graphs. Eight permits full DFlash2 windows; `--check-verify-graphs` exercises every enabled width, rollback and history reallocation. Requires decode graphs and `STRATA_GLM_VERIFY_GRAPHS`. |

CPU tiling preserves each row's dot-product and reduction order. Prepared Q8
operands and quartet caches preserve the existing W4A8 representation; they do
not introduce another quantization step. Small teams retain expert reduction
order. The original Q8 kernel remains the default. Scalar and AVX2 fallbacks
remain available.

Native route affinity is a separate lossy experiment; its bonus is not an
output-error bound. It can combine with the original 0.5 route-drop cap without
extending that cap. The bonus table follows the request's final per-device
admission, resets between requests, and is not compatible with adaptive cache
admission. Its throughput and coding checks are recorded with the
[DFlash2 experiments](GLM_DFLASH2_V100.md); no existing preset enables it.

Prefill queues require device routing, asynchronous uploads and the GPU arena.
Admission bounds the extra working set to 256 MiB per GPU. Grouped prefill also
requires each expert to have more than eight rows, limits padding to 25%, and
falls back to individual expert execution when those conditions do not hold.
It retains every prefill route and scatters contributions in expert order.
Admission uses each group’s row count; two pairs or one quartet may remain
in flight within the same bound. Grouping may change cuBLAS arithmetic and therefore requires model-level quality
checks. Reconstruction and GEMM still share one compute stream; this experiment
does not overlap those two stages.

The separate operand pipeline uses two 48 MiB sets of reconstructed FP16 expert
weights. It retains the existing GEMM shapes and scatter order, fences each
slot's reuse, and protects packed sources until retirement. All activation and
operand buffers are allocated before readers are queued. It requires SM70,
device routing, asynchronous uploads and the arena, with queue/batch one and
the bank and fused prefill disabled. Its admitted working set remains bounded
by 256 MiB per GPU; unsupported groups retain the existing path. The operand
buffers are released during the existing prefill-to-decode compaction.

The overlap revision adds a reconstruction stream between upload and GEMM.
Upload-ready and operand-ready events order the readers, and completion events
protect reuse of each operand slot. Pending owned sources also fence that
stream on exception. Dense prefixes finish before the routed input fence, so
their dequantization buffer can serve the two operand slots during the routed
phase. The routed streams finish before the next dense prefix reuses it.

The fixed FP16 cache is a separate decode experiment. It stores the exact half
operands formed by the original FP8 kernel, including its extreme-scale fallback,
in the same lane order. It preserves K partitioning and accumulation order at
verification widths 1–8. The original FP8 codes and scales on the GPU are replaced
by that cache; checkpoint files remain intact. Extra resident bytes can reduce
expert capacity. Borrowing charges them to scratch and can instead exhaust
temporary storage, so neither option is selected without model measurements.
Head partitioning and multiple FP8 accumulator chains are unsupported with this
cache. Bit 2 selects by format and dimensions, including any fixed matrix that
matches those bounds.
The optional extra-byte cap applies before allocation; it permits smaller
subsets when converting every matching matrix would exhaust the scratch
reserve. It supplies no automatic ranking or predicted speedup.

Dense splitting keeps the same partitioning across verification widths 1–8,
but changes the reduction tree relative to the original kernel. It therefore
needs numerical and model quality checks. The default retains the original
partitioning, including its existing small-matrix splits.

GPU reduction uses ordered additions. When peer access is unavailable, secondary
results pass through pinned host memory. Decode still needs CPU activation data
for nonresident experts. Unsupported route shapes use the existing host path.

Fused prefill uses the same rounded FP16 weight operands as the existing SM70
reconstruction, with a different accumulation order. It requires complete kernel
and model checks before selection. `glm_fp4_prefill_cuda_test` checks boundary
tiles and large activation outliers against reconstructed operands and cuBLAS;
`--bench` rotates three compressed matrices in CUDA graphs. Those synthetic
timings exclude host scheduling and do not establish model throughput.

Global cache admission keeps the current draft reserve and per-device expert
budget. Mode 1 retains parity ownership; mode 2 fills the combined capacity,
preferring parity until that device is full. Prefill still evaluates every route
with its existing ownership. Mode 3 ranks the prompt contributions that would
survive the current drop budget if every route used CPU execution. This avoids
ranking a weak contribution solely by frequency when it could be dropped during
decode. Admission can change which decode contributions
use CPU Q8 or GPU FP16 arithmetic and which are eligible for bounded route
dropping, so model quality must be checked even when the loss bounds stay fixed.
`FP4_CACHE_POLICY` reports selected slots, moved owners and prompt coverage.

FP8 accumulator chains leave K partitioning fixed across verification widths
1–8 and add no weight copies. They change the floating-point reduction tree.

Worker placement is a scheduling experiment. Explicit CPU IDs must belong to
the original process allocation; other platforms retain the default `-1`.

Adaptive caching uses decode routes that would survive the existing all-CPU
drop budget as its heat signal. It permits at most two promotions per GPU per
update, requires a score advantage of at least two, halves scores after each
update and keeps each layer's selected count and parity ownership. It retires
GPU readers before reclaiming demoted weights. It cannot be combined with
global cache admission. Admission changes CPU/GPU arithmetic and drop
eligibility, so it requires performance and quality checks; it is not enabled
in a selected preset.

Total-free cache sizing requires the GPU arena. It counts complete packed
projection allocations within each free hole and excludes smaller tails. It
retains the original 3072 / 1024 MiB decode reserves unless an explicit reserve
setting is provided. More cache admissions can change CPU/GPU arithmetic and
route-drop eligibility; output and quality checks remain necessary. The switch
is off by default.

Logical CPU placement is a separate worker-count experiment. The existing pool
first selects one CPU per physical core and repeats those IDs for larger worker
counts. This opt-in mapping uses distinct allocated logical CPUs instead. It
leaves one logical CPU for the coordinator and requires the worker count to fit
the remaining allocation. It changes placement before model execution and does
not change the row or reduction order. Higher worker counts still need measured
performance and output checks; they are not enabled in the selected preset.

The head partition retains the complete primary matrix for products wider than
eight rows, and places a packed copy of its second half on the secondary GPU.
On this checkpoint the extra matrix occupies about 321 MiB, plus bounded input
and output scratch. Narrow products use the original K reduction on each half;
the primary stream joins both output ranges before consuming logits. It
requires SM70 and peer access and does not support CUDA graph capture. The
extra allocation can change expert admission, so kernel parity alone does not
establish model quality or throughput. It is not selected in a preset.

The optional head reserve experiment subtracts the secondary head's allocation
from that GPU's original prefill and decode scratch reserves. It retains at least
256 MiB for scratch and leaves the primary reserve unchanged. It aims to keep
expert capacity from falling when the head copy is added; measured admission,
memory checks and model output still determine whether this succeeds. Head-only
accumulator overrides change the reduction tree and require separate quality
checks. Both experiments are off by default.

Block gate packing uses the existing FP8 lane layout and fixed-width K
partitioning on that measured matrix shape. It keeps one accumulator chain and
the original scale grid, without another resident weight copy. The reduction
tree changes, so its model performance and quality need separate checks.

The prefill bank borrows 1960 MiB per GPU from the main pass's draft reserve,
then releases it before draft priming. It requires two peer-accessible GPUs,
MTP, registered RAM, the GPU arena, device routing, GPU reduction, asynchronous
uploads and queue/batch one. Its stream uses preallocated buffers while the
coordinator runs fixed projections. Expert execution retains the original order
and packed weights; kept cache entries use separate storage after readers retire.
The allocation layout can affect later cache admission, so model output and
quality still require checking. Memory checks and quality checks must use at
least 4096 input tokens to exercise this path. It is not selected in a preset.

The slab experiment requires the bank and keeps its allocation size and expert
order. Its projection and frame views expire before their owning block. It
tests whether releasing one contiguous block improves decode admission; no
cache-capacity or throughput gain is assumed. A promising screen must pass an
instrumented 4096-input request and long-context training checks before full
timing qualification. The one-request memory diagnostic instruments all model
kernels, ends with a clean shutdown and supplies no throughput measurement.

The original route-drop entry point and selected presets retain their 0.5 cap.
An explicit higher cap permits a separate native-FP4 experiment using a larger
`STRATA_GLM_FP4_ROUTE_DROP` value. Selection still applies only to CPU routes,
retains at least one route, and measures discarded mass before renormalization.
The hard maximum is 0.75, which permits a renormalization factor as large as four;
this is not an output-error bound. Prefill routes and checkpoint weights stay
intact. Larger budgets need paired performance and quality measurements and are
not enabled by the KDA preset.
The CUDA parity test covers every chain count, native and prepacked FP8, partial
splits and large activation outliers. A model-level check remains necessary.

`STRATA_GLM_CUDA_PROFILE_REQUEST=N` captures a one-based valid serving request
with the CUDA profiler API; zero disables it. For two complete warmups, the
discarded one-token request and three timed requests, request six is the last
timed request. Activity traces are diagnostic runs, not throughput qualification.

`STRATA_GLM_FP4_PROFILE=1` adds `FP4_GPU_INTERVAL` diagnostics for reconstruction,
GEMM, skinny products, matrix copies and scale preparation. Event intervals can
include host enqueue gaps; intervals on separate streams overlap. A report is
flushed at a routed layer and can include preceding dense work. Do not sum these
intervals into request latency or interpret `input_wait_ms` as pure copy time.
The same opt-in emits `FP4_DENSE_INTERVAL` with the tensor name, dimensions,
actual token width and packed format. These intervals include FP16 input
conversion and the matrix product; they overlap the broader product intervals.
Request-end reporting drains remaining events, including the final vocabulary
projection, so its timing stays with its request. Profiling waits for the event
endpoints and is excluded from throughput qualification.

`FP4_TRANSFER` reports cumulative submitted matrix H2D bytes, split into startup,
request preparation/prefill and decode. Counts include compressed scale metadata
and staging padding, but exclude activation transfers, D2H traffic and expanded
GPU allocations. Subtract counters bracketing one request before using them in
the simulator. `FP4_BATCH` reports actual grouped coverage and padding.

The standalone `strata-glm-fp4-cpu-bench` target rotates 64 MiB of synthetic FP4
codes and checks bitwise parity before timing. Its metadata stays hot, and its
timings exclude quantization and full-model scheduling. Use it to investigate
kernels, not to predict end-to-end tokens per second.

Fresh coding fixtures are in `tools/glm_fp4_optimization_quality.py`: eight
training cases and twenty held-out cases with executable oracles. Freeze the
fixtures before experimentation and freeze the selected policy before running
the held-out cases. These checks are not a general quality or perplexity result.

## Keep re-selection during decode

Each request chooses the experts it keeps on the GPUs during decode from its last prefill batch's routes. After a
long prompt that choice covers fewer decode routes, so more decode experts run on the CPU (see
[long context](GLM_FP4_V100.md#long-context-opt-in)). A routing trace of one server answering six 512-token requests
(8K, 8K, 64K, 8K, 250K and 8K prompts; source v574, `--routing-trace=PATH`, which the serve mode now also takes)
measured the share of decode routes that land on a kept expert:

| Keep set | 8K | 8K | 64K | 8K | 250K | 8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| from the last prefill batch (what the engine does) | 0.51 | 0.40 | 0.34 | 0.51 | 0.38 | 0.44 |
| from the first quarter of the answer's own routes | 0.58 | 0.49 | 0.53 | 0.42 | 0.48 | 0.54 |
| from the answer's own routes (best possible) | 0.62-0.72 at every length | | | | | |

Keep sets from the whole prompt, or from its last 2,048 or 512 tokens, were no better than the last batch. The
trace runs with route affinity on, which biases routes toward the kept set and so favours the first row.

`STRATA_GLM_FP4_KEEP_RESELECT_ROWS=N` chooses each layer's kept experts again, once per request, after N decode
rows, from the sum of the prompt's and the answer's route shares ([experimental switches](#experimental-switches)). `..._SWAPS` caps the swaps per GPU and layer (each
needs a margin of two decode routes), and `..._PREFETCH` uploads the newly kept experts in the background after each
decode layer, instead of when they are first routed. Experts dropped from the kept set stay cached until their room
is needed. `FP4_KEEP_RESELECT` reports layers, swaps, bytes reclaimed and prefetch counts.

Timed pairs, measured like the [streamed prefill](GLM_FP4_V100_PREFILL.md#streamed-prefill-opt-in) pairs (streamed
lean profile, 12,288 context, three full warmups and three timed 8,192-input / 512-output requests per side, same
binary and job; every side gave identical repeats):

| Source, side | Prefill tok/s | Decode tok/s | Swaps | Note |
| --- | ---: | ---: | ---: | --- |
| v578, off | 788.69 | 41.27 | | |
| v578, after 128 rows, no cap | 694.31 | 34.75 | 7,941 | repeats differed: not qualified |
| v578, after 64 rows, no cap | 704.51 | 38.11 | 7,412 | CPU expert groups -42%; dropped experts freed |
| v579, off | 782.25 | 41.41 | | |
| v579, after 64 rows, cap 8 | 722.21 | 36.48 | 2,862 | |
| v579, after 64 rows, cap 8, prefetch 2 | 726.64 | 37.16 | 2,862 | 1,448 prefetched, 508 skipped |
| v579, after 64 rows, cap 4, prefetch 2 | 734.90 | 37.08 | 1,926 | 921 prefetched, 379 skipped |

Every re-selection side is below 0.94 of its off side in prefill and 0.93 in decode; the gate fixed before the
v579 run asked for 0.97 of both and at least 40 decode tok/s, so the coding panel was not run. Swapped experts must
be uploaded during decode, and the next prompt finds fewer of its experts cached.

A nine-request sequence on one server (256K context, FP8 latent cache, active pools, streamed prefill, 512-token
answers, single requests, decode tok/s):

| Request, prompt | off (v572) | after 128 rows (v578) | after 64 rows, cap 8, prefetch 2 (v579) |
| --- | ---: | ---: | ---: |
| 1, 8K | 15.36 | 16.79 | 15.50 |
| 2, 8K | 20.42 | 21.14 | 23.16 |
| 3, 8K | 26.38 | 24.14 | 26.89 |
| 4, 16K | 24.23 | 26.53 | 24.34 |
| 5, 8K | 27.56 | 29.60 | 28.86 |
| 6, 64K | 27.18 | 27.84 | 27.12 |
| 7, 8K | 31.39 | 31.53 | 33.21 |
| 8, 250K | 22.96 | 24.84 | 24.18 |
| 9, 8K | 33.19 | 33.80 | 34.20 |

Long prompts gain a little (250K +5.3% to +8.2%, 64K -0.2% to +2.4%), less than the 8K pairs lose. The v578 run
uploaded 24 GB of experts during decode per request against 7 GB with re-selection off. The three switches stay
off by default and are not part of any profile.

Two other attempts at the long-context decode cost, both timed pairs as above with the control's settings:

| Setting | Context | Decode tok/s | Control decode tok/s |
| --- | --- | ---: | ---: |
| `STRATA_GLM_FP4_DECODE_RESERVE_MIB=1024` (more experts kept) | 12,288 | 38.96 | 41.47 |
| same, streamed lean profile | 12,288 | 38.77 | 41.47 |
| `STRATA_GLM_INDEX_KEYS=f16` | 256K, FP8 cache, active pools | 36.40 | 39.64 |
| `STRATA_GLM_FP4_DECODE_RESERVE_MIB=1024` | same | 37.63 | 39.64 |
| both | same | 38.29 | 39.64 |

A smaller reserve keeps more experts but the arena then evicts and reloads more of them (4,174 physical evictions at
256K). FP16 index keys change the text and lower draft acceptance; time per verify round did not rise (sources v566
and v567).

## Measurement protocol

The October 19 campaign uses the same frozen 8,192 input token IDs as the earlier
reference. Full qualification generates 512 IDs (511 timed decode advances),
with two complete warmup requests, one discarded one-token request, then three
timed requests in one process. The discarded request is printed as `TRIAL 0`;
it is not a decode measurement. Short screens use one complete warmup and one
128-output timed request. Their decode rates cannot be compared directly with
the 512-output qualification. The discarded request emits the prefill token
and one decode token; it is one decode advance, with two output IDs. The
two-warmup retry protocol above is recorded separately in each JSON row.

All qualified timed requests must have zero major page faults and zero observed
swap, return the requested number of IDs without an EOS token, and exit cleanly.
Executable/library prefaulting happens outside the timed requests. The original
fast binary's first timed request still constructed 333 CPU scale-cache entries;
the following two constructed none. Report all three timings and use the median,
without treating that first request as fully warmed metadata.

The node has two V100 PCIe 32 GB cards, two Xeon Gold 6240 CPUs, 32 model workers,
CUDA 12.9 and GCC 12.2.1. The job requests 181 GiB RAM and uses
`numactl --interleave=all`. Clocks were not locked and installed DIMM speed was
not measured. Huge pages were disabled; the mapped weights used 4 KiB pages.
No administrator page-policy change was attempted.

`tools/glm_fp4_cpu_stream.cpp` supplies a separate read-only bandwidth probe.
Build it with `g++ -O3 -std=c++20 -pthread`, then run it under the same NUMA policy.
On this node, 32 pinned workers reading 4 GiB measured a median **117.816 GB/s**
across five samples after two warmups. Each sample makes fifteen complete passes
and verifies its checksum. This is a bandwidth scenario for the conditional
roofline, not a measurement of FP4 unpack, dot-product or scale-cache throughput.

The compact evidence is in
[`glm_fp4_v100_optimization_measurement.json`](glm_fp4_v100_optimization_measurement.json).
Raw output remains separate; hashes identify the frozen reports and executables.
The previous reference measurements and profiles remain unchanged.

## Short screens

These are single 8,192-to-128 timings, useful for rejecting candidates, not the
final 512-output throughput claim. Build labels identify immutable executables
in the JSON evidence; controls and output hashes are retained there.

| Candidate | Decode tok/s | Prefill tok/s | Finding |
| --- | ---: | ---: | --- |
| v1 control | 22.14 | 390.83 | Original fast settings |
| CPU row tile 2 / 4 | 20.27 / 19.86 | 391.70 / 390.93 | Slower |
| Quartet scales, 512 MiB cache | 8.97 | 390.66 | Cache construction/eviction overwhelms isolated kernel gain |
| Owning scale copies | 20.40 | 390.80 | Slower than borrowed metadata |
| MTP depth 5 / 7 | 18.02 / 15.73 | 391.53 / 391.36 | More speculative work; changed outputs; not selected |
| v2 control | 21.30 | 396.46 | Matched GPU experiment control |
| GPU reduction | 21.33 | 443.59 | Prefill gain; unchanged short output |
| Queue 4 + GPU reduction | 21.18 | 431.25 | Slower than reduction alone |
| Refined grouped pairs + reduction | 19.56 | 430.19 | Slower; changed arithmetic/output |
| Dense split + reduction | 16.24 | 444.09 | More MTP rounds and slower decode |
| v4 input Q8 reuse | 22.39 | 395.92 | Advance to full qualification |
| Input reuse + prepared operands | 22.46 | 396.98 | Near input reuse alone |
| Revised CPU teams 4 / 8 | 20.74 / 21.88 | 396.46 / 396.06 | Not selected |

The 6,144 MiB quartet-cache screen incurred 148 timed major faults and is
excluded. Queue 2 incurred 43; initial grouped pairs incurred 21; refined
grouped quartets with reduction incurred 32. Their raw records are preserved,
but none qualifies as a throughput result. The clean initial quartet request
admitted only pairs under its workspace rule, as confirmed by the batch counters.

For the real 8,192-by-4,096 MXFP8 attention projection, the dense split kernel
also lost in isolation: 64.54→72.48 microseconds at width 1,
68.97→75.05 at width 4, and 71.50→77.52 at width 8. That probe rotates three
matrix copies and times 1,000 CUDA-graph launches after warmup. It is one kernel
measurement, not a repeated model benchmark. Numerical checks passed, but the
changed reduction tree can alter MTP proposals; the model screen increased from
29 to 37 rounds. Dense splitting remains an unselected opt-in experiment.

## Updated resource bounds and gaps

Run the separate, unfitted native profile:

```sh
python3 tools/glm_runner_sim.py native-fp4 \
  --profile tools/sim/data/glm_fp4_v100_optimization.json \
  --target-decode 50 --target-prefill 800 --output tmp/native-fp4-roof.json
```

Use `--target-prefill 1000` for the stretch target. The original simulator
profile and its default results remain unchanged. An explicit `--cpu-gbps`
override is a scenario, even when a measured read probe exists in the profile.

| Metric | Selected measurement | Target | Throughput factor still needed |
| --- | ---: | ---: | ---: |
| Decode | 32.67 tok/s | 50 | 1.53× |
| Prefill | 454.29 tok/s | 800 | 1.76× |
| Prefill stretch | 454.29 tok/s | 1000 | 2.20× |

The median decode request took 15.6427 seconds for 511 advances. It executed
27,590 CPU expert groups, 625 main-model target rows and 146 speculative rounds
(3.5 advances per round). Reading each group's original FP4 codes once amounts
to 347.16 GB. At the separate 117.816 GB/s read-probe rate, that gives a
**2.947-second / 173.4 tok/s conditional streaming roof**. It omits scale
metadata, unpack/dot instructions, activation traffic and scheduling. The
observed 18.8% of this loose roof does not identify a single removable cost.

Verification consumed 13.5103 seconds, or 86.4% of decode time. Removing all
non-verification work would reach only **37.82 tok/s**. With non-verification
work fixed, verification must become **1.67× faster** to reach 50 tok/s.
The deeper-MTP screens and CPU-kernel experiments did not deliver that gain.

The median prefill took 18.0325 seconds. Its actual submitted matrix H2D traffic
was **70,177,140,736 bytes on GPU 0** and **53,274,591,616 on GPU 1**. Using the
previous concurrent DMA probe's 12.3075 / 12.3355 GB/s links and 24.615 GB/s
aggregate gives a **5.702-second / 1,436.7 tok/s conditional roof**, limited by
the primary link. Measured throughput is 31.6% of that optimistic bound.
The old compressed-extent proxy remains separately visible in the simulator;
it is not substituted for these phase counters.

Reaching 800 requires reducing prefill by 7.79 seconds; reaching 1000 requires
9.84 seconds. At unchanged matrix traffic, the primary link would need average
rates of 6.85 and 8.57 GB/s respectively. Both are below the isolated DMA probe,
but neither proves the targets achievable: activation traffic, reconstruction,
attention/KDA/mHC work, launches and resource contention are omitted, and the
bound assumes optimistic overlap. The phase counter ends at `compact`, so
post-compact cache preparation remains outside its prefill-byte count.

Further work should measure kernel-active time and host enqueue gaps separately
before replacing the decode kernels or prefill schedule. The subsequent screens
above tested tiled prefill kernels and added an opt-in reconstruction/GEMM
pipeline. Neither supplies a qualified model speedup so far. Simply increasing
queue depth or grouping cuBLAS calls did not improve the measured workload.

## Validation and limits

The final decoder's original greedy path, with all new switches off, produced
exactly the frozen reference's 128 token IDs on the 8,192-token input. This was
an output check only, without a full warmup. The frozen fast profile also kept
its complete 512-token output on the final build. Both original measurement
JSON files, both original presets, the original simulator profile and the
reference branch are unchanged.

CUDA SM70 and HIP decoder builds passed. The V100 ran the FP4 numerical tests,
including the new dense-split width checks; final CPU tests passed on VNNI,
forced AVX2 and scalar paths. The local GLM host suite accounted for twenty
passes and eight hardware/topology skips. Its socket transport test initially
failed inside the restricted sandbox and passed outside it. Benchmark/simulator
Python tests passed (36 tests). HIP GPU execution, single-GPU execution and the
non-peer transfer fallback were not exercised. The separate SYCL project does
not compile the changed engine sources.

A CPU-only build of every target still fails linking the unrelated
`strata-glm-exl3-check` and `strata-glm-exl3-dequant` tools because
`strata_env_is_one` is unresolved. Their source and link rules were unchanged;
the GLM test targets were built directly. This report does not claim that the
repository-wide all-target build passes.

Compute Sanitizer completed the selected GPU-reduction path at 128 input tokens
and eight outputs with **zero errors** and process exit zero. Its timings are
instrumented diagnostics only. The larger 2,048-token grouped-pair/reduction/
dense-split diagnostic was interrupted during its first prefill to stay within
the allocation; it has no completed clean result and is not counted as passed.

The additional fully metadata-warmed phase profile was skipped when only 322
seconds remained, below the queue's 330-second full-run guard. Earlier detailed
phase profiles remain available, explicitly labeled with their 333 first-timed
CPU scale-cache misses. They must not be presented as fully warmed metadata
costs. The selected request's aggregate MTP timers and exact matrix-transfer
counters used above come from the completed three-run qualification.
