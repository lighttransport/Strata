# GLM Q2 decode redesign plan

Goal: 15 delivered tokens/s for GLM-5.3-Flash UD-Q2_K_XL decode on the
Threadripper 1950X (two NUMA nodes, about 90 GB/s streaming), 128 GB DDR4 and
RTX 5060 Ti 16 GB, using at most about 120 GB of host RAM including the KV cache
and a 12 GiB GPU budget. PCIe is about 7.5 GB/s. MTP may be used.

This is a plan. Numbers marked **measured** come from runs listed below; numbers
marked **estimate** are arithmetic from those measurements and must be replaced
by measurements as each step lands.

## Progress (2026-10-05)

Measurements in this section ran while a 99 GB download and another agent's
compiler builds shared the machine (load average 5-8). CPU expert phases varied
by up to 40% between runs, so every comparison alternates its arms in the same
period, and GPU work is compared by the step trace's GPU-side time. These are
not quiet-machine results; repeat them on an idle machine before publishing.

Prime fixture, 4096 MiB dense cache, context 8192, 15 workers plus the host:

| Configuration | Ordinary tok/s | MTP depth 2 tok/s |
| --- | --- | --- |
| Earlier reference (quiet machine, docs) | 9.74 | 11.3 |
| This session, previous code, same period | 9.21 | 11.12 |
| Original IQ weights, new path + 3.5 GiB static GPU tier | 11.16, 11.23 | 12.96, 12.93 (CPU draft experts) |
| Same with GPU draft experts, no tier | | 12.45, 12.49 |
| Q2_K/Q3_K pack, mapped file, plain rows, 3.5 GiB tier | 12.28, 12.00 | 12.58, 13.68 |
| Q2_K/Q3_K pack, mapped node-owned rows, fused quantization, 3.5 GiB tier | 13.09, 13.10 | 13.97, 13.51 |

Each complete answer (122-145 tokens, ending on a stop token) passed the
400,538-case prime oracle. Outputs differ slightly between configurations
because GPU-resident experts use Q8_1 activations, the router GEMV sums in a
different order and the pack converts weights. The pack has not yet passed the
held-out KL/top-1 and HumanEval gates in
[GLM Q2 expert conversion](GLM_Q2_REQUANTIZATION.md); it remains opt-in.

In the last row the CPU reads 2.88 GB of routed experts per ordinary token at
about 55 GB/s (52 ms) and GPU work is about 24 ms. The real-weight expert
benchmark gives the same 55-63 GB/s for these formats, so scheduling overhead is
now small; CPU kernel throughput dominates. A three-token MTP verification reads
2.6x the expert bytes of one token, so MTP mostly amortizes GPU time.

An API check with [glm53f-q2-v2.json](../configs/glm53f-q2-v2.json) (original
weights, MTP, tier, new switches) returned a complete 122-token answer, survived a
cancelled stream, and produced an identical answer on repeat. Short prompts spend
most of a request in GPU prefill, which streams expert groups over PCIe even for
65 tokens; that is a separate prefill issue.

Corrections to the plan: per-layer handoff was not 7.4 ms. A GPU timeline of the
old path shows about 1.5-3 ms of handoff per token; the rest of the non-expert
time is GPU kernel time. Step 1 alone is therefore neutral (measured 8.79-8.84
versus 8.40-8.86 tok/s); its value is running GPU-resident experts with no host
round trip.

Implemented (all opt-in, environment variables):

| Switch | What | Measured effect |
| --- | --- | --- |
| `STRATA_GLM_STEP_PIPELINE=1` | GPU-driven step: launcher thread enqueues every layer; MoE layers hand off through a mapped mailbox (`glm_mailbox.cu`) | Bitwise-identical logits (verify-graph and verify checks, 64-token and 133-token outputs, MTP sweep); about neutral alone |
| `STRATA_GLM_Q8_DECODE=1` | Q8_0 MLA absorb and mHC projection GEMVs, FP32 router GEMV, no FP32 weight copies on the GPU | GPU time 25.4 to about 21.5 ms/step with the RMSNorm kernel; 551 MiB less GPU memory |
| (always) | 1024-thread RMSNorm for 1-8 rows | Included above |
| `STRATA_GLM_LAYER_FLOW=1` | One pool dispatch per MoE layer (`ExpertPool::run_layer_native`) | Bitwise equal to phased execution; +12% under heavy CPU interference, within noise otherwise |
| `--decode-cache-mib=N` with the pipeline | GPU tier experts run between publication and wait; CPU skips them | 7.3% fewer CPU bytes at 3 GiB, +5.3% tok/s |
| `STRATA_GLM_EXPERT_PRIOR=coverage.json` | Calibration counts blended with prompt routes for tier selection | Needed for short prompts |
| `STRATA_GLM_TIER_ADAPT=1` | Tier adapts during decode; staging copies use the service thread's idle waits | 3-7% fewer CPU bytes than static; about +1-2% once warm |
| `STRATA_GLM_STEP_TRACE=1` | Per-step head, CPU, between-layer and tail times | Diagnostics |
| `STRATA_GLM_MAPPED_OWNED=1` | Node-owned rows as views into the file mapping, pages moved by `mbind` | Pack: 12.0-12.3 to 13.1 tok/s ordinary |

Pack memory: copying the pack into 111 GB of owned arenas swapped while the
download filled the page cache, so its preparation stopped itself.
`STRATA_GLM_MAPPED_OWNED=1` keeps node-owned scheduling but points the shards into
the file mapping and moves each half's pages to its node with `mbind`
(`NumaTensor::View`); weights stay in reclaimable page cache. All pages moved in
both runs. `STRATA_GLM_Q2_NUMA_WEIGHTS=0` with `STRATA_GLM_PACK_FUSE_QUANT=0` keeps
a pack on its file mapping with plain rows.

Tried and not kept: the KDA column/row-part variants (no measurable change), the
upstream multi-column GEMV layout (`STRATA_GLM_MULTI_UPSTREAM=1`, slower), SMT
workers for Q2_K (19-53 versus 55-63 GB/s).

Next, in order of expected value:

1. Repeat the table on an idle machine, three trials each, all three coding
   fixtures, and record a measurement JSON.
2. Run the pack's quality gates (held-out KL/top-1, HumanEval, coding oracles).
3. Width-3 verification GPU work: the multi-column GEMV takes 21.9 ms per MTP
   round versus 16 ms for one token's GEMVs; router, mHC and MLA calls still run
   once per token.
4. Better tier selection for coding (a code-only routing prior, or cheaper
   adaptation; adaptive uploads currently add about 1 ms/step of GPU latency).
5. Faster Q2_K/Q3_K CPU kernels: 55-63 GB/s of about 90.

## Progress (2026-10-06): MTP verification and quality gates

Summary. The Q2_K/Q3_K pack passed the quality gates (held-out perplexity
+3.45%, HumanEval 155/164 against 153/164 for the original weights) and, with
MTP depth 2, decodes the three coding fixtures at 17.4, 16.8 and 15.6 tok/s
(prime, JSON, CSV; ordinary decode about 14 tok/s). The 20 tok/s target was not
reached. With the current structure a round needs at least about 144 ms (the
analysis is in "Toward 20 tok/s" below), and every hardware budget is close to
its limit: CPU expert bandwidth about 61 of 72 GB/s, 12.9 GB of GPU memory in
use, 116 GB of host memory resident. `STRATA_GLM_MTP_SKIP_ANCHOR=1` (exact, about
1.7 ms per round) is now in both v2 server configs.

### Verification-width GPU work

All of these keep every token of a verify window bitwise equal to a one-token
step. `--check-verify-graphs` and `--check-verify` (widths 1-4 and 8, history,
rollback, retained prefixes) pass with all of them enabled.

| Change | Where | Exactness check |
| --- | --- | --- |
| Q5_K/Q6_K multi-column GEMV unpacks each weight block once, not once per column | `native_mmvq.cu` | `mmvq_multi_parity`: 0 of 148,480 outputs differ |
| Optional 2 or 4 rows per block for the exact multi-column layout (`STRATA_GLM_MULTI_ROWS`) | `native_mmvq.cu` | same test at 1, 2 and 4 rows |
| Router, mHC coefficient/read/write: one launch per window instead of per token | `glm.cu` | `glm_q8_test`: bitwise equal to per-token calls |
| MLA k_b/v_b absorb: one window-wide Q8_0 GEMV before and after the per-token attention tiles | `glm_decode.cpp` | `--check-verify` (width 8 against sequential) |
| KDA recurrence writes the rollback history itself; one kernel per window instead of per token plus a 2 MB state copy | `glm_prefill.cu` | `--check-verify-graphs` with history and rollback |
| MTP resync starts after the anchor, whose draft step already used the same inputs (`STRATA_GLM_MTP_SKIP_ANCHOR=1`) | `generate_mtp` | identical draft state by construction; output tokens compared in the A/B |
| GPU draft experts build their groups on the device (no host round trip per draft step) | `moe_mtp` | same grouping as the host code |

An independent code review found a mid-step lookup race in the adaptive tier
(a victim's lookup entry was cleared while the GPU could still copy that layer's
lookup), a first-use pinned allocation inside the CPU service, page-binding
`mbind` calls that split the mapping into more VMAs than the default
`vm.max_map_count`, and a default-path change from the new RMSNorm kernel. These
are fixed: the lookup is no longer cleared mid-step, allocation happens at step
start, mapped node-owned rows use `move_pages`, and the RMSNorm kernel is part of
the opt-in decode kernels.

### Verification timing

Prime fixture, original weights, no GPU tier, GPU draft experts, two decode
repetitions per run, runs alternated in the same period while another agent's
builds paused. Every run produced the same 139 tokens. Per-round times come from
the step trace (GPU time is head + between-layer + tail, the part the CPU waits on).

| Run | tok/s | Rounds | Tokens/round | CPU experts ms/round | GPU ms/round | Resync ms/round |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before this work (session-2 binary) | 12.49 | 48 | 2.90 | 174.8 | 42.8 | 5.3 |
| Depth 2, all verify-width changes | 13.03, 13.06, 13.03, 13.02 | 48 | 2.90 | 174.3 | 34.3 | 4.8 |
| + skip anchor resync | 12.74, 12.80 | 48 | 2.90 | 179.9 | 34.6 | 3.1 |
| + skip anchor, 2 rows per GEMV block | 13.05, 13.08, 13.10, 13.11 | 48 | 2.90 | 174.6 | 35.1 | 3.1 |
| Depth 3, skip anchor | 13.34, 13.34 | 37 | 3.76 | 224.4 | 40.6 | 4.6 |

The verify-width changes cut GPU time per depth-2 round from 42.8 to 34.3 ms
(-20%) and raised throughput 4.4%. Skipping the anchor resync saves 1.7 ms per
round (the "skip anchor" row's slower CPU phase is interference; its GPU and
resync times match). Two rows per block does not change GPU time. Depth 3 now
beats depth 2 by 2.3% here: 92% of drafts are accepted (102 of 111, against 91
of 96 at depth 2), and a four-token verification costs 1.29x a three-token one. Earlier sessions measured
depth 3 below depth 2. In every row about 80% of a round is CPU expert time, so
GPU-side verification work is close to done; further MTP gains have to come from
CPU expert bytes or kernel speed.

### CPU kernel experiments

Expert-only benchmark (real routes, layers 3-44, node-owned rows, 15 workers,
interference possible; each figure is one of the last rounds of a run):

| Variant (two runs each, alternated) | Q2_K/Q3_K GB/s | Original IQ GB/s |
| --- | --- | --- |
| 4 KB pages | 64.2-65.7 (two rounds at 51.8 and 54.4) | 33.2-33.7; 40.0-41.0 |
| `madvise(MADV_HUGEPAGE)` on the owned arenas | 64.4-66.2 (one round at 54.2) | 40.1-40.6; 34.7-38.7 |
| No software prefetch (control) | 64.1-65.5; 59.1-60.0 | |
| Software prefetch 512 bytes ahead | 58.6-58.9; 47.4-56.2 | |
| 1024 bytes | 59.1-59.9; 58.6-58.8 | |
| 2048 bytes | 41.8-43.6; 49.1-56.6 | |
| 4096 bytes | 50.6-53.1; 56.2-57.3 | |
| Three-token jobs, prefetch 0 / 1024 / 2048 bytes | 56.9-57.3 / 21.1-21.6 / 51.0-51.4 | |

Huge pages make no difference beyond run-to-run noise, so the TLB is unlikely
to be the limiter. (The kernel's THP mode is `madvise` with direct defrag; the
control run sets `MADV_NOHUGEPAGE`. Huge-page backing was not read back during
the run.) Software prefetch only hurts (`STRATA_Q23_PREFETCH`, default off, kept
for reproduction). The worker-count scaling runs below 15 workers were not
usable (node-owned rows assume both nodes have workers).

### Final pack throughput (2026-10-06)

Prime fixture, dev binary with every switch in
[glm53f-q2-v2-pack.json](../configs/glm53f-q2-v2-pack.json) plus
`STRATA_GLM_MTP_SKIP_ANCHOR=1`, 3.5 GiB GPU tier, CPU draft experts, three decode
repetitions per run. Each run started after another agent's builds had been idle
for 30 seconds; they could resume during a run. Only the prime fixture was
measured, so these are not yet the three-fixture numbers the plan requires.

| Experts | Mode | tok/s (three repetitions) | Rounds | Drafts accepted |
| --- | --- | --- | ---: | --- |
| `q23` pack | Ordinary | 13.83, 13.85, 13.87 | | |
| `q23` pack | MTP depth 2 | 15.71, 17.24, 17.25 | 46 | 87 of 92 (95%) |
| `q23` pack | MTP depth 3 | 16.09, 15.98, 16.04 | 38 | 95 of 114 (83%) |
| `retain50` | Ordinary | 12.59, 12.68, 12.49 | | |
| `retain50` | MTP depth 2 | 10.28, 11.25, 11.36 | 47 | 86 of 94 (91%) |

All three `q23` runs produced the same 134 tokens (ordinary and MTP), ending on a
stop token, and the answer passes the 400,538-case prime oracle. The first
depth-2 repetition spent 790 ms drafting against about 310 ms afterwards (first
use of the draft layer). The pack's depth-2 round takes 168 ms: CPU experts 118
ms (2.5x one ordinary token's 47.9 ms), GPU 37 ms. Depth 3 is slower with the
pack because acceptance drops to 83% on this answer, so depth 2 stays the default.
`retain50` could not move 11.9 million pages (45.5 GiB) to their node: the original
and pack pages share the page cache, and those rows run as remote reads. Its MTP run is slower than its ordinary run;
as measured, `retain50` is not a usable speed option.

### Faster Q2_K/Q3_K CPU kernel (2026-10-06)

A single core running the old kernel on cache-resident weights reached only 4.1
GB/s, the same per-core rate the full model reached from DRAM (60-65 GB/s over
15 workers). The CPU expert phase was therefore instruction-bound, not
bandwidth-bound, which also explains why huge pages and prefetch did nothing.
`q23_avx2.cpp` now:

- unpacks each 32-value chunk once and builds both 16-value scales with one byte
  shuffle from a lane-duplicated scale vector, instead of two scalar broadcasts
  and an insert per chunk;
- keeps Q3_K codes unsigned (code + 4) and subtracts 4 x the activation pair
  sums (precomputed once per call) in int16 before the scale multiply, which
  removes the per-chunk sign handling;
- builds the decoded Q3_K scales from registers. Storing them as four 32-bit
  words and reloading 16 bytes defeated store forwarding; `perf annotate`
  attributed 23% of the Q3_K loop's samples to the reload.

Every lane's integer sum and the float accumulation order are unchanged, so
outputs are bitwise identical to the previous kernel (`q23_test`,
`q23_pool_test`, `flow_pool_test`, and a standalone comparison at 1-8 tokens).

| Single core, weights in L2 (3.4 GHz, no boost) | Old GB/s | New GB/s |
| --- | ---: | ---: |
| Gate/up (Q2_K) + down (Q3_K), one token | 4.06-4.11 | 6.01-6.06 |
| Gate/up only | | 5.77-6.03 |
| Down only | | 5.63-5.70 |
| Three tokens per expert | 2.10-2.12 | 2.20-2.21 |

Correction, same day: this kernel only runs with `STRATA_Q23_AVX2=1`. Every pack
measurement in this document, including the quality gates, used ggml-cpu's
per-token `vec_dot_q2_K_q8_K`/`vec_dot_q3_K_q8_K`, which is as fast as the new
kernel (measured in an otherwise idle queue slot, single core pinned to core 3):

| Kernel | One core, in cache | Three tokens, in cache | 15 threads from DRAM |
| --- | ---: | ---: | ---: |
| ggml-cpu (default pack path) | 5.77-6.32 GB/s | 2.10 GB/s | 43.6-44.7 GB/s (busy machine), 4.78 GB/s per thread (quiet) |
| New `q23_avx2` | 5.71-6.23 GB/s | 2.39 GB/s | 41.1-42.2 GB/s (busy), 4.71-5.04 GB/s per thread (quiet) |

The earlier "4.1 GB/s" was the previous `q23_avx2` kernel, which nothing used.
In the model, the pack with `STRATA_Q23_AVX2=1` gave the same throughput as the
default (MTP depth 2: 17.20-17.23 against 17.25 tok/s; ordinary 13.81-13.95
against 13.83-13.87). On the first 16 held-out sequences, KL is 0.1396 with the new kernel, 0.1382 with the same binary on ggml's kernel, and 0.1366 for the frozen quality binary (before this session's GPU changes): rounding-level differences.
The faster kernel stays opt-in. Software prefetch (0-4096 bytes ahead) does not
change the 15-thread streaming rate of the new kernel (4.70-5.04 GB/s per
thread). The earlier prefetch table measured nothing: the kernel benchmark ran
ggml's kernel, which has no prefetch hook.

Per core, the model reads experts at 4.0-4.3 GB/s while the same kernel runs at
about 6 GB/s from cache and 4.8 GB/s when 15 threads stream private buffers, so
the expert phase is memory-limited at about 58-61 GB/s, and a plain 15-thread
stream with this kernel reaches about 72 GB/s.

### Three-fixture benchmark, pack (2026-10-06)

`tools/glm_q2_coding_bench.py` (chat template, low reasoning, greedy, up to 512
tokens, three decode repetitions after warm-up, EOS-aware) with the `q23` pack,
mapped node-owned rows, 3.5 GiB GPU tier, CPU draft experts and the switches in
[glm53f-q2-v2-pack.json](../configs/glm53f-q2-v2-pack.json) plus
`STRATA_GLM_MTP_SKIP_ANCHOR=1`. Another agent's compiler jobs ran intermittently
throughout (each run started when total CPU use had stayed under 15% for 30
seconds); a process swap of up to 256 MiB was tolerated.

| Fixture | Tokens (ending on stop) | Ordinary tok/s | MTP depth 2 tok/s | Drafts accepted | Oracle |
| --- | ---: | --- | --- | ---: | --- |
| prime | 134 | 14.11, 14.07, 13.57 | 16.96, 17.38, 17.43 | 87/92 (95%) | 400,538 cases pass |
| json_escape | 210 | 14.68, 14.81, 14.84 | 16.66, 16.82, 16.54 | 137/144 (95%) | 2,052 cases pass |
| csv | 367 | not clean: both runs swapped (171 and 235 MB): 4.59, 9.96, 12.81 and 4.47, 8.35, 12.86 | 15.44, 15.43, 15.62 | 233/268 (87%) | fails one case (as in the gate run) |

The MTP answers differ from the gate run's because GPU-tier experts round
differently from CPU experts. Each fixture's three repetitions produced the same
tokens, and MTP produced exactly the ordinary decode's tokens on all three fixtures.

### Toward 20 tok/s: what was tried (2026-10-06)

Pack MTP depth 2 on the prime fixture takes about 160 ms per round for 2.78
delivered tokens: CPU experts 114 ms, GPU work the CPU waits on 35 ms (head,
between-layer and tail), drafting and resync about 12 ms. 20 tok/s needs about
139 ms.

| Attempt | Result |
| --- | --- |
| Faster Q2_K/Q3_K kernel | No gain in the model; ggml's kernel was already as fast (above) |
| Software prefetch, huge pages | No change |
| `STRATA_GLM_ROUTE_MIN_SHARE=0.06` (drop selected experts below 6% of the routed weight, opt-in) | 15.4% of routes dropped, CPU experts 114 to 104 ms/round, but draft acceptance fell from 89% to 83%: 17.42-17.46 tok/s, no net gain. Held-out (first 16 sequences): KL 0.188 against 0.137, top-1 85.9% against 87.9% |
| Same at 0.10 | KL 0.426, top-1 78.7%: unusable |
| `STRATA_GLM_TIER_ADAPT=1` with the pack | Tier hits 15.0% of routes against 6.8% static, but outputs vary between repetitions (resident experts round differently) and interference hid the throughput difference |

GPU timeline of one pack MTP run (`nsys`, 23 rounds, per round): the
multi-column GEMV takes 20.7 ms (q/k/v projections 8.5 ms at about 311 GB/s,
attention output 5.9 ms at about 396 GB/s, shared expert overlapped with the
CPU), tier experts 10.6 ms (overlapped), the MLA sparse-attention core in cuBLAS
SIMT SGEMM 2.5 ms (one call per token), KDA recurrence 2.5 ms, LM heads 2.7 ms,
and about 1,500 small kernels. Non-expert weights are 7.05 GB, so reading them
once per round at 448 GB/s would take about 14.5 ms.

Bound with the current structure: a round reads about 7 GB of routed experts on
the CPU. At the 72 GB/s that 15 threads reach streaming this kernel, that is
97 ms; with 35 ms of serial GPU work and 12 ms of drafting, a round takes at
least 144 ms, about 19.3 tok/s at prime's acceptance. Reaching 20 and above
needs one of:

1. Overlap: split each verify window into two micro-batches so the GPU runs
   one group's attention while the CPU runs the other group's experts. It
   hides most of the 35 ms but reads about 13% more expert bytes (the groups
   share fewer experts), so the estimate is about +9%. It needs a second set of
   step buffers, split layer graphs (pre-wait and post-wait), per-group
   mailbox slots and history offsets.
2. GPU fusion: q/k/v at full bandwidth, batched MLA attention for the verify
   window, fewer small kernels; about 5-10 ms per round. Not a quick fix:
   reusing each block's activation words across 4 rows (bitwise equal, tried
   and reverted) helps only while a matrix sits in the 32 MB L2. With weights
   read from DRAM (six copies rotated), three-column Q6_K 4096x8192 took
   120-142 us with 4 rows per block against 90-96 us with 1, and Q5_K 84-108
   against 66-71 us (another process was also using the GPU).
3. In-model expert bandwidth: little left. `strata-glm-q2-flow-bench`
   (`tools/glm_q2_flow_bench.cpp`: synthetic Q2_K/Q3_K experts in node-owned
   rows, random routes, the decoder's layer dataflow, no GPU) reproduces the
   model's rate: one token 66.7 GB/s (8 experts per layer, 1.09 ms), three
   tokens 61.1 GB/s (21.9 experts, 3.27 ms), phased dispatch 61.2/61.3 GB/s.
   Tasks per thread 4/8/16 give 61.0/63.8/66.4 GB/s (one token) and
   62.3/61.0/60.7 GB/s (three tokens); `STRATA_Q23_AVX2=1` gives 68.7 and
   64.4 GB/s. One-token layers already reach about 93% of plain 15-thread
   streaming with this kernel (72 GB/s); three-token windows lose a little more
   to the multiply-bound shared-expert jobs.
4. Fewer bytes: Q2_K down projections (about 8% fewer bytes) or other lossy
   formats, each needing a new pack and the full quality gates.

### Split verify: GPU attention overlapped with CPU experts (2026-10-06)

`STRATA_GLM_SPLIT_VERIFY=1` (opt-in, needs the step pipeline) runs each verify
window as two token groups: A is the first half (rounded down), B the rest. The
launcher enqueues, per layer, group A's combine of its previous MoE layer and its
layer up to the mailbox publication, then the same for group B. The GPU
therefore computes one group's attention while the CPU computes the other
group's routed experts. Group A runs ahead through the dense layers to its first
publication. Layer state still advances in token order (A's tokens before B's at
every layer), and every kernel in the window is width-invariant, so each token's
logits are those of the unsplit window.

What changed:

- `enqueue_moe_pipelined` is two halves, `enqueue_moe_publish` (router,
  publication, resident experts, shared expert) and `enqueue_moe_combine`
  (wait and add); the unsplit path calls them back to back.
- Group B has its own main buffers (`streams`, `x`, `y`, `hc_coeff`, router
  IDs/weights), mailbox slots after group A's, and its own resident-expert
  executor.
- The CPU service gets each pass's slot, width and position explicitly; the
  launcher thread is the only one that changes `batch_tokens` and `position`.
- Rollback history (`capture_state`, KDA snapshots) is offset by the group's
  first token.
- A KDA MoE layer's segment (previous combine through publication) is a CUDA
  graph per group, width, first token and combine flag. MLA layers stay eager,
  as in the unsplit path, because their kernels take the position as an
  argument. Without these graphs the launcher spent 119 ms per round launching
  kernels and the split was no faster.

Exactness: `--check-verify-graphs` (widths 1-4, warm-up, capture, replay,
history reallocation, rollback) and `--check-verify` (retained prefixes,
sequential logits, device rollback) pass with the split on; 256 split windows ran
in the checks. Pack MTP runs produce the same 134 tokens with and without it.

Pack, MTP depth 2, prime fixture, alternated in the same period (another agent
was active; both arms were slower than the quiet runs earlier in the day):

| Run | tok/s | CPU experts ms/round | GPU waited ms/round | Head ms/round |
| --- | --- | ---: | ---: | ---: |
| Split | 17.55, 18.12, 18.10 | 137.5 | 4.4 | 3.2 |
| Unsplit | 15.89, 15.96, 15.46 | 130.4 | 31.6 | 3.9 |
| Split again | 18.30 (later repetitions hit by interference: 7.34, 5.42) | | | |

The split hides almost all GPU work for about 5% more CPU expert time (experts
shared by both groups are read twice): about +14% here.

Three-fixture benchmark with the split (same settings as the pack fixture table
above plus `STRATA_GLM_SPLIT_VERIFY=1`; each fixture's repetitions produced the
same tokens; no swap):

| Fixture | Split MTP depth 2 tok/s | Unsplit, same period | Unsplit, earlier in the day |
| --- | --- | --- | --- |
| prime | 18.23, 18.28, 18.11 | 15.79, 15.79, 16.16 | 16.96, 17.38, 17.43 |
| json_escape | 18.40, 18.22, 18.39 | rejected (swapped 267 MB) | 16.66, 16.82, 16.54 |
| csv | 17.31, 17.51, 16.74 | not run | 15.44, 15.43, 15.62 |

With the adaptive tier on top (`STRATA_GLM_TIER_ADAPT=1`), outputs varied
between repetitions (134, 149, 131 tokens) and throughput was 15.86, 17.06,
17.55 tok/s (a compile overlapped the first repetition), so it is not part of the
recommended settings.

### REAP-50 GGUF (2026-10-06)

`/mnt/nvme01/models/glm53f/reap-50/GLM-5.3-Flash-REAP50-Q4_K_M.gguf` (99.3 GB,
one file): the same network with 144 routed experts per layer instead of 288,
top-8 routing, Q4_K gate/up and Q4_K or Q6_K down experts (15.1 MB per expert,
5.07 GB of routed experts per token against 3.09 GB for the Q2_K/Q3_K pack),
Q4_K/Q5_0 dense matrices, and the MTP layer.

Running it needed these changes (the original GGUF's behaviour is unchanged):

- Metadata from a newer converter: architecture `glm5-next`,
  `attention.hc.{mult,eps,sinkhorn_iters}`, `ssm.gate_lower_bound`, a scalar
  `swiglu_limit`, no `attention.indexer.kpool` (4) and no
  `expert_shared_feed_forward_length` (the expert width); tensors `hc_*` without
  `.weight` and `indexer.kpool_{ape,gate}`. The descriptor reads either spelling;
  `ModelArtifact::at` and the GPU weight cache map the new tensor names to the
  original ones.
- `ssm_a` is stored positive (the original stores it negative); every other
  shared FP32 tensor is bit-identical. Before the fix the first held-out sequence
  scored perplexity 93.5 and generation looped; `weight()` now negates an
  all-positive `ssm_a`.
- 144 experts: per-expert byte sizes come from the tensor's expert dimension
  instead of 288, the prefill uses `experts / 16` groups, and the prefill ring
  slot grows to 16 of the largest experts (about 231 MiB here instead of 208).
- Prefill experts: the build needs `-DSTRATA_MMQ_KQUANTS=ON`, which now also
  compiles the Q6_K MMQ instance.
- Node-owned rows over the file mapping (`STRATA_GLM_Q2_NUMA_WEIGHTS=1
  STRATA_GLM_MAPPED_OWNED=1`): `move_pages` moves only mapped pages, so each
  half is now pre-mapped (`MADV_POPULATE_READ`). Without it 22.2 million pages
  (85 GB) stayed where they were and decode ran at 7.3 tok/s.
- The GPU tier is not available yet: the GPU expert kernels have no Q6_K path.

`glm_quant_parity --dense` (new) checks every dense GEMV of a GGUF against the
dequantized rows; for this file all formats agree except the 128-wide Q5_0
projections, where the kernel (like llama.cpp's) applies the Q5_0 offset with
the unquantized activation sum, so the quantized-sum reference differs by about
1%.

First held-out sequence (512 tokens, decode path): perplexity 8.11 against 7.67
for the original UD-Q2_K_XL and 8.05 for the `q23` pack.

Prime fixture, all routed experts on the CPU (no tier), three repetitions,
outputs identical across modes (125 tokens):

| Mode | tok/s | CPU experts |
| --- | --- | --- |
| Ordinary | 10.34, 10.85, 10.91 | 66 ms/token (about 77 GB/s) |
| MTP depth 2 | 11.98, 11.87, 11.90 | 79/92 drafts accepted (86%) |
| MTP depth 2, split verify | 12.53, 12.54, 12.41 | |

REAP-50 at Q4_K_M reads 64% more expert bytes per token than the pack, so it is
slower here despite the CPU reading Q4_K faster (about 77 against about 60
GB/s). Its advantage is size: a Q2_K/Q3_K conversion of the 144 experts would
need about 56 GB, which leaves room for more node-local caching or a larger
share of experts on the GPU.

### Quality gate setup

Held-out evaluation uses the frozen corpus (64 sequences x 512 tokens, 32,768
targets, half code and half prose) through the same decode path for the baseline
and every candidate, with no GPU tier, so KL measures only the expert conversion.
The pipelined evaluator matches the non-pipelined one exactly (KL 2e-20, top-1
100%).

| Experts | Held-out perplexity | Change | KL from original (nats/token) | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| Original UD-Q2_K_XL (IQ2_XS gate/up, IQ3_XXS down) | 8.566 | | | |
| Gate/up converted to Q2_K, down kept (`q2` profile) | 8.850 | +3.32% | 0.146 | 83.8% |
| Gate/up Q2_K and down Q3_K (`q23` profile, full pack) | 8.861 | +3.45% | 0.158 | 83.2% |
| `retain50` profile: 62 of 126 projections kept in the original format | 8.625 | +0.70% | 0.076 | 88.4% |

Almost all of the loss comes from gate/up: IQ2_XS has six signed levels per
16-weight group, Q2_K four. A lossless SIMD-friendly re-encoding of IQ2_XS needs
about 3.3 bits/weight, which would not fit the 120 GB host budget.

The coding fixtures (one greedy answer each, new decode path) do not separate the
two: both pass prime (400,538 cases) and JSON (2,052 cases), and both fail CSV on
different cases. The original weights passed CSV with the older kernels, so one
greedy answer flips with rounding-level changes; HumanEval (164 tasks) is the
deciding coding check.

HumanEval (greedy chat, low reasoning, one sample per task, tests run in a
network-less bubblewrap sandbox, up to 2,048 generated tokens, frozen decoder
binary, process swap at most 352 MiB):

| Experts | Completed | Passed | pass@1 | Mean generated tokens |
| --- | ---: | ---: | ---: | ---: |
| Original UD-Q2_K_XL | 164/164 | 153 | 93.3% | 120.9 |
| Q2_K/Q3_K pack (`q23`) | 164/164 | 155 | 94.5% | 116.6 |

The two runs disagree on 10 tasks: 4 pass only with the original weights and 6
only with the pack (exact sign test p = 0.75). 45 of 164 answers are
token-identical. The pack therefore shows no measurable coding loss at this
sample size, despite its held-out KL. That is the gate the plan set for lossy
options, so `q23` is accepted for the throughput work below; it stays opt-in.

## Progress (2026-10-07): placement-independent experts, adaptive tier, routing affinity, 64 GB modes

Goal raised to 30 tok/s on the single node, plus a configuration for a 64 GB host. Everything
below is opt-in by environment variable; the previous configurations are unchanged. Work directory
`build-q2-v3/` (`run.sh`, `eval.sh`, `fixtures3.sh`, `summarize.py`, `nsys.sh`); server configurations
`configs/glm53f-q2-v3*.json`.

### Starting point (quiet machine, 2026-10-07)

Three-fixture benchmark, pack, MTP depth 2, split verify, 3.5 GiB static tier: prime 18.9, json_escape
20.6, csv 18.3 tok/s. A round is about 140 ms for 2.8 tokens: CPU experts 120 ms, in-step GPU waits 11
ms, draft and resync 13 ms. A token reads 3.06 GB of routed experts; the CPU reads them at 65-70 GB/s.
30 tok/s is 92 GB/s of expert traffic, so about 40% of it has to leave the CPU.

### Canonical expert arithmetic (`STRATA_GLM_CANON=1`)

`include/strata/kernels/canon_expert.hpp` defines one arithmetic for a routed Q2_K/Q3_K expert that the
CPU (`q23_avx2.cpp` dot, `canon_expert.cpp` quantizer and SwiGLU, built with `-ffp-contract=off`) and
the GPU (`src/kernels/cuda/glm_q23.cu`, one `__dp4a` per lane and chunk, explicit `__fmaf_rn`) produce
bit for bit: Q8_K activations, integer sums per superblock, the AVX2 kernel's eight float lanes and
reduction order, a shared polynomial `exp`, and the routed sum as one fma chain in descending route
order. `glm_q23_parity` checks every output row of random experts and real-sized layouts (both down
formats, one to six tokens, gate skipping) with `memcmp`; `canon_expert_test` checks the quantizer
against ggml's and the exp against libm. In the model, the tier contents never change a logit: the
ordinary decode of 511 tokens with 405 resident experts and with none gives identical tokens and
bit-identical final logits; `--check-verify` and `--check-verify-graphs` pass with a tier under split
verify. Held-out quality (first 16 sequences, against the original model's logits) is the legacy
path's within rounding: perplexity 4.825 (legacy 4.849), KL 0.1367 (0.1382), top-1 87.7% (87.7%).

The GPU kernels run at about 235 GB/s of expert bytes on 8-24 experts per launch (the old tier kernels
reached about 50 GB/s in the model) and 0.047 ms for one expert (a warp-parallel Q8_K quantizer, loads of
several superblocks issued together).

Two combine modes. Strict (lossless configurations): the GPU posts its resident rows into the mailbox
and the CPU adds every route in the canonical order, so logits do not depend on placement. Device
(`STRATA_GLM_CANON_COMBINE=device`, automatic when routing follows residency): the GPU adds its rows to
the CPU's partial sum on the device, which saves the row transfers; results then depend on placement at
rounding level only, and the tier's evolution is deterministic by step count.

### Adaptive GPU tier

`STRATA_GLM_TIER_ADAPT=1` is now usable: with canonical arithmetic its promotions do not change
outputs, and they follow step counts (selected in step s, staged on a background thread, uploaded after
step s+1, in the lookup at step s+3), so a run's residency sequence is reproducible. Scores start from
the selection's expected route rates (`STRATA_GLM_TIER_SEED`), and promotion is planned once per step
instead of per route on the decode thread. `STRATA_GLM_GPU_RESERVE_MIB=512` and
`--gpu-budget-mib=14336 --decode-cache-mib=5632` give a 4.8 GiB tier beside the desktop session; the
verify history allocation now persists across requests and the tier leaves 1.28 GiB for MTP's draft
weights, history and checkpoints (both had silently fallen back to replay steps or depth 1 when VRAM
was tight). `GPU_LIVE` prints device memory by category: dense weights 6154 MiB, state 333, MTP dense
221, verify history 291, tier the rest.

On the same 512-token prime answer (82% draft acceptance): static tier 18.0 tok/s, adaptive 21.0
(CPU bytes per round 8.26 -> 6.95 GB). Simulation on the three fixtures' routing traces
(`tools/glm_residency_sim.py`): prior-only static 6%, adaptive 23.5% of routes at 4.8 GiB, 28% at 6.5 GiB.

Three-fixture benchmark, lossless configuration (`configs/glm53f-q2-v3.json`: canonical + adaptive
tier + split verify): prime 20.6, json_escape 23.6, csv 17.5 tok/s (baseline 18.9 / 20.6 / 18.3). The csv answer changed (501
against 478 tokens) and its draft acceptance fell from 81% to 73%; one of its trials replayed 110 tokens.
Per round on json_escape: 125 ms (CPU 98, GPU waits 10, draft 7, resync 6) against 140 ms before.

### Routing affinity (`STRATA_GLM_ROUTE_AFFINITY=x`, lossy)

The router adds `x` (probability units) to the selection score of GPU-resident experts; weights keep
the true probabilities. Resident experts then win near ties, and the GPU share rises without more VRAM.
The same per-expert bonus table drives the prefill and decode routers; the evaluator builds a tier per
sequence so these modes are measured as they run.

| Margin | Routes changed (sim) | GPU share (sim) | Held-out ppl / KL / top-1 (16 seq.) | prime tok/s (warm tier) |
| --- | ---: | ---: | --- | --- |
| 0 | 0 | 23.5% | 4.825 / 0.1367 / 87.7% | 21.0 |
| 0.02 | 5.7% | 28.6% | not measured | 24.5 |
| 0.05 | 16% | 37.9% | 4.842 / 0.1426 / 87.5% | 25.4 |
| 0.10 | 33% | 52.8% | 4.966 / 0.1648 / 86.9% | 27.0-28.6 |
| 0.15 | 46% | 62% | 5.117 / 0.1983 / 85.9% | 28.1-28.6 |

Above 0.10 the CPU is no longer the bottleneck: at 0.15 the CPU expert phase is 49 ms per round but
the CPU waits 37 ms for the GPU. An nsys timeline at 0.10 (`build-q2-v3/nsys.sh`) gives per round:
dense Q5_K/Q6_K GEMVs 29.5 ms (split verify runs them once per token group, 10 GB per round at 346
GB/s), resident experts 12.7 ms, KDA 3.7, small Q8_0 GEMVs 6.1, LM heads 3.0, norms 2.3; about 72 ms
of kernel time in an 87 ms step. Both processors are busy about 70-80% of a step; the rest is the
per-layer ping-pong (one group's attention chain, most of it the MLA layers' eager kernels, is
longer than the other group's CPU pass).

### Gate-threshold row skipping (`STRATA_GLM_GATE_SKIP=thresholds.json`, lossy)

In canonical mode a unit whose |silu(gate)| is below its layer's threshold contributes exactly zero on
both devices, and the CPU does not read its up row (about a third of an expert's bytes).
`STRATA_Q23_GATE_STATS` records |silu(gate)| histograms; `tools/glm_gate_thresholds.py` turns them into
per-layer thresholds for a target share. Skipping 28% of units: perplexity 4.848 (+0.5%), KL 0.1455,
top-1 87.3%; 46%: 5.023 (+4.1%), KL 0.188. The 28% setting saves about 9% of CPU expert bytes.

### Q2_K down projections (`experts-q22.gguf`)

`strata-glm-q2-pack --down=q2 --reuse=experts-q23.gguf` converts only the down projections (the
gate/up bytes are copied from the q23 pack and checked), 95.9 GB instead of 105.5. Held-out: perplexity
4.867, KL 0.1637, top-1 86.5%, about as lossy as routing affinity 0.10 for 9% fewer bytes. Not timed;
`--all-layers` (also converts layer 11-44's IQ3_XXS/IQ4_XS exceptions) is implemented but no pack was
built.

### Prefill: chunk size

Each prefill chunk streams every routed expert to the GPU once (about 105 GB at about 7.5 GB/s over the
5060 Ti's x8 link, 14 s), so prefill speed is close to chunk / 14 s. 7,807-token prompt, 16k context:

| Chunk | tok/s | GPU allocated |
| --- | ---: | ---: |
| 256 | 13-15 | |
| 2048 | 89 | 8.95 GiB |
| 4096 | 153 | 10.1 GiB |
| 8192 | 220 | 12.4 GiB |

8192 is the default in the v3 configs and scripts. The GPU expert tier is built after prefill, so it
does not compete for this memory. The prefill arena needed headroom above an 8k context (`mla_value`
overflowed it); the overflow error now names the buffer.

### Long-context codegen fixture

`long_cpp` (`tools/glm_q2_coding_bench.py --fixtures long_cpp --context 16384 --tokens 4096`): two frozen
headers (`docs/fixtures/glm_q2_long_cpp`, 7,795 prompt tokens) and a request for a GoogleTest file. At
affinity 0.10: decode 21.1 and 19.4 tok/s, draft acceptance 73% and 59%, CPU expert rate 61 GB/s;
both trials stopped at the 4,096-token cap (17 coherent TEST cases, last one cut off).

### HumanEval

Lossless configuration (canonical, 4.8 GiB adaptive tier, MTP depth 2, split verify): 156/164 (95.1%;
original weights 153, legacy pack 155). The affinity runs were stopped (0.05: 53/56 when stopped).

### 64 GB host

Measured under a 60 GiB memory cgroup (`systemd-run --user --scope -p MemoryMax=60G -p MemorySwapMax=0`)
with the model files dropped from the page cache first (`build-q2-v3/drop_cache.py`). The pack was moved
to the x4-linked disk (`/mnt/nvme02/models/glm53f/q2-cpu-experiment/experts-q23.gguf`, symlinked from
the old path); the other drive (`nvme0n1`) trains at PCIe x1 in a x4 M.2 slot (a seating or Gen4-in-Gen3
training problem, not lane sharing).

| Mode | Warm tok/s | Held-out ppl / KL / top-1 |
| --- | ---: | --- |
| `exact`: lazy mapping, misses read on first use | 2.4 (1.5 on the x1 disk) | lossless |
| prefer-resident, frozen 48 GB set (bonus 4: never leaves the set) | 23 | 5.578 / 0.371 / 79.8% |
| prefer-resident, rebalance-only, hard ban | 22.9 | 5.641 / 0.405 / 79.1% |
| **hybrid, `STRATA_GLM_RAM_MARGIN=0.10`** (default in `glm53f-q2-v3-64g-resident.json`) | **15.1** | **4.888 / 0.150 / 87.0%** |
| hybrid, margin 0.05 | 10.9 | 4.850 / 0.143 / 87.8% |
| REAP-50 pack (55 GB) in RAM | 15.4-16.1 | 11.1 / 0.587 / 69.8% (all 64 sequences; original 8.57) |

`STRATA_GLM_MAPPED_LAZY=1` maps the pack without reading, moving or checksumming it (load 4 ms instead
of 135 s) and prefill stages only routed experts, placing each half's pages on its node.
`STRATA_GLM_PREFER_RESIDENT_MIB=N` keeps N MiB of experts in memory (`RamTier`): the router adds
`STRATA_GLM_RAM_MARGIN` to in-memory experts' selection scores; an expert outside the set that still
wins is read by the workers' page faults and joins the set, the coldest leaving at the end of the step.
Scores count the unbiased router's choices only (published through the mailbox, and from a second
unbiased routing pass in prefill); counting the routes taken made the set confirm itself. Bulk
rebalancing every 512 tokens (2048 past 4096 tokens of context) needs at least 256 tokens of evidence
and moves at most 5% of the budget. Background read-ahead between rebalances (`STRATA_GLM_RAM_ADMIT`)
cost half the speed (10,000 reads competing with decode, CPU expert rate 26 GB/s) and is off in the
config (`1e9`). The first repetition after a cold start runs at about half the warm rate while the set
moves to the prompt's topic.

### Other changes

- `tools/glm_q2_coding_bench.py --allow-different-trials` for residency-dependent routing;
  `tools/glm_q2_humaneval.py --gpu-budget-mib --decode-cache-mib --speculative --draft-depth`.
- `STRATA_GLM_ROUTER_SCORE_TRACE=path` (non-pipelined decode) records router logits for the simulator.
- `STRATA_GLM_STEP_TRACE=2` prints per-layer CPU waits and expert time for both token groups.
- The decode cache cap is 12288 MiB in the CLI and the server.

### Next steps

1. The GPU is the limit above affinity 0.10 (CPU waits 26-37 ms per round for it): fuse the dense
   chain's small kernels, make MLA layers graphable (position as device data), and run the dense
   GEMVs once per verify window instead of once per token group. Each would remove several ms per round.
2. Headless operation would add about 2 GiB of tier (about 5 points of GPU share).
3. HumanEval for the chosen fast configuration (affinity 0.05-0.10).

## Where a token goes today

Ordinary decode, prime fixture, current configuration
([glm53f-q2-mtp.json](../configs/glm53f-q2-mtp.json) environment), 9.74 tok/s:

| Part | ms/token | Source |
| --- | ---: | --- |
| CPU routed experts, gate/up (IQ2_XS, about 1.65 GB, about 37 GB/s) | 44.4 | measured, `CPU_EXPERT` |
| CPU hidden quantization (host thread only) | 1.45 | measured, `CPU_EXPERT` |
| CPU routed experts, down (IQ3_XXS, about 1.10 GB, about 42 GB/s) | 26.2 | measured, `CPU_EXPERT` |
| GPU kernels on the serial path | about 23.2 | measured 25.3 busy, minus about 2.1 of shared expert that overlaps CPU work (estimate) |
| Per-layer handoff: event wait, host job build, host sum, synchronous upload, launches | about 7.4 | remainder (estimate) |
| **Total** | **102.6** | measured |

GPU kernel time per decode token (nsys, 46 decode tokens, 1987 kernels/token):

| Kernel group | us/token | Note |
| --- | ---: | --- |
| Q5_K + Q6_K GEMV (5.1 GB) | 13,200 | 380-393 GB/s, already near the 448 GB/s limit |
| Q8_0 small GEMV, 226 launches (0.77 GB) | 2,417 | 319 GB/s |
| RMSNorm, 214 launches | 1,638 | 7.7 us each for 4096 values |
| MLA through FP32 cuBLAS/cutlass (gemv, sgemm, dot, gather, softmax, ...) | about 3,400 | `attn_k_b`/`attn_v_b` expanded to FP32 (about 0.8 GB VRAM) |
| KDA chunk + conv + output + gate | 1,714 | 40 us per layer for the recurrence |
| Router GEMV + selection | about 900 | |
| mHC, hc_read/write, quantization, other | about 1,100 | |
| Output head Q4_K | 845 | |

The trace's CPU timing is not usable (another process evicted model pages during
that run); GPU kernel durations are unaffected.

MTP depth 2 on the same fixture (measured): 252 ms per round for 2.89 delivered
tokens (11.3 tok/s). CPU experts take 190 ms for the union of three tokens
(6.77 GB, 2.46x one token's bytes); verify GPU work and handoffs take 51 ms;
drafting 6 ms and MTP resync 5 ms.

Hardware floors: 2.751 GB of routed experts at 90 GB/s is 30.6 ms; 6.45 GB of
fixed weights at 448 GB/s is 14.4 ms. The CPU is compute-bound in the IQ2_XS and
IQ3_XXS codebook decode, not bandwidth-bound: 37-43 GB/s of 90.

## What the measurements rule out

- Faster lossless IQ kernels are not a large lever. The table-sign kernel is close
  to ggml's design; gather, paired-row, compact-accumulator, SMT and compiler
  experiments were neutral or slower ([NUMA decode report](GLM_Q2_NUMA_DECODE.md)).
- Streaming experts to the GPU at 7.5 GB/s is 12x slower than reading them on the CPU.
- MTP alone cannot reach 15 tok/s: three verified tokens read 2.46x the bytes of one.
- A lossless SIMD-friendly re-encoding of IQ2_XS needs 3 bits/weight (six levels),
  which grows the routed weights from 99 GB to about 128 GB. It does not fit.

## Target architecture

### 1. GPU-driven decode step (lossless)

Replace per-layer host orchestration with one CUDA graph per decode step and
width (1-4): 45 layers, output head and greedy argmax. For each MoE layer the GPU:

1. runs the mixer, mHC, norm, router and selection;
2. writes the hidden activation already quantized to Q8_K (byte-identical to
   `quantize_row_q8_K`), the eight expert IDs and weights, and a sequence number
   into a mapped pinned mailbox;
3. runs the shared expert and any GPU-resident experts (part 3);
4. waits on `cuStreamWaitValue32(done[l] >= seq)`;
5. adds the CPU routed sum, read directly from mapped memory (16 KiB/token).

CPU workers stay pinned and spin on the mailbox. The host thread does not take
part per layer. The worker that finishes a layer's last task writes the weighted
routed sum in a fixed order and then the done flag.

**Measured** on this machine: a mailbox round trip with a 16 KiB payload each way
and three kernels takes 15.6-17.1 us, or 0.66-0.72 ms for 42 layers, with either
a spin kernel or `cuStreamWaitValue32`. That replaces about 7.4 ms of handoff.
Greedy argmax on the GPU also removes the 620 KB logits copy per token.

### 2. CPU expert service as a dataflow (lossless)

Today each layer is three global phases: gate/up rows, host-only intermediate
quantization (1.45 ms/token), then down rows. Replace them with per-expert
dependencies. The worker that completes an expert's last gate/up chunk applies
SwiGLU and quantizes that expert's intermediate, which releases its down chunks.
Down tasks start while other experts are still in gate/up, so no barrier waits.
Keep node-owned rows (`STRATA_GLM_Q2_NUMA_WEIGHTS` layout) so each node reads
its own DRAM. `strata-glm-q2-kernel-bench` with real routes is the test bed.
**Estimate:** 3-5 ms/token.

### 3. GPU expert tier (lossless weights)

Expert use is highly concentrated. From the 32K-token calibration counts
(code and prose, in-sample): the 400 most-used experts (3.1 GiB) carry 18.1% of
routed bytes, 500 (3.9 GiB) 20.5%, 750 (5.9 GiB) 25.5%. A uniform cache would
cover 3.3%, 4.1% and 6.2%. An earlier held-out trace covered 21.5% with 3 GiB.

The GPU computes its resident experts between the mailbox write and the wait,
while the CPU computes the rest. It is otherwise idle for about 55-70 ms per token.
Rank by calibration counts, adapt from the prompt's prefill routes, and include
MTP draft experts in the same tier. Free VRAM for it:

- replace the FP32 MLA copies with Q8_0 or F16 kernels (about 0.6 GB);
- trim decode-time workspaces, and let tier slots double as prefill staging.

GPU hits use Q8_1 activations, so outputs are not bit-identical to CPU-only
decode (already the case for the existing cache). An earlier 3 GiB attempt cut
CPU bytes 16.7% but did not raise throughput, because hits added host syncs and
copies. Part 1 removes them. **Estimate:** 18-20% of CPU expert time, about
13 ms/token, with about 4 GiB of slots.

### 4. GPU fixed path (lossless)

Target 25.3 to about 17.5 ms busy:

- fuse RMSNorm with hc_read/hc_write and mHC coefficients (214 launches);
- MLA decode from Q8_0 weights with fused gather, score, softmax and value for
  1-4 tokens, instead of FP32 cuBLAS;
- KDA recurrence near its state traffic (about 4 MB/layer, about 12 us);
- merge small Q8_0 GEMVs into fewer launches;
- fuse router GEMV, sigmoid and top-k;
- width 2-4 GEMVs that read each weight once. Width-3 verify GPU work plus
  handoffs is 51 ms today, versus about 30 ms at width 1.

### 5. MTP inside the step graph

Draft, verify and resync in graphs, with argmax on the GPU. Choose the depth per
round (1-3) from draft confidence, since each extra verified token adds about
0.73x of a token's CPU bytes.

### 6. Opt-in lossy accelerators (quality-gated)

- **Calibrated Q2_K/Q3_K experts.** Implemented in
  [GLM Q2 expert conversion](GLM_Q2_REQUANTIZATION.md); a 105.5 GB pack exists.
  Expert-only benchmark: 66.6 GB/s versus 40.1 GB/s for the IQ formats, with
  12.7% more bytes. Its qualification stopped because a concurrent `nvcc` build
  tripped the interference guard. A gate/up-only Q2_K profile keeps the down
  projection lossless.
- **Expert skipping by router weight.** Drop experts below a normalized weight
  threshold. Measure with the existing held-out KL/top-1 evaluator and HumanEval.
- **REAP-50.** Pruning halves the expert count, so the same VRAM tier covers
  twice the fraction. Q4_K_M reads about 1.8x the Q2 bytes per token, so it is
  slower unless requantized. Evaluate separately.

## Projected budgets (estimates)

| Configuration | CPU experts | GPU serial | Handoff | ms/token | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Today, ordinary (measured) | 72.0 | 23.2 | 7.4 | 102.6 | 9.7 |
| Parts 1-4, ordinary, original weights | about 55 | about 16 | about 0.7 | about 72 | about 14 |
| Parts 1-5, MTP depth 2, original weights, prime acceptance | about 148/round | about 24/round | about 1 | about 62 | about 16 |
| Same, CSV acceptance (79%) | | | | about 69 | about 14.5 |
| Parts 1-4 + Q2_K/Q3_K, ordinary | about 35 | about 16 | about 0.7 | about 52 | about 19 |
| Parts 1-5 + Q2_K/Q3_K, MTP depth 2 | about 92/round | about 24/round | about 1 | about 43 | about 23 |

With the original weights, 15 tok/s depends on MTP and high-acceptance output.
Reaching 15 tok/s without MTP needs a lossy step from part 6 or a larger GPU
tier. Each GiB of tier slots is worth about 3.5 ms/token. A long context costs
tier slots: the MLA latent cache needs about 22 KB per token in FP32 (11 layers x 512),
or about 11 KB in F16.

Host RAM: the original files peak at 102-105 GiB RSS today. The Q2_K/Q3_K pack is
105.5 GB in owned arenas plus the original fixed and draft tensors at load.
Neither change alters the KV cache, which is on the GPU.

## Order of work and acceptance checks

| Step | Work | Gate before the next step |
| --- | --- | --- |
| 0 | Per-token phase trace (GPU serial, handoff, CPU) without added syncs. On an idle machine, re-measure the baseline, the existing `--decode-prefill-cache`/`--decode-cache-mib` tier and the Q2_K/Q3_K pack. | Budget table confirmed or corrected |
| 1 | GPU-driven step with mailbox CPU service, widths 1-4 | Same greedy IDs as the current CPU path on all three coding fixtures; verify-graph checks; handoff at or below 1 ms/token |
| 2 | CPU dataflow scheduler | Bitwise-equal expert outputs in pool tests; kernel-bench gain |
| 3 | GPU expert tier in the step graph | Measured CPU-byte reduction translates into time; KL/top-1 against the CPU-only baseline unchanged within activation rounding |
| 4 | GPU kernel fusion and MLA Q8_0 | Primitive parity tests; GPU busy at or below about 18 ms/token |
| 5 | MTP in the graph, adaptive depth | Acceptance unchanged; delivered tok/s on three fixtures |
| 6 | Lossy options, one at a time | Held-out KL/top-1, HumanEval 164/164 completed, coding-fixture oracles |

Every throughput claim uses the existing harness: three repetitions on the
prime, JSON and CSV fixtures, EOS-aware delivered-token counts, the swap and
interference guard, and no other heavy process on the machine.
