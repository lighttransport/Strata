# GLM-5.3-Flash FP4-Local on 12 A64FX nodes: forecast, measured kernels, runner plan

Started 2026-10-22. The checkpoint is NVIDIA's quantization-aware-distilled `nvidia/GLM-5.3-Flash-FP4-Local` (revision
76348ddb): NVFP4 routed experts, an FP8 trunk. This note extends [GLM_A64FX_PLAN.md](GLM_A64FX_PLAN.md) (the q23 GGUF
pack on 6/8/12 nodes) to that checkpoint. Numbers marked "measured" ran on A64FX compute nodes; everything else is a
simulator output (`tools/sim/a64fx.py`) with its hypotheses printed next to it.

## 1. The checkpoint, byte by byte

`tools/glm_fp4_local_census.py` reads only the safetensors headers and writes `tools/sim/data/glm_fp4_local_trunk.json`;
its totals equal the census of the V100 FP4 work (`tools/sim/data/glm_fp4_v100.json`) to the byte.

| part | format | bytes |
|---|---|---:|
| routed experts, layers 3-44 (12,096 experts) | NVFP4: E2M1 codes, E4M3 scale per 16 (stored CSF32-compressed), F32 global | 162.5 GB stored, **171.2 GB** with the scales expanded |
| per decoded token, 8 x 42 experts, no reuse | NVFP4, scales expanded | **4.76 GB** (q23 GGUF pack: 3.09) |
| trunk, 45 layers | KDA q/k/v/o, dense FFN: MXFP8 (E4M3 + E8M0 per 32); MLA q_a/q_b/kv_a/o, shared experts: FP8 per 128 x 128 block; KDA f/g/b, MLA kv_b, indexer, routers, mHC: BF16 | 8.26 GB |
| per layer | dense 302.5 MB, KDA MoE layer 174.3 MB, MLA MoE layer 177.2 MB | |
| lm_head | MXFP8 | 0.654 GB |
| embedding | BF16, one row per token | 1.27 GB |
| MTP layer 45 | NVFP4 experts with plain E4M3 scales (4.08 GB), MXFP8 attention/shared, BF16 eh_proj | 4.30 GB |

Memory per node at 32K context (experts, TP12 trunk shares, replicated routers/mHC/embedding, KV, MTP, 1 GiB scratch):
**18.0 GiB at 12 nodes, 25.0 at 8, 32.2 at 6** against 29 GiB usable. FP4-Local does not fit 6 nodes.

## 2. How the simulator models it

`pack fp4_local` and `trunk fp4l` (the CLI picks the census trunk for this pack by itself). The trunk is no longer a
scale of the UD GGUF mix: every matrix streams its own format at the kernel rate for that format, capped by the
measured 504 GB/s streaming chain, with the share the busiest of the tensor-parallel ranks holds (64 attention heads
over 12 ranks: 6 on the busiest, 12.5 % above an even split). Routers and mHC weights stay inside their fixed site
costs. Two numerics modes:

- **fast**: W4A8 experts (int8 activations per 64 values, SDOT per K16 group), MXFP8 and block-FP8 trunk tensors
  requantized to int8 per (row, 32) at load (the bytes of ggml Q8_0), BF16 tensors as stored;
- **exact**: E2M1 x E4M3 products with F32 activations and F32 accumulation, the E4M3 trunk decoded exactly.

The fast mode has to pass a quality gate against the exact one before any fast number is quoted as the runner's.

## 3. Measured kernels (2026-10-22, one node, 48 threads)

Interactive job 52646912, one node, clang 21.1.0 `-O3 -mcpu=a64fx -msve-vector-bits=512`, `src/kernels/cpu/
glm_fp4_sve.c` (then in tools/a64fx_probe; run: `tools/a64fx_probe/run_fp4.sh`). Every kernel matches its scalar replica bit for bit and a
double-precision product of the source values within 1.1e-7 relative L2 (30 checks, 1-4 tokens, subnormal and zero
scales, outlier activations); the E4M3 vector decode is checked against all 256 codes at start-up. GB/s counts the
packed bytes (0.5625 B per NVFP4 weight, as the checkpoint with expanded scales), every thread streaming its own tiles
of first-touched copies larger than the caches.

| kernel | layout | 1 token | 2 | 3 | 4 |
|---|---|---:|---:|---:|---:|
| NVFP4 W4A8, 2048 x 4096 | rt16q: svtbl E2M1 decode, 4 SDOT-by-lane per K16 group | **403** | 356 | 301 | 235 |
| NVFP4 W4A8, 4096 x 2048 | | **405** | 356 | 303 | 237 |
| NVFP4 W4A16 exact, 2048 x 4096 | rt16p: two svtbl_f32 per byte pair | 180 | 124 | 95 | 66 |
| NVFP4 W4A16 exact, 2048 x 4096 | rt16q: byte extraction | 166 | 115 | 84 | 66 |
| MXFP8 exact, 768 x 4096 (one rank's KDA q) | E4M3 by sign-extending load + shift + F16 widen | 270 | 236 | 182 | 108 |
| block FP8 exact, 4096 x 1536 (one rank's MLA o) | same, F32 scale per 128 x 128 | 251 | 114 | 101 | 95 |
| Q8 (fast trunk), 768 x 4096 | int8 per (row, 32), SDOT by lane | 634 | 437 | 382 | 412 |
| BF16, 768 x 4096 | shift into F32 | 481 | 334 | 243 | 220 |

For comparison, the earlier A64FX FP4 kernels in the gemm repository measured 27-34 GB/s (exact NVFP4 with E4M3
gathers) and ~168 GB/s (fused MXFP4 SDOT); Strata's Q2_K SDOT kernel 182 GB/s. The W4A8 kernel is the first FP4
kernel on this machine near the streaming rate: 403 GB/s of 4.5-bit weights is 717 GMAC/s, against 554 GMAC/s for
Q2_K. A 4-way expert slice (512 x 4096) alone has 32 tiles for 48 threads and reaches 218 GB/s; in the runner a phase
streams the slices of all 8 routed experts at once. With an OpenMP barrier after every matrix (in-model phase mode) the
full 2048 x 4096 W4A8 matrix runs at 260 GB/s (18 us per matrix).

Prefill GEMM shapes (`--gemm M`, 2048 x 4096, tokens in blocks of 4 per decoded tile): W4A8 **3.24 TOPS** per node
at M = 16-128, W4A16 exact 1.06, Q8 2.98, MXFP8 exact 0.74. The simulator's prefill model is an int8 SDOT GEMM at
3.18 TOPS (measured in the gemm repository), so the measured W4A8 factor is 0.98 and the exact factors 3.0 (experts)
and 4.3 (trunk). Data: `tools/sim/data/measured_a64fx_kernels.json` sections `nvfp4_kernel_v1`, `trunk_kernel_v1`,
`nvfp4_gemm_v1`; the simulator reads them in place of its hypotheses (`a64fx.fp4_tables()`).

Reads of the checkpoint from the shared file system on compute nodes: one node alone 65-67 MB/s per stream (two
5.1 GB shards); 12 nodes reading 12 different shards at once 27-28 MB/s each, **316 MB/s aggregate**. Staging every
node's 15 GB share at that rate takes about 9 minutes per allocation.

## 4. Forecast, 12 nodes

`python3 tools/glm_a64fx_sim.py report --pack fp4_local`. Profile `redesign` = the proposed runner of GLM_A64FX_PLAN.md
section 4 (streaming phases at the measured chain rate, split-phase barriers, one collective per layer, mHC at today's
measured in-model 42.2 us per site), now with the measured FP4 kernels. Experts sliced over 4 ranks.

| numerics | decode 128 ctx plain / MTP2 | 8K | 32K | prefill 128 / 8K tokens | 8K step (ms) |
|---|---:|---:|---:|---:|---|
| fast | **79.2 / 113.9** | **70.4 / 95.0** | 63.1 / 81.0 | 520 / **740** | 14.2 = trunk 1.8 + experts 1.2 + collectives 2.6 + mHC 3.8 + context 1.6 + head 1.0 + phases and small steps 2.2 |
| exact | 64.0 / 89.5 | 58.2 / 77.4 | 53.1 / 67.8 | 212 / 334 | 17.2 = trunk 3.0 + experts 2.6 + collectives 3.0 + mHC 3.8 + context 1.6 + head 1.2 + 2.1 |

Profile `current` (the fixed costs of today's gemm runner) with the same kernels: 36.3 tok/s fast at 8K. Eight nodes
come out within 1 % of twelve in decode (69.7 fast at 8K; fewer tree steps per allreduce) and 18 % lower in prefill.

What moves the 12-node fast 8K number (plain / MTP2):

| change | tok/s |
|---|---:|
| as forecast | 70.4 / 95.0 |
| mHC fused to the standalone kernel cost (25 us per site) | 79.0 / 104.0 |
| mHC at today's legacy kernel cost (67 us) | 60.8 / 84.4 |
| bf16 collective payloads | 74.3 / 105.1 |
| KDA small steps 40 us instead of 12 | 66.0 / 86.3 |
| 6 us per phase instead of 3 | 65.6 / 91.6 |
| streaming chain 40 % of the HBM stream instead of 55 % | 67.3 / 92.6 |
| NVFP4 W4A8 at 250 GB/s (the pre-measurement hypothesis) | 66.4 / 88.0 |

The expert kernel is no longer the lever: 1.2 ms of the 14.2 ms step. Two thirds of the step are fixed costs the
runner has to bring down (mHC 3.8, collectives 2.6, phases 1.5, context 1.6, KDA small steps), the same conclusion as
for the q23 pack. Remaining FP4-specific hypothesis: the MLA attention core costs 5 us per layer at zero context once
its projections stream at the trunk rate. The general `redesign` hypotheses (KDA small steps as one 12 us phase, mHC
at 42.2 us per site, 3 us per phase, the 504 GB/s chain inside a whole step, prefill GEMM efficiency 0.70, MTP
acceptance 0.977 per position from the Q4 runner) are listed by `report`.

## 5. Golden reference and first runner checks

`tools/glm_fp4_cpu_reference.cpp` is the golden forward: every weight at its checkpoint value, double-precision dot
products, F32 activations, one position at a time, following the CPU reference of the GGUF runner step by step
(`vulkan/fast/reference.cpp`); only the routed experts differ (dequantized NVFP4 instead of the GGUF expert pool). On
a workstation (16 threads, Threadripper 1950X) it takes 13-15 s per position after a 3-minute first position.
Built with `-DSTRATA_GOLDEN_GGML` the same forward runs on a GGUF; on the first 32 positions of a chat prompt the
FP4-Local checkpoint and the UD-Q2_K_XL GGUF of the same model agree on 27 of 32 next-token predictions (mean KL
0.26, the cost of the 2-bit pack), which checks the FP4-Local name and format mapping end to end.

`src/cpu_runner/glm_cpu_runner.cpp` is the runner (section 6). At one rank on x86 with the portable kernel replicas,
exact numerics, the residual after each half-layer of the first five layers (three dense KDA layers, the first MLA
layer, the first KDA MoE layer) matches the golden within **1.3e-7 to 1.0e-6 relative L2** at positions 0 and 3.
Fast numerics (W4A8 experts, int8 trunk) drift 0.6-1.6 % per layer and 4.3 % after the first MoE layer at position 3;
the end-to-end KL gate decides whether that is acceptable.

On A64FX (measured 2026-10-22/23, clang 21 with libc++, MPI wrappers of the same LLVM module):

- one node, one rank, exact, first four layers: the same golden comparison gives 1.4e-7 to 1.0e-6 relative L2.
- **12 nodes, 12 ranks, all 45 layers, exact, teacher-forced over the first 32 positions of the chat prompt: mean
  KL 3.2e-12 against the golden logits, top-1 32 of 32** (interactive job 52646911). Heads 6/6/6/6/5 x 8 per rank,
  experts sliced over 4 ranks, 14.9 GB per rank.
- That first end-to-end build ran at 282 ms per token. Its timers named the fixes that followed: the KDA recurrence
  ran on one thread (120 ms; now split over head x column, bitwise equal), packed weights sat in the memory of the
  loading thread's CMG (88 GB/s per GEMV instead of 247-259 after first-touch placement by the GEMV's own thread
  split), every new token's embedding row was a page fault on the shared file system (50-400 ms stalls on random
  tokens; the BF16 table is now read into memory), and a 48-thread team shared its cores with the job's own daemons
  (47 threads, as the gemm runner uses).
- After those fixes (and fused GEMV regions, K-split of few-tile matrices, a deterministic uTofu all-reduce): **48 ms
  per token exact, 46 ms fast**, 12 nodes, first 32 positions (2026-10-23). The uTofu reduce-scatter + all-gather
  (ported from the gemm runner, rank-ordered sums, identical bits on every rank) takes 32 us per 16 KiB all-reduce
  alone; in the model the 90 all-reduces still cost ~10 ms per token, mostly waiting for the slowest rank (ranks 0-3
  hold 6 of the 64 heads, the others 5; each expert is sliced over 4 of 12 ranks).
- **Fast numerics against exact over 2048 teacher-forced positions: mean KL 0.124, top-1 agreement 87 %, perplexity
  45.26 vs 46.29, next-token top-1 787 vs 780 of 2047.** The perplexity and accuracy gates pass, the KL gate (0.03)
  does not: fast mode as built costs about as much KL as the whole Q2_K/Q3_K pack does against FP8 (0.117), so the
  exact mode is the reference result and lighter fast variants (W4A8 experts with the exact FP8 trunk, finer
  activation blocks) are next.
- Loading every rank's share from the shared file system takes 24 minutes (thousands of small reads per layer at
  ~27 MB/s per node). Each rank now writes a packed image of its share to node-local storage on the first load and
  replays it afterwards: 173.6 s -> 10.2 s for a four-layer one-node load (7.1 GB).

## 6. The runner

The runner is Strata ported to aarch64 (decided 2026-10-18), not a new runner in the gemm repository. Build: clang 21
with libc++ (`-stdlib=libc++`: the system libstdc++ is GCC 8 and has no `<span>`), `-mcpu=a64fx -msve-vector-bits=512`,
OpenMP; Fujitsu MPI and uTofu only behind a C interface. Milestones, each with its gate:

| step | where | gate |
|---|---|---|
| golden reference: fp64 CPU forward of FP4-Local on a workstation, per-layer dumps, routing traces, logits for a fixed corpus | x86 | itself checked against Strata's FP4 V100 path on the same prompts |
| `strata-glm-cpu` at world = 1 (opt-in `STRATA_ENABLE_CPU_RUNNER`), scalar kernels on x86, the kernels above on A64FX | x86, 1 node | per-layer relative L2 <= 1e-5 against golden, top-1 100 %, identical routing |
| 12-rank collectives (uTofu tree allreduce from the gemm repository's `utofu-tests`), bitwise identical results on all ranks | 12 nodes | within 1.5x of the measured 22.6 us (16 KiB fp32) |
| per-allocation staging of each rank's share from the shared file system | 12 nodes | <= 15 min |
| exact mode end to end, teacher-forced | 12 nodes | against golden: top-1 100 %, KL <= 1e-4, identical routing |
| fast mode against exact, >= 2K tokens | 12 nodes | KL <= baseline + 0.03, top-1 >= baseline - 0.01, perplexity +3 % at most |
| plain decode with stage timers against the simulator's terms | 12 nodes | >= 35 tok/s before tuning |
| persistent team + hardware barrier, split-phase barriers, 16 KiB prefetch, sharded sparse front, fused mHC phase, bf16 uTofu, KDA as one phase, MTP depth 2, batched prefill | 12 nodes | each lever against the term it targets; the goal is section 4 |

Partition: tensor-parallel trunk over 12 ranks (KDA and MLA heads 6 x 4 + 5 x 8, o_proj by the same heads, dense FFN
1024 rows per rank, shared expert in 16-row tiles, lm_head 12,912 vocabulary rows per rank), routers/mHC replicated
(their results must be bitwise identical on every rank), each routed expert sliced over 4 ranks (512 gate/up rows and
the matching down columns), the shared expert's partial result folded into the MoE allreduce.

### 6.1 Where the decode step goes (2026-10-23, 12 nodes, exact, first 32 positions)

The runner times every sub-phase (`RUNNER_TIMERS`, `RUNNER_SPREAD` min/max over ranks), times each phase alone after
an evaluation (`STRATA_RUNNER_PROBE_PHASES=1`), and benchmarks its GEMV path on synthetic matrices of the decode
shapes on one node without the model (`glm_cpu_runner --bench-gemv`).

| change | ms per token |
|---|---:|
| uTofu reduce-scatter + all-gather, experts sliced over 4 ranks | 48.0 |
| every expert sliced over all 12 ranks (`STRATA_RUNNER_EXPERT_PARTS=12`): all-reduce wait 10.0 -> 5.5 ms | 53.3 |
| K-split matrices placed by the K-split's own item split | 47.2 |
| top-k routing selection, SwiGLU and KDA prologue over the team | 44.2 |
| weights in fresh pages (below), K split only for matrices under team / 8 tiles | 41.8 |
| one team region per mHC site (and its RMS norm) and per MoE layer | 41.1 |
| BF16 / MXFP8 / block-FP8 kernels with their accumulators in registers (below) | 40.2 |
| mHC projection in its matrix's own K split; no allocation per call in the residual update or the MoE setup | **36.3** |

| KDA recurrence in SVE lanes (16 columns per vector, [head][group][row][16] state, 48 items); indexer scores with 8 heads per vector | 30.7 |
| (next allocation, slower nodes: the same build measured 33-35) LLVM OpenMP `dist` barriers, the KDA layer as one team region | 28.5-29.9 |
| residual update fused into the next mHC site; router sigmoids by their tile's thread | **28.1-28.7** |

Last build measured (2026-10-23 16:17): **30.7 ms per token (32.6 tok/s)**, exact, first 32 positions, KL 2.2e-12
against the fp64 golden (the same value as the 36.5 ms build: both new paths are bitwise equal to the loops they
replace), top-1 32/32. Over 2048 teacher-forced positions the mean is 34.7 ms (31.0 at position 256, 39.8 at 2047),
perplexity 46.29 as before. The two lane-parallel paths keep each column's or head's operation order; to be bitwise
equal they also had to match how clang compiled the old loops (its sums as an unfused multiply plus an ordered add,
the state update fused), checked under qemu. Two attempts did not help and are recorded so they are not repeated:
moving the KDA state and MLA cache pages to the CMGs of the threads that walk them (no change, and see the
`move_pages` note above), and the KDA recurrence in 64-column groups over row-major state with compiler vectorization
(2.5 vs 2.0 ms per token: only 12 work items, accumulators in memory).

Findings:

- **Placement was not happening.** `move_pages` on the packed matrices showed every page on one CMG: the packing
  buffers freed just before were heap memory the loading thread had already touched, and the matrices reused it, so
  first touch by the GEMV threads never placed anything. Matrices now come from an arena of fresh anonymous mappings;
  their pages split evenly over the four CMGs and the GEMV path went from 65-88 to 115-165 GB/s per call (KDA q/k/v
  87 -> 165, o_proj 66 -> 127, a 2048 x 4096 NVFP4 W4A16 matrix 75 -> 118, W4A8 expert slices 58 -> 218).
- **The kernel table's rates hold only for long runs.** The same kernels reach the section 3 rates when one region
  covers many matrices (fused: MXFP8 246, W4A16 182, W4A8 389 GB/s); one GEMV call per region, cold, gets 60-70 % of
  that. Decode regions therefore need to cover more matrices, or the next region's weights need to be in flight
  during the all-reduce. A software prefetch of each thread's whole share at region start made calls 20-50 % slower.
- **Splitting a matrix along K pays only when it has almost no rows** (the 24 x 16384 mHC projection: 25 vs 116 us). For
  the 12-tile expert slices it was 1.9-2.3x slower than whole tiles over the 16 slices of a region.
- Team forks cost ~5 us and barriers ~4 us at 47 threads; the probe showed small regions at 20-60 us each, so most of
  the cost is moving data between CMGs and serial steps between regions, not the synchronization itself. One mHC site
  (norm, projection, coefficients, combination, the following RMS norm) was ~100 us; it now runs as one region, and the
  MoE layer (router, top-k, gate/up, SwiGLU, down, weighted sum) too.
- **The BF16 and FP8 kernels kept their accumulators in memory.** `q[n][(k % 32) / 8]` with the index computed at run
  time made clang keep the four chains on the stack, a store and a load around every four FMAs; the per-tile time of a
  16 x 4096 BF16 tile was 29 us (about one multiply-add per cycle). With the chain index unrolled the loop has no
  vector stores, the results are bitwise unchanged (qemu parity test), and BF16 runs 2.3x faster (2048 x 4096: 190 ->
  438 GB/s; a router-sized call 30 -> 11 us). Block FP8 still spills a few registers per K32 step.
- The all-reduce costs 32 us alone; in the model it averages 60-100 us because the ranks with 6 of the 64 heads finish
  their KDA phase later than the ranks with 5, and because the replicated mHC step varies between nodes (1.1-3.7 ms per
  token for the same work: thread wake-up skew, not a core shared with system processes; 46 threads measured the same).
- **W4A8 experts alone (exact FP8 trunk) also fail the fast gate:** over 2048 positions KL 0.116 against exact, top-1
  agreement 88 %, perplexity +3.2 %. The int8 activations of the experts (one scale per 64 values) carry nearly all of
  fast mode's error, not the trunk requantization; the next fast variant puts one activation scale on each K16 group,
  which the W4A8 kernel already applies a weight scale to.

- **Fast mode's error is in the down projection.** Local drift after the first MoE layer (relative L2 against the
  golden, five layers): W4A8 on every expert projection 8.6e-3, with one activation scale per K16 group 4.8e-3 (the
  W4A8 kernel takes `group 16` activations: one extra vector multiply per group; bitwise against its replica under
  qemu); W4A8 on gate/up only (`STRATA_RUNNER_W4A8_ONLY=gate_up`, down stays W4A16) 2.5e-3, with K16 scales 2.1e-3.
  The down projection's inputs are SwiGLU outputs. Gate/up carry two thirds of the expert bytes, so `--numerics
  w4a8g16` with gate/up only was the next candidate. **It failed the 2048-token gate too:** KL 0.087 against exact,
  top-1 agreement 90 %, perplexity +1.9 %, though 28.1 ms per token over 2K (gate/up 1.3 instead of 2.6 ms). The KL
  grows with the position (0.037 over the first 256 tokens, 0.067 to 1024, 0.114 to 2048; median 0.015, the top 1 %
  of tokens carry 38 %): the activation error accumulates in the KDA state and the MLA cache, so the 5-layer drift
  underpredicts it, and any approximate activation path will drift with the context in this hybrid model. Fast mode
  is parked; exact numerics are the result.
- **The OpenMP runtime's barrier algorithm is worth 4-5 ms per token.** With LLVM's `dist` barrier
  (`KMP_FORKJOIN_BARRIER_PATTERN`, `KMP_PLAIN_BARRIER_PATTERN`, `KMP_REDUCTION_BARRIER_PATTERN` = `dist,dist`) a fork
  of 47 threads costs 4.3 us instead of 5.3 and a barrier 6.7 instead of 9.0, and the step went from 33-35 to
  28.5-29.9 ms on the same nodes; hypercube or tree barriers with branching 4 were slower (36-39 ms). The runner sets
  `dist` unless the environment chooses. Touching the next region's weights from the idle threads during each
  all-reduce was 1.7 ms slower with the default barriers and neutral with `dist`; it is opt-in
  (`STRATA_RUNNER_COMM_TOUCH=1`).
- **Attention scores at long context were serial chains too:** each (head, selected position) score was one chain of
  512 dependent double FMAs, 3.3-4.3 ms per token at 2K context. Eight pairs per work item now run as eight
  independent chains, each in its own order; a product of two floats is exact in double, so the sums do not depend on
  whether the multiply-add is fused.
- **8K context** (8191 teacher-forced positions of the 8K eval file): the mean went 31.9 -> 30.1 ms per token (34.7
  -> 31.4 ms at position 8190) after the indexer's pool selection became `nth_element` plus one pass (a serial stable
  sort of up to 2048 pool scores cost ~200 us per MLA layer; the selected set is the same, perplexity and next-token
  accuracy unchanged) and the attention score items were ordered so the heads of the same positions share a CMG's L2.
  Past 2K positions the step grows by ~0.5 ms per 1K (the indexer); attention itself stops growing at the 2048
  selected positions.
- **Multi-token passes** (`--multi T`, T = 1..4 consecutive positions per pass, the base for MTP verification and
  prefill): every GEMV reads its weights once for all T tokens; the mHC sites, the KDA convolution and recurrence,
  MLA attention and routing run per token in position order with the one-token code, so a pass is bitwise equal to T
  single steps (same golden KL fingerprint on 12 ranks). Measured per pass: T = 1, 2, 3, 4 -> 33, 52, 73, 87 ms
  (21.7 ms per token at T = 4). Per token at T = 4 the GEMV phases halve against a single step, but the routed
  experts do not (5.1 vs 5.0 ms: four tokens touch about four times the distinct experts), nor do the per-token mHC
  (3.3), KDA (1.9) and MLA attention (1.2) steps. MTP depth 2 verifies three positions per pass and accepts about
  2.2 tokens; it pays once a three-token pass costs about 46 ms instead of 73.
- **MTP drafting** (`--generate N --mtp K`, greedy): the checkpoint's draft block (layer 45: MLA with its own
  indexer and caches, a 288-expert MoE, MXFP8 attention weights, a plain residual instead of mHC) follows Strata's GPU
  decoder: cell q is `eh_proj([enorm(embedding of token q) | hnorm(h(q - 1))])`, `+= MLA(attn_norm)`,
  `+= MoE(ffn_norm)`, `shared_head_norm`, then the main `lm_head`; h is the main model's output-normed hidden (zero
  at q = 0) and a depth-2 step pairs the next draft with the previous step's post-norm hidden. A round drafts K
  tokens, verifies `[t, d1 .. dK]` in one multi-token pass, keeps the longest prefix whose drafts equal the main
  model's argmax plus that argmax, rolls the main model back to the snapshot after the last kept position (KDA
  states, conv histories, index key tails), restores the draft block to its first cell of the round (paired with
  the exact hidden), and recomputes the kept drafts' cells from the verified hiddens. The draft block's `kv_b` is
  MXFP8 (the main layers' is BF16); it is converted to BF16 at load, exactly.
  Measured (12 nodes, exact, 64-token prompt, 128 greedy tokens, every run's tokens identical to plain decoding):

  | | ms per token | accepted drafts | tokens per round | verify pass |
  |---|---:|---:|---:|---:|
  | plain | 26.1 | | 1 | |
  | MTP depth 1 | **24.4** | 52 / 77 | 1.66 | 38.4 ms (2 positions) |
  | MTP depth 2 | 26.8 | 65 / 128 | 2.00 | 50.1 ms (3 positions) |

  A second prompt (the first 128 tokens of the 2K eval file) gave the same: plain 26.4, depth 1 24.6 ms per token,
  52 of 76 drafts accepted.

  A draft step costs 1.2 ms, the rollback 0.4 ms, the draft block's resync 0.5 ms. Greedy tokens come from each
  rank's vocabulary slice (`MPI_MAXLOC`, the lower index on ties) instead of gathering the logits. What made the
  verification pass cheap enough: the MoE and KDA layers as one team region for all its positions, a persistent
  buffer pool for the per-pass expert slices and persistent buffers elsewhere in the pass (allocations with zero
  fill on one thread cost milliseconds per pass), and a rollback without copies: the KDA states live in a pool of
  placed buffers per layer, the verification pass writes each position's new state into its own buffer, and rolling
  back selects the buffer (the conv histories and index key tails, a few KB, are copied).
  The first draft is accepted 68 % of the time and the second about half as often, and the routed experts do not
  amortize over a pass (each position brings its own), so depth 1 is the better setting for now.
- **2026-10-24: verification and fixed costs** (12 nodes, exact, same prompt and tokens, every run's generated
  tokens identical to plain decoding):

  | change | plain | MTP depth 1 | verify pass (2 positions) |
  |---|---:|---:|---:|
  | before (table above, same nodes later in the day) | 24.3-24.5 | 23.0 | 36.1 |
  | own-line flag barrier for the team's sync points (1.6 vs 7.5 us per barrier) | | | |
  | mHC partial sums per thread, then per CMG, then on thread 0 (was: 24 row threads reading all 47 threads' lines, 2-20 us per site depending on where the lines were) | 24.4 | 22.95 | 36.1 |
  | MLA front sharded: q_a, kv_a, the indexer's projections and wq_b by rows over the ranks, one all-reduce of zero-padded rows per stage (exact: one value plus zeros) | 23.77 | 22.42 | 36.1 |
  | multi-token KDA: the convolution, q/k sums, normalization, the recurrence (each column group's positions in order) and the output gate as one phase each for all positions; per-head sums over the team | 23.61 | 21.45 | 33.6 |
  | persistent team: one OpenMP region per decode, every team region a flag dispatch (fork + join 4.3 us -> one barrier) | 22.16 | 20.40 | 32.0 |
  | default build: router whole again, attention maximum per thread and totals per (token, head) over the team, greedy picks by a uTofu all-reduce | 21.62 | 19.92 | 31.3 |
  | the indexer's selection for all positions of a pass at once; mHC projections copied chunk-major per CMG (a thread's blocks one local run instead of 8 places in 64 KB pages shared by two CMGs, 360 MiB per rank); mHC partials in per-thread buffers (256 tokens from here) | 21.28 | 19.03 | 30.8 |
  | the MoE layer's buffers persistent, its int8-activation check cached per layer (the check walked all 288 slice descriptors of the layer, ~3.6 MB per token on one thread) | 20.76 | **18.95** | |
  | greedy pick's scan of the rank's 12,912 vocabulary logits over the team (0.11 -> 0.01 ms per token; the first maximum as before) | 20.8 | 18.96 | |

  **Final build (2026-10-24 07:50, 256 tokens):**

  | prompt | plain | depth 1 | adaptive (`--mtp 3`) |
  |---|---:|---:|---:|
  | a: start of the eval document | 20.50 (48.8 tok/s) | **18.82 (53.1 tok/s)** | 19.71 |
  | b: eval document from token 1024 (code) | 20.66 | 16.29 | **13.52 (74.0 tok/s)** |
  | c: 8K eval file from token 5000 | 20.89 | 17.14 | **14.77 (67.7 tok/s)** |

  (09:16 build: + the parallel greedy-pick scan and the routing step's register threshold; run-to-run noise is
  ~0.2 ms. Re-measured 11:35 on a fresh allocation with the int8 draft head: a 20.58 / 18.82 / 19.46, b 20.50 /
  16.20 / 13.50, c 20.54 / 17.03 / 14.76, 8K mean 23.05, golden unchanged. 11:57, rollback copies over the team
  (0.5 -> 0.12 ms per round): b adaptive 13.47 ms, 74.2 tok/s, repeats within 0.03 ms. The KDA recurrence in 8-lane
  halves on 6-head ranks (`STRATA_RUNNER_KDA_HALVES=1`, bitwise) is 0.09 ms slower and stays off.) Prefill is the multi-token pass of at most 4 positions: a 64-token prompt takes 1.4 s (~45 tok/s); the
  section 4 prefill forecast needs GEMM kernels for long position blocks, which this runner does not have yet.

  Plain decoding over 8191 teacher-forced positions: mean **23.12 ms** (43.3 tok/s), perplexity 57.1163 (bitwise the
  same as before the attention and indexer changes). Golden: KL 1.74e-12, top-1 32/32. Against the section 4
  forecast (77 tok/s with MTP), the code prompt comes within 4 %; the document start, where the draft block's first
  draft is accepted 68 % of the time, reaches 53 tok/s. At the row before the final ones: depth 2 21.7 (65 of 128
  drafts, verify 40 ms); golden KL
  1.74e-12, top-1 32/32 (the same value with and without the persistent team and the router sharding). Plain
  decoding over 8191 teacher-forced positions: mean **24.29 ms** (was 29.26 before these changes, 27.16 before the
  persistent team and the attention changes), perplexity 57.1163 and next-token accuracy identical to the build
  without the attention changes (bitwise: the same selections). What did not help, so it is not repeated: the router sharded the same way (its all-reduce
  costs 0.87 ms per token and saves 0.15; opt-in `STRATA_RUNNER_ROUTER_SHARD=1`); gating drafts on the draft
  block's own probability (`STRATA_RUNNER_MTP_TAU1/2`: rejects few drafts, and the probability needs an extra
  all-gather); the libomp fork pattern (fork + join is 4.2-4.4 us at 47 threads for dist, hyper, tree and linear);
  44 threads with the last core of each CMG left free (the last thread of a CMG finishes the mHC site's first phase
  last at 47 threads too, so it is the team position, not the core); the next phase's weights prefetched in the
  current region (20.95 vs 20.76 ms) or touched during the all-reduce (22.3).
  The reduce-scatter's completion polling once per call instead of twice is 17.3 instead of 19.0 us (default).
- **Acceptance depends on the text, so the best depth does too** (default build, 64-token prompts, 128 greedy
  tokens; tokens identical at every depth):

  | prompt | plain | depth 1 | depth 2 | depth 3 |
  |---|---:|---:|---:|---:|
  | a: start of the eval document (the rows above) | 21.6 | **19.9** (68 %, 1.66 per round) | 21.7 (2.00) | 23.8 (2.25) |
  | b: eval document from token 1024 (code) | 21.6 | 17.0 (97 %, 1.97) | 15.1 (2.91) | **14.8 (3.66), 67 tok/s** |
  | c: 8K eval file from token 5000 | 21.5 | 17.9 (86 %, 1.86) | **16.7** (2.61) | 17.6 (3.05) |

  A round costs about 22 ms plus 10 ms per draft position (verify 31 / 40 / 49 ms for 2 / 3 / 4 positions, a draft
  step 1 ms), so one more draft pays when it is accepted more than ~45 % of the time given the ones before.
  **Adaptive depth** (`--mtp 3` with `STRATA_RUNNER_MTP_ADAPT=1`): each round takes the depth with the most expected
  tokens per ms, from a moving acceptance rate per draft position (given the drafts before it accepted) and that
  cost model (`STRATA_RUNNER_MTP_BASE` 22, `_STEP` 10 ms); every 16th round goes one deeper so that position keeps
  being judged. Over 256 tokens per prompt (ms per token):

  | prompt | depth 1 | depth 2 | depth 3 | adaptive | rounds at depth 1 / 2 / 3 |
  |---|---:|---:|---:|---:|---|
  | a | **19.7** | 20.7 | 22.4 | 20.4 (49 tok/s) | 123 / 19 / 5 |
  | b | 16.9 | 15.0 | **14.3** | 14.3 (70 tok/s) | 0 / 7 / 62 |
  | c | 17.8 | 16.2 | 16.3 | **15.7** (64 tok/s) | 0 / 9 / 67 |
  | mean | 18.1 | 17.3 | 17.7 | **16.8** | |
- **Fewer nodes are not faster.** With 8 ranks (experts in 8 parts, 22.5 GB per rank) plain decoding took 27.9 ms
  and depth 1 26.1, against 24.3 / 23.0 on 12 at the time: the bytes per rank grow by half while the all-reduce
  wait shrinks by much less. 6 ranks do not fit: a 6-rank load (experts in 6 parts) was killed for memory at layer
  40 of 45 (15.9 GB per rank at layer 25; a node has 28 GB free).
- **The 6 / 12 node question measured on the same layers** (first 30 layers, exact eval, 64 positions): 6 ranks 16.33
  ms per token (all-reduce 2.24, experts 3.49, KDA projections 3.34; 18.7 GB per rank), 12 ranks 13.42 (2.40, 2.56,
  2.15). Halving the nodes saves 0.16 ms of communication and doubles every rank's bytes, so 12 nodes are 22 %
  faster before 6 would even need to fit. The all-reduce itself costs 17.3 / 22.5 / 30.9 us for 1 / 2 / 4 hidden
  vectors; in a 4-position verification pass it averages ~61 us, half of it waiting for the slower nodes.
- **The draft block's head in int8** (default with MTP; `STRATA_RUNNER_DRAFT_HEAD=exact` keeps MXFP8): the MXFP8
  rows decoded and requantized per (row, 32) for the drafts only (every kept token stays the main model's exact
  pick). The same drafts on all three prompts; a draft step 1.0 -> 0.78 ms; adaptive depth b 13.41 ms (74.6
  tok/s), c 14.66, a 19.46.
- **Clock:** plain boost (2.2 GHz with all pipelines) is not available to this account; boost-eco (2.2 GHz, eco
  state 2) makes the issue-bound kernels slower (one token: W4A16 45.1 vs 36.5 us, MXFP8 52.2 vs 51.6; four tokens:
  MXFP8 109.9 vs 92.9), so the normal 2.0 GHz mode stays.
- **Relaxed numerics with MTP** (2026-10-24, 12 nodes, 256 tokens; KL / top-1 against the exact build over the same
  2048 teacher-forced positions as before; MTP tokens equal the same mode's plain tokens in every run). Fast mode's
  int8 trunk also needed a fix: the draft block's indexer projections are FP8 (BF16 in the main layers), so fast mode
  requantizes them and they must take the int8 activations (every fast-mode MTP run crashed before).

  | numerics | KL vs exact | top-1 vs exact | ppl (exact 46.29) | prompt a: plain / depth 1 | b: adaptive | c: adaptive |
  |---|---:|---:|---:|---:|---:|---:|
  | exact | 0 | 100 % | 46.29 | 20.6 / **18.9** (53 tok/s) | 13.5 (74) | 14.7 (68) |
  | W4A8 gate/up, K16 scales (`w4a8g16`, `STRATA_RUNNER_W4A8_ONLY=gate_up`) | 0.073 | 90.5 % | 47.22 | 20.0 / 19.0 | 13.3 (75) | 13.6 (74) |
  | W4A8 experts, exact trunk (`w4a8`) | 0.111 | 88.5 % | 47.50 | 19.2 / **17.2 (58 tok/s)** | **12.4 (81)** | **12.8 (78)** |
  | W4A8 experts + int8 trunk (`fast`) | 0.132 | 87.9 % | 45.49 | 17.9 / 17.9 | **11.8 (85)** | 13.5 (74) |

  Next-token accuracy on the text itself is the same in every mode (778-797 of 2047 against 780 exact). With a gate
  at KL <= 0.12, `w4a8` passes the 77 tok/s forecast on both non-opening prompts; the int8 trunk speeds up plain
  decoding but lowers the draft block's acceptance (prompt a 97 of 160 against 109 of 148), so it is not faster with
  MTP except on the code prompt.
  Over longer contexts the KL against exact keeps growing (the activation error accumulates in the KDA states and
  MLA caches): over the first 4096 positions `w4a8` 0.177 and gate/up-only 0.111 (from 0.111 / 0.073 over 2048),
  top-1 agreement 87.5 / 90.1 %. Predicting the text itself does not get worse: at 4K perplexity 44.53 / 43.92 and
  next-token top-1 1659 / 1669 against exact's 44.09 / 1657; at 8K perplexity 55.86 / 57.34 and 3507 / 3509 against
  57.12 / 3480 (plain decoding 21.9 / 22.5 ms per token at 8K against 23.1 exact). A KL gate therefore has to name
  its context length.
- **Depth 4 does not pay:** a 5-position verification pass (the kernels' 4-token accumulators run a second time
  over each cached weight tile for the fifth position; bitwise, golden KL unchanged) costs 67 ms against 49 for 4
  positions, so on the code prompt fixed depth 4 takes 14.1 ms per token (201 of 228 drafts accepted) and adaptive
  depth up to 4 13.9, against 13.5 with depth up to 3. Not kept.
- Keeping the draft block's cells of accepted drafts from drafting instead of recomputing them from the verified
  hiddens (`STRATA_RUNNER_MTP_LAZY=1`) lowers acceptance (code prompt 180 of 196 instead of 188 of 200) and is slower
  (13.8 vs 13.4 ms).
- **Spreading the split remainders** (`STRATA_RUNNER_REMAINDER_SPREAD=1`: the extra shared-expert unit, MLA front
  rows, vocabulary tiles and draft eh_proj rows on ranks 4-11 instead of 0-3, next to the 4 extra heads) is neutral
  (20.66 / 18.95 ms against 20.70 / 18.96; golden KL 1.99e-12): the critical path is the slowest nodes, not the extra
  units.
- **Where the remaining step goes** (plain, ms per token, rank 0): routed experts 4.0 (the exact W4A16 kernel is
  SIMD-issue bound: 165 GB/s for one token, 1.54x the time for two), mHC sites 3.4, KDA projections 3.4, all-reduce
  3.1 (about half of it waiting for slower ranks: 8 of the 12 nodes are 5-8 % slower in every compute phase; their
  weight pages are spread over the CMGs as evenly as on the fast nodes), router and shared expert 2.3, MLA 3.3, KDA
  recurrence and gates 0.8. A verify pass costs 1.43x a step: its two positions bring their own experts (8.6 vs
  4.1 ms) and per-position mHC, recurrence and attention.
- Never migrate pages with `move_pages` in these runs: moving the KDA state's heap pages to other CMGs made Fujitsu
  MPI fail in the next large all-to-all ("Tofu MRQ error", twice, different ranks). The library registers heap pages
  with the network interface by physical address.

Remaining step at 36.3 ms (rank max, ms per token): mHC sites 6.5 and the residual update 1.4, all-reduce 5.5, KDA 7.5
(projections 2.7, recurrence 2.0, o_proj and gates), MoE 8.4 (gate/up 3.0, down 2.1, router + shared 1.9, top-k and
setup 1.1), MLA 4.1, head 0.7, dense FFN 0.3. Against the section 4 exact forecast (58 tok/s, 17 ms) the gaps are the
mHC and KDA phases (the forecast has 3.8 ms of mHC), the all-reduce wait, and the exact W4A16 expert kernel, which is
compute-bound at ~115-180 GB/s.

### 6.2 Prefill (2026-10-24, 12 nodes, `--numerics w4a8`)

`w4a8` is the decode mode that meets the 77 tok/s target (section 6.1), so prefill was tuned in that mode. Test:
greedy generation after a 2048-token prompt (`eval-2048`), MTP 3 adaptive; the 64 generated tokens are identical in
every run (one md5 for every chunk size and every change below), and residual dumps of a 256-token eval at
`--multi 1` and `--multi 128` are identical on 12 ranks (270 of 270 files). The allocation had 12 of the slower
nodes (5-8 % behind the others in every phase).

| change (cumulative) | prefill tok/s (2048 tokens) |
|---|---:|
| one `step()` per prompt token (before) | 47.6 |
| multi-token passes of 4 / 16 / 32 / 64 positions (`STRATA_RUNNER_PREFILL_CHUNK`) | 83.1 / 96.7 / 101.1 / 103.8 |
| mHC sites batched 16 tokens per region; passes of 128 | 109.7 |
| exact trunk GEMMs for long passes: weights decoded once per tile, FMA chains interleaved | 122.3 |
| activations of a pass quantized in one region, routing per token over the team | 130.0 |
| expert plan over the team, pre-sized buffers | 133.4 |
| token-parallel mHC sites | 142.8 |
| all-reduce in 64K-float pieces (`STRATA_RUNNER_AR_MAX=65536`) | 148.3 |
| MLA scores in 4 x 4 (token, position) blocks, few-tile jobs split by tokens | 149.4 |
| expert outputs of a long pass in flat buffers that persist (no per-use allocation) | **155.8** (149.0 at the default 16K pieces) |

Other prompt lengths (final binary, 64K pieces): 64 tokens 117 tok/s (0.55 s), 128 tokens 134 tok/s (0.96 s, was
50.6 one step per token; the first pass still pays for first-touch allocations), 8000 tokens 133 tok/s (60.0 s; MLA attention 1.96 ms per prompt token there).

What each step does, and what did not work:

- **Long passes.** `step_multi` takes up to 256 positions (it took 4). GEMV jobs carry any number of token
  activations. A tile is run for 4 tokens at a time while it is in the core's cache. All-reduces longer than the
  uTofu buffer go in pieces (the per-element rank order of the sums is unchanged). Only the last position goes
  through the head. Passes of 256 are not faster than 128 (147.8).
- **Exact trunk GEMMs** (MXFP8 / block FP8 / BF16, `gf4_tile_many`). The per-token kernels decode every weight vector
  again for each 4-token group. They also give each accumulator four dependent FMLAs in a row, which fills A64FX's
  FP queues: one core with hot data reaches 7.4 of 32 MACs per cycle (0.5-0.56 TMAC/s per node at any token count,
  `STRATA_BENCH_NT` with `--bench-gemv`). The long-pass kernel decodes a 256-column slab of the tile once into F32
  scratch. Consecutive FMLAs go to 16 different (token, chain) accumulators, keeping each chain's own order, and
  the activations come from replicating loads: 12.3 MACs per cycle, bitwise equal to the per-token kernels (qemu
  test over all three layouts). Decode-once alone, without the reordering, gained nothing. Indexed FMLAs on
  quad-loaded activations spilled registers and were slower (6.3).
- **W4A8 experts.** The same interleaving for the int8 kernel (one K16 group decoded at a time, split SDOT chains)
  is bitwise but slower (15.6 vs 17.7 MACs per cycle on one core). Not kept. In a 128-token pass each expert slice
  sees ~3.6 tokens, so the experts stay at small-batch rates (1.3 ms per prompt token).
- **mHC** (`site_tokens`): each thread takes whole tokens and replays the team's split of `site()` for them, virtual
  thread by virtual thread (its units in order, then the per-CMG sums, then the CMG sums), so every sum keeps
  `site()`'s order. One region without barriers: 0.92 → 0.53 ms per prompt token.
- **All-reduce size.** 1 MB all-reduces take 501 us in 16K-float pieces and 337 us with a 256K buffer. A larger
  buffer makes 16K messages ~1.5 us slower (31.6 → 33.1 us), so the buffer stays at 16K floats unless
  `STRATA_RUNNER_AR_MAX` is set (+3-4 % prefill).
- **MLA attention.** When every position reads all earlier positions (contexts up to the 2048-position selection),
  scores are computed in blocks of 4 tokens x 4 positions per head: 0.39 → 0.32 ms per prompt token. A double copy
  of the latent cache was slower (twice the bytes per position) and was reverted.
- **Few-tile jobs** (router 18 tiles, shared expert ~11, per-head absorbs) are split into (tile, token block) items
  in long passes (`STRATA_RUNNER_TOKEN_SPLIT`). The fused MoE region then needs a barrier before each thread reads
  "its own" router rows: without it the generated tokens changed from run to run.
- **Decode is unchanged:** prompt b with MTP 3 adaptive takes 12.6 ms per token on these nodes, with the same tokens
  as before (md5 80357770; 12.4 on the faster nodes). The expert-plan buffers of the prefill are released before
  decoding; kept, they made decode 1.8 % slower.

- **Allocation.** Per-use output vectors grown on demand made the first pass slow (expert plan 2.55 ms per token on a
  128-token prompt, against 0.18 later: fresh mmaps and their page faults serialize in the kernel). Long passes now
  write gate/up/down rows into flat buffers that grow once and stay (plan 0.07 ms per prompt token).

Per prompt token at 156 tok/s (6.4 ms): routed experts 1.22 (of which plan 0.07), MLA attention 0.98 (scores 0.32,
indexer 0.26, average 0.21), KDA projections 0.92, all-reduce 0.78, router and shared expert 0.57, mHC 0.51, MLA
projections 0.46, KDA recurrence and gates 0.35 (scalar `expf` in the gates and the convolution's SiLU), draft
block over the prompt 0.24. Beyond 2048 positions every token selects its own positions, so the blocked scores do
not apply and attention grows with the context.

#### 6.2.1 Toward 400 tok/s (2026-10-25, 12 's' nodes)

Same test as above: `eval-2048` / `eval-8192` prompts, MTP 3 adaptive, 64 generated tokens. The md5 is unchanged by
every exact change below (2K `da2742e4`, 8K `4e9a5b36`, decode after prefill on prompt b `a3454d26`). The
baseline on these nodes is 169.6 tok/s at 2K (fast exp, section 6.3) and 145.1 at 8K.

| change | 2K tok/s | 8K tok/s |
|---|---:|---:|
| all-reduce copies over the team, one activation array per pass, attention in 128-token sub-blocks, blocked latent average (4 tokens x 32 latents) | 173.0 (c128) / 175.1 (c256) | 153.6 (c512) |
| comm thread + pipelined passes (`STRATA_RUNNER_PREFILL_PIPE=1`, opt-in) | 186.5 (c256) | 160.5 (c512) |
| union walk off again (see below) | | 167.4 (c512) |

- **Pipelined passes.** A thread pinned to the 48th core (`STRATA_RUNNER_COMM_CORE`) runs the uTofu all-reduces the
  main thread posts, in posting order. The same order on every rank keeps the collectives matched. A pass of 64+
  tokens runs as two halves, per layer: attention A, attention B, FFN A, FFN B. Each half's output all-reduce
  overlaps the other half's compute. The per-token arithmetic is unchanged, so the result is bitwise: dumps at 5 and 8
  layers are identical against single steps. The visible comm per prompt token drops from 1.04 ms (wire) to
  ~0.6 ms (0.24 waits + 0.36 in the MLA front, whose all-reduces still sit on the critical path). The thread
  stops after the prefill, so decode's small all-reduces go direct again (decode b unchanged, 11.5 ms).
- **Second all-reduce region** in `comm_utofu.c` (256K floats, `STRATA_RUNNER_AR_BIG`, default 262144) next to
  the 16K decode region: 1 MB in 334 us (501 in 16K pieces). Decode sizes are unchanged (4096 floats 17.4 us,
  16K 31.3 us).
- **Q8 prefill trunk** (`STRATA_RUNNER_PREFILL_TRUNK=q8`, +0.7 GiB per rank): the 8K eval gives ppl 59.30 and
  next-token 3479. That is outside the noise band (56.44-58.06 / 3489-3506), so it fails the gate and stays opt-in.
  It was only +5 % anyway (2K 182.0, 8K 155.0).
- **Union walk beyond 2K positions** (`STRATA_RUNNER_UNION_BLOCK=1`, opt-in): groups of 4 tokens walk the sorted
  union of their selections, reading each cache row once (scores 4 tokens x 4 rows; average 4 tokens x 3 heads
  x 16 latents). Bitwise (4096-token dumps on 12 ranks, 180 of 180), but slower: 8K 155.5 against 167.4 tok/s,
  `mla.avg` 0.99 against 0.58 ms. The unions of 4 neighbouring tokens are far larger than one selection.
- **No gain:** splitting the MLA front from the rest of attention so both halves' fronts run first (2K 185.6). The
  `MANY_KS` W4A8 GEMM microkernels (none beat the 4-token tile, removed).

Per prompt token at 2K c256, pipelined (5.4 ms): routed experts 1.08, KDA projections 0.80 (input projections 0.54),
MLA attention 0.76, MLA projections 0.66, mHC 0.48, router + shared expert 0.45, comm waits 0.60, draft block
0.25, KDA small phases 0.24.

### 6.3 Vector math and decode follow-ups (2026-10-24)

- Bitwise, default:
  - **W4A8 tiles use vector SDOTs on 4-byte activation broadcasts** instead of the indexed form, which A64FX issues
    as 2 uops: 5-8 % per core.
  - **The Sinkhorn iterations of the mHC coefficients run on one 16-lane vector.** Sums are formed with `svtbl` in
    the reference order, followed by vector divides; 20,000 random cases match the reference bitwise.
  - **Verification passes plan their experts over the team.**

  Decode with MTP 3 adaptive, on the slower nodes: code prompt 12.23 ms per token (12.74 with all three off),
  mid-document prompt 12.72 (13.18), same tokens. The code prompt is 12.4 ms on the faster nodes before these
  changes.
- **Approximate** (`STRATA_RUNNER_FAST_EXP`, the default since it passed the noise-band gate below; sites by
  `STRATA_RUNNER_FAST_EXP_MASK`):
  - FEXPA-based exp, sigmoid, SiLU and SwiGLU, within 1-6 ulp of libm; double exp within 1 ulp.
  - Sigmoid at 1.1 ns per value against 20 ns for libm.
  - Used for the KDA conv, decay, exp and gates, SwiGLU, the router and the MLA softmax: prefill 164.7 tok/s at 2K
    (+6 %), decode 11.97 ms per token (-2 %).
- **The KL gate cannot work as defined.** Against the bitwise w4a8 logits (a control run gives KL 0), changing
  only the router's sigmoid by a few ulp already gives KL 0.078 over 256 tokens (0.003 at the third token), only
  KDA 0.096, only SwiGLU 0.074, only the MLA softmax 0.0012. Perplexity moves both ways (104.8-108.4 around
  106.85). Any few-ulp perturbation moves this model's output distribution about as far as w4a8 moves it from exact
  (0.111), so a KL limit against one bitwise run cannot be passed by any non-bitwise change. Perplexity and
  next-token accuracy over long evals, against the spread of several perturbed runs, are a usable gate.

- **Noise band and the fast-exp default** (8K eval, w4a8; perplexity / next-token top-1 of 8190):

  | run | ppl | next top-1 |
  |---|---:|---:|
  | bitwise (libm) | 55.86 | 3507 |
  | fast exp in KDA only | 56.74 | 3494 |
  | fast exp in SwiGLU only | 58.06 | 3489 |
  | fast exp in the router only | 56.44 | 3506 |
  | fast exp everywhere | **56.03** | **3521** |

  Fast exp everywhere is inside the band of the single-site perturbations on both metrics, so it is the default
  (`STRATA_RUNNER_FAST_EXP=0` gives libm and the earlier bitwise tokens).
- **More bitwise defaults:**
  - The MXFP8 / block-FP8 / BF16 kernels for 1-4 tokens use vector FMLAs on replicated activations, with the
    chains interleaved (one core, MXFP8: 3.6 / 6.2 / 7.3 / 7.0 MACs per cycle at 1-4 tokens, against 3.2 / 4.3 /
    5.5 / 6.3).
  - The shared expert's tiles are rotated away from the router's threads at load.
  - The indexer skips its scores while every pool is selected (contexts up to 2048).
- **Decode with every default** (MTP 3 adaptive, 64-token prompts, 256 tokens): code prompt **11.69 ms (85.5
  tok/s)**, mid-document 12.37 (80.9), document start 15.40. On the same nodes before the float kernels and fast
  exp: 12.32 / 12.77.

### 6.4 Long context and several contexts (2026-10-24/25)

- **Memory per rank:** weights 15.0 GiB (+0.4 for the mHC copies). The context buffers are 25.5 KiB per position
  with a replicated fp32 latent cache (12 MLA layers including the draft block), so 256K positions take 6.4 GiB
  and fit. 512K (12.8 GiB) and 1M (25.5 GiB) do not. The runner prints the budget at load and warns under 2 GiB
  available.
- **Indexer at long context.** Beyond `STRATA_RUNNER_INDEX_SHARD_MIN` pools (default 16384, i.e. 64K positions),
  the indexer works like this:
  - each rank scores the pools of its 32-pool blocks (round-robin);
  - each rank keeps its best 512 by (score, then lower pool);
  - an allgather (4 KB per token per rank) merges them in the same order.

  Every pool of the global selection is among its owner's best 512, so the selection is the unsharded one. On 12
  ranks, sharding from 513 pools gives a 4096-token eval with residual dumps identical to the unsharded run (180
  of 180) and the same perplexity.

  The candidates travel over the runner's uTofu all-reduce, in zero-padded per-rank slots, carried as 16-bit halves
  in floats so that they survive the sums exactly. `MPI_Allgather` took 1-3 ms for 4-512 KB, and called between the
  runner's uTofu all-reduces it crashed in the Tofu transport (a segfault, also with never-freed buffers); in
  isolation it worked. Sharding starts at 65536 pools (256K positions): below that, the replicated scores cost less
  than the exchange (0.87 ms per token at a 4K context).
- **Hybrid latent cache** for 512K-1M (`STRATA_RUNNER_CACHE_HOT=W`, default 65536):
  - Every rank keeps a ring of the last W rows; older rows live only on the rank owning their 256-position block.
  - Passes that reach beyond the ring gather the absorbed queries of all heads.
  - Each rank scores its own cold rows for all heads under the global per-head maximum (one max all-reduce) and
    sends exp sums and weighted latent sums to the heads' ranks (one all-to-all). Those ranks add them to their
    hot part in rank order.
  - The absorbed queries of all heads travel in one all-reduce of a zero-padded all-heads buffer. The cold partial
    sums travel as fp32 in one all-reduce, summed in rank order; the per-head maximum is one small MPI all-reduce.
  - The result is deterministic but not bitwise: relative L2 up to 3.5e-7 against the replicated cache locally.
  - Gate (8K eval with a 2048-position ring, so most positions are cold): perplexity 56.42, next-token 3521 → 3519,
    against the default run's 56.03 / 3521. That is inside the band of section 6.3.
  - Speed: with SVE dot products and 16-row items (each row read once for all heads), an 8K eval token with a
    2048-position ring takes 9.7 ms against 7.0 replicated (17.4 with scalar dots). The text metrics are still inside
    the band (57.24 / 3498).
  - The hybrid cache only starts from `STRATA_RUNNER_CACHE_HYBRID_MIN` positions of capacity (default 300000).
    Below that the replicated cache fits (about 7 GiB at 290K) and is exact. A 64K run with the hybrid cache on
    decoded at 159 ms per token, every position past the ring.
  - 1M positions with a 64K ring: 1.6 + 1.9 GiB per rank.
- **Several contexts** (`--contexts C`):
  - Each context's per-sequence state (KDA states, rollback buffers and conv histories, MLA tails, pools and
    caches, the draft block, the positions) is swapped into the layers when that context is active.
  - A pass over several contexts runs the attention per context and everything else (mHC, FFN/MoE, all-reduces,
    head) over all tokens at once.
  - Two contexts in passes of 20 + 20 positions give each context exactly its single-context residuals.
  - `--generate N --contexts C` drafts every context's MTP window, verifies all windows in one pass, then accepts
    and rolls back per context.
  - The KDA and MLA projections run over every context's tokens at once; only the conv, recurrence, index
    selection and attention run per context. On 12 nodes every context's tokens equal its single-context run (md5).
  - Aggregate decode (64-token prompts, 256 tokens each, MTP 3 adaptive): 2 contexts 94.0 tok/s, 4 contexts
    **96.1 tok/s**, against 85.5 for the code prompt alone. Each token brings its own 8 experts, so batching
    sequences amortizes little of the step; the draft block still runs per context.
- **fp32 attention** (`STRATA_RUNNER_ATTN_F32=1`) fails the gate (8K perplexity 59.30, next-token 3477, both outside
  the band) for +2 % prefill (147.3 vs 144.0 tok/s at 8K): opt-in.
- The 8K prefill with every default is now **144 tok/s** (133 before fast exp and the kernel changes).
- **Long prompts** (replicated cache, synthetic text = the 8K eval repeated, 64 generated tokens, MTP 3 adaptive):

  | prompt | prefill | decode after it | context buffers per rank |
  |---|---:|---:|---:|
  | 64K | 105.8 tok/s (619 s) | 25.1 ms per token | 1.8 GiB (11.5 GiB left) |
  | 256K | 47.7 tok/s (92 min, before the indexer fix) | 55.5 ms per token (before the fix) | 6.6 GiB (6.8 GiB left) |

  - At 64K the indexer scores cost 0.45 ms per prompt token, against 2.1 before sharding in long passes.
  - The 256K run predates a fix: sharded tokens still walked every pool pair of the score loop, then skipped the
    ones they did not own. It spent 10.5 ms per prompt token and 29.6 ms per decoded token in the indexer, where
    the fixed loop at 64K spends 0.45 and 3.9.
  - The decode numbers include a low draft acceptance on the repeated text (1.9 tokens per pass).

**Remaining plan.** The prefill costs that grow with context are MLA attention and the indexer. Next steps:
1. **Indexer:** shard the scoring from 4096 pools in prefill passes (in the current build, not yet measured).
2. **Latent average:** block it like the scores, so rows are shared by the tokens of a pass whose selections
   overlap. At long context these are mostly the same top pools.
3. **Hybrid attention:** blocking over tokens as well as rows, then a gate run at 512K / 1M.
4. **Experts:** a W4A8 GEMM that wins with ~28+ tokens per slice. The decode-once version (`STRATA_RUNNER_W4A8_MANY`)
   is still slower than the per-tile 4-token kernel, so longer passes do not pay yet (chunk 128 stays the
   default).
5. **Prefill and decode together:** prefill chunks of one context interleaved with other contexts' decode passes
   (several contexts): the decode passes keep their latency while the prefill fills the nodes.

## 7. Reproduce

```
python3 tools/glm_fp4_local_census.py MODEL_DIR --out tools/sim/data/glm_fp4_local_trunk.json
python3 tools/glm_a64fx_sim.py report --pack fp4_local
python3 tools/glm_a64fx_sim.py estimate --pack fp4_local --profile redesign --numerics exact --mtp 2
python3 tools/glm_a64fx_sim.py sweep --pack fp4_local --profile redesign --grid nodes=8,12 --grid numerics=fast,exact
python3 -m unittest tools.test_sim_a64fx
# on a compute node inside an allocation:
STRATA=<synced tree> OUT=<shared dir> tools/a64fx_probe/run_fp4.sh
python3 tools/a64fx_probe/collect_fp4.py <OUT> --job <id> --write
```
The kernel test also runs under qemu on an x86 host (command in `tools/a64fx_probe/glm_fp4_test.c`).

Runner switches (12 ranks, one per node, 47 OpenMP threads each; defaults are the measured best and every switch
keeps the results bitwise unless it says otherwise):

| variable | default | |
|---|---|---|
| `STRATA_RUNNER_EXPERT_PARTS` | 4 | 12 slices every expert over all ranks (used for every number above) |
| `STRATA_RUNNER_LOAD_MTP=1` | off | load the draft block for plain runs too (one image for both) |
| `STRATA_RUNNER_PERSISTENT` | 1 | one OpenMP region per decode, team regions dispatched by flag (0: fork each) |
| `STRATA_RUNNER_MLA_SHARD` | 1 | MLA front rows over the ranks (0: replicated; a different image) |
| `STRATA_RUNNER_ROUTER_SHARD` | 0 | router rows over the ranks (slower; a different image) |
| `STRATA_RUNNER_KDA_BATCH` | 1 | multi-token KDA phases for all positions at once |
| `STRATA_RUNNER_MHC_PREFETCH` | 1 | each thread's mHC weight blocks prefetched at the site's start |
| `STRATA_RUNNER_RSAG=1` | off | the reduce-scatter's completions polled after the scatter too (the old order) |
| `STRATA_RUNNER_ARGMAX_MPI=1` | off | greedy picks by MPI MAXLOC instead of a uTofu all-reduce of (max, index) slots |
| `STRATA_RUNNER_DRAFT_HEAD` | int8 | `exact`: the draft block's picks from the MXFP8 head |
| `STRATA_RUNNER_MTP_ADAPT=1` | off | with `--mtp K`: the depth per round from 1 to K by expected tokens per ms (`_BASE` 22, `_STEP` 10 ms) |
| `STRATA_RUNNER_MTP_TAU1`, `_TAU2` | 0 | skip drafts under this draft-block probability (no gain measured) |
| `STRATA_RUNNER_INDEX_BATCH` | 1 | the indexer's selection for all positions of a pass at once (0: one position at a time) |
| `STRATA_RUNNER_MHC_REPLICATE` | 1 | chunk-major mHC projection copies, one per CMG (2: one shared copy, 0: none) |
| `STRATA_RUNNER_REMAINDER_SPREAD=1` | off | split remainders on ranks 4-11 (a different image; neutral) |
| `STRATA_RUNNER_CMG_THREADS` | 12 | threads per CMG for the barrier groups and the mHC partial sums (results depend on it) |
| `STRATA_RUNNER_IMAGE_KEEPCACHE=1` | off | keep the images' page cache (dropped by default: it takes HBM first touch needs) |
| `STRATA_RUNNER_PREFILL_CHUNK` | 128 | `--generate`: prompt positions per multi-token pass (1: one step per token) |
| `STRATA_RUNNER_AR_MAX` | 16384 | floats per uTofu all-reduce call (65536: prefill +3-4 %) |
| `STRATA_RUNNER_GEMM_DECODE` | 1 | long passes: exact trunk tiles decoded once for all tokens (`gf4_tile_many`) |
| `STRATA_RUNNER_SITE_TOKENS` | 1 | long passes: mHC sites with whole tokens per thread (0: batched groups of `STRATA_RUNNER_SITE_MAX`, default 16) |
| `STRATA_RUNNER_SCORE_BLOCK` | 1 | long passes within the selection size: MLA scores in 4 x 4 (token, position) blocks |
| `STRATA_RUNNER_TOKEN_SPLIT` | 1 | long passes: few-tile jobs as (tile, token block) items |
| `STRATA_RUNNER_FAST_EXP`, `_MASK` | 1, 15 | FEXPA vector exp / sigmoid / SiLU / SwiGLU (approximate); mask 1 KDA, 2 SwiGLU, 4 router, 8 MLA softmax |
| `STRATA_RUNNER_MHC_SVE`, `STRATA_RUNNER_PLAN_TEAM_MIN`, `GF4_W4A8_LANE=1` | 1, 2, off | vector Sinkhorn; passes of at least N tokens plan experts over the team; the indexed SDOT W4A8 form |
| `STRATA_RUNNER_INDEX_SKIP`, `STRATA_RUNNER_INDEX_SHARD_MIN` | 1, 16384 | skip indexer scores while every pool is kept; shard the scoring over ranks above this many pools |
| `STRATA_RUNNER_CACHE_HOT` | 65536 | hybrid latent cache beyond this many positions (ring + sharded cold blocks; 0: replicated) |
| `STRATA_RUNNER_ATTN_F32=1`, `STRATA_RUNNER_W4A8_MANY=1` | off | fp32 MLA scores/averages (approximate); W4A8 decode-once kernel (bitwise, slower) |
| `STRATA_RUNNER_ROT_SPREAD`, `GF4_FLOAT_LANE=1` | 1, off | shared-expert tile rotation away from the router; the indexed exact float kernels |
| `--contexts C` | 1 | C resident sequences (eval: segments; `--generate`: batched decode) |
| `STRATA_RUNNER_PREFILL_TIMERS=1`, `STRATA_BENCH_NT=N` | off | the prefill's phase timers per prompt token; `--bench-gemv` for N tokens per job |
| `STRATA_RUNNER_TIMERS_ALL=1`, `STRATA_RUNNER_NUMA_REPORT=1`, `STRATA_RUNNER_LOAD_TRACE=1`, `STRATA_RUNNER_SITE_TRACE=1` | off | per-rank timers, resident GiB per NUMA node, per-rank load progress, per-thread lateness in the mHC sites |

Images live in a node-local directory (`--image-dir`); a rank whose image write fails (a full disk) makes the
others rebuild on the next run, since every rank replays or none does.
