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
| MLA front all-reduces posted asynchronously in two stages, index-query transpose only for scoring tokens, `gf4_tile_many2` | **198.0** (c256, pipe) / 178.2 (default) | **175.6** (c512, pipe) |

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
- **Two-stage async MLA front.** In a pipelined pass, the front's two all-reduces (the sharded x projections, then
  `iq`) are posted on the comm thread. The waits then have the other half's work in front of them:
  `comm_mla_front` drops from 0.36 to 0.05 ms per prompt token.
- **`gf4_tile_many2`** (`STRATA_RUNNER_MANY2=0`: off): the FP32 long-pass kernel takes two tiles at once, 2 tiles
  x 3 tokens per register block. That is 24 chains and 5 loads per 6 FMLAs, against 16 chains and 5 loads per 4.
  The per-chain order is unchanged, so it is bitwise (node test over all three layouts). On one core it gives +14 %
  for MXFP8 (11.8 -> 13.4 MAC per cycle) and +21 % for block FP8 (13.0 -> 15.7). In the model it gives 2K 198.0 vs
  193.1 tok/s (`kda_proj` 0.74 vs 0.79 ms).
- **No gain:** splitting the MLA front from the rest of attention so both halves' fronts run first (2K 185.6). The
  `MANY_KS` W4A8 GEMM microkernels (none beat the 4-token tile, removed).

Per prompt token at 2K c256, pipelined (5.4 ms): routed experts 1.08, KDA projections 0.80 (input projections 0.54),
MLA attention 0.76, MLA projections 0.66, mHC 0.48, router + shared expert 0.45, comm waits 0.60, draft block
0.25, KDA small phases 0.24.

#### 6.2.1b Overheads found against the roofline (2026-10-25, 12 'k' nodes, through the iteration loop)

Every change is bitwise: 8-layer dumps against single steps, and the md5 suite at 2K / 8K / decode b. A different
allocation ('k' nodes) gives 2K 201.1 / 8K 178.6 tok/s on the 6.2.1 binary (vs 195.8 / 174.8 on the 's' nodes).

| change | effect (per prompt token, 2K c256) | 2K tok/s |
|---|---|---:|
| mHC: the projection's blocks with their row sums in registers (one SVE routine per block, no call or store) | `mhc.site` 0.506 -> 0.465 ms | |
| mHC: a token group's residual update before its projection | the projection reread values still in the store queue: 0.467 -> 0.331 | |
| pipelined pass: the summed sublayer outputs in per-token rows first touched by the token's owner | sites read them from their own CMG: 0.331 -> 0.298 | 201.8 ('s') |
| activation pools per call site (allocated once; only copied and quantized) | `acts_of` zero-filled T buffers on the main thread: router 0.167 -> 0.027, KDA 0.058 -> 0.010 | 209.1 (vs 198.4) |
| expert outputs summed in SVE (16 lanes, the same zero start and fused multiply-adds) | the scalar loop chased a row pointer per element: `experts.sum` 0.273 -> 0.037 | **219.4** |
| GEMV items balanced over all jobs of a call, (tile pair, token block) | KDA q/k/v bench 692 -> 749 GMAC/s on 47 threads; neutral in the model | 217-219 |
| indexer keys, pools and cache rows over the team (the snapshot callback was always passed, so this path never ran) | `mla.index` 0.132 -> 0.076 (8K 0.371 -> 0.311) | 218.5; 8K 198.8 |
| pipelined passes the default | | |

Measured on these nodes (one core, no memory traffic): FP32 32, FP64 16 and int8 SDOT **128** MAC per cycle. Eco
mode is off (the job requests `eco_state=0`). The roofline's int8 peak (6.2.2) is twice what it assumed.

Not kept:
- **Allocating the activations on their owner threads** (`STRATA_RUNNER_ACTS_PAR=1`): the GEMVs reading them got
  slower than the allocation saved (198.2 vs 200.1).
- **Finer GEMV items** (12 or 24 per thread): each tile's weights are decoded once per fewer tokens (669 / 544
  against 749 GMAC/s).
- **Sharding mHC by token over the ranks** (exact: rsag sums every element in rank order) waits for a 3-4 part
  pipeline. With two halves, the all-gather of the site outputs has only the other half's site to hide behind.

The full-node FP32 GEMV still runs at 7.9 MAC per cycle per thread against 10.7 on 12-24 threads and 14.5 for one
core. Per thread, the rate falls as CMGs are added (12: 10.7, 24: 10.1, 36: 9.5, 47: 7.9).

#### 6.2.1c The FP32 GEMVs (2026-10-25, 'k' nodes)

**The kernel.** `priv/a64fx-fp4/blk_bench.c` times the inner loop alone, with F32 weights in L1:
- 2 tiles x 3 tokens, the bitwise 4-chain order, runs at 17.5 MAC per cycle (peak 32).
- Six blockings fit one model within ~5 %: each 64-byte weight load costs ~1.8 cycles, each `ld1rw` broadcast ~0.7.
  Pre-broadcast activations in memory are slower (2-7).
- A hand-written asm block (exactly 2 loads, 3 broadcasts and 6 FMLAs per column) is bitwise but slower. The
  compiler's schedule issues the loads ahead of their FMLAs; with 24 accumulators there are only 8 spare
  registers, not enough to software-pipeline A64FX's long load latency.
- Relaxed orders (2 chains per tile and token, 3 x 4) reach 21.2 (+21 %). They would need the text gate, for
  ~0.1 ms per prompt token: not done.

The kernel stays as it is: 13-15 MAC per cycle at the model's shapes.

**The split over the team** was the loss. Whole items (tile pair x token block, or whole tiles per job) left the
busiest thread ceil(items / n) of them: 7 of 6.13 on 47 threads; 48 threads, 6 of 6, ran 13 % faster in total.
- The flattened (job, tile pair, token) space now goes in equal contiguous cost ranges, so each thread gets token
  ranges of 1-3 pairs. Bench on 47 threads: KDA q/k/v 749 -> 946 GMAC/s (10.1 MAC per cycle per thread, as on
  12-24 threads), router 500 -> 746.
- The KDA phase's plan holds two K-split matrices (`f_a`, `g_a`). The balanced split now covers the plan's other
  jobs too: `kdap.p1` 0.56 -> 0.415 ms.
- Bitwise (each token is computed once, by the same kernel).

md5 suite (all unchanged): 2K **234.8** tok/s (215.4 non-pipelined), 8K **208.6**, decode b 11.5 ms per token.

#### 6.2.1d Routed experts: balance and plan (2026-10-25, 'k' nodes)

- **Balance.** Busiest thread against the mean (c256): gate/up 0.404 / 0.368, down 0.434 / 0.417.
  - `gemv_share_cmg` keeps every expert tile in the CMG whose thread placed it, so the weights stream from that
    CMG's own memory. The CMG's tiles go in equal contiguous cost ranges, tile by tile, to its threads
    (`STRATA_RUNNER_EXPERT_BALANCE=0`: the placement split; long passes only, decode unchanged).
  - Down mean 0.419 -> 0.331 ms, gate/up unchanged; the expert phase 0.956 -> 0.896.
  - The prefill total did not move (2K 234.1 vs 233.9): `comm.wait` rose 0.106 -> 0.158. The likely mechanism:
    the shorter FFN halves no longer hide the previous layer's second-half FFN all-reduce (2 MB, ~0.67 ms on the
    wire), and the next MLA layer's front all-reduces queue behind it (one comm thread, the same order on every
    rank).
- **Plan.** The per-use token lists by a counting sort into one array (separately allocated per-use vectors cost
  ~0.1 ms per call in cache misses on thread 0): `experts.plan` 0.083 -> 0.068.
- **The kernel at expert shapes** (`--bench-gemv`, 47 threads, GMAC/s):

  | tokens per slice | gate/up | down |
  |---:|---:|---:|
  | 1 | 405 | 288 |
  | 2 | 659 | 506 |
  | 4 | 927 | 751 |
  | 8 | 1204 | 731 |

  The model runs ~4 tokens per slice. Down is capped by its K of 192 per rank with 12 expert parts; fewer parts
  would change the partial sums (gated) and the image layout.

md5 suite unchanged: 2K 234.0, 8K 209.9, decode b 11.48 ms.

#### 6.2.1e The exposed all-reduces (2026-10-25, 'k' nodes)

**Diagnosis.** Per prompt token at 2K:
- the comm thread is busy **1.45 ms** for ~1.8 MB of all-reduces, against 0.6 ms at the probe's 3 GB/s;
- the waits the pipeline exposes total 0.15-0.17 ms, spread over the slots (the largest: half A's attention
  output in KDA layers, 0.07);
- the rsag call copies the payload into its send slots, sums the 12 shards and copies the gathered result back:
  ~5 MB of single-core memory traffic per MB, slowed further by the 47 compute threads;
- placing the comm thread on CMG0, next to its buffers (team on cores 1-47), cut its time only 10 % (and slowed
  decode): locality is not the cause.

**Staged all-reduces** (default; `STRATA_RUNNER_AR_STAGED=0`: off).
- Four extra registered regions ("banks", one per pipeline slot: attention / FFN x half A / B;
  `STRATA_RUNNER_AR_BANK` floats each, default 1M = a 256-token half).
- The team writes each half's rows straight into the bank's send slots, shard by shard. The comm thread only sums
  (in rank order, as before) and Puts. The team reads the sums from the bank's result slots.
- A bank is reused four all-reduces later; every rank takes part in the three in between, so no slot is
  overwritten while in use.
- Bitwise.

| | before | staged |
|---|---:|---:|
| comm thread, ms per prompt token | 1.45 | 1.01 |
| exposed waits, ms | 0.15-0.17 | 0.09-0.11 |
| 2K tok/s (c256 / c512) | 234.0 / 239.7 | **238.5 / 244.3** |
| 8K tok/s | 209.9 | 211.2 |

#### 6.2.1f More pipeline parts; chunk size (2026-10-25, 'k' nodes)

**Pipeline parts** (`STRATA_RUNNER_PIPE_PARTS`, default 2; 2 staged banks per part). 3 and 4 parts are bitwise
(8-layer dumps), but slower. The comm thread's per-call costs multiply, and every part's GEMVs see fewer tokens:

| parts | 2K c256 | 2K c512 | comm thread (c512) | waits (c512) |
|---:|---:|---:|---:|---:|
| 2 | 232.7 | 240.6 | 0.63 | 0.038 |
| 3 | 223.3 | 237.6 | 1.76 | 0.073 |
| 4 | 213.8 | 227.0 | 2.79 | 0.159 |

**Chunk sweep** (one md5 per prompt at every chunk). The default prefill chunk is now **1024** (it was 128). The
multi-context path keeps 128: its passes hold every context's chunk.

| chunk | 128 | 256 | 512 | 1024 | 2048 |
|---|---:|---:|---:|---:|---:|
| 2K tok/s | 215.4 | 234.4 | 241.1 | **245.5** | 228.7 |
| 8K tok/s | | 201.6 | 209.8 | **216.7** | 204.5 |

md5 suite at the defaults: 2K **245.8**, 8K **215.5**, decode b 11.44 ms (all md5 unchanged).

#### 6.2.1g MLA attention (2026-10-25, 'k' nodes)

**Where it goes.** Per prompt token (chunk 1024):
- 2K: 0.67 ms (average 0.27, scores 0.17, expand 0.06, q prep 0.05, indexer 0.035);
- 8K: 1.52 ms (average 0.62, scores 0.40, indexer scores 0.24).

The FP64 dots ran at 7-15 % of the FP64 peak.

**The kernel on one core** (`priv/a64fx-fp4/att_bench.c`, 4 queries x 4 positions, 1024 positions):
- 5.2 MAC per cycle, and 9.2 with the cache rows L1-resident: the bound is streaming rows (or queries) from L2,
  1 byte per MAC.
- Not the cause: f64 queries, zip loads, an f64 cache copy (6.4 at most), padded strides, merging FMLAs, and a
  hand-written asm loop with exactly 28 instructions per step (5.4).
- A transpose tree reproduces `svaddv` bit for bit (FADDV's recursive pairwise order): +11 %, not used yet.

**Changes** (all bitwise: 8-layer dumps; 4096-token dumps against the morning's reference, 180 of 180, new suite
`t1l`; md5 suite):
- **Prefix scores:** an item covers 4 consecutive position blocks for the same 4 tokens and head, so the queries
  come from L2 once per 16 positions (`STRATA_RUNNER_SCORE_PB`, default 4): 0.176 -> 0.148 ms at 2K.
- **Gather scores and average** (beyond the 2048-position selection; `STRATA_RUNNER_GATHER_BLOCK=0`: off):
  - scores in items of 1 token x 3 heads x 4 positions over 4 blocks, each gathered row feeding 3 heads' dots in
    registers: 0.397 -> 0.315;
  - the average in items of 1 token x 3 heads x 32 latents (the scalar loop read each row segment per head):
    0.623 -> 0.400;
  - attention at 8K 1.515 -> 1.208 ms, 8K **216.5 -> 231.4 tok/s**.
- **Not kept:** sharding the indexer at 8K (`STRATA_RUNNER_INDEX_SHARD_MIN` 512 / 1024). The scores drop 0.24 ->
  0.05-0.09, but the synchronous candidate exchange costs 0.33-0.44 ms: 206.6 / 210.2 against 220.

md5 suite: 2K **245.2**, 8K **231.3**, decode b 11.52 ms (all unchanged).

**Then** (same checks, all bitwise):
- **The prefix average with each cache row segment shared by 2 heads** (4 tokens x 2 heads x 16 latents per item;
  `STRATA_RUNNER_AVG_HEADS=1`: as before): 0.280 -> 0.157 ms at 2K.
- **The scores' horizontal sums by the transpose tree** (`STRATA_RUNNER_TREE_SUM=0`: `svaddv`). It is bitwise
  in the model too, but neutral there (0.151 vs 0.157): the kernel is bound by L2, not by its reductions.

md5 suite: 2K **248.5**, 8K **234.5**, MLA attention 0.56 ms (2K) / 1.16 ms (8K) per prompt token.

#### 6.2.1h mHC sites (2026-10-25): no gain found

At chunk 1024 the sites take ~0.31-0.34 ms per prompt token.
- Per thread (`RUNNER_MHC_THREADS`): 0.18-0.28 ms; the slowest thread of each site 0.345 against a mean of 0.269.
  The spread is within the CMGs, not between them.
- Thread 0's projection runs at ~4.3 MAC per cycle (the inner kernel's load-bound rate is ~25).

Tried, all bitwise, all slower:
- tokens taken dynamically within a CMG: 0.407 vs 0.334 ms (the owner's L1 holds its tokens' residuals);
- each virtual thread's row sums kept in registers, one tile at a time: projection 0.198 vs 0.146;
- 12 tokens per pass over the weights (`STRATA_RUNNER_MHC_TOK`; each block streamed from L2 once per 12 tokens):
  0.391 vs 0.314.

The bound is not the weight stream, the store round trips or the balance; it stays open.

**The microbenchmark** (`priv/a64fx-fp4/mhc_bench.c`, the projection of 4 tokens on one core) reproduces it: 5.2-5.4
MAC per cycle. L1-resident weights or activations change nothing, and neither do row sums in registers. Without
the per-unit sum of squares it runs at 7.7.
- `sumsq` compiled to 128-bit NEON. It now runs in 8 double lanes with FADDV's tree, which is the loop's own order
  (bitwise: 0 mismatches in 6K blocks).
- The block kernel loads two BF16 columns at once (`ld1h`, widened by `zip` with zero) instead of one extending
  load and a shift per column.
- Bench 200.7 -> 159.8 us; in the model the projection drops 0.146 -> 0.124 ms (thread 0) and the site to
  ~0.30-0.35.
- md5 suite (bitwise, decode too: `sumsq` serves the norms): 2K **253.0**, 8K **235.6**.

#### 6.4b Batched multi-context decode, profiled (2026-10-25, 'k' nodes)

`generate_multi` (`--contexts C`; suite `mc`): 128-token prompts, 64 tokens each, MTP 3 adaptive, eval text (~50 %
of drafts accepted):

| C | ms per generated token | total tok/s | ms per verify pass |
|---:|---:|---:|---:|
| 1 | 18.3 | 54.6 | |
| 4 | 12.8 | 78.3 | 79.8 |
| 8 | 10.7 | 93.3 | 133.9 |
| 16 | 9.26 | 107.9 | 225.9 |

The shared phases amortize from C = 1 to 16 (ms per generated token): experts 3.44 -> 1.76, mHC 2.46 -> 0.78,
router 2.00 -> 0.69, all-reduce 1.96 -> 0.84. The per-context ones barely do:
- MLA attention 1.16 -> 1.09 (a sequence of team regions per context group, per layer);
- KDA conv / recurrence / gates 1.08 -> 0.76;
- MTP drafting 0.95 -> 0.86 (each context's drafts serial, single-token draft-block passes).

Together ~2.7 of the 9.3 ms at C = 16.

**Batched MTP drafts** (default; `STRATA_RUNNER_MTP_BATCH=0`: one context at a time).
- Each round drafts depth by depth: one draft-block pass per depth over every context still drafting at that
  depth (`mtp_forward_ctx`). The draft layer's attention runs per context through `mla_back`'s grouped path,
  with that context's draft layer state and its own cell; the projections, MoE and head run over all contexts'
  tokens.
- The post-verify MTP re-syncs of all contexts are one pass.
- Every context's tokens are unchanged (md5 per context at C = 4 / 8 / 16).
- MTP 0.861 -> 0.438 ms per generated token at C = 16. Total tok/s: C = 4 77.7 -> 80.0, C = 8 93.5 -> 96.7,
  C = 16 107.8 -> 112.9.

**One attention call for all contexts** (`STRATA_RUNNER_GROUP_ATTN=0`: per group).
- The verify pass (and the batched drafts) select per context. Then a single `mla_attention` covers every group's
  tokens, each reading its own context's latent cache through a per-token base pointer.
- Such a pass takes the per-token (gather) kernels, because the prefix kernels share cache rows between tokens.
- The per-group calls had cost ~190 us each for ~2 tokens.

**KDA recurrence items per (column group, context)** (`STRATA_RUNNER_KDA_REC_CTX=0`: per column group). Each item
walks only its context's tokens, instead of chaining all ~34 tokens' state loads.

Per-context md5s are unchanged in all four on/off combinations.

| at C = 16 | before | after |
|---|---:|---:|
| MLA attention, ms per generated token | 1.09 | 0.63 |
| KDA recurrence | 0.34 | 0.26 |
| MTP | 0.44 | 0.37 |
| total tok/s | 112.7 | **120.4** |

C = 4: 80.0 -> 81.8. Still per context: the selection (`mla.index` 0.15) and the KDA convolution's per-context
histories (0.23).

**Parallel per-context selection** (`STRATA_RUNNER_GROUP_SELECT=0`: one context after another).
- When every group has at most 8 tokens and stays inside the 2048-position selection, and the cache has no ring and
  the index skip applies, all contexts' `mla_select_multi` calls run in one team region, one context per item.
- Each item works on its context's `LayerState` directly. The timers are off inside the region, because the timer
  map is shared.
- Otherwise the old sequential loop runs, with its own parallel regions.

**KDA conv items per (part, context, channel)** (`STRATA_RUNNER_KDA_CONV_CTX=0`: per part and channel). Each item
walks one context's history buffer, instead of all contexts' tokens per channel.

The full suite passes with both on (t1 64/64, t1l 180/180, md5s; 2K 253.2, 8K 236.7 tok/s). Per-context md5s are
identical in all three variants.

| total tok/s | C = 4 | C = 16 |
|---|---:|---:|
| both on (default) | 83.1 | **123.6** |
| `GROUP_SELECT=0` | 82.7 | 121.3 |
| `KDA_CONV_CTX=0` | 83.1 | 122.7 |

At C = 16, the selection saves 0.15 ms per generated token and the conv items 0.06 (8.30 -> 8.09 ms).

**Work arrays kept across calls, and a vector conv** (2026-10-26). Bitwise: full suite, and per-context md5s at C = 4 / 16.
- `kda_multi`'s outputs, its activations (`km_afa`, `km_aga`, `km_ay`) and `mla_attention`'s three work arrays were
  allocated and zero-filled per call. They are now kept across calls; every element is written before it is read.
- At C = 16 this took `kdap.setup` from 0.172 to 0.048 ms per generated token and `mla.qprep` from 0.077 to 0.052.
  The conv loop that follows the setup went from 0.150 to 0.046 ms with its code unchanged.
- glibc malloc tunables (no mmap, no trim) made no difference, so the cost was not page faults.
- The KDA conv runs 16 channels per item with SVE. `svld3` / `svld4` read the histories and weights, and each lane
  does the same multiply and fused multiply-adds (`STRATA_RUNNER_KDA_CONV_VEC=0`: per channel).

| | 2K prefill | 8K prefill | C = 4 | C = 16 |
|---|---:|---:|---:|---:|
| before | 253.2 | 236.7 | 82.7 | 123.6 |
| work arrays kept | 257.1 | 242.4 | 86.0 | 129.2 |
| + vector conv | **269.5** | **245.9** | 86.1 | **129.2** |

Tried and dropped:
- `site_multi` for passes of up to 64 tokens. At C = 16 mHC rose from 0.73 to 1.90 ms per generated token.
- Its per-CMG sums spread over the whole team: no change.

`site_tokens`' imbalance at about 37 tokens remained: thread busy maximum 0.64 against a mean of 0.37 (fixed below by
sending these passes through `site_multi`).

**`site_multi` at 4-16 tokens** (2026-10-26). These are the mHC sites of decode passes: single-context MTP verify
(T = 4) and multi-context passes up to 16 tokens.

Phase clocks are inside the function (`STRATA_RUNNER_MHC_PHASES=1`: thread 0 after each barrier, every thread's
projection time, calls, bucketed by T), measured in C = 1 / 2 / 4 / 16 runs.

The projection dominated and grew 5-7 us per token per thread, against ~0.35 us of arithmetic. The cause was the
residual update: per (unit, token), 32 scalar stores, then vector reloads of the same values by the sum of squares and
the kernel. A64FX forwards stores to loads poorly, and `site_tokens` had already avoided this.

Changes, all bitwise:
- **Write first.** Each thread updates its chunks of every token's residual with SVE first: the same multiply and
  fused multiply-adds (`STRATA_RUNNER_MHC_WRITE_FIRST=0`: per unit).
- **Row sums in registers.** The projection uses `mhc_block_acc`, tokens in groups of 4 with this thread's units
  inside, the same unit order per token (`STRATA_RUNNER_MHC_UNITS=0`: `gf4_bf16k` partials).
- **`y` sized in parallel.** Each thread sizes its own tokens' `y` vectors; the serial zero fill had cost ~1.7 us per
  token.
- **Vector combination.** 16 lanes at a time. Its sum of squares is taken from the registers (zip to double lanes in
  `ld1uw`'s order, the same FADDV).
- A prefetch of the residual chunks (`STRATA_RUNNER_MHC_PREFETCH_R`) gained little (12 tokens: 89 -> 78 us) and is kept.

| us per call | T = 4 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|
| before | 59-63 | 89-128 | 149 | 240 |
| write first | 55-58 | 79-99 | 111 | 169 |
| all | **43** | **59-73** | **84** | **112** |
| of which projection (all) | 14 | 22-30 | 38 | 54 |

The rest at T = 4 (us): coefficients 5.9 (Sinkhorn, one thread per token), combination 6.4, and the four short phases
2.5-3.3 each, which is roughly the barrier cost.

| decode | before | after |
|---|---:|---:|
| C = 1, ms per token (`mc`, adaptive MTP) | 18.35 | **17.44** |
| decode b, ms per token (t2) | 11.39 | **10.87** |
| C = 2 total tok/s | 64.7 | **68.4** |
| C = 4 | 84.5 | **92.3** |
| C = 16 (mostly `site_tokens`) | 128.5 | 128.6 |

mHC at C = 4: 1.76 -> 1.02 ms per generated token.

**Merged phases: two barriers instead of five** (`STRATA_RUNNER_MHC_MERGE=0`: the five phases). After the
projection's barrier, thread `th` takes token `th % T` and a slice of that token's chunks. It does three things for
that token:
- it sums every thread's partials itself, per CMG and then over the CMG sums (the same order as the CMG leaders);
- it computes the coefficients, so each thread of the token gets the same values;
- it computes its chunks of x and their sums of squares.

After one more barrier it computes `inv_x` from all chunks and writes its slice of y. The results are bitwise: full
suite PASS, and the mc md5s are identical at C = 1/4/8/16.

| us per call (C = 4 run) | T = 4 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|
| five phases | 46.3 | 76.5 | 85.1 | 117.0 |
| merged | **39.5** | **71.8** | **82.2** | 118.0 |

At T = 16 there are only about 3 threads per token. Their x chunks come from residual lines the other threads wrote,
so the combination took 22 us instead of 14 + 3, and the merge doesn't pay at that size.

Decode:
- C = 1: 17.77 -> 17.35 ms per token, measured against `MERGE=0` in the same allocation.
- decode b: 10.87 -> 10.75 ms.
- C = 4: 91.3 -> 91.6 tok/s.
- C = 16: 128.3 -> 128.5 tok/s.

**Passes of 17-64 tokens through `site_multi`** (2026-10-26). At C = 16 (passes of about 37 tokens), `site_tokens`
gives one token to each of 37 threads and leaves 10 idle. Each busy thread streams all of `fn` (1 MB) for its one
token, so the CMGs with 12 busy threads were limited by L2 bandwidth: busy time per CMG was 0.56 / 0.55 / 0.44 /
0.16 ms per token.

`site_multi` splits by chunk instead: `fn` is read once by the team, and every thread's residual chunks are its own.
It had lost at this size before (0.73 -> 1.9 ms) because of the store-forwarding projection; with the fixes above it
wins.

Changes:
- `SITE_MAX` is 64, and the default `site_max` is 64 (`STRATA_RUNNER_SITE_MAX=16`: the old split).
- The merged phases apply at every size (`STRATA_RUNNER_MHC_MERGE_MAX`). At 24-48 tokens they beat the five phases
  too: 151-262 -> 139-233 us.
- **Pipelined residual prefetch.** `STRATA_RUNNER_MHC_PF_DIST`, default 2: token t + 2's lines are prefetched during
  token t's update. Prefetching all tokens at once means about 27 lines per token, more than a core keeps in flight.
  Distance 2 beat 4 and 8; the update at 36 tokens went from 62 to 51 us.
- **The sum of squares in the row partials' line.** It goes at offset 32 of each (token, thread) line, so the merged
  sums read 47 lines per token instead of 94. Sums and coefficients at 36 tokens: 26.6 -> 20.8 us.

All of this is bitwise. The full suite passes, and the mc md5s are identical at C = 1/4/8/16.

| C = 16 | `site_tokens` | `site_multi`, 5 phases | merged | + prefetch, one line |
|---|---:|---:|---:|---:|
| mhc ms per token | 0.795 | 0.739 | 0.673 | **0.603** |
| tok/s | 128.0 | 129.9 | 130.6 | **131.9** |

C = 8: 109.3 -> 115.9 tok/s. C = 4: 92.5. Prefill is unchanged (2K 256-261, 8K 242-243).

At 36 tokens a call takes 169 us:
- the projection, 107 us: the residual update 52 (latency-bound reads of other CMGs' lines), the blocks about 40, and
  the 2-vs-3-chunk split, whose busiest thread runs 11 % over the mean;
- the sums and coefficients, 21 us;
- the combination and output, 16 us;
- the serial setup and teardown, 14 + 11 us (the latter includes the phase timers themselves).

**KDA recurrence and the MTP block per context** (2026-10-26).

*Recurrence.* A (head, 16-column group) item runs two chains of 128 dependent adds per token, which every lane must
keep in the reference's order. One item at a time therefore ran at the add latency. `kda_groups_step<G>` interleaves
G column groups of one head; they share k, q and the decay, and each lane's operations are unchanged, so the result
is bitwise.

G (1, 2 or 4) is chosen from the rounds of items per thread times a step's cost at G (`STRATA_RUNNER_KDA_REC_G`
fixes it). The usual choices:
- C = 16: 768 items -> 192 (G = 4);
- one context: 48 items on 47 threads -> 24 (G = 2), so no thread takes a second item.

| | G = 1 | auto |
|---|---:|---:|
| kdam.rec, C = 16, ms per token | 0.268 | **0.137** |
| kdam.rec, C = 1 | 0.404 | **0.342** |
| C = 16 tok/s | 131.3 | **133.9** |
| C = 8 | 116.0 | **117.7** |
| C = 4 | 93.5 | **94.3** |

*MTP.* Sub-timers `mtp.front / attn / moe / head / argmax / calls / tokens`; `STRATA_RUNNER_MTP_DETAIL=1` also
attributes the shared timers to the block, as `mtpa.*` and `mtpm.*`. At C = 16 the block was 0.36 ms per token over
3.4 calls per pass of about 9 tokens each. Per call:
- attention, 1.08 ms: MLA projections 0.49, attention 0.37, all-reduces;
- MoE, 0.84 ms: experts 0.40, router 0.16;
- front, 0.40 ms;
- head, 0.39 ms.

Most of this is weight streaming per call, plus the experts the tokens route to.

A context's re-sync (the cells of its accepted drafts) now joins its next depth-1 draft: the cells follow on, so one
pass replaces two, and a context that is done needs none (`STRATA_RUNNER_MTP_FUSE_RS=0`: a re-sync pass after each
verify). md5s are identical.

| C = 16 | before | after |
|---|---:|---:|
| MTP calls per token | 0.141 | **0.100** |
| MTP ms per token | 0.361 | **0.342** |
| tok/s | 134.4 | **135.0** |

The calls carry more tokens, at about 0.24 ms per extra token (experts), so the gain is smaller than the call count
suggests. Single-context `generate()` (C = 1, decode b) does the same on its non-lazy path: MTP calls per token
1.33 -> 1.00, mtp 0.964 -> 0.849 ms, C = 1 17.14 -> 17.00 ms per token, decode b 10.69 ms; md5s identical.

**MLA expand and query prep per context** (2026-10-26). Sub-timers `mla.ex_jobs / ex_plan / ex_run / ex_copy /
ex_calls`. At C = 16 the expand took 0.108 ms per token, but its GEMV (already balanced by `balanced_items`) only 0.024.
The jobs' setup took 0.061 (about 110 us per call): `mla_attention` built fresh `std::vector<Activation>`s each call,
and thread 0 malloc'd and copied each (token, head) row in `set()` (T x heads of them). The absorb's `qh` (`mla.qprep`)
was the same.

The fix (`STRATA_RUNNER_MLA_ACTS_PAR=0`: as before), bitwise:
- both arrays persist (`ma_qh`, `ma_avs`), so their `x` stays allocated;
- for passes over 4 tokens, the team fills them;
- the `values` copy is spread over the team too.

| C = 16 | before | after |
|---|---:|---:|
| mla.expand, ms per token | 0.100 | **0.050** |
| mla.qprep | 0.047 | **0.010** |
| mla_attn | 0.430 | **0.293** |
| tok/s | 134.6 | **137.6** |

C = 8 119.0 -> 119.7, C = 4 94.9 -> 95.2, C = 1 16.96 -> 16.92 ms. Prefill uses the same code: 2K 268.6, 8K 252.4
(from about 260 / 243). Full PASS (t1 64/64, t1l 180/180, dec-b 10.54 ms); mc md5s identical.

**Routed experts per context: two balance attempts, both dropped** (2026-10-27). At C = 16 the experts take 1.70 ms per
token:
- gate/up GEMVs 0.69 mean / 0.77 max, and down 0.54 / 0.64; thread 0 waits 0.08 + 0.12 at the barriers;
- plan 0.15;
- weighted sum 0.08.

Neither attempt changes results (who computes a tile does not change its bits; md5s identical in every variant):
- **A fixed cost per tile** in `gemv_share_cmg`'s CMG split, `cols x (tokens + K)`: the down projection's busiest
  thread rose from 0.63 to 0.81 / 0.99 / 1.16 ms at K = 1 / 2 / 4, and C = 16 fell 137.1 -> 135.5 / 132.0 / 127.7.
  One cost model does not fit both the routed slices and the shared expert's down projection, which sit in one list.
- **Dynamic tiles per CMG**, from an atomic counter (chunks of 2-64 tiles, each CMG's own tiles, so memory stays
  local). Small chunks balanced the threads (wait 0.03-0.06) but raised the down projection's mean cost from 0.54 to
  0.69-1.17 ms: its tiles are tiny, so the counter's line is contended. Large chunks brought the imbalance back.
  C = 16 reached at best 135.7 (chunk 8) against 137.1 static.

What remains is kernel-bound:
- The routed slices see about 1.6 tokens each at C = 16, so the W4A8 kernel's per-weight decode dominates (the plan's
  expert GEMM, step 2, targets slices of 8+ tokens).
- `router.gemv` (router 288 x 4096 BF16, plus the shared gate/up) costs 7 us per token per layer at C = 16 and 11 at
  C = 1. That is about 4 MACs per cycle per core, against the 12 of `gf4_tile_many` in 128-token prefill: each
  thread's item is half a tile pair over about 20 tokens. `STRATA_RUNNER_GEMM_DECODE=0` (0.489) and
  `STRATA_RUNNER_MANY2=0` (0.472) are no better than the default (0.443 ms per token).

**W4A8 expert kernel: two bitwise variants, measured and dropped** (2026-10-27). The setup is a one-core microbench
(`priv/a64fx-fp4/w4p_test.c`, suite `w4p`) on the expert slice shapes: gate/up 192 x 4096 and 128 x 4096, down
4096 x 192. It runs 1-4 tokens, 64- and 16-value activation groups, L2-resident and streamed from a 32 MB set.
`gf4_w4a8` (`w4a8_tile`) runs at:

| MAC per cycle, L2 / streamed | 1 token | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| 192 x 4096, single tile | 12.9 / 12.0 | 16.7 / 16.1 | 18.4 / 17.9 | 19.2 / 18.8 |
| tile pairs | 15.0 / 13.3 | 18.2 / 16.8 | 19.2 / 18.1 | 21.8 / 20.7 |
| ILP form | 11.0 / 7.6 | 10.5 / 10.7 | 14.5 / 14.1 | 14.7 / 14.4 |
| 4096 x 192, single | 11.6 / 10.9 | 15.8 / 15.1 | 16.6 / 16.2 | 16.8 / 16.5 |
| tile pairs (earlier run) | 11.6 / 5.1 | 15.2 / 7.9 | 15.8 / 8.5 | 19.2 / 11.0 |

That is 10-17 % of the core's SDOT peak (2 pipes x 64 MACs).

The two variants:
- **Tile pairs** (`w4a8_tile2`): two tiles in lock-step, each activation broadcast used by both, and two independent
  float chains. On long rows that is +12-17 %; streamed short rows (down) ran at half speed, so pairs were limited to
  rows of 1024+ columns. In the runner it changed nothing: experts.up 0.688 vs 0.687 ms per token at C = 16, because
  `gemv_share_cmg` gives a thread only one or two tiles of a slice, so pairs seldom form.
- **ILP form**: the 4 SDOTs of a group into separate accumulators added as integers, and two superblocks at a time.
  It was slower everywhere (the extra adds, and spills). The SDOT chain's latency is not the limit, so the
  instruction mix (decode TBLs, scale conversions, SDOT, cvt and FMA per group) is.

Both were removed. A real gain needs fewer instructions per MAC (e.g. weights decoded once and reused across more
tokens, which decode passes do not have), not more parallel chains.

Full suite PASS with both changes (t1 64/64, t1l 180/180; 2K 263.2, 8K 243.7, dec-b 10.73 ms).

#### 6.2.2 Prefill roofline (2026-10-25)

`tools/a64fx_fp4/prefill_roofline.py` counts the MACs per prompt token from the census. Matrices: 16.4 GMAC,
of which routed experts 8.46 (W4A8), and the exact FP32 trunk 7.6 (KDA projections 4.68, MLA 1.37, router + shared
1.11, dense FFN 0.45, draft block 0.39). MLA attention adds 0.81 GMAC at 2K and 1.40 at 8K: 64 heads x 1024 MAC per
selected position, in FP64 (the FP32 form fails the gate).

Rates per node (47 cores, 2.0 GHz):
- **peaks:** FP32 3.0 TMAC/s, int8 6.0 TMAC/s, FP64 1.5 TMAC/s;
- **best measured kernels:** `gf4_tile_many2` 14.5 MAC per cycle (1.36 TMAC/s); W4A8 GEMM 1.62 TMAC/s (3.24 TOPS,
  section 3); attention at an assumed 30 % of FP64 peak. Head-split matrices carry the 6/5.33 heads imbalance.

| phase | measured (2K, pipe) | kernel roofline | peak roofline |
|---|---:|---:|---:|
| routed experts | 1.07 | 0.44 | 0.12 |
| KDA projections | 0.75 | 0.32 | 0.13 |
| MLA attention | 0.73 | 0.15 (8K: 0.26) | 0.05 |
| mHC sites | 0.51 | ~0.03 (35 M MAC per token, every rank) | |
| router + shared expert | 0.45 | 0.08 | 0.03 |
| comm waits | 0.36 | hidden (1.51 MB per token = 0.50 ms of all-reduce, under the compute) | |
| MLA projections | 0.32 | 0.09 | 0.04 |
| KDA conv / recurrence / gates | 0.25 | ~0.02 | |
| draft block | 0.24 | 0.03 | 0.01 |
| dense FFN, head, other | ~0.1 | 0.03 | 0.01 |
| **per prompt token** | **4.86 ms (206 tok/s from main_s)** | **1.13 ms (885 tok/s; 8K 806)** | 0.38 ms (2600) |

Weight streaming does not bind: a node's 15 GB per pass at ~800 GB/s is 0.07 ms per prompt token at c256. The
400 tok/s target (2.5 ms) is 2.2x the kernel roofline. The gaps are in overheads, not in the kernels:
- mHC (15x), the draft block (8x), router + shared (5.6x) and attention (4.9x) are far from their kernels;
- the experts and the projections are at 2.3-3.4x.

#### 6.2.3 Prefill chunk sweep (2026-10-27)

Current build, prefill tok/s and rank-0 ms per prompt token; the generated text is the same at every chunk (2K
md5 da2742e4, 8K f2f13c32):

| prompt, chunk | tok/s | experts | kda_proj | mhc.site | mla_attn | router | comm.wait |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2K, 256 | 262.6 | 0.981 | 0.566 | 0.322 | 0.445 | 0.332 | 0.098 |
| 2K, 512 | 270.1 | 0.923 | 0.519 | 0.339 | 0.448 | 0.314 | 0.043 |
| 2K, 1024 (default) | **271.3** | 0.823 | 0.541 | 0.310 | 0.415 | 0.295 | 0.189 |
| 2K, 2048 | 251.2 | 0.869 | 0.661 | 0.504 | 0.396 | 0.312 | 0.082 |
| 8K, 512 | 245.6 | 0.841 | 0.517 | 0.281 | 1.075 | 0.312 | 0.017 |
| 8K, 1024 (default) | **252.4** | 0.795 | 0.539 | 0.237 | 1.045 | 0.311 | 0.026 |
| 8K, 2048 | 237.6 | 0.817 | 0.661 | 0.267 | 1.061 | 0.315 | 0.014 |

The default chunk of 1024 stays the best. Going from 256 to 1024 positions per pass, there are 4x the tokens per
expert slice, but the expert time falls by only 16 %. A slice with many tokens still runs `gf4_w4a8` once per
4 tokens, and each run decodes the tile again; the expert GEMM of the plan's step 2 (decode once, a register block
across tokens) is what would turn more tokens per slice into speed. At 2048, the KDA projections (`kdap.p1`
0.41 -> 0.53) and the 2K mHC sites (0.31 -> 0.50) slow down, presumably from the larger per-pass working set.

#### 6.2.4 Expert GEMM for prefill: gf4_w4a8_gemm (2026-10-27)

`gf4_w4a8_gemm` (`src/kernels/cpu/glm_fp4_sve.c`) puts the tokens in the lanes:
- 16 tokens per vector, and a row's 4 weight bytes broadcast (`ld1rw`) against them, 4 rows x 2 token vectors per
  register block;
- each tile decoded once per slab of 8 superblocks for up to 32 tokens (the old path decoded it again for every
  4 tokens);
- the activations of a call transposed once to [block][4-column chunk][16 tokens x 4 bytes] with four rounds of
  zips, and the outputs transposed back the same way;
- a remainder of 1-8 tokens (and calls of up to 8 tokens) through `gf4_w4a8`.

Per (row, token) the integer dots, scale products and fused multiply-adds are `w4a8_tile`'s in its order, so it
is bitwise: 66 cases under qemu and on A64FX (`priv/a64fx-fp4/w4g_test.c`, suite `w4g`), and the full suite's
md5s and dumps unchanged. It is on by default in `tile_kernel`'s prefill branch; `STRATA_RUNNER_W4A8_GEMM=0` turns it
off, `STRATA_RUNNER_W4A8_GEMM_MIN` (8) sets the smallest token count.

One core, MAC per cycle (group 64; group 16 is 10-25 % lower):

| shape | tokens | 4-token path | gemm, all tiles in one call | gemm, one call per tile |
|---|---:|---:|---:|---:|
| gate/up 192 x 4096 | 16 | 17.6 | 27.8 | 20.4 |
| | 32 | 17.8 | 37.0 | 25.2 |
| down 4096 x 192 | 16 | 13.7 | 19.1 | 14.4 |
| | 32 | 13.7 | 24.1 | 17.8 |

The first version wrote its outputs with scalar stores; at the down projection's ldy of 4096 floats they all fell in
one L1 set, the down shape ran at 3.7 MAC per cycle, and prefill at 226 tok/s. Vector stores after the transpose
fixed that.

In the runner (2K, chunk 1024): experts 0.823 -> 0.807 ms per prompt token, prefill 271.3 -> 272.1 tok/s; full
suite 2K 274.9, 8K 255.5 tok/s. Down went 0.347 -> 0.253, up stayed 0.341 -> 0.338. The CMG split gives a thread
about one tile of a slice's gate or up matrix per call (12 tiles over 47 threads), so every call transposes all
of its tokens' activations for a single tile (the "one call per tile" column). Next step: transpose each expert
use's activations once per pass, shared by the team (gate and up of a slice share them; down's after the SwiGLU
blocks), and pass the transposed blocks to the kernel.

**Shared transposition** (`gf4_w4a8_gemm_prep` / `gf4_w4a8_gemm_t`, `STRATA_RUNNER_W4A8_XT`, default on): the plan's
per-use loop transposes the use's hidden-token activations once for its gate and up jobs; a pass after the SwiGLU
blocks does the same for down (`experts.xt_down`); `gemv_share_cmg` hands the blocks to the kernel. One core, one call
per tile: 37.8 MAC per cycle on prepared blocks at 32 tokens (25.9 transposing per call). In the runner (2K, chunk
1024; per-use growing buffers, the first version): up 0.338 -> 0.251, down 0.253 -> 0.249, xt_down 0.005, but
`experts.plan_jobs` 0.037 -> 0.273 and prefill 253.3 tok/s (md5 unchanged). The full suite passed on that version
(all md5s and dumps; 2K 250.4, 8K 251.0 tok/s). The second version uses one grow-only buffer for all uses, sized in
the plan: `plan_jobs` back to 0.044, up 0.236, down 0.229, experts 0.704 ms per prompt token (from 0.823 with the
4-token kernel), **prefill 2K 280.2 tok/s** (271.3 before this kernel), md5 da2742e4. Full suite PASS on this build (all md5s and dumps; 2K 280.4, 8K 259.4 tok/s,
decode b 10.43 ms).

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
4. **Experts:** a W4A8 GEMM that wins with ~28+ tokens per slice. Done 2026-10-27: `gf4_w4a8_gemm` with shared
   transposed activations (section 6.2.4); the earlier decode-once `gf4_w4a8_many` (slower) is removed.
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
| `STRATA_RUNNER_ATTN_F32=1` | off | fp32 MLA scores/averages (approximate) |
| `STRATA_RUNNER_W4A8_GEMM=0`, `STRATA_RUNNER_W4A8_GEMM_MIN`, `STRATA_RUNNER_W4A8_XT=0` | on, 8, on | prefill W4A8 experts through the 4-token kernel; smallest token count for `gf4_w4a8_gemm`; per-call instead of shared activation transposition (all bitwise) |
| `STRATA_RUNNER_ROT_SPREAD`, `GF4_FLOAT_LANE=1` | 1, off | shared-expert tile rotation away from the router; the indexed exact float kernels |
| `--contexts C` | 1 | C resident sequences (eval: segments; `--generate`: batched decode) |
| `STRATA_RUNNER_PREFILL_TIMERS=1`, `STRATA_BENCH_NT=N` | off | the prefill's phase timers per prompt token; `--bench-gemv` for N tokens per job |
| `STRATA_RUNNER_TIMERS_ALL=1`, `STRATA_RUNNER_NUMA_REPORT=1`, `STRATA_RUNNER_LOAD_TRACE=1`, `STRATA_RUNNER_SITE_TRACE=1` | off | per-rank timers, resident GiB per NUMA node, per-rank load progress, per-thread lateness in the mHC sites |

Images live in a node-local directory (`--image-dir`); a rank whose image write fails (a full disk) makes the
others rebuild on the next run, since every rank replays or none does.
