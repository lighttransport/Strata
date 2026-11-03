# MiniMax M3.1 Flash: decode projection for one V100

**Two kinds of numbers live here, and they are labelled.** The projection sections (from "What the projection uses"
to "What this says about reaching 25+ tok/s") are a model: byte counts are exact (read from the GGUF headers), hardware
constants were measured on this class of node with GLM, and three inputs are assumptions that the tables sweep. The
measured sections ("What was measured", "Measured: the preloaded expert pack" and "Measured: the decode path, step by
step") are runs of `strata-m3` on one Tesla V100 node. The projection predicted 12.3-19.6 tok/s for plain decode with
CPU-lane experts; the measured path reached 16.0 (15.8-16.0 over five fresh runs, 210-token prompt), inside that range.

## What was measured: the first smoke run

`strata-m3` (`src/minimax/m3_engine.cu`), UD-Q2_K_XL (4 shards, 143.9 GB), one Tesla V100 PCIe 32 GB, 72 CPUs and
181 GB of RAM in the allocation, CUDA 12.9.86, 8192 context, greedy, one run:

| | |
|---|---|
| Dense weights on the GPU | 9.35 GiB, loaded in 38 s |
| Expert slots in VRAM | 656 (23.3 MiB each, the largest expert), 14.96 GiB |
| Prompt, 210 tokens | 179 s (1.17 tok/s) |
| Decode, 64 tokens | 1823 s (**0.035 tok/s**, 28.5 s per token) |
| Expert copies, whole run | 13493 misses, 5954 hits, 230.5 GiB copied in 1997 s of host time (0.115 GiB/s) |

Tokenization matched llama.cpp on both smoke prompts (210 and 6960 tokens). The MSA kernels match the host
reference on the V100. Token parity with llama.cpp was not obtained: the reference server loaded only after a patch
and then produced nothing in 75 minutes.

Two numbers from the counters are used below. The prompt copied 5083 experts, so decode had 13493 − 5083 = 8410
misses over 63 steps × 57 layers × 4 experts = 14364 accesses: a **decode hit rate of 41.5 % at 656 slots (9.0 % of
the 7296 experts)**. That is one prompt, 64 tokens, an LRU of mixed-size slots, and it counts experts, not bytes.
The copy rate of 0.115 GiB/s is about a hundredth of PCIe's 12 GB/s, so the current "copy a missed expert from
the mapped file" path is not PCIe-bound. A direct measurement on the same node, after the run, points at the file
system (same V100 node, `dd` with `iflag=direct`, 16 MiB blocks, 144 GB model in four shards on shared storage):

| Cold read | GB/s |
|---|---|
| 1 stream, 1 GiB | 0.40 and 0.43 (two offsets) |
| 8 parallel streams, 4 GiB total | 0.42 |
| 32 parallel streams, 16 GiB total | 0.42 |

Host-to-device copies on the same V100 (`cudaMemcpyAsync`, 20 repetitions, 1-256 MiB): pinned 11.9-12.3 GB/s,
pageable 4.8-5.5 GB/s. Pageable staging alone therefore does not explain 0.115 GiB/s.

Cold reads reach about 0.42 GB/s on this node however many streams read, and the page cache held only 21 GB of the
144 GB model when checked (`Cached` in `/proc/meminfo`), so most experts were not in RAM. The memory-mapped path got
0.12 GB/s, about a third of what sequential reads give. This is consistent with the copy rate being the file system,
but it is two short probes on one node, taken after the runs; it has not been shown that every slow copy was a cold
read. Reading the whole 134 GB expert pack once into process memory at 0.42 GB/s would take about 5.3 minutes.

## What the projection uses

`tools/sim` is the GLM decode simulator; `minimax_m31` is a new model entry in it. Sources of every input:

| Input | Value | Source |
|---|---|---|
| Layers, experts, top-k, rows | 60 layers, 3 dense, 128 experts, top-4, 1 shared expert, 3072 rows | GGUF metadata |
| Routed expert size | 18.14 MB (layers 3-58), 24.48 MB (layer 59); 4.183 GB per token | GGUF tensor census, exact |
| Dense bytes read per token | 9.200 GB (attention, F32 indexer, shared experts, 3 dense FFNs, head) | GGUF tensor census, exact |
| Dense weights resident | 9579 MiB | census; the engine loaded 9.35 GiB |
| H2D (pinned), device memory | 12.3 GB/s, 576 GB/s | measured on the GLM V100 benchmark node |
| CPU expert flow | 81.25 GB/s, swept 50-120 | measured with GLM Q2 experts on that node; includes the arithmetic. Plain DRAM bandwidth and RAM speed were not measured |
| GPU GEMV efficiency, resident-expert kernel | 0.80 of 576 GB/s, 250 GB/s | **assumed** |
| GPU launch and attention costs | 44 kernels per layer, 40 µs attention per layer | **fitted on GLM**, not on MiniMax |
| Window union of experts for speculation | GLM's measured 1.80x for 2 tokens | **GLM value**; M3.1 has 128 experts, so it probably grows faster |
| VRAM tier hit rate | uniform 16 %, GLM curve 54 %, GLM curve scaled to the measured 41.5 % point: 56.5 % | **assumed**; one measured point |
| DSpark draft head | 2.0 GB Q8_0 read per round, about 4.9 ms; acceptance 0.5-0.85 | **assumed**; no acceptance measured |

The tier holds 1165 experts (16 % of 7296) in the projection because it assumes an engine built for decode (small
prompt chunk, FP16 K/V). The measured engine kept F32 K/V and 8192-token activation buffers, which left 656.

## Projection (tok/s)

Full output: `docs/benchmarks/minimax_m31_sim_20261101/projection.txt`. All rows use CPU-lane experts: experts that are
not in VRAM run on the CPU from RAM, overlapped with the GPU.

**Plain decode** (no speculation, next-layer PCIe prefetch on):

| GPU chain assumption | uniform routing (16 % hits) | curve-scaled (56.5 % hits) |
|---|---|---|
| as fitted on GLM | 12.3 | 16.0 |
| graph-captured chain (25 kernels per layer, half the exposed launch cost) | 13.9 | 18.9 |
| plus GEMV at 90 % of 576 GB/s and a 400 GB/s expert kernel | 14.4 | 19.6 |

At the last row the CPU side takes 26.6 ms and the GPU side 30.6 ms per token, yet a token takes about 51 ms, because
the attention and dense chain of each layer must finish before that layer's router runs; only the shared expert and
the resident experts overlap the CPU. A faster CPU therefore stops helping early: at 56.5 % hits the last row
plateaus at 23.3 tok/s for any CPU flow from 200 to 1000 GB/s (19.6 at 81 GB/s). The hit rate is the larger lever:
at an 80 % hit rate the same row gives 24.8 tok/s at 81 GB/s and 26.9 at 200 GB/s or more. **At the hit rates
assumed here (at most 56.5 %), plain decode stays below 25 tok/s under all three GPU-chain assumptions.** Whether
routing can reach an 80 % hit rate with 16 % of the experts resident is unknown; the one measured point is 41.5 % at 9 %.

**With the DSpark draft head** (last GPU-chain row; n = draft tokens per round, p = per-position acceptance):

| | p = 0.50 | p = 0.70 | p = 0.85 |
|---|---|---|---|
| 56.5 % hits, n = 1 | 19.0 | 21.5 | 23.4 |
| 56.5 % hits, n = 2 | 17.4 | 21.8 | **25.6** |
| 56.5 % hits, n = 3 | 15.1 | 20.5 | **25.7** |
| 16 % hits, n = 2 | 11.7 | 14.6 | 17.2 |

With n = 2 and 56.5 % hits the projection reaches 25 tok/s at a per-position acceptance of about 0.83. With uniform
routing it does not reach it at any acceptance. A draft that is accepted half the time is slower than no draft at all.

**UD-IQ1_M pack** (121 GB file, 3.74 GB of experts per token, 1317 slots), last GPU-chain row: 20.6 tok/s plain; with
two draft tokens 18.7 / 23.4 / 27.5 at acceptance 0.5 / 0.7 / 0.85.

## What this says about reaching 25+ tok/s

1. Streaming experts over PCIe cannot do it: 4.18 GB per token at 12.3 GB/s is 340 ms.
2. The first thing to fix is the copy path (0.115 GiB/s, 0.035 tok/s). Every row above assumes experts are already in
   RAM in a form the CPU kernels and the DMA engine read at full speed. With cold reads capped near 0.42 GB/s, that
   means reading the pack into process memory once at load (about 5 minutes for 134 GB) instead of faulting it in
   during decode, and not relying on the page cache to keep it.
3. Past that, the serial GPU chain (about 29 ms of the 51 ms) limits plain decode, not the CPU. Speculation amortizes
   that chain over several tokens; a smaller pack, a higher hit rate and a cheaper launch path each add a few tok/s.
4. The projection reaches 25 tok/s only if favourable assumptions hold together: either speculation with
   acceptance of about 0.83 or better at the curve-scaled hit rate, or an unusually high hit rate (about 80 %) without
   speculation, and in both cases the optimistic GPU chain. Treat 25 tok/s as a target this projection says is
   possible but not safe.

## Measured: the preloaded expert pack

`strata-m3 --preload-experts` (or `STRATA_M3_PRELOAD=1`; `--preload-threads N`, `--no-pin`) reads all routed experts
once at load into one block of process memory with large sequential `pread`s, drops each range from the page cache,
page-locks the block layer by layer, and serves a VRAM slot miss as one host-to-device copy from it. Without the flag
the engine is unchanged. `STRATA_M3_PRELOAD_CHECK=1` byte-compares 48 random experts of the pack with the mapped file.
It needs the pack (124.7 GiB) plus 12 GiB of free memory, and falls back to the mapped file with a message otherwise.

Same short smoke prompt, same V100 node class, same greedy settings, UD-Q2_K_XL, 8192 context, one run each (the
preloaded run had 646 expert slots instead of 656: 16.23 GiB of VRAM was free after load against 16.47 GiB, cause not isolated):

| | mapped file | preloaded pack |
|---|---|---|
| Load: pack read | none | 124.7 GiB in 135 s (0.99 GB/s, 16 threads), page-locked in 14 s |
| Prompt, 210 tokens | 179 s (1.17 tok/s) | 10.8 s (19.5 tok/s) |
| Decode, 64 tokens | 1823 s (0.035 tok/s) | 16.5 s (**3.83 tok/s**, 261 ms per token) |
| Greedy tokens | | all 64 identical to the mapped-file run |
| Pack bytes | | 48 random experts, 0 mismatching parts |
| Process memory | | 126 GiB peak resident |

The pack reads at about 1.0 GB/s here, more than twice the 0.42 GB/s the direct-read probe measured on the earlier
node, so that probe was not a hard limit for sequential buffered reads; the two numbers come from different nodes and
different times, and the cap's cause is unknown. The first logit of the preloaded run differs from the mapped run in
the second decimal (30.25 vs 30.31) with identical tokens; the cause is not isolated (the slot count differed, which
should not change arithmetic).

Decode is now bound by PCIe, as the arithmetic predicted: 133 misses per token x 18.4 MB is 2.4 GB, 200 ms at 12.3 GB/s,
against 261 ms measured. That is still far from 25 tok/s. Getting further needs the CPU-lane experts of the projection
rows above (misses computed from the pinned pack on the CPU, overlapped with the GPU), which are not implemented.

## Measured: the decode path, step by step

One V100 node (Tesla V100 PCIe 32 GB, 2x Xeon Gold 6240 with 4 NUMA nodes, 72 CPUs and 181 GB in the allocation, CUDA
12.9.86), UD-Q2_K_XL, the 210-token smoke prompt, 64 greedy tokens, one process per row (the rows with "runs" repeat the
request inside one process). Single sequence, no speculation, same weights and routing throughout. RAM speed was not
measured. Every row's 64 tokens were compared with the first mapped-file run; all matched except the 12.4-12.6 row, which
stayed identical for 32 tokens and then diverged (see the end of this section).

| Step (each includes the ones above) | decode tok/s | what it did |
|---|---|---|
| expert misses read from the mapped GGUF | 0.035 | page faults at ~0.12 GB/s |
| `--preload-experts`: pack in pinned RAM, one H2D copy per miss | 3.83 | PCIe-bound: ~133 misses x 18 MB per token |
| `--cpu-experts`: misses computed on the CPU from the pack, overlapped with the GPU | 6.8-7.9 | 35 threads, ggml-cpu dots, the pack placed by first touch |
| pack interleaved over the 4 NUMA nodes (`mbind`) | 11.0 | CPU lane 103 -> 44 ms/token; the benchmark measured 31 -> 65 GB/s |
| two VRAM slot sizes (1019 slots of 17.3 MiB instead of 761 of 23.3), AVX warp-per-slot MSA kernels | 11.7 | hit rate 38.8 -> 45.7 %; MSA 14.1 -> 5.5 ms/token on the GPU |
| router first (shared expert overlaps the CPU lane), pinned router staging, GPU argmax, indexer on a second stream, K/V sized to the request, embedding table in pinned RAM, 0.5 GiB reserve | 12.4-12.6 | 1101 slots |
| frequency-gated admission (up to 2 misses per layer enter VRAM, only if used at least as often as the slot's owner) | 13.1-14.3 | hit 52.9 -> 54 %, admissions 51 -> 28 per token |
| spin-wait stream sync, cache warmed from the prompt's own routing, never waiting on an in-flight admission copy | 14.6-15.9 (15.33 fresh, 14.56 and 15.86 on repeats; earlier builds of the same design 14.7-15.8) | the blocking sync wake-up cost ~250 us x 57 syncs, ~14 ms/token |
| CPU-lane workers pinned to the 35 physical cores (default; `STRATA_M3_PIN=0` unpins), large-expert slot pool 24 -> 8 | 15.2-15.7 mean over five runs (15.72 pinned, 15.22 unpinned, same node) | less straggling at the lane's barriers |
| the CPU lane's result and the step's token ids go through mapped pinned memory that the kernels read over PCIe instead of `cudaMemcpyAsync` | **16.02, 16.00, 15.79, 16.01, 16.01** (mean 15.97, min 15.79; five fresh runs) | on this GPU all H2D copies share one copy engine in issue order, so a 24 KB result queued behind a 17 MB admission copy waited up to 1.4 ms per layer; the run-to-run spread of the GPU-chain wait fell from 1060-1615 ms to 1121-1162 ms |

Run-to-run spread on the same configuration was about 0.4-0.6 tok/s until the last row, where it fell to 0.2. The warm-up copies ~940 experts in 1.4 s at the end
of the prefill; that time is not in the prompt figures above.

Where a token went (a profiled build before the pinning and mapped-copy changes, 72.5 ms): waiting for the GPU chain up to each layer's router 28 ms
(GPU timeline 24 ms: q/k/v matvecs 8.0, MSA 5.0, o-proj 4.4, router 2.4, rope/kv 1.4, norms 2.3), the CPU lane 34 ms
(gate/up 20, down 12; ~0.31 ms per expert, at the benchmark's floor), attention enqueue on the host 3 ms, head 1 ms.

**Long context.** The same engine on the 6960-token smoke prompt (55 blocks of 128, 16 selected per layer; a planted
code word that the answer must quote), 8192-token capacity, `--chunk 512`, 47 greedy tokens (the model stops there):

| build | decode tok/s | notes |
|---|---|---|
| first MSA kernels (before the rework) | 10.9-11.1 | the sparse-attention kernel took 410-450 us per layer call |
| reworked MSA kernels (1024-thread blocks, 4 slots in flight per warp, warp-level block selection) | 13.1 | 182 us per layer call; the 47 tokens, which quote the planted code word correctly, are identical to the first mapped-file run |
| + split-K sparse attention (a thread block per selected block, partial-softmax merge) | 13.5 | 147 us per layer call; the same 47 tokens |
| + pinned workers and the mapped copies (the final tree) | **14.0** | the same 47 tokens; the mapped copies did not change the long-context figure (13.996 against 14.021 before) |

Prefill at that length is slow with `--chunk 512`: 230 s (30 tok/s), because every 512-token chunk routes to nearly all
128 experts of every layer, so the whole 125 GiB pack crosses PCIe about 14 times (1.4 TiB copied). A larger chunk
needs more VRAM for activations and was not measured here.

MSA kernel microbenchmark (`minimax_msa_bench`, 1 query, 64 query heads over 4 kv heads, 128-wide, top-16 of the blocks;
CUDA events, 200 calls, microseconds per layer call; the host reference test `minimax_msa_parity` passes with all of them):

| cached positions | block scores | block selection | sparse attention | total |
|---|---|---|---|---|
| 270 (3 blocks) before -> after | 17.5 -> 17.4 | 8.9 -> 6.8 | 58.0 -> 28.4 | 84.4 -> 52.6 |
| 2048 (16 blocks) before -> after | 21.5 -> 21.4 | 41.2 -> 22.3 | 449.6 -> 102.8 | 512.3 -> 146.5 |
| 7000 (55 blocks) before -> after | 22.3 -> 23.5 | 43.9 -> 25.5 | 408.7 -> 98.2 | 475.0 -> 147.2 |

The sparse-attention row is the split-K version: each selected block of keys is its own thread block with a local
softmax, and a merge kernel combines the partials (133 us with one block per query head, 98 us split). Three attempts
were measured and dropped: unrolling 8 slots per warp (164 us, register spills); a block per (query, kv head) that
serves all 16 query heads of a group from one K/V read (2100 us: only 4 blocks on an 80-SM GPU); and the same grouping
combined with the split (94 us at 7000 positions but 85 us instead of 29 at 270: 16 warp reductions per key and 64
accumulator registers outweigh the traffic saved).

Pinning and the AVX2 row kernel, five runs each on one node (tok/s): unpinned 14.93, 14.65, 15.44, 15.28, 15.81; pinned
15.75, 15.74, 15.98, 15.74, 15.32; AVX2 rows 15.10, 15.40, 14.20, 15.89, 15.68; both 15.14, 15.28, 15.24, 16.02, 14.34.
The AVX2 rows (`STRATA_M3_IQ256=1`, matching the ggml path to 2e-6) gave no gain and stay opt-in.

What was tried and did not help: the AVX-512 i-quant kernel (47 GB/s against ggml's 63 on this CPU), the AVX2
multi-token kernel (+5-8 %, not adopted), more than 70 CPU threads (no gain past ~16-35), admitting every miss
(more hits but the compute stream waited on copy events: 13.1 tok/s at a 56 % hit rate before the wait was removed).
`nsys` crashed on this process with and without a local temp directory, so GPU timings come from CUDA events.

**The lossless bar used here.** It was set explicitly: equivalence at quantization-noise level, measured as teacher-forced
top-1 agreement and KL against the all-GPU path, not bit-identical free-running text. Bit-identical output in every cache
state would need canonical kernels (the CPU and GPU lanes rounding identically for IQ2_XS, IQ3_XXS and IQ4_XS, with a
matching `expf`); that was judged out of scope. **How "lossless" holds here.** Nothing approximate is used: no skipped or substituted experts, no reduced-precision
weights, no draft model. The CPU lane and the GPU lane evaluate the same quantized weights but quantize the activation
differently (Q8_K on the CPU, q8_1 on the GPU), as llama.cpp's two backends do, so results are not bit-identical across
cache states. Teacher-forced over the 64 baseline tokens against the all-GPU path: top-1 agreed on 63 of 64 steps, mean
KL 7.6e-4, max |dlogit| 1.05 (the single disagreement is a step whose top-1/top-2 logits differ by 0.17); four builds of the final
design agree on all 64 steps (mean KL 5.6e-4, 9.9e-4, 5.9e-4 and 6.1e-4, max 7.8e-3 down to 4.5e-3 and up to 1.7e-2,
mean of the per-step max |dlogit| 0.40-0.42, max 1.19-1.25). Two all-GPU
runs differed from each other by KL 9.1e-5 and max |dlogit| 0.79. Greedy text can therefore diverge at a near-tie: the 12.4-12.6 row, and a free-running repeat of the final build, generated "write a function" where the baseline had
"provide the function" at token 32.

## What is not predicted, and what is still owed

- The projection is not refit to the measured path. Its GPU attention chain and hit rates were assumptions; the measured
  path now gives the real ones (a per-layer GPU timeline above, hit rates of 46-54 % at 1100-1200 slots, and a routing
  trace from `STRATA_M3_ROUTE_TRACE` that `tools/minimax_cache_policy_sim.py` replays). Feeding them back into
  `tools/sim` is not done.
- Long context (6960 tokens, 14.0 tok/s) was accepted as the best-effort result. Not tried: fusing block scores + selection,
  128-thread split blocks, a cheaper grouped K/V read, a CUDA graph for the attention chain (estimated +0.4 to +1.0 tok/s),
  and an FP16 K/V cache (estimated +0.7 to +1.2 tok/s, but a rounding step beyond the lossless bar that was set).
- 25 tok/s was a projection target and is not reached: the measured path is 16.0 tok/s (15.8-16.0) at a 210-token
  prompt and 14.0 at 6960 tokens, below 15 at that length: the remaining ~9 ms per token against the short prompt is
  147 us of MSA kernels per layer (against 53 us), the K/V caches taking VRAM from expert slots (hit rate 46 % against
  49-54 %), and the larger indexer-key scan, plain (non-speculative) decode. The DSpark draft head is not implemented; no acceptance rate or
  draft-head timing has been measured on MiniMax.
- One prompt (plus the long one) was used for the decode numbers, 64 tokens each, on one node. The cache policies were
  tuned on a single 63-step routing trace; the frequency gate's gain over LRU may differ on other text.
- Not measured on this node: host memory bandwidth in isolation, RAM speed, and whether the file-system read rate is per
  node or shared with other jobs (0.42 GB/s in one probe, 1.0 GB/s while loading the pack on another node).
- Prefill is not optimized or projected: 20 tok/s at 210 tokens and 30 tok/s at 6960 with `--chunk 512`, plus
  135-150 s to load the pack and ~1.4 s to warm the cache from the prompt's routing.
- Free-running greedy text of the 12.4-12.6 tok/s configuration and of one final-build run diverged from the baseline at
  token 32 (a 0.17-logit near-tie); the other configurations matched all 64 tokens. Greedy output is not guaranteed
  identical across cache states (see "How lossless holds here").
- Lever-by-lever forecasts in this simulator have been off by several tok/s in both directions on GLM
  (`docs/GLM_RUNNER_SIM.md`).

## Reproduce

```bash
# the engine (CUDA build; needs ~137 GiB of free RAM for --preload-experts, a GPU with ~32 GB, token ids from the GGUF's tokenizer)
strata-m3 --model FIRST_SHARD.gguf --tokens-file prompt.ids --max-new 64 --max-context 8192 --chunk 512 \
    --preload-experts --cpu-experts          # STRATA_M3_PROF=2 prints the per-part timeline, STRATA_M3_ROUTE_TRACE=f the routes
python3 tools/minimax_m31_sim.py                       # the tables above
python3 tools/test_sim_minimax.py                      # census bytes, invariants (no GPU, no model)
build-dir/m3_cpu_expert_test                           # the CPU expert path against ggml's dequantizers (no GPU, no model)
python3 tools/minimax_cache_policy_sim.py trace.txt    # cache admission policies on a STRATA_M3_ROUTE_TRACE file
# one configuration through the CLI; it uses the GLM-fitted hit curve, so it equals the "as fitted on GLM" row for that curve (15.75 tok/s)
python3 tools/sim/glm_sim.py --model minimax_m31 estimate --hw tools/sim/data/hw_v100_m31_node.json \
    --pack minimax_m31_ud_q2_k_xl --speculation none --mtp 0 --context 8192 --placement cpu --threads 35 \
    --tier-policy lru --pcie-prefetch 1 --prefill-scratch-mib 512 --split-verify 0 --decode-cache-mib -1
# new census from other GGUF files (headers only):
python3 tools/minimax_sim_census.py FIRST_SHARD.gguf --name NAME --output tools/sim/data/minimax-m31-census-NAME.json
```

## Speculative decode and lossy levers (measured, one V100, UD-Q2_K_XL, 124.7 GiB pack in RAM)

Single runs unless noted; short prompt 210 tokens x 64 new, "code" prompt 487 tokens x 200 new. Quality is teacher-forced
against the unmodified path (64 steps). The K/V cache in FP16 (`--kv-f16`) gave KL 8.0e-4 and top-1 64/64, with no speed change at
short context. The window (verify) path matches single-token decode: KL 5.2e-4, top-1 64/64.

| Setting | Decode tok/s | Mean KL | Top-1 |
|---|---|---|---|
| none (short / code prompt) | 15.96 / 16.4 | - | - |
| top-3 router | 18.2 | 3.9e-3 | 62/64 |
| drop experts below 0.25 x max weight | 17.0 | 3.7e-3 | 63/64 |
| top-3 + 0.25 threshold (short / code) | 18.9 / 20.1 | - | - |
| top-2 (code) | 23.1 | 3.3e-2 | 60/64 |
| top-2 + 0.4 threshold (code) | 24.0 | 3.2e-2 | 60/64 |
| top-1 (code) | 29.5 | 1.1e-1 | 56/64 |
| renormalized weights | worse KL than not renormalizing | | |

Long prompt (6960 tokens, FP16 K/V): 14.2 plain, 16.2 with top-3, 16.5 with top-3 + 0.25 threshold.

N-gram (prompt lookup) speculation does not pay: 15-22 % of drafts accepted (1.2-1.3 tokens per step, also on a code-edit
prompt) and a 3-row verify step costs about 93 ms against about 62 ms per plain token, so it runs at 10.7-12.8 tok/s.
The GPU chain is 21.3 ms per step (CUDA-event timeline, top-2); the rest of a step is the CPU expert lane.
Prefill is bound by expert copies over PCIe: 30.6 tok/s at 6960 tokens, about 21 at 210.

### DSpark draft head (`--dspark FILE --spec dspark --draft D`), measured

The released DSpark head (5 layers, block 7, Q8_0, 2.1 GB on the GPU) is ported in `src/minimax/m3_engine.cu`: the target's
hidden states at the inputs of layers 4, 18, 32, 46, 60 are fused (`fc` + RMS norm) into the head's K/V cache for every
committed token; a draft is one pass over the block [last token, mask ...], the target's lm_head, then the Markov bias
chained by argmax. Verification is the exact window path, greedy accept. Same node, FP16 K/V, single runs, 200 new tokens
(120 for the first two rows of the code prompt).

| Prompt | Routing | Draft tokens | Accepted | Tokens/step | Decode tok/s | Without head |
|---|---|---|---|---|---|---|
| code, 487 tokens | exact (top-4) | 2 | 70 % | 2.41 | 18.4 | 16.2 |
| code | top-3, 0.25 threshold | 3 | 63 % | 2.90 | 23.1 | 20.1 |
| code | top-2, 0.4 threshold | 3 | 63 % | 2.90 | 30.1 | 23.9 |
| short, 210 tokens | exact | 2 | 83 % | 2.67 | 18.5 | 16.0 |
| short | top-3, 0.25 threshold | 3 | 63 % | 2.90 | 21.7 | 18.9 |
| short | top-2, 0.4 threshold | 3 | 73 % | 3.17 | 31.6 | 23.4 |

Top-1 routing with 2 drafts reached 39.9 tok/s (code, 120 tokens, 83 % accepted). The speculative output equals the greedy
output of the same (lossy) routing: the quality of each row is that of its routing setting (top-3 + 0.25: KL 4e-3; top-2
+ 0.4: KL 3.2e-2, top-1 agreement 60/64 on the short prompt). Not measured: the long (6960-token) prompt with the head,
sampling (temperature > 0), and acceptance on prompts other than these two.

Long prompt (6960 tokens, FP16 K/V, the head loaded; the answer ends at 47 tokens, so 15-17 verify steps per row and the figures
are noisy): top-2 + 0.4 threshold 20.5 plain, 23.9 with 3 drafts (71 % accepted, 3.13 tokens/step); top-3 + 0.25 threshold 19.5
with 3 drafts; exact routing 17.4 with 2 drafts (88 % accepted, 2.76 tokens/step).

## Prefill with FP16 tensor-core GEMMs (`--pf-gemm`, opt-in), measured

6960-token prompt, one V100, UD-Q2_K_XL pack in RAM, single runs (the prompt's routing is exact in all rows):

| Path | Chunk | Prefill tok/s | Time | Expert copies |
|---|---|---|---|---|
| q8_1 column-group matvecs (default) | 512 | 30.6 | 227 s | 83,029 (1420 GiB) |
| FP16 GEMM | 512 | 35.1 | 198 s | 82,990 |
| FP16 GEMM | 2048 | 73.3 | 95 s | 26,035 (445 GiB) |
| FP16 GEMM | 4096 | 102.3 | 68 s | 13,503 (231 GiB) |

Short prompt (210 tokens): 20.4 (default) to 22.3 tok/s (FP16 GEMM): the pack's PCIe traffic dominates. Quality of the GEMM path
(teacher-forced 64 steps against the default path): KL 1.2-1.6e-3, top-1 63-64/64. The larger chunk helps because each chunk
streams almost the whole pack once, so the copy volume per token falls with the chunk.

### Prefill, further steps (all opt-in, measured on the same node and prompt)

A 6960-token prompt in one 7168-token chunk, FP16 K/V, single runs, prefill tok/s (time). `--pf-gemm` is the FP16 GEMM path
above; the rows add one change each, in this order:

| Change | Prefill tok/s | Time |
|---|---|---|
| FP16 GEMM, chunk 4096 (two chunks) | 102 | 68 s |
| one chunk (7168): the pack streams once | 153 | 45 s |
| grouped sparse attention (one block per query and KV head) | 114 at chunk 4096 | 61 s |
| tensor-core sparse attention (WMMA, 16 heads x 128-key blocks) | 253 | 27.5 s |
| `--pf-prefetch`: next layer's experts copied while this layer computes | 384 | 18.1 s |
| staged, coalesced i-quant dequantization | 446-455 | 15.3-15.6 s |
| tensor-core block scores | 480-484 | 14.4-14.5 s |
| one FP16 conversion per shared input | 488-493 | 14.1-14.3 s |

Profile of the 45 s row (device time between syncs): sparse attention 21.5 s, expert copy waits 10.5 s, expert GEMMs 9.2 s,
block scores 1.6 s, other attention 1.7 s. In the 18 s row cuBLAS runs at about 70 TFLOP/s (385 TFLOP in 5.4 s); the dequantization
was 4.6 s before the staged kernel and 1.7 s after. Quality against the default path (teacher-forced 64 steps, short prompt): KL
1.1-1.3e-3, top-1 63-64/64 for every row, which is the level of FP16 K/V alone. Short prompts stay bound by streaming the pack:
210 tokens 22.8 tok/s, 487 tokens 38.6 tok/s. Not done: the 128 experts of the last layer (large slots, 8 of them) still copy
in line, about 0.25 s; prompts longer than the chunk need one pass over the pack per chunk.

## Decode under a quality gate: what the combined path reaches (measured, one V100, 2 x Xeon 6240)

Gate: teacher-forced 64 steps against the exact path, mean KL <= 0.01 and top-1 >= 95 % on **both** the 210-token and the 487-token
prompt, for the final combined configuration (FP16 K/V, DSpark head, routing levers). Speed is 200 new tokens on the same prompts
(short / code), `STRATA_M3_IQ256=1`.

| Routing | Draft | KL short / code | Top-1 short / code | tok/s short / code | Gate |
|---|---|---|---|---|---|
| exact | 2 | - | - | 19.4 / 19.1 | pass |
| top-3 | 2 | 6.0e-3 / 9.1e-3 | 61/64 / 63/64 | 22.8 / 23.4 | pass |
| top-3 | 3 | same | same | 21.3 / 22.6 | pass |
| top-3 + 0.25 threshold | 3 | 8.4e-3 / 1.06e-2 | 62/64 / 63/64 | 22.9 / 23.4 | **fails on code** |
| top-3 + 0.3 threshold | 3 | 6.9e-3 / 1.36e-2 | 62/64 / 63/64 | - | fails |

So the best configuration that meets the gate is **top-3 routing with the DSpark head and 2 drafted tokens: 22.8-23.4 tok/s**. The
earlier single-lever figure of 4e-3 for top-3 was measured on the short prompt only.

Per-layer sensitivity (a single MoE layer pruned to top-2, KL on the short / code prompt, 57 layers): 4e-4 to 4e-3, mostly 1e-3 to
2e-3. Dropping a second expert in one layer costs about 5e-4 KL per expert, against 7e-5 for the first one in every layer (top-3),
so after top-3 only about 4 more layers fit in the 0.01 budget: per-layer pruning cannot reach more than ~28-32 % of experts dropped
under this gate (`lay=` / `layfile=` REPL keys exist for it).

Where a speculative step goes (code prompt, top-3 + 0.25 threshold, draft 3, 70 steps, 8.45 s): 121 ms per step = CPU expert lane
73 ms (290 experts per step, 5.1 per layer, 72 GB/s of expert bytes), waiting for the GPU chain 25 ms, GPU enqueue 6 ms, draft,
inject, head and sync about 17 ms. Hardware limits measured on the node: 35 pinned threads read interleaved RAM at 121 GB/s
(16 threads 47 GB/s); one core decodes + dots i-quant rows at 2.4 GB/s of expert bytes with data in L2 (ggml 0.119 ns per weight,
the AVX2 multi-token kernel 0.127, the AVX-512 kernel 0.185, an AVX-512 variant with vector sign parity and gathered grids 0.234),
so 35 cores cap the lane near 85 GB/s of compute and it already runs at 72. Pinned SMT (70 threads) gave +7 % on a cold six-expert
round. `STRATA_M3_IQ256=1` is worth +2-5 % end to end. Reaching 35 tok/s inside the gate would need the lane at ~120 GB/s (a
decode kernel about 1.5-2x faster than ggml's AVX2 one, which three attempts did not produce) plus the chain and draft overhead
cut by ~15 ms; neither was found in the time of this allocation.

### Residency-aware pruning and cache-aware substitution (measured, same setup; draft 2, `STRATA_M3_IQ256=1`)

Only experts that miss the VRAM cache cost CPU time, so these levers drop only missed experts (`missk=` keeps at most that many missed
experts per token and layer, heaviest first, the heaviest expert of a token is never dropped; `missw=` is a weight floor; `sub=1/2`
replaces a dropped expert by the best resident one among the router's next four ranks, with its own weight (1) or the dropped
expert's weight (2)). KL and top-1 are on the 210-token / 487-token prompts, tok/s on the same prompts, 200 new tokens.

| Setting | Dropped | KL short / code | Top-1 short / code | tok/s short / code | Gate |
|---|---|---|---|---|---|
| top-3 (uniform) | 25 % | 6.0e-3 / 9.1e-3 | 61, 63 /64 | 22.8 / 23.4 | pass |
| missk=2 | 15-19 % | 6.6e-3 / 1.31e-2 | 61, 62 | 22.6 / 22.5 | fails (code) |
| missk=2 + top-3 | 28-30 % | 7.1e-3 / 1.62e-2 | 61, 62 | 25.7 / 26.1 | fails (code) |
| missk=2 + top-3 + sub=1 | 25-28 % | 5.9e-3 / 1.64e-2 | 63, 62 | 25.5 / 23.8 | fails (code) |
| missk=1 | 38-43 % | 2.6e-2 / 4.7e-2 | 60, 59 | 32.1 / 28.2 | fails |
| missk=1 + sub=1 | 25 % | 2.1e-2 / 4.8e-2 | 61, 58 | 32.2 / 29.6 | fails |
| missk=0 (every missed expert dropped) | 55 % | 7.2e-2 / 5.5e-2 | 58, 58 | 38.5 / 31.7 | fails |
| missk=0 + sub=1 | 37 % | 4.4e-2 / 5.9e-2 | 60, 55 | 33.7 / 34.0 | fails |

Targeting the missed experts does not buy quality: a dropped missed expert is heavier than the global lightest one, so each saved
CPU millisecond costs more KL than uniform top-3, and a resident stand-in recovers little of it. The speed/quality curve of routing
levers is monotone: 35 tok/s sits at KL 0.04-0.06, 2.5-6x over the gate.

Step breakdown of the gate-passing configuration (code prompt, top-3, DSpark draft 2, `STRATA_M3_IQ256=1`, 82 verify steps, 2.44
tokens per step, 103 ms per step, 23.6 tok/s): CPU lane 61.8 ms, waiting for the GPU chain 21.9 ms, GPU enqueue 4.3 ms, draft
pass 6.2 ms per call (5 head layers, lm_head, Markov chain; `M3_DSPARK` line), about 9 ms for the head, accept logic and the K/V
inject. The draft pass is already near its weight-read floor (about 2.1 GB at 576 GB/s = 3.6 ms plus the Markov tables).

### Confidence-gated draft length and the cache admission policy (measured; top-3 routing, DSpark, `STRATA_M3_IQ256=1`)

`pmin=` (REPL) drops drafts whose DSpark confidence head (`conf_proj`: sigmoid of the final-normed draft row and the previous
token's Markov row) is below the threshold, so low-confidence drafts add no rows to the verify window; the output is the same
greedy output as without it. tok/s on the code / short prompt, 200 new tokens, single runs:

| max draft, pmin | tokens per step (code / short) | tok/s code | tok/s short |
|---|---|---|---|
| 2, off | 2.44 / 2.53 | 23.3 | 22.9 |
| 4, 0.5 | 2.70 / 3.12 | 21.2 | 23.4 |
| 4, 0.7 | 2.41 / 2.44 | 24.2 | 24.3 |
| 4, 0.8 | 1.79 / 2.08 | 21.5 | 22.9 |
| **6, 0.7** | 2.56 / 3.12 | **24.5** | **25.0** |
| 6, 0.8 | 2.15 / 2.41 | 23.7 | 23.5 |

So the gate-passing best is now **24.5 tok/s (code) and 25.0 (short)**: top-3 routing, `draft=6 pmin=0.7` (KL as for top-3: 6.0e-3 / 9.1e-3).

Cache admission: on a routing trace of 296 decode steps a policy that admits an expert once it has been routed to twice looked
like +6 points of hit rate in the simulator (after excluding dropped `-1` experts, which the simulator had counted as accesses),
but in the engine it lost: admissions per step rise from ~8 to 20-40, copies in flight count as misses, and the hit rate fell
(code prompt, admissions per layer 1 / 2 / 3 / 4: 52.9 / 40.9 / 23.0 / 18.6 %, against 51.6 % for the frequency gate; short prompt
43.7 / 46.2 / 31.7 / 19.7 % against 41.7 %). The frequency gate stays. Offline bounds on the same trace (1100 slots, 15 % of the
experts): the best fixed set chosen with hindsight 61.4 %, a fixed set learned on the first half 47.3 %, a clairvoyant dynamic
cache that admits every miss 78.4 %.

Gate check of the best configuration (top-3 routing, DSpark `draft=6 pmin=0.7`, `STRATA_M3_IQ256=1`), teacher-forced 64 steps against
the exact path, short / code prompt: the 6-row verify window with top-3 routing KL 4.9e-3 / 8.9e-3, top-1 62/64 / 64/64 (the
single-token top-3 path measured 6.0e-3 / 9.8e-3, 61/64 / 63/64 in this run, 9.1e-3 in the earlier one: the code prompt sits just
under 0.01). The confidence cut only changes how many rows are verified, so it does not enter this measurement. Free-running greedy
text of plain top-3 decode against the speculative run agrees for 167 of 200 tokens (short prompt) and 7 of 200 (code prompt) before
the first difference: the window and the single-token paths round differently (CPU Q8_K against GPU q8_1 activations, summation
order), so near-tied tokens flip, as in the exact path (see the lossless bar above).

### Defaults (strata-m3 with `--dspark FILE`)

With the head loaded and no other flags, `strata-m3` now speculates (`--spec dspark --draft 6`, confidence floor 0.7) and runs the
AVX2 multi-token CPU rows (`STRATA_M3_IQ256=0` selects ggml's dots again). `spec=none`, `draft=`, `pmin=` override per REPL request.
Measured with no routing flags, 200 new tokens, code / short prompt: exact routing **19.1 / 21.0 tok/s** (16.4 / 15.4 with
`spec=none`), `topk=3` 24.3 / 24.9 tok/s. Without `--dspark` the engine is as before apart from the CPU rows.

The CPU decode kernel is not the lever either: with the weights resident in L2 (one thread, 64 gate + 64 up rows repeated, `m3_cpu_bench
--l1`) ggml's dots, the repo's AVX2 multi-token kernel and the same kernel rebuilt for 256-bit AVX512-VNNI (`vpdpwssd`/`vpdpbusd`
on the Xeon 6240, parity test passing) all take 0.115-0.119 ns per weight (2.4 GB/s of expert bytes per core). Three independent
implementations at the same speed point to a shared limit (the table-driven grid decode) rather than the multiply-add chain.

## Lossy decode settings judged by C++ coding tasks (measured; the gate is functional, not KL)

The KL gate above is replaced for the faster settings by `tools/minimax_cpp_eval.py`: 46 single-function C++ tasks (26 easy, 20
harder: an LRU cache class, Dijkstra, topological sort, an expression evaluator, knapsack, DP, string algorithms), each with unit
tests. The model answers with thinking off (the prompt ends in `</mm:think>`), greedy decoding, up to 450-500 new tokens; the first
code block is compiled with g++ -std=c++17 and run against the tests. "Pass" = compiles and every assert holds. `selftest` checks
that a reference solution passes each test. One run per setting (single greedy sample), DSpark speculation with the defaults
(draft 6, confidence floor 0.7), `STRATA_M3_IQ256=1`, FP16 K/V, one V100 + 2 x Xeon 6240. Decode tok/s is the mean over the
tasks' answers (about 230 tokens each); code answers speculate better than the prose prompts used above, hence the higher speeds.

| Setting (REPL key / CLI) | tok/s | Pass, 46 tasks | Easy 26 | Hard 20 | Pass vs exact |
|---|---|---|---|---|---|
| exact (`--speed exact`) | 24.1 | 41 | 26 | 15 | - |
| top-3 (`topk=3`, `--speed fast`) | **30.3** | 43 | 26 | 17 | 105 % |
| top-2 (`topk=2`, `--speed turbo`) | **39.2** | 37 | 26 | 11 | 90 % |
| top-2 + weight floor 0.4 (`topk=2 minw=0.4`) | 42.2 | 38 | 25 | 13 | 93 % |
| keep one missed expert (`missk=1`) | 38.1 | 38 | 24 | 14 | 93 % |
| drop every expert that misses VRAM (`missk=0`) | 47.2 | 39 | 25 | 14 | 95 % |
| missk=0 + resident stand-ins (`missk=0 sub=1`) | 42.8 | 36 | - | - | 88 %, slower |
| top-3 + missk=0 (`topk=3 missk=0`) | 49.0 | 37 | - | - | 90 % |
| **top-2 + missk=0 (`topk=2 missk=0`, `--speed max`)** | **50.6** | 39 | - | - | **95 %** |
| top-1 (`topk=1`) | 51.3 | 32 | 21 | 11 | 78 % |

Reading: the exact path fails 5 of the 46 tasks itself, so differences of one or two tasks are within what a single sample can show.
With "at least 80 % of the exact path's passes" as the bar (the user's), top-3 is the 30+ tok/s setting (it keeps every easy task and
passes 17 of 20 hard ones) and top-2, top-2 with a floor and `missk=0` are all above 35 tok/s with 90-95 % of the passes; top-1
(51 tok/s) falls just below the bar at 78 %. Dropping the experts that would run on the CPU (`missk=0`) is the best trade at the
top end: 47 tok/s for 95 % of the passes, and 50.6 tok/s for the same 95 % when top-2 is applied as well (`--speed max`). Not measured: other languages, prompts much longer than these, free-form (multi-function)
programs, sampling with temperature, and repeats of each setting.

### Where a `--speed max` step goes, and what did not help (measured; code tasks, up to 6 drafts)

`--speed max` (top-2, every expert that misses VRAM dropped except a token's heaviest): 47.5 / 51.8 / 50.2 tok/s on three tasks, 91-96 %
of drafts accepted, 3.7-4.3 tokens per verify step, about 82 ms per step: waiting for the GPU chain 40 ms, the CPU lane 20 ms (it
still computes the heaviest expert of a token when that one is not resident: 70 expert computations per step), the draft pass
10.3 ms (6.2 ms for 2 rows), the rest about 10 ms. The GPU chain of a 7-row window, per 57-layer step, from the CUDA-event timeline
(`STRATA_M3_PROF=3`): q/k/v matvecs 11.0 ms, o-projection 6.7, sparse attention 6.4, router 2.4, block scores 1.2, the rest 4.2:
31.9 ms against about 15 ms of dense weight reads at 576 GB/s, so the multi-column matvec kernels run at ~2.7x their bandwidth
floor with 7 columns. Tried and neutral: cache admission 1-8 experts per layer (the frequency gate admits the same ~440 experts per
run; LRU admits 5-10 times more, empties the CPU lane by 40 % and is slower, 43.9 against 46.8 tok/s), and the confidence floor
(means over four tasks: 0.4 -> 47.9, 0.55 -> 49.2, 0.7 -> 48.1 tok/s; within the run-to-run spread). Note the REPL keeps `admit=` and
`policy=` between requests: set them in every request of a sweep.

### Dense multi-column matvec: one row per block and the transposed reduction (measured 2026-11-02)

`dense_mmvq_bench` (new, `src/kernels/dense_mmvq_bench.cpp`) times the dense MMVQ kernels on MiniMax shapes for 1-8 columns. At 5-8
columns the kernels are limited by per-column instruction cost (25 % occupancy, 111 registers), not by bandwidth. Two existing knobs,
`STRATA_NO_MMVQ_ROWS2=1` (one row per thread block) and `STRATA_TSUM=1` (transposed warp reduction), run 15-20 % faster there with
bitwise the same sums. strata-m3 now sets both by default; the environment still wins (`=0` restores the library defaults).
Engine A/B on one V100 (max-mode routing `topk=2 missk=0`, `--dspark --kv-f16`, 300 tokens, mean of 3 runs, old / new tok/s):
dijkstra 49.8 / 52.3, lru_cache 48.9 / 51.3, knapsack 46.2 / 48.4, eval_expr 53.1 / 56.0 (+5 %). Exact routing with speculation,
200 tokens on two prompts: the generated token ids are identical with and without the knobs. Not measured: other GPUs.

### Where the rest of a `--speed max` step goes, and what was tried on it (measured 2026-11-03, one V100)

Per verify step (79 steps, `topk=2 missk=0`, dijkstra prompt, 300 tokens, 50.3 tok/s, 75 ms a step; `STRATA_M3_PROF=3` and `=4`):
attention part of the 57 layers up to the router 28.1 ms; after the router (shared expert 7.2, resident experts 18.6, wait for the CPU
0.1 ms per layer = 5.7, combine 3.1) 35 ms; the draft pass, head and host work between steps 11 ms. After the router the GPU, not the CPU
lane, is the longer side: the CPU result is ready 0.1 ms after the GPU has finished the resident experts. Host time of one window layer
(`M3_LANE_HOST`): filter + shared expert 0.014 ms, jobs + classify 0.003, CPU quantize + start 0.020, queueing the GPU experts 0.063,
gate/up join 0.122, quantize-h + down 0.100, accumulate + add 0.010 (sum 0.335 ms).

| Change | Result | Kept |
|---|---|---|
| Resident experts as 5 kernels instead of 8: gate and up as one matvec of 2 FF rows, the activation's q8_1 rows copied from the layer's one image, SwiGLU fused with its quantization (`STRATA_M3_NO_FUSE_EXPERT=1` restores the 8) | max-mode 55.1 / 54.0 tok/s against 53.5 / 52.5 (+3.2 % / +2.8 %, dijkstra / lru_cache, 2 runs each); generated tokens identical; exact routing 27.12 against 27.15 (the CPU lane dominates there) | yes, default |
| CPU gate/up phase started on the workers before the GPU experts are queued (`Par::start` / `finish`) | 50.3 -> 51.0 tok/s (one run each), tokens identical | yes |
| DSpark Markov kernel: 17 16-byte loads per vocabulary row instead of 256 byte loads | Markov + argmax + confidence part of the draft pass 2.31 -> 1.48 ms, draft pass 9.7 -> 8.9 ms, tokens identical | yes |
| Dense matvec, activation-sum (dot2) removed (wrong results, upper bound only) | 0-2 % at 1-5 columns, up to 7 % at 6 | no |
| Dense matvec, minimum blocks per SM 4 -> 6 / 8 | q: 138 -> 136 / 133 us at 8 columns, at most 4 % | no |
| Dense matvec on tensor cores (WMMA 32x8x16, fp16 dequantized weights, Q5_K) | correct (relative error 4e-4) but flat at 115 us for 1-8 columns against 50 us (1 column) and 125-136 us (7-8) for the dp4a kernel | no |

Estimated, not built: splitting a window into two half-windows so one half's CPU experts overlap the other half's GPU work would hide the
CPU lane but doubles the launches and re-reads the dense weights (about +15 ms of GPU time against 16.7 ms hidden). A draft head
restricted to 16K-32K frequent tokens would save about 3.5 ms a step; tokens of the benchmark answers are covered 98.1 % (16K) and
99.2 % (32K) by a list made from this repository's code and docs, so the loss in acceptance is about the same size as the gain.
The fused SwiGLU + quantize kernel computes the activation with libdevice `expf` and IEEE operations on purpose: `native_mmvq.cu` is
built with `--use_fast_math`, and the first version (plain `expf`, `/`) changed the generated tokens on 2 of 4 prompts.

The resident experts of a window layer on side streams (`STRATA_M3_EXP_STREAMS=S`, default 4, 0 = off): each expert job has its own scratch and
output rows, and one kernel adds the jobs into `moe` in the original job order, so every element sees the same operations in the same order
(generated tokens identical in all rows). One V100, `--dspark --kv-f16`, 300 tokens, two runs each (dijkstra / lru_cache, tok/s):

| side streams | max-mode (`topk=2 missk=0`) | exact routing, 200 tokens |
|---|---|---|
| 0 | 55.3 / 54.2 | 27.8 / 26.5 |
| 2 | 56.9 / 55.7 | - |
| 3 | 57.0 / 55.9 | 27.9 / 26.6 |
| 4 | 57.3 / 56.1 | - |

With the stream default at 4 only the max-mode rows were measured at 4 (not the exact-routing rows); the CPU lane is the longer side in exact routing.

### A second V100 as extra expert cache (`STRATA_M3_GPU1=1`, opt-in; measured 2026-11-03, two V100 PCIe on different sockets)

GPU 0 keeps everything it had (dense weights, K/V, draft head, prefill, its own slots). GPU 1 holds only expert slots (1800 of 17.3 MiB, so
2.9k slots in total instead of 1.1k) and its own copy stream; the warm-up after the prefill fills it with the experts the prompt used most. A
decode-window layer sends the q8_1 image of `h` to GPU 1 by peer copy, runs the experts resident there on GPU 1's side streams (the same
kernels as on GPU 0) and copies their output rows back into the job's rows on GPU 0; the ordered accumulate adds all jobs in the original order.
The prefill never uses GPU 1's slots; the one-token step goes through the window lane when a second GPU is on, so it uses them too.
Peer access works between the two cards (copies: 28 KB 4.3 us, 100 KB 11 us, 200 KB 21 us; pinned host-to-device 12.3 GB/s on each).

Admissions onto GPU 1 take at most `STRATA_M3_GPU1_ADMIT` (default 2) experts per layer. Measured with the final code, two runs each, mean tok/s
(dijkstra / lru_cache; 200 tokens, max-mode 300 and 261 tokens, the lru_cache answer differs in length between runs on one and two GPUs):

| mode | one GPU | two GPUs | hit rate one -> two GPUs |
|---|---|---|---|
| exact routing | 27.0 / 25.8 | 38.8 / 38.7 (+44 % / +50 %) | 53 -> 76-78 % |
| `--speed fast` (top-3) | 32.9 / 33.3 | 46.7 / 46.8 (+42 % / +40 %) | 58 -> 81-83 % |
| `--speed max` (top-2, `missk=0`) | 56.7 / 55.5 | 62.7 / 59.3 (+11 % / +7 %) | 83 -> 92-93 % |

The 46-task C++ benchmark, two GPUs: exact routing 41 of 46 passes at 37.6 tok/s mean (one GPU: 41 at 25.5); `--speed max` settings 38 of 46 at
62.7 tok/s mean (one GPU: 36 and 36 of 46 at 56.8 and 56.9). With admissions off (`STRATA_M3_GPU1_ADMIT=0`, only the warm-up fills GPU 1) exact
routing gives 34.6 / 35.3 and `--speed fast` 41.8 / 41.0 tok/s; one admission a layer: 38.2 / 38.1 and 46.4 / 43.5; two: 38.6 / 38.7 and 46.4 / 46.7.
The simulator predicted 79.7 % at 2800 slots (52.4 % at 1100) for the exact-routing code trace; the engine reached 78.8 %.
Exactness of the second GPU (`STRATA_M3_GPU1_CHECK=1`, debug and slow: every second-GPU job is also run on GPU 0 from the host copy of the expert and
the output rows compared bit for bit): 20,487 jobs (6,610 on dijkstra, 13,877 on lru_cache; exact routing, speculation on), 0 differ. A second-GPU
expert is computed bit for bit as on GPU 0. Whole-run tokens, placement made equal (`STRATA_M3_GPU1_ONLY=1` with `admit=0` and `STRATA_M3_T1_MULTI=1`, debug): the warm-up puts the
same 1019 experts a one-GPU run would keep into GPU 1's slots (and empties GPU 0's small-class slots and any other second-GPU slot), admissions are
off, and the one-token step goes through the window lane in both runs, so both run the same experts on a GPU and the same on the CPU. Exact routing,
speculation on, 200 tokens: the generated token ids are identical on all four prompts (dijkstra 26062ead, lru_cache 0d4d31ee, knapsack 1a4a28d3,
eval_expr 2ee63a99), and the CPU / GPU expert counts are equal (for example knapsack 35913 / 15159). Before the one-token step and the stale second-GPU slots were
handled the hashes differed on three of the four prompts, with unequal counts: equal placement is what matters.
In normal use (second-GPU admissions on, a cache 2.6 times larger) the two runs place different experts on the CPU lane, and the CPU lane and
the GPU round differently in the last bits, so whole-run tokens differ from a one-GPU run in exact routing by that placement alone. Teacher-forced 64 steps against the exact references
(`sw/*-exact.f32`), short / code prompts: one GPU KL 4.7e-4 / 6.0e-4, top-1 64/64 and 64/64; two GPUs KL 6.3e-4 / 7.8e-4, top-1 63/64 and 64/64;
two GPUs against one GPU KL 7.1e-4 / 6.2e-4, top-1 63/64 and 64/64 (the gate is KL <= 0.01 and top-1 >= 95 %). The 46-task benchmark's 41 of 46 in exact
routing equals the one-GPU result. The one-GPU default path's generated tokens are unchanged by all of this (hashes of four
prompts equal the earlier runs). Peer access works between the two cards (copies: 28 KB 4.3 us, 100 KB 11 us, 200 KB 21 us; pinned host-to-device
12.3 GB/s on each).

With the one-token step also going through the window lane (final code, one run each, dijkstra / lru_cache): exact routing 38.5 / 38.6, `--speed fast` 46.5 / 47.5,
`--speed max` 63.3 / 60.0 tok/s (261 tokens in the second answer), unchanged from the table above within the run-to-run spread.

Found on the way: a window layer with more than 16 distinct experts (4 tokens of exact routing often have more) overflowed the job list of the
side-stream path and fell back to the sequential loop, which queues every expert on GPU 0, so an expert resident on GPU 1 was read over PCIe by
GPU 0's kernels: one prompt lost 22 % with admissions on. The capacity is now 64 jobs (top_k x window) and the ordered accumulate runs in launches
of 16 jobs. Not the cause, and not kept: copy-engine transfers replaced by mapped pinned host memory (slower, 59.3 against 63.4 tok/s in max-mode)
or by direct peer reads (`STRATA_M3_GPU1_DIRECT=1`, 60.9). Not measured: more than two GPUs.

## The simulator fitted on the measured path, and projections for batched decode and prefill (2026-11-03)

`tools/sim/minimax_step.py` is a MiniMax step-time model (the GLM-derived model above was never fitted on this engine); its inputs are
in `tools/sim/data/minimax_m31_measured.json` (dense matvec bench for 1-8 columns, the max-mode step profile, the routing union law from
`STRATA_M3_ROUTE_TRACE`, cache-simulator hit rates for B sequences, the six single-stream decode speeds, the prefill points). Only these are
fitted: the CPU-lane constants (250 us a job, 130 us a layer, 40 us a further token), a host term of 120 us a layer (stream wake-up, routing
filter, queueing), the effective stream job time (23 us) and three prefill constants. `python3 tools/minimax_m31_fit.py` prints everything
below (`docs/benchmarks/minimax_m31_sim_20261103/fit.txt`); `python3 tools/minimax_m31_sim.py --measured` prints the measured speeds next to the
old GLM-fitted projection; `python3 tools/test_sim_minimax.py` checks the fit.

Single-stream decode with DSpark (4.1 rows a window, 3.8 tokens a step), mean of the dijkstra / lru_cache runs, measured against the fitted model:

| | exact | fast (top-3) | max (top-2, `missk=0`) |
|---|---|---|---|
| one V100 | 26.4 / 26.6 | 33.1 / 33.0 | 56.1 / 55.0 |
| two V100s | 38.8 / 38.0 | 46.8 / 46.0 | 61.0 / 59.9 |

Plain decode without speculation (the first engine build, exact routing, one GPU) was 16.0 tok/s, which the GLM-fitted projection also gave;
every number above includes speculation, FP16 K/V, the fused experts, side streams and (two GPUs) the second GPU's cache.

**Batched decode, PROJECTION.** The engine decodes one sequence of at most 8 rows today. The model runs B sequences, each with a 4.1-row
window, 2048-token context (K/V 154 KB a token, so more sequences leave fewer expert slots), shared dense matvecs and experts, separate attention.
Aggregate tok/s (per-sequence tok/s is that divided by B), B = 1 / 2 / 4 / 8 / 16 / 32:

| | exact | fast | max |
|---|---|---|---|
| 1 GPU, matvecs as today (8 columns a launch) | 27 / 31 / 36 / 39 / 44 / 48 | 33 / 39 / 45 / 49 / 54 / 58 | 55 / 67 / 82 / 90 / 97 / 100 |
| 2 GPUs, matvecs as today | 38 / 46 / 53 / 58 / 62 / 64 | 46 / 56 / 66 / 71 / 75 / 76 | 60 / 70 / 86 / 97 / 104 / 108 |
| 1 GPU, + tensor-core dense path above 8 columns | 27 / 34 / 42 / 47 / 55 / 61 | 33 / 43 / 54 / 62 / 72 / 79 | 55 / 79 / 117 / 150 / 172 / 180 |
| 2 GPUs, + tensor-core dense path | 38 / 51 / 66 / 78 / 85 / 90 | 46 / 64 / 86 / 104 / 112 / 116 | 60 / 87 / 140 / 196 / 228 / 245 |

Reading: with today's kernels batching gains little (every launch re-reads the weights for each 8 columns); a dense path that reads the
weights once (the tensor-core dequantize-and-multiply prototype is flat in the columns, 115 us for the q matrix) raises the max-mode
aggregate to about 2.7x (1 GPU, B = 8: 150) and 3.3x (2 GPUs, B = 8: 196) the single-stream speed, at 19 and 25 tok/s a sequence. Exact and fast
routing stay limited by the CPU lane (B = 8, 1 GPU: 4.3-5.8 CPU jobs a layer) and their hit rate falls with B (cache shared by the sequences:
64 % at B = 8 against 84 % in max mode); the draft pass (14 ms at B = 8 with the tensor-core path, 40 ms without) and per-sequence attention
(32 us + 23 us a row a layer) are the next terms. Not modelled: the engine work (K/V per sequence, several sequences in one window, 32-row
buffers), sequences of different lengths, and any scheduling loss.

**Prefill, PROJECTION.** The model: each chunk streams the experts its tokens touch over PCIe (one link per GPU) while the GPU computes;
most of the compute hides under the stream. Fitted on 6960 tokens (14.2 s), 210 tokens (22.8 tok/s) and 487 tokens (38.6 tok/s); it has a
fixed 3.5 s a prompt (warm-up, last layer, tail), which is a fitted number, not a measured one. tok/s:

| | chunk 2048 | 4096 | 7168 | 14336 |
|---|---|---|---|---|
| 14336-token prompt, 1 GPU, as fitted | 204 | 337 | 555 | 575 |
| same, stream fully hidden | 209 | 350 | 588 | 593 |
| 14336-token prompt, 2 GPUs (each streams half the experts), as fitted | 387 | 618 | 870 | 895 |

Many short prompts in one chunk share one pass over the pack: 1 x 210 tokens 22 tok/s, 8 x 210 129, 32 x 210 479, 64 x 210 541 (1 GPU; 32 / 203
/ 699 / 854 with two GPUs); long prompts approach 600 (1 GPU) and 1000 (2 GPUs) tok/s, bound by the 125 GiB pack over 12.3 GB/s links.
Not measured: the split of the 14.2 s between GPU work and waiting for the stream (the model assumes 10 s of GPU work); a profile of
`--pf-gemm --pf-prefetch` would pin the fixed cost and the overlap down. Forecasts in this family of models have been several tok/s off on
GLM; treat the batched and prefill rows as ranges, not promises.

## Batched decode: the multi-sequence engine and the weights-once kernel (measured 2026-11-03, one V100)

**What exists.** `strata-m3 --seqs N` reserves K/V for N sequences (before the expert slots are sized: each costs 154 KB a context token) and the REPL command
`runb prompts=a.ids,b.ids,... n=200 draft=6 [topk=2 missk=0 ...]` prefills each prompt on its own (exact routing), warms the expert cache once from all of
them, then decodes them together: every step drafts per sequence (DSpark, one pass each), runs the windows of all active sequences as ONE forward
(`forward_multi`: the dense projections, router and experts shared over all rows, the attention per sequence on its own K/V and positions), and
verifies each sequence on its own rows. A batch step takes at most 32 rows (`kMaxWin`; a single sequence still takes 8: `win_cap`). A batch of one gives the
single-sequence tokens exactly (exact routing, `admit=0`: hash bdf2ef0a both; 21.98 against 21.90 tok/s), and two copies of one prompt give that prompt's tokens in both
sequences (30.4 against 22.0 tok/s aggregate). Not done: prefill of several prompts as one chunk, a batched draft pass, sequences of different lengths in one K/V
allocation, `runb` on two GPUs (untested).

**The weights-once kernel** (`src/kernels/cuda/native_tc_mmvq.cu`, test `tc_mmvq_test`): Q4_K / Q5_K / Q6_K blocks are decoded to fp16 in shared memory and
multiplied on tensor cores (WMMA 32x8x16, up to 64 columns in one pass over the weights); the activations are fp16 and the sums fp32, so the results are not bitwise
those of the q8_1 kernel (every output within 0.5 % of the largest value, 0.03 of the rms) and it is used only for batch steps of more than 8 rows. Microseconds a call:

| columns | q (Q5_K 6144x8192) | o (8192x6144) | shared gate (6144x3072) | lm_head (Q4_K 200064 rows) | q with the 8-column launches (engine settings) |
|---|---|---|---|---|---|
| 8 | 118 | 115 | 57 | 1860 | 138 |
| 16 | 158 | 147 | 74 | 2594 | 276 |
| 32 | 267 | 216 | 111 | 4088 | 552 |
| 64 | 378 | 477 | 187 | 7062 | 1104 |

It is not flat in the columns (the activation fragments come from global memory and the 4 warps take turns): about 117 us at 8 columns, the q matrix at 300 GB/s
effective, 2-3 times faster than the launches above 16 columns, and the lm_head 3x at 32 columns. A shared-memory staging of the activation tile is the obvious next step.

**Batched decode, `--speed max` settings, DSpark, different prompts, context 2048, 200 tokens a sequence** (aggregate tok/s; one run each):

| sequences | rows a step | tok/s with the weights-once kernel | tok/s with 8-column launches | per sequence | forward ms / draft ms a step |
|---|---|---|---|---|---|
| 1 | 4.1 | 55.5 | 55.5 | 55.5 | 57.8 / 8.8 |
| 2 | 8.0 | 68.5 | 64.1 | 34.2 | 92.6 / 17.5 |
| 4 | 11.6 | 70.5 | 66.2 | 17.6 | 125.6 / 29.5 |
| 8 (3 drafts) | 17.4 | 81.2 | 73.8 | 10.2 | 164.1 / 36.7 |

Exact routing, 4 sequences: 29.9 (weights-once) and 28.4 tok/s; the batch is bound by the CPU lane there (hit rate 48 %, 354 ms a forward). The simulator had projected
82 / 117 for 4 sequences and 90 / 150 for 8 (8-column launches / weights-once): too optimistic because it assumed one attention launch for all sequences and a batched
draft pass. The engine runs the draft once per sequence (7.3-7.5 ms each, 37 ms of a 200 ms step at 8 sequences) and launches the attention kernels per sequence
(refit: 90 us a sequence and layer beyond the first, 12 kernels). The fitted model (`tools/minimax_m31_fit.py`, measured against model per row) reproduces the batches
to within 7 % up to 4 sequences (8 sequences: model 71, measured 81) and says what the next two engine changes are worth at 4 / 8 sequences (aggregate tok/s):
a batched draft pass 88 / 95, plus one attention launch for all sequences 96 / 110, plus a second GPU 112 / 136. The longer-run table of the projection (steady state,
3.8 tokens a sequence step, which the 4-sequence run does not reach because its sequences finish at different steps) is in `docs/benchmarks/minimax_m31_sim_20261103/fit.txt`.

Quality and the default path (same day, one V100): the 46 C++ tasks decoded four at a time (`runb`, batches of 4, `--speed max` settings, draft 6) pass 39 of 46 at 72.7 tok/s
aggregate (mean over the batches); the same settings one sequence at a time pass 36-37 of 46 at 56.8 tok/s and the exact path 41 of 46. The default single-sequence path is
unchanged by the multi-sequence work: exact routing, admissions on, the four prompts' generated tokens hash to the same values as before (dijkstra 1ee67a8d, lru_cache 38fb7620,
knapsack cd04cbf5, eval_expr e30269a7; 27.7 / 26.4 / 25.0 / 23.5 tok/s). One change was needed for that: the batch-sized buffers (32 logit rows, the fp16 activation buffer) are
allocated only when `--seqs` is above 1, because the larger logits buffer alone cost the single-sequence engine one expert slot (1036 against 1037), which moved the exact-routing
tokens of two prompts.

## Prefill at 8-16K tokens: where the time goes and what was changed (measured 2026-11-03, one V100 PCIe, UD-Q2_K_XL, draft head on)

Profile of one 16000-token chunk (`STRATA_M3_PFPROF=1`, `--chunk 16384 --pf-gemm --pf-prefetch --kv-f16`, 537 tok/s with injection on tensor cores, 29.8 s; PFPROF itself adds a few percent):
sparse attention 6.1 s, expert and dense GEMMs (cuBLAS, about 65 TFLOP/s) 10.5 s, i-quant dequantization 2.2 s, indexer block scores 2.0 s, f32 to f16
conversions 0.56 s, indexer projections 0.65 s, block selection 0.35 s, draft-head injection 2.0 s (8-row loop). The single 125 GiB expert stream (10.9 s)
is hidden under this at 16K; the pipeline is compute-bound, and the 3.5 s "fixed cost" in the fitted prefill model is attention that grows with the
prompt, not a fixed term.

Changes, all opt-in except the exact one (tokens/logits compared as noted):

| change | 16K single chunk | 8K single chunk |
|---|---|---|
| `STRATA_M3_PF_INJECT=1` (draft head on; the baseline of the rows below): draft-head K/V injection on tensor cores (`native_tc_mmvq`, now with Q8_0); injection 2.0 -> 0.8 s at 16K, 0.88 -> 0.35 s at 7K, tokens and acceptance unchanged | 537 tok/s | 497 tok/s |
| `STRATA_MSA_TC2=1`: tensor-core prefill attention with 47 KB shared memory (two blocks per SM, 64-key units, 32-lane softmax, merge factors computed once); standalone 1.6x (46 vs 75 ms at 8192 queries, output max difference 9e-5 on values of rms 0.04) | 613 tok/s | 541 tok/s |
| block-score kernel skips key blocks no query of the group can see (exact; always on in prefill with FP16 caches): scores 2.0 -> 1.06 s at 16K, 0.5 -> 0.27 s at 8K | 634 tok/s | |

Quality of `STRATA_MSA_TC2=1` (teacher-forced 64 steps against the default q8_1 path, `--pf-gemm`): short prompt KL 5.4e-4 (without the flag 1.3e-3), top-1
64/64; 7K prompt KL 1.37e-3 and 1.67e-3 in a second run (without the flag 1.36e-3 and 1.16e-3), top-1 47/47. Run-to-run spread of the same configuration is
about 1e-4 in KL because the CPU/GPU expert placement is not fixed; the 7K figure of the flagged path sits at the 1.6e-3 gate line.

What is left (16K, after the changes): sparse attention 6.1 s is still the largest item at about 11 TFLOP/s effective (latency bound on the fragment loads);
cuBLAS 10.5 s; dequantization 2.2 s is serialized with the GEMMs. The ncu profile of the attention kernel did not finish (kernel replay of a process holding
125 GiB of pinned memory); not measured. Not done: dequantize/GEMM overlap, second per-layer host sync, 2-GPU prefill.
