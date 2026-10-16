# Muse Spark Open 816B-A42B on TR16 + B550 (plan, measurements and forecast, 2026-10-16)

Can Strata run Meta's Muse Spark Open at UD-IQ1_S (175.6 GB) on the two machines at once, TR16 (125 GB) and the B550
(62 GB), over the InfiniBand link? This note gives the architecture as the GGUF headers state it, what was measured,
the simulator's forecast for decode and prefill, and the plan for the runner on top of the GLM one. No Muse weights
were downloaded (no disk space): everything below comes from the shard headers (the first megabyte of each shard,
13 MB for the metadata shard) and from kernel benchmarks on synthetic blocks.

**Forecast: decode about 5.4 tok/s, prefill about 250-340 tok/s at 8K**, both with today's measured CPU kernel and
InfiniBand QDR. Decode is bound by TR16's CPU expert lane (123 of 186 ms per token). Stock ggml-cpu would give
0.7 tok/s on these CPUs (see the IQ1_S kernel below).

## The model

From `tools/muse_gguf_census.py` over the UD-IQ1_S shard headers (`tools/sim/data/census-muse-spark-ud-iq1s.json`)
and llama.cpp's `src/models/muse-spark.cpp` (master, 2026-10-16), which is the reference for the arithmetic:

| | Muse Spark Open | GLM-5.3-Flash (for comparison) |
|---|---|---|
| layers | 62, all MoE (no dense layer, no shared expert, no MTP head) | 45 + 1 MTP, 3 dense |
| hidden | 8192 | 4096 |
| attention | GQA 128 q / 16 kv heads of 64, sigmoid output gate, QK-norm; 46 layers with a 2048-token sliding window and RoPE, 16 full-attention layers (1, 5, ..., 61) without RoPE | KDA + 11 MLA layers |
| MoE | router on the 8192 stream (sigmoid + selection bias, top-8 of 256, weights renormalized); experts run in a 4096 latent (`ffn_latent_down` 8192 -> 4096, experts 4096 -> 4096 -> 4096, `ffn_latent_up` 4096 -> 8192); each expert's output is RMS-normalized (`ffn_post_exp_norm`) before the router weight | top-8 of 288 experts of 4096 -> 2048 -> 4096, 1 shared |
| residual | norm-preserving: `alpha * x + beta * branch` per channel, after weightless RMS norms | mHC (4 streams) |
| head | vocab 202,048, logits x 0.177 then softcap 20 | vocab 154,880 |
| UD-IQ1_S file | 175.63 GB: experts 156.03 GB (IQ1_S, 9.83 MB each), block trunk 17.31 GB (Q6_K q/o, Q8_0 k/v/gate/latent, F32 router), output 1.36 GB (Q6_K), token_embd 0.93 GB (Q4_K) | Q23 pack 111 GB |

Per decoded token the GPUs read the trunk and the head, **18.67 GB**, and the CPU lanes read up to 4.88 GB of routed
experts (8 x 62 x 9.83 MB). That is the reverse of GLM-5.3-Flash, where the experts dominate: Muse's trunk does not fit
one 16 GB card, so it has to be split across both cards by layer.

## Fit: the model needs both nodes

TR16 can give the experts about 116 GiB and the B550 about 56 GiB (8 and 6 GiB kept for the OS, desktop and buffers).
The experts are 145.3 GiB, so the B550 must hold at least 0.21 of them and can hold at most 0.385. The engine splits
every expert's rows in whole 256-value blocks (`strata_muse_row_split`, `include/strata/host/muse_names.h`), so the
largest share that fits is 6/16 = 0.375: the B550 holds rows [2560, 4096) of every expert (54.5 GiB), TR16 the rest
(90.8 GiB) plus the trunk copies it uploads. `strata-muse-inspect` prints this from the headers:

```
$ strata-muse-inspect s1.gguf h2.gguf h3.gguf h4.gguf h5.gguf
muse-spark: 62 layers (16 full attention, 46 sliding window 2048), hidden 8192, latent 4096,
  256 experts of 4096 x 4096 (top-8, sigmoid router, weights renormalized), 128/16 heads of 64, vocab 202048, context 1048576
  rope base 500000 (sliding-window layers only), rms eps 1e-05, post-norm eps 1e-08, logit scale 0.176777, softcap 20
footprint: 175.63 GB = experts 156.03 (9.83 MB each) + block trunk 17.31 (279.2 MB per layer) + output 1.36 + token_embd 0.93
  per decoded token: 18.67 GB of trunk + output on the GPUs, 4.88 GB of routed experts (8 x 62)
two nodes, 56 GiB for node B's expert rows: node B holds rows [2560, 4096) of every expert (share 0.375), 54.5 GiB; node A keeps 90.8 GiB of expert rows
ok
```

Neither node needs the whole file: a packer can fetch each expert tensor by HTTP range and write the two row slices,
so TR16 needs about 117 GB of disk (its expert rows plus the trunk, head and embedding) and the B550 about 59 GB
(67 GB when its card holds trunk layers).

## The IQ1_S CPU kernel (measured)

ggml-cpu's AVX2 IQ1_S dot product decodes the grid indices with BMI2 `pdep`, which Zen 1 and Zen 2 run in microcode.
On both of these CPUs it is an order of magnitude slower than the other i-quants:

| 1 GiB of rows, decode-shaped (1 token), Q8_K activations | TR16 1950X, 15 threads | B550 3950X, 16 threads |
|---|---:|---:|
| ggml-cpu IQ1_S (`pdep`) | 1.62 GB/s (8.3 GMAC/s) | 1.33 GB/s (6.8 GMAC/s) |
| the same arithmetic without `pdep` | 20.0-21.4 GB/s (102-110 GMAC/s; one run 77) | 17.4 GB/s (89.1 GMAC/s) |
| ggml IQ2_XXS / Q2_K / Q4_K, for scale | - | 17.5 / 24.8 / 25.3 GB/s |

The engine's kernel is a new format in the AVX2 multi-token family (`src/kernels/cpu/iq_avx2.cpp`, `Fmt32<19>`): a
weight is `d * (2s+1) * (g +- 1/8)` with `g` in {-1, 0, 1}, so `8g +- 1` is an odd integer in [-9, 9] whose
magnitude feeds `maddubs` and whose value is the `vpsignb` operand, and each 32-value chunk is decoded once for every
token of a group. It is reached only where the caller sets `NativeFmt::iq1s_rows` (the Muse loader) or
`STRATA_IQ1S_256=1`, so a model that already runs IQ1_S experts on ggml-cpu keeps its rounding. Whole Muse experts
(gate/up, the hidden's Q8_K, down) through the engine dispatch, `iq1s_expert_parity --bench --threads 15 --mb 1024`:

| TR16, 15 threads, 2026-10-16 16:27, idle | 1 token per expert | 2 | 4 |
|---|---:|---:|---:|
| Strata IQ1_S kernel | 18.0 GB/s, 92.2 GMAC/s | 156.9 GMAC/s | 209.2 GMAC/s |
| ggml-cpu | 1.49 GB/s, 7.6 GMAC/s | 7.6 | 7.3 |

`iq1s_expert_parity` (a ctest) checks the rows against ggml-cpu (rel 1.8e-7 gate/up, 1.3e-7 down) and bit for bit
across 1..8 tokens, on TR16 and on the B550. The B550's engine-level rate is not measured: a 26-thread miner held
its CPU during this session (runs under it gave 14-25 GMAC/s). Even the new kernel is compute bound (18 GB/s against
TR16's 65.6 GB/s expert-lane stream), so the CPU lanes, not the link or the RAM, set the decode speed.

## Forecast (simulator)

`tools/sim/muse.py`, `python3 tools/muse_sim.py report`. Per decode token, layer by layer: the GPU chain on the card that
holds the layer's trunk (bytes over the measured GEMV bandwidth, ~5 us per launch, 18 launches), the GPU-resident
experts (the leftover VRAM filled with each card's hottest experts), and the slower of the two CPU lanes plus its
InfiniBand round trip (ucomm: rtt/2 + bytes at 3.26 GB/s, plus the B550 worker's 0.15 ms per request). Prefill
streams each layer's experts to a GPU per chunk; its GEMM rates are fitted to the GLM 8K prefill trace.

Two arrangements of the two 16 GB cards: **split-gpus** (TR16 + RX 9070 XT, B550 + RTX 5060 Ti; docs/GLM_MULTINODE.md)
and **gpus-on-tr16** (both cards in TR16, the B550 gives RAM and CPU only; how the machines stood on 2026-10-16).

### Decode, 8K context, InfiniBand QDR, best plan (no speculation)

| CPU kernel | trunk on the GPUs | split-gpus | gpus-on-tr16 | bound by |
|---|---|---:|---:|---|
| ggml-cpu (stock) | as shipped | 0.67 | 0.68 | CPU (1.4 s per token) |
| **Strata IQ1_S (measured)** | **as shipped** | **5.37** | **5.56** | TR16 CPU lane 117-123 ms + GPU chain 57 ms |
| Strata IQ1_S | Q4_K copies (lossy) | 6.67 | 6.94 | |
| a DRAM-bound kernel (not written) | as shipped | 9.06 | 9.30 | GPU chain 56 ms, CPU 45-48 ms |
| a DRAM-bound kernel | Q4_K copies | 11.42 | 11.63 | |

The row split wins in every case: whole-expert placement (hottest on the B550) loses 0.6-0.8 tok/s to the per-token
imbalance of 8 routes over two nodes, and a layer split (each node its own layers' experts) runs the two CPUs one
after the other (3.47; impossible with both cards in TR16, which cannot hold every expert). With the shipped trunk the
two cards' leftover VRAM holds about 960 experts (6 % of each card's layers), which take about 30 % of the routes on the
borrowed GLM hit curve.

| decode sensitivity (split-gpus, measured kernel) | tok/s |
|---|---:|
| baseline | 5.37 |
| B550 CPU shared with the miner (20 GMAC/s at 1 token) | 3.51 |
| 1GbE instead of InfiniBand | 4.22 |
| routing skew 0.7x / 1.3x the GLM curve | 5.00 / 5.80 |
| context 32K / 128K | 5.11 / 4.01 |
| top-6 routing (lossy) | 6.45 |
| n-gram drafts, width 2, 1.4 accepted per step | 4.60 |

Speculation does not pay without a draft model: at width 2 the two tokens share few experts (0.93 distinct experts per
route on GLM), so the CPU work nearly doubles while the kernel only amortizes the decode of experts both tokens use.
Muse Spark ships no MTP head.

### Prefill, best plan and chunk

| tokens | split-gpus | gpus-on-tr16 |
|---:|---:|---:|
| 2,048 | 202 | 156 |
| 8,192 | 249 | 338 |
| 32,768 | 290 | 299 |

At 8K the dense projections (287 M weights per layer on one card at a time) take about 15 s and the expert GEMMs 10-26 s
of the total; the experts stream once per chunk (2.5 GB per layer over 14.3 GB/s and ~13 GB/s, or the IB link for a
node without a card).

### What the forecast does not know

- Muse routing: the GPU-cache hit rates use GLM-5.3-Flash's measured LRU curve (sensitivity above).
- The GPU side: the trunk GEMV rates (Q6_K/Q8_0 at hidden 8192), the GQA attention and an IQ1_S GPU expert kernel are
  taken from the GLM runner's measured rates or estimated; prefill attention (20/15 TFLOP/s) is an estimate.
- The B550's engine-level IQ1_S rate (estimated from its quiet vec_dot run and TR16's engine/vec_dot ratio).
- Quality: nothing here measures what UD-IQ1_S does to the model.

## The runner: plan on top of the GLM one

Done in this change (all without the weights, no GPU needed for the tests):

| piece | where | checked by |
|---|---|---|
| census of a GGUF from its headers | `tools/muse_gguf_census.py` | the 175,626,404,864 bytes of the release |
| the simulator | `tools/sim/muse.py`, `tools/muse_sim.py`; `model.py` registers the geometry (`glm_sim.py --model muse`: single-node sizing only) | `tools/test_muse_sim.py` |
| block tensor names, shapes, formats, the expert row split (C11, shared by every program) | `include/strata/host/muse_names.h`, `src/host/muse_names.c` | `muse_names_test` |
| geometry from the keys, tensor directory check, footprint, row-split plan | `include/strata/artifact/muse_model.hpp`, `strata-muse-inspect` | `muse_model_test` (synthetic file; `STRATA_MUSE_HEADERS=dir` adds the real headers) |
| IQ1_S in the reader's block table | `include/strata/artifact/gguf_reader.hpp` | `muse_model_test` |
| IQ1_S CPU expert kernel, opt-in | `iq_avx2.cpp` `Fmt32<19>`, `native_expert.cpp` (`NativeFmt::iq1s_rows`), `kNativeFF` 4096 | `iq1s_expert_parity`, `iq_avx2_parity` |
| the block's arithmetic as an f32 reference | `include/strata/muse/reference.hpp` | `muse_reference_test` |

Next, in order (each with a parity test against `reference.hpp` before it is timed):

1. **One layer for real.** Fetch one layer's tensors by HTTP range (about 750 MB: the trunk plus three expert tensors)
   and check `reference.hpp` with dequantized weights against llama.cpp's `muse-spark` graph on the same input
   (`llama-eval-callback` dumps), so the reference is checked against the original before anything is built on it.
2. **The GPU trunk** (CUDA and HIP): the GLM runner's dense GEMVs (Q6_K, Q8_0) for q/k/v/gate/o and the latent
   projections; a GQA decode attention with a 2048-position ring for the 46 sliding-window layers and a full cache for
   the 16 others (64 KB per token at f16); the router (sigmoid + bias top-8) on the GPU as GLM's; CUDA graphs per layer
   range. The trunk is split by layer across the two cards (32/30 for decode).
3. **The CPU expert lane**: the existing `ExpertPool` with `NativeFmt::iq1s_rows` on 4096-wide latents; the GPU-resident
   hot experts need an IQ1_S tier kernel (llama.cpp's MMVQ IQ1_S is the reference).
4. **Two nodes**: extend the remote TP worker (`glm_tp_worker.cpp`, ucomm over IB) for the row split. A request is the
   latent (16 KB f32) and the routes; because `ffn_post_exp_norm` normalizes each expert's whole output, the worker
   returns one partial latent per (token, expert), about 88 KB per layer, and the node that owns the layer adds the
   partials per expert before the norm and the router weight. With split-gpus the layer owner alternates between the
   nodes (layers 0-31 on TR16's card, 32-61 on the B550's): the hidden state crosses the link twice per token.
5. **Prefill**: chunks of 4-8K, each node's card streaming the experts its node holds; activations over IB.
6. **Levers**, ranked by the forecast: a faster IQ1_S CPU kernel (up to 9.1 tok/s if it reached DRAM speed), Q4_K trunk
   copies (+1.3 tok/s, lossy), top-6 routing (+1.1, lossy).

## Reproduce

```
# headers only (see tools/muse_gguf_census.py for the curl lines)
python3 tools/muse_gguf_census.py s1.gguf h2.gguf h3.gguf h4.gguf h5.gguf --name ud-iq1s --output tools/sim/data/census-muse-spark-ud-iq1s.json
strata-muse-inspect s1.gguf h2.gguf h3.gguf h4.gguf h5.gguf
python3 tools/muse_sim.py report
python3 tools/muse_sim.py decode --gpu-layers 32,30 --set b550.cpu_gmacs.iq256=20
iq1s_expert_parity && iq1s_expert_parity --bench --threads 15 --mb 1024 --nt 1,2,4
```
