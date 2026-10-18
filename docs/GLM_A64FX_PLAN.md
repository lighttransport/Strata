# GLM-5.3-Flash on A64FX: simulated decode and prefill at 6 and 12 nodes, and a runner proposal

Written 2026-10-18. Everything below that is not marked "measured" is a simulator output: an estimate with a stated
calibration band and a printed list of hypotheses. The first version ran nothing on the cluster; on 2026-10-18 one-node
and 12-node probes measured the expert kernels, barriers, HBM stream and the collectives at 6/8/12 ranks, and the
simulator and the tables below use those numbers (section 5).

The question: Strata's GLM-5.3-Flash Q2Q3 expert pack (`experts-q23-fp8.gguf`, 110.2 GB, Q2_K gate/up + Q3_K down
on all 42 MoE layers) on A64FX nodes with 32 GiB of HBM2 each, 6 nodes (192 GiB) or 12 nodes (384 GiB). What decode
and prefill speed is reachable, with which layout, and what runner would get there.

## 1. Where the numbers come from

Three bodies of measured work feed the model; none was redone.

- **The native 12-node runner** in the gemm repository (`a64fx/glm5`, one MPI rank per node, 47 OpenMP threads,
  uTofu collectives, UD-Q4_K_XL experts with a Q8_0 trunk). Qualified: **35.46 decode / 412.27 prefill tok/s** at
  8K (GLM53F_STRATA.md:447,502), 39.90 / 285.31 at a 128-token prompt, 31.70 / 372.39 at 32K (:504-505). Best opt-in:
  **53.35 tok/s with MTP depth 2** at 512 tokens (handoff-decode100-resume-20261004.md:11). Plain decode is 28 ms per
  token split as mHC 6.6, KDA 7.1, MoE 8.0 (local 4.9, allreduce 2.8), sparse attention 4.5, head/embed/dense 0.9 ms
  (:78-85). Per rank per token: 1.55 GB of weights (experts 429 MB, KDA 315 MB, sparse front 272 MB replicated),
  1,482 OpenMP barriers (4.9 ms of waiting), 90 allreduces (handoff-decode100-20261004.md:29-31). fapp shows no
  region is HBM-bound (IPC 0.5-0.7, under 12 % L2-miss wait): the runner is 7x off its 4 ms bandwidth floor because of
  synchronization and small per-thread work, not because of HBM (:99-110).
- **Tofu-D and node micro-benchmarks** (`a64fx/utofu-tests/summary.md`, 12-node 2x3x2 torus): Put 0.97-1.13 us one
  way, 6.34 GB/s per link, per hop 1.23 us + bytes / 6.36 GB/s, recursive-doubling allreduce of 16 KiB 23.47 us fp32 /
  13.61 us bf16 (5 steps at 12 ranks), MoE dispatch + combine at B=1 34.6 us over multi-TNI Puts, all-to-all 86 GB/s
  aggregate uniform / 36 GB/s under routing skew, 8 MiB reduce-scatter/all-gather 5.7 ms (glm5/README.md:25). Node
  read bandwidth 807-915 GB/s with per-CMG buffers (glm5/matvec.md:5, summary:88); standalone cold Q8 matvec 351 GB/s
  (handoff-decode100-20261004.md:52); int8 SDOT GEMM 1.87 / 2.60 / 3.06 / 3.18 TOPS per node at M = 8 / 16 / 32 / 64
  (glm5/CALIBRATION.md).
- **The one-node decode chain** of the clair simulator campaign (`a64fx/llm-guided-opt/sim-accuracy/glm53f/STATUS.md:266-335`):
  the full per-rank chain of 270 GEMV stages (1.56 GB) runs in **2.75 ms = 63 % of the 900.8 GB/s roof** with split-phase
  barriers (1.3 us) and 16 KiB prefetch, roof 1.74 ms. This is the measured efficiency a well-structured streaming
  runner reaches on one node, and it anchors the `redesign` profile below. The clair qlair simulator itself is not
  used for throughput: all its backends are unqualified (stage errors -1 to -43 %), its cluster mode has one scalar
  collective, and the model does not fit an instruction-level run.

The model's expert byte census is Strata's own (`tools/sim/model.py`): 3.09 GB of routed experts per token for q23
with no reuse, 6.45 GB of trunk per token at the UD Q5_K/Q6_K mix (1.49x as Q8_0, 0.86x as Q4_K). The UD-Q4_K_XL pack
reconstructed from the same census gives 15.33 GB per rank and 426 MB per rank per token against the measured 15 GB
and 429 MB.

## 2. The model (`tools/sim/a64fx.py`, CLI `tools/glm_a64fx_sim.py`)

A decode step is the sum over the 45 layers of mixer (KDA or sparse MLA), two mHC sites, router, shared expert,
routed experts and the collectives, plus the head. Every term is `fixed + bytes / rate` and carries a node-count
scaling class: **S** streamed bytes divided by the trunk split (twice the bytes per rank at 6 nodes), **C** constant
(replicated tensors, per-phase latency), **L** collective (`ceil(log2 N)` tree steps, plus one for a non-power-of-two
count: 5 at 12, 4 at 6, 7 at 48, plus the straggler wait), **X** context (fitted on the 128 / 8K / 32K rows: 0.39 us
per token below 8K, 0.136 above). Routed experts are split into `parts` slices per expert over that many ranks
(`parts = 1` whole experts, `parts = ranks` every rank holds a slice of every expert; the runner today uses 4 of 12);
the busiest rank's load over the mean comes from the Monte-Carlo of `glm53f_expert_shard_sim.py` (3.35x at 12 ranks
whole experts, 1.52x at 4 slices, 1.0 at 12), narrower slices pay the measured penalty 1.0 / 1.16 / 1.37 at 1 / 2 / 4
slices, and the combine waits for the busiest rank. MTP rounds use `routing.tokens_per_round` with two measured
acceptance fixtures (BEST binary 42/43 per position; the legacy sweep 96/128, 147/256, 163/384, 163/512) and a
per-binary verify width curve. Prefill is an int8 SDOT GEMM roofline at the chunk's M per expert plus the trunk
reduce-scatter/all-gather per layer (bandwidth-bound, nearly node-count-invariant), a per-layer fixed cost and a
context term. Memory fit counts expert share, trunk share, replicated tensors, KV (26.1 KB per token) and KDA state,
MTP block and scratch against 29 GiB usable per node (7 GiB per CMG rank).

Two profiles: **`current`** is the runner as it is, fitted stage by stage to the 28 ms split (its fixed costs do not
shrink with fewer nodes; its per-phase overheads are inside the stage times and are only reported). **`redesign`** is
the proposed runner: streaming at the one-node 63 % of roof (504 GB/s), mHC at its 23 us standalone kernel time per
site, KDA small steps as one full-team phase, 8 phases per MoE layer at a 1.38 us split-phase barrier (the measured libhwb barrier) + 1.6 us dispatch,
one collective per layer where the layout allows, 5 us straggler wait, no overlap credit. Its unmeasured constants
are printed with every report.

### Validation against the measured 12-node rows

```
record                                                          measured predicted   error  ok
TP12 Q4 plain decode, 8K context (qualified capacity4096)          35.46     34.22   -3.5%  yes
TP12 Q4 plain decode, 128-token prompt                             39.90     38.35   -3.9%  yes
TP12 Q4 plain decode, repeated 32K                                 31.70     30.78   -2.9%  yes
TP12 Q4 plain decode, 512 context, BEST option set (stage spli     35.70     35.99    0.8%  yes
   stages (predicted/measured ms): mhc 6.55/6.6  kda 7.08/7.1  moe 8.66/8.0  sparse 4.51/4.5  head 0.98/0.9
TP12 Q4 MTP depth 2, 512 context, BEST option set                  53.35     53.32   -0.1%  yes
TP12 Q4 MTP depth 3, 512 context, BEST option set                  47.07     48.82    3.7%  yes
TP12 Q4 MTP depth 1, legacy sweep binary                           28.99     30.30    4.5%  yes
TP12 Q4 MTP depth 2, legacy sweep binary                           26.20     26.03   -0.6%  yes
TP12 Q4 MTP depth 3, legacy sweep binary                           22.00     21.19   -3.7%  yes
TP12 Q4 MTP depth 4, legacy sweep binary                           17.90     17.22   -3.8%  yes
TP12 prefill 8K, chunk 4096 (qualified)                           412.27    406.51   -1.4%  yes
TP12 prefill 128 tokens                                           285.31    280.88   -1.6%  yes
TP12 prefill repeated 32K, chunk 4096                             372.39    368.42   -1.1%  yes
13 of 13 rows within tolerance
```

Every row is within 5 %; the records and their sources are in `tools/sim/data/measured_a64fx_12n.json`. The
`current` profile owes this to being fitted on these rows, so its 12-node columns below reproduce the runner by
construction; its 6-node and q23 columns are the forecast. The `redesign` profile is not fitted to any whole-model
row; its anchor is the one-node chain.

## 3. Results

Pack q23_fp8 (110.2 GB routed experts), q8 trunk. Simulator outputs, not measurements.

### Memory per rank (GiB)

| nodes x ranks | pack | experts | trunk+replicated | KV 32K | total | usable | fits |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6 x1 | q23_fp8 | 17.1 | 1.8 | 0.94 | 21.2 | 29 | yes |
| 6 x1 | ud_q4kxl | 28.5 | 1.8 | 0.94 | 33.0 | 29 | **no** |
| 6 x4 | q23_fp8 | 4.3 | 0.6 | 0.94 | 6.2 | 7 | yes |
| 6 x4 | ud_q4kxl | 7.1 | 0.6 | 0.94 | 9.1 | 7 | **no** |
| 8 x1 | q23_fp8 | 12.8 | 1.4 | 0.94 | 16.5 | 29 | yes |
| 8 x1 | ud_q4kxl | 21.4 | 1.4 | 0.94 | 25.3 | 29 | yes |
| 8 x4 | q23_fp8 | 3.2 | 0.5 | 0.94 | 5.0 | 7 | yes |
| 8 x4 | ud_q4kxl | 5.4 | 0.5 | 0.94 | 7.2 | 7 | **no** |
| 12 x1 | q23_fp8 | 8.6 | 1.0 | 0.94 | 11.7 | 29 | yes |
| 12 x1 | ud_q4kxl | 14.3 | 1.0 | 0.94 | 17.6 | 29 | yes |
| 12 x4 | q23_fp8 | 2.1 | 0.4 | 0.94 | 3.8 | 7 | yes |
| 12 x4 | ud_q4kxl | 3.6 | 0.4 | 0.94 | 5.3 | 7 | yes |

### Decode, profile `current` (tok/s)

| nodes | layout | 128 ctx plain / MTP2 best / MTP2 sweep | 8K | 32K | 8K step ms |
|---|---:|---:|---:|---:|---:|
| 6 | sliced over 4 ranks (runner today) | 32.2 / 47.7 / 35.0 | 29.2 / 43.3 / 31.8 | 26.7 / 39.6 / 29.0 | 34.2 = trunk 5.4 + experts 8.9 + coll 4.2 + mHC 5.9 + ctx 3.2 + head 1.3 |
| 6 | sliced over every rank | 32.2 / 47.8 / 35.0 | 29.3 / 43.4 / 31.8 | 26.7 / 39.6 / 29.0 | 34.2 = trunk 5.4 + experts 10.5 + coll 2.7 + mHC 5.9 + ctx 3.2 + head 1.3 |
| 6 | whole experts (EP) | 30.7 / 45.4 / 33.3 | 28.0 / 41.4 / 30.4 | 25.6 / 38.0 / 27.8 | 35.8 = trunk 5.4 + experts 6.5 + coll 8.2 + mHC 5.9 + ctx 3.2 + head 1.3 |
| 6 | sliced over 2 ranks | 32.6 / 48.3 / 35.4 | 29.6 / 43.8 / 32.1 | 27.0 / 40.0 / 29.3 | 33.8 = trunk 5.4 + experts 7.3 + coll 5.4 + mHC 5.9 + ctx 3.2 + head 1.3 |
| 6 | sliced over 4 ranks (runner today) x4 CMG ranks | 23.3 / 34.5 / 25.3 | 21.7 / 32.1 / 23.5 | 20.3 / 30.0 / 22.0 | 46.1 = trunk 10.7 + experts 8.9 + coll 11.0 + mHC 5.9 + ctx 3.2 + head 1.3 |
| 6 | sliced over every rank x4 CMG ranks | 19.5 / 28.9 / 21.2 | 18.4 / 27.3 / 20.0 | 17.4 / 25.7 / 18.9 | 54.3 = trunk 10.7 + experts 24.6 + coll 3.5 + mHC 5.9 + ctx 3.2 + head 1.3 |
| 6 | whole experts (EP) x4 CMG ranks | 18.7 / 27.7 / 20.3 | 17.7 / 26.2 / 19.2 | 16.7 / 24.8 / 18.1 | 56.6 = trunk 10.7 + experts 6.5 + coll 23.8 + mHC 5.9 + ctx 3.2 + head 1.3 |
| 6 | sliced over 2 ranks x4 CMG ranks | 21.6 / 32.0 / 23.5 | 20.2 / 30.0 / 22.0 | 19.0 / 28.1 / 20.6 | 49.4 = trunk 10.7 + experts 7.3 + coll 15.8 + mHC 5.9 + ctx 3.2 + head 1.3 |
| 8 | sliced over 4 ranks (runner today) | 36.8 / 54.5 / 39.9 | 32.9 / 48.8 / 35.8 | 29.7 / 44.1 / 32.3 | 30.3 = trunk 4.7 + experts 6.7 + coll 3.4 + mHC 5.9 + ctx 3.2 + head 1.1 |
| 8 | sliced over every rank | 35.4 / 52.4 / 38.4 | 31.8 / 47.2 / 34.6 | 28.8 / 42.7 / 31.3 | 31.4 = trunk 4.7 + experts 9.0 + coll 2.1 + mHC 5.9 + ctx 3.2 + head 1.1 |
| 8 | whole experts (EP) | 33.7 / 49.9 / 36.6 | 30.5 / 45.1 / 33.1 | 27.7 / 41.1 / 30.1 | 32.8 = trunk 4.7 + experts 4.9 + coll 7.7 + mHC 5.9 + ctx 3.2 + head 1.1 |
| 8 | sliced over 2 ranks | 36.1 / 53.4 / 39.2 | 32.4 / 48.0 / 35.2 | 29.3 / 43.4 / 31.8 | 30.9 = trunk 4.7 + experts 5.5 + coll 5.1 + mHC 5.9 + ctx 3.2 + head 1.1 |
| 8 | sliced over 4 ranks (runner today) x4 CMG ranks | 25.2 / 37.4 / 27.4 | 23.4 / 34.7 / 25.4 | 21.7 / 32.2 / 23.6 | 42.7 = trunk 10.0 + experts 6.7 + coll 10.6 + mHC 5.9 + ctx 3.2 + head 1.1 |
| 8 | sliced over every rank x4 CMG ranks | 20.6 / 30.5 / 22.4 | 19.3 / 28.7 / 21.0 | 18.2 / 27.0 / 19.8 | 51.7 = trunk 10.0 + experts 23.2 + coll 3.1 + mHC 5.9 + ctx 3.2 + head 1.1 |
| 8 | whole experts (EP) x4 CMG ranks | 20.2 / 29.9 / 21.9 | 19.0 / 28.1 / 20.6 | 17.9 / 26.5 / 19.4 | 52.7 = trunk 10.0 + experts 4.9 + coll 22.4 + mHC 5.9 + ctx 3.2 + head 1.1 |
| 8 | sliced over 2 ranks x4 CMG ranks | 23.3 / 34.5 / 25.3 | 21.7 / 32.1 / 23.6 | 20.3 / 30.0 / 22.0 | 46.1 = trunk 10.0 + experts 5.5 + coll 15.2 + mHC 5.9 + ctx 3.2 + head 1.1 |
| 12 | sliced over 4 ranks (runner today) | 39.2 / 58.1 / 42.6 | 34.9 / 51.8 / 37.9 | 31.4 / 46.5 / 34.1 | 28.6 = trunk 4.0 + experts 4.4 + coll 4.7 + mHC 5.9 + ctx 3.2 + head 0.9 |
| 12 | sliced over every rank | 37.1 / 55.0 / 40.3 | 33.2 / 49.2 / 36.1 | 30.0 / 44.4 / 32.6 | 30.1 = trunk 4.0 + experts 7.6 + coll 3.0 + mHC 5.9 + ctx 3.2 + head 0.9 |
| 12 | whole experts (EP) | 35.5 / 52.6 / 38.6 | 32.0 / 47.3 / 34.7 | 28.9 / 42.9 / 31.4 | 31.3 = trunk 4.0 + experts 3.3 + coll 8.5 + mHC 5.9 + ctx 3.2 + head 0.9 |
| 12 | sliced over 2 ranks | 38.3 / 56.7 / 41.6 | 34.2 / 50.6 / 37.1 | 30.8 / 45.6 / 33.4 | 29.3 = trunk 4.0 + experts 3.7 + coll 6.1 + mHC 5.9 + ctx 3.2 + head 0.9 |
| 12 | sliced over 4 ranks (runner today) x4 CMG ranks | 26.7 / 39.6 / 29.0 | 24.7 / 36.5 / 26.8 | 22.8 / 33.8 / 24.8 | 40.5 = trunk 9.3 + experts 4.4 + coll 11.4 + mHC 5.9 + ctx 3.2 + head 0.9 |
| 12 | sliced over every rank x4 CMG ranks | 21.2 / 31.4 / 23.0 | 19.9 / 29.4 / 21.6 | 18.7 / 27.7 / 20.3 | 50.3 = trunk 9.3 + experts 21.7 + coll 3.9 + mHC 5.9 + ctx 3.2 + head 0.9 |
| 12 | whole experts (EP) x4 CMG ranks | 21.6 / 31.9 / 23.4 | 20.2 / 29.9 / 21.9 | 18.9 / 28.1 / 20.6 | 49.5 = trunk 9.3 + experts 3.3 + coll 21.6 + mHC 5.9 + ctx 3.2 + head 0.9 |
| 12 | sliced over 2 ranks x4 CMG ranks | 24.7 / 36.6 / 26.8 | 22.9 / 34.0 / 24.9 | 21.3 / 31.6 / 23.2 | 43.6 = trunk 9.3 + experts 3.7 + coll 15.3 + mHC 5.9 + ctx 3.2 + head 0.9 |

### Prefill, profile `current` (tok/s)

| nodes | layout | 128 tokens | 8K (chunk 4096) | 32K | 8K ms/position |
|---|---:|---:|---:|---:|---:|
| 6 | sliced over 4 ranks (runner today) | 167 | 262 | 245 | 3.82 = compute 2.82 + coll 0.91 + fixed 0.01 + ctx 0.09 |
| 6 | sliced over every rank | 167 | 262 | 245 | 3.82 = compute 2.82 + coll 0.91 + fixed 0.01 + ctx 0.09 |
| 6 | whole experts (EP) | 166 | 258 | 242 | 3.88 = compute 2.82 + coll 0.96 + fixed 0.01 + ctx 0.09 |
| 6 | sliced over 2 ranks | 166 | 258 | 242 | 3.88 = compute 2.82 + coll 0.96 + fixed 0.01 + ctx 0.09 |
| 6 | sliced over 4 ranks (runner today) x4 CMG ranks | 162 | 249 | 234 | 4.02 = compute 2.82 + coll 1.11 + fixed 0.01 + ctx 0.09 |
| 6 | sliced over every rank x4 CMG ranks | 163 | 253 | 237 | 3.96 = compute 2.82 + coll 1.05 + fixed 0.01 + ctx 0.09 |
| 6 | whole experts (EP) x4 CMG ranks | 162 | 249 | 234 | 4.02 = compute 2.82 + coll 1.11 + fixed 0.01 + ctx 0.09 |
| 6 | sliced over 2 ranks x4 CMG ranks | 162 | 249 | 234 | 4.02 = compute 2.82 + coll 1.11 + fixed 0.01 + ctx 0.09 |
| 8 | sliced over 4 ranks (runner today) | 207 | 316 | 293 | 3.16 = compute 2.11 + coll 0.96 + fixed 0.01 + ctx 0.09 |
| 8 | sliced over every rank | 207 | 316 | 293 | 3.16 = compute 2.11 + coll 0.96 + fixed 0.01 + ctx 0.09 |
| 8 | whole experts (EP) | 205 | 311 | 288 | 3.22 = compute 2.11 + coll 1.01 + fixed 0.01 + ctx 0.09 |
| 8 | sliced over 2 ranks | 205 | 311 | 288 | 3.22 = compute 2.11 + coll 1.01 + fixed 0.01 + ctx 0.09 |
| 8 | sliced over 4 ranks (runner today) x4 CMG ranks | 200 | 300 | 279 | 3.33 = compute 2.11 + coll 1.12 + fixed 0.01 + ctx 0.09 |
| 8 | sliced over every rank x4 CMG ranks | 203 | 306 | 284 | 3.27 = compute 2.11 + coll 1.06 + fixed 0.01 + ctx 0.09 |
| 8 | whole experts (EP) x4 CMG ranks | 200 | 300 | 279 | 3.33 = compute 2.11 + coll 1.12 + fixed 0.01 + ctx 0.09 |
| 8 | sliced over 2 ranks x4 CMG ranks | 200 | 300 | 279 | 3.33 = compute 2.11 + coll 1.12 + fixed 0.01 + ctx 0.09 |
| 12 | sliced over 4 ranks (runner today) | 268 | 390 | 355 | 2.56 = compute 1.41 + coll 1.06 + fixed 0.01 + ctx 0.09 |
| 12 | sliced over every rank | 272 | 399 | 362 | 2.51 = compute 1.41 + coll 1.00 + fixed 0.01 + ctx 0.09 |
| 12 | whole experts (EP) | 268 | 390 | 355 | 2.56 = compute 1.41 + coll 1.06 + fixed 0.01 + ctx 0.09 |
| 12 | sliced over 2 ranks | 268 | 390 | 355 | 2.56 = compute 1.41 + coll 1.06 + fixed 0.01 + ctx 0.09 |
| 12 | sliced over 4 ranks (runner today) x4 CMG ranks | 262 | 379 | 346 | 2.64 = compute 1.41 + coll 1.13 + fixed 0.01 + ctx 0.09 |
| 12 | sliced over every rank x4 CMG ranks | 267 | 389 | 354 | 2.57 = compute 1.41 + coll 1.07 + fixed 0.01 + ctx 0.09 |
| 12 | whole experts (EP) x4 CMG ranks | 262 | 379 | 346 | 2.64 = compute 1.41 + coll 1.13 + fixed 0.01 + ctx 0.09 |
| 12 | sliced over 2 ranks x4 CMG ranks | 262 | 379 | 346 | 2.64 = compute 1.41 + coll 1.13 + fixed 0.01 + ctx 0.09 |

Hypotheses in `current`: expert_q2k_gbps 78 GB/s is ggml's standalone SVE rate; the runner's in-model Q2 rate is not measured; context slope applied as one term on the sparse layers; 6-node attention split: 64 heads / 6 is not integral (TP-6 needs a 60+4 or TP-4 split)

### Decode, profile `redesign` (tok/s)

| nodes | layout | 128 ctx plain / MTP2 best / MTP2 sweep | 8K | 32K | 8K step ms |
|---|---:|---:|---:|---:|---:|
| 6 | sliced over 4 ranks (runner today) | 73.7 / 97.0 / 71.1 | 66.1 / 82.9 / 60.8 | 59.6 / 72.1 / 52.8 | 15.1 = trunk 2.3 + experts 3.7 + coll 2.3 + mHC 1.9 + ctx 1.6 + head 1.0 + phases 1.5 |
| 6 | sliced over every rank | 74.9 / 99.8 / 73.1 | 67.0 / 84.9 / 62.3 | 60.4 / 73.6 / 53.9 | 14.9 = trunk 2.3 + experts 4.0 + coll 1.8 + mHC 1.9 + ctx 1.6 + head 1.0 + phases 1.5 |
| 6 | whole experts (EP) | 68.2 / 85.7 / 62.8 | 61.6 / 74.5 / 54.6 | 56.0 / 65.6 / 48.1 | 16.2 = trunk 2.3 + experts 3.3 + coll 3.8 + mHC 1.9 + ctx 1.6 + head 1.0 + phases 1.5 |
| 6 | sliced over 2 ranks | 72.7 / 94.6 / 69.3 | 65.3 / 81.2 / 59.5 | 59.0 / 70.7 / 51.8 | 15.3 = trunk 2.3 + experts 3.4 + coll 2.7 + mHC 1.9 + ctx 1.6 + head 1.0 + phases 1.5 |
| 6 | sliced over 4 ranks (runner today) x4 CMG ranks | 68.5 / 81.2 / 59.5 | 61.8 / 71.1 / 52.1 | 56.1 / 63.0 / 46.1 | 16.2 = trunk 2.2 + experts 3.7 + coll 4.8 + mHC 1.4 + ctx 1.6 + head 1.0 + phases 1.1 |
| 6 | sliced over every rank x4 CMG ranks | 66.7 / 81.1 / 59.4 | 60.4 / 71.0 / 52.1 | 54.9 / 62.9 / 46.1 | 16.6 = trunk 2.2 + experts 6.3 + coll 2.7 + mHC 1.4 + ctx 1.6 + head 1.0 + phases 1.1 |
| 6 | whole experts (EP) x4 CMG ranks | 50.5 / 63.1 / 46.3 | 46.8 / 56.8 / 41.7 | 43.5 / 51.5 / 37.8 | 21.4 = trunk 2.2 + experts 3.3 + coll 10.4 + mHC 1.4 + ctx 1.6 + head 1.0 + phases 1.1 |
| 6 | sliced over 2 ranks x4 CMG ranks | 61.7 / 70.1 / 51.4 | 56.2 / 62.4 / 45.8 | 51.5 / 56.1 / 41.1 | 17.8 = trunk 2.2 + experts 3.4 + coll 6.7 + mHC 1.4 + ctx 1.6 + head 1.0 + phases 1.1 |
| 8 | sliced over 4 ranks (runner today) | 87.2 / 119.0 / 87.2 | 76.7 / 98.5 / 72.2 | 68.1 / 83.5 / 61.2 | 13.0 = trunk 1.8 + experts 2.8 + coll 1.7 + mHC 1.9 + ctx 1.6 + head 0.9 + phases 1.5 |
| 8 | sliced over every rank | 87.2 / 119.9 / 87.9 | 76.7 / 99.1 / 72.6 | 68.1 / 84.0 / 61.6 | 13.0 = trunk 1.8 + experts 3.2 + coll 1.3 + mHC 1.9 + ctx 1.6 + head 0.9 + phases 1.5 |
| 8 | whole experts (EP) | 78.3 / 100.2 / 73.5 | 69.7 / 85.3 / 62.5 | 62.6 / 73.8 / 54.1 | 14.3 = trunk 1.8 + experts 2.5 + coll 3.3 + mHC 1.9 + ctx 1.6 + head 0.9 + phases 1.5 |
| 8 | sliced over 2 ranks | 84.1 / 112.0 / 82.1 | 74.3 / 93.6 / 68.6 | 66.2 / 80.0 / 58.7 | 13.5 = trunk 1.8 + experts 2.6 + coll 2.3 + mHC 1.9 + ctx 1.6 + head 0.9 + phases 1.5 |
| 8 | sliced over 4 ranks (runner today) x4 CMG ranks | 78.3 / 93.0 / 68.1 | 69.7 / 80.0 / 58.6 | 62.6 / 69.8 / 51.2 | 14.3 = trunk 1.7 + experts 2.8 + coll 4.5 + mHC 1.4 + ctx 1.6 + head 0.9 + phases 1.1 |
| 8 | sliced over every rank x4 CMG ranks | 75.5 / 92.0 / 67.4 | 67.5 / 79.2 / 58.1 | 60.8 / 69.3 / 50.8 | 14.8 = trunk 1.7 + experts 5.5 + coll 2.3 + mHC 1.4 + ctx 1.6 + head 0.9 + phases 1.1 |
| 8 | whole experts (EP) x4 CMG ranks | 55.5 / 70.0 / 51.3 | 51.0 / 62.4 / 45.7 | 47.1 / 56.0 / 41.1 | 19.6 = trunk 1.7 + experts 2.5 + coll 10.0 + mHC 1.4 + ctx 1.6 + head 0.9 + phases 1.1 |
| 8 | sliced over 2 ranks x4 CMG ranks | 65.6 / 86.4 / 63.3 | 59.5 / 75.1 / 55.0 | 54.2 / 66.1 / 48.4 | 16.8 = trunk 1.7 + experts 2.6 + coll 7.1 + mHC 1.4 + ctx 1.6 + head 0.9 + phases 1.1 |
| 12 | sliced over 4 ranks (runner today) | 90.6 / 117.6 / 86.2 | 79.3 / 97.5 / 71.5 | 70.2 / 82.9 / 60.7 | 12.6 = trunk 1.3 + experts 1.9 + coll 2.7 + mHC 1.9 + ctx 1.6 + head 0.8 + phases 1.5 |
| 12 | sliced over every rank | 90.5 / 118.5 / 86.9 | 79.2 / 98.2 / 72.0 | 70.2 / 83.3 / 61.1 | 12.6 = trunk 1.3 + experts 2.4 + coll 2.2 + mHC 1.9 + ctx 1.6 + head 0.8 + phases 1.5 |
| 12 | whole experts (EP) | 80.9 / 99.5 / 72.9 | 71.8 / 84.7 / 62.1 | 64.2 / 73.4 / 53.8 | 13.9 = trunk 1.3 + experts 1.7 + coll 4.2 + mHC 1.9 + ctx 1.6 + head 0.8 + phases 1.5 |
| 12 | sliced over 2 ranks | 87.3 / 111.1 / 81.4 | 76.8 / 93.0 / 68.2 | 68.2 / 79.6 / 58.3 | 13.0 = trunk 1.3 + experts 1.8 + coll 3.2 + mHC 1.9 + ctx 1.6 + head 0.8 + phases 1.5 |
| 12 | sliced over 4 ranks (runner today) x4 CMG ranks | 82.1 / 94.9 / 69.6 | 72.7 / 81.4 / 59.6 | 65.0 / 70.9 / 52.0 | 13.8 = trunk 1.2 + experts 1.9 + coll 5.3 + mHC 1.4 + ctx 1.6 + head 0.8 + phases 1.1 |
| 12 | sliced over every rank x4 CMG ranks | 78.4 / 92.8 / 68.0 | 69.8 / 79.9 / 58.5 | 62.7 / 69.7 / 51.1 | 14.3 = trunk 1.2 + experts 4.7 + coll 3.1 + mHC 1.4 + ctx 1.6 + head 0.8 + phases 1.1 |
| 12 | whole experts (EP) x4 CMG ranks | 59.6 / 75.4 / 55.3 | 54.5 / 66.6 / 48.8 | 50.0 / 59.4 / 43.6 | 18.4 = trunk 1.2 + experts 1.7 + coll 10.1 + mHC 1.4 + ctx 1.6 + head 0.8 + phases 1.1 |
| 12 | sliced over 2 ranks x4 CMG ranks | 70.2 / 92.4 / 67.7 | 63.3 / 79.5 / 58.3 | 57.3 / 69.5 / 50.9 | 15.8 = trunk 1.2 + experts 1.8 + coll 7.5 + mHC 1.4 + ctx 1.6 + head 0.8 + phases 1.1 |

### Prefill, profile `redesign` (tok/s)

| nodes | layout | 128 tokens | 8K (chunk 4096) | 32K | 8K ms/position |
|---|---:|---:|---:|---:|---:|
| 6 | sliced over 4 ranks (runner today) | 303 | 473 | 443 | 2.12 = compute 1.61 + coll 0.46 + fixed 0.00 + ctx 0.05 |
| 6 | sliced over every rank | 303 | 473 | 443 | 2.12 = compute 1.61 + coll 0.46 + fixed 0.00 + ctx 0.05 |
| 6 | whole experts (EP) | 298 | 461 | 433 | 2.17 = compute 1.61 + coll 0.51 + fixed 0.00 + ctx 0.05 |
| 6 | sliced over 2 ranks | 298 | 461 | 433 | 2.17 = compute 1.61 + coll 0.51 + fixed 0.00 + ctx 0.05 |
| 6 | sliced over 4 ranks (runner today) x4 CMG ranks | 292 | 445 | 419 | 2.25 = compute 1.61 + coll 0.59 + fixed 0.00 + ctx 0.05 |
| 6 | sliced over every rank x4 CMG ranks | 297 | 458 | 430 | 2.18 = compute 1.61 + coll 0.52 + fixed 0.00 + ctx 0.05 |
| 6 | whole experts (EP) x4 CMG ranks | 292 | 445 | 419 | 2.25 = compute 1.61 + coll 0.59 + fixed 0.00 + ctx 0.05 |
| 6 | sliced over 2 ranks x4 CMG ranks | 292 | 445 | 419 | 2.25 = compute 1.61 + coll 0.59 + fixed 0.00 + ctx 0.05 |
| 8 | sliced over 4 ranks (runner today) | 379 | 576 | 533 | 1.74 = compute 1.21 + coll 0.48 + fixed 0.00 + ctx 0.05 |
| 8 | sliced over every rank | 379 | 576 | 533 | 1.74 = compute 1.21 + coll 0.48 + fixed 0.00 + ctx 0.05 |
| 8 | whole experts (EP) | 371 | 558 | 517 | 1.79 = compute 1.21 + coll 0.53 + fixed 0.00 + ctx 0.05 |
| 8 | sliced over 2 ranks | 371 | 558 | 517 | 1.79 = compute 1.21 + coll 0.53 + fixed 0.00 + ctx 0.05 |
| 8 | sliced over 4 ranks (runner today) x4 CMG ranks | 364 | 541 | 502 | 1.85 = compute 1.21 + coll 0.59 + fixed 0.00 + ctx 0.05 |
| 8 | sliced over every rank x4 CMG ranks | 372 | 559 | 519 | 1.79 = compute 1.21 + coll 0.53 + fixed 0.00 + ctx 0.05 |
| 8 | whole experts (EP) x4 CMG ranks | 364 | 541 | 502 | 1.85 = compute 1.21 + coll 0.59 + fixed 0.00 + ctx 0.05 |
| 8 | sliced over 2 ranks x4 CMG ranks | 364 | 541 | 502 | 1.85 = compute 1.21 + coll 0.59 + fixed 0.00 + ctx 0.05 |
| 12 | sliced over 4 ranks (runner today) | 492 | 706 | 643 | 1.42 = compute 0.80 + coll 0.56 + fixed 0.00 + ctx 0.05 |
| 12 | sliced over every rank | 506 | 737 | 668 | 1.36 = compute 0.80 + coll 0.50 + fixed 0.00 + ctx 0.05 |
| 12 | whole experts (EP) | 492 | 706 | 643 | 1.42 = compute 0.80 + coll 0.56 + fixed 0.00 + ctx 0.05 |
| 12 | sliced over 2 ranks | 492 | 706 | 643 | 1.42 = compute 0.80 + coll 0.56 + fixed 0.00 + ctx 0.05 |
| 12 | sliced over 4 ranks (runner today) x4 CMG ranks | 483 | 688 | 627 | 1.45 = compute 0.80 + coll 0.60 + fixed 0.00 + ctx 0.05 |
| 12 | sliced over every rank x4 CMG ranks | 498 | 719 | 653 | 1.39 = compute 0.80 + coll 0.54 + fixed 0.00 + ctx 0.05 |
| 12 | whole experts (EP) x4 CMG ranks | 483 | 688 | 627 | 1.45 = compute 0.80 + coll 0.60 + fixed 0.00 + ctx 0.05 |
| 12 | sliced over 2 ranks x4 CMG ranks | 483 | 688 | 627 | 1.45 = compute 0.80 + coll 0.60 + fixed 0.00 + ctx 0.05 |

Hypotheses in `redesign`: expert_q2k_gbps and its width costs are the standalone SDOT kernels (one node); in-model rate unmeasured; mHC 23 us per site in-model (standalone kernel time; the in-model cost was never reproduced); KDA small steps 12 us as one full-team phase (today 82 us on 6 threads); prefill int8 efficiency 0.70 (today 0.40 fitted; the 94 % CMG microkernel exists only standalone); 4 ranks per node: collective costs scaled from the 6/8/12-rank measurements to 24/48 ranks

With a **Q4_K trunk** instead of Q8_0 (lossy on the trunk; the 0.86x byte scale of `model.DENSE_FORMATS`), the
`redesign` decode rows of the runner layout become:

| 6 | sliced over 4 ranks (runner today) | 79.0 / 100.6 / 73.7 | 70.3 / 85.5 / 62.7 | 63.0 / 74.0 / 54.3 | 14.2 = trunk 1.5 + experts 3.7 + coll 2.3 + mHC 1.9 + ctx 1.6 + head 0.8 + phases 1.5 |
| 8 | sliced over 4 ranks (runner today) | 92.7 / 123.0 / 90.2 | 80.9 / 101.2 / 74.2 | 71.4 / 85.5 / 62.7 | 12.4 = trunk 1.2 + experts 2.8 + coll 1.7 + mHC 1.9 + ctx 1.6 + head 0.7 + phases 1.5 |
| 12 | sliced over 4 ranks (runner today) | 94.5 / 120.2 / 88.1 | 82.3 / 99.3 / 72.8 | 72.5 / 84.1 / 61.7 | 12.2 = trunk 0.9 + experts 1.9 + coll 2.7 + mHC 1.9 + ctx 1.6 + head 0.7 + phases 1.5 |

### Reading the tables

1. **Six nodes are feasible only because of the Q2Q3 pack.** UD-Q4_K_XL needs 28.5 GiB of experts per node at 6 nodes
   and does not fit; q23-from-FP8 needs 17.1 GiB and leaves 8 GiB of margin with a 32K context. At 12 nodes either
   pack fits.
2. **The runner as it is, moved to 6 nodes with q23: about 30 tok/s decode (44 with MTP depth 2 at the BEST
   acceptance, 32 at the sweep acceptance) and about 260 tok/s prefill at 8K.** Twelve nodes with q23: about 35 / 52 /
   38 and 390. ggml's SVE Q2_K/Q3_K dots run at 97/56 GB/s per node (measured), slower per byte than the runner's
   Q4_K path, which cancels the smaller pack: q23 is no faster than Q4 on the current runner. Halving the node count costs only 15 % because 60 % of the current
   step is latency that does not scale with the node count at all (mHC 5.9, context 3.2, KDA small steps, router).
3. **The redesigned runner: about 79 tok/s plain at 12 nodes, 77 at 8 and 66 at 6 (8K context); 97 / 98 / 83 with
   MTP depth 2 at the BEST acceptance, 71 / 72 / 61 at the sweep acceptance; prefill about 700-740 at 12 nodes, 576 at
   8 and 473 at 6.** A Q4_K trunk adds 3-6 %. The step is 12.6-15 ms: trunk 1.3-2.3, experts 1.9-3.7, collectives
   1.7-2.7, mHC 1.9, context 1.6, phases 1.5, head 0.8-1.0. The expert term uses the measured SDOT kernels as the
   runner would build them (clang 21 -mcpu=a64fx, best kernel per width: 159.9 GB/s per node for the q23 mix; ggml's
   would cost 8 tok/s at 12 nodes) and their measured per-token cost in the MTP verify step (1.47x at 2 tokens, 1.74x
   at 3: the kernels are compute-bound, so extra tokens are not free; this is what pulled MTP2 from 107 to 97). Otherwise the gain over today comes from removing the sync-bound fixed costs (mHC 5.9
   -> 1.9, KDA fixed 3.5 -> 0.5, barriers 5.5 -> 1.5).
4. **How experts are sliced is not the lever at 6-12 nodes.** Slicing every active expert over 4 ranks (today), over
   every rank, or over 2 ranks lands within 3 % of each other under both profiles: wider slices save bandwidth per
   slice but the busiest rank waits longer. Whole-expert EP loses 5-8 % because top-8 of 288 over 6-12 owners is
   badly imbalanced (busiest rank 2.2-3.4x the mean). Keep intra-expert slicing; fix the synchronization.
5. **Four ranks per node (one per CMG) loses under both profiles**: every collective runs over 24-48 ranks (6-7 tree
   steps) and the replicated sparse-attention front is read by every CMG rank at a quarter of the bandwidth. Its only
   merit is smaller teams; the model gives it 0.7x fixed costs in `redesign` and it still comes out 4-20 % behind.
6. **MTP pays only with high acceptance.** With the BEST fixture (0.977 per position) depth 2 adds 25-35 %; with the
   legacy sweep fixture (0.75 / 0.53) it adds nothing or loses under `redesign` because the verify step's expert
   bytes grow with the window's union ratio (2.46x at width 3) while 2.27 tokens come out. The runner measured both
   outcomes on different binaries; which one applies depends on the draft head and the prompt.
7. **Eight nodes are the decode sweet spot.** Eight is a power of two, so every allreduce takes 3 tree steps instead
   of 4 at 6 nodes or 5 at 12, and 2x2x2 is a clean torus. Measured on 8 ranks of a 12-node allocation: the 16 KiB
   fp32 tree allreduce takes 12.5 us against 18.5 at 6 ranks and 22.6 at 12. Under `redesign` 8 nodes decode within
   3 % of 12 (13.0 vs 12.6 ms per step: 1.0 ms less in collectives pays for most of the extra expert bytes) and are
   ahead with MTP depth 2 (98.3 vs 97.4); under `current` they recover 33 of the 35 tok/s of 12 nodes. Prefill still
   scales with node count (576 vs 706 under `redesign`). Memory is 15.9 GiB per node.
8. **Prefill is compute-bound on int8 SDOT and collective-bound second.** At 12 nodes today: 1.41 ms per position of
   GEMM at an effective 1.42 TFLOPS per node (40 % of the measured 3.18 TOPS GEMM) plus 1.0 ms of reduce-scatter/
   all-gather that does not shrink with node count. The `redesign` 700+ assumes 70 % GEMM efficiency and one
   collective per layer in bf16; 2,000 tok/s would need more than 50 % of the int8 peak of 12 nodes with no collective
   cost and is out of reach of this design.

### Calibration band

The `current` 12-node rows are fitted (within 5 %). Moving to 6 nodes relies on the scaling classes; the pessimistic
edge, every streamed term scaling exactly 1/N with no fixed part, lands at 27-28 tok/s instead of 30. The `redesign`
numbers rest on the hypotheses printed under each table. The expert kernels are now measured standalone (SDOT, 159.9
GB/s per node for the q23 mix on one node; the same Q2_K kernel ran 9 % faster on another node); if they ran in-model
at ggml's rate instead (97 GB/s) the 12-node step grows 1.5 ms and decode drops to 71 tok/s, at 120 GB/s to 75. The
largest remaining unknown is the mHC in-model cost (23 us per site assumed; at today's 78 us the step grows 4.6 ms and
decode drops to 58 tok/s). Treat `redesign` as 58-85 tok/s at 12 nodes and 50-72 at 6 until the mHC phase is measured
in-model.

## 4. The runner: Strata ported to A64FX

Decision (2026-10-18): the runner is a port of Strata's engine to aarch64, not a new runner in the gemm repository.
What Strata has and lacks today:

- `strata-glm-decode` (`src/program/glm_decode.cpp`, 12.3k lines) is built only with CUDA or HIP and calls the CUDA
  runtime directly; the mixer (KDA, MLA), mHC, router, shared expert, dense layers and head exist as GPU kernels and
  as scalar references (`include/strata/kernels/glm_reference.hpp`).
- Portable CPU code: the expert pool (`include/strata/kernels/cpu/pool.hpp`), `native_expert.cpp` (ggml-cpu `vec_dot`
  fallback; upstream ggml has 512-bit SVE Q2_K/Q3_K dots in `ggml-cpu/arch/arm/quants.c`), the canonical expert
  path (`canon_expert.cpp`, bit-exact CPU/GPU), `strata_host` (C11), `ucomm` (C11; ib verbs or UDP; `UCOMM_MAX_WORLD`
  8), the remote expert worker (`include/strata/net/expert_tp.hpp`, `glm_tp_worker.cpp`).
- The **Vulkan runner** (`vulkan/`, 4.3k lines) is the only GLM runner without CUDA: a device runs projections, mHC,
  KDA and MLA while the CPU pool runs experts and norms/conv/SwiGLU run on the CPU; it already has batched prefill,
  a resident expert tier and verified speculative decoding behind flags. It is the template for the port.
- x86-only code to lift: `pool.cpp` (`immintrin.h`, `_mm_pause`, `__rdtsc`, UMWAIT), `expert_layout.cpp` (cpuid),
  the `*_avx*.cpp` kernels (guard by ISA), `exl3/math.hpp` and `ucomm_file.c` (guards exist), and the two-node NUMA
  split hard-coded in `ExpertJobMulti::numa[2]` and `numa_weights.hpp`.

The design, in the order the simulator says it matters:

1. **One process per node, four CMG teams.** Generalize the pool's NUMA split from 2 to `numa_nodes` (4), first-touch
   every weight on its CMG (`XOS_MMM_L_PAGING_POLICY=demand`), one worker team per CMG, a split-phase flag barrier
   (1.3 us measured at 48 threads) instead of park/unpark. Four processes per node is scored and rejected above.
2. **A CPU projection layer, `src/cpu_runner/`**, replacing `vulkan::Compute`: SVE kernels for Q8/Q4_K GEMV, Q2_K/Q3_K
   expert dots on int8 SDOT, KDA recurrence, MLA attention with the top-2048 indexer, mHC Sinkhorn, router and head,
   each parity-tested against `glm_reference.hpp` and the GPU kernels' recorded outputs. Every sub-layer is one phase
   whose rows are statically partitioned over all 48 cores to the HBM stream; no 6-thread phases. Target 8 phases per
   MoE layer against about 33 barriers today.
3. **Trunk tensor-parallel over all ranks, experts sliced over 4-12 ranks** (the simulator's best, equal within 3 %),
   the Q2Q3 experts taken straight from the Strata pack and sharded per node by `ucfile`; the sparse-attention front
   sharded too (today it is replicated: 272 MB per rank per token).
4. **One or two collectives per layer over a uTofu `ucomm` backend**: `ucomm_utofu.c` with one-sided Puts and
   memory-sentinel completion, multi-TNI Puts to distinct peers, bf16 tree allreduce as in
   `utofu-tests/tp_allreduce.h`; `UCOMM_MAX_WORLD` raised to 48; `ucomm_reduce_scatter`/`ucomm_allgather` and
   `ucomm_alltoallv` added for prefill; bootstrap from the MPI or `pjsub` rank environment instead of the TCP star.
   The MoE combine is folded into the next trunk allreduce where the dataflow allows.
5. **mHC and norms fused into one phase per site** (the 23 us standalone kernel; 70-86 us in-model today for reasons
   never found). This is the single largest term in the forecast and the first thing to measure in-model.
6. **MTP** reused from Strata's draft/verify path at depth 2 once plain decode is at or under 15 ms per token; the
   window's verify bytes grow with the union ratio, so acceptance under 0.9 per position does not pay.
7. **Prefill**: Strata's batched prefill with chunk 4096, Q2_K/Q3_K dequantized to int8 tiles once per chunk, int8
   SDOT GEMMs with per-expert token batching, tiled MLA attention, reduce-scatter/all-gather in bf16 32-token slabs.
8. **Build**: `-DSTRATA_ENABLE_CPU_RUNNER=ON`, CUDA and HIP off, `-march=armv8.2-a+sve`, cross-compiled with the
   A64FX clang toolchain; `strata_kernels_cpu` gains an `sve` ISA rung next to `avx`, `avx2`, `avx512`.

## 5. Measurements (2026-10-18) and what is still open

All raw logs sit with the probe outputs; the simulator reads `tools/sim/data/measured_a64fx_kernels.json` and
`tools/sim/data/measured_a64fx_comm.json`. Tooling: `tools/a64fx_probe/` (`run_probe.sh` builds ggml with FCC in clang
mode on a compute node and runs the kernel, parity and node benches; `run_comm.sh` runs gemm's pure-uTofu MoE dispatch
bench on rank subsets of one 12-node 2x3x2 allocation).

1. **Expert kernels, measured.** ggml's SVE dots (llama.cpp fbec0ddd8) are instruction-bound and linear in threads.
   Strata's SDOT kernels (`tools/a64fx_probe/q23_sdot_sve.c`, `iq_sdot_sve.c`) replace them for the four formats the
   plans use. Q2_K/Q3_K: the Q8_K activation is permuted once per token so each 2-bit plane of a 256-weight block is
   one 512-bit vector; one SDOT per plane per token; Q2_K weights are pre-multiplied by their 4-bit scales in int8
   (v2), Q3_K keeps int32 lane multipliers (its scaled weights overflow int8) and unpacks its 6-bit scales with two
   `svtbl`. IQ2_XXS/IQ1_S: eight 8-weight grid entries per 512-bit vector by one 64-bit gather; IQ2_XXS signs gathered
   from `ksigns64` (5-12 % faster than rebuilding them from popcount parity in registers), IQ1_S's +-1/8 delta folded
   into the weights as 8g +- 1. Every kernel matches ggml's `vec_dot` and a scalar double reference within 2.4e-7 at
   1-4 tokens (73 checks). One node, one session (job 52438878), 48 threads, GB/s of weight bytes:

   | GLM shapes | 1 token | 2 | 3 | 4 | ggml, 1 token |
   |---|---:|---:|---:|---:|---:|
   | Q2_K gate/up 2048 x 4096, SDOT v2 | 180.3 | 119.6 | 79.6 | 57.1 | 91.6 |
   | Q3_K gate/up 2048 x 4096, SDOT v2 | 171.8 | 113.8 | 84.9 | 58.4 | 54.5 |
   | Q3_K down 4096 x 2048, SDOT v2 | 138.0 | 89.7 | 71.3 | 54.1 | 52.4 |
   | IQ2_XXS 2048 x 4096, SDOT gather signs | 81.1 | 56.4 | 48.2 | 34.9 | 11.1 |
   | IQ1_S 2048 x 4096, SDOT | 58.4 | 42.3 | 34.4 | 27.3 | 8.2 |

   DS4.1F shapes, one token: Q2_K gate/up 2304 x 5120 192.2 (ggml 94.7), Q2_K down 5120 x 2304 133.4 (81.2), IQ2_XXS
   84.8 (11.1), IQ1_S 60.3 (8.2). The kernels are compute-bound (HBM streams 915 GB/s), so GB/s falls with the token
   count and the redesign charges the measured cost per extra token in the MTP verify step. What did not help: the
   first Q3_K version unpacked its scales through the stack (88 GB/s, fixed); processing two rows per call to share
   the activation loads is slower everywhere at 1-2 tokens (register spills; 120 vs 180 GB/s on Q2_K). Node spread:
   the same Q2_K kernel ran 198 GB/s on another node (ggml 96 vs 92), so compare kernels within one run. Gather-free
   IQ kernels (job 52439675): an in-register `svtbl` decode of the IQ2_XXS grid matches the gathers only once the
   decode is amortized over 2-4 tokens, a +12 % repack adds 4-6 % at one token, and expanded 2-bit codes double the
   rate at +61-64 % memory (data: `iq_gather_free`). Compiler and pipelining (job 52440185, same sources linked into
   the same harness): clang 21 with `-mcpu=a64fx` makes 2-4 token steps 25-45 % faster than FCC 4.12 on the K-quants
   (Q2_K at 4 tokens 59 -> 85 GB/s, Q3_K down 56 -> 70) and ties it at one token; software-pipelined v3 kernels
   (block i+1 decoded, its gathers issued, before block i's token math) are roughly neutral (+2-7 % for IQ1_S and Q3_K
   down). The runner configuration the simulator now uses: clang 21 -mcpu=a64fx, best kernel per width (data:
   `sdot_kernel_v3`). In the
   simulator: `current` uses ggml's mix (78 GB/s), `redesign` the SDOT q23 mix (159.9 GB/s, derived from the data file).
2. **Node constants, measured** (`node_probe.c`): HBM read 915 GB/s at 48 threads, 229 per CMG; libhwb hardware
   barrier 1.38 us hierarchical over 48 threads, 0.12 us inside one CMG; a flat OpenMP barrier 5.29 us at 48 threads
   and 1.73 at 12; an empty `omp parallel` fork-join 5.24 us. The redesign's split-phase barrier (1.4 assumed) is the
   measured 1.38; the chain rate stays at the measured 504 GB/s absolute.
3. **Collectives at 6, 8 and 12 ranks, measured** on GLM geometry (288 experts, top-8, hidden 4096, uniform routing),
   two sweeps within 4 %:

   | ranks | tree allreduce fp32 1 / 3 tokens (us) | bf16 1 / 3 | dispatch+combine bf16 1 / 3 |
   |---:|---:|---:|---:|
   | 6 | 18.5 / 44.0 | 11.7 / 24.5 | 18.3 / 47.1 |
   | 8 | 12.5 / 27.6 | 8.3 / 15.8 | 21.1 / 76.9 |
   | 12 | 22.6 / 55.3 | 14.3 / 30.0 | 25.9 / 61.0 |

   The step model (recursive doubling, 4.69 us per step) was within 5 % at 6 and 12 ranks but 13-27 % high at 8; the
   simulator now uses the table at these rank counts and the model elsewhere. The 8-rank 3-token dispatch is slow for
   this routing seed (reproducible); one realization, not an average over routings.
4. **Still open:** the mHC site cost as one fused phase in-model (today 70-86 us against 23 us standalone; worth
   about 5 ms per token), the expert kernels' in-model rate inside a streaming chain, KDA small steps as one team
   phase, and the int8 expert GEMM efficiency at the chunk-4096 M per expert inside a full prefill (today 40 %).

Reproduce: `python3 tools/glm_a64fx_sim.py validate`, `python3 tools/glm_a64fx_sim.py report` (these tables),
`python3 tools/glm_a64fx_sim.py estimate --nodes 6 --profile redesign --mtp 2 --set profile.expert_q2k_gbps=120`,
`python3 -m unittest tools.test_sim_a64fx`.
