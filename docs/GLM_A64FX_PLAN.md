# GLM-5.3-Flash on A64FX: simulated decode and prefill at 6 and 12 nodes, and a runner proposal

Written 2026-10-18. Everything below that is not marked "measured" is a simulator output: an estimate with a stated
calibration band and a printed list of hypotheses. Nothing was run on the A64FX cluster for this document.

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
site, KDA small steps as one full-team phase, 8 phases per MoE layer at 1.4 us split-phase barrier + 1.6 us dispatch,
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
| 6 | sliced over 4 ranks (runner today) | 32.7 / 48.5 / 35.5 | 29.7 / 44.0 / 32.2 | 27.0 / 40.1 / 29.4 | 33.7 = trunk 5.4 + experts 8.5 + coll 4.2 + mHC 5.9 + ctx 3.2 + head 1.3 |
| 6 | sliced over every rank | 32.7 / 48.5 / 35.6 | 29.7 / 44.0 / 32.2 | 27.1 / 40.1 / 29.4 | 33.7 = trunk 5.4 + experts 10.0 + coll 2.7 + mHC 5.9 + ctx 3.2 + head 1.3 |
| 6 | whole experts (EP) | 31.2 / 46.2 / 33.9 | 28.4 / 42.1 / 30.9 | 26.0 / 38.5 / 28.2 | 35.2 = trunk 5.4 + experts 6.2 + coll 7.9 + mHC 5.9 + ctx 3.2 + head 1.3 |
| 6 | sliced over 2 ranks | 33.1 / 49.1 / 36.0 | 30.0 / 44.5 / 32.6 | 27.3 / 40.5 / 29.7 | 33.3 = trunk 5.4 + experts 7.0 + coll 5.3 + mHC 5.9 + ctx 3.2 + head 1.3 |
| 6 | sliced over 4 ranks (runner today) x4 CMG ranks | 23.7 / 35.1 / 25.8 | 22.1 / 32.7 / 24.0 | 20.6 / 30.5 / 22.4 | 45.3 = trunk 10.7 + experts 8.5 + coll 10.6 + mHC 5.9 + ctx 3.2 + head 1.3 |
| 6 | sliced over every rank x4 CMG ranks | 20.0 / 29.6 / 21.7 | 18.8 / 27.9 / 20.4 | 17.7 / 26.3 / 19.3 | 53.1 = trunk 10.7 + experts 23.4 + coll 3.5 + mHC 5.9 + ctx 3.2 + head 1.3 |
| 6 | whole experts (EP) x4 CMG ranks | 19.2 / 28.4 / 20.8 | 18.1 / 26.8 / 19.6 | 17.1 / 25.3 / 18.6 | 55.3 = trunk 10.7 + experts 6.2 + coll 22.8 + mHC 5.9 + ctx 3.2 + head 1.3 |
| 6 | sliced over 2 ranks x4 CMG ranks | 22.1 / 32.7 / 24.0 | 20.6 / 30.6 / 22.4 | 19.3 / 28.7 / 21.0 | 48.4 = trunk 10.7 + experts 7.0 + coll 15.2 + mHC 5.9 + ctx 3.2 + head 1.3 |
| 8 | sliced over 4 ranks (runner today) | 37.1 / 54.9 / 40.3 | 33.2 / 49.2 / 36.1 | 30.0 / 44.4 / 32.5 | 30.1 = trunk 4.7 + experts 6.3 + coll 3.5 + mHC 5.9 + ctx 3.2 + head 1.1 |
| 8 | sliced over every rank | 35.8 / 53.0 / 38.8 | 32.1 / 47.6 / 34.9 | 29.1 / 43.1 / 31.6 | 31.1 = trunk 4.7 + experts 8.6 + coll 2.3 + mHC 5.9 + ctx 3.2 + head 1.1 |
| 8 | whole experts (EP) | 34.1 / 50.5 / 37.0 | 30.8 / 45.6 / 33.5 | 28.0 / 41.5 / 30.4 | 32.5 = trunk 4.7 + experts 4.7 + coll 7.5 + mHC 5.9 + ctx 3.2 + head 1.1 |
| 8 | sliced over 2 ranks | 36.4 / 53.9 / 39.5 | 32.7 / 48.4 / 35.5 | 29.5 / 43.7 / 32.1 | 30.6 = trunk 4.7 + experts 5.2 + coll 5.1 + mHC 5.9 + ctx 3.2 + head 1.1 |
| 8 | sliced over 4 ranks (runner today) x4 CMG ranks | 25.7 / 38.1 / 27.9 | 23.8 / 35.2 / 25.8 | 22.1 / 32.7 / 24.0 | 42.1 = trunk 10.0 + experts 6.3 + coll 10.3 + mHC 5.9 + ctx 3.2 + head 1.1 |
| 8 | sliced over every rank x4 CMG ranks | 21.1 / 31.2 / 22.9 | 19.8 / 29.3 / 21.5 | 18.6 / 27.5 / 20.2 | 50.6 = trunk 10.0 + experts 22.0 + coll 3.1 + mHC 5.9 + ctx 3.2 + head 1.1 |
| 8 | whole experts (EP) x4 CMG ranks | 20.7 / 30.6 / 22.4 | 19.4 / 28.7 / 21.1 | 18.2 / 27.0 / 19.8 | 51.6 = trunk 10.0 + experts 4.7 + coll 21.4 + mHC 5.9 + ctx 3.2 + head 1.1 |
| 8 | sliced over 2 ranks x4 CMG ranks | 23.7 / 35.2 / 25.8 | 22.1 / 32.7 / 24.0 | 20.6 / 30.5 / 22.4 | 45.3 = trunk 10.0 + experts 5.2 + coll 14.6 + mHC 5.9 + ctx 3.2 + head 1.1 |
| 12 | sliced over 4 ranks (runner today) | 39.6 / 58.7 / 43.0 | 35.2 / 52.2 / 38.2 | 31.6 / 46.8 / 34.3 | 28.4 = trunk 4.0 + experts 4.2 + coll 4.7 + mHC 5.9 + ctx 3.2 + head 0.9 |
| 12 | sliced over every rank | 37.5 / 55.6 / 40.7 | 33.6 / 49.7 / 36.4 | 30.2 / 44.8 / 32.8 | 29.8 = trunk 4.0 + experts 7.2 + coll 3.1 + mHC 5.9 + ctx 3.2 + head 0.9 |
| 12 | whole experts (EP) | 36.0 / 53.3 / 39.1 | 32.3 / 47.9 / 35.1 | 29.2 / 43.3 / 31.8 | 30.9 = trunk 4.0 + experts 3.1 + coll 8.3 + mHC 5.9 + ctx 3.2 + head 0.9 |
| 12 | sliced over 2 ranks | 38.7 / 57.3 / 42.0 | 34.5 / 51.1 / 37.4 | 31.0 / 45.9 / 33.7 | 29.0 = trunk 4.0 + experts 3.5 + coll 6.0 + mHC 5.9 + ctx 3.2 + head 0.9 |
| 12 | sliced over 4 ranks (runner today) x4 CMG ranks | 27.2 / 40.2 / 29.5 | 25.0 / 37.1 / 27.2 | 23.1 / 34.3 / 25.1 | 40.0 = trunk 9.3 + experts 4.2 + coll 11.0 + mHC 5.9 + ctx 3.2 + head 0.9 |
| 12 | sliced over every rank x4 CMG ranks | 21.7 / 32.1 / 23.6 | 20.3 / 30.1 / 22.0 | 19.0 / 28.2 / 20.7 | 49.2 = trunk 9.3 + experts 20.7 + coll 3.9 + mHC 5.9 + ctx 3.2 + head 0.9 |
| 12 | whole experts (EP) x4 CMG ranks | 22.0 / 32.6 / 23.9 | 20.6 / 30.5 / 22.4 | 19.3 / 28.6 / 21.0 | 48.5 = trunk 9.3 + experts 3.1 + coll 20.7 + mHC 5.9 + ctx 3.2 + head 0.9 |
| 12 | sliced over 2 ranks x4 CMG ranks | 25.2 / 37.3 / 27.3 | 23.3 / 34.5 / 25.3 | 21.7 / 32.1 / 23.5 | 42.9 = trunk 9.3 + experts 3.5 + coll 14.7 + mHC 5.9 + ctx 3.2 + head 0.9 |

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

Hypotheses in `current`: expert_q2k_gbps (Q2_K/Q3_K in-model rate, never measured on A64FX); context slope applied as one term on the sparse layers; 6-node attention split: 64 heads / 6 is not integral (TP-6 needs a 60+4 or TP-4 split)

### Decode, profile `redesign` (tok/s)

| nodes | layout | 128 ctx plain / MTP2 best / MTP2 sweep | 8K | 32K | 8K step ms |
|---|---:|---:|---:|---:|---:|
| 6 | sliced over 4 ranks (runner today) | 82.2 / 115.2 / 84.4 | 72.8 / 95.9 / 70.3 | 65.1 / 81.7 / 59.9 | 13.7 = trunk 2.3 + experts 2.4 + coll 2.2 + mHC 1.9 + ctx 1.6 + head 1.0 + phases 1.5 |
| 6 | sliced over every rank | 83.2 / 117.5 / 86.1 | 73.6 / 97.4 / 71.4 | 65.7 / 82.8 / 60.7 | 13.6 = trunk 2.3 + experts 2.6 + coll 1.9 + mHC 1.9 + ctx 1.6 + head 1.0 + phases 1.5 |
| 6 | whole experts (EP) | 77.6 / 105.5 / 77.3 | 69.2 / 89.1 / 65.3 | 62.2 / 76.7 / 56.2 | 14.5 = trunk 2.3 + experts 2.2 + coll 3.1 + mHC 1.9 + ctx 1.6 + head 1.0 + phases 1.5 |
| 6 | sliced over 2 ranks | 81.4 / 113.2 / 82.9 | 72.2 / 94.5 / 69.2 | 64.6 / 80.6 / 59.1 | 13.9 = trunk 2.3 + experts 2.2 + coll 2.4 + mHC 1.9 + ctx 1.6 + head 1.0 + phases 1.5 |
| 6 | sliced over 4 ranks (runner today) x4 CMG ranks | 79.5 / 102.5 / 75.2 | 70.7 / 86.9 / 63.7 | 63.4 / 75.1 / 55.0 | 14.1 = trunk 2.2 + experts 2.4 + coll 4.1 + mHC 1.4 + ctx 1.6 + head 1.0 + phases 1.1 |
| 6 | sliced over every rank x4 CMG ranks | 78.1 / 102.6 / 75.2 | 69.6 / 87.0 / 63.8 | 62.5 / 75.1 / 55.1 | 14.4 = trunk 2.2 + experts 4.0 + coll 2.7 + mHC 1.4 + ctx 1.6 + head 1.0 + phases 1.1 |
| 6 | whole experts (EP) x4 CMG ranks | 61.8 / 86.3 / 63.2 | 56.4 / 75.0 / 54.9 | 51.6 / 66.0 / 48.4 | 17.7 = trunk 2.2 + experts 2.2 + coll 7.9 + mHC 1.4 + ctx 1.6 + head 1.0 + phases 1.1 |
| 6 | sliced over 2 ranks x4 CMG ranks | 73.3 / 91.8 / 67.3 | 65.8 / 79.1 / 58.0 | 59.4 / 69.2 / 50.7 | 15.2 = trunk 2.2 + experts 2.2 + coll 5.3 + mHC 1.4 + ctx 1.6 + head 1.0 + phases 1.1 |
| 8 | sliced over 4 ranks (runner today) | 94.9 / 135.4 / 99.2 | 82.6 / 109.5 / 80.2 | 72.8 / 91.3 / 66.9 | 12.1 = trunk 1.8 + experts 1.8 + coll 1.7 + mHC 1.9 + ctx 1.6 + head 0.9 + phases 1.5 |
| 8 | sliced over every rank | 95.0 / 136.1 / 99.8 | 82.6 / 109.9 / 80.6 | 72.8 / 91.6 / 67.2 | 12.1 = trunk 1.8 + experts 2.1 + coll 1.4 + mHC 1.9 + ctx 1.6 + head 0.9 + phases 1.5 |
| 8 | whole experts (EP) | 87.7 / 120.3 / 88.2 | 77.1 / 99.4 / 72.8 | 68.5 / 84.2 / 61.7 | 13.0 = trunk 1.8 + experts 1.7 + coll 2.7 + mHC 1.9 + ctx 1.6 + head 0.9 + phases 1.5 |
| 8 | sliced over 2 ranks | 92.4 / 129.9 / 95.2 | 80.7 / 105.9 / 77.6 | 71.3 / 88.8 / 65.1 | 12.4 = trunk 1.8 + experts 1.7 + coll 2.1 + mHC 1.9 + ctx 1.6 + head 0.9 + phases 1.5 |
| 8 | sliced over 4 ranks (runner today) x4 CMG ranks | 90.4 / 116.8 / 85.6 | 79.2 / 97.0 / 71.1 | 70.1 / 82.5 / 60.4 | 12.6 = trunk 1.7 + experts 1.8 + coll 3.7 + mHC 1.4 + ctx 1.6 + head 0.9 + phases 1.1 |
| 8 | sliced over every rank x4 CMG ranks | 88.3 / 116.2 / 85.1 | 77.5 / 96.5 / 70.8 | 68.8 / 82.1 / 60.2 | 12.9 = trunk 1.7 + experts 3.6 + coll 2.3 + mHC 1.4 + ctx 1.6 + head 0.9 + phases 1.1 |
| 8 | whole experts (EP) x4 CMG ranks | 67.5 / 95.1 / 69.7 | 61.0 / 81.5 / 59.8 | 55.5 / 71.0 / 52.1 | 16.4 = trunk 1.7 + experts 1.7 + coll 7.7 + mHC 1.4 + ctx 1.6 + head 0.9 + phases 1.1 |
| 8 | sliced over 2 ranks x4 CMG ranks | 77.0 / 112.1 / 82.2 | 68.7 / 93.7 / 68.7 | 61.8 / 80.1 / 58.7 | 14.6 = trunk 1.7 + experts 1.7 + coll 5.8 + mHC 1.4 + ctx 1.6 + head 0.9 + phases 1.1 |
| 12 | sliced over 4 ranks (runner today) | 96.9 / 131.2 / 96.2 | 84.1 / 106.7 / 78.2 | 74.0 / 89.4 / 65.5 | 11.9 = trunk 1.3 + experts 1.3 + coll 2.6 + mHC 1.9 + ctx 1.6 + head 0.8 + phases 1.5 |
| 12 | sliced over every rank | 97.0 / 131.9 / 96.7 | 84.2 / 107.2 / 78.6 | 74.0 / 89.7 / 65.8 | 11.9 = trunk 1.3 + experts 1.6 + coll 2.3 + mHC 1.9 + ctx 1.6 + head 0.8 + phases 1.5 |
| 12 | whole experts (EP) | 89.3 / 117.1 / 85.8 | 78.3 / 97.2 / 71.2 | 69.4 / 82.6 / 60.5 | 12.8 = trunk 1.3 + experts 1.1 + coll 3.6 + mHC 1.9 + ctx 1.6 + head 0.8 + phases 1.5 |
| 12 | sliced over 2 ranks | 94.4 / 126.3 / 92.6 | 82.2 / 103.4 / 75.8 | 72.5 / 87.1 / 63.8 | 12.2 = trunk 1.3 + experts 1.2 + coll 3.0 + mHC 1.9 + ctx 1.6 + head 0.8 + phases 1.5 |
| 12 | sliced over 4 ranks (runner today) x4 CMG ranks | 92.7 / 114.6 / 84.0 | 80.9 / 95.5 / 70.0 | 71.4 / 81.4 / 59.6 | 12.4 = trunk 1.2 + experts 1.3 + coll 4.6 + mHC 1.4 + ctx 1.6 + head 0.8 + phases 1.1 |
| 12 | sliced over every rank x4 CMG ranks | 90.1 / 113.2 / 83.0 | 78.9 / 94.5 / 69.3 | 69.9 / 80.6 / 59.1 | 12.7 = trunk 1.2 + experts 3.1 + coll 3.1 + mHC 1.4 + ctx 1.6 + head 0.8 + phases 1.1 |
| 12 | whole experts (EP) x4 CMG ranks | 71.1 / 99.4 / 72.9 | 64.0 / 84.7 / 62.1 | 57.9 / 73.4 / 53.8 | 15.6 = trunk 1.2 + experts 1.1 + coll 8.0 + mHC 1.4 + ctx 1.6 + head 0.8 + phases 1.1 |
| 12 | sliced over 2 ranks x4 CMG ranks | 80.9 / 116.1 / 85.1 | 71.8 / 96.5 / 70.7 | 64.3 / 82.1 / 60.2 | 13.9 = trunk 1.2 + experts 1.2 + coll 6.2 + mHC 1.4 + ctx 1.6 + head 0.8 + phases 1.1 |

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

Hypotheses in `redesign`: expert_q2k_gbps 250 GB/s (47 x 7.1 GB/s x 0.75): the Q2_K SVE dot has never been timed on A64FX; mHC 23 us per site in-model (standalone kernel time; the in-model cost was never reproduced); KDA small steps 12 us as one full-team phase (today 82 us on 6 threads); prefill int8 efficiency 0.70 (today 0.40 fitted; the 94 % CMG microkernel exists only standalone); EP dispatch+combine at 6 owners scaled from the 12-owner measurement; 4 ranks per node: collective costs scaled from 12-rank measurements to 24/48 ranks

With a **Q4_K trunk** instead of Q8_0 (lossy on the trunk; the 0.86x byte scale of `model.DENSE_FORMATS`), the
`redesign` decode rows of the runner layout become:

| 6 | sliced over 4 ranks (runner today) | 88.8 / 120.3 / 88.2 | 77.9 / 99.4 / 72.8 | 69.1 / 84.2 / 61.7 | 12.8 = trunk 1.5 + experts 2.4 + coll 2.2 + mHC 1.9 + ctx 1.6 + head 0.8 + phases 1.5 |
| 8 | sliced over 4 ranks (runner today) | 101.4 / 140.7 / 103.1 | 87.5 / 112.9 / 82.7 | 76.5 / 93.7 / 68.7 | 11.4 = trunk 1.2 + experts 1.8 + coll 1.7 + mHC 1.9 + ctx 1.6 + head 0.7 + phases 1.5 |
| 12 | sliced over 4 ranks (runner today) | 101.4 / 134.4 / 98.5 | 87.4 / 108.8 / 79.8 | 76.5 / 90.9 / 66.6 | 11.4 = trunk 0.9 + experts 1.3 + coll 2.6 + mHC 1.9 + ctx 1.6 + head 0.7 + phases 1.5 |

### Reading the tables

1. **Six nodes are feasible only because of the Q2Q3 pack.** UD-Q4_K_XL needs 28.5 GiB of experts per node at 6 nodes
   and does not fit; q23-from-FP8 needs 17.1 GiB and leaves 8 GiB of margin with a 32K context. At 12 nodes either
   pack fits.
2. **The runner as it is, moved to 6 nodes with q23: about 30 tok/s decode (44 with MTP depth 2 at the BEST
   acceptance, 32 at the sweep acceptance) and about 260 tok/s prefill at 8K.** Twelve nodes with q23: about 35 / 52 /
   38 and 390. The Q2_K kernel's per-byte cost (a hypothesis: 1.5x the Q4_K kernel's) cancels the smaller pack, so
   q23 is no faster than Q4 on the current runner. Halving the node count costs only 15 % because 60 % of the current
   step is latency that does not scale with the node count at all (mHC 5.9, context 3.2, KDA small steps, router).
3. **The redesigned runner: about 84 tok/s plain at 12 nodes and 73 at 6 (8K context); 107 / 96 with MTP depth 2 at
   the BEST acceptance, 78 / 70 at the sweep acceptance; prefill about 700-740 at 12 nodes and 470 at 6.** A Q4_K trunk
   adds 4-7 %. The step is 12-14 ms: trunk 1.3-2.3, experts 1.3-2.6, collectives 1.9-2.6, mHC 1.9, context 1.6,
   phases 1.5, head 0.8-1.0. No single term dominates; the gain over today comes from removing the sync-bound fixed
   costs (mHC 5.9 -> 1.9, KDA fixed 3.5 -> 0.5, barriers 5.5 -> 1.5), not from a faster kernel.
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
   of 4 at 6 nodes or 5 at 12, and 2x2x2 is a clean torus. Under `redesign` 8 nodes decode as fast as 12 (12.1 vs
   11.9 ms per step: 0.9 ms less in collectives pays for the extra trunk and expert bytes) and are ahead with MTP
   depth 2; under `current` they recover 33 of the 35 tok/s of 12 nodes. Prefill still scales with node count
   (576 vs 706 under `redesign`). Memory is 15.9 GiB per node. The 8-node collectives are extrapolated from the
   12-rank measurements by step count; no 8-node allocation has been measured.
8. **Prefill is compute-bound on int8 SDOT and collective-bound second.** At 12 nodes today: 1.41 ms per position of
   GEMM at an effective 1.42 TFLOPS per node (40 % of the measured 3.18 TOPS GEMM) plus 1.0 ms of reduce-scatter/
   all-gather that does not shrink with node count. The `redesign` 700+ assumes 70 % GEMM efficiency and one
   collective per layer in bf16; 2,000 tok/s would need more than 50 % of the int8 peak of 12 nodes with no collective
   cost and is out of reach of this design.

### Calibration band

The `current` 12-node rows are fitted (within 5 %). Moving to 6 nodes relies on the scaling classes; the pessimistic
edge, every streamed term scaling exactly 1/N with no fixed part, lands at 27-28 tok/s instead of 30. The `redesign`
numbers rest on the hypotheses printed under each table; the two that move the result most are the Q2_K/Q3_K SVE
dot rate (250 GB/s assumed in-model; at 120 GB/s the 12-node step grows 1.3 ms and decode drops to 76 tok/s) and the
mHC in-model cost (23 us per site assumed; at today's 78 us the step grows 5 ms and decode drops to 59 tok/s). Treat
`redesign` as 60-90 tok/s at 12 nodes and 55-80 at 6 until those two are measured.

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

## 5. Next measurements (one node each, before any port work)

1. ~~The Q2_K/Q3_K SVE dot rate per core~~ **Measured 2026-10-18** (one node, interactive, `tools/a64fx_probe/run_probe.sh`,
   ggml-cpu from llama.cpp fbec0ddd8 built with FCC in clang mode, `tools/sim/data/measured_a64fx_kernels.json`):

   | ggml SVE `vec_dot`, GLM shapes | GB/s per core | GB/s per node (48) | GMAC/s per node |
   |---|---:|---:|---:|
   | Q2_K | 2.02 | 97.0 | 296 |
   | Q3_K | 1.16 | 55.6 | 129 |
   | Q4_K | 1.60 | 76.0 | 135 |
   | IQ2_XXS / IQ2_S / IQ1_S | 0.23 / 0.22 / 0.17 | 11.2 / 10.5 / 8.2 | 43 / 33 / 42 |
   | Q8_0 | 9.0 | 432 | 407 |

   Every format scales linearly from 1 to 48 threads and matched a double dequantize-and-dot (rel 1e-7), so all are
   instruction-bound, not HBM-bound (only Q8_0 reaches half the stream rate). The old hypothesis (7.1 GB/s per core,
   Zen 1 scaled) was 3.5x too high: the `current` profile now uses 78 GB/s per node for the q23 pack (bytes-weighted
   Q2_K/Q3_K), which leaves the 12-node decode forecast unchanged (35 tok/s, expert time is a small slice there) and
   trims 6 nodes to 29 tok/s. The redesign's 250 GB/s is now explicitly a new-kernel target: ggml's codebook kernels
   cannot reach it; it needs the int8-SDOT tile scheme of gemm's MXFP4 path (~380 GMAC/s per node). The IQ formats
   are unusable as they stand on A64FX (10 GB/s per node): the DS4.1F 6/8-node plans need the SVE W1A8 kernel
   (`svtbl` over an int8 grid + `svdot_s32`) before they are worth running.
2. The mHC site cost as one fused phase in-model (today 70-86 us against 23 us standalone). Worth 5 ms per token.
3. Dispatch + combine and tree allreduce at 6 ranks on a 6-node allocation (the model scales them from 12).
4. The int8 expert GEMM efficiency at the chunk-4096 M per expert inside a full prefill (today 40 % effective).

Reproduce: `python3 tools/glm_a64fx_sim.py validate`, `python3 tools/glm_a64fx_sim.py report` (these tables),
`python3 tools/glm_a64fx_sim.py estimate --nodes 6 --profile redesign --mtp 2 --set profile.expert_q2k_gbps=120`,
`python3 -m unittest tools.test_sim_a64fx`.
