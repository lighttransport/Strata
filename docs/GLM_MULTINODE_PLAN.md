# GLM-5.3-Flash across nodes: plan for TR16 + B550 (2026-10-10)

## What exists

Strata already runs a two-node expert tensor-parallel mode for the GLM decoder
(`docs/GLM_REMOTE_TP_COMM.md`, runner `tools/glm_remote_tp.sh`, worker
`src/program/glm_tp_worker.cpp`, wire protocol `include/strata/net/expert_tp.hpp`):

- The decoder node keeps the first `keep` FFN rows of every CPU-routed expert
  (`keep` is a multiple of 512 of 2048, so the local share is 0.25, 0.5 or 0.75)
  and the worker computes the rest on its CPU, returning one router-weighted
  partial sum per token. GPU-tier experts, dense and attention work and all
  prefill stay on the decoder node.
- One sequential round trip per MoE layer, 42 per token: a 4.8 KB request and an
  8 KB (f16) or 16 KB (f32) reply per token. Measured on the ConnectX-3 link in
  its current PCIe-lane-degraded state: 20.9 µs decode-shaped round trip,
  1.55–1.65 GB/s bulk (a healthy QDR link does about 3.2 GB/s), 0.023 ms exposed
  wait per layer after buffer registration. Over 1 GbE (UDP backend): 401 µs round
  trip, 109–113 MB/s, about 200 µs wake-up floor.
- Measured decode with the REAP-50 Q23 pack, 75% local: ordinary decode 10.5–10.9
  → 16.0–16.2 tok/s (+50%); with the GPU tier plus MTP2 the two-node run gave no
  gain (16.1 vs 16.1), and an unpaired tier+MTP run reached 19.2–20.3. Quality:
  KL 0.019 (f32 reply) / 0.023 (f16), top-1 95%.
- Limits: the worker holds its rows in RAM (`--reserve-gib`), memlock on B550 is
  about 8 GB so rows are registered per layer, the first prefill is slow (the
  worker's rows are read from disk), the worker computes in phases, and only one
  worker is supported.

## Simulator: `glm_sim.py multinode`

`tools/sim/glm_sim.py multinode --node DECODER --node WORKER [--node WORKER …]
--link ib|ib-qdr|gbe|10gbe|rtt_us,gbps,exposed_us … --split row|expert|both
<config options>` scores every share assignment for up to three workers with
each worker's own CPU roofline (`tools/sim/hw.py` `WorkerNode`), its link's
round trip, reply wire time and exposed wait, the RAM fit of every node and the
start-up shipping time, and prints the best split and the single-node baseline
(`tools/sim/multinode.py`; tests in `tools/test_sim_multinode.py`). The legacy
one-worker presets keep their calibrated numbers. A pipeline split is not scored:
single-stream decode gains nothing from serial stages. Predictions carry the
±30% band of the one-worker InfiniBand validation; several workers and the
expert-index split are unvalidated and not implemented.

Results with the TR16 calibration (`tools/sim/data/params_tr16_20261009.json`):

| Scenario | Single node | Best split | Predicted | Change |
|---|---:|---|---:|---:|
| Qualified profile (Q23, MTP1, affinity 0.08) + B550 over the degraded IB | 26.4 | B550 0.25 of rows | 26.7 | +1.2% |
| Same over 1 GbE | 26.4 | 0.25 | 16.3 | −38% |
| Fast candidate (Q22, Q4 dense, MTP3 soft acceptance) + B550 over IB | 28.3 | 0.25 | 31.4 | +11% |
| Qualified profile + a second TR16-class node over healthy QDR IB | 26.4 | 0.25–0.5 on the fast node | 28.4 | +8% |

Why the gain is small: the B550 streams expert rows at 26–29 GB/s against the
TR16's 72 GB/s, so with the only share that keeps it off the critical path
(0.25) the decoder still reads 75% of the bytes, and the 42 serial round trips
add about 1–2 ms per token. Over 1 GbE the round trips alone cost more than the
work they remove. A worker only pays when its DRAM bandwidth is at least half
the decoder's, the link is InfiniBand-class, and the split is close to the
bandwidth ratio (which needs finer than 512-row steps).

## Plan

1. **Measure before building.** Run the existing two-node mode on the full
   model (Q23 pack on TR16, B550 holding 25% of the rows = 25.9 GiB, which fits
   its 60 GiB) with the qualified profile and with the fast candidate, over IB:
   `tools/glm_remote_tp.sh b550 <worker-binary> <b550-ib-ip>:PORT --threads 16 --reserve-gib 2 -- <decoder command> --remote-tp-share=0.25`.
   Compare with the same-day single-node run (held-out 2K/8K, 512×3). This
   validates the simulator rows above; the expected gain is 0–11%.
2. **If the IB link stays degraded**, do not pursue multinode for throughput; the
   link's PCIe lanes are the first thing to fix (the card should report x4/x8;
   `lspci -vv` on both nodes). The simulator's `ib-qdr` link preset shows what a
   healthy link buys.
3. **Finer row split** (256-row blocks, needs the fused hidden quantization off)
   would let a 29 GB/s worker take 0.28 instead of 0.25 of the rows; gain ≤ 3%.
   Low priority.
4. **Expert-index split** (whole experts per node): same message shape, uneven
   load, no implementation; the simulator's imbalance model is the only evidence.
   Only worth building for a node pair with similar DRAM bandwidth.
5. **More than one worker**: engine work (N queue pairs, N partial sums); the
   simulator supports up to three to rank hardware purchases, nothing more.
6. **Capacity rather than speed**: the real use of the B550 is the 64 GB
   REAP-50 configuration as an independent server, or holding the full model's
   rows for a decoder node with less than 128 GB (the simulator's RAM-fit notes
   show when that is the only option).

## Feasibility study: 32 GB RAM + 8 GB GPU (RTX 3070 class), target about 10 tok/s

Simulator preset `b550_32g_3070` (Ryzen 3950X, 30 GiB usable at 34 GB/s, 7.9 GB
VRAM, 2.5 GB/s NVMe), weights 32–63 GB, `tools/sim/glm_sim.py plan --hw
b550_32g_3070 --target-decode 10 --hypothetical`:

| Mode | Weights | Predicted decode | Prefill | Loss | Comment |
|---|---|---:|---:|---|---|
| RAM-tier **exact** (misses read from NVMe) | REAP-50 Q23 (51 GB experts) | 2.2 tok/s | 69 | none | disk-bound; the RAM set (24 GiB) serves 74% of routed bytes |
| RAM-tier **hybrid** (routing margin toward resident experts) | REAP-50 Q23 | 8.6 (MTP1) | 69 | KL about 0.2 | model is optimistic: on the 64 GB machine the same mode measured 5.5–6.0 against a 15 predicted |
| RAM-tier frozen/hybrid + affinity 0.1 + skip 0.1 + Q4 dense | q2_orig / q22 / REAP-50 | 12–13 | 26–69 | KL 0.19–0.69 | every >10 row is lossy on two or three axes at once |
| implemented, lossless only | — | no candidate fits 30 GiB | — | — | `plan --implemented-only` and `--lossless` return nothing |

Verdict: **about 10 tok/s is not reachable at 32 GB with the current packs
without a large quality loss**, and the 8 GB card leaves no room for a resident
expert tier (the plan allocates 0–64 MiB), so prefill is also disk-bound at
30–70 tok/s. The realistic routes are:

- a pack whose routed experts fit in about 22 GiB of RAM: a REAP-75-class
  pruning (72 experts, about 26 GB at Q2_K/Q3_K; no such GGUF exists upstream)
  or a 1.6-bit REAP-50 (IQ1-class, about 30 GB, quality unknown); then the
  single-node model predicts 10–14 tok/s with the same CPU;
- otherwise accept the hybrid RAM tier at KL about 0.2 and 5–9 tok/s, which the
  earlier 64 GB experiment judged not worth shipping (`docs/GLM_64G_16G_SAFE.md`).

The queued next step is a simulator sweep over hypothetical pack sizes
(`--pack` with a synthetic 72-expert REAP entry) once such weights exist; no
engine work is proposed for this configuration now.
