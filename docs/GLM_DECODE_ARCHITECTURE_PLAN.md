# GLM-5.3-Flash decode: architecture plan for 50+ tok/s on one GPU

Target machine: Threadripper 1950X (16 cores, 2 NUMA nodes, 72 GB/s expert stream), RTX 5060 Ti 16 GB on PCIe
Gen3 x8, 124 GiB RAM. Today's measured single-stream decode is 20-24 tok/s (q23 pack, MTP depth 2, split
verify, adaptive 4.8 GiB tier, lossless). This plan uses the simulator in `tools/sim` (docs/GLM_SIM.md,
calibrated within 5 % on the quiet measurements) to rank architecture changes. Every number below that is not
marked measured is a prediction, with the simulator command that produced it.

## 1. Where a round goes today

MTP depth 2, split verify, prime fixture (measured 125-140 ms per 2.9-token round):

| Part | ms per round | Note |
|---|---:|---|
| CPU routed experts | 98-108 | 2.4 GB per token at 63-65 GB/s, memory-bound |
| GPU dense chain, shared and resident experts | 65-68 | hidden under the CPU pass by split verify |
| MTP drafts + resync | 13 | sequential GPU/CPU steps |
| head, tail, handoffs | 8 | |

The CPU expert pass is the critical path, and 78 % of its bytes are experts the GPU does not hold.

## 2. Hardware ceilings (roofline, pipeline-independent)

```
python3 tools/sim/glm_sim.py estimate --hw tr16 --pack q22 --mtp 3 --acceptance prime    # prints both ceilings
```

| Resource | Limit | Consequence |
|---|---|---|
| DRAM stream for experts | 72 GB/s (65 in the model) | q22, MTP window, 4 GiB tier: 1.9 GB/token -> 39 tok/s max |
| CPU quantized dot (15 Zen 1 cores) | 257 GMAC/s | 30 tok/s aggregate for all experts on the CPU, at any batch size |
| GPU dense weights | 6.45 GB per read at 381 GB/s = 17 ms | one read per token: 59 tok/s; 2 reads per 3.7-token round: 110 |
| GPU resident experts | 235 GB/s today, ~380 possible | 1.7 GB/token at 62 % share: 7.2 ms (138 tok/s) |
| PCIe Gen3 x8 | 7.2 GB/s | too slow to stream experts: 3 % of layer bytes already stalls the pipeline |

Two conclusions follow. First, with the experts on the CPU, lossless single-stream decode cannot exceed about 30
tok/s on this CPU, and a realistic pipeline reaches 25-27. Second, 50 tok/s needs at least 60 % of expert bytes
served from VRAM, which only routing bias (affinity) delivers on 16 GB, plus about half the GPU work per token.

## 3. The architecture

Ranked by predicted gain; "lossy" means the greedy output differs from the model's own.

**A. Resident-biased routing (affinity 0.15-0.20), lossy.** Already implemented (`STRATA_GLM_ROUTE_AFFINITY`);
measured 0.15 gives 62 % GPU share at KL 0.198 and top-1 85.9 %. It is the one lever large enough: CPU bytes fall
from 2.1 to 0.9 GB per token. Needs the KL/HumanEval gates and probably a per-layer margin (the first MoE layers
carry more weight). Pair with the adaptive tier and a denser tier format (D).

**B. Block drafter instead of sequential MTP, lossless by itself.** A DFlash2-style block-diffusion drafter
(block 8, about 512 MiB, one forward per round, conditioned on the target's hidden states; HyperDFlash shows how
to align it with mHC residual streams) replaces depth-4 MTP's 16 ms of drafts and 10 ms of resync with 3 ms, and
delivers 5.4 accepted tokens per round on code (published DFlash2 numbers). It does nothing while the CPU experts
are the bottleneck (verify bytes grow with the window: 5.3x one token at block 8, measured from the routing traces),
and becomes the main gain once A holds: 43 -> 51 tok/s. Keep the block at 8; blocks of 12-16 lose again.

**C. Expert skipping 10-20 % by gate threshold, lossy.** Measured: skipping 28 % of gate/up units cost about 9 %
of bytes; `ROUTE_MIN_SHARE 0.06` dropped 15 % of routes but cut acceptance from 89 to 83 %. Worth 2-4 tok/s per
10 %. Prefer dropping the lowest-weight routes only when their weight is below a per-layer threshold learned on
the calibration corpus.

**D. VRAM for the tier.** Q4_K copies of the dense weights (+840 MiB, GPU bytes -14 %, lossy in the dense path),
reserve 256 MiB and prefill scratch 512 MiB at decode time (+770 MiB), and a denser tier format for resident
experts (1.4x slots; lossy for the hottest experts, so gate it). Each GiB of tier is worth about 3.5 ms per token.

**E. GPU chain work.** CUDA graphs for the whole per-layer chain (44 -> ~10 launches per layer), one launch for all
resident experts of a layer tiled over the window's tokens (235 -> ~380 GB/s), fused router + mHC + mailbox
kernels (100 -> 50 us per layer-group). These do not move single-stream speed while the CPU binds, but they lift
the GPU ceiling from 36 to 46 tok/s and are required for A and B to pay off.

**F. Batched decode.** Rows share the dense GEMVs (2 reads per round regardless of rows) and dedupe some experts.
Aggregate throughput: 2 rows 51-58 tok/s, 4 rows 60. The CPU quantized-dot ceiling (30/(1 - GPU share)) caps 8
rows; a 2x faster dot kernel only helps there.

**G. From the literature survey (docs/GLM_DECODE_RESEARCH.md).** Two lossless additions: a cross-layer route
predictor that prefetches next-layer experts over PCIe during the CPU pass (+6 % here, +18 % on a Gen4 x16 link)
and a recency-based tier policy (+3 %). Two lossy ones: KTransformers-style expert deferral, which gives
ordinary and batched decode the overlap that split verify gives MTP (15 -> 24 tok/s single token, 24 -> 31 at
4 rows), and AcceptMoE-style verifier sizing (4 experts for draft positions >= 2: 23.7 -> 28.3 on the lossless
base at a 2 % acceptance cost, 57 -> 64 on the lossy stack). Together they move the one-card figure to 26 tok/s
lossless, 31-33 with tapering, and the lossy stack to 64 (75 with two rows).

**Not worth it here:** PCIe co-streaming of experts (Gen3 x8: -5 % even at 3 %; needs about 50 GB/s to help),
MTP deeper than 4, blocks wider than 8, a faster CPU dot kernel below 8 rows, codebook 2-bit formats on the CPU
(compute-bound at 37-42 GB/s). A second 16 GB card is the lossless route: 32-33 tok/s single-stream.

## 3b. Polished algorithms, our own additions, and four machines

The literature round (docs/GLM_DECODE_RESEARCH.md) added four algorithms; this round refines them and adds two
of our own. All are simulator knobs; `plan` now switches the lossless ones on by default (`--baseline` for today's
algorithms).

- **Graded expert deferral** (`--deferral-share f`). The all-or-nothing version overstated the gain: hiding the
  whole GPU chain under the CPU pass of a single token would defer 85 % of the routed work. With f = 0.3 ordinary
  decode goes 15.3 -> 16.7 tok/s, with f = 0.6 to 20.3; batched 4 rows 24.3 -> 28.1 at f = 0.3. Split-verify MTP
  already overlaps, so deferral is a tool for the ordinary and batched paths only.
- **Adaptive window** (ours; `--adaptive-window 1`, builds on `STRATA_GLM_MTP_CONTINUATION_MARGIN`). Drafting
  stops when the draft's confidence is low, so about half of the positions that would be rejected are never
  verified. Lossless; +1-3 % on MTP, +15 % on an 8-block drafter with chat-like acceptance.
- **Prediction-driven disk prefetch** (ours, same predictor as the PCIe prefetch). In the RAM tier the next
  layer's predicted misses are read from NVMe before they are needed, so only bandwidth remains: 64 GB exact mode
  4.5 -> 5.6 tok/s on one drive, 10.2 with two.
- **Tail-only affinity** (ours; `--tail-affinity x`). The routing bias is applied only to draft positions >= 2,
  the tokens most likely to be rejected anyway: 26.1 -> 30.1 tok/s at x = 0.10 (full affinity 0.10 gives 32.5),
  with the quality change confined to about half of the accepted tokens. Combined with 4-expert tapering and
  MTP5: 33.3.
- **Verifier tapering** (`--spec-tail-topk 4`) now also scales the routed MACs: lossless base 23.7 -> 28.3 (MTP3),
  29.5 (MTP5); the lossy 50 tok/s stack 57 -> 64, 75 with two rows.

Predicted decode tok/s per machine (prime fixture; lossless rows change no output, "experts lossless" keeps the
experts exact but uses Q4 dense copies; today's GPU kernels unless noted):

| Machine | Lossless, polished | Mild lossy | Full lossy stack |
|---|---:|---:|---:|
| tr16: 1950X, 124 GiB, 5060 Ti 16 GB | 26.1 (q22, MTP3, LRU + prefetch) | 32-33 (tail top-k 4, tail affinity); 36-38 (+ Q4 dense, skip 10 %, affinity 0.05) | 47 with today's GPU kernels; 64 with graphs, a 380 GB/s tier kernel and a block drafter |
| tr16 with 60 GiB RAM | 5.7 (exact RAM tier, one NVMe); 10 with two drives | 26-33 (REAP-50 or a frozen q22 set) | 46 (REAP-50, dflash, affinity 0.10) |
| Xeon 6240 x2 + 1 V100 32 GB | 42.6 (q23, MTP3); 52.6 (q2_orig, dflash 8) | 58 (dflash + tail top-k 4) | 61 |
| Xeon 6240 x2 + 2 V100 32 GB | 45.8 (MTP3); 58 (dflash 8); batch 4: 90 | 62 | 66 |
| B550: 3950X, 32 GB DDR4-2666, 3070 8 GB (unmeasured) | experts lossless: 2.8 (one NVMe), 4.5 (two), 6.4 (four); no verify history fits, so no MTP | 15.9 (frozen 22 GB q22 set, affinity 0.10, skip 10 %) | 18.7 (+ deferral 0.5) |

What the table says: with two V100s the experts are 95 % resident and a block drafter is the right speculation
(58 lossless); with one V100 and 20 GB of tier the same drafter already wins (52.6); on 16 GB cards the CPU
stream binds and MTP3 with LRU + prefetch is the lossless optimum; on 8 GB cards the dense copies fill the VRAM,
so only RAM-side levers (frozen or hybrid expert set, deferral) and faster disks matter.

## 4. Predicted ladder (single stream, q22 pack, prime fixture)

```
B="--pack q22 --acceptance prime --dense-format q4 --reserve-mib 256 --prefill-scratch-mib 512 \
   --param gpu_kernels_per_layer=10 --param gpu_layer_fixed_us=50 --set gpu.tier_gbps=380"
python3 tools/sim/glm_sim.py estimate --hw tr16 $B --mtp 4 --expert-skip 0.2 --affinity 0.15 --tier-compress 1.4
python3 tools/sim/glm_sim.py estimate --hw tr16 $B --speculation dflash --draft-model-mib 512 --expert-skip 0.2 --affinity 0.15 --tier-compress 1.4
```

| Step | Change | tok/s | CPU GB/token | GPU share | Bound | Lossless |
|---|---|---:|---:|---:|---|---|
| L0 | q22, MTP3, adaptive tier (today's design) | 23.7 | 2.14 | 22 % | CPU | yes |
| L1 | + Q4 dense copies, reserve 256, scratch 512 (D) | 24.8 | 2.02 | 26 % | CPU | no |
| L2 | + graphs, 380 GB/s tier kernel (E) | 25.0 | 2.02 | 26 % | CPU | no |
| L3 | + expert skip 10 % (C) | 26.9 | 1.82 | 26 % | CPU | no |
| L4 | + affinity 0.10 (A) | 36.4 | 1.15 | 53 % | CPU | no |
| L5 | affinity 0.15 | 40.4 | 0.92 | 63 % | GPU | no |
| L6-L8 | + MTP4, 1.4x tier format, skip 20 % | 42.1 | 0.76 | 65 % | GPU | no |
| L10 | + fused per-layer kernels (E) | 43.0 | 0.76 | 65 % | GPU | no |
| M4 | L10 with block drafter 8 instead of MTP4 (B), code text | **50.7** | 0.95 | 64 % | CPU = GPU | no |
| M5 / M6 | same, math / chat text | 59.8 / 46.8 | 0.80 / 1.03 | 64 % | | no |
| M9 | M4 with affinity 0.20 | 58.0 | 0.76 | 71 % | GPU | no |
| M11 | M4 with today's GPU kernels | 46.8 | 0.95 | 64 % | GPU | no |

Batched, same base (`--batch N`): MTP2 per row 39 / 51 / 60 tok/s aggregate for 1 / 2 / 4 rows; block drafter
51 / 58 / 61. Lossless with only the GPU work (E) done: 24 / 27 / 30 for 1 / 2 / 4 rows (MTP3).

## 5. Maximum decode performance on this machine (assessment)

| Mode | Predicted | Hard ceiling | What binds |
|---|---:|---:|---|
| single stream, lossless, one GPU | 25-27 (26.1 with LRU + prefetch) | 30 (CPU dot), 39 (DRAM with 4 GiB tier) | CPU expert bytes |
| single stream, lossless, two 16 GB GPUs | 32-33 | 44 | CPU expert bytes |
| single stream, lossy (A + B + C + D + E) | 47-60 by text type, 50 on code | 59 CPU / 58 GPU at 64 % share | both, balanced |
| single stream, affinity 0.20 | 58 | 72 CPU / 57 GPU | GPU dense reads |
| batched 4 rows, lossy | 60 | 67 CPU / 74 GPU | CPU dot rate |
| batched 8 rows, lossless | 32-35 | 30/(1 - share) | CPU dot rate |

Beyond about 60 tok/s the dense GEMVs (two reads per round) and the 257 GMAC/s quantized-dot rate bind; only a
faster CPU, a second card, or a smaller dense path moves them.

## 6. Order of work and what to measure first

1. GPU chain (E): graphs, batched resident-expert kernel, fused layer kernels. Measure the resident kernel's GB/s
   at widths 4-8 and the per-layer GPU time with nsys; the plan assumes 380 GB/s and 50 us.
2. VRAM (D): Q4_K dense copies, decode-time scratch release. Measure the tier size that results (plan: +1.6 GiB).
3. Affinity 0.15 with the quality gates (A), then expert skip (C) behind the same gates. Measure GPU share per
   fixture; the plan assumes 62-65 %.
4. Block drafter (B): train a DFlash2-style drafter on GLM hidden states (mHC-aligned); measure accepted length on
   prime / json / csv / coding fixtures. The plan assumes 5.4 on code; below 4.4 it falls back to MTP depth 4.
5. Batched decode (F) with per-row drafts in one window; check that the dense GEMVs really run once per group.

Re-run `python3 tools/sim/glm_sim.py validate` after adding each measurement to `tools/sim/data/measured_tr16.json`
so the remaining predictions are re-calibrated.

## 7. Risks

- Affinity 0.15-0.20 is a visible quality change (KL 0.2); the gates may stop at 0.10, which caps the plan at
  about 40 tok/s single-stream (block drafter + 53 % share).
- The block drafter's accepted length on GLM is unmeasured; DFlash2 numbers are for Qwen3.8-27B.
- The union-bytes model for 8-token windows comes from three coding/text traces; conversational text may share
  fewer experts between consecutive tokens.
- The simulator's GPU terms were calibrated with today's kernels; the 380 GB/s and 50 us assumptions are targets,
  not measurements.
