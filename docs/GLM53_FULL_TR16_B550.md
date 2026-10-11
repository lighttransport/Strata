# GLM-5.3 (743B, non-Flash) on TR16 + B550: plan and simulator study (2026-10-10)

Can the full GLM-5.3 run on the two-node setup of TR16 (Threadripper 1950X, 128 GB
DDR4 at about 72 GB/s, RTX 5060 Ti 16 GB) and B550 (Ryzen 3950X, 62 GB DDR4-2133 at
about 26–29 GB/s, RX 9070 XT 16 GB), 192 GB of RAM in total, joined by a
ConnectX-3 InfiniBand link (currently PCIe-lane-degraded, 1.65 GB/s) and 1 GbE?
Short answer: **only as a REAP-pruned model, at roughly 9–10 tokens/s predicted,
after a new model implementation in Strata.** Nothing below is measured; every
speed is a simulator prediction (`tools/sim/glm_sim.py --model glm53`) and every
quality number is quoted from a community 4-bit MLX build.

## The model is not Flash

From [zai-org/GLM-5.3 config.json](https://huggingface.co/zai-org/GLM-5.3):

| | GLM-5.3-Flash (supported) | GLM-5.3 |
|---|---|---|
| Architecture | `glm5next`: KDA + MLA hybrid, mHC 4-stream residual | `glm_moe_dsa`: DeepSeek-V3.2-style MLA with a lightning-indexer sparse attention (DSA), plain residual |
| Layers | 45 + 1 MTP, 3 dense, 42 MoE | 78 + 1 MTP, 3 dense, 75 MoE |
| Hidden / expert rows | 4096 / 2048 | 6144 / 2048 |
| Routed experts | 288, top-8 | 256, top-8, 1 shared |
| Attention | 11 MLA + 31 KDA | 78 MLA (q_lora 2048, kv_lora 512, 64 heads, qk 256, v 256); indexer weights on 21 layers (index top-k 2048), the other 57 reuse the previous selection |
| Routed parameters | 170B | 724.8B (37.75M per expert) |
| Total | about 180B | 743B |
| License | MIT | "GLM-5.3 License" (permissive, notice required) |

**Strata cannot load it today.** The decoder, the expert-pack tool and the
assembler are written for `glm5next` (hidden 4096, KDA, mHC); `strata-glm-q2-pack`
rejects other geometry. Required engine work:

1. A `glm_moe_dsa` model path: MLA everywhere (Flash's MLA kernels are a starting
   point; head size and ranks differ), no KDA, no mHC, the lightning indexer and
   its layer-sharing schedule (`indexer_types`), sigmoid routing with
   `noaux_tc` correction bias and `routed_scaling_factor` 2.5, MTP layer 78.
2. Generalize the GGUF descriptor, the CPU expert pool, the GPU tier, the pack
   tool (hidden 6144, 256 experts) and the remote-TP worker.
3. A REAP or pruning step (below).

The indexer only matters for prompts longer than 2048 tokens; a first version
could run dense MLA up to 2K and add DSA second.

## Sizes

Upstream GGUFs from [unsloth/GLM-5.3-GGUF](https://huggingface.co/unsloth/GLM-5.3-GGUF):
UD-IQ1_S 216.7 GB, UD-IQ1_M 228.5, UD-IQ2_M 238.6, UD-Q2_K_XL 253.9, UD-IQ3_XXS
281.7, UD-Q3_K_XL 343.0, UD-IQ4_XS 365.3, UD-Q4_K_XL 467.3 GB. **None fits 192 GB**:
each node needs about 7 GB of headroom, so at most about 117 GB of experts on
TR16 and 53 GB on B550 (about 170 GB in total, less once B550's share is rounded).

Simulator pack table (routed experts only; fixed weights about 12.6 GB at the UD
mix, 9.9 GB as Q4_K on the GPU):

| Pack (`--pack`) | Experts kept | Routed GB | GB read per token | Fits 192 GB split |
|---|---:|---:|---:|---|
| `glm53_q23` (Q2_K/Q2_K/Q3_K) | 256 | 262.4 | 8.20 | no |
| `glm53_iq2` (IQ2_XS/IQ2_XS/IQ3_XXS) | 256 | 232.2 | 7.25 | no |
| `glm53_reap25_iq2` | 192 | 174.1 | 7.25 | only with an expert split at 0.35 (not implemented) |
| `glm53_reap25_q23` | 192 | 196.8 | 8.20 | no |
| `glm53_reap37_q23` | 160 | 164.0 | 8.20 | yes, rows 0.75 / 0.25 (114.5 + 38.2 GiB) |
| `glm53_reap37_q22` | 160 | 148.6 | 7.43 | yes, 0.75 / 0.25 |
| `glm53_reap50_q23` | 128 | 131.2 | 8.20 | yes, 0.75 / 0.25 |
| `glm53_reap50_q22` | 128 | 118.9 | 7.43 | yes (also fits TR16 alone, barely) |

REAP keeps the bytes read per token unchanged (top-8 of fewer experts); it only
cuts what must be resident. Bits per weight set the speed.

## Quality evidence for pruning

The only GLM-5.3 REAP numbers available are from
[pipenetwork's MLX builds](https://huggingface.co/pipenetwork/GLM-5.3-REAP50-MLX-4bit)
(REAP saliency over 65,536 calibration tokens; wikitext-2 perplexity, 4-bit, all
on identical windows):

| Build | Size | Perplexity | vs full 4-bit |
|---|---:|---:|---:|
| full, 4-bit | 418.6 GB | 2.864 | — |
| full, mixed 3/6-bit | 332.6 GB | 3.034 | +5.9% |
| REAP-25 (192 experts), 4-bit | 316.6 GB | 3.287 | +14.8% |
| REAP-37 (160 experts), 4-bit | 267.2 GB | 3.852 | +34.5% |
| REAP-50 (128 experts), 4-bit | 214.7 GB | 5.030 | +75.6% |

Pruning costs much more on GLM-5.3 than on Flash, and 2–3-bit experts will add
to it (on Flash, Q2_K/Q3_K experts cost +3.95% perplexity with Q4 dense weights and +2.07% with affinity 0.08 on top of unchanged Q23). REAP-37 is the
deepest prune that plausibly stays useful; REAP-50 is a 76% perplexity hit
before quantization.

## Predicted speed (simulator, TR16 calibration, Q4_K fixed weights, 3000 MiB GPU tier)

`python3 tools/sim/glm_sim.py --model glm53 multinode --node tr16 --node b550 --link ib --params tools/sim/data/params_tr16_20261009.json --pack <pack> --dense-format q4 --speculation mtp --mtp 1 --decode-cache-mib 3000`

| Pack | Best split | Decode, IB | Decode, 1 GbE | Bottleneck |
|---|---|---:|---:|---|
| `glm53_reap37_q23` | rows 0.75 / 0.25 | 9.2 (8.9 with MTP depth 3) | 6.7 | TR16 DRAM |
| `glm53_reap37_q22` | 0.75 / 0.25 | 10.0 | — | TR16 DRAM |
| `glm53_reap50_q23` | 0.75 / 0.25 | 9.2 | — | TR16 DRAM |
| `glm53_reap50_q22` | 0.75 / 0.25 | 10.0 | — | TR16 DRAM |
| `glm53_reap25_iq2` | experts 0.65 / 0.35 (hypothetical split) | 7.0 | — | CPU compute (IQ codebook decode) |

Prefill with a single 8K pass and streaming prefetch: 263 tokens/s predicted for
REAP-37 Q23 (GPU-bound: mixers 20 s, expert GEMMs 11 s, PCIe 23 s for 164 GB).

Why it tops out near 10 tokens/s: a token reads 7.4–8.2 GB of routed experts.
TR16 alone would need 105–115 ms for that at 72 GB/s, and B550 can take only a
quarter of the rows before it becomes the slow side (26–29 GB/s). With 75 MoE
layers the IB round trips cost about 1.6 ms per token, small next to the 85 ms of
expert reads, so the split gains about 24% here (more than on Flash). Over 1 GbE
the round trips cost 30 ms per token and the split loses.

The second GPU (B550's 9070 XT) is unused in this layout: the remote worker is
CPU-only. Giving it a resident expert tier for B550's rows (4–6 GB, about 3% of
the experts) is worth about 2–4% and is not a priority.

## Recommended configuration

**REAP-37 (160 experts) with Q2_K gate/up and Q3_K down, rows split 75/25 over
InfiniBand, fixed weights in Q4_K on TR16's GPU, MTP depth 1.** It fits (114.5
GiB on TR16, 38.2 GiB on B550), predicts about 9 tokens/s and 260 tokens/s
prefill at 8K, and is the deepest prune with a plausible quality (REAP-37 4-bit
measured +35% perplexity; Q2/Q3 adds more). REAP-37 Q22 is the faster, lossier
alternative (10 tokens/s). If quality matters more than speed, REAP-25 needs
about 175 GB even at IQ2/IQ3 and does not fit 192 GB with the implemented row
split.

## Plan

1. **Engine bring-up of `glm_moe_dsa`** on TR16 alone with a small artificial
   checkpoint, then layer-by-layer parity against `transformers` on short
   prompts (dense MLA path, prompts ≤ 2048 tokens). Largest single work item.
2. **REAP-37 calibration.** Full-model calibration does not fit 192 GB at any
   precision above 2 bits. Two routes: (a) take pipenetwork's kept-expert sets if
   they publish them (their REAP builds are subsets of the 256 experts; the
   indices can be recovered by matching expert weights against the BF16 release
   layer by layer), or (b) count router statistics with Strata's prefill path on
   the UD-Q2_K_XL (254 GB, file-mapped from NVMe: each 8K prefill chunk streams
   all experts once, about 2 minutes per chunk from disk) over the 32,768-token
   calibration corpus, then rank by `router_weight × ‖expert_output‖`.
3. **Pack conversion** to Q2_K/Q3_K for the kept experts (generalized
   `strata-glm-q2-pack`), then the row split shipped to B550 by the remote-TP
   setup (38 GiB, about 25 s over IB).
4. **Measure** decode and prefill on held-out 2K/8K prompts against the
   simulator rows above, and quality on the 8,192-target corpus and HumanEval
   against the REAP-37 4-bit numbers.
5. **Fix the IB link** (PCIe lanes) before measuring: a healthy QDR link halves
   the round trip; the simulator's `ib-qdr` preset gives the expected effect.

## Simulator changes

`tools/sim/model.py` adds `GLM53_GEOMETRY`, estimated fixed-weight bytes and VRAM
tables, and the `glm53*` pack family (full, REAP-25/37/50 × Q23/Q22/IQ2/Q4);
`model.use_model()` switches every module, and `glm_sim.py --model glm53 …`
selects it for any subcommand. Fixed-weight sizes are estimated from the config at
an assumed 5.5 bits per weight (not parsed from a GGUF header), the GPU kernel and
CPU expert parameters are the TR16 Flash calibration, and MTP acceptance, tier
hit curves and quality estimates are Flash-derived. Treat every GLM-5.3 number as
an order-of-magnitude guide (±40%) until a GLM-5.3 run exists. Tests:
`tools/test_sim_multinode.py` (`Glm53ModelTest`), with the Flash numbers pinned
unchanged after switching models.
