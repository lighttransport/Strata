# GLM-5.3-Flash on Strata: which weights for which PC

A guide to picking a weight variant and configuration from the host RAM, the
GPU memory and the quality you can accept. Rows marked **measured** come from
the named machine and document; rows marked *estimated* or *untested* are
arithmetic from file sizes and the measured memory model, not runs. GLM-5.3-Flash
has 45 transformer layers, 42 of them MoE with 288 routed experts (top-8); the
REAP-50 variant keeps 144 experts per layer.

## How memory is used

- **Host RAM** holds the routed experts (file-mapped, read once per token at
  DRAM speed) plus about 6 GB of fixed weights and the MTP draft layer. The
  decoder's resident set is close to the expert bytes plus 3–7 GB; the machine
  needs roughly 8 GB above that for the OS, page cache churn and the server.
  Measured: full Q2/Q3 experts (111 GB) peak at 107.8–114 GiB RSS on a 128 GB
  host; the REAP-50 Q23 model (55 GB of experts) peaks at 52.9–55.9 GiB inside a
  60 GiB cgroup.
- **GPU memory** holds the dense weights (5.1–6.2 GiB), the KV/state for the
  context, verification history and the resident expert tier that cuts CPU
  traffic (6.6 GiB on a 16 GB card with a desktop attached). Prefill streams
  every expert through the GPU once per chunk, so a 16 GB card is enough for any
  variant; more VRAM mainly means a larger resident tier.
- **Decode speed** is bounded by routed expert bytes per token divided by host
  DRAM bandwidth (Q2_K/Q3_K experts read at 66–78 GB/s on the Threadripper, about
  26–30 GB/s on the dual-channel Ryzen), then raised by the GPU tier hit rate,
  speculative decoding (MTP) and the lossy routing affinity.

## Presets

| Host RAM | Weights (size) | Experts per token | Configuration | Measured / expected decode | Quality vs original UD-Q2_K_XL | Status |
|---:|---|---|---|---|---|---|
| 64 GB | `reap50/GLM-5.3-Flash-REAP50-Q23-assembled.gguf` (63.2 GB) | 144 experts, Q2_K/Q3_K; 1.5–2.4 GB/token | `configs/glm53f-reap50-q23-b550-60g-9070xt-experimental.json` in a 60 GiB cgroup; 16 GB GPU | **measured** 10.2 tok/s (B550, affinity 0), 16.5 at affinity 0.06; **measured** 23.5–24.0 on TR16's DRAM in a 60 GiB cgroup | PPL +29.8%, KL 0.59, top-1 70% (pruned model) | released |
| 64 GB | full model + `experts-q23.gguf` in RAM-tier modes (`STRATA_GLM_PREFER_RESIDENT_MIB`, `STRATA_GLM_RAM_MARGIN`) | 288 experts, part resident | `docs/GLM_Q2_DECODE_REDESIGN.md` "64 GB host" | **measured** 2.4 tok/s exact (disk fetch); 15.1 hybrid at margin 0.10 with PPL +1.3% in one run, 5.5–6.0 in the safe retest | hybrid KL 0.15; frozen 48 GB set KL 0.37 | research, not recommended |
| 96 GB | REAP-50 `IQ4_XS` (88.0 GB) or `Q4_K_S` (93.5 GB) from patrickbdevaney, no Strata pack | 144 experts, 4-bit; 3.0–3.4 GB/token | run the GGUF directly; needs `-DSTRATA_MMQ_KQUANTS=ON`; GPU tier unavailable for Q6_K/Q4_K experts | *untested*; the Q4_K_M base measured 10.3–12.5 tok/s on TR16 (64% more bytes than Q23) | better than REAP-50 Q23 (4-bit experts) but still pruned | *untested* |
| 96 GB | REAP-50 with a Q3_K gate/up, Q3_K down pack (*"q33"*) | 144 experts; ~2.2 GB/token | would need a `--gate-up=q3` option in `strata-glm-q2-pack` (today gate/up is always Q2_K) | *estimated* 18–22 tok/s on TR16-class DRAM | *untested* | not built |
| 128 GB | Unsloth UD-Q2_K_XL (108.7 GB) + `experts-q23.gguf` (105.5 GB; 120.9 GB effective) | 288 experts, Q2_K/Q3_K; 3.09 GB/token before tier hits | `configs/experimental/glm53f-q23-tr16-aff08.json`, 16 GB GPU, two NUMA nodes interleaved | **measured** 23.1 tok/s decode, 131 (2K) / 428 (8K) prefill | PPL +2.07%, KL +0.018, top-1 −0.29 pt; HumanEval 153 vs 154 | released, qualified |
| 128 GB | same + `experts-q22.gguf` (95.9 GB) | 2.8 GB/token | research profiles (soft acceptance depth 3, guard) | **measured** 31 tok/s (affinity 0.08), 36–38 with affinity 0.14 + top-k 6 | +3.95% PPL at 0.08; affinity 0.14 lost 10 HumanEval tasks (144/164) | research only |
| 128 GB | single-file assembled full model (*planned*, about 120.9 GB with Q23 experts, 111 GB with Q22) | as above | same profiles; no Unsloth shards, no fingerprint check | same as the sidecar runs (byte-identical tensors) | same | planned, see `docs/GLM_RELEASE_Q2Q3.md` |
| 160 GB | Unsloth UD-Q3_K_XL (147.5 GB) unpacked, or UD-IQ3_XXS (120.4 GB) | 288 experts, 3-bit; ~4.3 GB/token (Q3_K_XL) | run the GGUF directly; IQ formats decode at only 37–43 GB/s on the CPU (codebook bound) | *estimated* 12–15 tok/s (Q3_K_XL) on TR16-class DRAM; IQ3_XXS measured about 7.3 tok/s at bring-up | lossless w.r.t. that quantization | *untested* with the current engine |
| 192 GB | Unsloth UD-IQ4_XS (156.8 GB) | ~4.8 GB/token, IQ4_XS (codebook decode) | run directly | *estimated* ≤10 tok/s | — | *untested* |
| 256 GB | Unsloth UD-Q4_K_XL (199.7 GB) | ~5.9 GB/token, Q4_K (about 77 GB/s on TR16) | run directly with K-quant MMQ prefill | *estimated* 10–12 tok/s single node | near-lossless 4-bit | *untested* |
| 192 GB (TR16 128 + B550 64, two nodes) | GLM-5.3 (743B) REAP-37 Q2_K/Q3_K (164 GB), rows 75/25 over IB | 8.2 GB/token | needs a new `glm_moe_dsa` engine path | *simulated* about 9 tok/s | REAP-37 4-bit measured +35% perplexity upstream | planned, see `docs/GLM53_FULL_TR16_B550.md` |
| 64 GB | MiMo-V2.6-Flash REAP-50 Q2_K/Q3_K (about 59 GB) | 128 experts; 3.43 GB/token | needs a `mimo_v2` engine path | *simulated* 9–10 tok/s on B550 | REAP-50 loss unpublished | planned, see `docs/MIMO_V26_TR16_B550.md` |
| 128 GB | MiMo-V2.6-Flash (309B) Q2_K/Q3_K (about 114 GB) | 256 experts; 3.43 GB/token | needs a `mimo_v2` engine path | *simulated* 20 tok/s on TR16 | ≈ +5.5% perplexity (community 3.0-bpw measurement) | planned, see `docs/MIMO_V26_TR16_B550.md` |
| 192 GB (two nodes) | MiMo-V2.6-Pro (1T) REAP-50 Q2_K (164 GB) or REAP-62 Q2_K/Q3_K (136 GB) | 6.8–7.5 GB/token | needs `mimo_v2` plus the remote worker | *simulated* about 8 tok/s | unknown (no Pro REAP exists) | planned |
| 2 × 128 GB (two nodes) | full model split across nodes over IB | half the experts per node | see `docs/GLM_MULTINODE_PLAN.md` | *planned* | — | planned |

Notes:

- The 64 GB and 128 GB rows are the released configurations. Everything from
  96 GB upward without a Strata pack runs the upstream GGUF as-is and loses the
  Q2_K/Q3_K fast paths (canonical kernels, fused quantization, the GPU tier for
  IQ/K-quant experts is not implemented), so more RAM does not mean more speed
  today; it means less quantization loss.
- A larger pack family (Q3_K gate/up for 96 GB REAP-50, or a Q4_K-expert pack
  for 192–256 GB full-model hosts) needs pack-tool and tier work before it can
  be measured; the sizes above are the only facts.
- VRAM below 16 GB: 8 GB was tested once on TR16 with a 28 GiB RAM cgroup
  (`docs/GLM_32G_8G.md`) and failed to reach usable speed (0.3 tok/s). With
  128 GB host RAM an 8–12 GB card should still work with a smaller decode cache
  (`--decode-cache-mib`), at reduced speed; not measured.
- Dual GPUs on one node: measured only on 2×V100 32 GB (28.9 tok/s with the
  unpacked UD-Q2_K_XL, 160 GB host), see `docs/GLM53_V100.md`.
