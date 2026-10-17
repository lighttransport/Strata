# llvq, GSQ and RCO for GLM-5.3-Flash experts: one-layer study (2026-10-16)

Three recent methods were checked for the routed experts: **llvq** (Leech-lattice vector quantization,
github.com/pjmalandrino/llvq, Rust), **GSQ** (Gumbel-Softmax learned scalar rounding, IST-DASLab/GSQ) and **RCO**
(budget-constrained bit allocation, IST-DASLab/RCO). They were run on 16 routed experts of layer 20 (8 most-routed,
8 median by the calibration coverage) next to the GGUF types the engine already runs, the stored EXL3 K2 pack and
our shipped Q23 file. Background and the earlier method survey: [GLM_QUANT_RESEARCH.md](GLM_QUANT_RESEARCH.md).

## Inputs

- **Ground truth**: the original BF16 weights of those experts, read with HTTP range requests from
  zai-org/GLM-5.3-Flash-BF16 (`tools/quant_study/fetch_bf16_experts.py`, 0.80 GiB).
- **FP8 is as good a source as BF16.** The official release zai-org/GLM-5.3-Flash is FP8 E4M3 with 128 x 128 block
  scales (328 GB). Decoded, it differs from BF16 by 3e-6 relative error; Q2_K, Q3_K and IQ2_S quantized from FP8
  versus from BF16 differ by less than 0.03% in error against BF16 (12 matrices). Requantization can start from
  the FP8 release.
- **EXL3 K2 dequant** of the `glm53f-exl` pack (`strata-glm-exl3-dequant`, `tools/glm_exl3_dequant.cpp`; matches the
  engine's EXL3 matrix product to 1.6e-6). Re-quantizing it ranks the methods exactly as the BF16 input does
  (each method's error against its input is the same for both inputs), but every result then also carries K2's own
  6.9% error against the truth; it is a usable proxy for ranking, not for absolute quality.
- **Importance**: per-expert diagonal second moments of the expert inputs from the REAP calibration
  (`prune/calibration/blk.20.{gu,down}.f32`), used as imatrix weights and as llvq/GSQ diagonal Hessians.
- **Activations**: layer-20 MoE inputs on the 8,192-token gate corpus (full 288-expert Q23, decode path,
  `STRATA_GLM_ACTIVATION_TRACE` with the new opt-in `_LAYER` / `_POSITIONS`), giving per-expert Hessians
  (`tools/quant_study/hessians.py`, down-projection inputs recomputed with the BF16 gate/up and the clamped SwiGLU).
  Split 80/20 by token: the held-out part measures **output error** `tr(D H Dᵀ)/tr(W H Wᵀ)` with `D = Ŵ − W`.

## Results

All 48 matrices (16 experts x gate/up/down), error against BF16; "weighted" uses the diagonal input moments:

| method | bits per weight | error | weighted |
|---|---:|---:|---:|
| EXL3 K2 (stored pack) | 2.01 | 6.9% | 6.9% |
| llvq Tetra, identity Hessian, rotated | 2.06 | 8.5% | 8.5% |
| llvq Tetra, diagonal Hessian, rotated | 2.06 | 8.9% | 8.0% |
| GSQ 2-bit, group 128, diagonal Hessian (layer-wise) | 2.125 | 15.0% | 14.9% |
| IQ2_XS + imatrix | 2.31 | 9.2% | 8.7% |
| IQ2_S + imatrix | 2.56 | 6.3% | 6.0% |
| Q2_K | 2.625 | 8.9% | 8.9% |
| Q2_K + imatrix | 2.625 | 7.4% | 7.0% |
| **shipped Q23** (Q2_K gate/up, Q3_K down) | 2.90 | **12.1%** | **11.8%** |
| IQ3_XXS + imatrix | 3.06 | 3.4% | 3.2% |
| Q3_K + imatrix | 3.44 | 2.1% | 2.1% |
| IQ3_S + imatrix | 3.44 | 1.9% | 1.8% |
| Q4_K + imatrix | 4.50 | 0.5% | 0.5% |

Hot and median experts behave the same for every method (within 0.1 pp).

Held-out output error on the 8 hot experts (682-1,586 routed tokens each in the corpus):

| method | bits per weight | output error |
|---|---:|---:|
| EXL3 K2 (stored) | 2.01 | 0.055 |
| llvq, diagonal Hessian | 2.06 | 0.064 |
| llvq, full Hessian | 2.06 | 0.093 |
| GSQ, diagonal Hessian | 2.125 | 0.121 |
| GSQ, full Hessian | 2.125 | 0.175 |
| IQ2_XS + imatrix | 2.31 | 0.068 |
| IQ2_S + imatrix | 2.56 | 0.048 |
| Q2_K + imatrix | 2.625 | 0.055 |
| shipped Q23 | 2.90 | 0.087 |
| IQ3_XXS + imatrix | 3.06 | 0.025 |
| Q3_K + imatrix | 3.44 | 0.016 |

The full-Hessian rows are worse than the diagonal ones: with 550-1,270 fitting tokens for a 4,096-wide Hessian,
GPTQ fits the sample, not the layer (the llvq paper calibrates on ~6,100 sequences). A fair full-Hessian comparison
needs a calibration run tens of times longer; the diagonal rows are the reliable ones here.

## What this means

1. **The shipped Q23 is the weakest point, and the fix needs no new code.** Its experts were requantized from
   Unsloth's UD-Q2_K_XL, already a 2-bit model, and carry 12% error against the original; the same Q2_K/Q3_K types
   quantized from the original with the importance weights the engine already collects reach 7.0% (gate/up) and
   2.1% (down), on the same kernels and at the same size. Requantizing from the FP8 release (now downloading to
   `models/glm53f/fp8`) is the first step.
2. **IQ2_S is the best 2.5-bit option of the existing types** (output error 0.048 at 2.56 bpw against Q2_K's 0.055 at
   2.625). IQ2_S already runs in the CPU expert lane, the GPU tier, the decode cache and the mixed worker (CUDA and
   HIP, non-canonical); only the canonical sum (`STRATA_GLM_CANON`) is Q2_K/Q3_K only. `strata-glm-q2-pack --fp8=DIR
   --gate-type=iq2_s --down-type=iq2_s` now writes IQ2_S (or any of the IQ/K types) straight from the FP8 release
   (layer 10 with the calibration importance: normalized MSE 0.062 gate/up, 0.045 down; 4.4 min per layer on 14
   threads).
3. **llvq** reaches IQ2_XS-level quality at 2.06 bpw (0.25 bits less) but trails EXL3 K2 at the same rate in the
   settings that could be calibrated here. The native port in [GLM_LLVQ.md](GLM_LLVQ.md) (2026-10-16) is bit-exact with
   the Rust encoder and encodes 153,800 blocks/s on the RX 9070 XT (Rust: ~25,000 on 16 threads), with AVX2, CUDA and
   HIP decoders; the full-Hessian question is still open (it costs ~1.5 PFLOP for all experts).
4. **GSQ** as released cannot be applied end to end: no GLM model wrapper, no GGUF K-quant projection in the code (the
   README's GGUF results are not reproducible from it), and its gains come from end-to-end block training with
   activations on multi-GPU nodes. Its layer-wise loop (reproduced here with its own quantizer and hyper-parameters)
   halves its GPTQ starting point's error (30.5% to 15.0% on one matrix) but stays far behind K-quants at similar
   size, because the 2.125-bit symmetric grid has less side information than Q2_K's 16-element sub-block scales and
   minimums.
5. **RCO** needs the model's loss through an HF forward over a multi-bitwidth database; it cannot run on GLM here.
   Its idea (pick a type per tensor under an exact size budget) can be applied with a proxy: the per-matrix errors
   above, weighted by routing share, solved as a knapsack over Q2_K / IQ2_S / IQ3_XXS / Q3_K.

## Reproduce

```sh
B=models/glm53f/quant-study; PY=$HOME/venvs/gsq/bin/python   # numpy + torch (cu128) + GSQ deps
python3 tools/quant_study/fetch_bf16_experts.py --layer 20 --experts 21,29,... --shared --out $B
build-glm-release/strata-glm-exl3-dequant models/glm53f/glm53f-exl 20 21,29,... $B/exl3-layer20.safetensors
# tools: $B/bin/ggml_quant (tools/quant_study/ggml_quant.cpp, libggml-base), $B/bin/llvq_layer
#        (tools/quant_study/llvq_layer, cargo +1.93.0, llvq ddecad31), GSQ checkout at 03fc1648 in $B/GSQ
$PY tools/quant_study/evaluate.py --root $B --methods q2_K,q2_K-im,...,llvq-diag-rot
$PY tools/quant_study/hessians.py --trace $B/trace-layer20.bin --layer 20 --root $B
$PY tools/quant_study/evaluate.py --root $B --oerr --inputs bf16 --experts 29,117,140,141,220,244,251,263 --methods ...
$PY tools/quant_study/evaluate.py --root $B --table        # or --oerr --table
```

Receipts: `$B/results-layer20.jsonl` (all 48 matrices), `$B/results-layer20-oerr.jsonl` (held-out output error),
`$B/bf16-layer20.manifest.json` (sha256 of every fetched tensor).
