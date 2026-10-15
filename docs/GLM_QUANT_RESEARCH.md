# Quantization options for GLM-5.3-Flash experts (research note, 2026-10-16)

What this fork ships today: **Q23**, the Unsloth UD-Q2_K_XL model with the routed experts requantized to Q2_K
(gate/up) and Q3_K (down). Full model 120.9 GB; REAP-224 (224 of 288 experts per layer, sliced without
requantization) 95.5 GB at +4.9% perplexity and HumanEval 152/164 (150 on a rerun); REAP-50 (144 experts, Q23 in a
Q4_K_M base) 63.2 GB at +30% perplexity against the 2-bit original. The question tracked in
[TODO.md](TODO.md) section 2 / lighttransport/Strata#2: smaller expert weights at the same quality, or better
quality at the same size. This note reviews recent methods for fit with this engine, where routed experts run on
the CPU from RAM (any ggml type through `native_fmt` in `src/kernels/cpu/native_expert.cpp`) and the GPU-resident
tier, the expert worker and the canonical CPU/GPU sum handle Q2_K/Q3_K only.

## Methods reviewed

### GSQ: Gumbel-Softmax scalar quantization (IST DASLab)

[arXiv 2604.18556](https://arxiv.org/html/2604.18556v2), code [IST-DASLab/GSQ](https://github.com/IST-DASLab/GSQ).
Each weight gets learnable logits over candidate grid values; Gumbel-Softmax sampling makes the assignment
differentiable, scales are learned jointly, and the run is warm-started from GPTQ (Lion optimizer, annealed
temperature). Results: at 2 bits it closes most of the gap to vector quantization (Llama-3.1-70B zero-shot average
75.6 against QTIP 77.3 and EfficientQAT 71.4); at 3 bits it is within 0.2-0.5 of QTIP.

Two properties make it the best fit here:

- **It can start from an existing GGUF K-quant checkpoint and project the result back into the same format**
  (named types: Q2_K, Q3_K, Q3_K_M; the GGUF code or grid index of each weight is the starting point). The output
  keeps the Q23 layout, so the CPU expert pool, the canonical Q2_K/Q3_K kernels, the GPU tier, the worker and REAP
  slicing need no change. It is a weight-values improvement only.
- **It quantizes MoE models expert by expert.** For Kimi-K2.5 (1T parameters) each of the 384 experts per layer is
  optimized independently and attention is left unquantized; 2.13 bits per parameter kept most benchmark scores
  (AIME25 93.0 vs 95.3 base, LiveCodeBench 69.4 vs 61.4). The per-expert split keeps each subproblem small.

Cost: end-to-end 24 hours on 8x H200 for Kimi-K2.5 and 68 hours for dense Llama-3.1-70B. GLM-5.3-Flash is about
321B parameters with 288 experts per layer; a run needs a rented multi-GPU node, or a much longer run on local 16 GB
cards (feasible in principle because of the per-expert decomposition; not measured). The auxiliary logits add
memory proportional to the number of candidate grid values per weight (4 logits per weight at 2 bits, 5 at 3 bits).
Calibration in the paper used OpenThoughts (plus OpenPlatypus and FineWeb-Edu for Kimi-K2.6); our own calibration
corpus (`tools/glm_q2_corpus.py`) and routing statistics already exist.

### RCO: per-tensor type allocation under a size budget

Used with GSQ by ISTA-DASLab (Helcig and Alistarh 2026; we did not read the RCO paper itself). It assigns one GGUF
type per tensor so the file meets an exact size budget. Their
[Qwen3.8-Flash-Next GSQ-RCO GGUFs](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) (512 experts,
48 layers) range from Q2_0 at 2.40 bpw to IQ3_S at 3.50 bpw; their card reports benchmark parity with BF16 at
IQ3_XXS (task average 92.6 vs 93.1) and a small loss at Q2_0 (89.1). Allocation is per layer, not per expert, and
experts get most of the budget (95% of the searchable weights). The formats used are Q2_0, IQ2_XS, IQ3_XXS and
IQ4_NL; Q2_0 decodes about 3.4x faster at prompt time than IQ2_XS on their measurements. Upstream Strata already
runs these files (Q2_0 and IQ kernels in `src/prefill/moe_mmq.cu`, `src/kernels/cpu/iq_avx*.cpp`); for GLM the CPU
side would work through `native_fmt`, but the GLM GPU tier, worker and canonical sum would need Q2_0/IQ paths on
CUDA and HIP.

### GyRot (KAIST, HPCA 2026)

[arXiv 2607.27694](https://arxiv.org/abs/2607.27694). An algorithm-hardware co-design for **4-bit weights and
activations** on an INT4 systolic-array accelerator: coarse rotation scope combined with fine (32-channel)
asymmetric group quantization, a "harmonic-aligned permutation" that maps outlier channels onto Hadamard rows, and
zero-point rounding for integer-only dequantization; evaluated on LLaMA-family models against QuaRot and SpinQuant.
**Not a fit**: it targets W4A4 compute on custom hardware, not 2-3-bit weight-only experts streamed from RAM, and has
no MoE evaluation. One finding transfers as a caution: rotation and fine-grained group quantization interfere unless
the rotation scope is coarser than the group, which matters if we ever try Hadamard rotation on K-quant experts
(256-element super-blocks, 16-element sub-blocks).

### Trellis and vector quantization (QTIP, EXL3, IQ*_KT)

[QTIP](https://arxiv.org/pdf/2406.11235) and its streamlined variant EXL3
([exllamav3](https://github.com/turboderp-org/exllamav3)) give the best quality per bit at 2 bits;
[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) adds integer trellis types (IQ1_KT to IQ4_KT) with
reasonable CPU speed, and reports IQ2_KT at ~2.125 bpw below IQ2_KS in perplexity (secondary source; quantization
is about five times slower). This fork has a native EXL3 loading path for existing EXL3 models
([GLM_EXL3.md](GLM_EXL3.md)). Cost here: trellis decode kernels on the CPU expert pool and on both GPU vendors, and
EXL3 MoE CPU offload exists only in third-party forks. A later option.

### Sharing structure across experts

[MoBE](https://arxiv.org/abs/2508.05257) (mixture of shared bases), [D2-MoE](https://arxiv.org/abs/2502.17298)
(shared base plus low-rank per-expert deltas), [SLBF](https://arxiv.org/abs/2610.09342) (data-free shared low-rank
bases, EMNLP Findings 2026, 16B-122B models) and BitsMoE (shared SVD basis, quantized expert-specific factors; seen
only as a listing excerpt) reduce expert bytes by factoring experts, orthogonal to scalar quantization and to REAP.
They need new expert kernels (a basis product plus per-expert factors) and change the compute pattern of the CPU
lane. Research-grade for this engine.

### Mixed precision across experts

Q-Strata ([arXiv 2608.30564](https://arxiv.org/pdf/2608.30564)) and Colla-Q
([arXiv 2609.18131](https://arxiv.org/pdf/2609.18131)) allocate bits per expert or block (Colla-Q: 4-bit for the top
25% of blocks, 2-bit for the rest, about 2.54 average bits). We already have per-expert routing frequencies (the
coverage priors) and REAP saliency to drive such an allocation. Q2_K/Q3_K/Q4_K mixing per projection already works;
per-expert type mixing needs a pack-format change (one tensor per layer holds all experts today).

### What the community GLM-5.3-Flash GGUFs show

Builds that keep attention, shared experts, router and MTP at higher precision and lower only the routed experts
do best; IQ2_XS and below degrade sharply; Unsloth's UD-IQ3_XXS (~120 GB) is the usual 128 GB choice (model cards
of [Unsloth](https://unsloth.ai/docs/models/glm-5.3-flash) and community uploads; self-reported). This matches the
Q23 design.

## Recommended path

| step | what | engine work | gate | cost |
|---|---|---|---|---|
| 1 | GSQ refinement of the Q23 experts in place (K-quant projection mode, per expert), for the full model, REAP-224 and REAP-50 | none | corpus `tr16-35` and HumanEval against current Q23 | rented GPU node; Kimi-K2.5 took 24 h on 8x H200 |
| 2 | per-layer type allocation among Q2_K / Q3_K / Q4_K under a size budget (RCO-style), using coverage and REAP saliency | none (types already supported) | same | CPU/GPU hours for the search, no new kernels |
| 3 | Q2_0 / IQ2_XS / IQ3_XXS experts (GSQ-RCO-style files) | GLM GPU tier, worker and canonical sum for the new types, CUDA and HIP | same | kernel work |
| 4 | trellis experts; shared-basis expert compression | new CPU and GPU expert kernels | same | research |

GyRot: no action. Expected effect of step 1, from the paper's results only: part of REAP-224's +4.9% perplexity
recovered at 95.5 GB, or Q23-level quality at a REAP-50-class size; not measured on GLM. Step 1 needs a decision on
GPU budget (local cards or a rented node) before it starts.
