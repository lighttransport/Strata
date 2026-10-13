# REAP-224 on TR16: plan for 600+ tok/s prefill and 35+ tok/s decode

Optimization stopped at the user's request on 2026-10-13. This document retains the plan and historical
results. Failed runtime experiments were removed during release preparation; see
[the release record](GLM_RELEASE_20261013.md) for the retained code and validation. The future work below
is not scheduled as part of the release.

Architectural follow-up, 2026-10-13: the next two-hour iteration raised the prefill target to 700 tok/s at 8K.
Implemented opt-in ready/early-device expert merging, fused KDA verification, uneven verify groups,
device-resident draft hidden rows, selective MTP extension, and long-prefill group/top-k budgets and MMQ tile
hints. Decode did not reach 35 at both widths. The default profile is unchanged; 2,048 default-output tokens
matched the frozen baseline. CUDA and HIP compile, and the new functional checks pass. Quality gates were
deferred at the user's request. See "Architectural continuation" in [GLM_PRUNE_224.md](GLM_PRUNE_224.md) and
[its evidence](fixtures/glm_tr16_arch_20261013/summary.json) for the prefill screens and repeats. Nine groups
with top-k 6 reached 718.39 tok/s median over three 8K passes in one process, with quality unqualified.

The user authorized a further one-hour iteration. Staged admission now measures B after a prefix of A's
CPU experts: front sizes 1, 2 and 4 measured 29.30, 29.48 and 29.64 tok/s at 2K, below the preceding 30.06
control. The largest prefix saved only 1.0% of CPU expert bytes. An additional opt-in route-first GPU order
publishes both groups before their resident/shared experts; with a two-expert prefix it measured 29.27
tok/s at 2K. Its checks and measurements are recorded in
[the extension evidence](fixtures/glm_tr16_incremental_20261013/summary.json). Neither experiment changes
the tuned profile. Final default controls measured 29.89/31.55 tok/s at 2K/8K, with all 2,048 warmup and
timed tokens matching the frozen baseline. The decode target remains unmet. The exact expert-MMA and
fused MLA kernels below remain future work. The new lossy prefill budgets later failed the release
rejection screen and were removed; a speed screen alone was insufficient.

Decode follow-up, 2026-10-13: the coarse CPU schedule and causal window-scoring experiments missed their 5%
gates. Existing recency/unbiased tier screens also stayed below 35 tok/s; readback tracing did not establish
0.5 ms/token of removable stalls. The tuned profile remains unchanged. See "Decode-35 experiments" in
[GLM_PRUNE_224.md](GLM_PRUNE_224.md) for the fresh 29.87/31.55 tok/s held-out baseline, protocol and results.
The continuation used mapped weights and did not retry huge-page allocation or force memory compaction.

Status on 2026-10-12, measured on TR16 (Threadripper 1950X, DDR4, RTX 5060 Ti 16 GB on PCIe 3.0 x8, 7.2 GB/s),
`configs/experimental/glm53f-q23-reap224-tr16-tuned.json`, quality gate passed (corpus perplexity +4.90%, KL +0.0636,
top-1 −1.90 pp vs 288-expert Q23; HumanEval 152/164, the floor):

| | 2K prompt | 8K prompt |
|---|---:|---:|
| Prefill (tok/s) | 168 | 496 |
| Decode (tok/s) | 29.0 | 31.6 |

Details and every rejected lever: `docs/GLM_PRUNE_224.md`, "Tuning the 224-expert Q23 file".

## 1. What bounds each number

**Prefill below about 7K tokens is PCIe-bound, not kernel-bound.** A 2K chunk routes 16,384 token-expert pairs per
layer over 224 experts (about 73 tokens each), so nearly every expert is used and all 88.5 GB of routed weights
cross the 7.2 GB/s link once per chunk: 12.3 s, i.e. 167 tok/s at 2K, which is what is measured (168). The link
allows 333 tok/s at 4K, 500 at 6K and 682 at 8K. No GPU kernel changes these numbers; only moving fewer bytes
per prompt token does.

**Prefill at 8K is GPU-compute-bound** (16.65 s of kernels against 12.3 s of transfer). Nsight Systems, 8K
single pass: expert MMQ 5.2 s (Q2_K/Q3_K at about 28 TOPS), FP16 dense projections 3.4 s (about 36 TFLOPS, the
FP32-accumulate peak), MLA attention 2.6 s, KDA 1.3 s, FP16 casts 0.5 s, the rest about 3.7 s.

**Decode is host-memory-bound.** The CPU reads about 1.56 GB of non-resident expert weight per token at about
70 GB/s (55% of routes already hit the 6.75 GB GPU tier); the GPU waits on the CPU about 80% of each verify round.
Rejected: higher affinity and per-layer profiles (corpus gate), Q2_K down projections (HumanEval 139/164), more
threads or NUMA-local placement (no gain), PCIe streaming of cold experts (decode −30% under a 7.2 GB/s copy,
because the decoder's own PCIe traffic queues behind it).

## 2. Prefill plan

Target: 600+ tok/s sustained for prompts of 8K and longer on TR16; the best achievable below that, with the
PCIe limit stated.

### P0. Measurement and quality tooling (first, about 1 day)
- **Prefill-path quality check.** The corpus gate runs teacher-forced verify windows and never exercises batch
  prefill, so a prefill-only precision change cannot be judged today (an FP16-accumulate trial diverged at token
  13 and could not be scored). Add an evaluation that prefills each corpus sequence in one batch and compares
  every position's logits with the reference (`--eval-corpus` with `--eval-prefill`), reporting the same
  perplexity, KL and top-1 as the existing gate.
- **Per-phase prefill timing** in the decoder log (expert GEMM, projections, attention, KDA, transfer wait per
  layer) so each kernel's gain is measured in the real pipeline, not only in Nsight.
- Gate for every lossy prefill step: prefill-path KL within +0.01 of the unchanged path, plus the existing corpus
  gate and HumanEval.

### P1. Expert GEMM: int8 MMA kernel for Q2_K / Q3_K (largest item, 5.2 s → about 2 s; 4–6 days)
- Today: llama.cpp MMQ, which unpacks K-quant scales inside the inner loop (measured on this card, 4096 x 512 x
  14336: Q2_K 30.8 TFLOPS, Q3_K 37.1, Q8_0 50.7).
- New kernel: expand each prefetched expert group once on the GPU into an MMA-friendly layout: int8 codes
  premultiplied by their 4-bit sub-block scale (exact: at most 45 for Q2_K), one FP32 super-block scale per 256,
  and the Q2_K minimum term kept as a separate per-16 correction applied through the activations' 16-element
  sums (as MMQ does). Activations stay q8_1 (per-32 scale), so the arithmetic matches MMQ's.
- Grouped over the routed experts with the existing dest/source permutation; cp.async double buffering and
  m16n8k32 s8 MMA, reusing the pipelined grouped-GEMM structure from `experiment/fp4/gpu/mma_native.cu`, which
  reached 74 TFLOPS (MXFP4) on this card.
- Target 70–90 TOPS. Expansion costs one read of the group and one 8-bit write (about 0.3 s per 8K pass in VRAM
  bandwidth), overlapped with the previous group's GEMM.
- Validation: parity against MMQ on real expert groups (relative L2 < 1e-6 against an FP64 reference of the same
  quantized inputs), then the prefill-path check (should be exact up to FP32 summation order).

### P2. Dense projections in block-scaled FP8 (3.4 s → about 1.3 s; 3–4 days, lossy)
- The 1,800 FP16 GEMMs (MLA q_b and the other dense projections, 256–1,024-token tiles, 16,384-wide outputs) run
  at the FP16/FP32-accumulate peak. `kind::mxf8f6f4` with UE8M0 scales per 32 runs at 151 TFLOPS on this card
  (measured in `experiment/fp4`).
- Dequantize the dense weights once at load to E4M3 with per-32 scales (from Q4/Q5/Q8 sources the error is small
  next to their own quantization); quantize activations per 32 in the producer kernel (removing the 0.5 s of
  FP16 casts at the same time).
- Alternative if quality fails: FP16 accumulation (measured +7% end to end, 485 → 521 tok/s), only for the
  projections the prefill-path check accepts.

### P3. Fused MLA prefill attention (2.6 s → about 1.2 s; 3–4 days)
- Today: gather to FP16, cuBLAS QK, softmax, cuBLAS PV, magma SGEMM (five launches per tile, the score matrix
  through VRAM).
- Replace with a flash-attention-style kernel over the latent cache (FP16 MMA, online softmax, no score matrix
  in memory), causal tiles skipped.

### P4. Overlap and small items (1–2 days)
- Skip the transfer of experts already resident in the decode tier during prefill (6.75 GB, about 8% of the
  bytes): reuse the resident copies in the prefill path instead of streaming them again.
- KDA tiling (1.3 s; chunk/row/column variants measured flat, so a kernel change is needed for more than 10–20%).

### Expected result
- GPU work at 8K falls from 16.65 s to about 9.5–10.5 s, below the 11.3–12.3 s of transfer, so prefill becomes
  PCIe-bound again: about 670–720 tok/s at 8K, about 1,300 at 16K with `--prefill-batch=16384`.
- Below 8K the PCIe limit stays (2K 167–180, 4K about 360, 6K about 540) unless fewer bytes move:
  - **Batching prompts** across requests in the server, so one expert transfer serves several prompts (best
    for concurrent use; no quality cost).
  - **Lossy low-count expert skipping in prefill**: at 2K many experts get only a few tokens; routing those few
    tokens to their next-best already-transferred expert cuts the bytes. Gate it with the prefill-path check.
    Only this or a faster link moves 2K materially.
  - **Hardware:** the 5060 Ti supports PCIe 5.0 x8; a Gen4 x16 or Gen5 slot would raise the link to
    14–28 GB/s and the 2K figure in proportion. On TR16's Gen3 x8 link, 600+ at 2K is not possible without
    skipping most expert transfers.

## 3. Decode plan (target 35+ at 2K and 8K; +21% and +11% needed)

Each item is measured alone at 2K and 8K (512 tokens x 3), with the CPU expert GB/s and bytes per token.

### D1. Host memory: huge pages and NUMA-replicated hot cold-experts (lossless; 2–3 days)
- The expert arenas request transparent huge pages, but after page-cache churn the kernel often cannot provide
  them (2026-10-12: kcompactd at 98% at decode start). Measure the decoder's AnonHugePages; if low, the expert
  pass pays TLB misses. Option A (needs root, user action): reserve hugetlbfs pages (`vm.nr_hugepages`) and map
  the owned arenas with MAP_HUGETLB (the pinned allocator already supports it).
- The pruned file freed about 25 GB of RAM. Replicate the most-used non-resident experts on both NUMA nodes so
  each node's threads read local memory (the 1950X's cross-die bandwidth is the weak link); route each CPU expert
  job to the node holding a local copy.
- Expected +8–15% (the read test on this machine reaches 84–88 GB/s against the expert pass's 70).

### D2. Draft layer without CPU experts (lossless for output; 1 day)
- The draft layer still runs four CPU experts per proposal (1.4 s of a 17 s trial). Draft with GPU-resident
  experts only (route the draft to its best resident experts, or keep a small fixed VRAM set for the MTP layer);
  acceptance may fall slightly; outputs are always verified.
- Expected +4–7%.

### D3. Window expert sharing (lossy; 3–4 days)
- In a verify window (4 tokens), bias each token's routing toward experts already chosen by the other tokens of
  the window, within a small logit margin, so the window reads fewer distinct experts. Unlike affinity toward
  the GPU tier, the shared experts are ones the neighbouring tokens actually need.
- Implement in the batched router; measure distinct experts per window, GB/token, corpus gate and HumanEval.
- Expected 10–20% fewer bytes per token at a KL cost to be measured. Budget: the gate has +0.016 KL and about
  1 pp top-1 left; drop Q4 dense (about +0.012 KL of its cost) if needed to make room.

### D4. Recover HumanEval margin (2–3 days)
- HumanEval is exactly at the floor, which blocks every further lossy step. Requantize the 224 kept experts with
  their own calibration (the Q23 pack was calibrated with all 288 present), and compare the 224 plan with a
  plan that weights saliency by the coding part of the calibration corpus. Any HumanEval margin gained can then
  pay for D3.

### D5. Smaller items (1 day)
- Free VRAM for the tier: KV/state, verify history, MTP dense weights (+0.3–0.5 GB, about +2%).
- Prompt-conditioned tier: after prefill, seed the tier from the prompt's own routing counts.

### Expected result
D1 + D2 (lossless) take decode from 29.0 / 31.6 to about 33–37 / 36–40 tok/s; with D3 inside the gate,
35+ at 2K becomes likely. D1's gain is the main uncertainty, and the first measurement (AnonHugePages and the
expert pass's GB/s with a hugetlb arena) decides how much of the plan rests on D3.

## 4. Order and checkpoints

1. P0 tooling and the D1 measurements (hugepage state, NUMA traffic): 1–2 days. They decide the order below.
2. D2 (fast, lossless) and D1, then decode re-measure. **Checkpoint:** if 2K decode is still below about 33,
   start D3 and D4.
3. P1 expert int8 kernel, then P3 attention, then P2 FP8 projections (behind the prefill-path check). Re-measure
   8K and 16K after each.
4. Full gate (corpus, prefill-path check, HumanEval 164) on the final profile; CUDA and HIP builds; HIP keeps the
   existing kernels where the new ones are CUDA-only (sm_120 MMA).

Total effort: about 3–4 weeks of kernel and pipeline work. Root is needed for hugetlbfs (D1 option A).
