# GLM-5.3 Flash on two Tesla V100 32GB cards, 2026-10-14

With the qualified 4K recipe and opt-in device greedy selection, Q2 measured 29.01–29.25 decode tokens/s and full Q23 measured 23.36–23.39. Order-swapped off/on pairs gained 2.18–2.42% and 1.28–1.83%, respectively, with identical token sequences. The feature defaults off.

Current Q2 f16 prefill was about 14% slower than the preserved reference. MMQ recovered prefill throughput but lost two held-out coding tasks under the same token cap, so f16-batched remains the qualified recipe.

The current source was synced to an isolated remote checkout and built for `sm_70`. The existing remote checkout and its binaries were preserved. Source baseline: `7a33c9c796337b9871f0b3639ece7d8281a5f391`. This report covers the full UD-Q2_K_XL and assembled Q23 models, not a pruned expert pack.

The remote candidate binary is retained at `tmp/v100-20261014-7a33c9c7/build-candidate/strata-glm-decode`, relative to the remote Strata directory. Raw records remain at `slurm-gpu/tmp/runs/20261014-v100`. GPU allocations were released after the work completed.

## Hardware and timing

Two Tesla V100-PCIE-32GB cards, 32,768 MiB each, 250 W limits; two Xeon Gold 6240 CPUs, 36 physical cores, 72 logical CPUs, four NUMA domains. Runs used 35 workers, automatic CPU affinity and interleaved memory. OS MemTotal was 195,834,796 KiB; the Slurm memory limit was 160 GiB. RAM speed was unavailable. Red Hat Enterprise Linux 8.10, kernel 4.18.0-553.155.1.el8_10.x86_64, CUDA 12.9.86, GCC 12.2.1 and driver 580.178.04 were used.

Each qualified 4K process loaded the model once, excluded one three-token warmup request, then ran three 128-token requests and three 512-token requests. Context allocation was 131,072 tokens; prompt length was 4,096. GPU budgets were 32,768 MiB each. Prefill used `f16-batched`, batch 8,192; decode used the `tools/glm_v100.env` recipe with extend cache, 256-token window, adaptive cache and graphs. Each resident `GEN` resets decoder request state and freshly processes the full prompt; these are not conversation-prefix reuse timings. The model stays resident and uses the stated adaptive expert-cache policy. Timings exclude model load and warmup; prefill includes reset and CPU/cache preparation. Decode timing covers generated tokens minus one because prefill supplies the first token. Model mappings were byte-identical private copies locked in RAM to prevent shared-filesystem invalidations.

Whole-process 1-second samples (including startup) observed Q2 baseline peak client RSS of 99.65 GiB and GPU peaks of 31,716/31,874 MiB. Q23 mapped baseline A observed 110.96 GiB RSS and 31,694/31,892 MiB GPU peaks. These are sampled process observations, not minimum RAM requirements. System swap-in and swap-out deltas were zero for those runs. Sampled SM clocks ranged from 1,230 to 1,380 MHz; this range includes startup, not just active decode.

## Current source versus the preserved reference

The reference is a preserved historical binary, identified by SHA-256 in the evidence, rather than a fully reproducible clean-release source build. The order was reference/current/current/reference on the same allocation.

| Q2 run | Prefill tokens/s | 128-token decode tokens/s | 512-token decode tokens/s |
|---|---:|---:|---:|
| Reference A | 499.66 | 27.58 | 28.57 |
| Current A | 430.22 | 27.65 | 28.55 |
| Current B | 430.21 | 27.33 | 28.26 |
| Reference B | 500.04 | 27.48 | 28.44 |

All reference/current 128-token and 512-token sequences matched and the strict parser oracle passed 400,532 cases. Current Q2 prefill was approximately 14% slower. Decode was essentially unchanged. The cause of the prefill difference was not established. The historical binary rejects Q23's Q3_K prefill tensors, so there is no historical Q23 timing comparison. Current Q23 controls measured 465.52/465.18 prefill tokens/s and 23.08/22.98 tokens/s for 512-token decode. Their repeated token sequences matched, and the same strict parser oracle passed.

## Opt-in device greedy selection

`STRATA_GLM_DEVICE_GREEDY=1` selects the next token on the GPU after the vocabulary head. It reads back an 8-byte token/non-finite result instead of 619,520 bytes of FP32 logits for this model's 154,880-token vocabulary. Scratch is 456 bytes and is allocated once. Integer ordering preserves FP32 subnormals, signed-zero ties and the first-index tie rule. Any NaN or infinity still rejects the row.

The switch defaults off and uses exact-one truthiness. It applies to ordinary single-row GGUF CLI and resident-server greedy decoding. Prefill, logits dumps, evaluation, diagnostics, speculative decoding, parallel requests and EXL3 retain their existing paths.

Order-swapped pairs used the same candidate binary with the switch off/on and on/off. Each cell below is the median of three 512-token requests.

| Model | Pair | Off tokens/s | On tokens/s | Change |
|---|---|---:|---:|---:|
| UD-Q2_K_XL | A | 28.6218 | 29.2456 | +2.18% |
| UD-Q2_K_XL | B | 28.3242 | 29.0099 | +2.42% |
| Full Q23 | A | 22.9658 | 23.3867 | +1.83% |
| Full Q23 | B | 23.0660 | 23.3609 | +1.28% |

All 128-token and 512-token sequences matched within each model across these runs, including all repetitions. Q2 short-request gains were 1.80% and 2.60%; Q23 gains were 1.70% and 1.58%. Prefill was effectively unchanged within each pair. Models produce different continuations, so their throughput is not a controlled quantization comparison.

For this assembled Q23 pack, dual-GPU direct upload also requires `STRATA_GLM_Q2_NUMA_WEIGHTS=0` and `STRATA_GLM_PACK_FUSE_QUANT=0`.

## Longer prompt pairs

Each length uses one pair with three 512-token requests per arm after warmup, still allocating context 131,072. The 8K order was off/on; 16K reversed it to on/off. These are different prompts, not two order-swapped pairs at the same length. The full original parser task was preserved after unrelated C++ source and padding; exact prompt text and token/hash manifests are included.

| Model | Actual prompt tokens | Decode off tokens/s | Decode on tokens/s | Change | Strict parser oracle |
|---|---:|---:|---:|---:|---|
| Q2 | 8,192 | 28.1055 | 28.7256 | +2.21% | Both pass |
| Q2 | 16,384 | 28.2877 | 28.4439 | +0.55% | Both fail |
| Q23 | 8,192 | 23.2282 | 23.5887 | +1.55% | Both pass |
| Q23 | 16,384 | 23.0553 | 23.3962 | +1.48% | Both pass |

All three token sequences matched off/on at both lengths for each model. The 16K difference is small and a single pair does not establish a gain beyond run variation. Both 16K responses put `parse_u64` in `strata::telemetry`, while the strict fixture calls a global function; compilation therefore fails in both modes. Keep these as timing observations with failed coding qualification, not successful coding throughput. The distinct long-prefix FNV task and 15,991-token recall check passed in both modes as described below.

## Validation and limits

The GPU component test passed 156 cases and 39 CUDA graph replays, covering ties, signed zero, subnormals, finite bit patterns, NaN/infinity, boundaries and scratch reuse. Five selected CUDA tests passed. Component and full-model Compute Sanitizer runs reported zero errors. The full Q2 model run used two GPUs, context 131,072, batch 8,192, 24,576 MiB budgets, one warmup and two 16-token requests; its instrumented timings are not performance results. The baseline build passed ten selected tests. Baseline and candidate each passed 49 Python regression tests.

The decoder, shared kernel and new test compiled for HIP `gfx1201` with AMD clang 24.0.0git. HIP GPU execution was unavailable. SYCL uses a separate engine and was not built. Default-path verification compared baseline, candidate-unset and candidate-zero with all other Strata switches cleared: final logits and token output were byte-identical for a 48-token prompt and eight generated tokens, context 4,096, one GPU. This is a scoped byte comparison, not an assertion about all prompts.

Quality uses 20 held-out C++ tasks with independent compiled oracles, a long-prefix C++ task and two exact recall checks. Coding generation stops naturally or at 1,024 tokens; recall is capped at 512. The long coding prompt was 8,291 tokens; recall prompts were 7,879 and 15,991 tokens. Only the final answer after the reasoning delimiter is graded. Original outputs and raw grades are retained; derived final-answer grades identify their raw source hash. A cap before the final answer counts as a failure.

Q2 off/on produced identical token sequences on all 23 tasks. Both scored 16/20 on the held-out coding tasks, with four caps; the long coding task and both recall checks passed. Q23 off/on also matched on all 23 token sequences. Both scored 13/20, with seven caps; the long coding task and both recall checks passed.

These checks do not measure perplexity, HumanEval or equivalence to BF16. The models were hashed and tensor extents checked; no source-provided checksum was available for comparison. Q23 native expert parity passed at layers 3, 11, 44 and 45.

Nsight Systems 2026.5.1 loaded CUPTI 13.4 and failed with `CUPTI_ERROR_INVALID_DEVICE`; captures contained no usable CUDA trace data. No performance claim uses those traces.

## MMQ prefill comparison

The same candidate binary with device greedy enabled was measured f16/MMQ/MMQ/f16. Only `--prefill-experts` changed. Each process used three 128-token and three 512-token requests after the excluded warmup.

| Run | Prefill tokens/s | 128-token decode tokens/s | 512-token decode tokens/s |
|---|---:|---:|---:|
| f16 A | 430.60 | 28.14 | 29.09 |
| MMQ A | 486.32 | 27.56 | 28.49 |
| MMQ B | 485.36 | 27.74 | 28.62 |
| f16 B | 429.92 | 28.31 | 29.28 |

MMQ prefill improved 12.94%/12.90%, while long decode fell 2.06%/2.23%. Median full request latency fell 7.07%/7.08% for 128 generated tokens and 1.80%/1.79% for 512. MMQ outputs matched each other across repetitions and runs, but differed from f16 starting at zero-based token 17. Both methods passed the strict parser oracle's 400,532 cases. MMQ scored 14/20 on the held-out C++ tasks with six caps, versus f16's 16/20 with four caps. `add_i64` and `merge_overlap` regressed by reaching the cap before a final answer; no previously failing task improved. The long-prefix coding task and both recall checks still passed. Keep f16-batched as the qualified recipe for this workload; the MMQ throughput gain does not justify accepting these observed failures.

## Other screens and unsuccessful attempts

Single timings are screening results only. The existing sm70/new-IQ, staging and admission variants did not establish a decode improvement over the dual-GPU recipe. MTP depth one increased short decode throughput but incurred substantially slower prefill and changed outputs. Ordered KDA with four parts matched the serial variant's output but did not improve overall throughput; eight parts were unsupported.

Initial pipeline-labelled auto-cache screens actually used the fallback path because auto cache enables decode-prefill-cache. Corrected fixed-cache single-GPU screens verified positive layer-flow time. At the 12 GiB explicit decode-cache limit, plain/fused/stream prefill measured 246.75/246.78/197.96 tokens/s and decode 17.36/17.17/17.19 tokens/s. They did not beat the dual-GPU recipe. A 16 GiB serve-cache attempt and a 24 GiB CLI-cache attempt were rejected before inference; both interfaces enforce the 12 GiB limit.

The Q23 canonical/direct-expert dual-GPU screen reached READY but rejected the warmup request: canonical experts require the plain CPU expert path or pipelined tier. It produced no qualified timing. The canonical-plus-Q8 variant was skipped because it shares this unconditional restriction; it was not measured.

A separate Q23 single-GPU CLI pipeline screen at 12 GiB cache completed with positive layer-flow time: prefill 232.83 tokens/s (two recorded prefills), decode 18.11 tokens/s (one warmup plus one timed 128-token replay). The two decode token sequences matched. This is cached-prefix CLI replay, not the resident-server protocol; resource sampling and a 512-token coding oracle were not run for this screen. It does not establish a replacement for the qualified dual-GPU recipe.

An unlocked default diagnostic was interrupted during shared-filesystem rereads and did not complete its comparison; its exit marker alone is not evidence of success. The locked retry completed. The first full-model sanitizer attempt used batch 512 and exhausted prefill scratch before inference; only the completed batch-8,192 retry is counted. A reference timing controller encountered a shell error after completing timing; strict coding checks were recovered separately. Frozen raw records retain these failures.

## Reproduce the resident comparison

Use the source and candidate hashes in `provenance.json`. `candidate.patch` freezes the five-file candidate relative to the recorded base commit. Build with CMake Release, CUDA enabled, tests enabled and `CMAKE_CUDA_ARCHITECTURES=70`. The ggml dependency was pinned to `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Set `MODEL` to the first Q2 shard or the full Q23 file and `DECODER` to the built `strata-glm-decode` executable. Run each arm as a fresh process with a fresh output filename. Clear inherited `STRATA_*` variables before loading the recipe.

```bash
source tools/glm_v100.env
export STRATA_GLM_DEVICE_GREEDY=0  # Repeat with 1; reverse order for pair B.
# Full assembled Q23 additionally needs:
# export STRATA_GLM_Q2_NUMA_WEIGHTS=0 STRATA_GLM_PACK_FUSE_QUANT=0
numactl --interleave=all python3 tools/glm_v100_bench.py \
  "$MODEL" docs/fixtures/glm53_cpp_quality/chat_prompt.txt \
  --decoder "$DECODER" --prompt-tokens 4096 --generated-tokens 128 \
  --repetitions 3 --validation-tokens 512 --validation-repetitions 3 \
  --context 131072 --threads 35 --gpu-devices 0,1 --gpu-budget-mib 32768 \
  --prefill-batch 8192 --prefill-experts f16-batched \
  --prefill-expert-cache-mib auto --decode-prefill-cache \
  --decode-cache-mib extend --decode-graphs --decode-cache-window 256 \
  --decode-cache-adapt --speculative none --draft-depth 1 --output "$OUTPUT"
```

`measurements.json` preserves every measured repetition, settings, effective Strata environment, source-record hashes and canonical token-sequence hashes. Token hashes are SHA-256 of comma-separated decimal token IDs encoded as UTF-8. `quality.json` records each completed task, cap, final-answer grade and off/on token identity. `telemetry.json` contains sampled whole-process resource ranges. `ranges.json` gives min/median/max for each process; `oracles.json` retains strict parser outcomes, and `pipeline-cli-screen.json` preserves the separate CLI screen. The engine diff, new test and report sources pass whitespace checks. The frozen patch has three blank context-marker lines (18, 32, 84) flagged when checked as an added text artifact; its bytes and hash are preserved.

The frozen raw records remain in the isolated remote run directory and local ignored study directory; binaries, model weights, raw logits and failed profiler captures are not included in the compact evidence.

The quality runner uses the same resident protocol and recipe. After setting the recipe and model-specific overrides, run each mode with a new output directory:

```bash
GLM_WORKERS=35 python3 bench/results/2026-10-14-glm-v100/reproduce_quality.py \
  --model "$MODEL" --decoder "$DECODER" --tokens 1024 --output "$QUALITY_OUTPUT"
python3 bench/results/2026-10-14-glm-v100/grade_final_answers.py "$QUALITY_OUTPUT"
```

The second command derives `results.final-answer-v3.json` without changing the raw generations. The portable scripts differ from the measured private runners only in repository/fixture discovery, requiring the model argument, and allowing non-Slurm execution with a 35-worker default. Their measured-run source hashes are recorded in the evidence. For MMQ, additionally set `GLM_QUALITY_PREFILL=mmq`; all other conditions are unchanged.
