# Experimental GLM Vulkan runner

`STRATA_ENABLE_VULKAN=ON` builds `strata-glm-vulkan` without CUDA, HIP or ROCm.
The first runner targets the GLM-5.3f NoPE, four-stream REAP50 GGUF model on one
Vulkan GPU plus system RAM. Validation hardware is an RX 9070 XT, Ryzen 3950X
in ECO mode, X570S AERO G and 64 GiB DDR4-2666. An RX 570 with 16 GiB VRAM
also passed the short correctness check below. Windows execution has not been tested.
Vulkan avoids ROCm's architecture selection;
it still requires a working Vulkan driver and the compute/memory features
checked at startup.

The GPU runs dense and per-head projections, mHC mixing, KDA recurrence and
sparse MLA attention. KDA state and MLA caches remain resident between tokens.
The existing CPU pool runs routed experts. Normalization, convolution,
SwiGLU, index pooling and selection currently run on the CPU. Inputs and
outputs cross a host boundary between projection and mixer calls.

This is a standalone experimental runner. Setup and the API server do not yet
select it. Sequential prefill remains the default. Opt-in paths provide batched
prefill, a resident GPU expert tier and verified speculative decoding as
described below. CUDA graphs and multi-GPU execution are unsupported.

## Build

On Linux, install a C++20 compiler, CMake, Ninja, Vulkan development headers,
`glslc` and SPIR-V headers. Mesa RADV is the tested Vulkan driver; ROCm is not
needed by this executable. Use a separate build directory:

```sh
cmake -S . -B build-vulkan -G Ninja \
  -DSTRATA_ENABLE_VULKAN=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF
cmake --build build-vulkan --target strata-glm-vulkan glm_vulkan_parity -j 8
```

The build uses the same pinned GGML checkout as the CPU expert path. An existing
checkout can be supplied with `-DSTRATA_GGML_DIR=<llama.cpp checkout>`. For a
Vulkan SDK outside the system prefix, supply `CMAKE_PREFIX_PATH` and
`Vulkan_GLSLC_EXECUTABLE`. The custom SPIR-V shaders are embedded in the binary;
no shader files or ROCm libraries are needed at runtime.

The tested workspace build used glslc 2023.8 and SPIR-V headers 1.4.309.
That compiler did not enable GGML's integer-dot-product shader feature. This
limits the measured implementation; the results do not establish Vulkan's
maximum possible speed on RDNA4.

## Validate and run

Run `build-vulkan/vulkan/glm_vulkan_parity` first. It compares FP32 and quantized
projections with CPU oracles, including Q4_0/Q5_0's original-input-sum correction,
checks repeated cached projections, five KDA steps and retained state, mHC,
sparse MLA and rejection of invalid selected rows. Projection and custom-kernel
checks use a relative L2 limit of `2e-5`; they do not prove full-model quality.

```sh
build-vulkan/vulkan/strata-glm-vulkan \
  --model /path/to/GLM-5.3-Flash-REAP50-Q23-assembled.gguf \
  --prompt-file /path/to/prompt.ids --output /tmp/vulkan-result.json \
  --context 4096 --generate 64 --trials 3 --threads 12 --gpu-budget-mib 15360
```

Prompts contain comma-separated integer token IDs. Generated IDs go to stdout;
measurements go to stderr and the requested JSON file. `--stop-ids` accepts a
comma-separated list, including the model's end-of-turn tokens. Existing output
files are rejected. Three trials include an untimed prefill warmup; cold expert
page misses can still affect the first generated response, so inspect all trials
and I/O evidence.

Quantized two-dimensional projections reproduce Q8_1 activation scale/rounding.
Q4_0/Q5_0 also retain the original-input-sum term used by the HIP/CUDA dot
product. Per-head MLA products keep FP32 activations. FP16 operand conversion
and cooperative-matrix paths are disabled in the initial strict projection
profile, with FP32 accumulation requested explicitly.

Custom working buffers prefer host-cached coherent memory so the CPU does not
read result vectors through uncached mapped VRAM. Persistent state prefers
host-visible device-local memory and can use coherent system memory when a
visible-VRAM heap cannot satisfy an allocation. That fallback was not exercised
on the validation card. Allocations, tensor geometry, selected rows
and append order are checked. Each custom dispatch waits for its own fence,
with a 60-second bound; a failed fence exits without freeing in-flight buffers.

## Fault evidence and comparisons

### RX 570 16 GiB correctness check, October 14

The [RX 570 config](../configs/experimental/glm53f-reap50-vulkan-rx570-64g-16g.json)
uses the strict Vulkan path, 12 CPU workers, a 4096-token context and a
15360 MiB GPU allocation budget. It points to the REAP50 Q23 assembled model
under `models/glm53f-reap/`. It enables no performance options.

On a Ryzen 3950X with 64 GB installed RAM (60 GiB reported by Linux), an
RX 570 reporting 16 GiB VRAM and Mesa RADV 26.0.8, primitive and model-group
parity passed the existing gates. Two guarded runs of a 23-token arithmetic
prompt produced identical 58-token outputs, answered `4` and stopped at the
end-of-turn token. Peak service memory was 40.3 GiB with zero swap; sampled
global VRAM peaked at 5297 MiB. No new GPU fault or PCI identity change was
recorded. RAM speed was not measured.

Run with a Python environment containing Strata's tokenizer dependencies:

```sh
.venv-glm/bin/python tools/glm_guarded_run.py \
  configs/experimental/glm53f-reap50-vulkan-rx570-64g-16g.json \
  bench/fixtures/glm53_vulkan_prefix32.ids \
  --output /tmp/rx570-reap50 --tokens 64 --trials 2 --timeout 1200
```

The guard preserves that Python interpreter in its service. The
[validation record](benchmarks/glm_vulkan_rx570_20261014.json) describes the
arithmetic prompt and evidence. This is a short correctness smoke test;
long-context, coding and perplexity qualification remain open. The 15 GiB
budget permits allocations but does not force them: this run used about
5.2 GiB VRAM, so allocation near the full 16 GiB was not tested.

The same day's untuned performance baseline used the frozen 32-token prompt
above, 64 generated tokens per trial, five trials and one untimed prefill
warmup. Median sequential prefill was 3.679 tok/s; median decode was
3.601 tok/s. Decode trials were 3.155, 3.601, 3.617, 3.617 and 3.596 tok/s,
with the first decode retained in the median. Each decode timing covers
63 model steps; the first generated token comes from prefill logits.
All five 64-token outputs matched. Peak RAM was 41.22 GiB, sampled global
VRAM was 5298 MiB and swap remained zero. No new GPU fault was recorded.
The [baseline record](benchmarks/glm_vulkan_rx570_baseline_20261014.json)
includes individual timings and source/build hashes. This is a short-prompt
baseline, with no same-day comparison against another backend.

The [guarded diagnostic workflow](GLM_GPU_DIAGNOSTICS.md) also accepts benchmark
configs with `backend: "vulkan"`. It retains the 60 GiB RAM/zero-swap limit,
physical GPU headroom, owned-service cleanup, PCI identity checks and timestamped
telemetry. Vulkan runs additionally hash their backend and shader sources.
Engine phase tracing is optional and must be disabled for performance timings.

`STRATA_GLM_STATE_TRACE_DIR` saves the first token's residual after every layer.
It refuses to overwrite files. The HIP reference rejects this diagnostic in
pipelined execution, where synchronization can wait for host work not yet
started. Layer dumps are diagnostic artifacts, not inputs for simulator fitting.

The October 12 comparison uses the same Q23 model and token IDs, 12 CPU workers,
no expert tier, sequential prefill, three trials and the same resource limits.
The initial Vulkan implementation matched 192 generated tokens but was slower
than that HIP control. Earlier smoke tests also found a full-distribution
mismatch; both failures and subsequent corrections are retained. See the
[measurement record](benchmarks/glm_vulkan_9070xt_20261012.json) for final numbers,
quality checks and the remaining limitations. Full held-out coding and
perplexity qualification are separate from these short comparisons.

The corrected warm results were:

| Backend | Sequential prefill tok/s | Decode tok/s | Sampled peak global VRAM MiB |
| --- | ---: | ---: | ---: |
| HIP / ROCm 10.1, no expert tier | 8.641 | 8.534 | 6,723 |
| Vulkan, corrected staging and affine dots | 3.784 | 3.739 | 5,487 |

Vulkan reached 43.8% of this matched HIP control's decode rate, saved 1,236 MiB
of sampled global VRAM, and improved 2.45 times over its first warm baseline
(1.527 tok/s). All 192 generated IDs matched across three trials. This is a
comparison against the sequential, no-tier control, not the optimized HIP tier.
The frozen prompt IDs are in `bench/fixtures/glm53_vulkan_prefix32.ids`.

On a separate 53-token C++ prime prompt, both backends passed 400,538 compiled
oracle cases, including unsigned overflow/boundary inputs. HIP generated 129
tokens and Vulkan 138. These traced cold runs are correctness checks rather
than performance measurements. They do not replace the full held-out suite.

The corrected 32-token prefill still had KL 0.0624 from HIP to Vulkan, above the
0.01 numerical gate. The remaining discrepancy is unresolved. Keep the runner
experimental; these measurements do not qualify it as the default coding path.
No GPU/PCI hardware fault was recorded during this campaign. At that time, older AMD cards,
batched prefill, long-context quality and speculative decoding remain untested.

Vulkan engine traces also bracket each named projection and custom shader with
its context position and layer. A failed GPU fence exits without freeing
in-flight buffers. Read the last persisted operation after a reboot using the
diagnostic status tool; an interrupted run is never a passing measurement.


## Opt-in decode profile measured on October 12

The later five-trial comparison on the same 3950X in 65 W ECO mode,
64 GiB DDR4-2666 and RX 9070 XT reached the 80% decode target:

| Backend | Workers | Sequential prefill tok/s | Decode tok/s | Sampled peak global VRAM MiB |
| --- | ---: | ---: | ---: | ---: |
| ROCm 10.1 control | 12 | 8.643 | 8.530 | 6,723 |
| Vulkan fast profile | 12 | 7.000 | 6.972 | 5,517 |

The Vulkan median is 81.7% of the matched ROCm median, and 1.86 times the
previous 3.739 tok/s Vulkan result. All four warm Vulkan decode trials were
above 80% of the control; the cold first decode was much slower. Both runs used
the frozen 32-token prompt, 64 generated tokens per trial, five trials, one
untimed prefill warmup, no expert tier and the same RAM/VRAM limits. Reported
medians include the cold first decode. All 320 generated IDs matched. These
numbers describe the sequential, CPU-expert control, not a tuned ROCm GPU tier.
The [optimization record](benchmarks/glm_vulkan_optimization_9070xt_20261012.json)
contains individual timings, source/build hashes, quality results and failures.

Use the separate
[fast config](../configs/experimental/glm53f-reap50-vulkan-fast-64g-16g.json).
The original Vulkan config and CUDA/HIP defaults keep their existing settings.
The profile enables these options explicitly:

- `STRATA_VULKAN_ASYNC_STAGING=1`: reusable pinned upload/download storage,
  followed by one synchronization after queued transfers and graph execution.
  Pending transfers are synchronized before storage is freed on an exception.
- `STRATA_VULKAN_JOINED_PROJECTIONS=1`: persistent graphs for independent
  KDA/MLA projections and FFN gate/up pairs, including distinct low-rank inputs.
  Shared inputs are uploaded once. Original-sum affine formats use individual
  products so their correction remains in place.
- `STRATA_VULKAN_OVERLAP_EXPERTS=1` and
  `STRATA_VULKAN_CPU_PLACEMENT=split`: CPU routed experts overlap the shared
  expert on Vulkan. A persistent coordinator owns the reserved host core,
  and GPU submission uses an unused physical core. This requires a spare core.
  GPU work stays on the submitting thread; captured CPU storage is joined on
  success and exception paths.
- `STRATA_VULKAN_CPU_MHC=1`: mHC coefficients/read/write use the existing CPU
  reference instead of transferring residual vectors for a small dispatch.
- `STRATA_VULKAN_FUSED_KDA=1`: one shader performs convolution/history updates,
  decay/beta gates, the recurrence, output RMS normalization and the output gate.
  Per-head normalization and decay calculations are shared across columns.
- `STRATA_VULKAN_AVX2_ACTIVATIONS=1`: vectorized Q8_1 activation reconstruction,
  preserving scalar rounding, scale storage, clipping and negative zero.
  This requires a native-host build on an AVX2 CPU.
- `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`: GGML's existing queue option also keeps
  transfers on the compute queue on this RADV card, avoiding cross-queue waits.
  `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` remains set in the measured profile.

The tested fast executable was built with:

```sh
cmake -S . -B build-vulkan-native \
  -DSTRATA_ENABLE_VULKAN=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_VULKAN_NATIVE_HOST=ON \
  -DSTRATA_VULKAN_NATIVE_DOTS=ON \
  -DSTRATA_GGML_DIR=/path/to/pinned/llama.cpp \
  -DVulkan_GLSLC_EXECUTABLE=/path/to/glslc \
  -DCMAKE_PREFIX_PATH=/path/to/vulkan/development/prefix
cmake --build build-vulkan-native --target strata-glm-vulkan \
  glm_vulkan_parity glm_vulkan_executor_check -j 8
build-vulkan-native/vulkan/glm_vulkan_executor_check
```

`STRATA_VULKAN_NATIVE_HOST` uses native CPU instructions with floating-point
contraction disabled and finite checks retained; it does not create a portable
CPU binary. The tested compiler was
[shaderc v2026.3](https://github.com/google/shaderc/tree/v2026.3), commit
`2c8cae778eec0283b44acbe7ed1a386865d78799`, built with its pinned dependencies.
Both advertised upstream Linux binary artifacts returned HTTP 404, so the
compiler was built in workspace scratch without installing system packages.

`STRATA_VULKAN_NATIVE_DOTS` patches a build-local copy of the pinned GGML vector
shaders; it preserves the supplied checkout. It removes rounded half sums from
Q4_K/Q5_K minima and leaves Q4_0/Q5_0 original-input-sum correction to the host.
The unpatched integer path failed the unchanged primitive accuracy gate; the
patch passed. Integer dots are nevertheless disabled in the measured fast
profile. This option is independent of native host compilation.

For GPU parity, set `STRATA_VULKAN_ASYNC_STAGING=1` and
`STRATA_VULKAN_AVX2_ACTIVATIONS=1`, then run
`glm_vulkan_parity /path/to/model.gguf` under the guarded GPU precheck. The model
argument additionally checks grouped projections with shared/distinct inputs.
Fused KDA checks cover five recurrent steps, convolution lengths one and four,
state/output/history and reset. Existing `2e-5` limits remain unchanged.
The CPU-only coordinator check covers exception propagation and joining queued
work before captured storage is released.

The final prefill KL is still 0.058, above the unchanged 0.01 gate. Matching
greedy IDs does not establish distribution equivalence, and the short coding
oracle does not replace held-out coding or perplexity tests. Keep the profile
experimental. No GPU/PCI hardware fault was recorded during the optimization
campaign. At that time, older AMD GPUs, CUDA execution, SYCL builds, batched prefill and
speculative decoding were not tested.

## RX570 optimization candidates

The RX570 campaign uses the modified 16 GiB card, a Ryzen 3950X and 64 GiB
system RAM. Defaults remain sequential prefill, CPU routed experts and no
speculation. These new paths are experimental:

- `--prefill-batch N` (2–1024) batches independent projections and expert inputs
  while retaining ordered KDA recurrence and causal MLA selection.
- `--prefill-experts gpu` streams canonical Q2_K/Q3_K experts into device memory
  during batched prefill. IQ experts retain their CPU path.
- `--decode-cache-mib N` reserves up to 8192 MiB for canonical experts ranked by
  prompt route counts. Actual Vulkan allocation sizes determine admission.
  Missing experts run on the CPU; reduction retains the original order.
- `--speculative lookup|mtp --draft-depth N` enables draft proposals (depth 1–3).
  Target argmax verifies every proposal. `--verify-batch` uses a GPU checkpoint
  and restores/replays the accepted prefix after a rejection. The MTP path
  requires the model's single next-token MLA block.
- `STRATA_VULKAN_DEVICE_STATE=1` places persistent custom state and caches in
  unmapped device memory. Transfers and zeroing use fenced Vulkan commands.
- `STRATA_VULKAN_RELEASE_DENSE_HOST=1` discards original mapped dense pages after
  their first upload. `STRATA_VULKAN_GPU_INDEXER=1` batches index scoring on the
  GPU; causal pool counts and stable host selection retain their ordering.
  Canonical expert dots and GPU index scoring require `shaderFloat64`; the
  RX570 reports this feature. The strict path does not require it.
- `STRATA_VULKAN_BATCH_GEMV=1` is a diagnostic fallback that uses individual
  projection products within a prefill chunk.
- `STRATA_VULKAN_BATCH_MMV=1` limits dense products to eight-column vector
  tiles and retains individual per-head products. Real Q4_K, Q6_K and FP32
  2/3/8-column primitive results matched individual products bit for bit on
  RX570. General GEMM batching changed the parser training output; this path
  avoids that reduction change while retaining weight reuse within each tile.

The expanded model primitive check covers all 21 distinct two-dimensional
block tensor geometries, including Q5_0's 128-input KDA products and small router
and mHC outputs. All 147 comparisons (2–8 columns per geometry) were
bit-identical to individual products in portable and native-host RX570 builds.

The primitive suite additionally checks canonical GPU experts with 1, 2 and 8
inputs, causal batched MLA, index scores and checkpoint restoration. Short
32-input/64-output checks matched the strict RX570 greedy IDs for CPU batched
prefill, streamed GPU experts and serial MTP. These checks do not qualify
8K/16K prefill or coding quality. No 10–15 decode or 200 prefill target has been
qualified. Final timings are deferred at the user's request while other builds
are active.

Separate [decode](../configs/experimental/glm53f-reap50-vulkan-rx570-decode-candidate.json)
and [prefill](../configs/experimental/glm53f-reap50-vulkan-rx570-prefill-candidate.json)
candidate configs expose these switches without changing the strict preset.
They are starting points for qualification, not selected performance presets.
The combined fast path failed the complete `parse_u64` training output gate:
both runs stopped, but the strict control generated 503 tokens and the candidate
330, with the first difference at token 6. Short-prefix matches do not supersede
that failure. The subsequent multi-vector candidate passed the coding holdouts
described below; its first long-context comparison failed as described below.
Removing fused KDA and CPU mHC while retaining general GEMM still changed the
parser output (first difference at token 9). The eight-column vector candidate
matched all three complete strict training outputs (503, 931 and 462 tokens),
including 33 rejected speculative windows and replay in the parser case.
The strict parser response itself violated the requested
single-code-block format; matching it establishes no acceleration regression,
not a passing coding oracle.
The slice and routed-reduction training responses passed their compiled C++
oracles as well as the strict token comparison.

An 8192 MiB expert budget admitted 942 experts (8183.62 MiB of actual expert
allocations). Two 32-input/64-output trials matched strict IDs. Sampled total
device VRAM use reached 13721 MiB and the service's RAM peak was 38.24 GiB,
with zero swap and no guard fault rejection. This tests the modified card beyond
8 GiB; it is an allocation/correctness result, not a throughput qualification.
Both portable and native-host primitive suites passed, including device state,
all 2–8 column vector tiles and actual 32-head/128-dimension index scores.
The latest corrected build matched two default 64-token trials and produced byte-identical
32-token prefill logits against the frozen original Vulkan executable.
All 30 pinned HumanEval holdouts matched complete strict outputs and passed
their independent Python tests. Their longest prompt plus response was 783
tokens, so these cases did not exercise sparse pool pruning beyond 2048 tokens.
The first 8K comparison failed at token 15:
the strict control generated 138 tokens and the candidate 157, both
stop-terminated and within the resource guards. Both generated C++ responses
passed the independent prime oracle (400538 checks each). This is an output
mismatch without an observed functional failure on that case; the strict
token gate remains failed. The remaining three long
holdouts were not run. Short coding success does not qualify long-context
execution or throughput.
Oversized MLA/index dispatches use a two-axis grid: 1024 tokens times the
model's 64 attention heads exceed Polaris's 65535-group X limit. Boundary tests
passed on RX570 and software Vulkan. The dispatch extension retained byte-identical
one-axis shader binaries; the later index-score rounding fix changes that
shader separately. The held-out campaign uses its frozen executable; this boundary
extension was checked separately, with the owned reference process briefly
paused and resumed. Its timings remain unqualified.
The near-limit working-buffer test also covers a 32K cache capacity, a
1024-token/64-head/512-latent batch and a subsequent 20 MiB expert upload.
It exposed simultaneous charging of old and replacement scratch allocations.
Releasing fenced scratch before growing, and discarding large attention
temporaries after readback, passed at 1013.06 MiB on RX570 within a 1024 MiB
working budget. Native-host software Vulkan passed at 1015.06 MiB with device
state enabled.

Long runs can print milliseconds in scientific notation. The harness now parses
that format. `glm_vulkan_quality.py recheck-timing` can separately revalidate a
saved correctness run rejected solely by that completion-parser error. It
checks the original resource/fault evidence, input/executable/output hashes,
token counts and stop termination; it preserves raw failed records and never
qualifies timings. The first 8K control used this recovery.

A stricter primitive check subsequently reproduced index-score rounding
differences at the actual 32-head/128-dimension geometry. Passing CPU-rounded
normalization roots and using FP64 division made 257-pool, three-query scores
bit-identical on RX570 and native-host software Vulkan. Larger projection
outputs (up to 8192 rows), joined projections, and batched attention at 64 heads,
512 latent dimensions and 2049–2051 selected rows also matched their individual
RX570 products bit for bit. This fixes a demonstrated numerical difference; it
does not establish the cause of the failed 8K comparison. A separate 3072-token
parser training case passed all 402 strict output IDs and its compiled C++
oracle. The corrected 8K regression then matched all 138 strict IDs, stopped
normally and passed the same 400538-check C++ oracle. Peak service memory was
53.14 GiB with zero swap and no reported GPU fault. The original failed evidence
remains unchanged; this rerun is a known-failure regression, not an unseen
holdout. The other 8K case and two 16K cases remain untested.
The final streamed-expert candidate also matched two strict 32-input/64-output
trials with a 24576-token cache capacity and clean resource guards. This is a
short execution check, not long-prompt or throughput qualification for that
prefill configuration.

`tools/glm_vulkan_qualify.py --correctness-only` records output/resource checks
without qualifying timings. `tools/glm_vulkan_quality.py` freezes and resumes
three training cases, 30 pinned HumanEval holdouts and four 8K/16K cases; it
requires complete stop-terminated output matching the strict Vulkan control.
Keep failed evidence and use training cases to diagnose changes before running
holdouts. `tools/sim/vulkan.py` uses separate RX570 phase measurements; its cache
and speculative estimates are hypotheses, and long-context estimates remain
unqualified. The 224 GB/s specification is a ceiling, not a fitted bandwidth.
