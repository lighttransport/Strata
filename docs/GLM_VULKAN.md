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

### Opt-in fast engine: October 14 diagnostic checkpoint

`--engine fast` uses one Vulkan device and recorded token windows. It remains
experimental: the numerical gate against strict has not passed. The guarded
runner accepts `"engine": "fast"`, a resident expert tier up to 12288 MiB,
and the fast routing affinity/min-share options. Strict remains the default.

On RX570 16 GiB and Ryzen 9 3950X, the frozen 32-token prompt followed by
64 generated tokens produced identical fast greedy and MTP1 IDs with a
9216 MiB tier. All 31 draft proposals were accepted on this repetitive prompt.
The same-day strict control matched the saved strict prefill logits byte for
byte and its 32 generated IDs. These are single short correctness runs,
not throughput or quality qualification.

The current fast prefill KL against strict is 0.204947, above the 0.01 limit.
Diagnostic `STRATA_VULKAN_FAST_Q8_EMULATE=1` with CPU-style nearest-even half
scales reduced KL to about 0.11932; it also fails the limit. This option now
rounds every column of a window and honors input strides. Activation rounding
matched the CPU exactly on sampled offset/strided widths 1–4. Resident Q2_K
gate/up and Q3_K down products at those widths had relative L2 below 2.8e-7
against sampled dequantized CPU products; shared hit-union slots and weights
also passed. These primitive checks do not establish end-to-end quality.

`STRATA_VULKAN_FAST_PROFILE=2` with `STRATA_VULKAN_FAST_PROFILE_T=2` profiles
two-token target windows. Reports divide by the tokens actually profiled,
and omit GPU timing when a phase has no matching window. Native and portable
Vulkan targets built; GPU checks used the native build. The 33 relevant Python
tests and three host rule tests passed. Seven CUDA tests could not execute
without a CUDA device; three other GLM tests skipped. Perplexity, training
oracles, HumanEval and long-context gates remain untested. Compact evidence is
in `docs/benchmarks/glm_vulkan_fast_resume_20261014.json`.

### Opt-in fast engine: October 15 progress

On the same RX570 16 GiB and Ryzen 9 3950X with 64 GiB DDR4-2666,
five guarded trials of 32 input tokens and 128 generated tokens measured
15.8224 decode tok/s median. This uses MTP depth 3, a 9216 MiB resident tier,
routing affinity 0.08, the REAP50 coverage prior, and the two options below.
The runner performed one untimed prefill warmup; each trial includes its
first decode work. The resource and GPU guards passed. This is a short,
repetitive prompt measurement; it does not establish coding throughput.

- `STRATA_VULKAN_FAST_EXPERT_Q8=1` rounds resident expert activations to the
  CPU Q8_K grid and uses the CPU SwiGLU approximation. Two sampled experts
  at widths 1 and 4 had whole-expert relative L2 error at most 4.41e-7.
- `STRATA_VULKAN_FAST_CANONICAL_SUM=1` requires that option and combines
  CPU and resident expert rows in router order. Mixed CPU/GPU cancellation
  checks at widths 1, 2 and 4 matched the CPU FMA reduction byte for byte.

MTP depths 2 and 3 now retain per-prefix MLA index-ring snapshots for
rollback. Eighteen position/keep checks passed. On a parser training case,
greedy and MTP3 produced the same 161 token IDs and passed the compiled
oracle. Skipping discarded draft attention/FFN work preserved all three
training outputs byte for byte; all three oracles passed. The frozen
164-case HumanEval run stopped after a later GPU fault, with 111 cases
completed and 100 isolated oracle passes. No full HumanEval score is available.

Large prefill windows up to 2048 and tiled quantized GEMM are experimental.
Early RX570 Q4_K 4096-by-8192 products with 256 columns measured about
1.27 TFLOPS; later vector-decode measurements are below. An initial
32-token prefill window changed logits (KL about 0.199 against the earlier
width-4 run). Prefill has not reached 100 tok/s, and
the strict numerical, perplexity and long-context gates remain unresolved.
Defaults remain unchanged. Compact measurements and frozen executable/source
hashes are in `docs/benchmarks/glm_vulkan_fast_progress_20261015.json`.

`configs/experimental/glm53f-reap50-vulkan-rx570-fast-decode.json` records
the decode candidate options with the native build executable path. Its
HumanEval qualification is pending. Automatic GEMM selection still uses
64- or 128-column tiles; the optional tile override described below also
accepts 256. Software Vulkan checks cover six weight types, one/two heads,
256/768 K dimensions, row/column tails, strides and offsets (216 cases).
Bulk MLA indexing matches serial indexing at 12 window boundaries in
software Vulkan. These software checks do not measure RX570 performance.

On October 16, the original frozen cache-only candidate completed all 164
HumanEval cases with 145 isolated oracle passes; 143 also stop-terminated.
Two valid solutions reached the fixed 2048-token generation cap. The same
executable, configuration, prompts and oracles were retained across both
boots, and all run identities and fixture hashes were checked after completion.
This qualifies that immutable candidate's functional gate; it does not qualify
the later large-prefill binary or its throughput. The aggregate and hashes are
in `docs/benchmarks/glm_vulkan_humaneval_cache_rx570_20261016.json`.

The prefill prototype also accepts `STRATA_VULKAN_FAST_PREFILL_GPU=1`, with
a chunk greater than 4 and both canonical expert options enabled. It groups
frequently routed Q2_K/Q3_K experts into GPU matrix products while CPU workers
process the remaining experts. Uploads and expert products run before the
host-waiting layer tail is submitted. Routing rows are combined in the same
CPU order after both groups finish. Decode and the strict engine do not use
this path.

This is unqualified work toward the 100 tok/s target. The initial FP32 expert
GEMM failed a whole-expert numeric check (relative L2 3.8182e-4). A replacement
keeps integer-valued partial dots and the CPU's eight-lane scale/reduction
order. Six software Vulkan checks of two real experts, at 5, 33 and 129 token
columns, pass the 2e-4 error limit; the largest relative L2 is 1.63085e-4.
They are not byte-identical CPU results. Later native numeric checks are
below; RX570 timings and end-to-end prefill quality remain unqualified.

Within this opt-in experiment, `STRATA_VULKAN_FAST_PREFILL_GPU_MAX_EXPERTS`
accepts 0–144 (initial value 72) and
`STRATA_VULKAN_FAST_PREFILL_GPU_MIN_TOKENS` accepts 1–2048 (initial value 32).
Zero GPU experts provides a submission-only control. These values have not
been qualified for functional quality; later RX570 balance measurements are
recorded below. `PREFILL_EXPERT_SPLIT` reports route counts
and packing, upload and queued-work wall times to diagnose the next bottleneck.

Later RX570 checks on Mesa 26.0.8 passed all 72 sampled matrix-product cases
(maximum relative L2 1.40939e-6). With direct vector unpacking, the Q4_K
4096-by-8192 product at 256 columns measured 6.264 ms / 2.74264 TFLOPS for
64-column tiles; Q6_K measured 6.77664 ms / 2.53516 TFLOPS. Each value is a
five-submission median after one discarded submission. The 128- and
256-column tiles were slower on those two shapes. These are primitive GPU
timings, not inference qualification. `STRATA_VULKAN_FAST_PREFILL_TILE` can
select 64, 128 or 256 explicitly; 0 retains the prior automatic selection.

Streamed experts now accept the CPU's existing Q8_K activation blocks: 4672
bytes per 4096-value token instead of 16384 FP32 bytes. The unpacked input
grid matches the CPU byte for byte on RX570. Six native whole-expert checks
pass the same 2e-4 limit (maximum relative L2 1.63085e-4). They do not match
CPU outputs byte for byte. Tiled weight packing matches the original bytes
in 48 CPU shape/type cases, including row tails and an unaligned source; it
is used only for streamed experts.

The large-window numeric diagnosis found that changing routed reduction
from GPU to CPU changes rounding. A 512-case cancellation probe on this
driver did not match CPU `std::fma`, even with a GLSL `precise` local.
Ordinary GLSL Fma does not guarantee a fused operation; see
[Khronos' FMA extension rationale](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_shader_fma.html).
Large canonical windows now use the same GPU router-order reduction as
small windows. With `STRATA_VULKAN_FAST_PREFILL_MATCH_GEMV=1` retaining the
existing four-token product arithmetic, the 32-token control matches all
45 attention traces, all 45 layer-output traces and final logits byte for
byte. This is a diagnostic control, not a prefill speed qualification.
The tiled-GEMM path still changes arithmetic: its sampled width-32 logit
KL against width 4 was 0.126279 before this reduction fix. Its end-to-end
quality remains unqualified.

The first 2048-token repeated throughput attempt triggered an amdgpu compute
queue timeout and driver reset before completing a trial. The resource/fault
guard stopped the owned test. There is no valid throughput result from that
attempt. GPU execution is stopped for this boot. The HumanEval scheduler was
stopped at a safe boundary with 111 completed cases preserved: 100 isolated
oracle passes, including 99 stop-terminated passes. This is not a full score.

CPU-side follow-up changed large prefill to submit each layer's segments
after the preceding CPU handoff, keeping future host-waiting tails out of
the submitted queue. Storage-buffer bindings now reject out-of-bounds ranges
and ranges above the device limit before submission. Neither change has been
GPU-validated after the fault, and the timeout's root cause is unresolved.
A healthy boot and guarded smaller-window checks are needed before another
2048-token trial. The 100+ tok/s target, strict KL, perplexity and large-window
functional quality gates remain incomplete.

For the next diagnosis, `STRATA_VULKAN_FAST_PROFILE_T` accepts a width up to
the configured prefill chunk. Large-window query layouts reset when scratch
commands are recreated, and timestamp-capacity checks reject oversized
detail traces before submission. `STRATA_VULKAN_FAST_PREFILL_PROGRESS=1`
prints allocation and per-layer submission/expert/completion milestones.
These optional diagnostics build but await GPU validation after a healthy boot.

A CPU-only follow-up removes the temporary joined gate/up tensor from streamed
expert packing. The direct two-source packer matches the concatenated-source
format in 48 shape/type cases, including unequal row counts and unaligned
inputs. For the Q2_K 4096-by-4096 gate/up matrix on the Ryzen 9 3950X, nine
interleaved CPU timing pairs measured 1.653 ms median for joining then packing
and 1.486 ms for direct packing, after one warmup per path. The helper also
uses 5.25 MiB less temporary memory. Six software Vulkan whole-expert checks
retain the previous numeric errors (maximum relative L2 1.63085e-4).
This changes only opt-in streamed prefill and awaits RX570 inference testing;
these packing timings do not establish 100 tok/s prefill.

### October 15 checks after reboot

The fresh boot passed all 33 small Vulkan window checks without a GPU fault.
The committed prefill submission changes then completed guarded 32-, 128-,
256- and 512-token windows. The width-32 arithmetic-matching control retained
the previous prefill logits byte for byte. These runs do not establish the
cause of the earlier 2048-token timeout.

On the same RX570 16 GiB, Mesa 26.0.8, Ryzen 9 3950X and DDR4-2666 system,
three warmed 256-token trials measured 28.1536 prefill tok/s with CPU experts
and 26.437 with the previous streamed CPU/GPU expert path. The prompts repeat
the frozen 32-token prefix; each trial generated only two tokens, without MTP
or a decode expert tier. These measurements do not qualify coding throughput.

`STRATA_VULKAN_FAST_PREFILL_GPU_PACK=1` adds opt-in GPU conversion of native
Q2_K/Q3_K expert rows into the existing streamed layout. It requires
`STRATA_VULKAN_FAST_PREFILL_GPU=1`. The CPU uploads contiguous native gate,
up and down rows; the product command converts their layout before use.
This removes CPU packing and its host temporary, adds about 8.69 MiB of raw
device weight scratch, and leaves the quantization and dot arithmetic alone.
GPU conversion time is included in `queued_work_ms`; `pack_ms` measures
CPU packing only.

The GPU packer matches every byte and destination sentinel in 48 shape/type
cases on both RX570 and software Vulkan. Six native whole-expert checks
retain the prior maximum relative L2 error of 1.63085e-4 against the CPU.
Both the disabled control and enabled 256-token path preserve the prior
streamed prefill logits byte for byte. Three warmed trials with GPU packing
measured 34.1901 prefill tok/s, about 29% above 26.437. The 512-token path
completed cleanly but was slower per token; increasing the window alone
does not reach 100 tok/s. Native and portable Vulkan builds and the 35
Python checks passed. Compact evidence is in
`docs/benchmarks/glm_vulkan_post_reboot_20261015.json`.

`STRATA_VULKAN_FAST_PREFILL_GPU_VECTOR=1` optionally reuses decoded integer
weight metadata in the canonical expert matrix products. It requires the
streamed GPU path. Six whole-expert cases match the original loader byte for
byte on both RX570 and software Vulkan. Three warmed 512-token trials measured
33.9919 tok/s versus about 30 tok/s with GPU packing alone; the final prefill
logits match byte for byte. The 1024-token execution check also completed
without a fault. These results apply to the frozen pre-safety builds recorded
in the compact evidence.

The subsequent 2048-token check triggered another amdgpu compute-queue timeout.
The last progress marker was the first routed-expert handoff (layer 3), and
the guard killed its owned test before any prefill trial completed. The
per-layer submission change did not prevent this failure. Its cause remains
unconfirmed. GPU testing stopped for that boot. The unchanged frozen HumanEval
candidate has 107 isolated oracle passes from 119 completed cases; its
remaining 45 cases and the full 140-pass gate are incomplete.

A CPU-side follow-up guards the Q8 maximum-index lookup for non-finite blocks
and rejects non-finite prefill expert inputs and final logits. Large dense and
shared SwiGLU work uses at most 1024-token slices of the original shader;
smaller widths retain their original dispatch. Eight software Vulkan cases
through 2048 tokens match per-token shader bytes and sentinels. Ten Q8 cases
(four finite, six non-finite) pass with and without diagnostic robust buffers,
and six whole-expert cases retain the earlier errors and loader byte matches.
`STRATA_VULKAN_FAST_ROBUST_BUFFERS=1` enables Vulkan robust buffer access for
diagnosis; it remains off by default. Device logs now include dispatch and
storage-range limits. These follow-ups build in both host configurations but
have not run on RX570 after the reset. They do not establish a timeout fix.

The October 15 postmortem found a two-second scheduler timeout in the installed
`7.0.0-38-generic` amdgpu module. Its module metadata says 2000 ms, its compiled
timeout initializer sets all four queues to 2000 ticks with `CONFIG_HZ=1000`,
and the `lockup_timeout` parameter is empty. The loaded and installed module
source versions match. This agrees with the
[Linux 7.0 timeout initializer](https://github.com/torvalds/linux/blob/v7.0/drivers/gpu/drm/amd/amdgpu/amdgpu_device.c).

The first prefill command still contains all three dense layers and layer-3
attention/router. In the successful 1024-token detailed run it took 1228.47 ms;
the 512-token detailed run took 539.89 ms. The log field `first_segment_ms` is
divided by `profiled_tokens`, so its logged 1024-token value of 1.19968 means
milliseconds per token. Linear doubling would put that command at 2456.94 ms
for 2048 tokens. That is an estimate, not a completed 2048-token measurement.
An oversized initial submission is therefore the leading timeout explanation.
Splitting SwiGLU dispatches inside the same command does not bound the total
submission time. A future diagnostic should split the dense prefix at layer
boundaries with separate submissions and completion checks, then measure each
submission before attempting 2048 tokens again.

No OOM event or PCIe error was recorded around the reset. The driver announced
a coredump, but it was no longer present at inspection. The exact overdue
command and shader remain unknown. The earlier PRT warning reports a change
to fault handling; the
[Polaris memory-controller code](https://github.com/torvalds/linux/blob/v7.0/drivers/gpu/drm/amd/amdgpu/gmc_v8_0.c)
masks some unmapped-access faults when PRT is enabled, so absent VM-fault logs
cannot rule out an access error. This investigation submitted no GPU work and
changed no driver settings. The compact evidence records its limits.

`STRATA_VULKAN_FAST_PREFILL_SPLIT_PREFIX=1` prepares that submission check for
large-window prefill. It records the initial dense attention and FFN work into
separate commands, adds a device memory dependency at each boundary, and waits
for each command's fence before submitting the next. Decode and the disabled
prefill path retain their original command geometry and shaders.

The default completed-command limit is 500 ms. An explicit
`STRATA_VULKAN_FAST_PREFILL_SUBMIT_LIMIT_MS` accepts 1 through 1500 ms and
requires prefix splitting. `PREFILL_SUBMISSION` reports host wall time from
submission through fence completion; it is not a GPU timestamp. A slow
completed prefix fails the run. This cannot preempt a stuck shader, and does
not change the driver watchdog. With profiling and prefill progress enabled,
`PREFILL_SEGMENT` also reports absolute GPU milliseconds for each command,
separately from the existing per-token aggregate. Native and portable Vulkan
compilation passed. Physical controls at 32, 128, 256, 512 and 1024 tokens
completed without faults and matched the appropriate frozen runner's final
vocabulary logits byte for byte. At 1024 tokens the six dense-prefix host
durations were at most 132.643 ms; the largest GPU segment was 458.216 ms.
The old failed 2048-token runner was not retried, and wider submissions remain
untested. This diagnostic does not establish an instruction-level timeout
cause, full functional quality or 100 tok/s. Hashes and measured limits are in
`docs/benchmarks/glm_vulkan_prefill_submission_controls_rx570_20261016.json`.

`STRATA_VULKAN_FAST_PREFILL_GPU_NTILE=16|32|64` selects the column tile for
streamed GPU experts; 64 retains the original kernels. The narrower variants
reduce padding for small expert token groups and retain the same eight dot
lanes, block-scale operations and reduction order. The four default scalar
and vector Q2_K/Q3_K shader binaries remain byte-identical after compilation.
Full-model checks and inference throughput for the new tiles are pending.

On RX570, two experts at 5, 33 and 129 tokens each passed full output-byte
comparisons for both new tile sizes. Paired timings used three warmup pairs
and five measured pairs with alternating order, identical GPU packing and
the same vector loader. At 129 tokens, tile 32 reduced queued expert work by
about 6.8% and total call time by about 5.2%; at 33 tokens it was slightly
slower. Tile 16 was about 14% slower in queued work at both 33 and 129 tokens.
It improved only the five-token case. These whole-expert microbenchmarks do
not establish an end-to-end prefill win; the default remains 64. Counts,
medians and hashes are in
`docs/benchmarks/glm_vulkan_expert_tiles_rx570_20261016.json`.

The checker accepts `--tile-columns` and optional `--boundary-checks`. The
expanded software test exposes a CPU-reference relative L2 error of
2.24746e-4 at 17 columns with the unchanged 64-column kernel. Both narrower
tiles match its output bytes and hit the same existing error limit. That
limit remains unchanged; the extra boundary screen fails and is retained as
an unresolved numeric check. Tile equivalence does not establish strict CPU
parity or complete inference quality.

`STRATA_VULKAN_FAST_PREFILL_REUSE_EXPERT=1` optionally retains one GPU expert
scratch set across routed layers and prefill windows of a prompt. It requires
streamed GPU experts. The set grows to a power-of-two capacity as needed;
the previous set is freed before growth, and all retained scratch is freed
before allocating the decode expert tier. Each call still uploads its own
weights and inputs. Up to 128 commands are retained, keyed by token count,
input encoding and exact SwiGLU clamp bits. Each replay completes on a fence
before those inputs or buffers can be reused.

Layer diagnostics separate scratch setup, input/job preparation, CPU work,
command recording and submit/wait time; counters are per-layer differences
even when scratch is shared. Native and portable compilation passed.
Twelve software Vulkan expert cases and 24 clamp-change comparisons retain
the original output bytes using 256-token scratch for 5, 33 and 129 tokens,
with both CPU and GPU weight packing. Physical final-vocabulary controls at
128 and 256 tokens also match the frozen runner. The native microbenchmark
below shows no clear gain from command replay alone; full functional quality
remains unqualified.

`STRATA_VULKAN_FAST_PREFILL_REUSE_HOST=1` retains the host activation, result,
weight, sum and job buffers across routed layers of a large prefill window.
It requires a chunk greater than four. Result and routing-weight rows are
cleared exactly as before; the arithmetic and job order are unchanged.
Previous CPU jobs and GPU commands complete before buffers can grow, and
the buffers are released with prefill scratch before decode. On the RX570
system above, 512 repetitive input tokens and two output tokens measured
33.969 to 37.0698 prefill tok/s median, using one untimed warmup and three
trials in each guarded process. Both performance guards passed, and the
first measured trial's final vocabulary logits are byte-identical. This 9.1%
gain is a short prompt
measurement; full-model quality and combined decode qualification remain
pending. Evidence is in
`docs/benchmarks/glm_vulkan_prefill_host_reuse_rx570_20261016.json`.

With host reuse, limiting streamed GPU work to the busiest 20 experts per
layer instead of at most 72 measured 37.0698 to 46.4556 prefill tok/s on the
same 512-token prompt and three-trial protocol. CPU work and GPU upload/compute
time became similar. Final logits differ: the top token matches and KL from
the 72-expert result is 0.00247912. This compares two experimental prefill
settings, so it does not establish strict-engine or functional quality.
Defaults remain unchanged. Measurements and limits are in
`docs/benchmarks/glm_vulkan_prefill_balance_rx570_20261016.json`.

The frozen 512-token, 20-expert candidate passed two of three fixed training
oracles. The routed-reduction task repeated reasoning until the unchanged
1536-token cap and failed the complete-code-block check. The original
candidate passed that task in 452 tokens, and the new binary with the original
four-token prefill configuration reproduced all 452 tokens exactly. The new
prefill settings therefore remain unqualified; diagnostic training controls
are isolating expert offload from batched main projections. No new full
HumanEval score is claimed for this candidate.

For that training case, keeping CPU experts while restoring the original
GEMV arithmetic for all main projections reproduced the 452-token baseline
exactly. Batched GEMM with CPU experts still failed its functional oracle.
`STRATA_VULKAN_FAST_PREFILL_MATCH_F32=1` is a narrower diagnostic: it retains
the original four-column GEMV calls for F32 projections, including routing
and mHC, while quantized products stay batched. Native and portable builds
pass. F32-only matching with the 20-expert GPU setting still fails this
training oracle; it passes with CPU experts alone. This remains a diagnostic,
with no qualified optimized-candidate quality or measured speedup. Fixed caps,
outcomes and identities are in
`docs/benchmarks/glm_vulkan_prefill_training_controls_rx570_20261016.json`.

The global zero-affinity variant passed two of the same three training
oracles; its parser response failed the single-code-block requirement.
Restoring main GEMV arithmetic while retaining GPU experts passed only one
of three. These completed failures are preserved with their original caps
and oracles. Neither setting repairs the optimized candidate's quality.

`STRATA_VULKAN_FAST_PREFILL_GPU_MTILE=32|64` selects the expert row tile.
The 32-row variant halves each workgroup's row accumulators and weight
scratch while retaining the dot and scale-operation order. Four software
configurations (scalar 64 columns, and vector 16/32/64 columns) each pass
six whole-expert byte comparisons and the existing CPU-error screen.
All default shader binaries remain byte-identical. The native paired
microbenchmark found
that 32-row tiles take about 18–24% longer in queued work at the tested
5/33/129-token groups. Command replay alone showed no clear timing gain.
The default remains 64 rows; the 32-row variant is not selected for further
full-model tuning. Measurements are in
`docs/benchmarks/glm_vulkan_expert_row_reuse_rx570_20261016.json`.
Combined row-tile and command-replay checks also passed six expert cases and
12 clamp-change byte comparisons in each of the native and portable builds,
using software Vulkan and 256-token scratch. Commands, source hashes and
scope are recorded in
`docs/benchmarks/glm_vulkan_expert_reuse_20261016.json`.

### Standalone hardware diagnostics

`glm_vulkan_hw_diag` checks memory and arithmetic without model weights. Build
with `cmake --build build-vulkan-rx570-native --target glm_vulkan_hw_diag -j8`.
For physical hardware use `tools/glm_vulkan_hw_diag.py`; without `--run` it only
prints a plan and does not open a Vulkan device. Example from the repository
root, after a healthy boot:

```bash
.venv-glm/bin/python tools/glm_vulkan_hw_diag.py --preset smoke --robust --output build-vulkan-rx570-native/hw-results/smoke --run
.venv-glm/bin/python tools/glm_vulkan_hw_diag.py --preset vram-full --output build-vulkan-rx570-native/hw-results/vram-full --run
.venv-glm/bin/python tools/glm_vulkan_hw_diag.py --preset compute --output build-vulkan-rx570-native/hw-results/compute --run
```

Run each stage only after the previous stage passes. Use fresh output
directories. `--pci` defaults to `0000:07:00.0` and `--device` to `RX 570`;
they must identify the same card when testing another machine.

| Preset | Memory payload | Memory patterns | Compute checks |
| --- | --- | --- | --- |
| `smoke` | 64 MiB | 9, one round | 2 seeds, one integer and FP32 result per seed |
| `vram` | Up to 14 GiB | 9, one round; 100 ms hold per pattern | None |
| `vram-full` | Entire detected heap as target, up to 16 GiB; tests all memory it can allocate | 73, one round; 100 ms hold per pattern | None |
| `compute` | Small compute buffers | None | 1000 seeds, 16 integer and FP32 results per seed |
| `extended` | Up to 14 GiB | 73, two rounds; 1 s hold per pattern | 256 seeds, 16 integer and FP32 results per seed |

For the original presets, the launcher reduces the payload to retain a 1 GiB VRAM
reserve plus a 64 MiB allowance for staging and other allocations. An explicit
`--memory-mib` request must fit that limit. It caps system RAM at 2 GiB with
zero swap and uses one UUID-owned service. It rejects an existing kernel
fault, another Strata GPU workload, changed PCI identity or concurrent hardware
diagnostic. It never resets a device or changes driver timeouts.

`vram-full` removes the 14 GiB cap and the 1 GiB reserve. On the 16 GiB RX570
its target is 17179869184 bytes. It rechecks free VRAM before launch, rounds
the free capacity down to MiB, and passes that allocation limit to the executable.
The limit includes the diagnostic's
own device-local staging buffers, guards and padding. Near that limit it uses
smaller blocks down to 4 KiB payloads; allocation OOM reduces the probe size.
Other Vulkan errors fail the test. All successful allocations remain held
while their contents are checked, so the payload is no longer capped at
14 GiB. Each allocation records its logical begin/end and allocation size.
`vram` remains available as the conservative scan.

The full-heap result distinguishes `memory_target_bytes`,
`memory_payload_bytes` and `unavailable_target_bytes`. Their accounting is
validated by the launcher; a target of 16 GiB is not reported as 16 GiB tested
when allocations fall short. Driver-owned memory, auxiliary allocations,
planning granularity and allocation padding are included in that remainder.
`physical_range_complete`
remains false: buffer offsets do not identify physical VRAM cells, and the
driver can migrate allocations. Vulkan's
[heap size](https://docs.vulkan.org/refpages/latest/refpages/source/VkMemoryHeap.html)
and [memory budget](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceMemoryBudgetPropertiesEXT.html)
are distinct; budgets can change with other activity. A successful result
means the allocated payload passed the requested checks, with the untested
remainder explicitly reported.

Memory is held in device-local allocations of at most 64 MiB with 256-byte
sentinels at both ends. All allocations are filled before verification to
exercise aliasing between them. Patterns cover zero, ones, alternating bits,
logical addresses, inverted addresses, seeded random data and rotating bits;
the full sweep adds each walking-one and walking-zero bit at every word.
GPU verification uses permuted access order and in-place inversion. Every
word is also checked by a CPU oracle after a device-to-device copy and staged
readback. A CPU-generated random upload tests the opposite transfer direction.
The reported word offsets are logical; Vulkan does not expose physical VRAM
addresses, and a pass does not establish coverage of every physical cell.

Compute checks compare unsigned 32-bit rotate/multiply/add/XOR results and
128-by-256-by-128 FP32 matrix products against CPU oracles. Matrix inputs are
multiples of one quarter; their products and bounded sums are exactly
representable. Both signs of zero are accepted. Each repetition is read back
and checked before the next one; CPU oracles are reused within a seed to
reduce idle time. The compute and extended presets use 256 ALU iterations
per element. FP16, FP64 and inference kernels are outside this check's scope.

Submissions process at most 1 MiB per memory window by default, or one small
compute kernel. They have separate completion checks and GPU timestamps.
A completed submission above 500 ms fails the protocol. This check happens
after completion; it cannot preempt a stalled shader. The driver watchdog
remains enabled. Logs retain mismatch stage, logical word, expected and actual
bits, XOR, seeds, progress, submission counts and the largest submission time.
The launcher also saves kernel logs and sensor counters. After a new fault it
attempts a two-second, 4 MiB coredump-prefix read, recording unavailable access
without requesting elevated privileges. Raw dumps stay in the result directory
with mode 0600 and should remain outside Git.

Validation used only LVP software Vulkan: native build 24 tests
and portable build 10 checks passed. Cases include all 73 patterns over two
rounds, 65 MiB across two allocations, a non-power-of-two window, robust access,
and deliberate one-bit errors in each of the three checkers. The full-heap
extension also checks allocation OOM recovery with unequal block sizes,
accurate unavailable-byte reporting, the final MiB ending exactly at the
16 GiB logical boundary, and a one-bit error near that boundary without
offset wraparound. The logical-boundary test allocates only 1 MiB in LVP;
it does not claim a 16 GiB physical memory test. These are checker
validation results; RX570 execution awaits a healthy boot. Passing this suite
narrows the failure investigation but does not qualify inference or fix the
large-prefill submission timeout.

### October 16 hardware execution

On the RX570 16 GiB, RADV/Mesa 26.0.8 and kernel `7.0.0-38-generic`, the
prepared smoke, full-heap and compute presets all passed after a new boot.
The full-heap sweep retained 256 allocations and checked 17172393984 payload
bytes with all 73 patterns in 1148.85 seconds. It reports 7475200 target
bytes as unavailable. All 3638335 submissions completed; the largest GPU
timestamp interval was 4.27392 ms. No new guard-matched kernel fault or PCI
identity change was recorded.

Global telemetry sampled a maximum of 17081454592 bytes in VRAM and
167579648 bytes in GTT during that sweep. The highest sampled GPU temperature
was 54 C. These counters do not identify which allocations were resident;
the driver may migrate buffers, so this is not a physical-cell coverage map.
Compute stress checked 1000 seeds, each with 16 individually verified integer
and FP32 results, in 42.12 seconds. Its largest GPU submission was 0.19152 ms.
These passes did not reproduce a memory or arithmetic error in the tested
workloads. The earlier model timeout's cause and inference quality remain
unresolved. Compact counts and hashes are in
`docs/benchmarks/glm_vulkan_hw_diag_rx570_20261016.json`.
