# Experimental GLM Vulkan runner

`STRATA_ENABLE_VULKAN=ON` builds `strata-glm-vulkan` without CUDA, HIP or ROCm.
The first runner targets the GLM-5.3f NoPE, four-stream REAP50 GGUF model on one
Vulkan GPU plus system RAM. Validation hardware is an RX 9070 XT, Ryzen 3950X
in ECO mode, X570S AERO G and 64 GiB DDR4-2666. Other AMD cards and Windows
execution have not been tested. Vulkan avoids ROCm's architecture selection;
it still requires a working Vulkan driver and the compute/memory features
checked at startup.

The GPU runs dense and per-head projections, mHC mixing, KDA recurrence and
sparse MLA attention. KDA state and MLA caches remain resident between tokens.
The existing CPU pool runs routed experts. Normalization, convolution,
SwiGLU, index pooling and selection currently run on the CPU. Inputs and
outputs cross a host boundary between projection and mixer calls.

This is a standalone experimental runner. Setup and the API server do not yet
select it. Prefill is sequential, with no resident GPU expert tier, speculative
decoding, CUDA graphs or multi-GPU execution. The benchmark rejects unsupported
settings. It is not a claim that the optimized HIP runner has been fully ported.

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
No GPU/PCI hardware fault was recorded during this campaign. Older AMD cards,
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
campaign. Older AMD GPUs, CUDA execution, SYCL builds, batched prefill and
speculative decoding were not tested.
