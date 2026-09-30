# GLM-5.3-Flash hardware configurations and throughput estimates

Recorded on 2026-09-30 for the native GGUF backend described in [GLM53_FLASH.md](GLM53_FLASH.md). These are bandwidth calculations and conditional scenarios, not measured Strata throughput benchmarks. GB means decimal gigabytes; cache settings in the decoder use MiB.

## Model and runtime configuration

Artifact: `/path/to/models/glm53f/iq3/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf`, with all four sibling shards. The Q3 label describes a mixed-quantization artifact, not uniformly three-bit tensors.

| Quantity | Inspected value |
| --- | ---: |
| Main layers | 45: 34 KDA, 11 sparse MLA |
| FFNs | 3 dense, 42 MoE |
| Experts per MoE layer / selected per token | 288 / 8 |
| Hidden width / expert intermediate width | 4096 / 2048 |
| Main expert weights | 109.867696128 GB |
| Main fixed weights | 7.690382584 GB |
| Main weights combined | 117.558078712 GB |
| MTP draft block | 2.799972480 GB |
| Total tensor payload | 120.358051192 GB |
| Selected main expert weights per token, S | 3.051880448 GB |

Reproduce the census with `build-glm/strata-model-inspect <any shard>`. The current experimental runtime defaults to six CPU workers plus the host, a 4096 MiB dense cache, prefill batches of eight, and no GPU expert cache. A 512 MiB expert cache was validated separately. MTP execution and conversation-prefix reuse are not implemented. The runnable serving configuration is in [GLM53_FLASH.md](GLM53_FLASH.md#build-and-run).

The optimized scenarios below assume CPU experts execute directly from resident weights, fixed projections remain on GPU, and unnecessary host staging copies are removed. They do not describe the current default runtime: its 4 GiB dense cache cannot hold all fixed GPU weights, and CPU misses are copied into temporary expert blobs. The estimates also assume enough free RAM for the model and overhead, a GPU available to this workload, and short or moderate context. GPU kernel efficiency, CPU quantized-dot throughput and recurrent-state costs remain additional limits.

## Hardware configurations

| Component | Existing environment | Proposed Ryzen configuration |
| --- | --- | --- |
| CPU | Threadripper 1950X, 16 cores / 32 threads, two NUMA nodes; observed with `lscpu` | Ryzen 9 9950X-class, 16 cores / 32 threads |
| RAM capacity | 160 GB, user supplied | At least comparable capacity assumed; actual DIMM layout must be specified |
| RAM bandwidth | 40 GB/s, user supplied; not measured in this session | 100 GB/s initially assumed; revised to 60-75 GB/s for DDR5-5600 planning, not an exact four-DIMM measurement |
| GPU | RTX 5060 Ti 16 GB | Same GPU |
| GPU PCIe link | Gen3 x8, observed with `nvidia-smi`; about 7.9 GB/s theoretical per direction | Gen5 x8; about 31.5 GB/s theoretical per direction |
| SSD | NVMe, 1 GB/s user supplied | Same SSD assumed |

A Gen5 x16 motherboard slot does not give this GPU a 63 GB/s link: the 5060 Ti uses eight lanes. A true x16 GPU would have approximately 63 GB/s theoretical bandwidth. [MSI 5060 Ti 16 GB specifications](https://www.msi.com/Graphics-Card/GeForce-RTX-5060-Ti-16G-GAMING-OC/Specification)

### DDR5 evidence and correction

No verified benchmark for exactly a 9950X with four DIMMs operating at DDR5-5600 was found during this research. The closest firsthand results were:

| Configuration | Benchmark | Measured result |
| --- | --- | ---: |
| 9950X, 2x48 GB, DDR5-5600 JEDEC | Intel Memory Latency Checker | Approximately 61.5 GB/s |
| 9950X, 4x48 GB, tuned DDR5-6000, 32-39-39-99 timings | AIDA64 memory read | 79,027 MB/s, approximately 79.0 GB/s |

The measurements are reported in the [Level1Techs thread](https://forum.level1techs.com/t/192gb-ddr5-9950x-amd5/217337?page=2); the four-DIMM result is visible in its [benchmark screenshot](https://level1techs.us-east-1.linodeobjects.com/original/4X/5/b/4/5b4b8c50bcba8ddf2b9b377005593463d86e0203.png). They use different benchmarks and are not a controlled DIMM-count comparison. Neither establishes the exact four-DIMM DDR5-5600 bandwidth.

The provisional 60-75 GB/s range is an inference for planning. Four DIMMs still use two memory channels, so DDR5-5600 has a theoretical maximum of `5600 million transfers/s * 16 bytes = 89.6 GB/s`. Sustained 100 GB/s therefore cannot be assumed for DDR5-5600 DRAM traffic. AMD officially specifies DDR5-5600 with two DIMMs and DDR5-3600 with four; a four-DIMM 5600 configuration operates above that specification. [AMD 9950X specifications](https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x.html)

## Decode estimates

For RAM bandwidth R, the expert-read ceiling is `R / S`. A simple serial timing model adds 16 ms per token for fixed GPU weight traffic: `tok/s = 1 / (S / R + 0.016)`. The 16 ms is an approximate bandwidth allowance, not a measured GPU execution time; it does not model every tensor expansion, cache effect or overlap.

| Sustained RAM bandwidth | Expert-read ceiling | With the assumed 16 ms GPU cost |
| ---: | ---: | ---: |
| 40 GB/s: existing assumption | 13.1 tok/s | 10.8 tok/s |
| 60 GB/s: revised planning range, lower end | 19.7 tok/s | 15.0 tok/s |
| 75 GB/s: revised planning range, upper end | 24.6 tok/s | 17.6 tok/s |
| 100 GB/s: hypothetical faster memory configuration | 32.8 tok/s | 21.5 tok/s |

The earlier rounded 10-12 tok/s and 20-25 tok/s scenarios allowed for overlap and useful GPU cache hits. For DDR5-5600 planning, the revised conditional estimate is approximately 15-18 tok/s before CPU quantization overhead. It is not measured performance.

Streaming all selected experts to GPU instead gives transfer-only ceilings of about 2.6 tok/s on Gen3 x8, 10.3 tok/s on Gen5 x8, or 20.6 tok/s on a hypothetical true Gen5 x16 GPU. PCIe protocol overhead, staging and GPU execution reduce these figures. CPU expert execution avoids transferring missed expert weights over PCIe.

A uniformly useful 5 GB GPU expert cache covers only about 4.6% of main expert bytes. At 40 GB/s it raises the expert-read ceiling from 13.1 to 13.7 tok/s. A coding-specific profile could achieve a higher traffic hit rate if routing is concentrated; no GLM coding hit-rate measurements are available. Keeping the entire 2.8 GB MTP block on GPU would reduce space for this main expert cache.

## Prefill estimates

Assume independent uniform top-8 routing over 288 experts. For batch B, expected distinct experts per layer are `288 * (1 - (35/36)^B)`. If each distinct expert is read once per batch, expert traffic per token is `S * (36/B) * (1 - (35/36)^B)`.

| Batch | Expert GB/token | RAM ceiling, 40 GB/s | RAM ceiling, 60-75 GB/s | RAM ceiling, 100 GB/s | GPU transfer ceiling, Gen5 x8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 2.771 | 14.4 tok/s | 21.7-27.1 tok/s | 36.1 tok/s | 11.4 tok/s |
| 32 | 2.039 | 19.6 tok/s | 29.4-36.8 tok/s | 49.0 tok/s | 15.4 tok/s |
| 64 | 1.434 | 27.9 tok/s | 41.8-52.3 tok/s | 69.7 tok/s | 22.0 tok/s |
| 128 | 0.835 | 47.9 tok/s | 71.9-89.8 tok/s | 119.8 tok/s | 37.7 tok/s |
| 256 | 0.429 | 93.3 tok/s | 139.9-174.9 tok/s | 233.2 tok/s | 73.5 tok/s |

These are expert-memory or transfer ceilings, not full prefill throughput. The legacy CPU path supports up to eight tokens; the opt-in GPU path now supports 2048/4096-token chunks (measured below). Correlated routing changes reuse, and quantized CPU computation can dominate at large batches. Routed FFNs require approximately 16.9 billion multiply/add operations per token; 120 tok/s requires about 2 trillion such operations/s before quantization overhead. The CPU and GPU columns describe different execution paths and cannot be added without a model of how work is divided.

## Coding speculation and Strata techniques

The largest latency improvement for repeated coding prompts may come from conversation-prefix reuse. Reusing 20,000 tokens and processing only 500 new tokens reduces prompt processing work from 20,500 to 500 tokens, approximately 41x, excluding checkpoint and lookup overhead. This is a work reduction, not a 41x increase in processing throughput.

For an illustrative MTP scenario, draft three tokens and verify four positions. Assume conditional acceptance probability a = 0.8 at every draft position. Expected useful progress is `L = 1 + a + a^2 + a^3 = 2.952` tokens. Let u be expert traffic for the four-position batch divided by traffic for four separate positions. The memory-only speedup is `L / (4*u)`.

The following output rates additionally assume fixed GPU traffic costs 16 ms once per verification batch: `tok/s = L / (4*S*u/R + 0.016)`. This optimistic amortization requires efficient batched GPU execution. Draft execution, rejected-token recovery and additional CPU work are excluded.

| Verification routing scenario | u | Memory-only speedup | At 40 GB/s RAM | At 60-75 GB/s RAM | At 100 GB/s RAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent uniform routing | 0.959 | 0.77x | 9.6 tok/s | 14.0-17.2 tok/s | 22.2 tok/s |
| Moderate expert reuse, assumed | 0.65 | 1.14x | 13.8 tok/s | 19.9-24.2 tok/s | 31.0 tok/s |
| Strong expert reuse, assumed | 0.45 | 1.64x | 19.3 tok/s | 27.4-33.1 tok/s | 41.6 tok/s |

Neither the 80% acceptance nor the moderate/strong reuse factors have been measured for this artifact. The earlier 15-20 tok/s coding scenario on DDR4 and 30-40 tok/s scenario at 100 GB/s were favorable conditional estimates. They are not applicable automatically to DDR5-5600, and weak reuse can make speculation slower. The [GLM MTP implementation author](https://github.com/ggml-org/llama.cpp/pull/27917) reports 15-30% improvement on a different CPU-offloaded configuration with a Q4 draft; that result does not establish performance for this supplied Q2_K/Q3_K draft block.

Strata's prompt-lookup/suffix drafting is particularly relevant when edits copy existing code. Adaptive draft depth should use measured accepted tokens per second and verification cost, rather than acceptance alone. CPU kernels must also reuse quantized weights efficiently across verification positions; otherwise speculative work adds computation without sufficient savings.

Device-side recurrent checkpoints are necessary for efficient rejection recovery. GLM's 34 KDA state matrices occupy approximately 143 MB. Saving and restoring those matrices through Gen3 x8 alone takes about 36 ms theoretically, or about 9 ms through Gen5 x8, before other state. The current full host snapshot mechanism is a correctness check, not an optimized speculative rollback path.

## SSD and measurement requirements

At the assumed 1 GB/s SSD speed, reading the whole 120.36 GB artifact takes about 120 seconds before initialization overhead. Fully cold expert reads have a transfer-only ceiling of about 0.33 decode tok/s. Random-access behavior can lower throughput. SSD reads should leave the steady-state path once weights are resident in sufficient RAM.

Before treating any estimate as an achievable rate, measure sustained read bandwidth with the intended DIMM population and operating speed, CPU expert throughput for the actual mixed quantization, PCIe transfer bandwidth, dense-weight residency, routing overlap across coding tokens, draft acceptance by position, and rollback cost. Record prompt length, prefill batch, generated length, CPU worker count, GPU cache budgets, warm/cold state, and free GPU memory alongside the throughput result.


## Measured GPU prefill, 2026-09-30

On this Threadripper 1950X / 160 GB DDR4 / RTX 5060 Ti 16 GB machine,
4096-token coding prefill exceeded 100 tok/s. Actual GPU link is PCIe Gen3 x8.
The prompt was the first 4096 tokenizer tokens of a C++ decoder source review;
all 45 main layers and the final vocabulary projection were timed. No prefix
reuse, artificial routing, speculative drafting, or MTP execution was used.

| Warm run | Time (s) | Prefill tok/s |
| --- | ---: | ---: |
| 1 | 36.4487 | 112.377 |
| 2 | 39.7009 | 103.171 |
| 3 | 36.0310 | 113.680 |
| Median | 36.4487 | 112.377 |

Context allocation was 8192 tokens, prefill width 4096, CPU expert workers six,
and staging workers two (NUMA-local CPUs 8 and 9). The mixed native IQ3_XXS
artifact was unchanged. Every run streamed 109,867,696,128 native expert bytes
through 756 groups. CPU staging took 22.77-26.47 seconds and overlapped GPU work.
Two subsequent greedy decode tokens were 286 and 595; this is not a decode speed
measurement.

Peak explicitly managed GPU allocation was 10,987.5 MiB (10.73 GiB). The final admission policy reserves
1024 MiB for CUDA/runtime allocations, giving 12,011.5 MiB, below the
12,288 MiB process budget. The allocation counter excludes runtime allocations. An NVML sample during
full-model validation reported 11,870 MiB for the process, below the cap; this
sample does not establish a continuous NVML peak. Desktop allocations are separate. Initialization checks available device
memory and can select 2048 automatically. Larger contexts or desktop pressure
can reduce the available budget or fail admission rather than silently spill.

Weights were prefaulted into RAM, then one complete untimed warmup preceded
three reset-and-recompute measurements. Cold SSD loading, initialization, and
warmup are excluded. This is a measured warm coding-prefix result, not a promise
for every prompt or context length. Initial sequential KDA implementation reached
only 99.48 tok/s median; persistent 64-token KDA chunks removed repeated state
loads and redundant normalization to reach 128.53 tok/s before the chunk-consistency correction. The final fixed
projection shapes measured 112.38 tok/s median; CPU-only compilation overlapped
some trials. All three final trials reported zero major page faults. See
[raw measurements](glm53_flash_prefill_measurement.json).

The GPU path keeps fixed weights resident, uses FP16 dense tensor-core GEMM,
native quantized expert MMQ with Q8_1 activations, and double-buffered 16-expert
streaming. Routing, mHC, recurrent state and sparse-MLA attention remain FP32.
Dense projections use fixed 2048-token tiles and sparse-index GEMM uses the
allocated pool capacity as its leading shape, making chunk sizes numerically
consistent. Decode continues on the CPU-expert/native-CUDA path. Small GPU prompt chunks are
inefficient because streaming groups amortize poorly; retain the default legacy
width eight for short prompts unless measuring the GPU path explicitly.

Reproduce with a UTF-8 coding prompt of at least 4096 tokens:

```sh
PYTHONPATH=tools python3 tools/glm_prefill_bench.py \
  /path/to/models/glm53f/iq3/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf \
  coding-prompt.txt --tokens 4096 --batch 4096 --context 8192 \
  --gpu-budget-mib 12288 --repetitions 3 --output prefill.json
```

The JSON records individual measurements, median, actual selected batch, memory
counter, decode IDs, and decoder log. Use `--batch auto` for memory-based admission
or `--batch 2048` for the smaller workspace. GPU prefill requires the pinned
llama.cpp MMQ backend used by this build.


Full-model 2048-versus-4096 chunk checks pass the 2% normalized L2 limit for
all saved states and vocabulary logits, with the same first greedy token (286).
Before fixing dense and index projection shapes, the check failed with 12.7%
logit error despite the same token; the threshold was not relaxed.
CUDA GLM primitive parity passes, including persistent KDA at 1e-6 scaled FP32
tolerance. Python regression suite: 93 passed, three skipped, 47 subtests passed.
The generic expert_multi_test requires AVX-512/VNNI/VBMI and cannot run on this
Threadripper; the GLM native expert path supports this CPU.


Memory admission and serving checks also passed: explicit 4096-token prefill
was rejected at an 11,000 MiB budget; `auto` selected 2048 with 9739.36 MiB
explicit allocation and a 1024 MiB runtime reserve. A STOP during prefill returned
a completion, then the same process handled a fresh three-token prompt and
returned greedy token 220, confirming the partial state was reset.


The smaller 2048-token coding-prefix benchmark (same source prefix, shorter
prompt; context allocation still 8192) measured 68.0832, 74.9802, and 69.1519 tok/s,
median **69.15 tok/s**. Times were 30.0808, 27.3139, and 29.6160 seconds. All three
runs had zero major page faults. Explicit allocation was 9739.36 MiB (9.51 GiB);
a process NVML sample was 10,622 MiB. Each run still streamed all main expert bytes,
so smaller chunks amortize transfer less efficiently. The 100+ target is achieved
at width 4096, not by this fallback. The benchmark wrapper itself produced these
[2048 measurements](glm53_flash_prefill_2048_measurement.json).

CUDA and CPU-only builds succeeded. Final relevant CTest selection: eight passed;
CPU-only selection: six passed. The backend remains opt-in and experimental;
16K/32K context benchmarks, continuous NVML peak tracking, and broad comparisons
against an independent GLM implementation remain future validation work.


## Measured decode and prompt lookup, 2026-10-01

Same DDR4 / Threadripper 1950X / RTX 5060 Ti / Gen3 x8 environment, same
4096-token coding prefix, context allocation 8192, six CPU expert workers,
and 33 generated tokens. Rates time the 32 decode transitions after prefill;
startup, prefill, the first prefill-derived token, and explicit parity checks
are excluded. These are single paired runs, not three-run medians.

| Decode path | Time for 32 transitions (s) | tok/s |
| --- | ---: | ---: |
| Initial CPU expert path | 55.0689 | 0.5811 |
| Clamped AVX2 CPU path, before removing packing | 43.4516 | 0.7365 |
| Zero-copy clamped AVX2 CPU path | 11.7624 | 2.7205 |
| Zero-copy CPU path, 16 workers | 9.90886 | 3.2294 |
| Zero-copy CPU + lookup, 16 workers | 15.0679 | 2.1237 |
| Zero-copy CPU target + prompt lookup, depth 3 | 16.7664 | 1.9086 |
| Selected-expert GPU + optimized KDA | 31.6749 | 1.0103 |
| GPU target + prompt lookup, depth 3 | 42.4701 | 0.7535 |

The CPU zero-copy path was the fastest single-token mode measured here. It
measured 2.7205 tok/s alone and 1.9086 tok/s with lookup (0.702x). It
removed per-layer native blob allocation, zero initialization, and gate/up/down
copying; native expert kernels now read the mapped tensors directly. Greedy
output was identical to the packed AVX2 path, and eight-position verification
and rollback passed bitwise checks.

The GPU lookup run accepted six of 15 drafts across five windows and replayed
seven input positions. Its speedup was 0.746x (25.4% slower), despite producing
exactly the same 33 greedy IDs as the GPU single-token run. The CPU lookup path
before zero-copy packing similarly measured 0.3601 tok/s. Coding-task speculation
is conditional on both acceptance and routing reuse; this prefix does not show a
speedup. These results supersede using earlier bandwidth-only ceilings as practical
decode predictions for this CPU.

Single-token GPU expert traffic was 97,660,174,336 bytes for 32 transitions
(3.052 GB/transition), with CPU staging totaling 11.877 seconds. Lookup transferred
130,349,858,816 bytes and staged for 15.999 seconds, including verification and
replay. Sparse routing is honored; GPU decode does not transfer all experts.
Native quantized GPU expert evaluation and PCIe/staging cost both matter.

Managed allocation peaked at 11,137.9 MiB in the checked single run and
11,135.5 MiB in lookup. Adding the 1024 MiB runtime allowance stays below the
12,288 MiB cap. An NVML sample during GPU lookup reported 12,018 MiB for the
process. This is a sample, not a continuous NVML peak. Device rollback storage
was 145.605 MiB. Longer context admission can fail instead of exceeding the cap.

Eight-position batched logits and rollback were bitwise identical to sequential
GPU execution, and full generated IDs matched in the paired benchmark. Real
expert tests also cover single/multi clamped AVX2 execution, mapped split tensors,
and CPU pool equivalence at layers 3, 11, 12 and 44. CPU and GPU targets can differ
in low-bit activation quantization and float reduction order, so cross-backend
bitwise identity is not claimed.

[Raw paired decode results](glm53_flash_decode_measurement.json) include per-step
latencies, acceptance/replay counts, output IDs, allocation counters and logs.
The learned GLM MTP block is still not executed; these speculative figures describe
prompt lookup only. Broad coding-task acceptance measurements and an optimized
commit path that avoids replay remain future work.


Increasing CPU expert workers from six to 16 measured **3.2294 tok/s** for single
decode (9.90886 seconds for 32 transitions), versus **2.1237 tok/s** with lookup
(15.0679 seconds). Both produced the same 33 greedy IDs as the six-worker zero-copy
path; lookup again accepted six of 15 drafts and replayed seven positions. For
this fixture the recommended configuration is GPU prefill, CPU decode, 16 workers,
and lookup disabled. This is one hardware/prompt sample, not a broad optimum.

The exact 4096-token UTF-8 coding prefix is saved as
[bench/fixtures/glm53_coding_prefix.txt](../bench/fixtures/glm53_coding_prefix.txt).
Use it with `tools/glm_decode_bench.py --decode-experts cpu --threads 16` to repeat
the paired experiment. Model MTP execution and speculative speedup on favorable
copy/edit tasks are not established by this benchmark.


Final regression checks: CUDA and CPU-only builds succeed; eight relevant CUDA
CTest checks and six CPU-only checks pass; Python suite passes 93 tests with three
skips and 47 subtests. Mapped expert pool outputs match packed direct execution,
and single/multi clamped outputs match bitwise at the four tested main layers.
Serving with GPU decode and lookup enabled passes cancellation, clean next-request
state, and automatic 2048 memory fallback at 11,000 MiB. A small eight-token
prefill allocation also completed with GPU decode/lookup enabled, returning
220, 220, 16 for the standard three-ID smoke prompt. That smoke was not a warm
performance run. The generic AVX-512-only expert test remains unsupported on this
Threadripper, as recorded above.
