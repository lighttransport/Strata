# GLM-5.3-Flash hardware configurations and throughput estimates

This document records initial estimates from 2026-09-30 followed by chronological implementation and measurement updates. The initial sections are historical bandwidth calculations and conditional scenarios; later sections contain measured results. For the current accepted configuration, quality limitations and hardware forecasts, use [the GLM guide](README_GLM53_FLASH.md). Backend details are in [GLM53_FLASH.md](GLM53_FLASH.md). GB means decimal gigabytes; cache settings in the decoder use MiB.

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

## Decode optimization and desktop incident (2026-10-01)

Experimental changes use GPU KDA and sparse MLA for CPU-expert decode and add
learned MTP proposals checked against greedy target outputs. A 4096-token coding
prefix measured 5.15663 tok/s for 32 decode transitions with 16 CPU workers;
sequential verification and device rollback matched. A 64-token MTP smoke test
accepted 10/15 proposals but measured only 3.99535 tok/s. These are exploratory
runs, not the planned three-fixture, 256-token acceptance benchmark. The 10 tok/s
target has not been achieved.

Two interrupted full CPU packing experiments coincided with desktop failures at
02:01 and 02:09 JST. System logs show systemd-oomd killing GNOME Shell and other
session processes after memory pressure exceeded 50% for over 20 seconds
(observed 77.22% and 82.12%). The second run ended while packing layer 41, before
any decode timing. No NVIDIA Xid appeared in the inspected kernel log. This
supports host-memory reclaim pressure as the cause of the session failures.

Lossless CPU preparation needs 127674 MiB of anonymous allocations in addition
to the original mapped weight working set. It remains disabled by default.
Packing now requires 64 GiB of host capacity beyond the complete packed weights,
32 GiB MemAvailable beyond each next allocation, and low memory pressure. The
full experiment is therefore rejected on the 157 GiB host. Failed preparation
clears partial entries so they cannot be used for later inference.

GPU allocations retain the configured process-owned budget and additionally
check physical free memory, preserving 2 GiB for display growth and library
allocations. Allocation bounds now check offsets before unsigned subtraction.
This physical reserve supplements the existing 1024 MiB runtime allowance;
it does not measure all driver allocations. Small GLM primitive and prefill
parity suites passed CUDA Compute Sanitizer memcheck with zero errors, including
sparse attention, batch boundaries and selection ties. These tests do not prove
all full-model executions are free of CUDA defects.

A guarded follow-up run (64-token coding prefix, 17 output tokens, CPU16,
128-wide prefill, 8192 context, 10240 MiB GPU budget, packing disabled) completed
both decode repetitions and verification/rollback checks. The second repetition
measured 5.45663 tok/s; explicit peak allocation was 9039.51 MiB. Physical free
VRAM during verification was approximately 5.2 GiB and host PSI avg10 remained
zero. This short-prefix result is not a 4096-token throughput measurement.

### Matched decode follow-up

[Raw measurements](glm53_flash_decode_optimization_measurement.json) use the
4096-token C++ coding fixture, 65 generated tokens, 8192 context, 16 CPU workers,
three restored-state repetitions, and no CPU packing. IQ2_S scales are decoded
once per quantization block rather than reconstructed for each 32-value half;
real-expert parity passed with unchanged single/multi-token results. An
interleaved gate/up experiment was removed because it slowed IQ3_S.

| Mode | Median tok/s | Trial tok/s | Accepted proposals | Replayed tokens |
| --- | ---: | --- | --- | ---: |
| Single decode | 5.34647 | 5.44200, 5.34647, 5.31779 | N/A | N/A |
| MTP depth 3 | 5.22723 | 5.28915, 5.22115, 5.22723 | 44/60 per trial | 0 |

The single and MTP greedy outputs match across all three repetitions. These
65-token runs do not satisfy the planned 256-token, three-fixture acceptance
gate; 10 tok/s is still unachieved. MTP priming for 4096 tokens took 12046.6 ms,
excluded from decode timing. The owned allocation peaks were 11137.9 MiB and
11138.1 MiB respectively, including prefill. A driver sample during single
decode reported 11870 MiB of process VRAM. Host PSI avg10 stayed zero in sampled
checks.

Verification now optionally retains intermediate recurrence, convolution and
pending indexer state. Rejection restores an accepted prefix without running
its experts again; retained-prefix and next-token logits matched replay in
seven-prefix checks. The compact decode arena is 32 MiB. State history adapts
to both the owned allocation budget and physical VRAM reserve, up to seven
prefixes. Missing history slots use the original replay path. A 4096-token
lookup-depth-7 run eliminated replay but accepted only 23/84 proposals and
measured a 3.59727 tok/s median. A short-prefix MTP depth-3 run with 32 workers
measured 4.34010 tok/s; additional workers did not help this CPU.

Final validation: CUDA and CPU-only builds passed; the real model checked all
seven retained prefixes plus their next-token logits against replay. The Python
suite ran 50 tests with three skips and no failures. CPU model/pool tests passed.
The separate S2 expert_multi_test cannot run on this AVX2-only CPU because its
entry point requires AVX512-VNNI/VBMI; real GLM AVX2 expert parity passed instead.

### Expert cache coverage and NUMA inspection

A single 256-token coding completion (255 timed transitions) measured 5.14882
tok/s: 49525.9 ms total, 19519.8 ms CPU gate/up, 443.567 ms CPU quantization,
and 12729.0 ms CPU down phases. The new `--routing-trace=path` flag records main
CPU-target expert selections. It adds buffered trace output, so this is a
routing diagnostic rather than the three-repetition acceptance benchmark.

[Cache coverage estimates](glm53_flash_cache_coverage_measurement.json) train
rankings on the first 128 decode positions and evaluate the remaining 127.
They do not execute a GPU cache. With 3072 MiB of slots, 21.47% of held-out
expert bytes are covered. Scaling CPU phase time by uncovered bytes and assuming
zero GPU-hit/fill cost gives only 5.9991 tok/s. A ranking built using the future
evaluation routes covers 24.96% and gives a 6.1643 tok/s estimate; it is an
optimistic static ranking, not a deployable predictor or a strict throughput
bound. Cache coverage alone appears insufficient for the 10 tok/s target.

Reproduce the analysis using `tools/glm_routing_cache.py MODEL TRACE
--decode-log=LOG --output=RESULT.json`. It reads GGUF headers and allocates no
GPU memory. Unit tests reject repeated/incomplete traces and distinguish
training-only selection from future-informed selection.

NUMA inspection found 64322 MiB of capacity on node 0 and
96718 MiB on node 1. System file pages were approximately 31748 MiB and
91109 MiB respectively. Sampling model file pages every 16 MiB found 1562
samples on node 0 and 5615 on node 1 (78.2% on node 1). A small 128-page
migration test succeeded without modifying weight files. Binding the decoder
process to node 1 with 16 workers measured only 3.05295 tok/s median
(3.04222, 3.05295, 3.07973), so that affinity setting is rejected.

`tools/glm_numa_balance.py MODEL` reports a proposed placement without applying
it. The proposed full experiment uses `--apply --node0-percent=40`, touches
only main expert file pages, and stops below 32 GiB MemAvailable or above 5%
PSI avg10. It uses private mappings and does not write weights. Automatic
approval review rejected the full operation because of the scale of memory/I/O
pressure after the earlier OOM incident. It has not run and awaits explicit
user approval. No NUMA rebalancing speedup has been measured.

### Q2_K_XL model experiment

The Q2_K_XL export keeps the same architecture and top-8 expert routing. Its
tensor payload is 108,710,550,904 bytes (101.25 GiB), versus 120,358,051,192
bytes (112.09 GiB) for IQ3_XXS. Main expert tensors occupy 99,033,808,896 bytes
versus 109,867,696,128 bytes: about 9.9% less. Main fixed tensors occupy
6,888,746,232 bytes, and the MTP layer occupies 2,787,995,776 bytes.
This mixed export uses IQ2_XS gate/up and mostly IQ3_XXS down experts, with
Q5_K/Q4_K fixed matrices; the model name does not mean every tensor is 2-bit.

Decode now dispatches fixed matrices through the existing generic native
MMVQ implementation, and grouped GPU down experts also dispatch IQ3_XXS.
Real-weight CPU/GPU parity on layers 3, 11, 44 and 45 passed; CUDA memcheck
reported zero errors. CPU prepacking remains disabled and the owned GPU
budget remains 12288 MiB, including its 1024 MiB runtime reserve, with a
separate 2 GiB physical free-memory guard. Smaller weights alone do not make
the earlier full CPU prepacking experiment safe.

[Q2 measurements](glm53_flash_q2_decode_measurement.json) use the same
4096-token coding prefix, 8192 context, 16 CPU workers and three warm decode
repetitions, producing 65 tokens (64 timed transitions). The matched Q3
results above were collected separately, not interleaved; quantization also
changes the generated sequence, so this is a workload comparison rather than
an identical-routing microbenchmark.

| Model | Single median tok/s | MTP depth-3 median tok/s | Single / MTP peak owned MiB |
| --- | ---: | ---: | ---: |
| IQ3_XXS | 5.34647 | 5.22723 | 11137.9 / 11138.1 |
| Q2_K_XL | 5.34491 | 5.39861 | 10598.7 / 10442.5 |

Q2 single trials were 5.42875, 5.34491 and 4.92137 tok/s; MTP trials were
5.44444, 5.26990 and 5.39861. Single and MTP produced identical greedy IDs,
including across repetitions. Each MTP trial accepted 46/54 proposals in
18 rounds and replayed zero tokens. MTP priming took 16214.5 ms and is
excluded from decode throughput. Q2 lowers owned peaks by roughly 539 MiB
for single decode and 696 MiB for MTP, but provides no material single-decode
speedup and only about 3.3% over the separately measured Q3 MTP median.

The two Q2 prefill observations were 87.958 and 152.351 tok/s, with staging
times of 32159.6 and 14011.3 ms. These are single observations with different
weight-cache conditions, not a stable prefill median. Sampled host PSI avg10
remained below 5% and returned to zero; sampled physical free VRAM was about
2.4 GiB during prefill. No OOM or desktop crash was observed in this run.
The 10 tok/s target and the 256-token, three-fixture acceptance gate remain
unachieved. Four focused CUDA/model/pool regression tests and six benchmark
parser/routing-cache unit tests passed.

The subsequent [256-token depth-7 comparison](glm53_flash_q2_depth7_measurement.json)
measured a 6.10450 tok/s single-decode median (6.05479, 6.12733, 6.10450),
but only 3.73243 tok/s with MTP (3.79525, 3.73243, 3.67251). Greedy outputs
matched across modes and repetitions. Each MTP trial accepted 203/358 drafts
in 52 rounds and replayed 134 target tokens: physical VRAM headroom allowed
only four retained verification prefixes at the time of allocation.
The first MTP trial spent 1359.18 ms drafting, 48700.9 ms verifying and
663.573 ms resynchronizing; its 67189.3 ms total also includes restore/replay
and checkpoint operations. Draft generation is not the principal bottleneck.
New MTP timing counters expose those phases. MTP now caps the effective draft
depth to the available history slots when any slots fit, avoiding rejection
replay caused by overlong proposals. Performance of this cap needs a new run;
the linked depth-7 measurements precede this change.

The [history-capped follow-up](glm53_flash_q2_adaptive_depth_measurement.json)
fit five retained prefixes and capped requested depth 7 to effective depth 5.
Every trial replayed zero tokens and accepted 196/292 drafts in 59 rounds.
Its MTP median was 5.25623 tok/s (5.57040, 5.22358, 5.25623), compared with
the paired single median of 5.72824 (5.85842, 5.72526, 5.72824). This eliminates
the observed replay penalty, but MTP remains slower than single decode.
The median MTP trial spent 1145.85 ms drafting, 46573.1 ms verifying and
676.627 ms resynchronizing, out of 48513.8 ms total. Peak owned allocations
were 10453.7 MiB single and 10735.1 MiB MTP. Greedy IDs matched across all
three repetitions and between modes; sequential/device-rollback logits also
matched. Physical headroom changes the available slot count, so effective
depth is logged per trial. The 10 tok/s target is still unachieved.

`tools/glm_decode_bench.py MODEL CODING_FIXTURE --draft-depths=1,2,3`
compares MTP depths using one single-decode baseline and records every variant
in `mtp_depth_sweep`. Each variant must match the baseline's greedy tokens,
including variants slower than the measured winner. The summary selects the
highest median throughput among the requested depths; it does not imply a
speedup over single decode. Trial records also expose CPU expert phase times
and MTP draft/verify/resynchronization times when the decoder emits them.

For decode-only GPU profiling, `--profile-decode` brackets each decode trial
with CUDA profiler start/stop calls after prefill, MTP priming and any snapshot
restoration. Use Nsight Systems `--capture-range=cudaProfilerApi
--capture-range-end=stop --sample=none --cpuctxsw=none --trace=cuda,cublas`
with one decode repetition to collect that range. Keep profiling reports
outside the repository. Profiling timings are diagnostic and should not be
used as the throughput acceptance measurement; the ordinary benchmark keeps
this option disabled.

The [shallow Q2 depth sweep](glm53_flash_q2_shallow_depth_sweep.json) compared
three 256-token repetitions per mode on the 4096-token C++ prefix:

| Mode | Median tok/s | Trial tok/s | Accepted / proposed per trial |
| --- | ---: | --- | ---: |
| Single | 6.04773 | See measurement JSON | N/A |
| MTP depth 1 | 6.63088 | 6.63088, 6.64787, 6.61372 | 124 / 131 |
| MTP depth 2 | 6.52444 | 6.54291, 6.36846, 6.52444 | 164 / 180 |
| MTP depth 3 | 6.37792 | 6.37792, 6.38259, 6.28751 | 182 / 217 |

Every speculative trial replayed zero tokens and all greedy outputs matched
single decode. Depth 1 is the measured winner for this Q2 coding prefix,
about 9.6% above the paired single median, but remains below 10 tok/s. This
does not establish the best depth for Python/edit fixtures or other prompts.

An isolated [CPU accumulator comparison](glm53_flash_q2_cpu_accumulator_experiment.json)
used real layer-11 IQ2_XS gate and IQ3_XXS down slices with synthetic Q8_K
activations. One integer accumulator instead of two retained bitwise results
at 1, 2, 3, 4 and 8 tokens. Larger IQ2_XS batches improved by about 7–8%, but
one- and two-token batches showed no consistent useful gain across the two
formats. Since depth 1 verifies at most two tokens, the production kernel
remains unchanged. These warm-cache microbenchmarks are not whole-model
throughput measurements.

Decode-only Nsight profiling of 16 transitions at a 64-token prefix found
the serial router consuming 383.742 ms across 672 calls (about 0.571 ms per
call), 47.2% of summed GPU kernel time. Q5_K MMVQ contributed 113.726 ms and
Q6_K MMVQ 100.189 ms. The diagnostic run took 2740.12 ms total, including
1637.742 ms in CPU expert phases; profiler overhead makes this unsuitable
as an acceptance throughput result. Device/host memory operations totaled
about 3.84 ms on the GPU, so actual transfer execution was a small fraction
of this trace. CPU CUDA API waits overlap GPU execution and must not be
added to kernel time.

For up to 512 experts, the router now uses one warp, caches each score once
and selects winners with first-index tie ordering. Lane zero retains the
original selected-weight summation and normalization order. Larger expert
counts retain the serial path. Random and equal-score tests cover expert
counts 1, 31, 32, 33, 127, 128, 287, 288, 511, 512 and 513. GLM primitive
parity passed under CUDA memcheck with zero errors. Full-model throughput
and greedy equivalence after this change are being measured separately.

The [full-model warp-router comparison](glm53_flash_q2_warp_router_measurement.json)
then measured 7.19913 tok/s single (7.21128, 7.19913, 7.02633) and
7.49783 tok/s MTP depth 1 (7.27190, 7.57257, 7.49783), using the same
4096-token C++ prefix and three 256-token repetitions. All greedy IDs match
between modes, across repetitions and against the prior serial-router sweep.
The single median improves about 19.0% over that prior sweep; MTP improves
about 13.1%. CPU expert phases in the median single trial total 25933.163 ms
out of roughly 35421 ms, so CPU expert work remains the dominant cost.
The 10 tok/s target remains unachieved.

The benchmark accepts `--pool-spin-us=N` to set the existing expert-pool
`STRATA_POOL_SPIN_US` control for its child decoders and records the setting
as `cpu_pool_spin_us`. The pool's default spin-before-sleep interval is
20000 microseconds. Shorter values may reduce idle worker activity during
GPU execution, but introduce wakeup overhead; this is a measured tuning
choice, not a change to expert arithmetic or the default runtime policy.

The [100-microsecond spin experiment](glm53_flash_q2_short_spin_measurement.json)
measured 5.94827 tok/s single and 6.11235 tok/s MTP depth 1, below the
default-spin warp-router results. Greedy outputs matched the default setting.
This tuning is rejected; the default remains 20000 microseconds. Read-only
16 MiB-stride sampling of Q2 file pages found 2922 samples on NUMA node 0 and
3561 on node 1 (54.9% on node 1), a less skewed placement than the earlier
Q3 sample. No page migration was performed for Q2.

An optional `--decode-cache-mib=N` single-decode experiment selects main
experts by their frequency during GPU prefill. It uploads their original
quantized gate/up/down slices into allocations tracked by the existing
owned-memory budget. CPU misses retain direct mapped weights. The cache
stays fixed during decode; it does not use future decode routes. Allocation
preserves the 2 GiB physical display reserve and another 384 MiB for runtime
scratch/checkpoints; actual cache size may be below the requested budget.
The prefill arena is compacted before allocating the cache. This option
requires CPU single decode without MTP, lookup, CPU prepacking or the old
expert cache. It is disabled by default; full-length validation and
throughput measurement remain pending. GPU hits use the existing Q8_1 activation
arithmetic, while CPU misses use Q8_K, so equality to an uncached completion
must be evaluated separately from repeatability and verification parity.

Small-cache validation at a 64-token prefix requested 256 MiB and allocated
32 experts / 251.375 MiB. Two 17-token completions were identical and matched
the earlier uncached 17-token completion. Sequential/batched logits, rollback,
seven retained prefixes and next-token logits matched. CUDA memcheck finished
with zero errors; peak owned GPU allocation was 8556.25 MiB. Instrumented
timing is excluded from performance claims. These short checks do not prove
long-completion equality to uncached
decode or the 10 tok/s target.

For single-cache measurements, use `tools/glm_decode_bench.py MODEL FIXTURE
--single-only --decode-cache-mib=3072 --generated-tokens=256 --repetitions=3`.
The result records requested cache size and actual resident slots/MiB, and
checks output repeatability across trials. It explicitly marks the greedy
comparison scope as repetitions only; it does not claim a speculative
comparison. Cache preparation is outside the timed decode interval.

The [3072 MiB cache run](glm53_flash_q2_cache3072_measurement.json) allocated
394 experts / 3069.94 MiB and measured 7.15729 tok/s median (7.15729,
7.29580, 7.12195). Verification and repeated outputs passed, but the 256-token
completion differs from uncached decode, consistent with mixed GPU Q8_1 and
CPU Q8_K activations. It is a workload comparison, not an identical-route
speedup. CPU expert bytes fell to 584,323,170,304 per trial (about 16.7% below
the uncached 701,489,479,680), yet throughput did not improve. Peak owned
allocation was 10646.8 MiB. This configuration has not achieved 10 tok/s.

Cache hit metadata uploads now use reusable pinned buffers on the model's
CUDA stream, and result readback is enqueued before CPU misses execute.
The stream completes before either pinned buffer is read/reused, retaining
its source lifetime and allowing GPU transfers to overlap CPU work. This
replaces synchronous hit-metadata copies and a later blocking result copy;
it needs a fresh correctness/performance comparison against the cache run
above. Extra pinned host storage is roughly 1 MiB for this geometry.

The asynchronous transfer path passed a fresh 64-token-prefix check: both
17-token completions matched the prior cached run exactly, and sequential,
batched, rollback and seven-prefix/next-token verification passed. Its
3072 MiB, three-repetition long-prompt results follow below.

The [asynchronous cache follow-up](glm53_flash_q2_cache3072_async_measurement.json)
measured 6.51619 tok/s median (5.21643, 6.51619, 6.74968). Physical headroom
limited this run to 388 experts / 3022.62 MiB, versus 394 / 3069.94 MiB in the
earlier run. Cache membership and the resulting completion differ, so this
is not a controlled timing comparison of the two transfer implementations.
Within the new run, repeated output and verification passed. CPU expert phases
varied considerably: 37511.269 ms in the first trial, 28443.122 ms in the
second and 27161.420 ms in the third. CPU expert bytes were 567,280,992,256
per trial; peak owned allocation was 10453.9 MiB. It has not established a
throughput benefit over uncached MTP, which remains the measured winner.

Host topology inspection confirmed 16 physical cores with SMT pairs
0/16 through 15/31. A 15-worker, automatic-affinity experiment assigns the
host thread to core 0 and workers to cores 1–15, testing one active thread
per physical core instead of 16 workers plus the host. It changes process
thread placement only; no NUMA page migration or system setting is changed.

The first physical-core run stopped before prefill because its requested
4096-token batch would consume the display reserve. The retry uses 2048-token
prefill batches with the same 4096-token prefix, 8192 context and 256 generated
tokens. This keeps the safety guard intact. The completed
[physical-core run](glm53_flash_q2_physical_core_measurement.json) measured
6.75347 tok/s single and 7.54068 tok/s MTP depth 1 (three repetitions each).
All generated IDs match the earlier uncached warp-router run and match
between single and speculative modes. Peak owned allocation was 10149.9 MiB
in MTP mode. This is the current best measured Q2 median, still below target.

At the time of this experiment, the optimization goal was GLM-5.3-Flash Q2 decode at **10+ tok/s** on
the DDR4 / 160 GB RAM / RTX 5060 Ti machine. Either single or verified
speculative decode may satisfy it. Retain all model layers and top-8 routing,
the 12288 MiB total GPU budget and physical display-headroom checks. Require
three warm 256-token trials at a 4096-token prefix and 8192 context, with
output/correctness checks; extend validation across the coding fixtures
before declaring the goal achieved. The goal remains unmet.

A separate [DDR4 bandwidth test](glm53_flash_ddr4_bandwidth_measurement.json)
used 16 threads and 1.5 GiB of anonymous arrays with parallel first touch.
Five-trial medians were 50.7009 GB/s for read-only streaming and 35.33 GB/s
for a triad, counting logical bytes. These access patterns differ from the
quantized expert kernels, and triad accounting excludes write-allocation
traffic; neither number is a strict decode bound. They support investigating
expert arithmetic and scheduling rather than assuming 40 GB/s is the
measured limit of this machine.

The native CPU expert pool now accepts the experimental environment setting
`STRATA_NATIVE_TASKS_PER_THREAD=1..16`; its default remains 3. It controls
row-range granularity in gate/up and down phases without changing arithmetic,
expert selection, or output accumulation. Invalid values fail before worker
creation. A sequential short-prefix sweep compares 3, 1, 6, 12, then 3 again,
with three 65-token trials per process and exact token equality checks. Its
[incremental measurements](glm53_flash_q2_task_granularity_measurement.json)
are screening evidence, not proof of the long-prefix throughput target.

The completed screening medians were 5.31640 (3 ranges), 5.44774 (1),
6.46837 (6), 6.16887 (12), then 6.38024 (3 again) tok/s. All outputs
matched. The baseline drift is much larger than the apparent advantage of
6 over the final baseline; an inadvertently concurrent compiler microbenchmark
also overlapped part of the sweep. No default is changed from this evidence.

The subsequently isolated [compiler scheduling experiment](glm53_flash_q2_cpu_tuning_experiment.json)
compared identical kernel source with and without `-mtune=znver1` on real
layer-11 IQ2_XS and IQ3_XXS expert slices. All results were bitwise identical.
Single/two-token speed ratios were 1.010/1.018 for IQ2_XS and 1.005/0.992
for IQ3_XXS; three-token IQ2_XS fell to 0.794. This does not justify a
production compiler-flag change.

A full-prefix follow-up tests 31 CPU workers plus the host with automatic
thread placement, default row granularity, 2048-token prefill batches, and
MTP depth 1. It starts after both preceding workflows finish and preserves
the GPU memory guards. The completed
[SMT run](glm53_flash_q2_smt_measurement.json) measured 3.88241 tok/s single
and 3.27814 tok/s MTP depth 1. All output IDs match between modes and
repetitions and match the earlier uncached baseline. Peak owned allocation
was 9055.21 MiB single and 10149.9 MiB MTP. This worker configuration is
rejected for throughput.

An isolated IQ2_XS sign-table experiment runs after this workflow,
preserving arithmetic and requiring bitwise output equality against vector
parity reconstruction on the same expert slices. A separate row-prefetch
comparison follows it; neither has changed the production kernels.

The [sign-table microbenchmark](glm53_flash_q2_sign_lookup_experiment.json)
measured IQ2_XS speed ratios of 1.061, 0.967 and 0.847 for one, two and
three tokens, with bitwise-identical outputs. A production specialization
therefore uses the existing sign table only for one-token rows and retains
vector parity reconstruction for multiple tokens. Rechecking the actual
specialization against the original kernel passed bitwise equality at
1, 2, 3, 4 and 8 tokens for both tested formats; its one-token IQ2_XS ratio
was 1.054. Model and pool tests passed. A full-prefix single/MTP comparison
with verification is running; no full-decode speedup is claimed yet.

The [row-prefetch microbenchmark](glm53_flash_q2_cpu_prefetch_experiment.json)
found no consistent benefit across the tested formats/token counts at
one or four rows ahead. Prefetching remains an isolated experiment and is
not added to the production kernel.

The one-token sign specialization's full-prefix follow-up has reported
6.88363 tok/s single-decode median; MTP trials are still pending. This
does not establish a decode throughput improvement. A combined 512 KiB
signed-grid lookup experiment and a comparison with ggml's existing CPU
dot kernels are serialized after this run. Both use the same expert slices
and activations, with output parity checks, before any further production
change is considered.

Correction: the earlier isolated accumulator, compiler, sign, prefetch,
combined-grid and ggml-dot microbenchmarks used incorrectly labeled slices.
Layer 11 gate is IQ3_XXS (18), and down is IQ4_XS (23), but these tests
parsed them as IQ2_XS (17) and IQ3_XXS (18). Their numerical and timing
conclusions above are invalid. Their JSON reports are explicitly marked
invalidated. The one-token production sign specialization has been removed.
Full-model decode reports use the correct tensor metadata and remain valid:
the specialization follow-up measured 6.88363 tok/s single and 7.21094 tok/s
MTP with outputs matching the earlier baseline, without establishing a gain.
Corrected isolated tests use layer 3 expert 7: gate type 17 / 2424832 bytes,
down type 18 / 3211264 bytes. Sizes are checked against GGUF metadata and
every block scale is finite; outputs must also be finite before parity can
pass. These corrected comparisons are running sequentially.

The corrected comparisons completed with finite outputs. The
[sign lookup](glm53_flash_q2_sign_lookup_corrected_experiment.json) measured
1.064 / 0.964 / 0.836 speed ratios at 1 / 2 / 3 tokens, bitwise equal.
The [combined grid](glm53_flash_q2_signed_grid_corrected_experiment.json)
measured 1.012 / 0.965 / 0.862 with bitwise equality.
The [ggml comparison](glm53_flash_q2_ggml_dot_corrected_experiment.json)
passed relative-error checks (about 5e-8), but was slower: one-token ratios
were 0.977 for IQ2_XS and 0.935 for IQ3_XXS. Original vector kernels remain
in production; the isolated sign gain has not established a full-decode gain.

The [read-only NUMA locality probe](glm53_flash_q2_numa_locality_measurement.json)
sampled eight experts per main layer, three row-page addresses per matrix
in each 128-row chunk. Of 10752 sampled down chunks, 10537 (98.0%) had
all samples on one node. Gate/up pairs were less local: 3252 of 5376
(60.5%) had all six samples on one node. Sampled pages were 60.3% node 1
and 39.7% node 0. No pages were migrated and no private model copy was
created. Sparse samples do not prove whole-chunk placement or current
worker locality, but support testing node-preferred queues with stealing
under the existing epoch/completion protocol. The probe is available as
`tools/glm_numa_locality.py MODEL --output=REPORT.json`; it has no migration
mode. No scheduler or throughput change is claimed from this probe alone.

An experimental native down-projection scheduler is now opt-in with
`STRATA_NATIVE_NUMA_LOCAL=1` (default 0). It samples existing row pages
using query-only `move_pages` with NULL targets and flags 0, assigns each
task to the node with more known samples, and allows workers to steal
after exhausting their local queue. Unsupported topology/query failure
falls back to the global queue. Row arithmetic, model layers and expert
selection are unchanged. Page queries and queue preparation are included
in timed decode work; no page migration or model copy is requested.

The [short scheduler check](glm53_flash_q2_numa_scheduler_smoke.json)
passed identical 17-token completions at a 64-token prefix for both modes
and repetitions, plus sequential/batched and rollback verification.
The enabled run logged actual local queue activation with known resident
pages. These short timings are not a throughput acceptance claim.
Full-prefix 256-token single/MTP trials are running with 15 pinned workers
plus the host, 2048-token prefill batches and the existing GPU guards.
A CPU-only regression test, `native_pool_test`, is queued afterward to
check 384 synthetic native batches against finite serial outputs, with
sentinels for missing rows, host participation on/off and varying job/token
counts. Its result is pending.

The [completed local-queue run](glm53_flash_q2_numa_scheduler_measurement.json)
measured 5.16138 tok/s single and 6.63929 tok/s MTP depth 1. Outputs
matched the earlier physical-core baseline and between modes/repetitions.
This does not establish a throughput benefit. Concurrent CPU-heavy
conversion/compiler processes were observed near the end of this workflow,
so attribution against older runs is uncertain. The setting remains off
by default. `native_pool_test` passed 384 batches against finite serial
outputs, including host participation on/off, varied job/token counts,
repeated batches and sentinel checks for missing rows.

Opt-in scheduling now reports `NATIVE_NUMA_TIMING prepare_ms=... queries=...`,
including page-query and queue-building work; the benchmark parser records
these per-trial values. A serialized short diagnostic runs default, local,
then default again, checking exact token equality and baseline drift.
It follows the completed native pool test and remains under the GPU guards.

The diagnostic completed with exact token equality across all three runs:
[default first](glm53_flash_q2_numa_diagnostic_0_measurement.json) 6.63659,
[local](glm53_flash_q2_numa_diagnostic_1_measurement.json) 6.53504, and
[default last](glm53_flash_q2_numa_diagnostic_2_measurement.json) 6.67845
tok/s. Local preparation took 218.457 / 220.147 / 211.135 ms over 2688
queries per trial, about 2.2% of total decode time. These short-prefix
results identify query overhead, but do not demonstrate a scheduling gain.

An experimental `--mtp-experts=cpu` option now runs draft routed experts
directly from mapped weights through the existing native CPU pool. Draft
attention, shared FFN and output head remain on GPU. The default remains
`--mtp-experts=gpu`. CPU drafting avoids allocating the 2502 MiB resident
draft-expert bank; it uses reusable activation/results buffers rather than
a private model copy. CPU Q8_K activations may change draft proposals
relative to GPU Q8_1, so greedy target verification is still required.
CPU expert-phase counters include draft work in this mode; MTP draft and
verify timings remain separately reported. Model and native-pool tests
passed. A short-prefix CPU-draft comparison with verification is running;
no decode benefit or larger main-model cache is claimed yet.

The [CPU-draft short comparison](glm53_flash_q2_cpu_mtp_smoke_measurement.json)
completed with all 65 generated IDs matching between single and speculative
modes and across three repetitions, with verification enabled. Single
median was 6.73541 tok/s; CPU-draft MTP median was 6.45631 (5.30034,
7.16411, 6.45631), so it has not demonstrated a throughput benefit.
Draft expert residency was zero; allocated GPU memory after 64-token
priming was 7355.27 MiB. Peak owned allocation during the speculative
process was 9055.21 MiB, including prefill. Draft/verify timings varied,
with draft 214.333 / 173.849 / 190.524 ms and verify 11469.7 / 8493.51 /
9411.22 ms. This validates the low-VRAM draft path on a short prefix,
not the full 4096-prefix acceptance gate. Combining CPU drafting with a
main-model expert cache remains unimplemented and unmeasured.

CPU-draft MTP can now be combined with `--decode-cache-mib`. GPU drafting,
lookup speculation, CPU prepacking and streamed target experts remain
incompatible with this cache mode. Cache preparation reserves 768 MiB of
owned/physical headroom for workspace and verification in addition to the
existing 2 GiB display-headroom check. Total GPU budget remains 12288 MiB.
The cache report includes a deterministic resident-set fingerprint.
The benchmark passes the measured single-run slot count to speculative
decode via `--decode-cache-slots`, requires exact actual-cache equality,
and stops if headroom cannot reproduce that set. It still requires
identical greedy output across modes and repetitions. This prevents a
timing comparison from silently using different cached experts.

The [combined short test](glm53_flash_q2_cached_cpu_mtp_smoke_measurement.json)
passed identical cached single/speculative 65-token outputs across three
repetitions at a 64-token prefix, exact resident-set equality, and baseline
verification. Model/native-pool tests and seven benchmark-parser tests
passed, including rejection of a mismatched resident fingerprint even
when token output matches. The [full-prefix 256-token trials](glm53_flash_q2_cached_cpu_mtp_measurement.json) with a requested
3072 MiB cache have completed with identical greedy outputs between modes; actual cache size remained subject to the GPU
guards. No 10 tok/s result was established.

### Accepted decode performance and C++ quality validation

The user accepted the prior best 7.54 tok/s median; further pursuit of 10 tok/s
is paused. The subsequent [C++ validation](fixtures/glm53_cpp_quality/README.md)
used 4096 chat-template input tokens and a 512-token output cap. Single and GPU
MTP depth-1 answers ended naturally at 393 and 398 tokens. Both generated C++17
functions compiled with warnings treated as errors and passed 400,532 cases each
under ASan/UBSan. They are coherent and correct for this parsing task, but their
greedy tokens differ beginning at index 96; exact speculative equivalence is not
established on this prompt. Measured prefill was 84.48/84.21 tok/s and decode
2.15/2.12 tok/s respectively, demonstrating that the earlier 7.54 rate does not
predict this task's decode rate. See the linked report for configuration and scope.
