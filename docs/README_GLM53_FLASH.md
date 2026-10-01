# GLM5.3Flash with Strata

This fork runs GLM-5.3-Flash GGUF weights through the experimental `strata-glm-decode` backend. The current reference model is the mixed-quantization **UD-Q2_K_XL**, with all four shards. It runs on a Threadripper 1950X, 160 GB DDR4 and RTX 5060 Ti 16 GB, keeping fixed projections on GPU and executing routed decode experts directly from RAM.

The accepted best decode benchmark is **7.54 tok/s median** with GPU MTP depth 1. A separate C++ task produced correct, coherent code at **about 2.1 tok/s**. **10+ tok/s remains a hardware/implementation forecast, not an achieved result or a general coding-speed claim.**

## Measured results

| Workload | Model / mode | Input / output | Prefill | Decode |
| --- | --- | --- | ---: | ---: |
| Earlier warm source-prefix prefill | Q3, GPU batch 4096 | 4096 / 2 tokens | 112.38 tok/s median | Not a decode measurement |
| Optimized source-prefix benchmark | Q2, GPU MTP depth 1 | 4096 / 256 generated tokens | See raw record | **7.54 tok/s median**, 3 trials |
| C++ strict decimal parser | Q2, single | 4096 / 393 tokens including stop | 84.48 tok/s | 2.15 tok/s |
| Same C++ task | Q2, GPU MTP depth 1 | 4096 / 398 tokens including stop | 84.21 tok/s | 2.12 tok/s |

Sources: [Q3 prefill measurement](glm53_flash_prefill_measurement.json), [best Q2 decode measurement](glm53_flash_q2_physical_core_measurement.json), and [C++ validation measurement](fixtures/glm53_cpp_quality/measurement.json). The C++ rows are one run each, with a 512-token cap; both ended naturally. Startup, weight loading and MTP priming are excluded from throughput. The earlier decode fixture and the chat-template task are different workloads.

Both generated C++17 parsers compiled with warnings treated as errors and passed **400,532 cases each** against an independent `std::from_chars` oracle. Tests covered overflow, UINT64_MAX, invalid bytes, NUL, signs, whitespace, leading zeroes, unchanged output on failure and randomized inputs. ASan/UBSan checks passed; leak detection was disabled for sandbox compatibility. See the [answers, generated code, test harness and report](fixtures/glm53_cpp_quality/README.md).

Single and speculative outputs first differ at token index 96, in a variable name, followed by comments and explanatory wording. Both are correct on this task. Exact speculative token equivalence is **not established** for general prompts, and one parsing task does not establish broad coding quality.

## Terminal demo

Use the [simple ASCII chat + speed counter](../demos/glm53_chat/README.md) for a direct terminal demo with live and final decode tok/s. The [Pi + asciinema setup](../demos/glm53_pi/README.md) records GLM code generation in Pi's TUI. It uses an isolated profile and disables agent tools because GLM tool-call parsing is not implemented.

## Model size and memory

The inspected Q2 artifact has 45 main layers, 34 KDA and 11 sparse MLA layers, three dense FFNs and 42 MoE FFNs. Each MoE layer selects 8 of 288 experts. Hidden width is 4096 and expert intermediate width is 2048.

| Q2 tensor quantity | Size |
| --- | ---: |
| Main expert weights | 99.03 GB |
| Main fixed weights | 6.89 GB |
| MTP draft block | 2.79 GB |
| Total tensor payload | **108.71 GB / 101.25 GiB** |
| Selected main expert weights per decoded token | **2.751 GB** |

Q2 describes a mixture of native quantization formats, not uniformly two-bit weights. All layers and top-8 routing are retained. Plan for **160-192 GB or more system RAM** to keep weights resident while leaving room for the OS, mapped pages and workspaces. 128 GB leaves limited headroom and has not been validated as a desktop configuration here. A larger GPU does not automatically remove the current backend's mapped-RAM requirement.

The reference GPU limit is `--gpu-budget-mib=12288`: owned allocations must leave a 1024 MiB CUDA/runtime allowance, and allocation guards preserve at least 2 GiB of physically free GPU memory. Desktop use can require more headroom. Peak owned allocation in the C++ test was 9055.21 MiB for single decode and 10149.90 MiB for GPU MTP. Keep the guards enabled; insufficient headroom should reject a run.

## Build and run

Use the repository's pinned ggml dependency. The following CUDA architecture is for the tested RTX 5060 Ti; select the architecture appropriate to another GPU.

```sh
cmake -S . -B build-glm -G Ninja \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DSTRATA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-glm -j 4

export GLM_Q2_MODEL=/path/to/models/glm53f/q2/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
build-glm/strata-model-inspect "$GLM_Q2_MODEL"
```

Python text helpers require `regex` and `jinja2` in your Python environment. To reproduce the 4096-input-token C++ generation:

```sh
STRATA_NATIVE_NUMA_LOCAL=0 python3 tools/glm_cpp_quality.py "$GLM_Q2_MODEL" \
  --decoder build-glm/strata-glm-decode --output /tmp/glm-cpp-quality
```

This runs single and GPU-MTP decoding sequentially, saves prompts, token IDs, answers and timing logs, and reports token equality separately from quality. It uses the embedded chat template with low reasoning effort. Compilation and behavioral checks are described in the [validation report](fixtures/glm53_cpp_quality/README.md).

For a short text request:

```sh
python3 tools/glm_generate.py "$GLM_Q2_MODEL" \
  "Write a C++17 function that counts ASCII digits in a string_view." \
  --decoder build-glm/strata-glm-decode --tokens 512 \
  --threads 15 --cpu-affinity auto --prefill-batch 2048 \
  --gpu-budget-mib 12288 --decode-experts cpu \
  --speculative mtp --draft-depth 1 --cpu-prepack-mib 0
```

A large GPU prefill batch is the measured long-prompt configuration; it is inefficient for a tiny prompt. Use `--prefill-batch 8` for the legacy short-prompt path. The text helper's default chat-template reasoning differs from the dedicated low-effort validation runner, so this short example is not an identical benchmark.

Reference long-prompt settings: **15 workers plus the host**, automatic physical-core affinity, context 8192, prefill batch 2048, CPU main experts, GPU MTP depth 1, zero main expert cache and zero CPU prepacking. MTP consumes about 2502 MiB of GPU expert residency and takes additional priming time. CPU drafting and main GPU caches are experimental alternatives, not the accepted fastest configuration. Full CPU prepacking previously exhausted host memory and should remain off.

The original `strata` executable and installer remain the Qwen engine. GLM uses a separate experimental entry point. See [backend and API setup](GLM53_FLASH.md) for integration; do not assume the original README's Qwen speeds, model menu, multimodal features or multi-GPU support apply to GLM. The current GLM text API is greedy-only and does not reuse conversation prefixes.

## Forecast: CPU, RAM, motherboard and GPU combinations

The practical upgrade target is faster **native CPU expert execution with enough resident RAM**, alongside a GPU that fits fixed layers and the draft. Increasing PCIe speed helps GPU prefill and streamed-expert experiments, but CPU decode misses do not transfer all their weights to GPU.

The table gives **conditional planning scenarios**, not benchmarks of these machines. `G` is the required speedup of the current CPU-dominated portion of decode, combining CPU arithmetic, effective memory bandwidth and scheduling. No measurements establish that a listed machine achieves its assigned `G`.

| CPU / motherboard class | RAM population | GPU combination | Assumed CPU-phase gain G | Earlier benchmark forecast | C++ task forecast |
| --- | --- | --- | --- | ---: | ---: |
| Existing 1950X / X399 | Existing 160 GB DDR4 | RTX 5060 Ti 16 GB | Reference | **7.54 measured** | **2.12 measured** |
| Ryzen 9 9950X / AM5 board supporting the chosen high-capacity kit and CPU-connected Gen5 GPU slot | 192 GB, 4x48 GB DDR5 UDIMM; actual stable speed must be measured | Keep RTX 5060 Ti 16 GB | 2-3x, hypothetical | **13-17 tok/s** | **4-6 tok/s** |
| Threadripper 9960X / TRX50 with compatible BIOS | 256 GB, 4x64 GB ECC RDIMM; populate all four channels | 5060 Ti 16 GB; optional NVIDIA 24-32 GB card for additional workspace | 4-6x, hypothetical | **21-26 tok/s** | **7.5-10.5 tok/s** |
| Threadripper PRO 9965WX-class / WRX90 with compatible BIOS | 256 GB, 8x32 GB ECC RDIMM; populate all eight channels | NVIDIA 24-32 GB card; a true x16 interface benefits streaming experiments | 6-10x, hypothetical | **26-32 tok/s** | **10.5-15.5 tok/s** |

### How the forecast is calculated

Use a simple latency model with **20 ms/token of fixed GPU/other cost**, chosen as a planning allowance rather than a measurement:

```text
T_new = 0.020 + (T_reference - 0.020) / G
forecast tok/s = 1 / T_new
T_reference = 1 / 7.54068 for the earlier benchmark
T_reference = 1 / 2.11979 for the C++ MTP task
```

GPU cache benefits and speculative improvements are not added again; the reference already uses depth-1 MTP. This model does not isolate actual phase times or predict different routing, thermals, CPU ISA dispatch, cache behavior or platform contention. The older benchmark crosses 10 tok/s at approximately **1.41x** CPU-phase gain. The C++ task needs approximately **5.65x**. This is why 10+ on a modern desktop is plausible for the favorable benchmark, while sustained 10+ on varied coding prompts remains a larger and unverified goal.

RAM bandwidth alone gives the loose Q2 expert-read ceiling `R / 2.751`, with decimal GB/s. At assumed sustained 60-75 GB/s this is 21.8-27.3 tok/s; at hypothetical 120-180 GB/s it is 43.6-65.4; at hypothetical 180-280 GB/s it is 65.4-101.8. These exclude CPU quantized-dot work and all other execution, and the higher ranges are scenario inputs, not measured platform bandwidth. They must not be presented as achievable end-to-end rates. The current machine measured 50.70 GB/s in a separate streaming-read test, yet the C++ task decoded at only 2.1 tok/s.

### Platform constraints and source evidence

- **AM5 remains dual-channel with four DIMMs.** DDR5-5600 therefore has a 89.6 GB/s theoretical payload maximum, not 100 GB/s sustained. AMD lists DDR5-5600 for two-DIMM configurations and DDR5-3600 for four-DIMM configurations on the [9950X specifications](https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x.html). Treat four-DIMM 5600 operation as above-spec and validate the exact board, kit and BIOS.
- Previous research found approximately 61.5 GB/s for 2x48 GB JEDEC-5600 and 79.0 GB/s for tuned 4x48 GB DDR5-6000, using different benchmarks. Neither is an exact four-DIMM DDR5-5600 measurement. See [DDR5 evidence and links](GLM53_FLASH_PERFORMANCE.md#ddr5-evidence-and-correction); 60-75 GB/s is a provisional planning range.
- **TRX50 exposes four memory channels; WRX90 exposes eight with a PRO CPU.** A PRO CPU in TRX50 does not turn it into an eight-channel platform. AMD documents these [platform differences](https://www.amd.com/en/products/processors/workstations/ryzen-threadripper.html), and the [9960X specification](https://www.amd.com/en/products/processors/ryzen-threadripper/9000-series/amd-ryzen-threadripper-9960x.html) lists DDR5 up to 6400 MT/s. Supported CPU/BIOS lists and RDIMM qualification matter; AM5 UDIMMs cannot be reused as workstation RDIMMs.
- **The 5060 Ti is Gen5 x8.** A Gen5 x16 motherboard slot still gives this GPU at most an x8 link: about 31.5 GB/s theoretical per direction. A true Gen5 x16 GPU has about 63 GB/s theoretical per direction, before overhead. See [MSI's GPU specification](https://www.msi.com/Graphics-Card/GeForce-RTX-5060-Ti-16G-GAMING/Specification). The existing machine negotiated Gen3 x8, about 7.9 GB/s theoretical.
- A 24-32 GB NVIDIA GPU provides more room for resident experts, workspace and desktop use, but a main-cache experiment has not established a decode speedup here. The forecasts above give it no automatic speed multiplier. Raising the GPU budget needs fresh admission and output validation; GLM multi-GPU execution is not a validated upgrade path.

For a cost-conscious upgrade, the 9950X/AM5 scenario is a candidate for exceeding 10 tok/s on the earlier benchmark while keeping the existing GPU. For a **10+ tok/s varied-coding target**, a four- or eight-channel workstation is the stronger experimental direction because it increases both CPU resources and memory supply. Benchmark the real native expert kernel and multiple coding prompts before treating either as a purchase guarantee.

A roughly 1 GB/s NVMe SSD is adequate for loading but cannot sustain decode from cold experts: `1 / 2.751` is only about 0.36 tok/s before random-access and compute costs. Faster SSDs shorten startup and recovery; enough RAM should keep disk reads out of steady-state decode.

## What to validate on upgraded hardware

Measure the actual DIMM operating speed and sustained read bandwidth, negotiated PCIe width, native mixed-quantization expert throughput and GPU headroom. Then rerun both the 4096-prefix/256-output repeated benchmark and the chat-template C++ quality task. Report cold/warm state, output token count, MTP priming time, single/speculative rates and output checks separately. A claim of general 10+ tok/s needs multiple realistic coding prompts, not only the favorable source-prefix fixture.

The [full performance record](GLM53_FLASH_PERFORMANCE.md) retains earlier experiments and theoretical estimates. Its historical sections describe implementations and targets at the time of measurement; this guide summarizes the current accepted configuration and known quality limitations.
