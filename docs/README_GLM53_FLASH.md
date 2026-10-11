# GLM5.3Flash with Strata

Current configurations and reproduction: [experimental release candidate](GLM_RELEASE_CANDIDATE.md), covering Threadripper 1950X and Ryzen 9 3950X / RX 9070 XT. The initial Q2 results below are historical.

For the 128 GB, two-node Threadripper configuration and current Q2 decode work, see [the NUMA decode report](GLM_Q2_NUMA_DECODE.md).

For Xeon Gold with one or two Tesla V100 GPUs, see the [V100 hardware measurement report](GLM53_V100.md). It covers the dual-GPU reference, single-socket worker comparison and 16GB VRAM-capacity forecast.

This fork runs GLM-5.3-Flash GGUF weights through the experimental `strata-glm-decode` backend. The current reference model is the mixed-quantization **UD-Q2_K_XL**, with all four shards. It runs on a Threadripper 1950X, 160 GB DDR4 and RTX 5060 Ti 16 GB, keeping fixed projections on GPU and executing routed decode experts directly from RAM.

Initial Q2 measurements are **approximately 7-8 tok/s decode**: 7.33 tok/s median for short chat, 7.26 tok/s single decode and 8.35 tok/s GPU MTP for a 4096-token C++ task. Each mode used three stable trials; all six long-prompt outputs were token-identical. Rates describe these prompts and this hardware configuration.

## Initial measured results

| Workload | Model / mode | Input / output | Decode |
| --- | --- | --- | ---: |
| Short chat | Q2, single | 53 / 133 tokens including stop | **7.33 tok/s median**, 3 trials |
| C++ strict decimal parser | Q2, single | 4096 / 393 tokens including stop | **7.26 tok/s median**, 3 trials |
| Same C++ task | Q2, GPU MTP depth 1 | 4096 / 393 tokens including stop | **8.35 tok/s median**, 3 trials |

Source: [initial decode measurements](glm53_flash_q2_initial_decode_measurement.json). Each mode repeats decode three times after one prefill, with a 512-token cap for the long prompt; answers ended naturally. Startup, weight loading, prefill and MTP priming are excluded from decode throughput. A fresh [terminal demo](../demos/glm53_chat/README.md) measured **7.51 tok/s** for the short C++ task; the README animation shows eight seconds at original speed with prefill skipped.

### Initial warm 4K prefill

The same 4096-token C++ chat prompt was benchmarked separately with an untimed full-prefill warmup followed by three reset-and-prefill trials:

| GPU batch | Trial tok/s | Median tok/s | Median elapsed | Peak owned GPU memory |
| --- | --- | ---: | ---: | ---: |
| 2048 | 98.478, 98.281, 98.289 | **98.29** | 41.673 s | 9055.21 MiB |
| 4096 | 157.983, 158.251, 158.573 | **158.25** | 25.883 s | 10303.40 MiB |

Sources: [batch 2048](glm53_flash_q2_prefill_2048_initial_measurement.json) and [batch 4096](glm53_flash_q2_prefill_4096_initial_measurement.json). Both use Q2, 15 workers, automatic affinity, context 8192, dense cache 4096 MiB and total GPU budget 12288 MiB. All six timed trials had zero major page faults. Peak owned allocation plus the 1024 MiB runtime reserve fits the budget; owned allocation is not a measurement of total driver VRAM use. The physical free-memory guard remained enabled. Both batch widths returned the same first two greedy token IDs; this is a limited output sanity check.

Batch 4096 was **1.61x faster** on this prompt. It streams each routed expert set once instead of twice, reducing measured native expert staging from about 198 GB to 99 GB per prefill. These warm results exclude model loading, weight prefaulting and the untimed warmup; cold startup and tiny prompts have different costs.

```sh
PYTHONPATH=tools python3 tools/glm_prefill_bench.py "$GLM_Q2_MODEL" \
  docs/fixtures/glm53_cpp_quality/chat_prompt.txt \
  --tokens 4096 --batch 4096 --context 8192 --threads 15 --cpu-affinity auto \
  --gpu-budget-mib 12288 --repetitions 3 --output /tmp/glm-prefill.json
```

Use `--batch 2048` for the smaller workspace.

Both generated C++17 parsers compiled with warnings treated as errors and passed **400,532 cases each** against an independent `std::from_chars` oracle. Tests covered overflow, UINT64_MAX, invalid bytes, NUL, signs, whitespace, leading zeroes, unchanged output on failure and randomized inputs. ASan/UBSan checks passed; leak detection was disabled for sandbox compatibility. See the [answers, generated code, test harness and report](fixtures/glm53_cpp_quality/README.md).

Single and speculative outputs are token-identical in these initial measurements. The generated parser passed 400,532 cases under ASan/UBSan. These checks validate this prompt, not general lossless speculation or broad coding quality.

The trial rates were 7.324 / 7.334 / 7.327 for short chat; 7.263 / 7.287 / 7.256 for long single decode; and 8.362 / 8.347 / 8.305 for long GPU MTP. Within-mode spreads were 0.14%, 0.42% and 0.67%. One-second samples found no competing process using a full CPU core; aggregate competing activity averaged about 0.57 / 0.59 / 0.38 of one core during decode, with a brief aggregate peak of 1.02 cores in the long single run. This is a quiet-desktop check, not CPU isolation. No other process was stopped or modified.

Reproduce on Linux with the model environment variable from the setup section:

```sh
python3 tools/glm_decode_stability.py "$GLM_Q2_MODEL" --output /tmp/glm-stability.json
```

The helper records three snapshot-reset decode trials per prompt/mode and samples competing CPU usage from `/proc`. Runtime logs and answers remain in the printed temporary directory. The terminal demo reports its own backend timing; playback retains the original generation speed.

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

## Forecast from the initial measurements: CPU, RAM, motherboard and GPU combinations

The initial prefill baseline is **98.29/158.25 tok/s** at batch 2048/4096 on the existing GPU and Gen3 x8 link. Faster PCIe can reduce streaming cost, but these measurements do not isolate transfer latency from staging and GPU work. No numeric prefill forecast for upgraded platforms has been validated; a fourfold link-speed increase does not establish a fourfold prefill gain.

The practical upgrade target is faster **native CPU expert execution with enough resident RAM**, alongside a GPU that fits fixed layers and the draft. Increasing PCIe speed helps GPU prefill and streamed-expert experiments, but CPU decode misses do not transfer all their weights to GPU.

The table gives **conditional planning scenarios**, not benchmarks of these machines. Every forecast cell is in tok/s. `G` is the required speedup of the current CPU-dominated portion of decode, combining CPU arithmetic, effective memory bandwidth and scheduling. No measurements establish that a listed machine achieves its assigned `G`.

| CPU / motherboard class | RAM population | GPU combination | Assumed CPU-phase gain G | Single forecast | MTP forecast |
| --- | --- | --- | --- | ---: | ---: |
| Existing 1950X / X399 | Existing 160 GB DDR4 | RTX 5060 Ti 16 GB | Reference | **7.26 measured** | **8.35 measured** |
| Ryzen 9 9950X / AM5 board supporting the chosen kit and CPU-connected Gen5 GPU slot | 192 GB, 4x48 GB DDR5 UDIMM; actual stable speed must be measured | Keep RTX 5060 Ti 16 GB | 2-3x, hypothetical | **12.7-16.9 tok/s** | **14.3-18.8 tok/s** |
| Threadripper 9960X / TRX50 with compatible BIOS | 256 GB, 4x64 GB ECC RDIMM; populate all four channels | 5060 Ti 16 GB; optional NVIDIA 24-32 GB card for workspace | 4-6x, hypothetical | **20.2-25.2 tok/s** | **22.2-27.3 tok/s** |
| Threadripper PRO 9965WX-class / WRX90 with compatible BIOS | 256 GB, 8x32 GB ECC RDIMM; populate all eight channels | NVIDIA 24-32 GB card; true x16 benefits streaming experiments | 6-10x, hypothetical | **25.2-31.5 tok/s** | **27.3-33.4 tok/s** |

### How the scenarios are calculated

Use a simple latency model with **20 ms/token of fixed GPU/other cost**, chosen as a planning allowance rather than a measurement:

```text
T_new = 0.020 + (T_reference - 0.020) / G
illustrative tok/s = 1 / T_new
T_reference = 1 / 7.26309 for single decode
T_reference = 1 / 8.34745 for GPU MTP
```

The assumed gains are not CPU benchmark results. This model does not isolate actual phase times or predict different routing, thermals, CPU ISA dispatch, cache behavior or platform contention. GPU cache and speculative gains are not added automatically. Each column uses its own initial measured reference, so MTP is not added as an extra multiplier. Keeping acceptance and expert reuse constant while scaling CPU work is itself an assumption; the MTP column is conditional, not a proven hardware speedup.

RAM bandwidth alone gives the loose Q2 expert-read ceiling `R / 2.751`, with decimal GB/s. At assumed sustained 60-75 GB/s this is 21.8-27.3 tok/s; at hypothetical 120-180 GB/s it is 43.6-65.4; at hypothetical 180-280 GB/s it is 65.4-101.8. These exclude CPU quantized-dot work and all other execution, and the higher ranges are scenario inputs, not measured platform bandwidth. They must not be presented as achievable end-to-end rates. The current machine measured 50.70 GB/s in a separate streaming-read test; the initial coding rate is 7.26 tok/s.

### Platform constraints and source evidence

- **AM5 remains dual-channel with four DIMMs.** DDR5-5600 therefore has a 89.6 GB/s theoretical payload maximum, not 100 GB/s sustained. AMD lists DDR5-5600 for two-DIMM configurations and DDR5-3600 for four-DIMM configurations on the [9950X specifications](https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x.html). Treat four-DIMM 5600 operation as above-spec and validate the exact board, kit and BIOS.
- Previous research found approximately 61.5 GB/s for 2x48 GB JEDEC-5600 and 79.0 GB/s for tuned 4x48 GB DDR5-6000, using different benchmarks. Neither is an exact four-DIMM DDR5-5600 measurement. See [DDR5 evidence and links](GLM53_FLASH_PERFORMANCE.md#ddr5-evidence-and-correction); 60-75 GB/s is a provisional planning range.
- **TRX50 exposes four memory channels; WRX90 exposes eight with a PRO CPU.** A PRO CPU in TRX50 does not turn it into an eight-channel platform. AMD documents these [platform differences](https://www.amd.com/en/products/processors/workstations/ryzen-threadripper.html), and the [9960X specification](https://www.amd.com/en/products/processors/ryzen-threadripper/9000-series/amd-ryzen-threadripper-9960x.html) lists DDR5 up to 6400 MT/s. Supported CPU/BIOS lists and RDIMM qualification matter; AM5 UDIMMs cannot be reused as workstation RDIMMs.
- **The 5060 Ti is Gen5 x8.** A Gen5 x16 motherboard slot still gives this GPU at most an x8 link: about 31.5 GB/s theoretical per direction. A true Gen5 x16 GPU has about 63 GB/s theoretical per direction, before overhead. See [MSI's GPU specification](https://www.msi.com/Graphics-Card/GeForce-RTX-5060-Ti-16G-GAMING/Specification). The existing machine negotiated Gen3 x8, about 7.9 GB/s theoretical.
- A 24-32 GB NVIDIA GPU provides more room for resident experts, workspace and desktop use, but a main-cache experiment has not established a decode speedup here. The forecasts above give it no automatic speed multiplier. Raising the GPU budget needs fresh admission and output validation; GLM multi-GPU execution is not a validated upgrade path.

The AM5 scenario keeps the existing GPU while upgrading CPU execution and memory. Four- or eight-channel workstation platforms provide more memory supply and CPU resources, but need platform-specific measurements. Benchmark the real native expert kernel and multiple coding prompts before treating any scenario as a purchase guarantee.

A roughly 1 GB/s NVMe SSD is adequate for loading but cannot sustain decode from cold experts: `1 / 2.751` is only about 0.36 tok/s before random-access and compute costs. Faster SSDs shorten startup and recovery; enough RAM should keep disk reads out of steady-state decode.

## What to validate on upgraded hardware

Measure the actual DIMM operating speed and sustained read bandwidth, negotiated PCIe width, native mixed-quantization expert throughput and GPU headroom. Then rerun both the 4096-prefix/256-output repeated benchmark and the chat-template C++ quality task. Report cold/warm state, output token count, MTP priming time, single/speculative rates and output checks separately. General throughput claims need multiple realistic coding prompts and sampled competing CPU activity, not only the favorable source-prefix fixture.

The [full performance record](GLM53_FLASH_PERFORMANCE.md) retains earlier experiments and theoretical estimates. Its historical sections describe implementations and targets at the time of measurement; this guide summarizes the current accepted configuration and known quality limitations.

## RTX 3070: bounded Q23 prefill (experimental)

The opt-in [3070 preset](../configs/glm53f-reap50-q23-64g-rtx3070-streamed-experimental.json)
uses the unchanged REAP50 Q23 assembly, CPU decode experts, unbiased top-8 routing,
4K context and a 2048-token logical prefill batch. The development host is a
Ryzen 9 3950X, four 16 GiB DIMMs reporting 2400 MT/s, and an RTX 3070 8 GiB.
Linux reports 60.68 GiB RAM. Inference tests use a 56 GiB cgroup with swap disabled.
The 300 tok/s prefill target has not been qualified.

During prefill, `STRATA_GLM_PREFILL_DENSE_MIB=4096` bounds the dense GPU cache.
A fixed subset, including the output head, remains resident; the other matrices
stream unchanged through the cache. Current-layer operands cannot be evicted
while callers retain their pointers. Before decode-cache admission, prefill
buffers are released and all fixed decode weights are restored. Repeated requests
restore the large prefill workspace after draining transfers and invalidating
captured graphs. This mode is for one GGUF GPU with MMQ experts and no MTP,
parallel sequences or prefill expert cache. Allocation guards remain enabled.

`STRATA_GLM_WARM_ACTIVE_ONLY=1` warms CPU-required weights without rereading dense
source pages already owned by the GPU. Streamed dense source pages remain reusable;
only the fixed GPU subset's source pages are discarded. No model files are edited
and no whole-model COW copies or registrations are made.

`STRATA_GLM_PREFILL_PROJECTION_ROWS=128` keeps the original small-batch projection
shapes while amortizing expert transfers over the logical batch. On a frozen
1024-target teacher-forced code prefix, these panels reproduced every reference
logit bit. Larger projection panels failed the gate (mean KL 0.01093, top-1
agreement 96.48%) and are not selected. Expert weight precision is unchanged.

`STRATA_GLM_PREFILL_MLA_TILE` accepts 8, 16, 32 or 64 (default 64); admission and
attention execution use the same tile. The preset retains 64. Two staged prefetch
groups reuse the ordinary upload slots, including on REAP50's 144-expert geometry.
`STRATA_GLM_STAGE_SPREAD=1` spreads staging workers over allowed physical cores.
All new options are absent from existing presets and do not change engine defaults.

For timing, the memory harness accepts `benchmark_decode_warmup: true` to perform
an untimed ordinary decode and restore its starting state before repeated trials.
The equivalent CLI switch is `--decode-warmup`, with `--decode-bench=3` or more.
`STRATA_GLM_PHASE_FAULTS=1` records decode major/minor faults separately from prefill.
`require_deterministic: true` rejects differing repeated greedy outputs.
The coding evaluator accepts `--config` for the preset's batch, context, CPU,
cache and environment settings; specify `--decoder` separately. Configured REAP
evaluations use the API's user/observation turn stops. Existing evaluator defaults
remain unchanged.

For phase-aware teacher-forced checks, set `STRATA_GLM_EVAL_DECODE_PREFIX=128`
with a decode-cache budget and `--eval-corpus` (without `--eval-prefill`). The
runner scores the initial prefix through prefill, trains residency on its routes,
then scores remaining targets one token at a time. This exercises resident expert
combination and one-token approximation options. The prefix must leave at least
one target; ordinary evaluation remains unchanged when the variable is absent.

Use the repository's CUDA build instructions with `CMAKE_CUDA_ARCHITECTURES=86`.
On the development host, CUDA 12.4 and GCC 13 build the runner in
`build-glm-3070-cuda12`. Set the preset's model path to the assembled file, or put
it under the repository's `models/glm53f-reap/`. Keep cold startup, warmed timing,
profiling overhead, page faults, cache admission and quality results separate.

Two alternating same-day rounds used the same frozen 2048-token coding prefix,
128 generated tokens and three trials per round. Load and an untimed 127-step
decode warmup were excluded. The medians were:

| Configuration | Prefill, round 1 / 2 (tok/s) | Decode, round 1 / 2 (tok/s) |
|---|---:|---:|
| Reference batch 128 | 26.360 / 26.379 | 10.210 / 10.212 |
| Streamed batch 2048, projection rows 128 | 206.762 / 206.430 | 10.217 / 10.216 |

This is a 7.8x prefill improvement on this fixture. Decode meets 10 tok/s;
prefill does not meet 300 tok/s. Candidate peak RAM was 55.17 GiB, inference
swap was zero, and sampled free VRAM stayed at least 746 MiB. Candidate prefill
major faults were 0/0/0 and 1/0/0; all timed decode major faults were zero.
All repeated and baseline/candidate output tokens matched. Decode cache admission
was 107 reference slots (929.56 MiB) versus 108 candidate slots (938.25 MiB).
The [machine-readable record](benchmarks/glm_rtx3070_q23_20261011.json) retains
the protocol, individual timings, quality checks and validation limits.

Both configurations passed HumanEval tasks 0 through 29 with identical generated
token hashes on all 30 tasks (1024-token generation limit). This is a matched
subset check, not a full HumanEval score. CUDA builds and selected-path parity
checks passed; the existing optional ordered-KDA bitwise test fails on SM86, so
that path remains disabled. HIP was not built because its toolchain is unavailable;
the GLM runner has no SYCL target.

## RTX 3070: approximate decode (experimental)

The separate [approximate decode preset](../configs/glm53f-reap50-q23-64g-rtx3070-decode-approx-experimental.json)
keeps the same Q23 weight file and the bounded prefill configuration. It drops
selected routes whose normalized share is below 0.07, uses device-side expert
combination, disables adaptive GPU-cache replacement, and defers up to two
low-weight cold routes in layers 10 onward. Deferral changes when the next
hyper-connection mixer sees their contribution. The existing strict preset is
unchanged; these approximations are opt-in.

On the frozen 2048-token prefix, three warmed 127-transition trials measured
15.559, 15.579 and 15.560 tok/s. Repeated greedy outputs matched, timed decode
major faults were zero, peak RAM was 54.91 GiB, and sampled free VRAM stayed at
least 746 MiB. This timing includes every decode transition and excludes loading
and the untimed ordinary decode warmup. It is not a general chat throughput rate.
The 0.07 threshold pruned about 32% of selected routes in this run; it is not a
7% bound on total omitted weight.

Phase-aware teacher-forced scoring used 128 prefill targets, trained GPU residency,
and 896 one-token decode targets. Perplexity was 2.827 strict versus 2.953
approximate, an increase of 4.5%. Against the saved strict batch-128 prefill logits,
candidate top-1 agreement was 94.6% and mean KL was 0.0581; the phase-aware strict
run itself had 97.1% agreement and KL 0.00788 against that reference because the
decode projection shapes differ. Both presets passed HumanEval tasks 0 through
29, with no newly failing tasks. Nine completions retained identical token hashes;
21 changed. This subset does not establish a full HumanEval score or universal
quality equivalence.

See the [decode measurement record](benchmarks/glm_rtx3070_decode_20261011.json)
for configuration hashes, memory limits, validation and the final preset timing.
The public preset's separate three-trial confirmation measured 15.576, 15.550
and 15.582 tok/s (median 15.576), with identical repeated outputs and zero timed
decode major faults. Its median prefill rate was 205.76 tok/s.

Further RTX 3070 experiments are recorded in the
[20 tok/s optimization record](benchmarks/glm_rtx3070_goal20_20261011.json).
Registered expert uploads with the experimental prefill arithmetic measured
377.42 tok/s at 2K and 448.64 tok/s at 8K on the same 3950X, DDR4-2400 and
3070. These rates exclude loading and response preparation. The 18.26 tok/s
decode candidate increased matched perplexity by 7.9%, but introduced a failure
on HumanEval/10 and was rejected. The public preset above remains unchanged;
the 20 tok/s decode target and quality qualification across contexts are pending.

For long-context diagnostics, `STRATA_GLM_EVAL_DECODE_ONLY=1` together with
`STRATA_GLM_EVAL_DECODE_PREFIX=<tokens>` feeds the prefix in configured prefill
batches and scores only the subsequent one-token decode targets. It avoids
retaining prefix logits, requires scalar scoring without saved/reference logit
files, and reports the prefix separately. Compare strict and candidate runs on
identical prefix and target tokens. The original evaluation mode remains the
default.

## RTX 3070: DDR4-2666 presets (experimental)

These presets were measured on a Ryzen 9 3950X, ASUS TUF GAMING B550-PLUS,
user-reported DDR4-2666, 60.69 GiB of Linux-visible RAM and RTX 3070 8 GiB.
They use the original REAP50 Q23 weights. The selected configurations combine
registered expert uploads, ordered two-part KDA prefill, coalesced dequantization
and shared causal attention keys. TF32/FP16 arithmetic, decode route pruning and
partial expert deferral remain opt-in. Existing presets are unchanged.

| Measured prefix / preset | Context capacity | Decode tok/s | Prefill tok/s | Matched PPL increase |
| --- | ---: | ---: | ---: | ---: |
| [2K, tile 32](../configs/glm53f-reap50-q23-64g-rtx3070-shared-prefill-tile32-2k-experimental.json) | 4096 | 19.49 | 464.26 | 5.69% |
| [4K](../configs/glm53f-reap50-q23-64g-rtx3070-shared-prefill-4k-experimental.json) | 8192 | 16.92 | 505.75 | 8.92% |
| [8K](../configs/glm53f-reap50-q23-64g-rtx3070-shared-prefill-8k-experimental.json) | 12288 | 17.63 | 487.60 | 7.57% |

These rates were rechecked after code cleanup; prefill and final benchmark logits
were byte-identical to the accepted build. Rates are medians of three trials
with 128 generated tokens and 127 timed decode transitions, after an untimed decode warmup. Loading, registration and response
preparation are excluded. Runs used a 56 GiB cgroup memory limit, no swap and a
512 MiB physical GPU reserve. Timed decode major faults were zero; startup still
involves paging and reclaim. Prompt contents affect throughput.

Perplexity scored 1,024 frozen decode targets after each stated prefix against
its matched strict baseline. Each configuration passed the same 30 frozen short
HumanEval prompts, satisfying the accepted gate of at most 10% perplexity increase
and no new coding-test failures. Full HumanEval and long-prefix coding were not
qualified. The 20 decode / 500 prefill tok/s target was not reached at all three
contexts.

The 4K and 8K presets use compact FP16 attention scratch and a 3072 MiB prefill
dense budget; the 2K preset uses a 4096 MiB budget. All use four dedicated
prefetch buffers. The active attention stride and shared causal key matrix change
GEMM geometry and rounding, so they remain explicit numerical experiments.

Short attention rows exposed a shared-memory race between reading the softmax
maximum and replacing it with partial sums. A block barrier fixes the race.
CUDA regression checks cover a CPU reference, masked values, row guards and
bitwise repeats across 12 widths; CUDA Racecheck reported zero hazards.
Ordered KDA checks preserve the original FP32
state and output bits for two- and four-part execution.

Earlier qualified alternatives remain available: coalesced
[2K](../configs/glm53f-reap50-q23-64g-rtx3070-coalesced-2k-experimental.json),
[4K](../configs/glm53f-reap50-q23-64g-rtx3070-coalesced-4k-experimental.json),
[8K](../configs/glm53f-reap50-q23-64g-rtx3070-coalesced-8k-experimental.json),
and [shared-prefix 2K with tile 64](../configs/glm53f-reap50-q23-64g-rtx3070-shared-prefill-2k-experimental.json).
Their measurements and rejected experiments are retained in the
[DDR4-2666 benchmark report](benchmarks/glm_rtx3070_2666_20261011.json).

Use `check_prefill_repeats: true` in a benchmark configuration to compare each
independent prefill's logits and route counts with the warmup. This requires at
least two trials and runs outside timing. The existing `require_deterministic`
check covers decode replay from a saved prefill state.

For a guarded CLI measurement, supply a token-ID prompt and a fresh output stem:

```bash
.venv-glm/bin/python tools/glm_low_memory_bench.py \
  configs/glm53f-reap50-q23-64g-rtx3070-shared-prefill-tile32-2k-experimental.json \
  /path/to/prompt.ids --output /tmp/glm-shared-prefill-bench \
  --tokens 128 --trials 3 --ram-gib 56 --timeout 1200 \
  --gpu-capacity-mib 8192 --single
```
