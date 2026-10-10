# B550 BF16 WMMA prefill experiment, 2026-10-10

This experiment is opt-in and is not a selected inference preset. The new
`bf16-batched` expert-prefill mode rounds activations and dequantized expert
weights directly to BF16, uses FP32 GEMM accumulation/output, and retains the
existing dense projections, serial KDA and CPU decode recipe. No DDR or GPU
clock changes. Hardware: Ryzen 9 3950X, RX 9070 XT gfx1201, installed HIP 7.14.60850,
DDR 2133 MT/s; original REAP50 Q23 plus the unchanged Q22 down-only sidecar.

The reference requested as `~/Work/gemm/main` exists at
`references/gemm`. `reference-manifest.json` records source hashes.
The 128x128/32-K fragment layout comes from its RDNA4 VLM experiment; vector
loads, LDS packing and one-panel prefetch follow its tuned LLM/VLM variants.
The reference tree was read only. Its MIT notice is retained in the port.

## Kernel correctness and speed

The GPU test checks batch/tail bounds against library BF16 GEMM, samples a
CPU double-precision dot oracle, validates masked/padded expert gathers and
BF16 rounding (including ties, large values and NaNs), and compares direct
BF16 dequantization against FP32 values for Q2_K, Q3_K and i-quant formats.
Final HIP CTest passed both `glm_bf16_prefill_test` and `glm_prefill_parity`.

Microbenchmark: BF16 X[B,T,K] * W[B,N,K]^T -> FP32, three warmups and twenty
GPU-event-timed calls. These numbers exclude routing, transfers, dequantization,
KDA and decode; they are not full-model throughput.

| Variant | T=128,N=4096,K=4096,B=2 TFLOPS | T=256,N=4096,K=2048,B=2 TFLOPS |
| --- | ---: | ---: |
| v1 scalar loads | 15.6412 | 7.09735 |
| v2 vector loads / vector-packed LDS | 32.4337 | 32.9940 |
| v3 one-panel prefetch | 43.4089 | 42.6401 |
| Library BF16, paired with v3 | 98.2181 | 97.1319 |

The port improved substantially but remains slower than library GEMM on these
shapes. `wmma-v1.cu` and `wmma-v2.cu` preserve the earlier implementations;
the final implementation is `src/prefill/wmma_gfx12.cu`. Paired library numbers
for each version are retained in `wmma-v*.parity-bench.txt`.

## Full-model measured timing

Frozen code prefixes, one first-prefill trial, 128 generated tokens, 60 GiB inference cgroup,
zero cgroup swap, GPU budget 15360 MiB and physical reserve 512 MiB. All completed
rows below were clean under the benchmark guard. They are single runs, not medians.

| Record | Context / input tokens | Prefill tok/s | Decode tok/s | Peak cgroup RAM GiB | Min VRAM free MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mmq8k-control` | 8192 / 7936 | 57.101 | 33.3381 | 44.058 | 1158 |
| `bf16-blas8k` | 8192 / 7936 | 389.076 | 33.4788 | 48.605 | 1310 |
| `bf16-wmma8k` | 8192 / 7936 | 392.713 | 33.4540 | 48.291 | 1357 |
| `bf16-wmma16k` | 16384 / 16128 | 426.786 | 33.4496 | 48.798 | 1362 |
| `bf16-wmma32k` | 32768 / 32512 | 445.692 | 32.1624 | 49.104 | 1370 |
| `bf16-wmma8k-final` | 8192 / 7936 | 405.242 | 33.5823 | 48.292 | 1354 |

The first 8K library/custom BF16 difference was under 1%; the final-build WMMA
repeat was 405.242 tok/s. These single-run data do not establish
a WMMA advantage over library BF16. The 32K row has no qualified same-run MMQ
control and therefore supports an absolute timing only. The faster MMQ 8192-batch
control hit its 300-second benchmark/service limit with changed file-page
accounting and low GPU activity; its `.progress.json` and `.runner.txt` are
retained, with no completed result. Do not compare against its partial timing.

The normal MMQ 2048-batch control reproduced all 128 tokens from the earlier
default reference exactly. Its 57.1 tok/s prefill was slower than earlier runs;
current shared-system conditions limit comparisons with earlier measurements.
The 35+ coherent decode target remains unmet. Fast timing does not override the
quality failures below. `summary.json` and `build-manifest.json` link results
and identify the decoder/source used. `timing-build-manifest.json` identifies
the earlier timing build; `repeat-timing-build-manifest.json` records the repeat
binary. `build-manifest.json` identifies the final build after comment-only
cleanup. Both backend builds passed again after that cleanup.
The final host-API isolation leaves generic dequantization format coverage and
legacy entry points unchanged. New IQ BF16 stores are reachable through the
opt-in expert entry point. Final HIP/CUDA rebuilds and both GPU parity tests
passed after this change.

## Quality gate

The matched greedy coding subset uses HumanEval 10/11/17/25, frozen ~14K C++
background and 1024 output tokens at context 16384. Original MMQ passed 3/4.
Library BF16 completed 2/4: tasks 10/11 passed, 17/25 failed, introducing new
failures on tasks 17/25. It fails the conservative selection gate.

Resident WMMA completed task 10 (failed, as MMQ did), then hit the engine's
300-second silence timeout on task 11 after prefill and cache preparation.
It is an incomplete run, not a scored four-task result. Its guard recorded
concurrent upload, build and reclaim activity; the cause was not isolated.
Global swap was full and an unrelated `hf` process used roughly 15 GiB RAM.
That process was left alone. Inference cgroups retained a 60 GiB RAM limit and
zero swap. Timing from functional quality runs does not qualify performance.

Fresh-process WMMA probes have a 240-second service cap. Task 17 completed and
failed to produce a usable function. These probes use a different process
protocol from the earlier resident control and do not establish matched parity.
Task 25 also completed and failed its tests. BF16 remains unselected. Qualified throughput measurements are recorded above.
The fresh functions omitted required imports; the evaluator/protocol was not
changed to forgive those failures.

## Reproduction

Select `prefill_experts: "bf16-batched"` in an experimental config, or
`--prefill-experts=bf16-batched` for the decoder CLI. Absent
`STRATA_GLM_BF16_WMMA`, the mode uses library strided batched BF16 GEMM.
`STRATA_GLM_BF16_WMMA=1` requests the custom gfx1200/gfx1201 kernel and fails
rather than silently falling back on unsupported hardware. The experiment
currently requires one GPU. Existing MMQ/F16 modes are unchanged.

The legacy-named `STRATA_GLM_F16_PREFILL_MIN_TOKENS=4096` also gates BF16:
short batches use MMQ. The fixture configs retain serial KDA row parts 1,
prefill columns 32/chunk 256/minimum 4096 and expert row bucketing step 64.
The new option is accepted by the server config and the coding harness.

```bash
build-hip/glm_bf16_prefill_test --wmma --bench
ctest --test-dir build-hip -R '^(glm_bf16_prefill_test|glm_prefill_parity)$' \
  --output-on-failure -j1
```

For model timing, pass a fixture config and the corresponding frozen
`docs/fixtures/glm_b550_prefill_20261010/code-*.ids` to
`tools/glm_low_memory_bench.py`, with a unique output prefix, RAM 60 GiB,
128 outputs, one trial, GPU capacity 16304 MiB and used limit 15792 MiB.
Do not overlap qualified timing with GPU jobs or compilation.

HIP and CUDA decoder/test builds passed. CUDA is compile-only here; no NVIDIA
GPU is available. These GLM GPU targets do not build on SYCL, and no SYCL
compiler was found. Existing Python tests passed: 5 server and 11 benchmark.
