# B550 simulator calibration and the path to 20 tok/s

Status on 2026-10-09: **Lossless decode remains below 20 tok/s. Opt-in lossy routing sustained 21.27–22.64 tok/s over 512-token outputs, but is not quality-qualified.** See the [new speed and quality experiments](../bench/results/2026-10-09-b550-glm-reap50-speed/README.md). A fresh 8K document chat at affinity .12 measured **20.0 tok/s decode / 182.5 tok/s prefill** in one request; both it and the unbiased reference gave coherent but capped answers with a shared threshold interpretation error. The corrected larger-cache configuration measures **11.38 tok/s decode and 231.06 tok/s warm prefill** for 2,048 input + 512 output tokens, median of three repetitions. Held-out tasks measure 11.19–11.61 decode and 227.94–230.89 warm prefill. A HIP canonical-rounding bug found during validation was fixed; captured-expert, all-CPU/cached, and adaptive-residency checks now pass. The web demo remains stopped. The serving checkout and its defaults were not changed; builds and experiments used the isolated checkout.


A later [quality-first sweep](../bench/results/2026-10-09-b550-glm-sweet-spot/README.md) recommends opt-in affinity **.06**, measured at **16.46 decode / 229.12 warm prefill tok/s** over 2K + 512, three repetitions. It completes the 8K document and 4K C++ tasks; its parser passes 5,020 boundary/malformed-input cases. Faster .08/.10 variants failed the capped C++ answer requirement. This remains a narrow experimental recommendation, not 128K or general quality qualification.

A subsequent [fixed-affinity .06 optimization](../bench/results/2026-10-09-b550-glm-aff006-tuning/README.md) reaches **16.5 decode / 250.2 prefill tok/s** on the real 7,358-token document, median of three fresh requests with an 8K batch and 32-token cache probe; first-request prefill is 228.7. Preparation time is included. Batch-only reaches 294.2 prefill / 15.5 decode median, first prefill 265.3. **The 18–20 decode target remains unmet.** All document factual answers are correct, but action-list metric errors persist; the probe C++ parsers pass in two of three trials, with one cap and one parser failing 678/5,020 cases. These profiles remain experimental; the earlier quality-first profile and serving defaults are preserved. Probe state/next-logit restoration passes, HIP/CUDA compile checks pass (CUDA uses an alternate available GGML revision and no GPU execution), and the report records memory guards and reproduction. The existing routing calibration rejects the new cache-learning/probe records; it is not calibrated for these profiles.

## Machine and target

Measurements used B550: Ryzen 9 3950X, 16 cores / 32 threads, 62.693 GiB physical RAM, RX 9070 XT (`gfx1201`), 16,304 MiB reported VRAM, PCIe 4 x16, ROCm 7.14 development build. GPU clocks were automatic. The benchmark guard allowed 60 GiB RAM, no swap, and a physical GPU reserve. CPU governor was `powersave` with boost enabled. These are whole-machine results, not a bandwidth specification.

The target was `/mnt/disk01/models/glm53f-reap/GLM-5.3-Flash-REAP50-Q23-assembled.gguf`, SHA256 `c3b0cfb144556bd28e265d8255d48eeafce3d127aba5b9b5dd4e326322a34aa9`: 58.87 GiB file, 144 experts per MoE layer, top 8, 42 main MoE layers. Main routed weights are 55,094,280,192 bytes. The unchanged Q23 model is the quality reference; this does not establish equivalence to the original unpruned model.

The secondary Q3_K_M file has 72,137,834,496 main routed bytes (67.18 GiB), Q3_K gate/up and Q4_K down. Its header census is included, but no new throughput claim is made for it. Its larger expert working set makes resident-memory assumptions particularly unsafe on this host.

The source baseline was `15b787b5`. Work was built in the isolated B550 directory `/home/syoyo/work/Strata-b550-calibration-15b787b5`; the existing dirty serving checkout was preserved. Local TR16 performance jobs were not run.

## Measurements

All throughput rows below are medians of three repetitions. **Prefill is warm:** `--bench=3` runs an untimed prefill first, after which model pages are resident. A separate cold diagnostic took about 69.6 seconds for 2K input (29.4 tok/s); the 230 tok/s rates do not include that initial SSD read. The first six use 128 generated tokens and the same 2K prompt; the final row uses 512. These are pre-fix screening measurements, not held-out quality-qualified results. Cache sizes are actual admitted sizes, not requested limits.

| Configuration | Cache MiB | Decode tok/s | Prefill tok/s | Minimum free VRAM MiB |
|---|---:|---:|---:|---:|
| Baseline, no MTP | 6142 | 10.30 | 230.83 | 3002 |
| Requested 7 GiB cache | 7167 | 11.14 | 230.65 | 1822 |
| Requested 8 GiB cache | 7741 | 10.96 | 230.52 | 1162 |
| MTP depth 1 | 7715 | 10.74 | 231.11 | 764 |
| MTP depth 2 | 7706 | 8.54 | 231.02 | 756 |
| MTP depth 3 | about 7688 | 7.84 | 230.57 | about 740 |
| 8 GiB request, 256 MiB runtime headroom, 512 output | 8184 | **11.59** | **230.51** | **652** |

The corrected-kernel baseline subsequently measured **10.4652 decode / 230.833 prefill tok/s** on 2K + 512, three repetitions, with identical greedy output in all three. Minimum free VRAM was 3002 MiB. These corrected results replace the pre-fix data for the current baseline calibration. The corrected 8 GiB request (8183.62 MiB admitted) measured **11.3763 decode / 231.056 prefill tok/s**, with 652 MiB minimum free VRAM. This is an 8.7% median decode gain over the corrected baseline on separate process states, not the 20 tok/s target.

Corrected held-out prompts (2K input + 512 output, three repetitions, 8 GiB requested cache) were frozen before fitting. Every repetition within each case emitted identical tokens. All passed resource guards with at least 652 MiB free VRAM:

| Held-out task | Decode tok/s | Prefill tok/s | Decode prediction error |
|---|---:|---:|---:|
| C++ code review | 11.606 | 230.887 | −3.3% |
| Document summary/Q&A | 11.193 | 229.923 | +0.3% |
| Installation chat | 11.265 | 227.935 | −0.3% |

The simulator was fitted only to the corrected baseline and independent CPU anchors. All twelve current decode/prefill records, including the MTP1 screen below, meet their 10%/15% tolerances. These short, fixed-output benchmarks measure speed and repeatability; they do not certify complete chat answers or the long-context quality ladder.

Corrected MTP1 was screened separately at 2K + 128, three repetitions: **10.8698 decode / 231.462 warm prefill tok/s**, 44 accepted drafts out of 82 proposals per repetition (53.66%), 764 MiB minimum free VRAM. Its token streams repeat exactly. It does not beat ordinary decode and was not promoted to a 512-token candidate. The historical acceptance profile predicts this corrected screen within 0.7%, but MTP2/3 and other-domain MTP acceptance are not requalified.

The original serving binary separately measured 10.27 decode / 229.84 prefill. The rebuilt baseline agrees closely. The 512-token result passed memory guards without swap or OOM. Before the rounding fix, different repetitions produced different greedy token streams, including the original binary, so those cache comparisons do not isolate identical output workloads.

The pinned-transfer probe measured 27.21 GB/s H2D alone. With eight CPU read workers it measured 20.36 GB/s H2D plus 10.32 GB/s CPU reads; with twelve, 10.88 plus 19.63 GB/s. Each point is a three-run median, with 9 MiB transfers from a 512 MiB pinned allocation, two seconds per sample, and copied-byte verification. The CPU load is a read proxy, not the expert kernel. This establishes contention and does not justify adding CPU and standalone PCIe rates as independent resources.

Measured CPU expert counters give about 27.99 GB/s logical weight bandwidth. Baseline CPU expert reads were roughly 1.94–2.01 GB per decoded step. The corrected 512-token repetitions read 1.936–1.949 GB per step at 28.00–28.07 GB/s. Decode repetitions reuse the same process and cache; they are not independent cold-start requests. CPU reads dominate this configuration.

### Real-input CPU and cache replay

A corrected 64-token capture supplies real activations and routes. The CPU-only kernel matrix uses one window after skipping 16 positions, all 42 MoE layers, five timed repetitions after warmup, canonical arithmetic and layer dataflow. It includes expert computation and a uniform-weight routed reduction, not full decode. Output hashes agree across all worker counts at each width.

| CPU workers (plus host participation) | Width 1 GB/s | Width 2 GB/s | Width 4 GB/s |
|---|---:|---:|---:|
| 4 | 25.00 | 24.90 | 24.43 |
| 8 | 27.85 | 27.83 | 27.68 |
| 12 | 28.93 | 28.91 | 28.78 |
| 16 | 28.84 | 28.70 | 28.30 |

The profiling timeline spends about 24.7 ms/step outside CPU expert service (head, between-layer waits, tail). Holding that observed overhead fixed, a 50 ms ordinary decode step would leave about 25.3 ms for CPU service, or roughly 0.71 GB at 28 GB/s. That implies about 77% routed-byte residency versus the observed 43.5%. This is a conditional budget, not a guarantee that overhead stays fixed under a redesign.

The real routing trace contains 2,646 executions (63 steps × 42 layers). Its reconstructed CPU bytes exactly match the engine's 108,986,105,856-byte counter. Current adaptive residency serves 43.48% of routed bytes and uploads 2.915 GB in 320 promotions. Immediate-admission replay from the same initial inventory gives:

| Policy | Routed-byte hit share | Uploaded GB over 63 steps |
|---|---:|---:|
| Current adaptive, observed | 43.48% | 2.915 |
| Frozen initial inventory, replay | 38.72% | 0 |
| LRU, optimistic replay | 41.99% | 111.856 |
| Frequency-protected LRU, optimistic replay | 44.49% | 107.046 |

The small protected-policy hit gain costs far more uploads. These policies do not pass the cost gate for implementation. `policy-replay.json` reports a staging traffic estimate that assumes a source read, staging write and DMA read reach DRAM; CPU cache reuse can reduce this estimate. It also reports standalone upload time and explicitly omits promotion delay, staging capacity and GPU execution. It is not a throughput forecast.

Reproduce the component checks while the B550 is idle:

```sh
PY=/home/syoyo/work/Strata/.venv-glm-hip/bin/python
$PY tools/glm_b550_calibrate.py --output runs/repro-trace --tokens 64 --trials 1 \
  --cases cache8 --trace --capture --env STRATA_GLM_TIER_RUNTIME_RESERVE_MIB=256
$PY tools/glm_b550_kernel_matrix.py /mnt/disk01/models/glm53f-reap/GLM-5.3-Flash-REAP50-Q23-assembled.gguf \
  runs/repro-trace/cache8.capture --output runs/repro-kernels
python3 tools/glm_b550_replay.py docs/fixtures/glm_b550_sim_20261009/corrected-events.jsonl \
  --request 1 --capacity-mib 8192 \
  --engine-log docs/fixtures/glm_b550_sim_20261009/corrected-trace.log
```

The importer rejects traced or parity-check runs as throughput fitting data.

## Simulator changes and limits

`tools/sim` now accepts exact header censuses for both REAP packs, replays 144-expert traces, accounts for B550 phase-specific VRAM and compact MTP history, supports per-position conditional acceptance, and bounds concurrent CPU/DMA traffic by host memory bandwidth. Routing JSONL preserves rejected speculative branches and repeated token positions. The replay helper reports uploaded bytes as well as hits; its immediate-admission policies are exploratory upper bounds, not implementations of the engine's delayed promotion policy.

Fit rows and held-out rows are separate. The optimizer does not use held-out rows by default, and validation fails if any required row exceeds its tolerance. `--observed-routing` records are conditional component diagnostics, not workload forecasts. The B550 planner defaults to implemented ordinary/MTP candidates; hypothetical architectures require explicit opt-in. `lossless` describes the selected arithmetic/model transformations, not a passed runtime quality certificate.

`params_b550.json` is **provisional, fitted only to the corrected 6 GiB baseline**. The corrected larger-cache run and new workload fixtures are held out. Earlier pre-fix fit/records are preserved under the fixture directory, separate from the current fit. CPU bandwidth is anchored to corrected-run counters, width penalty to the independent real-input CPU matrix, and prefill staging to the recorded 10.53 GB/s. Exposed graph-launch fraction and effective prefill GEMM time remain aggregate fit terms. The very small fitted launch fraction must not be interpreted as an independent kernel measurement. New-workload acceptance, actual GPU component timing, and long-context memory growth still need validation. The pre-fix ordinary-decode predictions were within about 2%; generic acceptance overpredicted its MTP1/2/3 rows by 23%, 49%, and 51%. Those failed forecasts are retained as historical evidence. The corrected capacity holdout is predicted within 1.3%. The planner’s `b550_screen` acceptance profile comes from the historical screen. Corrected MTP1 checks it on the same prompt, but it is not a forecast for arbitrary prompts or a requalification of MTP2/3.

## Quality gate

Fixed-residency sequential graph/direct launches and rollback checks passed. Synthetic canonical CPU/GPU expert parity, mailbox checks, and GLM primitive parity passed. Additional half-subnormal scale cases also passed.

An opt-in same-checkpoint diagnostic teacher-forces 32 identical tokens through fixed residency and adaptive residency. It found differences in target logits after promotions, although the first mismatching row still had the same greedy token. Disabling graph replay and adding an explicit stream fence did not resolve this. Uploaded expert bytes matched the GGUF source in the readback diagnostic. These results rule out those simple explanations; they do not establish a root cause.

Actual resident-row comparisons reproduced the mismatch outside the decoder with one captured expert. Input Q8 bytes and floating hidden values were identical; hidden Q8 quantization differed in two bytes (one quantized value and its block sum). HIP's `__fmul_rn` / `__fadd_rn` wrappers allowed contraction because `glm_q23.cu` was missing from the HIP `-ffp-contract=off` source list. Adding it fixes the captured case bit for bit. A synthetic boundary case (input float bits `0x3f8ccccd`, `0xbc54dced`) now checks this without distributing model weights.

After the fix, the 32-step all-CPU versus resident and adaptive-residency logit checks, sequential graph/direct check, retained verification prefixes, next-token logits and rollback all pass. The corrected 6 GiB and 8 GiB processes each repeat their 512-token output exactly, but their separate processes differ at output index 1. The same-checkpoint placement checks pass; the cross-process/prefill difference is not yet isolated. Do not claim that the two end-to-end runs have identical target state.

The 8 GiB diagnostic with a 256 MiB runtime reserve was stopped by the physical VRAM guard at 506 MiB free: its extra checkpoint consumes memory absent from normal throughput runs. No OOM occurred. The same diagnostic passes with the 6 GiB cache.

This is a correctness fix, not evidence that the 20 tok/s or long-context gates pass. The 8K/32K regression ladder and 128K, 90%-input/10%-output chat qualification remain separate follow-up gates once a throughput candidate qualifies.

## Reproduce

Use a quiet B550 and stop its demo first:

```sh
cd /home/syoyo/work/Strata
bash tools/glm_b550_60g.sh stop
```

Build the isolated source with the changes in this checkout:

```sh
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_PREFILL_MMQ=ON \
  -DCMAKE_BUILD_RPATH=/opt/rocm/lib
cmake --build build-hip --target strata-glm-decode glm_q23_parity \
  strata-glm-q2-kernel-bench -j8
```

The calibration driver uses the original checkout's experimental config and frozen `build-hip-glm-v11/single-2048.ids` by default. Pass `--source-root` and `--prompt` to use another location. Each run writes the config, manifest, raw output/log, resource samples, parsed results and repetition checks. Existing result directories are refused. Use the installed Strata Python environment:

```sh
PY=/home/syoyo/work/Strata/.venv-glm-hip/bin/python
$PY tools/glm_b550_calibrate.py --output runs/repro-screen \
  --tokens 128 --trials 3 --cases baseline cache7 cache8 mtp1 mtp2 mtp3
$PY tools/glm_b550_calibrate.py --output runs/repro-512 \
  --tokens 512 --trials 3 --cases cache8 \
  --env STRATA_GLM_TIER_RUNTIME_RESERVE_MIB=256
$PY tools/glm_b550_calibrate.py --output runs/repro-quality \
  --tokens 32 --trials 1 --cases baseline --fixed-tier --checks decode_graphs \
  --env STRATA_GLM_CHECK_TIER_ADAPT=1 \
  --env STRATA_GLM_CHECK_TIER_UPLOADS=1
```

The last command failed before the HIP contraction fix and passes with it. Checks are diagnostics and must not be used for throughput timing. `--fixed-tier` removes the presence-based adaptation variable; setting it to `0` does not disable it.

For future independent workloads, freeze the chat-template fixtures before fitting:

```sh
$PY tools/glm_b550_fixtures.py /mnt/disk01/models/glm53f-reap/GLM-5.3-Flash-REAP50-Q23-assembled.gguf runs/prompts
$PY tools/glm_b550_calibrate.py --output runs/repro-holdout-code \
  --prompt runs/prompts/holdout-code.ids --tokens 512 --trials 3 --cases baseline cache8
```

Run the simulator locally without loading weights or using a GPU:

```sh
python3 tools/glm_b550_fit.py
python3 tools/sim/glm_sim.py validate --records tools/sim/data/measured_b550.json \
  --params tools/sim/data/params_b550.json
python3 tools/sim/glm_sim.py plan --hw b550 --pack reap50_q23 \
  --quality-reference reap50_q23 --implemented-only --lossless \
  --target-decode 20 --prompt 2048 --generate 512 --threads 12 \
  --gpu-budget-mib 15792 --reserve-mib 512 --decode-cache-mib 8192 \
  --set runtime_vram.reap50_q23.runtime_headroom=256 \
  --params tools/sim/data/params_b550.json
python3 tools/test_sim.py
python3 tools/test_sim_b550.py
python3 tools/test_glm_low_memory_bench.py
```

## Next decision gates

1. Keep the new rounding and same-checkpoint residency checks as mandatory gates. The captured failure and 32-step gate pass after the HIP build fix; broaden this to held-out inputs and verification widths before promotion.
2. Capture real expert inputs and branch-preserving routing events separately from throughput. Widths 1, 2 and 4, worker counts, cache uploads, and three ordinary-decode holdouts are measured here. Next measure actual speculative execution unions and GPU resident timings; validate MTP acceptance on new domains without fitting those holdouts.
3. LRU and the tested protected policy failed the upload-cost screen. Evaluate any replacement with upload bytes, shared bandwidth, staging capacity and the engine's promotion delay included. Implement a policy only if its conservative gain clears the measurement noise and quality gate. The current immediate-admission replay is insufficient evidence.
4. Test target-verified MTP1 and existing canonical expert-group merging on the measured workloads. Depths 2 and 3 were slower in this screen; higher acceptance must be demonstrated before using optimistic profiles.
5. At 28 GB/s, 20 tok/s allows **at most 1.40 GB CPU reads per delivered token even if all other work were free**. The actual allowance is smaller. A solution needs substantially more reuse or fewer CPU reads per accepted token; simply streaming cold experts to the GPU does not remove their host-memory traffic. If exact methods cannot clear this bound on held-out workloads, report the gap rather than silently changing quantization or routing.
6. After a candidate clears correctness and 2K throughput, run 8K/32K regressions and the separate 128K chat-quality task under memory guards. Keep the demo stopped until explicitly relaunched.

Raw screening evidence is in [fixtures/glm_b550_sim_20261009](fixtures/glm_b550_sim_20261009/). Simulator input censuses, measured rows and fit metadata are in `tools/sim/data/`.

The long-context runner can now use an isolated decoder and leave the demo stopped. After the short-context throughput/quality gates pass, run the independent chat ladder (natural EOS, 90% input / 10% maximum output):

```sh
$PY tools/glm_long_chat_check.py --contexts 8192 32768 131072 \
  --config /home/syoyo/work/Strata/configs/glm53f-reap50-q23-b550-60g-9070xt-experimental.json \
  --decoder build-hip/strata-glm-decode --port 8081 \
  --leave-demo-stopped --output runs/long-chat-corrected
```

This uses a temporary loopback server, stops it after each case, and does not relaunch the web demo. Add `--prepare-only` to freeze requests without starting a server. The runner deliberately uses a smaller 3 GiB cache and a 2 GiB GPU reserve for the context ladder; its throughput must be reported separately from the 8 GiB-cache short-context screen. Review `answer.md`, expected facts, finish reason, truncation and resource telemetry; a completed HTTP request alone is not a quality pass.

Prepared long-chat fixtures are in the isolated B550 checkout at `runs/long-chat-prepared`: 7,363 + 819 tokens at 8K, 29,482 + 3,276 at 32K, and 117,955 + 13,107 at 128K. Output counts are maximum allowances; requests reserve template/context slack. These requests have **not been executed** in this experiment.

Validation: 88 Python tests pass; the HIP build, synthetic Q23 parity including the rounding boundary and subnormal scales, captured real-expert parity, wide-kernel parity, and same-state residency/rollback checks pass. CUDA/SYCL builds were not run. B550's demo service is inactive and ports 8080/8081 have no listeners at completion.
