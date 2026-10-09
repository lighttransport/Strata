# B550 fixed-affinity .06 tuning — 2026-10-09

**Result: 16.5 decode / 250.2 prefill tok/s median on the 8K document with probe 32, at affinity .06. Prefill exceeds 200; 18–20 decode is not achieved. Quality failures keep the new profiles experimental.**

Target: 18–20 decode and 200+ prefill tok/s on the 8K document Q&A, preserving scalar affinity .06 and the REAP50 Q23 weights. Work used the isolated B550 checkout; serving defaults and the stopped web demo are preserved.

Hardware: Ryzen 9 3950X, 62.693 GiB physical RAM, RX 9070 XT gfx1201 with 16,304 MiB reported VRAM, PCIe 4 x16, automatic GPU clocks, powersave CPU governor with boost. ROCm 7.14 development build, Linux 7.0.0-38-generic. RAM guard 60 GiB, zero swap, physical VRAM reserve 512 MiB. Model `GLM-5.3-Flash-REAP50-Q23-assembled.gguf`, SHA256 `c3b0cfb144556bd28e265d8255d48eeafce3d127aba5b9b5dd4e326322a34aa9`, 58.87 GiB, 144 experts, top 8, 42 MoE layers, gate/up Q2_K and down Q3_K. Source baseline `15b787b5` plus the recorded experimental changes.

## Screen methodology

Every screen is one real API document request: 7,358 prompt tokens, natural EOS honored, 768-token cap, context 8,192. The same synthetic dossier and questions from the quality-first sweep are used. Model pages are warmed within each service cgroup before readiness. Each setting gets a fresh server; loading is excluded from reported API timings, while request reset and cache preparation are included by the engine's DONE timing. No prefix reuse. These are one-run screens, not medians. Output workloads can differ because residency affects .06 routing; no same-token speedup is implied. Configurations, responses, logs, and memory telemetry are attached.

## Conventional tuning

| Screen | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| screen: baseline | 185.3 | 15.6 |
| cache-screen: recent256 | 186.5 | 15.8 |
| cache-screen: recent512 | 185.6 | 15.6 |
| cache-screen: sdma0 | 179.7 | 15.5 |
| hardware-screen: pp4096 | 234.9 | 15.4 |
| hardware-screen: tasks2 | 186.7 | 15.5 |
| hardware-screen: tasks1 | 186.3 | 15.1 |
| hardware-screen: kda4 | 196.6 | 15.5 |
| hardware-screen: threads8 | 186.4 | 15.1 |
| hardware-screen: threads16 | 186.3 | 15.3 |
| adapt-screen: adapt20 | 186.3 | 15.5 |
| adapt-screen: adapt10 | 186.5 | 15.3 |
| adapt-screen: adaptfast | 186.2 | 15.4 |
| learn-screen: baseline4k | 239.8 | 15.5 |
| learn-screen: unbiased4k | 239.7 | 15.2 |
| learn-screen: unbiasedfast4k | 240.8 | 15.2 |

The 4K batch initially failed the 1,024 MiB prefill scratch arena, not physical VRAM. Raising the opt-in arena to 2,048 MiB permits it. Ordinary CPU worker/task tuning, recent-route cache windows, SDMA disablement, and faster cache adaptation did not reach the decode target. Their exact variants are recorded in the JSON files alongside this report.

## Opt-in prototypes

`STRATA_GLM_TIER_LEARN_UNBIASED=1` learns cache popularity from the router before residency bonus. Actual execution keeps .06 affinity and the same weight probabilities. It uses the existing mailbox's desired-route channel and adds a second router call. The initial screen slowed decode slightly, so it is not selected by itself.

`STRATA_GLM_CACHE_PROBE=8..64` runs a short all-CPU canonical decode before cache admission, records the original router's experts, restores the full recurrent/KV state, and ranks the tier from those counts. The existing 256-pseudo-token prior remains; probe counts default to 1,024 pseudo-tokens, with opt-in `STRATA_GLM_CACHE_PROBE_WEIGHT=256..8192`. It does not change the user-visible initial logits or commit probe output. It skips probing if there are fewer than eight context slots available, and rejects unsupported placement/speculative modes.

Probe preparation is logged separately and included in the API's reported prompt time. It must not be added a second time when computing effective prefill. Diagnostic `STRATA_GLM_CHECK_CACHE_PROBE=1` compares every restored state buffer byte-for-byte and recomputes the first next-step logits. The assembled GGUF uses the expert-pack marker internally; the first compatibility guard mistakenly rejected it. Those rejected runs are retained outside successful results.

These are opt-in prototypes. The old lossless/routing fits explicitly reject probe and unbiased-learner records until separate calibration exists. No performance claim is made for other models, CUDA hardware, or 128K contexts.

## Further screens

| Screen | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| wide: 8K batch, no probe | 264.0 | 15.5 |
| wide: 8K batch, probe 32 | 228.5 | 16.5 |
| wide: 8K batch, probe 64 | 201.5 | 16.7 |
| weighted: probe 24, weight 8192, 4K batch | 217.4 | 16.6 |
| weighted: probe 32, weight 4096, 4K batch | 207.0 | 16.4 |
| prefetch: probe 64, prefetch 0 | 192.2 | 15.0 |
| prefetch: probe 64, prefetch 128 | 198.3 | 15.6 |
| prefetch: probe 64, prefetch 1024 | 203.1 | 16.8 |
| native arithmetic: 8K batch, no probe, document | 258.5 | 13.0 |
| native arithmetic: C++ review | 261.3 | 15.8 |

The native arithmetic C++ response repeated its reasoning until the 1,280-token cap and supplied no final answer or parser. That mode is slower on the document and is not selected. The prefetch-1024 probe answer has all eight correct facts, but one action conflates verifying remaining crates with the checksum agreement threshold. The probe-32 and no-probe 8K answers explicitly keep these metrics separate. Stronger probe weighting and the unbiased cache learner did not produce an 18 tok/s decode result.

The probe-32 screen spent 4,401 ms probing plus 1,258 ms admitting the cache, already included in its prompt timing. Its counter records 456,595,668,992 logical CPU expert bytes and 16,189 ms of CPU layer flow over the probe plus response; these counters include shadow-probe work and cannot be treated as generation-only hardware bandwidth. DDR expert bandwidth measured earlier was roughly 28–29 GB/s. CPU expert work remains material even with GPU residency. These observations support a bandwidth/CPU-work bottleneck; they are not a complete critical-path attribution.

The 8K batch needs a 4,096 MiB prefill scratch arena. It stays inside the memory guards on this fixture; this is not a recommendation for arbitrary model or context sizes. The report retains failed scratch/probe compatibility screens separately from successful timings. No server-default or weight-quantization change is selected.

The 10 GiB cache request admitted only 933 slots / 8,105 MiB without probing and 932 slots / 8,097 MiB with probe 24. Physical GPU free-memory checks stop admission while preserving reserve and runtime headroom; the requested size is not allocated size. Screens were 265.4 / 15.5 tok/s without probing and 231.8 / 16.4 tok/s with probe 24 (prefill / decode). There is no evidence of extra cache capacity on this hardware with these dense weights and guards.

## Build and reproduction

The final engine source is the baseline plus [engine-from-15b787b5.patch](engine-from-15b787b5.patch), including earlier opt-in routing/check tooling and the HIP canonical-rounding fix. Apply only in a clean isolated checkout, or use the current working tree. The source and binary hashes, dependency caveat and backend coverage are in [build-validation.json](build-validation.json). Final HIP and CUDA compile logs are attached. The CUDA check uses CUDA 13.2, architecture 120 and an available reference GGML revision (`e45ca254`), not the pinned dependency; no local GPU or model test was run. The separate SYCL engine has no GLM target. This is experimental build evidence, not release certification across all backends.

In the idle B550 isolated checkout, build the configured HIP target and run the exact validation sequentially:

```sh
cmake --build build-hip --target strata-glm-decode -j 2
PY=/home/syoyo/work/Strata/.venv-glm-hip/bin/python
$PY tools/glm_b550_quality_sweep.py \
  --config configs/experimental/glm53f-reap50-q23-b550-quality-first.json \
  --decoder build-hip/strata-glm-decode \
  --output runs/repro-aff006-final \
  --affinities .06 --trials 3 --tasks document cpp-review \
  --variants bench/results/2026-10-09-b550-glm-aff006-tuning/final-variants.json
```

Use a fresh output directory. This runs a temporary loopback server and enforces 60 GiB RAM, zero swap and the 512 MiB physical GPU reserve. It leaves the web demo stopped. `run.json` records the binary/runner hashes; each case's `config.json` is authoritative for overrides. The inner `document/fixture.json` describes the original long-chat builder fixture; the root `document.fixture.json` and `document.request.json` describe the shortened Q&A request actually measured. No 32K/128K throughput or quality claim follows from this 8K experiment.

The remaining decode gap is measurable: 16.5 tok/s means about 60.6 ms/token; 18 requires 55.6 ms and 20 requires 50.0 ms. Another 5–11 ms/token must be removed. More requested cache cannot do this under the measured GPU guard, and the tested CPU task/thread/prefetch settings did not do it. A future exact implementation needs faster CPU expert kernels or greater reuse per host read; the current routing fit is not evidence that either has been achieved. This experiment does not raise scalar affinity or alter quantization to fill that gap.

## Repeated validation and decision

Final HIP binary SHA256: `8c2c754264f5c327997826cbe468b953888c3cbb9d8c165833074a1d8e1df98f`. Each profile ran three document requests then three C++ requests in one fresh server, resetting state and the expert tier per request. Later requests retain buffers; the first-request and three-request median prefill figures are deliberately shown separately. Loading is excluded, cache/probe preparation is included, and prefix reuse remains zero.

| Profile / task | First prefill | Median prefill | Median decode | Decode range | Natural stops |
|---|---:|---:|---:|---:|---:|
| prefill8k / document | 265.3 | 294.2 | 15.5 | 15.4–15.5 | 3/3 |
| prefill8k / cpp-review | 261.8 | 261.9 | 16.1 | 15.9–16.6 | 0/3 |
| prefill8k-probe32 / document | 228.7 | 250.2 | 16.5 | 16.5–16.6 | 3/3 |
| prefill8k-probe32 / cpp-review | 204.8 | 206.4 | 16.2 | 16.2–16.4 | 2/3 |

All six document answers contain the eight correct factual answers. Two of three batch-only answers and all three probe answers still partly conflate crate completion and checksum agreement in their action lists; one probe answer simultaneously says to keep the metrics separate. This is a quality limitation, not an unconditional document-task pass.

Batch-only C++ reviews hit the cap in all three trials and provide no parser. Probe-32 reviews identify the five core defects; trial 0 supplies a correct parser passing 5,020 cases, trial 1 supplies an incorrect overflow check failing 678/5,020 cases, and trial 2 supplies a correct parser passing 5,020 but reaches the cap during a trailing Registry comment, leaving its code fence unclosed. The function itself is unchanged and complete, so it is tested separately from response completion. Wrong numeric triggering examples remain, including claiming `12x` leaves zero rather than 12. The [manual audit](quality-audit.json) and [parser checks](parser-checks/summary.json) retain these failures.

**The 18–20 decode target is not met at .06.** The 200+ prefill target is met by both larger-batch profiles, including preparation time. The [batch-only profile](../../../configs/experimental/glm53f-reap50-q23-b550-aff006-prefill8k.json) and [probe-32 profile](../../../configs/experimental/glm53f-reap50-q23-b550-aff006-document8k-probe32.json) are opt-in experiments, not replacements for the earlier quality-first profile. In particular, neither is qualified for general coding. The original serving configuration, weights, clocks and stopped demo remain unchanged.

Both profiles stayed within guards: zero swap, zero cgroup max/OOM events, peak physical GPU use 15,656 MiB with at least 648 MiB free. Detailed per-profile RAM charges and GPU peaks are in [validation-memory.json](validation-memory.json). The probe restoration diagnostic passed byte-identical recurrent/KV state and next-step logits; this does not establish byte-identical biased generation or general quality equivalence.

Re-run parser checks from the repository root, after manually reviewing supplied code:

```sh
python3 bench/results/2026-10-09-b550-glm-aff006-tuning/check-parsers.py \
  runs/repro-aff006-final runs/repro-aff006-parser-checks
```

The harness tests the original supplied function: empty input, ASCII validation, overflow, UINT64_MAX, leading zeros, embedded NUL, non-ASCII, long strings, unchanged output on failure and 5,000 fixed-seed generated inputs. These are 5,020 function inputs, not independent model prompts. The checks use one low-priority CPU compiler and no local GPU/model jobs. New cache-learning/probe records are excluded from the existing routing fit; no new simulator fit is claimed for these profiles.

The last staging screen keeps the original 2K prefill batch and changes only staging workers from 4 to 8. It measured **182.0 prefill / 15.4 decode** on the document and **203.4 / 17.9** on the C++ request; the C++ answer repeats its reasoning until the 1,280-token cap and supplies no final answer. This does not improve the document target and is not selected. Both cases stopped without swap/OOM. Their raw responses are retained in `raw/aff006-staging-screen-20261009`.

Validation and cleanup: commit verification ran 91 Python simulator/low-memory tests against an export of the staged tree (90 passed, one trace test skipped because the export has no local traces); final syntax and source/data whitespace checks pass, both final engine builds pass, and the representative probe state/logit parity check passes. The B550 temporary quality service and demo service are inactive, no decoder process remains, and ports 8080/8081 have no listener. Final experimental sources, profiles and this report are synced to the isolated B550 checkout; no serving-default change or release promotion is made because the decode and quality gates remain open.
