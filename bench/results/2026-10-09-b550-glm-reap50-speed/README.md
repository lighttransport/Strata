# B550 REAP50 decode speed experiments — 2026-10-09

Routing bias cleared 20 tok/s in short speed screens. It is a lossy experiment, not a qualified serving setting. The web demo remains stopped and the serving checkout is unchanged.

## Machine and model

Ryzen 9 3950X (16 cores / 32 threads), 62.693 GiB RAM, RX 9070 XT gfx1201 with 16,304 MiB reported VRAM, PCIe 4 x16. GPU clocks automatic; CPU powersave with boost. ROCm 7.14 development build, source baseline `15b787b5` plus the experimental changes recorded in per-run manifests. Experiments use `/home/syoyo/work/Strata-b550-calibration-15b787b5`, not the serving checkout. Linux kernel 7.0.0-38-generic; binary hashes are recorded in manifests; SSD model and GPU power limit were not separately recorded.

Model: `GLM-5.3-Flash-REAP50-Q23-assembled.gguf`, 58.87 GiB, main gate/up Q2_K and down Q3_K, 144 experts, top 8, 42 MoE layers. SHA256 `c3b0cfb144556bd28e265d8255d48eeafce3d127aba5b9b5dd4e326322a34aa9`. This unchanged REAP50 model is the reference, not the original unpruned model.

## Throughput

All rows use 2,048 input tokens and three capped output repetitions in one process, retaining adaptive residency between repetitions. The prompt is the existing C++ test-generation fixture; its path and hash are in manifests. Prefill is warm: the engine performs an untimed prefill before the measured repetitions. Loading and cold SSD reads are excluded. Ordinary decode times N−1 steps after the initial token; the table lists the emitted cap N. No MTP reasoning budget or sampling changes were made. CPU workers 12, context 4,096, GPU budget 15,792 MiB, requested expert cache 8,192 MiB, physical reserve 512 MiB. Lookup reserves additional verification history. Resource guards enforce 60 GiB RAM and zero swap.

| Case | Output cap | Median decode tok/s | Decode range | Warm prefill tok/s |
|---|---:|---:|---:|---:|
| cache8 | 128 | 10.97 | 10.63–11.03 | 231.16 |
| aff10 | 128 | 18.59 | 17.62–20.86 | 230.63 |
| aff20 | 128 | 24.85 | 23.49–25.34 | 230.15 |
| aff30 | 128 | 27.14 | 26.86–27.31 | 230.97 |
| aff12 | 128 | 20.36 | 19.06–20.99 | 230.63 |
| aff15 | 128 | 23.00 | 21.08–23.06 | 231.17 |
| mtp1 | 128 | 10.48 | 9.79–10.50 | 231.44 |
| mtp2 | 128 | 8.11 | 7.55–8.20 | 230.61 |
| mtp3 | 128 | 7.25 | 6.91–7.34 | 230.19 |
| lookup3 | 512 | 11.00 | 11.00–11.00 | 228.59 |
| lookup7 | 512 | 10.65 | 10.64–10.65 | 230.19 |

`affNN` adds NN/100 to resident-expert selection scores, while retaining original probabilities for selected expert weights. Positive affinity also selects the existing device combine path. Thus this comparison includes routing changes and a combine change. The same-day cache8 baseline was 10.97 tok/s. Positive-affinity cases also set 256 MiB tier runtime headroom; the zero-bias cache8 screen retained the original larger runtime reserve and admitted less cache. Its ratio is therefore not a matched-residency bias-only comparison. Per-run configurations and actual admission counters are attached. Cache and routes adapt across repetitions; biased outputs differ between repetitions, so these are different output workloads. Do not interpret their ratios as an exact same-token kernel speedup.

Merged MTP (`STRATA_GLM_MOE_MERGE_GROUPS=early`) and target-verified cached suffix lookup did not improve speed. Lookup is a CLI-only opt-in prototype; the web server still rejects that combination. Lookup speed output repeats exactly. Its original 64-token parity check proposed no suffixes and is insufficient to qualify speculative acceptance.

## Quality limits

Separate preliminary 64-row teacher-forced comparisons measured next-token top-1 agreement of 51/64 at affinity .10, 45/64 at .12, 54/64 at .15 and 41/64 at .20. These used different prefills/cache inventories, so their nonmonotonic scores are not a paired affinity curve. Mean KL was .706, .422, .666 and .952 respectively. A zero-affinity control had exact logits (8/8 rows, KL 0). These metrics do not certify semantic quality.

Decoded capped continuations include repetition and garbled text, including the unbiased baseline. They are not complete chat evaluations. Paired grid, layer-preservation, 512-token validation and document chat results are reported below. No 128K quality claim is made here.

The original lossless simulator underpredicts positive-affinity speed. Resident-call counters rise from 40.3% at zero bias to 75.1%, 78.2%, 83.2%, 87.3% and 92.3% at .10, .12, .15, .20 and .30. This supports reduced CPU weight traffic as the main observed gain; it does not isolate the device-combine contribution. These runs remain excluded from ordinary throughput fitting.

## Layer-preserving screens

Leaving layers 11, 12 and 44 unbiased measured 21.59 tok/s at affinity .15 and 22.75 at .20 (2K + 128, three repetitions). These layer choices are hypotheses, not a measured sensitivity ranking. The profiles are in `configs/experimental/`.

A paired 64-row grid leaving seven layers unbiased (3, 4, 5, 11, 12, 43, 44) measured:

| Affinity | Mean KL | Next-token top-1 agreement |
|---|---:|---:|
| 0 | approximately 0; exact logits | 64/64 |
| .12 | .627 | 51/64 |
| .15 | .591 | 51/64 |
| .20 | .728 | 45/64 |

Each row uses the same teacher-forced continuation and fixed resident inventory in one process. This is conditional on that process's prefill state, not an end-to-end chat equivalence check. The three-layer preliminary quality run used an older binary and did not execute the requested grid; it is excluded from paired evidence. The final rerun is attached below.

## Separate routing simulation

`routing-sim-params.json` fits only the affinity response constant (.168) and an aggregate positive-affinity resident GPU cost scale (.70), using the .10 and .20 speed screens. Every CPU bandwidth and ordinary parameter stays fixed. This scale is a fitted cost term, not a measured increase in GPU bandwidth or an isolated combine speedup. Held-out .12 and .15 predictions are 20.23 and 22.86 tok/s, errors −0.66% and −0.59%. The .30 prediction is 23.89, error −11.97%, outside the 10% tolerance. It is not qualified for aggressive bias, layer overrides, other prompt domains, longer output or quality.

Reproduce the fit with `tools/glm_b550_affinity_fit.py`, passing the five scalar-affinity result files, `--fit aff10.result aff20.result`, and a fresh `--output` directory. Validation rows and training selection are included in the JSON artifacts. A regression check verifies these two calibration terms cannot change zero-affinity predictions.

## Reproduce

Build HIP for gfx1201 in the isolated checkout, with the same Q23 canonical-rounding fix described in [the calibration report](../../../docs/GLM_B550_SIM.md). Stop other model jobs first. The driver refuses to clear pages while a decoder is active.

```sh
PY=/home/syoyo/work/Strata/.venv-glm-hip/bin/python
$PY tools/glm_b550_calibrate.py --output runs/repro-speed --tokens 128 --trials 3 \
  --cases cache8 aff10 aff12 aff15 aff20
$PY tools/glm_b550_calibrate.py --output runs/repro-paired --tokens 32 --trials 1 \
  --cases aff20 --env STRATA_GLM_CHECK_AFFINITY=64 \
  --env STRATA_GLM_AFFINITY_QUALITY_GRID=0,0.05,0.10,0.12,0.15,0.20
$PY tools/glm_b550_calibrate.py --output runs/repro-long --tokens 512 --trials 3 \
  --cases cache8 aff12 aff15
```

Raw JSON, manifests, timing logs, token IDs and decoded first repetitions are under `raw/runs/`. Quality-check runs must not be used as throughput calibration. No release default changed. HIP and CUDA builds pass; CUDA was compile-only, using an existing local GGML dependency (see `build-provenance.json`). No local GPU performance jobs or SYCL tests were run. Python checks: 90 passed.

## Paired quality grids

These are 64 teacher-forced rows, one frozen resident inventory per profile and a common prefill state within each process. Different profiles have different states and must not be ranked as a paired layer-sensitivity experiment. Zero affinity restores exact logits in all three grids.

| Profile | Affinity | Mean KL | Top-1 agreement |
|---|---:|---:|---:|
| All layers | 0 | 0.000 | 64/64 |
| All layers | 0.05 | 0.430 | 57/64 |
| All layers | 0.1 | 0.591 | 54/64 |
| All layers | 0.12 | 0.573 | 54/64 |
| All layers | 0.15 | 0.414 | 54/64 |
| All layers | 0.2 | 0.543 | 56/64 |
| Preserve 3 | 0 | 0.000 | 64/64 |
| Preserve 3 | 0.12 | 0.736 | 52/64 |
| Preserve 3 | 0.15 | 0.625 | 53/64 |
| Preserve 3 | 0.2 | 0.905 | 53/64 |
| Preserve 7 | 0 | 0.000 | 64/64 |
| Preserve 7 | 0.12 | 0.627 | 51/64 |
| Preserve 7 | 0.15 | 0.591 | 51/64 |
| Preserve 7 | 0.2 | 0.728 | 45/64 |

Preserving seven layers at affinity .20 measured **20.37 decode / 229.72 warm prefill tok/s**, median of three 128-output runs. It still changes 19/64 next-token predictions in its paired test. The three-layer grid has now been rerun with the final binary; only the final directory is attached.

## Sustained 512-token speed

2,048 input + 512 emitted output tokens, three repetitions; all passed 60 GiB/no-swap and VRAM guards. Zero-bias runtime reserve admitted less cache, as noted above.

| Case | Median decode | Decode range | Warm prefill | Min free VRAM MiB |
|---|---:|---:|---:|---:|
| cache8 | 11.03 | 11.02–11.07 | 229.93 | 1162 |
| aff12 | 21.27 | 20.97–21.44 | 229.96 | 652 |
| aff15 | 22.64 | 22.63–22.75 | 231.08 | 652 |

The short-screen routing fit was not changed for these runs. It predicts the 512-token .12 and .15 measurements with errors -4.89%, 0.97%. This is a longer-output holdout on the same prompt, not a new-domain validation.

Decoded longer code continuations are repetitive and fail to produce a useful answer, including the unbiased baseline. The faster settings are not ready for serving.

## Fresh 8K document chat

One request per setting, the same 7,363-token dossier prompt and an 819-token output cap (90% input / 10% output budget). No prefix reuse (`cached_tokens=0`). These are fresh server-request timings, not the warm 2K benchmark or three-run medians. Exact counts matched the prepared fixture.

| Setting | Prefill tok/s | Decode tok/s | Output | Finish |
|---|---:|---:|---:|---|
| Zero bias | 176.7 | 11.3 | 819 | length |
| Affinity .12 | 182.5 | 20.0 | 819 | length |

Both answers are coherent and include the eight factual answers: superseding date, funding and allocations, 525 unverified crates / 37.5% completion, rollback condition and owner, lead and mitigation, privacy requirement, version phrase and absent external auditor. Both incorrectly compare completion percentage with checksum accuracy, which are different quantities. Both finish at the cap; the biased answer reaches only the final action-list heading while the reference starts several actions. Neither is a fully completed answer. No new factual error or repetitive loop was observed in the .12 answer, but one task is insufficient to establish general preservation of quality.

The .12 API rate is rounded to 20.0; 819 / 40.8572 seconds is approximately 20.045 tok/s. All output tokens count. Both temporary servers were stopped afterward; the serving demo remains stopped. No 32K/128K qualification, original-model comparison or C++ correctness pass is claimed for these routing variants.

To repeat the document comparison, set scalar affinity to 0 or .12 in a copy of the configuration, use the isolated decoder, and run:

```sh
$PY tools/glm_long_chat_check.py --contexts 8192 --output runs/repro-chat \
  --config configs/your-experimental-config.json --decoder build-hip/strata-glm-decode \
  --cache-mib 8192 --reserve-mib 512 --port 8081 --leave-demo-stopped
```

The configuration also needs GPU budget 15,792 MiB and tier runtime reserve 256 MiB. Run settings sequentially, with a new output directory each time. Requests, prompt, fixtures, configuration, complete API response, answers and memory telemetry are attached under the two chat directories.
