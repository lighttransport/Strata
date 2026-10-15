# GLM-5.3-Flash with Strata: release notes, measurements, weights and setup (October 2026)

This is the reference page for the GLM-5.3-Flash work in this fork: what runs on which machine, how fast,
with which quality evidence, where the weights are, and how to set it up. Speeds are medians or listed
trials of timed runs on the machines named; a token is about three quarters of a word. Quality claims are
only the ones that were measured; "not gated" means no quality run exists for that configuration.

Attribution: this work studied [Project Maya](https://github.com/mw00/project-maya) (peasantsmith, MIT License),
a separate GLM-5.3-Flash engine that also started from Strata. The ideas taken from it are listed in
[the comparison below](#how-this-differs-from-project-maya); no Maya code or weights are used.

## The machines

| name | CPU, RAM | GPU(s) | links |
|---|---|---|---|
| TR16 | Threadripper 1950X (16 cores, two NUMA nodes), 128 GB DDR4-3200 | RTX 5060 Ti 16 GB (PCIe 3 x8) and RX 9070 XT 16 GB (PCIe 3 x16) | [TR16 runner plan](GLM_TR16_RUNNER_PLAN.md), [single GPU](GLM_TR16_OPTIMIZATION.md), [two GPUs](MULTI_GPU.md) |
| B550 | Ryzen 9 3950X (16 cores, one node), 64 GB DDR4-2133 | RX 9070 XT 16 GB (ROCm) | [B550 REAP-50](GLM_B550_REAP50.md), [B550 simulator](GLM_B550_SIM.md) |
| zen5 | Ryzen 9 3950X, 64 GB DDR4-2666 (ASUS TUF B550-PLUS) | RTX 3070 8 GB (CUDA 12.4, SM86) | measurement records `docs/benchmarks/glm_rtx3070_*_20261011.json`; presets and engine switches live on the `glm53f` branch of that machine (checkpoint `6e63bc20`), not yet merged here |
| V100 server | 2x Xeon Gold 6240, 160 GiB | 2x Tesla V100 PCIe 32 GB | [V100](GLM53_V100.md), [V100 decode](GLM53_V100_DECODE_20261003.md) |

## Measured speed and quality

Decode is single-stream tokens per second; prefill is the prompt rate at the stated length. "Q23" is the
fork's expert requantization (Q2_K gate/up, Q3_K down) assembled into the Unsloth UD-Q2_K_XL base;
"REAP-224" keeps 224 of 288 routed experts per layer, "REAP-50" keeps 144.

| machine | configuration | decode tok/s | prefill tok/s | quality evidence |
|---|---|---:|---:|---|
| TR16, RTX 5060 Ti alone | REAP-224 Q23, tuned profile (`configs/experimental/glm53f-q23-reap224-tr16-tuned.json`) | 29.0 (2K), 31.6 (8K) | 168 (2K), 496 (8K) | HumanEval 152/164 (a later rerun scored 150/164); corpus perplexity +4.9%, KL 0.064, top-1 -1.9 pp vs unpruned Q23 |
| TR16, RX 9070 XT alone | REAP-224 Q23, affinity 0.14 | 34.1 (8K), 35.0 (8K chat) | 561 (8K) | same profile family; no HumanEval or corpus run on this card |
| TR16, two GPUs (decode) | REAP-224 Q23, CUDA primary + HIP expert worker, affinity 0.14, fused publish, tier inflight 4 (`configs/experimental/glm53f-reap224-tr16-two-gpu-decode.json`) | 43.6 / 41.9 / 43.8 (8K, 512 tokens), 44.1 / 41.3 / 40.2 (1K, 256) | 465 (8K, this orientation) | corpus gate `tr16-35` passed (KL 0.210, top-1 0.853, perplexity 5.08 against limits 0.253 / 0.839 / 5.21); HumanEval 164 in progress |
| TR16, two GPUs (prefill) | REAP-224 Q23, HIP primary + CUDA expert worker, fraction 0.79 (`configs/experimental/glm53f-reap224-tr16-two-gpu-prefill.json`) | about 37 (short samples) | 870-875 (8K) | lossy F16 attention and F16 partial return; not gated separately |
| TR16, two GPUs, fastest | as the decode row with affinity 0.30 | 49.0 (1K), 46.1 (8K) | 510 (8K) | fails the corpus gate (KL 0.310, top-1 0.817); not a release configuration |
| TR16, RTX 5060 Ti, full model | full Q23 (288 experts), affinity 0.08, MTP depth 1 | 23.1 (2K), 23.0 (8K) | 131 (2K), 428 (8K) | HumanEval 153/164 |
| B550, RX 9070 XT, 16 GB VRAM | REAP-50 Q23, default service, 60 GiB RAM cap | 10.2 (2K) | 230 (2K) | vs UD-Q2_K_XL: perplexity 11.12 vs 8.57, KL 0.587, top-1 69.8%; no HumanEval |
| B550, RX 9070 XT, 16 GB VRAM | REAP-50 Q23, affinity 0.06 | 16.5 (2K and a 7.4K document) | 229-250 | passes the fork's C++ coding review task; higher affinity did not |
| zen5, RTX 3070 8 GB, 64 GB | REAP-50 Q23, strict routing, bounded dense prefill cache (`STRATA_GLM_PREFILL_DENSE_MIB=4096`), 2K prefix | 10.2 | 207 (2K) | bit-exact against the reference logits on the frozen prefix; HumanEval 0-29 subset 30/30 |
| zen5, RTX 3070 8 GB, 64 GB | REAP-50 Q23, approximate decode preset (route share < 0.07 dropped, device combine, deferred cold routes) | 15.6 (2K) | 206 | matched perplexity +4.5%, KL 0.058 vs strict; HumanEval 0-29 subset 30/30 |
| zen5, RTX 3070 8 GB, 64 GB (DDR4-2666) | REAP-50 Q23, shared-prefill presets with registered expert uploads, ordered KDA, coalesced dequantization | 19.5 (2K), 16.9 (4K), 17.6 (8K) | 464 (2K), 506 (4K), 488 (8K) | matched perplexity +5.7 / +8.9 / +7.6% (gate: at most +10%), HumanEval 0-29 subset passed at each context; full HumanEval not run |
| 32 GB RAM, 8 GB VRAM budget | full experts ([GLM_32G_8G.md](GLM_32G_8G.md)) | 0.3 | 0.6 | not usable |
| V100 server, two V100s | full UD-Q2_K_XL, 4K prompt | 28.9 greedy, 29.1 MTP depth 1 | 790 (16K) | logit parity, memcheck, 400,532 C++ oracle cases; no HumanEval |
| V100 server, 16 GiB per card cap | same | 18.1 (one GPU), 21.9 (two) | | |

The RTX 3070 rows come from the four `docs/benchmarks/glm_rtx3070_*_20261011.json` records (medians of three
trials, 128 generated tokens, 56 GiB cgroup, no swap, 512 MiB VRAM reserve); the presets they name
(`configs/glm53f-reap50-q23-64g-rtx3070-*.json`) and their engine switches are on that machine's branch and are
not in this tree yet, so they are evidence, not something this build can run.

Notes on the two-GPU rows. The decode and prefill configurations have opposite primaries and cannot run in one
process at these speeds; a server picks one. The decode row's quality gate is the one agreed for this work
(perplexity within +6%, KL within +0.08, top-1 within 3 pp of the unpruned Q23 run, plus HumanEval at or above
152/164); affinity 0.20 and 0.30 fail it. Removing every lossy setting from the two-GPU stack gives 20 tok/s,
so the speed comes from gate-tested approximations: routing affinity 0.14, soft draft acceptance (margin 6,
top-k 8), no adaptive worker-cache refresh and the fused worker return. The full measurement history, the
profiles and what did not work are in [GLM_TR16_RUNNER_PLAN.md](GLM_TR16_RUNNER_PLAN.md) (sections 12-15) and
the simulator that ranked the work in [GLM_RUNNER_SIM.md](GLM_RUNNER_SIM.md).

## Weights

Hugging Face: [`lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts`](https://huggingface.co/lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts).
The model card there has the full file table, the conversion recipe and the quality numbers.

| file | size | what it is |
|---|---:|---|
| `GLM-5.3-Flash-Q23-assembled.gguf` | 120.9 GB | the full model, single file: Unsloth UD-Q2_K_XL dense/attention/router weights with the routed experts requantized to Q2_K (gate/up) and Q3_K (down). Needs about 128 GB of RAM. |
| `reap224/GLM-5.3-Flash-Q23-REAP224-assembled.gguf` | 95.5 GB | 224 of 288 experts per layer, sliced from the file above (REAP-style saliency ranking with the fork's own calibration, no further requantization). The TR16 release rows use it. |
| `reap224/coverage-keep224.json`, `plan-keep224.txt`, `plan-keep224.report.json` | small | expert prior for the GPU-resident tier (`STRATA_GLM_EXPERT_PRIOR`), the kept experts, the per-layer saliency report |
| `reap50/GLM-5.3-Flash-REAP50-Q23-assembled.gguf` | 63.2 GB | REAP-50 (144 experts) with Q23 routed experts in the Q4_K_M base; the B550 rows use it. Fits 64 GB of RAM. |
| `reap50/reap50-coverage.json`, `reap50-q23-assembled.provenance.json` | small | expert prior and provenance |
| `experts-q23.gguf`, `experts-q22.gguf` | 105.5 / 95.9 GB | expert sidecars for the four original Unsloth shards (`--expert-pack`); the loader fingerprints the shards by size, header and modification time, so a fresh shard download needs the sidecar regenerated or the timestamps restored ([GLM_RELEASE_Q2Q3.md](GLM_RELEASE_Q2Q3.md)) |
| `calibration/coverage-q23-tr16.json`, `*.quality.json` | small | expert prior for the sidecars; per-projection reconstruction error |

SHA-256 of the large files (first 16 hex digits): full `8e280fd86714b606`, REAP-224 `3ef5726a`, REAP-50
`c3b0cfb144556bd2`, experts-q22 `53adf0f48d2723fe`; the model card lists the full digests.

## Setup

For a machine that is not one of the measured ones, the step-by-step bring-up, tuning and benchmarking
protocol is in [GLM53F_NEW_HARDWARE.md](GLM53F_NEW_HARDWARE.md); the commands below are the short form.

### Build

CUDA (NVIDIA; set the architecture of your card, 120 is the RTX 5060 Ti, 70 the V100):

```sh
cmake -S . -B build-glm -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_BUILD_TESTS=ON
cmake --build build-glm -j 8 --target strata-glm-decode strata-glm-gpu-worker strata-model-inspect
```

HIP (AMD, ROCm 7 or later; gfx1201 is the RX 9070 XT):

```sh
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DSTRATA_BUILD_TESTS=ON
cmake --build build-hip -j 8 --target strata-glm-decode strata-glm-gpu-worker
```

Tests: `ctest --test-dir build-glm -L glm` (and the same for `build-hip`); the labels are described in
[GLM_TESTS.md](GLM_TESTS.md). Python helpers need `numpy`, `regex` and `jinja2`.

### Download

```sh
pip install -U huggingface_hub
# 128 GB machine: the full model
hf download lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts --include 'GLM-5.3-Flash-Q23-assembled.gguf' --local-dir models
# TR16 release rows: REAP-224
hf download lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts --include 'reap224/*' --local-dir models
# 64 GB machine: REAP-50
hf download lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts --include 'reap50/*' --local-dir models
build-glm/strata-model-inspect models/reap224/GLM-5.3-Flash-Q23-REAP224-assembled.gguf
```

### Run on one GPU

The service and the one-shot decoder take a profile JSON. The single-GPU TR16 profile is
`configs/experimental/glm53f-q23-reap224-tr16-tuned.json`; set `model` and `STRATA_GLM_EXPERT_PRIOR` to your
paths. For the service:

```sh
python3 -m serve.server --engine glm --config configs/experimental/glm53f-q23-reap224-tr16-tuned.json
```

For a timed run from the shell, the one-shot decoder with the same settings (the exact argument list is in
[GLM_RELEASE_Q2Q3.md](GLM_RELEASE_Q2Q3.md#configuration-a-128-gb-ram-16-gb-nvidia-gpu-tr16)). On a machine with two NUMA nodes run it under
`numactl --interleave=all`; keep swap off for the process (the measurements use a systemd scope with
`MemorySwapMax=0`) and leave 1-2 GB of VRAM free for the desktop.

B550-class machines (64 GB, one RX 9070 XT): [GLM_B550_REAP50.md](GLM_B550_REAP50.md) has the profile,
the 60 GiB cgroup and the `tools/glm_b550_60g.sh start` launcher. The V100 server: the `build-glm-v100`
recipe and profile in [GLM53_V100.md](GLM53_V100.md).

### Run on two GPUs (TR16)

One process is the primary (it runs the model) and a second process on the other card is the expert
worker; they share a memory-mapped exchange buffer and a socket. `tools/glm_mixed_run.py` starts both:

```sh
# decode orientation: CUDA primary, HIP worker (the 43 tok/s row)
python3 tools/glm_mixed_run.py --combine device \
  --primary build-glm/strata-glm-decode --worker build-hip/strata-glm-gpu-worker \
  --model models/reap224/GLM-5.3-Flash-Q23-REAP224-assembled.gguf \
  --worker-cache-mib 14336 --worker-gpu-budget-mib 15360 -- <decoder arguments>
```

The environment for the two release orientations is the `env` block of
`configs/experimental/glm53f-reap224-tr16-two-gpu-decode.json` and `...-prefill.json`;
`tools/glm_mixed_bench.py --config <that file> --cases cuda-hip` (or `hip-cuda` with
`--fractions 0.79` for the prefill orientation) runs the measured benchmark with the right arguments and
writes `results.json` with the speeds and stream-health checks. For serving, `tools/glm_mixed_decode.py`
is a drop-in decoder for `serve.server` that launches the pair; it reads `STRATA_GLM_MIXED_PRIMARY` and
`STRATA_GLM_MIXED_WORKER` for the two binaries.

For another pair of cards, `tools/glm_mixed_fraction_calibrate.py` finds the prefill expert split
(whole 16-expert groups, hill-climbing on measured runs), and
`tools/glm_runner_sim.py` ([GLM_RUNNER_SIM.md](GLM_RUNNER_SIM.md)) forecasts decode and prefill from
a calibrated kernel table.

### Quality checks

`tools/glm_tr16_quality.py --quality-gate tr16-35` runs the 35-track corpus gate against a baseline run;
`tools/glm_q2_humaneval.py` runs the 164-task HumanEval with isolated tests (about two hours on TR16);
`tools/glm_mixed_bench.py` reports distinct 4-gram fractions and repeat-guard counts so loops show up.

## How this differs from Project Maya

[Project Maya](https://github.com/mw00/project-maya) (studied at v1.1.8, commit `5c970a86`) and this fork both
start from Strata and both run GLM-5.3-Flash with experts outside VRAM, but they answer different questions.

| | Project Maya | this fork |
|---|---|---|
| target machine | one to sixteen NVIDIA GPUs (AMD experimental), mostly 24-48 GB cards; its published results lead with 4x RTX 4090 (118 tok/s) and 2x RTX 3090 (53 tok/s decode, 885 prefill, Threadripper 3970X, 128 GB) | commodity desktops: one 8-16 GB card, or two 16 GB cards of different vendors (RTX 5060 Ti + RX 9070 XT), DDR4 desktops with 64-128 GB; plus an old V100 server |
| weights | its own quantizations of the FP8 release (Maya-S, S24, M, L: 94.7-156.3 GB), made with statistics from the FP8 model | the Unsloth UD-Q2_K_XL base with the routed experts requantized to Q2_K/Q3_K ("Q23"), optionally REAP-pruned to 224 or 144 experts per layer (95.5 / 63.2 GB) |
| where experts run | a VRAM / RAM / NVMe tier hierarchy: the hottest experts in VRAM, the next in RAM, the rest on the SSD, moved as the conversation goes; the engine pins most free RAM for its tier, so the model need not fit in RAM | routed experts execute on the CPU directly from RAM (a NUMA-aware expert pool), with a GPU-resident tier for the hottest experts; the model must fit in RAM |
| several GPUs | layer-range split of the model across GPUs | expert-parallel: one primary GPU runs the model, a worker process on the second GPU (any vendor) runs a share of the experts over a shared-memory exchange; a layer split was built and measured slower on this hardware (12.5-17.6 against 36 tok/s, plan section 11) and removed |
| drafting | the model's MTP block, optionally a separate DFlash2 drafter | the model's MTP block, with draft experts on the GPU and bounded soft acceptance |
| tuning | `--calibrate` measures the machine and sets the configuration | profiles per machine, a calibrated simulator ([GLM_RUNNER_SIM.md](GLM_RUNNER_SIM.md)) to rank changes, and a [bring-up guide](GLM53F_NEW_HARDWARE.md) |
| quality evidence | zero-shot accuracy of each quantization relative to FP8 | every lossy setting gated against the unpruned Q23 model: 35-track corpus perplexity/KL/top-1 limits and HumanEval at or above 152/164 |

What was taken from Maya, all reimplemented independently: the aged route-count admission rule for the
GPU expert cache (`STRATA_GLM_MIXED_ADAPT_MIN_ROUTES` / `MIN_GAIN_PCT`, default off; it did not help on
held-out prompts here), the idea of pricing CPU expert service against host-to-device transfer as
concurrent lanes (the simulator's miss-lane model), separating fresh-prompt from repeated warmed-prompt
measurements, and the layer-range split, which was built and then removed for the reason above. Its
RDNA expert kernels target IQ2/IQ3 on gfx1100/gfx1151 and were not applicable to Q2_K/Q3_K on gfx1201.
The study is recorded in [GLM_TR16_MIXED_REDESIGN.md](GLM_TR16_MIXED_REDESIGN.md#project-maya-follow-up).

The published numbers are not a controlled comparison. Maya's closest machine, two RTX 3090 (48 GB of VRAM)
on a Threadripper 3970X, decodes 53 tok/s and prefills 885 tok/s with its own quantization; the two-GPU
desktop here (32 GB of VRAM over two vendors, an older Threadripper 1950X) decodes 40-44 and prefills 870
with the gated Q23 REAP-224 configuration. Quantization, quality measure, VRAM and prompts all differ.

## Documents

Current: this page, [GLM53F_NEW_HARDWARE.md](GLM53F_NEW_HARDWARE.md) (new-machine guide and benchmark protocol), [GLM_TR16_RUNNER_PLAN.md](GLM_TR16_RUNNER_PLAN.md) (two-GPU work, sections 12-15 are
the release-phase measurements), [GLM_RUNNER_SIM.md](GLM_RUNNER_SIM.md), [MULTI_GPU.md](MULTI_GPU.md),
[GLM_RELEASE_Q2Q3.md](GLM_RELEASE_Q2Q3.md) (full-model and sidecar recipes), [GLM_PRUNE_224.md](GLM_PRUNE_224.md),
[GLM_B550_REAP50.md](GLM_B550_REAP50.md), [GLM53_V100.md](GLM53_V100.md), [GLM_TESTS.md](GLM_TESTS.md).

Historical (kept as records, numbers superseded by the table above): [README_GLM53_FLASH.md](README_GLM53_FLASH.md),
[GLM53_FLASH_PERFORMANCE.md](GLM53_FLASH_PERFORMANCE.md), [GLM_RELEASE_20261013.md](GLM_RELEASE_20261013.md),
[GLM_RELEASE_CANDIDATE.md](GLM_RELEASE_CANDIDATE.md), [GLM_TR16_MIXED_REDESIGN.md](GLM_TR16_MIXED_REDESIGN.md),
[GLM_TR16_OPTIMIZATION.md](GLM_TR16_OPTIMIZATION.md), the `GLM_Q2_*`, `GLM_64G_16G_SAFE`, `GLM_32G_8G` and
`GLM53_V100_*` studies, and the `glm53_flash_*_measurement.json` records.
