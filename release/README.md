---
license: mit
base_model:
  - zai-org/GLM-5.3-Flash
  - unsloth/GLM-5.3-Flash-GGUF
  - patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF
tags:
  - gguf
  - strata
  - quantization
  - reap
---

# GLM-5.3-Flash-Strata-Q2Q3-Experts

Experimental GGUF weights for running **GLM-5.3-Flash** on a normal PC with the
[LightTransport Strata fork](https://github.com/lighttransport/Strata): expert
sidecars and a complete full model for a 128 GB host with a 16 GB NVIDIA GPU,
a smaller REAP-224 model, and a complete REAP-50 model for a 64 GB host with a
16 GB AMD GPU. The sidecars requantize
routed expert gate/up weights from IQ2_XS to Q2_K and down weights from
IQ3_XXS to Q3_K; this changes weight values and is neither a lossless repack nor
a fine-tune. REAP-224 keeps 224 of 288 routed experts per layer; REAP-50 keeps
144. Read the quality numbers below before choosing a model.

Current measurements, setup for one and two GPUs, and context-length limits are in
[`docs/GLM53F_RELEASE.md`](https://github.com/lighttransport/Strata/blob/glm53f/docs/GLM53F_RELEASE.md)
on the fork's `glm53f` branch; bringing it up on other hardware is described in
[`docs/GLM53F_NEW_HARDWARE.md`](https://github.com/lighttransport/Strata/blob/glm53f/docs/GLM53F_NEW_HARDWARE.md).
The conversion recipe is `docs/GLM_RELEASE_Q2Q3.md`.

## Default since 2026-10-17: experts quantized from the FP8 release

**On a 128 GB machine use `GLM-5.3-Flash-Q23-FP8-assembled.gguf`** (or `experts-q23-fp8.gguf` with the Unsloth
shards); **on a 64 GB machine use `reap50/GLM-5.3-Flash-REAP50-Q23-FP8-assembled.gguf`.** Their routed experts have
the same types as Q23 (Q2_K gate/up, Q3_K down) but are quantized once, directly from the official FP8 weights,
instead of from Unsloth's 2-bit experts (REAP-50: from its Q4_K_M experts). Measured against exact logits of the FP8
release on 16 held-out sequences (8,192 tokens):

| Weights | Size | Perplexity (FP8: 4.449) | KL vs FP8 | Top-1 agreement | Decode | HumanEval |
|---|---:|---:|---:|---:|---:|---:|
| **Q23 from FP8** (default, 128 GB) | 119.9 GB | **4.606 (+3.5%)** | **0.117** | **88.3%** | 19.6 tok/s | **155/164** (unchanged routing), 151 (affinity 0.08) |
| Q23 (first release) | 120.9 GB | 4.850 (+9.0%) | 0.212 | 84.7% | 19.6 tok/s | 154/164, 156 (affinity 0.08) |
| Unsloth UD-Q2_K_XL as shipped | 108.7 GB | 4.688 (+5.4%) | 0.144 | 87.5% | | |
| **REAP-50 Q23 from FP8** (default, 64 GB) | 63.2 GB | **5.184 (+16.5%)** | **0.393** | **80.4%** | 16.1 tok/s | **154/164** |
| REAP-50 Q23 (first release) | 63.2 GB | 5.632 (+26.6%) | 0.479 | 77.6% | 15.9 tok/s | 149/164 |

Threadripper 1950X with an RX 9070 XT, the same decoder for every row: speed is 512 generated tokens after a 2K
prompt; the full models use MTP depth 1 and affinity 0.08, the REAP-50 rows the B550 profile with unchanged routing
inside a 60 GiB / no-swap cgroup; HumanEval as noted, REAP-50 with unchanged routing and MTP depth 1. The method, the
determinism checks and the per-projection errors are in the fork's `docs/GLM_FP8_REFERENCE.md`. The first-release
files below stay published for the earlier measurements.

## Choosing for output quality

**Use the full, unpruned `GLM-5.3-Flash-Q23-assembled.gguf` when output quality
is the priority and the model fits in RAM.** It retains all 288 routed experts
per layer and is 120.9 GB. Q23 is still a quantized model. REAP-224 is a smaller,
experimental 95.5 GB alternative that trades quality for memory headroom.

Full HumanEval results on the Threadripper 1950X, 128 GB DDR4-3200 and RTX 5060 Ti
16 GB used one greedy chat sample per task, low reasoning, a 2,048-token cap and
isolated tests:

| Weights and configuration | Date | Passed |
|---|---|---:|
| Full Q23, unchanged routing, strict MTP depth 2, canonical experts and adaptive GPU tier | 2026-10-07 | 156/164 (95.12%) |
| Full Q23, unchanged routing, strict MTP depth 1 | 2026-10-09 | 154/164 (93.90%) |
| Full Q23, routing affinity 0.08, strict MTP depth 1 | 2026-10-09 | 153/164 (93.29%) |
| REAP-224 Q23, tuned profile (RTX 5060 Ti), first run | 2026-10-13 | 152/164 (92.68%) |
| REAP-224 Q23, tuned profile, rerun | 2026-10-13 | 150/164 (91.46%) |
| REAP-224 Q23, two GPUs (RTX 5060 Ti + RX 9070 XT), affinity 0.14, soft MTP acceptance | 2026-10-16 | 146/164 (89.02%) |

These runs use different builds and settings; their score differences do not
isolate pruning or speculative depth. The fork's release floor is 152/164 plus a
corpus gate (perplexity within +6%, KL within +0.08, top-1 within 3 points of the
unpruned Q23 model). The REAP-224 tuned profile sits at that floor (152, then 150 on
a rerun); the faster two-GPU configuration passes the corpus gate but not HumanEval.

For quality priority, keep routing unchanged and use strict MTP acceptance.
Depth 2 has the highest recorded score above, but depth 1 was faster in a later
matched 2K-prompt timing check: 20.28 versus 18.63 tok/s, with identical output
on that prompt. The measured speed/quality compromise is full Q23 with affinity
0.08 and depth 1: about 23 tok/s, at the quality cost shown above. These timings
are specific to the TR16 machine. See `docs/GLM_Q2_DECODE_REDESIGN.md`,
`docs/GLM_TR16_OPTIMIZATION.md` and `docs/GLM_RELEASE_20261013.md` for the records.

## Files

| Path | Size | Use with | Notes |
|---|---:|---|---|
| `GLM-5.3-Flash-Q23-FP8-assembled.gguf` | 119.9 GB | nothing else | **default single-file full model**: the four Unsloth UD-Q2_K_XL shards with `experts-q23-fp8.gguf` assembled in (126 expert checksums verified); token-identical to the sidecar setup in a 128-token check in canonical mode (three trials). SHA-256 `ec0895e8330e771fb5c44610a2876ecc763c6e884a7ecd3e61e8ef2a8f3821a9` |
| `experts-q23-fp8.gguf` | 110.2 GB | the four Unsloth UD-Q2_K_XL shards | **default sidecar**: every routed projection of layers 3-44 quantized from the FP8 release, Q2_K gate/up, Q3_K down. SHA-256 `c2f06cd16ac5ded094e0c22040fdf170e132e7dcb3b59ae0e588cc83dc8f198e` |
| `experts-q23-fp8.gguf.quality.json` | small | — | per-projection reconstruction error against the FP8 weights |
| `reap50/GLM-5.3-Flash-REAP50-Q23-FP8-assembled.gguf` | 63.2 GB | nothing else | **default for 64 GB**: REAP-50 (the same 144 experts per layer as the first REAP-50 file) with its routed experts quantized from the FP8 release. SHA-256 `03d09db917f1498ae6cc2a18a117a92043955268bd5930ade8fae1bbea491401` |
| `reap50/reap50-expert-map.txt` | 23 KB | — | the original expert index behind each REAP-50 expert, per layer (recovered from the routers) |
| `GLM-5.3-Flash-Q23-assembled.gguf` | 120.9 GB | nothing else | **single-file full model**: the four Unsloth UD-Q2_K_XL shards with `experts-q23.gguf` assembled in (byte copy, checksums verified, layers 11/12/44 kept in their original formats). No fingerprint check, no shards needed; token-identical to the sidecar setup in a 256-token check. Needs a Strata build with the assembled-file loader change. SHA-256 `8e280fd86714b606211c285552e77cde7b8c31ac0e9483df85c72ab1174a50e7` |
| `reap224/GLM-5.3-Flash-Q23-REAP224-assembled.gguf` | 95.5 GB | nothing else | **single-file pruned model**, 224 of 288 routed experts per layer, sliced from full Q23 without further requantization. Experimental; see the quality results above. SHA-256 `3ef5726ae4a9768fa8311453927674401f3958a8c3884effddac92fc41262003` |
| `reap224/plan-keep224.txt`, `reap224/plan-keep224.report.json` | 35 KB / 32 KB | — | kept expert IDs and per-layer saliency report |
| `reap224/coverage-keep224.json` | 103 KB | the REAP-224 model | expert prior re-indexed to the kept experts |
| `experts-q23.gguf` | 105.5 GB | the four Unsloth UD-Q2_K_XL shards | Q2_K gate/up, Q3_K down; layer 11 unchanged. SHA-256 `d358456cf90e7aef45e74806cc7eb96f05ea69761422e96a2d0fae256ea68aff` |
| `experts-q22.gguf` | 95.9 GB | the four Unsloth shards | as above with 39 down projections in Q2_K (9.1% fewer expert bytes); used by the faster research profiles |
| `experts-q2{2,3}.gguf.quality.json` | small | — | per-projection reconstruction error and the source fingerprint |
| `calibration/coverage-q23-tr16.json` | 106 KB | both sidecars | expert prior for the GPU-resident tier (`STRATA_GLM_EXPERT_PRIOR`) |
| `reap50/GLM-5.3-Flash-REAP50-Q23-assembled.gguf` | 63.2 GB | nothing else | complete REAP-50 model (144 experts, top-8) with Q2_K/Q3_K routed experts assembled into the Q4_K_M base. SHA-256 `c3b0cfb144556bd28e265d8255d48eeafce3d127aba5b9b5dd4e326322a34aa9` |
| `reap50/reap50-q23-assembled.provenance.json` | small | — | source model, pack, fingerprint, checksums |
| `reap50/reap50-coverage.json` | 60 KB | the REAP-50 model | expert prior |

**The sidecars are not a complete model.** Keep all four original UD-Q2_K_XL
shards from [Unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF); dense,
attention, router, MTP and unconverted expert weights load from them. They need
Strata's `--expert-pack` loader; other engines have not been tested.

**Simplest option on a 128 GB machine:** download `GLM-5.3-Flash-Q23-FP8-assembled.gguf` alone and pass it as the model (no `--expert-pack`).

**Portability limitation of the sidecars.** The loader fingerprints the original
shards by size, header and file modification time, and there is no override, so
a fresh download fails with `source fingerprint mismatch`. Either regenerate the
sidecar against your shards (the fork's `docs/GLM_Q2_REQUANTIZATION.md`) or
restore the original modification times (untested workaround, exact timestamps
in `docs/GLM_RELEASE_Q2Q3.md`). The assembled REAP-50 file is self-contained and
has no such check.

## Measured results

Single-stream decode and prompt (prefill) rates, tokens per second. Full trials and conditions:
`docs/GLM53F_RELEASE.md` in the fork.

| Machine | Weights and configuration | Decode | Prefill | Quality |
|---|---|---:|---:|---|
| Threadripper 1950X, 128 GB, RTX 5060 Ti 16 GB | `reap224/…assembled.gguf`, tuned profile, MTP depth 3 | 29.0 (2K), 31.6 (8K) | 168 (2K), 496 (8K) | HumanEval 152/164 (150 on rerun); perplexity +4.9%, KL 0.064 vs unpruned Q23 |
| same, RX 9070 XT 16 GB alone (ROCm) | `reap224/…`, affinity 0.14 | 34.1 (8K) | 561 (8K) | not separately gated |
| same, RTX 5060 Ti + RX 9070 XT (expert worker on the second card) | `reap224/…`, affinity 0.14 | 40-44 (1K-8K) | 465 (8K); 870 with the HIP card as primary | corpus gate passed; HumanEval 146/164, below the floor |
| same, RX 9070 XT 16 GB alone | `GLM-5.3-Flash-Q23-FP8-assembled.gguf` / `experts-q23-fp8.gguf` (full), affinity 0.08, MTP depth 1 | 19.6 (2K), 19.6 (8K) | 178 (2K), 497 (8K) | vs FP8: perplexity +3.5%, KL 0.117; HumanEval 151/164, 155/164 with unchanged routing |
| same, RTX 5060 Ti | `GLM-5.3-Flash-Q23-assembled.gguf` (full), affinity 0.08, MTP depth 1 | 23.1 (2K), 23.0 (8K) | 131 (2K), 428 (8K) | HumanEval 153/164; perplexity +2.07%, KL +0.018 vs unchanged routing |
| Threadripper 1950X in a 60 GiB / no-swap cgroup, RX 9070 XT | `reap50/GLM-5.3-Flash-REAP50-Q23-FP8-assembled.gguf`, the B550 profile, unchanged routing | 16.1 (2K), 16.0 (8K) | 219 (2K), 217 (8K) | vs FP8: perplexity +16.5%, KL 0.393, top-1 80.4%; HumanEval 154/164 (first REAP-50 file under the same settings: +26.6%, 0.479, 77.6%, 149/164, 15.9 / 16.2 tok/s) |
| Ryzen 9 3950X, 64 GB, RX 9070 XT 16 GB (ROCm) | `reap50/…assembled.gguf` (first release), default service config | 10.2 | 230 | vs original UD-Q2_K_XL: perplexity 11.12 vs 8.57 (+29.8%), KL 0.587, top-1 69.8% |
| same | `reap50/…`, affinity 0.06 | 16.5 | 229-250 | passes the fork's C++ review task |
| Ryzen 9 3950X, 64 GB DDR4-2666, RTX 3070 8 GB | `reap50/…`, presets per context | 19.5 (2K), 16.9 (4K), 17.6 (8K) | 464-506 | matched perplexity +5.7 to +8.9%; 30-task HumanEval subset passed |

Speeds are medians of timed 128-512-token continuations after a warm-up, with swap disabled, on the
machines named; they are not promises for other hardware. Context length: about 26 MB of VRAM per 1K tokens
on the primary GPU, taken from the GPU expert cache; roughly 128K is practical on a 16 GB card (see the
fork's release notes).

## Reproduce the conversions

Sidecars (128 GB machine, CUDA): calibrate the Unsloth model with the fork's
corpus, then convert and verify.

```bash
git clone https://github.com/lighttransport/Strata.git && cd Strata
git checkout e5232ca3 && git submodule update --init --recursive
cmake -S . -B build-glm -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_MMQ_KQUANTS=ON
cmake --build build-glm -j 8 --target strata-glm-decode strata-glm-q2-pack strata-glm-assemble
hf download unsloth/GLM-5.3-Flash-GGUF --include 'UD-Q2_K_XL/*' --local-dir models/glm53f
MODEL=models/glm53f/UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf; WORK=build-q2-redesign
python tools/glm_q2_corpus.py "$MODEL" "$WORK/corpus"
STRATA_GLM_CALIBRATION_DIR="$WORK/calibration" STRATA_GLM_LOCK_RUNTIME=1 \
  python tools/glm_q2_run_guard.py --output "$WORK/calibration-run" -- \
  numactl --interleave=all build-glm/strata-glm-decode "$MODEL" 0 1 4096 15 \
  --context=8192 --prefill-batch=256 --gpu-budget-mib=12288 --cpu-affinity=numa --decode-graphs \
  --eval-corpus="$WORK/corpus/calibration.ids"
build-glm/strata-glm-q2-pack "$MODEL" experts-q23.gguf --calibration="$WORK/calibration" --threads=16
build-glm/strata-glm-q2-pack "$MODEL" experts-q22.gguf --calibration="$WORK/calibration" --threads=16 --down=q2 --reuse=experts-q23.gguf
build-glm/strata-glm-q2-pack --verify "$MODEL" experts-q23.gguf
```

FP8-derived files (the defaults): download [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
and add `--fp8=<its directory> --gate-type=q2_k --down-type=q3_k` to the `strata-glm-q2-pack` command above, then
`strata-glm-assemble "$MODEL" experts-q23-fp8.gguf GLM-5.3-Flash-Q23-FP8-assembled.gguf`. For REAP-50, recover the
kept experts with `tools/glm_reap_expert_map.py` (every REAP-50 router row equals exactly one original row), pack
against the published REAP-50 file with `--expert-map=`, and assemble with `strata-glm-assemble --reassemble`. Full
commands: `docs/GLM_RELEASE_Q2Q3.md`, "Reproducing the FP8-derived weights".

REAP-50 assembled model (first release): the same calibration and `strata-glm-q2-pack` run on
`GLM-5.3-Flash-REAP50-Q4_K_M.gguf` from
[patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF),
followed by `strata-glm-assemble REAP50-Q4_K_M.gguf reap50-q23.gguf GLM-5.3-Flash-REAP50-Q23-assembled.gguf`,
which copies bytes without requantizing and verifies every expert checksum.

These commands reproduce the workflow, not the published bytes; identical output
needs the same revision, corpus, settings and toolchain.

## Run

128 GB machine: use an assembled file as the model. The default weights were measured with
`configs/experimental/glm53f-q23fp8-tr16-9070xt-aff08.json` (and `-aff0.json` for unchanged routing). Other profiles in the fork:
`configs/experimental/glm53f-q23-reap224-tr16-tuned.json` (REAP-224, one GPU),
`configs/experimental/glm53f-reap224-tr16-two-gpu-decode.json` and `…-prefill.json` (two GPUs, run through
`tools/glm_mixed_run.py`), and `configs/experimental/glm53f-q23-tr16-aff08.json` (full model; with the
sidecars instead, pass the first Unsloth shard and `--expert-pack=/path/to/experts-q23.gguf`). Profiles work with
`python3 -m serve.server --engine glm --config …`. 64 GB machine: use
`reap50/GLM-5.3-Flash-REAP50-Q23-FP8-assembled.gguf` (the default; or the first REAP-50 file) as the model with
`configs/glm53f-reap50-q23-b550-60g-9070xt-experimental.json` via
`tools/glm_b550_60g.sh start`, inside the 60 GiB memory cgroup it sets up; with an 8 GB card, the
`configs/glm53f-reap50-q23-64g-rtx3070-*.json` presets. Full command lines are in `docs/GLM53F_RELEASE.md`
and `docs/GLM_RELEASE_Q2Q3.md`.

## License and attribution

The original [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is by
**Z.ai** under the [MIT license](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE),
copyright (c) 2026 Z.AI Co., Ltd. The UD-Q2_K_XL quantization is by **Unsloth**
([MIT](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)). The REAP-50 pruned
GGUF base is by **patrickbdevaney** (MIT, base model zai-org/GLM-5.3-Flash). The
original **Strata** engine is by **Niko1221 and the Strata contributors**
([MIT](https://github.com/Niko1221/Strata/blob/main/LICENSE)); this conversion
uses the [LightTransport fork](https://github.com/lighttransport/Strata).
Quantization kernels come from [GGML](https://github.com/ggml-org/ggml) (MIT).
These derived weights are distributed under MIT; retain the upstream notices
when redistributing. This is a community conversion, not an official Z.ai,
Unsloth, REAP or upstream Strata release.
