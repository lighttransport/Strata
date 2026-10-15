---
license: mit
base_model:
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

The full reproduction guide, with build commands, run commands, measured
speeds and quality, is `docs/GLM_RELEASE_Q2Q3.md` in the Strata fork.

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
| REAP-224 Q23, tuned profile, local release candidate | 2026-10-13 | 150/164 (91.46%) |

These runs use different builds and settings; their score differences do not
isolate pruning or speculative depth. The latest REAP-224 run completed all
164 tasks but missed its 152-pass release floor. It had no test timeouts or new
capped tasks. Shared CPU/GPU activity was allowed, so it supports no performance
claim. An earlier version of the tuned profile scored 152/164; that historical
result does not qualify the current candidate.

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

**Simplest option on a 128 GB machine:** download `GLM-5.3-Flash-Q23-assembled.gguf` alone and pass it as the model (no `--expert-pack`).

**Portability limitation of the sidecars.** The loader fingerprints the original
shards by size, header and file modification time, and there is no override, so
a fresh download fails with `source fingerprint mismatch`. Either regenerate the
sidecar against your shards (the fork's `docs/GLM_Q2_REQUANTIZATION.md`) or
restore the original modification times (untested workaround, exact timestamps
in `docs/GLM_RELEASE_Q2Q3.md`). The assembled REAP-50 file is self-contained and
has no such check.

## Measured results

| Machine | Weights | Decode tok/s | Prefill tok/s | Quality |
|---|---|---:|---:|---|
| 128 GB, Threadripper 1950X, RTX 5060 Ti 16 GB (PCIe 3.0 x8) | shards + `experts-q23.gguf`, affinity 0.08, MTP depth 1 (`configs/experimental/glm53f-q23-tr16-aff08.json`) | 23.1 (2K prompt), 23.0 (8K) | 131 (2K), 428 (8K single pass) | vs unchanged Q23: perplexity +2.07%, KL +0.018, top-1 −0.29 pt; HumanEval 153/164 (baseline 154) |
| same | shards + `experts-q22.gguf`, Q4_K dense, MTP depth 3 soft acceptance, affinity 0.14, top-k 6, repetition guard (research binary) | 36.2 (2K), 38.1 (8K) | 143 (2K) | perplexity +8.45%, KL +0.100, top-1 −3.6 pt; not qualified |
| 64 GB, Ryzen 3950X, RX 9070 XT 16 GB (ROCm) | `reap50/…assembled.gguf`, default service config | 10.2 | 230 | vs original UD-Q2_K_XL: perplexity 11.12 vs 8.57 (+29.8%), KL 0.587, top-1 69.8% |
| same | affinity 0.06 profile | 16.5 | 229 | passes the fork's C++ review coding task; higher affinity did not |

Speeds are medians of three timed 512-token (128 for the B550 default)
continuations after a warm-up, measured under a quiet-machine guard. They are
not promises for other hardware.

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

REAP-50 assembled model: the same calibration and `strata-glm-q2-pack` run on
`GLM-5.3-Flash-REAP50-Q4_K_M.gguf` from
[patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF),
followed by `strata-glm-assemble REAP50-Q4_K_M.gguf reap50-q23.gguf GLM-5.3-Flash-REAP50-Q23-assembled.gguf`,
which copies bytes without requantizing and verifies every expert checksum.

These commands reproduce the workflow, not the published bytes; identical output
needs the same revision, corpus, settings and toolchain.

## Run

128 GB machine: pass the first Unsloth shard as the model and add
`--expert-pack=/path/to/experts-q23.gguf`; the measured profile is
`configs/experimental/glm53f-q23-tr16-aff08.json` (also usable with
`python3 -m serve.server --engine glm --config …`). 64 GB machine: use the
assembled file as the model with
`configs/glm53f-reap50-q23-b550-60g-9070xt-experimental.json` via
`tools/glm_b550_60g.sh start`, inside the 60 GiB memory cgroup it sets up. Full
command lines are in `docs/GLM_RELEASE_Q2Q3.md`.

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
