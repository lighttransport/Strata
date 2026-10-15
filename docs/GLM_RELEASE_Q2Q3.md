# GLM-5.3-Flash on Strata: reproducing the released Q2/Q3 expert weights

This document describes how to run GLM-5.3-Flash with the weights published at
[huggingface.co/lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts](https://huggingface.co/lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts)
on two kinds of PC, and how those weights were produced. Every number is a
measurement on the named machine; nothing here is a guarantee for other hardware.
The weights are experimental conversions: the expert sidecars change weight
values (they are not lossless repacks), and the REAP-50 model removes half the
experts. See the quality tables before relying on either.

**For output quality, prefer the full, unpruned `GLM-5.3-Flash-Q23-assembled.gguf`
(288 experts, 120.9 GB) when RAM permits.** Full Q23 is still quantized. REAP-224
(224 experts, 95.5 GB) is an experimental alternative for lower memory use.
Its latest tuned release candidate scored 150/164 HumanEval, below its 152-pass
floor; see the [2026-10-13 validation report](GLM_RELEASE_20261013.md).

On TR1950X / 128 GB DDR4-3200 / RTX 5060 Ti 16 GB, full Q23 with unchanged routing
scored 156/164 with strict MTP depth 2 on October 7 and 154/164 with depth 1 on
October 9. These are different builds/settings, not a controlled depth comparison.
Keep routing unchanged and use strict acceptance for quality priority. The later
matched timing check favored depth 1 (20.28 versus 18.63 tok/s at 2K, identical
output on that prompt). The measured speed/quality compromise below uses affinity
0.08 and depth 1: about 23 tok/s and 153/164. All scores use the pinned greedy chat
protocol, low reasoning and a 2,048-token cap. See the
[depth-two result](GLM_Q2_DECODE_REDESIGN.md#humaneval) and
[later qualification](GLM_TR16_OPTIMIZATION.md#quality-and-promotion).

| Configuration | Host RAM | GPU | Weights | Measured decode / prefill |
|---|---:|---|---|---|
| A: full model, 288 experts | 128 GB | 16 GB NVIDIA (RTX 5060 Ti, PCIe 3.0 x8) | Unsloth UD-Q2_K_XL shards + `experts-q23.gguf` | 23.1 tok/s decode, 131 tok/s prefill at 2K (qualified); research candidates up to 36–38 |
| B: REAP-50 pruned model, 144 experts | 64 GB | 16 GB AMD (RX 9070 XT, ROCm) | `reap50/GLM-5.3-Flash-REAP50-Q23-assembled.gguf` (standalone) | 10.2 tok/s decode, 230 tok/s prefill (default service); 16.5 decode with affinity 0.06 |

## Files in the repository

| Path | Size | What it is | Needs |
|---|---:|---|---|
| `GLM-5.3-Flash-Q23-assembled.gguf` | 120.9 GB | Single-file full model: Unsloth UD-Q2_K_XL shards + `experts-q23.gguf`, assembled with `strata-glm-assemble <shard 1> experts-q23.gguf <out>` (split base and retained original projections supported since this release). Token-identical to the sidecar setup (256 tokens, qualified profile). SHA-256 `8e280fd8…a50e7`. | a Strata build with the assembled-file loader change |
| `reap224/GLM-5.3-Flash-Q23-REAP224-assembled.gguf` | 95.5 GB | Single-file REAP-224 model, sliced from full Q23 without further requantization. SHA-256 `3ef5726ae4a9768fa8311453927674401f3958a8c3884effddac92fc41262003`. See [quality and pruning details](GLM_PRUNE_224.md). | a Strata build with the pruned-model loader |
| `reap224/plan-keep224.txt`, `reap224/plan-keep224.report.json` | 35 KB / 32 KB | Kept expert IDs and per-layer saliency report. | — |
| `reap224/coverage-keep224.json` | 103 KB | Expert prior re-indexed to the 224 kept experts. | the REAP-224 model |
| `experts-q23.gguf` | 105.5 GB | Expert sidecar for the full model: routed gate/up weights requantized IQ2_XS→Q2_K, down IQ3_XXS→Q3_K (layer 11 unchanged). SHA-256 `d358456c…68aff`. | the four Unsloth UD-Q2_K_XL shards |
| `experts-q22.gguf` | 95.9 GB | Same, but 39 down projections also in Q2_K (9.1% fewer expert bytes; used by the faster research profiles). | the four Unsloth shards |
| `experts-q2{2,3}.gguf.quality.json` | small | Per-projection reconstruction error and the source fingerprint the sidecar was built against. | — |
| `calibration/coverage-q23-tr16.json` | 106 KB | Expert-popularity prior from the calibration corpus (`STRATA_GLM_EXPERT_PRIOR`); seeds the GPU-resident expert tier. | — |
| `reap50/GLM-5.3-Flash-REAP50-Q23-assembled.gguf` | 63.2 GB | Complete single-file REAP-50 model (144 experts, top-8) with Q2_K/Q3_K routed experts assembled into the Q4_K_M base. SHA-256 `c3b0cfb1…4aa9`. | nothing else |
| `reap50/reap50-q23-assembled.provenance.json` | small | Source model, source pack, fingerprint, checksums and byte count of the assembly. | — |
| `reap50/reap50-coverage.json` | 60 KB | Expert-popularity prior for the REAP-50 model. | — |

Upstream sources: the UD-Q2_K_XL shards are from
[unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF);
the REAP-50 base (`GLM-5.3-Flash-REAP50-Q4_K_M.gguf`, 99.3 GB, 144 experts) was
downloaded from
[patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF).
The original model is [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT).

## Portability warning for the expert sidecars (configuration A)

The single-file `GLM-5.3-Flash-Q23-assembled.gguf` avoids everything in this section: pass it as the model and drop `--expert-pack`.


The sidecar loader fingerprints the **original shard files**: size, header bytes
and the file modification time at full precision
(`include/strata/core/model.hpp`, `source_fingerprint()`), and refuses to load
with `expert pack: version or source fingerprint mismatch`. There is no
override. A fresh download of the same shards has new modification times, so
**the published sidecars do not load on a fresh installation as-is.** Two ways
around it:

1. Regenerate the sidecar locally against your shards (about 70 minutes of
   calibration on a 16-core machine plus the pack conversion; commands below).
   This reproduces the workflow, not the published bytes.
2. Restore the modification times the sidecars were built against (untested
   workaround; only valid while the Unsloth repository revision is unchanged,
   so sizes and headers still match):

   ```sh
   cd models/glm53f/UD-Q2_K_XL
   touch -d '2026-09-07 08:25:41.224191838 +0900' GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
   touch -d '2026-09-07 09:47:37.416159536 +0900' GLM-5.3-Flash-UD-Q2_K_XL-00002-of-00004.gguf
   touch -d '2026-09-07 11:16:01.801103295 +0900' GLM-5.3-Flash-UD-Q2_K_XL-00003-of-00004.gguf
   touch -d '2026-09-07 11:32:10.324522715 +0900' GLM-5.3-Flash-UD-Q2_K_XL-00004-of-00004.gguf
   ```

The assembled REAP-50 file (configuration B) does not have this problem: it is
self-contained and the loader only checks its own embedded pack metadata.

## Configuration A: 128 GB RAM, 16 GB NVIDIA GPU (TR16)

Measured on a Threadripper 1950X (16 cores, two NUMA nodes, 128 GB DDR4) with
an RTX 5060 Ti 16 GB on PCIe 3.0 x8, CUDA 13, Ubuntu 24.04. Peak resident set
is about 108 GiB (the experts are file-mapped; host memory below about 118 GB
will swap), peak GPU allocation 12.6–12.8 GiB with the desktop sharing the GPU.

### Build

```sh
git clone https://github.com/lighttransport/Strata.git && cd Strata
git checkout e5232ca3          # revision of the research builds; 6ecf3bcb for the qualified profile
git submodule update --init --recursive
cmake -S . -B build-glm -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_ENABLE_HIP=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_MMQ_KQUANTS=ON
cmake --build build-glm -j 12 --target strata-glm-decode strata-glm-q2-pack
python3 -m venv .venv && .venv/bin/pip install huggingface_hub regex jinja2
```

Set `CMAKE_CUDA_ARCHITECTURES` for your GPU. `numactl` is required on two-node
machines.

### Download

```sh
hf download unsloth/GLM-5.3-Flash-GGUF --include 'UD-Q2_K_XL/*' --local-dir models/glm53f
hf download lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts --include 'experts-q23.gguf' 'calibration/*' \
  --local-dir models/glm53f-strata
```

Apply the portability workaround above, then verify the sidecar:

```sh
build-glm/strata-glm-q2-pack --verify models/glm53f/UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf \
  models/glm53f-strata/experts-q23.gguf
```

### Qualified profile: affinity 0.08, MTP depth 1 (23 tok/s)

`configs/experimental/glm53f-q23-tr16-aff08.json` is the measured profile
(2K prompt: 23.06 decode / 130.9 prefill tok/s; 8K prompt with
`--prefill-batch=8192` and `STRATA_GLM_PREFILL_SCRATCH_CAP_MIB=2048`: 23.00 /
427.9). Against unchanged Q23 it measured perplexity +2.07%, KL +0.018, top-1
agreement −0.29 points on the 8,192-target corpus and 153/164 HumanEval versus
154/164. The direct command (environment from the profile's `env` block):

```sh
env STRATA_GLM_QUANT_ONCE=1 STRATA_NATIVE_NUMA_LOCAL=0 STRATA_POOL_SPIN_US=20000 \
 STRATA_GLM_LAYER_GRAPHS=1 STRATA_GLM_VERIFY_GRAPHS=1 STRATA_NATIVE_TASKS_PER_THREAD=12 \
 STRATA_GLM_STEP_PIPELINE=1 STRATA_GLM_Q8_DECODE=1 STRATA_GLM_LAYER_FLOW=1 STRATA_GLM_MAPPED_OWNED=1 \
 STRATA_GLM_EXPERT_PRIOR=$PWD/models/glm53f-strata/calibration/coverage-q23-tr16.json \
 STRATA_GLM_MTP_SKIP_ANCHOR=1 STRATA_GLM_SPLIT_VERIFY=1 STRATA_GLM_CANON=1 STRATA_GLM_TIER_ADAPT=1 \
 STRATA_GLM_GPU_RESERVE_MIB=512 STRATA_GLM_TIER_OWNED_RESERVE=1 STRATA_GLM_Q23_WIDE=1 \
 STRATA_GLM_ROUTE_AFFINITY=0.08 STRATA_GLM_MLA_GRAPHS=1 STRATA_GLM_STAGE_WORKERS=4 \
 STRATA_GLM_DEFER_COPY_WAIT=1 STRATA_GLM_DEQUANT_ONCE=1 STRATA_GLM_MLA_VECTOR_GATHER=1 \
 STRATA_GLM_KDA_PREPARE=1 STRATA_GLM_KDA_COLUMNS=128 STRATA_GLM_STAGE_PREFETCH=1 \
 STRATA_GLM_PREFETCH_GROUPS=12 STRATA_GLM_PREFETCH_REUSE_DEVICE=1 STRATA_GLM_PREFILL_SCRATCH_CAP_MIB=1024 \
 STRATA_GLM_MLA_F16=1 STRATA_GLM_PREFILL_KDA_ROW_PARTS=4 STRATA_GLM_MAPPED_LAZY=1 \
 STRATA_GLM_COMPACT_STAGING=1 STRATA_GLM_RELEASE_GPU_DENSE_HOST=1 STRATA_GLM_NUMA_AFTER_PREFILL=1 \
 STRATA_GLM_TIER_PREALLOCATE=1 STRATA_GLM_COMPACT_VERIFY_HISTORY=1 STRATA_GLM_MAILBOX_PREALLOCATE=1 \
 STRATA_Q23_PREFETCH=512 \
numactl --interleave=all build-glm/strata-glm-decode \
  models/glm53f/UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf @prompt.ids 512 4096 15 \
  --context=12288 --prefill-batch=2048 --gpu-budget-mib=14336 --cpu-affinity=numa --decode-experts=cpu \
  --decode-cache-mib=5632 --prefill-experts=mmq --decode-graphs \
  --expert-pack=models/glm53f-strata/experts-q23.gguf --speculative=mtp --draft-depth=1 --mtp-experts=cpu
```

Positional arguments are model, prompt (`@file` of token ids), tokens to
generate, dense cache MiB and CPU threads. The same JSON profile drives the
Python server: `python3 -m serve.server --engine glm --config
configs/experimental/glm53f-q23-tr16-aff08.json --port 8080` (greedy text only;
the server does not add `numactl`, use `exe: tools/glm_numa_interleave.sh` with
`env.STRATA_GLM_NUMA_EXECUTABLE` on two-node hosts).

### Faster research profiles (not quality-qualified)

Measured 2026-10-10 with `experts-q22.gguf`, dense weights requantized to Q4_K
at load (`STRATA_GLM_DENSE_Q4=all`), MTP depth 3 with soft acceptance
(`STRATA_GLM_MTP_ACCEPT_MARGIN=4`, `TOP_K=6`, `MAX_SOFT=3`), the repetition
guard (`STRATA_GLM_MTP_REPEAT_GUARD=4`) and a 6656 MiB decode cache. These need
the research binary built from `docs/fixtures/glm_tr16_goal_20261010/draft-route.patch`
on `e5232ca3`; the flags do not exist in the plain tree. Held-out, 512 generated
tokens, three timed repetitions:

| Profile | 2K decode | 8K decode | Full corpus vs Q23 | HumanEval |
|---|---:|---:|---|---|
| affinity 0.14 + top-k 6 | 36.2 | 38.1 | PPL +8.45%, KL +0.100, top-1 −3.6 pts | see `docs/GLM_TR16_OPTIMIZATION.md` |
| affinity 0.16 + top-k 6 | 36.7 | 36.4 | pilot PPL +10.7% (rejected) | 19/20 on the first 20 tasks |
| soft margin 4 control (affinity 0.08) | 31.2 | 34.3 | PPL +3.95%, KL +0.058, top-1 −2.3 pts | looped without the guard |

Any routing change above affinity 0.08 needs the repetition guard; without it
the soft-acceptance decoder loops on long continuations. Full details, the
rejected levers and the measurement protocol are in
`docs/GLM_TR16_OPTIMIZATION.md`.

### Reproducing the sidecars

```sh
MODEL=models/glm53f/UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
WORK=build-q2-redesign
.venv/bin/python tools/glm_q2_corpus.py "$MODEL" "$WORK/corpus"        # 32,768 calibration targets, code and prose
STRATA_GLM_CALIBRATION_DIR="$WORK/calibration" STRATA_GLM_LOCK_RUNTIME=1 \
  .venv/bin/python tools/glm_q2_run_guard.py --output "$WORK/calibration-run" -- \
  numactl --interleave=all build-glm/strata-glm-decode "$MODEL" 0 1 4096 15 \
  --context=8192 --prefill-batch=256 --gpu-budget-mib=12288 --cpu-affinity=numa --decode-graphs \
  --eval-corpus="$WORK/corpus/calibration.ids"
build-glm/strata-glm-q2-pack "$MODEL" experts-q23.gguf --calibration="$WORK/calibration" --threads=16
build-glm/strata-glm-q2-pack "$MODEL" experts-q22.gguf --calibration="$WORK/calibration" --threads=16 \
  --down=q2 --reuse=experts-q23.gguf
build-glm/strata-glm-q2-pack --verify "$MODEL" experts-q23.gguf
```

The calibration writes `coverage.json`, which is the expert prior shipped as
`calibration/coverage-q23-tr16.json`. Allow 110 GiB of free disk per sidecar.

## Configuration B: 64 GB RAM, 16 GB AMD GPU (B550)

Measured on a Ryzen 9 3950X (16 cores), 62 GB DDR4-2133 (expert reads about
26–30 GB/s) and an RX 9070 XT 16 GB (gfx1201), ROCm 7.14, Ubuntu 24.04. The
service runs inside a 60 GiB memory cgroup; peak use is 52.9 GiB in the
benchmark and 55.9 GiB during start-up with warm-up. Keep the GPU power policy
automatic (`echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level`).

### Build

```sh
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_BUILD_RPATH=/opt/rocm/lib
cmake --build build-hip -j 4 --target strata-glm-decode strata-glm-assemble
python3 -m venv --system-site-packages .venv-glm-hip
```

### Download

```sh
hf download lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts --include 'reap50/*' --local-dir models/glm53f-strata
sha256sum models/glm53f-strata/reap50/GLM-5.3-Flash-REAP50-Q23-assembled.gguf   # c3b0cfb1…4aa9
```

No other model file is needed. Point `model` and `env.STRATA_GLM_EXPERT_PRIOR`
in the config at the downloaded `.gguf` and `reap50-coverage.json`.

### Default guarded service (10.2 tok/s decode, 230 tok/s prefill)

`configs/glm53f-reap50-q23-b550-60g-9070xt-experimental.json`, started by
`tools/glm_b550_60g.sh start` (stop, status, start-lan), which runs
`systemd-run --user -p MemoryMax=60G -p MemorySwapMax=0 … tools/glm_guarded_serve.py <config> --ram-gib 60 --reserve-mib 2048 --host 127.0.0.1 --port 8080`.
The decoder equivalent (12 threads, 4096 context, affinity 0):

```sh
build-hip/strata-glm-decode models/glm53f-strata/reap50/GLM-5.3-Flash-REAP50-Q23-assembled.gguf @prompt.ids 128 4096 12 \
  --context=4096 --prefill-batch=2048 --gpu-budget-mib=14336 --cpu-affinity=auto --decode-experts=cpu \
  --prefill-experts=f16-batched --decode-graphs --decode-cache-mib=6144
```

with the config's `env` exported (`STRATA_GLM_MAPPED_OWNED=1`,
`STRATA_GLM_MAPPED_LAZY=1`, `STRATA_GLM_Q2_NUMA_WEIGHTS=1`, `STRATA_GLM_CANON=1`,
`STRATA_GLM_TIER_ADAPT=1`, `STRATA_GLM_PACK_FUSE_QUANT=0`,
`STRATA_GLM_FLOATING_Q8=all`, `STRATA_GLM_RELEASE_GPU_DENSE_HOST=1`,
`STRATA_GLM_GPU_RESERVE_MIB=2048`, `STRATA_NATIVE_TASKS_PER_THREAD=6`,
`GLIBC_TUNABLES=glibc.cpu.x86_non_temporal_threshold=1048576`,
`STRATA_GLM_EXPERT_PRIOR=…/reap50-coverage.json`).

### Faster opt-in profile (affinity 0.06, 16.5 tok/s)

`configs/experimental/glm53f-reap50-q23-b550-quality-first.json`: 8192 context,
8192 MiB decode cache, `--gpu-budget-mib=15792`, `STRATA_GLM_GPU_RESERVE_MIB=512`,
`STRATA_GLM_TIER_RUNTIME_RESERVE_MIB=256`, `--route-affinity=0.06`. Measured 16.46
decode / 229 prefill tok/s (2K prompt, 512 tokens, three trials). The affinity
sweep on this machine (2K prompt, 256 tokens) gave 11.6 / 13.9 / 16.3 / 17.8 /
19.8 / 21.1 tok/s at 0 / .04 / .06 / .08 / .10 / .12, but only 0 and 0.06
passed the C++ review coding task; stay at 0.06 or below. For 8K documents
`glm53f-reap50-q23-b550-aff006-prefill8k.json` (single 8K prefill pass, 4 GiB
scratch cap) measured 294 prefill / 15.5 decode tok/s.

### Quality of the REAP-50 model

Against the original UD-Q2_K_XL on 32,768 targets: perplexity 11.12 versus 8.57
(+29.8%), KL 0.587, top-1 agreement 69.8%. Against canonical Q23 on a 512-target
smoke: +7.6% perplexity, KL 0.55. This model trades half its experts for
fitting in 64 GB; use configuration A when the host has 128 GB.

### How the assembled file was produced

On TR16: calibration of the REAP-50 Q4_K_M base with the same corpus
(`build-q2-v2/reap-calibrate.sh`, 61 minutes, 92 GiB peak RSS), the pack
conversion `strata-glm-q2-pack REAP50-Q4_K_M.gguf reap50-q23.gguf --calibration=… --threads=16`
(55.1 GB, Q2_K gate/up, Q3_K down, 126 verified projections), then
`strata-glm-assemble REAP50-Q4_K_M.gguf reap50-q23.gguf GLM-5.3-Flash-REAP50-Q23-assembled.gguf`,
which copies bytes without requantizing, checks the fingerprint and all expert
checksums, and refuses split bases. `reap50-q23-assembled.provenance.json`
records the inputs and the SHA-256.

## Uploading or mirroring the weights

`tools/hf_release_upload.py` uploads a manifest of files with verification
against the repository's LFS hashes and unlimited retries with backoff; it is
how this release was pushed from two machines and can be re-run at any time to
resume or verify.

## Licenses

GLM-5.3-Flash is by Z.ai under MIT (copyright (c) 2026 Z.AI Co., Ltd.). The
UD-Q2_K_XL quantization is by Unsloth (MIT). The REAP-50 pruned base GGUF was
published by patrickbdevaney under MIT with `zai-org/GLM-5.3-Flash` as its base
model. Strata is MIT (Niko1221 and the Strata contributors); the LightTransport
fork builds on it; quantization kernels come from GGML (MIT). The derived
weights are distributed under MIT; keep the upstream notices when
redistributing. This is a community conversion, not an official release of any
of those parties.
