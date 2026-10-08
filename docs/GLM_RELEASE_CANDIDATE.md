# GLM experimental release candidate — 2026-10-08

This candidate collects the current GLM full-model and REAP50 configurations,
HIP integration, model assembler, memory guards and reproduction evidence.
It is an experimental source snapshot, not an upstream Strata release or a
claim that pruned/requantized GLM preserves the original model's quality.
No 30 tok/s single-sequence target has been reached.

## Measured configurations

| Host | CPU / GPU | Model; RAM cap | Prompt / output | Warm prefill tok/s | Decode tok/s | Qualification |
| --- | --- | --- | --- | ---: | ---: | --- |
| TR16 | Threadripper 1950X / RTX 5060 Ti 16 GB | Full Q23; 124 GiB | 1024 / 512 | 65.890 | 19.662 | Three-trial medians, clean |
| TR16 | Same | Full Q23; 124 GiB | 8192 / 128 | 374.359 | 21.231 | Diagnostic: kernel reclaim/compaction |
| TR16 | Same | REAP50 Q23 Q8; 60 GiB | 1024 / 512 | 102.820 | 23.486 | Three-trial medians, clean |
| TR16 | Same | REAP50 Q23 Q8; 60 GiB | 8192 / 128 | 407.436 | 23.967 | Three-trial medians, clean |
| B550 | Ryzen 9 3950X / RX 9070 XT 16 GB | Assembled REAP50 Q23; 60 GiB | 2048 / 128 | 230.060 | 10.212 | Three-trial medians, clean |

TR16 uses two NUMA nodes and verified MTP depth one. B550 has one NUMA node,
uses no speculation, 12 CPU workers, a 6144 MiB decode cache and FP16 batched
expert prefill. These are different inference recipes, not a controlled CPU or
GPU comparison. All preserve at least 2048 MiB of actual free VRAM and disable
swap. B550 reported 16304 MiB total VRAM, with 3002 MiB minimum free in the
qualified benchmark. Peak RAM was 52.90 GiB (55.86 GiB during service startup).

TR16 results are historical measurements from the recorded v11 runs. The host
is now running another task; no new TR16 performance or quality run was started
for this candidate. See [TR16 evidence](GLM_DUAL_128G_64G.md) and
[B550 evidence](GLM_B550_REAP50.md) for exact trials and limitations.

## B550 reproduction

Use the candidate source archive and its per-file SHA-256 manifest. The base
merge commit `036c926e6db07e8f66d6b125b3139ef282dad327` alone does **not** contain
the GLM integration; the snapshot includes the working-tree changes. Upstream
Strata 0.1.40.3 was merged before this work. Existing research is retained;
older experiments are historical, not alternative supported defaults.

The tested environment is Linux, ROCm HIP 7.14.60850, gfx1201, hipBLASLt 1.4.1,
Python with NumPy 2.4.3, transformers 5.5.3 and tokenizers 0.22.2. No GLM
hipBLASLt tuning table was used. Keep GPU power policy automatic; the historical
manual 96 MHz memory-clock results are not the current profile. Idle 96 MHz
alone does not indicate a problem.

Build with the pinned llama.cpp revision
`3cf03257f219afbe7334045ff7c6a06ac68c627d` (CMake fetches it when GGML_DIR is omitted):

```sh
cmake -S . -B build-hip-glm-v11 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DCMAKE_BUILD_RPATH=/opt/rocm/lib
cmake --build build-hip-glm-v11 -j 4 --target strata-glm-decode strata-glm-assemble
python3 -m venv --system-site-packages .venv-glm-hip
```

Follow [AMD build requirements](AMD_HIP.md) for the installed ROCm toolchain
and [GLM API requirements](GLM53_FLASH.md) for Python dependencies. This is the
experimental GLM entry point, not the default Qwen installer/model menu.

Use the processed REAP50 expert pack with its original Q4_K_M base. The pack
alone is not a complete model. Assembly copies bytes without requantization:

```sh
build-hip-glm-v11/strata-glm-assemble ORIGINAL-Q4_K_M.gguf reap50-q23.gguf OUTPUT.gguf
sha256sum OUTPUT.gguf
```

The tested standalone model is 63,205,406,688 bytes (58.87 GiB), SHA-256
`c3b0cfb144556bd28e265d8255d48eeafce3d127aba5b9b5dd4e326322a34aa9`.
Weights are not included in the source candidate. Copy the model and matching
coverage JSON to paths in
`configs/glm53f-reap50-q23-b550-60g-9070xt-experimental.json`, or adjust its
`model` and `env.STRATA_GLM_EXPERT_PRIOR` fields. Keep guards enabled.
The tested B550 model path is `/mnt/disk01/models/glm53f-reap/`.

```sh
tools/glm_b550_60g.sh start
tools/glm_b550_60g.sh status
# Stop the API before any benchmark using the GPU.
tools/glm_b550_60g.sh stop
.venv-glm-hip/bin/python tools/glm_low_memory_bench.py \
  configs/glm53f-reap50-q23-b550-60g-9070xt-experimental.json \
  docs/fixtures/glm_cpp_review_20261008/perf-2048.ids \
  --output build-hip-glm-v11/reproduction --ram-gib 60 \
  --tokens 128 --trials 3 --timeout 900 --single \
  --gpu-capacity-mib 16304 --gpu-used-limit-mib 14256
tools/glm_b550_60g.sh start
```

The frozen performance IDs reproduce the original 2048-token benchmark, not
the separate C++ review fixture. Do not add `--fit-only` to qualification runs.
An untimed initial prefill precedes the measured warm prefills. SSD warmup and
cold first-request latency are separate costs. The API binds loopback only;
its current default context is 4096 and sampling is greedy (`temperature: 0`).

## TR16 reproduction (deferred while this host is busy)

The historical executable and archive hashes are preserved in
[the TR16 build manifest](fixtures/glm_cpp_review_20261008/tr16-historical-build.json).
It was linked against frozen local CUDA archives before the upstream/HIP merge.
Do not label a new build of this candidate as a repeat of that exact binary.
The CUDA source candidate has not been requalified during this release task.
For a new build, use CUDA 13.2 and architecture 120 as in the historical machine:

```sh
cmake -S . -B build-glm-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=ON
cmake --build build-glm-release -j 4 --target strata-glm-decode
```

Copy the selected v11 config to a local reproduction config. For REAP, set
`exe` to `build-glm-release/strata-glm-decode`. For the full model retain
`exe: tools/glm_numa_interleave.sh` and set its
`env.STRATA_GLM_NUMA_EXECUTABLE` to the absolute new binary path. Adjust model,
expert-pack and coverage paths to the original recipe; keep all precision,
NUMA, cache and MTP settings. The standalone B550 assembly is not the full model.

```sh
# Only after TR16's other task has finished; choose one model at a time.
python3 tools/glm_low_memory_bench.py LOCAL-REAP-CONFIG.json \
  docs/fixtures/glm_cpp_review_20261008/tr16-perf-8192.ids \
  --output build-glm-release/reap-long --ram-gib 60 --tokens 128 --trials 3 \
  --gpu-capacity-mib 16384 --gpu-used-limit-mib 14336
# Full model: use its config, --ram-gib 124, and a different output prefix.
```

Report the new executable hash, guard status and individual trials. A run with
reclaim/compaction must stay diagnostic. See the TR16 report for short-case
settings and the original model/sidecar paths in the checked-in profiles.

## Quality and release limits

The FP16 prefill comparison over 2048 teacher-forced targets measured perplexity
9.0374 → 9.1036 (+0.73%), mean KL 0.0351 and 91.85% top-token agreement versus
MMQ. FP16 prefill is not bit-exact. The MMQ reference configuration is retained.
This measures an incremental backend change, not the effect of pruning.
The earlier 32768-target REAP comparison measured perplexity 11.115 versus
8.566 for its reference; the REAP recipe remains experimental.

The new long-context C++ review procedure and raw answers are collected in
[the quality fixture](fixtures/glm_cpp_review_20261008/README.md). It checks
five seeded correctness bugs and a lifetime-safe control. B550 completed 6511
input / 1965 output tokens at 172.1 tok/s prefill and 10.6 tok/s decode. It
found all five bugs without a spurious defect finding, but its explanation and
test plan contain errors. Peak RAM was 57.18 GiB and minimum free VRAM 2816 MiB,
with no swap, limit hit or guard rejection. This longer first-request prefill
did not meet 200 tok/s. The 12K config is a test profile; the default stays 4K.
A single review
cannot establish general coding quality or replace a held-out benchmark.

Prior validation: six HIP GLM kernel tests, native Q3 expert parity at layers
3/11/44, five GLM API tests, eleven memory-harness tests, four assembler tests
and the warmup rejection fixtures passed. Runtime guards reject inadequate
headroom. Original model files and earlier research records are preserved.

This candidate is prepared locally. No tag, push, model upload, public release,
or upstream version bump is implied. The archive excludes models, generated
binaries, raw large logits, caches and unrelated local simulation work.

## Prepare and verify the source candidate

```sh
python3 tools/glm_prepare_release.py --output build-release-glm-20261008-rc1
(cd build-release-glm-20261008-rc1 && sha256sum -c SHA256SUMS)
```

The output contains a source tarball, per-file manifest, working-tree patch,
status inventory and SHA256SUMS. The archive includes new GLM files that a
Git diff alone would omit. The script refuses an existing output directory;
use a new candidate name after changes. It performs no commit, push or upload.
Check `excluded_untracked` in the manifest before publishing. Historical
experiments remain explicitly experimental; unrelated `tools/sim/` work is
excluded and untouched. Deployment weights and build products are excluded.

## RC2: long document-chat evidence

The follow-on [16K–128K chat screen](GLM_LONG_CHAT_QA.md) is included in RC2.
All cases fit the same 60 GiB / 2 GiB-reserve limits with a smaller 3072 MiB
GPU expert cache. At 128K capacity, 117955 input / 1056 actual output tokens
measured 173.4 tok/s prefill and 9.2 tok/s decode. The 13107-token output allowance
was not filled; this is not a sustained 13K-output qualification.

The 16K, 32K and 128K cases answered all eight factual questions correctly.
The 64K answer confused checksum agreement with crate-completion percentage,
and a supplementary follow-up repeated that error (five of six corrections
succeeded). A 32K next-action sentence also reversed a local-copy handling rule.
This remains an experimental candidate, with explicit model-quality limitations.
The source and evidence update is packaged separately as
`build-release-glm-20261008-rc2`; RC1 remains intact. Use that directory name in
the packaging/checksum commands above to prepare another copy of this revision.
