# Experimental GLM Q2 expert conversion

This is an opt-in experiment for GLM-5.3-Flash UD-Q2_K_XL on the
Threadripper 1950X, two NUMA nodes, 128 GB RAM and RTX 5060 Ti 16 GB.
The original GGUF files and existing server configurations are unchanged.
**15 delivered tokens/s has not been demonstrated by this conversion.**

## Implemented path

`strata-glm-q2-pack` converts eligible routed projections into a separate GGUF:
IQ2_XS gate/up becomes Q2_K, and IQ3_XXS down becomes Q3_K. Layer 11's
IQ3 gate/up and IQ4_XS down exceptions keep their original representation.
Dense, attention, router and MTP weights stay in the original files.

`--fp8=DIR` converts every routed layer in range from the FP8 release
(zai-org/GLM-5.3-Flash, `strata/artifact/fp8_experts.hpp`: E4M3 times the 128x128
`weight_scale_inv`, rounded to f32) instead of requantizing the GGUF's own 2-bit
experts. With it, `--gate-type` (gate and up) and `--down-type` take any of
q2_k, q3_k, q4_k, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s and iq4_xs; sizes come
from `ggml_row_size`. The pack carries `strata.expert_pack.fp8`, and only for
such packs the loader accepts every type with native kernels in place of the
GGUF's expert types. Without `--fp8` the conversion and the accepted types are
unchanged. Layer 10 to IQ2_S with the calibration importance (`--calibration`) took
4.4 min on 14 threads; importance-weighted normalized MSE against FP8: gate/up
0.062, down 0.045. IQ2_S experts run on the non-canonical CPU, GPU-tier and worker
paths; the canonical sum (`STRATA_GLM_CANON`) is Q2_K/Q3_K only.

`--resume` continues an interrupted conversion. Each finished projection is recorded in `OUTPUT.partial.progress`
(name, checksum, error terms) after its payload is synced. A resumed run must have the same plan: source
fingerprint, options and layout. It rewrites only the projections that are not recorded. A layer-10 IQ2_S pack killed
with SIGKILL during its second projection and resumed was byte-identical to an uninterrupted one. Without `--resume`
a partial file is refused, as before.

### Full IQ2_S pack from FP8 (TR16, 2026-10-16)

All routed layers 3-44, gate/up and down IQ2_S, with the calibration importance:

```
build-glm-release/strata-glm-q2-pack GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf experts-iq2s.gguf \
  --calibration=prune/calibration --fp8=fp8 --gate-type=iq2_s --down-type=iq2_s --threads=14
build-glm-release/strata-glm-q2-pack --verify GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf experts-iq2s.gguf
```

The pack has 126 projections of 773,849,088 bytes each, 97.5 GB in total, against 111 GB for Q23 (Q2_K gate/up, Q3_K
down). It took 2.7 h on 14 threads, with the llvq GPU encoder running beside it. `--verify` read all 126
projections back and checked their checksums in 3 minutes.

Importance-weighted normalized MSE against FP8, per projection over the 42 layers:

| Projection | Mean | Min | Max |
|---|---:|---:|---:|
| gate | 0.0613 | 0.0590 (layer 44) | 0.0617 (layer 6) |
| up | 0.0613 | 0.0590 (layer 44) | 0.0617 (layer 5) |
| down | 0.0466 | 0.0142 (layer 19) | 0.0576 (layer 23) |

Gate and up are uniform across layers. Down varies by layer: the lowest values are layers 19, 3, 12, 16 and 27
(0.014-0.031), and most layers are 0.045-0.058.

Corpus gate (`tools/glm_tr16_quality.py --eval-prefill`, 16 held-out sequences, 8,192 tokens), run on the
RX 9070 XT with the same non-canonical settings for both models:

| Model | Experts | Perplexity | KL | Top-1 |
|---|---|---:|---:|---:|
| UD-Q2_K_XL (the reference itself) | unsloth's | 4.645 | 0 | 100% |
| Q23, assembled | Q2_K/Q3_K from the UD-Q2_K_XL experts | 4.850 | 0.1471 | 86.99% |
| IQ2_S | from FP8 | 4.673 | 0.1750 | 86.19% |

IQ2_S passes the `tr16-35` gate against Q23 (perplexity within +6%, KL within +0.08, top-1 within 3 points): its
perplexity is 3.7% lower than Q23's and 0.6% above the reference model's, with KL 0.028 higher and top-1 0.8 points
lower. The Q23 run was repeated after a GPU reset and gave the same values to every printed digit (the evaluation is
deterministic).

KL and top-1 are measured against the original UD-Q2_K_XL model, not against FP8. A pack that is closer to the
original weights than UD-Q2_K_XL still shows a KL against that reference, so perplexity is the direct quality
comparison. The 64-sequence reference from 2026-10-06 (CUDA build) was deleted. It was regenerated for these 16
sequences with the HIP build, using the same arguments, so these numbers are not bit-comparable with the
2026-10-09 baseline (KL 0.1367).

The runs used `--allow-shared-gpu` and `--max-swap-mib 1024` (new, opt-in options of `tools/glm_tr16_quality.py`):
another job used the GPU part of the time, and with a full page cache the kernel swapped a few MiB of the mapped
model. Neither changes the evaluation's numbers, but these runs are not timing results. HumanEval and decode speed
for the IQ2_S pack were not measured.

The loader checks the source identity, names, shapes, paired gate/up formats,
payload bounds and per-tensor checksums. Source identity is an FNV-64 fingerprint
of shard sizes, modification times and headers; it is not a cryptographic hash
of every original weight. Converted payloads have independent FNV-64 checksums.
Conversion refuses to overwrite either an existing pack or a partial pack.

Each expert's output rows are split between two node-bound arenas. GPU prefill
stages the same converted bytes from those arenas. Hidden activation
quantization runs with the gate/up row tasks. A fixed-order parallel reduction
sums routed outputs, followed by an asynchronous upload from pinned memory.

`--cpu-expert-backend=auto` currently selects GGML's native Q2_K/Q3_K row kernels.
The custom AVX2 shared-token kernel (`STRATA_Q23_AVX2=1`) and reversible 32-row
tiles (`packed-dot` and `packed-lut`) remain experimental alternatives.
`STRATA_GLM_PACK_FUSE_QUANT=0` disables fused hidden quantization for comparison.
`STRATA_GLM_PACK_HUGE=1` requests aligned, transparent-huge-page arenas; actual
huge-page backing and a whole-model speed benefit must be measured separately.

The sidecar automatically enables owned NUMA rows and requires pinned workers
on two nodes. The default packed-weight cap is 112 GiB, with a further 16 GiB
host-capacity check. TP, CPU format prepacking, direct upload and the legacy
expert cache are incompatible with this path.

## Calibration and quality tools

Use the Python environment that provides Strata's tokenizer and server
dependencies. Example shell variables below refer to the original model and a
new output path on a volume with at least 110 GiB free:

```sh
MODEL=/mnt/nvme01/models/glm53f/q2/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
PACK=/mnt/nvme01/models/glm53f/q2-cpu-q23.gguf
PY=build-exl3-venv/bin/python
WORK=build-q2-redesign

$PY tools/glm_q2_corpus.py "$MODEL" "$WORK/corpus"
STRATA_GLM_CALIBRATION_DIR="$WORK/calibration" STRATA_GLM_LOCK_RUNTIME=1 \
  $PY tools/glm_q2_run_guard.py --output "$WORK/calibration-run" -- \
  numactl --interleave=all build-glm/strata-glm-decode "$MODEL" 0 1 4096 15 \
  --context=8192 --prefill-batch=256 --gpu-budget-mib=12288 \
  --cpu-affinity=numa --decode-graphs --eval-corpus="$WORK/corpus/calibration.ids"

build-glm/strata-glm-q2-pack "$MODEL" "$PACK" \
  --calibration="$WORK/calibration" --threads=16
build-glm/strata-glm-q2-pack --verify "$MODEL" "$PACK"
$PY tools/glm_q2_pack_profiles.py "$PACK.quality.json" "$WORK/profiles"
```

The corpus tool freezes 32,768 calibration and 32,768 evaluation targets, half
code and half prose, from committed repository files. Splits are disjoint by
file/content hash. Its manifest records revision, source hashes and token-file
hashes. This is an engineering corpus, not a general model benchmark.
Calibration collects per-expert squared input moments for gate/up and down.
Unvisited experts use the layer's mean; `coverage.json` records visit counts.
The converter uses those moments in GGML's weighted quantizer and reports
weighted reconstruction errors. Sixteen converter workers use less than
513 MiB of float scratch, plus packed output buffers.

The profile generator produces full Q2/Q3, gate/up-only Q2, and profiles that
retain 25%, 50% or 75% of projections, prioritizing reconstruction error. Gate
and up are retained together. These rankings propose quality experiments;
they do not establish a quality threshold. The `original` profile retains every
projection, allowing the new execution path to be checked without changing weights.

For held-out evaluation, run the decoder with `--eval-corpus=.../evaluation.ids`
and `--eval-save-logits=/large-volume/baseline.logits` on the original model.
Then use `--eval-reference=/large-volume/baseline.logits`, `--expert-pack=...`
and optionally `--expert-pack-profile=...` for each candidate. The evaluator
reports teacher-forced NLL/perplexity, full-vocabulary KL from the baseline and
top-1 agreement. Full FP32 baseline logits need approximately 20 GiB. Do not
place them on a small build volume. Sequence IDs and lengths must match.

`tools/glm_q2_humaneval.py` implements one greedy chat sample per task, using
HumanEval revision `6d43fb980f9fee3c892a914eda09951f772ad10d` and a pinned dataset
SHA-256. Generated Python runs in a network-disabled Bubblewrap namespace with
only read-only `/usr`, a temporary filesystem, resource limits and a timeout.
It refuses to execute generated code if the isolation self-test fails. Its
chat prompting differs from the original raw-completion HumanEval protocol.
Use the same token cap and protocol for baseline and candidates, and require
all 164 tasks before treating the reported fraction as a complete pass@1 run.

## Validation and performance

The CPU tests compare Q2_K/Q3_K dots with a scalar dequantized reference,
including zero/extreme inputs, widths 1–4 and partial row ranges. Tile
pack/unpack is byte-exact. Owned rows, fused/unfused hidden quantization,
fixed-order reduction and huge-page alignment have separate pool checks.
`prefill_mmq_q23_test` covers permuted rows, multiple experts, zero input and
tail batches at the model's actual matrix dimensions. The test found a
Q2_K activation-padding read; initializing the padding eliminated non-finite
outputs and passed Compute Sanitizer's uninitialized-memory checks.

The completed 32K calibration used all 84 gate/up and down moment tables, with
262,144 routed observations per table. Two expert moment rows were unvisited
and used the documented fallback. Peak sampled RSS was 100.23 GiB, with zero
process swap. A 16-target control retaining every original weight matched the
saved baseline's evaluation metrics through the new owned/fused/asynchronous
path (KL 0, top-1 agreement 100%). These are correctness checks, not speed
measurements; conversion was running concurrently with the control.
The layer-3-converted verification graph check passed bitwise logit equality
for widths 1–4, repeated replay, history reallocation and accepted-prefix
rollback, with zero sampled swap.

The strict one-token-versus-17-token GPU prefill check failed its unchanged
2% logit/state threshold for both original and layer-3-converted weights:
logit relative L2 was 10.26% and 10.05%, respectively. Both selected token
1704. This check is not reported as passing; agreement on one greedy token
does not establish state parity. The layer-3 conversion also failed the
128-versus-256-token chunk check (13.64% logit relative L2, same greedy token
1424); that chunk comparison has not yet been repeated with original weights.
See the [validation record](glm53_flash_q23_validation.json).

An initial layer-3 experiment on this host used real routed weights, synthetic
activations, width 3, 15 workers plus the host, owned NUMA rows and 20 rounds:

| Kernel | Median packed traffic rate |
| --- | ---: |
| Original IQ formats | 40.09 GB/s |
| GGML Q2_K/Q3_K rows | 66.57 GB/s |
| Custom shared-token Q2_K/Q3_K AVX2 | 52.09 GB/s |
| Exact int16 tile lookup | 22.48 GB/s |

These are preliminary expert-only results, not delivered-token rates or a
matched full-model speedup. These initial runs had no continuous external-work
guard. [Individual rounds](glm53_flash_q23_kernel_measurement.json) are saved.
The direct bitplane tile also needs measurement.
Real-activation replay accepts traces from `STRATA_GLM_ACTIVATION_TRACE` or
`glm_q2_coding_bench.py --capture-experts`; CSV traces use synthetic inputs.

Use `glm_q2_coding_bench.py --expert-pack=... --mtp-sweep --repetitions=3`
for ordinary decode and MTP depths 1–3, across all three coding fixtures.
The harness checks greedy IDs across modes within each profile. Quantized
candidates may legitimately differ from the original model, so quality must
be checked separately. Alternate baseline/candidate runs and use
`--reject-competitors` for timing. Retain EOS-aware delivered-token counts,
acceptance, CPU expert time and memory samples, not just the fastest trial.

`glm_q2_run_guard.py` rejects any sampled process swap, aggregate experiment
RSS above 118 GiB or host available memory below 4 GiB. It includes child
processes, samples every 100 ms, and can reject sustained external CPU work
with `--require-idle`. Calibration quality runs need not be idle, but their
wall times are not performance results. The optional Linux runtime lock
protects small writable mappings within the existing memlock allowance;
it does not change system swappiness or exempt a run from the swap guard.

## Long qualification run

`tools/glm_q2_qualify.py` runs conversion, payload verification,
a full-pack graph check and short prime/MTP screen, all five
held-out profiles, alternating coding trials, isolated C++ correctness checks,
and full HumanEval for the original, Q2 and Q2/Q3 profiles. It waits for the
calibration guard record and checks all 84 projection counts before starting.
It needs exclusive GPU access and a large output volume. Keep the binaries
fixed during the run: completed-stage markers include executable hashes and
settings, and are reused only when these match. Failed or partial stages stop
the run; they are not silently reused or promoted.

```sh
$PY tools/glm_q2_qualify.py "$MODEL" \
  --work="$WORK/qualification" --store=/mnt/nvme01/models/glm53f/q2-cpu-experiment \
  --calibration-record="$WORK/calibration-run.memory.json" \
  --calibration="$WORK/calibration" --corpus="$WORK/corpus" \
  --dataset="$WORK/HumanEval.jsonl.gz"
```

For a local API that was unloaded for the experiment, optional
`--restore-api=http://127.0.0.1:8095` reloads its existing configuration on exit,
checks health and requests a short completion. The runner does not unload a
service itself or install a converted profile. `status.json` records stage
progress; successful full runs write `measurement.json`. Individual guard
records, model logs, generated answers and correctness results remain beside it.

`--defer-api-unload` with `--restore-api` keeps the existing API loaded during
CPU-only conversion and payload verification, then unloads it before model
validation. The short prime screen uses 128 tokens and one trial; it does not
replace the complete timing and quality matrix. Runtime locking also covers
the converter and verifier when `STRATA_GLM_LOCK_RUNTIME=1`. An earlier full
conversion was rejected for 16 KiB of swapped heap and did not publish a pack.
