# Experimental DFlash2 bridge calibration

These tools calibrate the drafter's target-feature bridge. They do not change
the target checkpoint or the selected inference preset. No calibrated adapter
has a qualified repeated throughput gain or coding-quality result yet.

The optional worker argument `--fc-adapter ADAPTER_DIR` loads two FP32 low-rank
matrices. Their correction is added only to the `fc` feature projection.
The loader checks the base checkpoint identity, adapter checksum, dimensions,
dtype and finite values. Without this argument the worker uses its existing
path. Adapter weights are experiment artifacts; do not add them to Git.

The separate `--adapter-targets fc-mlp-down` calibration option adds residual
updates to each draft layer's MLP output projection as well as FC. Run the
result with `--linear-adapters ADAPTER_DIR`; it cannot be combined with
`--fc-adapter`. The loader resolves all target dimensions from the unpacked
base matrices before allocating adapters and checks the `linear.safetensors`
checksum, FP32 dtype and finite values. The two modes use separate manifest
schemas. Neither changes the base checkpoint files.

The opt-in `fc-mlp-down-selector` mode also adapts the selector's hidden
projection. `--selector-loss-weight 1 --prefix-decay 0.8` adds a teacher-forced
loss over the current top 16 candidates and gives earlier positions more
weight. Labels outside those candidates still train the unary head.
`--selection-metric selector-prefix` selects by the mean correct prefix from
the serving selector, then selector agreement and cross entropy. Evaluation
uses its own preceding choices; the selector loss uses preceding teacher IDs.
The default loss and its gradients remain ordinary cross entropy.

The worker's optional `--warm-protocol` materializes binary export and selector
work before readiness; it requires `--gemm-f32`. `--trace-protocol N` logs the
first N operations to stderr, with N limited to 64. Both are off by default.
On one V100, the source205 synthetic socket probe returned prepare, draft and
select in 171.55, 16.12 and 1.89 ms after warmup. This does not establish native
inference throughput or resolve the earlier native worker timeout.

## Head operands

Build the CUDA-only `strata-glm-fp4-export-head` target, then run:

```sh
strata-glm-fp4-export-head MODEL_DIR NEW_HEAD_DIR
```

The tool resolves `output.weight` through the normal native artifact loader.
It reconstructs the same FP16 operands used by the SM70 compressed head,
checks valid head output bits at widths one and eight, and writes row-major
`weights.f16`, `allowed.u8` and `metadata.json`. It refuses an existing output
directory. Invalid vocabulary rows are zeroed in this training copy and must
also be excluded from the training softmax. The checkpoint stays read-only.

On a Tesla V100 PCIe 32 GB, the GLM head exported 154,880 rows by 4,096 columns,
with 154,856 valid vocabulary rows. Its FP16 file is 1,268,776,960 bytes. Both
native output checks passed. This was an operand check, not a model timing.
The exporter is excluded from the HIP and SYCL builds.

## Capture contract

Calibration consumes committed protocol-2 feature and embedding captures, plus
the complete prompt and generated token IDs. It verifies file and frame
checksums, contiguous positions, anchor IDs, and the accepted proposal prefix.
The capture's first logical position can be nonzero.

For a draft request ending at position `S`:

| Input or label | Absolute token positions |
| --- | --- |
| Target-feature context | `S-2048` through `S-1` |
| Anchor embedding | `S` |
| Seven prediction labels | `S+1` through `S+7` |

Future target features must never enter the input. Samples without complete
past features or all seven labels are skipped. Four CPU tests check alignment
and corruption; a fifth checks a nonzero absolute feature origin.

## Calibration

Run calibration after the target process releases its GPU allocations:

```sh
python tools/glm_dflash2_calibrate.py \
  --dataset CAPTURE_DIR --head HEAD_DIR --head-proof HEAD_PROOF_JSON \
  --model DRAFT_MODEL_DIR --gemm-f32 GEMM_HELPER \
  --output NEW_ADAPTER_RUN_DIR --device cuda:1 \
  --rank 8 --alpha 8 --steps 256 --learning-rate 0.0003
```

The base drafter stays frozen. Linear forward products use its FP32-output
helper. Backward products use frozen half-weight values promoted to FP32;
attention uses serving half-output rounding with FP32 surrogate gradients.
This avoids FP16 output overflow and small-gradient underflow. A zero-adapter
comparison with the ordinary serving core must pass before any optimizer step.

Prompt indices whose remainder modulo four is three form the calibration
evaluation group. The run saves adapters every 64 steps and selects by that
group's token agreement, then cross entropy. These measurements do not establish
inference speed or general coding accuracy. Fresh coding holdouts are not
calibration data.

Twenty-four CPU tests passed for the worker, adapters and capture contracts.
Real-weight V100 checks matched the ordinary serving forward at context lengths
eight and 2,048, before and after one synthetic update. Both FC-only and FC/MLP
modes passed. Base weights stayed unchanged and every adapter received finite
gradients. Packed skinny forward had relative L2 error 0.00254 for FC-only and
0.00340 for FC/MLP against their differentiated references on these probes;
neither was bit-identical. Probe peak reserved memory was 10,085,203,968 and
10,601,103,360 bytes respectively. These checks do not measure model tok/s.

The first calibration used rank 8, 256 optimizer steps, 1,064 training windows
and eight held-out synthetic windows. On one V100 it took 102.49 seconds.
Agreement rose from 28/56 to 31/56 labels, and cross entropy fell from 2.944
to 2.432. This small evaluation group does not establish general accuracy.

A direct-worker depth-three screen on two V100 PCIe 32 GB cards with 181 GiB
RAM measured 30.35 / 32.02 decode and 503.66 / 504.54 prefill tok/s for the
base / calibrated drafter. Each used three full warmups and one timed
512-output request. Target and actual GPU-worker residency checks passed.
The 5.47% decode increase is a single-request screen; separate repeated timing
and coding checks remain required. It is below the earlier 36.33 tok/s
candidate that passed both reused coding panels, and below the 50 tok/s goal.
At depth seven, the same protocol measured 23.95 / 24.97 decode and
504.01 / 503.25 prefill tok/s. This longer draft was slower than depth three.

Larger calibration runs on the same eight evaluation windows reached 34/56
unary agreement with rank-32 FC and 35/56 with rank-8 FC/MLP adapters. An
offline packed-worker selector check instead preferred FC/MLP step 320, with
mean correct prefix 3.875 versus 3.375 for the base worker. These small proxy
results do not establish a native speed increase.

The source205 seven-adapter synthetic GPU probe passed, but its first real
selector-aware training run failed before an optimizer update: evaluation
cached an inference-only frozen operand that autograd could not reuse.
The fix creates a normal frozen operand even when evaluation initializes it.
All 31 CPU tests pass, including evaluation followed by training. The revised
real-weight V100 check also passed that transition, serving equality, finite
gradients for every adapter and unchanged base weights. Packed skinny relative
L2 error was 0.00276; peak reserved memory was 10,661,920,768 bytes.
The failed source and artifacts are retained.

The corrected selector-aware run completed 512 rank-8 updates in 187.09
training seconds, or 266.56 seconds including preparation. Selection chose
step 320: mean correct prefix rose from
3.375 to 3.875 on the same eight calibration windows, with 31/56 selected
tokens matching the teacher.

Its matched native screens measured 30.15 / 30.07 decode tok/s at depth three
and 23.90 / 25.00 at depth seven, base / trained. Each side used three full
warmups and one timed 512-output request on the same two V100s and 181 GiB
allocation. Target and worker residency checks passed. At depth three,
accepted proposals fell from 343 to 340. At depth seven, rounds fell from
147 to 141 and verification time from 18.97 to 17.99 seconds, but the longer
block remained slower than depth three. Neither pair qualifies for promotion;
coding quality was not scored. The calibration proxy does not establish a
native speed gain.

With protocol warmup enabled, the earlier rank-32 FC adapter completed a
matched depth-three screen at 30.06 / 33.39 decode and 504.05 / 504.20 prefill
tok/s, base / trained, on the same two V100s and 181 GiB allocation. Both
used three full warmups and one timed 512-output request and passed residency
checks. This 11.1% single-request gain remains below the 36.33 tok/s candidate
that passed both reused coding panels. Repeated timing and coding quality for
rank-32 remain unqualified.

Each adapter still needs repeated native inference timing, target and worker
residency checks, and separate coding checks. Freeze a final policy before
scoring fresh holdouts. Keep original baselines, failed runs and all artifact
identities intact.
