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

Each adapter still needs repeated native inference timing, target and worker
residency checks, and separate coding checks. Freeze a final policy before
scoring fresh holdouts. Keep original baselines, failed runs and all artifact
identities intact.
