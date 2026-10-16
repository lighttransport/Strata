# Quantization research probes: sm_86 and Zen 2

These standalone probes implement the three follow-ups from the quantization
survey. They do not modify the engine, convert models, or reproduce published
model-quality or accelerator-speed results.

## Coverage

`bench_research_cpu` and `bench_research_gpu` independently encode the same seeded
Gaussian source matrix with channel outliers for each format:

| Probe | Stored weights and metadata | Timed execution |
|---|---|---|
| EXL3 K2/MCG | cyclic 16x16 packed tiles; FP32 scale per output row | FP32 decode, input Hadamard |
| EXL3 K2/mul1 | independently encoded cyclic tiles; FP32 scale per output row | exact FP16 codebook reconstruction with FP32 input; separate q8 affine probe |
| XOR-Trellis | cyclic two-bit transitions, four palettes; one positive E5M3-style scale per row/group of 32 | packed FP32 GEMV, no rotation |
| scalar HIGGS-style INT4 | Lloyd-Max Gaussian grid, packed four-bit indices; FP32 RMS scale per row/group of 128 | lookup GEMV, input Hadamard |
| Qift INT2 | packed indices for +/-0.5 and +/-1.5; FP32 RMS scale per output row | lookup GEMV, input Hadamard |

The cyclic trellis encoder makes three greedy sweeps, then packs the resulting
two-bit branches. Runtime states are extracted from that packed ring. The first
seven states can change at the wrap boundary; reported weight NMSE includes that
effect. This is a bounded diagnostic encoder, **not** upstream Viterbi/LDLQ,
curvature-aware XOR-Trellis quantization, or a calibrated EXL3 converter. Its
quality cannot rank those algorithms. MCG bytes are never reinterpreted as mul1.

`--encoder viterbi` enables exact dynamic programming over all 65536 states with
a fixed boundary chosen from the greedy ring. The last eight transitions are
constrained to close the ring, and traceback verifies the initial/final state.
The objective includes the stored scales. It does not search all possible
boundary states and does not implement LDLQ feedback. `--curvature` additionally
weights XOR reconstruction error by input second moments from eight separate
seeded calibration vectors. This is a diagonal sensitivity proxy, not the
diagonal D of an LDL factorization. Encoding is outside the timing region and
can be expensive on full matrix shapes. `--format NAME` selects one format;
`--rows N` permits small reference cases with output widths divisible by 16.

`--boundary-trials N` tries 1..8 fixed boundaries per tile with Viterbi. Trial
one is the existing greedy boundary; remaining trials use zero and deterministic
XOR offsets. Every path closes its own ring. Selection evaluates the full weighted
reconstruction objective of the closed path, and retains the original candidate
if no alternative improves it. This is bounded boundary search, not globally
optimal tail-biting quantization. Default N is one.

`--xor-scale-search` then fits each XOR group scale against its fixed decoded
palette values. A least-squares estimate proposes a stored byte; five nearby
valid bytes are compared with the original byte using actual source reconstruction
error, weighted by the supplied diagonal when present. The payload stays packed
and the total storage budget is unchanged. The original byte remains a fallback.
This improves only the declared source/diagonal objective, not necessarily
held-out projections or model quality. Boundary and scale searches currently
apply to the independent encoder and cannot be combined with LDL feedback.
The runner records baseline, four-boundary and four-boundary-plus-scale results
separately for CPU and CUDA.

`--calibrate-xor-scales` fits XOR scale bytes to calibration **projection** error,
with fixed packed palette states. For each output row it forms group projection
features from supplied activations, then performs three coordinate sweeps over
stored scales. Each coordinate tries nearby bytes around a least-squares proposal
and retains its current byte unless the full row projection residual improves.
Interactions between groups are included; this differs from source-only fitting.

This mode requires separate `--calibration-f32` and `--validation-f32` inputs.
The candidate is accepted only if training projection MSE does not increase and
paired validation gain exceeds two standard errors. It otherwise returns the
original matrix. It always uses that paired rule; `--validation-guard` controls
the LDL tuner separately. Reports preserve candidate training/validation loss,
gain/standard error, changed scale count, acceptance and baseline held-out error.
Held-out data is not used during fitting or acceptance. Payload bytes, decoding
kernels and storage size remain unchanged. This mode is currently independent
of LDL feedback and remains an opt-in experiment.

`--ldl-feedback` adds a stronger XOR-only calibration experiment and requires
`--calibration-f32`. It forms input covariance in independent 128-channel blocks,
adds damping (`--hessian-reg`, default 0.01 times each block's mean diagonal),
and factorizes each block as `H = L D L^T` in double precision. It runs input
trellis tiles from high to low and propagates finalized reconstruction error
through L to earlier tiles. Viterbi uses the actual D diagonal and fixed stored
scales, with the same fixed boundary as its baseline.

Within a 16-input trellis tile, the path objective remains diagonal; feedback
does not optimize every within-tile correlation. Cross-128-block covariance is
also omitted. This is an LDLQ-style block-feedback approximation, not a full
XOR-Trellis paper reproduction or model converter.

The candidate is accepted only when its **calibration** projection MSE is below
the D-weighted fixed-boundary Viterbi baseline's MSE. Reports include
`ldl_baseline_mse`, `ldl_candidate_mse`, `ldl_selected` and the selected weights'
`calibration_projection_mse`. This guard does not promise improvement on the
separate held-out activation or any model task. Hessian construction, factorization
and encoding stay outside GEMV timing; no Hessian is needed during inference.

`--tune-ldl --validation-f32 FILE` adds a declared damping sweep at 0.01, 0.1
and 1.0. Each setting contributes its baseline and calibration-accepted feedback
encoder, for six evaluations. Selection uses validation projection MSE and
includes the first baseline as a fallback. Final held-out activations are not an
argument to the tuner and are never used for selection. Supplied training,
validation and held-out vectors are checked for exact-vector overlap. This does
not guarantee improvement on held-out data.

Reports add the baseline/selected validation MSE, selected damping, whether
feedback was chosen and the trial count. The original `--ldl-feedback` remains
available separately; all calibration modes are opt-in. The supplied
`--hessian-reg` applies to the single-setting feedback mode; tuning uses its fixed
sweep instead. Runtime GEMV and stored weight formats do not change with tuning.

`--validation-guard paired` is an optional conservative selector; the existing
`mean` rule remains the default. It computes per-vector loss differences against
the fixed first baseline and requires positive mean improvement greater than
twice its standard error before replacing that baseline. It then selects the
lowest validation MSE among eligible candidates. Reports include gain, standard
error and the count of lower-mean candidates rejected by the guard. At least two
validation vectors are required.

This is a declared 2SE heuristic, not a formal confidence guarantee: multiple
candidates are evaluated and real activation samples may be correlated. It can
reject useful improvements, and it still cannot guarantee better held-out or
model-level quality. Candidate selection does not read the held-out vectors.
The runner preserves mean-only and paired-guard results in separate reports.

Rotation is an orthonormal 128-wide Hadamard on both the source weights and
input. HIGGS additionally applies deterministic seeded random signs before the
transform and counts the four-byte seed in storage; vector grids remain absent.
Qift omits the paper's W4 activation/KV pipeline and full calibration recipe.
EXL3 uses a simplified per-row scale rather than a complete model export's
two-sided scales and rotations. These differences are deliberate baseline scope.
The XOR scale emulation uses unsigned normal values
`(1 + mantissa/8) * 2^(exponent-15)` and nearest rounding, reserving byte zero for
exact zero. All 256 byte values are finite; this is a declared experimental
storage convention, not a native FP4 GEMM
format or a file format compatible with upstream code.

`--scale fit` adds alternating nearest-level assignment and least-squares scale
fitting for Qift and scalar HIGGS. It tries six initial scale factors and up to
16 fitting steps each, retaining the lowest source reconstruction error including
the RMS baseline. `--scale both` reports RMS and fitted variants separately. The
default is RMS for comparison with earlier probes. This is scalar reconstruction
fitting, not model-loss optimization or the paper's complete quantization recipe.

The mul1 q8 path uses the existing AVX2 integer tile-dot probe on CPU and an
unsigned-byte dp4a reconstruction on GPU. It includes input quantization time.
It approximates the exact rounded FP16 codebook with its affine form; activation
rounding and codebook rounding are both included in `own_weight_rel_l2`.
FP32 paths use a diagnostic AVX2 decode/FMA baseline on CPU and warp GEMV on GPU.
They are initial kernels, not claimed optimized implementations.

`--cpu-kernel simd` adds AVX2 packed-tile FP32 paths for MCG, mul1 and XOR-Trellis;
`both` reports them alongside the reference. MCG/mul1 use shared 65536-entry
FP32 decoder tables with gather; XOR uses vector parity folding and a 16-entry
signed-byte shuffle table. The reference remains the binary's default; the runner
requests both. Each path reads the same packed weights/scales and accumulates
over 16-input tiles. Scalar and SIMD reductions can round differently.
MCG/mul1 decoder tables use 256 KiB each of shared runtime workspace, initialized
during warmup and excluded from serialized per-matrix `total_bytes`. These probes
do not expand a matrix to FP32.

`--gpu-kernel packed` adds CUDA alternatives for XOR-Trellis and Qift; `both`
reports them alongside the reference, which remains the binary's default.
XOR computes the same parity-derived palette index and uses byte permutations
over four packed signed-byte tables instead of dynamic float-array indexing.
Qift loads one packed byte and four FP32 activations per lane, decodes the fixed
four levels in registers and applies the per-row scale after reduction. The
Qift path supports both RMS and fitted scales. Accumulation order differs from
the reference. The runner requests both variants.

`--check-gpu-decoder` exhaustively compares all 65536 XOR states decoded by the
CUDA byte-permutation path with the scalar codebook. This check bypasses timing
and projection execution; it requires the CUDA binary.

All formats now represent exactly zero weight groups with zero scale, and reject
nonfinite transformed source values or overflowing scales. The runner includes
separate CPU/CUDA checks for zero weights with nonzero input and zero activations
with nonzero weights, including the q8 activation quantizer's zero-peak case.

Smoke mode uses 128x128 matrices, one corpus copy and one repetition. Full mode
uses 2048x4096 gate/up and 4096x2048 down, 64 physical copies of one encoded
matrix and 20 passes. Payload copies rotate outside cache; metadata stays shared
and warm. Encoding, reference calculation and uploads are outside timing. Input
transforms, activation quantization, CPU dispatch and GPU synchronization are
inside timing. No expanded weight matrix is used by the timed kernels.

`total_bytes` includes packed weights, scales and scalar lookup grids. CSV error
columns compare kernel outputs with an FP64 projection of **their own decoded
weights**, while `weight_nmse` compares with the source weights in the same basis.
Neither establishes model quality. Smoke checks require finite output and successful
exit; numerical error metrics are diagnostics, not tolerance qualification.

`source_projection_rel_l2` compares the final kernel output to a direct FP64
projection of the unquantized input matrix and activation. This includes weight
quantization, transforms and activation approximation; `own_weight_rel_l2` instead
isolates execution error against the decoded weights. `--report-csv FILE` writes
these diagnostic metrics without timing columns and is allowed in smoke mode.

`--heldout-f32 FILE` evaluates one or more separate input vectors against the
original source matrix. It records aggregate projection MSE/relative L2 and
worst-token relative L2; the reader rejects vectors also present in the supplied
calibration set. These are untimed FP64 projections of the decoded weights and
exclude the q8 activation approximation and runtime kernel reduction. The
evaluation may temporarily expand decoded weights on the host, outside GEMV
timing. For LDL feedback it also evaluates the exact D-weighted baseline on the
same held-out vectors. Held-out data never participates in candidate selection.

## Supplied weights and activations

Both research binaries accept raw **little-endian float32** inputs:

```sh
/tmp/strata-fp4-sm86/bench_research_cpu \
  --weights-f32 /tmp/weights.f32 --cols 4096 --rows 2048 \
  --activation-f32 /tmp/heldout-activation.f32 \
  --format qift2 --scale both --report-csv /tmp/qift-diagnostic.csv
```

Weights are `[rows,cols]` in row-major order. An activation contains exactly
`cols` values; calibration contains one or more `cols`-wide vectors. Input widths
must be divisible by 128 and at most 16384; output widths must be divisible by 16
and at most 4096. Files must be finite, nonempty, at most 512 MiB and exactly match
the declared geometry. Missing/short/misaligned/nonfinite files fail explicitly.
Calibration is used only by the XOR diagonal proxy; `--curvature` is required
when supplying it. Omit calibration for other formats. Keep calibration vectors
separate from held-out activations.

Without supplied activations the input remains a synthetic sine vector. Supplied
weights therefore do not imply real workload calibration or model-quality
qualification. This reader does not load GGUF/safetensors or dequantize existing
checkpoints; export the intended original source matrix as float32 first. Source
files are only read and are never modified or uploaded.

The runner creates a small synthetic float32 fixture with separate calibration
and held-out vectors, plus SHA256 hashes, in its output directory. CPU/CUDA
file-input cases independently quantize that fixture with fixed-boundary Viterbi
and test both scale variants. These smoke cases exercise the real-input path but
do not use actual model weights.
Each fixture additionally includes 16 held-out vectors, distinct from its eight
calibration vectors and the single runtime activation. File, wider-shape and LDL
checks report these multi-vector metrics.
Separate seeded files add 96 training and 32 validation vectors for tuning.
Their generation preserves the original weights, eight-vector calibration and
held-out fixture bytes, so the earlier overfitting case remains reproducible.

## GyRot-inspired accuracy study

`gyro_study.py` needs only Python's standard library. It sweeps rotation scopes
0/32/128/512/1024 (limited by input width), groups 32/64/128, symmetric and
asymmetric W4A4 RTN, and FP32 versus integer metadata. A calibration-energy
permutation places outlier channels at Sylvester harmonic rows whose signs are
constant within each quantization group. We label this `harmonic_slots`: it is a
proxy for HAP, not an official implementation.

Eight calibration activation vectors determine the permutation; separate held-out
vectors determine error. Both weights and activations receive the same permutation
and rotation, and unquantized projection error checks this change of basis.
Integer metadata uses an unsigned byte for scale and a signed ceiling-rounded
byte for the unscaled activation bias, with shared power-of-two quanta. The study
measures weight/activation NMSE and projection relative L2; it does not implement
GPTQ, KV quantization, the GyRot accelerator, or model-level evaluation. Storage
estimates explicitly exclude permutations and activation metadata. No timings are
collected.

The default study uses 1024 input channels, eight output rows and two held-out
tokens, so the smoke check covers the G32/R1024 design point. The larger extent
uses 32 output rows and 16 held-out tokens.

Optional `--weights-json FILE` accepts a finite `[output,input]` matrix, with an
input width that is a power of two and at least 128. Activations remain synthetic,
so this still does not qualify an actual model or workload.

## Running

```sh
cmake --build /tmp/strata-fp4-sm86 -j 2
python3 experiment/fp4/scripts/bench_sm86_zen2.py \
  --build /tmp/strata-fp4-sm86 --output /tmp/strata-quant-research-smoke
# Structural/small-fixture checks without the larger legacy workloads:
python3 experiment/fp4/scripts/bench_sm86_zen2.py --checks-only \
  --build /tmp/strata-fp4-sm86 --output /tmp/strata-quant-packed-wide-checks
# Accuracy-study sweep only, with no performance timings:
python3 experiment/fp4/research/gyro_study.py \
  --benchmark --output /tmp/strata-gyro-study
# Small fixed-boundary Viterbi check, without performance qualification:
/tmp/strata-fp4-sm86/bench_research_cpu --encoder viterbi \
  --rows 16 --format xor_trellis --curvature
# Supplied calibration covariance and block feedback:
/tmp/strata-fp4-sm86/bench_research_cpu \
  --weights-f32 /tmp/weights.f32 --cols 128 --rows 16 \
  --activation-f32 /tmp/heldout-activation.f32 \
  --calibration-f32 /tmp/calibration.f32 --format xor_trellis \
  --ldl-feedback --report-csv /tmp/ldl-diagnostic.csv
```

The main runner includes the earlier MXFP4/K-quant/MCG probes, both new research
binaries, and the GyRot study. Add `--benchmark` to the runner only on an idle
machine to collect repeated timing CSVs and the larger study extent. Use a fresh
output directory for each run. The original default Blackwell build also compiles
these CUDA probes; HIP/SYCL sources and engine paths are untouched.
`--checks-only` runs only structural and small fixture processes and cannot be
combined with `--benchmark`. Fixture checks cover 128x16 and 256x32 projections,
including reference/AVX2 CPU and reference/packed CUDA variants.
The runner writes `diagnostic-summary.md` after successful execution. It lists
storage budgets and held-out errors without timing columns or a cross-bitrate
winner, and reports LDL's held-out change separately from its calibration guard.
The summary can also be regenerated with
`python3 experiment/fp4/research/summarize_checks.py --input OUTPUT_DIR`.

The runner also executes `check_research_formats` and small CPU/CUDA Viterbi
cases. Structural checks cover every packed word/ring boundary with independent
branch extraction, fixed XOR palette goldens, recovery of an exactly representable
closed ring, ordered/symmetric Lloyd-Max levels, RHT projection invariance and
AVX2 MUL1 integer parity. These checks establish implementation properties, not
the quantizer's model quality or speed.
Hessian checks reconstruct a damped rank-deficient covariance from L/D and
verify the quadratic-form identity independently of the encoder.

Additional checks exhaust all 65536 XOR states through the AVX2 decoder, compare
SIMD and reference projections with varying row/group scales, require exact
zero reconstruction and verify rejection of source-transform overflow.

Execution check on 2026-10-13: Ryzen 9 3950X (Zen 2), RTX 3070 8 GB (sm_86),
GCC 15.2 and CUDA 13.4. Both the sm_86 and default sm_120a configurations
compiled. All ten runner processes exited successfully while the miner was
active; the GyRot study also completed at input width 1024. Only compilation,
execution and finite diagnostics were checked. Full timing mode, real-model
conversion, model quality and optimized-kernel performance remain unqualified.

The follow-up with fixed-boundary Viterbi and randomized HIGGS preprocessing
also compiled for sm_86 and sm_120a. All 13 suite processes exited successfully,
including the structural checks and CPU/CUDA curvature-proxy XOR cases. Small
CPU Viterbi cases also completed for MCG and mul1. These remain synthetic
execution checks while the miner is active.

The float32-input/scale-fitting follow-up compiled in both configurations and
completed all 16 runner processes. Reader checks covered exact size and NaN
rejection; scalar fitting passed the check that stored weight error does not
increase relative to RMS. CPU and CUDA diagnostic CSVs each contain eight rows
for the shared synthetic file fixture. Actual model weights, model quality and
performance remain unqualified.

The AVX2/zero-input follow-up compiled for both configurations and completed all
20 runner processes. Exhaustive XOR-state and SIMD projection checks passed.
Zero-weight and zero-activation reports each produced exact zero outputs in all
11 CPU and eight CUDA diagnostic rows. Timings taken with the miner active are
still unqualified; no CPU or GPU speedup is claimed from this run.

The packed-CUDA follow-up compiled for sm_86 and sm_120a. The original-shape full
suite completed 21 processes; the focused suite with the wider fixture completed
14 processes. Exhaustive CUDA XOR-byte parity passed. On the 128x16 and 256x32
synthetic fixtures, packed XOR/Qift paths had relative L2 error against their
decoded-weight FP64 projection below 1.1e-7. Zero-input checks also passed.
These are implementation checks on the RTX 3070, with no qualified timing result.

The block-LDL follow-up compiled in both configurations and passed all 16 focused
checks, including CPU/CUDA feedback fixtures. With the synthetic 128x16 source
and eight calibration vectors, baseline/candidate calibration MSE was
209.284721/204.470008 and both backends selected the same candidate. Held-out
projection relative L2 was about 0.443; this single synthetic projection is not
model quality evidence. Calibration improvements and GEMV timings remain distinct.

The multi-vector follow-up compiled in both configurations and passed all 16
focused processes. On 16 disjoint synthetic held-out vectors, LDL baseline versus
selected relative L2 was 0.325983 versus 0.430355 (worst selected token 0.666080),
despite lower calibration MSE. This exposes overfitting of the eight-vector
calibration fixture; it does not support recommending the feedback encoder for
model quality. The baseline and feedback results are preserved in the generated
diagnostic summary. No held-out selection or timing qualification was performed.

The validation-tuning follow-up compiled in both configurations and passed all
18 focused processes. With 96 training, 32 validation and the unchanged 16
held-out vectors, CPU/CUDA both selected damping 1.0 with feedback. Validation
MSE improved from 231.005344 to 220.589122, while held-out relative L2 changed
from baseline 0.289518 to selected 0.294332. The larger calibration fixture gave
a better baseline than the earlier eight-vector case, but validation-selected
feedback still did not beat that baseline on this held-out fixture. This remains
synthetic evidence and does not justify a default change or model-quality claim.

The paired-guard follow-up compiled for both configurations and passed all 20
focused processes. CPU/CUDA both rejected two lower-mean candidates whose paired
improvements did not clear the 2SE heuristic and retained the damping-0.01
baseline, with held-out relative L2 unchanged at 0.289518. Mean-only selection
remains recorded separately at 0.294332. This fallback avoids the observed
regression on the reused synthetic fixture; it establishes no generalization
guarantee or model-level advantage.

The boundary/scale-search follow-up compiled in both configurations and passed
all 26 focused processes. For the shared 128x16 synthetic fixture, the independent
baseline and four-boundary variant both had weight NMSE 0.093061; adding stored
scale fitting lowered it to 0.088871 at the same 576-byte budget. Held-out relative
L2 changed from 0.316255 to 0.319735. CPU/CUDA agreed. Source reconstruction
improved, but this fixture does not show a held-out quality benefit, so the
refinement remains opt-in. No inference speedup is claimed.

The projection-calibrated scale follow-up compiled in both configurations and
passed all 28 focused processes. On the shared synthetic 128x16 fixture it
changed 45 scale bytes and reduced training MSE from 294.033848 to 224.976987.
Validation gain was 59.027793 with standard error 24.325131, clearing the declared
2SE guard. Held-out relative L2 decreased from 0.316255 to 0.305334 at unchanged
576-byte storage; CPU/CUDA agreed. This is a synthetic diagnostic improvement,
not actual-model quality qualification or an inference speedup measurement.

The robustness follow-up passed all 37 focused processes on the same hardware
while the miner was active. Fixture generation now accepts `--seed`; the runner
adds fixed fresh seeds 5678, 9012 and 3456 to the reused development seed 1234.
Each uses 96 training, 32 validation and 16 held-out vectors, with the same
scale-fitting procedure and paired 2SE acceptance rule. Held-out vectors never
select a candidate. CPU/CUDA agreed for all three fresh fixtures:

| Seed | Baseline held-out relative L2 | Selected relative L2 |
|---|---:|---:|
| 5678 | 0.286017 | 0.245515 |
| 9012 | 0.313077 | 0.289595 |
| 3456 | 0.263678 | 0.235737 |

All three candidates passed validation and improved this synthetic held-out
metric at unchanged storage. `scale-seed-summary.csv` preserves every fixture's
acceptance and baseline/selected error, counting each fixture once in the
Markdown summary. These small Gaussian fixtures do not establish actual-model
quality or speed. Summary regeneration skips its own generated CSV.

## Real weights, stress profiles and algorithm fidelity

`export_layer.py` reads only unquantized F32/F16/BF16 2D GGUF tensors. It exports
a bounded row/column slice without expanding the model, checks finite values
and input geometry, and records hashes, offsets and declared input provenance.
Inputs must use the tensor's full input width before cropping. Training,
validation and held-out vectors must be disjoint. Cropped projections omit
other input channels and do not qualify the full layer or model.

```bash
python3 experiment/fp4/research/export_layer.py \
  --model /path/model.gguf --tensor blk.10.ffn_gate_inp.weight \
  --inputs /path/layer-inputs --input-source captured \
  --cols 128 --rows 16 --output /tmp/layer-fixture
python3 experiment/fp4/scripts/bench_sm86_zen2.py --checks-only \
  --build /tmp/strata-fp4-sm86 --output /tmp/layer-checks \
  --real-fixture /tmp/layer-fixture --real-only
```

The input directory contains `calibration-large.f32`, `validation.f32`,
`heldout.f32` and one-vector `activation.f32`. Declare `--input-source synthetic`
for synthetic activations; the exporter does not infer provenance from numbers.
Capture separate prompt sets at the corresponding layer input for real quality
work. Activation moments and final logits cannot replace those inputs.

Three local GLM-5.3-Flash-REAP50 FP32 router tensors were checked: layers 10,
20 and 30, cropped respectively to 128x16, 256x32 and 512x16 (input x output).
All used independent synthetic input sets (96 training, 32 validation and
16 held-out vectors). CPU/CUDA decoded-weight held-out metrics agreed. Scale
calibration retained the baseline for all three. No original dense FP16/BF16
expert weights or captured layer-input corpus were found; actual-model
calibration and quality qualification remain pending those inputs.

The fixed stress profiles use strong channel outliers, correlated inputs,
rank-deficient inputs, held-out outlier relocation and zero-signal validation.
CPU/CUDA agreed. Selected held-out relative L2 was:

| Profile | Baseline | Selected | Candidate accepted |
|---|---:|---:|---|
| Outliers | 0.308133 | 0.153982 | yes |
| Correlated | 0.335540 | 0.162529 | yes |
| Rank-deficient | 0.385551 | 0.274962 | yes |
| Distribution shift | 0.311694 | 0.528289 | yes |
| Zero-signal validation | 0.308133 | 0.308133 | no |

The distribution-shift regression is retained. Passing the paired validation
heuristic does not establish robustness to changed activation distributions.

`--hessian-block 128|256|384|512` extends LDL feedback across input tiles in
larger covariance blocks. Width must divide the input width; 128 remains the
default. The runner compares 128 and 256 on the correlated 256-input fixture.
For 128-input blocks, paired validation selected damping 1 without feedback;
held-out relative L2 changed from 0.343104 to 0.295219. For 256-input blocks,
it retained damping 0.01 without feedback and unchanged relative L2 0.351499.
These baselines differ; the larger block showed no gain in this check.
This retains cross-128 covariance, while still using a diagonal trellis cost
inside each 16-input tile. It is not full within-tile LDLQ or full-layer
covariance for layers wider than the selected block.

`higgs_reference.py` extracts only literal grids from a supplied official
[Transformers HIGGS source](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/higgs.py),
without executing it. It follows block-norm scaling, unnormalized-Hadamard
normalization, nearest-vector assignment and FP16 stored grids/scales for p=2 grids
of 16/64/256 vectors (2/3/4 payload bits per weight), plus scalar W4. It records
the upstream SHA256 and checks unquantized rotation invariance. Python arithmetic
uses FP64. FLUTE packing, duplicated scales, lookup tables and CUDA execution
are not reproduced; its logical storage estimate must not be used as an
upstream export size.

```bash
python3 experiment/fp4/research/higgs_reference.py \
  --upstream /path/transformers/src/transformers/integrations/higgs.py \
  --fixture /tmp/layer-fixture --output /tmp/higgs-reference
python3 experiment/fp4/research/gyro_study.py --fixture /tmp/layer-fixture \
  --outlier-score maxabs --output /tmp/gyro-reference
```

For layer 10's 128x16 real-weight slice with synthetic inputs, scalar W4 held-out
relative L2 was 0.081950; vector W4 was 0.079192. Logical storage including the
small-slice grid was 1088 versus 2080 bytes. This is a bounded arithmetic
diagnostic, not model quality or a speed comparison.

The GyRot study now accepts exported weights and supplied disjoint inputs.
Its optional maximum-magnitude ranking follows the outlier alignment described
in [GyRot section IV](https://arxiv.org/html/2607.27694v1); structural checks
verify constant-sign harmonic rows within each group. Coarse rotation, fine
grouping and ceiling-rounded unscaled bias are represented, with
calibration-only shared activation metadata quanta, frozen for
held-out inputs; scale and bias bytes saturate when their range is exceeded.
It remains an RTN reference: GPTQ weight fitting, KV quantization, model-wide transform fusion
and the proposed integer accelerator are absent. No official HAP implementation
was located in the inspected paper, so the permutation is labeled paper-guided.

The existing EXL3 experiment is still a K2 codebook/packing comparison.
[Upstream quantization](https://github.com/turboderp-org/exllamav3/blob/master/exllamav3/modules/quant/exl3_lib/quantize.py)
uses codebook markers, tensor-core ordering, fitted scaling, transforms and
calibrated quantization. Matching decode arithmetic does not reproduce that
export pipeline. Official EXL3 conversion and FLUTE execution were not run:
Torch, ExLlamaV3, Transformers and FLUTE are absent from the local experiment
environment. These comparisons remain pending the upstream runtime and original
model/activation inputs; no replacement implementation is claimed equivalent.

The final follow-up compiled the sm_86 and default configurations. The expanded
runner passed 61 processes including one real-weight slice; two additional
real-slice runs passed five processes each. Python exporter checks covered
F32/F16/BF16 orientation, truncation and input overlap; harmonic-slot and frozen
metadata saturation checks passed. HIGGS references ran on the synthetic fixture
and all three real slices; GyRot ran on synthetic and real-slice fixtures.
The default CPU diagnostic CSV for the first real slice stayed byte-identical
after adding configurable Hessian widths. All CUDA execution used RTX 3070;
the default architecture configuration was compile-only for GPU execution.
The miner remained active, so these are execution and arithmetic checks only.

### Captured-input conversion and fixture integrity

`activation_trace.py` reads the binary GLM expert-input records written by
`STRATA_GLM_ACTIVATION_TRACE`. Supply separate traces for training, validation
and held-out prompt sets; multiple trace files per split are accepted. The
converter selects one layer and preserves its full 4096-wide inputs. It records
trace SHA256 values and selected record/token positions without storing source
paths. It rejects repeated trace content, overlapping vectors across splits,
duplicate layer/token positions, malformed routing, nonfinite selected inputs
and truncated records. A per-split cap of 2048 vectors rejects excess rather
than silently sampling. Each split needs at least two selected vectors.

```bash
python3 experiment/fp4/research/activation_trace.py \
  --train-trace /path/train.capture \
  --validation-trace /path/validation.capture \
  --heldout-trace /path/heldout.capture \
  --layer 10 --experts 144 --output /tmp/captured-layer-inputs
python3 experiment/fp4/research/export_layer.py \
  --model /path/model.gguf --tensor blk.10.ffn_gate_inp.weight \
  --inputs /tmp/captured-layer-inputs --input-source captured \
  --cols 128 --rows 16 --output /tmp/captured-layer-fixture
```

Choose 144 or 288 experts to match the captured GLM model. These traces contain
expert inputs, so automatic exporter matching currently accepts only the
corresponding unfiltered router tensor. It rejects a different layer or an
expert-filtered trace. The converter cannot establish independent prompt sets
or the originating model identity from this binary format; callers must retain
that information. `activation.f32` reuses the first training vector solely for
runtime implementation checks, not held-out quality measurement.

The exporter verifies supplied capture hashes and retains capture provenance
in the fixture manifest. It validates even channels omitted by a crop, checks
overlap again after cropping, and canonicalizes signed zero for vector identity.
The replay runner, HIGGS reference and GyRot study now verify file hashes,
geometry, finite values and disjoint splits before consuming a fixture. Invalid
real-fixture input clears an old success status and fails before kernel execution.

Structural tests use generated trace records, not captured model activations.
They cover conversion, correct router matching, wrong-layer rejection, truncated
and ambiguous records, malformed routing, nonfinite values, changed files and
crop-induced leakage. The follow-up passed five small CPU/CUDA replay processes
on the existing real-weight/synthetic-input slice, plus HIGGS and GyRot checks.
No C++ or kernel files changed in this follow-up; actual captured-input quality
and performance qualification remain pending.

## Primary sources

- [EXL3 codebook and packing conventions](https://github.com/turboderp-org/exllamav3/blob/master/exllamav3/modules/quant/exl3_lib/quantize.py).
- [QTIP](https://github.com/Cornell-RelaxML/qtip).
- [XOR-Trellis equations and palettes](https://arxiv.org/html/2610.00432v1).
- [HIGGS](https://arxiv.org/abs/2411.17525).
- [Qift](https://arxiv.org/abs/2606.02823).
- [GyRot](https://arxiv.org/html/2607.27694v1).
