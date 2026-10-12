# TR16 Q23 optimization, 2026-10-09–10

This experiment targets one request on TR16 (Threadripper 1950X, 128 GB RAM,
RTX 5060 Ti 16 GB). It uses the existing Q23 overlay of the Q2 model. The original
Q2 files are retained; the overlay supplies routed expert weights. The optimized
runner measurements use frozen source builds. Serving defaults and concurrent
shared engine work are preserved. The state-only experiment remains in isolated
research builds; its assembled integration was removed after an 8K control
reproducibility failure.

## Ongoing 30/50-token/s search

As of 2026-10-10 12:08 JST, neither 30 tokens/s with lower loss nor 50 tokens/s
with moderate loss is qualified. The best fully qualified profile remains near
23 tokens/s. Q4 short screens reached 25.65 tokens/s, but the full corpus
failed the provisional top-1 agreement limit. Q3 short screens reached about
25 tokens/s and all three precision pilots failed the stricter quality limits.
Earlier runs above 30 and 50 tokens/s produced repeated-output loops and were
rejected. Corrected interleaved Q3 kernel timing passed bitwise parity; the
completed GPU draft and multi-token Q3 screens remained below 30 tokens/s.
Shared projections slowed both tested configurations. Q22 down projections
improved the short 2K prefill screen to 143.31 tokens/s, but the matched decode
screen measured 25.42 versus Q23's 25.63. Q22 with depth-three relaxed drafting
reached 28.09 in a short screen. Both Q22 corpus pilots failed the lower-loss
limits; perplexity increased 4.87%, inside the moderate exploratory budget.
The protected-route pruning experiment passed CUDA/HIP builds, five CUDA parity
suites and 20 routing-oracle configurations. All three short full-model controls
passed. The strongest candidate reached 29.35 tokens/s, but its 2K corpus
pilot increased perplexity 20.68%, so it is rejected. The milder candidate
reached 25.87 tokens/s with perplexity +7.56%; it fails the lower-loss gates
and remains eligible only for moderate-loss exploration. Larger-cache and
drafting screens completed: depth three with acceptance margin four reached
30.44 tokens/s in a 128-token timed screen. Held-out 512-token runs, with
three timed repetitions each, reached medians of 31.25 tokens/s at 2K and
34.34 at 8K. All eight continuations, including warm-ups, were inspected
without an obvious sustained repetition loop; correctness remains untested.
The full 8,192-token corpus check measured perplexity +3.95%, KL +0.0578,
and top-1 agreement −2.27 percentage points. All three provisional lower-loss
limits fail; the perplexity result is within the moderate exploratory budget.
The margin-four task evaluation found a repeated-code loop on HumanEval/31,
which the baseline passed. That candidate is rejected, and its full task
evaluation was stopped. The margin-two replacement measured 28.87 tokens/s at 2K and 30.12 at 8K
in the same held-out protocol. Its inspected continuations showed no obvious
loop, but it misses the 2K speed target and is not quality-qualified. No experimental
result in this search changes the selected profiles below.

## Selected opt-in profiles

All rows use a held-out prompt, 512 generated tokens, one discarded decode warm-up
and three timed repetitions. Rates are medians in tokens/s. Profiles are under
`configs/experimental/`; their runtime settings were checked against the saved
measurement configurations.

| Profile | Prompt | Decode | Prefill | Validation |
|---|---:|---:|---:|---|
| `glm53f-q23-tr16-aff08.json` | 2K | 23.0641 | 130.864 | Full corpus and HumanEval budget passed |
| `glm53f-q23-tr16-aff08-8k.json` | 8K | 23.0002 | 427.889 | Same qualified affinity setting |
| `glm53f-q23-tr16-stream-mtp1.json` | 2K | 20.3272 | 131.527 | Exact output parity |
| `glm53f-q23-tr16-stream8k-mtp1.json` | 8K | 19.3478 | 474.917 | Exact output parity |

Affinity is the fastest qualified decode setting tested. Exact streaming is the
choice for faster 8K prefill. Combining them is an unqualified experiment, as
described below. The 130-token/s 2K prefill target is met; the 30-token/s decode
and 500-token/s 8K prefill targets remain unmet. The original calibration profile
and exact committed-runner MTP1 profile are retained for comparisons.

## Measured hardware limits

The GPU is connected at PCIe 3.0 x8. Pinned host-to-device copies measured
7.2128 GB/s. A 15-worker host read test measured 78.2878 GB/s; the Q23 expert-flow
probe measured 72.35 GB/s at width one and 71.43 GB/s at width two. These are
independent probes, not fitted end-to-end throughput.

The Q23 routed tensors contain 111,188,901,888 bytes. Together with fixed and
draft tensors, effective weights total 120,865,643,896 bytes (112.565 GiB).
The overlay itself is smaller because some tensors remain in the original model.
The original Q2 effective weights total 108,710,550,904 bytes (101.245 GiB).

An ordinary Q23 token selects about 3.089 GB of routed weights before cache hits.
At 72.35 GB/s, reading those bytes alone costs 42.69 ms, or 23.4 tokens/s.
This is a no-cache CPU component ceiling, not a whole-engine ceiling: resident
experts remove CPU reads, while attention, dense work and synchronization add time.
Speculation changes both expert unions and tokens delivered per verification.

One all-expert Q23 upload takes at least 15.42 seconds at the measured PCIe rate.
Ignoring compute and layer boundaries gives about 132.9 prefill tokens/s at 2K
and 531.4 at 8K. The targets are 30 decode tokens/s, 130 prefill at 2K and 500
prefill at 8K. Targets are not measurements or promised attainable rates.

## Measurements

The 2K held-out code prompt generated 512 tokens. Each mode had one discarded
decode warm-up and three timed repetitions:

| Mode | Decode tokens/s, median | Prefill tokens/s, median | Output |
|---|---:|---:|---|
| MTP2, unchanged calibration config | 18.6273 | 131.350 | Identical in all repetitions |
| MTP1, otherwise unchanged | 20.2822 | 131.325 | Identical to MTP2 and in all repetitions |

MTP1 gained 8.9% on this held-out prompt. These are repeated timings within one
process per configuration, not independent process replicates. The original
30-token/s decode target is not reached.

The 8K held-out code prompt with the same 512-token protocol measured 20.0982
decode tokens/s and 443.041 prefill tokens/s. FP16 expert bucketing reached
359.689 prefill tokens/s on that prompt and was rejected as slower than MMQ.
Prefill throughput excludes cold model loading and decode-cache preparation.
The one-time NUMA relocation took about 18 seconds in a fresh process; a resident
server keeps that placement for subsequent requests.

The short screens used 128 generated tokens and one timed decode repetition.
MTP1 baselines ranged from 19.52 to 20.19 tokens/s. Affinity 0.05, 0.08 and 0.10
measured 21.64, 22.67 and 21.97 respectively; all three changed output between
repetitions. Affinity 0.08 passed the later full quality checks below; 0.05 and
0.10 remain unqualified. Unsplit verification regressed to 17.86;
continuation margin 2 and batched resident execution stayed near 20.1–20.3.
Neither showed a clear improvement over split MTP1.

At 8K, the MMQ screen measured 435.77 prefill tokens/s. Batched FP16 measured
197.89 and was rejected as slower. Both repeated their own output. The 500-token/s
8K target is not reached by either screen.

An isolated opt-in implementation started the next layer's weight prefetch after
the preceding layer enqueued its final expert products. Existing completion events
protected pool-slot reuse. Three timed 8K prefills measured 439.42 tokens/s with
the flag off and 438.36 with it on (-0.24%). All four 128-token decode repetitions
matched between modes. The patch was rejected for lack of speedup and was not
copied into the shared engine source. Its patch and raw results are retained in
the fixture directory. This experiment was built and exercised on CUDA only.

A second implementation lets the foreground consume each prefetched expert group
as soon as its upload is queued, instead of joining the entire producer first.
The foreground owns the group map; an atomic release/acquire handoff publishes
each recorded CUDA event. The fallback uploader still joins the producer before
reusing its two pinned buffers. The option is `STRATA_GLM_PREFETCH_STREAM=1`, with
18 prefetch groups. It is off by default and restricted to the existing staged,
single-GPU path.

| Streaming prefill comparison | Control tokens/s | Streaming tokens/s | Gain |
|---|---:|---:|---:|
| 8K training prompt, 128 generated tokens | 426.395 | 474.209 | 11.21% |
| 8K held-out prompt, 512 generated tokens | 428.063 | 474.917 | 10.95% |

Each row used three timed prefills and one discarded decode warm-up. All four
generated token sequences matched between modes, including all 512 tokens on the
held-out prompt. Streaming is a prefill improvement: paired held-out decode was
19.5186 versus 19.3478 tokens/s. The final 2K, 512-token check measured 131.527
prefill and 20.3272 decode tokens/s. All four 512-token outputs matched the
committed MTP1 baseline. Its peak RSS was 107.75 GiB, process swap stayed zero,
and minimum observed GPU free memory was 806 MiB.

The 2K target is met. Streaming remains 25.083 tokens/s (5.02%) below the 500-token/s
8K target. Exact MTP1 decode remains about 20 tokens/s, short of the 30-token/s
target. Affinity's approximately 23-token/s results are separate measurements;
the combined settings were measured separately after qualification:

| Streaming plus affinity 0.08 | Decode tokens/s | Prefill tokens/s |
|---|---:|---:|
| 2K, 512 generated tokens, three timed repetitions | 22.7364 | 131.166 |
| 8K, same protocol | 22.0450 | 468.977 |

At 2K, each of the four generated sequences matched its corresponding
non-streaming affinity trial. At 8K, all four differed. The resident tier had the
same 627 slots but a different trained-cache fingerprint; timed draft acceptance
and decode throughput varied more. The combined profile has no full paired
quality evaluation and is not promoted. The full HumanEval qualification below
applies to the committed runner with affinity 0.08. Exact streaming has its own
output-parity checks. These are separate opt-in choices, rather than an assumed
combination of independently measured gains.

A final CUDA-only prototype combined per-group streaming with earlier next-layer
uploads. Its paired 8K screen measured 468.736 tokens/s for streaming alone and
468.383 for the combination (-0.075%). All four 128-token outputs matched between
modes. It failed the predefined 3% gain threshold and was rejected; its patch and
measurements are retained. No held-out confirmation or HIP build was pursued for
this rejected prototype.

## Reproduction and isolation

`build-tr16-20261009/source` is an archive of commit `6ecf3bcb`; the CUDA baseline
was built from that archive. Pre-existing uncommitted engine work was saved in
`build-tr16-20261009/pre-existing.patch` and excluded from the baseline. A separate
session was also changing engine files, so the root working tree is not a valid
identifier for the measured binary. Each run records its binary hash.
The concurrent B550 lever commit `e5232ca3` and later shared-tree edits are outside
these frozen builds.
Later runs also retain a hash-verified copy of the Python runner. The initial
`screen-v2` predates that snapshot feature and retains its hash and command only;
`runner-audit.json` records this limitation.

The baseline configure used Release, CUDA architecture 120, and the existing
local ggml checkout:

```sh
cmake -S build-tr16-20261009/source -B build-tr16-20261009/baseline -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_GLM_MODEL_TESTS=ON \
  -DSTRATA_GGML_DIR=/mnt/nvme02/work/gemm/main/tmp/qwen38/reference-build/source
cmake --build build-tr16-20261009/baseline --target strata-glm-decode \
  glm_q23_parity glm_q8_test glm_mailbox_test glm_prefill_parity -j12
```

Frozen training and held-out prompts and hardware inventory are saved in
[`fixtures/glm_tr16_20261009`](fixtures/glm_tr16_20261009). The prompts were
generated with the existing B550 fixture tool; each length has code, documentation
and chat inputs. A held-out input is never used to fit simulator parameters.

The opt-in config is
`configs/experimental/glm53f-q23-tr16-calibration.json`. Its paths are local to
TR16; replace them when reproducing elsewhere. The Q23 pack and prior must already
exist. Use a Python environment with the existing benchmark dependencies
(`jinja2` and `regex`), plus CUDA access and `numactl`.

Exact opt-in profiles retain the measured binaries:
`configs/experimental/glm53f-q23-tr16-mtp1.json` uses the committed baseline,
and `configs/experimental/glm53f-q23-tr16-stream-mtp1.json` uses the isolated
streaming runner with a 2K prefill batch. For the measured 8K throughput, use
`configs/experimental/glm53f-q23-tr16-stream8k-mtp1.json`: it sets an 8K batch and
2 GiB scratch cap. An 8K prompt split into four 2K chunks repeats expert uploads
and does not have the reported 8K throughput. The qualified approximate profile is
`configs/experimental/glm53f-q23-tr16-aff08.json`, with MTP1 and affinity 0.08.
Its 8K-batch counterpart is `configs/experimental/glm53f-q23-tr16-aff08-8k.json`.
The shared C++ working tree has concurrent changes from another
session, so it is deliberately not overwritten. Recreate the streaming source
from the frozen commit and the saved patch:

```sh
mkdir -p build-tr16-reproduce/source
git archive 6ecf3bcb | tar -x -C build-tr16-reproduce/source
patch --directory=build-tr16-reproduce/source -p1 < \
  docs/fixtures/glm_tr16_20261009/stream-prefetch.patch
```

Configure and build that source with the CUDA command above, changing the source
and build directories. Set the profile's `exe` to the resulting binary. Passing
`--config` to the guarded runner selects the profile without changing defaults.
The experimental cases reject binaries that do not contain the required flag.

```sh
build-exl3-venv/bin/python tools/glm_tr16_calibrate.py \
  --prompt build-tr16-20261009/prompts2k/train-code.ids \
  --output build-tr16-20261009/new-screen \
  --cases mtp1 mtp2 --interleave --tokens 128 --trials 1
```

The runner waits for a quiet interval, rejects competing CUDA clients and sustained
external CPU work, checks zero process swap, keeps at least 4 GiB host memory
available and 512 MiB GPU memory free. It does not stop other users' jobs. A run
interrupted by another session compiling was excluded from calibration.
The first decode repetition is discarded. Final timing uses 512 generated tokens
and three timed repetitions; 128-token screens only select candidates.

`--require-repeat-equal` checks token identity across repetitions. Affinity can
change routes and generated tokens; a clean timing run alone does not establish
quality or determinism. `--allow-interference` produces diagnostic records that
the simulator importer refuses as calibration input.

## Simulator changes

Prefill upload accounting now sums each layer's actual format, including the
different format in layer 11 and metadata. The decode GPU ceiling no longer counts
the first three dense layers twice. `--tier-owned-reserve` exposes the existing
memory-planner option through the CLI.

`tools/glm_tr16_sim_records.py` accepts only complete, quiet measurements.
Training records must be named explicitly with `--fit`. Forecasts freeze MTP
acceptance from those records; `--observed-routing` is a separately labelled
component diagnostic and cannot fit parameters. Only an ordinary-decode training
run can fit aggregate launch overhead; only an 8K prefill training run can fit
prefill GEMM cost. A PCIe-bound 2K measurement cannot identify GEMM throughput.

The opt-in fit is `tools/sim/data/params_tr16_20261009.json`, with forecast records
in `measured_tr16_20261009.json` in the same directory. Training is restricted to
the first ordinary/MTP2 screen, the first MTP1 baseline, and the MMQ 8K training
prompt. Q23 bandwidth and the width-two penalty come from the independent expert
probe. Only aggregate launch overhead and prefill GEMM cost are fitted.

The 36 forecast records have 2.4% mean absolute training error. The two
512-token held-out decode errors are -6.1% and -4.8%. Affinity 0.10 is an explicit
miss (+14.7%, outside the 10% decode tolerance); it is not fitted away. The
separate `measured_tr16_20261009_conditional.json` substitutes observed acceptance
and cache size and must not be reported as forecast accuracy. In particular,
affinity 0.10's measured draft acceptance fell to 0.636, versus 0.726 at 0.08.
The 8K exact holdout errors are -3.9% for decode and -2.0% for prefill; the final
affinity 0.08 decode errors are +7.2% and +7.5%. No holdout was used to refit the
parameters. The streaming overlap is not represented by this simulator and is
excluded from its fit and forecast accuracy claims.

```sh
python3 tools/sim/glm_sim.py validate \
  --records tools/sim/data/measured_tr16_20261009.json \
  --params tools/sim/data/params_tr16_20261009.json
```

This command deliberately exits nonzero while the affinity forecast misses its
tolerance. Historical default parameters are unchanged.

## Remaining target gap

The frozen exact-MTP1 forecast predicts 1.722 tokens per 89.17 ms round, including
74.17 ms of CPU expert work. These are simulator components, not measured kernel
times; the complete breakdown is in `exact-mtp1-forecast-breakdown.json` in the
fixture directory. Its CPU-only ceiling is 23.22 tokens/s and GPU-only ceiling is
29.80 tokens/s. Removing launch overhead alone cannot reach 30 under this model.
At the same acceptance rate, 30 tokens/s requires a 57.40 ms round, including all
work: at least a 22.6% reduction in the CPU component even with perfect overlap.
Further decode work must reduce expert bytes or the cost of reading them, then
remeasure acceptance and quality. The measured unsplit-verification regression
rules out simply enabling that existing option.

One concrete follow-up is to audit the measured 6,153.68 MiB dense GPU allocation
for storage needed only during prefill, then measure whether freeing any such
storage allows a larger resident expert tier. The current estimates assume no
memory saving from that untested change.

For 8K prefill, 500 tokens/s requires 16.384 seconds versus the streaming result's
17.25 seconds. The upload floor is about 15.42 seconds, leaving less than a second
for exposed compute and layer boundaries at the target. Further work should
measure those exposed intervals before adding more overlap or GPU buffers; the
current memory reserve must remain part of each candidate's guard.

## Quality and promotion

For approximate settings, the provisional lower-loss budget against unchanged Q23 is perplexity
at most 3% higher, KL at most 0.03 higher against the same frozen reference, top-1
agreement at most one percentage point lower, and at most one fewer HumanEval
pass out of 164. Existing regression thresholds are unchanged.

`tools/glm_tr16_quality.py` evaluates the full regression corpus with the candidate's
actual resident tier and affinity, preserving the corpus and raw evaluation logs.
Run it first with unchanged Q23, then pass that `result.json` through `--baseline`
for the candidate. It rejects incomplete or incomparable evaluations and nonfinite
metrics. Passing this corpus gate does not claim HumanEval qualification.

Both full evaluations scored 8,192 tokens over the same 16 frozen sequences:

| Setting | Perplexity | KL | Top-1 agreement |
|---|---:|---:|---:|
| Unchanged Q23 | 4.826016 | 0.136762 | 87.7075% |
| Q23, affinity 0.08 | 4.926113 | 0.155009 | 87.4146% |

Affinity 0.08 passes this corpus gate: perplexity +2.07%, KL +0.01825 and top-1
agreement -0.293 percentage points. The unchanged-Q23 result also passes the
repository's stricter regression tolerances. The 512-token affinity timing checks measured 23.0641
decode tokens/s at 2K and 23.0002 at 8K, with differing outputs between repetitions.

The complete unchanged-Q23 HumanEval baseline scored 154/164; affinity 0.08 scored
153/164 and passes the agreed gate, using the full one-pass allowance. This uses the
repository's pinned greedy chat protocol with low reasoning, a 2,048-token
generation cap, 12K context, one sample per task and isolated Python tests.
The candidate lost tasks 81, 132 and 137, and recovered tasks 97 and 130: net -1.
The corpus and HumanEval manifests passed the input and protocol consistency
checks; `qualification.json` records the complete decision.
HumanEval guards permit and record external CPU work, so their wall times are
excluded from performance claims. Both runs had zero process swap and at least
818 MiB observed GPU free memory. Per-task results, manifests and consolidated
generated code are retained with the fixtures.

Approximate settings must not be promoted until both quality gates and held-out
timings pass. Any experimental engine patch must additionally pass CUDA parity,
repeatable output checks and a repeatable speed gain before being copied into the
shared source tree.

## Build and correctness checks

The isolated streaming runner passed `glm_q23_parity`, `glm_q8_test`,
`glm_mailbox_test` and `glm_prefill_parity` on CUDA, plus the paired full-model
output checks above. It also compiled and linked with HIP for `gfx1201`, using
`-DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=ON`,
`-DCMAKE_HIP_ARCHITECTURES=gfx1201` and
`-DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0`. No AMD runtime test was performed.
This GLM decoder target has no SYCL implementation. Build and parity logs are
retained with the fixtures. All 91 tests in `test_glm_tr16_calibrate`, `test_sim`
and `test_sim_b550` passed. No serving defaults or shared engine files were
changed by this experiment.

The stop-token audit covers all 34 complete clean timing runs in the fixture
directory. None encountered EOS, EOT or EOM before the requested generation
length. GPU memory checks include the desktop; host `MemAvailable` includes
reclaimable mapped weights and must not be interpreted as unused physical RAM.

## Follow-up 30/50 tokens/s exploration (in progress)

The follow-up on the same TR16 uses isolated source at `e5232ca3`, the same Q2
model and Q23 expert sidecar. It has not established either 30 tokens/s with
lower loss or 50 tokens/s with acceptable loss. The qualified affinity-0.08
profile above remains the measured quality baseline for deployment.

Skipping unused MTP resynchronization outputs preserves the persistent draft
state in 32 byte-for-byte diagnostic comparisons, including append-only caches.
Three short full-model comparisons also preserve all 256 emitted tokens in each pair.
At depth one, one 128-token timed training-prompt screen changed from 20.3617 to
20.7303 tokens/s. With affinity 0.08 and pruning 0.12, the corresponding pair
changed from 27.1541 to 27.9471 tokens/s. These short measurements do not establish
sustained gains. The change is opt-in. Its independent patch has now built with
CUDA and HIP and passed five CUDA parity tests. The standalone default matches
256 prior control tokens; enabling state-only matches all 256 tokens from that
fresh control (20.3862 versus 20.7181 tokens/s in the short screen). Its 8K
diagnostic also passed 32 full-state comparisons. The opt-in change is now
integrated in the shared decoder, preserving the prior deferral edits. Fresh
assembled CUDA/HIP builds and held-out 2K/8K pairs remain pending. HIP runtime
behavior has not been tested.

In an unpruned affinity-0.20 configuration with relaxed acceptance, the same
state-only change increased the median from 35.1410 to 36.4788 tokens/s over three
512-token timed trials. All 2,048 warmup and timed output tokens matched between
the two runs. The per-trial gains were 3.5–4.1%. This configuration also produced
repetitive continuations and remains unqualified; the comparison measures the
resynchronization change, not achievement of the lower-loss 30-token/s target.
The same-prompt exact control measured 20.0934 tokens/s median over three
512-token trials and did not show the repeated-sentence pattern. All four of
its warmup and timed streams were identical. This comparison concerns repetition,
not the factual correctness of the generated review.

An aggressive combination of affinity 0.35, pruning 0.12, Q4 dense repacking and
relaxed draft acceptance reached 55.4266 tokens/s median over three 512-token
training-prompt trials (58.1545, 55.4266 and 52.8727). Every inspected continuation
collapsed into repeated sentences. This result does **not** achieve the useful
50-token/s target. High draft acceptance on repeated text is not evidence of
useful throughput.

The matched four-sequence, 2,048-target-token corpus pilot also rejects the
following combinations. All candidate rows use affinity 0.08 and the same fast
kernel flags; the last two additionally renormalize the remaining route weights.
This scoring does not assess relaxed speculative acceptance.

| Pilot configuration | Perplexity | KL | Top-token agreement |
|---|---:|---:|---:|
| Unmodified control | 3.772909 | 0.102885 | 0.892578 |
| Pruning 0.12 | 5.043133 | 0.471369 | 0.792480 |
| Pruning 0.12, renormalized | 5.585812 | 0.562422 | 0.744629 |
| Pruning 0.16, renormalized | 18.322836 | 1.823126 | 0.541504 |

The unpruned affinity-0.20 configuration, scored without relaxed acceptance,
measured perplexity 4.165690 (+10.41%), KL 0.208000 and top-token agreement
0.862793 (down 2.98 percentage points) on the same 2,048 targets. It fails the
configured pilot gate. Preventing repeated generated phrases would not remove
this measured distribution change. A first attempt was interrupted by the
system file indexer; these numbers come from the complete, clean retry.

Q4 dense repacking with affinity zero passed this pilot: perplexity 3.788316
(+0.41%), KL 0.113818 (+0.01093), and top-token agreement 0.894043
(up 0.15 percentage points). All 2,048 targets were scored and no CPU or GPU
interference was recorded. This is a component pilot, not full-corpus or
generated-task qualification. Subsequent quality runs may record CPU contention
without treating their wall time as a performance measurement; GPU exclusivity
and memory limits remain enforced.

Adding affinity 0.08 to Q4 scored perplexity 3.854981 (+2.18%), KL 0.138642
(+0.03576), and top-token agreement 0.878906 (down 1.37 percentage points).
Perplexity passes the pilot limit, while KL and agreement narrowly fail. This
prompted a fresh paired 8,192-target evaluation without changing the limits.

The fresh 8,192-target baseline has completed. All 16 cumulative result rows,
including final perplexity 4.826016, KL 0.136762 and agreement 0.877075, match
the prior full baseline exactly. Corpus and model input identities also match;
no interference was recorded. The paired Q4 candidate also completed without
recorded interference: perplexity 4.924513 (+2.04%), KL 0.166410 (+0.02965),
and agreement 0.865723 (down 1.135 percentage points). Perplexity and KL pass;
agreement narrowly misses the one-percentage-point limit, so the full corpus
gate remains failed. Paired 20-task generated-code smoke tests are complete;
these do not replace a full 164-task evaluation.

The generated-code smoke baseline passed all first 20 pinned HumanEval tasks.
Each generated token hash and pass outcome matches the corresponding task in
the prior full baseline run. Q4 also passed all 20 tasks, with 7 identical token
hashes out of 20 paired outputs. No claim about the other 144 tasks follows
from this check.

Q4 with affinity 0.08 and renormalized pruning 0.08 failed all three pilot
limits: perplexity 4.262414 (+12.97%), KL 0.243813, and top-token agreement
0.848145 (down 4.44 percentage points). It was also slower than the unpruned
Q4 screen, so it is excluded from the lower-loss candidate path.

The combined experimental runner built with CUDA and HIP and passed five CUDA
parity checks, ten routing-oracle cases and the GLM-shaped interleaved dense
kernel parity test. No AMD runtime test was performed. The new decoder/kernel
experiments remain isolated patches; none is a qualified replacement profile.
The search continues with unpruned Q4 configurations, smaller acceptance margins,
component corpus checks and paired generated-code tests.

The completed Q4 sweep used one 128-token warmup and one timed trial per case
on the same 2K training prompt. Its fastest unpruned configuration measured
24.8212 tokens/s with affinity 0.08 and one draft token. Depths two and three
with acceptance margins one or two measured 20.8729–22.8731 tokens/s. Adding
renormalized pruning at 0.10 reached 27.3725 tokens/s; its quality is unqualified.
Prefill remained about 131 tokens/s. These are short screens, not held-out or
sustained measurements.

For that 24.8212-token/s run, removing the measured draft time while holding
everything else fixed gives 26.57 tokens/s. Accepting every depth-one proposal
at the measured average round cost gives 27.36 tokens/s. These counterfactuals
are not hardware rooflines or forecasts: changing acceptance changes the work,
and GPU drafting consumes memory otherwise available to target experts.
The four split-verification screens measured 25.5109 tokens/s at depth one
without relaxed acceptance, 23.4204 at depth two with margin one, and
23.0166/23.9213 at depth three with margins one/two. Output tokens differ from
the unsplit cases, so these are not exact-output speedup measurements. The
one-draft configuration remains the fastest unpruned Q4 screen. CPU/GPU draft
comparisons remain queued.

An additional experiment blocks only relaxed accepts that would repeat
a recent emitted n-gram. It leaves exact target matches alone and therefore
cannot prevent repetition already preferred by the target model. Its default
is off. Its CPU test passed 21,504 differential cases, and bounded-acceptance
regression tests passed against the same header. CUDA/HIP builds, five CUDA
parity tests, ten routing-oracle cases and dense-kernel parity have now passed.
All eight full-model control pairs passed, each comparing 256 emitted tokens.
The exact repetition-guard case preserved acceptance counts and blocked zero
tokens, as expected. Paired 512-token continuations are now running; no quality
or speed benefit from the guard is established.

The long unpruned control reproduced all 2,048 tokens of the earlier state-only
run and measured 36.7702 tokens/s. A four-token repetition guard measured
34.8530 tokens/s and blocked 1–5 soft accepts per continuation. Inspection of
all four 512-token outputs found that one of the three timed continuations
still entered a long repeated-function-signature loop. The other two and the
warmup lacked a comparable sustained loop, which does not establish their
correctness. The candidate remains unqualified. Because continuations differ,
the 5.21% rate decrease is not guard-check overhead alone.

The eight-token guard measured 36.5707 tokens/s. It blocked three proposals
during warmup and none during the timed trials. Two of the three timed
continuations still entered long loops, so this variant is also unqualified.
These results match the policy's limitation: it cannot reject a repeated token
that already matches the target's greedy choice.

The depth-three proposal controls preserve all 256 warmup and timed target
tokens per pair. On the same 2K training prompt, one 128-token timed trial gave:

| Recursive hidden blend | Decode tokens/s | Accepted drafts per round |
|---|---:|---:|
| 1.0, unchanged | 15.6364 | 1.3333 |
| 0.0, retain previous hidden | 13.9268 | 1.0656 |
| 0.5 | 15.9907 | 1.3774 |
| 0.75 | 15.5455 | 1.3333 |

The half blend gains 2.27% in this short screen, but remains slower than depth
one. The other blends provide no measured benefit. These are grouped runs,
not interleaved process replicates. Different round boundaries also mean that
per-position acceptance counts are not paired predictions on identical prefixes.
Unpruned draft routing with global pruning disabled matched all tokens and
acceptance counts, as expected. The exact repetition-guard control also matched
all tokens and acceptance counts (15.6798 tokens/s).

A further opt-in Q3 dense conversion is prepared but unbuilt. It uses the same
tensor-selection rules as Q4 and rejects simultaneous Q3/Q4 conversion. Planned
screens separate conversion from a larger resident expert cache, followed by
component corpus pilots. No Q3 dense speed or quality result is available.

Two additional quality pilots are prepared for the existing unbiased cache
learner, at affinity 0.20 and at Q4 plus affinity 0.08. They will test whether
learning popularity before the residency bonus reduces distribution loss; no
benefit is established yet. The B550 shadow-probe option rejects this MTP setup
and is not being silently enabled. TR16 simulator import now rejects both
unmodeled cache-learning/probe options.

Completed raw evidence and research patches are retained in
[`fixtures/glm_tr16_goal_20261010`](fixtures/glm_tr16_goal_20261010/README.txt).
Its `progress.json` records the incomplete checks explicitly. Full corpus,
paired 164-task HumanEval and held-out timing remain necessary for any finalist.

The aggressive repetition-guard run measured 50.6875 tokens/s, but all four
512-token continuations (warmup and three timed trials) collapsed into repeated
passages. It is rejected as a useful 50-token/s result.

The initial GPU-draft comparisons were rejected before loading because the
runner forbids GPU draft with the target cache. The valid CPU controls measured
20.968, 24.4148 and 21.0157 tokens/s; there is no corresponding GPU measurement.
An isolated, default-off experiment now reserves GPU draft weights and rollback
history before target-cache admission, retaining the existing memory guards.
It still requires fresh CUDA/HIP builds and guarded model validation.

The assembled root CUDA and HIP builds and five CUDA parity tests passed.
On the held-out 2K prompt, state-only resynchronization measured 23.8950
tokens/s versus a same-day control of 23.4275 (+2.00%), each with three timed
512-token trials after warmup. All 2,048 output tokens matched between the
pair; the control also matched the prior qualified profile. The 8K pair and
full generated-task qualification remain pending.

The fresh 8K control was clean (23.3119 decode and 435.016 prefill tokens/s),
but did not reproduce the archived control: the first differing token in
each 512-token trial was 240, 63, 60 and 37. Both admitted the same 627 cache
slots and fingerprint. The archived binary predates the current base commit.
The batch stopped at this check; an original-binary/prepatch/state-only audit
is running. No 8K output-equivalence claim is made.

The 8K audit reproduced all archived tokens with both the original binary
and the current-base pre-optimization binary (23.5278 and 23.5073 tokens/s).
Thus the divergence is confined to the newly assembled build so far, not
established as a base-commit change. That build includes the independent
state-only change and preserved deferral edits. Standalone off/on tests are
queued to separate them; integration is not promoted.

The assembled 8K state-only run measured 24.0787 tokens/s and matched all
2,048 archived tokens. It did not match the first assembled control, which
remains the only divergent run in this audit. Therefore their nominal rate
difference is not reported as an output-equivalent speedup. A repeat of that
control is queued; neither a deterministic patch regression nor a transient
cause has been established.

The independent standalone 8K off/on pair passed: 23.5539 versus 24.0469
tokens/s (+2.09%), with all 2,048 tokens identical between the pair and the
archive. Both runs used three timed 512-token trials after warmup. This
validates the isolated optimization on that workload; the earlier assembled
control discrepancy remains recorded and its repeat is pending.

The 2,048-target affinity-0.20 pilot with unbiased cache learning completed.
Perplexity was 4.09272 (+8.48% versus the exact baseline), compared with
4.16569 (+10.41%) under the prior learner. KL was 0.196093 and top-1 agreement
0.867188, a 2.54 percentage-point drop versus baseline. All provisional low-loss
limits still fail. The perplexity increase is within the provisional 10%
moderate-loss budget, but full corpus, generated tasks and speed remain
unqualified. This quality-only run does not establish a throughput gain.

Unbiased learning with Q4 plus affinity 0.08 also completed the 2,048-target
pilot: perplexity 3.83597 (+1.67%), top-1 agreement 0.883789 (-0.879 percentage
points), and KL 0.138000 (+0.035115). Perplexity and top-1 pass the provisional
low-loss limits, but KL still exceeds +0.03. Full corpus and throughput for
this learner remain unmeasured.

The isolated Q3 build now also includes a prepared interleaved Q3 MMVQ path.
It has no default table entry and uses the existing Q3 dot expression. CUDA/HIP
builds, bitwise parity (including small and odd-sized matrices), and benchmarks
are queued. Native Q3 model screens still run first. A measured two-token
table is tested in the model only if a Q3 pilot stays within the provisional
moderate perplexity budget; this is not full quality qualification.

The assembled control repeat failed again: 23.2198 decode tokens/s, with a
difference from token zero in every trial and a different prefill-trained cache
fingerprint. Binary, prompt, model input identities and settings match the
first control; only the run label differs. The cause remains unresolved.
The main decoder has therefore been restored to its pre-integration contents,
preserving the other session's 22-line deferral edits. The standalone state-only
patch, builds and successful paired evidence are retained as research artifacts.
No default-preservation claim is made for the rejected assembled integration.

The first native Q3 dense screen completed at 19.9972 decode tokens/s without
affinity (128-token warmup and one timed 128-token continuation). The same 271
tensors occupy 2465.547 MiB versus Q4's 3227.625 MiB, saving 762.078 MiB.
This run does not establish a speed gain or quality qualification. Matched
affinity/cache variants and corpus pilots remain pending; the interleaved
Q3 path was inactive in this run.

At affinity 0.08 and the same requested 6656 MiB cache budget, native Q3
measured 24.1844 versus Q4's 25.6458 tokens/s. Actual admission differed:
766 Q3 slots versus 734 Q4 slots. Q3 accepted 51 drafts in 75 timed rounds,
versus Q4's 56 in 71; its average verification time per round was slightly
lower. The different outputs and cache admission prevent treating this as
a pure kernel comparison. Increasing the requested Q3 cache to 7424 MiB
raised the short-screen rate to 24.7067 tokens/s. Quality is still pending.

Attention-only Q3 conversion measured 24.9286 tokens/s in the short screen
(146 converted tensors), still below Q4. The first Q3 quality pilot, without
affinity, completed 2,048 targets: perplexity 3.94715 (+4.62%), KL 0.153105
(+0.050220), and top-1 agreement 0.881836 (-1.074 percentage points). It fails
all provisional low-loss limits. Its perplexity remains within the provisional
moderate budget, but neither full task quality nor a target speed is qualified.

All three native Q3 pilots have now completed 2,048 targets. The larger-cache
Q3 variant has perplexity 4.01299 (+6.36%), KL 0.179046 (+0.076161), and
top-1 agreement 0.865723 (-2.686 percentage points). Attention-only Q3 has
perplexity 4.01232 (+6.35%), KL 0.172163 (+0.069278), and top-1 agreement
0.871094 (-2.148 percentage points). Both fail every provisional low-loss
limit. Their perplexity alone remains within the moderate pilot budget;
this does not qualify either profile for deployment.

The isolated Q3 CUDA and HIP builds and bitwise interleaved-kernel parity
checks passed. The original benchmark invocation placed its arguments in
the wrong order, so it ran parity without timing. The completed native
screens and quality pilots are retained. A separate corrected timing run
is underway, followed by matched CPU/GPU draft screens; no timing result
is inferred from the parity checks.

Corrected Q3 IL timing and bitwise parity completed. For example, the
4096x12288 two-token kernel measured 98.0 us native versus 63.7 us for
interleaved rows=1. The selected model screen preserved all 256 tokens and
measured 25.0188 versus 24.9286 tokens/s. Its depth-one split verification
uses one-token groups, so this small difference does not establish a gain
from the multi-token Q3 kernel. A matched depth-three native/IL comparison
has been added to the pending combined screens to exercise that path.

The first matched GPU draft attempt reserved 2502 MiB of draft weights and
rollback history before target-cache admission, but a second runtime guard
rejected it before decode. It supplies no GPU throughput evidence. A new
isolated build narrows that guard to allow only the explicitly enabled,
pre-reserved, single-GPU fixed-cache configuration. Fresh CUDA/HIP builds,
parity checks and a CPU token control precede new paired screens; the failed
run and its original binaries are retained.

The corrected GPU-cache build passed CUDA and HIP compilation, five CUDA
parity tests, and the interleaved-kernel bitwise checks. Source comparison
confirms only the runtime admission guard changed from the previous isolated
build. Fresh model controls and GPU draft timings are now running.

The first valid GPU draft pair completed at affinity zero with a 3584 MiB
requested target cache: CPU 20.9860 versus GPU 21.1651 tokens/s, a nominal
0.85% increase in one timed 128-token screen. Both had 74 rounds and 53
accepted proposals, identical actual target-cache capacity, and all 256
warmup/timed tokens matched. GPU drafting occupied 2502 MiB and left
1685 MiB free at the reported decode checkpoint. Draft time fell from
4.551 to 3.700 ms per round; verification measured 75.037 versus 75.359 ms.
This is a valid short comparison, not full quality qualification or evidence
that GPU drafting reaches either speed target.

At affinity 0.08 with the same 3584 MiB requested target cache, the paired
short screen measured CPU 24.3322 versus GPU 24.7472 tokens/s (+1.71%
nominal). Actual cache capacity matched, and both 128-token trials matched
their respective CPU/GPU counterpart exactly. Warmup and timed outputs
within each backend differed. GPU drafting saved 0.840 ms per round; this
comparison still does not reach 30 tokens/s or qualify the profile.

All six matched Q4 draft screens completed cleanly. At depth three with
soft-acceptance margin 1, CPU measured 20.7765 versus GPU 20.9985 tokens/s.
Cache capacity matched, but outputs first differed at token 49 in warmup
and token 6 in the timed continuation. The nominal 1.07% difference cannot
be attributed to backend cost alone. Both are slower than depth one and
remain unqualified. The Q3 multi-token kernel comparisons run next.

The matched depth-three Q3 native/interleaved comparison preserved all 256
tokens and acceptance statistics. Native measured 23.4585 and interleaved
23.5738 tokens/s (+0.49% nominal in one short trial). Thus the standalone
kernel gain gives little end-to-end benefit in this measured configuration.
Neither precision quality nor either throughput target is qualified.

With Q3, depth three and soft margin 2, GPU drafting measured 26.1162
versus CPU 23.5738 tokens/s. Actual cache capacity matched, but outputs
first differed at warmup token 8 and timed token 2. GPU accepted 2.256
proposals per round versus CPU 1.930, reducing the timed round count from
43 to 39. Draft time fell from 10.273 to 7.952 ms per round, while verification
rose from 110.014 to 113.306 ms. The 10.78% nominal throughput difference
therefore includes a changed output workload and acceptance pattern. It is
not an isolated backend gain, and the candidate remains unqualified.

The final Q3 margin-4 GPU screen measured 26.0980 versus CPU 23.8394
tokens/s. Outputs differed at warmup token 8 and timed token 2; actual
cache capacity matched. All seven combined Q3 screens completed cleanly,
but none reached 30 tokens/s. Shared-projection and per-layer affinity
experiments are now running on the same Q23 weights before testing Q22.

Sharing attention input projections across split verification groups preserved
all 256 tokens for Q4 depth one but slowed the fresh paired screen from
25.6470 to 22.4984 tokens/s (-12.28%). Keep this option disabled for that
configuration. The upcoming Q22 overlay has identical tensor names/shapes
to Q23; a header-only audit found 39 down projections changing Q3_K to Q2_K,
reducing expert bytes by 9.071%. Payload equality and a speed gain are not
implied by this header audit.

Shared projections also slowed Q3 depth-three verification from 23.5738
to 20.3630 tokens/s (-13.62%), while preserving all 256 tokens. The option
is rejected for both tested configurations; the existing unshared path
remains selected for subsequent screens.

On TR16 with Q4 dense weights and ordinary depth-one acceptance, layer
affinity 0.06/0.10/0.14 measured 25.0580 tokens/s, while 0.08/0.14/0.18
measured 26.7316 versus the flat 0.08 control at 25.6470. The groups are
layers 3–16, 17–23 and 24–44. These are short screens with different
routing; their quality pilots are pending, and neither reaches 30 tokens/s.

The 0.06/0.12/0.20 layer-affinity profile measured 24.9453 tokens/s, leaving
the 0.08/0.14/0.18 profile fastest at 26.7316. All six projection/profile
screens completed cleanly. The fastest and gentlest profiles now undergo
2,048-target quality pilots before any Q22 combination is selected.

The fastest layer-affinity profile failed all provisional low-loss limits
on the completed 2,048-target pilot: perplexity 3.92497 (+4.03%), KL
0.166487 (+0.063602), and top-1 agreement 0.879395 (-1.318 percentage
points). Its perplexity alone falls within the moderate exploratory budget;
this permits testing a Q22 combination but does not qualify the profile.

The gentler profile completed its 2,048-target pilot with perplexity 3.84559
(+1.93%, within the provisional low-loss limit), KL 0.142567 (+0.039682),
and top-1 agreement 0.882324 (-1.025 percentage points). KL and top-1 fail
the limits, so neither tested layer profile is qualified. The Q22 speed
stage now begins with a fresh Q23 control, followed by the intentional
down-projection precision change and a conditional mix2 combination.

The fresh Q23 control measured 25.6326 decode and 131.284 prefill tokens/s,
preserving all 256 tokens from the prior Q4 control. Q22 without affinity
measured 22.0011 decode and 143.437 prefill tokens/s. At matched affinity
0.08, Q22 measured 25.4214 decode and 143.310 prefill tokens/s. Each result
uses a 2K training prompt, one 128-token warm-up and one 128-token timed
decode; these are screens, not held-out qualifications. Prefill excludes
cold loading and cache preparation.

With the same 6656 MiB cache request, Q22 admitted 845 slots (6654.38 MiB)
versus Q23's 734 (6376.62 MiB). In the matched-affinity timed trial, reported
CPU expert bytes fell from 219.786 to 204.759 GB and layer-flow time from
3046.64 to 2850.36 ms. Draft time rose from 341.91 to 438.765 ms, with
71/72 verification rounds and 56/55 accepted draft tokens respectively.
The changed weights, routing and outputs prevent attributing the total
throughput difference to one component. Reported expert bytes are work
counters, not unique physical DRAM transactions. Q22 pruning, deeper
drafting and layer-affinity screens continue before quality evaluation.

Q22 pruning at minimum route shares 0.04 and 0.08 measured 24.4473 and
24.6502 decode tokens/s, both below the 25.4214 unpruned screen. The 0.08
case reported 193.354 GB of CPU expert work and 2662.41 ms of layer-flow
time, but accepted only 48 drafts in 78 rounds, versus 55 in 72 unpruned.
Reducing expert work alone did not improve delivered throughput.

An isolated follow-up adds `STRATA_GLM_ROUTE_PRUNE_COLD_ONLY=1`. For the
fixed-cache experiment with positive residency affinity, it protects routes
with positive bonuses and prunes only nonpositive-bonus routes. A null
bonus disables pruning, including for the draft router. Renormalization,
when enabled, still rescales retained weights. This is not a general
residency query for other bonus policies. The default is off.

CUDA/HIP builds, five CUDA parity suites and all 20 routing-oracle configurations
passed. HIP runtime behavior was not tested on AMD hardware. The oracle covers null, mixed, all-positive and all-zero bonus
tables, serial/warp/batched/mailbox routing, both renormalization modes,
and empty-retained-mass fallback. Full-model controls and four candidate
screens have started with the unchanged-output controls; no speed or quality benefit from
this option has been measured.

All seven Q22 speed screens completed cleanly. Depth-three drafting with
acceptance margin 2 was fastest at 28.0924 decode tokens/s (143.2665
prefill), with 87 accepted drafts in 40 timed verification rounds. Fifteen
accepted drafts used relaxed acceptance; their summed logit regret was
15.0247 and maximum regret 1.88914. The two 128-token continuations had
distinct four-gram fractions of 0.992 and no obvious sustained repetition
on inspection. This short diagnostic does not qualify long generation.

The Q22 layer-affinity combination measured 24.9695 decode tokens/s,
below flat affinity 0.08. The queued 2,048-target pilots evaluate the flat
and fastest Q22 configurations with relaxed acceptance disabled. They
measure precision/routing loss, not the generated-task quality of the
margin-2 policy. Cache/drafting follow-up screens are prepared separately;
they are queued after the cold-pruning speed and quality stages.

The flat-affinity Q22 pilot completed all 2,048 targets: perplexity
3.956559 (+4.8676% versus Q23), KL 0.165847 (+0.062961), and top-1
agreement 0.868164 (-2.4414 percentage points). It fails all provisional
lower-loss limits. Perplexity is inside the moderate exploratory 10%
budget, permitting further moderate-loss tests but not qualification.

Both Q22 pilot configurations produced identical 2,048-target metrics with
relaxed acceptance disabled. The bandwidth-only prefill estimate from
101,622,624,768 reported upload bytes per pass and the independently measured
7.2128 GB/s H2D rate is 14.0892 seconds, or 145.36 tokens/s at 2K. The
143.31 measurement is 98.59% of that estimate. This is a component estimate
anchored to measured bandwidth, not a theoretical device limit. It excludes
compute and synchronization. At 8K a single upload pass would imply 581.44
tokens/s before those costs; no corresponding Q22 8K measurement is claimed.

The new pruning build's first full-model control, with the option disabled,
preserved all 256 tokens and the 845-slot cache fingerprint from the previous
Q22 runner. Decode measured 25.4149 versus 25.4214 tokens/s previously.
The existing-pruning control also preserved all 256 tokens and its cache
fingerprint, measuring 24.5569 versus 24.6502 tokens/s previously. The
zero-threshold control passed the same token and cache checks at 25.4205
tokens/s. All three short model controls passed; this does not qualify the
changed-routing candidates or their long-generation behavior.

At minimum share 0.08, protected-route pruning measured 25.7503 tokens/s
without renormalization and 25.8718 with it, versus the fresh 25.4149
control. In the non-renormalized trial, reported CPU expert work fell from
204.759 to 178.945 GB and layer-flow time from 2840.47 to 2508.73 ms, while
throughput increased only 1.32%. Different routing and generated outputs
prevent treating this as a pure kernel comparison. Both candidates remain
below 30 tokens/s, with quality pilots pending.

At minimum share 0.12, the non-renormalized candidate reached 29.3498
tokens/s with ordinary depth-one acceptance; renormalization measured
28.1126. The fastest case reported 97.447 GB of CPU expert work and
1476.92 ms of layer-flow time, versus 204.759 GB and 2840.47 ms for the
control. Its speed gain was 15.48%, despite roughly halving that CPU work.
All seven controls/screens completed cleanly. The fastest mild and strong
variants now undergo corpus pilots. Short excerpts showed no obvious
sustained generation loop; repeated quoted signatures reduced four-gram
uniqueness in some excerpts. No long-generation or task-quality claim is
made from these 128-token samples.

### Remaining decode-time gap, 2026-10-10

The Q22/Q4 depth-three margin-four held-out runs measured 31.25 tokens/s
at 2K and 34.34 at 8K (512 generated tokens, three timed repetitions).
Reaching 50 on those median timings requires total time reductions of
37.5% and 31.3%, respectively. The reported expert-flow timer alone was
23.28–23.99 ms per generated step at 2K and 20.56–21.25 ms at 8K, compared
with a 20 ms total budget at 50 tokens/s. These timers can include waiting
and overlap other work; they are not additive components or a hardware
ceiling. Logical expert bytes divided by flow time were about 72 GB/s;
this counter is not a measurement of physical DRAM transactions. Further
work needs to reduce target verification and expert-flow costs together,
while preserving task quality. See the held-out `target50-cost-analysis.json`.

### Q22 repetition-guard regression, 2026-10-10

The previously built guard-off decoder reproduced all 256 reference tokens
and the cache fingerprint on the Q22 short screen (30.46 tokens/s). Guard
four measured 30.58 with identical tokens and zero blocked proposals there;
that sample establishes no active-guard quality benefit. On a fresh
HumanEval/31 request, guard off reproduced the original failing 2,048-token
loop exactly, including its token hash. Guard four blocked two proposals,
produced 174 tokens and passed the isolated test. The paired task used the
same prompt, cap, model and build. Full guarded HumanEval was stopped at the user’s wrap-up request with
9 of 164 tasks completed, all nine passing (`complete=false`). Guarded
held-out timing and overall quality remain unqualified. See
[the paused handoff](../resume-tr16.md) for remaining work and a resume prompt.

## 35/45-token/s follow-up, 2026-10-10 afternoon

The user set new targets: 35+ decode tokens/s with lesser loss, and 45+ with
moderate loss where coding quality stays nearly lossless. The authorized window
was four hours of serial GPU time from 13:16 JST; quality evaluation was
deferred to the next window by that decision. Agreed numeric gates are recorded
in `build-tr16-35-20261010/goal.json`: the 35 track allows perplexity +6%, KL
+0.08 and a top-1 drop of 3 percentage points against unchanged Q23, with at
least 152/164 HumanEval passes; the 45 track allows +10%, +0.12 and 5 points
with the same HumanEval floor, and is defined on the 8K code held-out prompt
with the 2K rate reported alongside. The old provisional lower-loss gate is
unattainable for any candidate built on the Q22 pack, which alone measured
+3.95% perplexity. No profile below is quality-qualified; speed screens select
candidates only.

Evidence is under `build-tr16-35-20261010/` (gitignored) with an export in
`docs/fixtures/glm_tr16_35_20261010/`. All screens use the gpu-cache-v2 research
binary (`47ddc27e…`), the Q22 pack, Q4 dense weights, affinity 0.08, CPU draft
depth 3 and soft margin 4 unless a row says otherwise, on the 2K training code
prompt with one 128-token warm-up and one timed 128-token continuation.

### Control and decomposition

The fresh control reproduced all 256 reference tokens, the 845-slot /
6654.38 MiB tier and its fingerprint, at 29.98 tokens/s (the archived screen
measured 30.44). Token-identical reruns during the day ranged 29.98–30.45, so
differences under about 2% are noise at this protocol.

A `STRATA_GLM_STEP_TRACE=2` diagnostic of the same configuration (run outside
the calibrator, never a throughput record) split the timed trial's 94.5 ms
verification round into 81.1 ms waiting on the CPU expert pass, 7.5 ms of GPU
dense work between MoE layers, 3.3 ms head and 2.6 ms tail (0.9 ms sync,
0.5 ms planning, 1.1 ms copies). Per generated token that is 23.4 ms of
expert wait against 3.9 ms of everything else the GPU side exposes. Exact
kernel levers can therefore address at most about 4 ms per token; bytes per
token and tokens per round decide the rest.

### Single-lever screens

| Lever | Decode tok/s | Change | Tokens/round | GB/token | Outcome |
|---|---:|---:|---:|---:|---|
| Control | 29.98 | — | 3.46 | 1.654 | reference |
| Soft margin 5 / 6 | 29.06 / 29.01 | −3.1% / −3.2% | 3.37 | 1.755 | rejected |
| Max soft accepts 4 / 5 | 30.27 / 30.45 | noise | 3.46 | 1.654 | token-identical, inert at depth 3 |
| Accept top-k 6 | 30.76 | +2.6% | 3.56 | 1.642 | ingredient |
| Draft depth 4 / 5 | 26.75 / 26.52 | −10.8% / −11.5% | 3.88 / 4.41 | 1.974 / 2.012 | rejected: union bytes outgrow tokens |
| Lookup drafts depth 7, min match 6 | 30.67 | +2.3% | 3.66 | 1.691 | ingredient; lossy under soft accept |
| Lookup depth 7, min match 3 | 34.26 | +14.3% | 4.92 | 1.584 | **rejected: looped** (distinct 4-gram 0.53) |
| `STRATA_GLM_MTP_REUSE_DRAFT=1` | 31.64 | +5.5% | 3.66 | 1.605 | kept |
| Recursive blend 0.5 (draft-route binary) | 30.75 | +2.6% | 3.66 | 1.674 | ingredient, guard binary only |
| Affinity 0.10 | 31.17 | +4.0% | 3.56 | 1.577 | kept |
| Affinity 0.16 | 36.06 | +20.3% | 3.76 | 1.141 | kept for the 45 track |
| Affinity 0.12 + unbiased learner | 29.19 | −2.6% | 3.28 | 1.577 | rejected |
| Per-layer profile mix2 (0.08/0.14/0.18) | 32.40 | +8.1% | 3.46 | 1.377 | kept |
| `FUSE_ROUTE=1` | 30.26 | noise | 3.46 | 1.654 | token-identical, no gain |
| MLA kernels with graphs off | 29.27 | −2.4% | 3.46 | 1.725 | rejected, tokens changed |
| Tier 6796 / 6898 / 7284 MiB | 29.17 / 28.81 / 28.85 | −2.7% to −3.9% | 3.20–3.37 | 1.68–1.74 | rejected: routing changed, more bytes |
| GPU draft with reserved cache | 27.56 | −8.1% | 3.20 | 1.931 | rejected: tier fell to 4292 MiB |

The affinity 0.16 continuation quoted the reviewed code's function signature
twice, which lowered its distinct 4-gram fraction to 0.90; the decoded text is
coherent and the screen rule was changed to an absolute 0.85 floor plus a
no-repeated-32-gram check, recorded as `decision_v1`/`decision` in the
summaries. The margin 5/6 and depth 4/5 rows were already slower, so the rule
change promoted only affinity 0.16. One affinity 0.12 attempt was rejected by
the quiet-run guard because another session's `npm exec chrome-devtools-mcp`
start-up used a full core for a minute; it was rerun afterwards.

### 512-token combinations and confirmations

The 128-token screens misled on every acceptance-sensitive lever. At 512
generated tokens on the same prompt the control measured 32.97 tokens/s (the
longer continuation amortizes per-request overhead), and:

| Configuration (512 tokens, one timed trial) | Decode tok/s | Change | Tokens/round | GB/token | Distinct 4-grams (warm-up, timed) | Outcome |
|---|---:|---:|---:|---:|---|---|
| Control | 32.97 | — | 3.44 | 1.495 | 0.988, 0.864 | reference |
| Affinity 0.10 + reuse-draft | 31.05 | −5.8% | 3.20 | 1.534 | 0.904, 0.957 | rejected |
| Affinity 0.12 + reuse-draft | 32.63 | −1.0% | 3.20 | 1.411 | **0.497**, 0.957 | rejected, warm-up looped |
| Mix2 + reuse-draft | 35.64 | +8.1% | 3.56 | 1.295 | **0.607, 0.589** | rejected: looped |
| Affinity 0.16 + reuse-draft | 38.65 | +17.3% | 3.74 | 1.029 | 0.969, **0.711** | rejected: looped |
| Affinity 0.16 + reuse + top-k 6 + lookup 7 | 46.61 | +41.4% | 5.12 | 0.990 | **0.505, 0.371** | rejected: looped |
| Mix2 + reuse + top-k 6 | 33.66 | +2.1% | 3.35 | 1.291 | 0.912, 0.971 | noise |
| Affinity 0.16 alone | 32.34 | −1.9% | 3.12 | 1.226 | 0.890, 0.890 | no gain |
| Mix2 alone | 32.72 | −0.7% | 3.26 | 1.370 | 0.866, 0.961 | no gain |
| Reuse-draft alone | 29.85 | −9.5% | 3.08 | 1.674 | 0.908, 0.959 | rejected |
| Affinity 0.12 alone | 36.64 | +11.1% | 3.61 | 1.259 | **0.303, 0.426** | rejected: looped |
| **Affinity 0.16 + top-k 6** | **39.60** | **+20.1%** | 3.85 | 1.037 | 0.898, 0.917 | candidate; text inspected, coherent (quotes the reviewed source verbatim, which drafts easily) |

`STRATA_GLM_MTP_REUSE_DRAFT=1` is rejected: alone it loses 9.5% at 512 tokens
and every combination containing it looped. Routing levers alone reduce bytes
per token by 15–30% but lose draft acceptance over long continuations and end
flat. The 46.6-token/s row is the same failure as the earlier 50+ runs: a
repeated passage that every draft predicts. Loops appear without the repeat
guard whenever routing changes under soft acceptance; the same three routing
settings run on the draft-route binary with `STRATA_GLM_MTP_REPEAT_GUARD=4`
stayed clean at 33.82 (affinity 0.12), 33.58 (affinity 0.16) and 33.96 (mix2)
tokens/s, +1.9% to +3.0% over the 512-token control. The guard is therefore a
requirement for any routing candidate, and the merged binary (guard plus the
gpu-cache-v2 features) must be built before quality work.

On the 8K held-out code prompt (512 tokens, one timed trial; same-day control
34.08 tokens/s, 3.58 tokens/round, 1.530 GB/token): top-k 6 measured 35.15
(+3.1%), mix2 + top-k 6 35.73 (+4.8%), affinity 0.12 34.37, mix2 alone 33.88,
and lookup drafts 30.10 (−11.7%; suffix matches displaced better proposals).
None looped.

### Held-out timings, 512 generated tokens, three timed repetitions

Held-out code prompts (`build-tr16-20261009/prompts{2k,8k}/holdout-code.ids`)
were used only here. Controls are the archived soft4 runs of the same binary
and configuration (31.2495 at 2K, 34.3397 at 8K). The guard-free gpu-cache-v2
binary was used, so these are speed records only; the guarded merged binary
must re-time any finalist before quality work.

| Configuration | 2K median (trials) | 8K median (trials) | Tokens/round 2K / 8K | GB/token 2K / 8K | Distinct 4-grams, all streams |
|---|---:|---:|---:|---:|---|
| Soft4 control (archived) | 31.25 | 34.34 | 3.30 / 3.57 | 1.70 / 1.53 | inspected earlier |
| **Affinity 0.16 + accept top-k 6** | **36.72** (37.76, 36.59, 36.72) | **36.35** (36.35, 35.84, 37.26) | 3.51 / 3.48 | 1.17 / 1.18 | 0.97–0.99 / 0.95–0.98, no repeated 32-gram |

Per-position acceptance for the candidate was 0.95 / 0.85 / 0.71 at 2K and
0.96 / 0.84 / 0.67 at 8K, with 293 and 267 soft accepts over the 1,536 timed
tokens (mean logit regret 0.27 and 0.24 per token). Prefill was unchanged at
143 tokens/s. The 2K gain comes from bytes per token (1.17 versus 1.70 GB);
at 8K the control already had high acceptance, so the gain is 5.8%.

The two 8K-only keeps were confirmed at 512×3 on the 8K held-out prompt:
top-k 6 alone measured 34.79 (35.16, 34.79, 33.98) and mix2 + top-k 6 34.56
(35.34, 34.45, 34.56) tokens/s, both clean but below 35 on the full protocol
(+1.3% and +0.7% over the archived control). They are not finalists.

### Status at the end of the window

- **35 track:** affinity 0.16 + top-k 6 is the only configuration above 35 on
  both held-out prompts (36.72 / 36.35). Its distribution loss is unmeasured;
  affinity 0.20 earlier measured +10.4% perplexity and 0.08 +2.07%, so 0.16 is
  more likely to fit the 45-track corpus gate than the 35-track one. Guarded
  re-timing on the merged binary and the full quality ladder are required.
- **45 track:** no non-looping run reached 45 at any context. The DRAM bound
  (1.17 GB/token at about 72 GB/s is already 16 ms of a 22 ms budget) leaves
  no admissible lever in this set; every run above 40 looped.
- Prepared, not run: `build-tr16-35-20261010/build-merged.py` (merged source
  with the cold-prune features and the draft-route repeat guard, one hunk
  merged by hand in the routing-oracle test), `quality-ladder.py` (task 31,
  20-task smoke, 2,048-target pilot, full 8,192 corpus, full 164 HumanEval
  against the agreed gates), `finalize.py` for held-out timing. The 128-token
  screening protocol is retired for acceptance-sensitive levers; `screen.py`
  keeps its decisions and per-trial loop proxies in each `summary.json`.

A final pair of 512-token runs of the candidate on the draft-route binary with
`STRATA_GLM_MTP_REPEAT_GUARD=4` measured 39.20 tokens/s on the 2K training
prompt (unguarded 39.60) and 36.02 on the 8K held-out prompt (unguarded
first trial 36.35). The guard blocked zero proposals and the acceptance and
byte counters matched the unguarded runs, so guard overhead is within noise
for this candidate. Its full held-out 512×3 timing on the merged binary and
the quality ladder remain for the next window.

### Extended window (to 21:16 JST): guarded candidates and quality

The user extended the window to eight hours. The draft-route binary
(`39afb8af…`, validated earlier with CUDA parity suites, a HIP build, model
controls and 21,504 guard cases) supports every flag the candidates use, so
the merged build was not needed for them and was left unbuilt.

Guard-on 2K training-prompt screens (512 tokens, one timed trial, control
32.97): affinity 0.12 + top-k 6 33.50; affinity 0.14 + top-k 6 37.63; mix2 +
top-k 6 35.38; affinity 0.16 + top-k 8 38.10; affinity 0.16 + top-k 6 +
recursive blend 0.5 35.69 (blend lowers acceptance; rejected). All clean.

Held-out 512×3 with the guard (archived control 31.25 / 34.34):

| Configuration (guard 4) | 2K median (trials) | 8K median (trials) | Tokens/round | GB/token | Guard blocked (per trial) |
|---|---:|---:|---:|---:|---|
| Affinity 0.14 + top-k 6 | 36.21 (34.76, 36.76, 36.21) | 38.14 (38.72, 38.14, 36.50) | 3.46 / 3.63 | 1.25 / 1.18 | 2K 0/0/1, 8K 0/4/3 |

No stream repeated a 32-gram; one 8K stream restated a sentence once (a mild
soft repetition, inspected, not a sustained loop).

Quality ladder for affinity 0.16 + top-k 6 (guard 4, same binary): HumanEval/31
passed without reaching the cap; the first 20 tasks scored 19/20 (baseline
20/20, lost HumanEval/10, no cap hits); the 2,048-target corpus pilot then
measured perplexity 4.1765 (+10.70% versus the unchanged-Q23 pilot control
3.7729), KL +0.1259 and top-1 agreement −3.76 percentage points. That fails
the 35-track limits even at the pilot's 1.5× allowance, and would also exceed
the 45-track corpus limits, so the ladder stopped before the full corpus and
the 65-minute full HumanEval. The Q22 pack alone measured +4.87% on this pilot,
so affinity 0.16 adds roughly six points of perplexity.

Corpus pilots (2,048 targets, margin 0, versus the unchanged-Q23 pilot control;
the Q22 pack alone is +4.87% here) for the guarded top-k 6 candidates:

| Candidate (guard 4, top-k 6) | Perplexity | KL | Top-1 drop | 35-track pilot gate (1.5× limits) |
|---|---:|---:|---:|---|
| Affinity 0.16 | +10.70% | +0.126 | 3.76 pp | fails |
| Affinity 0.14 | +9.47% | +0.113 | 2.78 pp | fails on perplexity |
| Affinity 0.12 | +7.67% | +0.085 | 2.15 pp | passes |
| Mix2 profile | +6.92% | +0.088 | 3.22 pp | passes |

Affinity 0.14 + top-k 6 was still evaluated on the full 8,192-target corpus
because it is the only candidate above 35 tokens/s at both held-out contexts:
perplexity 5.233740 (+8.45% versus 4.826016), KL +0.0998 and top-1 agreement
−3.61 percentage points. It fails every 35-track limit (+6%, +0.08, 3 pp) and
would fit the 45-track corpus limits, whose speed it does not meet. Its full
HumanEval was run regardless, as the user's stated priority is coding quality;
the result is recorded below.

### Prefill plan from the simulator (extended window)

While the quality chain held the GPU, the calibrated TR16 simulator
(`tools/sim/data/params_tr16_20261009.json`, Q22 pack, Q4 dense, MTP depth 3,
affinity 0.14) was used to rank prefill levers. Its 2K prediction matches the
measurement (136.7 unstreamed versus 143.3 measured; 144.7 with streaming),
and it reproduces the archived streaming result's order of magnitude at 8K.

| Prefill configuration (Q22) | Predicted tok/s | Bottleneck | Note |
|---|---:|---|---|
| 2K prompt, one 2K chunk | 136.7 | PCIe (101.6 GB in 14.2 s) | measured 143.3 |
| 2K, streaming prefetch | 144.7 | PCIe | upload floor; only fewer bytes help further |
| 2K, CPU assist | 137.3 | PCIe | +0.4%, not worth an implementation |
| 8K prompt, four 2K chunks | 136.7 | PCIe, 4 uploads | **what the candidate configs use today** |
| 8K, two 4K chunks | 289.3 | PCIe | |
| 8K, one 8K pass | 393.1 | PCIe | config change only (`prefill_batch` 8192, 2 GiB scratch cap) |
| 8K, one pass, streaming prefetch | 492.4 | **GPU** (mixers 10.4 s, GEMMs 6.2 s) | needs the streaming patch ported |
| 8K, one pass, streaming + CPU assist | 497.8 | GPU | +1% |

At one 8K pass the candidate's prefill becomes GPU-compute bound, so beyond
streaming the remaining levers are attention/mixer and MMQ GEMM time, not
PCIe. The streaming patch (`docs/fixtures/glm_tr16_20261009/stream-prefetch.patch`)
applies cleanly onto the draft-route source; the ported source is
`build-tr16-35-20261010/stream-source` with `stream-incremental.patch`, built
as `stream-cuda` at low priority beside the HumanEval run. The post-chain
measurements (`run-stage3i.sh`) time the candidate with a single 8K pass on the
validated binary, then on the streaming binary after its GPU parity tests.

The first full HumanEval attempt for affinity 0.14 + top-k 6 (guard 4) was
rejected by the quiet-run guard after 30 tasks (all 30 passed) when a desktop
Chrome GPU process appeared as an external CUDA client; its partial result in
`q-g6-aff014-topk6-guard4-full/` is interrupted evidence, not a score. The
chain was relaunched sequentially (`run-chain2.sh`) with hour-long idle waits
and a fresh HumanEval output name.

### Coding quality verdict and prefill measurement (evening)

The full 164-task HumanEval for affinity 0.14 + top-k 6 (guard 4) scored
**144/164 against the 154 baseline**: twelve tasks lost (36, 37, 49, 59, 71, 94,
100, 103, 109, 120, 132, 134), two gained (110, 129), and task 116 hit the
2,048-token cap. That is a net loss of ten, far outside the agreed limit of two,
so the affinity-based 35-token/s candidates are not "coding nearly lossless"
and none is promoted. Mix2 + top-k 6 measured 30.84 (2K) / 32.93 (8K) on the
held-out prompts with the guard and perplexity +6.94%, KL +0.085, top-1 −3.09
points on the full corpus: below the speed target and outside the loss limit.

The remaining open question is the soft-acceptance control itself (affinity
0.08, margin 4, top-k 4, guard 4, about 31 tok/s at 2K and 34 at 8K, corpus
+3.95%): its full HumanEval was started at the end of the window and is the
first number to read in `build-tr16-35-20261010/q-soft4-aff08-guard4-he/`.

Prefill: with `--prefill-batch=8192` (2 GiB scratch cap) the candidate measured
**425.2 tokens/s** on the 8K held-out prompt (423.7 and 426.7 in the two timed
passes) against 143 with four 2K chunks, with decode and memory unchanged
(peak 12,814 MiB). The simulator predicted 393. Every 8K profile should use the
single pass. The streaming-prefetch build (`stream-cuda`, draft-route source plus
the archived streaming patch) was built but its timed run was rejected by the
quiet-run guard (another process compiled during it); it is queued to rerun after
the HumanEval, with its GPU parity tests first.

### Soft-acceptance control: full HumanEval (final result of the window)

Q22 pack, Q4 dense, affinity 0.08, MTP depth 3, soft margin 4, top-k 4, max
soft 3, repetition guard 4 (draft-route binary): **151/164** against the 154
baseline. Lost 62, 94, 100, 113, 132, 137, 163; gained 97, 110, 116, 129; no task
reached the 2,048-token cap (the unguarded margin-4 loop on task 31 is gone).
The agreed limit is 152, so it misses the coding gate by one task; its corpus
loss (perplexity +3.95%, KL +0.058, top-1 −2.27 pt) is inside the 35-track
limits, and its speed is 31.25 (2K) / 34.34 (8K) held-out, below 35. It is the
fastest configuration within one task of the coding gate, but it qualifies for
neither track. The qualified profile remains affinity 0.08 MTP1 on Q23 at 23
tokens/s (153/164).

### Streaming prefetch on a single 8K pass (2026-10-10 22:50)

The archived streaming-prefetch patch (`STRATA_GLM_PREFETCH_STREAM=1`, 18 groups),
ported onto the draft-route source and built as `build-tr16-35-20261010/stream-cuda`
(CUDA parity suites `glm_q23_parity`, `glm_q8_test`, `glm_mailbox_test`,
`glm_prefill_parity`, `glm_parity` passed), measured **510.1 tokens/s** (510.4,
509.8) on the 8K held-out code prompt with `--prefill-batch=8192`, against 425.2 for
the same configuration without streaming (+20%). Configuration: Q22 pack, Q4 dense,
affinity 0.14, top-k 6, guard 4, MTP depth 3; peak GPU 12,814 MiB in both. The
simulator predicted 492. This passes the original 500 tokens/s 8K prefill target
for the first time on TR16 (the earlier best was 474.9 with Q23 experts and an
8K batch). Decode in
the 128-token check was 32.25 tokens/s (one short trial, not comparable to the
512×3 numbers). Two earlier attempts were rejected by the quiet-run guard (another
session's compile, then kernel memory compaction while 118 GB of page cache was in
use); evicting the cache of unused large files fixed the second.

## Speculative decoding bounds on TR16 (2026-10-11)

Question: can a different speculative algorithm reach 40+ tokens/s losslessly? Inputs: the soft4 control's
measured per-position acceptance (0.98 / 0.88 / 0.71 at depth 3, 3.58 tokens per round, 8K held-out code) and
the simulator's verify cost per window width (`tools/sim`, Q22 / Q4 dense / affinity 0.08, scaled to the
measured 91 ms at width 4): 45.9, 58.9, 83.9, 94.5, 110.9, 127.5, 146.5, 165.7 ms for widths 1-8, draft 4.0 ms
per MTP token. Scripts: `build-spec/adaptive_bound.py`, `build-spec/suffix_replay.py` (research directory).

- **Draft length.** Fixed depth 3 predicts 33.2 tokens/s (measured 34.1). An oracle that verifies exactly up to
  the first rejection, depth up to 7, reaches 38.1. A confidence gate with a realistic predictor (AUC 0.75-0.95)
  gives 32.3-35.0. Each extra verified token costs 16-19 ms of CPU expert bytes, so utility-optimal adaptive
  depth (EVICT, arXiv 2605.00342) cannot reach 40 here.
- **Model-free drafting.** Replaying the logged held-out outputs through a suffix matcher: on the review/explain
  prompts a cost-aware hybrid (suffix draft only when its expected tokens per ms beats MTP) never chooses the
  suffix draft and equals MTP; the old rule (any match replaces MTP) loses 6-9%, matching the measured -11.7%.
  Suffix drafting pays only when the answer repeats the context (code edits, refactors, agent tool loops).
- **Planner.** `glm_sim.py plan --target-decode 40` finds nothing lossless at 40; every 40+ candidate combines a
  block drafter (DFlash) with lossy routing levers.
- **Cost-aware lookup policy** (`STRATA_GLM_MTP_LOOKUP_POLICY=cost`, patch `build-spec/cost-policy.patch` on the
  streaming source, binary `build-spec/cost-cuda`): a lookup (suffix) draft replaces the MTP draft only when its
  expected tokens per ms is higher. Per-token acceptance is learned per match-length bucket from optimistic starts
  (0.55 / 0.75 / 0.88 / 0.95 for 4-7 / 8-15 / 16-23 / 24+ tokens); matches of 24+ tokens are always taken;
  verify cost per window width starts from TR16 measurements (4 wide 101-103 ms, 8 wide 156-179 ms) and is
  refined online; lookup tokens are accepted only on exact target matches (lossless). Indicative screen
  (2026-10-12, another agent's builds on the machine; soft4 Q22 base, MTP depth 3, 512 tokens x 3, medians):

  | Prompt | MTP only | Any match replaces MTP (old rule) | Cost policy |
  |---|---:|---:|---:|
  | edit-cpp (rewrite a C++ file with comments) | 31.9 | 39.3 | **38.6 (+21%)** |
  | edit-py (reformat a Python file) | 33.8 | 32.1 | **33.6 (−0.6%)** |
  | 8K held-out code review | 28.5 | 27.6 | **28.2 (−1%)** |

  Lookup acceptance: edit-cpp 92%, edit-py about 53-62%, review about 41%. Earlier versions show why each rule
  matters: pessimistic priors (0.3-0.7) took no lookup at all (edit-cpp 31.2); without the 24+ rule, +10-15%.

  Ported to the main decoder (`src/program/glm_decode.cpp`, opt-in, exact acceptance only). Screen on TR16
  (2026-10-12, machine quiet; `configs/glm53f-q2-v3.json`: assembled Q23, MTP depth 3, lookup depth 7 with
  minimum match 4, GPU budget 13824 MiB, 512 tokens x 3, medians):

  | Prompt | MTP only | Any match replaces MTP | Cost policy, research-build costs | Cost policy, this setup's costs |
  |---|---:|---:|---:|---:|
  | edit-cpp | 20.1 | 21.5 (+7%) | 20.8 (+3.5%) | 20.9 (+4%) |
  | edit-py | 22.6 | 19.9 (−12%) | 21.4 (−5%) | 22.1 (−2%) |
  | 8K held-out code review | 14.7 | 14.3 (−3%) | 15.3 | 15.3 |

  On the review prompt the policy took no lookup round and ran the same rounds as MTP only; the 14.7 vs 15.3
  difference is run-to-run drift, not the policy. The cost-policy output matched MTP only token for token on
  all three prompts; the replace rule's output differed on the review prompt. On this configuration verify
  time grows about 31 ms per window token (4-token windows 147-149 ms, 7-token lookup windows 239-255 ms,
  derived from the replace runs), twice the research build's slope, so lookup windows gain less here. The
  shipped starting costs are these measurements; the research-build costs (4 wide 102 ms, about 15 ms per
  token) made the policy take losing lookup rounds on edit-py. One edit-py cost run was stopped by the GPU
  memory reserve guard (483 MiB free of the 512 MiB reserve at a 13824 MiB budget) and was repeated.
