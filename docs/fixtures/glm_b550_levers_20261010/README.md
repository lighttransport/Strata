# B550 decode optimization, 2026-10-10

Result: **33.57 / 33.61 tok/s** with coherent short code on stock DDR4-2133, using Q22 fit4.
The measured **35.07 tok/s** variant failed the coding gate and is not selected.

Target: 35+ decode tok/s with coherent generated code. DDR clock unchanged.
Hardware: Ryzen 9 3950X, RX 9070 XT 16 GB, DDR4-2133 (all four configured DIMM speeds confirmed with udevadm).
Model: GLM-5.3-Flash REAP-50 Q23 assembled GGUF. Same 2048-token prompt, 128 greedy generated tokens,
4096 context, 12 CPU threads, 15360 MiB GPU budget and 12288 MiB decode expert cache as levers2.

All runs use the current `build-hip/strata-glm-decode`, the step-1 environment, and the opt-in
HC_FUSED / TIER_DIRECT / MLA_KERNELS / HEADS_BLOCKS / FUSE_ROUTE switches. Affinity and deferral are lossy.
Defaults are unchanged. Results are exploratory until confirmed in alternating rounds and quality checked.

| Candidate | tok/s | CPU / GPU wait / head / tail ms per token |
|---|---:|---|
| mix2 + defer 2, same-day baseline | 28.1172 | 21.3 / 10.3 / 2.2 / 1.6 |
| mix2 + defer 3 | 26.8444 | 23.9 / 9.2 / 2.2 / 1.6 |

Baseline tail: stream synchronization 0.906, tier planning 0.335, logits copy and validation 0.318 ms/token.
The third deferred route raises CPU time and loses throughput. No quality test is planned for that regression.

`prof-fit*.json` explores six layer groups (3–9, 10–16, 17–23, 24–30, 31–37, 38–46), allocating
larger residency bonuses to later groups based on the previous sensitivity scan. Profile files and exact
commands are retained beside raw logs. `steptrace.sh` and the quality harnesses were copied from ~/work/kexp,
redirected into this directory, and the speed harness now points to the current decoder. The initial `baseline`
attempt used the calibration binary and did not complete; exclude it.

## Additional exploratory runs

| Candidate | tok/s | CPU / GPU wait / head / tail ms per token |
|---|---:|---|
| fit1 + defer 2 | 26.8495 | 24.2 / 9.0 / 2.2 / 1.6 |
| fit2 + defer 2 | 28.1440 | 21.0 / 10.4 / 2.2 / 1.6 |
| fit3 + defer 2 | 29.7173 | 17.5 / 12.2 / 2.2 / 1.6 |
| fit4 + defer 2 | 31.1979 | 13.9 / 14.1 / 2.2 / 1.6 |
| fit5 + defer 2 | 32.3254 | 11.2 / 15.8 / 2.2 / 1.5 |
| fit6 + defer 2 | 33.2688 | 8.1 / 17.9 / 2.2 / 1.6 |
| mix2 + defer 2 + KDA row parts 4 | 27.7059 | 22.8 / 9.3 / 2.1 / 1.6 |
| mix2 + defer 2 + KDA row parts 4, column tiles | 27.7128 | 22.8 / 9.3 / 2.1 / 1.6 |
| mix2 + defer 2 + dense Q4=q5 | 28.1124 | 21.3 / 10.3 / 2.2 / 1.6 |
| mix2 + defer 2 + dense Q4=all | 28.1078 | 21.4 / 10.2 / 2.2 / 1.5 |

The q5 conversion finds no eligible tensors in this assembled GGUF. The all conversion applies to some Q6_K
matrices, reduces dense allocation by about 153 MiB, and increases expert slots from 906 to 923 (7870.88 to
8018.56 MiB), without improving speed in this prompt. Most large dense weights are already Q4_K.
The parallel KDA paths change FP32 reduction order and the generated route sequence; lower GPU wait is offset
by more CPU work. They are not selected by these exploratory measurements.

## Rejected speed-only results

- fit7 + defer 2: 33.9259 tok/s; once-per-token lookup upload: 34.0578. HumanEval smoke: **4/10**.
- fit8 + full deferral + once-per-token lookup: **35.0154 tok/s**, CPU 3.6 / GPU wait 21.1 / head 2.1 /
  tail 1.5 ms. HumanEval smoke: **3/10**, with repeated expressions and 1024-token loops on several simple tasks.
  This reaches the speed number once, but fails the coding-coherence requirement. Do not select it.
- fit6 without deferral: 30.4181; full deferral: 33.5009; KDA columns 64: 33.2173; dense Q4=all: 33.3342.
- Once-per-token lookup on mix2 + defer 2: 28.0746 off, 28.1390 on, identical 128 generated token IDs.
  This small difference needs repeat measurements.

The pending repeat queue for fit8 was stopped after its ten-task smoke completed; the proposed extra speed
runs in `qualify-first.sh` were not executed. Historical first-ten-task scores for aff 0.08, mix2, and mix2 +
defer 1 were all 10/10. These ten-task subsets are rejection screens, not full HumanEval pass@1 estimates.

## Opt-in code experiments

`STRATA_GLM_TIER_LOOKUP_ONCE=1` uploads stable one-token pipeline residency before the layers. Other cache modes
keep their per-layer copies. `lookup-once.patch` isolates this experiment from the inherited uncommitted changes.

`STRATA_GLM_NORM_Q8_FUSED=1` produces normalized floats and native Q8_1 blocks together when a one-token decode
uses Q8 kernels and QUANT_ONCE. Its Q8 buffer persists across reset_phase(), is allocated before graph capture,
and shared activation state is cleared at batch boundaries and incompatible norms. Prefill and wider decode
calls retain the existing kernels. The HIP kernel test passed 32 bitwise cases: widths 32/256/4096/16384, one/four
rows, null/ordinary/large weights, and zero inputs. Existing glm_q8_test checks also passed.

`STRATA_GLM_DENSE_Q4=q3mixers` converts eligible large Q4_K/Q5_K/Q6_K mixer matrices to owned Q3_K GPU copies.
The first three layers, draft, shared experts, routers, output head and original GGUF remain unchanged.
This is lossy and unqualified until measured and quality checked. GPU prefill uses those copies too.

Full strata-glm-decode and glm_q8_test targets built on HIP (gfx1201) and CUDA (sm_89, CUDA 13.2).
CUDA was compiled only: this machine has an AMD GPU. SYCL does not build these GLM targets.

## Down-only Q22 sidecar

Generated `/mnt/disk01/models/glm53f-reap/REAP50-Q22-down-only-experimental.gguf`, 16,647,198,848 bytes,
42 Q2_K down-projection tensors. Source: the existing assembled Q23 GGUF. Gate/up remain in the source;
conversion is explicitly uncalibrated. Conversion completed in 16 min 13 s; original source fingerprint was
checked unchanged. The converter refuses existing output/partial files. Model headers remain bound to the
exact source generation and payload FNV checksums.

The opt-in `--down=q2` converter now accepts assembled Q3_K down inputs. `ModelArtifact::overlay_experts`
allows these down-only Q3_K -> Q2_K transitions on a validated assembly, retaining type/source/bounds/checksum
validation and rejecting a second sidecar. Native assembled overlays can use the existing single-node CPU
path; original-source overlays and packed CPU backends retain their NUMA constraints. The initial decoder
startup rejected the sidecar under the legacy two-node check; that run has no speed result and is saved as
`q22-startup-legacy-numa.err`.

Validation: seven synthetic assembly/overlay tests pass; real pack integration checks pass (correct source,
version/source mismatch, truncation, corrupted payload); HIP canonical expert, Q8, Q23 expert and mailbox tests
pass. The Q23 parity covers Q2_K/Q2_K expert arithmetic. Two test executables were initially absent and were
built before rerunning those tests. CUDA/HIP decoder and converter targets rebuilt after the single-node change.

The matched first-30-task results below qualify only that short-code subset. Run the sidecar with the original model plus `--expert-pack=<sidecar>`;
it is not a standalone model. `sweep-q22.sh` records the first comparisons.

## Q22 speed exploration (before the matched quality checks)

All runs here retain canonical expert arithmetic and the fused norm/Q8 switch unless noted.

| Candidate | tok/s | CPU / GPU wait ms per token |
|---|---:|---|
| Q22 mix2 + defer 2 | 30.0614 | 18.8 / 10.3 |
| Q22 fit4 + defer 2 | 33.4852 | 9.1 / 16.7 |
| Q22 fit4 + Q3 mixers + defer 2 | 33.6577 | 11.5 / 14.2 |
| Q22 fit10 + Q3 mixers + defer 2 | 33.0666 | 10.4 / 15.7 |
| Q22 fit4 + Q3 mixers + defer 2 cap .16 | 32.3537 | 7.9 / 18.9 |
| Q22 fit6 + Q3 mixers + defer 2 cap .16 | 32.0769 | 7.8 / 19.3 |
| Q22 fit4, GPU budget 16000 | 33.0832 | 10.6 / 15.6 |
| Q22 fit4, budget 16000 / tier reserve 256 | 33.3239 | 9.7 / 16.2 |
| Q22 fit4 + Q3 mixers, budget 16000 / reserve 256 | 33.8335 | 10.4 / 15.0 |
| Q22 fit8 + Q3 mixers + defer 2 cap .16 | 34.9952 | 2.8 / 21.9 |
| Q22 fit8 + defer 2 cap .16, budget 16000 / reserve 256 | 34.2080 | 2.6 / 22.8 |

Q22 cache at budget 15360: 1074 slots / 8457.75 MiB versus Q23's 906 / 7870.88. Q3 mixers increase Q22
cache to 1174 / 9245.25. Q22 fit4 at budget 16000: 1132 / 8914.5, free GPU memory 1010 MiB after decode;
reserve 256 increases it to 1164 / 9166.5. Larger cache changes affinity-selected routes and the generated
sequence, so speed does not improve monotonically. These are greedy prompt measurements, not matched-token
microbenchmarks. None is selected without coding checks.

The Q23 non-canonical native path regressed to 20.58 tok/s (CPU 29.6 / wait 15.0). On Q22, its generic GPU
expert kernel rejects down type 10. **Keep STRATA_GLM_CANON=1 with this Q22 sidecar.** The failed native run
has no speed result; its queue stopped at the unsupported type, so subsequent entries were not executed.

The new capped Q22 fit8 combination is distinct from the rejected uncapped Q23 fit8 configuration. Its
HumanEval rejection screen is reported below; rounded speed alone is insufficient evidence for selection.

The capped Q22 fit8/Q3 near-target combination failed its ten-task HumanEval rejection screen: **3/10**,
seven token-cap hits. It is rejected too. A post-run wrapper error (`12288: command not found`) came from
editing the shell harness while it was active; the inner Python evaluation completed all ten tasks, and
measurement.json and generated artifacts are intact. The current harness passes bash syntax validation.
Do not alter an active harness file. No speed confirmation is warranted for this rejected configuration.

The Q22 MTP sweep disables expert deferral and tests draft depths one/two. MTP speed is the emitted-token
SPECULATIVE tok_s statistic; STEP_TRACE phase averages in these runs are per verification batch, not per
emitted token. This repeats the earlier MTP investigation only because Q22 changes the CPU/GPU balance.

## Matched coding screen and KL (first 30 HumanEval chat tasks)

Greedy chat, low reasoning, 1024-token cap, isolated Python tests. This is a 30-task comparison, not full
164-task HumanEval. The two Q22 settings concatenate their first-ten smoke with tasks 10–29; tasks are distinct.

| Configuration | Speed exploration | Passes / 30 | Valid syntax / 30 | Token-cap hits | KL vs original Q23 aff 0 | PPL |
|---|---:|---:|---:|---:|---:|---:|
| Q23 mix2 + defer 2 reference | ~28.2 | 29 | 30 | 0 | 0.286 (previous session) | |
| Q22 fit4 + defer 2 + lossless switches | ~33.6 | 26 | 30 | 0 | 0.423130 | 5.652501 |
| Q22 fit5 + defer 2 + KDA row parts 4 | 35.0717, one run | 23 | 24 | 2 | 0.462702 | 5.828970 |

KL protocol: four 512-token sequences, one-token teacher-forced decode, existing aff-0 Q23 logits reference.
No MTP, no Q3 dense repack in these two Q22 finalists. **The 35.07 setting fails coding coherence and is not
selected**: six malformed outputs, including repetition. Fit4 loses three functional passes versus the
reference, but all thirty outputs have valid syntax and none hit the cap. Larger/long-form coding remains
unqualified. Results and failed task IDs: `quality-summary.json`.

The local experimental config is `configs/experimental/glm-b550-q22-fit4-2133.json`. It uses the original
Q23 model plus the checked down-only sidecar, CANON=1, 12 threads, GPU budget 15360, expert cache request 12288,
4096 context, mmq prefill batch 2048, fit4 profile, two-route deferral and the lossless switch stack.
It changes no default. `confirm-final.sh` repeats the original and fit4 in two alternating rounds.

## Confirmation and selected experimental result

Two alternating rounds with the original Q23 mix2 reference and Q22 fit4:

| Round | Q23 reference tok/s | Q22 fit4 tok/s |
|---|---:|---:|
| 1 | 26.1032 | 33.5708 |
| 2 | 28.3086 | 33.6053 |
| Reference recheck | 28.3065 | |

The first reference is an unexplained timing outlier: same 128 tokens, same 906 slots and cache fingerprint,
CPU 21.2 ms/token, but head 3.7 and GPU wait 11.6 instead of about 2.2 and 10.2. Preserve it, but use the
repeatable 28.31 baseline to assess the typical gain (about 19%). Q22 confirms near 33.59 with CPU 9.1 /
GPU wait 16.6 / head 2.2 / tail 1.6 ms per token. No CPU/GPU compilation or other inference ran during timings.

The four Q22 fit4 failures are valid Python: three omit typing imports required by the prompt; one includes
an incorrect self-test assertion. Scores remain 26/30 under the unchanged evaluation protocol. Do not replace
the missing imports or remove assertions to improve the reported score. There are no syntax-invalid outputs
or token-cap hits in this subset. The rejected 35.07 variant has six syntax-invalid outputs and two cap hits.

No default configuration, DDR clock, source model file, or commit changed. All new engine features are opt-in.
The down-only sidecar is uncalibrated. Full 164-task HumanEval, long-form code and other contexts are not tested.
The goal of **35+ with coding coherence is not achieved** in this round. The experimental fit4 config records
the best validated result; it is not a replacement for an existing user/default config.
