# B550 measurement and simulator checkpoint — 2026-10-09

The user requested a current summary, simulator/config update, commit, then resumption. The sweep paused after clean run `022-base-ordinary256`: **21 qualified runs, two failed attempts**. This is a partial checkpoint; all worker and pure-prefill cases, failed-case retries, and the supported-maximum cache measurement remain pending.

Ryzen 9 3950X, RX 9070 XT, REAP-50 Q23, 60 GiB cgroup, zero swap; decoder SHA256 `8c2c754264f5c327997826cbe468b953888c3cbb9d8c165833074a1d8e1df98f`. The engine and serving configuration are unchanged. Every successful decode trial emits exactly 512 IDs; the reported mean excludes trial 1. BASE reproduced at **10.71575 decode / 231.129 prefill tok/s**, within the initial 10% gate. Subsequent BASE prefill rates remain near 230 tok/s; no other prompt size is qualified yet.

BASE neighbours differ across processes despite canonical mode. Each lossless lever reports its token mismatch; ratios cannot be treated as identical-token speedups. Affinity is lossy, and its two clean timing runs await replacement brackets because control `011` was disqualified by system-update CPU activity. All IDs and guard diagnostics are retained. No quality pass is claimed.

The initial largest-cache request was a measurement-plan mistake: 14336 MiB exceeds the unchanged CLI's 12288 MiB cap. Its error is retained. The sweep retries failures at the end; `run_max_cache.py` then adds fresh BASE / supported-maximum request / BASE runs so the actual largest admitted slots/MiB are measured.

## Completed measurements

| Lever | Kind | Measured tok/s | Neighbour BASE tok/s | Ratio | IDs equal to BASE | Actual slots / MiB | Start / end load | Start / end / loaded peak MCLK MHz | Status |
|---|---|---|---|---|---|---|---|---|---|
| BASE | decode | 10.7157 | — | — | — | 707 / 6142.06 | 0.00 / 12.16 | 772.0 / 96.0 / 1258 | complete |
| mtp1 | decode | 10.5800 | 10.7681 | 0.983 | no | 707 / 6142.06 | 0.21 / 12.32 | 772.0 / 96.0 / 1258 | complete |
| mtp2 | decode | 10.3447 | 10.8072 | 0.957 | no | 707 / 6142.06 | 0.13 / 12.45 | 772.0 / 96.0 / 1258 | complete |
| mtp3 | decode | 8.3249 | 10.8227 | 0.769 | no | 707 / 6142.06 | 0.14 / 13.71 | 772.0 / 96.0 / 1258 | complete |
| mtp2-unsplit | decode | 8.2960 | 10.7302 | 0.773 | no | 707 / 6142.06 | 0.13 / 13.71 | 772.0 / 96.0 / 1258 | complete |
| mtp2-aff05 | decode | 12.0534 | — | — | — | 707 / 6142.06 | 0.14 / 11.64 | 772.0 / 96.0 / 1258 | complete |
| mtp2-aff10 | decode | 15.6094 | — | — | — | 707 / 6142.06 | 0.12 / 12.72 | 772.0 / 96.0 / 1258 | complete |
| mtp3-margin2 | decode | 8.8679 | 10.5763 | 0.838 | no | 707 / 6142.06 | 0.13 / 13.54 | 772.0 / 96.0 / 1258 | complete |
| static-tier | decode | 10.1458 | 10.5180 | 0.965 | no | 707 / 6142.06 | 0.29 / 12.27 | 772.0 / 96.0 / 1258 | complete |
| cache3072 | decode | 9.7130 | 10.7329 | 0.905 | no | 353 / 3066.69 | 0.14 / 13.42 | 772.0 / 96.0 / 1258 | complete |
| largest-admitted-cache | decode | — | 10.5612 | — | — | — | 0.14 / 0.19 | 772.0 / 96.0 / — | failed |

Rates are means of trials 2–3. Each ratio uses the mean of the two neighbouring matched BASE runs. Token equality uses all three 512-token streams and both BASE neighbours. Prefill has no decoder steps; its one emitted token per trial was already computed by prefill. End load includes the benchmark itself. A 96 MHz clock after exit is an idle reading; loaded peaks are reported separately.

Successful runs: 21; failed attempts: 2. Failed attempts are retained separately from importable records, with null tok/s. See each manifest, result, log, token file and record for exact flags and diagnostics.

## Simulator update

Raw `measured_b550_levers.json` rows remain `fit:false`. Separate frozen forecast and conditional-acceptance views contain 42 timing rows. Only unchanged-flags BASE decode/prefill timing trains aggregate launch/GEMM terms; CPU counter bandwidth is 28.061 GB/s, with the prior independent width and staging anchors retained. B550 CLI prefetch defaults to zero, and implemented-only REAP plans exclude the unsupported staged-prefetch path. The static lever is correctly mapped to `static_prior`, because it keeps BASE's expert prior. Prefill explicitly uses zero prefetch groups; the previous imported calibration inherited eight groups from the simulator default.

The B550 CLI now uses a 512-token frozen-fixture acceptance prior, pooling MTP1/2 counters; its third-position estimate remains historical. Unsplit has a separate prior. The old `b550_screen` profile is still selectable, and other hardware defaults are unchanged.

Forecast error averages **6.26% over the nine decode levers**, **3.46% over all decode rows**, and **2.01% over all 42 decode/prefill rows**, with continuation-margin 2 **12.84% high** and outside tolerance. Conditional acceptance diagnostics average **1.56%**, maximum **9.10%**, with all rows within tolerance. The detailed results are in `simulator-update-validation.json`. These are same-input timing checks; acceptance counters were calibration inputs, and the rows are not independent workload or output-quality qualifications. Small exposed-launch and effective-GEMM terms are aggregate model parameters, not independent kernel measurements.

Reproduce this frozen calibration before resuming/appending raw measurements:

```sh
python3 tools/glm_b550_fit_levers.py
python3 tools/sim/glm_sim.py validate --records tools/sim/data/measured_b550_levers_forecast.json \
  --params tools/sim/data/params_b550.json  # reports the retained margin-2 miss
python3 tools/sim/glm_sim.py validate --records tools/sim/data/measured_b550_levers_conditional.json \
  --params tools/sim/data/params_b550.json
python3 tools/test_sim.py
python3 tools/test_sim_b550.py
```

The pre-update parameter file and original checksum remain archived. `measurement-plan.json` records the user's changed authorization and the new expected checksum. Subsequent measurements do not automatically refit parameters. Raw logs, manifests, results and token files are committed; redundant progress snapshots and final-logit dumps remain runtime artifacts.
