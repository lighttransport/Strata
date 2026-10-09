# B550 step 1 matrix — 2026-10-09

This replaces the terminated lever sweep at the user's request. The old sweep stopped in its quiet gate before run 024; its completed 8-worker run 023 is retained without a trailing control. Step 1 appends new `step1-` rows to `tools/sim/data/measured_b550_levers.json`. No engine changes, parameter refit or new commit are made.

The binary is pinned to SHA256 `8c2c754264f5c327997826cbe468b953888c3cbb9d8c165833074a1d8e1df98f`. REAP-50 Q23, frozen 2048-token input, 12 workers, 4096 context, 2048 prefill chunk, adaptive tier, three decode trials and 512 emitted tokens per trial match the original setup. Stop IDs are absent. STEP_TRACE is enabled consistently. Reported speed averages trials 2–3; trial 1 warms the tier.

All cases request a 12288 MiB cache and a 512 MiB physical GPU reserve. Allocation budgets are tried in order: 15360, 14848, 14336 MiB. A GPU headroom/allocation refusal abandons that budget epoch and restarts the entire nine-run set at the next budget. Earlier attempts remain recorded and are excluded from the selected-budget comparison. The physical GPU is 16304 MiB; the guard limits global usage to 15792 MiB, independently of the engine allocation budget.

The previously approved **512 MiB runtime headroom** is applied consistently to BASE15 and MTP cases. This is separate from the physical GPU reserve; this decoder rejects 256 MiB runtime headroom for MTP. The adjustment preserves matched controls, and actual tier slots/MiB are reported rather than assuming the simulator's tier prediction.

Before each run the one-minute load must stay below 3 for 180 seconds. Read-only AMD/ROCm telemetry verifies MCLK above 100 MHz; no clocks are changed. During decode, `rocm-smi --showmeminfo vram --showclocks --showuse` is sampled every five seconds. Only readings confirmed to remain inside a decode trial before and after the query count toward the summary. Raw ROCm output, sysfs MCLK and the guard's independent one-second global VRAM telemetry are retained. All three decode trials require ROCm coverage. A 96 MHz idle reading is distinguished from the loaded peak.

The run order is BASE15, affinity 0.06 ordinary, BASE15, affinity 0.08 ordinary, BASE15, MTP1 without affinity, BASE15, affinity 0.08 + MTP1, BASE15. Nonbudget failures and their affected comparisons are retried at the end with fresh BASE15 controls on both sides. CPU OOM and swapping are not silently treated as GPU-budget refusals.

Token matching is calculated separately for every trial against both neighbouring controls. First differing indices are zero-based. A failed MTP1/no-affinity identity check is reported as a bug; BASE-control reproducibility is recorded separately, so the result does not imply an isolated root cause. No output-quality pass is claimed from throughput alone.

`records.json` holds this set; `prior-records.json` freezes the previous rows for append preservation. `plan.json` records the protocol and simulator references (12.3 / 15.6 / 16.9 / 17.9 tok/s). `SUMMARY.md` contains actual results and `audit.json` verifies the final evidence. `run_step1.py` uses only existing engine flags and the unchanged memory guard. `report_step1.py` audits records without refitting parameters. Runtime unit: `strata-b550-step1-20261009.service` in the isolated B550 checkout.

Parameter SHA256 stays `6786b52fe952e5ec3b86af363964c35d80570ba1a3f8e62573e1cb5c563ad9b6`. The frozen simulator forecast/conditional datasets from commit `79ab77df` remain unchanged.

## Outcome

All nine timing runs completed at the first allocation budget, **15360 MiB**, with no budget fallback, OOM, swapping or guard failure. The best measured rate is ordinary affinity 0.08 at **18.1856 tok/s**; affinity 0.06 reaches **17.07225 tok/s**. MTP1 alone reaches **11.13825 tok/s**, and affinity 0.08 + MTP1 reaches **17.3076 tok/s**. The no-affinity MTP identity gate failed and is reported as a bug. BASE controls also disagree, so the difference has not been isolated to MTP. No engine change, parameter refit or new commit was made. Both measurement supervisors are inactive after completion. The detailed table, first-difference indices and match counts are in `SUMMARY.md`; all 27 streams contain exactly 512 IDs.
