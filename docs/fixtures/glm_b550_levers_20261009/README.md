# B550 lever measurements — 2026-10-09

Measurement sweep: Ryzen 9 3950X, RX 9070 XT, REAP-50 Q23 assembled GGUF, 60 GiB RAM cgroup, no swap. Engine, model and serving configuration are unchanged. The original protocol prohibited fitting/committing; after run 022 the user explicitly requested a current summary, simulator/config update, commit and measurement resumption. That checkpoint is documented in [CHECKPOINT.md](CHECKPOINT.md). Raw measurement rows remain `fit:false`; separate frozen simulator views identify training and timing checks. Further runs use the same engine flags.

The unchanged-flags BASE gate uses the user-authorized current B550 binary SHA256 `8c2c754264f5c327997826cbe468b953888c3cbb9d8c165833074a1d8e1df98f`; the historical manifest's executable has been replaced. Trials 2–3 measure **10.71575 decode / 231.129 prefill tok/s**, versus reference means 10.46455 / 230.93, so both pass the 10% gate. All three streams contain exactly 512 IDs and match each other.

After that gate, STEP_TRACE is enabled for every interleaved run, as approved by the user. The cohorts have matched BASE controls:

- MTP: original BASE settings with runtime reserve 512 MiB. The binary rejects the original 256 MiB override with MTP. MTP1/2/3, MTP2 unsplit, MTP2 affinity .05/.10, and MTP3 continuation margin 2 are compared against ordinary decode controls at that same reserve.
- Ordinary decode: original 256 MiB runtime reserve, testing static tier, 3072 MiB cache, the allocator's largest admitted cache, and 8/16 workers. The initial plan mistakenly requests 14336 MiB, above this CLI's 12288 MiB cap. Its rejection and retry are retained. After the main sweep, `run_max_cache.py` adds fresh BASE / 12288 MiB request / BASE runs; allocator admission enforces live-budget and physical-reserve limits, and the actual slots/MiB are reported.
- Prefill: approved 2048 MiB scratch arena, shared 8192 context, and zero decoder steps. The unchanged CLI requires `steps >= 1` and `prompt + steps <= context`, so these runs emit only the first token already computed by prefill, once per trial. The shared 8K context permits a 4096-token input. The measurement adapter reuses the existing cgroup/GPU guard and retains its original result as `*.guard-result.json`; it reclassifies completion for three PREFILL lines and no DECODE lines. BASE controls use 2048 input / 2048 chunk with the same arena/context.

The static case unsets `STRATA_GLM_TIER_ADAPT`: this binary checks presence, so literal `0` would leave adaptation enabled. Zero prefetch groups are represented by absent prefetch flags; this binary rejects literal `PREFETCH_GROUPS=0`. Nonzero staged-prefetch requests use the requested `STAGE_PREFETCH=1` and group count. Engine errors are retained; unsupported paths are not implemented or patched.

Decode runs emit exactly 512 tokens per trial without stop IDs. Every run has three timed trials; the reported rate is the arithmetic mean of trials 2–3. Raw trial 1 is retained but excluded from the performance mean. Each lever is bracketed by matched BASE controls. Failed levers receive one retry at the end with BASE on both sides and unchanged lever flags. Successful records remain `fit:false`; failed attempts with null rates are kept separately.

Before every run, the 1-minute load average must remain below 3 for at least 180 seconds. Samples, starting/ending load, ROCm SMI output, AMD SMI clock output and sysfs clock readings are recorded. AMD SMI supplies read-only telemetry when runtime-suspended RDNA4 returns empty ROCm clock fields. No clocks are reset or forced. The GPU may return to 96 MHz after exit; loaded clock peaks from the guard telemetry distinguish idle from being stuck. End load includes the benchmark's own CPU workers.

Prefill inputs are prefixes of the frozen 2048-token BASE input. The 4096-token input repeats that frozen input twice. Their exact IDs and hashes are retained; this is a performance fixture, not a chat-quality evaluation.

`measurement-plan.json` captures the protocol decisions and the unchanged simulator-parameter checksum. `SUMMARY.md` contains the comparison table. Each run has its config, manifest, raw log, result, IDs and standalone simulator record. The combined data is `tools/sim/data/measured_b550_levers.json`.

Reproduce on an otherwise idle B550, in the isolated checkout, using a new fixture directory to avoid reusing results:

```sh
PY=/home/syoyo/work/Strata/.venv-glm-hip/bin/python
$PY docs/fixtures/glm_b550_levers_20261009/run_measurements.py \
  --output docs/fixtures/glm_b550_levers_20261009
# Only after the gate passes:
$PY docs/fixtures/glm_b550_levers_20261009/run_sweep.py
# After the main sweep finishes, measure the supported maximum cache request:
$PY docs/fixtures/glm_b550_levers_20261009/run_max_cache.py
```

The scripts verify the pinned binary hash. A completed prior record is reused when resuming the same directory; a fresh measurement requires a new directory with the same scripts/base config. The sweep writes only new measurement files, and never invokes a build, parameter fit or Git commit.

Early ID check: MTP1 and its trailing BASE control agree in all three trials, while the leading BASE control differs beginning at the second emitted token. Each run repeats identically internally. This is BASE-control output instability, so the bracket ratio is not a controlled identical-token speedup and cannot isolate an MTP correctness failure. Full streams and first-mismatch diagnostics are retained; no engine fix is attempted.
