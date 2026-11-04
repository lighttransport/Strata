# Resume DeepSeek V4.1 Flash Coder work after reboot

Updated 2026-11-04, about 10:00 UTC. This file replaces the old handoff entirely.

Resume the user's paused optimization goal only when asked to resume. The goal is:
**30+ decode tok/s with exact greedy output, 60+ tok/s in an explicitly lossy mode; speculative decoding is allowed. Engram lookup must not become the bottleneck.** The original work window was six hours; about three hours of active work remain. Check the goal tool for authoritative status/time. Neither target has been achieved. Do not mark the goal complete or describe a screen as qualification.

## Scope and rules

- Work in this Strata checkout. Read AGENTS.md and docs/AI_REFERENCE.md. Preserve unrelated dirty GLM/user files. No commits or push are authorized. Do not spawn subagents.
- Target and drafter inference must run on the Threadripper 1950X GPU machine (16 cores/32 threads, 128 GB installed RAM, RX 9070 XT 16 GB, gfx1201). The B550 machine is for Engram/file serving only, over IB ucomm. No remote inference.
- The user selected the Coder IQ2_XXS mixed pack. Do not substitute a different model/quantization.
- Keep actual model paths, SSH settings and endpoints in ignored local configuration, not source/docs/reports. The authoritative connection settings are in `priv/dsv41-coder-remote/config.json`.
- New features remain opt-in. Preserve the existing default path. Engine environment reads use glm_env helpers without changing their truthiness semantics.
- Exact means identical greedy token IDs to the frozen full six-route reference, not numerical tolerance. Proposed qualification: single sequence, 32K total context, roughly 4K and near-32K active prefixes, ten coding prompts/context, 512 output tokens, five repetitions. This has NOT been completed.
- Lossy quality floor is not selected. Measure the speed/correctness frontier before choosing a setting; do not enable a lossy setting silently.
- Proposed Engram gate: exposed wait p95 <= 0.5 ms per committed token and <= 3% throughput cost against exact-row replay. Replay comparison is not implemented/qualified. Keep >= 1 GiB physical VRAM and >= 12 GiB host memory available.

## Reboot checkpoint

The user paused work for reboot. The owned calibration process group was terminated with SIGTERM after validating its process identity and ancestry. It was running the last task's top2 variant. The local API was already stopped for benchmarking, and the benchmark controller was stopped too, so its finally block did not restart the API.

The remote file-service parent was left running with **24 reader threads**, selected in the ignored config. It may survive a reboot of only the GPU machine. Its old PID is NOT a safe identity after reboot; use the launcher's identity cookie/status. Do not kill processes based on PIDs in old logs. There should be no local inference process left from this checkpoint. Confirm current state rather than assuming it.

`priv/dsv41-coder-remote/coding-calibration/INTERRUPTED_FOR_REBOOT.txt` marks the interruption. Preserve raw partial outputs. Existing benchmark directories reject overwrite; use a new directory or run only missing variants into new directories.

## Start here after reboot

1. Check goal status and current hardware/process state. Review tracked changes with `git status --short --untracked-files=no`; the checkout has hundreds of unrelated untracked experiment files, so avoid dumping all of them.
2. Read `priv/dsv41-coder-remote/OPTIMIZATION_STATE.md` for earlier diagnostics, then the latest checkpoint below. Use the virtualenv interpreter `priv/dsv41-coder-remote/venv/bin/python`; system Python lacks required dependencies. Do not use bare `python`.
3. GPU/network commands require the available escalation mechanism. The HIP runtime is under `/opt/rocm/core-10.1`; existing ignored run config contains LD_LIBRARY_PATH. Do not print connection secrets/settings unnecessarily.
4. Inspect `tools/dsv41/remote_engram.py status --config priv/dsv41-coder-remote/config.json` with the virtualenv Python. The remote parent validates/hashes the whole table on restart, which takes several minutes. A surviving parent avoids the rehash. Do not use generic launcher `stop` just to stop the local API: it stops the remote parent too.
5. HIP and CUDA builds now include resident routing bias (built during the local commit audit). Its model runtime/quality checks have NOT been run. Rebuild if source changes before exercising it. Do not compile while timing inference. Existing configured build trees are `build-dsv41-remote` and `build-dsv41-remote-cuda`; target `strata-dsv41`. CUDA is compile-only SM86 here. There is no DSV41 SYCL target; shared CPU/SYCL execution is unvalidated.
6. Grade the completed coding calibration outputs with the existing private script before making quality claims. Then decide the next controlled measurements. If restoring the API first, run the generic launcher `start`; it serves only localhost:8080, with the original six-route/no-drafter settings. Check logs for readiness, not just the PID.

Useful commands (from repo root, substituting the virtualenv interpreter for PY):

```sh
PY=priv/dsv41-coder-remote/venv/bin/python
$PY tools/dsv41/remote_engram.py status --config priv/dsv41-coder-remote/config.json
cmake --build build-dsv41-remote --target strata-dsv41 -j 4
cmake --build build-dsv41-remote-cuda --target strata-dsv41 -j 4
$PY -m unittest discover -s tests/terminal -p test_dsv41_remote_engram.py
$PY -m unittest discover -s tools/dsv41 -p test_bench_decode.py
```

The grading script executes generated C++ only inside a bubblewrap namespace: no network, no home/repository mount, only system tools/libraries and the individual generated source. Bubblewrap namespace creation fails inside the default tool sandbox but passed outside it. Request escalation for the **isolated grading command**, keeping its bwrap isolation; do not remove isolation to run generated code with unrestricted access.

```sh
priv/dsv41-coder-remote/venv/bin/python priv/dsv41-coder-remote/grade_calibration.py
```

That script is prepared but has NOT been run. It compiles the repository's independent training-task oracles, records diagnostics, and writes `coding-calibration/quality.json`. Partial/missing variants must remain unqualified.

## Working deployment already implemented

Production `strata-dsv41` supports the six-shard Coder layout with five local shards and the combined IQ4_NL Engram table served remotely through native IB ucomm. It validates source size/SHA-256, preserves exact row bytes/order/duplicates, fails closed on incomplete reads, and sends idle heartbeats. The remote machine performs only file reads. A parent holds the verified table open and forks a fresh session child for each client.

- Launcher: `tools/dsv41/remote_engram.py`
- Private config: `priv/dsv41-coder-remote/config.json`
- Generated engine config: `priv/dsv41-coder-remote/state/run.json`
- Local API cookie/logs: `priv/dsv41-coder-remote/state/`
- Pack: `priv/dsv41-coder-remote/pack`
- Catalog: `data/dsv41/coder-set.json`
- Docs: `docs/DSV41_REMOTE_ENGRAM.md`
- Latest pre-routing-bias builds succeeded in `/tmp/ds4-lookup-fullscratch-hip-build.log` and `/tmp/ds4-lookup-fullscratch-cuda-build.log` (tmp logs may disappear at reboot).

Original API settings remain CPU15, reserve3072 MiB, six experts, DSpark none, vision off, mmap on, prefill8192, total context32768, parallel1, read-only expert profile, 256 MiB Engram row cache. No experimental decode knob or resident routing bias was enabled in this API config.

Earlier validation passed: 2053 ordered remote rows byte-identical to direct remote pread; idle/cancellation/failure/recovery checks; local pack default output unchanged. See private notebook/evidence for precise test scopes. Most recently, nine launcher tests and four decode-harness tests passed. Newly added routing bias has no build/runtime validation yet.

## Implemented decode experiments

- `STRATA_IQ1M_DIRECT=1`: single-token IQ1_M vectors consumed immediately, avoiding the four-step decoded stack frame. Default unchanged. Row/expert pipeline bit checks passed; whole-model gain modest.
- `DS4_IQ_PF_COLD`: existing nonzero flag now skips warm prefetch hints using Linux residency checks; its default semantics remain unchanged. `STRATA_IQ_PREFETCH=0` was separately screened.
- `--dspark-vram-mib`: optional balanced drafter expert budget per GPU group; default -1 preserves prior drafter-first placement.
- `--target-vram-counts`: fixed target residency per layer with shape/range/budget checks. Frozen target mask **f60dd546f883ae02**, counts **55 followed by 39 copies of 17**, 718/7680 resident. Physical free VRAM changes can make this fail closed; reserve2816 was needed in later screens.
- `DS4_PLACE_HASH`, `DS4_CPU_HASH`: diagnostic placement and CPU-call hashes.
- `DS4_TOKLOG`: exposed Engram wait p50/p95/max per committed token (round-amortized under speculation).
- `--lookup-draft 0..15`, `--lookup-min 3..32`: weight-free greedy suffix drafting using the existing GLM SuffixDrafter; every proposal target-verified. One sequence, default off; sampled requests bypass it. Adapter test and original suffix test passed.
- `--lookup-check`: serial/lookup/serial from one frozen prefix. Requires checkpoint(1) BEFORE pin(); pin alone does not snapshot. Fixed-length ignores EOS. Generic verifier scratch is now allocated lazily even without DSpark; earlier null-scratch fault was fixed. The last HIP/CUDA builds include that fix.
- `tools/dsv41/bench_decode.py`: sequential CLI screens with raw outputs, hashes, explicit overrides, timing/wait/placement counters, immutable `--reference-ids` option, and error/token-count validation. Four unit tests passed.
- Remote reader tuning: optional strict integer `remote_readers` 1..64 in launcher config; default preserves ambient remote environment or service default8. Saved process cookie/status report the count, or unknown for old cookies. Explicit mismatched count requires stop/restart. Nine tests passed, including owned-process lifecycle and ambient override handling. Docs updated.

### Last experiment: resident routing bias — built, runtime untested

Just added `Options::resident_route_bias` (default0), CLI `--resident-route-bias X`, help, and docs. Files: `src/dsv41/ds4_model.hpp`, `ds4_main.cpp`, `ds4_model.cu`, `docs/DSV41_REMOTE_ENGRAM.md`.

`apply_resident_route_bias()` runs after expert loading. Zero returns before device operations or allocations. Positive bias requires one GPU, `--no-expert-cache`, `--dspark none`, `--vision off`, and at least k resident experts per target layer. It reads each target layer's gate selection bias, subtracts X for nonresident experts, and uploads it. Selection changes in BOTH prefill and decode; route weights still use the selected experts' original gate scores before normalization. It emits an explicit LOSSY log. It does not modify files on disk. The CLI rejects malformed/nonfinite/negative numbers.

Purpose: measure speed/quality when favoring GPU-resident experts, including whether 60 tok/s has any usable-quality operating point. It may severely damage output. Do not present GPU-only gibberish as success. Review implementation, build both backends, test default0 unchanged and validation paths, then evaluate on coding tasks. It has not been enabled in inference. Current HIP/CUDA binaries recognize the flag; model runtime and quality validation remain outstanding.

## Measured evidence — screens, not final qualification

All speeds are local single-sequence decode on the above hardware. Context, output length and placement matter; compare same-day controlled runs rather than mixing rows as a single progression.

- Original full six-route baseline, about4028 prefix/256 output: **7.18 tok/s**, CPU about115 ms/token, warm Engram about0.15 ms.
- CPU-thread and exact-kernel screens mostly **6.8–8.2 tok/s**. One direct+no-prefetch fixed-placement screen **7.92**, same128 IDs, Engram p95 **0.174 ms**.
- Early reduced-routing screen: top4 **9.48**, top2 **14.62**, top1 **22.24**; outputs changed. No quality score from that screen.
- Frozen-prefix serial/lookup/serial check: **7.144 / 7.110 / 7.422 tok/s**, all128 IDs identical. Lookup proposed12/accepted3 over3 rounds. Evidence `lookup-check-k4-m3-n128-r3/result.json`.
- DSpark drafter staged locally with SHA verification. Bounded128 MiB drafter plus fixed target mask did not help: loaded-off **6.75**, draft1 **5.99**, draft2 **6.51**, draft4 **4.34**; first-position acceptance57%, about2.18 committed/round. Default drafter placement displaced target weights and slowed prefill badly. Keep disabled.
- Stage14b: copy/interleave **7.46**; zero-copy PCIe streaming **8.72** with changed tokens. CPU/GPU expert lane output checks were not bit-identical (per32 q8_1 GPU versus per256 Q8_K CPU by default), so this cannot be called exact. 59.9 GB raw copies pinned successfully; >50 GB host headroom observed.
- Stage15b (fixed mask, direct1/prefetch0, reserve2816): physical16 **6.84**, SMT24 **6.94**, SMT30 **4.87**, SMT32 **4.31**. More SMT threads did not help. Engram p95 respectively0.719/1.263/2.975/3.248 ms.
- Stage16: remote readers24, same stage15 physical16 settings: **7.89 tok/s**,128 output, CPU103.76 ms/token, Engram mean1.04 ms and p95 **1.484 ms**. CPU compilation briefly overlapped part of this screen; not qualified. Parent was freshly restarted/hashed, so cache state differs. More readers are not established better. Default source remains8; ignored config currently explicitly24.
- Private IQ1_M 80-byte prepacked prototypes: 65536 rows bit-identical versus ggml, 1.429x storage. Bit-plane layout/SIMD-sign variant showed only small variable gain (roughly9–12 ms vs10–12 ms per CPU test). Not integrated. Evidence `bitplane-probe-results.log`, `calibration-driver.log`. Do not consume extra12+ GB host RAM for this without convincing whole-model benefit.

### Latest 4K coding calibration, interrupted for reboot

Uses the repository's three **training** specifications/oracles (`tools/glm_coding_quality.py`), not held-out tasks. Exactly4096 prompt tokens, greedy, 512-token maximum, direct1/prefetch0, CPU16, six/four/two/one experts, no adaptive expert movement. EOS can stop before512: `completed_requested_tokens=false` here usually means EOS, not inference failure. These are variable-length answer timings, not a fixed512-token qualification. No correctness grading has run yet.

Artifacts: `priv/dsv41-coder-remote/coding-calibration/`, prepared/run/grade scripts in the same private parent directory.

| Task | Six experts | Four | Two | One |
|---|---:|---:|---:|---:|
| parse_u64 | 8.92 (171 tokens) | 12.26 (171) | 19.15 (163) | 25.51 (512 cap) |
| checked_slice | 8.91 (130) | 13.35 (130) | 18.67 (153) | 25.49 (512 cap) |
| reduce_routed | 8.41 (191) | 11.55 (191) | interrupted | not run |

Parser six-route output was manually decoded and was a complete plausible C++ answer, but only execution of its independent oracle establishes correctness. Top1 hitting the cap may indicate degraded/repetitive output; inspect/grade it. Engram p95 in these runs ranged0.452–1.818 ms; the latency gate is not uniformly met. Do not claim quality scores, 30 tok/s exact, or 60 tok/s lossy from these results.

## Important unresolved exactness issue

Fresh repeated prefill can yield different eventual greedy sequences even with the same fixed target expert mask. CPU-call hashes first diverged at layer13 INPUT while earlier outputs matched; GPU prefix captures also showed differences. Serial decode from one frozen prefix repeated identically. Root cause is NOT established. Do not assert a proven pre-existing race.

`ROCBLAS_DEFAULT_ATOMICS_MODE=0` was tested and both handles queried as mode0; it did not fix the discrepancy. `HSA_ENABLE_SDMA=0` diagnostic is prepared as stage12 but has NOT been run. Preserve `sub-*.bin` private raw captures; never add them to git. Changed CPU/GPU placement causing rounding differences is a separate, confirmed issue.

## Next work after explicit resume

1. Complete model runtime/default-parity and quality checks for resident routing bias before using it. HIP and CUDA compilation passed during the commit audit. Keep API defaults unchanged.
2. Grade completed calibration results in bwrap, preserve interruption, and finish missing calibration variants in new output directories. Do not rerun the old all-task controller unchanged: it rejects existing output directories. Do not tune on the repository's held-outs.
3. Screen small resident bias values with quality checks and honest changed-output labels. Record resident mask, CPU expert count, memory headroom, Engram tails, token count/EOS/cap, and native binary SHA. If quality collapses, report that instead of calling the speed target met.
4. Continue exact decode using frozen-prefix comparisons. Investigate prefill nondeterminism separately; the current exact proof is only one128-token frozen-prefix case. More SMT, current DSpark settings and existing PCIe auto streaming did not solve the bottleneck.
5. Engram reader24 is experimental. Compare controlled8/24 with equivalent cache state before retaining it; a service restart rehashes the entire source and changes page-cache state. Uniform sub0.5 ms p95 and the replay throughput gate remain unmet/unverified.
6. Run representative short and near32K coding qualification only after finding promising settings. Keep final qualification separate from training/calibration and microbenchmarks.

Never kill unrelated CPU/GPU jobs, rewrite unrelated files, commit model weights/raw captures, or claim unproduced measurements. An engine fault previously hung its HSA process; identify owned argv/ancestry before terminating it, and keep RLIMIT_CORE soft0 for private benchmark controllers.

## Local commit/audit validation update

The subsequent commit/pre-push audit rebuilt HIP and CUDA successfully, ran nine launcher tests, four decode-harness tests, thirteen pack tests, the lookup adapter test, and 67 server tests (one skipped). The native loopback transport test passed. A fresh IQ1_M exact-row check had zero bit differences across7136 rows. These are validation checks, not resumed optimization or speed qualification. Metal build wiring now includes the shared environment-helper source; Metal and SYCL builds cannot be validated in the available Linux toolchain. Routing-bias model execution remains untested. The user has not authorized a push.
