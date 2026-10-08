# B550 long-context C++ review smoke test

The frozen prompt contains an unrelated real `pool.hpp` as background followed
by a synthetic C++17 review target. The target has five known correctness bugs
and one correct lifetime example. Only `review.cpp` is in scope. The rubric is
stored separately and is not sent to the model. This is a single synthetic case;
it does not establish general coding quality or retrieval across all positions.

The server tokenized the prompt to 6511 tokens. The test uses context capacity
12288, 2300 maximum generated tokens (including reasoning), temperature zero,
low reasoning effort, and the same assembled REAP50 model as the current B550
profile. Only context capacity and the log destination change. Prefill batch
remains 2048; no speculation or prefix reuse. The 60 GiB cgroup, zero swap, and
2048 MiB minimum free global VRAM guards remain enabled.

## Reproduction

Run on B550 from the repository root with no other GPU inference process:

```sh
tools/glm_b550_60g.sh stop
systemd-run --user --unit=strata-glm-quality \
  --property=WorkingDirectory="$PWD" \
  --property=MemoryMax=60G --property=MemorySwapMax=0 \
  --property=OOMPolicy=kill --property=KillMode=control-group \
  "$PWD/.venv-glm-hip/bin/python" tools/glm_guarded_serve.py \
  configs/glm53f-reap50-q23-b550-60g-9070xt-quality-12k.json \
  --ram-gib 60 --reserve-mib 2048 --port 8080 \
  --telemetry build-hip-glm-v11/quality-12k-memory.json
# Wait for journalctl --user -u strata-glm-quality to report ready.
curl --fail http://127.0.0.1:8080/health
.venv-glm-hip/bin/python tools/glm_cpp_review_check.py \
  --output build-hip-glm-v11/quality-12k-f16
systemctl --user stop strata-glm-quality
tools/glm_b550_60g.sh start
```

Save the guard telemetry before replacing it with another run. Inspect the
complete answer and `finish_reason`; reaching a token limit is not a completed
review. Evaluate mechanisms, triggers and corrections against `rubric.json`,
not keyword matches. The helper writes request JSON, raw response/timings and
a readable final answer. It does not score itself or call another model.

The `perf-2048.ids` and `tr16-perf-8192.ids` files are historical throughput
inputs, separate from this quality prompt. The TR16 build manifest records the
historical binary's local link inputs; no TR16 workload was launched here.

## Observed result — 2026-10-08

The request completed with `finish_reason: stop`: 6511 input tokens and 1965
output tokens including reasoning, in 224.08 seconds. The final review contains
748 whitespace-separated words. Engine timings were **172.1 tok/s prefill**
(37.839 s) and **10.6 tok/s decode** (186.188 s). This was the first request
after service expert-page warmup, not a repeated warm-prefill benchmark.
The earlier 2048-token warm throughput result remains 230.06 tok/s; the longer
request here did not meet 200 tok/s prefill. Do not substitute one rate for the
other or claim a general 200+ prefill guarantee.

Peak cgroup RAM, including startup, was 57.18 GiB; summed RSS peaked at 53.10
GiB. Minimum free global VRAM was 2816 MiB. Swap and memory-limit/OOM counters
were zero, with no guard rejection. The recorded engine exit `-15` reflects
our deliberate service stop after the successful answer, not a generation
failure. The standard 4096-context service was then restarted.

Manual assessment: **5/5 seeded bugs identified**, all five core corrections
appropriate, **zero spurious defect findings**, and the `Snapshot::read`
lifetime control correctly accepted. However, the answer contains these errors:

- `UINT64_MAX - 1 + 2` wraps to **0**, not 1. Its example still demonstrates
  the overflow defect, but the arithmetic explanation is wrong.
- A variable declared in the `for` initializer lives for the whole loop, not
  one iteration. Its lifetime ends before the returned callbacks execute;
  capturing it by value is the correct fix.
- The proposed `display_name` test uses the first view after the next call,
  although the contract guarantees it only until that call. That expectation
  is invalid for the suggested reusable storage.
- TSan is not a sufficient primary check for a single-thread dangling stack
  reference; use ASan with stack-use-after-scope detection and behavioral tests.

Verdict: **qualified smoke result, not an error-free review or broad release
quality pass**. No complete model-generated patch was compiled. This test
compares against a known rubric, not an MMQ output or the unpruned model.

Evidence: [raw API response](quality-12k-f16.json),
[final review](quality-12k-f16.md), [assessment](assessment.json),
[request](quality-12k-f16.request.json), and
[guard telemetry](quality-12k-memory.json). The tested executable hash is
`685b1344f3e46b9d29e5fe599f96b01db6372218a5cab235bd7b2331399a150f`.
