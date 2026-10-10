# B550 longer-context prefill experiments, 2026-10-10

The optional [8K MMQ preset](../../../configs/experimental/glm-b550-q22-fit4-prefill8k-2133.json)
uses batches up to 8192, prepared serial KDA columns 32 and chunks 256 for
batches of at least 4096 tokens. Short batches inherit the existing geometry.
The original fit4 preset and engine defaults are unchanged. FP16 is not selected.

Hardware: Ryzen 9 3950X, RX 9070 XT (gfx1201, 16304 MiB usable VRAM),
ROCm 7.1.4, four DIMMs at **2133 MT/s**. No DDR or GPU clock changes.
Original REAP50 Q23 assembly plus the experimental Q22 down-only sidecar;
original gate/up and model files were not edited. Routing and decode settings
come from `configs/experimental/glm-b550-q22-fit4-2133.json`.

## Qualified 8K comparison

Frozen 7936-token code prefix, 128 generated tokens, greedy decode without
speculation, fresh 60 GiB cgroup, cgroup swap disabled, GPU allocation budget
15360 MiB and at least 512 MiB physical VRAM reserve. These are first-prefill
measurements; loading and host registration are outside the timer.

| Record | Batch | Prefill tok/s | Decode tok/s | Peak RAM GiB | Minimum VRAM free MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| `final-control-8192-01` | 2048 | 87.4782 | 33.3769 | 48.716 | 1160 |
| `final-mmq-8192-01` | 8192 | 118.546 | 29.6735 | 48.370 | 1254 |
| `final-mmq-8192-02` | 8192 | 107.489 | 33.3234 | 48.570 | 1254 |

The repeat was **22.9% faster in prefill**, with decode 33.32 versus 33.38 tok/s.
The first candidate run was 35.5% faster in prefill but 11.1% slower in decode.
Both runs were clean under the guard, so that timing variation remains reported.
Both reproduced all 128 control tokens and the decode-cache fingerprint exactly.
The coherent 35+ decode tok/s target remains unmet.

Increasing the batch reduces repeated expert-weight transfers: at this prefix,
one full expert pass replaces four. The optional preset omits benchmark-only
step tracing and logit dumps. `final-comparison.json` records the paired ratios
and token/cache parity; `summary.json` links the raw measurements.

## Quality gates and longer-context limits

- The final MMQ short coding smoke test passed 27/30, with the same two malformed
  outputs and one token cap as the control. **All 30 token sequences matched.**
- With frozen ~14K unrelated C++ background, HumanEval tasks 10/11/17/25 passed
  3/4, and **all four token sequences matched the control**, including its known
  task-10 failure. Three outputs hit the 1024-token cap. This is not full
  164-task pass@1 or a broad long-context quality qualification.
- Strict four-fact JSON recall passed at 8K, 12K and 24K with normal stop tokens
  and clean guards. The 12K and 24K prompts exercise short final batches.
- At 16K, all four expected values appeared inside a JSON fence, but a prose
  prefix failed the frozen strict-format scorer. The matched original-MMQ
  control produced **the same 256 tokens**, including this failure. The failure
  remains recorded; the scorer was not loosened.
- The 32K MMQ recall service timed out at 15 minutes 30 seconds, without a
  completed result or scored answer. A 16K performance retry and a later
  original-MMQ 16K control showed sustained reclaim or the same changing
  file-page accounting and low GPU activity. Those runs were stopped and their
  progress records retained. An unrelated `hf` process used about 12 GiB RAM
  during diagnosis. Shared-system pressure is a possible cause, not a confirmed
  root cause. No 16K or 32K preset is selected from these runs.

A completed 16K candidate code timing measured 115.237 prefill / 33.1407 decode
but was excluded for `kswapd0` interference. Separate clean recall timings at
16K and 24K measured 114.864 and 112.997 prefill tok/s on their own frozen
prompts; they are not paired throughput comparisons against the code prefixes.

FP16 was considered under the user's conditional quality approval. The fastest
row-part-4 candidate measured 453.489 / 595.696 / 616.434 prefill tok/s at
8K / 16K / 32K. Despite short-prompt gating preserving the 30 short outputs,
its long coding result regressed to 2/4 (new failure on task 25). Serial-KDA
FP16 also scored 2/4 (new failure on task 17). **Both are rejected for selection.**
Single-position logit diagnostics are retained, but do not override these gates.

## Reproduction

Public metadata uses [portable paths](../../PORTABLE_BENCHMARK_PATHS.md).
Run from the Strata checkout root and supply the model and external inputs there.

`prompt-manifest.json` freezes source and token hashes for 7936, 16128 and
32512-token code prefixes. `recall-manifest.json` freezes the four facts and
five chat prompts. `background-manifest.json` records the coding backgrounds.
Token files contain no trailing newline. Do not regenerate them from edited code.

```bash
python3 tools/glm_low_memory_bench.py \
  docs/fixtures/glm_b550_prefill_20261010/q22-c8192-b8192-mmq-gated.config.json \
  docs/fixtures/glm_b550_prefill_20261010/code-7936.ids \
  --output docs/fixtures/glm_b550_prefill_20261010/UNIQUE-PREFIX \
  --ram-gib 60 --tokens 128 --trials 1 --timeout 300 \
  --gpu-capacity-mib 16304 --gpu-used-limit-mib 15792 --single
```

Use `q22-c8192-b2048.config.json` for the matched control. The guard clears model
pages before a run and excludes overflow or sustained external CPU/reclaim work.
Do not overlap qualified timing with another GPU job or compilation. Two trials
include an untimed prefill; one trial times the first prefill. Global system swap
is separate from the inference cgroup's zero-swap limit.

```bash
systemd-run --user --scope -q -p MemoryMax=60G -p MemorySwapMax=0 \
  python3 docs/fixtures/glm_b550_prefill_20261010/run_quality.py \
  docs/fixtures/glm_b550_prefill_20261010/q22-c8192-b8192-mmq-gated.config.json \
  docs/fixtures/glm_b550_prefill_20261010/UNIQUE-QUALITY-OUTPUT --limit 30
```

For the long subset, use the 16K config, `--task-indices 10,11,17,25` and
`--background docs/fixtures/glm_b550_prefill_20261010/background-14000.txt`.
The pinned dataset is `datasets/HumanEval.jsonl.gz`. Generated Python
executes inside bwrap without the model, repository, home directory or network.
Quality-run timing is diagnostic, not qualified performance.

For recall, use `recall-cCONTEXT.ids`, one trial and 256 output tokens, then run
`score_recall.py PREFIX --context CONTEXT`. The decoder emits the requested
count even after a turn ends; the frozen scorer uses only the first assistant
turn and requires exact JSON values and a stop. Run `summarize.py` to refresh
`summary.json` from the raw records.

## Opt-in engine controls and validation

- `STRATA_GLM_F16_PREFILL_MIN_TOKENS`: 0..16384, default 0. With an FP16
  expert-prefill mode, shorter batches use MMQ. GPU decode retains its mode.
- `STRATA_GLM_PREFILL_KDA_COLUMNS`: 32/64/128; absent inherits global columns.
- `STRATA_GLM_PREFILL_KDA_CHUNK`: 64/128/256/512/1024/2048; absent uses 64.
- `STRATA_GLM_PREFILL_KDA_MIN_TOKENS`: 0..16384, default 0. Shorter batches
  inherit global columns/row parts and 64-token chunks. Decode and rollback
  capture retain their previous geometry.
- `--dump-prefill-logits=PATH`: optional last-prompt-token FP32 distribution,
  before cache preparation or generated tokens. The benchmark config enables
  it with `record_prefill_logits: true`.

HIP and CUDA decoder/parity builds succeeded. GPU parity ran on RX 9070 XT;
CUDA was compile-only, with no NVIDIA GPU available. These GLM GPU targets are
not enabled for SYCL. Expanded prepared/unprepared serial-KDA tests passed
bitwise for every output and final state, including 8192-token inputs,
columns 32/64/128 and chunks 64/256/2048. Parallel row parts use tolerance tests.
Serial kernel parity does not establish full-model FP16 parity.

With the new controls absent, the engine reproduced the frozen 128-token
baseline output exactly. The benchmark-tool unit suite passed 11 tests;
changed Python tools compile, and `git diff --check` passes.

Other excluded records: `base8k` (trailing-newline parser failure),
`base32k-01` and `hybrid-8k-01` (external `hf` CPU interference),
`hybrid-b16384-32k-01` (cold-cache accounting failure before launch).
`prepared32-profile8k` changes decode scheduling through profiling; its decode
rate and tokens do not qualify throughput or default parity.
