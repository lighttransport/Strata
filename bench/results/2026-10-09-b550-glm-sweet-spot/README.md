# B550 REAP50 speed/quality sweep — 2026-10-09

**Recommendation: scalar affinity .06, an experimental quality-first profile.** It sustains **16.46 tok/s decode / 229.12 tok/s warm prefill** on 2K + 512, and completes the document and C++ tasks with a tested parser. Faster .08/.10 settings failed the C++ answer requirement at the same cap. The [opt-in configuration](../../../configs/experimental/glm53f-reap50-q23-b550-quality-first.json) uses an 8K context, 60 GiB RAM guard when launched through the guarded runner, and a 512 MiB VRAM reserve. It requires the tested HIP build and B550 model paths. Serving defaults are unchanged; the demo stays stopped.

Hardware: Ryzen 9 3950X, 62.693 GiB physical RAM, RX 9070 XT gfx1201, 16,304 MiB VRAM, PCIe 4 x16, automatic GPU clocks, CPU powersave with boost. Linux 7.0.0-38-generic, ROCm 7.14 development build. Engine source baseline `15b787b5` plus the prior canonical-rounding fix and experimental tooling; binary SHA256 `e067fcd6977afa7722f06058838e3ae06d38a8f937a0ccb4bd1e2f6d4a8c6544`. The isolated B550 checkout is `/home/syoyo/work/Strata-b550-calibration-15b787b5`. No local GPU performance tests were run.

Model: `GLM-5.3-Flash-REAP50-Q23-assembled.gguf`, 58.87 GiB, 144 experts, top 8, 42 MoE layers, gate/up Q2_K and down Q3_K. Model SHA256 `c3b0cfb144556bd28e265d8255d48eeafce3d127aba5b9b5dd4e326322a34aa9`. This is a comparison against unchanged REAP50 Q23, not the original unpruned model.

## Measurement design

The scalar routing bias is swept at 0, .04, .06, .08, .10 and .12. Every case uses the same 8,192 MiB requested cache, 15,792 MiB GPU budget, 512 MiB physical reserve and 256 MiB tier runtime headroom. This fixes the cache/reserve confound in the earlier baseline comparison. Zero bias uses canonical strict combine; positive bias uses the existing device combine, so the difference includes both route selection and combine behavior. CPU workers 12, prefill batch 2,048, context 4,096, ordinary greedy decode. Guards enforce 60 GiB RAM and zero swap.

Speed uses the same 2,048-token fixture and 256 emitted output tokens, three repetitions per process. Decode times 255 steps after the initial token. Model loading and cold prefill are excluded; measured prefill is warm after an untimed prefill. Adaptive residency persists between repetitions, and biased outputs can differ. All per-run settings, hashes, timing lines and token IDs are retained.

Quality uses a common-state 64-row teacher-forced logit grid, plus real chat requests: a roughly 7.3K-input document Q&A and a roughly 4K-input C++ review. One fresh loopback server per bias; requests run in the same document-then-code order. Model pages and buffers stay warm, but the decoder resets state and rebuilds the expert tier for each request; no prefix reuse. API timings are fresh-request timings and are not warm benchmark medians. The document response asks for eight concise answers and three actions, avoiding the previous redundant summary. The C++ review contains empty-input, overflow, failure-state preservation, bounds and retained-view lifetime/thread-safety issues. It asks for concrete triggers and a corrected parser. Exact rendered prompt counts and hashes are saved. Natural EOS is honored; output caps are 768 and 1,280 tokens. These are small synthetic quality checks, not general model certification.

## Reproduce

Run only while B550 is idle, in the isolated checkout:

```sh
PY=/home/syoyo/work/Strata/.venv-glm-hip/bin/python
$PY tools/glm_b550_calibrate.py --output runs/repro-sweet-speed \
  --tokens 256 --trials 3 --cases aff00 aff04 aff06 aff08 aff10 aff12
$PY tools/glm_b550_calibrate.py --output runs/repro-sweet-paired --tokens 32 --trials 1 \
  --cases aff12 --env STRATA_GLM_CHECK_AFFINITY=64 \
  --env STRATA_GLM_AFFINITY_QUALITY_GRID=0,0.02,0.04,0.06,0.08,0.10,0.12
$PY tools/glm_b550_quality_sweep.py \
  --config configs/glm53f-reap50-q23-b550-60g-9070xt-experimental.json \
  --decoder build-hip/strata-glm-decode --output runs/repro-sweet-quality \
  --affinities 0 .04 .06 .08 .10
```

Run each command sequentially, using fresh output directories. The quality runner stops its temporary service and leaves the serving demo stopped. No release default changes.

## Speed sweep

Three-run medians and ranges, tok/s. All cases passed guards; minimum free VRAM was 652 MiB.

| Bias | Decode median | Decode range | Warm prefill median |
|---|---:|---:|---:|
| 0.00 | 11.60 | 11.43–11.63 | 230.12 |
| 0.04 | 13.94 | 13.59–14.74 | 229.88 |
| 0.06 | 16.31 | 16.14–17.03 | 230.40 |
| 0.08 | 17.80 | 17.20–18.60 | 229.93 |
| 0.10 | 19.83 | 19.18–20.27 | 229.53 |
| 0.12 | 21.05 | 20.15–21.42 | 229.54 |

The preceding routing fit was frozen before this sweep. All six predictions meet the 10% tolerance; errors are -3.2%, 1.5%, -4.8%, -4.5%, -6.4%, -3.9%. No new fit was used to choose the measured speed band. Biased repetitions are not byte-identical because routing follows adaptive residency.

## Common-state quality diagnostic

64 teacher-forced rows, fixed resident inventory, one prefill state. Zero bias restores exact logits. These are diagnostic probabilities, not a semantic quality score.

| Bias | Mean KL | Top-1 agreement | Mean relative logit L2 |
|---|---:|---:|---:|
| 0.00 | 0.000 | 64/64 | 0.000 |
| 0.02 | 0.266 | 54/64 | 0.167 |
| 0.04 | 0.244 | 51/64 | 0.183 |
| 0.06 | 0.301 | 49/64 | 0.204 |
| 0.08 | 0.350 | 50/64 | 0.223 |
| 0.10 | 0.390 | 47/64 | 0.237 |
| 0.12 | 0.461 | 49/64 | 0.255 |

The semantic tests use 7,358 document prompt tokens and 4,082 C++ prompt tokens. Neither reuses a conversation prefix. Positive bias can produce repetition even at .04; the full answers therefore take priority over the KL ranking.

## Real chat quality

One request per task and setting, all output tokens including reasoning counted. These are not multi-run medians. Document has 7,358 input tokens, C++ review has 4,082. `stop` means natural EOS; `length` means the cap was reached.

| Bias | Task | Fresh prefill tok/s | Decode tok/s | Output tokens | Finish |
|---|---|---:|---:|---:|---|
| 0.00 | document | 183.7 | 11.2 | 407 | stop |
| 0.00 | cpp-review | 208.6 | 11.0 | 1280 | length |
| 0.04 | document | 185.8 | 14.1 | 411 | stop |
| 0.04 | cpp-review | 208.5 | 16.8 | 1280 | length |
| 0.06 | document | 185.1 | 15.6 | 420 | stop |
| 0.06 | cpp-review | 207.8 | 16.1 | 1165 | stop |
| 0.08 | document | 185.8 | 17.1 | 475 | stop |
| 0.08 | cpp-review | 208.4 | 17.6 | 1280 | length |
| 0.10 | document | 183.8 | 18.2 | 415 | stop |
| 0.10 | cpp-review | 207.8 | 20.7 | 1280 | length |

All five document answers contain the eight correct factual answers and three actions, and finish naturally. Minor action-list issues remain: .06 infers that no external audit engagement exists from the absence of a named auditor; .08 has ambiguous rollback wording; .10 partly conflates remaining verification with the checksum agreement target. The response-format change removes the earlier threshold confusion from the numbered answers.

C++ results distinguish the settings:

- **0:** Identifies the five main defects and supplies a complete parser; it passes 5,020 compiled boundary/malformed-input cases. The response cap truncates the last registry suggestion. It incorrectly claims `parse_u64("12x", out)` leaves `out == 0`; the shown buggy code leaves 12.
- **.04:** Repetitive incorrect numeric reasoning until the cap, no final answer or parser.
- **.06:** Identifies the five main defects, finishes naturally and supplies a parser passing the same 5,020 cases. Its `"7x"` example wrongly claims `out == 0` instead of 7, matching the baseline's error pattern. Its safe parser handles empty input, overflow, ASCII validation and failure-state preservation correctly.
- **.08:** Partial review only; reaches the cap before providing a parser. Adds an incorrect overflow example/output. No parser to test.
- **.10:** Repetitive reasoning until the cap, no final answer or parser.

This supports **.06 as the quality-first experimental profile** among the tested 15–20 tok/s settings. It does not establish general equivalence to the unbiased model: its teacher-forced top-1 changes 15/64 rows, and only one document and one C++ task were used. No 32K/128K quality qualification, original-model comparison or general coding certification is claimed. The 5,020 parser tests exercise the supplied function; they are not 5,020 independent model prompts.

Parser checks used a single low-priority local CPU compile/run, with no local model or GPU performance jobs. Tests cover UINT64_MAX, overflow, signs, whitespace, embedded NUL, non-ASCII, leading zeroes, long strings, failure-state preservation and 5,000 fixed-seed generated inputs. GCC version and source are recorded under `parser-checks/`; the reusable harness template is `parser-harness.cpp.in`.

## Sustained candidates and recommendation

2,048 input + 512 emitted output tokens, three repetitions, same cache/reserve settings. Both passed guards with at least 652 MiB free VRAM, no swap or OOM.

| Bias | Median decode tok/s | Decode range | Warm prefill tok/s | Quality decision |
|---|---:|---:|---:|---|
| 0.06 | 16.46 | 16.32–17.18 | 229.12 | Recommend experimentally |
| 0.08 | 18.31 | 18.05–18.37 | 229.33 | C++ answer incomplete at tested cap |

Use `STRATA_GLM_ROUTE_AFFINITY=0.06` with the attached quality-first configuration. The configuration is opt-in, for the isolated HIP build and B550 model paths; it does not start the demo or enforce a RAM cgroup by itself. The guarded runners impose 60 GiB/no-swap. The eight-K context in the profile matches the chat quality checks; the throughput table uses a four-K context for the two-K prompt. Do not extend the speed/quality claim to 128K.

Reproduce sustained checks:

```sh
$PY tools/glm_b550_calibrate.py --output runs/repro-sweet-sustained \
  --tokens 512 --trials 3 --cases aff06 aff08
```

Validation: 90 Python checks pass, syntax checks and `git diff --check` pass. This turn changes only opt-in benchmark tooling, an experimental configuration and reports; the tested HIP/CUDA engine build is unchanged. No new GPU backend build was required. No default, model weight or serving checkout changed. All temporary servers and decoder processes stopped at the end; ports 8080/8081 are closed.
