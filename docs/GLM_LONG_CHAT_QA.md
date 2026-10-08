# B550 document-chat context ladder

This test checks the assembled REAP50 Q23 model on Ryzen 9 3950X and RX 9070 XT
at 16K, 32K, 64K and 128K context capacities (K = 1024). The request reserves
approximately 90% for the fully templated document prompt and 10% for generated
tokens, including reasoning. Eight additional tokens are left for the server's
context safety reserve. Normal end-of-answer stops are honored: output capacity
is not the same as actual generated output or full-window occupancy.

The GPU expert cache is 3072 MiB throughout the ladder, down from the 6144 MiB
used by the normal 4K B550 service. This leaves room for growing attention state
and makes the four context cases comparable to each other. These rates should
not be labeled repeats of the earlier 6 GiB-cache throughput benchmark.
Other settings retain FP16 batched expert prefill, batch 2048, 12 CPU workers,
greedy generation, low reasoning effort and no speculation. The guard enforces
60 GiB RAM, no swap and at least 2048 MiB of free global VRAM. TR16 is not tested
because the user has another task running there. The GGUF declares a 1048576-token
model context; that metadata is not a claim of usable capacity on this machine.

## Fixture and scoring

Each deterministic fictional museum-project dossier has eight decision records
spread through a larger set of numbered background inspection notes. The facts
include provisional and superseding launch dates, funding allocations, migration
progress, an exact rollback rule, owners and risk mitigation, a privacy rule,
and a version phrase. The task asks for a summary, eight cited answers and next
actions. The auditor question checks that the model does not invent an identity.
The test also requires combining 315 verified crates with the total 840:
525 remain and 37.5% are complete. Budgets must not count the 90000-credit
contingency twice.

This is a synthetic, low-density retrieval/summarization smoke test with repeated
background prose. It is not a held-out natural-document benchmark, comprehensive
long-context evaluation, or evidence that REAP preserves full-model quality.
Answers are assessed manually against independently stored expected values.
Transport success, normal completion, factual quality and memory fit are reported
separately. No automatic keyword score is presented as semantic correctness.

## Reproduction

From the B550 repository root, with the GPU available exclusively:

```sh
.venv-glm-hip/bin/python tools/glm_long_chat_check.py \
  --contexts 16384 32768 65536 131072 \
  --output build-hip-glm-v11/chat-context-reproduction
```

The output directory must not exist. The runner prepares all fixtures with the
model's tokenizer and the server's chat template, stops the standard API, starts
a separate guarded service for each case, saves raw requests/responses and memory
telemetry, and restarts the standard API when it finishes. It never changes GPU
clocks or model weights. Failed cases are recorded rather than silently shortened.
`--prepare-only` builds fixtures without stopping or starting services.

A case's `fixture.json` records actual templated token count, output allowance,
SHA-256 and approximate fact positions. `result.json` contains raw API output
and engine timings; `answer.md` is the final answer; `memory.json` contains the
memory guard record. `config.json` captures every runtime setting. First-request
prefill follows expert-page warmup and is not a repeated warm-prefill timing.
There is one request per length, with the same eight questions across lengths;
these are not independent examples or three-trial throughput qualifications.

## Completed cases

| Context capacity | Templated input | Output allowance | Actual output | Prefill tok/s | Decode tok/s | QA correctness | Finish |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 16384 | 14736 | 1638 | 1255 | 197.4 | 9.5 | 8/8 | stop |
| 32768 | 29482 | 3276 | 965 | 195.6 | 9.4 | 8/8 | stop |
| 65536 | 58973 | 6553 | 1075 | 185.6 | 9.3 | 7/8 full; 1 partial | stop |
| 131072 | 117955 | 13107 | 1056 | 173.4 | 9.2 | 8/8 | stop |

Output counts include reasoning. All four completed cases had zero swap, no memory
limit/OOM event and no guard rejection. The 16K case peaked at 56.98 GiB RAM
with 6262 MiB minimum free VRAM; 32K peaked at 57.00 GiB with 5888 MiB free.

The 16K summary and answers were accurate. Its suggestion to hold contingency
unspent is a conservative next action, not an explicit permanent prohibition
in the source. The 32K answer also got all eight questions and citations right,
but one next-action sentence reverses the local-copy handling sequence: copies
must remain local **until** supervisor checks, rather than becoming local-only
after those checks. It also places actions before the questions instead of at
the requested end. Factual QA success does not imply a flawless overall response.

Raw cases and manual assessments are under
[fixtures/glm_long_chat_20261008](fixtures/glm_long_chat_20261008/).

The 64K case fits (57.12 GiB peak RAM, 5140 MiB minimum free VRAM, no swap,
limit/OOM event or rejection), but it is **not a clean quality pass**. It gets
525 remaining crates and 37.5% complete right, then compares that completion
percentage against the 99.7% checksum-agreement threshold. Its next-action list
invents an acceptance goal of about 838 verified crates. The source defines no
such goal: verification progress and checksum agreement are different metrics.
This is an interpretation error, not a missing-fact or context-fit failure.

The 128K case completed normally with all eight answers and evidence IDs correct.
Peak RAM was 57.16 GiB, with 3644 MiB minimum free VRAM, zero swap and no memory
limit/OOM event or guard rejection. Its answer kept checksum agreement separate
from crate completion. The 64K error did not recur, so this small test does not
establish a monotonic context-related quality decline or a reliable cutoff.

**Capacity and output limit:** 128K here means a 131072-token configured window,
with 117955 input tokens and a 13107-token output allowance. The model naturally
stopped after 1056 output tokens; it did not generate a 13K-token answer. Thus
this tests a roughly 118K document plus a normal summary/QA answer, not quality
across a fully consumed 128K window or a sustained 13K output. The output allowance
includes reasoning. The request took about 13.25 minutes (680.26 s prefill and
114.41 s decode), excluding startup and expert-page warmup.

## Supplementary multi-turn chat

After the ladder, a second user turn is appended to the completed 32K document
conversation. It deliberately proposes the superseded date, double-counts
contingency, assigns rollback authority to the wrong person, misreads a single
99.8% audit, treats 99.7% checksum agreement as a crate-count goal, and asks about
sharing an unchecked working copy. This is a targeted diagnostic follow-up,
including an error observed during the ladder, not an independent held-out test.

The follow-up has 30552 input tokens and a 2208-token output allowance. It uses
the remaining 32K window after adding the previous answer and new question;
it is supplementary to, rather than another row of, the 90/10 ladder.
The server re-prefills the conversation; no prefix-cache reuse is claimed.

```sh
.venv-glm-hip/bin/python tools/glm_long_chat_check.py \
  --followup-from build-hip-glm-v11/chat-context-reproduction/32768 \
  --output build-hip-glm-v11/chat-followup-reproduction
```

The follow-up stopped normally after 537 output tokens: 200.6 tok/s prefill,
9.1 tok/s decode, 57.11 GiB peak RAM and 5888 MiB minimum free VRAM. Swap and
limit/OOM counters remained zero; there was no guard rejection.

Manual score: **5/6 correction checks succeed**. It corrects the launch date,
funding, rollback condition, authorizer and working-copy handling. However, it
explicitly accepts the false premise that 838 verified crates meets the 99.7%
rule. That applies a checksum-agreement threshold to a different metric. The
cited OPS-043 does not support the conclusion. This reproduces the same type of
interpretation error as the 64K answer, even when the user asks it to check the
assumption. The report therefore does not qualify reliable document reasoning
from the successful fact retrieval or 128K memory fit.

[Follow-up answer](fixtures/glm_long_chat_20261008/followup-32k/answer.md),
[manual assessment](fixtures/glm_long_chat_20261008/followup-32k/assessment.json),
and [combined machine-readable results](benchmarks/glm_b550_long_chat_20261008.json).

## Conclusion

All four context configurations fit this workload under 60 GiB RAM and the 2 GiB
VRAM reserve. The largest first-request prefill measured 173.4 tok/s and decode
9.2 tok/s with a 3 GiB expert cache. Those are below the earlier 10+/200+ targets;
the earlier 2K warm benchmark used a 6 GiB cache and remains a separate result.
No timing here is a three-trial median. No full 13K-token generated answer was
tested. The synthetic QA succeeds at most lengths, but the threshold confusion
in one long answer and the follow-up is a material quality limitation. No weights,
precision setting or prompts were revised to conceal those failures.
