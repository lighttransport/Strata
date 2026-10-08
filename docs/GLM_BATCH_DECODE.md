# GLM independent decode batches (experimental)

The GLM GGUF engine can keep up to eight independent sequences on one GPU. Each slot owns its
KDA recurrence and convolution state, MLA latent cache and pooled index keys, and token position.
Dense projections and routed expert work are shared across the active rows. Slots may have different
prompt lengths and finish independently.

This is an experimental implementation. The targets are 30–35 single-stream tokens/s and about
200 aggregate tokens/s with eight active requests; these are targets, not measured results.
The v3 configs remain the qualified configurations. The v4 configs are candidates for measurement.
The single-stream candidate enables the owned-allocation cache allowance and leaves MLA graphs
off, following the measurements below. Its 12K context still needs separate memory qualification.

## Serving

Set `"parallel": 8` in a GLM GGUF server config. The Python server uses its existing request queue,
slot admission, cancellation and streaming protocol. The engine advertises `batch_slots=8`.
Requests enter batch slots directly, including the first request. GLM cannot reuse a solo prefix during
promotion, so the adapter avoids re-prefilling generated tokens. Requests are greedy, as in the existing GLM engine. Native EXL3, multiple GPUs, legacy expert
caches and lookup speculation are not supported with GLM sequence batching.

The engine reserves sequence state before choosing its prefill workspace. It reduces the prefill
chunk size if the requested workspace cannot fit. Long contexts leave less memory for resident
experts. Admission resets a slot and reads its prompt; there is no prefix-cache reuse for this path.
A failed shared decode step terminates the engine because several slots may have changed state.

## Optional optimizations

- `STRATA_GLM_MLA_GRAPHS=1`: pass the MLA position through device memory, enabling graph replay
  at changing positions. Applies to the single-sequence target model, including split verification.
- `STRATA_GLM_TIER_OWNED_RESERVE=1`: allocate MTP weights/history before sizing a CLI decode cache
  and account for those owned allocations when sizing its runtime allowance. A single sequence
  with weights and history allocated keeps 512 MiB for checkpoints and later decode buffers;
  sequence batching retains the larger allowance, deducting at most 512 MiB already owned.
  Device and physical memory limits remain in force. This needs memory qualification on the target system.
- `STRATA_GLM_SPLIT_PROJECTIONS=1`: compute token-independent attention input projections across
  a whole split verification window. This experiment can trade reduced weight reads for more
  waiting between groups; measure it on the intended workload before selecting it.
- `STRATA_GLM_Q23_WIDE=1`: use an eight-column canonical expert tile for windows wider than four.
  CPU/GPU arithmetic and reduction order stay unchanged. Value `2` separately launches groups
  of up to four rows and groups of five to eight rows; this is an additional experiment.
  Shared-memory staging uses the actual projection width.
- `STRATA_GLM_BATCH_SPLIT=1`: split independent decode rows into two groups for CPU/GPU overlap.
  Requires `STRATA_GLM_SPLIT_VERIFY=1`; graph capture is disabled for the per-slot mixer state.
- `STRATA_GLM_BATCH_MTP=1`: opt into independent speculative sequences, with MTP depth one or two.
  Verification packs consecutive tokens from multiple sequences into a window of at most eight
  rows. Rejected prefixes are restored and replayed; expert placement remains fixed through a
  speculative window. This requires MTP in the config and does not support the RAM tier or legacy
  adaptive decode cache. Ordinary independent decoding is the default.

## Measurement

`tools/glm_batch_bench.py` drives concurrent requests through the production `GlmEngine` adapter.
It records prompt IDs, delivered output IDs, arrival times, config, environment and engine logs.
It reports decode throughput separately from wall time including admission. `full_batch_decode_tok_s`
counts only steps with all requested slots active. A run with early EOS is marked by
`all_reached_limit=false`; it does not establish throughput for the requested full output length.
This measures the API engine adapter, not HTTP transport overhead.

For example, using the repository's Python environment:

```sh
numactl --interleave=all build-exl3-venv/bin/python tools/glm_batch_bench.py \
  configs/glm53f-q2-v4-batch-short.json --prompt-tokens 1024 --tokens 1024 \
  --repetitions 3 --output build-q2-v4/short-1024
```

Use `--prompt-tokens 8192` and the long-context config for the long case, and repeat with
`--tokens 2048`. Output directories must be new. Run under `tools/glm_q2_run_guard.py` for memory
and external CPU interference checks. `--require-idle` rejects a run when another process uses
sustained CPU time. For a long functional run, `--record-interference` records those intervals without
stopping generation; a nonempty interference record is not a clean timing result. Memory limits
remain enforced in either mode. Do not run simultaneous GPU benchmarks.

For correctness, `--check-sequences=8` on the decoder compares batched and isolated logits at
widths one through eight and checks two consecutive three-token windows. The benchmark's
`--check-sequential` compares complete batched and isolated output token streams and requires
routing affinity zero. `--cancel-slot N` exercises cancellation after four delivered tokens.
Changes to routing affinity or other lossy options still require the established held-out and
coding quality gates in `GLM_Q2_DECODE_REDESIGN.md`.

## Kernel measurements (2026-10-07)

RTX 5060 Ti 16 GB, Threadripper 1950X, canonical Q2_K gate/up and Q3_K down, 4096 hidden and
2048 intermediate dimensions. The table gives median milliseconds per call from three alternating
runs of `glm_q23_parity --bench` (50 calls per case). It includes activation quantization and both
projections. These are kernel measurements, not end-to-end token rates.

| Expert groups | Rows per expert | Four-column tile | Eight-column tile |
| --- | --- | ---: | ---: |
| 8 | 8 | 0.575 ms | 0.496 ms |
| 24 | 8 | 1.641 ms | 1.400 ms |
| 24 | varying from 1 to 8 | 1.102 ms | 0.992 ms |

All tested output rows were bitwise equal to the canonical CPU reference. The eight-column tile
reduced time by about 14–15% for the fully shared eight-row cases. The bucketed variant measured
1.031 ms for the varying-width case, so it was slower than the direct eight-column launch there.
Logs are in `build-q2-v4/q23-mode*-r*.log` on the measurement machine.

## Single-stream measurements (2026-10-07)

On the same machine, one 1024-token coding prompt and 1024 generated tokens, three decode trials,
MTP depth two and routing affinity 0.10 gave the following results. These CLI measurements used
context 8192, prefill chunk 2048, 15 CPU workers and normal NUMA placement. They do not qualify
the candidate config's different context and workspace sizes.

| Variant | Resident experts | Decode tokens/s, three trials |
| --- | ---: | --- |
| Frozen baseline, fixed cache | 3066.69 MiB | 22.02, 21.76, 20.60 |
| MLA graphs, same cache | 3066.69 MiB | 21.75, 21.38, 20.70 |
| Owned-allocation allowance, MLA graphs off | 3761.69 MiB | 22.04, 22.73, 22.71 |

All three runs completed without process swap or recorded external CPU interference. Baseline
and MLA graph output streams were identical across all 3072 generated tokens. Graph replay
reduced CPU enqueue time but did not improve total decode time in this comparison. The larger
cache run reported 975–1016 MiB GPU free memory at trial ends; these are end measurements,
not the minimum free memory throughout generation. Its peak host RSS was about 113.74 GiB.

Logs and memory records are under `build-q2-v4/single-fixed-baseline-r1*`,
`single-fixed-mla*` and `single-owned-plain2*`. The 30–35 tokens/s target remains unmet.
The new candidates have not completed the full held-out and HumanEval quality gates; lossless
kernel parity and selected output parity do not qualify the routing-affinity settings.

Additional full-length runs use the owned-allocation allowance, MLA graphs off, MTP depth two,
affinity 0.10 and prefill chunk 2048. Short prompts use context 8192; long prompts use 12288.

| Prompt tokens | Output tokens | Resident experts | Decode tokens/s, three trials | Recorded interference samples |
| ---: | ---: | ---: | --- | ---: |
| 1024 | 2048 | 3839.88 MiB | 21.35, 21.11, 21.27 | 13 |

All listed trials reached the requested output length without process swap. These runs recorded
external CPU activity and are not clean timing qualifications. Records use
`build-q2-v4/single-owned-<prompt>-<output>-final-guard.{log,stdout,memory.json}`.

## Independent M=8 measurements (2026-10-07)

Eight different coding prompts, ordinary greedy decoding, routing affinity 0.05, one trial per
case on the same machine. Decode rates sum delivered tokens across the eight requests; the full
batch column includes only steps with all eight active. Admission includes reading each prompt
and allows existing requests to advance between prompt chunks. Wall rates exclude process startup.

| Prompt tokens per request | Output tokens per request | Full batch decode tok/s | All decode tok/s | Including admission tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 23.50 | 23.43 | 16.20 |
| 1024 | 2048 | 22.42 | 22.39 | 18.45 |
| 8192 | 1024 | 21.01 | 20.92 | 7.66 |
| 8192 | 2048 | 20.41 | 20.37 | 11.06 |

The first case completed all 8192 requested tokens without process swap. Its guard recorded one
`kswapd0` interval, so this is not labeled clean timing. Resident expert capacity was 2962.44 MiB.
Results and per-request output IDs are in `build-q2-v4/m8-short-1024-final/`; the sibling
`m8-short-1024-final-guard.memory.json` records memory and interference.
The 2048-output case completed all 16384 requested tokens without process swap; its guard
recorded three host reclaim activity intervals. Its records use `m8-short-2048-final`.
The 8192-prompt/1024-output case also completed all requested tokens without swap, but recorded
35 interference samples including desktop CPU activity. Its prefill chunk was reduced to 2048
tokens to fit the eight states; resident experts occupied 1520.31 MiB at the final admission.
Its records use `m8-long-1024-final`. These single-trial results do not establish clean performance.
The 8192-prompt/2048-output case completed all 16384 requested tokens without swap, with 54
interference samples. Its final admission cache was 1537.69 MiB; records use `m8-long-2048-final`.
All four cases completed without a memory-guard rejection. Neither requested throughput target
was reached by these runs.

For the 1024-output case, the engine counted 16.78 TB of logical CPU expert reads and 267.19 seconds
of CPU expert work, an effective 62.8 GB/s. There were 8184 decode tokens after the eight initial prefill tokens, giving
about 2.05 GB of CPU expert traffic per decode token. At unchanged traffic, 200 tokens/s would
require about 410 GB/s for that work alone. This calculation explains why the kernel microbenchmark
gain is insufficient on this workload; reaching the target requires much less CPU expert traffic.
