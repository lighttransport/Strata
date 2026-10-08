# GLM single-stream optimization experiments

These are development measurements on October 7, 2026, not a qualified release
configuration. The current targets are 35 tokens/s for single-stream decode and
400 tokens/s for prefill. P11 exceeded the prefill target in a clean 8K trial,
but its precision changes still need full quality qualification. The decode
target has not been reached. Batched-decode
optimization stopped; its earlier results are in [GLM_BATCH_DECODE.md](GLM_BATCH_DECODE.md).

The eight-hour experiment window is 08:55:57–16:55:57 UTC. Trials alternate
decode, prefill, decode, prefill. One GPU experiment runs at a time, with no
concurrent builds. The local handover is `resume.md`; local raw records and
deadline-enforced runners are under `build-q2-v5/`.

## Machine and workload

- Ryzen Threadripper 1950X, 16 physical cores, two NUMA nodes, 125 GiB usable RAM.
- RTX 5060 Ti, 16 GiB, PCIe Gen3 x8. Desktop applications share the GPU.
- GLM-5.3-Flash UD-Q2_K_XL with the canonical Q2_K gate/up and Q3_K down expert
  pack, 111,188,901,888 bytes. Expert pages are placed by NUMA node before timing.
- Decode comparisons use the same 1,024-token prompt, 1,024 generated tokens,
  three repetitions, MTP, and routing affinity 0.10. This affinity changes
  routing; these timings do not qualify its output quality.
- Prefill comparisons use an 8,192-token prompt and chunk, context capacity
  12,288, MMQ experts, and three timed repetitions after warm-up.
- Runs called clean below had no process swap or recorded external CPU
  interference according to `tools/glm_q2_run_guard.py`.

## Completed comparisons

All rates are tokens/s. Each cell lists the three timed repetitions.

| Trial | Baseline | Candidate | Decision |
| --- | --- | --- | --- |
| D01: split verification vs unsplit | 22.85 / 22.53 / 23.18 | 19.75 / 19.45 / 20.04 | Retain split |
| P01: two vs four staging workers, persistent FP32 dense copies | 225.36 / 218.48 / 216.09 | Allocation failed | No candidate timing |
| D02: MTP depth 2 vs 3 | 22.22 / 23.46 / 22.83 | 21.11 / 21.70 / 22.13 | Retain depth 2 |
| P02: two vs four staging workers, Q8 dense storage | 225.34 / 223.79 / 224.72 | 227.97 / 227.97 / 227.44 | Four workers remain a candidate |
| D03: narrow Q23 GPU kernels | Kernel comparison only | No consistent improvement | Not retained |
| P03: host vs copy-stream buffer reuse wait | 223.68 / 221.97 / 222.86 | 225.79 / 223.06 / 227.87 | Small gain; no default change |
| D04: four-row Q5/Q6 dense scheduling | Kernel comparison only | Slower in all six projection cases | Reverted scheduling change |
| P04: reuse dequantized panels and vectorize MLA gather | 221.96 / 221.49 / 220.98 | 239.69 / 238.73 / 239.41 | Retain as the prefill candidate |
| D05: Q23 vs Q22 expert pack | 22.05 / 21.73 / 22.33 | 21.83 / 23.71 / 22.54; loading interference | Retain Q23 |
| P05: prepare exact-order KDA inputs | 237.05 / 237.96 / 238.00 | 240.10 / 241.21 / 241.23 | Retain as an opt-in candidate |
| D06: concurrent split-verification CUDA streams | 21.48 / 21.20 / 21.89 | 21.22 / 20.96 / 21.62 | Removed; slower in every pair |
| P06: FP16 MLA attention | 237.64 / 236.47 / 240.91 | 265.11 / 266.52 / 266.30 | Precision candidate; quality gates pending |
| D07: MTP depth 2 vs 1, fixed cache | 21.81 / 21.71 / 22.43 | 21.42 / 21.53 / 21.80; browser interference | Retain depth 2 |
| P07: staged expert prefetch, shared pinned buffers | 240.21 / 238.25 / 240.70 | 317.80 / 319.12 / 319.16 | Retain opt-in; bitwise equality and cancellation checks passed |

D01 used a fixed 4,096 MiB cache budget, with 4,091.81 MiB allocated in both
arms. Both were clean and all 3,072 generated token IDs matched. D02 requested
5,632 MiB but available GPU memory limited the actual caches to 4,552.25 and
4,474.06 MiB respectively; both runs were clean. Changing MTP depth therefore
also changed the available cache allowance.

P01's four-worker arm failed after desktop GPU usage increased. P02 restored
`STRATA_GLM_Q8_DECODE=1`, reducing GPU allocation from 12,897.4 to 12,346.8 MiB.
P02 and P03 were clean, and the two final output IDs matched across their arms.
Those two IDs alone are not a full prefill-logit or quality validation.

D03's six alternating kernel runs passed canonical CPU/GPU bitwise parity.
Median baseline/narrow times were 0.230/0.236 ms for eight experts and one token,
0.241/0.236 ms for eight experts and two tokens, and 0.636/0.713 ms for 24 experts
and two tokens. The narrow dispatch and extra kernel instantiations were removed;
the additional two-token validation cases remain.

D04 extended the existing four-row dense layout to single-column Q5_K/Q6_K
calls. It passed 207,901 bitwise comparisons, including a Q6_K matrix with an
uneven output size. Three alternating benchmark pairs on 4,096-input,
16,384-output matrices found no win for one, two, or three activation columns.
For example, median single-column Q5_K time rose from 0.126 to 0.137 ms and
Q6_K from 0.166 to 0.188 ms. The scheduling change was reverted; the stronger
reference comparison and optional dense benchmarks remain in
`mmvq_multi_parity`.

The preceding D04 decode profile captured 512 outputs over 219 MTP rounds.
Q5_K/Q6_K projections took an aggregate 6.13 seconds on the GPU and resident
expert kernels 2.41 seconds. CPU expert flow took 16.60 seconds and read
1.128 TB of logical expert bytes. The GPU mailbox wait kernel spins while
waiting for CPU results, so its duration is not additional arithmetic work.
Nsight's report importer appeared in the interference record after capture;
this run is diagnostic and is not a clean throughput qualification.

P04 used `STRATA_GLM_DEQUANT_ONCE=1` and `STRATA_GLM_MLA_VECTOR_GATHER=1`
together, with four staging workers, Q8 dense storage, and the deferred copy
wait in both arms. Both runs were clean. All 154,880 saved final logits were
bit-identical, as were the two generated token IDs. The combined change improved
the mean prefill rate by about 8%; this comparison does not separate each
option's contribution. These settings remain opt-in pending broader workload
qualification.

D05 fixed the cache budget at 4,096 MiB. Q23 was clean. Q22's first attempt
was stopped during loading after process swap reached 48,080 KiB. After
releasing only the unused Q23 pack's file-cache pages, the retry completed
without swap. It recorded five CPU-interference samples during the first
12 seconds of loading, from that cache release and kernel memory compaction;
it is not a clean qualification. Its median gain was only about 2%, and the
additional quantization loss has not passed the full coding gates, so Q23
remains the selected pack.

P05 moves KDA input normalization and scalar preparation out of the recurrent
loop while preserving its state accumulation order. Kernel comparisons checked
all output and state bits at five token counts, including 8,192. Both full-model
arms were clean; all 154,880 final logits and both output IDs matched exactly.
The median gain was about 1.4% on top of P04. `STRATA_GLM_KDA_PREPARE=1`
with 128 columns remains opt-in.

D06 gave each split-verification group a separate CUDA stream, cuBLAS handle,
and 32 MiB scratch arena, with per-layer events preserving recurrent state
order. Both handles must use pedantic math: leaving the new handle at its
default caused a sequential-logit mismatch. After that correction, all graph,
eight-token sequential, and retained-prefix checks passed. Both timing arms
were clean and all 3,072 output IDs matched, but the candidate was slower in
every pair. The prototype was removed; its patch and tested binary are retained
locally under `build-q2-v5/`.

P06 used the existing `STRATA_GLM_MLA_F16=1` path. The first baseline
recorded a browser CPU spike during loading; the table uses its clean repeat.
The candidate was also clean. Both generated token IDs matched, but all final
logits changed (maximum absolute difference 1.83). A 512-token held-out prefill
smoke check gave perplexity 8.082 versus 8.124 for FP32, KL 0.0205, and 92.6%
top-token agreement. This is one sequence, not a quality qualification.

The new optional `--eval-prefill` flag evaluates teacher-forced prefill rows
using `--prefill-batch`; ordinary `--eval-corpus` still evaluates decode.
It requires the decode expert cache disabled. The output head processes at
most eight rows per native projection call, allowing all prefill logits to
be saved or compared in the existing evaluation format. Both smoke arms
completed successfully; full held-out and coding gates remain pending.

D07 kept the cache budget at 4,096 MiB. Depth 2 was clean; depth 1 recorded
browser CPU activity during timing. Depth 1 reduced drafting time, but its
additional target rounds increased verification time. It showed no measured
win and is not a clean qualification, so depth 2 remains selected.

## Prefill profile

The P02 follow-up CUDA profile used Q8 dense storage, four staging workers, and
the same 8K prefill geometry. Profiled timings are diagnostic, not throughput
qualification. Aggregate durations included:

- SIMT matrix products: 5.46 seconds in the two largest kernel groups.
- Tensor Core matrix products: 3.27 seconds in the largest kernel group.
- KDA chunk kernels: 3.20 seconds.
- MLA cache gather: 2.01 seconds.
- Host-to-device copies: 15.49 seconds, transferring approximately 111.2 GB.

Kernel and transfer times can overlap and must not be added to estimate wall
time. Scratch allocation peaked at 1,862.81 MiB within a 4,096 MiB arena. This
suggests unused capacity for this geometry; it does not establish a safe smaller
arena for other chunk sizes or context capacities.

The host embedding vector now releases after its synchronous device copies.
The profiled run and P03 completed successfully with this change. Long-context
decode, full prefill-logit comparisons, held-out quality, and HumanEval still
need qualification for any selected final configuration.

## Staged expert prefetch

P07 adds an opt-in worker that stages upcoming expert groups and queues their
GPU copies while the layer mixer runs. It joins before expert consumption and
uses events to protect host and device buffer reuse. The scratch-cap option
keeps the existing allocation bounds checks. These options currently support
one GGUF GPU.

The first 8K comparison measured 231.75 / 239.09 / 241.36 tokens/s without
prefetch and 315.49 / 315.65 / 309.02 with nine groups prefetched. Final logits
and output IDs matched exactly. Both arms had zero swap but recorded kernel
memory-reclamation activity during timing, so neither is a clean qualification.

The implementation now reuses the existing two pinned staging buffers instead
of allocating an additional 1,495.51 MiB of pinned host memory. A 512-token
held-out check compared all 79,298,560 logits bit for bit with the FP32
reference and passed, with no recorded interference or swap. The full 8K
retest was clean: 317.80 / 319.12 / 319.16 tokens/s versus 240.21 / 238.25 /
240.70, with a 2,304 MiB scratch cap in both arms. All final logits and output
IDs matched. A single-request server check passed prefill and decode
cancellation, then produced identical 32-token replies in the same process.
That correctness check used lazy NUMA placement and is not a throughput result.

The retained options are `STRATA_GLM_STAGE_PREFETCH=1`,
`STRATA_GLM_PREFETCH_GROUPS=9`, and
`STRATA_GLM_PREFILL_SCRATCH_CAP_MIB=2304`, together with the P05 lossless stack.
The cap was measured for this 8K geometry; other shapes retain bounds checks
and may need more scratch. The options remain disabled by default.

## D08 mild gate skipping

The 30-percent gate-mask candidate measured 21.78 / 22.70 / 23.49 tokens/s
versus 22.29 / 22.06 / 22.96 without skipping, on the same 1K prompt/output
geometry. Both arms were clean with zero swap. The median gain was only 1.9%;
the lossy candidate is not selected and its combined quality remains unqualified.

## P08 GPU staging buffer reuse

`STRATA_GLM_PREFETCH_REUSE_DEVICE=1` reuses the first two prefetch buffers as
the regular GPU staging ring. Their shared completion events protect both uses.
This saves 416 MiB and allows twelve prefetched groups with only 82.5 MiB more
GPU allocation than the nine-group P07 setup. The option requires at least two
staged groups and remains disabled by default.

The paired 8K trial measured 327.16 / 333.16 / 333.61 tokens/s with twelve groups,
versus 293.42 / 293.31 / 292.72 with nine, both with 2,304 MiB scratch. Both arms
were clean with zero swap. All 154,880 final logits and output IDs matched.
Separate nine- and twelve-group checks each matched all 79,298,560 held-out
logits. The baseline was slower than P07's earlier measurement; this comparison
uses the paired runs. Retain twelve groups and device reuse as opt-in candidates.

## D09 adaptive MTP continuation

Keeping two proposals and attempting a third only when the draft's top-two
logit margin reached 2.0 passed a 256-token unbiased output comparison. It
measured 22.59 / 20.76 / 22.67 tokens/s versus fixed depth two at 22.52 /
22.37 / 23.09; both runs were clean with zero swap. The median gain was 0.3%
and the mean fell 2.9%. The prototype was removed; its function and executable
remain in the experiment directory. Fixed depth two remains selected.

## P09 eighteen prefetched groups

The eighteen-group, 512-token correctness check matched all 79,298,560 logits.
At 8K, twelve groups measured 324.38 / 324.11 / 324.38 tokens/s cleanly.
Eighteen groups with a 2,048 MiB scratch cap exceeded the GPU allocation budget
before timing, so there is no eighteen-group throughput result. Keep twelve.

## D10 CPU task sizing

A canonical expert-flow screen compared 4, 6, 8, 12 and 16 tasks per worker
with one- and two-token groups, 96 synthetic experts, 1,000 measured layers,
15 workers, and two reversed repetitions. All final output checksums matched.
No setting won consistently across both group widths; retain twelve tasks.
For example, six tasks measured 1,048 / 1,058 microseconds for one token versus
1,095 / 1,056 at twelve, but 2,068 / 2,132 for two tokens versus 2,057 / 2,078.
This is a CPU microbenchmark, not end-to-end decode throughput.

## P10 FP16 MLA with staged prefetch

With twelve prefetched groups, GPU ring reuse and a 2,048 MiB scratch cap,
FP16 MLA measured 383.03 / 371.15 / 372.91 tokens/s versus FP32 at 299.97 /
305.72 / 297.69. Both arms had zero swap. The baseline was clean; the FP16
arm recorded ChatGPT CPU work during loading and kernel memory reclamation
during timing, so its timing is not a clean qualification. Output IDs matched, but
all 154,880 final logits changed (maximum absolute difference 1.831). Carry
FP16 forward for full quality qualification; it is not yet qualified.

## D11 larger decode cache

Requesting 5,120 MiB allocated 4,995.31 MiB of experts and measured 23.20 /
24.06 / 23.72 tokens/s. Requesting 4,096 MiB allocated 4,091.81 MiB and measured
22.96 / 22.66 / 23.44. Both runs were clean with zero swap. Carry the larger
request forward for full qualification; existing live-memory limits still cap
the actual tier. Routing affinity makes output quality depend on tier contents.

## P11 prefill-only KDA row partitioning

`STRATA_GLM_PREFILL_KDA_ROW_PARTS=4` selects the existing four-part recurrence
for fast prefill. Decode retains its original recurrence and rollback history.
The switch is opt-in and changes floating-point reduction order.

With FP16 MLA, twelve prefetched groups and a 2,048 MiB scratch cap, four parts
measured **403.18 / 402.48 / 402.34 tokens/s**, clean with zero swap. One part
measured 367.81 / 367.33 / 372.81, but recorded external CPU work during timing.
All final logits changed (maximum absolute difference 2.020); output IDs matched.
The 512-token smoke measured perplexity 8.1053 versus 8.0824, KL 0.01935,
and top-1 agreement 93.95%. This is not full quality qualification.

The candidate has been frozen as `build-q2-v5/selected.json` and a separate
executable, with hashes in `selected-build.json`. Full quality and short/8K
prompt plus 1K/2K output checks are the next step.

## Frozen candidate qualification (in progress)

The selected configuration combines FP16 MLA and four-part prefill KDA with
twelve prefetched groups, a 2,048 MiB scratch cap, requested 5,120 MiB expert
cache, MTP depth two, and routing affinity 0.10. Two consecutive 8K requests
in the same server process produced identical 16-token outputs, with no swap
or recorded interference. This correctness check used lazy NUMA placement.

Full-workload timing uses normal NUMA placement and three repetitions:

| Prompt / output tokens | Prefill tokens/s | Decode tokens/s | Guard result |
| --- | --- | --- | --- |
| 1,024 / 1,024 | 66.06 / 66.05 / 66.05 | 22.14 / 23.10 / 22.57 | Clean |

The other three performance cases and full quality gates are still running.
The 400+ result above is for the 8K prefill optimization trial; the decode
target remains unmet. No qualified-release claim is made.
