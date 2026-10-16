# V100s optimization measurements, 2026-10-16 JST

The selected profile reached **50.3616 decode tok/s** and **850.218 prefill
tok/s** on the training prompt, with a 2 GiB primary expert cache, a 28 GiB
worker cache and the three-token repeat guard. Held-out code reached **49.9607
tok/s**, and held-out chat **52.3375 tok/s**. All twelve streams passed the
stop and repetition checks. The separate best prefill median was **855.043
tok/s**. These are opt-in experimental throughput measurements; model quality
has not been qualified.

Hardware: two Tesla V100 PCIe 32 GB GPUs and two Xeon Gold 6240 CPUs, 36 physical
cores / 72 logical CPUs, four NUMA domains (two per socket), 160 GiB allocation,
190,991 MiB OS-visible RAM. RAM speed was unavailable. GNU 12.2.1, CUDA 12.9,
sm70, NVIDIA 580.178.04. Both GPU budgets are 30,720 MiB. GPU 0 is near NUMA
node 0; GPU 1 is near node 2.

Weights: GLM-5.3-Flash REAP-224 Q23, 95,538,419,872 bytes, SHA-256
`3ef5726ae4a9768fa8311453927674401f3958a8c3884effddac92fc41262003`.
The base source is `6e4277070684cc20a1663cd7981c7fce6eadec80`; the final code
commit is `0c1598183878b989e50b540b2625a78f8b355829`.
[Validation](validation.json) records build, parity and placement checks.
[Results](results.json) preserve individual source-file, patch, binary, prompt and
raw-output hashes. [Profiles](profiles.json) contain the environment settings with
repository-relative artifact paths; each result lists its case overrides.

## Complete trials

| Configuration | Prefill tok/s | Decode tok/s | Scope |
| --- | ---: | ---: | --- |
| Merged build, earlier release flags | 352.603 | 38.2241 | 8K / 512 |
| Merged build, updated release decode flags | 353.696 | 44.0683 | 8K / 512 |
| Frozen build, later decode control | 425.994 | 41.6281 | 8K / 512 |
| Fast prefill, worker ordered additions and direct upload | 855.043 | — | 8K, 8 generated tokens |
| Combined fast configuration, code prompt | 854.443 | 43.2910 | 8K / 512 |
| Combined fast configuration, held-out code | 854.630 | 43.5333 | 8K / 512 |
| Combined fast configuration, held-out chat | 853.009 | 44.8017 | 8K / 512 |
| Socket-owned rows and token grids, 10/14 GiB caches | 852.490 | 46.1849 | 8K / 512 |
| Local counters and batched convolution, 10/14 GiB caches | 851.779 | 45.6654 | 8K / 512 |
| Same, 0/28 GiB caches | 852.279 | 47.8005 | 8K / 512 |
| Same, 2/28 GiB caches | 852.421 | 51.3874 | 8K / 512 |
| Same, small-batch Q23 kernels | 850.871 | 51.6943 | 8K / 512 |
| Matched control, small-batch kernels off | 850.061 | 51.3342 | 8K / 512 |
| Final code, small-batch kernels and registration cleanup | 851.326 | 51.5549 | 8K / 512 |
| Final code, small-batch kernels off | 849.437 | 51.4822 | 8K / 512 |
| Final code, guard 4, held-out chat | 849.677 | 51.4638 | 8K / 512 |
| Selected guard 3, training code | 850.218 | 50.3616 | 8K / 512 |
| Selected guard 3, held-out code | 850.874 | 49.9607 | 8K / 512 |
| Selected guard 3, held-out chat | 850.753 | 52.3375 | 8K / 512 |

Each row uses a prefill warmup and three timed prefills. Full decode rows use a
512-token warmup followed by three timed 512-token repeats restored from the same
prefill state. Decode rates count generated tokens after the initial token.
Staging and hashing are untimed. Adaptive cache learning continues across repeats,
so decode token streams can differ. This protocol does not meet the stricter
identical-output gate in `tools/glm_v100_optimize.py`; complete warm trials and
output-health checks are reported separately from that gate and model quality.
The earlier target-reaching repeats were 51.3874, 51.2108 and 51.6327 tok/s
(warmup 40.4906). The guard-4 final-code repeats were 51.5549, 50.8429 and
51.8462 tok/s.
Selected guard-3 training repeats were 50.1444, 50.3616 and 50.4059 tok/s
(warmup 42.1182). All four generated 512 tokens with no early stop or
consecutive repetition cycle. This is a warm-cache result.

The direct worker's separate best prefill passes were 855.169, 855.043 and
854.096 tok/s. Every repeat matched warmup **last-prompt logits and aggregate
route counts**. With four worker cores local to GPU 1 (18,19,20,23), staged upload
plus ordered additions reached 853.494 tok/s; repeats 854.230, 853.494 and
852.015 also passed. Direct upload needs one worker core (35); the primary uses
cores 0–34. No profiler runs overlap the accepted timings.

## Changes

The worker honors `STRATA_GLM_PREFILL_ORDERED_SCATTER=1`, using the same ordered
weighted additions as the primary. The existing atomic path remains the default.
`STRATA_GLM_MIXED_DIRECT_UPLOAD=1` registers read-only model pages and copies
expert gate/up/down weights straight to the device. It handles disjoint tensor
ranges and shared end pages; registration stays alive until DMA completes.
The direct path omits unused CPU staging buffers and copy threads.

`STRATA_GLM_NORM_Q8_FUSED_BATCH=1` extends existing fused normalization/Q8 reuse
to short verification batches. `STRATA_GLM_POINTWISE_BATCH=1` groups independent
mHC reads, MLA head projections and KDA gates/outputs into token grids.
`STRATA_GLM_CONV_BATCH_HISTORY=1` computes convolution and per-token rollback
histories in a token grid, then updates the live history in a separate ordered
launch. Per-token arithmetic and reduction order are preserved.

`STRATA_GLM_SOCKET_ROWS=1` assigns expert row halves to physical CPU sockets,
rather than treating the first two SNC NUMA domains as two sockets. Socket IDs
need not be consecutive. It requires pinned workers in both sockets,
`STRATA_GLM_Q2_NUMA_WEIGHTS=1`, lazy mapped expert views, externally placed pages,
and no layer pipeline or post-prefill page migration. Claims and completion
counts stay local to each socket; the host acquires both completion sequences.
The global scheduler remains the default.

One sealed read-only RAM copy is pinned by its owner and shared by both
processes. Half of each expert projection's rows is first-touched on each
socket, interleaved across that socket's two NUMA domains. Other model bytes
are interleaved across all four nodes. The
[placement plan](socket-row-plan.json) describes 129 tensor byte ranges, including
the MTP block, and is bound to the full model size and SHA-256. Pages crossing a half boundary
have no unique owner and are excluded from placement samples. The plan introduces no
new tensor-name or assembled-pack rules. The resident-model helper accepts this
plan through `socket_row_plan=`. It checks complete byte coverage, overlap,
full SHA-256, seals and sampled page placement before children run.

`STRATA_GLM_Q23_SMALL_BATCH=1` specializes canonical expert kernels for one
and two submitted tokens, reducing shared activation storage and omitting wider
accumulator paths. Q2_K/Q3_K CPU-oracle comparisons passed with the option off
and on, including underflow and quantization-rounding cases. In isolated V100
microbenchmarks, eight single-token experts went from 0.182 to 0.170 ms and 24
from 0.471 to 0.429 ms. Wider-batch behavior and the default path are preserved.
The full-model medians were 51.6943 tok/s enabled and 51.3342 disabled.
Corresponding warmup/timed token streams matched byte for byte across the two
runs. Final-source medians were 51.5549 enabled and 51.4822 disabled. These small
timing differences should not be read as a precise speedup.

## Reproduction

Use the `v100s` entry in [profiles.json](profiles.json). The profile is scoped to
the hardware above; CPU masks must be adapted to another machine's physical
socket and PCI topology. In particular, CPU IDs 18–21 are not all in GPU 1's
NUMA domain on this node.

The 8K prompts already live in
`docs/fixtures/glm_tr16_20261009/prompts/8k/`: `train-code.ids`, `holdout-code.ids`
and `holdout-chat.ids`. Their hashes are in each result. The expert prior is
`tools/sim/data/v100s_release/coverage-keep224.json`.

Run the owner under `numactl --interleave=all`, with `PYTHONPATH=tools`, enough
RAM and a sufficient memlock limit. The owner must stay alive for the entire
mixed run:

```python
import json, os, subprocess, sys
from pathlib import Path
from glm_resident_model import ResidentModel

folder = Path('docs/benchmarks/v100s_opt_20261016')
profile = json.loads((folder / 'profiles.json').read_text())['v100s']
plan = json.loads((folder / 'socket-row-plan.json').read_text())
env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
env.update(profile['env'])
with ResidentModel('models/model.gguf', profile['model_sha256'],
                   max_bytes=128 * 1024**3, socket_row_plan=plan) as resident:
    subprocess.run([
        sys.executable, 'tools/glm_mixed_run.py',
        '--primary', 'build/strata-glm-decode',
        '--worker', 'build/strata-glm-gpu-worker',
        '--model', str(resident.path), '--primary-device', '0', '--worker-device', '1',
        '--primary-cpus', '0-34', '--worker-cpus', '35', '--combine', 'device',
        '--worker-cache-mib', '28672', '--worker-gpu-budget-mib', '30720',
        '--report', 'tmp/v100s-run-memory.json', '--',
        '@' + profile['prompt'], '512', '4096', '34', '--context=12288',
        '--prefill-batch=8192', '--gpu-budget-mib=30720', '--gpu-devices=0',
        '--cpu-affinity=numa', '--decode-experts=cpu', '--bench=3', '--decode-bench=4',
        '--decode-cache-mib=2048', '--prefill-experts=f16-batched',
        '--prefill-expert-cache-mib=0', '--decode-graphs', '--speculative=mtp',
        '--draft-depth=3', '--mtp-experts=gpu', '--warm-weights',
        '--check-prefill-repeats',
    ], env=env, check=True)
```

Use `--warm-weights`: `--lock-weights` would create a private COW copy and defeat
sharing. The public socket copier was exercised on a 64 MiB fixture on both
sockets, including sealing and full SHA-256. The public placement verifier also
checked ten panels in the actual resident 95 GB model. The timed runs used the
preceding equivalent placement prototype; a complete 95 GB copy by the public
helper has not yet been tested.

## Validation and limits

GPU tests covered 80 norm/Q8 cases, 80 pointwise/guard comparisons, 20 strided
MLA head comparisons, and 180 convolution/history cases. CPU flow tests compared
588 executions with an independent whole-row phased oracle, both with a host
worker and without one; the latter also exercised sleeping workers. Four- and
eight-token model verification, retained-prefix logits and rollback passed
with batched norm and pointwise operations enabled. A width-eight check with
both options off reproduced the same mismatch on the frozen base build.
With new options off and on, a 64-token prompt followed by 16 greedy tokens
matched frozen baseline token bytes and both 154,880-float logit arrays.

The fast prefill profile uses FP16 MLA absorb/index, exchange and partials;
repeatability does not establish fidelity against the original arithmetic.
The selected decode profile uses repeat guard 3, affinity 0.14, soft MTP
margin 6 / top-k 8 / max-soft 3, draft top-k 4 and dense Q4. The imported release profile is itself experimental;
its reported HumanEval result was 146/164, below its 152/164 qualification floor.
Completion, absence of early stops and absence of repetition cycles are limited
output-health checks, not coding or language accuracy measurements.

The faster guard-4 profile reached 51.5549 tok/s on training and 51.4638 on
chat, but one timed held-out code stream cycled (period 34 from token 335). Its
49.5951 tok/s code median is excluded. Guard 3 removed that observed cycle;
guard 2 also passed stream checks but fell to 45.8213 tok/s on held-out code.
These checks do not establish coding correctness.

Excluded measurements include failed repeated-prefill comparisons, MTP depth
5/7 runs with cycles, an 8/28 GiB cache run with a cycle, and the 4/28 GiB cache
run (50.2226 tok/s) with a cycle. The CUDA-event profile's 54.1041 tok/s has
different lookup acceptance and is instrumented, so it is excluded. Its graph
times include CPU mailbox waits. Nsight failed at decode and produced no usable
trace. Existing custom one-query MLA products were slower than cuBLAS. A
correct-shape Q4_K vocabulary-head microbenchmark measured about 0.50 ms for
one token and matched per-column output bits; it did not justify a new head.

CUDA sm70, HIP and the Vulkan backend builds pass. Nine host tests and six
resident-model tests pass. The local pool stress test skipped for unavailable
AVX-512 features. AMD GPU execution, a SYCL build (no compiler installed),
full-288 weights, server latency, long-context accuracy and paired coding
quality evaluation have not been performed for these changes.
