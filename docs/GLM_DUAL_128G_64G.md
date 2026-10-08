# GLM single decode: full model on 128 GB and REAP on 64 GB

For the current cross-host summary and reproduction scope, see
[the experimental release candidate](GLM_RELEASE_CANDIDATE.md). These are recorded
TR16 results; no new benchmark was run while the host was occupied by another task.

The target is 30 delivered tokens/s for one sequence. It has not been reached.
The measurements below use a Threadripper 1950X, 128 GB DDR4, two NUMA nodes,
and an RTX 5060 Ti 16 GB. Full-model runs have a 124 GiB memory cgroup; REAP
runs have a 60 GiB cgroup. Both prohibit swap and leave at least 2,048 MiB of
actual free VRAM, including the desktop and CUDA driver allocations.

Numbers are greedy generation with verified MTP, not throughput across several
requests. The short case has 1,024 prompt tokens and 512 generated tokens,
repeated three times. All listed short cases generated 512 tokens per trial
without an earlier end token.

## Short measurements

| Model and change | Prefill tokens/s, median | Decode tokens/s, median | Status |
| --- | ---: | ---: | --- |
| Full Q23, lazy page placement | 65.873 | 14.347 | Loading recorded kernel compaction |
| Full Q23, NUMA placement after prefill, MTP depth two | 65.939 | 17.508 | Clean; exact outputs and final logits |
| Full Q23, MTP depth one | 65.918 | 19.243 | Exact; first decode trial recorded kernel compaction |
| Full Q23, MTP1, CPU prefetch and interleaved auxiliary allocation | 65.890 | 19.662 | Clean; exact outputs and final logits |
| REAP Q23 control | 102.478 | 20.214 | Clean |
| REAP Q23, NUMA placement after prefill | 102.783 | 21.018 | Clean; exact outputs and final logits |
| REAP Q23, Q8 MLA/mHC and 2K context | 102.993 | 22.037 | Clean; precision change, quality smoke only |
| REAP Q23, Q8 MLA/mHC and 12K context | 102.895 | 23.022 | Clean; 8K category comparison passed |
| Same plus CPU software prefetch | 102.820 | 23.486 | Clean; exact outputs and final logits |

The 2K context row is a short-context experiment. The new profiles retain
12K context. The matched 8,192-prompt / 128-output measurements were:

| Model | Prefill tokens/s, median | Decode tokens/s, median | Peak cgroup RAM | Minimum free VRAM | Guard |
| --- | ---: | ---: | ---: | ---: | --- |
| Full Q23, MTP1, NUMA placement, CPU prefetch and interleaved allocation | 374.359 | 21.231 | 108.17 GiB | 2,356 MiB | Kernel reclaim and compaction |
| REAP Q23, MTP1, NUMA, Q8 MLA/mHC and CPU prefetch | 407.436 | 23.967 | 55.47 GiB | 2,358 MiB | Clean |

The REAP decode trials were 21.032 / 23.967 / 24.593 tokens/s. Full-model
trials were 18.969 / 21.231 / 21.595. Both completed without swap or OOM.
Full-model long timings are diagnostic because the guard recorded kernel CPU
work, including reclaim and compaction in this run. The preceding long recipe measured
373.933 prefill / 20.152 decode and was also diagnostic. Interleaving did not
eliminate the interference in the long case.
The REAP long control before CPU prefetch measured 407.835 prefill / 23.469
decode. Both updated long cases matched all 384 generated IDs and final logits
of their respective controls exactly.
Earlier REAP measurements are in
[GLM_REAP50_64G_16G.md](GLM_REAP50_64G_16G.md).

NUMA relocation moved every routed-weight page successfully. It adds about
24 seconds of setup for the full model and 12 seconds for REAP, outside the
prefill and decode timing intervals. Output IDs and final logits matched the
unmodified placement in all three 512-token trials.

The full-model MTP1 test read about 1.46 TB of logical CPU expert bytes per
512-token trial. Its last trial spent 20.15 seconds in the CPU layer pass.
This is why a small GPU-kernel improvement alone cannot supply the remaining
decode gain. REAP halves the stored expert count, but still selects eight
experts per token.

## Quality screens

These are two 256-target tuning sequences, one code and one prose. They are
feasibility screens, not full qualification. The additional limit is 1% per
category; combined perplexity cannot hide a failing category.

| Change | Code perplexity change | Prose perplexity change | Screen |
| --- | ---: | ---: | --- |
| REAP Q8 MLA/mHC, against current REAP Q23 | -0.10% | -3.82% | Pass; broader checks needed |
| Same plus routing margin 0.05 | +3.37% | -1.73% | Reject |
| Same plus routing margin 0.02 | -0.87% | +4.66% | Reject |
| Full model, Q4 large mixer matrices | +6.30% | +0.51% | Reject |
| Full model, Q4 large Q5 matrices, retain Q6 | +7.04% | +2.06% | Reject |
| Full model, Q4 attention output projections | +1.13% | +0.71% | Borderline; not selected |

The REAP model's earlier regression against full canonical Q23 remains. Its
existing 512-target baseline measured +4.58% code and +10.60% prose perplexity.
The Q8 screen above measures an additional change relative to that REAP pack;
it does not qualify pruning as a replacement for the full model.

A matched 8,192-target tuning comparison (4,096 code and 4,096 prose) also
passed. Code perplexity was 3.717257 / 3.693623 before/after Q8, a 0.64%
decrease. Prose was 18.405764 / 18.395855, a 0.05% decrease. Both runs were
clean, completed all sixteen sequences, and used the 60 GiB cap with zero
swap. This larger check is still tuning data; HumanEval and independent
held-out qualification have not been run for Q8.

## Implemented experiments

All new switches are opt-in. Original model files are unchanged.

- `STRATA_GLM_NUMA_AFTER_PREFILL=1` populates and relocates lazy routed-weight
  mappings after prefill, before decode. A read-only page-location probe found
  only 78.8% of sampled full-model pages on their intended node before this fix.
- `STRATA_GLM_TIER_PREALLOCATE=1` creates pinned expert-upload staging before
  the model fills RAM. `STRATA_GLM_MAILBOX_PREALLOCATE=1` also creates the
  decode mailbox early. Neither removed full-model kernel compaction in the
  measured sustained MTP1 cases.
- `STRATA_GLM_COMPACT_VERIFY_HISTORY=1` stores one recurrent-state base plus
  the KDA inputs needed to reconstruct retained prefixes. Seven slots use
  233 MiB instead of about 1 GiB. Mixed-width graph and rollback checks pass.
- `STRATA_GLM_FLOATING_Q8=mla|all` converts otherwise expanded MLA matrices,
  and optionally mHC matrices, into owned Q8 GPU copies. It also avoids an
  unused FP32 copy of already-Q8 draft matrices. This is a precision change
  when the source is not already Q8.
- `STRATA_GLM_DENSE_Q4=mixers|all|q5|outputs` tests narrower dense conversions.
  The output head, routers, first three dense layers, and routed experts are
  preserved. None of these Q4 variants has passed the category screen.
- `STRATA_GLM_MTP_CONTINUATION_MARGIN` can shorten low-confidence draft
  continuations. Per-position proposal and acceptance counts are logged;
  this policy has not yet been selected.
- `STRATA_Q23_PREFETCH=512` enables software prefetch in the existing CPU dot
  kernels. A guarded synthetic Q23 flow sweep measured 74.79 GB/s for width one,
  against 70.34 / 70.70 GB/s in controls. The whole-model REAP gain was smaller:
  23.022 to 23.486 tokens/s, with all 1,536 IDs and final logits identical.

The full-model profiles use the Linux-only `tools/glm_numa_interleave.sh`
launcher, which requires `numactl`. `STRATA_GLM_NUMA_EXECUTABLE` names the frozen
decoder. This interleaves auxiliary CPU allocations; expert rows are still
relocated to their designated nodes. One three-trial 1K run was clean,
19.385 / 19.750 / 19.662 tokens/s. The preceding prefetch-only run measured
19.719 median but recorded compaction. The benchmark worker already used
`numactl --interleave=all` in both runs; adding the wrapper did not introduce
a new allocation policy in that comparison. The wrapper applies that same
policy when the server launches the decoder. The clean repeat does not show
that the wrapper removed compaction.

Smaller task counts and compiling the Q23 kernel with `-mtune=znver1` did not
give a repeatable benefit. The instruction-tuning test retained exact output
hashes and passed 5,952 scalar-dot comparisons; it is not selected.

An exact CPU expert-group merge was also implemented. It can share a weight
read across both verification groups and publish the first group before the
second finishes. All 312 CPU comparisons, GPU graph checks, and retained-prefix
checks passed. However, sustained REAP decode fell from 22.037 to 18.538
tokens/s, despite about 9% fewer CPU bytes. It loses too much GPU overlap and
remains disabled (`STRATA_GLM_MOE_MERGE_GROUPS=0`).

Long suffix lookup proposals likewise failed to improve sustained REAP speed.
Direct seven-token lookup measured 19.264 tokens/s; chaining only long suffix
matches measured 20.156, against the 20.214 control. Outputs and final logits
remained exact. Lookup remains disabled.

The measured profiles are
`configs/glm53f-q23-v11-128g-16g-experimental-{short,long}.json` and
`configs/glm53f-reap50-q23-q8-v11-64g-16g-experimental-{short,long}.json`.
They reference the frozen local executable. Existing v10 profiles remain
available. The Q8 profiles stay experimental.

Run from the repository root, with a fresh output prefix:

```sh
build-exl3-venv/bin/python tools/glm_low_memory_bench.py \
  configs/glm53f-q23-v11-128g-16g-experimental-long.json \
  build-q2-v4/single-8192.ids --output build-q2-v11-dual/full-reproduction \
  --ram-gib 124 --tokens 128 --trials 3 \
  --gpu-capacity-mib 16384 --gpu-used-limit-mib 14336

build-exl3-venv/bin/python tools/glm_low_memory_bench.py \
  configs/glm53f-reap50-q23-q8-v11-64g-16g-experimental-long.json \
  build-q2-v4/single-8192.ids --output build-q2-v11-dual/reap-reproduction \
  --ram-gib 60 --tokens 128 --trials 3 \
  --gpu-capacity-mib 16384 --gpu-used-limit-mib 14336
```

Use one GPU process at a time. The memory harness clears the selected model's
inactive clean file cache before entering the cgroup. When switching models,
also clear the other model's inactive cache as recorded in these experiments.

The baseline NUMA/Q8 recipes passed server checks with repeated requests, cancellation
after four tokens, and successful recovery in the same engine process. The
repeated and recovered 32-token outputs matched exactly. Full-model mixed-width
GPU graph, retained-prefix and rollback checks also passed with the final
frozen executable. CPU prefetch and the explicit launch wrapper were then checked
separately for output/logit equality. These are functional checks, separate
from timing.

Raw configurations, frozen build hashes, memory records, and equality reports
are in `build-q2-v11-dual/`; the collected records are in
[glm53_flash_dual_v11_measurement.json](glm53_flash_dual_v11_measurement.json).
Earlier full-model cases marked with kernel compaction remain diagnostic.
The short repeat with the explicit wrapper passed the same guard. No 30-token/s or
fully qualified new precision profile is claimed.
