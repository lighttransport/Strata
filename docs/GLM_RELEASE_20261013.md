# GLM REAP-224 local release candidate, 2026-10-13

This is a local source candidate on `glm53f`, based on `889136c3`, for the existing REAP-224 TR16 tuned
profile. Full HumanEval completed with **150/164 passes**, below the 152/164 release floor. The candidate
is not qualified for release. No version tag or publication is part of this preparation. Compact measurements, source/build identities and audit
findings are in [the evidence record](fixtures/glm_release_20261013/summary.json).
The 35 tok/s decode target was not met. The 700 tok/s prefill screen is not a qualified release claim.

For output quality, prefer the full, unpruned Q23 model (288 experts, 120.9 GB)
when RAM permits, with unchanged routing and strict MTP acceptance. Earlier full
Q23 runs scored 156/164 at MTP depth 2 and 154/164 at depth 1 on different builds;
these results do not isolate the effect of depth or pruning. The
[full-model guide](GLM_RELEASE_Q2Q3.md) records the measured quality/speed choices.
REAP-224 remains available as a smaller experimental model; availability does not
qualify this candidate for release.

## Scope

Removed the unsuccessful runtime experiments from the follow-up optimization: coarse CPU scheduling,
ready/early-device/staged merges, route-first GPU ordering, fused KDA verification, uneven verification
splits, device draft state, selective draft extension and the MMQ column hint. Frozen patches, binaries
and raw measurements remain in the ignored optimization directories; compact historical evidence remains
under `docs/fixtures/`. The tuned numerical settings are unchanged. Model and expert-prior paths in the
checked-in profile are relative examples; set them to the local files before running.

Retained bounded expert-service replay, hidden-state readback tracing, causal tier replay and guarded
fresh-process comparison tools. Kept the Clang/oneAPI shuffle-immediate fix and expanded CPU flow and Q8
MMQ coverage. The replay path is diagnostic and the throughput runner rejects it.

The optional batched FP16 accumulation path now honors the same lossy-prefill permission as ordinary
FP16 accumulation. Short batches retain FP32 accumulation even when the batched option is enabled.
The precision test exercises both batched layouts, ordinary products, partial output tiles, row padding
and short/long/short transitions. Added direct tests for bounded MTP acceptance, stop-token rejection,
tie ordering and the repeat guard. Environment-helper migrations preserve each switch's truthiness.

Q8 transcoding now observes the general lossy-prefill threshold as well as its own threshold. Its side
stream and events are drained and released during decoder teardown. The FP16/Q8 prefill experiments
reject multiple GPUs because their scratch buffers and events belong to the primary device.

The source packager includes the new replay dependencies, tests and compact release evidence. It omits
raw `.logits.bin` benchmark dumps, the historical `strata-numa-ab.sh` script with a literal authentication
value, and 14 older community environment snapshots containing a possible bearer credential. The manifest
records these omissions without changing the frozen originals. The same exclusions apply to the
accompanying patch. Archive membership and checksums are covered by a packaging test; broken symlinks
are rejected. The profile-path diff is omitted from the patch to avoid reproducing the old machine paths;
the archive includes the corrected profile, and the manifest records this separate patch exclusion.
The extracted source draft also configured, built and passed all five host GLM tests without CUDA, HIP,
model files or downloads.

Both new long-prefill budgets were also removed after the rejection screen: 16 sequences / 8,192 scored
tokens, 512-token prefill batches with the lossy threshold lowered to 512. Control perplexity/KL/top-1
were 4.9270 / 0.172604 / 86.4136%; nine groups and top-k six produced 7.3389 / 0.654512 / 72.3267%.
Incremental KL was +0.481907 against a +0.01 limit. This was a short-corpus rejection screen, not an 8K
quality qualification. Both runs were complete, guarded and used the same frozen experimental binary.

Added barriers before the FP32 and fused FP16 softmax kernels reuse shared reduction storage. Tests cover
small, partial and full rows, and compare the fused result with FP32 softmax followed by an FP16 cast.

## Validation

CUDA and HIP builds passed. All 19 CUDA/host GLM tests, four HIP-build CPU tests and the oneAPI shared
CPU flow test passed. The Q2 lookup test passed 5,952 cases; CUDA racecheck reported zero hazards in the
modified softmax kernels. The Python GLM suite passed 162 tests after dependency/sandbox-specific reruns.
Installer tests passed 462 tests with one Windows-only test skipped. The initial sandbox failures and
rerun logs are retained. All 2,048 warmup and timed tokens across 2K and 8K match the frozen tuned-profile
baseline. The 2K check lowered the Q8-specific threshold to one token to verify that the general
4,096-token precision guard takes precedence.
The complete corpus passed: perplexity 5.08287, KL 0.201527 and top-1 agreement 85.5347% across 8,192
targets, using 256-token prefill batches and evaluation width four. The decoder returned the expected
`trailing evaluation baseline records` notice after scoring all 16 sequences because the reference holds
additional sequences. There was no guard rejection, swap use or GPU reserve violation in the corpus run.
The full HumanEval rerun completed all 164 tasks: **150/164 (91.46%)**, compared with **152/164 (92.68%)**
for the frozen baseline. It used the same binary, profile, pinned dataset and greedy chat protocol with
low reasoning and a 2,048-token cap. Binary, compiled-source and input identities were checked before
and after the run. Generated solutions ran in isolated tests. The engine and resource guard exited
successfully; all 14 failures were test failures, with no wall-time test timeouts.

Five previously passing tasks failed in this run:

| Task | Function | Observed output error |
|---|---|---|
| 113 | `odd_count` | Omits an `n` in the required string: `4put` instead of `4nput`. |
| 118 | `get_closest_vowel` | Excludes index 1 from its reverse scan, missing valid answers for `full` and `bad`. |
| 147 | `get_max_triples` | Counts the polynomial's residues incorrectly; returns 4 for input 5 instead of 1. |
| 155 | `even_odd_count` | Skips zero digits; returns `(0, 0)` for input 0 instead of `(1, 0)`. |
| 158 | `find_max` | Selects the fewest unique characters instead of the most. |

Three baseline failures passed: task 129 (`minPath`), 139 (`special_factorial`) and 154 (`cycpattern_check`).
Only tasks 32 and 130 reached the token cap; both were already capped in the baseline, so there are no new
capped tasks. The five new failures are output logic/formatting errors, not timing failures. No failed
generation was resampled or replaced. This single run establishes the reported score; it does not isolate
whether the changed outputs came from runtime changes or run-to-run variation. The 152-pass gate remains unmet.

Shared CPU/GPU activity was explicitly allowed for this quality-only rerun. The guard permitted up to
1 GiB of process swap, retained its host-memory limits, and recorded GPU headroom without enforcing a
reserve. Observed peak swap was 909,776 KiB (888.5 MiB), and GPU free memory stayed at or above 858 MiB.
These relaxed timing conditions do not change the quality thresholds and support no performance claim.
The earlier 72-task run interrupted by unrelated GPU work and a subsequent startup attempt rejected by
the zero-swap guard are preserved and excluded from the final score.

Measurements use a Threadripper 1950X with 128 GB DDR4-3200 and an RTX 5060 Ti 16 GB at 150 W, PCIe 3.0 x8.
Corpus and HumanEval runs record CPU interference and make no timing claim. These are short-prefill
quality checks; the 8K output-parity check does not establish new long-prefill quality. AMD and Intel GPU
execution and a full SYCL GPU-engine build are unavailable here. Existing compiler warnings remain.

## Local publication audit

The prospective Git destination is `origin/glm53f` (`lighttransport/Strata`), with remote tip `4894c996`.
There is one existing outgoing commit, `889136c3`, and the update would be a fast-forward. No commits,
tags, pushes or release uploads were made during this preparation. Existing upstream GLM discussions
are [issue 2067](https://github.com/Niko1221/Strata/issues/2067) and
[PR 1420](https://github.com/Niko1221/Strata/pull/1420); no messages were posted.

Gitleaks reported no findings in the outgoing commit. The new working files' 38 generic matches were
output SHA-256 hashes. The full source preflight also matched older hash manifests and tensor names;
one historical loopback script contained a literal authentication value. Further review identified
`CRAWL4AI_BEARER_TOKEN` in 14 existing community snapshots; hexadecimal form alone does not make a value
a checksum. These 15 files are excluded from the archive, and the frozen originals are preserved.
The token's owner/status and any required rotation remain unresolved, so Git publication is blocked.
No credential value is reproduced in this report. The archive also omits 33 raw logit dumps.

The existing outgoing commit still contains the original machine-specific model/prior paths and an
assistant session URL in its message. The current source profile uses relative paths, but preparing a
Git publication requires cleaning that unpublished history and repeating the range audit. This task
has not rewritten history. The source archive is a working-tree snapshot with a per-file manifest.
