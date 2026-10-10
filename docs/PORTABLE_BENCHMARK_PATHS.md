# Portable paths in benchmark evidence

The unpublished 2026-10-10/11 benchmark records were normalized for public
history. This is an explicit path-only edit to public copies of the evidence,
not a new measurement. Original unmodified blobs are retained in a local Git
bundle and backup ref; they are not part of the outgoing branch history.

Run the fixture scripts and engine configs from the Strata checkout root.
Paths now use this layout:

| Purpose | Portable location |
| --- | --- |
| Engine build and fixture/config files | Relative to the checkout, such as `./build-hip/strata-glm-decode` and `docs/fixtures/...` |
| Model assembly, expert sidecar and coverage data | `models/glm53f-reap/...` |
| Other GLM model variants | `models/glm53f/...` |
| Pinned HumanEval input and external reference logits | `datasets/...` |
| External GEMM source used by the experiment | `references/gemm/...` |

Supply those inputs locally, or copy the engine config and point its model,
expert-pack and coverage fields to your existing files. No model weights or
downloaded datasets are included in these commits. The reference checkout is
needed only to reproduce the reference-source study, not to run Strata.

Historical command arrays record the same invocation with normalized paths.
When reconstructing a `systemd-run` invocation, use the actual absolute
checkout path for its `WorkingDirectory` property (for example, `$(pwd)` from
the checkout). The maintained benchmark harness computes that path itself.

Timing values, output token IDs, cache fingerprints and recorded model,
source, binary and token/logit hashes were left unchanged. Recorded hashes
identify the original measured inputs and builds, not a regenerated run or a
hash of the normalized public metadata file. Generated model answers were not
edited. RAM-clock changes and unqualified/failed runs retain their original
labels and limitations.
