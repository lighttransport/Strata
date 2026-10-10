# B550 optimization pre-push audit, 2026-10-10

Scope: the local opt-in GLM decode kernel/transfer/deferral stack, fused norm/Q8, down-only Q22 overlay,
converter checks, experimental B550 config and measured results. No default config or DDR clock changed.
Local virtual environment, build scripts/output and resume-b550.md are excluded from the commit.

## Findings fixed

- CPU mailbox route mask was 32 bits, while eight-token decode can address 64 routes. Changed it to uint64_t;
  all 64 default bits and a bit-63-only mask pass an independent UBSan check using the source declaration.
- The new MLA scratch-size header now includes cstddef and declares std::size_t, so it is self-contained.
- Server ignores top-level route_affinity. The experimental config now sets STRATA_GLM_ROUTE_AFFINITY=0.08
  in env; its fit4 per-layer profile still takes precedence. This config uses absolute B550-local paths.
- Removed a historical local commit reference absent from this checkout and clarified completed quality checks.

## Validation

- HIP gfx1201 and CUDA sm89 full decoder, converter and Q8-test builds pass after the audit fixes.
  CUDA compilation only; no NVIDIA GPU is available. Existing narrowing/ignored-return/indentation warnings
  remain in the build logs. SYCL does not consume the changed GLM kernel headers or build these GLM targets.
- HIP canon_expert_test, glm_q8_test, glm_q23_parity and glm_mailbox_test: 4/4 pass.
  Additional STRATA_GLM_HEADS_BLOCKS=1 glm_q8_test passes; norm/Q8 includes 32 bitwise cases.
- tools/test_glm_assemble.py: 7/7 pass. Real sidecar integrity/rejection checks passed during this round
  before the audit (valid source, bad source/version, truncated and corrupt payload).
- Final decoder logits match the saved pre-round binary byte-for-byte on 4 x 64 scored tokens under the fixed
  Q23/mix2/two-route-deferral/kernel-stack recipe, with new lookup/norm switches off. SHA256:
  8e440b20833fcaf1bae078782779db37d97b2c69b13642079d47ca14c1d89292.
  Earlier new-switch-on parity has the same hash. This is a fixed experimental recipe check, not a complete
  stock-release or all-context equivalence proof.
- Fixture JSON, Python harness syntax, all shell harness syntax and regenerated summaries pass.
  New artifact credential-pattern scan passes; each new artifact is below 10 MiB.
- Source, harness, config and documentation whitespace checks pass. Full staged whitespace check retains
  findings in raw rejected-model outputs and unified patch context: these records are deliberately unedited.
  No model weights, virtual environments, logits or build binaries are staged. No active custom pre-push hook is configured in this checkout.

## Measurement and publication limits

Selected experimental Q22 fit4 repeats 33.5708 / 33.6053 tok/s at DDR4-2133, with 26/30 functional passes,
30/30 valid Python syntax, zero cap hits. Matched Q23 control passes 29/30. The 35.0717 variant is rejected
(23/30 passes, six malformed outputs, two cap hits). Full HumanEval and long-form code are untested.
Older kernel switches change FP32 reduction order; affinity, expert deferral, Q22 and Q3 repack are lossy.
The down-only Q22 sidecar is explicitly uncalibrated. No 35+ coding-coherent claim is warranted.

## Existing outbound branch history

Fetched origin before auditing. origin/glm53f = 27deb918; parent of this commit = 6ecf3bcb.
The parent is ahead by 415 commits and behind by zero; this commit adds one more. A normal branch push would
publish all 416 commits, not just this optimization. The existing range contains 3244 changed files and was
not subjected to a complete semantic/backend review here. Its diff whitespace audit emits 1185 report lines,
including old benchmark output and the PR template. No existing outbound blob exceeds GitHub's 100 MiB limit
(largest 1,552,923 bytes). Those historical findings were not rewritten during this commit.

The audited optimization commit passes its checks. The entire 416-commit branch is not certified by this
scoped audit. No push was performed.
