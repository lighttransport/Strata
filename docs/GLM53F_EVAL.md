# Coding-quality evaluation suite for GLM-5.3-Flash configurations

HumanEval no longer decides whether a configuration is good enough. It has 164 short Python functions, is close to
saturated and likely in the training data, and moves by 2-3 tasks between runs with identical settings (152 vs 150 on
the REAP-224 profile). This suite replaces it with recent benchmarks in C++ and Python, closer to real coding work. It
judges every candidate against a reference run on the same tasks, with paired statistics, instead of against an
absolute floor.

## What a candidate is compared with

- **Quality reference**: the unpruned Q23 model with strict routing,
  `configs/experimental/glm53f-q23-tr16-calibration.json` (affinity 0, exact MTP acceptance).
- **Candidates**: any profile with a lossy switch (routing affinity, soft MTP acceptance, F16 prefill paths), a
  pruned model (REAP-224, REAP-50) or new expert weights.

Both runs use the same tasks, greedy decoding, low reasoning effort and the same token caps.

## Tiers

| tier | when | contents | time on TR16 |
|---|---|---|---|
| 0 | every candidate | corpus gate `tools/glm_tr16_quality.py --quality-gate tr16-35` (perplexity +6%, KL +0.08, top-1 3 pp); the C++ parser oracle (`docs/fixtures/glm53_cpp_quality/`) on 3 runs; 8K stream health (`tools/glm_mixed_bench.py`) | ~15 min |
| 1 | before a profile is published | EffiBench-X subset (100 problems, Python and C++) and Aider polyglot (26 C++ and 34 Python exercises, two attempts) | not measured yet; estimated 5-6 h |
| 2 | release | OmniCode sample: 40 Python and 40 C++ bug-fix instances, 20 + 20 test-generation instances, solved by SWE-agent | not measured yet; estimated 1-2 days |

HumanEval stays as a 30-task smoke test (`tools/glm_q2_humaneval.py --limit 30`), not a score.

## Benchmarks

| benchmark | tasks | how it is judged | adapter |
|---|---|---|---|
| [EffiBench-X](https://github.com/EffiBench/EffiBench-X) (NeurIPS 2025, Apache-2.0) | 623 problems from LeetCode, AtCoder, CodeChef, Aizu and Codeforces, released from November 2023; frozen subset of 100 (54 function-style, 46 stdin/stdout), [effibench_x_subset.json](fixtures/glm_eval/effibench_x_subset.json) | 100 generated tests per problem in a container; runtime and peak memory against the dataset's reference solution in the same language | `tools/glm_eval_effibench.py` |
| [Aider polyglot](https://github.com/Aider-AI/polyglot-benchmark) | the hardest 225 Exercism exercises; we use the 26 C++ and 34 Python ones | the model edits the stub files through Aider (diff edit format); the exercise's unit tests decide; on failure the test output is shown and the model gets a second attempt | `tools/glm_eval_aider.py` |
| [OmniCode](https://github.com/seal-research/OmniCode) | bug fixing and test generation on real pull requests; Python (Django, SymPy, Scrapy and 11 more repositories) and C++ (nlohmann/json, fmt, Catch2, simdjson); frozen sample [omnicode_sample.json](fixtures/glm_eval/omnicode_sample.json) | the agent's patch is applied in the repository container and the instance's tests decide (SWE-bench and Multi-SWE-bench harnesses) | `tools/glm_eval_omnicode.py` |

Not used: [CodeGolf Bench](https://arxiv.org/abs/2605.30394). It submits solutions to the online code.golf service
and measures program length, not correctness regressions; the suite stays offline.

The samples are drawn per stratum (EffiBench-X: source and difficulty; OmniCode: repository) in a fixed hash order,
so they are reproducible and are not chosen by looking at results. Only IDs are committed; the problem data comes from
the benchmarks' own releases.

## Decision rule

`tools/glm_eval_compare.py reference/results.json candidate/results.json --margin 0.03` pairs the tasks and reports:

- tasks only the reference solved and tasks only the candidate solved;
- an exact McNemar test on those two counts;
- a bootstrap 95% interval for the difference in pass rate.

A candidate fails a benchmark when the McNemar test shows a significant loss (p < 0.05, more losses than wins), or
when the lower end of the interval is below -3 points. A tier passes when every benchmark in it passes. With 60-200
tasks per benchmark the interval is wide; the rule catches real regressions and does not certify equality.

## Running it

Everything that executes model-written code runs in containers on rootless podman; the adapters find its
Docker-compatible socket at `/run/user/$UID/podman/podman.sock` (`systemctl --user start podman.socket`). The harness
checkouts and run data live outside the repository (on TR16: `/mnt/nvme02/work/eval-harness`). A run starts
`serve.server --engine glm --config <profile>` itself, or uses `--base-url` for a server that is already running.
The two-GPU configuration is not served by `serve.server` yet, so it cannot be evaluated this way.

```sh
H=/mnt/nvme02/work/eval-harness; PY=$H/.venv/bin/python   # venv with EffiBench-X requirements + pyarrow
# EffiBench-X: dataset (2.2 GB) into $H/EffiBench-X/data/hf, then
$PY tools/glm_eval_effibench.py prepare  --harness $H/EffiBench-X --out $H/runs/effibench/problems
$PY tools/glm_eval_effibench.py generate --harness $H/EffiBench-X --problems $H/runs/effibench/problems \
    --config configs/experimental/glm53f-q23-tr16-calibration.json --run $H/runs/effibench/q23-strict
# stop every model server first: runtimes are measured on the same CPU
$PY tools/glm_eval_effibench.py sandbox  --harness $H/EffiBench-X &
$PY tools/glm_eval_effibench.py evaluate --harness $H/EffiBench-X --problems $H/runs/effibench/problems \
    --canonical $H/runs/effibench/canonical                  # once: the reference solutions
$PY tools/glm_eval_effibench.py evaluate --harness $H/EffiBench-X --problems $H/runs/effibench/problems \
    --run $H/runs/effibench/q23-strict
$PY tools/glm_eval_effibench.py report --problems $H/runs/effibench/problems --run $H/runs/effibench/q23-strict \
    --canonical $H/runs/effibench/canonical --profile q23-strict

# Aider polyglot
python3 tools/glm_eval_aider.py build --aider $H/aider
python3 tools/glm_eval_aider.py run --aider $H/aider --exercises $H/polyglot-benchmark \
    --config configs/experimental/glm53f-q23-tr16-calibration.json --run $H/runs/aider/q23-strict
python3 tools/glm_eval_aider.py report --run $H/runs/aider/q23-strict --profile q23-strict

# compare
python3 tools/glm_eval_compare.py $H/runs/effibench/q23-strict/results.json $H/runs/effibench/<candidate>/results.json --margin 0.03
```

OmniCode needs a Python 3.13 environment with its requirements and its four submodules (SWE-agent, SWE-ReX,
SWE-bench, multi-swe-bench; the `.gitmodules` URLs are SSH, clone them over HTTPS), the dataset
(`hf download seal-research/OmniCode --repo-type dataset --local-dir data`) and the prebuilt
`omnicodeorg/omnicode:*` images, which `evaluate` removes after each repository. Check the harness with the
reference patches first (`evaluate --gold`), then run `agent`, `evaluate` and `report` per language and task.

## Limits

- The GLM server decodes greedily only, so every benchmark is pass@1 with one sample; run-to-run variation comes from
  numerics (MTP acceptance, tier state), not sampling.
- EffiBench-X runtimes are measured with the sandbox running 8 workers in parallel. They are comparable only within
  one evaluation session on one machine. The reference solutions are re-measured in the same session.
- Aider's diff edit format measures edit-format discipline as well as coding. A malformed edit counts as a failure,
  as it would for a user.
- OmniCode tasks are agentic and long (tens of model calls per instance at 30-45 tok/s), so tier 2 uses a sample
  and runs only for releases.
