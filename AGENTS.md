# AGENTS.md

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Contributing a change or report

- Never push without the user's explicit permission. This includes push dry runs.
  Requests to review, commit locally, prepare for push, or run a pre-push audit do not authorize a push.
  Complete the local work and report readiness; wait for explicit push authorization.
- Search the open issues and pull requests first, and add to a thread that already covers your point.
- Open an issue with the form that fits (bug report, feature request or question).
- One change per pull request. Say what it changes and what it leaves alone.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).

## Pre-push audit procedure

Complete this audit before requesting permission to push. An audit does not authorize a push, including a
push dry run. Inspect every outgoing commit, not just HEAD or the final combined diff: a secret or unwanted
file added and later deleted still leaves the machine in Git history.

1. **Identify the exact destination and range.** Record the local branch, remote, destination branch or tag,
   and current remote tip. Refresh remote-tracking refs when needed. For an existing branch, use the actual
   destination ref as the base, rather than assuming `origin/main` or the configured upstream is correct.
   For a new branch, identify and report the intended base. List all outgoing commits and their summaries,
   and determine whether the update is a fast-forward. For a proposed force push, also list the previously
   published commits that would no longer be reachable from the destination ref.

   ```bash
   git status --short
   git branch -vv
   # Replace this ref with the actual destination/base identified above.
   audit_base=origin/glm53f
   audit_range="${audit_base}..HEAD"
   git log --oneline "$audit_range"
   git log -m --format=fuller --stat --patch "$audit_range"
   git diff --check "$audit_range"
   ```

2. **Check credentials and sensitive information throughout the range.** Inspect each commit's patch,
   message, added blobs, source comments, configs, documentation and logs. Check for API keys, access tokens,
   passwords, private keys, credential files, authenticated URLs, private endpoints, customer/proprietary
   model names and personal machine paths. Do not exclude Markdown, text files or benchmark records.
   Use installed secret scanners such as `gitleaks` or `trufflehog` over the same commit range, with redacted
   output and offline verification where supported. Record unavailable scanners and use a local pattern
   scan plus manual review instead; a pattern scan is not proof that no secrets exist. Report findings by
   file/commit and type without printing the secret. Use relative paths or placeholders for machine-specific
   reproduction paths; do not silently alter frozen raw evidence to sanitize it.

3. **Check build outputs and scratch files.** Reject unintended build trees, executables, object files,
   libraries, CMake/Ninja state, virtual environments, `__pycache__`, package caches, local release packages,
   and agent scratch directories. Inspect `git status --short` and the files in every outgoing commit.
   `.gitignore` does not protect files already tracked or added with `-f`.

4. **Check binary data and fixture scope.** Inspect binary additions and the largest outgoing blobs.
   Reject model weights (`.gguf`, `.safetensors`, checkpoints), downloaded datasets, archives, large captures
   and raw logit dumps unless their inclusion is explicitly authorized. Existing curated documentation and
   small test fixtures are allowed when deliberately scoped; measure their size and explain additions.
   Keep reproducible performance summaries, source/config hashes and necessary compact evidence. Preserve
   raw frozen outputs; report their whitespace warnings separately from source defects. Use an agreed
   external artifact destination or Git LFS for approved large assets.

5. **Review behavior and validation.** Confirm the commits match the requested scope and inspect the complete
   diff for correctness, bounds, resource lifetime and compatibility. Build every applicable backend touched
   and run relevant tests. Verify opt-in behavior and unchanged default output as required above. For measured
   changes, record hardware, RAM speed, build/config hashes, baseline, timing protocol and quality checks.
   Distinguish compile-only checks from GPU execution, single timings from repeated results, and failed or
   interrupted runs from qualified measurements. State skipped checks and unresolved failures explicitly.

6. **Resolve findings and repeat the audit.** A suspected credential leak blocks push preparation until
   resolved; notify the user, treat a confirmed credential as compromised and arrange rotation without
   exposing it. Removing a secret or artifact in a later commit is insufficient: remove it from the outgoing
   history, then repeat every audit check. Do not rewrite published history without explicit permission.
   Resolve unintended assets with the user rather than guessing where to publish them.

7. **Report readiness and obtain push permission.** Summarize the remote/destination, exact commit count and
   one-line summaries, fast-forward or force status, checks passed, skipped checks and outstanding issues.
   If matching explicit authorization for this destination and audited commit set is absent, ask before
   running any push command. Preparing, committing, credentials being available, or approval for an unrelated
   earlier push does not grant permission. Changes to the destination, audited commit set or force mode
   require matching authorization. After an authorized force push, report the previous remote tip for recovery;
   use `--force-with-lease` rather than `--force`. Do not delete remote branches or publish releases as part
   of an ordinary push unless explicitly requested.
