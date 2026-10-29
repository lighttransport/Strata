# DS41 C++ parser task

This is one coding task, not a broad coding benchmark. The prompt asks for a C++17 unsigned decimal parser.
The existing independent oracle checks 400,532 cases, including overflow, unchanged output on failure,
embedded NUL and non-ASCII bytes. Compare exact baseline and candidate under the same prompt and output cap.

Use a Python environment with `tokenizers` and `jinja2`:

```bash
"$DS41_GATE_PYTHON" experiment/ds41f_tr16_decode/cpp_gate/prepare.py
cd experiment/ds41f_tr16_decode
DS41_GATE_PYTHON="$DS41_GATE_PYTHON" bash cpp_gate/run.sh NAME ENV=VALUE
```

Run generation only when other two-node trials have finished. The runner uses best2n.env and the existing
memory/VRAM guards, with 512 output tokens. It refuses to overwrite evidence. The grader extracts exactly
one fenced C++ block and compiles and executes it in a networkless bwrap namespace with resource limits.
It records compile/run failures as `passed: false`; a grader process exit of zero means a record was written,
not that the candidate passed. Inspect the JSON `passed` value. The frozen decoder log remains the evidence.
The compile uses C++17 warnings as errors and fatal UBSan. Python syntax and shell syntax checks passed.

Harness checks: the existing known-good parser passes all 400,532 cases; an always-false parser compiles and
fails case 2. These checks do not qualify a DS41 output. Qualified exact and trim-0.125 outputs passed this task; see [the results](../CANDIDATES.md) and the
per-run grade records. This does not qualify other candidates.

`merge_parity.cpp` separately compares the original and opt-in direct merge paths bit-for-bit across 96
configurations, including mixed local/external routes and synchronous/asynchronous workers and vector tails. It uses small
F32 local experts; full-model teacher-forced parity remains necessary for the merge optimization.
