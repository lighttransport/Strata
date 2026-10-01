# Record GLM5.3Flash in Pi's terminal UI

This demo records **live code generation** from the local Strata GLM Q2 backend inside the Pi coding agent's TUI, using asciinema's terminal-text `.cast` format. It is a text-only demonstration: Pi's tools are disabled because the GLM API currently rejects tool-call requests. It does not demonstrate autonomous editing or test execution by Pi.

Prerequisites: a built `build-glm/strata-glm-decode`, all four Q2 model shards, enough resident RAM, NVIDIA/CUDA, Python dependencies `regex` and `jinja2`, and `pi` plus `asciinema` on PATH. Setup was checked with Pi 0.85.1 and asciinema 2.4.0. No install or changes to your normal Pi profile are needed.

From the repository root:

```sh
export GLM_Q2_MODEL=/path/to/models/glm53f/q2/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
python3 tools/glm_pi_record.py "$GLM_Q2_MODEL" --prepare-only
python3 tools/glm_pi_record.py "$GLM_Q2_MODEL"
```

The first command validates prerequisites and writes isolated settings without starting the model. The second starts a loopback-only GLM server on port 18095, waits for readiness, then records Pi at 110 columns by 36 rows. After the answer finishes, **press Ctrl+D** to leave Pi and end the recording. The launcher stops the server it created. Do not run alongside another GPU benchmark; the same 12288 MiB budget and physical display-headroom guards apply.

```sh
asciinema play recordings/glm53-pi.cast
asciinema play --speed 2 recordings/glm53-pi.cast
```

The recording retains actual generation timing. Faster playback does not change the measured model speed. Nothing is uploaded automatically. `recordings/` is ignored by git; review a cast before publishing it yourself. Use `--output /path/to/demo.cast` for another destination, `--port 18096` for another port, and `--decoder /path/to/strata-glm-decode` for another build.

The launcher creates a fresh `PI_CODING_AGENT_DIR` under `/tmp`, disables startup network operations, telemetry, extensions, skills, context-file discovery and session saving. Its provider uses a dummy local key, greedy temperature 0, low model reasoning effort, and a 512-token cap. The demo runs in an isolated temporary directory with the supplied [C++ prompt](prompt.txt), so normal Pi sessions and repository context do not appear. Backend configuration/logs stay in the printed runtime directory and are not part of the cast.

The server defaults to single decode with CPU main experts, 15 workers with automatic physical-core affinity, 2048-token GPU prefill, context 8192 and no CPU prepacking. This short TUI prompt is **not** the earlier 4096-token quality benchmark. The UI may take time before the first token while prefill runs. Optional `--speculative mtp` enables GPU drafting at depth 1, but requires more free VRAM; desktop pressure caused an initial MTP attempt to be rejected by the display-headroom guard. This recording is a demonstration, not a new performance claim.

For repeated measured rates (7.33 tok/s short chat; 7.26 single / 8.35 GPU MTP on the 4K coding prompt), generated-code tests and hardware scenarios, see the [GLM guide](../../docs/README_GLM53_FLASH.md). Full edit-and-test agent operation requires implementing and validating GLM tool-call parsing first.

## Captured run

A live recording was produced locally at `recordings/glm53-pi.cast`: approximately 121 seconds at 110x36. Pi sent 195 input tokens and received 284 output tokens, finishing with a normal stop. The backend reported approximately 4.6 decode tok/s on this short prompt and 9055.21 MiB peak owned GPU allocation. These results describe this demo only. The cast was checked as valid asciicast v2 with output-only events, the requested function and success return visible, no GLM error, and no private home/model paths or credential patterns. It is deliberately excluded from git.
