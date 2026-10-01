# Simple GLM terminal chat and decode-speed recording

This small ASCII TUI uses **Strata's native GLM decoder directly**. It does not require Pi, an HTTP server or llama-cli. The backend and model are the same ones used for the GLM measurements, so the displayed rate measures this run rather than a different inference engine.

Requirements: built `build-glm/strata-glm-decode`, GLM Q2 shards, Python with `regex` and `jinja2`, and a terminal supporting curses. Asciinema is only needed for recording.

```sh
export GLM_Q2_MODEL=/path/to/models/glm53f/q2/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
python3 tools/glm_chat_tui.py "$GLM_Q2_MODEL"
```

The initial prompt asks for a complete C++17 primality function. After an answer, type another message and press Enter. Press `q` with an empty input to quit, or Ctrl+C to cancel. Follow-ups retain text history but recompute the prompt; conversation-prefix reuse is not implemented. The UI shows the tail of long responses. Use `--save recordings/chat` to retain the final answer and timing JSON.

- **LIVE decode:** visible generated token IDs after the first token divided by elapsed time since the first token. Updates during generation; it excludes the first-token wait.
- **FINAL decode:** the backend's generated-token count divided by its decode timer. This count includes the stop token, if emitted. This is the authoritative final rate for the demo.
- **Prompt time:** separate prefill time. Optional MTP priming is included in this phase by the resident decoder protocol.
- **Total time:** includes prompt processing and decode, excluding initial model startup.

The default is single decode, 15 workers with automatic affinity, context 8192, 2048-token GPU prefill, a 12288 MiB GPU budget and no CPU prepacking. Physical display-memory guards remain enabled. `--speculative mtp` selects GPU MTP depth 1 when enough free VRAM is available. `--prefill-batch 8` selects the legacy short-prompt path; measure it separately if comparing performance. `--tokens 512` raises the default 256-token answer cap.

## Record and replay

The demo option exits automatically four seconds after the first answer. The following produces a local recording, output text and timing JSON. Choose a new filename for each run; asciinema refuses to overwrite an existing recording by default.

```sh
asciinema rec --cols 110 --rows 36 --env TERM \
  --title 'GLM5.3Flash Q2 - Strata chat - live decode speed' \
  --command 'python3 tools/glm_chat_tui.py "$GLM_Q2_MODEL" --demo --save recordings/glm53-chat' \
  recordings/glm53-chat.cast

asciinema play recordings/glm53-chat.cast
```

`recordings/` is ignored by git. The recording contains actual terminal output and actual elapsed time; nothing is uploaded. You can use `asciinema play --speed 2` for faster viewing, but playback speed is not inference speed. Runtime engine logs remain under a temporary directory rather than appearing in the UI.

This is a short coding example, not the earlier 4096-token benchmark. Rates depend on prompt, routing and machine load. See the [GLM guide](../../docs/README_GLM53_FLASH.md) for the accepted 7.54 tok/s benchmark and the separate C++ validation at approximately 2.1 tok/s.

## Captured demo

The local `recordings/glm53-chat.cast` contains a completed live run (about 68 seconds, including startup and the final display pause). The backend measured **5.27 decode tok/s**, generating **133 tokens including stop** from a **53-token prompt**. Prefill took **26.83 seconds** and decode **25.25 seconds**. The generated primality function compiled with C++17 warnings treated as errors and passed checks for every input from 0 through 9999 plus two boundary inputs near UINT32_MAX. These are short-prompt demo results, not a replacement for the 4096-token measurements. The recording was validated as output-only asciicast v2 and passed a credential scan.
