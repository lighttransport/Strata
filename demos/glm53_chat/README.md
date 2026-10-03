# Simple GLM terminal chat and decode-speed recording

This small ASCII TUI uses **Strata's native GLM decoder directly**. It does not require Pi, an HTTP server or llama-cli. The backend and model are the same ones used for the GLM measurements, so the displayed rate measures this run rather than a different inference engine.

Requirements: built `build-glm/strata-glm-decode`, GLM Q2 shards, Python with `regex` and `jinja2`, and a terminal supporting curses. Asciinema is only needed for recording.

```sh
export GLM_Q2_MODEL=/path/to/models/glm53f/q2/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
python3 tools/glm_chat_tui.py "$GLM_Q2_MODEL"
```

The initial prompt asks for a complete C++17 primality function. After an answer, type another message and press Enter. Press `q` with an empty input to quit, or Ctrl+C to cancel. Follow-ups retain text history but recompute the prompt; conversation-prefix reuse is not implemented. The UI shows the tail of long responses. Use `--save recordings/chat` to retain the final answer and timing JSON.

- **LIVE decode:** visible generated token IDs after the first token divided by elapsed time since the first token. Updates during generation; it excludes the first-token wait.
- **FINAL decode:** the backend's generated-token count minus one, divided by its decode timer. The first token comes from prefill; the remaining tokens are timed decode advances, including a stop token if emitted. This is the authoritative final rate for the demo.
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

This is a short coding example, not the 4096-token benchmark. Initial measurements are 7.33 tok/s median for short chat and 7.26 single / 8.35 GPU MTP on the 4K coding prompt. Rates depend on prompt, routing and machine load; see the [GLM guide](../../docs/README_GLM53_FLASH.md).

## Captured demo

The refreshed local `recordings/glm53-chat-initial.cast` contains a completed live run. The backend measured **7.51 decode tok/s**, generating **133 tokens including stop** from a **53-token prompt**. Prefill/prime took **23.35 seconds** and decode **17.71 seconds**. This is a short-prompt demo, separate from the initial three-trial decode medians above.

The [README animation](../../docs/media/glm53-live-decode.gif) shows eight seconds of actual live decoding at original speed, starting about five seconds after the first live token. Startup and prefill are skipped. Local GIF and WebM exports are in `recordings/glm53-live-decode.gif` and `recordings/glm53-live-decode.webm`; recordings remain outside Git. The source cast is output-only asciicast v2. See the [demo measurement](measurement.json).

## Xeon Gold and dual-V100 recording

The V100 demo uses the full 4K C++ parser prompt and the measured MTP depth-1
profile. It runs three untimed 128-token warmup requests before the visible request.
Stop-token termination is disabled for the fixed 512-token benchmark cap.
Use the [hardware report](../../docs/GLM53_V100.md) for RAM requirements,
build settings and the separate three-trial benchmark medians.

```sh
source tools/glm_v100.env
export STRATA_GLM_RESIDENT_MULTI2=1 STRATA_GLM_BATCHED_RESIDENT=1
asciinema rec --cols 110 --rows 36 --env TERM \
  --title 'GLM5.3Flash Q2 - Xeon Gold and dual V100 live decode' \
  --command 'numactl --interleave=all .venv-glm/bin/python tools/glm_chat_tui.py "$GLM_Q2_MODEL" --decoder build-glm-v100/strata-glm-decode --context 131072 --threads 35 --gpu-devices 0,1 --gpu-budget-mib 32768 --prefill-batch 8192 --prefill-experts f16-batched --prefill-expert-cache-mib auto --decode-prefill-cache --decode-cache-mib extend --decode-graphs --decode-cache-adapt --lock-weights --speculative mtp --warmup-tokens 128 --warmup-repetitions 3 --ignore-stop --tokens 512 --prompt-file docs/fixtures/glm53_cpp_quality/chat_prompt.txt --raw-prompt --demo --save recordings/glm53-v100-chat' \
  recordings/glm53-v100-chat.cast
```

Export an eight-second excerpt after choosing a start time during live decode:

```sh
.venv-glm/bin/pip install pyte pillow
.venv-glm/bin/python tools/glm_chat_render.py \
  recordings/glm53-v100-chat.cast docs/media/glm53-v100-live-decode.gif \
  --start <decode-clip-start-seconds> --duration 8 \
  --title 'Old server. New model. Live GLM decode / original speed'
```

The renderer preserves terminal event timing. Its title is a presentation
caption; the live and final counters come from the captured terminal output.
Raw casts and engine logs stay in the ignored recording directory.

The [V100 README animation](../../docs/media/glm53-v100-live-decode.gif) is an
eight-second live excerpt starting at 270 seconds in the fresh output-only
recording. The request measured **28.17 decode tok/s** (511 timed advances
over 18.1421 seconds) with a 4096-token prompt and a fixed 512-token cap.
Prefill/prime took 14.034 seconds. The separate three-trial MTP benchmark
median is **29.06 tok/s**; the caption keeps these figures distinct. The
generated C++17 parser compiled with `-Wall -Wextra -Werror` and passed
400,532 oracle cases. See the [demo measurement](v100_measurement.json).
