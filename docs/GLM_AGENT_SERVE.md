# GLM-5.3-Flash as a coding-agent backend (2026-10-18)

The GLM server (`python3 -m serve.server --engine glm --config ...`) can serve Claude Code (`/v1/messages`),
OpenAI-compatible agents (`/v1/chat/completions` with tools) and Codex (`/v1/responses`). Four opt-in pieces make
that work; without them the server behaves as before (greedy text only, no tools, every request read from scratch).

| Piece | Config key | What it does |
|---|---|---|
| Tool calls | none (always on with a GLM template) | GLM's `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>` is parsed, streamed and typed by the tool's schema; parallel calls, forced calls (`tool_choice`), calls written inside the reasoning |
| Prefix reuse | `"prefix_cache": {"checkpoints": K}` | a request continues from a saved state of an earlier prompt instead of reading it again |
| Sampling | `"glm_sampling": "honor"` | temperature, top_p, top_k, min_p and seed are sampled by the engine, with MTP drafts still exact |
| Several conversations | `"parallel": N` with `"prefix_cache"` | N slots, each with its own prompt and checkpoints |

Profile: [`configs/experimental/glm53f-q23fp8-tr16-9070xt-agent.json`](../configs/experimental/glm53f-q23fp8-tr16-9070xt-agent.json)
(2 slots of 32K, 2 checkpoints each, sampling honored, temperature 1.0 / top_p 0.95 as the defaults for requests
that set none, `fit_max_tokens`). Set `model` (the single file `GLM-5.3-Flash-Q23-FP8-assembled.gguf`, or the shards
plus `expert_pack`) and `STRATA_GLM_EXPERT_PRIOR` to your paths, and run the HIP build with ROCm's libraries on
`LD_LIBRARY_PATH`.

## Tool calls

- The template's dialect is detected from its source (`<arg_key>`), so the same code serves Qwen-style templates
  unchanged. GLM's call syntax is rewritten on the fly into the form the existing output parser reads, so every rule
  of that parser applies: streamed arguments, calls inside the reasoning held until the reasoning ends, `<tool_call>`
  quoted in prose or code stays text, a call to an undeclared tool name stays text.
- String arguments are passed through raw; other types are JSON-decoded by the tool's schema.
- Tool-call ids are passed to the template (OpenAI `tool_call_id`, Anthropic `tool_use_id`), which sorts a block of
  results into the order of the calls.
- GLM's argument tags are special tokens: one quoted inside a message is encoded as text.
- `tool_choice` "required" / a named tool writes `<tool_call>` / `<tool_call>NAME` as the start of the reply.

## Reasoning effort and thinking

GLM's template knows `Reasoning Effort: Low`, `High` and `Max` (its default) and always opens `<think>`. The
server maps a client's effort: none to low becomes Low, medium becomes High, high / xhigh / max or none given
become Max. Thinking off (`enable_thinking: false`, Anthropic `thinking.disabled`) closes the block at once
(`<think></think>`), the template's own form for a past turn without reasoning.

- `"reasoning_effort_lock": "low" | "high" | "max"` renders every request at one level. The effort line is the first
  line of the prompt, so a client that changes effort between turns otherwise forces a full re-read.
- `"glm_clear_thinking": true` drops the reasoning of turns before the last user message from the prompt. It saves
  context but changes the prompt at every new user turn, so prefix reuse stops at that point.

## Prefix reuse

GLM-5.3-Flash has recurrent (KDA) layers, so a prompt cannot be cut back to an arbitrary point: a request can only
continue from a saved state. `--prefix-cache=K` keeps, per slot:

- the token ids the slot's attention rows were written from (the last prompt), and
- up to K checkpoints in pinned host memory: the KDA states, the MLA index rings and the draft block (146 MiB each
  for this model), saved at the end of each prompt and at a pinned prefix (`strata_prefix`).

A new prompt resumes from the latest checkpoint inside its common prefix with the slot's tokens; MLA rows below it
are still valid in place (they are append-only). Only the rest is read. Saving takes about 12 ms, restoring about
280 ms (the request setup that a full read also pays).

An agent turn's prompt is the previous prompt plus the reply plus the tool results, so the end-of-prompt checkpoint
is always inside it, whatever the client does to the reply when it sends it back. The generated tokens themselves
are read again (a few hundred tokens) because clients re-render them (JSON reformatted, whitespace stripped).

Exactness: a resumed request gives bit-identical logits and tokens to a fresh read whose prompt is split into chunks
at the same point (`GEN ... split=N`, a debug key). Against a fresh read with the default chunking the first-token
KL was 0.04 (the prefill chunk boundary changes the floating-point summation order), with the same top-1 token.

Measured on TR16 (RX 9070 XT, FP8-derived Q23 single file, a Claude Code session of ~16K-token prompts): the first
request reads the whole prompt (~120 s); later turns reuse 15.9K tokens and take 15-28 s. `cached_tokens` /
`cache_read_input_tokens` in the response report the reused tokens.

Limits: one GPU, GGUF weights, device KV, no remote TP, legacy GPU layout.

## Sampling

`"glm_sampling"`:
- `"reject"` (default): only greedy settings are accepted (temperature 0, top_p 1, ...); others are a 400.
- `"ignore"`: any settings are accepted and the reply is greedy.
- `"honor"`: temperature, top_p, top_k (at most 64), min_p and seed are sampled; penalties, typical_p and mirostat are
  still a 400. The config's `"sampling"` block gives defaults.

The sampler is the device kernel's chain on the host (`include/strata/spec/host_sampler.hpp`: top_k, top_p, min_p,
temperature, a Philox draw keyed by seed and the token's absolute position; `glm_sampler_test` checks it against the
kernel's reference). With MTP or lookup drafts, a draft token is accepted iff it equals the target's draw at that
position, which is exact sampling for point-mass drafts. Measured: the same seed gives the same reply, and MTP on and
off give the same tokens at a fixed seed. A request without a seed gets a random one.

## Several conversations

`"parallel": N` (2..8) with `"prefix_cache"` gives every slot its own prompt and checkpoints; the server sends a
conversation back to the slot that holds the longest prefix of it. Memory per slot: the MLA latent cache at 26.1 KB
per token of context (3.4 GB at 128K, 0.85 GB at 32K) on the GPU, 146 MiB of KDA state, and K x 146 MiB of pinned
host memory for its checkpoints. Measured with two slots: two conversations alternating each resume from their own
slot (cached 831 / 827 / 849 / 841 tokens on their second and third turns), and run together in the batch.

## Clients

Claude Code:
```sh
ANTHROPIC_BASE_URL=http://HOST:PORT ANTHROPIC_API_KEY=<the server's api_key> \
CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192 ANTHROPIC_SMALL_FAST_MODEL=<model name> ANTHROPIC_DEFAULT_HAIKU_MODEL=<model name> \
  claude --model <model name> --permission-mode acceptEdits
```
Claude Code's default 32K output request does not fit a 32K context (`fit_max_tokens` shortens it). Its auto
permission mode asks the same model to classify every action with a 60 s timeout, which a local model does not meet:
use another permission mode.

OpenAI-compatible agents: base URL `http://HOST:PORT/v1`, the model name from `/v1/models`, tools as usual.

Codex (`~/.codex/config.toml`):
```toml
model = "<model name>"
model_provider = "strata"
[model_providers.strata]
name = "Strata GLM"
base_url = "http://HOST:PORT/v1"
wire_api = "responses"
env_key = "STRATA_KEY"
```

## End-to-end checks (TR16, 2026-10-18)

With the profile above on the RX 9070 XT:
- **Claude Code** (`claude -p`, `--permission-mode acceptEdits`): wrote a file with `Write`, called `Bash`, answered;
  31,855 of the run's prompt tokens were reused from the cache.
- **Codex** (`codex exec`, Responses API): created `add.py`, ran it, reported `5`.
- **OpenAI chat completions**, streamed: two parallel `get_weather` calls with schema-typed arguments, `finish_reason`
  `tool_calls`.

## Tests

- `serve/test_glm_tools.py`: the real GLM template (`serve/fixtures/glm53f_chat_template.jinja`) rendered and parsed
  back, streamed one character at a time and whole, forced calls, id ordering, effort and thinking, sampling modes,
  prefix keys.
- `tests/host/glm_prefix_cache_test.cpp`, `tests/host/glm_sampler_test.cpp` (ctest label `glm`).
