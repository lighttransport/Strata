# GLM-5.3-Flash experimental backend

For the current Q2 setup and C++ output validation, see [the GLM guide](README_GLM53_FLASH.md). Initial Q2 measurements reached 7.33 tok/s short chat, 7.26 tok/s long-prompt single decode and 8.35 tok/s GPU MTP, with three stable trials per mode and identical long-prompt outputs. The initial notes below predate GPU MTP drafting and verification; both are now implemented.

This implements the model-loading and numerical foundations of the GLM port,
plus a separate experimental decoder with GPU batched prefill. The existing `strata` executable remains the Qwen engine;
`strata-glm-decode` is the experimental GLM entry point.

Hardware configurations, measured DDR5 reference results, and conditional prefill/decode estimates are recorded in [GLM53_FLASH_PERFORMANCE.md](GLM53_FLASH_PERFORMANCE.md). The four-DIMM DDR5-5600 planning assumption is 60-75 GB/s; no exact measurement for that configuration was verified.

## Implemented

- Read any member of a split GGUF, including a tensorless first shard. Validate
  split metadata, tensor counts, duplicate names, payload extents and overlaps.
- Describe dense/MoE and KDA/sparse-MLA layers from metadata. Separate main and
  draft blocks and report their exact native byte counts.
- Preserve mixed expert quantization. Generalize CPU expert scratch and row
  scheduling to 4096/2048, add asymmetric SwiGLU clamping, IQ3_S GPU down
  projection, and Q2_K/Q3_K grouped execution.
- CUDA FP32 sigmoid routing with selection-only correction bias, mHC Sinkhorn
  mixing, KDA recurrence and convolution, learned IndexPool compression, radix
  pool selection with a causal unpooled tail, and absorbed NoPE MLA.
- Honor the GGUF conversion of KDA `ssm_a` to `-exp(A_log)`; the original
  Transformers parameter cannot be used interchangeably with this stored tensor.
- Layer-batched prefill (one to eight tokens) and greedy decoding with CPU routed experts and native
  CUDA dense projections. A bounded LRU holds dense weights; MLA keeps only
  512-dimensional latents per token.
- Optional static GPU expert cache with a separate budget per MoE layer. Prompt
  execution admits experts, then generation freezes admission. Resident CUDA
  experts and the shared FFN overlap CPU miss processing; repeated experts in a
  prompt batch share a CPU weight read.
- Snapshot/replay checks restore recurrent matrices, convolution history,
  latent caches, pooled keys, pending pool keys/gates, and the token position.
- GLM4 BPE pre-tokenization and a text frontend using the GGUF chat template.
- Resident text API integration through `serve.server --engine glm`, with the
  GGUF template and stop-token IDs, progress heartbeats, context validation and
  cancellation between prefill batches or decode steps. Requests reset model
  state while retaining mapped weights and caches. Sampling and tool calls are
  rejected because the experimental backend implements greedy text only.

The decoder currently requires normalized sigmoid routing, one expert group,
one shared expert, four mHC streams and NoPE MLA. It checks tensor shapes before
execution. Other GLM configurations need additional backend implementations.

## Build and run

Use the repository's pinned ggml dependency. `STRATA_GGML_DIR` can instead point
to a compatible llama.cpp checkout, but an arbitrary newer checkout may not build.

```sh
cmake -S . -B build-glm -G Ninja \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DSTRATA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-glm -j 4

MODEL=/path/to/models/glm53f/iq3/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf
build-glm/strata-model-inspect "$MODEL"
build-glm/strata-glm-decode "$MODEL" 154822,154824,154828,154841 2 4096 6
```

Arguments after the token IDs are generated-token count, dense-cache MiB and CPU
worker count. The host also participates in CPU expert work. The decoder emits
one greedy token ID per line. It allocates state for the supplied prompt and
generation length; it does not reserve the model's entire maximum context.
The dense cache must fit its largest individual tensor (about 497 MiB here).

For text input, use an environment with the existing Python frontend dependencies
`regex` and `jinja2`:

```sh
python tools/glm_generate.py "$MODEL" "Hello" \
  --decoder build-glm/strata-glm-decode --tokens 16
python tools/glm_generate.py "$MODEL" "Hello" --dry-run
```

Prefill defaults to batches of eight; `--prefill-batch 1` selects sequential
execution. `--expert-cache-mib 512` enables the optional resident expert cache
(zero is the default). Its budget must fit at least one expert per MoE layer.
CPU and GPU experts quantize activations differently, so cache hits can change
logits. This frontend is for text prompts; it
does not implement image/video input or tool execution.

For the existing OpenAI/Anthropic API frontend, create a config such as:

```json
{
  "exe": "/path/to/Strata/build-glm/strata-glm-decode",
  "model": "/path/to/models/glm53f/iq3/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf",
  "model_name": "glm-5.3-flash",
  "context": 4096,
  "dense_cache_mib": 4096,
  "expert_cache_mib": 0,
  "threads": 6,
  "prefill_batch": 8
}
```

Start it with `python3 -m serve.server --engine glm --config glm.json --port 8080`.
Each conversation request prefills its full prompt; prefix reuse and multi-user
batching are not implemented. Cancellation is observed after the current batch
or token completes.

## Validation

```sh
ctest --test-dir build-glm --output-on-failure \
  -R '^(gguf_reader_test|model_test|glm_parity)$'
PYTHONPATH=tools python -m unittest tools.test_glm_tokenizer
PYTHONPATH=tools python3 -m unittest serve.test_glm serve.test_server serve.test_detok
build-glm/glm_quant_parity "$MODEL"
build-glm/native_expert_parity "$MODEL" 3 11 44 45
build-glm/strata-glm-decode "$MODEL" 154822,154824,154828 3 4096 6 --check-replay
build-glm/strata-glm-decode "$MODEL" 154822,154824,154828,154841,9703,220,108714,100461 2 4096 6 --check-prefill --check-replay
build-glm/strata-glm-decode "$MODEL" 154822,154824,154828 2 4096 6 --expert-cache-mib=512 --check-replay
```

Validated against the supplied UD-IQ3_XXS shards:

- 1,412 tensors, 45 main blocks, one draft block; 34 KDA and 11 MLA blocks,
  three leading dense FFNs, and 288 experts with top-8 routing.
- Main expert bytes: 109,867,696,128; main fixed bytes: 7,690,382,584;
  draft bytes: 2,799,972,480. Selected main expert bytes per token: 3,051,880,448.
- CUDA primitive parity at actual GLM dimensions, including eight recurrent
  steps, asymmetric clamps, routing ties, IndexPool boundaries and compressed
  MLA versus a double-precision softmax reference.
- Real quantized-row dequantization matches ggml exactly for the five expert
  formats. Same-Q8 dot checks cover Q2_K, Q3_K, IQ2_S, IQ3_S and IQ4_XS.
  IQ2_S retains the pinned CUDA dot's integer scaled-sum truncation.
- CPU pool outputs match direct native expert outputs; sampled CPU and GPU
  experts agree with dequantized float references within activation-rounding
  tolerances. Full model prefill/decode produces finite vocabulary logits.
- State restore/replay produced identical full-vocabulary logits across a pool
  completion boundary and the subsequent partial tail.
- Eight-token batched prefill matched sequential execution bitwise for both
  complete state and vocabulary logits. Optional 512 MiB expert-cache execution
  exercised 22 resident entries out of 1,680 routed entries (including replay),
  with bitwise replay agreement.
- Repeated resident requests returned identical outputs; oversized requests
  were rejected. A cancelled request stopped and the next request returned the
  expected token. API regressions passed (44 tests, with three existing skips).
- A separate CPU llama.cpp GLM reference at commit
  `1665c0e6f6e291a6d0b28b1f2d5e0e53d1e85ff9` produced the same greedy token
  (220) after token IDs `154822,154824,154828`. Full-vocabulary relative L1
  difference was 0.05923, with maximum absolute logit difference 0.84482.
  Only its architecture-name mapping was adapted from `glm5-next` to the
  artifact's `glm5next`; model equations were unchanged. This single-prefix
  comparison is not an accuracy benchmark or a parity pass. CPU and native CUDA
  dense projections use different activation-quantization contracts, but the
  source of every end-to-end difference has not been isolated.

Use `--dump-logits=path` with `strata-glm-decode` to save the final processed
token's prediction logits as little-endian float32 vocabulary order. With one
generated token, this records the final prompt token's logits. These checks do
not establish end-to-end numerical parity or model-quality benchmarks.

## Historical full-port status before MTP implementation

At this initial stage, captured layer graphs, MTP draft-block execution and speculative verification,
broader independent full-model logit comparisons, and performance
and quality benchmarks were unimplemented. Snapshot support and draft-format
parity provided foundations for MTP; the decoder at that stage did not execute block 45. Later updates implemented GPU and CPU MTP draft execution and verification. Broader independent model-quality evaluation remains outstanding.
The correctness baseline does not substantiate the earlier architectural speed
estimates.

The equations follow the official
[Transformers GLM5-Next implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm5_next)
and [SGLang GLM5-Next implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/glm5_next.py).
GGUF parameter transformations follow the
[llama.cpp GLM converter](https://github.com/timkhronos/llama.cpp/blob/1665c0e6f6e291a6d0b28b1f2d5e0e53d1e85ff9/conversion/glm.py).


## Large GPU prefill

Select `--prefill-batch=4096` or `2048` on `strata-glm-decode`, or `auto` to
choose based on memory. `--context=8192 --gpu-budget-mib=12288` matches the measured
configuration. The cap includes a 1024 MiB runtime reserve; fixed weights, recurrent
state, attention caches, staging slots and reusable workspaces count against it.
GPU prefill and the legacy static expert cache cannot be enabled together.

`tools/glm_generate.py` forwards these batch/budget options. For serving, configure
`prefill_batch` as 4096, 2048, or `"auto"`, and `gpu_budget_mib` as 12288. The default
batch remains eight. Cancellation resets partially processed recurrent and cache
state before the next request. Decode remains the CPU expert path.

`--check-prefill --reference-batch=2048 --prefill-batch=4096` compares full-model
state and logits across chunk widths. GPU checks require normalized L2 error at
most 0.02 and the same first greedy token; legacy checks remain bitwise. CUDA
primitive tests cover causal convolution, routing/grouping, mHC, persistent KDA,
IndexPool boundaries and top-k ties, and gathered MLA attention.

Measured warm 4096-token prefill is **112.38 tok/s median**, with **10.73 GiB**
explicit allocation. See the performance document for timings, budget scope and
reproduction. Longer 16K/32K contexts and broad generation-quality comparisons
remain unbenchmarked; this backend remains experimental.


## Single-token and speculative decode

`--decode-experts=cpu` remains the default. Clamped GLM experts now use the
AVX2 shared dot kernels and can read gate/up/down directly from their GGUF
mappings, avoiding a new packed expert copy for every layer and token. Static
expert-cache configurations retain the packed path. Shared AVX reductions can
change rounding relative to the earlier ggml-only CPU path; do not assume old
and new greedy sequences are identical.

`--decode-experts=gpu` streams only the experts selected by each decode or verify
window. It retains native quantization, uses native Q8_1 GPU expert execution,
and uses the optimized KDA kernel. Fixed weights and prefill allocations remain
resident. This is a separate target arithmetic path from CPU expert execution;
its single-token and batched verifier have been checked for exact agreement.

`--lookup-depth=3` enables prompt-lookup speculation; zero (the default) disables
it, and 1-7 selects the maximum draft depth. The suffix drafter proposes copied
or repeated continuations, then the target evaluates all positions in a window.
Only drafts matching greedy target tokens are accepted. Rejection restores a
145.61 MiB device checkpoint and replays the accepted prefix. Checkpoints store
recurrent/convolution state and pending IndexPool state; append-only latent and
pooled caches use the restored position to exclude rejected future entries.
This implementation prioritizes correctness; replay adds work on rejection.

`--check-verify` compares eight-position verification with sequential logits
bitwise and checks rollback followed by another decode. It needs eight spare
context positions. No learned GLM MTP draft execution is included in this lookup
path. Server config keys `decode_experts` (`"cpu"` or `"gpu"`) and `lookup_depth`
(default 0) forward these choices. The text frontend accepts the same flags.

Reproduce paired single/lookup measurements, with an exact greedy-ID comparison:

```sh
PYTHONPATH=tools python3 tools/glm_decode_bench.py \
  /path/to/models/glm53f/iq3/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf \
  bench/fixtures/glm53_coding_prefix.txt --prompt-tokens 4096 --generated-tokens 33 \
  --decode-experts cpu --threads 16 --lookup-depth 3 --check-verify --output decode.json
```

Decode timing excludes prefill and the first token available from prefill, and
includes checkpoints, verification, replay and output delivery. Lookup remains
opt-in because it was slower on the measured coding continuation. See the
performance document for actual timings and limits.


The initial Q2 configuration measures approximately 7-8 tok/s with single or GPU MTP decode. Prompt lookup did not demonstrate a speed advantage in its separate tests; keep `lookup_depth=0` for this workload. The measured fast CPU decode
configuration keeps fixed weights resident through GPU prefill; the legacy 4096
MiB dense LRU can evict weights and reduce decode speed. GPU prefill/decode with
lookup reserves checkpoint/headroom during memory admission, including `auto`
batch selection. GPU prefill and static expert caches remain mutually exclusive.
