# GLM on multiple GPUs and sockets

The experimental GLM decoder has an opt-in contiguous layer layout. It uses the
existing Q2 and assembled Q23 GGUF files, tensor loaders, CPU experts and GPU
kernels. `gpu_layout` defaults to `legacy`; existing configurations, including
ones with the generic Qwen `layer_split` setting, keep their existing path.

## Configuration

For the native GLM engine in the Python server, add:

```json
{
  "gpu_devices": [0, 1],
  "gpu_layout": "layer",
  "layer_split": "24",
  "layer_pipeline": "mtp",
  "speculative": "mtp",
  "draft_depth": 1,
  "threads": 32,
  "prefill_batch": 256,
  "prefill_experts": "mmq",
  "prefill_expert_cache_mib": 0,
  "decode_cache_mib": 4096,
  "decode_cache_adapt": false,
  "lock_weights": true
}
```

These are additional fields in an existing GLM config containing `exe`, `model`
and `context`. Cache budgets are MiB **per stage**. `threads` is the total worker
budget; each stage also has a submission thread that participates in CPU work.
The coordinator and submission cores are reserved before distributing workers.
A small CPU allowance or insufficient worker budget is rejected.

Optional `layer_workers` and `layer_cache_mib` lists set each stage's worker
count and fixed cache budget. For example, `layer_workers=[16,15]` needs a
`threads` budget of at least 31. Each stage's submission thread is additional
and also participates in CPU work. `layer_pin_coordinator=true` pins the
coordinator to its reserved physical core after the stage threads are created.
The CLI forms are `--layer-workers=16,15`,
`--layer-cache-mib=24576,22528` and `--layer-pin-coordinator`. These controls are
opt-in; measured plans that use them record and validate them as version 2.

`layer_split` is a comma-separated list, or a JSON list, of first-layer indices
on later devices. `24` means layers `[0,24)` on GPU 0 and `[24,45)` on GPU 1.
Device order is preserved. Without calibration, `auto` supplies balanced layer
counts as an initial placement. To use a measured placement, collect at least
two comparable explicit-cut records with `tools/glm_v100_bench.py`, then run:

```sh
python3 tools/glm_layer_tune.py cut24.json cut26.json --model "$MODEL" --output layer-plan.json
```

Set `layer_split=auto` and `layer_plan=layer-plan.json`, or pass
`--layer-split=auto --layer-plan=layer-plan.json`. Selection maximizes measured
single-stream decode throughput, including tail draft work and rejection,
among placements that actually ran. The engine validates the model filename and
size, decoder identity, CPU/GPU types, device order, settings and environment
before applying the cut, then performs normal stage memory preflight. Calibration
records must share a binary, node, allocation, prompt, environment, cache policy
and timing protocol; incomplete, instrumented or unstable records are rejected.
The plan includes SHA-256 provenance for its input records and decoder. Retune
after changing hardware or the build. Placement screening is separate from
coding-quality qualification and does not prove optimality for every workload.

`layer_pipeline=none` runs the stages serially and accepts 2–16 GPUs.
`layer_pipeline=mtp` overlaps a single unconfirmed head input with confirmed
work on the tail. It currently requires exactly two GPUs and draft depth one.
`speculative=mtp, layer_pipeline=none` computes the same draft sequence without
issuing speculative head work, providing a serial reference.

The named CLI options are `--gpu-layout=layer`, `--layer-split=24` and
`--layer-pipeline=mtp`. The server appends them after all 27 legacy model and
settings fields. GEN/STOP/QUIT, PP, T and DONE retain their existing protocol.
The generation wrapper and V100 benchmark accept the same options.

Use [the layer recipe](../tools/glm_v100_layer.env) in a clean environment.
The legacy V100 recipe selects direct weight registration and GPU helpers;
those settings cannot be combined with layer ownership. Native EXL3, external
expert overlays, route bias, tensor/expert GPU splitting, mutable RAM tiers,
requantized dense weights and automatic cache budgets are rejected. Adaptation
within a fixed budget is opt-in, as described below. An assembled Q23 file uses its original mapped expert
bytes; it is not copied into the legacy two-node expert arenas.

## Coexistence with the TR16 runner

The TR16 mixed CUDA/HIP worker's controls and measurements are in
[GLM_TR16_RUNNER_PLAN.md](GLM_TR16_RUNNER_PLAN.md). The release cleanup after
`3e01337d` removes its process-based layer-range prototype, stage protocol and
`tools/glm_stage_run.py`. The native `gpu_layout=layer` path described here
retains its shared model mappings, tail-owned head and independent transport
within one backend. Mixed-worker descriptors and retired stage descriptors
cannot be combined with `gpu_layout=layer`.

The imported decode levers remain opt-in. The fastest TR16 settings include
lossy routing and soft acceptance and have deferred quality checks; their rates
are not V100 measurements. The earlier native-layer V100 report freezes the v9/v10
source and binaries. It does not measure this combined revision. Rebuild and
collect fresh timings before claiming a speed change; measured plans from the
older binary will fail the decoder-identity check.

The [v100s node release-profile report](benchmarks/v100s_runner_sim_20261016/README.md)
measures the integrated `83f3cac9` engine with two V100 PCIe 32 GB GPUs, two
Xeon Gold 6240 sockets and four NUMA domains. It uses the CUDA expert-worker
runner, REAP-224 Q23, affinity 0.14 and MTP depths 2 and 3. Its measured warm
decode rates and mode-specific simulator fits are separate from the native
one-token-ahead pipeline and from quality qualification.

`tools/glm_mixed_run.py` also accepts two CUDA binaries. Supply
`--primary-device=0 --worker-device=1` using ordinals visible inside the
allocation. Both binaries report their visible UUIDs and PCI identities; the
launcher checks that they agree and gives each child one GPU UUID. Explicit
`--primary-cpus` and `--worker-cpus` masks must stay within the launcher's
allowed cpuset. Reserve separate physical cores, including SMT siblings, when
constructing those masks. Worker cache and GPU budgets are configurable and
checked against actual GPU capacity. Existing CUDA/HIP defaults are unchanged.
`--combine=native` is an additional CUDA/CUDA experiment; `strict` remains the
launcher's default. Changing arithmetic requires a fresh quality comparison.

For serving through `tools/glm_mixed_decode.py`, set the same placement in the
server config's `env` block:

| Environment variable | Launcher option |
| --- | --- |
| `STRATA_GLM_MIXED_PRIMARY_DEVICE` | `--primary-device` |
| `STRATA_GLM_MIXED_WORKER_DEVICE` | `--worker-device` |
| `STRATA_GLM_MIXED_PRIMARY_CPUS` | `--primary-cpus` |
| `STRATA_GLM_MIXED_WORKER_CPUS` | `--worker-cpus` |

GPU values are allocation-visible ordinals. CPU values are comma-separated
IDs or ranges, such as `0-15,32-47`; choose them from the actual topology and
allowed cpuset. The adapter forwards these settings to the same launcher
checks. Unset variables retain the existing placement behavior.

On dedicated cards, [VRAM preallocation](GLM_KV_OFFLOAD.md) also allows a
numeric primary `decode_cache_mib` above 12,288 MiB, within `gpu_budget_mib`.
Set `STRATA_GLM_MIXED_WORKER_CACHE_MIB` and
`STRATA_GLM_MIXED_WORKER_GPU_BUDGET_MIB` separately for the worker. Both
processes inherit the preallocation switch. Check each process's admitted
cache and live allocations as well as total GPU memory use: reserving a pool
alone does not populate it with useful expert weights.

The [full UD-Q3_K_XL quality report](benchmarks/v100s_q3_20261017/README.md)
uses the original 288 experts on the v100s node: two V100 PCIe 32 GB GPUs,
two Xeon Gold 6240 sockets and a 160 GiB allocation. The 137.4 GiB shared
model passed full SHA and socket-placement checks. Greedy decoding scored
154/164 HumanEval and 24/24 C++; affinity 0.08/MTP3 scored 153/164 and 21/24,
failing the held-out C++ criterion. Use its greedy profile as the Q3 quality
reference. Both cards reached nearly full VRAM use, with no observed process
swap. These are quality and memory checks; Q3 throughput was not measured.

`tools/glm_v100_optimize.py` runs a frozen, resumable 8K/512-token matrix.
Screening, diagnostics, and final timing eligibility are separate. Final
eligibility requires full warmups and validations, stable streams, no observed
swap, and no detected cycles or generation beyond a real stop token.

Saved CUDA expert-worker profiles can also be matrix cases. Set `profile` to
the experimental configuration JSON and `prompt` to an 8,192-token ID file.
Optional `settings` override `threads`, `prefill_batch` or `decode_cache_mib`;
`env` and `unset` apply explicit environment changes. Keep routing and MTP
acceptance fixed when comparing scheduling changes. For example:

```json
{"cases":[
  {"name":"q3-greedy","profile":"configs/experimental/glm53f-ud-q3-k-xl-v100s-greedy.json","prompt":"docs/fixtures/glm_tr16_20261009/prompts/8k/train-code.ids"},
  {"name":"q3-aff008-mtp3","profile":"configs/experimental/glm53f-ud-q3-k-xl-v100s-aff008-mtp3.json","prompt":"docs/fixtures/glm_tr16_20261009/prompts/8k/train-code.ids"}
]}
```

These cases require a live shared resident owner. Save its `resident.identity`
as JSON and pass `resident.path` directly, preserving the `/proc` descriptor
path. Run from the repository root with the project Python environment active:

```sh
python3 tools/glm_v100_optimize.py --matrix tmp/q3-matrix.json \
  --model "$RESIDENT_MODEL" --resident-identity tmp/resident-identity.json \
  --decoder build/strata-glm-decode --worker build/strata-glm-gpu-worker \
  --decode-target 50 --prefill-target 700 --output tmp/q3-timing
```

The profile's context, precision, cache budgets and speculation mode are used.
One prefill warmup precedes three timed prefills, each checked against the
warmup's logits and route counts outside timing. One full decode warmup
precedes three timed decodes restored from the same target state. Both modes
count generated tokens after the initial prefill-selected token. Cache
learning can continue across repeats: the report distinguishes complete
measurements from repeatable decode timings and quality qualification. Renewed
allocations may use a new verified resident descriptor without changing the
frozen experiment identity. Freeze the candidate using training prompts, then
run two fresh processes on training and held-out prompts before accepting its
measured result. Use `--baseline` and
`--baseline-worker` for cases marked `build=baseline` when comparing binaries.

`STRATA_GLM_IQ512_CLAMP=1` selects the clamped IQ gate/up CPU kernel on
AVX512F/BW/DQ/VL processors, including Skylake and Cascade Lake. It supports
separate NUMA mappings for gate and up rows. The default remains AVX2 for
clamped rows; the new path changes the floating-point reduction order and
requires a fresh quality comparison. `STRATA_NO_IQ512` and the forced ISA cap
disable it. The older unclamped AVX-512 kernel retains its VNNI/VBMI check.

When testing `STRATA_SM70_TABLE=1` on Volta with a clamped model, also set
`STRATA_GLM_FUSED_CLAMP=1`. The fused gate/up kernel then applies the same
asymmetric SwiGLU clamp as the standalone activation. Compare normal and
high-amplitude activations with `native_expert_bench`; its optional last
argument scales the test activations. Neither switch changes the default
kernel selection.

`tools/glm_v100_corpus.py` scores a saved benchmark configuration against full
Q23 reference logits; legacy configurations use the actual target verification
path through `--eval-verify`. Corpus scores alone do not qualify a candidate.
HumanEval generation can be resumed with the same measured engine arguments;
`tools/glm_humaneval_grade.py` grades frozen generations with the existing
isolated runner. Partial or ungraded generations cannot qualify.

## Ownership and scheduling

A single shared `ModelArtifact` owns all mappings and tensor metadata. Stages
keep global layer indices and allocate target state and dense weights only for
their range. The first stage reads embeddings. The tail owns output norm,
output projection, captured target hidden states and the draft block. The
shared C11 rules in `glm_layer.h` validate cuts and tensor ownership.

Each GPU has one persistent submission thread and a CPU pool with an explicit
physical-core list. Linux placement respects the original allowed cpuset,
finds each GPU's PCI NUMA node, and prefers cores from that node's **socket**.
It does not assume that two sockets imply two NUMA domains. Pools have disjoint
cores and the diagnostic hook reports all live pools.

Locking materializes one byte-identical private COW mapping before `mlock`,
including draft tensors when MTP is enabled. The existing lock-limit preflight
runs before touching pages; a separate host budget reserves 4 GiB plus transport buffers and 512 MiB per stage. During the
single materialization pass, Linux requests interleaving across allowed memory
nodes and logs whether the policy applied. Transport stays pinned and portable
across devices. NUMA policy failures are visible in the log. The optional
`layer_memory=socket` / `--layer-memory=socket` policy materializes each stage's
owned tensors on the allowed memory nodes of its CPU socket. It requires locked
weights and still uses the single shared mapping. `interleave` remains the
default; measure both policies on the host before choosing one.

The tail loads draft experts before admitting its fixed target cache. New layer
mode uses GPU draft experts by default; `mtp_experts=cpu` in the server config
or `--mtp-experts=cpu` selects the CPU draft reference. The draft changes only
proposals; confirmed target tokens still select every emitted output. The
legacy restriction on combining resident draft experts and a decode cache is
unchanged. Original IQ tensors use the native expert kernels. `STRATA_GLM_CANON=1`
currently permits canonical GPU cache admission only for Q2_K/Q3_K expert pairs;
it leaves IQ experts on the CPU and can therefore empty the Q2 model's cache.

A boundary carries four FP32 residual streams: 65,536 bytes per GLM token.
Two pinned slots have occupancy and generation checks. A stage completes its
stream before publishing its slot through a future; the consumer releases it
only after its own copy and execution complete. This conservative host fence
requires no peer access or cross-device event support. Prefill chunks advance through stages in diagonal waves: each stage preserves
token order while independent stages process different chunks. A wave drains
all submitted work before publishing or reusing transport slots.

For MTP, the head runs the confirmed input and checkpoints recurrent KDA,
convolution and open indexer state. The tail confirms that input while the head
runs the proposed next token in the other slot. On mismatch, the scheduler
restores the checkpoint and replays the actual token. The tail receives only
confirmed tokens, and its next draft may overlap head replay. Count/EOS exits
drain the head, restore it and discard the unused future residual. Cancellation
and failures drain outstanding futures before restoration; serving resets every
stage before accepting another request. Append-only latent and completed pool
rows are masked by committed position and overwritten during replay.

With a positive fixed cache budget, `decode_cache_adapt=true` enables confirmed
windows of `decode_cache_window` consumed inputs. Both stages drain to the same
frontier before residency changes. The head checkpoint includes route counters;
rejected and discarded future work does not affect admission. Each stage blends
one quarter of its previous route score with 32 times the committed window's
counts, then deterministically selects experts within its admitted byte budget.
It retains unchanged experts, compacts their views inside the slab with a
bounded device scratch copy, transfers new experts, and invalidates graphs and
lookup views before resuming. A policy boundary
replays even a matching head proposal under the new residency policy. The serial
reference uses the same boundaries and policy. DONE timing includes adaptation.

The new layer path disables lossy prefill. Its reference must use the same
prefill panels and arithmetic switches. This setting applies only to the new
layout; the legacy prefill policy is unchanged.

## Checks and measurements

CPU-only tests:

```sh
cmake --build build --target glm_layer_test glm_layer_pipeline_test
ctest --test-dir build -R '^glm_(layer|layer_pipeline)_test$' --output-on-failure
python3 -m unittest serve.test_glm
```

The pipeline test uses the production scheduler with a recurrent fake backend.
It checks acceptance, forced misses, committed cache-policy changes, open
four-token pools, append-only ring positions, cancellation, failure and slot reuse around 3/4, 255/256 and
8191/8192. This is scheduler validation, not GPU arithmetic validation.

Full-model serial parity, requiring space for an additional full-layer GPU
reference as well as the two stages:

```sh
source tools/glm_v100_layer.env
strata-glm-decode "$MODEL" 1,2,3,4,5,6,7,8 16 4096 32 \
  --gpu-layout=layer --gpu-devices=0,1 --layer-split=24 \
  --context=8192 --prefill-batch=8 --gpu-budget-mib=32768 \
  --lock-weights --check-layer-parity
```

The check compares every prefill panel and eight decode steps with an unsplit
reference, including owned recurrent, convolution, latent and indexer buffers.
A pass requires bitwise-equal logits and state. `--check-layer-stages=3` or
`--check-layer-stages=4` emulates those logical ranges on the selected physical
GPUs for this diagnostic; `--gpu-devices=0` permits a single-GPU check. Logical
stages sharing a device use one serialized submission owner. Production device
lists still reject duplicates, and emulation is not a throughput result.

`--check-layer-pipeline` with `--speculative=mtp --draft-depth=1` compares serial
and overlapped execution, emitted token IDs and committed state. Add
`--layer-force-miss-every=1` to perturb every proposal and exercise rejection.
With a fixed cache budget and `--decode-cache-adapt --decode-cache-window=8`, it
also requires identical cache fingerprints at every confirmed boundary and
identical committed route statistics. This diagnostic copies every resident
expert back and checks its payload byte for byte against the GGUF after initial
admission and every compaction; those checks are excluded from timing runs. Append-only future cache rows are outside
the committed state comparison; open pool rows and recurrent state are included.


No throughput target follows from compiling this code or passing scheduler
tests. Qualify Q2 and Q23 separately with a same-day serial control, two full
512-token warmups, three 512-token timed runs, three validation runs, coding
quality and strict-parser checks, and 8K/16K recall. Record RAM speed if known,
GPU and CPU models, source/binary hashes, environment and cache budgets.
Do not count whole-device utilization or aggregate conversation throughput as
single-stream overlap. CUDA sm_70 execution and Compute Sanitizer are separate
from local CUDA sm_80 or HIP compile-only checks.

The [dual-V100 measurement report](../bench/results/2026-10-15-glm-layer-v100/README.md)
records the implemented path, its serial controls, quality gates and remaining
limits. The measured environment is opt-in. In particular, native and canonical
expert arithmetic can produce different greedy sequences, and changing a layer
cut can also change numerical results. Equal tokens are checked between serial
and overlapped execution at the same cut and settings; cross-cut tuning is not
an identical-token speed comparison. The current prefill/cache preparation path
also needs work before claiming a general request-latency improvement over the
legacy decoder.

## Maya source study

Reference: [project-maya at d1871953](https://github.com/mw00/project-maya/tree/d18719534828bea35dc8b347367e66de59ffb2e1).
Its [MIT notice](licenses/project-maya.txt) is retained. The implementation here
is adapted to Strata's existing GGUF decoder; no Maya model-pack conversion is
required.

| Source in that revision | Idea used or considered here |
| --- | --- |
| `include/strata/core/glm_model.hpp`, `src/core/glm_model.cu`: `load_pack_split` | Ordered contiguous ranges, global layer IDs, first-stage embeddings and tail head/MTP; stage memory feasibility before expert residency. |
| `src/core/glm_model.cu`: placement planner | Tail draft cost belongs in the two-stage critical path. Count-balanced `auto` is the fallback; the measured-plan tool selects among completed full-model placements. |
| `src/core/glm_fast_path.cu`: fast speculative path | One head input ahead, head checkpoint/rollback, confirmed-only tail, two alternating boundary slots, draft overlap with replay, drain on exit. |
| `src/core/glm_fast_path.cu`: CPU placement and pinned allocation | Disjoint physical cores, GPU PCI locality, allowed cpus/mems, interleaved model memory; preserve transport lifetime while CPU experts block. |
| `src/core/glm_fast_path.cu`: CPU/PCIe calibration | Calibration must rotate through experts beyond LLC. A hot single-expert microbenchmark cannot select the full-model transfer split. Not yet ported. |
| `src/core/glm_fast_path.cu`, `tools/calibrate_glm.py`: split selection | Compare CPU work with PCIe and GPU work, then choose an integer number of cold experts for each lane. Strata's V100 experiment first measures the CPU pool and a pinned-transfer lower bound on 64 real experts of each format, separately on both sockets. A transfer-only result does not establish GPU compute cost or full-model benefit. |
| `src/core/glm_multi.cu` | Multiple-conversation pipelining measures aggregate throughput. It is not the single-stream objective of this port. |
| `src/core/glm_window.cu` | DFlash2 and batched speculation are separate algorithms, outside this initial one-ahead port. |

Maya's published dual-V100 results use different quantizations, memory budgets
and prompts. They motivate investigating this architecture; they are not
measurements of this implementation. Alternative batched MTP schedules are outside the behavior studied in the
pinned source.
