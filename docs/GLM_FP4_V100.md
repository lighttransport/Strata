# Native GLM FP4 on two V100 GPUs

The experimental native loader accepts the full GLM-5.3-Flash ModelOpt mixed
precision checkpoint directory. It follows `model.safetensors.index.json`,
including indexed shards in subdirectories. The checkpoint stays in its
published weight formats: NVFP4 E2M1 codes with group-16 E4M3 scales and an
FP32 global scale, lossless CSF32 scale encoding, MXFP8 with E8M0 scales,
block FP8 with FP32 128x128 scales, and protected BF16/FP32 tensors.

`--fp4-kernel=reference` is the default for this format. The opt-in
`--fp4-kernel=sm70` uses Volta FP16 tensor cores and FP32 accumulation.
By default, both use W4A16 arithmetic: inputs and reconstructed matrix operands are FP16;
this does not emulate ModelOpt's FP4 activation quantizer. Original weight
codes and scales retain their compressed checkpoint values. GGUF, EXL3 and
FP8-reference invocations retain their existing selection and defaults.

## Build and inspect

Use CUDA 12.x; CUDA 13 no longer compiles SM70. The GLM engine also needs its
usual GGML dependency. An existing dependency checkout can be supplied with
`-DSTRATA_GGML_DIR=...` to avoid downloading it again.

```sh
cmake -S . -B build-glm-v100 -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_BUILD_TESTS=ON
cmake --build build-glm-v100 --target strata-glm-decode \
  strata-glm-fp4-check glm_fp4_cuda_test glm_fp4_cpu_test glm_fp4_codec_test -j
build-glm-v100/strata-glm-fp4-check models/glm53f-fp4
ctest --test-dir build-glm-v100 -R '^glm_fp4_' --output-on-failure
python3 -m pip install -r requirements.txt -r requirements-exl3.txt
```

The host checker validates bindings and reconstructs a selected expert matrix.
It does not run inference or certify all checkpoint payload values. To check
and time a real GPU projection:

```sh
build-glm-v100/glm_fp4_cuda_test models/glm53f-fp4 \
  model.language_model.layers.3.mlp.experts.0.gate_proj.weight --bench
```

The CUDA test checks exact scale/reconstruction bytes, FP64 dot-product
references, outlier inputs, partial output tiles, and fixed first-row
arithmetic across verification widths 1 through 8. The optional microbenchmark
uses resident operands and CUDA graphs; its rate excludes loading and CPU work.

## Run

Keep the actual login alias and model path in ignored local configuration.
For the standalone session launcher, request two GPUs and `181G` host memory:

```sh
python3 slurm-gpu/gpu_session.py start --gpus 2 --memory 181G --cpus 72
```

The model inspected during implementation has 177421592008 indexed payload
bytes (about 165.2 GiB); whole indexed files occupy 177463841024 bytes (about
165.3 GiB). The host admission check includes draft and vision payloads and
reserves another 6 GiB with the default CPU scale cache. It bounds this against physical RAM, visible cgroup
limits and `SLURM_MEM_PER_NODE`. `181G` is an allocation request, not proof of
residency; check RSS, swap, page faults and the job memory controller during a
run. Locking needs a sufficient existing memlock allowance.

If a shared filesystem does not retain its mapped pages, opt in to
`STRATA_GLM_FP4_RAM_COPY=1`. This reads each indexed shard into one shared RAM
mapping, releases source cache pages in 4 MiB chunks, and protects the finished
mapping read-only. Tensor reads and weight locking then use that same copy.
It preserves the original compressed bytes and uses the same host admission
check. The normal file mapping stays the default. On a NUMA machine,
`numactl --interleave=all` can distribute these pages across memory controllers;
record that placement with the throughput result.

For repeated experiments, `tools/glm_ram_cache.py` keeps one sealed, locked
RAM copy across child processes. It hashes every source shard and reads back
the sealed copy to check its SHA256. The child opens its own read-only file
descriptors. The cache directory contains metadata and links to the owner's
descriptors; it contains no second weight set. The owner removes that directory
and releases the RAM after its child exits. Choose a fresh cache directory and
report path. This tool requires Linux and Python 3.9 or newer with `os.memfd_create`.
`--direct-read` uses aligned `O_DIRECT` reads for complete pages and buffered
reads for each shard's final partial page. It avoids populating a second source
page cache; the filesystem must support direct reads. Source and sealed-copy
SHA256 checks still cover every byte.

`STRATA_GLM_FP4_RAM_FDS=1` is set by the cache tool. The loader accepts only
write-sealed tmpfs files owned by the current user in this mode. Its normal
path and symlink checks remain in force for file-backed checkpoints.

```sh
python3 tools/glm_generate.py models/glm53f-fp4 'Explain NUMA placement.' \
  --decoder build-glm-v100/strata-glm-decode --fp4-kernel sm70 \
  --gpu-devices 0,1 --context 12288 --prefill-batch 8192 --lock-weights
python3 -m serve.server --engine glm \
  --config configs/experimental/glm53f-fp4-dual-v100-sm70.json
```

For the optional RAM copy, prefix the command with
`STRATA_GLM_FP4_RAM_COPY=1 numactl --interleave=all` when `numactl` is available.

Copy the supplied profile to ignored local configuration to select the local
model and executable. The profile requests native MTP at depth 3 with exact
greedy acceptance; its throughput and quality must be checked together.
Ordinary greedy decoding uses `speculative: "none"`. Text only, one request
at a time, one or two CUDA GPUs, and context capacity 12288 are the target
configuration. Vision, GGUF sidecars/caches, remote workers, parallel requests
and HIP execution are outside this native path.

One process owns the shared read-only host mapping and persistent GPU workers.
Fixed matrices stay compressed on the primary GPU. Each GPU owns a disjoint
expert cache; nonresident decode experts run W4A16 on the CPU pool using the
original mapped codes. A bounded 2 GiB LRU holds expanded CPU scale planes and
small tables of FP16-rounded weight magnitudes. Protected BF16 matrices cache
their FP16 multiplication operands in the same number of bytes; their original
BF16 values remain in the host mapping. MXFP8 cache packing permutes the original
codes into the quadpair layout and duplicates scales at group width 16. Block
FP8 keeps its row-major layout, which measured faster for the shared-expert shape.
GPU uploads expand CSF32 directly into the prepacked layout. Larger prefill
products reconstruct one projection in bounded FP16 scratch for cuBLAS.
Repeated requests shrink the decode cache before restoring prefill scratch.
No complete FP16 or FP32 expert copy is made on the host.

The following options apply only to the native experiment:

| Setting | Behavior |
| --- | --- |
| `STRATA_GLM_FP4_CPU_SCALE_CACHE_MIB` | CPU scale-cache limit, 128–10240 MiB; default 2048. Host admission reserves at least this limit plus 4 GiB. |
| `STRATA_GLM_FP4_CPU_CSF32=1` | Lossless CPU row layout for CSF32 scales: original row bases, packed scale nibbles and sparse exceptions. It retains the exact logical scale bytes and does not change weight codes. |
| `STRATA_GLM_FP4_CPU_SPARSE=1` | Experimental lossless row layout with sorted exception records instead of a mask and rank for every group. Supports native expert matrices with up to 256 scale groups per row. |
| `STRATA_GLM_FP4_CPU_Q8=1` | Experimental lossy CPU W4A8 arithmetic; implies the sparse scale layout. Activations use signed int8 blocks of 64 values. Original FP4 codes are retained; their integer E2M1 magnitudes use a shared scale factor instead of per-weight FP16 rounding. AVX-512 VNNI accelerates the dot products; other CPUs use a scalar reference. GPU arithmetic is unchanged. |
| `STRATA_GLM_FP4_CPU_BORROW=1` | With the sparse or Q8 path, read original tiled scale bases and nibbles from the checkpoint mapping. Only exception records and row offsets occupy the CPU scale cache. The mapping owner outlives the cached views. |
| `STRATA_GLM_FP4_CPU_FUSED=1` | Run CPU gate/up, hidden quantization and down products in one worker dispatch, with barriers between phases. It also enables vector FP16 conversion with the scalar codec's rounding. Q8 uses vector conversion with either schedule. |
| `STRATA_GLM_FP4_CPU_SPIN=1` | Keep native FP4 workers active between dispatches. This uses CPU cores while the decoder lives, including between requests; use dedicated cores and measure the worker count. Other formats retain sleeping workers. |
| `STRATA_GLM_FP4_ROUTE_DROP=0.20` | Opt-in lossy decode: discard the smallest CPU expert contributions while their combined weight is at most this fraction of the token's original total routing weight. Range 0–0.5; default 0 disables dropping. GPU-resident experts, prompt prefill and the MTP draft layer retain all routes. Remaining weights are not renormalized. This bounds routing mass, not output error. |
| `STRATA_GLM_FP4_ROUTE_DROP_MAX=2` | Also cap the number of dropped experts per token and main layer. Range 0–7, default 2. Ties use the original route order. Reports include discarded counts and actual mean/maximum routing-mass shares. |
| `STRATA_GLM_FP4_ROUTE_RENORM=1` | After dropping, rescale retained CPU and GPU routing coefficients to restore the token's original total routing weight. Default off. The drop budget and reported discarded share refer to the original coefficients. Checkpoint weights, prefill and the draft layer retain their existing behavior. |
| `STRATA_GLM_FP4_DEVICE_ROUTING=1` | Gather prefill inputs and accumulate each GPU's expert results on the device, in expert order. Duplicate routes use the host path. |
| `STRATA_GLM_FP4_DECODE_RESERVE_MIB` | Opt-in reserve for allocations made after decode-cache selection, 256–8192 MiB per GPU. Unset or `-1` retains the original 3072 MiB primary and 1024 MiB secondary reserves. Explicit settings count all free preallocated pool blocks for the cache budget. Individual allocations still require a large enough block. A smaller reserve admits more resident experts; prefill limits and the allocator's memory checks still apply. |
| `STRATA_GLM_FP4_DECODE_ROUTING=1` | Gather narrow decode inputs and accumulate resident expert outputs on each GPU. Defer per-expert synchronization until the device sum is copied back; keep in-flight expert buffers protected until completion. Default off; duplicate routes retain the host path. |
| `STRATA_GLM_FP4_REGISTER_WEIGHTS=1` | Register immutable RAM mappings for read-only DMA. Requires RAM copy or sealed-RAM mode and GPU support for read-only registration. |
| `STRATA_GLM_FP4_ASYNC_UPLOAD=1` | Prepare the next expert on a separate copy stream. Requires device routing and VRAM preallocation. Host staging is bounded to two 16 MiB frames per GPU. Registered sources bypass the staging copies and cache validated immutable metadata. |
| `STRATA_GLM_FP4_TENSOR_ATTENTION=1` | Supply the existing MLA tensor helper for native prefill. With `STRATA_GLM_MLA_F16=1`, it uses FP16 attention operands and FP32 accumulation for long prompts. `STRATA_GLM_MLA_ABSORB_F16=1` also enables FP16 absorb operands. Native short prompts retain the original path below `STRATA_PREFILL_LOSSY_MIN_TOKENS` (default 4096). |

`STRATA_GLM_VRAM_PREALLOC=1` with a 1024 MiB margin avoids CUDA's per-allocation
rounding overhead for many small expert matrices. Repeated prefill evicts
experts before restoring the arena and growing scratch. The allocator and
arithmetic used by ordinary GGUF and EXL3 invocations retain their existing
defaults. Full-model GGUF/EXL3 regression runs were not available for this test.

The SM70 prefill reconstructor uses the same FP16 scale roundings as its skinny
decode core. Tests compare the operand bytes exactly. Those roundings can differ
from the reference kernel's FP32 global-scale calculation. Tensor attention is
also an opt-in precision change; its quality must be checked separately.

The SM70 core adapts the quadpair layout and shift/rebias decoder from
[v100-skinny at the studied revision](https://github.com/dnv2003/v100-skinny/tree/5b589c0dc81223e0ba65bcb3e755874723f8b515).
Its MIT notice and the decoder's Apache-2.0 attribution are retained in
`third_party/licenses/v100-skinny.txt`. K splitting and reduction stay fixed
across verification widths. CPU AVX2/AVX-512 and scalar implementations also
keep each row's arithmetic independent of verification width. Backend and
kernel choices still have different floating point reduction orders; compare
quality and generated tokens before claiming equivalence.

The measured 5.26 tokens/s reference is commit `a11bf931`, with the sparse and
Q8 CPU options unset. Its measurement record below remains the reference for
later decode experiments. MTP with a positive `STRATA_GLM_MTP_ACCEPT_MARGIN`
is also lossy: it can accept a draft token within the configured logit margin
and rank, subject to the per-window limit and repetition guard. These options
require their own full-model quality and throughput measurements.
Unused vocabulary slots masked with negative infinity do not disable bounded
acceptance. A masked draft cannot be softly accepted; soft acceptance also
rejects NaNs and positive infinity. A zero margin retains exact greedy acceptance.
`STRATA_GLM_VERIFY_GRAPHS` permits native ModelOpt verification graphs at widths
1 through 4 when decode graphs are enabled. As with the existing switch, its
presence enables it, including a value of `0`. `--check-verify-graphs` requires
actual captures and compares mixed widths, history reallocation, replay and
accepted-prefix rollback against direct execution. Ordinary EXL3 verification
graphs retain their existing restriction.

## Component measurements

On 2026-10-18, one Tesla V100-PCIE-32GB with CUDA 12.9 passed the synthetic
matrix suite, selected real checkpoint projections, and Compute Sanitizer
memcheck with zero errors. The suite covers CSF32, both FP8 scale formats,
BF16/FP32, cached operands, outliers, partial output tiles and widths 1 through 8.
The host codec and three CPU ISA paths also passed.

These are single samples of 1000 resident CUDA-graph launches, with rotating
operands larger than L2. They exclude routing, uploads and CPU work:

| Projection | Rows × columns | Reference µs | SM70 µs |
| --- | --- | ---: | ---: |
| NVFP4 expert gate | 2048 × 4096 | 47.59 | 9.94 |
| NVFP4 expert down | 4096 × 2048 | 40.36 | 11.86 |
| MXFP8 query, packed cache | 8192 × 4096 | 205.09 | 64.68 |
| Block-FP8 shared gate, row-major | 2048 × 4096 | 87.27 | 33.42 |

The CPU lookup measured 1467 µs versus 2093 µs for multiply/round on one
Xeon Gold 6240 thread at width 1, over a 72 MiB working set. Width 4 was
3631 versus 3717 µs; width 8 was 6736 versus 6512 µs. CPU scaling across
35 workers is not established by these single-thread measurements. GPU clocks
were not locked, and another job could use the other GPU and CPUs on the node.
The [compact evidence](glm_fp4_v100_core_measurement.json) records source and
binary hashes, the full-checkpoint manifest digest, working sets, comparisons
and validation scope. These results
do not establish full-model speed, residency or coding quality.

The CPU FP4 loops now specialize verification widths 1 through 8 at dispatch,
keeping their FMA and reduction order. On the same CPU model, an isolated
comparison against `df3f0443` used a 2048 × 4096 projection, a 72 MiB rotating
working set, and three alternating samples of 64 pool calls. The medians were:

| CPU workers | Width | Before µs | Fixed width µs |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 1497.61 | 1198.17 |
| 1 | 4 | 3641.79 | 3183.77 |
| 17 | 1 | 148.48 | 138.19 |
| 17 | 4 | 284.06 | 254.43 |

The 17 workers used one socket. The comparison checked bitwise output and
first-row stability across widths 1 through 8; AVX-512, AVX2 and scalar parity
tests also passed. These measurements include pool dispatch and exclude model
loading, routing, GPU work and scale-cache misses. A separate 35-worker run
gave 317.99 → 307.48 µs at width 1 and 315.95 → 292.20 µs at width 4;
width 3 was effectively unchanged. That matrix used single-thread first-touch
placement, so these runs do not establish the best full-model worker count.
They do not measure full-model throughput. The
[CPU comparison evidence](glm_fp4_v100_cpu_width_measurement.json) records all
samples, hashes, hardware and build scope.

## Full-model validation

The full checkpoint ran on two Tesla V100-PCIE-32GB GPUs, two Xeon Gold 6240
sockets, 35 CPU workers and a `181G` Slurm allocation on 2026-10-18. Runs used
`numactl --interleave=all` for weight allocation. GPU and CPU clocks were not locked;
RAM speed was not recorded. An independent 256 MiB sealed-RAM DMA test measured
12.31 and 12.34 GB/s concurrently, about 24.62 GB/s in aggregate over three
samples. These copy rates are not inference rates.

Short trials used the frozen 8192-token prompt, one full warmup, 64 output
tokens and one timed request. Device routing improved prefill from 155.09 to
226.45 tokens/s in a matched pair; decode was 9.14 and 9.59 tokens/s. The next
operand-reconstruction control measured 236.32 prefill and 9.14 decode tokens/s.
The final prefill candidate's short trial measured 388.39 prefill and 7.74
decode tokens/s. These single samples do not meet the qualification protocol.

With a 6 GiB CPU scale cache, the full protocol below completed three timed
512-output requests after one full warmup. Median prefill was 393.80 tokens/s
and median decode was 5.26 tokens/s; the ranges were 388.29–394.27 and
5.23–5.29 respectively. All three generated token sequences matched bitwise.
Timed requests had no major faults or swap, and peak sampled decoder RSS was
173.52 GiB. The 6 GiB and failed 8/10 GiB runs also produced identical warmup
tokens. Startup and RAM registration were excluded from these rates.

Sequential versus batched verification and rollback passed bitwise comparisons
with both kernel modes. The 64-token sequential-versus-prefill check failed the
existing 2% tolerance in both reference and SM70 modes, while their first greedy
tokens agreed. This remains unresolved. Synthetic GPU tests, real expert
operand checks and component Compute Sanitizer runs passed; those results do
not resolve the full-model parity failure.

For the frozen 8192-token prompt, switching the attention helper from FP32 to
FP16 operands gave final-row logit relative L2 0.08125 and KL divergence
0.02171 (FP32 to FP16). The first greedy token agreed. This is one prompt and
one logit row, not a corpus perplexity result. The faster prefill measurements
above use the FP16 attention options; their long-prompt quality remains open.

512-output runs with 8 and 10 GiB CPU caches completed their warmups and then
lost the decoder during the next prefill. Neither produced a timed sample.
The final step memory controller recorded two OOM kills across the job; its
181 GiB ancestors reported no OOM kills or limit failures. The decoder exit
statuses were not captured, so individual kill attribution remains unconfirmed.
These are failed runs, not throughput results. Raw evidence is retained
separately from the compact report. The completed 6 GiB run and quality checks
are recorded in the [full-model evidence](glm_fp4_v100_model_measurement.json).

Ordinary SM70 greedy decoding passed two of three frozen training coding tasks
and 19 of 20 held-out tasks with independent C++ oracles and a 1536-token output
cap. These checks do not establish reference-mode equivalence or corpus
perplexity, and they do not close the prefill consistency failure.
Reference greedy decoding also passed two of three training tasks. Its two
passing outputs matched SM70 bitwise; the failing task's outputs differed.
The reference mode was not run on the 20 held-out tasks.

## Opt-in decode measurement

A later run on the same two V100-PCIE-32GB GPUs and two Xeon Gold 6240
sockets used 32 CPU workers, the same frozen 8192-token prompt and a `181G`
allocation. Weights occupied one sealed, locked RAM copy with interleaved NUMA
placement. All 42 source shards and their sealed copies passed full SHA256
checks; their canonical manifest matched the preserved reference.

| Mode | Output tokens | Timed requests | Median prefill tok/s | Median decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Preserved W4A16 reference, 35 workers | 512 | 3 | 393.80 | 5.26 |
| Q8 CPU arithmetic, bounded route dropping, greedy | 512 | 3 | 394.74 | 19.08 |
| Q8 CPU arithmetic, bounded route dropping, MTP and suffix lookup | 512 | 3 | 390.70 | 31.02 |

The last two rows followed two full warmup requests and a discarded prefill request.
Thirteen mapped executable and shared-library files (1,038,543,808 bytes) were
read into the OS page cache in 3.97 seconds during the second discarded warmup.
The preparation record verifies that these reads finished before timed requests.
This preparation does not change weights or inference settings.
Decode ranged from 28.31 to 31.30 tokens/s; timing covers 511 advances after the
prefill token. Its three output sequences matched, with no EOS inside the
512-token window. Every timed request had zero major faults; sampled swap was
zero. Peak sampled decoder RSS was 167.55 GiB.
The 181 GiB memory controller recorded a whole-job peak of 172.89 GiB,
with no limit failures or OOM kills. Startup, the RAM copy, registration and
shutdown are outside these rates. The reference used one full warmup. Its
same-day rerun had timed major faults in two samples, so the table retains
the earlier clean reference measurement. Earlier attempts with the selected
settings failed the residency gate because their third timed requests had
30 and 32 major faults. Those attempts and an
unsuccessful preparation run remain excluded controls in the report.

This profile is lossy and opt-in. CPU products use block-64 int8 activations
with original FP4 codes. Runtime route dropping is limited to 50% of the
original routing mass and seven CPU contributions per token and main layer;
retained coefficients are renormalized. Actual mean discarded mass was
38.57%, with a maximum of 49.9999% and a maximum coefficient multiplier of 2.
Those counters include warmups and all evaluated verification rows, including
rejected rows.
Checkpoint files stay intact, and prefill and the draft layer retain all routes.

MTP uses depth 3, a logit margin of 3, rank limit 8, at most three soft accepts
per window and a four-token repetition guard. Suffix lookup uses depth 7,
minimum match 6, chaining and the existing cost policy. GPU decode routing
uses device gather/scatter and defers per-expert synchronization. Decode graphs
are off. The original GPU cache reserves are retained: smaller-reserve controls
measured slower. CUDA graphs and the narrower MTP-margin control were also
slower in 128-token trials; these short controls are not 512-token qualification.

Settings were selected from throughput and three training tasks, then frozen
before the held-out run. The selected build passed 2/3 training tasks
and 18/20 held-out coding tasks, compared with the preserved baseline's
2/3 and 19/20. The selected policy failed `mul_u64` and `merge_overlap`; no
settings were tuned on these held-out results. Each answer used temperature
zero, real EOS stops, a 1536-token cap and an independent compiled C++ execution
oracle. This small check does not establish general quality or perplexity.
The earlier prefill consistency failure and FP16-attention limitations remain
open.

The final V100 build passed rebuilt CPU tests in native VNNI, forced AVX2 and
scalar modes, plus the pool and C11 policy checks. Host/device routing logits,
mixed-width graph replay, verification history and prefix rollback passed
bitwise checks. With all new options off, 128 generated tokens matched the
frozen `a11bf931` reference. Additional CUDA SM86 and HIP gfx1201 builds passed
compile checks; GPU execution on those backends, full-model GGUF/EXL3
regressions and a new full-model Compute Sanitizer run were not performed.
The [decode evidence](glm_fp4_v100_decode20_measurement.json)
records configurations, all repeats, source/build hashes, failed controls and
quality results. The original [5.26 tok/s record](glm_fp4_v100_model_measurement.json),
profile and `fp4-v100-reference` branch remain preserved.

The experimental [fast profile](../configs/experimental/glm53f-fp4-dual-v100-fast.json)
expects a sealed RAM cache at `tmp/glm-fp4-ram`. From the repository root, with
the rebuilt executable and two GPUs allocated:

```sh
numactl --interleave=all python3 tools/glm_ram_cache.py models/glm53f-fp4 \
  --cache-dir tmp/glm-fp4-ram --report tmp/glm-fp4-ram-load.json --direct-read -- \
  python3 -m serve.server --engine glm \
    --config configs/experimental/glm53f-fp4-dual-v100-fast.json
```

Choose a fresh cache directory/report, and keep the profile's model directory
equal to the cache directory. The owner must stay alive while the server runs.
The first request is slower; the table measures warm resident requests.
Workers spin while the engine lives, including between requests. The profile
binds the API to `127.0.0.1`. The 20 tok/s decode target was met for this
protocol; the original 800 tok/s prefill target remains unmet.

## Qualification protocol

Use batch one, a frozen 8192-token prompt, 512 generated tokens, capacity
12288, prefill batch 8192, one full untimed warmup and at least three timed
resident requests. Use one configuration for both throughput metrics. Keep
ordinary greedy, exact MTP and kernel microbenchmarks separate.

```sh
python3 tools/glm_v100_bench.py models/glm53f-fp4 \
  docs/fixtures/glm_cpp_review_20261008/tr16-perf-8192.ids --prompt-ids \
  --decoder build-glm-v100/strata-glm-decode --fp4-kernel sm70 \
  --context 12288 --prompt-tokens 8192 --generated-tokens 512 \
  --prefill-batch 8192 --threads 35 --gpu-devices 0,1 \
  --gpu-budget-mib 32768 --speculative mtp --draft-depth 3 \
  --full-warmups 1 --repetitions 3 --output tmp/fp4-sm70-mtp3.json
```

Repeat with `--speculative none`, then with `--fp4-kernel reference` for the
same checkpoint. Keep raw evidence under an ignored local directory. The
adapter records metadata hashes, indexed file sizes, binary hash, prompt
hash, token IDs, per-device counters and host memory/fault samples. It rejects
swapping and major faults during timed native requests. Metadata hashes do
not substitute for complete weight payload hashes. Inspect the job memory
limit, RAM configuration, CPU interference, GPU clocks and link topology as
part of the measurement record.

Before reporting a speed qualification, run paired reference/SM70 logit and
corpus checks, independent repeated/reset requests, and ordinary/exact-MTP
coding checks on frozen inputs. A compile-only test or a kernel rate cannot
establish end-to-end throughput or coding quality.

The registered-RAM experiment can be reproduced with the following options.
Keep its raw report in an ignored directory. The 6 GiB CPU cache leaves more
room for the large checkpoint's runtime allocations than the failed 8/10 GiB runs.

```sh
export STRATA_GLM_VRAM_PREALLOC=1 STRATA_GLM_VRAM_PREALLOC_MARGIN_MIB=1024
export STRATA_GLM_FP4_CPU_SCALE_CACHE_MIB=6144 STRATA_GLM_FP4_CPU_CSF32=1
export STRATA_GLM_FP4_DEVICE_ROUTING=1 STRATA_GLM_FP4_ASYNC_UPLOAD=1
export STRATA_GLM_FP4_REGISTER_WEIGHTS=1 STRATA_GLM_FP4_TENSOR_ATTENTION=1
export STRATA_GLM_MLA_F16=1 STRATA_GLM_MLA_ABSORB_F16=1
numactl --interleave=all python3 tools/glm_ram_cache.py models/glm53f-fp4 \
  --cache-dir tmp/fp4-ram --report tmp/fp4-ram.json -- \
  python3 tools/glm_v100_bench.py '{model}' \
  docs/fixtures/glm_cpp_review_20261008/tr16-perf-8192.ids --prompt-ids \
  --decoder build-glm-v100/strata-glm-decode --fp4-kernel sm70 \
  --context 12288 --prompt-tokens 8192 --generated-tokens 512 \
  --prefill-batch 8192 --threads 35 --gpu-devices 0,1 \
  --gpu-budget-mib 32768 --speculative none --full-warmups 1 \
  --repetitions 3 --output tmp/fp4-resident.json
```

The benchmark allows a longer untimed shutdown when read-only RAM registration
is enabled. Startup, registration and cleanup are excluded from request rates.

The opt-in profile met the 30 tokens/s decode target with a 31.02 tokens/s
median under the resident protocol above. The 800 tokens/s prefill target
remains unmet.
