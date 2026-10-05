# GLM-5.3-Flash native EXL3

The experimental GLM decoder accepts the native safetensors directory as well as
GGUF shards. The current native contract is the routed-only K2/MCG pack, with
`.trellis`, `.suh`, `.svh` and `.mcg` for each expert projection. It executes the
existing quantization; it does not requantize the model. Other EXL3 codebooks,
bit widths and packed scales are refused.

## Run

Install the normal Python requirements and the optional tokenizer:

```sh
python3 -m pip install -r requirements.txt -r requirements-exl3.txt
cmake --build build-glm --target strata-glm-decode strata-glm-exl3-check
build-glm/strata-glm-exl3-check /mnt/nvme01/models/glm53f/glm53f-exl
python3 tools/glm_generate.py /mnt/nvme01/models/glm53f/glm53f-exl \
  'Explain NUMA placement.' --decoder build-glm/strata-glm-decode --tokens 32
python3 -m serve.server --engine glm --config configs/glm53f-exl.json
```

Use an existing CUDA-enabled GLM build directory; the targets require the same
GGML dependency as the GGUF GLM decoder. The server binds to its usual loopback
address. Native defaults are context 8192, dense cache 4096 MiB, GPU budget
10240 MiB, prefill batch 256, 15 CPU workers and NUMA affinity. A larger prompt
needs an explicit `--context`, up to 65536 in this implementation. Greedy text
CLI and API requests use the existing token stream, cancellation and reset
protocol. The supplied chat template always opens `<think>` and defaults to
max reasoning effort; `enable_thinking=false` does not disable that template. Vision, MTP, lookup speculation, multiple GPUs, GGUF expert caches,
remote tensor parallel and CUDA graph replay are not qualified for this format.

## Execution and memory

Decode runs routed experts on persistent CPU workers. Each worker is pinned to
an allowed physical core, leaving the first physical core for the host. Output
rows are split on 128-element Hadamard boundaries and repacked into contiguous
output tiles. Workers allocate and first-touch their rows in 256 MiB anonymous
arenas. The startup log lists the actual worker CPUs. Placement follows Linux
first-touch policy; start with the normal local allocation policy, rather than
an inherited interleave policy. Selected experts remain resident across requests;
loading is lazy, so initial requests include disk reads and packing.

GPU prefill groups tokens by expert, uploads its packed rows and reconstructs
one projection into reusable FP32 scratch for cuBLAS. Hadamard transforms and
scale application run on the GPU. It never expands the complete expert model.
Both Hadamard transforms use the upstream normalization, `1 / sqrt(128)`.
The head retains its padded shape; IDs absent from the tokenizer are masked
before greedy selection. Batches convert BF16 dense weights in panels of 512
output rows in GPU scratch before an FP32 cuBLAS product. Single-token products
normally use a BF16-weight GPU GEMV with FP32 accumulation. The supplied config
sets `STRATA_EXL3_CPU_DENSE=1`: decode instead reads BF16 matrices directly on the
CPU workers, transferring only their input and output vectors. Attention and
recurrent state remain on the GPU. This avoids repeated whole-matrix uploads;
its FP32 reduction order differs from the GPU GEMV. Clear that environment
setting to use GPU dense products.

On AVX2 CPUs the routed K2 kernel decodes eight output columns together, reusing
packed-word vectors and preserving scalar accumulation order. Other CPUs use
the scalar path. Set `STRATA_EXL3_SCALAR=1` to force scalar expert math. The GPU
fixed-weight cache retains a subset up to half its budget, leaving the rest for
streamed matrices. `STRATA_EXL3_LRU=1` restores plain LRU for comparisons.

`--weight-pages=huge` (server `weight_pages: "huge"`) requests Linux transparent
huge pages for the anonymous routed-weight arenas. The default explicitly
requests 4 KiB pages. THP advice is not a guarantee of huge-page backing, and
requires no reserved hugetlb pool. Compare steady-state decode after weights
are resident, and inspect `/proc/PID/smaps` for `AnonHugePages` before attributing
a speed change to huge pages.

On 2026-10-05, inspection of the local pack found 120 shards and 150226 tensors.
Actual payload was 97709588472 bytes. Main routed weights were 71.2904 GiB,
main fixed weights 16.6172 GiB, MTP 2.04167 GiB and vision 1.04984 GiB. MTP and
vision are not materialized by the native text path. The index's `total_size`
counts whole shard files, including headers; individual tensor sizes are checked
against their payload spans. Admission reserves 6 GiB of host RAM beyond main
weights. Temporary dense conversions also use that reserve.

Sparse MLA latent and pooled caches currently stay on the GPU. FP32 cache data
for 64K context is approximately 1.461 GiB, and KDA recurrent state is another
136 MiB. A 1M context needs approximately 23.375 GiB for these MLA caches alone;
host KV offload is required before enabling it on a 12–16 GiB card. Fitting the
weights in 128 GB host memory does not establish 1M-context support.

## Validation

```sh
cmake --build build-glm --target exl3_cpu_test exl3_cuda_test safetensors_test
ctest --test-dir build-glm -R '^(exl3_|safetensors_test)' --output-on-failure
build-glm/exl3_cpu_test /mnt/nvme01/models/glm53f/glm53f-exl
python3 tools/test_glm_artifact.py
python3 tools/test_glm_generate.py
python3 -m unittest serve.test_glm
```

The CPU tests check FP16 round trips, independent Clair codebook/tile golden
hashes, explicit Hadamard signs, native A_log conversion, malformed safetensors
and exact scalar-versus-sharded results on real gate/up/down projections in
layers 3, 7 and 44, experts 0, 17 and 287, plus MLA K/V extraction. The optional CUDA test checks trellis reconstruction,
normalized Hadamard, BF16 GEMV/conversion and a complete synthetic projection,
with explicit skip status when no CUDA device is available. Pass the model
directory to `exl3_cuda_test` to check full-sized real gate/up/down projections.

On 2026-10-05, CUDA was checked on the two-node Threadripper 1950X with an
RTX 5060 Ti 16 GB (driver 615.71.09). Real layer-3 expert-17 gate/up/down GPU
projections matched the scalar CPU reference with relative L2 errors of
1.16e-6, 1.22e-6 and 8.39e-7. Full-model single-token CPU-expert versus
GPU-expert prefill state/logit parity passed, including the same greedy token.
Two-token batched versus sequential prefill also passed, with greedy token
2282 on both paths. The single-token check peaked at 20671.1 MiB host RSS and 4813.2 MiB engine GPU allocation;
it loads only selected experts and does not measure complete model residency.
All 118 existing mock/API server regressions and the GLM boundary tests passed.
A short chat CLI run used 18 prompt tokens and generated the coherent thinking
prefix "The user is asking me to reply with". Prefill took 50.404 s, including
lazy loading/packing; seven subsequent CPU decode steps took 54.299 s
(0.129 tokens/s). Host peak RSS was 39378.9 MiB and engine GPU allocation peaked
at 4815.06 MiB. This scalar CPU kernel is much slower than the existing Q2 path.
The loopback server also returned HTTP 200 for health, model listing and a
four-token chat completion ("The user is asking", truncated reasoning).
See [the runtime measurement](glm53f_exl_runtime_measurement.json) for scope
and settings. 8K/32K/64K throughput and 1M-context execution are not established.

EXL3 equations follow ExLlamaV3 commit
`c5d9c657966ffeeaa9353f0cc899f18629da4a13` and the portable Clair decoder.
The upstream MIT notice is in `third_party/exl3/LICENSE`. The safetensors
implementation follows the bounded-reader design in `gemm/glm53f/common`.

## Decode bottlenecks measured on the two-node host

A follow-up on 2026-10-05 used the same Threadripper 1950X and RTX 5060 Ti,
with the EXL3 server unloaded during Q2 measurements. Q2's 53-token short chat
produced the same 133 output IDs in three repetitions, matching the earlier
verified output. Plain CPU-expert decode measured 8.53 / 8.86 / 8.92 tokens/s;
page-local down scheduling measured 8.78 / 8.76 / 8.87 tokens/s. The generated
C++ passed 400538 independent sieve and unsigned-boundary checks. Peak Q2 host
RSS was 102511 MiB and engine GPU allocation was 8219 MiB. Main Q2 payload is
98.648 GiB, excluding its MTP block, so it fits in the 125 GiB usable host RAM.

The median Q2 trial spent 50.28 ms/token in gate/up, 1.41 ms in activation
quantization, 27.62 ms in down and 33.62 ms elsewhere. Its expert kernels process
2.751 GB of packed weights per token at about 35.3 GB/s equivalent throughput.
This includes dequantization and arithmetic; it is not a DRAM-counter bandwidth
measurement. The 90 GB/s streaming result does not establish the same rate for
these kernels. CUDA profiling found dense Q5/Q6 projections and about 2180
launch calls per token, with fixed weights retained on the GPU.

A separate EXL3 trace captured three decode steps. It transferred 50.854 GB from
host to GPU: **16.951 GB/token**, taking **3.308 s/token** on the device transfer
track. Its 4096 MiB weight cache cannot retain the native fixed weights across a
complete layer traversal. The trace also recorded 3768 allocations and frees;
BF16 GEMV kernels themselves used only 127.944 ms across the three steps.
An eight-second CPU sample near the end of decode attributed 77.64% of sampled
cycles to the scalar expert apply loop. EXL3's legacy `CPU_EXPERT` counters are
not populated, so their zero values do not represent zero CPU work.

These traces point to vectorizing the EXL3 expert loop and retaining a selected
set of fixed weights instead of uploading the complete fixed-weight traversal
every token. Page advice cannot remove either dominant cost. Profiling changes
timings; the instrumented EXL3 result (0.112 tokens/s) is diagnostic rather than
a steady-state throughput claim. Full settings, trial data and profile scope
are in [the remeasurement](glm53_flash_q2_short_numa_remeasurement.json).

## Decode optimization results (2026-10-05)

On the same two-node Threadripper 1950X, 125 GiB usable RAM and RTX 5060 Ti
16 GB, Q2's 53-token prompt and 132 decode steps per trial measured:

| Q2 path | Median tokens/s (three trials) |
| --- | ---: |
| Direct launches, previous expert kernel | 8.856 |
| CUDA graphs and shared input quantization | 9.246 |
| Graphs, shared quantization and table signs | 9.675 |

A clean reverse-order graph/shared-quantization control measured 9.377 tokens/s,
so the table-sign change added about 3.2% on this host. An earlier reverse
control overlapped CPU compilation and was discarded.

The table-sign path replaces IQ2_XS sign-byte expansion with four lookups in the
existing 1 KiB sign table. It preserves integer dot products and FP32
accumulation order. It is enabled by `STRATA_IQ2_TABLE_SIGNS=1`; unset it to
restore vector sign expansion. All 399 generated IDs matched the previous
baseline, whose generated C++ passed 400538 independent checks. The real-weight
expert parity test passed layers 3, 11, 44 and 45, including multi-token CPU
products. CUDA graphs separately passed eight sequential bitwise-logit checks
and checkpoint rollback. Peak Q2 host RSS was 102553 MiB; peak engine allocation
was 8365 MiB with graph verification and 8219 MiB in the table-sign timing run.

For EXL3, the 18-token greeting prompt generated four IDs per trial (three decode
steps). After the first repetition had loaded all selected expert matrices:

| EXL3 path | Warm median tokens/s (last two trials) |
| --- | ---: |
| Scalar routed experts, plain GPU LRU | 0.1575 |
| AVX2 routed experts, persistent fixed-weight subset | 0.2336 |
| AVX2 experts, persistent subset, host BF16 dense products | 0.5090 |

The three paths produced identical short token streams. AVX2 expert products
matched the scalar result bitwise for 27 real projections and 256 randomized
tiles. Host BF16 arithmetic passed scalar-accumulator and double-precision
reference checks. Full-model GPU batched prefill versus CPU dense reference
passed the existing 2% relative-L2 state/logit threshold and selected token 2282
in both paths. Host BF16 and GPU GEMV use different summation orders; full
logits are not claimed bitwise identical. Peak EXL3 host RSS was 37446 MiB and
engine GPU allocation 4815 MiB for the host-dense path. These are short decode
measurements, not long-context or complete-answer quality tests.

The EXL3 server config enables host BF16 decode. The Q2 config enables graphs,
shared input quantization and table signs. Use:

```sh
python3 -m serve.server --engine glm --config configs/glm53f-exl.json
numactl --interleave=all python3 -m serve.server --engine glm \
  --config configs/glm53f-q2.json
```

Do not inherit Q2's interleave policy for EXL3: its packed routed rows rely on
worker-local first touch. The configs describe this tested 16 GB GPU and 128 GB
RAM host; they are not universal hardware defaults. Loading and GPU prefill are
excluded from decode timing; EXL3's first repetition also includes lazy expert
loading. Trial timings, hashes, memory peaks and validation scope are saved in
[the optimization measurement](glm53f_decode_optimization_measurement.json).

## Integer CPU expert probe (2026-10-05)

`strata-glm-exl3-kernel-bench` tests 336 expert calls using independent copies
of real layer-3 expert-17 matrix shapes: 2113929216 packed bytes, beyond L3.
On the two-node Threadripper 1950X, 15 pinned workers and worker-local first
touch, the three-trial median was 1463.50 ms for native MCG FP32 arithmetic,
503.715 ms for a synthetic mul1/int8 probe, and 869.873 ms for a two-pass
integer probe. Hadamard transforms, scales, SwiGLU and worker coordination
are included; attention, fixed projections, routing and sampling are excluded.
The fastest probe therefore permits only about 1.99 expert-only tokens/s on
this host. It does not establish a 10 tokens/s CPU path.

The integer implementation passed 256 randomized tile comparisons. The probe
uses the mul1 integer codebook over the existing packed shapes; it is not
inference with the MCG model and has no model-quality result. Its two-pass
activation representation uses base 127. Source inspection used upstream
EXL3 revision `16a49792a3c93d8432d72e6c4bce800841566577`. MCG cannot be
converted to mul1 by relabeling its packed bits. The original model files
remain unchanged.

## Experimental resident fixed weights and native GPU experts

`STRATA_EXL3_DENSE_QUANT=1` packs large BF16 fixed matrices into runtime
blocks of 32 weights with one FP16 scale. `STRATA_EXL3_DENSE_BITS=6` selects
signed six-bit values; `8` selects signed eight-bit values. The output head
always uses eight bits and matrices smaller than 8 MiB stay BF16. Activations
remain FP32. The six-bit mixed set occupies 6694.31 MiB on this model. Source
safetensors are not changed. This is additional quantization, not exact BF16
inference, and remains opt-in.

`STRATA_EXL3_GPU_EXPERT_CACHE_MIB=N` enables an LRU of native MCG packed
experts on the GPU, with direct packed GEMV and Hadamard transforms. It requires
resident fixed-weight quantization. Each complete expert uses about 6.035 MiB.
The cache also evicts when actual free VRAM approaches the GPU's runtime
reserve, with another 256 MiB of margin. CUDA graphs are not enabled for this
path. Cache hit/miss and copied-byte counters accompany each trial; a short
warm repetition does not predict throughput on a longer response.

`--check-native-dense` compares eight natural teacher-forced continuation
positions with original BF16 fixed weights, using matched GPU batched prefill.
It reports KL divergence, raw logit relative L2 and greedy agreement. Its
smoke threshold requires KL at most 0.01 at every position and eight matching
greedy predictions. Raw logit L2 is reported separately; this smoke check does
not establish general model quality or long-context accuracy.

On the tested host, `configs/glm53f-exl-fast.json` enables this experimental
mixed Q6/Q8/BF16 path with an 8192-token context, a 10240 MiB fixed-weight
budget and a GPU expert-cache ceiling of 2816 MiB. The cache may shrink below
that ceiling. `configs/glm53f-exl.json` retains the original BF16 fixed-weight
path as the accuracy-preserving alternative. Start the experimental server:

```sh
python3 -m serve.server --engine glm --config configs/glm53f-exl-fast.json
```

The 18-token greeting smoke check matched all eight BF16-reference greedy
predictions, with maximum KL divergence 0.00786035 and maximum raw-logit
relative L2 0.115307. Thus it passes the probability/greedy smoke threshold,
but does not pass the older 2% raw-logit criterion used for arithmetic parity.
After this reference pass, three decode steps per repetition measured
1.99961, 2.01463 and 2.03373 tokens/s. The last two repetitions each missed
708 of 1008 expert calls and copied 4272.89 MiB. Actual cache size was
2100.23 MiB under the physical-memory guard. These short measurements do not
demonstrate 10 tokens/s.

Without the BF16 quality-reference pass, a 32-token response (31 timed decode
steps) measured 0.926864 tokens/s on its first repetition, including lazy host
expert loading, and 2.17656 / 2.20704 tokens/s on warm repetitions. All 96 IDs
matched across repetitions. Peak host RSS was 47397.3 MiB and peak engine GPU
allocation 10396.1 MiB. The adaptive cache settled at 2263.18 MiB. Each warm
trial had 3381 hits and 7035 misses, copying 42457.3 MiB: about 1.34 GiB per
decode step. The observed GPU link was PCIe Gen3 x8.

A per-layer cache experiment produced the same 96 IDs and warm rates of
2.25805 / 2.20157 tokens/s. Its roughly 2% mean gain was insufficient to retain
the added policy complexity, so the final path uses adaptive global LRU.
CPU compilation did not overlap these timing runs. CPU quantization bounds,
CUDA Q6/Q8 conversion and GEMV parity, three real packed CUDA projections,
three tokenizer checks and three API boundary checks passed. Complete timing,
quality rows, counters and token hashes are in
[the fast-path measurement](glm53f_exl_fast_measurement.json).

The experimental server was restored at `http://127.0.0.1:8095/v1` with the
fast config. Health and model discovery returned HTTP 200 with the model
loaded. Two four-token API requests both returned the same reasoning text,
`The user is asking`, confirming generation and repeated-request reset. API
throughput counts the first prediction from prefill as a generated token;
use the CLI decode-step rates above for throughput comparisons.
