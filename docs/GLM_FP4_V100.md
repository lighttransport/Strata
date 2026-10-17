# Native GLM FP4 on two V100 GPUs

The experimental native loader accepts the full GLM-5.3-Flash ModelOpt mixed
precision checkpoint directory. It follows `model.safetensors.index.json`,
including indexed shards in subdirectories. The checkpoint stays in its
published weight formats: NVFP4 E2M1 codes with group-16 E4M3 scales and an
FP32 global scale, lossless CSF32 scale encoding, MXFP8 with E8M0 scales,
block FP8 with FP32 128x128 scales, and protected BF16/FP32 tensors.

`--fp4-kernel=reference` is the default for this format. The opt-in
`--fp4-kernel=sm70` uses Volta FP16 tensor cores and FP32 accumulation.
Both use W4A16 arithmetic: inputs and reconstructed matrix operands are FP16;
this does not emulate ModelOpt's FP4 activation quantizer. Original weight
codes and scales remain available in the mapped checkpoint. GGUF, EXL3 and
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
reserves another 6 GiB. It bounds this against physical RAM, visible cgroup
limits and `SLURM_MEM_PER_NODE`. `181G` is an allocation request, not proof of
residency; check RSS, swap, page faults and the job memory controller during a
run. Locking needs a sufficient existing memlock allowance.

```sh
python3 tools/glm_generate.py models/glm53f-fp4 'Explain NUMA placement.' \
  --decoder build-glm-v100/strata-glm-decode --fp4-kernel sm70 \
  --gpu-devices 0,1 --context 12288 --prefill-batch 8192 --lock-weights
python3 -m serve.server --engine glm \
  --config configs/experimental/glm53f-fp4-dual-v100-sm70.json
```

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

The SM70 core adapts the quadpair layout and shift/rebias decoder from
[v100-skinny at the studied revision](https://github.com/dnv2003/v100-skinny/tree/5b589c0dc81223e0ba65bcb3e755874723f8b515).
Its MIT notice and the decoder's Apache-2.0 attribution are retained in
`third_party/licenses/v100-skinny.txt`. K splitting and reduction stay fixed
across verification widths. CPU AVX2/AVX-512 and scalar implementations also
keep each row's arithmetic independent of verification width. Backend and
kernel choices still have different floating point reduction orders; compare
quality and generated tokens before claiming equivalence.

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

The 30 tokens/s decode and 800 tokens/s prefill numbers are qualification
targets. They are not measured results in this document.
