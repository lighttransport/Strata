# NVFP4 W4A16 on RDNA4

This notebook covers experimental tools and subsequent opt-in CPU decoder
changes. Each dated section records its own integration and validation scope;
early statements about unchanged dispatch apply to that experiment.

The repository includes sources, tests and selected compact verification
evidence. Large raw benchmark collections, downloaded weights and local build
trees remain outside the commit. Historical `results/` references describe
those local collections when the corresponding files are absent from checkout.

The design adapts [JohnTDI-cpu's RDNA4 WMMA guide](https://github.com/JohnTDI-cpu/rdna4-wmma-guide)
(CC BY 4.0). The guide publishes a design description, not its GEMM source.
This implementation uses the documented column-distributed accumulator mapping,
LDS lookup decoding, vector loads and register tiling. Our prior BF16 kernel
already used the same accumulator mapping.

`tools/glm_fp4_gpu.cu` adds an explicitly selected HIP variant: one wave computes
16x32 outputs using two accumulators, reusing activation fragments. Each K32
iteration executes two K16 WMMA operations per output tile. Each lane loads
8 packed weights with one 32-bit load and 8 activations with two float4 loads.
Unlike the guide's MXFP4 example, this NVFP4 implementation preserves each K16
E4M3 scale, uses BF16 fragments and applies the F32 global scale in the epilogue.
It does not reinterpret E2M1 codes as INT4 or quantize activations to FP8.
The bounded test API accepts 1 through 16 tokens. The guide's full 64x64 block
and 2x2 register tiling were not implemented for this small-token workload.

Build and run from the repository root:

```sh
make -f experiment/glm_fp4_local/Makefile
build-fp4-local/glm-fp4-rdna4-test "$PARTIAL_MODEL_DIR"
```

The input is a safetensors directory containing expert 0 gate/up/down weights,
CSF scales and global scales from layers 3 and 4 of
`nvidia/GLM-5.3-Flash-FP4-Local`, revision
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`.
Weights are external artifacts and are not included here.
The shared C11 CSF decoder expands scales before transfer to the GPU; this is
not inline GPU decoding of compressed CSF scales.

For the existing two-node test, append `ib tiled` to the primary's arguments:

```sh
build-fp4-local/glm-fp4-node-hip 0 "$IB_ROOT" "$PARTIAL_MODEL_DIR" ib tiled
# On the CUDA worker, retain its baseline kernel:
./node-cuda 1 "$IB_ROOT" unused ib
```

## Measurements, 2026-10-19

Hardware: Threadripper 1950X, RX 9070 XT (gfx1201), 128 GB installed RAM;
worker Ryzen 3950X, RTX 5060 Ti 16 GB, 64 GB RAM. Both miners remained running.
RAM speed was not recorded. HIP clang 24 (ROCm core 10.1 installation), CUDA
13.2 compile check. These are contention-affected smoke measurements, not
throughput estimates for a full model or miner-free performance claims.

- 18 asymmetric synthetic cases include partial N tiles, K16 tails, all FP4
  codes, and a range of finite E4M3 scales. The same test data generator covers
  every finite nonnegative E4M3 scale in its largest case.
- 18 real projection cases: layers 3 and 4, expert 0, three projections, M=1/7/16.
  All 36 cases match the original kernel bit-for-bit. Every output also passes
  a scalar GPU FP64 accumulation error bound; 32 outputs per case are checked
  against an independent host decoder and FP64 reference.
- Each real case has two warmup pairs then eight measured pairs, with alternating
  order and a 100 ms pause per pair. HIP events measure only the kernel, excluding
  allocations, transfers, CSF expansion and reference work. Median speed ratios
  range from 0.954 to 3.655. Two cases regress slightly; the variant stays opt-in.
- The real two-node IB test passes four split MLP cases (layers 3/4, M=1/16).
  Local FFN width is 1408 and remote width is 640. The primary uses the new tiled
  kernel; the worker uses the prior CUDA kernel. Combined outputs pass comparison
  with the unsplit baseline. Logs identify the IB backend on both ranks.
- HIP and CUDA GPU source compilation passed. The new kernel is HIP-only and
  does not add a SYCL path. Full-model loading, routing and generation are not
  validated by these tests; production multi-node integration remains pending.

Raw compact logs and source/binary hashes are in `results/20261019/`.

## Persistent routed partial-weight runner

`tools/glm_fp4_moe_node.cpp` extends the test to experts 0 and 1 in layers 3 and 4.
It loads four experts once on each node, then accepts bounded, versioned run
commands with checked sequence numbers. The coordinator gathers token groups,
sends activations, runs its partition concurrently with the worker, sums the
partial outputs and scatters weighted results to the original token positions.
The worker requires no model directory or extra checkpoint copy.

The suite uses synthetic top-1 and top-2 routes, reversed token gathering,
unequal groups, skipped experts and batches of 1, 7, 16 and 3. Each scenario
chains the two layers with a bounded synthetic residual, then repeats from the
same input. Per-layer comparisons use unsplit experts on the primary with the
same incoming activation. Repeat checks require identical final output bits.
This tests MoE dispatch and communication; it does not load the model router,
attention, normalization, shared expert, mHC, or tokenizer. It is not a full
transformer forward pass or a text-generation test.

Fetch just the required slices, then build both workers:

```sh
python3 tools/glm_fp4_fetch_partial.py \
  --source "$CHECKPOINT_DIR" --output "$PARTIAL_MODEL_DIR"
make -f experiment/glm_fp4_local/Makefile nodes NVCC=nvcc
```

The extractor reads complete local shards where available and uses exact HTTP
ranges for the rest. It rejects full-shard responses, limits individual reads
to 16 MiB and tensor payload to 64 MiB, and records each tensor's SHA256 and source
range. Use a new output directory. Connection addresses and SSH aliases belong
in local configuration, not this report.

Copy `build-fp4-local/glm-fp4-moe-node-cuda` to the worker, then run:

```sh
python3 tools/glm_fp4_multinode.py \
  --ssh "$WORKER_SSH_ALIAS" --root "$IB_ROOT" \
  --model "$PARTIAL_MODEL_DIR" --worker-binary "$WORKER_BINARY" \
  --output "$RESULT_DIR" --tiled
```

The launcher uses existing SSH configuration, retains separate node logs, checks
both nodes' exit status and IB completion markers, and bounds worker lifetime
with `timeout`. It does not copy weights to the worker or manage mining processes.
A failed coordinator may leave the remote worker alive until that timeout.
The 2 GiB free-VRAM guard in the GPU primitive remains enabled. Reference checking
runs on every projection, so total wall time is not an inference performance
measurement. The fixed two-node split is still 1408/640 FFN channels.

### Persistent runner results, 2026-10-19

On the same two GPUs with mining left running, all 16 routed layer cases and
24 worker requests passed over IB. The second pass produced bit-identical final
outputs. Maximum per-layer relative L2 error against the unsplit reference was
`1.31363043e-5` (0.001314%). Every projection also passed the GPU FP64 bound and
independent host sample checks. No weight reload occurred between requests.

The fixture has 84 tensors and 53,745,916 payload bytes (51.3 MiB). Expanded-scale
packed partitions account for 38,928,432 bytes on the primary and 17,694,768 bytes
on the worker. The primary additionally holds 56,623,152 bytes of packed unsplit
reference matrices. These counts exclude runtime context, activations, scratch
buffers and the miners. No end-to-end throughput claim is made.

Both HIP and CUDA executables built; the C++ warning check and Python syntax
checks passed. The launcher verified both exit codes and explicit IB success
markers. Evidence is under `results/20261019/moe-persistent/`. Full-model routing
and generation remain unimplemented by this experimental runner.

## Three-node tree, partial weights

`tools/glm_fp4_tree_node.cpp` uses two separate IB contexts: primary to middle
node, then middle node to leaf. This matches the tested topology's routed IP
connection and separate middle-node IB ports; it requires no network changes
and does not assume direct verbs connectivity between primary and leaf.
The middle node owns its downstream context in one asynchronous task while
its main thread computes its GPU expert. It forwards leaf inputs and outputs.
The primary computes its own expert concurrently with the remote request.

Each node holds one whole expert from each of layers 3 and 4: expert 0 on the
primary, expert 1 on the middle, expert 2 on the leaf. Weights are transferred
once at startup. Both workers need only the executable. This is whole-expert
placement, not a three-way FFN partition. Protocol and sequence checks remain
bounded to two layers and 16 tokens. Empty groups are supported. The suite
covers top-1 routes, weighted top-3 routes, primary-only and leaf-only requests,
unequal token groups, reversed gathering, two-layer chaining and exact repeats.
Routes and residuals remain synthetic; this is not model generation.

```sh
python3 tools/glm_fp4_fetch_partial.py --source "$CHECKPOINT_DIR" \
  --output "$TREE_MODEL_DIR" --experts 3 --reuse "$PARTIAL_MODEL_DIR"
make -f experiment/glm_fp4_local/Makefile tree NVCC=nvcc
# Install glm-fp4-tree-cuda at WORKER_BINARY on both SSH workers.
python3 tools/glm_fp4_tree.py \
  --relay "$MIDDLE_SSH_ALIAS" --leaf "$LEAF_SSH_ALIAS" \
  --up-root "$PRIMARY_IB_ROOT" --down-root "$MIDDLE_IB_ROOT" \
  --model "$TREE_MODEL_DIR" --worker-binary "$WORKER_BINARY" \
  --output "$TREE_RESULTS" --tiled
```

The default port assignment is primary port 1, middle upstream port 1,
middle downstream port 2, leaf port 1; command-line options can override it.
The CUDA binary contains sm_86 and sm_120 code. The primary uses gfx1201.
The extractor's three-expert budget is 96 MiB; reuse tensors are checked against
their recorded SHA256. On this run, complete local shards were available, so
creating the 80,619,104-byte (76.9 MiB) fixture required no new network download.

### Three-node results, 2026-10-19

Primary and middle hardware are as above. The third machine reports a Ryzen 9
3950X, 65,166,020,608 bytes of RAM and an RTX 3070 8 GB. Its SSH alias is not
an indication of CPU architecture. Available RAM before the test was about
52.7 GiB. The miner used 4827 MiB VRAM and reported 100% GPU utilization both
before and after testing. The GPU negotiated PCIe 4.0 x16; the IB adapter
negotiated PCIe 3.0 x4 and QDR 40 Gb/s. RAM speed was not measured.

- All 16 routed layer cases passed, with 16 protocol requests at each worker
  (including idle requests). Repeated final outputs were bit-identical.
- Largest relative L2 error versus the primary's unsplit reference was
  `9.27607847e-5`, or 0.009276%. Every projection also passed its FP64 bound and
  independent host sample checks. Cross-backend results are not bit-identical.
- Each node holds 28,311,576 bytes of packed expert data (about 27 MiB).
  The primary additionally holds six reference experts, 84,934,728 bytes.
  Runtime context, activations, scratch space and mining allocations are extra.
- All three processes exited successfully with explicit IB completion evidence.
  The middle log confirms both IB ports. No miner or system limit was changed.
- HIP and dual-architecture CUDA builds passed, as did C++ warning, Python
  syntax and whitespace checks. No production engine default changed.

The bounded IB probe registers 256 KiB, runs two warmups and 20 measured round
trips per size. These short samples were taken with mining active. Results
include host software overhead and are not sustained bandwidth measurements.

| Payload each way | Primary–middle median round trip | Middle–leaf median round trip |
|---|---:|---:|
| 16 KiB | 20.256 us | 28.159 us |
| 256 KiB | 144.549 us | 177.717 us |

An initial attempt using the general `ucomm_bench` failed when it tried to
register 64 MiB against existing 8 MiB locked-memory limits on the primary/leaf.
Those failed logs are retained as `tree-link-*`; they are not qualified timings.
The bounded replacement succeeded (`tree-probe-*`). Run it with:

```sh
make -f experiment/glm_fp4_local/Makefile build-fp4-local/ib-probe
# Start rank 0 and rank 1 on the two endpoints of one edge.
./ib-probe "$EDGE_RANK" "$EDGE_ROOT" "$IB_PORT"
```

Logs and source/binary/fixture hashes are in `results/20261019/tree/`.
Full-model integration, GPU cache eviction, adaptive placement, a three-way
FFN-split comparison, GPU transfer bandwidth, and end-to-end throughput remain
untested here. These correctness and small-message tests do not establish a
three-node speedup over two nodes.

## Direct workers, no relay

The physical IB connections were subsequently verified to form a triangle:
primary port 1 to worker one port 1, worker one port 2 to worker two port 1,
and **primary port 2 to worker two port 2**. The IP route used earlier did not
fully describe that topology. A direct native-verbs probe passed on the third
edge; no network configuration was changed.

Add `--direct` to connect the primary separately to each worker. Both bootstrap
endpoints must belong to the primary and use different TCP ports. The verified
run used LAN addresses for TCP bootstrap; all weight, activation and result
payloads used native IB verbs. Each primary IB context has one asynchronous
owner while the primary GPU computes. Worker one never receives or forwards
worker two's weights, activations, or outputs. The existing relay mode remains
available without `--direct`.

```sh
python3 tools/glm_fp4_tree.py --direct --tiled \
  --worker-one "$FIRST_SSH_ALIAS" --worker-two "$SECOND_SSH_ALIAS" \
  --up-root "$PRIMARY_BOOTSTRAP_ONE" --down-root "$PRIMARY_BOOTSTRAP_TWO" \
  --primary-port 1 --relay-up-port 1 --primary-leaf-port 2 --leaf-port 2 \
  --model "$TREE_MODEL_DIR" --worker-binary "$WORKER_BINARY" \
  --output "$DIRECT_RESULTS"
```

`--worker-one`/`--worker-two` are aliases for the older `--relay`/`--leaf` options;
log names remain `primary.log`, `relay.log`, and `leaf.log` for compatibility.
In direct mode `--relay-down-port` is unused. The protocol handshake distinguishes
direct mode from relay mode before weight transfer. No production engine default
changed, and the same 76.9 MiB fixture is reused without another weight download.

Direct mode passed all 16 routed layer cases on the three GPUs, with exact
repeat checks and the same maximum relative L2 error (`9.27607847e-5`) as the
relay run. A fresh relay regression also passed all 16 cases. The eight final
output FNV64 fingerprints match between modes (fingerprints, not a stored raw
output byte comparison). All nodes exited successfully. Each worker still
holds about 27 MiB of packed expert data; miners were left running.

On the direct primary–worker-two edge, the bounded probe measured median round
trips of 27.416 us for 16 KiB each way and 176.538 us for 256 KiB each way, using
two warmups and 20 samples. These are short contention-affected communication
measurements, not a model throughput or direct-versus-relay speedup result.
HIP and CUDA builds, the C++ warning check, Python syntax check and whitespace
check passed. Evidence is in `results/20261019/direct/`.

## Extended validation and low-duty endurance

`--extended` adds zero inputs, alternating-sign inputs, varying top-2 selection,
all-idle layers, and token counts 2 and 15. It runs eight scenarios through two
layers, three times: 48 layer cases per topology. Repeat equality now compares
raw bytes, including signed zero. `--dump-outputs` retains a bounded local F32
file for exact cross-topology comparison. Raw outputs stay outside the shared
report; hashes and compact logs are recorded here.

The 2026-10-19 direct and relay extended runs both passed all 48 cases. Their
3,293,184 output bytes matched exactly, SHA256
`fa3301c1fe4ad09863ca1cabd5e51e76d7c62d2965ea48a4fc5877bf5f6df410`.
The maximum relative L2 error versus the unsplit reference was `0.000122505225`
(0.012251%). Original scenario fingerprints are unchanged. The measured process
wall times include validation and intentional pauses and are not inference
performance measurements. Evidence is in `results/20261019/extended/`.

Command validation is shared C11 under `src/host/glm_fp4_protocol.c`, linked into
`strata_host` and both experimental workers. Invalid sequence, version, operation,
layer, token count, stop payload or worker destination is rejected before reading
activation payloads. CPU tests cover every legal token-count pair, malformed
fields, null input and destination restrictions. Three launcher tests check
success, missing worker evidence and failure cleanup without GPUs or real SSH.
The launcher now writes `summary.json` on success or runtime failure, and checks
exact case counts, request counts, topology, dump size and completion markers.

```sh
cc -O2 -std=c11 -shared -fPIC -Iinclude src/host/glm_fp4_protocol.c \
  -o build-fp4-local/libprotocol.so
python3 tools/test_glm_fp4_protocol.py build-fp4-local/libprotocol.so
python3 tools/test_glm_fp4_launcher.py
```

HIP, dual-architecture CUDA and a fresh CMake `strata_host` build passed. An older
CPU build directory could not regenerate because its external dependency path
was stale; the fresh host-only configuration disabled native GGML experts.
No full production decoder or SYCL GPU validation is claimed here.

`tools/glm_fp4_soak.py` provides bounded endurance testing while a checkpoint is
being downloaded. Each cycle starts fresh workers with persistent expert caches
for the 48-case suite, alternating direct and relay topology. This tests repeated
connections and result stability across mining contention; it does not test a
worker staying resident for hours. It checks results against a qualified output
SHA256. Defaults are one run every 600 seconds, at most 24 runs. It stops on the
first failure, a download completion marker, a `STOP` file in its output directory,
or the run limit. A STOP request is processed between tests. The underlying
launcher has a 300-second deadline; remote workers have their own timeout.

Store connection settings in an ignored local JSON file with these keys:

```json
{
  "worker_one": "worker-one",
  "worker_two": "worker-two",
  "model": "build-fp4-local/partial-tree",
  "worker_binary": "./glm-fp4-tree-cuda",
  "download_dir": "CHECKPOINT_DIRECTORY",
  "direct_roots": ["PRIMARY_BOOTSTRAP_ONE", "PRIMARY_BOOTSTRAP_TWO"],
  "relay_roots": ["PRIMARY_BOOTSTRAP", "MIDDLE_BOOTSTRAP"],
  "expected_output_sha256": "fa3301c1fe4ad09863ca1cabd5e51e76d7c62d2965ea48a4fc5877bf5f6df410"
}
```

```sh
python3 tools/glm_fp4_soak.py --config "$LOCAL_CONFIG" --output "$SOAK_RESULTS"
```

Each run retains separate node logs, a JSON summary and about 3.14 MiB of raw
outputs locally. The supervisor writes an atomic `status.json` and uses a lock to
prevent duplicate supervisors in the same output directory. Completion-marker,
stop-file and evidence-preservation checks passed. Future endurance runs are not
part of the qualified results above; consult their live status and per-run logs.

## Faster FP4 decode plan and CPU baseline (2026-10-19)

This is a proposed implementation order, not a GPU speed claim. GPU miners were
left running and no GPU benchmark was performed for this plan. The current
experimental runner is still a partial-MoE correctness harness, not a complete
FP4 model decoder.

### CPU measurements

The CPU-only benchmark is `tools/glm_fp4_cpu_bench.py`. Frozen output, raw repeat
times, input-scale digest, decoder source hash and binary hash are in
`results/20261019/cpu/tr16-csf-copy.json`. Build:

```sh
cc -O2 -std=c11 -shared -fPIC -Iinclude src/host/glm_fp4.c -lm \
  -o build-fp4-local/libcsf-bench.so
python3 tools/glm_fp4_cpu_bench.py --model "$MODEL_DIR" \
  --library build-fp4-local/libcsf-bench.so
```

Hardware: Threadripper 1950X, 16 physical cores / 32 threads, 128 GiB installed
RAM, two NUMA nodes. Compiler: Ubuntu GCC 15.2.0. RAM clock was not recorded.
The CPU was available per the operator; GPU mining remained active, so background
memory contention is not ruled out. Affinity uses one logical CPU per physical
core. The 8-core case uses CPUs 0–7 on one NUMA node; 16 cores use CPUs 0–15
across both nodes. Memory is first-touched by the assigned workers.

The CSF test decodes actual scales from layers 3–4, experts 0–2, all three
projections: 18 matrices. Seven timed repeats follow a warmup; each worker decodes
the set four times per repeat. This is warm-cache throughput of the existing
validated C decoder, with Python/ctypes dispatch and affinity overhead included.
Parallel outputs are compared against serial outputs; this is not an independent
format-correctness oracle. Tensor reads and allocations are outside timing.
The separate memory test copies a 512 MiB source into a 512 MiB destination eight
times per repeat. Payload bandwidth counts bytes copied once; physical memory
traffic is higher and must not be equated with the simulator's DRAM bandwidth.

| Physical cores | CSF median Gscales/s (min–max) | Copy median payload GB/s (min–max) |
|---|---:|---:|
| 1 | 0.663 (0.661–0.664) | 13.14 (11.74–14.39) |
| 8 | 2.810 (2.765–2.874) | 22.82 (21.87–23.28) |
| 16 | 5.549 (5.325–7.041) | 37.69 (37.07–38.37) |

The single-core CSF result can inform a warm-cache serial sensitivity scenario;
the aggregate rate cannot replace it unless the execution path actually decodes
independent matrices concurrently. Neither rate measures full-checkpoint cold
memory behavior, disk loading, pinned-memory DMA or other nodes. Keep the default
hypothetical simulator profile unchanged until those distinctions are modeled.

### Implementation order and acceptance checks

1. **Prepare scales once on CPU.** Decode independent expert matrices in a bounded
   pool, validate using the shared C11 contract, and retain packed E2M1 codes plus
   expanded E4M3 scale bytes in NUMA-local RAM. Replace compressed buffers after
   validation rather than retaining both permanently. Record startup time, peak
   RAM, local versus remote NUMA reads, and repeat full-sized streaming tests.
   CPU parallelism belongs in loading/prefetch, not as an assumed 16-fold speedup
   of one serial miss. This removes repeated CSF work from decode.
2. **Remove runner overhead.** Add persistent activation/output/workspace buffers
   and a bounded pinned upload pool. Keep gate/up intermediates, SiLU and down
   inputs on GPU. Replace per-call allocation, device synchronization and host
   round trips with stream events. Add per-node asynchronous dispatch so one
   expert upload can overlap another's compute without buffer reuse races.
   Preserve the synchronous reference mode for comparisons.
3. **Specialize batch-1 GEMV first.** The existing kernel issues 16-row WMMA for a
   single useful activation row. Prototype a wave/warp reduction over K with
   packed contiguous code loads, one scale load per K16 group, FP32 accumulation
   and final global scaling. Compare output-row tiling and bounded split-K to
   expose enough waves without excessive partial-output traffic. Fuse gate/up
   scheduling to reuse activations, then SiLU where synchronization permits.
   This is a hypothesis: unused WMMA rows do not imply a 16x speedup, since
   memory traffic and unpack can dominate.
4. **Keep a separate small-batch WMMA path.** For M=2/4/8/16, compare GEMV against
   the existing tiled BF16 WMMA implementation. Try vectorized code loads,
   register reuse of K16 scales, and packed-bit/LUT conversion that avoids
   per-value exponent computation. Tune output tiles, K staging and occupancy
   separately on RDNA4, SM120 and SM86. Dispatch thresholds must come from
   repeated measurements, not a shared hardware-peak assumption.
5. **Use a bounded two-tier expert cache.** Retain cold experts in packed FP4;
   optionally retain genuinely hot experts in BF16. BF16 requires 50,331,660 bytes
   per expert versus 14,155,788 packed, so compare saved unpack time against lost
   hit coverage at the same VRAM budget. Test eviction, promotion and prefetch
   together using real routing traces. Do not assume an 80% hit rate from capacity
   alone. A CPU AVX2 conversion path is a candidate for preparation, but uploading
   BF16 costs more bandwidth; GPU expansion once on promotion is another candidate.
6. **Qualify correctness and full-model speed.** Exhaustively check all 2,032
   E2M1/finite nonnegative E4M3 products (all are exactly representable in BF16;
   rechecked on CPU). Keep the global FP32 scale separate. Preserve BF16 activation
   rounding in the exact W4A16 comparison. Changed reduction order needs numerical
   bounds and relative-error checks against FP64, not an assumption of bitwise
   equality. Test K16 boundaries, tails, extreme scales, actual partial layers and
   direct/relay output agreement. Build HIP and CUDA for shared GPU changes.
   Measure kernel-only and end-to-end costs separately, then full-decoder quality
   and throughput when that path exists. GPU timings under mining are explicitly
   qualified, and tuning decisions require repeat distributions and a same-day
   baseline.

INT4/INT8/FP8 WMMA are separate activation-quantized experiments, not the default
exact W4A16 replacement. E2M1 cannot be mapped to one signed INT4 code with one
common scale; arbitrary BF16 activations also cannot be represented exactly by
INT4/INT8/FP8. Keep the lossless software-unpack/BF16 path as the reference.

The 20 tok/s target permits 50 ms for the entire decoder. First measure its
non-MoE budget; the current arbitrary 50–150 ms allowance cannot establish target
feasibility. Allocate the remaining time between MoE compute, cache misses and
communication. Report separately whether a proposed kernel reduces computation,
weight traffic or host overhead; faster arithmetic alone cannot hide a streaming
bandwidth deficit. No 20 tok/s claim is made by this plan.

### CPU optimization follow-up

The shared CSF decoder now specializes its common 32x2 block: one packed code
byte per row, two explicit output bytes, no variable-width inner loop. The
validation pass, exception ordering and odd-column fallback remain unchanged.
This is lossless and does not change quantization or routing. The CPU test suite
passed all six tests both normally and under UBSan, including independent format
fixtures, real matrix geometry, tails and malformed-input rejection.

An alternating baseline/candidate benchmark on the same actual 18 scale matrices
produced the following warm-cache medians. Each side has seven samples per core
count. Frozen repeat times and source/binary hashes are in
`results/20261019/cpu/tr16-csf-paired.json`. Use the benchmark's
`--compare-library BASELINE_SO --compare-source BASELINE_C` options to alternate
implementations; source paths are not emitted into the report.

| Physical cores | Original Gscales/s | Specialized Gscales/s | Ratio |
|---|---:|---:|---:|
| 1 | 0.613 | 1.352 | 2.21x |
| 8 | 2.461 | 5.318 | 2.16x |
| 16 | 4.642 | 9.740 | 2.10x |

These ratios are routine-level, not model decode speedups. Absolute rates varied
between sessions, which is why the alternating comparison supersedes comparisons
between separately timed runs. No CPU DRAM or FLOPS utilization claim follows
from a warm-cache CSF decoder result.

The existing standalone K-quant benchmark was also rebuilt and its selftests
passed. A synthetic two-expert-set, seven-iteration screen at 16 cores measured
73.8 GB/s read bandwidth, 42.7 GB/s Q2_K gate/up, 32.4 GB/s Q3_K down and 35.3
GB/s for its Q23 MoE layer (47.8% of the same-run read reference). Its CSV and
reproduction metadata are in `results/20261019/cpu/tr16-q23-screen*`. This is the
standalone MiMo-shape K-quant port, not the engine's newer `q23_avx2.cpp`, not a
real GLM routing replay, and not an end-to-end tok/s result. The next comparison
must include that engine kernel before deciding which unpack/dot changes to port.
The 90% target is not met. Establish actual memory clocks/channel configuration
and a matching read/FMA reference before reporting theoretical-peak efficiency.

### Lessons from the newer Xeon/V100 checkout

The inspected source snapshot is `53ad731b63ab9a42bbb1533da1013662f39f08f0`.
Its `docs/GLM_FP4_V100_EXPERIMENTS.md` reports 32.10 decode tok/s and 520.89
prefill tok/s on dual Gold 6240 and two V100 PCIe 32 GB cards with a 181 GiB
allocation. This is the qualified MTP/suffix profile, not a 35 tok/s exact greedy
W4A16 result. The underlying `docs/GLM_FP4_V100.md` lists a 5.26 tok/s preserved
W4A16 reference, 19.08 tok/s Q8/route-drop greedy, and 31.02 tok/s with MTP and
suffix reuse before subsequent scheduling improvements. The fast profile allows
50% routing-mass dropping with renormalization; observed mean discarded mass was
38.57%. These quality-changing choices must remain explicit, opt-in experiments.

The transferable scheduling idea is CPU execution of nonresident experts,
overlapping GPU-resident expert work, rather than uploading every cache miss.
Its CPU implementation has bounded scale caches, fused worker phases and
verification-width specialization. Its Q8 dot dispatch uses AVX-512 VNNI or a
scalar fallback; TR16 lacks VNNI. A proper AVX2 integer-dot implementation is a
specific missing optimization to investigate alongside the exact W4A16 path.
Any port must preserve group-16 E4M3 scales; the standalone MXFP4 group-32
power-of-two-scale arithmetic cannot be substituted unchanged.

### Local-layout AVX2 W4A8 projection prototype

`src/kernels/cpu/fp4_local_q8.cpp` adds an opt-in CPU experiment for the local
checkpoint's adjacent-nibble E2M1 codes and K16 E4M3 scales. It does not reuse the
GGUF NVFP4 byte layout, change engine defaults, prune routes or require VNNI.
Inputs are already quantized signed bytes with FP32 scales per K64 and signed
sums per K16. Activation quantization is lossy relative to W4A16 and is outside
these timings. The exact W4A16 path remains the reference for model quality.

The strict AVX2 mode keeps the scalar K16 FMA order and matches its output bits.
Its unsigned offset lookup handles -128 activations without byte-negation
overflow. The separate tree-reduction option moves activation signs onto the
small weights, performs exact integer products, accumulates independent FP32
lanes and reduces once per output row. Two- and four-row variants reuse activation
loads. They change floating reduction order; their numerical gate is explicit.
All modes apply the weight's global FP32 scale last.

```sh
make -f experiment/glm_fp4_local/Makefile cpu-q8
build-fp4-local/glm-fp4-q8-cpu-test
OMP_NUM_THREADS=16 OMP_PROC_BIND=spread OMP_PLACES=cores \
  build-fp4-local/glm-fp4-q8-cpu-test "$PARTIAL_MODEL_DIR" --local-rows
```

The partial directory must contain layers 3 and 4, experts 0/1/2 and all three
projections. The experiment's existing partial-tree checkpoint supplies these
18 matrices (84,934,728 bytes of codes, expanded scales and globals). Loading,
CSF expansion, allocation and row-local first touch are outside timing. The
projection calls and OpenMP dispatch are timed. Each implementation receives one
warmup and seven rotated measurements. SIMD availability is checked at runtime;
the standalone benchmark executable itself is built for AVX2/FMA.

With 16 physical cores on the TR16, matched row-local allocation:

| Kernel | Median for 18 matrices | Weight+scale GB/s |
|---|---:|---:|
| Strict AVX2 | 3.433 ms | 24.74 |
| Tree, one row | 2.222 ms | 38.22 |
| Tree, two rows | 2.156 ms | 39.39 |
| Tree, four rows | 2.093 ms | 40.57 |

The four-row result is 1.64x the strict variant in this run. Row tiling did not
help the single-core case, so no automatic tile selection is established. The
worst real-matrix relative L2 difference from scalar Q8 was 1.8888e-7. These are
seeded quantized activations, not captured model activations or a full-model
quality test. Tests cover widths 1–8, K=64/128/2048/4096, partial-row guards,
all 2,032 finite E2M1/E4M3 combinations, -128/+127 activations, row-tile tails,
strict bit parity and independent FP64 bounds. The generic-dispatch UBSan build
also passed. No GPU backend was changed or executed.

Frozen output and hashes are in `results/20261019/cpu-q8/`. A subsequent matched
16-core, first-touch read reference measured 82.453 GB/s (80.682–83.698), using a
1 GiB array and seven post-warmup samples. The FP4 prototype reaches about 49%
of that measured reference, not 90%. It includes unpack and arithmetic, whereas
the reference only reads. This is not a memory-controller utilization counter.
The theoretical DDR bandwidth target is still unqualified: firmware access to
configured DIMM speeds was denied. If the historical DDR4-3200/four-channel
configuration is active, its raw arithmetic ceiling is 102.4 GB/s and 90% is
92.16 GB/s; those conditional numbers are not fresh hardware verification.

### Engine Q2/Q3 screen

The current `kq_cpu_bench` and `q23_test` were rebuilt. The ordinary and Q2 LUT
parity tests passed 5,952 scalar-dot checks, verification widths 1–4, partial
rows and extreme inputs. A separate synthetic streaming screen uses the engine
entry points with 117 expert blobs (1,066 MB), hidden width 4096, expert width
2048, one token, NUMA interleaving and CPU affinity 0–15. Three rotated process
rounds each report the median of seven timed passes; blob generation is outside
timing. It is not an actual GLM route replay or complete MoE activation chain.

| Existing option | Three round medians, GB/s | Median of medians |
|---|---|---:|
| GGML path | 50.2, 53.2, 51.1 | 51.1 |
| Native Q23 AVX2 | 32.3, 51.7, 52.0 | 51.7 |
| Native with Q2 LUT | 53.4, 51.8, 51.3 | 51.8 |
| Above with prefetch distance 1024 | 55.7, 31.4, 54.6 | 54.6 |

This screen is not qualified for changing defaults: CPU load varied and some
rounds overlapped prototype compilation. The large outliers are retained in the
raw evidence. Prefetch remains a candidate for a clean matched route replay,
not a demonstrated general improvement. Q23's roughly 51–55 GB/s is also short
of the 90% target. Next work must distinguish unpack/scale instruction costs,
worker scheduling and memory placement rather than treating these kernel rates
as full-model tok/s.

### Partial CPU chain and activation quantization

The CPU-only `cpu-chain` target includes input quantization, gate/up projections,
BF16-rounded clamped SiLU multiplication, hidden quantization, down projection
and fixed-order expert combination. It loads layers 3 and 4, three experts per
layer. Inputs, routing weights and residuals are synthetic. This is not a full
model decode benchmark and its reciprocal must not be reported as model tok/s.

```sh
make -f experiment/glm_fp4_local/Makefile cpu-chain cpu-q8
OMP_PROC_BIND=spread OMP_PLACES=cores \
  build-fp4-local/glm-fp4-cpu-chain "$PARTIAL_MODEL_DIR"
taskset -c 2 build-fp4-local/glm-fp4-q8-cpu-test --quant-bench
```

On the Threadripper 1950X with 16 physical cores, seven rotated samples of ten
chains each gave these medians:

| Dispatch and quantizer | Partial-chain time |
|---|---:|
| Separate teams, scalar quantizer | 2.39489 ms |
| Persistent team, scalar quantizer | 2.28764 ms |
| Persistent team, AVX2 quantizer | 2.28649 ms |

Persistent dispatch reduced elapsed time by about 4.5% in this run. AVX2
quantization did not measurably improve the whole chain. In a separate pinned
single-thread benchmark, 131,072 blocks took 8.1793 ms scalar and 5.33059 ms AVX2
(1.53x). Its 84,992-byte working set fits in L2; this is not streaming bandwidth.
CPU-time medians were 8.168 and 5.323 ms respectively.

Quantizer tests verify identical scales, signed bytes and group sums, including
ties-to-even, zero, subnormals, large finite inputs and nonfinite rejection
without output writes. Both dot-product and partial-chain undefined-behavior
sanitizer checks passed. Outputs match across dispatch and quantizer variants.
Tree reduction versus strict Q8 gives relative L2 0.000208805 after the complete
partial chain; neither is a quality comparison against W4A16 model inference.

The real-weight Q23 benchmark now repacks already-Q23 inputs for requested
packed layouts, and verifies pack/unpack bytes before timing. Previously only
requantized inputs were repacked. An optional `--fuse-quant` exposes the existing
owned-row fused quantization path. Pool and flow tests passed. Three captured
route positions matched output hashes across five row-layout scheduling modes.
Timing varied substantially, so no scheduling default changes are justified.

Background activity measured about 21% CPU busy with our benchmarks stopped;
eight-thread chain samples and Q23 scheduling comparisons were particularly
noisy. Raw samples, validation logs and source/binary hashes are preserved in
`results/20261019/cpu-chain/`. No GPU work or production default changed. The
next performance priority remains projection unpack/scale arithmetic and its
memory access, rather than further quantizer optimization.

### K64 projection experiment

The optional `wide_k` CPU argument processes 64 columns per iteration using
256-bit nibble unpacking. Two accumulators per row keep K16 scale groups in
their original lanes and avoid cross-lane weight assembly. It supports batch
one with tree reduction and row tiles 1/2/4; all earlier calls keep K32 behavior.
This changes floating-point reduction order and remains experimental.

The expanded projection test checks the new loop against independent FP64
bounds, row-tail guards, extreme signed activations and every finite code/scale
pair. The generic-dispatch UBSan build passed. On the 18 real matrices, the
maximum relative L2 across old and new tree variants remained 1.8888e-7 versus
scalar Q8. The existing partial-chain correctness check also passed after the
API extension; the chain continues using K32.

Two timing screens do not establish an improvement. At 16 cores, the first
screen measured K32 row-one at 61.47 GB/s and K64 row-one at 60.41 GB/s. A repeat
measured 43.14 and 40.09 GB/s respectively; K64 row-four reached 42.91 GB/s.
Background CPU contention remains, and the repeat additionally overlapped a
brief chain correctness check. Do not interpret the higher first-screen rate
as an improvement over older measurements or as verified DRAM utilization.
Raw samples and qualifications are in `results/20261019/cpu-wide/`. No production
default changed. Wider unpacking alone is insufficient to justify adopting
this variant on the 1950X.

### Fused hidden work and Q8_K chunks

The partial-chain benchmark now compares row tiles 1/2/4 and a fused worker
phase: after each 64-row gate/up job, the same worker computes SiLU and quantizes
that hidden chunk before the layer barrier. Down projection still waits for all
hidden chunks. This avoids a separate work-sharing phase and retains local
intermediate data. All seven variants produce identical partial-chain output
at 1/8/16 threads, and the updated chain passes UBSan.

In one noisy 16-core screen, persistent vector quantization with row-four took
2.71580 ms; fused row-four took 2.36146 ms. Row-one without fusion took 2.75027 ms
and fused row-one 2.36718 ms. Samples included large outliers, so these medians
identify a candidate, not a qualified speedup or production default. Results
are preserved under `results/20261019/cpu-fused/`.

For existing Q2/Q3 execution, `native_quant_h_rows` now uses the same AVX2/AVX1
Q8_K quantizer dispatch as `native_quant_h`. Canonical arithmetic and other
activation formats retain their existing paths. This affects the opt-in fused
hidden-chunk path. Chunk parity tests cover Q2_K, Q3_K and IQ3_S formats, interior
and empty ranges, output guards and the existing extreme/sparse/zero cases.
This dispatch change has no end-to-end throughput claim yet.

The subsequent `q8k_quant_parity --chunks-bench` isolates this dispatch on one
pinned 1950X core. Seven alternating samples, each with 16,384 quantizations of
2,048 hot input values, compare generic and native chunk calls. Every sample
matches the full-vector reference bytes.

| Chunk size | Generic time per 2,048 values | Native dispatch | Ratio |
|---|---:|---:|---:|
| 256 (used by the fused pool) | 2.808 us | 1.018 us | 2.76x |
| 512 | 2.825 us | 0.986 us | 2.87x |
| 2,048 | 2.807 us | 0.961 us | 2.92x |

A real-weight, two-layer Q23 screen used 15 workers, owned NUMA rows, fused
hidden quantization and the same captured top-eight routes. Three alternating
process rounds gave generic/AVX2 medians of 2.552/2.458, 2.717/2.499 and
2.411/2.724 ms. All output hashes matched. Background load remains and the
generic environment setting also disables initial activation SIMD quantization.
These results establish the primitive gain, but not a reliable whole-expert
speedup. Raw samples and provenance are in `results/20261019/q8-chunks/`.

### CPU chain row placement

`glm-fp4-cpu-chain PARTIAL_MODEL_DIR --placement-bench` compares loader-allocated
weights with fresh anonymous mappings first-touched by the persistent chain's
expert/64-row worker schedule. Huge pages are disabled for these experimental
mappings. Setup and byte-for-byte copy verification are outside timing; each
team size gets new mappings. The benchmark retains original weights as the
control, so its temporary memory footprint includes both copies.

The final placement validation sampled 1,296 pages at both 8 and 16 workers:
648 were on each NUMA node, with zero mismatches against the node of the worker
that copied each sampled row. This verifies sampled placement, not all pages
or actual DRAM traffic. CPU IDs are recorded because OpenMP core places may
schedule either SMT sibling of a physical core.

All complete-chain outputs and copied weights matched. Address and undefined-
behavior sanitizer checks passed with leak detection disabled; LeakSanitizer
could not run under this environment's tracing. The host CSF object was covered
by UBSan only. Sanitizer timings are not performance measurements.

Placement alone did not produce a consistent speedup under current contention.
At 16 workers, loader/placed row-one medians were 2.252/2.050 ms initially and
2.091/2.374 ms on repeat. Row-four measured 1.982/2.140 and 2.090/2.252 ms.
No placement default changes are justified. The benchmark now makes ownership
explicit for future comparisons; evidence is in `results/20261019/cpu-placement/`.

### Cache-resident instruction counters

`glm-fp4-q8-cpu-test --kernel-bench MODE REPEATS` runs one projection variant on
a synthetic 128x4096 matrix (294,916 weight/scale bytes). Modes 1/2/4 are strict
K32, tree K32 row-one and tree K32 row-four; modes 5/7 are K64 row-one/four.
Pin the process to one physical core and use `perf stat -e cycles:u,instructions:u`.
The benchmark reports both wall and process CPU time, checks output bounds
against scalar Q8, and performs no GPU work. Perf counts include setup/reference
checking; they are not counters restricted precisely to the timed loop.

On the 1950X, three rotated 2,048-repeat process runs found that the original
K64 row-four retired about 27% fewer instructions than K32 row-four, but used
only 3.3% fewer median cycles. K64 row-one retired fewer instructions than K32
row-one yet used more cycles. This cache-resident result does not establish
DRAM utilization. Generic cache-event names are preserved without interpreting
them as memory-controller counters.

A cycles profile attributed 99.47% of sampled cycles to K64 row-four. Inspection
found stack-resident accumulators in its inner loop. The experimental four-row
specialization now uses one accumulator per row, while smaller tiles retain
two. This changes reduction order within the already opt-in tree path. In three
alternating 4,096-repeat comparisons, median instructions fell 1.9% and median
cycles 1.4%; one candidate run regressed. Its synthetic relative L2 error was
3.39056e-7 versus scalar Q8. Unit bounds, extreme inputs, all finite code/scale
pairs and UBSan passed.

The real-weight follow-up still favored K32 row-four: 43.02 GB/s versus 40.81
GB/s for the revised K64 row-four at 16 workers. No general speedup is claimed;
the chain and production defaults do not use this variant. Raw counters,
samples, validation and hashes are in `results/20261019/cpu-counters/`. The next
kernel investigation must consider dependency chains and execution resources,
not instruction count alone.

### Multi-token projection measurements

The real-weight benchmark now accepts `--batch=1..8` and `--threads=1..32` in
addition to `--local-rows`. Multi-token runs compare scalar, strict SIMD and
tree SIMD; the single-token-only row tiles and K64 variants are excluded. Output
allocation, scalar parity and numerical bounds cover every token. Reports give
both complete batch time and time per token. For isolated counters,
`--kernel-bench MODE REPEATS BATCH` accepts widths 1..8 in strict/tree modes 1/2.

An initial 16-worker screen on the same 18 real matrices gave tree projection
times of 1.388, 1.960, 5.366 and 7.659 ms for widths 1/2/4/8, respectively. Per
token these are 1.388, 0.980, 1.342 and 0.957 ms. CPU contention varied across
the runs, so the nonmonotonic per-token figures are retained without claiming a
batching speedup. These are prequantized synthetic inputs. Real speculative
execution also depends on expert-route overlap, acceptance, attention and
drafting; these numbers cannot be used as delivered model tok/s.

A candidate loaded each token's activation inside the one-row dot loop rather
than keeping all activation vectors live. It matched the original kernel bits
in 72 UBSan comparisons across widths 1..8, K64/128/4096, signed extremes and
partial rows. Three noisy batch-eight real-weight comparisons favored it, but
pinned cache-resident counters showed 2.5% more instructions and about 1.1%
more median cycles. The candidate was rejected and the kernel restored exactly;
the broader benchmark remains. Evidence is in `results/20261019/cpu-batch/`.

### Exact FP16 scale-cache experiment

The optional CPU helper `fp4_local_half_scale` converts each valid E4M3 scale
to exact FP16 bits representing scale/2. All nonzero converted values are normal
FP16, so this loses no precision. The matching batch-one AVX2/FMA/F16C kernel
converts pairs of scale values and retains the existing K32 tree multiplication
and accumulation order. Invalid E4M3 codes are rejected. No checkpoint format,
loader behavior or production default changes.

The extra storage is one byte per K16 scale: expanded weight codes plus scales
grow from 84,934,728 to 94,371,912 bytes for the 18-matrix fixture, approximately
11.1%. The benchmark retains both scale representations for comparison, so its
temporary allocation is larger still. This percentage describes the expanded
CPU representation, not the compressed checkpoint's total size. Modes 8/9 of
the isolated kernel benchmark select FP16 scales with row tiles one/four.

Unit tests match the original tree output bits, including every finite scale,
signed activation extremes and partial rows. The real-weight benchmark also
checks exact tree-output parity for the scale-cache variants. UBSan and the
unchanged partial-chain check passed. CPUs without F16C skip these benchmark
modes; requesting the specialized API without support fails explicitly.

The tradeoff did not pay off on the 1950X. In three alternating pinned-core
counter rounds, median cycles rose from 765.03M to 773.62M for row-one and from
697.62M to 719.22M for row-four, despite fewer instructions. The initial real-
weight 16-worker screen measured 2.395/2.430 ms for byte/FP16 scales at row-one
and 2.480/2.692 ms at row-four. Background contention still applies. The cache
remains an isolated experiment and is not recommended for the chain or full
model. Evidence is in `results/20261019/cpu-half-scales/`.

### Size-preserving K32 CPU packing

`strata_glm_fp4_repack_k32` defines a reversible CPU layout in the shared C11
host code. Each 16-byte block stores values 0..15 in its low nibbles and values
16..31 in its high nibbles. Scales remain row-major E4M3. This lets the K32
kernel omit two byte-interleave operations per iteration without increasing
resident weight size. It is a separate in-memory representation, not a change
to the checkpoint format. Exact in-place conversion is supported; invalid
lengths, selectors and partial overlap are rejected before output writes.

The dedicated AVX2/FMA entry point preserves the original tree arithmetic order.
Tests cover inverse conversion, independent nibble layout checks, in-place use,
overlap rejection, token widths 1..8, signed extremes, partial output rows and
finite scale/code pairs. The real matrices round-trip exactly and their packed
kernel outputs match the original tree bits at widths one and eight. C11 host
and kernel UBSan checks passed.

In three alternating pinned-core counter rounds (4,096 repeats of a synthetic
128x4096 projection), median cycles changed as follows:

| Row tile | Adjacent nibbles | K32 packed | Reduction |
|---|---:|---:|---:|
| One | 764.43M | 713.42M | 6.7% |
| Four | 699.26M | 649.01M | 7.2% |

On the 18 real matrices with 16 workers, single-token medians changed from
2.147 to 2.038 ms for row-one and 2.230 to 2.048 ms for row-four. At width eight,
row-one changed from 10.476 to 10.183 ms. CPU contention still limits these
wall-clock comparisons; counters include setup and reference checking.

The chain's `--layout-bench` option converts weights in place outside timing,
warms each variant, then compares seven rotated samples of ten fused chains.
It verifies the complete output after each layout transition and after restoring
the original layout. At 16 workers, adjacent/packed medians were 1.850/1.783 ms
for row-one and 1.903/1.845 ms for row-four. Eight-worker row-four regressed, so
this is not a universal recommendation. The complete layout-transition chain
also passed UBSan. No default changed. Conversion cost is
excluded and no full-model tok/s claim follows from these partial chains.
Evidence and exact counter-run source/binary hashes are in
`results/20261019/cpu-planar/`.

### K32 conversion startup cost

The shared C11 converter now handles eight byte pairs directly instead of
extracting and inserting 32 individual nibbles per block. A temporary 16-byte
block preserves exact in-place operation. Validation and overlap rejection are
unchanged. Both versions were compiled with the same portable `cc -O2 -std=c11`
flags, without architecture-specific intrinsics.

`tools/glm_fp4_pack_bench.py` compares two shared libraries on a 128 MiB buffer,
first-touched by pinned physical-core workers. It uses seven alternating samples
after warmup and checks every output byte against an independent repeated
all-byte-value pattern outside timing. Thread-pool dispatch and affinity calls
are timed. Rates count payload bytes once, not total memory read/write traffic.

| Workers | Original pack time | Revised pack time | Revised payload rate |
|---|---:|---:|---:|
| 1 | 396.759 ms | 15.460 ms | 8.68 GB/s |
| 8 | 57.559 ms | 9.665 ms | 13.89 GB/s |
| 16 | 56.505 ms | 7.088 ms | 18.94 GB/s |

Reverse conversion improved as well: at 16 workers, 52.739 to 8.054 ms.
Eight host tests passed normally and under UBSan; the kernel tests and complete
chain layout-transition checks passed after rebuilding. These are startup
conversion results, not an additional decode speedup or a full-model loading
estimate. Background contention remains uncontrolled. Raw samples, library and
source hashes are in `results/20261019/cpu-pack-startup/`.

### Packed K64 candidate under contention

The packed CPU API now has an optional `wide_k` selector for batch one. Its
256-bit load consumes two packed K32 blocks directly, removing the K64 loop's
byte interleaves as well. It uses the existing K64 reduction order; repacking
does not change those output bits, but K64 and K32 have different floating
reduction orders. No resident storage is added. Kernel-benchmark modes 12/13
select packed K64 with row tiles one/four.

Unit checks and UBSan passed, including exact parity with matching adjacent-
layout K64 kernels, partial rows, signed extremes and finite scale/code pairs.
The real 18-matrix test also checks that matching-K64 output bits are unchanged.

Three pinned-core counter rounds measured 777.28M median cycles for packed K32
row-four and 702.54M for packed K64 row-four, a 9.6% reduction in this screen.
Packed K64 row-one instead took more cycles than packed K32 row-one. These are
not qualified performance recommendations: full-core samples varied from about
2 ms to 63 ms. With our benchmarks stopped, a subsequent three-second sample
showed 85.3% host CPU busy and a one-minute load average of 30.78. The candidate
remains opt-in; neither the chain nor production defaults use it. Raw counters,
all outliers, source/binary hashes and validation are preserved in
`results/20261019/cpu-planar-wide/`.

### Q8 activation error against a W4A16 arithmetic reference

`glm-fp4-cpu-chain PARTIAL_MODEL_DIR --quality-check` runs a CPU-only numerical
comparison on the two real-weight layers and three experts per layer. The new
reference rounds inputs to BF16, decodes weights without Q8 activation
quantization, accumulates in FP64 and applies the global scale last. Every finite
E2M1/E4M3 weight product is checked to be exactly representable in BF16. Closed-
form fixtures check scale/global arithmetic, BF16 ties and signs. This is a
numerical oracle, not an emulator of GPU WMMA accumulation order.

Four deterministic synthetic input patterns are tested: sine, seeded normal,
normal with one value of 16 per K64 block, and sparse normal with one nonzero per
block. Input hashes are recorded. Both paths retain the prototype's BF16-rounded
clamped SiLU, uniform expert combination and synthetic residual. They do not run
attention or a real router, and residual averaging can hide expert-output error.

For the fast Q8 tree path versus the BF16-input reference:

| Input | Layer 3 expert-down relative L2 | Layer 4 expert-down relative L2 | Final residual relative L2 |
|---|---:|---:|---:|
| Sine | 1.185% | 1.164% | 0.135% |
| Normal | 1.415% | 1.411% | 0.165% |
| Outlier | 2.792% | 2.558% | 1.188% |
| Sparse | 0.887% | 1.068% | 0.00789% |

Strict Q8 and tree Q8 are reported separately in the raw log; the latter also
changes FP32 reduction order. These comparisons therefore include quantization
and accumulation effects. K32 repacking remains bit-identical to the adjacent
Q8 tree output for every pattern. No threshold is asserted as acceptable model
quality: real activation traces, full-model perplexity and task evaluations are
still needed. The reported pass checks finiteness, oracle fixtures and packing
parity, not quality acceptance. The comparison passed UBSan with the current
host converter and CPU kernels, and the unchanged default chain checks passed.
Evidence is in `results/20261019/cpu-quality/`.


### CPU W4A16 without activation quantization

The experimental `cpu-w4a16` Makefile target builds scalar and runtime-dispatched
AVX2/FMA FP4 GEMV checks. Its inputs are finite F32 buffers already rounded to
BF16 by the caller, with validated finite E4M3 scale codes. Weights are decoded
exactly before FP32 accumulation; global scale is applied last. This removes Q8
activation error but does not emulate GPU WMMA accumulation order. The production
engine and default Q8 chain are unchanged.

The two-layer, three-expert partial chain now supports W4A16 and
`--precision-bench`. On the synthetic outlier input, final residual relative L2
error against the FP64 oracle fell from 1.188% for Q8 to 0.00630% for W4A16.
Sine and normal inputs had identical final BF16 residuals to the oracle; sparse
input relative L2 was 0.000216%. This is a numerical experiment, not full-model
quality acceptance.

The initial seven-sample timing screen on Threadripper 1950X measured 2.191 ms
for Q8 planar row-one and 3.479 ms for W4A16 planar row-one at 16 workers.
Host load was uncontrolled; these are partial-chain timings, not decode tok/s.
Scalar/SIMD FP64 rounding bounds, adjacent/planar parity, all finite scale/code
pairs, outliers and row guards passed. Recovered AddressSanitizer/UBSan logs
contain PASS, with leak detection disabled; the originating exit status was not
recovered. Evidence and source hashes are in `results/20261019/cpu-w4a16/`.


A subsequent two-row W4A16 candidate reused activation loads across rows while
preserving each row's FP32 reduction order. All unit checks and partial-chain
output parity passed. It regressed in the paired screen: at eight workers,
row-one/row-two medians were 3.279/4.418 ms; at sixteen, 1.838/2.090 ms.
The candidate was discarded and all four edited source files restored to their
recorded baseline hashes. The same screen's Q8 row-one median was 1.287 ms at
sixteen workers. Absolute times differed substantially from the earlier screen;
use within-run comparisons, not cross-run speedup claims. Evidence is in
`results/20261019/cpu-w4a16-r2/`. The source of the regression has not been
isolated; reduced activation loads alone did not improve this kernel.


### W4A16 loop indexing and counter benchmark

`glm-fp4-w4a16-cpu-test --kernel-bench planar 16384` now measures a deterministic
128x4096 cache-resident matrix, with warmup, adjacent/planar output checks and
CPU/wall time. `adjacent` selects the original nibble layout. Counter runs include
startup and reference setup; this benchmark is not a DRAM-bandwidth measurement.

Disassembly showed repeated index shifts and sign extensions in every K32
iteration. Replacing the column index with a block index preserves the weight
loads and accumulation order while reducing that address work. On the
Threadripper 1950X, pinned to CPU 0, five alternating 4096-call samples reduced
median instructions from 2.646B to 2.309B (12.7%) and cycles from 1.168B to 1.139B
(2.5%). Three longer alternating 16384-call samples gave median cycles of
4.672B versus 4.567B (2.2%) and instructions of 10.537B versus 9.189B (12.8%).
Host load was uncontrolled, and all samples, including outliers, are retained.

The simpler loop is retained in the experimental CPU kernel. No production
backend or default precision changed. Normal and UBSan kernel checks passed;
the partial-chain quality log is byte-identical to the previous W4A16 log.
Three alternating full partial-chain screens had large timing variation and do
not establish a multi-worker speedup. Neither these counters nor the chain
establish full-model throughput or 90% of a hardware peak. Evidence, source and
binary hashes are in `results/20261019/cpu-w4a16-index/`.


### Rejected FP16-bit magnitude decoder

A CPU candidate replaced signed-byte-to-FP32 conversion with a shuffle table of
FP16 upper bytes, byte-to-word expansion, an eight-bit shift and F16C conversion.
All signed doubled E2M1 magnitudes are exactly representable in FP16. The scale
multiplication and FMA order stayed unchanged. Unit tests passed, and a new
diagnostic digest over all output bit patterns matched the baseline. The digest
is useful for regression comparison, not proof of universal numerical parity.

Five alternating pinned-CPU-0 screens on Threadripper 1950X, each with 8192
calls on a cache-resident 128x4096 matrix, measured median cycles of 2.272B for
the retained integer conversion and 2.611B for FP16 conversion (15.0% slower).
Instructions rose from 4.602B to 5.274B (14.6%). Background load was uncontrolled;
raw samples are retained. The extra unpack/shift sequence did not pay off on
this CPU. The candidate was discarded, restoring the previous kernel exactly;
the output digest remains in the test tool. This does not change production
precision, GPU behavior or model decode performance. Evidence and hashes are in
`results/20261019/cpu-w4a16-halfbits/`.


### Q2/Q3 cache-resident kernel screen

`q23_test --bench TYPE TOKENS REPEATS LAYOUT` adds a CPU-only benchmark to the
existing test executable: TYPE is 10 (Q2_K) or 11 (Q3_K), TOKENS is 1–4, and
LAYOUT is row/direct/lut. It checks the 128x4096 quantized matrix against an
independent decoded-weight FP64 dot before timing, then checks bitwise output
stability. Packing, activation quantization and reference work are outside the
reported CPU/wall time. Process-wide perf counters include that setup and must
not be interpreted as kernel-only counters or used to infer batching benefits.

On Threadripper 1950X, pinned CPU 0, three rounds of 1024 calls gave these median
CPU times (milliseconds). Inputs are synthetic, data are cache-resident, and
background load was uncontrolled:

| Format / tokens | Row | Packed direct | Packed LUT |
|---|---:|---:|---:|
| Q2 / 1 | 29.244 | 104.055 | 108.310 |
| Q2 / 2 | 54.202 | 152.528 | 165.568 |
| Q2 / 4 | 112.567 | 264.988 | 303.480 |
| Q3 / 1 | 41.014 | 133.284 | 116.306 |
| Q3 / 2 | 57.606 | 197.418 | 176.870 |
| Q3 / 4 | 109.428 | 332.046 | 331.991 |

This screen favors ordinary rows. Four-token work reduces CPU time per token
by about 4% for Q2 and 33% for Q3 relative to single-token calls in this screen.
It does not establish DRAM bandwidth or full-model speculative throughput.

The separate `STRATA_Q2K_LUT=1` option applies to ordinary Q2 rows, not the packed
LUT layout. Three alternating 4096-call comparisons measured:

| Tokens | Default row CPU ms | Row-LUT CPU ms | Row-LUT change |
|---|---:|---:|---:|
| 1 | 115.591 | 118.315 | 2.4% slower |
| 2 | 210.617 | 202.013 | 4.1% faster |
| 4 | 428.411 | 375.845 | 12.3% faster |

The existing option remains off by default. It is a candidate for multi-token
CPU verification, not a single-token decode recommendation. Both unit-test modes
passed and their complete output dumps matched byte-for-byte. No kernel source
or production dispatch changed. Model activation replay was not performed;
these measurements use synthetic inputs. Evidence and hashes are in
`results/20261019/q23-kernel-screen/`.


### Q2/Q3 weight-ring screen beyond cache capacity

The benchmark accepts an optional final WEIGHT_COPIES argument (1–1024).
It allocates distinct physical copies of the selected packed matrix, validates
all copies, warms them, then cycles through them during timing. Values repeat;
addresses do not. First touch and execution are pinned to CPU 0 in this screen.
For example, `q23_test --bench 10 4 4096 row 1024` uses a 176,160,768-byte Q2
weight ring. Q3 uses 230,686,720 bytes. This is synthetic kernel work, not a
captured route or activation replay. No GPU is used.

Three alternating forward/reverse case rounds on Threadripper 1950X gave these
median CPU times for 4096 calls with 1024 copies:

| Format / tokens | Default row CPU ms | Q2 row-LUT CPU ms |
|---|---:|---:|
| Q2 / 1 | 118.958 | 124.451 |
| Q2 / 4 | 448.317 | 367.285 |
| Q3 / 1 | 166.060 | — |
| Q3 / 4 | 445.432 | — |

Q2 row-LUT was 18.1% faster for four tokens and 4.6% slower for one token in this
screen. The paired cache-resident four-token results varied strongly: default
429.9–506.4 ms, row-LUT 365.0–523.2 ms. Background load was uncontrolled; do not
promote a universal speedup or default from these results. The streaming result
makes row-LUT a candidate for real multi-token CPU service replay.

`payload_GB_s` counts packed bytes once per call divided by wall time; it is not
measured DRAM traffic or a hardware peak. No claim of 90% utilization follows.
Existing unit tests passed; extra checks covered Q2/Q3, all three layouts,
widths one/four, two physical copies, every-copy output parity and invalid copy
bounds. Production kernels and defaults are unchanged. Full records and source
hashes are in `results/20261019/q23-stream/`.


### Q2 row-LUT with the CPU expert pool

`q23_pool_test --bench TOKENS WORKERS REPEATS` now runs a synthetic CPU expert
layer using production pool calls and NUMA-owned rows. The benchmark uses eight
experts, H4096/F2048, Q2 gate/up and Q3 down, 72,876,032 packed weight bytes,
and uniform routing weights. Gate/up, SiLU, hidden Q8 quantization, down and
routed reduction are timed. Initial input quantization, packing and NUMA
placement are excluded. The benchmark does not execute attention or a router.

On Threadripper 1950X with 15 workers plus the pinned host, local NUMA scheduling
and prefetch off, three alternating process comparisons produced these medians
of seven timed samples (milliseconds):

| Tokens / mode | Process 1 | Process 2 | Process 3 | Median of medians |
|---|---:|---:|---:|---:|
| 1 / default | 1.233 | 1.284 | 1.227 | 1.233 |
| 1 / row-LUT | 1.312 | 10.020 | 1.218 | 1.312 |
| 4 / default | 5.669 | 4.534 | 5.053 | 5.053 |
| 4 / row-LUT | 4.563 | 4.566 | 4.597 | 4.566 |

The four-token median is 9.6% lower with row-LUT, but one paired comparison
slightly regressed. The single-token results include a large row-LUT outlier.
Background load was uncontrolled. This supports continued evaluation rather
than changing a default; neither single-token nor full-model decode improvement
is established. The earlier single-core 18.1% result does not transfer directly
to this complete CPU expert layer.

The existing pool test passed. Timed expert and reduced outputs were checked
byte-for-byte against warmup; output hashes agreed across the two modes and all
processes for each width. Production kernels and dispatch remain unchanged.
No GPU or IB work was run. Raw samples and hashes are in
`results/20261019/q23-pool-screen/`.


### Reusable CPU W4A16 projection

`strata::kernels::cpu::Fp4LocalCpu` provides a host-resident projection API for
partial-model execution. It owns one planar packed code/scale representation,
rounds finite inputs to BF16, runs the existing FP32-accumulating W4A16 kernel,
and applies global scale last. K16 tails are zero-padded to K32. No original
weight copy or expanded BF16 weights are retained. Constructor thread count is
explicit (1–32, default one); calls accept 1–16 tokens. Tokens currently use
independent dot products, not a weight-reusing batched kernel.

This is groundwork for CPU experts in the distributed runner; it is not yet
connected to node scheduling, cache-miss dispatch or production decode. GPU
classes and defaults are unchanged. The CPU arithmetic does not claim GPU WMMA
bit equivalence or model-quality acceptance. Optional `kernel_ms` includes
OpenMP dispatch and excludes rounding, allocations and FP64 verification.

Build with `make -f experiment/glm_fp4_local/Makefile cpu-projection`, then run
`glm-fp4-cpu-projection-test PARTIAL_MODEL_DIR`. The normal test passed 36
synthetic shape/batch cases and 12 real-weight cases from layers 3–4, expert 0,
all three projections. Checks include independent FP64 dots using original
adjacent-layout weights, internal packed references, bitwise single/batched and
one/four-thread parity, owned-weight lifetime, invalid scale/extents, nonfinite
inputs and BF16 overflow rejection. Synthetic tests also passed AddressSanitizer
and UBSan with the host converter instrumented; leak detection was disabled.
No performance measurement is claimed. Evidence is in
`results/20261019/cpu-projection/`.


### CPU backend in the three-role partial-model runner

`make -f experiment/glm_fp4_local/Makefile tree-cpu` builds `glm-fp4-tree-cpu`
without CUDA or HIP runtime dependencies. The same bounded tree protocol now
executes complete FP4 gate/up–SiLU–down experts through `Fp4LocalCpu`, with four
threads per projection and FP64 verification enabled. It caches host weights
once and supports the existing direct and relay routing cases. This is a separate
binary; the default GPU backend and arithmetic are unchanged. The GPU-only
`tiled` selector is rejected by the CPU binary before networking begins.

An explicit `udp` argument selects the existing UDP transport; omitting it still
selects IB. For local validation without GPU use, run from the repository root:

```sh
python3 tools/test_glm_fp4_cpu_tree.py PARTIAL_MODEL_DIR
```

Three local CPU processes on Threadripper 1950X passed both direct and relay
UDP tests using real weights from layers 3–4, experts 0–2. Each topology passed
16 layer checks with zero relative L2 against the local reference, batch sizes
1/7/16/3, and bit-identical repeated residuals. The complete 884,736-byte output
dumps matched across topologies. Raw dumps stay in the ignored build directory;
shared evidence records their hash. Three malformed-command tests also passed.

CPU, HIP and CUDA tree binaries built and linked. GPU execution was skipped;
no new cross-backend parity claim is made. These local tests do not validate
cross-machine IB, mixed CPU/GPU execution, automatic CPU cache-miss dispatch,
CPU/GPU overlap, model quality or full-model throughput. They establish that the
CPU W4A16 backend can execute the partial-model distributed protocol. Evidence
and hashes are in `results/20261019/cpu-tree-local/`.


### CPU W4A16 across three physical machines over IB

The CPU node now passes both direct and relay tests on the three configured
machines over QDR IB. Each worker was built from the same 24-file source package
in a separate experiment directory, avoiding runtime incompatibility with the
older worker compiler. Source-package hashes were verified remotely, and both
worker binary hashes and compiler versions were recorded. No GPU was used.

Each topology passed 16 layer checks using layers 3–4, experts 0–2 and batches
1/7/16/3. Relative L2 against the primary's local CPU reference was zero on every
check; repeated outputs were bit-identical. The full 884,736-byte direct IB,
relay IB and earlier local UDP output dumps match. All three node processes
exited normally in each successful run. Raw output dumps remain local; the
shared report contains their hash.

An initial direct attempt failed before any layer test because the leaf used
port 1. The existing direct topology requires leaf port 2; relay uses leaf port
1. The corrected mapping passed. This is a run-configuration error, not evidence
of a kernel failure. Both failed workers had exited by the cleanup check.

The launcher accepts `--ssh-config FILE` for explicit local connection settings,
and `--worker-binary` can select the separately built CPU node. The primary uses
`--primary-binary build-fp4-local/glm-fp4-tree-cpu`; actual aliases, endpoints and
remote experiment directories remain in ignored local configuration. IB remains
the default transport. HIP and CUDA targets also built and linked, without GPU
execution.

This validates CPU expert execution on the physical distributed topology. It
does not yet implement CPU/GPU overlap, a cache-miss scheduler or full-model
inference. The runs include FP64 verification, startup and deliberate sleeps;
their elapsed times are not decode throughput measurements. Evidence is in
`results/20261019/cpu-tree-ib/`.


### CPU projection thread-count measurements

`glm-fp4-cpu-projection-test --bench PARTIAL_MODEL_DIR` measures API wall time
and kernel time separately. It runs six real matrices (layers 3–4, expert 0,
gate/up/down), deterministic synthetic inputs and seven rotated samples per
thread count. Each sample establishes its OpenMP team size before timing.
Packing and FP64 checks are outside timing; every measured output is checked
byte-for-byte against a verified reference. Kernels include OpenMP dispatch;
API time also includes allocation, BF16 rounding and finite-output checks.

On Threadripper 1950X with spread/core binding, the sum of six projection medians
was as follows (milliseconds):

| Tokens | 1 thread | 4 threads | 16 threads |
|---|---:|---:|---:|
| 1 | 9.407 | 3.315 | 1.037 |
| 4 | 40.668 | 11.759 | 4.267 |

These are warm-cache projection measurements under uncontrolled background
load. Summing independent medians is not an end-to-end expert-chain measurement.
No DRAM-bandwidth, full-model throughput or peak-utilization claim follows.

The CPU node now accepts an explicit `threads=N` selector (1–32); the launcher
exposes `--primary-cpu-threads N` without changing worker settings. The default
remains four threads. A three-machine direct IB test with a 16-thread primary
and four-thread workers passed all 16 layer checks and produced a byte-identical
output dump to the previous four-thread-primary run. The 16-thread primary is
therefore an available option for the CPU partial runner, not a new production
default. Invalid counts are rejected before networking. CPU/HIP/CUDA binaries
built and linked; GPU execution was not run. Evidence is in
`results/20261019/cpu-projection-threads/`.


### Paired CPU gate/up projections

`Fp4LocalCpu::run_pair` computes two projections with one BF16 input-rounding
pass and one OpenMP dispatch. It keeps each projection's own packed weights,
global scale and output rows; matching input widths and thread counts are
required. Dot-product arithmetic and accumulation order are unchanged. The
original `run` method and node defaults remain unchanged; nodes do not yet select
the paired API.

`glm-fp4-cpu-projection-test --pair-bench PARTIAL_MODEL_DIR` alternates separate
and paired calls on real gate/up weights from layers 3–4, expert 0. At 16 threads
on Threadripper 1950X, the sum of the two one-token gate/up medians fell from
0.611 to 0.580 ms in one screen and from 0.725 to 0.656 ms in a repeat (5.0% and
9.5%). Four-thread one-token results were +2.3% and -0.5%; four-token 16-thread
results varied from 12.7% to 1.2% faster. Background load was uncontrolled, and
these sums are not expert-chain timings. The API remains an explicit candidate,
not a default or a full-model decode speedup claim.

All measured outputs matched separately verified projections byte-for-byte.
Synthetic tests cover K16 padding, unequal row counts/global scales, batch widths
1/3/16 and rejected incompatible input widths, thread counts and token extents.
AddressSanitizer and UBSan checks passed with leak detection disabled. The CPU
tree target rebuilt; no GPU files changed or GPU execution ran. Evidence is in
`results/20261019/cpu-pair/`.


### Complete CPU expert and paired-node validation

`Fp4LocalCpuExpert` owns gate/up/down projections and implements the partial
runner's BF16-rounded, clamp-10 SiLU. Separate projection calls remain its
default. `run(..., paired=true)` explicitly selects paired gate/up. The CPU node
now uses this reusable expert; `paired` enables pairing, and the launcher exposes
`--primary-cpu-paired` without changing worker options. GPU expert code and the
four-thread separate CPU default remain unchanged.

`glm-fp4-cpu-projection-test --expert-bench PARTIAL_MODEL_DIR` measures complete
single-expert execution, including SiLU and down, with verification outside the
timed region. In one nine-sample alternating screen on Threadripper 1950X,
16-thread medians were:

| Layer / tokens | Separate ms | Paired ms |
|---|---:|---:|
| 3 / 1 | 0.436 | 0.409 |
| 4 / 1 | 0.474 | 0.473 |
| 3 / 4 | 2.031 | 1.969 |
| 4 / 4 | 1.963 | 1.941 |

These small, variable savings do not establish a general speedup; four-thread
results also included a slight regression. Weights are real but activations are
synthetic, background load was uncontrolled, and routing/reduction/attention are
excluded. Pairing remains opt-in.

Normal and ASan/UBSan tests passed analytic zero, positive and both clamp
conditions across threads 1/4, batches 1/3/16 and both pairing modes. Incompatible
shapes are rejected. Leak detection was disabled. All measured real expert
outputs matched verified references byte-for-byte.

A direct three-machine IB run with a 16-thread paired primary and unchanged
four-thread separate workers passed all 16 layer checks at zero relative L2.
Its final dump matches the original CPU IB baseline byte-for-byte. CPU, HIP and
CUDA node builds passed; GPUs were not run. The integration is still a bounded
partial-model runner, not a cache-miss scheduler or full-model decoder. Evidence
is in `results/20261019/cpu-expert-paired/`.


### Fused CPU expert with one team across phases

`Fp4LocalCpuExpert::run_fused` keeps one OpenMP team for the complete expert call.
Each gate/up row job computes its BF16 clamp-10 SiLU immediately, writing padded
hidden rows. An implicit barrier precedes the down-projection jobs. This shares
input rounding and dispatch, parallelizes SiLU, and avoids copying/re-rounding
the already-BF16 hidden input. It retains per-call scratch allocations; it is not
a persistent decode-wide worker pool. Projection accumulation order is unchanged.

The mode is opt-in: the CPU node accepts `fused`, and the launcher exposes
`--primary-cpu-fused`. It is mutually exclusive with `paired`. The default remains
four-thread separate execution. In the repeat nine-sample screen on Threadripper
1950X, 16-thread complete-expert medians were:

| Layer / tokens | Separate ms | Paired ms | Fused ms |
|---|---:|---:|---:|
| 3 / 1 | 0.429 | 0.406 | 0.388 |
| 4 / 1 | 0.482 | 0.479 | 0.432 |
| 3 / 4 | 1.780 | 1.700 | 1.588 |
| 4 / 4 | 2.073 | 2.004 | 1.902 |

This is an 8–11% reduction versus separate execution in that repeat. Background
load was uncontrolled, inputs synthetic and weights real; the earlier screen
included a single-token case slower than paired execution. Keep these as
candidate measurements, not a full-model speedup or peak-utilization claim.

Normal and ASan/UBSan tests passed zero/clamp cases, padded H16/48/80 geometry,
batches 1/3/16, threads 1/4, exact output parity, input rejection and nonfinite
intermediate rejection. Leak detection was disabled. All real-weight measured
outputs matched separately verified expert outputs.

A 16-thread fused primary passed all 16 direct IB layer checks with unchanged
four-thread separate workers. Relative L2 was zero and the entire output dump
matches the original CPU IB baseline byte-for-byte. CPU/HIP/CUDA builds passed;
no GPUs were run. Evidence is in `results/20261019/cpu-expert-fused/`.


### Two-token CPU W4A16 weight reuse

The optional low-level `fp4_local_w4a16_batch_rows` entry point decodes each K32
weight block once for two tokens. It keeps each token's original FMA and reduction
order, handles an odd final token with the original kernel, and retains scalar
fallback. Existing projection and expert dispatch do not select it yet.

On Threadripper 1950X, pinned to logical CPU 0, synthetic K4096/N128 planar weights
and BF16-rounded inputs gave these median CPU times per call. Each mode ran 1024
calls in three separate processes, with alternating mode order:

| Tokens | Separate ms | Reuse ms | CPU time reduction |
|---|---:|---:|---:|
| 2 | 0.1624 | 0.0945 | 41.8% |
| 3 | 0.2394 | 0.1780 | 25.7% |
| 4 | 0.3162 | 0.1924 | 39.1% |
| 7 | 0.5517 | 0.3579 | 35.1% |
| 16 | 1.2647 | 0.7572 | 40.1% |

This is a cache-resident primitive screen under uncontrolled background load,
not a DRAM bandwidth measurement or full-model decode result. It provides no
single-token speedup and needs multi-token expert integration and measurement
before any speculation throughput claim.

Scalar/SIMD and adjacent/planar tests compare batches 1/2/3/4/7/16 byte-for-byte
against original single-token calls, including untouched output guards. Invalid
token extents and empty row ranges are checked. Normal and ASan/UBSan tests passed
(leak detection disabled). GPUs were not run. Reproduce the timing with
`taskset -c 0 build-fp4-local/glm-fp4-w4a16-cpu-test --batch-bench reuse 4 1024`
and replace `reuse` with `separate` for the baseline. Compact evidence and source
hashes are in `results/20261019/cpu-w4a16-batch/`.


### Weight reuse in the complete CPU expert

`run_fused(input,tokens,verify,reuse_weights=true)` now groups two tokens per
row job for gate, up and down. SiLU processes both token rows before the existing
team barrier. The original fused dispatch remains the default. Odd batches use
the original single-token kernel for the tail; padded K16 geometry is supported.

The repeat screen on Threadripper 1950X, 16 threads, real layer 3/4 expert-0
weights and synthetic activations measured the following nine-sample medians.
Timing includes input rounding, scratch allocation and the complete expert.
Verification runs outside timing and all timed results match the reference bytes.

| Layer / tokens | Fused ms | Reuse ms | Reduction |
|---|---:|---:|---:|
| 3 / 1 | 0.4879 | 0.5020 | -2.9% |
| 3 / 2 | 1.0107 | 0.6367 | 37.0% |
| 3 / 3 | 1.5223 | 1.2589 | 17.3% |
| 3 / 4 | 2.1457 | 1.3622 | 36.5% |
| 3 / 7 | 3.6451 | 2.6313 | 27.8% |
| 3 / 16 | 29.0499 | 18.3308 | 36.9% |
| 4 / 1 | 0.5233 | 0.5326 | -1.8% |
| 4 / 2 | 1.1100 | 0.7167 | 35.4% |
| 4 / 3 | 1.5441 | 1.1768 | 23.8% |
| 4 / 4 | 2.0953 | 1.3206 | 37.0% |
| 4 / 7 | 3.6412 | 2.6404 | 27.5% |
| 4 / 16 | 8.4531 | 5.3544 | 36.7% |

Background load was uncontrolled. The first screen also overlapped a sanitizer
build/run; the repeat had no agent-started concurrent build. These are candidate
expert timings, not full-model throughput or proof of 90% peak utilization.

Normal and ASan/UBSan tests passed (leak detection disabled), including exact
reference parity at padded H16/48/80 and batches 1/2/3/4/7/16 with 1/4 threads.
The CPU node rebuilt successfully, but no node flag selects this new mode yet.
No GPU kernel changed and GPUs were not run. Evidence is in
`results/20261019/cpu-expert-reuse/`.


### CPU W4A16 streaming weight screen

The low-level batch benchmark accepts an optional `WEIGHT_COPIES` argument
(1–1024). Each copy has separate storage, with identical deterministic values;
initialization is outside timing. The timed loop rotates through copies. One
copy is 294,912 packed code/scale bytes; 1024 copies occupy 301,989,888 bytes.

A Threadripper 1950X CPU-0 screen used three interleaved process rounds of 8192
calls, synthetic K4096/N128 weights, and batches 1/2/4. Median CPU ms per call:

| Copies | Tokens | Separate ms | Reuse ms | Reduction |
|---|---:|---:|---:|---:|
| 1 | 1 | 0.0827 | 0.0791 | 4.4% |
| 1 | 2 | 0.1701 | 0.1049 | 38.4% |
| 1 | 4 | 0.3179 | 0.1889 | 40.6% |
| 1024 | 1 | 0.0863 | 0.0798 | 7.6% |
| 1024 | 2 | 0.1611 | 0.1044 | 35.2% |
| 1024 | 4 | 0.3194 | 0.1905 | 40.3% |

Background load and clock variation remain uncontrolled. The ring test shows
whether the reuse benefit survives a larger working set; it does not measure
hardware DRAM transactions, establish a bandwidth roof, or imply full-model
decode throughput. Final outputs matched the original single-token reference.
Normal kernel tests and invalid-copy-bound checks passed; kernels and production
dispatch did not change. No GPU execution ran.

Reproduce with `taskset -c 0 build-fp4-local/glm-fp4-w4a16-cpu-test
--batch-bench reuse 4 8192 1024` (one shell line), then compare `separate`.
Evidence is in `results/20261019/cpu-w4a16-stream/`.


### Four-token CPU weight reuse

The opt-in batched W4A16 path now decodes each K32 block for four tokens,
followed by two-token and single-token tails. Each token keeps the same FMA
and reduction order. The fused expert's explicit `reuse_weights=true` mode
uses four-token row jobs. Default single-token and non-reuse dispatch remain
unchanged. The planar four-token inner loop generated by GCC 15.2 has no vector
stack spills in the inspected assembly.

Three alternating CPU-0 process rounds on Threadripper 1950X used 2048 calls
per mode over 1024 physical copies of a synthetic K4096/N128 matrix (302 MB).
Four-token reuse reduced median CPU time versus two-token reuse by 27.1% at
batch 4, 25.1% at batch 7 and 10.4% at batch 16.

In a subsequent real-weight complete-expert screen, nine rotated timings per
mode at 16 threads gave these batch-4 medians:

| Layer | Fused without reuse ms | Four-token reuse ms | Reduction |
|---|---:|---:|---:|
| 3 | 2.0094 | 1.0085 | 49.8% |
| 4 | 2.1198 | 1.0659 | 49.7% |

Background load was uncontrolled and odd-batch results were mixed: layer 4
batch 7 regressed 16.7% versus fused without reuse. This is not a universal
performance recommendation or full-model throughput result. Single-token
execution gains no weight reuse.

Normal kernel and expert tests passed exact reference parity. ASan/UBSan
expert tests passed with leak detection disabled, including padded dimensions
and batches 1/2/3/4/7/16. The CPU node rebuilt; no GPU execution ran and no node
flag selects reuse yet. Raw compact timing/test evidence and hashes are in
`results/20261019/cpu-w4a16-four/`.


### Rejected Q2 lookup-loop unrolling

The existing Q2/Q3 row kernels already unpack each chunk once across tokens.
An isolated candidate forced eight-way unrolling of the optional Q2 lookup
loop (`STRATA_Q2K_LUT=1`). Production source was not edited. Both candidates
were built from current source against the same CPU libraries. Normal dump
tests matched byte-for-byte across Q2/Q3 cases.

On Threadripper 1950X, three alternating CPU-0 screens over a 176 MB weight
ring gave batch-4 median CPU time 370.012 ms baseline and 360.650 ms candidate
for 4096 calls (2.5% lower). Single-token results were noisy; the compiler
already unrolls that specialization. The batch-4 function grew from 1724 to
3172 bytes. Three longer cache-resident counter screens measured median
7.5445B to 7.3851B instructions and 6.1853B to 5.2061B cycles. These counters
include process setup/reference work, and background load was uncontrolled.

The complete synthetic expert-layer screen used 15 workers plus the host,
eight experts, H4096/F2048, Q2 gate/up, Q3 down, batch 4 and 15 samples per
process. It includes SiLU, hidden quantization and routed reduction, while
input quantization and loading are outside timing:

| Alternating process pair | Baseline median ms | Unrolled median ms |
|---|---:|---:|
| 0 | 3.5770 | 3.6165 |
| 1 | 3.5559 | 3.5650 |
| 2 | 3.5631 | 3.8464 |

All output hashes matched. The candidate lost all three complete-layer
comparisons, so it was rejected despite the isolated-kernel improvement.
No production change or default change was made; no GPU execution ran.
Evidence, hashes and the small rejected patch are retained in
`results/20261019/q23-unroll/`.


### Q2/Q3 worker and dataflow scheduling screen (2026-10-20)

The pool benchmark now accepts optional `split|flow` after the repeat count.
`split` remains its default. The new mode exercises existing `run_layer_native`
scheduling, which can begin an expert's down work when its gate/up and hidden
quantization finish. It verifies every timed result against the phased path
byte-for-byte, including routed sums; production scheduling was not changed.
Phase timings are emitted only for split mode because the flow mode overlaps
phases and does not populate those counters.

TR16 tests used synthetic H4096/F2048, eight experts, NUMA-owned Q2 gate/up and
Q3 down, `STRATA_Q2K_LUT=1`, 20 timed repeats and uncontrolled background load.
Seven workers returned the documented skip status 77 because their placement
did not cover both NUMA nodes. Eleven versus fifteen workers gave 2.1006 versus
1.4164 ms for one token, but 6.8438 versus 7.1044 ms for four tokens; these single
process comparisons do not qualify a worker-count recommendation. Hidden
quantization was about 1% of the recorded phased latency.

With fifteen workers plus the host, the scheduler comparison was:

| Process order | Tokens | Split median ms | Flow median ms |
|---|---:|---:|---:|
| Split then flow | 1 | 1.3810 | 1.1946 |
| Split then flow | 4 | 4.5832 | 3.8562 |
| Flow then split | 1 | 1.4813 | 1.1199 |
| Flow then split | 4 | 5.1873 | 3.8528 |

Flow was 13–26% faster in these screens, with identical expert and routed-sum
output hashes across modes and worker counts. The reverse-order repeat
explicitly set `STRATA_NATIVE_NUMA_LOCAL=0`; it was unset in the first screen
(the existing default is false). Input quantization, model loading and actual
GLM routing are outside scope; inputs/weights are synthetic. This is not a
full-model decode measurement or proof of peak bandwidth utilization.

The benchmark was compiled against the same existing CPU archives for both
modes; binary/archive hashes, phase timings, skipped configurations and raw
results are in `results/20261020/q23-scheduling/`. Invalid mode rejection passed.
No GPU backend source changed or GPU execution ran.


### Q2/Q3 dataflow task-size tuning

`q23_pool_test --bench TOKENS WORKERS REPEATS flow-sweep` reuses one loaded
fixture and worker pool while rotating through 1/2/3/4/6/8 tasks per thread.
It calls the existing diagnostic setter only between completed batches, warms
each selected schedule outside timing, then checks expert outputs and routed
sums byte-for-byte against phased execution. Normal benchmark and production
defaults are unchanged.

TR16, fifteen pinned workers plus host, synthetic H4096/F2048/eight-expert
Q2/Q3 data and NUMA-owned rows; `STRATA_Q2K_LUT=1` and
`STRATA_NATIVE_NUMA_LOCAL=0`. Eighteen rotated samples per setting in each of
two processes produced these medians (ms):

| Tasks/thread | Token 1, first | Token 1, repeat | Tokens 4, first | Tokens 4, repeat |
|---|---:|---:|---:|---:|
| 1 | 1.5152 | 1.5051 | 5.6776 | 5.0835 |
| 2 | 1.1861 | 1.2391 | 4.3451 | 4.2692 |
| 3 (existing default) | 1.1820 | 1.1148 | 4.1023 | 4.2332 |
| 4 | 1.1052 | 1.0976 | 4.1036 | 3.9107 |
| 6 | 1.1310 | 1.1000 | 3.8729 | 3.8567 |
| 8 | 1.1299 | 1.1112 | 3.8971 | 3.8334 |

Four tasks per thread reduced single-token latency by 1.5–6.5% relative to
three; six reduced four-token latency by 5.6–8.9%. These are opt-in tuning
candidates for the tested shape, not a global recommendation. The single-token
1.098 ms result processes 72.876 MB of packed expert weights, or roughly
66.4 GB/s of logical payload; it does not measure DRAM traffic or prove 90% of
the hardware peak. Background load was uncontrolled. No full-model run or GPU
execution occurred. All 432 timed outputs matched their reference hashes.
Evidence and binary/archive hashes are in `results/20261020/q23-flow-tasks/`.


### Prefetch with tuned Q2/Q3 dataflow

Three process-order screens on TR16 tested existing `STRATA_Q23_PREFETCH`
distances 0/256/1024 on the synthetic one-token expert layer, fifteen workers
plus host, four tasks/thread, Q2 LUT enabled, NUMA-owned rows and dataflow
scheduling. Each process reports the median of twenty timed calls:

| Screen | Prefetch off ms | 256 bytes ms | 1024 bytes ms |
|---|---:|---:|---:|
| Ascending distance | 1.8598 | 2.0364 | 1.1360 |
| Descending distance | 1.1506 | 1.1187 | 1.1290 |
| Rotated distance | 1.0969 | 1.1509 | 1.0533 |

1024 bytes beat off in all three process comparisons, but the first screen's
large slowdown makes its apparent gain unreliable. The latter two comparisons
gave 1.9% and 4.0% lower latency. This remains an opt-in candidate, not a default
change or full-model recommendation. All 180 timed outputs matched the phased
reference, including routed sums. No production source changed.

A fresh, separate 16-core first-touch memory reference measured 80.8905 GB/s
median read throughput (77.745–81.060), using one-GiB arrays and 4-KiB pages.
The best expert-process median above implies 69.2 GB/s of packed-weight payload,
about 85.5% of that read reference. These workloads differ: the expert set is
72.876 MB and includes arithmetic and output writes, while the reference is a
large sequential read. This ratio is not hardware bandwidth utilization and
does not establish the requested 90% of theoretical peak. No DIMM-rate or
memory-controller counter measurement was added.

The existing engine selects dataflow with `STRATA_GLM_LAYER_FLOW=1`; the
corresponding tuning knobs are `STRATA_NATIVE_TASKS_PER_THREAD=4` and
`STRATA_Q23_PREFETCH=1024`. The tests also used `STRATA_Q2K_LUT=1`. These flags
identify a configuration to validate on actual routing; this screen does not
qualify changing the user's normal launch configuration. Background load was
uncontrolled; no GPU execution ran. Evidence is in
`results/20261020/q23-flow-prefetch/`.


### CPU weight reuse in the node protocol

The CPU node accepts the explicit `reuse` selector for fused expert execution
with four-token weight sharing and two-/one-token tails. The launcher exposes
`--primary-cpu-reuse`, mutually exclusive with paired and fused selectors. It
changes only the primary; remote worker arguments remain unchanged. Existing
defaults are preserved. The local smoke harness accepts `--primary-mode reuse`
and checks the corresponding startup marker.

Three local CPU processes passed both direct and relay loopback-UDP topologies,
16 layer checks each, with a reuse primary and separate-mode workers. Batches
1/3/7/16 used real partial layer-3/4 weights with synthetic activations and routes.
Relative L2 was zero and repeated outputs were exact. Both 884,736-byte final
dumps match the prior separate-CPU baseline byte-for-byte (SHA-256 recorded in
the evidence). CPU mode conflicts and repeated selectors are rejected before
network setup.

CPU/HIP/CUDA node builds passed; no GPU execution ran. The experimental runner
has no SYCL backend. FP64 verification remains enabled throughout, so this is
correctness evidence, not timing or full-model throughput. Physical IB testing
of the new reuse selector remains outstanding. Reproduce with
`python3 tools/test_glm_fp4_cpu_tree.py PARTIAL_MODEL_DIR --primary-mode reuse
--output NEW_OUTPUT_DIR` on one shell line. Evidence is in
`results/20261020/cpu-tree-reuse/`.


### Physical IB validation of the reuse primary

The new `--primary-cpu-reuse` selector passed the physical three-machine direct
IB test with sixteen primary CPU threads and unchanged four-thread CPU workers.
The two real partial layers exercised batches 1/3/7/16 with repeated synthetic
routes/activations. All sixteen layer checks reported zero relative L2; all
three nodes completed successfully. The final 884,736-byte dump matches both
the local reuse tests and the original separate-CPU baseline byte-for-byte.

The test reused the existing remote CPU deployments, changed no remote source,
and ran no GPU kernels. FP64 verification remained enabled; elapsed wall time
is not decode throughput. Physical relay testing of reuse was not repeated
(the local relay test passed in the preceding section). This does not establish
a full-model decoder or CPU/GPU cache-miss overlap. Validation-line excerpts,
full-raw-log hashes and output/source hashes are in
`results/20261020/cpu-ib-reuse/`; private connection configuration stays local.


### Local FP4 to production NVFP4 layout adapter

`strata_glm_fp4_pack_nvfp4` is a shared C11 layout conversion: adjacent E2M1
nibbles and decoded row-major E4M3 scales become GGML NVFP4 blocks, each with
four scale bytes and 32 code bytes for 64 weights. Columns must be divisible
by 64. Code and scale bytes are preserved without requantization; the global
scale stays separate for the engine's existing expert scale tail. This changes
no production dispatch and does not yet assemble or run a native expert.

The converter validates exact buffer extents, multiplication overflow, nulls,
overlap and finite nonnegative scale codes before writing. Output cannot
overlap either input. Storage equals the original codes plus expanded scale
bytes; this is not a claim about compressed checkpoint size.

The CPU-only CMake test `glm_fp4_nvpack_test` checks all 2032 finite E2M1/E4M3
combinations and multi-row geometry against GGML's independent decoder. With
a partial-model directory argument, it checked six real expert-0 projections
from layers 3/4: 50,331,648 weights, exact code/scale payload and exact numeric
decoded values, including the separately applied global scale. GGML's integer
E2M1 table collapses negative zero on decode; the original sign nibble remains
in the packed bytes. Float-bit equality for signed zero is not claimed.

Normal and ASan/UBSan synthetic/real-weight tests passed; leak detection was
disabled and the linked existing GGML archive was not instrumented. CMake
build and CTest passed the new test plus glm_names_test, glm_env_test and
glm_expert_pack_test. GPU execution was not used. Evidence and hashes are in
`results/20261020/cpu-nvpack/`. Next is assembling the three converted
projections plus global-scale tail and validating actual native CPU execution.


### Native CPU execution of converted FP4 experts and clamp regression

`glm_fp4_native_test` assembles the three converted local projections at the
engine's `NativeFmt` offsets and appends `{s_gate,s_up,s_down,0}`. It executes
native input quantization, gate/up, hidden quantization and down, checking all
hidden and output values against direct GGML dot products with matching global
scales, clamp and SiLU. This is Q8_0 activation arithmetic, not the experimental
BF16 W4A16 contract; matching this reference is not a full-model quality gate.

The test exposed an existing defect: NVFP4 fast gate/up ignored a positive
`swiglu_limit`, although the fallback applied it. The new
`nvfp4_256_gu_rows_clamped` applies the upper gate bound and symmetric up bound
after global scaling, before SiLU. The old unclamped API and its template
specialization remain intact. Positive-limit native dispatch uses this AVX2
path or GGML; the existing AVX512 gate/up path is bypassed for positive limits
because it has no clamp contract. Unclamped AVX512 dispatch is unchanged.

On TR1950X, the old implementation failed a clamp-triggering analytic fixture.
After the fix, 80 synthetic combinations passed bitwise parity (limits 0/10,
inputs 0/1/20/-20, widths 1–9/16), as did four real-weight expert runs from
layers 3/4 at widths 1/4. Existing AVX2 NVFP4 parity and five CTests passed.
ASan/UBSan covered the converter, native dispatch, AVX2 kernel and test with
synthetic and real weights; existing GGML/other CPU objects were uninstrumented,
and leak detection was disabled. AVX512 execution was unavailable; no GPU
execution ran. No throughput result is claimed yet. Evidence and the failing
before-fix log are in `results/20261020/cpu-native-fp4/`.


### Native FP4 CPU pool timing against GGML

`glm_fp4_native_test --bench PARTIAL_MODEL_DIR` now times the production worker
pool with one converted real expert-0 blob from each of layers 3/4. It checks
the full expert against direct GGML dots before timing, then verifies every
pooled output and routed sum byte-for-byte outside the timed interval. Input
quantization, loading, conversion and allocation are excluded; gate/up, SiLU,
hidden Q8_0 quantization, down and routed reduction are included.

TR1950X used fifteen pinned workers plus the host, ordinary complete blobs
(14,155,792 bytes each), and synthetic positive inputs. Each fixture rotated
nine split/flow samples. Two process rounds reversed AVX2 versus GGML order;
AVX512 was disabled, and GGML fallback was selected with
`STRATA_NO_NVFP4_256=1`. Repeat-process medians were:

| Layer / tokens | Schedule | GGML ms | AVX2 ms |
|---|---|---:|---:|
| 3 / 1 | split | 0.2101 | 0.2042 |
| 3 / 1 | flow | 0.2053 | 0.1976 |
| 4 / 1 | split | 0.2140 | 0.2046 |
| 4 / 1 | flow | 0.2034 | 0.2038 |
| 3 / 4 | split | 0.6964 | 0.4783 |
| 3 / 4 | flow | 0.7396 | 0.4601 |
| 4 / 4 | split | 0.6422 | 0.4708 |
| 4 / 4 | flow | 0.6619 | 0.4458 |

Four-token AVX2 saved 27–38% in the repeat and 25–35% in the first screen.
Single-token differences were small and mixed. All 288 timed outputs matched
the independent reference and expected routed sums. Both FP4 CTests passed
after adding the benchmark. Background load was uncontrolled; the one-expert
working set and ordinary allocation do not establish DRAM bandwidth, a full
routed-layer rate or full-model decode throughput. These Q8_0 measurements
are not directly comparable to BF16 W4A16 quality or API timings.

No production default changed and no GPU execution ran. NUMA-owned NVFP4
rows remain unqualified: separate projection pointers and global-tail addressing
need validation before using that placement. Evidence and binary/archive hashes
are in `results/20261020/cpu-native-pool/`.


### Native FP4 NUMA projection ownership

Native row calls now accept a separate scale-tail owner pointer. Both phased
and dataflow NUMA jobs pass their original complete expert blob as that owner,
while gate/up/down weights come from their node-local shards. Missing owners
for formats with scale tails are rejected before dispatch. NVFP4 AVX2 gate/up
honors separate up storage with clamped or unclamped arithmetic; fast down
paths honor separate down storage. Separate-up execution uses AVX2 or GGML,
since the existing AVX512 gate/up API assumes contiguous projections.

Complete-blob calls retain their existing default arguments. Q2/Q3 jobs, which
have no scale tail, retain their previous behavior. The original scale-tail
owner must remain alive for the job; this change does not alter allocation or
checkpoint-loading policy.

The native FP4 test now checks independently allocated half-row gate/up/down
buffers against its complete-blob and direct GGML reference for every existing
synthetic/real fixture. `--bench-numa PARTIAL_MODEL_DIR` additionally exercises
actual `NumaTensor` placement with fifteen pinned workers plus host, both split
and flow schedules, and explicit rejection of a missing scale-tail owner.

On TR1950X, all 80 synthetic cases and four real expert runs passed exact
parity. Both owned pool schedules and the GGML fallback passed. ASan/UBSan
instrumented the pool, native dispatch, AVX2 kernel, converter and test; separate
shards and actual NUMA jobs passed (leaks disabled, other linked CPU/GGML
objects uninstrumented). Four CTests, including existing Q2/Q3 and dataflow
pool tests, passed, as did existing AVX2 NVFP4 parity. AVX512 execution was
unavailable, and no GPU execution ran.

This is correctness support, not a measured NUMA speedup. The first timings
were uncontrolled, the fallback run overlapped sanitizer compilation, and
sanitizer timings are not performance evidence. Multi-expert timing and
full-model quality remain outstanding. Evidence is in
`results/20261020/cpu-native-numa/`.


### Three real native FP4 experts: placement and scheduling

`glm_fp4_native_pool_bench PARTIAL_MODEL_DIR` loads actual experts 0/1/2 from
layers 3/4, converts their projections, and compares complete blobs versus
NUMA-owned rows under split and dataflow scheduling. Each three-expert set
occupies 42,467,376 packed bytes including global-scale tails. Original blobs
are allocated/filled on the pinned host; NUMA copies use the existing per-node
first-touch allocator. Synthetic inputs and fixed route weights 1/6,2/6,3/6
are used; this is not a captured router trace.

TR1950X used fifteen pinned workers plus host, three tasks/thread, native NUMA
local mode disabled, and AVX512 disabled. Each process rotated nine samples
per mode with an untimed schedule warmup. Two process rounds reversed AVX2
and GGML fallback order. Single-token AVX2 dataflow medians were:

| Layer | First complete ms | First NUMA ms | Repeat complete ms | Repeat NUMA ms |
|---|---:|---:|---:|---:|
| 3 | 1.1525 | 0.7293 | 1.2761 | 0.9269 |
| 4 | 1.2772 | 0.8808 | 1.2816 | 1.0812 |

NUMA placement saved 16–27% in the repeat and 31–37% in the first screen.
Four-token placement effects were negligible in the repeat: complete/NUMA
dataflow was 2.6724/2.6901 ms for layer 3 and 2.4864/2.5112 ms for layer 4.
For comparison, repeated GGML fallback NUMA dataflow took 3.1933/3.2960 ms
at four tokens. Single-token AVX2 versus GGML results were mixed.

All 576 timed outputs matched their sequential complete-blob references, and
expert/routed-output hashes matched across AVX2, GGML, placements and schedules.
Timing includes gate/up, SiLU, hidden Q8_0 quantization, down and routed reduction;
loading, conversion, allocations, input quantization and reference checks are
excluded. Background load was uncontrolled. These are partial expert-service
timings, not full-model tok/s, a quality gate or theoretical-peak utilization.
No production defaults changed and no GPU execution ran. Evidence and hashes
are in `results/20261020/cpu-native-multiexpert/`.


### Single-token NVFP4 biased integer dot candidate

An isolated AVX2 candidate replaces signed-weight magnitude/sign transfer with
unsigned `(weight + 12)` and subtracts precomputed `12 * activation` pair sums.
The integer pair bounds stay below int16 saturation. Scale multiplication,
FMA and final reduction order remain unchanged. Raw Q8_0 -128 inputs select the
original arithmetic because sign transfer has different wrap behavior there.
Only one-token slices select this candidate; it adds 4 KiB of activation
correction storage per call. Production source and defaults remain unchanged.

On TR1950X, a pinned-core K4096/N128 probe ran 16,384 calls per process.
Three alternating-order comparisons reduced whole-process cycles by 26.7%,
8.6% and 9.8%; instructions fell 6.7%. The first comparison was slower in
both absolute baseline and candidate time. A row-count sweep processing about
2^21 rows per process gave these cycle reductions:

| Rows/call | First | Reverse-order repeat |
|---|---:|---:|
| 8 | 20.8% | 6.0% |
| 32 | 7.1% | 9.0% |
| 512 | -20.3% | 9.0% |

These include activation preprocessing and process setup. The probe checks
exact GGML parity before timing and after changing a raw activation to -128.
All 84 synthetic/real native cases, including independent NUMA row shards,
passed bitwise parity; ASan/UBSan also passed. Sanitizers instrumented the
candidate, test, dispatch, pool and host converter, but not GGML or the other
linked CPU archive objects; leak detection was disabled.

The three-real-expert NUMA/dataflow screen improved single-token medians
by 5.8–9.1% in the first round, then regressed by 28.9–65.2% in the reverse
round. Unchanged four-token arithmetic also slowed by 45.5–46.5% in that
round, so the pool comparison is inconclusive under uncontrolled load.
All cross-binary output hashes matched. The candidate is retained for further
pool testing, not promoted based on isolated instruction savings. There was
no GPU execution, full-model quality validation, tok/s measurement or proof
of 90% hardware utilization. Compact logs, exact patch, probe source and hashes
are in `results/20261020/cpu-offset-dot/`.

To reproduce the isolated probe, compile the baseline or patched kernel with
`-O3 -DNDEBUG -std=c++20 -mavx2 -mfma -mf16c`, using `include` and
`third_party/ggml` includes. Link `probe.cpp` and that object ahead of the
GGML CPU/base archives with `-lm -pthread`, using the matching GGML public
headers. Run `perf stat -e cycles:u,instructions:u -- taskset -c 0 PROBE ROWS`.
Use alternating baseline/candidate order; omit ROWS for the 128-row case.


### Same-process FP4 offset-dot pool comparison

The offset-dot candidate and baseline were linked under separate symbols into
one experimental executable. A wrapper selects the kernel only between completed
pool jobs. Both variants pay the same atomic-load/branch overhead. The test reuses
the same three-real-expert weights and pinned worker pool, rotates all eight
kernel/placement/schedule combinations, and takes 21 samples per combination.
The sequential reference always uses the baseline. Two runs completed on TR1950X.

Single-token NUMA/dataflow medians were:

| Run | Layer | Baseline ms | Candidate ms | Reduction |
|---|---:|---:|---:|---:|
| First | 3 | 0.9643 | 0.9338 | 3.2% |
| First | 4 | 0.6414 | 0.5944 | 7.3% |
| Repeat | 3 | 0.6409 | 0.5531 | 13.7% |
| Repeat | 4 | 0.6398 | 0.5982 | 6.5% |

All 1,344 timed expert/routed outputs matched the baseline exactly, with matching
hashes across variants. Four-token NUMA/dataflow controls changed by +1.0%,
+1.3%, -0.9%, and +7.4% in latency reduction, so background timing variation
remains material. This supports further tuning of the one-token candidate, but
does not justify a universal speedup or default change. No production source
changed. The exact experimental sources, logs and metadata are retained in
`results/20261020/cpu-offset-pool-ab/`. No full-model throughput, quality or
90% peak-utilization claim follows from this screen.


### Opt-in single-token NVFP4 offset dots

`STRATA_NVFP4_OFFSET_DOT=1` now enables the biased integer dot path in the
AVX2 NVFP4 kernel for calls with exactly one token. Only the exact string `1`
enables it, through the shared C11 environment helper; the value is cached on
first single-token use. Other token counts retain the original path. This does
not force AVX2 over AVX512; the existing backend dispatch remains in control.

The default `Acts` structure and original row-dot arithmetic remain intact.
The opt-in path uses a separate derived activation structure with 4 KiB of
corrections, shared across rows within the call. Any raw Q8_0 -128 selects the
original sign-transfer arithmetic. Clamping, separate NUMA row storage, scales,
FMA order and final reduction are preserved. This remains NVFP4/Q8_0 arithmetic,
not an exact BF16 W4A16 replacement.

Validation covered unset/0/1/2 settings, 84 synthetic and real expert cases per
setting, plus 256 raw-code outputs across four dimensions, all 16 FP4 codes,
extreme signed Q8 values and a -128 at the final activation position. Every
output matched GGML exactly. The existing random-block test passed; integrated
pool tests matched all 288 timed outputs and off/on hashes. Those pool timings
are not performance evidence because compilation ran concurrently. ASan/UBSan
passed with the test, kernel, dispatch, pool and converter instrumented; linked
GGML/other archive objects were uninstrumented and leak detection was disabled.

Affected CPU targets built under HIP and a fresh CUDA configuration using GCC;
the CUDA native tests and real-weight output matched HIP. Native default/offset,
Q2/Q3 pool and flow-pool CTests passed. Standalone Clang compilation passed.
SYCL build validation is outstanding because the configured compiler is absent
and the cached build tool path is stale; Clang compilation does not replace it.
No GPU code was executed or changed by this step.

After builds completed, a pinned-core K4096/N128 probe of the integrated binary
ran 16,384 calls with the switch off/on in alternating order. Whole-process cycles
fell 14.2% and 9.1%; instructions fell 5.1% in both rounds. This is an isolated
kernel screen, not full-model throughput or 90% peak utilization. Default latency
has not been qualified. Evidence and the integration patch are in
`results/20261020/cpu-offset-optin/`.

Example CPU validation with a partial model fixture:

```sh
STRATA_NO_NVFP4_512=1 STRATA_NVFP4_OFFSET_DOT=1 \
  BUILD_DIR/glm_fp4_native_test PARTIAL_MODEL_DIR
```


### FP4 streaming exposes a limit of offset dots

`tools/glm_fp4_stream_bench.cpp` adds a CPU-only streaming probe. Sixteen pinned
physical-core workers first-touch their own uninitialized packed buffers, then
reuse the same worker threads for timing. Each owns 64 K4096/N256 matrices;
total packed weights are 603,979,776 bytes. Read-only and NVFP4 row-dot modes
alternate over the same buffers. A separate post-check barrier excludes exact
GGML comparison from the next measured interval. One warmup is excluded and
nine timings are retained per mode. Allocation, initial reference dots and
input preparation are untimed. This is plain row streaming, not an expert chain.

On TR1950X, the integrated opt-in offset kernel lost performance at the default
2 KiB prefetch distance despite its cached probe gains:

| Prefetch bytes | Baseline GB/s, first / repeat | Offset GB/s, first / repeat |
|---|---:|---:|
| 0 | 48.33 / 47.48 | 48.24 / 46.93 |
| 512 | 45.03 / 46.48 | 38.68 / 47.23 |
| 1024 | 42.42 / 44.33 | 41.09 / 41.85 |
| 2048 (default) | 48.73 / 46.25 | 42.98 / 42.88 |
| 4096 | 47.58 / 45.23 | 47.63 / 45.58 |

Same-buffer read controls in the initial default-prefetch comparisons reached
78.82–81.45 GB/s. The best row-dot payload rate was about 60% of its associated
read control, not 90% of hardware peak. These are logical payload bytes; no
memory-controller traffic or DIMM-clock measurement was made. Background load
was uncontrolled. All twenty normal runs checked every output after every pass
against independent GGML dots, with identical reference hashes. A small two-worker
ASan/UBSan run passed; linked GGML/other CPU archive objects were uninstrumented
and leak detection was disabled.

Do not enable offset dots globally based on the cached probe. Prefetch 0 or 4096
removed most of the streaming penalty here, but did not establish an improvement
over the baseline. Keep defaults unchanged and evaluate the actual expert working
set. This result directs further work toward the streaming loop and memory-access
costs, rather than assuming fewer instructions means higher sustained bandwidth.
Evidence and hashes are in `results/20261020/cpu-fp4-stream/`.

Build the standalone probe with C++20, `-O3 -mavx2 -mfma -mf16c`, repository
`include` and `third_party/ggml` plus matching GGML public headers. Link
`libstrata_kernels_cpu.a`, `libstrata_host.a`, `libggml-cpu.a`,
`libggml-base.a`, `-lm -pthread`. Run `PROBE 16 64`, selecting
`STRATA_NVFP4_OFFSET_DOT=0` or `1` and `STRATA_NVFP4_PREFETCH=BYTES`.
Use alternating variant order. The configured CPU set must permit all workers.


### SMT and two-row streaming screens

The streaming probe now accepts `[physical|logical]` after thread/copy counts.
Linux logical placement appends allowed SMT siblings after the physical-core
list. It reports the exact pinned CPUs and rejects affinity failures. The default
remains one worker per physical core. Compare `PROBE 16 64 physical` with
`PROBE 32 32 logical`: both use 603,979,776 packed bytes. Synthetic data differ
across thread partitions because generation uses per-worker seeds.

With prefetch disabled, two reverse-order runs on TR1950X gave:

| Workers / kernel | First GB/s | Repeat GB/s |
|---|---:|---:|
| 16 physical / baseline | 44.57 | 43.83 |
| 32 logical / baseline | 40.13 | 46.11 |
| 16 physical / offset | 39.48 | 42.19 |
| 32 logical / offset | 39.54 | 51.51 |

SMT read controls also dropped to 55.78–61.40 GB/s, compared with physical-worker
controls of 73.21–80.48 GB/s. This is not stable evidence for changing worker
placement. All row outputs matched the appropriate independent GGML reference.
No engine scheduling policy changed.

A second isolated candidate pairs gate/up rows and adjacent down rows, sharing
Q8 loads and activation-scale broadcasts. Each row retains its original FMA
order. It passed native raw-Q8 and synthetic/real expert parity, but lost
streaming throughput in all four comparisons:

| Run / prefetch | Baseline GB/s | Two-row GB/s |
|---|---:|---:|
| First / 0 | 43.67 | 38.58 |
| First / 2048 | 40.95 | 35.89 |
| Repeat / 0 | 43.70 | 40.52 |
| Repeat / 2048 | 45.42 | 36.88 |

The two-row candidate is rejected, with its patch retained for reproducibility.
It was not installed in production or promoted to a new flag. No sanitizer or
full-backend qualification was performed for this rejected candidate. The
streaming benchmark remains available for future kernels. Compact logs, source
and binary hashes, and the rejected patch are in
`results/20261020/cpu-stream-smt-pair/`. No GPU execution, full-model throughput
or theoretical-peak claim follows from either experiment.


### Half-precision scale lookup rejected

An isolated FP4 candidate stores all 256 halved E4M3 scale values as exact FP16
bits in a constexpr table, assembles four values, converts them with F16C, then
broadcasts the lanes. It preserves the original sentinel and high-bit decoding
behavior and does not introduce a startup ISA-dependent constructor.

Native synthetic/real and raw-Q8 parity passed, as did every streaming output.
On TR1950X with sixteen physical workers, prefetch disabled and offset dots off,
the two baseline/candidate streaming comparisons were 47.88/40.85 and
42.94/40.41 GB/s. The candidate is rejected; no production change or additional
sanitizer/backend qualification was pursued. Patch, logs and hashes are in
`results/20261020/cpu-scale-half/`. No full-model throughput claim follows.


### Q2/Q3 queue groups: small mixed same-process result

An isolated pool candidate queues gate/up then down for groups of four experts
within each node/stage, rather than all experts' gate/up followed by all down.
Both node queues keep each expert's producers ahead of its consumers. Existing
reference arithmetic and output order are unchanged. This is not integrated.

The TR1950X screen uses E8/H4096/F2048 synthetic NUMA-owned Q2_K gate/up and Q3_K
down, Q2 LUT enabled, fifteen workers plus host, prefetch off, and four tasks/thread
for one token or six for four tokens. Two reverse-order process comparisons gave
baseline/candidate medians of 1.1398/1.1530 and 1.1840/1.1620 ms at one token;
four-token medians were 4.1806/3.9753 and 3.8908/6.2388 ms. Background timing
variation makes this insufficient evidence for promotion.

A same-process harness then alternated queue orders on identical allocations and
worker pool, with an untimed warmup after each selection and twenty samples each:

| Tokens | All-expert queue ms | Groups-of-four queue ms |
|---|---:|---:|
| 1 | 1.1651 | 1.1400 |
| 4 | 3.9896 | 3.9888 |

All 240 timed outputs across both experiments matched their phased reference,
with matching hashes per token count. The existing Q2/Q3 pool correctness test
also passed. This small mixed result does not justify a new production setting.
The isolated candidate was not sanitizer/backend-qualified, and real routing,
FP4 and early-group-completion timing remain untested for this schedule. Logs,
patches and the alternating harness are in `results/20261020/cpu-q23-queue-groups/`.


### FP4 profiling, inlining and Zen 1 tuning

A whole-process `perf record -e cycles:u -F 997` run of the 16-worker streaming
probe collected 1,279 samples in `nvfp4_256_rows`. Loop-tail comparison and FMA
instructions had the largest local sample percentages. The disassembly reloads
the loop bound from the stack in this inlined specialization. Sampling skid
prevents treating those percentages as exact instruction costs or proof of a
bottleneck. Setup and read controls are included in the process totals; timings
from the profiled run are not a qualified performance comparison.

An isolated no-inline row-dot candidate passed native parity but did not help:
streaming baseline/candidate comparisons were 39.11/38.97 and 52.56/38.65 GB/s.
The out-of-line loop still reads its bound from the activation structure.
The candidate is not integrated; reduced inlining alone is not a demonstrated fix.

Compiling the unchanged kernel with `-mtune=znver1` retained the same AVX2/FMA/F16C
ISA and exact native outputs. Two alternating-order pinned CPU0 probes used
2.793/3.392 and 2.808/3.358 billion cycles for generic/Zen1 tuning: tuning was
20–21% worse. Streaming results were mixed, 53.72/51.96 and 42.46/50.99 GB/s.
Generic tuning is retained. Background load remained uncontrolled; no new default,
full-model throughput or 90% peak claim is justified. These candidates were not
sanitizer/backend-qualified because neither demonstrated a reliable improvement.

Compact profile summaries, logs, no-inline patch and source/profile hashes are in
`results/20261020/cpu-fp4-profile-tuning/`; the raw capture and full disassembly
remain in ignored scratch storage.


### 128-bit FP4 candidate and CPU-time accounting

An isolated single-token candidate retains two 128-bit accumulators corresponding
to the low/high halves of the original AVX2 accumulator. It preserves each lane's
FMA sequence and the final high-plus-low reduction. The initial implementation
decoded each 8-byte subblock separately; a revision shares the 16-byte nibble
decode across both subblocks, removing redundant shuffles. Both passed all native
synthetic/real and raw-Q8 parity cases.

The revised pinned CPU0 probe used 20–40% fewer cycles in two comparisons, but
baseline cycle counts varied strongly. Its streaming tests did not establish a
repeatable wall-time gain. Same-process real three-expert comparisons also varied
widely, including large changes in unchanged four-token controls. All 1,344 timed
same-process results matched their baseline reference exactly. The candidate
remains experimental; no production flag or default changed.

The standalone streaming probe now records process CPU milliseconds and
`active_cores_median`, the median ratio of process CPU time to elapsed time over
its timed region. These include all benchmark threads and the main thread, and
are not a direct hardware utilization or preemption counter. Independent medians
are reported, so median CPU time / median wall time need not equal the reported
median ratio. The separate verification rendezvous keeps checks out of subsequent
measurements. In two new 16-worker comparisons:

| Run / kernel | Wall ms | Process CPU ms | Median CPU-seconds / wall-second |
|---|---:|---:|---:|
| 1 / baseline | 19.613 | 210.250 | 10.73 |
| 1 / 128-bit | 16.614 | 189.334 | 11.40 |
| 2 / baseline | 30.391 | 224.217 | 7.20 |
| 2 / 128-bit | 26.213 | 194.104 | 7.27 |

The candidate used about 10–13% less process CPU time, while the measured
parallelism of roughly 7–11 on sixteen workers indicates material scheduling or
load-balance delays. The source of that variation is not established. These
observations justify retaining the candidate, not claiming a reliable decode
speedup. Sanitizer/backend qualification remains outstanding. The exact patches,
logs, hashes and accounting results are in `results/20261020/cpu-fp4-128/`.


### Stronger FP4 edge checks and expert-pool CPU accounting

`glm_fp4_native_test` now checks seven finite FP16 activation scales: zero,
smallest/largest subnormal, smallest normal, one, largest finite, and minus one.
Combined with four dimensions and four raw signed-Q8 patterns, including -128
in the last block, this makes 112 input configurations per run. Each compares
all sixteen output rows and a guarded thirteen-row subrange starting at row one:
3,248 selected outputs and 336 untouched output guards. These use the independent
GGML dot reference and cover all sixteen FP4 codes.

The current default and offset paths pass these checks in HIP- and CUDA-configured
CPU CTests. The experimental 128-bit candidate also passes ASan/UBSan runs, including
80 synthetic expert cases and four real expert cases with independent row shards.
Offset disabled exercises the 128-bit candidate; offset enabled retains the 256-bit
biased path. Sanitizers cover the test, candidate kernel, dispatch, pool and host
converter. Linked GGML/other CPU archive objects remain uninstrumented and leak
detection is disabled. Candidate full-backend/SYCL qualification remains open.

`glm_fp4_native_pool_bench` now reports process CPU-ms and median CPU/wall ratios,
using the same definitions as the streaming probe. Verification stays outside
timing. A validation run checked all 144 timed results and matched all 16 output
hashes against the previous baseline. CPU metrics were finite and positive; this
run is not a new performance claim. Both CPU test/benchmark targets built under
HIP and CUDA configurations, without GPU execution. No production kernel or
default changed. Logs and hashes are in
`results/20261020/cpu-fp4-128-validation/`.


### Combined 128-bit biased dots and corrected CPU accounting

The experimental 128-bit layout now also has a biased-dot variant. It shares
nibble decoding across two subblocks and subtracts the existing activation
corrections in 128-bit lanes. Raw -128 selects the signed 128-bit fallback. Both
bias off/on passed the expanded raw finite-scale/partial-range tests and 84 native
synthetic/real cases. ASan/UBSan passed with test, kernel, dispatch, pool and host
converter instrumented; other linked CPU/GGML archive objects were uninstrumented
and leak detection was disabled. This candidate is not integrated.

Pinned-core probes used 1–4% fewer cycles than signed 128-bit and 11–22% fewer than
the existing 256-bit biased kernel. A final same-process paired real-expert screen
with process-clock sampling removed from latency measurements gave:

| Run | Layer | Baseline 256-bit biased ms | Candidate 128-bit biased ms |
|---|---:|---:|---:|
| 1 | 3 | 0.600338 | 0.582477 |
| 1 | 4 | 0.607808 | 0.578058 |
| 2 | 3 | 1.036700 | 0.953728 |
| 2 | 4 | 0.566927 | 0.527216 |

All four single-token NUMA/dataflow comparisons improved 3–8%. Four-token controls
were mixed, and background load remained uncontrolled. All 1,344 latency outputs
matched exactly, as did 64 final CPU-batch checks. The candidate is retained for
backend qualification, with no default change or full-model throughput claim.

**CPU-accounting correction:** individual sub-millisecond pool calls produced
unreliable process-clock readings, sometimes exceeding physically plausible
parallelism. Preserve earlier raw measurements, but do not use their per-call
CPU metrics to infer utilization or CPU-cost improvements. The production
benchmark tool now measures CPU cost separately over 32 calls per mode and reports
`cpu_batch_calls`, `cpu_avg_ms`, and `cpu_batch_active_cores`. Its single-call
latency distribution does not sample the process clock. Batch ratios remain
approximate diagnostics, not hardware utilization counters. Streaming benchmarks
use longer intervals; very small streaming fixtures should likewise not support
CPU-cost claims.

The revised benchmark target built in HIP and CUDA configurations and retained
its exact reference checks; no GPU code executed. Candidate full-backend
qualification remains outstanding. Final paired results, sanitizer/parity logs,
patch, harness and source hashes are in `results/20261020/cpu-fp4-128-bias/`.


### Experimental 128-bit CPU path integrated

`STRATA_NVFP4_128=1` now selects the two 128-bit accumulator layout for single-token
calls in the AVX2 NVFP4 CPU path. It is independent of
`STRATA_NVFP4_OFFSET_DOT=1`: enable both for biased 128-bit dots, or only the new
switch for signed 128-bit dots. Only exact `1` enables either switch, through the
shared C11 environment helper, cached on first use. Neither switch forces AVX2
over AVX512. This remains NVFP4/Q8_0 arithmetic; it does not replace the BF16
W4A16 reference with an exact equivalent.

Dispatch occurs once per row range in the public entry points. Separate range
helpers contain the new activation types and loop; no selection check was added
inside the original row loops. The original signed/biased 256-bit row-dot bodies
remain text-identical to the pre-integration snapshot. Default and multi-token
arithmetic remain unchanged. Raw -128 in biased mode falls back to signed dots
with the selected lane width. Global scales, clamping, separate NUMA shards and
final reduction order retain their prior contracts.

All four flag combinations pass exact GGML checks, including the expanded raw
finite-scale/partial-row cases, 80 synthetic expert cases and 4 real expert cases.
Final GCC, Clang and sanitizer transcripts match. ASan/UBSan instrumented the
test, kernel, dispatch, pool and converter; other linked CPU/GGML objects were
uninstrumented and leak detection was disabled. Six final HIP-configured CPU
CTests passed, including Q2/Q3 and flow-pool regression checks. Four CUDA-configured
native-mode CTests passed. The existing random-block parity check passed. All 288
integrated pool latency outputs and 32 final CPU-batch checks matched, with
identical hashes across off/on modes. No GPU execution ran. SYCL build validation
remains outstanding because its configured compiler is unavailable; standalone
Clang validation does not replace that build.

**Performance remains experimental.** Final integrated pinned-core off/on cycle
counts were 2.585/2.580 billion and 2.594/2.938 billion. One real-layer screen improved
single-token NUMA/dataflow medians from 0.5749 to 0.5445 ms and 0.6122 to 0.5690 ms,
but unchanged four-token controls also moved. The earlier prototype's 3–8% paired
gain is not a guaranteed gain from this integrated build. Do not enable it as a
global default on this evidence. No full-model throughput or 90% peak claim is made.

To reproduce the tested partial CPU configuration:

```sh
STRATA_NO_NVFP4_512=1 STRATA_NVFP4_128=1 STRATA_NVFP4_OFFSET_DOT=1 \
  STRATA_NVFP4_PREFETCH=0 BUILD_DIR/glm_fp4_native_test PARTIAL_MODEL_DIR
```

Use `glm_fp4_native_pool_bench` instead of the test executable for the partial
expert benchmark. Final evidence, integration patch and source hashes are in
`results/20261020/cpu-fp4-128-optin/`.

### Q2/Q3 task tuning with fused hidden quantization (2026-10-20)

The new `q23_pool_test --bench TOKENS WORKERS REPEATS flow-quant-sweep`
mode compares tasks/thread 3, 4 and 6 with hidden quantization fusion off/on.
Earlier task sweeps inherited `NativeFmt`'s unfused default. The assembled-pack
paths in `src/program/glm_decode.cpp` already enable fusion unless disabled or
incompatible with the selected path. This benchmark correction matters when
choosing a setting for that decoder configuration; fusion is not a new feature.

On Threadripper 1950X with 128 GiB RAM, 15 pinned workers plus the host used
NUMA-owned Q2_K gate/up and Q3_K down weights: H=4096, F=2048, eight synthetic
experts, 72,876,032 packed bytes. Each process rotated six configurations for
20 samples each with an untimed warmup. Input quantization was excluded;
gate/up, SiLU, hidden quantization, down and routed reduction were included.
`STRATA_Q2K_LUT=1`, `STRATA_Q23_PREFETCH=0` and
`STRATA_NATIVE_NUMA_LOCAL=0` were used. RAM speed was not reverified.

Fused-mode median latency in milliseconds:

| Tokens | Run | Tasks 3 (default) | Tasks 4 | Tasks 6 |
| --- | --- | --- | --- | --- |
| 1 | 1 | 1.3050 | 1.2700 | 1.2624 |
| 1 | 2 | 1.8661 | 1.8089 | 1.8694 |
| 4 | 1 | 5.0365 | 4.6061 | 4.6652 |
| 4 | 2 | 4.9837 | 4.6159 | 4.6415 |

Four tasks reduced latency against three by 2.7–3.1% for one token and
7.4–8.5% for four tokens. Six was less consistent. Four is therefore the next
candidate for this fused/dataflow configuration; the earlier unfused result
favoring six for four tokens does not establish the fused setting. At this
geometry fused gate/up fixes its quantization-aligned task partition, while the
task setting still affects down. Off/on comparisons at other task settings
therefore also change the gate/up partition.

All 480 primary timed outputs and reductions matched the unfused phased
reference bitwise. The regular CPU pool parity test passed in HIP- and
CUDA-configured builds; the CUDA-configured six-case smoke sweep also matched.
These are CPU executions, not GPU checks. No production default or kernel was
changed by this benchmark addition. Synthetic routing and uncontrolled host
load limit the result: absolute one-token times drifted substantially between
runs. No full-model speed, quality or 90% theoretical-peak claim follows.

Reproduce with:

```sh
STRATA_Q2K_LUT=1 STRATA_Q23_PREFETCH=0 STRATA_NATIVE_NUMA_LOCAL=0 \
  BUILD_DIR/q23_pool_test --bench 4 15 20 flow-quant-sweep
```

Raw primary logs, validation logs, benchmark source snapshot, source hashes and
machine-readable medians are in `results/20261020/q23-fused-task-sweep/`.

### Q2/Q3 fusion depends on CPU expert count (2026-10-20)

The benchmark now accepts an optional expert count (1–8, default 8) after its
mode. A follow-up on the same Threadripper 1950X tested one, three and four
experts, one/four tokens, tasks/thread 3/4/6 and fusion off/on. Two runs per
geometry rotated six variants with 20 samples each. All 1,440 timed expert
outputs and routed sums matched the unfused phased reference bitwise. This
uses the same H=4096, F=2048, NUMA ownership, 15 workers plus host and environment
as the eight-expert screen above. CPU tests in HIP- and CUDA-configured builds
passed; the latter also passed a six-variant one-expert smoke run. No GPU ran.

At the default three tasks/thread, one-expert median latency was:

| Tokens | Run | Unfused ms | Fused ms | Unfused latency reduction |
| --- | --- | --- | --- | --- |
| 1 | 1 | 0.2667 | 0.4097 | 34.9% |
| 1 | 2 | 0.1783 | 0.2618 | 31.9% |
| 4 | 1 | 0.6128 | 1.0405 | 41.1% |
| 4 | 2 | 0.6213 | 1.1034 | 43.7% |

The penalty persisted at task settings four and six. The fused gate/up schedule
has eight tasks total for this one-expert geometry, only four per NUMA node,
while the unfused schedule can subdivide rows more finely. Q8_K hidden
quantization requires complete 256-value blocks, so simply halving fused chunks
would violate that contract. More gate/up parallelism with a separate hidden
quantization step is a plausible explanation for the measured benefit.

Three- and four-expert results were mixed, with large absolute timing drift.
These data do not justify a universal fusion cutoff or task setting. The next
production candidate is opt-in expert-count-aware fusion selection, first for
one CPU expert, retaining the same quantization arithmetic. No production default
changed in this screen. Real routing, mixed CPU/GPU overlap, full-model quality
and end-to-end throughput remain untested by these synthetic cases.

```sh
STRATA_Q2K_LUT=1 STRATA_Q23_PREFETCH=0 STRATA_NATIVE_NUMA_LOCAL=0 \
  BUILD_DIR/q23_pool_test --bench 1 15 20 flow-quant-sweep 1
```

The full distributions, source snapshot/hash and medians are frozen in
`results/20261020/q23-expert-count/`. This is partial CPU service evidence, not
full-model tok/s or a measurement of theoretical peak utilization.

### Opt-in single-expert Q2/Q3 unfused selection (2026-10-20)

`STRATA_Q23_SINGLE_UNFUSED=1` now lets the CPU dataflow pool disable requested
hidden-quantization fusion for the measured one-expert case. The selection is
limited to owned NUMA Q2_K gate/up and Q3_K down, H=4096, F=2048, 16 CPU
participants, one/four tokens and noncanonical arithmetic. Other geometries,
canonical groups and phased fallback retain their previous behavior. Only
exact `1` enables the option, using `strata_env_is_one`, cached on first eligible
scheduler entry. The default is off.

A local format copy changes only `fuse_h_quant`. The existing synchronous
scheduler executes it, parks all workers, and returns before that copy expires.
The caller's format and quantization arithmetic are unchanged. The existing
unfused path exposes more gate/up parallelism for this one-expert geometry.

On Threadripper 1950X, 720 timed outputs/reductions matched the unfused reference
bitwise across settings 0, 1 and 2, tasks/thread 3/4/6, and one/four tokens. The
exact output hashes also matched across processes. With the option enabled,
requested-fused versus unfused one-token medians at tasks=3 were 0.1577/0.1562 ms;
with it disabled they were 0.3926/0.2704 ms. Setting 2 retained the fused penalty
(0.2747/0.1656 ms). These cross-process values show substantial host-load drift;
they are not a qualified integrated speedup. Four-token enabled timings were
also noisy (0.8883/0.8336 ms). The preceding same-process fusion sweep is the
stronger performance evidence. Production defaults remain unchanged.

Use this only with the existing dataflow path (`STRATA_GLM_LAYER_FLOW=1`) when
screening a decoder configuration. This benchmark does not establish full-model
quality or tok/s. Logs, medians, the selection block and source hashes are in
`results/20261020/q23-single-unfused/`.

Both HIP- and CUDA-configured CPU builds passed `q23_pool_test` and
`flow_pool_test` with the option enabled. A CUDA-configured one-expert four-token
six-variant smoke sweep also passed. No GPU execution was performed. SYCL build
validation remains unavailable because its configured `icpx` compiler is missing;
this shared CPU change is not ready for cross-backend review on that evidence.

### Real FP4 hidden-quantization scheduling screen (2026-10-20)

`glm_fp4_native_pool_bench` now includes fused NUMA split/flow modes alongside
its four previous modes, and accepts optional `EXPERTS` (1–3, default 3) after
the partial model directory. It still rotates modes, checks every timed output
and sum, and samples process CPU time only in separate 32-call batches. The
selected expert weights remain `(expert+1)/6`, including the one-expert subset;
this is not normalized real routing. The reference is native NVFP4/Q8_0, not
exact BF16 W4A16. Input quantization/loading remain excluded.

Two runs each with one and three real experts from layers 3/4 showed the same
one-expert fusion penalty seen with Q2/Q3. On Threadripper 1950X with 15 workers
plus host, one-expert NUMA flow medians across layers/runs were 0.235–0.360 ms
unfused versus 0.509–0.633 ms fused for one token, and 0.692–0.785 versus
1.251–1.410 ms for four tokens. These are paired modes within each process;
range endpoints do not define a speedup. Three-expert results were mixed.

A temporary prototype changed NVFP4 fused gate/up chunks from 256 to 128 rows,
retaining complete 32-value Q8_0 blocks. For F=2048 this exposes 16 gate/up tasks
rather than eight. Two equivalent runs reduced the one-expert fused medians to
0.272–0.315 ms for one token and 0.749–0.830 ms for four tokens. Same-process
unfused controls were 0.231–0.253 and 0.694–0.760 ms respectively. Thus smaller
chunks substantially reduced the fusion penalty but did not consistently beat
the existing unfused path. Host-load variation prevents treating the separate
prototype/baseline processes as a qualified speedup.

The prototype switch/helper was removed from production source after this
screen. Only the benchmark coverage was retained; defaults and production pool
behavior remain as before this experiment. All 1,728 latency outputs and 192
final CPU-batch checks matched bitwise, including matching hashes across both
implementations. The prototype passed six HIP-configured CPU regression tests.
The final restored source built in HIP- and CUDA-configured trees; a real
one-expert CUDA-configured benchmark passed all six modes (216 further latency
checks and 24 final CPU-batch checks). No GPU execution or SYCL build ran.

Logs, benchmark snapshot, prototype helper and source hashes are frozen in
`results/20261020/fp4-fused-scheduling/`. The next candidate is single-expert
unfused selection for FP4; more fused chunk variants need evidence that they can
beat that existing path before integration. No full-model throughput or peak
utilization claim follows from this partial expert screen.

### Opt-in single-expert FP4 unfused selection (2026-10-20)

`STRATA_NVFP4_SINGLE_UNFUSED=1` now selects the existing unfused CPU dataflow
schedule when fusion was requested for one owned NUMA expert with NVFP4 gate/up
and down, H=4096, F=2048, 16 CPU participants, one/four tokens and noncanonical
arithmetic. It shares the synchronous local-format-copy mechanism with the
Q2/Q3 option, but the two switches are independent. Only exact `1` enables it,
through the shared C11 environment helper, cached on scheduler entry. The
caller format, packed weights, global scales, quantization and reduction order
are unchanged. Other expert counts, phased execution and fallback paths keep
their prior selection. Default behavior is unchanged.

Real partial weights from layers 3/4, one/three experts and one/four tokens were
checked with switch values 0, 1, 2 and a second run at 1. The benchmark rotates
six execution modes and uses nine latency samples per mode; separate CPU-time
batches contain 32 calls. All 1,728 latency checks and 192 final CPU-batch checks
matched bitwise. Hashes matched across switches and also matched the frozen
pre-integration results for every geometry, including three-expert controls.
This verifies partial native NVFP4/Q8_0 output, not full-model quality or exact
BF16 W4A16 equivalence.

For requested-fused one-expert NUMA flow on Threadripper 1950X, the off run gave
0.429/0.311 ms for one token on layers 3/4 and 1.166/0.936 ms for four tokens.
The two enabled runs gave 0.215–0.239 ms and 0.703–0.748 ms respectively. Their
same-process unfused controls gave 0.217–0.238 ms and 0.698–0.743 ms. Enabled
fused requests now track the existing unfused control, as intended. Separate
process timing drift remains substantial (especially the value-2 run), so these
numbers do not establish a precise decoder-level speedup. The existing
same-process fusion comparison is the stronger evidence for this selection.

Both HIP- and CUDA-configured builds passed all six scoped CPU regression tests
with both single-expert options enabled. A CUDA-configured real one-expert run
passed another 216 latency checks and 24 final CPU-batch checks. No GPU ran.
SYCL validation remains unavailable because the configured compiler is missing.
No default or full-model throughput claim is made.

For partial CPU reproduction:

```sh
STRATA_NVFP4_SINGLE_UNFUSED=1 STRATA_NO_NVFP4_512=1 \
  STRATA_NVFP4_128=0 STRATA_NVFP4_OFFSET_DOT=0 STRATA_NVFP4_PREFETCH=0 \
  STRATA_NATIVE_NUMA_LOCAL=0 \
  BUILD_DIR/glm_fp4_native_pool_bench PARTIAL_MODEL_DIR 1
```

Decoder experiments additionally need the existing `STRATA_GLM_LAYER_FLOW=1`
path to use this selection. Logs, medians, source hashes and the selection block
are in `results/20261020/fp4-single-unfused/`.

### CPU service model audit (2026-10-20)

`compare_cpu_service.py` compares the frozen real partial-weight CPU results
against the simulator's earlier 50/40/40 Gweight/token/s assumptions and runs an
explicit node-rate sensitivity sweep. It changes no simulator defaults and
prints no full-model tok/s. Run it from the repository root:

```sh
python3 experiment/glm_fp4_local/compare_cpu_service.py
```

The primary-only one-expert model gives 0.514 ms at one token and 2.054 ms at
four tokens. Measured requested-fused dataflow service with the single-expert
unfused option gives 0.215–0.239 and 0.703–0.748 ms respectively. For three
experts the model gives 1.541/6.163 ms; observed medians range 0.632–1.156 and
1.747–10.998 ms. Preserve that large four-token outlier: these data do not
establish a stable replacement rate. Measurements exclude input quantization;
the model includes it. Fixed-input repeated partial weights also differ from
full-model expert streaming.

With the previous 42-layer, 3/3/2 assignment, 50% cache-hit and CPU/GPU-overlap
assumptions, changing only the primary CPU rate from 50 to 75 Gweight/token/s
leaves MoE time at 51.638 ms. Doubling only the middle-node rate from 40 to 80
reduces it to 45.573 ms; doubling only the third-node rate leaves it at 51.638 ms.
Doubling all three gives 42.574 ms. These are uncalibrated sensitivity results,
not measured throughput or forecasts. At primary rates 100/125 the scheduler
instead gives 51.937/53.260 ms: altered event timing and shared-resource
contention make this fixed scheduling model nonmonotonic. Do not interpret that
as proof that a faster CPU kernel slows real hardware.

This shifts the next measurement priority toward the middle node and realistic
routing/overlap. A scalar CPU rate misses expert-count-dependent scheduling and
small-batch reuse; a future measured service table needs token count, CPU expert
count and execution mode, plus input-quantization and memory-traffic treatment
to avoid double counting. Do not fit the full three-node model to primary-only
partial timings. Non-MoE decoder work remains unmeasured by this simulator.

The diagnostic output and source/evidence hashes are in
`results/20261020/cpu-model-service-audit/comparison.json`. Existing FP4 simulator
unit tests passed; no production kernel or simulator default changed in this
audit.

### Middle-worker CPU streaming under load (2026-10-20)

A bounded CPU-only test on the middle worker identified its CPU as Ryzen 9
3950X (16 cores, 32 threads, one socket). Observed load averages before the test
were 30.12/30.20/30.19; the miner and other workloads were left running. These
are contention-conditioned measurements, not idle-system peaks. RAM speed was
not checked. No GPU work or system-library changes were performed.

The current `glm_fp4_stream_bench.cpp` was built with GCC, AVX2/FMA/F16C and
static runtime libraries, using the current CPU/host/GGML archives. An initial
dynamically linked binary failed before execution because the worker lacked its
required GLIBC_2.43 math symbol. The static binary passed a local smoke check
and then all four remote runs. Each run used 150,994,944 packed bytes, physical
core affinity, first touch, one warmup and nine timed passes of each mode.
Every FP4 output matched GGML and every read checksum matched. The 16-thread
variants produced identical output hashes. The four-thread data partition has
a different seed distribution and is not expected to share that hash.

| Threads | Dot layout | Read GB/s | FP4 packed GB/s | FP4 Gweight/s |
| --- | --- | --- | --- | --- |
| 4 | 256-bit | 16.10 | 12.31 | 21.89 |
| 16 | 256-bit, first | 20.18 | 17.97 | 31.95 |
| 16 | 128-bit | 19.94 | 18.35 | 32.61 |
| 16 | 256-bit, repeat | 22.16 | 18.34 | 32.60 |

Offset dots and prefetch were disabled. The 128-bit variant does not establish a
win over the repeated 256-bit baseline. FP4 achieves 83–89% of the read-only
control in the 16-thread baseline runs; the 128-bit run's 92% ratio comes with a
lower read control and does not establish a kernel improvement. None of these
ratios measures 90% of theoretical hardware bandwidth or FLOPS.

The observed 32 Gweight/s streaming rate is below the simulator's assumed
40 Gweight/token/s for this worker. This is a synthetic row-dot workload, not
whole expert service: it excludes activation quantization, SiLU, routing and
network work. It cannot directly replace the simulator's service rate, and an
idle-worker rate remains unknown. Current evidence favors investigating memory
contention, cache placement and concurrent scheduling before more arithmetic
variants on this worker.

Logs, exact static-binary/source hashes and configuration are in
`results/20261020/worker-cpu-stream/`. The shared log omits only its deployment
path; the original scratch log is retained locally. No model weights, binaries
or private connection configuration are included in the report.

### Middle-worker thread budget and L3 affinity (2026-10-20)

A follow-up with the same static FP4 streaming binary compared eight, twelve
and sixteen physical threads under the existing worker load. It also read the
CPU/cache topology: Ryzen 3950X exposes one NUMA node and four L3 groups of four
physical cores each. Compact eight-core placement uses CPU IDs 0–7; spread
placement uses 0,2,4,6,8,10,12,14. No affinity policy of other processes changed.
Offset dots, 128-bit dots and software prefetch remained disabled.

Eight/sixteen/eight runs used the same 150,994,944-byte total with 32/16 copies
per worker. Eight-thread FP4 bandwidth was 15.26/15.49 GB/s against 18.81 GB/s
for the intervening sixteen-thread run: 81–82% of throughput. Median process
CPU time was 65.57/63.51 ms versus 106.82 ms for the same byte count, a 39–41%
reduction. This suggests a CPU-budget tradeoff for future communication/overlap
tests, not a free throughput improvement.

Spread/compact/spread eight-thread runs used identical data and hashes. Spread
FP4 bandwidth was 14.05/15.15 GB/s versus compact 15.55 GB/s. The read controls
also favored compact placement (17.77 versus 16.43/16.36 GB/s). Spreading across
all L3 groups is not supported as an improvement under this load; this does not
establish an idle-worker affinity policy.

Twelve/sixteen/twelve runs gave 16.10/18.96/16.56 GB/s. Twelve threads retained
85–87% of sixteen-thread throughput, with about 22% lower process CPU time per
byte. Twelve threads used 21 copies each (148,635,648 bytes, 1.56% fewer than the
sixteen-thread run), so CPU cost is normalized by bytes. Different thread counts
also change synthetic data partitioning; identical hashes are required only
within matching counts. All 81 timed FP4 passes plus nine warmup passes matched
GGML, and all read-control checksums passed.

Keep sixteen threads for maximum isolated throughput on current evidence. Eight
is a candidate only when CPU budget for other decoder work matters; twelve did
not retain enough extra throughput here to establish a preferable compromise.
Actual routing/network overlap must be measured before changing a decoder
configuration. Process CPU-time ratios do not measure hardware utilization;
none of these tests proves 90% theoretical peak. Source/binary provenance is
shared with the preceding streaming test; topology, logs and summary are in
`results/20261020/worker-cpu-thread-affinity/`.

### Real FP4 service on the single-NUMA worker (2026-10-20)

The partial native pool benchmark now supports single-NUMA machines: it runs
complete-blob split and flow modes without constructing unavailable NUMA shards.
Machines with owned NUMA rows retain all six modes. Optional `WORKERS` follows
`EXPERTS`; the default remains 15, with the host participating as before. This
changes benchmark coverage, not the production pool or its defaults.

A statically linked build and only the 77 MiB partial fixture were deployed to a
temporary directory on the Ryzen 3950X worker. Three runs with 7/15/7 workers
(8/16/8 participants) tested real experts 0–2 from layers 3/4 at one/four tokens.
The miner and other workloads remained running. All 216 timed output/reduction
checks and 24 final CPU-batch checks passed. Hashes matched the frozen primary
host's results for every geometry. Final HIP- and CUDA-configured benchmark
builds passed local six-mode real-weight checks. No GPU execution occurred.

Single-token complete-blob flow latency was 2.250–2.535 ms with eight participants
versus 1.419–1.467 ms with sixteen. Split latency was 2.251–2.487 versus
1.567–1.685 ms. Eight participants did not consistently save process CPU time
in flow mode: measured CPU cost was 23.8–26.3 ms/call versus 21.5–22.0 ms with
sixteen. This weakens the streaming-only case for reducing decoder threads.
Four-token flow latency varied from 4.59–6.17 ms with eight to 1.71–4.31 ms with
sixteen; these noisy data do not qualify a stable four-token speedup.

The three-expert packed working set is 42,467,376 bytes. It is repeatedly reused,
so cache coverage and CPU affinity can affect these service timings differently
from the larger streaming test. CPU contention and cache effects were not
separated. Keep sixteen participants for the next worker service experiment;
do not infer full-checkpoint throughput, idle-system peak or a universal thread
policy. Input quantization, loading and networking are excluded, while gate/up,
SiLU, hidden quantization, down and reduction are included. The next calibration
needs varying expert sets to distinguish cache-resident and streaming service.

```sh
STRATA_NO_NVFP4_512=1 STRATA_NVFP4_PREFETCH=0 \
  STRATA_NVFP4_128=0 STRATA_NVFP4_OFFSET_DOT=0 \
  BUILD_DIR/glm_fp4_native_pool_bench PARTIAL_MODEL_DIR 3 15
```

Logs, fixture/source/static-binary hashes and benchmark snapshot are in
`results/20261020/worker-real-pool/`. No model or executable is included in the
shared evidence. SYCL build validation remains unavailable; this benchmark
change does not establish cross-backend review readiness.

### Worker FP4 task granularity: no default change (2026-10-20)

The real-weight pool benchmark accepts optional `task-sweep` after its worker
count. It rotates tasks/thread 3, 4 and 6 together with available execution modes
within each process, retaining the same weights, allocations and persistent
pool. The setter runs only between completed calls, followed by an untimed
warmup. Default benchmark operation and production defaults are unchanged.

An initial separate-process worker screen suggested six tasks could help
single-token service. Two same-process sweeps on Ryzen 3950X with real layers
3/4, three experts and sixteen participants did not confirm a stable setting:

| Run | Layer | Tokens | Tasks 3 ms | Tasks 4 ms | Tasks 6 ms |
| --- | --- | --- | --- | --- | --- |
| 1 | 3 | 1 | 2.5419 | 1.6036 | 1.4039 |
| 1 | 4 | 1 | 1.6987 | 1.4759 | 1.4656 |
| 2 | 3 | 1 | 1.3877 | 1.4005 | 1.4030 |
| 2 | 4 | 1 | 1.4164 | 1.4360 | 1.3939 |
| 1 | 3 | 4 | 5.5417 | 4.7260 | 5.9989 |
| 1 | 4 | 4 | 5.6787 | 4.2059 | 5.4873 |
| 2 | 3 | 4 | 6.6332 | 4.9775 | 5.7061 |
| 2 | 4 | 4 | 5.2093 | 6.2209 | 5.6965 |

These are complete-blob flow medians from nine rotated samples. Background load
remained uncontrolled and the miner was not changed. Keep the existing setting
of three: neither four nor six has a repeatable advantage across this screen.
Do not transfer the earlier TR16 fused-Q2/Q3 setting to this FP4 worker.

All 432 paired timed outputs/reductions and 48 final CPU-batch checks matched
bitwise, including hashes from the preceding worker benchmark. The expanded
sweep also built and passed real-weight checks in HIP- and CUDA-configured CPU
builds on the primary host. No GPU ran; SYCL validation remains unavailable.
Raw initial/paired logs, benchmark snapshot and binary/source hashes are in
`results/20261020/worker-task-granularity/`.

```sh
STRATA_NO_NVFP4_512=1 STRATA_NVFP4_PREFETCH=0 \
  STRATA_NVFP4_128=0 STRATA_NVFP4_OFFSET_DOT=0 \
  BUILD_DIR/glm_fp4_native_pool_bench PARTIAL_MODEL_DIR 3 15 task-sweep
```

### FP4 service phase accounting (2026-10-20)

The real-weight pool benchmark now reports split-mode phase means from the
pool's existing cumulative timers. `gu_avg_ms` includes gate/up, clamp/SiLU and
its worker phase; `hidden_quant_avg_ms` covers the serial hidden-quantization
interval; `down_avg_ms` includes down projection and its worker phase.
`other_avg_ms` is the same-call total minus those three components, covering
remaining setup, reduction and outer timing overhead. Fused modes include hidden
quantization within gate/up. Flow modes have no comparable phase breakdown and
do not print these fields. Input quantization remains excluded.

Components are arithmetic means of the same nine timed calls, not independently
selected medians. Their sum was checked against mean total latency to printed
precision. This avoids subtracting unrelated median timings. Worker wait time
and contention remain inside phase timings: they are not isolated kernel times.

Two HIP-configured runs on Threadripper 1950X with real layers 3/4 and three
experts show hidden quantization accounting for roughly 0.15–1.00% of unfused
NUMA split service at one/four tokens. Gate/up accounts for 53–81%, down for
19–44%, and other work for 0.19–2.91%. Absolute timings varied substantially
under uncontrolled load. These percentages locate work within these observed
calls; they are not a stable full-model profile. Even eliminating this hidden
quantization interval would save only about 1% or less in these cases, whereas
fusion can reduce gate/up parallelism. Projection work and scheduling remain
the more promising targets.

Two HIP-configured runs and one CUDA-configured CPU run passed all 648 timed
output/reduction checks and 72 final CPU-batch checks, with hashes matching the
frozen pre-instrumentation results. No production kernel or default changed.
No GPU ran. Source snapshot/hash, raw phase data and checked sums are in
`results/20261020/fp4-service-phases/`.

### Absolute-activation FP4 dot prototype (2026-10-20)

An isolated AVX2 prototype precomputed absolute Q8 activation bytes per row
range, then transferred activation signs to signed FP4 values before
`maddubs`. This replaces repeated absolute-weight work with a reused activation
buffer. Integer pair sums remain within int16 bounds. Raw Q8 -128 retained the
original signed-dot fallback, preserving legacy wrap behavior. FMA scaling,
accumulation and reduction order were unchanged.

The prototype reused the offset-dot selector only in its scratch binary and
disabled the narrow path there. Production source and flag meanings were not
changed. The existing raw finite-scale/extreme-input tests and synthetic/real
expert tests passed bitwise, including final-block -128 and partial-row guards.

A 604 MB, sixteen-physical-thread streaming screen used baseline/candidate/
candidate/baseline ordering. Baseline packed bandwidth was 29.826 and 27.699
GB/s; candidate was 27.761 and 27.832 GB/s. All 36 timed passes plus warmups
matched GGML; output hashes agreed across runs. Host load varied and these
measurements establish no repeatable improvement. The candidate was not
integrated. No GPU or cross-backend production change was made.

Raw logs, parity transcript, source/prototype hashes and compact prototype diff
are in `results/20261020/fp4-absolute-activations/`. The result shows why reducing
one inner-loop operation is insufficient evidence of faster service when it
also adds activation-buffer traffic. That explanation is a hypothesis, not a
measured hardware-counter attribution.

### Host-load and FP4 counter audit (2026-10-20)

An unchanged streaming baseline had fallen substantially below earlier results.
A read-only host-level two-second sample found approximately 6.28 CPU cores of
browser activity plus one core of Python work. The sandbox process namespace
showed only three processes, versus 827 from the host-level inspection, so an
empty sandbox process-usage list did not prove an idle machine. No workload was
stopped or modified. Configured DIMM speed could not be obtained; the theoretical
memory-bandwidth denominator remains unverified.

A cached-row counter probe of the absolute-activation prototype was pinned to
CPU 11 after a short per-core activity sample. It performed 2,097,152 row dots
at K=4096 with 128 stored rows and checked GGML parity before and after. The
baseline/candidate/candidate/baseline order produced:

| Version | Instructions, billion | Cycles, billion | Task CPU ms | Context switches |
| --- | --- | --- | --- | --- |
| Baseline | 7.247 | 2.813 | 819.82 | 312 |
| Candidate | 7.144 | 2.947 | 876.07 | 345 |
| Candidate repeat | 7.153 | 4.763 | 1507.71 | 647 |
| Baseline repeat | 7.257 | 4.878 | 1548.11 | 683 |

The candidate reduces retired instructions by about 1.4%, but unchanged cycle
counts vary by roughly 1.7x. Context switches and observed host load qualify the
measurement; they do not alone identify every source of variation. This probe
supports the instruction-count observation, not a reliable cycle improvement.
The preceding streaming test also showed no repeatable gain, so the prototype
remains outside production. Do not use these runs as idle-system peak evidence.

Counter logs, parity output, probe snapshot and source hashes are in
`results/20261020/fp4-counter-load-audit/`. Future small-gain comparisons need
same-process controls and workload/CPU-time records; no background workload
change is implied or required by this report.

### Rotating real FP4 weight buffers (2026-10-20)

The pool benchmark accepts `fixed|task-sweep` followed by optional `BANKS`
(1–16, default 1). Banks are physically separate copies of the same real expert
weights, including scale tails. Every warmup, timed call and CPU-batch call
advances the bank. For BANKS>1 only complete-blob split/flow modes run; NUMA-owned
shards are not constructed. Allocation/copying is outside timing. This increases
working-set size without downloading additional layers; it does not replay a
real routing trace or guarantee a cache miss on each access.

On the single-NUMA Ryzen 3950X worker, three experts use 42,467,376 packed bytes
per bank; eight banks rotate through 339,739,008 bytes. Four processes ran in
1/8/8/1 bank order with sixteen participants, unchanged default dot arithmetic,
software prefetch off, and the existing background load. Single-token flow
medians in milliseconds were:

| Layer | One bank, first | Eight banks, first | Eight banks, repeat | One bank, repeat |
| --- | --- | --- | --- | --- |
| 3 | 1.4820 | 1.8469 | 1.7670 | 1.4239 |
| 4 | 1.7402 | 4.5178 | 1.7673 | 1.5689 |

Layer 3 shows a repeatable working-set penalty in this screen; layer 4 includes
a large outlier. Four-token timings were also noisy and did not give a stable
bank-size ordering. The result reinforces that repeated partial weights can be
optimistic for full-checkpoint service, but does not isolate cache effects from
all load variation or establish a replacement simulator rate. Physical buffer
copies also lack the variation of distinct experts and activation/routing traces.

All 288 worker timed checks and 32 final CPU-batch checks matched bitwise,
including hashes from the prior real-weight test. HIP- and CUDA-configured local
runs also passed four-bank real-weight checks. Existing single-bank operation
retains its six modes where NUMA ownership is available. No production kernel,
default, GPU workload or miner setting changed. No SYCL validation was available.

```sh
STRATA_NO_NVFP4_512=1 STRATA_NVFP4_PREFETCH=0 \
  STRATA_NVFP4_128=0 STRATA_NVFP4_OFFSET_DOT=0 \
  BUILD_DIR/glm_fp4_native_pool_bench PARTIAL_MODEL_DIR 3 15 fixed 8
```

Logs, source/static-binary hashes and benchmark snapshot are in
`results/20261020/worker-rotating-weights/`. Large buffers and downloaded weights
remain outside shared evidence.

### Worker prefetch with rotating FP4 weights (2026-10-20)

The Ryzen 3950X worker's three-expert service was screened with eight physical
weight banks (339,739,008 packed bytes), sixteen participants, original 256-bit
dots and prefetch distances 0/512/2048/4096/0 bytes. A follow-up used 512/0/512.
All runs reused the same benchmark binary and partial fixture. The miner and
other workloads remained running. This tests the existing prefetch knob; no
kernel or default changed.

Single-token flow medians for the follow-up were:

| Prefetch bytes | Layer 3 ms | Layer 4 ms |
| --- | --- | --- |
| 512, first | 1.6670 | 1.6180 |
| 0 | 1.7668 | 1.7582 |
| 512, repeat | 1.7090 | 1.7639 |

For layer 3, 512 bytes improved latency by 3.3–5.6% against the intervening
control. Layer 4 improved 8.0% initially but was effectively tied in the repeat
(0.3% slower). The initial sweep also favored 512 bytes for single-token flow:
1.626/1.652 ms versus no-prefetch endpoints 1.827/1.744 and 1.779/1.839 ms.
Larger distances were inconsistent. Four-token results varied widely and do not
qualify a setting; no general batch-size speedup is claimed.

512 bytes is a candidate for a quieter single-token worker validation, not a
new default or an established full-decoder improvement. Separate-process host
variation and repeated underlying expert contents limit the conclusion. Keep
the raw controls and do not select the best run alone. All 576 timed outputs and
64 final CPU-batch checks matched bitwise and retained prior worker hashes.
No GPU execution occurred. Logs and binary provenance are in
`results/20261020/worker-rotating-prefetch/`.

### Real-weight FP4 clamp stress coverage (2026-10-20)

The native FP4 test previously used amplitude 1 for its real layer 3/4 expert
fixtures, which produced zero clamped rows. Synthetic tests already exercised
clamping, but the real-weight path needed explicit boundary coverage before
further gate/up optimization. Non-benchmark real-fixture tests now also use
synthetic activation amplitudes +4096 and -4096 at one/four tokens. Each new
case fails unless it exercises all three boundaries: gate above +10, up above
+10 and up below -10. Benchmark inputs and production behavior are unchanged.

Both HIP- and CUDA-configured CPU builds passed all four combinations of the
128-bit and offset-dot options, with AVX512 disabled. Each invocation covers
80 existing synthetic expert cases, 12 real expert cases (eight clamp stress
cases) and the existing raw-Q8 finite-scale/extreme-input checks. All 64 new
real stress cases across the eight invocations matched the clamped GGML
reference bitwise for complete blobs and independently allocated NUMA row
shards, including hidden quantization and down projection. Counts of each clamp
direction are preserved in the output. These are tests with real weights and
artificially extreme activations, not a model-quality evaluation or realistic
performance workload. No GPU ran; SYCL validation remains unavailable.

Logs, test-source snapshot/hash and checked boundary counts are in
`results/20261020/fp4-real-clamp-stress/`. Pass a partial model directory to
`glm_fp4_native_test` to run the added real cases; model-free CTests continue
to cover the synthetic/raw cases without downloads.

### SSE clamp prototype: not integrated (2026-10-20)

Object inspection confirmed scalar `fminf`/`fmaxf` call relocations in the AVX2
NVFP4 gate/up code. An isolated prototype replaced the three clamp operations
with SSE scalar min/max, preserving positive-limit clipping for finite inputs.
All four AVX2 option combinations passed synthetic/raw tests and real-weight
clamp stress tests, including separate NUMA row shards.

However, a separate scalar-clamp probe over one million pseudorandom float bit
patterns at three positive limits found 5,985 NaN-input disagreements with the
current standard-library expression. It found no disagreements among sampled
non-NaN inputs. This is a sampled compatibility check, not an exhaustive proof;
exceptions/NaN payload behavior must not be inferred from finite-value tests.
The prototype therefore does not preserve the full observed scalar contract.

The baseline/candidate/candidate/baseline real-pool screen also gave no stable
speedup. Unfused NUMA flow single-token medians for layer 3 were
0.552/0.549/0.568/0.554 ms; layer 4 was 0.585/0.572/0.556/1.043 ms, including a
large baseline outlier. Four-token timings were similarly mixed. All 864 timed
real outputs and 96 final CPU-batch checks matched, as expected for these finite
fixtures. No production source or default changed, and no GPU ran. No further
cross-backend validation was attempted for this rejected scratch prototype.

The compact diff, parity logs, sampled clamp probe, source hashes and pool
results are in `results/20261020/fp4-sse-clamp/`. Keep scalar clamp semantics;
removing three library calls has not established a worthwhile compatible gain.

### Q2/Q3 complete-blob service on the worker (2026-10-20)

The synthetic Q2/Q3 pool benchmark now supports single-NUMA machines by
assembling complete gate/up/down expert blobs when owned NUMA rows are
unavailable. The existing owned-row path remains intact. A requested fused
quantization sweep is rejected on the unowned path rather than silently running
a different mode. This is benchmark coverage only; no kernel/default changed.

A static build ran on the Ryzen 3950X worker with eight synthetic experts,
H=4096, F=2048, Q2_K gate/up and Q3_K down, fifteen workers plus host. Packed
weights total 72,876,032 bytes. Twenty timed flow calls per process include
GU/SiLU/hidden quantization/down/reduction and exclude input quantization. The
Q2 lookup table was tested in off/on/on/off order; software prefetch and native
local scheduling were disabled. Background load and miner settings were unchanged.

| Option/run | One-token median ms | Four-token median ms |
| --- | --- | --- |
| LUT off, first | 2.9588 | 3.7780 |
| LUT on, first | 3.1083 | 3.8747 |
| LUT on, repeat | 3.0795 | 3.6656 |
| LUT off, repeat | 3.0685 | 5.1334 |

Single-token results show no LUT advantage in this configuration; four-token
results are mixed, including a slow final control. Keep the worker LUT setting
off pending a better-qualified workload comparison. This does not negate the
separate TR16 experiments or establish a universal CPU policy. The data are
synthetic and repeatedly reused, not a full-model routing trace or tok/s result.

All 160 timed outputs and reductions matched the phased reference bitwise;
hashes agreed across LUT settings. The regular pool parity test passed in both
HIP- and CUDA-configured CPU builds, and the primary host retained its NUMA
benchmark path. No GPU ran. Source/static-binary hashes, final benchmark snapshot
and logs are in `results/20261020/worker-q23-lut/`. SYCL validation remains
unavailable.

### Conditional 90% CPU bandwidth budget (2026-10-20)

`cpu_peak_budget.py` makes the bandwidth target explicit without claiming an
observed peak. Existing reports describe four-channel DDR4-3200, but current
configured DIMM speed remains unverified. Under that assumption, 64 payload
bits per channel give 102.4 GB/s theoretical bandwidth and a 90% target of
92.16 GB/s. Native NVFP4 stores 36 bytes per 64 weights, so packed weights alone
would require 163.84 Gweights/s. Activation/output and other traffic are extra.

For sixteen physical cores, that target allows at most 0.293 cycles/weight at
an assumed 3.0 GHz, or 0.332 at an assumed 3.4 GHz. These are explicit frequency
scenarios, not measured sustained all-core clocks. The prior cached-row probe
had whole-process baseline counts of 0.327 and 0.568 cycles/weight. Perfect
sixteen-core scaling at 3.4 GHz would project 93.44 and 53.88 GB/s respectively.
This is an idealized arithmetic projection, not measured bandwidth or a hardware
ceiling; the counters include initialization/reference checking and vary under
host load. Even the faster sample leaves little room for shared-resource costs.

The absolute-activation candidate's corresponding projections are 89.21 and
55.19 GB/s, again not a demonstrated improvement. Integer-dot/unpack throughput
must not be compared with advertised FP32 FMA FLOPS as a utilization percentage.
The target requires both sufficient DRAM bandwidth and sufficient dot/unpack
throughput concurrently. The earlier approximately 81 GB/s read-only reference
also falls below 92.16 GB/s; it is a measured control, not a proof of physical
peak or a fixed upper bound. A 90%-of-read-control ratio cannot establish the
requested 90%-of-theoretical target.

```sh
python3 experiment/glm_fp4_local/cpu_peak_budget.py \
  --memory-mts 3200 --channels 4 --cores 16 --ghz 3.0 3.4
```

The unit arithmetic was checked against 102.4 GB/s, 92.16 GB/s and 163.84
Gweights/s. Output and explicit assumptions are in
`results/20261020/cpu-conditional-peak/budget.json`. No simulator default, kernel,
clock, memory setting or workload changed. This report does not establish full
model tok/s or completion of the utilization target.

### FP4 huge-page advice and NUMA placement (2026-10-20)

The standalone streaming benchmark accepts a final page-advice selector after
placement: `default` (unchanged allocation behavior), `huge` or `nohuge`. On
Linux, advice applies only to complete pages inside each weight allocation,
before pinned first touch. No global THP policy is changed. The benchmark now
reports advised bytes, process-wide anonymous huge-page KiB and anonymous NUMA
page counts. These counters are diagnostics, not buffer-specific residency.

On Threadripper 1950X with sixteen physical threads and 603,979,776 packed bytes,
the system THP policy was `madvise`. A nohuge/huge/huge/nohuge screen obtained
557,056 KiB of process anonymous huge-page backing in both advised runs, versus
zero in controls. FP4 bandwidth was 43.41/29.96/33.46/49.11 GB/s respectively.
The large regression justified checking page placement rather than promoting
the advice based on expected TLB savings.

A follow-up with placement counters showed:

| Advice | Anonymous pages N0 / N1 | Huge KiB | Read GB/s | FP4 GB/s |
| --- | --- | --- | --- | --- |
| huge | 111088 / 37392 | 540672 | 48.67 | 42.69 |
| nohuge | 74727 / 73745 | 0 | 79.01 | 50.36 |

Huge-page advice coincided with approximately 75% of anonymous pages on node 0,
while the nohuge control was nearly balanced. This is consistent with NUMA
placement contributing to the slowdown, but does not prove the allocation
mechanism or exclude host-load effects. Do not interpret these runs as the
inherent performance of correctly placed huge pages. Pinned first touch alone
was insufficient to guarantee balanced placement here. Keep default allocation
behavior; explicit placement validation is needed before another huge-page
optimization is integrated.

All 54 timed FP4 passes plus warmups matched GGML; read checksums and output
hashes agreed. The final tool also linked against the CUDA-configured CPU
archives and passed a small huge-advice smoke run. No GPU ran. Only benchmark
code changed; production allocation and miner settings are untouched. Logs,
source snapshot/hash and metadata are in `results/20261020/fp4-page-advice/`.

### Explicitly bound FP4 page-advice test (2026-10-20)

The standalone stream benchmark adds `huge-bind` and `nohuge-bind` selectors.
Each discovers its selected CPU's NUMA node through sysfs, binds only complete
pages inside that worker's allocation with `MPOL_BIND`, then applies page advice
before first touch. Unsupported nodes or a failed binding syscall fail the run;
there is no silent fallback. Other modes retain their previous behavior. No
system-wide policy or another process's memory placement is changed.

On Threadripper 1950X, a bound nohuge/huge/huge/nohuge screen restored nearly
balanced process anonymous page counts in all four runs. The huge runs each
obtained 286,720 KiB (280 MiB) of anonymous huge pages, less than half of the
576 MiB packed-weight allocation; these remain process-wide counters. FP4
bandwidth was 43.91/41.67/38.74/38.19 GB/s. This shows no repeatable huge-page
advantage under current conditions. Balanced node totals alone do not prove
all pages reside beside their intended worker, and the counters do not identify
why the remaining allocation used small pages. Do not infer a specific
fragmentation mechanism from these data.

All 36 timed FP4 passes plus warmups matched GGML and all read checksums passed.
The final benchmark also linked against CUDA-configured CPU archives and passed
a small bound-huge smoke test. No GPU or production allocation change was made.
Logs, source snapshot/hash and configuration are in
`results/20261020/fp4-bound-pages/`. The diagnostic supports testing placement
explicitly; it does not establish the 90% utilization target or justify enabling
huge pages by default.

### Per-worker FP4 page residency check (2026-10-20)

The stream benchmark now queries residency after worker completion, taking one
page sample per 2 MiB of each weight buffer and comparing its node with the
assigned worker CPU's sysfs node. `move_pages` uses null destination nodes and
zero flags: this only queries residency and never migrates memory. Unsupported
or failed queries are reported as unknown with the errno, not as local pages.
The query is outside all benchmark intervals.

On Threadripper 1950X, three 604 MB runs gave:

| Mode | Local / remote / unknown samples | Read GB/s | FP4 GB/s |
| --- | --- | --- | --- |
| huge, unbound | 152 / 136 / 0 | 32.98 | 25.83 |
| huge, bound | 288 / 0 / 0 | 79.08 | 42.06 |
| nohuge, bound | 288 / 0 / 0 | 78.29 | 37.93 |

The unbound run had nearly half its sampled pages remote to their assigned
worker. Both bound runs had entirely local samples. This corroborates the NUMA
placement concern from aggregate counters. Sampling is post-run and sparse; it
is not proof of every page's location throughout execution, nor does this small
screen isolate host-load effects. Bound huge pages do not become a recommended
default on this evidence. The production `NumaTensor` owned-row allocator already
binds arenas; this diagnostic is not a newly implemented production placement
optimization.

All 27 timed FP4 passes plus warmups matched GGML, with matching checksums/hashes.
The final diagnostic also linked against CUDA-configured CPU archives and passed
a small bound-buffer smoke run. No GPU or production behavior changed. Logs,
per-worker residency counts, source snapshot/hash and summary are in
`results/20261020/fp4-page-residency/`.

### Dot variants with sampled-local streaming buffers (2026-10-20)

The existing signed/biased 256-bit and 128-bit FP4 options were re-screened with
16 physical threads, 604 MB, `nohuge-bind` allocation and prefetch disabled.
Each run's 288 post-run residency samples were local, with zero remote/unknown
samples. This removes the demonstrated unbound huge-page placement issue from
this particular comparison, though sparse post-run samples do not prove every
page's location during timing.

The first baseline/biased/narrow/combined/baseline sequence gave FP4 packed
bandwidth 36.62/40.71/44.84/41.97/43.22 GB/s. A signed-narrow/baseline/narrow
follow-up gave 44.84/39.60/42.83 GB/s; read controls were 75.75/74.16/75.01 GB/s.
Process CPU time in that follow-up was 137.06/145.65/138.74 ms. Signed 128-bit
therefore remains a plausible streaming candidate with local buffers, but this
screen does not establish a decoder gain or eliminate host-load confounding.

The subsequent real three-expert NUMA/dataflow baseline/narrow/narrow/baseline
screen was mixed. Single-token layer-3 medians were 1.030/2.217/0.685/0.712 ms;
layer 4 was 0.896/0.963/0.800/0.755 ms. The unchanged four-token path also varied
substantially, so the real-service data do not qualify a stable improvement.
Retain the opt-in status of `STRATA_NVFP4_128`; do not change its default from
streaming numbers alone.

All 72 timed streaming passes and 864 timed real-service checks matched their
references; the 96 final real CPU-batch checks also matched, with equal hashes
across variants. No source/kernel/default changed for this screen and no GPU
ran. Full logs, residency evidence and source hashes are in
`results/20261020/fp4-local-variant-screen/`. This remains partial CPU evidence,
not 90% theoretical peak or full-model throughput.

### Opt-in unfused selection for one or two FP4 experts (2026-10-20)

Two-expert real-weight comparisons on Threadripper 1950X showed lower unfused
NUMA/dataflow latency in all eight layer/token comparisons across two runs.
Reductions ranged from 3.2% to 26.6%, with substantial host-load variation. This
extends the evidence beyond the earlier one-expert case, but not to three or
more experts or arbitrary CPU geometries.

`STRATA_NVFP4_SMALL_UNFUSED=1` now selects the existing unfused schedule for one
or two NVFP4 CPU experts when fusion was requested. It retains the prior limits:
owned NUMA rows, H=4096, F=2048, sixteen participants, one/four tokens and
noncanonical arithmetic. Only exact `1` enables it, using the shared C11 helper.
The previous `STRATA_NVFP4_SINGLE_UNFUSED=1` retains its one-expert-only behavior;
the Q2/Q3 option also retains its existing scope. All switches default off.
The existing synchronous local-format-copy path keeps caller state and dot,
quantization, clamp and reduction arithmetic unchanged.

Integrated validation covered one/two/three experts, the new switch values
0/1/2, and both token widths. All 1,080 timed checks and 120 final CPU-batch
checks matched bitwise. The preceding two-expert screen had 432 timed checks.
In the integrated two-expert enabled run, requested-fused versus explicitly
unfused single-token medians were 0.422/0.419 ms (layer 3) and 0.413/0.421 ms
(layer 4); four-token medians were 1.195/1.184 and 1.193/1.182 ms. This confirms
that the selection tracks the existing unfused path. Separate off/on process
timings are not a qualified full-decoder speedup. Value 2 leaves the switch off,
and three-expert selection remains unchanged.

Six scoped CPU regressions passed in each of the HIP- and CUDA-configured
builds. A CUDA-configured two-expert real run passed another 216 timed checks
and 24 final CPU-batch checks. No GPU ran. SYCL validation remains unavailable
because its configured compiler is missing. Default production behavior remains
unchanged; this is an opt-in candidate, not a full-model quality or tok/s claim.

Use `STRATA_NVFP4_SMALL_UNFUSED=1` with the existing dataflow path
(`STRATA_GLM_LAYER_FLOW=1`) when testing this configuration. Source hashes,
selection block, initial/integrated logs and test evidence are in
`results/20261020/fp4-small-unfused/`.


### Q2/Q3 two-expert unfused selection (2026-10-20)

On Threadripper 1950X, two repeated synthetic H=4096/F=2048 two-expert
Q2_K gate/up plus Q3_K down sweeps favored unfused execution. At the default
three tasks per participant, one-token unfused/fused medians were
0.329/0.403 and 0.327/0.399 ms; four-token medians were 1.211/1.512 and
1.278/1.524 ms. These are approximately 18% and 16–20% latency reductions,
respectively, under uncontrolled host load. Weights were repeatedly reused;
input quantization was excluded. This is CPU expert service, not model decode.

`STRATA_Q23_SMALL_UNFUSED=1` extends the existing opt-in scheduling choice to
one or two Q2_K/Q3_K experts. It requires owned NUMA rows, H=4096/F=2048,
sixteen participants, one/four tokens, requested fusion and noncanonical
arithmetic. The shared C11 helper accepts only exact `1`. The previous
`STRATA_Q23_SINGLE_UNFUSED` still selects only one expert. Defaults remain off;
the FP4 selection and all arithmetic are unchanged. Decoder testing also needs
the existing `STRATA_GLM_LAYER_FLOW=1` path.

The integrated two-expert sweep covered switch values 0/1/2 and both token
counts, with 720 timed output/reduction comparisons matching the phased
reference bitwise. Enabled requested-fused/unfused medians at three tasks were
0.401/0.394 ms for one token and 1.566/1.560 ms for four tokens. Values 0 and 2
retained the fusion penalty in the same-process comparisons. Logged
`fuse_h_quant` describes the caller request, before the opt-in selection.

Initial and integrated evidence, medians, the selection block and source hashes
are in `results/20261020/q23-small-unfused/`. These measurements do not establish
90% theoretical peak utilization or an end-to-end tok/s improvement.

Six scoped CPU regressions passed in each HIP- and CUDA-configured build.
An additional CUDA-configured sweep with the option enabled covered one,
two and three experts at both token counts: all 720 timed comparisons matched
bitwise. Three experts remain outside the source selection guard. These were
CPU executions; no GPU benchmark ran. SYCL validation remains unavailable
because the configured compiler is missing.


### Q2/Q3 rotating NUMA weights (2026-10-20)

The CPU benchmark now accepts an optional final `BANKS` argument (1–16,
default 1). Each bank has separately allocated NUMA gate/up/down tensors, or a
separate complete blob on single-NUMA systems. Allocation and copying are
outside timing; pointer selection is inside each call. Warmup also advances
the bank. Copies contain identical values, so every timed output and reduction
can be checked bitwise against the original phased reference. This is a memory
working-set experiment, not a replay of model routing or proof of cold caches.

Threadripper 1950X, H=4096/F=2048, two Q2_K/Q3_K experts, sixteen participants,
LUT enabled, prefetch disabled, both unfused-selection switches disabled:
eight banks occupy 145,752,064 packed bytes versus 18,219,008 for one bank.
At the default three tasks per participant, two eight-bank runs gave unfused/
fused medians of 0.401/0.478 and 0.560/0.578 ms for one token. Four-token
medians were 1.993/2.085 and 2.055/2.161 ms. Thus unfused latency reductions
were 3.1–16.2% for one token and 4.4–4.9% for four, smaller and less stable than
the earlier repeatedly reused two-expert results. Do not apply the earlier
16–20% four-token gain to streaming expert service.

One-bank controls were 0.358/0.435 and 0.374/0.449 ms for one token, and
6.578/3.296 and 1.407/1.699 ms for four tokens. The large first four-token
outlier is retained: host load was uncontrolled. All 960 timed output/reduction
comparisons passed. Initial performance binaries preceded a comment/indentation
cleanup only. No production default changed and no full-model throughput or
90% peak claim follows. Evidence is in `results/20261020/q23-rotating-weights/`.

Reproduce with `q23_pool_test --bench 1 15 20 flow-quant-sweep 2 8`, setting
`STRATA_Q2K_LUT=1`, `STRATA_Q23_PREFETCH=0`, `STRATA_NATIVE_NUMA_LOCAL=0`,
`STRATA_Q23_SINGLE_UNFUSED=0` and `STRATA_Q23_SMALL_UNFUSED=0`.

The final benchmark source built in HIP- and CUDA-configured trees; the Q2/Q3
CPU regression passed in both. A CUDA-configured eight-bank run with
`STRATA_Q23_SMALL_UNFUSED=1` passed another 120 timed bitwise comparisons.
No GPU ran. SYCL remains unvalidated because the configured compiler is absent.


### Real FP4 rotating NUMA weights (2026-10-20)

The real partial-weight CPU benchmark now retains all six execution modes when
`BANKS>1` on a NUMA-capable host. Each bank has separately allocated gate/up/down
NUMA tensors; each invocation selects matching row shards and a complete blob
that owns the global scale tail. Complete-blob modes rotate their copies too.
Allocation and copying remain outside timing. Warmup and CPU-batch calls also
advance the bank. Bank count defaults to one; production code is unchanged.

Threadripper 1950X, sixteen participants, real layers 3/4, experts 0/1,
H=4096/F=2048, baseline AVX2 dot, prefetch disabled, one/four tokens: eight banks
represent 226,492,672 packed bytes per layout, versus 28,311,584 for one bank.
Both complete and owned layouts are allocated, so this reported working set is
not total process memory. Input quantization is excluded. Fixed inputs, repeated
identical bank contents and sequential bank rotation are not real routing replay
or proof that each access misses cache. The native path uses Q8 activations,
not the exact BF16 W4A16 reference.

With both unfused-selection switches off, eight-bank NUMA/dataflow medians were:

| Run | Layer | Tokens | Unfused ms | Fused ms |
|---|---:|---:|---:|---:|
| 1 | 3 | 1 | 0.583 | 0.906 |
| 1 | 4 | 1 | 0.525 | 0.728 |
| 2 | 3 | 1 | 0.555 | 0.727 |
| 2 | 4 | 1 | 0.681 | 0.888 |
| 1 | 3 | 4 | 2.014 | 2.197 |
| 1 | 4 | 4 | 1.436 | 1.828 |
| 2 | 3 | 4 | 1.483 | 1.858 |
| 2 | 4 | 4 | 2.156 | 2.264 |

Unfused latency was lower in every comparison: 23.3–35.6% for one token and
4.8–21.4% for four. One-bank controls and all raw latency samples are retained.
Host load was uncontrolled; these comparisons support retaining the existing
opt-in, not changing a default or claiming end-to-end decode throughput.

The benchmark built in HIP- and CUDA-configured trees. Two initial one-bank
and two eight-bank runs passed 864 timed output/reduction checks and 96 final
CPU-batch checks. Enabled `STRATA_NVFP4_SMALL_UNFUSED=1` eight-bank runs in both
builds passed another 432 timed and 48 final CPU-batch checks, with identical
reference hashes. Requested-fused and explicit-unfused enabled results tracked
within about 9%, consistent with host variation; this is not a separate-process
speedup measurement. No GPU ran. SYCL remains unavailable. No 90% peak claim.

Reproduce with `glm_fp4_native_pool_bench PARTIAL_MODEL_DIR 2 15 fixed 8`,
setting `STRATA_NO_NVFP4_512=1`, `STRATA_NVFP4_128=0`,
`STRATA_NVFP4_OFFSET_DOT=0`, `STRATA_NVFP4_PREFETCH=0`,
`STRATA_NATIVE_NUMA_LOCAL=0`, `STRATA_NVFP4_SINGLE_UNFUSED=0` and
`STRATA_NVFP4_SMALL_UNFUSED=0` (or 1 for the integrated selection).
Source hashes, logs and summaries are in
`results/20261020/fp4-numa-rotating-weights/`.


### FP4 prefetch on rotating NUMA weights (2026-10-20)

The larger-working-set benchmark tested the existing AVX2 software-prefetch
setting on Threadripper 1950X with two real experts, layers 3/4, sixteen
participants, eight banks (226,492,672 bytes per layout), baseline dot arithmetic
and `STRATA_NVFP4_SMALL_UNFUSED=1`. Input quantization was excluded; background
load was uncontrolled. Distances were 0/512/2048/4096/0 bytes, followed by
2048/0/2048. No source or production default changed.

The first sweep mixed modest gains with regressions and large outliers. The
4096-byte run's layer-4 single-token requested-fused dataflow median was
3.982 ms, versus 0.848/0.731 ms in the surrounding no-prefetch runs. This is
retained as observed, not attributed solely to prefetch.

The follow-up requested-fused (effectively unfused) NUMA/dataflow medians were:

| Prefetch bytes | Layer 3, one token ms | Layer 4, one token ms | Layer 3, four tokens ms | Layer 4, four tokens ms |
|---:|---:|---:|---:|---:|
| 2048 | 0.552 | 0.546 | 1.449 | 1.511 |
| 0 | 0.559 | 0.518 | 1.470 | 1.520 |
| 2048 | 0.838 | 0.518 | 1.933 | 1.463 |

There is no repeatable improvement across layers/token widths. The first
2048-versus-zero CPU-cost batches were also nearly identical for layer 3
(one token 8.969/8.967 CPU ms, four tokens 23.936/23.931 CPU ms). These timings
do not justify changing the existing 2048-byte production default or
recommending a different distance for this host. Prefetch tuning alone has not
closed the throughput gap; a new kernel instruction/traffic reduction is a
better next experiment than extending this distance sweep.

All 1,728 timed output/reduction comparisons and 192 final CPU-batch checks
passed bitwise. CPU execution used the HIP-configured build previously
validated for this benchmark; no new backend code or GPU execution. This is
not a full-model decode measurement or a theoretical-peak utilization claim.
Logs, selected-mode summary and source hashes are in
`results/20261020/fp4-numa-prefetch/`. Reproduce the preceding rotating-NUMA
command with `STRATA_NVFP4_SMALL_UNFUSED=1` and the distances listed above.


### FP4 packed activation-load prototype (2026-10-20)

A scratch AVX2 prototype copied Q8 activation bytes from their 34-byte blocks
into aligned 32-byte vectors when preparing each row range. The inner loop then
loaded contiguous vectors; scale conversion, integer dots, FMA ordering and
reduction were unchanged. The existing Q8 pointers were retained for other
specializations. Copying and the larger activation scratch were included in
service timing. This was a prototype, not a production switch or format change.

The prototype passed the synthetic/raw-Q8 and real partial-weight parity suite,
including positive/negative high-amplitude clamp stress. Baseline/candidate/
candidate/baseline tests used Threadripper 1950X, sixteen participants, real
layers 3/4 with two experts, eight NUMA banks, small-expert unfused selection on,
baseline 256-bit signed dots and prefetch off. Requested-fused NUMA/dataflow
medians (effectively unfused) were:

| Variant | Layer 3, one token ms | Layer 4, one token ms | Layer 3, four tokens ms | Layer 4, four tokens ms |
|---|---:|---:|---:|---:|
| Baseline | 0.551 | 0.575 | 1.474 | 1.482 |
| Candidate | 0.635 | 0.537 | 2.076 | 1.476 |
| Candidate | 0.548 | 0.789 | 1.476 | 1.880 |
| Baseline | 0.531 | 0.544 | 1.483 | 1.442 |

There is no consistent latency improvement. Host load remained uncontrolled;
these data do not isolate copy cost from scheduling interference. All 864 timed
output/reduction comparisons and 96 final CPU-batch checks matched bitwise,
including reference hashes across baseline/candidate processes. The candidate
is not integrated: extra activation copying/storage is not justified by these
results. Production source and defaults remain unchanged.

This scratch object was linked ahead of the HIP-configured CPU archive for
CPU-only testing. CUDA/SYCL compilation of the rejected prototype and GPU
execution were not performed. No full-model or peak-utilization claim follows.
The compact patch against the recorded source hash, parity output and raw pool
logs are in `results/20261020/fp4-prepacked-activations/`.


### Four-token 128-bit FP4 dot prototype (2026-10-20)

A scratch AVX2/FMA prototype split each four-token accumulator into low/high
128-bit halves, retaining the baseline lane/FMA/reduction order. Each decoded
weight pair was shared by four tokens. Only `row_dot<4>` used this prototype;
one-token execution and all production source remained unchanged. The intent
was to reduce 256-bit lane assembly on Threadripper 1950X without reloading
weights separately for each token.

Synthetic/raw-Q8 and real partial-weight parity passed, including large positive
and negative activation clamp stress. Baseline/candidate/candidate/baseline
runs used sixteen participants, two real experts from layers 3/4, eight NUMA
banks, small-expert unfused selection enabled and prefetch disabled. The
requested-fused NUMA/dataflow four-token medians were:

| Variant | Layer 3 ms | Layer 4 ms | Layer 3 CPU ms/call | Layer 4 CPU ms/call |
|---|---:|---:|---:|---:|
| Baseline | 2.081 | 1.479 | 37.672 | 23.415 |
| Candidate | 1.502 | 1.518 | 24.427 | 24.418 |
| Candidate | 1.551 | 2.033 | 24.428 | 35.797 |
| Baseline | 1.432 | 1.449 | 22.939 | 23.911 |

Host load was uncontrolled; retain the slow baseline and candidate samples.
The unchanged one-token control also varied (layer 3: 0.723/0.549/0.555/0.535 ms).
The candidate does not establish a throughput win. In less-disrupted samples,
its four-token latency and process CPU cost are slightly higher than baseline.
Do not integrate this prototype or extend the current one-token 128-bit option
on the strength of these results.

All 864 timed output/reduction comparisons and 96 final CPU-batch checks passed
bitwise, with matching baseline/candidate hashes. Scratch CPU compilation and
linking used the HIP-configured libraries; CUDA/SYCL compilation and GPU testing
of this rejected candidate were not performed. Production defaults are unchanged;
full-model decode and 90% peak utilization remain unverified. Patch, source hashes,
parity output and timing logs are in `results/20261020/fp4-four-token-128/`.


### Four-token prototype instruction audit (2026-10-20)

A single-thread hardware-counter comparison investigated the rejected 128-bit
four-token prototype. Threadripper 1950X, logical CPU 11, K4096, 128 repeatedly
reused rows, four distinct Q8 activation vectors, 16,384 calls, prefetch off.
GGML parity was checked before and after timing, including a raw -128 activation.
`perf stat` measured the whole process, including initialization and validation;
these are not exclusively inner-loop counts or DRAM bandwidth measurements.

| Order | Variant | Instructions, billion | Cycles, billion | Task-clock ms | Context switches |
|---:|---|---:|---:|---:|---:|
| 1 | Baseline | 15.889 | 8.401 | 2745.45 | 1041 |
| 2 | Candidate | 24.486 | 9.599 | 3148.31 | 1197 |
| 3 | Candidate | 24.478 | 8.535 | 2729.34 | 1018 |
| 4 | Baseline | 15.891 | 9.604 | 3069.26 | 1161 |

The candidate executes approximately 54% more instructions in both runs.
Cycles/task-clock remain affected by uncontrolled load, so they do not establish
a stable latency ratio. The instruction result gives a firmer reason to reject
this implementation than the earlier service timings alone. Smaller SIMD width
has not reduced executed work for four-token reuse. Future candidates should
first demonstrate lower instruction cost without sacrificing exact arithmetic,
then pass memory-service tests; no peak-utilization gain is established here.

No production code changed. Raw counters, probe snapshot/hash, timing and parity
outputs are in `results/20261020/fp4-four-token-counters/`. Reproduce with
`STRATA_NVFP4_PREFETCH=0 perf stat -x, -e cycles,instructions,task-clock,context-switches`
followed by `taskset -c CORE PROBE 128`, using the baseline or scratch prototype
object described in the previous experiment. The chosen core and existing host
load must be recorded when comparing results.


### Two-way FP4 block-loop unrolling (2026-10-20)

A scratch candidate added `#pragma GCC unroll 2` to the baseline signed AVX2
block loop, preserving vector layout, scales and FMA/reduction order. The
synthetic/raw-Q8/real clamp-stress suite passed. No production source changed.

The initial four-token counter screen omitted production `-fPIC` on the scratch
object. Its raw evidence is retained as exploratory only. Both baseline and
candidate were then rebuilt with matching O3/NDEBUG/native-expert/PIC/C++20 and
AVX2/FMA/F16C flags. Whole-process counters on Threadripper 1950X logical CPU 11,
128 repeatedly reused rows, K4096, 16,384 calls and prefetch off showed:

| Tokens | Baseline instructions (two runs), billion | Candidate instructions (two runs), billion |
|---:|---|---|
| 1 | 7.248 / 7.247 | 7.045 / 7.045 |
| 4 | 15.895 / 15.886 | 15.823 / 15.820 |

Instruction reductions were about 2.8% for one token and 0.4% for four. Single-
token cycles were 2.901/2.823 billion baseline versus 2.817/2.807 candidate;
four-token cycles varied substantially. All probes passed GGML parity including
raw -128. Counts include initialization and checks. Object text grew from
125,824 to 164,375 bytes (30.6%); this is not a measurement of instruction-cache
misses or the hot-loop footprint alone.

Matching-PIC real partial-weight service tests used sixteen participants, two
experts, layers 3/4, eight NUMA banks and small-expert unfused selection enabled.
Prefetch was off. Requested-fused NUMA/dataflow medians were:

| Variant | Layer 3, one token ms | Layer 4, one token ms | Layer 3, four tokens ms | Layer 4, four tokens ms |
|---|---:|---:|---:|---:|
| Baseline | 0.552 | 0.544 | 1.537 | 1.490 |
| Candidate | 0.733 | 0.523 | 2.073 | 1.454 |
| Candidate | 0.735 | 0.551 | 2.028 | 1.555 |
| Baseline | 0.616 | 0.600 | 1.652 | 1.596 |

The modest instruction saving did not establish a consistent service improvement.
Host load was uncontrolled; the slower layer-3 samples cannot be attributed
solely to unrolling. Keep the candidate out of production: this evidence does
not justify the larger code footprint. All 864 timed output/reduction and 96
final CPU-batch checks matched bitwise. These were CPU-only scratch builds linked
with HIP-configured libraries; the rejected pragma was not validated with
CUDA/SYCL compilers. No full-model throughput or 90% peak claim follows.

Compact patch, source hashes, both counter screens, probe snapshots, parity and
service logs are in `results/20261020/fp4-block-unroll2/`.


### Coarse tasks with rotating FP4 weights (2026-10-20)

The real partial-weight benchmark's `task-sweep` now includes one and two tasks
per participant as well as three/four/six. Its fixed mode and production default
remain unchanged. This checks whether larger row ranges reduce repeated
activation setup and task scheduling overhead without starving participants.

Threadripper 1950X, sixteen participants, two real experts, layers 3/4, eight
NUMA banks, small-expert unfused selection enabled, prefetch off, baseline AVX2
dots. Two repeated requested-fused NUMA/dataflow single-token comparisons gave:

| Run/layer | Tasks 1 ms | Tasks 2 ms | Tasks 3 ms | Tasks 4 ms | Tasks 6 ms |
|---|---:|---:|---:|---:|---:|
| 1 / 3 | 0.783 | 0.576 | 0.510 | 0.504 | 0.512 |
| 1 / 4 | 0.967 | 0.813 | 0.782 | 0.784 | 0.775 |
| 2 / 3 | 0.828 | 0.689 | 0.741 | 0.725 | 0.722 |
| 2 / 4 | 0.787 | 0.612 | 0.553 | 0.579 | 0.540 |

One task per participant was slower in every single-token comparison. Two tasks
was mixed, slower than three in three of four comparisons. Four-token results
were noisy: first layer-3 medians ranged 4.667–7.067 ms, while the repeat ranged
1.485–2.080 ms. These outliers are retained. Host load was uncontrolled; do not
attribute all timing differences to task overhead. There is no basis here to
recommend fewer tasks or replace the production default of three.

Both sweeps passed all 2,160 timed output/reduction and 240 final CPU-batch
checks bitwise. Input quantization is excluded; banks reuse identical weight
values. No full-model throughput or 90% peak-utilization claim follows. Logs,
summary and source hashes are in `results/20261020/fp4-coarse-tasks/`. Reproduce
with the preceding real rotating-NUMA command, replacing `fixed` with
`task-sweep` and retaining `STRATA_NVFP4_SMALL_UNFUSED=1`.

The expanded benchmark built in HIP- and CUDA-configured trees. An additional
CUDA-configured sweep passed 1,080 timed and 120 final CPU-batch comparisons,
matching the HIP reference hashes. These were CPU executions; no GPU ran.
SYCL validation remains unavailable. Only benchmark coverage changed.


### Worker biased-dot FP4 check (2026-10-20)

The Ryzen 9 3950X worker tested the existing `STRATA_NVFP4_OFFSET_DOT` option in
0/1/1/0 order using its previously deployed static CPU benchmark and real
partial weights. Sixteen participants, three experts, layers 3/4, eight rotating
complete-blob banks (339,739,008 bytes), prefetch off, narrow-128 option off.
The host has one NUMA node, so this checks complete-blob split/dataflow modes.
Each process was bounded to 45 seconds; no miner or GPU state was changed.
Pre-test load averages were 30.08/30.38/30.50 on 32 logical CPUs.

| Offset option | Layer 3, one token ms | Layer 4, one token ms | Layer 3, four tokens ms | Layer 4, four tokens ms |
|---:|---:|---:|---:|---:|
| 0 | 1.665 | 1.737 | 2.970 | 1.984 |
| 1 | 1.730 | 1.775 | 4.759 | 1.908 |
| 1 | 1.824 | 1.781 | 4.570 | 6.989 |
| 0 | 2.896 | 1.795 | 5.382 | 5.994 |

These are dataflow medians. The biased-dot option changes only single-token
execution; the large four-token variation is an unchanged-path control showing
substantial timing interference. Single-token results do not establish a stable
benefit. Do not enable this option by default on the worker. This comparison
does not calibrate the full-model simulator or demonstrate peak utilization.

All 288 timed output/reduction and 32 final CPU-batch checks passed bitwise,
with matching off/on reference hashes. No new source or backend build was
needed: this tests an existing deployed option. Binary and partial-fixture
hashes, compact host metadata and all logs are in
`results/20261020/worker-offset-dots/`. Use the existing worker benchmark with
`PARTIAL_MODEL_DIR 3 15 fixed 8`, toggling `STRATA_NVFP4_OFFSET_DOT=0/1`, with
`STRATA_NO_NVFP4_512=1`, `STRATA_NVFP4_128=0`, `STRATA_NVFP4_PREFETCH=0` and
`STRATA_NATIVE_NUMA_LOCAL=0`.


### CPU activation quantization budget (2026-10-20)

Added `glm_cpu_quant_bench` under `STRATA_BUILD_TESTS`. It calls the actual native
input/hidden quantizers for Q2_K/Q3_K and NVFP4 at H4096/F2048, one/four tokens,
and 32 rotating finite synthetic input banks. Seven samples each contain 8,192
calls. All 32 banks are checked byte-for-byte against the active GGML quantizer
before timing; the final output of each timed batch is checked again. This is
not a stress test of every floating-point input and does not check every timed
call individually. Allocation/reference work is outside timing. No production
quantization behavior changed.

Threadripper 1950X, pinned logical CPU 11, uncontrolled host load. In the initial
run, one-token input quantization was 2.121 us for Q2/Q3 and 1.931 us for FP4;
hidden quantization was 1.042 and 0.973 us respectively. Four-token times were
8.394/7.708 us input and 4.169/3.866 us hidden. A repeat had slower Q2/Q3 input
3.417 us and hidden 1.839 us; FP4 input was 2.761 us and hidden 1.004 us.
CUDA-configured CPU execution gave 2.132/2.065 us input and 1.144/1.013 us
hidden for Q2/Q3 and FP4 respectively. These process differences are not evidence
of a backend performance difference.

For scale only, 42 layers with one input quantization and eight expert hidden
quantizations per layer would cost about 0.41 ms at the initial FP4 medians:
42*(1.931+8*0.973) microseconds. This is arithmetic on isolated measurements,
not a full-decoder measurement; scheduling and placement are excluded, and
hidden quantization is already included in the expert service timings. Do not
add it again to the simulator's expert service. Input quantization remains a
small omitted cost, not an explanation for tens of milliseconds of decode time.

The new target built in HIP- and CUDA-configured trees. Both linked CPU paths
ran successfully, and a final CMake-built HIP run passed the same checks. The
first manual runs precede only explicit include and unavailable-native guard
additions. No GPU ran; SYCL validation remains unavailable. All logs and source
hashes are in `results/20261020/cpu-activation-quantization/`. Reproduce with
`cmake --build BUILD_DIR --target glm_cpu_quant_bench`, then
`taskset -c CORE BUILD_DIR/glm_cpu_quant_bench`; record core placement and load.


### Native NVFP4 K32 layout prototype (2026-10-20)

A scratch native NVFP4 layout stores the first sixteen values of each K32 pair
in low nibbles and the next sixteen in high nibbles. Four group-16 scale bytes
and the 36-byte K64 block size remain unchanged. This removes two 64-bit byte
interleaves from `values32`; vector lanes, scales, integer dots and FMA/reduction
order remain unchanged. It applies the earlier local-layout idea to the native
NVFP4/Q8 kernel, not to the exact BF16 W4A16 helper.

Matching-PIC O3 AVX2/FMA/F16C builds on Threadripper 1950X, logical CPU 11,
128 reused rows, K4096, prefetch off: one-token whole-process instruction counts
were 7.256/7.248 billion baseline versus 6.712/6.712 candidate; four-token counts
were 15.893/15.886 versus 15.350/15.357 billion. Reductions are about 7.4% and
3.4%. Probes retain original-layout weights for GGML reference checks, including
raw -128 activation. Counts include setup/verification and do not measure DRAM
traffic. Cycles varied with background load.

Real partial-weight baseline/candidate/candidate/baseline runs used two experts,
layers 3/4, sixteen participants, eight NUMA banks and small-expert unfused
selection. Candidate weights were repacked before copies/shards were created,
outside timing; scale tails were unchanged. Requested-fused dataflow medians:

| Variant | Layer 3, one token ms | Layer 4, one token ms | Layer 3, four tokens ms | Layer 4, four tokens ms |
|---|---:|---:|---:|---:|
| Baseline | 0.541 | 0.544 | 1.470 | 2.266 |
| Candidate | 0.587 | 0.522 | 1.641 | 1.471 |
| Candidate | 0.865 | 0.591 | 2.049 | 1.668 |
| Baseline | 0.855 | 0.572 | 2.023 | 1.515 |

No stable service speedup follows under this uncontrolled load. All 864 timed
output/reduction and 96 final CPU-batch comparisons passed within each process;
reference hashes also match across original and repacked processes. This is
not exhaustive validation of all scales, shapes or kernel variants. The 128-bit
alternative expects the old layout and was explicitly disabled.

A separate serial in-place repacking test on CPU 11 used 75,497,472 bytes. Two
conversions restored every original byte in each of seven samples; median time
per conversion was 34.713 ms, or 2.175 logical GB/s (not measured DRAM bandwidth).
This simple repacker is a loading-time candidate, not suitable for each decode
call. Production integration would require a shared C11 conversion contract,
explicit layout ownership and matching kernels/upload paths; passing repacked
weights to existing consumers would be incorrect. No integration or default
change was made. Scratch CPU compilation used HIP-configured libraries; no
CUDA/SYCL prototype validation or GPU execution was performed.

Preserve this as an instruction-saving layout candidate requiring stronger
service/format validation. No full-model throughput or 90% utilization claim.
Counter probes, compact kernel/pool patches, repack test and all logs are in
`results/20261020/native-fp4-k32-layout/`.


### Shared native K32 conversion contract (2026-10-20)

`strata_glm_nvfp4_repack_k32` now provides the experimental native layout
conversion in the shared C11 host library. It accepts complete 36-byte K64
blocks only, excluding global scale tails. The nibble permutation is its own
inverse. It preserves all four scale bytes verbatim, including opaque nonfinite
codes; it does not validate or reinterpret scales. Exact in-place and disjoint
buffers are supported. Invalid byte lengths, null nonempty buffers, address
wrap and partial overlap fail before output writes. Empty buffers succeed.
This API is distinct from the earlier adjacent-nibble local FP4 converter.

No loader, checkpoint, GPU format, native dispatch or production default selects
this layout. Existing kernels other than the scratch matching candidate cannot
consume it. An explicit shared representation contract is a prerequisite for
future integration, not a performance or full-model-support claim.

The model-free C11 `glm_nvfp4_k32_test` checks 9,216 byte-position/value fixtures,
independently decodes all 64 nibbles per block, checks scale preservation and
output guards, and exercises disjoint/in-place inverse behavior plus rejection
without writes. A separate single-block case supplements the multi-block cases.
AddressSanitizer/UBSan passed with leak detection disabled. The existing NVFP4
packer test now checks the new layout's codes/scales and in-place inverse too.
It passed its synthetic fixtures and six real matrices (layers 3/4, expert 0,
gate/up/down) in both HIP- and CUDA-configured CPU builds. Each real matrix
contains 8,388,608 values, all checked against the original local payload.

Both scoped tests passed in both builds. No GPU ran; SYCL validation remains
unavailable. The original NVFP4 packer still produces the same GGML layout and
passes its independent dequantization comparison. Reproduce with
`cmake --build BUILD_DIR --target glm_nvfp4_k32_test glm_fp4_nvpack_test`, then
`ctest --test-dir BUILD_DIR -R '^(glm_nvfp4_k32_test|glm_fp4_nvpack_test)$'`
and `BUILD_DIR/glm_fp4_nvpack_test PARTIAL_MODEL_DIR` for real-weight coverage.
Logs and source hashes are in `results/20261020/native-fp4-k32-contract/`.


### Explicit native K32 dot entry point (2026-10-20)

`nvfp4_256_k32_rows` now exposes the experimental packed layout through a
separate AVX2/FMA/F16C entry point. It shares the signed-dot implementation with
the original kernel through a compile-time layout parameter; ordinary calls
retain the original byte interleaves. The explicit K32 API requires the shared
C11 repacker's output and never selects old-layout offset/narrow alternatives.
The existing prefetch setting still applies. It supports K64..4096, one to
32 tokens, padded row strides and partial ranges; token widths above eight use
balanced slices. Invalid geometry/null nonempty buffers throw before work, and
empty ranges do not access buffers. Callers must check CPU feature support and
provide correctly sized allocations.

The new model-free test checks 1,232 configurations against GGML over original-
layout weights: K64/128/640/4096; widths 1..9,16,32; seven finite half-scale
patterns; signed Q8 extremes including all -128 and a final-block -128; varied
FP4 nibbles/group scales, positive/negative global multipliers, padded strides
and partial-row output guards. Every output bit matched. A run with both
old-layout single-token options enabled also passed, confirming the explicit
API does not select them. The new kernel passed ASan/UBSan synthetic checks
with leak detection disabled. Geometry/null-buffer rejection was exercised.

An optional partial-model argument checks six complete real matrices from
layers 3/4, expert 0, gate/up/down, at token widths 1/4/9. All 18 projections
matched independent GGML dots with global scaling bit-for-bit in both HIP- and
CUDA-configured CPU builds. Six scoped tests passed in each build, including
the C11 conversion and all four original native FP4 option combinations.
The final test added real-weight coverage after the synthetic sanitizer run;
the kernel implementation was unchanged. No GPU ran; SYCL remains unavailable.

No loader, expert scheduler, GPU path or default selects the new representation.
This is a validated explicit projection primitive, not yet integrated K32 expert
execution or a full-model speedup. The earlier scratch performance screen is
not a performance qualification of every new entry-point shape. Reproduce with
`cmake --build BUILD_DIR --target glm_nvfp4_k32_dot_test`, then
`BUILD_DIR/glm_nvfp4_k32_dot_test [PARTIAL_MODEL_DIR]`.
Logs and source hashes are in `results/20261020/native-fp4-k32-dot-api/`.


### Explicit K32 gate/up and SiLU primitive (2026-10-20)

`nvfp4_256_k32_gu_rows_clamped` adds matching gate/up execution to the explicit
K32 projection API. Gate/up reuse one prepared activation set per row range.
The existing SiLU, global-scale multiplication and clamp arithmetic are shared
through a compile-time layout parameter. Both combined blobs and independent
packed up buffers are supported; no temporary gate/up output vectors are added.
The old-layout options cannot select a different kernel through this explicit
API. Geometry/null-buffer validation is shared with K32 plain rows; combined
up-offset overflow is rejected before writes, and separate-up calls ignore the
combined offset. No loader or expert-pool dispatch selects K32 yet.

The 1,232 synthetic dot configurations now also check gate/up with clamp off/on,
combined/full-row and separate/partial-row buffers. Up weights reverse the gate
row order and use a different global multiplier, preventing identical gate/up
inputs from masking addressing errors. Reference dots use original-layout GGML
weights, followed by the same scalar clamp/SiLU operations. Full results and
partial-row guards match bitwise across widths 1..9/16/32, K64/128/640/4096,
finite scale patterns and signed Q8 extremes. Combined-offset overflow and
separate-up offset independence are checked explicitly.

Real layers 3/4, expert 0, gate/up pairs passed widths 1/4/9 at amplitudes
1/+4096/-4096: 18 GU comparisons per backend, in addition to the 18 plain
projection comparisons. Every stress case hit gate-high, up-high and up-low
clamps; counts are preserved in the logs. Real tests ran with both old-layout
single-token options enabled, confirming explicit K32 isolation. Six scoped
CPU regressions passed in each HIP-/CUDA-configured build. The final synthetic
suite passed ASan/UBSan, leak detection disabled. Real checks precede only the
addition of offset-boundary test cases; the kernel was unchanged. SYCL remains
unavailable and no GPU ran.

This completes explicit K32 projection and gate/up primitives, not integrated
expert scheduling, a default change, or a measured full-model speedup. Logs and
source hashes are in `results/20261020/native-fp4-k32-gu-api/`. Reproduce with
`glm_nvfp4_k32_dot_test PARTIAL_MODEL_DIR` from either configured build.


### Explicit K32 CPU expert execution (2026-10-20)

`NativeFmt::nvfp4_k32` explicitly selects the validated K32 gate/up and down
primitives. Both projection types must be NVFP4, geometry must fit the AVX2
kernels, and canonical/lossless/Q2-Q3 layouts cannot be combined with it.
CPU feature checks precede packed dispatch; there is no fallback to an
incompatible GGML/AVX-512 kernel. Rebuilding a native format clears the flag.
The field must only be set after all three projections have been converted.
It is an explicit CPU representation choice, independent of old-layout dot
options. No loader or environment switch automatically enables it.

The real pool benchmark accepts a final `ggml|k32` argument, default `ggml`.
K32 mode repacks only bytes before the global scale tail through the shared
C11 helper, then constructs rotating banks and NUMA shards. It retains original-
layout blobs for independent reference execution, including normal quantization,
clamp, down scaling and routed reduction. Conversion/reference work is outside
timing; retaining the reference increases benchmark memory and is not a loader
memory recommendation. Unsupported AVX2 hosts skip K32 mode before pool work.

Complete expert tests on Threadripper 1950X used real layers 3/4, two experts,
sixteen participants, one/four tokens and eight banks. Combined-blob split/flow,
NUMA split/flow and requested-fused NUMA split/flow all matched original-layout
reference outputs and routed sums bitwise: 216 timed and 24 final CPU-batch
checks per build, 432/48 total across HIP-/CUDA-configured CPU builds. The
small-expert unfused option was enabled. Original-format references have K32
explicitly cleared and use separate unconverted storage, not the candidate
kernel to compute its own expected output.

Eight scoped regressions passed in both builds (K32 host/dot, four original
native FP4 variants, Q2/Q3 pool and dataflow). Additional format checks reject
K32 selection on Q2/Q3 before output writes and verify `native_fmt` clears stale
layout selection. The HIP real run precedes only the added benchmark CPU-skip
guard and these format tests; execution arithmetic was unchanged. No GPU ran;
SYCL validation remains unavailable. Timing remains affected by uncontrolled
load and is not a qualified K32 speedup or full-model decode measurement.

Reproduce with `glm_fp4_native_pool_bench PARTIAL_MODEL_DIR 2 15 fixed 8 k32`
(or final `ggml` for the original layout), using the previous CPU benchmark
settings. Default model loading and GPU format remain unchanged. CPU/GPU cache
or multi-node loading would need explicit layout ownership before using this
representation. Logs, mode summaries and source hashes are in
`results/20261020/native-fp4-k32-experts/`.


### Alternating complete-expert layouts (2026-10-20)

The pool benchmark accepts final layout `paired`. It holds original and K32
complete blobs, banks and NUMA tensors in the same process. Layout/container
selection happens before warmup/timing; synchronous completion ensures no worker
retains pointers while containers swap. Each layout has its own bank cursor.
At banks=1, selection updates blob and shard pointers too. Both layouts are
checked against one original-layout sequential expert/reference sum.

The first implementation rotated the variant list, which did not consistently
reverse each layout pair's order. Its three logs are retained as preliminary.
The final schedule keeps each pair adjacent, alternates original-first and
K32-first every round, and rotates the execution-mode order. CPU-cost batches
remain separate fixed-order diagnostics, not balanced pair timings. The default
single-layout mode and production dispatch are unchanged.

Between screens, a one-second read-only host sample showed 61.5% aggregate CPU
busy and load averages 32.78/27.34/21.20; many logical CPUs were near 100% busy.
The cgroup reported no throttled time. This demonstrates that the host was not
idle at that observation, not which workload caused individual slow samples.
No miner or unrelated process was changed.

Final eight-bank, two-expert, sixteen-participant requested-fused NUMA/dataflow
medians with the small-expert unfused option enabled:

| Run | Layer | Tokens | GGML ms | K32 ms |
|---|---:|---:|---:|---:|
| 1 | 3 | 1 | 1.002 | 0.871 |
| 1 | 4 | 1 | 0.787 | 0.745 |
| 2 | 3 | 1 | 0.952 | 1.178 |
| 2 | 4 | 1 | 0.723 | 0.729 |
| 1 | 3 | 4 | 2.352 | 3.618 |
| 1 | 4 | 4 | 4.045 | 4.673 |
| 2 | 3 | 4 | 3.282 | 2.884 |
| 2 | 4 | 4 | 2.133 | 2.833 |

Same-process pairing does not establish a stable K32 service speedup under this
load. Retain the instruction-count result separately; it does not imply these
wall-clock results must improve. Do not enable K32 by default or claim a
full-model throughput/90% peak gain. Paired mode retains two layouts and their
NUMA copies; its memory footprint is deliberately larger than deployment needs.

The final HIP-configured eight-bank runs and CUDA-configured one-bank run passed
1,296 timed output/reduction and 144 final CPU-batch checks. The three preliminary
runs passed another 1,296/144. All mode/layout reference hashes match. This
validates repeated layout switching and banks=1/8, not a cache-miss routing
trace. Both CPU builds passed; no GPU ran and SYCL remains unavailable.
Reproduce with `glm_fp4_native_pool_bench PARTIAL_MODEL_DIR 2 15 fixed 8 paired`
and the preceding explicit benchmark environment. Logs, host observation,
summaries and source hashes are in `results/20261020/native-fp4-k32-paired/`.


### Worker paired K32 expert comparison (2026-10-20)

The paired-layout CPU benchmark was statically linked and deployed to the Ryzen
9 3950X worker's temporary directory. Existing real partial weights were reused;
no checkpoint download or miner change was needed. Three experts, layers 3/4,
sixteen participants, three tasks per participant, eight banks per layout,
baseline signed AVX2 dots and prefetch off. The single-NUMA host uses complete-
blob split/dataflow modes. Each layout's packed working set is 339,739,008 bytes;
paired allocation is larger. Each invocation was bounded to 60 seconds.
Pre-test load averages were 30.08/30.28/30.50 on 32 logical CPUs.

Dataflow medians, with adjacent layout pairs reversing order each round:

| Run | Layer | Tokens | GGML ms | K32 ms |
|---|---:|---:|---:|---:|
| 1 | 3 | 1 | 1.836 | 1.799 |
| 1 | 4 | 1 | 4.940 | 4.442 |
| 2 | 3 | 1 | 1.841 | 1.897 |
| 2 | 4 | 1 | 1.712 | 1.710 |
| 1 | 3 | 4 | 5.126 | 4.355 |
| 1 | 4 | 4 | 5.996 | 6.594 |
| 2 | 3 | 4 | 4.422 | 7.465 |
| 2 | 4 | 4 | 5.176 | 6.262 |

No repeatable service advantage is established. The single-token results mostly
track each other outside the large layer-4 first-run variation; four-token
results are mixed and often slower for K32. Do not promote K32 on this worker
based on the primary CPU's instruction-count saving. This does not isolate
memory bandwidth from scheduling contention or calibrate full-model throughput.

All 288 timed output/reduction and 32 final CPU-batch checks passed bitwise
against the original-layout reference. The same binary supports both layouts;
no GPU ran and production defaults remain unchanged. Reproduce with the static
pool benchmark using `PARTIAL_MODEL_DIR 3 15 fixed 8 paired`,
`STRATA_NATIVE_TASKS_PER_THREAD=3`, `STRATA_NATIVE_NUMA_LOCAL=0`,
`STRATA_NO_NVFP4_512=1`, `STRATA_NVFP4_128=0`, `STRATA_NVFP4_OFFSET_DOT=0`,
and `STRATA_NVFP4_PREFETCH=0`. Logs, compact host metadata and binary/source
hashes are in `results/20261020/worker-k32-paired/`.


### Q3_K decoded-scale scratch layout (2026-10-20)

A scratch Q3_K layout keeps the 96 code/high-mask bytes, stores sixteen decoded
signed scale bytes instead of twelve packed scale bytes, then retains the FP16
block scale. Blocks grow from 110 to 114 bytes (3.64%). The corresponding scratch
kernel loads signed scales directly, avoiding the scalar bit extraction and
bias subtraction in every row/block. Integer dots and floating accumulation
order are unchanged. This is not a GGML-compatible block and is not integrated
into any loader, native format or production dispatch.

Threadripper 1950X, logical CPU 11, prefetch off, synthetic 128x4096 rows, one/four
tokens, matching O3/PIC/AVX2/FMA/F16C builds. Baseline/candidate/candidate/baseline
cache-resident tests used 8,192 calls. Whole-process instructions were
2.486/2.485 billion baseline versus 1.926/1.927 candidate for one token (22.5%
lower); four-token counts were 4.976/4.975 versus 4.457/4.453 billion (10.5%
lower). Counts include quantization, conversion and reference checking; they
are not exclusively inner-loop counters.

| Variant | One-token kernel CPU ms | Four-token kernel CPU ms |
|---|---:|---:|
| Baseline | 382.116 | 1288.400 |
| Candidate | 342.825 | 1269.070 |
| Candidate | 341.318 | 993.053 |
| Baseline | 380.083 | 1155.710 |

Single-token cache-resident CPU time fell approximately 10%; four-token timing
varied more. A 1,024-copy ring, 4,096 calls, grew from 230,686,720 to 239,075,328
bytes. One-token CPU times were 362.578/338.099/345.278/204.359 ms in the same
order: no stable streaming gain. Four-token times were
545.332/504.247/509.081/674.544 ms; both candidate runs were lower than both
baseline runs, but this remains a noisy synthetic single-core screen.

Every run passed the independent decoded-weight FP64 tolerance check and its
own bitwise post-timing stability check. All copies passed output stability,
and full-output hashes match between original and expanded layouts for each
token width. This does not replace exhaustive scale-code/shape validation or
complete-expert testing. The prototype remains isolated; conversion/allocation
are excluded from kernel times, and the larger weight traffic must be included
in any memory/throughput comparison. No CUDA/SYCL prototype validation or GPU
execution was performed. No full-model or 90% utilization claim follows.

The result merits shared C11 format validation and complete-expert measurement,
not a production default change. Compact kernel/probe patches, original/source
hashes, raw counters and all timing logs are in
`results/20261020/q3-expanded-scales/`.

### Q3 expanded-scale shared contract (2026-10-20)

The experimental Q3 scale representation now has a shared C11 converter in
`strata_host`: `strata_q3k_expand_scales` and `strata_q3k_compact_scales`.
Each 256-weight block grows from 110 to 114 bytes (+3.636%). Code/high-mask
bytes and opaque FP16 bytes remain unchanged; sixteen signed scale bytes replace
twelve packed bytes. This format cannot be passed to existing GGML/GPU kernels.
No production kernel, loader selection, or default representation changed.

The model-free test covers all 64 scale values at each of sixteen positions,
17-block arbitrary-byte round trips, guards, extent/null/overlap/address-overflow
rejection, and all 192 invalid signed scale bytes at each position in the final
block. Invalid input leaves the entire output untouched. The test passed in the
HIP and CUDA configured CPU builds and under AddressSanitizer/UBSan. LeakSanitizer
was disabled because it cannot run under this environment's tracing; SYCL was
not tested because its compiler is unavailable. No GPU execution was needed.

The scratch AVX2 expanded-scale probe now uses the shared converter. For 128
rows, K=4096, and token counts 1/2/3/4, output hashes match the original-layout
baseline and the independent decoded-weight checks pass. These short 16-call
runs establish connectivity/correctness only, not performance. The earlier
instruction reduction still needs complete-expert and streaming validation;
full-model throughput and the 90% peak target remain unproven. Evidence and
source hashes are in `results/20261020/q3-expanded-scale-contract/`.

### Q3 expanded-scale kernel qualification (2026-10-20)

`q3_expanded_parity.cpp` tests the scratch expanded-scale kernel alongside the
original kernel in one process. Compile the scratch kernel with exported names
renamed using `-Dq23_rows=q3_expanded_rows` and
`-Dq23_gu_rows=q3_expanded_gu_rows`, then link it with the normal CPU/host/GGML
archives. The patch in `results/20261020/q3-expanded-kernel-parity/kernel.patch`
applies to the recorded baseline source hash and is for a scratch copy only.

All 1,200 configurations pass bitwise comparison: K=256/512/768/2048/4096,
widths 1 through 8, six raw Q8 patterns including -128 and +127, arbitrary Q3
code/scale bytes with finite FP16 scales, seven padding bytes per expanded row,
partial row guards, and gate/up execution with clamping, canonical activation,
and gate skipping. Dot-only cases also pass an independent GGML-dequantized
FP64 reference check. The shared C11 converter prepares expanded weights.

UBSan initially rejected the scratch kernel's reference to an original GGML
struct at an odd-aligned expanded row. Reading code and high-mask fields by
byte offset removes that invalid reference. After this fix, the instrumented
kernel and driver pass ASan/UBSan against the HIP-configured CPU libraries;
those linked reference libraries themselves were not instrumented. An optimized
build against the CUDA-configured CPU libraries also passes. LeakSanitizer was
disabled because of tracing incompatibility. SYCL and GPU execution were not
tested. Source hashes and the corrected scratch patch are frozen in the result
directory. These are correctness checks, not performance measurements; no
production kernel or default layout changed. Complete-expert integration and
streaming measurement remain the next gates.

### Q3 expanded scales in complete Q2/Q3 expert service (2026-10-20)

The scratch pool probe converts Q3 down weights with the shared C11 converter,
updates the down row/blob extents, and builds NUMA-owned row shards afterward.
Q2 gate/up bytes and activation quantization stay unchanged. This does not add a
production format selection. The scratch executable requires `--bench` and
`STRATA_Q23_AVX2=1`; it must never fall back to original-layout GGML kernels.

On the Threadripper 1950X, two synthetic experts (H=4096, F=2048), fifteen workers
plus the caller, eight separate weight banks, Q2 LUT enabled, prefetch off,
three tasks/thread, and the existing small-expert unfused option were tested.
Timed dataflow calls include gate/up, SiLU, hidden quantization, down projection,
and routed reduction; input quantization, weight conversion and NUMA setup are
excluded. Banks rotate across calls; this is not a real routing trace. Both
executables were built with O3/NDEBUG against the same HIP-configured CPU archive.
No GPU was used or miner changed. Host load was uncontrolled.

Each token width used baseline/expanded/expanded/baseline order with nine timed
calls per run. All 72 timed calls match their local split reference exactly;
output hashes covering expert outputs and routed sums also match across layouts.
Medians in run order, milliseconds:

| Tokens | Baseline 1 | Expanded 1 | Expanded 2 | Baseline 2 |
|---|---:|---:|---:|---:|
| 1 | 1.02436 | 0.577756 | 1.91804 | 0.546765 |
| 4 | 1.47804 | 1.35052 | 2.13964 | 2.85784 |

These timings do not establish a repeatable complete-expert speedup. The prior
single-core instruction reduction does not justify adopting a larger weight
layout by default. Keep the candidate experimental pending controlled scheduling
and streaming measurements. No full-model throughput or 90% peak claim follows.
Logs, configuration, source hashes, and scratch patches are in
`results/20261020/q3-expanded-experts/`. CUDA/SYCL complete-expert builds were not
run in this experiment; the previous CUDA CPU kernel parity check is separate.

### Q2/Q3 phase budget after expanded-scale screening (2026-10-20)

A separate split-phase run used the same two scratch binaries and configuration
as the preceding complete-expert experiment: two experts, eight banks, fifteen
workers plus caller, Q2 LUT on, prefetch off, three tasks/thread, Threadripper
1950X. Both widths used baseline/expanded/expanded/baseline order, nine timed
calls per run. All 72 calls retain exact output/sum hashes across layouts.
Input quantization and setup remain excluded. The phase clocks include worker
scheduling and waits; they are not isolated kernel CPU-time measurements.

Across the four baseline runs, medians of per-call phase/total ratios were:
Q2 gate/up 57.8–59.9%, Q3 down 37.7–40.4%, hidden quantization 0.52–0.80%.
These ratios come from each call before taking the median; medians across
phases need not sum to one. Illustratively, holding other costs fixed, a 10%
reduction in down-phase time would yield only 1.039–1.042x complete-expert
speedup. Even doubling down speed would yield 1.232–1.253x. This is an Amdahl
calculation from this synthetic split schedule, not a full-model forecast or
proof that kernel speed translates proportionally into phase speed.

The first one-token baseline down median was 0.2190 ms; the two expanded runs
were 0.2115 and 0.1987 ms, while the final baseline rose to 0.3518 ms. Four-token
down medians were 0.5591 / 0.5430 / 0.5532 / 0.5438 ms in run order. The final
baseline matches or beats the candidates at four tokens. Background contention
still prevents claiming an expanded-scale benefit. Prioritize Q2 gate/up work
and scheduling alongside FP4, rather than treating the Q3 unpack instruction
saving as a major decoder gain. No default changed. Logs, phase fractions and
illustrative bounds are in `results/20261020/q3-expanded-phases/`; binary hashes
are the same as `q3-expanded-experts/summary.json`.

### Q2 lookup two-way unroll screening (2026-10-20)

A scratch candidate adds `#pragma GCC unroll 2` to the eight-chunk Q2 LUT loop,
matching its existing two-chunk integer accumulation groups. This differs from
the previously rejected eight-way unroll: batch-4 function text shrinks from
1,724 to 1,544 bytes; batch-1 shrinks from 1,265 to 809. Production source and
defaults remain unchanged. Matching O3/NDEBUG/PIC AVX2/FMA/F16C compilation and
HIP-configured CPU archives were used. The normal Q2/Q3 dump suite matches
byte-for-byte, including its independent scalar references and widths 1–4.

CPU-11 cache-resident screens on Threadripper 1950X used K4096, 128 rows,
8,192 calls, Q2 LUT enabled, prefetch disabled, and baseline/candidate/candidate/
baseline order. Kernel process CPU times in milliseconds:

| Tokens | Baseline 1 | Candidate 1 | Candidate 2 | Baseline 2 |
|---|---:|---:|---:|---:|
| 1 | 449.428 | 653.765 | 420.014 | 289.154 |
| 4 | 1208.150 | 945.307 | 944.921 | 1501.810 |

The apparent batch-4 kernel improvement did not survive a complete-expert
screen. Two experts, H4096/F2048, eight NUMA-owned banks, fifteen workers plus
caller, dataflow, three tasks/thread, and the existing small-expert unfused
selection gave medians 1.42549 / 1.54357 / 2.72288 / 1.42300 ms in the same
run order (15 timed calls each). All 60 calls match expert outputs and routed
sums across builds. Input quantization and loading are outside timing; hidden
quantization and routed reduction are included. Host load remained uncontrolled.
Do not adopt this candidate from the isolated kernel timings. No CUDA/SYCL
build or GPU execution was performed for this rejected scratch candidate.
Patch, logs, symbol sizes, hashes and summaries are in `results/20261020/q2-unroll2/`.

### Same-process Q2 candidate screening (2026-10-20)

`q23_paired_probe.cpp` links the baseline and a renamed candidate into one
process. Compile the candidate with `-Dq23_rows=q23_candidate_rows` and
`-Dq23_gu_rows=q23_candidate_gu_rows`, then link its object before the normal
CPU/host/GGML archives. The interface is `paired_probe TOKENS COPIES`. It uses
shared weights/activations, K4096, 128 rows, 512 calls per timed batch, twelve
pairs and reversed first/second order on alternating pairs. Each pair traverses
the same ring positions. Packing and reference work stay outside timing; every
batch's final complete output must match baseline bytes. Baseline output also
passes the independent decoded-weight FP64 check. This is a kernel screen,
not expert service or model throughput.

The two-way Q2 unroll was re-screened on Threadripper 1950X, CPU 11, Q2 LUT on,
prefetch off, with matching O3/NDEBUG/PIC AVX2/FMA/F16C objects and the same
HIP-configured CPU archive. Ratios below are medians of candidate/baseline
process CPU time within each pair; lower than one would favor the candidate.

| Tokens | Weight copies | Median CPU-time ratio | Candidate wins / pairs |
|---|---:|---:|---:|
| 1 | 1 | 1.476 | 1 / 12 |
| 4 | 1 | 1.091 | 2 / 12 |
| 1 | 1024 | 1.431 | 0 / 12 |
| 4 | 1024 | 1.064 | 0 / 12 |

The ring holds 176,160,768 weight bytes. All 96 timed batches passed output
comparison. This stronger paired evidence fails to reproduce the prior
separate-process apparent four-token win and supports rejecting the unroll
candidate. Background load is still uncontrolled, and process CPU time does
not remove frequency/cache/SMT contention. Future kernel candidates should use
this paired screen before complete-expert trials. No production default or
kernel changed; no CUDA/SYCL or GPU execution was needed for this experiment.
Logs and hashes are in `results/20261020/q2-paired-screen/`; the candidate patch
is retained under `q2-unroll2/`.

### Paired Q3 expanded-scale screen on both NUMA nodes (2026-10-20)

`q3_expanded_paired_probe.cpp` compares original and expanded Q3 rows in one
process using the already qualified scratch kernel and shared C11 conversion.
It follows the Q2 paired protocol: K4096, 128 rows, 512 calls per batch, twelve
pairs with alternating first layout, identical ring indices, and final-output
byte comparison on every batch. Original output passes the decoded-weight FP64
check; all expanded ring slots are checked against original output before timing.
Both layouts coexist in memory. At 1024 copies their respective payloads are
230,686,720 and 239,075,328 bytes (+3.636%); total resident allocation is larger.
No conversion is included in timed execution.

On Threadripper 1950X, the same optimized binary was run first on CPU 11
(NUMA node 1), then CPU 3 (node 0). Affinity applies before allocation and
initialization, but physical page placement was not separately measured.
Prefetch was disabled, host load uncontrolled, and miners unchanged. Results
are median paired candidate/baseline process CPU-time ratios:

| CPU | Tokens | Copies | Ratio | Candidate wins / pairs |
|---|---:|---:|---:|---:|
| 11 | 1 | 1 | 0.907 | 11 / 12 |
| 11 | 4 | 1 | 0.967 | 9 / 12 |
| 11 | 1 | 1024 | 0.910 | 11 / 12 |
| 11 | 4 | 1024 | 0.960 | 8 / 12 |
| 3 | 1 | 1 | 0.898 | 11 / 12 |
| 3 | 4 | 1 | 0.964 | 11 / 12 |
| 3 | 1 | 1024 | 0.917 | 12 / 12 |
| 3 | 4 | 1024 | 0.954 | 12 / 12 |

All 192 timed batches passed output comparison. Unlike Q2 two-way unrolling,
the Q3 expanded-scale benefit survives this paired kernel screen: approximately
8–10% less CPU time at one token and 3–5% at four in median pairs. Preserve the
individual noisy ratios; these are not isolated hardware-cycle measurements.
This warrants paired complete-expert evaluation, not default adoption. The
previous separate-process pool test did not show a repeatable service benefit,
and the phase budget limits the likely gain from a down-only optimization.
There is no new full-model throughput or 90% utilization claim.

Logs and hashes are in `results/20261020/q3-expanded-paired-screen/` and
`q3-expanded-paired-screen-node0/`. Link the driver with the scratch kernel's
`q3_expanded_rows` symbol, the normal CPU/host/GGML archives, and run
`taskset -c CPU paired_probe TOKENS COPIES`. Only HIP-configured CPU libraries
were used; this experiment did not execute GPU code or test CUDA/SYCL builds.

### Paired complete-expert Q3 layouts (2026-10-20)

`q3_expanded_paired_pool.cpp` holds both Q3 down layouts in one process with
shared Q2 gate/up banks, activation inputs, pool workers and baseline references.
The scratch dispatcher in `q3_expanded_paired_dispatch.cpp` distinguishes the
explicit down row extents and calls separately named baseline/expanded kernels.
The driver requires NUMA-owned rows and `STRATA_Q23_AVX2=1`. This is not a
production loader/format dispatch. Compile the original Q23 kernel with
`q23_rows=q23_baseline_rows` and `q23_gu_rows=q23_baseline_gu_rows`, and link
alongside the previously qualified expanded kernel's `q3_expanded_rows` exports,
the scratch dispatcher and the normal CPU/host/GGML libraries.

On Threadripper 1950X, two experts, H4096/F2048, fifteen workers plus caller,
eight banks, Q2 LUT enabled, prefetch off, three tasks/thread and the small-expert
unfused option were tested. Each process has twenty pairs, reversing first
layout every round. Independent layout cursors visit matching bank indices;
each call advances a bank, including warmup. With eight banks and one warmup
before each timed call, odd-indexed banks are timed and even-indexed banks warm
the schedule. Q2 banks are shared, so order is balanced but cross-layout cache
reuse is possible. Both Q3 allocations coexist; this is not an equal-total-RAM
cache-capacity comparison. Format/shard selection occurs only after synchronous
pool completion. Every timed output and routed sum matches original-layout
split execution byte-for-byte.

Two full invocations of each token width/mode produced these median paired
candidate/baseline latency ratios (below one favors expanded scales):

| Mode | Tokens | First run | Repeat |
|---|---:|---:|---:|
| Split | 1 | 0.958 | 0.979 |
| Split | 4 | 0.974 | 0.978 |
| Dataflow | 1 | 0.970 | 0.961 |
| Dataflow | 4 | 0.978 | 0.977 |

All 320 timed calls passed. Split down-phase median paired ratios were
0.895/0.921 at one token and 0.952/0.940 at four across the two runs. Phase
clocks include scheduling waits. The first dataflow one-token run's standalone
layout medians were 0.5351/0.5379 ms despite a paired ratio of 0.970: a median
of ratios is not a ratio of medians. Preserve both and the raw outliers.
Background load was uncontrolled; these measurements show a modest paired
synthetic-service benefit, not a full-model speedup or 90% peak utilization.
Input quantization, allocation and conversion remain outside timing.

This supports evaluating an explicit opt-in expanded-Q3 native format; do not
change defaults or infer a model tok/s increase. Current production kernels
remain unchanged. Evidence, driver/dispatcher hashes and binary hashes are in
`results/20261020/q3-paired-experts/` and `q3-paired-experts-repeat/`. This
experiment used HIP-configured CPU archives only, with no GPU execution or
CUDA/SYCL build. The scratch candidate's broader kernel parity/sanitizer checks
are recorded separately above.

### Explicit expanded-Q3 AVX2 APIs (2026-10-20)

The qualified expanded-scale arithmetic is now exposed as
`q3k_expanded_rows` and `q3k_expanded_gu_rows` in `q23_avx2.hpp`. They accept
only the layout produced by the shared C11 converter; no default dispatch or
loader uses them. A compile-time layout parameter retains the original Q2/Q3
branches and their operation order. The expanded branch uses byte offsets so
odd padded row strides do not bind misaligned GGML structs.

The API checks K256..4096, widths 1..8, row ranges, minimum stride, extent
multiplication overflow, and all required token/buffer pointers before writes.
Empty ranges accept null buffers after validating geometry. Callers remain
responsible for CPU AVX2/FMA/F16C support, valid converted scale bytes, and actual
buffer capacities. GU requires both expanded weight planes. This API does not
infer representation from row stride and cannot fall back to GGML layout.

`glm_q3_expanded_dot_test` runs the previous 1,200 parity configurations against
original-layout kernels and independent decoded-weight dot references, plus
invalid geometry/null-token/null-up/overflow rejection and empty-range checks.
The new API, host converter, original Q23, and Q23 pool tests all pass in both
HIP- and CUDA-configured CPU builds (four tests per build). ASan/UBSan pass with
the kernel and driver instrumented; linked GGML/reference libraries were not
instrumented. LeakSanitizer was disabled because of tracing incompatibility.
SYCL is untested because its compiler is unavailable; no GPU execution ran.

Original Q2/Q3 dumps from both rebuilt trees match a saved pre-change binary
byte-for-byte with the Q2 LUT both off and on (24,657 bytes each, including test
status text). This establishes unchanged tested default outputs, not a claim
about every possible input. The initial test build lacked the CPU feature-check
header; that compile error was corrected before these passing checks. Evidence
and source hashes are in `results/20261020/q3-expanded-api/`. Native-format and
loader integration remain outstanding; no full-model speed claim follows from
adding these explicit APIs.

### Explicit native Q3 down-format integration (2026-10-20)

`NativeFmt::q3_expanded_down` now selects the expanded-Q3 down API directly.
It defaults off and `native_fmt` clears it when constructing a normal format.
The supported combination is ordinary Q2_K gate/up and Q3_K down with Q8_K
activations, AVX2, whole K256 dimensions up to 4096, original Q2 offsets, and
exact expanded down extents. Canonical, tiled, lossless, NVFP4-K32, scale-tail,
and incompatible type/extent combinations are rejected before GU/down output
writes. Conversion remains explicit via the C11 converter: only down bytes
change, then d_row and total bytes must describe the converted representation.
No environment switch, loader auto-selection, fallback to original Q3 bytes,
or default change was added.

The paired pool driver now sets this real native flag and links only normal
CPU libraries; the historical scratch dispatcher is not linked. HIP- and
CUDA-configured CPU builds each passed 160 timed complete-expert comparisons
(two modes, two token widths, twenty pairs), all against original-layout split
outputs and routed sums. Existing shared-Q2 bank/warmup limitations still apply.
These runs are integration validation, not a stronger performance claim: load
was variable, test recompilation overlapped the start of the HIP screen, and
CUDA batch-4 split had a median paired ratio 1.028. Preserve the raw timings.
Evidence is in `results/20261020/q3-native-experts-hip/` and
`q3-native-experts-cuda/`.

The expanded dot test additionally covers complete native blobs and separate
down buffers at widths 1/4, partial-output guards, 24 incompatible-format
rejections across GU/down, and reset of the flag by normal format construction.
Both configured builds passed six CPU regressions: Q3 converter, Q3 expanded
API, FP4 K32 API, Q23 kernel, Q23 pool and dataflow pool. The extended dot test
was rebuilt and passed afterward in both trees. Original Q23 dumps remain
byte-identical to the saved pre-change binary with Q2 LUT on/off. Existing
unused umwait warnings appeared while rebuilding the pool. SYCL remains
unavailable; no GPU execution or full-model decode test ran. Source/check
records are in `results/20261020/q3-native-format/`.

### Checked complete-expert preparation (2026-10-20)

The shared C11 `strata_q2q3_scale_layout` now computes original/expanded complete
Q2_K gate/up + Q3_K down extents with checked size arithmetic. Its companion
`strata_q2q3_expand_down` requires exact, disjoint complete blobs, copies gate/up
unchanged, and expands only down scales. Invalid geometry, null buffers,
size mismatch, address overflow and overlap are rejected before writes. Layout
failure leaves the caller's layout object unchanged. This keeps the byte-layout
rule in `strata_host`, rather than duplicating it in the native caller.

`native_prepare_q3_expanded_down` validates an original native format, prepares
converted bytes in a temporary vector, and commits the expanded format and blob
only after success. The source format may alias the output format, and input
bytes may reside in the output vector: old storage remains valid through
conversion. Wrong input sizes, incompatible formats or allocation failures
leave both outputs unchanged. Preparation holds old and new bytes transiently
for one expert; callers should process bounded expert batches rather than
retain a second full-model copy. No loader or default selection was added.

C11 tests cover four complete-expert geometries, retained GU bytes, exact down
round trips, guards, null/extent/overlap rejection and arithmetic overflow. The
native API test exercises aliased preparation, wrong-size rollback, rejection
of already-expanded input, and complete/separate-down execution parity. Both
HIP- and CUDA-configured CPU builds passed the C11 layout, expanded-dot and
Q23 pool tests (three per build). The shared C11 tests also pass ASan/UBSan
with LeakSanitizer disabled for tracing compatibility. The no-native-experts
fallback passed a compile-only check; it throws without modifying outputs.
SYCL remains unavailable; no GPU or new throughput measurement ran. Existing
unused umwait warnings remain. Source hashes are recorded under
`results/20261020/q3-expert-preparation/`.

### Preparation helper through NUMA expert execution (2026-10-20)

The paired pool driver now assembles each original expert and calls
`native_prepare_q3_expanded_down`, verifies unchanged gate/up bytes, and places
the returned down bytes into NUMA-owned banks. It no longer constructs expanded
format offsets manually. Temporary complete-blob storage is bounded to one
expert at a time; persistent paired layout banks retain the earlier footprint.

The HIP-configured CPU build passed 160 timed comparisons across split/dataflow,
one/four tokens, two experts, eight banks and twenty alternating pairs per
configuration. Every output and routed sum matches original-layout execution.
Median paired candidate/baseline latency ratios were split 0.9505/1.0108 and
dataflow 0.9602/0.9852 at one/four tokens. Host load remained uncontrolled;
the batch-4 split result does not support a universal service speedup.

Measured preparation for two H4096/F2048 experts was 34.04–43.33 ms, including
original-blob assembly, conversion, prefix verification, allocation and copying
expanded down bytes, but excluding NUMA bank construction. These are four
single startup observations under load, not isolated converter throughput.
Each expert grew from 9,109,504 to 9,240,576 bytes (+1.439%). This preparation
cost is far larger than the per-call service saving: use retained prepared
weights and measure initialization separately; converting on every decode or
cache miss is not justified by these numbers. No new default or loader hook
was added. Evidence is in `results/20261020/q3-prepared-pool/`.

Decoder inspection identifies additional integration work: `glm_decode.cpp`
selects layer formats separately in its single/multi-token paths, supports both
prepared and NUMA-owned tensors, and applies canonical-format selection after
prepared-format selection. The current expanded native format rejects canonical
mode. A loader hook must explicitly resolve those combinations and preserve
original GPU/upload bytes; simply replacing a tensor pointer is insufficient.

### Faster shared Q3 scale conversion (2026-10-20)

The C11 scale expander now unpacks four scale bytes per 32-bit word instead of
sixteen separate bit-field calculations. Bytewise masks reconstruct the six-bit
values; low five bits plus sign extension produce identical two's-complement
bytes without cross-byte subtraction borrows. Loads/stores use memcpy, avoiding
alignment and aliasing assumptions. Arithmetic operates independently within
each byte; execution has been tested on x86-64, not a big-endian target.
The inverse and all validation rules remain unchanged.

`q3_conversion_paired.cpp` links old and new implementations in one process,
reverses first/second order over twelve pairs per size, and compares every
batch's complete output. On Threadripper 1950X CPU 11, matching O3/NDEBUG C11
builds gave the following process CPU-time measurements under uncontrolled load:

| K256 blocks | Original ms/call | New ms/call | Median paired new/original ratio |
|---|---:|---:|---:|
| 2,048 | 0.04409 | 0.02291 | 0.513 |
| 32,768 (one H4096/F2048 down matrix) | 0.50842 | 0.37044 | 0.719 |
| 262,144 | 4.87100 | 4.19388 | 0.860 |

The new implementation won all 36 pairs; all 72 timed batches matched bytes.
Buffers are allocated before timing and reused. Calls per batch are 32 for the
two smaller sizes and four for the largest; the benchmark excludes complete
expert assembly, allocation and NUMA placement. It therefore cannot attribute
the previous 34–43 ms complete preparation cost to this conversion loop: the
observed isolated converter is much cheaper. Allocation/copying and scheduling
remain relevant preparation costs.

Integrated into the shared C11 implementation after exhaustive scale-position,
round-trip, rejection and complete-blob tests passed, including ASan/UBSan
(LeakSanitizer disabled for tracing compatibility). Both HIP- and CUDA-configured
CPU builds passed converter and expanded native-dot tests after integration.
No loader/default selection or arithmetic result changed. No new steady-state
decode speed claim follows; this optimizes one-time preparation. SYCL and GPU
execution were not tested. Frozen old/new patch, logs, hashes and summaries are
in `results/20261020/q3-word-conversion/`.

### Canonical arithmetic with expanded Q3 down weights (2026-10-20)

Expanded native Q3 down weights now permit the existing canonical Q2/Q3 mode.
Canonical input/hidden Q8 quantization and original Q2 gate/up computation are
unchanged; only down scale loading differs, with the same integer sums and
floating-point accumulation order. The earlier blanket canonical rejection was
removed. Other type/layout/extent/ISA checks remain. No decoder loader hook,
GPU representation change, or default selection was added.

Native complete-blob/separate-down parity now runs with canonical both off and
on. Canonical expert tests and expanded API tests pass in HIP- and CUDA-configured
CPU builds. The paired preparation-to-NUMA pool driver additionally accepts an
optional canonical test mode: 0 ordinary, 1 canonical with no gate skipping,
2 canonical with an explicit 0.5 gate-skip threshold for testing only.
This does not enable skipping in production or assert that skipping is lossless.

Final complete-expert validation covered split/dataflow, one/four tokens, two
experts, eight banks and twenty pairs per configuration, with and without the
test skip threshold. All 320 timed calls match original-layout canonical expert
outputs and routed sums exactly. Skip-case hashes differ from no-skip hashes
and are nonzero, demonstrating that the test affects the result while preserving
layout parity. The initial 0.01 threshold produced unchanged hashes and was not
counted as proof of effective skipping; an additional high-threshold diagnostic
confirmed all-zero output before the final 0.5-threshold run. Original initial
logs are retained under `q3-canonical-experts/`; final logs and hashes are under
`results/20261020/q3-canonical-verified/`.

Complete-expert execution used HIP-configured CPU libraries; CUDA validation
here was the model-free canonical/expanded test suite. Host load was uncontrolled
and compilation overlapped some validation runs; no throughput claim is made.
No GPU execution or SYCL validation ran. Decoder integration still must retain
original GPU/upload bytes and reconcile prepared versus NUMA-owned sources.

### Direct expanded-Q3 NUMA arena construction (2026-10-20)

`NumaTensor` has an explicit final `expanded_q3` constructor argument, default
false. When selected, source rows remain original Q3_K (110 bytes/K256) and
output rows must have the expanded extent (114 bytes/K256). Each node's bound
worker converts its half of each expert directly into its destination arena
using the shared C11 converter. No whole-layer expanded temporary is allocated.
Mixed packed-tile/expanded requests, incompatible columns/row extents, and
source extent overflow are rejected. The existing thread-error cleanup applies.

`copy()` deliberately returns the expanded arena bytes. Callers must not put
these tensors in the original-GGML GPU-upload source map; GPU uploads need the
original artifact. Decoder inspection found that copied NUMA tensors normally
enter that map, so loader integration must explicitly omit expanded down arenas
and update CPU-only format selection in both decode paths. That loader change
has not been made yet; current decoder defaults and upload sources are unchanged.

The Q23 pool test now checks direct expanded Q3 arenas with ordinary and huge
pages, canonical Q2/Q3 output parity against original rows, fused hidden
quantization on/off, and copying expert 1 back against independent C11
conversion. Existing Q2/IQ4 and tiled-layout cases remain exercised. Final Q23
pool and general NUMA pool tests pass in HIP- and CUDA-configured CPU builds
(two tests each). No throughput measurement, full decoder build, GPU execution
or SYCL validation is claimed by this step. Source hashes are in
`results/20261020/q3-direct-numa/`.

### Decoder copied-NUMA integration (2026-10-20)

The decoder now recognizes the explicit startup option
`STRATA_GLM_Q3_EXPANDED_DOWN=1`, read with the shared exact-one environment
helper. It requires `STRATA_GLM_Q2_NUMA_WEIGHTS=1`, AVX2, and Q2_K gate/up plus
Q3_K down in every owned MoE layer. Existing copied-NUMA restrictions still
apply: pinned workers on two nodes, one GPU, no legacy expert cache, CPU
prepack or direct-upload mode. Expanded mode additionally rejects mapped/tiled
rows, remote tensor parallelism, layer-coordinator shared staging and LLVQ.
This is a Q2/Q3 option, not a change to the FP4 model loader.

Tensor types, shapes and original byte extents are checked before allocation.
The NUMA budget includes expanded down bytes. Each node converts directly into
its own arena. Both decoder CPU expert paths select expanded down metadata
only after successful preparation; canonical quantization, fusion and routing
settings are preserved. With the option off, the format-selection helper exits
before looking up a layer. Conversion failure clears all partial NUMA ownership.

Expanded down tensors are deliberately excluded from the original-GGML upload
source map. GPU uploads continue to read original artifact bytes; those files
are not modified. The resident-row CPU/GPU diagnostic reconstructs an original
Q3 reference format before reading artifact down bytes or writing a diagnostic
fixture. CPU jobs that also retain original pointers use original expert byte
strides, while their node-owned shards use expanded strides. Original down file
pages may need to be faulted back in for uploads; GPU/CPU overlap and full-model
performance with this option are not yet measured. The flag stays off by default.

Integration validation: the full HIP and CUDA decoder targets built, and both
executables returned their expected usage message (exit 2) without loading a
model. The HIP smoke required a scoped ROCm runtime library search path. Five
CPU tests passed in each configured tree: scale layout, expanded dot products,
Q23 pool, canonical expert and NUMA pool. These checks do not exercise the
new decoder loader or GPU upload path. SYCL validation is unavailable.

A subsequent review tightened the opt-in loader preflight to the native pool's
`kNativeFF` intermediate-width limit (2048), before arena allocation; the dot
kernel's standalone 4096 limit is not the pool's capacity. Both full decoder
rebuilds for this final guard passed, followed by successful usage smoke checks.
Source hashes and `runtime-smoke.json` identify the final guard revision and
its successfully built binaries. No new throughput or peak-utilization claim follows.

### FP4 gate/up scale-gather screen (2026-10-20)

An isolated AVX2 candidate replaced four scalar UE4M3 table loads in the signed
row kernel with one four-lane gather plus scale broadcasts. Weight layout,
integer dot products, FMA order and SwiGLU/clamping were unchanged. Candidate
symbols were renamed and linked beside the baseline from the HIP-configured
CPU archive; no production source or default was changed. Disassembly confirmed
that the candidate emitted `vgatherdps`.

On Threadripper 1950X, pinned CPU 3, twelve alternating same-process pairs per
case gave these median candidate/baseline process-CPU-time ratios:

| Tokens | Packed working set | Candidate/baseline |
|---|---:|---:|
| 1 | 589,824 bytes | 1.422 |
| 1 | 37,748,736 bytes | 1.405 |
| 4 | 589,824 bytes | 1.152 |
| 4 | 37,748,736 bytes | 1.164 |

The candidate lost all 48 pairs. Wall-time ratios closely matched process CPU
time. All 96 timed batches and initial per-bank comparisons matched baseline
output bits. Each batch made 64 calls; the large case rotated 64 independently
initialized gate/up banks. Synthetic data used all byte-valued scale/code
inputs and raw signed Q8 activation bytes, including possible -128 values;
clamping was 7 with global gate/up scales 0.001. Prefetch, offset-dot and
128-bit options were disabled. Host load was uncontrolled.

The candidate is rejected. This tests GU kernels only, not full expert service,
NUMA streaming or model decode. The existing production implementation is the
parity reference; this screen adds no independent numerical-quality proof.
No GPU, CUDA/SYCL candidate build or sanitizer validation ran. The reusable
paired harness is `fp4_scale_paired_probe.cpp`; logs, candidate patch and hashes
are in `results/20261020/fp4-scale-gather/`. The next scale-decoding candidate
must avoid gather overhead rather than merely combine scalar loads.

### FP4 arithmetic scale decoding screen (2026-10-20)

A second isolated candidate formed four FP32 scale values directly from the
four UE4M3 bytes using SIMD integer exponent construction. It handled exponent
zero with mantissa times 2^-10, cleared the exact 0x7f sentinel, and retained
the production table's treatment of the high bit (including nonzero 0xff).
An exhaustive helper check covered every byte value in each of four lanes and
matched the table bitwise. Dot arithmetic and reduction order were unchanged.

The same paired GU harness, CPU 3 on Threadripper 1950X, prefetch/128-bit/offset
options disabled, gave these median candidate/baseline process-CPU-time ratios:

| Tokens | Packed working set | Candidate/baseline |
|---|---:|---:|
| 1 | 589,824 bytes | 1.136 |
| 1 | 37,748,736 bytes | 1.129 |
| 4 | 589,824 bytes | 1.082 |
| 4 | 37,748,736 bytes | 1.089 |

The candidate lost 46 of 48 alternating pairs. All 96 timed batches and initial
per-bank checks matched baseline output bits. The candidate is rejected;
production source/defaults are unchanged. Alongside the gather result, this
provides no evidence for replacing scalar scale-table loads on this CPU. It
does not prove that scale processing has zero cost or that every alternative
is slower. Host load was uncontrolled; no full expert/model speed, theoretical
utilization, GPU execution, sanitizer result or CUDA/SYCL candidate validation
is claimed. The generator, patch, exhaustive checker, logs and hashes are under
`results/20261020/fp4-scale-arithmetic/`.

### FP4 compiler code-generation screen (2026-10-20)

The unchanged current AVX2 source was compiled with Ubuntu Clang 21.1.8 and
symbol-renamed so the paired GU harness could call it beside the GCC 15.2.0
baseline from the HIP-configured CPU archive. Both used O3, C++20, PIC and
AVX2/FMA/F16C; no fast-math option, ISA extension or numerical change was added.
This is separate from the previously rejected GCC Zen 1 tuning experiment.

Twelve alternating pairs per case, 64 calls per timed batch, gave median
Clang/GCC process-CPU-time ratios below. The repeat used CPU 11 on the other
NUMA node and reversed case order. Host load remained uncontrolled.

| Tokens / working set | CPU 3 | CPU 11 |
|---|---:|---:|
| 1 / 589,824 bytes | 1.104 | 1.074 |
| 1 / 37,748,736 bytes | 1.019 | 1.060 |
| 4 / 589,824 bytes | 0.974 | 0.940 |
| 4 / 37,748,736 bytes | 0.980 | 0.949 |

All 192 timed batches and initial per-bank checks matched baseline output bits.
Four-token Clang won 46/48 pairs; one-token Clang won only 3/48. These results
support investigating batch-four code generation but reject a blanket compiler
switch for single-token decode. They do not yet establish a full-expert gain:
input preparation, hidden quantization, down projection, scheduling and reduction
remain outside this GU screen. Production compiler settings and kernels are
unchanged. No GPU, CUDA/SYCL candidate validation, full-model decode or peak
utilization claim follows. Logs, compiler versions and source/object/binary
hashes are in `results/20261020/fp4-clang-screen/`.

### Paired Clang/GCC complete FP4 expert service (2026-10-20)

A scratch dispatch wrapper links symbol-renamed GCC and Clang builds of the
unchanged FP4 source into one executable. An atomic selection is changed only
between synchronous pool calls; both variants pay the wrapper cost and use the
same original-GGML weights, scales, inputs and routing weights. The existing
partial-weight pool harness was adapted to alternate compiler selection rather
than K32 layout. Production sources and compiler settings are unchanged.

Two runs used real layers 3/4, two experts per layer, eight rotating banks,
15 workers plus host on Threadripper 1950X, tokens 1/4, split/dataflow and owned
NUMA/fused-hidden-quantization variants. Weight payload per eight-bank set is
226,492,672 bytes. Timing includes GU, hidden quantization, down projection,
scheduling and routing reduction; input quantization, setup and NUMA arena
construction are excluded. Prefetch, 128-bit, offset-dot and AVX-512 paths were
disabled. Host load was uncontrolled.

Median across configuration ratios (each ratio compares the two nine-sample
latency medians) was:

| Tokens / ownership | First Clang/GCC | Repeat Clang/GCC |
|---|---:|---:|
| 1 / ordinary | 1.022 | 0.979 |
| 1 / NUMA-owned | 1.066 | 1.044 |
| 4 / ordinary | 0.971 | 0.946 |
| 4 / NUMA-owned | 0.955 | 0.966 |

Clang won 22/24 four-token configuration comparisons. The owned four-token
ranges were 0.922–0.962 initially and 0.924–1.034 on repeat, so it did not win
every case. Single-token results were mostly slower or noisy and do not support
a global compiler switch. All 864 timed checks and 96 final CPU-batch checks
matched complete expert output and routing reduction bits. Reference outputs
were generated with GCC before either timed variant. CPU-cost diagnostic
batches are sequential 32-call batches, not alternating latency samples.

This supports further batch-specific code-generation work, not a full-model
tok/s or 90%-of-peak claim. No GPU or CUDA/SYCL candidate validation ran. The
source generator, dispatch wrapper, logs and source/object/binary hashes are in
`results/20261020/fp4-clang-pool/`. Its generator uses the unchanged-source Clang
candidate object from the preceding compiler screen; it does not install that
object into a production build.

### Four-token GU-only compiler selection and identical-code control (2026-10-20)

Phase records from the preceding compiler screen showed GU improvements in
most split configurations, but down-phase results were mixed. A narrower scratch
wrapper now selects Clang only for ordinary four-token GU entry points. Down,
K32 and every other token count use GCC. This keeps the known one-token Clang
regression out of the candidate. The original source/defaults remain unchanged.

Two real-weight complete-expert runs retained the previous two-layer/two-expert,
eight-bank, 15-worker setup and parity checks. A third executable forced GCC
for both compiler labels, supplying an identical-code control. Four-token
median configuration latency ratios were:

| Selection | Ordinary rows | NUMA-owned rows |
|---|---:|---:|
| GU-only Clang, first | 0.965 | 0.957 |
| GU-only Clang, repeat | 0.961 | 0.977 |
| GCC/GCC control | 0.995 | 1.008 |

The GU-only candidate won 21/24 four-token comparisons. This remains a small
candidate gain, not a precise speedup estimate. Four-token NUMA control ratios
ranged 0.978–1.029, and the repeat candidate ranged 0.716–1.048. One-token calls
also execute identical GCC code in the candidate probe: their first-run NUMA
ratios ranged 0.575–1.359 despite identical computation. This exposes host-load
noise and prevents interpreting individual outliers as kernel improvements.
No samples were discarded or corrected using the separately timed control.

All 1,296 timed checks and 144 final CPU-batch checks passed bitwise expert and
routing-reduction parity. The `compiler=clang` label in `control.log` denotes
the benchmark slot only; it executes GCC. Input quantization/setup remain
excluded, and no full-model, GPU or peak-utilization measurement was made.
Wrappers, hashes and complete compact logs are in
`results/20261020/fp4-clang-gu-only/`. The next implementation candidate should
preserve GCC single-token behavior and isolate the four-token GU improvement;
a global compiler change is not supported by these results.

### GCC partial-dot source-order screen (2026-10-20)

Disassembly of the four-token dot function showed Clang completing the first
32-value partial dot/FMA before starting the second, while GCC kept additional
intermediate values live. An isolated source candidate explicitly scoped each
half's activation load, dot and scaled FMA in sequence. Accumulator update order
was unchanged. GCC still controls instruction scheduling; source scopes alone
do not enforce an assembly schedule.

GCC 15.2.0 emitted a 1,901-byte four-token ordinary row function versus 1,944
bytes for baseline and 908 bytes for Clang. Function size includes control paths
and is not an instruction-count or throughput measurement. The paired GU screen
on TR1950X CPU 3 gave candidate/baseline process-CPU medians 1.004/0.979 for one
token (589,824-byte/37,748,736-byte weight sets) and 1.014/0.999 for four tokens.
The four-token candidate won only 10/24 pairs. All 96 timed batches matched
baseline output bits. Options and protocol matched the earlier scale screen.

This source-order change does not reproduce Clang's four-token gain and is not
integrated. Host load remained uncontrolled; no full expert, model, GPU,
CUDA/SYCL or sanitizer qualification was performed for the rejected candidate.
The patch, compact logs and hashes are in `results/20261020/fp4-half-schedule/`.
The batch-specific Clang result remains a candidate; neither these assembly
observations nor the smaller function prove why its measured service was faster.

### Q2/Q3 compiler screen (2026-10-20)

The unchanged Q2/Q3 AVX2 source was compiled with Clang 21.1.8, exported under
candidate names and linked beside the GCC 15.2.0 baseline. The paired row probe
now accepts optional TYPE 10/11 (Q2_K/Q3_K); omitting it retains Q2_K. Every
screen checks baseline against an FP64 decoded-weight reference, then checks
each timed candidate batch bitwise against baseline. ISA flags and O3 match;
no fast-math option or production source/compiler setting changed.

On Threadripper 1950X CPU 3, median Clang/GCC process-CPU ratios were:

| Format / tokens | One bank | 64 banks |
|---|---:|---:|
| Q2 ordinary / 1 | 1.098 | 1.096 |
| Q2 ordinary / 4 | 0.985 | 0.986 |
| Q2 LUT / 1 | 1.168 | 1.172 |
| Q2 LUT / 4 | 1.002 | 0.995 |
| Q3 / 1 | 1.026 | 1.064 |
| Q3 / 4 | 0.814 | 0.807 |

Q3 repeated on CPU 11, reversing case order, gave 0.820/0.825 for four tokens
and 1.075/1.044 for one token. Four-token Q3 won all 48 paired comparisons
across both cores and working sets. This is a consistent 17.5–19.3% reduction
in this row-kernel CPU time; it is not a complete-expert or model speedup.
Q2 LUT shows no useful four-token gain and a substantial one-token regression.
The promising next selection is therefore Q3 four-token down projection only,
retaining GCC Q2 gate/up and single-token code.

All 384 timed batches passed. Each batch had 512 calls; shape was K4096,
128 rows, with one or 64 copies. Weight sets are 172,032/11,010,048 bytes for
Q2 and 225,280/14,417,920 bytes for Q3. The larger rings exceed a local cache
slice but are not a full-model DRAM workload. Host load was uncontrolled;
no GPU, full expert/model, CUDA/SYCL or sanitizer qualification ran for this
screen. Compact logs and source/binary hashes are in
`results/20261020/q23-clang-screen/`.

### Q3 four-token Clang down projection in complete expert service (2026-10-20)

`q3_compiler_paired_pool.cpp` adapts the expanded-layout paired pool probe to
select between compiler objects while sharing the same original Q2/Q3 NUMA
weight banks. The scratch wrapper dispatches only type-11, four-token ordinary
row calls to Clang. All Q2 GU, one-token, expanded-Q3 and other calls use GCC.
Selection changes between synchronous pool calls through an atomic flag; the
reference is generated before enabling the candidate. No production source or
compiler setting changed.

TR1950X, 15 workers plus host, two synthetic H4096/F2048 experts and eight banks
(145,752,064 packed bytes) gave these median paired candidate/baseline complete
service latency ratios over twenty alternating pairs per case:

| Mode | First | Repeat | Canonical enabled |
|---|---:|---:|---:|
| Four-token split | 0.916 | 0.923 | 0.899 |
| Four-token dataflow | 0.924 | 0.909 | 0.894 |

Four-token candidate calls won 114/120 pairs. The first split down-phase paired
ratio was 0.818, consistent with the row-kernel screen at the actual down width
2048. One-token identical-code controls gave 0.994 split and 0.995 dataflow;
both labels execute GCC. All 320 timed outputs and routing reductions matched
reference bits, including canonical mode. Q2 LUT and Q23 AVX2 were enabled;
software prefetch was disabled. No gate skipping was enabled.

These measurements support a batch-four Q3-down implementation candidate, with
roughly 8–11% lower complete-expert latency in these cases. They do not prove
full-model performance or that changing the default compiler is safe. Input
quantization and setup are excluded. Warm and timed calls each advance the bank,
so four banks are timed and four are warmed; both compilers share those banks.
The data are synthetic and background load was uncontrolled. Real-model routing,
expanded-Q3 interaction, GPU execution and SYCL qualification remain untested.
The wrapper, compact logs and source/object/binary hashes are under
`results/20261020/q3-clang-pool/`.

### Q3 compiler × expanded-layout compatibility (2026-10-20)

A four-variant extension of the paired pool probe compares original/expanded
Q3 rows under GCC/Clang in one pool. Only four-token down calls may select
Clang; Q2 GU and single-token code remain GCC. Each round rotates variant order,
uses shared Q2 banks and the corresponding original or expanded down arenas,
and checks against the original-GCC split output. Twenty rounds cover one/four
tokens, split/dataflow, and four-token canonical mode. All 480 timed expert
outputs and routing reductions matched reference bits.

This establishes compatibility in these synthetic complete-expert cases. It
fails to establish that layout and compiler gains add. Median paired four-token
combined/original-GCC latency ratios ranged from 0.603 to 1.042 across modes;
combined/original-Clang also changed direction. These broad ratios are retained
as noisy evidence, not reported as a 40% speedup or an additive improvement.
The corresponding original-layout Clang/GCC ratios ranged 0.807–1.006, also less
consistent than the earlier two-variant screen.

A separate two-second host observation around completion showed 52.35%
aggregate CPU busy time, CPU-pressure avg10 8.40 and memory-pressure avg10 zero.
This is not per-call attribution or a proof of the cause of each outlier. Miners
and other workloads were left running. No samples were discarded, timings were
not corrected using the host sample, and production settings remain unchanged.

Hardware/configuration otherwise match the preceding Q3 compiler pool screen.
Input quantization and preparation are untimed; weight data are synthetic. No
GPU, full-model decode or SYCL validation ran. The compatibility result permits
future expanded-Q3 qualification, but performance promotion still needs stronger
evidence. Harness, wrapper, logs and hashes are in
`results/20261020/q3-compiler-layout/`.

### GCC Q3 forced-inlining screen (2026-10-20)

The preceding compiler objects differ in an observable way: GCC emits separate
four-token dot functions while Clang inlines dot processing into row functions.
A scratch GCC candidate added `always_inline` to the dot template, preserving
all arithmetic. This tests whether inlining alone reproduces the Clang gain;
it does not establish why the compilers differ in performance.

The paired Q3 row screen on TR1950X gave median process-CPU-time ratios:

| Tokens / weight banks | CPU 3 | CPU 11 |
|---|---:|---:|
| 1 / 1 | 0.956 | 0.989 |
| 1 / 64 | 0.977 | 0.974 |
| 4 / 1 | 1.018 | 0.993 |
| 4 / 64 | 0.982 | 0.994 |

Four-token inlining won only 27/48 pairs and did not reproduce the earlier
17.5–19.3% Clang row-kernel reduction. One-token inlining won 39/48 pairs, a
small possible lead (1–4% CPU-time reduction) that remains unqualified at full
expert scope. Some CPU 11 wall ratios differed substantially from process-CPU
ratios, consistent with uncontrolled scheduling effects; wall improvements are
not reported as kernel gains. All 192 timed batches matched baseline bits.

No production source or compiler setting changed. The blanket template
attribute is only a prototype and has not been qualified for other types,
expanded layouts, full expert service, GPU, CUDA/SYCL or sanitizers. The patch,
compact logs and hashes are in `results/20261020/q3-forced-inline/`. The result
rules out inlining alone as an explanation for the large batch-four compiler
gain and leaves the single-token candidate for a separate service-level test.

### Single-token Q3 forced-inline complete-service check (2026-10-20)

The compiler-paired pool harness was linked with the GCC forced-inline object,
selecting it only for ordinary Q3 down calls at one token. Q2 GU, expanded rows
and four-token calls stayed on baseline GCC. Both objects in this experiment
use GCC; the reused `set_q3_compiler` interface name does not imply Clang.

With the previous synthetic two-expert H4096/F2048, eight-bank, 15-worker setup,
median paired complete-service candidate/baseline ratios were:

| Mode | Ordinary quantization | Canonical mode |
|---|---:|---:|
| One-token split | 1.039 | 1.358 |
| One-token dataflow | 1.131 | 0.976 |

The candidate won only 34/80 single-token pairs. Four-token identical-code
controls gave 1.008 split and 1.009 dataflow. All 240 timed expert/reduction
outputs matched reference bits. This fails to carry the small single-core row
benefit into complete service, so the forced-inline candidate is not promoted.
Large individual phase/timing ratios remain host-load-conditioned evidence,
not a precise intrinsic slowdown estimate. No samples were discarded.

Input quantization/setup are excluded; Q2 LUT and Q23 AVX2 were enabled and
prefetch disabled. Host load was uncontrolled. No production change, GPU work,
full-model decode or CUDA/SYCL qualification occurred. Logs, wrapper and hashes
are in `results/20261020/q3-inline-pool/`. The batch-four Clang Q3-down candidate
remains the stronger lead; this screen provides no reason to alter single-token
compiler behavior.

### Opt-in Q3 four-token Clang integration (2026-10-20)

The experimental CMake option `STRATA_Q3_CLANG_NT4` builds a second, symbol-renamed
copy of the existing Q2/Q3 CPU source with Clang. It is off by default and is
currently restricted to native Linux x86-64 GNU host builds. The custom object
uses libstdc++, O3, PIC, AVX2/FMA/F16C and a dependency file for included headers.
`STRATA_Q3_CLANG_COMPILER` can select the clang++ executable; tested version is
21.1.8. This does not change the GPU compiler or the normal CPU compiler.

Runtime activation is separate and also off by default. The exact-one C11
helper reads `STRATA_Q3_CLANG_NT4=1` once, at the first original-Q3 four-token row
call. Only those calls dispatch to the Clang object. Value 0, value 2 and unset
retain baseline behavior. Q2 GU, single-token calls and expanded Q3 retain the
existing path. An ordinary build without the option ignores this environment
variable. The existing Q23 AVX2 path must also be selected when needed:

```sh
cmake -S . -B BUILD_DIR -DSTRATA_Q3_CLANG_NT4=ON
cmake --build BUILD_DIR --target strata-glm-decode -j 4
STRATA_Q23_AVX2=1 STRATA_Q3_CLANG_NT4=1 BUILD_DIR/strata-glm-decode MODEL TOKEN_IDS
```

Configure an appropriate HIP/CUDA build as usual before using these commands.
The flag is an experiment, not a default recommendation or a model throughput
claim. Its supporting measurement remains the earlier synthetic complete-expert
batch-four screen. The expanded-layout/compiler combination is deliberately
not selected by this flag because its additional performance is unproven.

The default-disabled Q23 CPU object compiled with the original build command
is byte-identical to its saved pre-change object (SHA-256
`abea083f4be5ba6a16a8c42a1be7c1845ab22359e6eff2e72b2834b4507336e6`).
Enabled builds passed five CPU tests each in HIP- and CUDA-configured trees:
Q23 rows, Q23 pool, flow pool, expanded Q3 dot and canonical expert. Runtime-off
value 0 passed Q23 rows/pool in HIP; value 2 passed those tests in CUDA. These
are CPU execution checks, not GPU execution or a full model quality check.
Full decoder build completion is recorded separately under
`results/20261020/q3-clang-integration/`. SYCL is unsupported by this opt-in
build path and was not tested. No new full-model tok/s or peak-utilization claim
is made.

Final integration build checks: full HIP and CUDA decoder targets both completed
successfully. Rebuilding after the final CMake guard reported no work to do in
both trees. Both executables returned the expected no-argument usage message
(exit 2); the HIP smoke used a scoped ROCm runtime library path. These startup
checks did not load weights or execute GPU kernels. Final binary hashes and
check results are in the integration `checks.json`.

### Integrated Q3 runtime-switch service measurement (2026-10-20)

A CPU-only service probe now links the actual opt-in production archive, without
the scratch compiler dispatcher or manually injected candidate object. Separate
processes select runtime flag 0/1/1/0 (ABBA). Each process runs twenty measured
four-token calls using synthetic H4096/F2048, two experts, eight shared NUMA
banks and 15 workers plus host. Q23 AVX2/Q2 LUT are enabled; prefetch is off.
Input quantization and setup are excluded. All 160 timed checks passed, and
complete expert plus routing-reduction hashes match across all eight processes.

| Runtime flag / run | Split median ms | Down median ms | Dataflow median ms |
|---|---:|---:|---:|
| 0 / first | 1.1732 | 0.5177 | 1.0903 |
| 1 / first | 1.0712 | 0.4186 | 0.9802 |
| 1 / second | 1.1111 | 0.4181 | 1.5149 |
| 0 / second | 1.1759 | 0.5050 | 1.4235 |

Split service retains roughly 5–9% lower latency, with down-phase medians
17–19% lower. Dataflow is mixed: the first enabled run is faster, while the
second is slower than the final disabled run. Keep that negative result;
separate-process ABBA cannot remove changing host contention. These measurements
confirm integrated runtime operation and split-service behavior, not a uniform
end-to-end speedup, full-model tok/s or theoretical peak utilization. The earlier
same-process paired screen remains the cleaner compiler comparison.

No GPU ran. The probe, logs, output hash and source/archive/binary hashes are in
`results/20261020/q3-integrated-service/`. This is a synthetic CPU-layer check;
real-model routing and full decoder performance remain unmeasured.

### Opt-in four-token NVFP4 Clang GU integration (2026-10-20)

`STRATA_NVFP4_CLANG_NT4` is a separate CMake option, off by default, for the
previous four-token GU compiler candidate. As with the Q3 option, it supports
native Linux x86-64 GNU host builds and compiles a symbol-renamed copy of the
existing CPU source using clang++/libstdc++, O3, PIC and AVX2/FMA/F16C. The
compiler executable is selectable through `STRATA_NVFP4_CLANG_COMPILER`.

Runtime `STRATA_NVFP4_CLANG_NT4=1` is separately required. The shared exact-one
C11 helper is read once on the first eligible call. Only ordinary four-token
AVX2 GU entry points, clamped and unclamped, select Clang. Single-token modes,
down rows and K32 APIs retain GCC. This does not change AVX-512 selection; on a
CPU with that path enabled, `STRATA_NO_NVFP4_512=1` is needed to test AVX2 GU.
Unset, 0 and 2 leave runtime selection off. A default build without the option
ignores the environment variable.

```sh
cmake -S . -B BUILD_DIR -DSTRATA_NVFP4_CLANG_NT4=ON
cmake --build BUILD_DIR --target strata-glm-decode -j 4
STRATA_NVFP4_CLANG_NT4=1 BUILD_DIR/strata-glm-decode MODEL TOKEN_IDS
```

Use the normal backend configuration first. The option is experimental and
independent of `STRATA_Q3_CLANG_NT4`; neither changes a production default.
Both full HIP and CUDA decoder builds passed, as did three enabled CPU tests
in each tree (NVFP4 parity, K32 dot and Q23 pool). Runtime 0 passed NVFP4 parity
in HIP, and runtime 2 passed it in CUDA. Both decoder no-argument usage smokes
passed (exit 2); HIP used a scoped ROCm runtime library path. No GPU ran.

The default-disabled FP4 object is byte-identical to its saved pre-change object,
SHA-256 `53ec0c87aa7c9de1d656ed3a9678c0e3c122353357d0c7e58511e080d2fc9f54`.
Integrated real partial layers 3/4, two experts, eight banks and 15 workers plus
host passed 864 timed checks and 96 final CPU-batch checks. Output/reduction
hashes agree across separate runtime 0/1/1/0 processes. Four-token median
configuration latency ratios were 0.950/0.837 for ordinary rows and 0.933/0.848
for NUMA-owned rows in the two enabled-versus-neighboring-disabled comparisons.
Enabled four-token cases won 21/24 comparisons. These unusually broad gains
remain separate-process host-load-conditioned results: the earlier paired
GU-only screen supports a smaller approximately 2–4% service improvement and
is the cleaner comparison. Do not promote the 15–16% separate-process numbers
to a precise kernel or model speedup. One-token code is unchanged; its grouped
ratios still varied from 0.988 to 1.046.

Input quantization and setup are excluded. No full-model decode, theoretical
peak-utilization, GPU execution or SYCL validation is claimed. Logs, source and
binary hashes, parity counts and timings are under
`results/20261020/nvfp4-clang-integration/`.

### Ryzen worker validation of the FP4 compiler option (2026-10-20)

The Ryzen 9 3950X worker (16 cores/32 threads, one NUMA node) was checked using
its existing real partial-weight fixture. Before testing it had about 56 GiB
available RAM and load averages 31.31/31.05/30.66. Miners and other workloads
were left running. A statically linked integrated CPU benchmark passed a local
smoke, was copied under a content-derived temporary name, and its remote hash
was verified. No model download, GPU execution or system-library change occurred.

Separate runtime 0/1/1/0 processes used two experts from layers 3/4, eight banks
and 15 workers plus host. All 288 timed checks and 32 final CPU-batch checks
passed. Cross-process hashes matched. Timings were strongly unstable: the
four-token configuration-median enabled/baseline ratios were 2.128 and 2.087,
with individual four-token medians spanning roughly 1.3–5.5 ms. These runs do
not support enabling the option on this worker or establishing an intrinsic
compiler slowdown.

A bounded follow-up used the same-process GU-only compiler dispatcher. Its
static binary also passed locally and was hash-verified on the worker. All
144 timed checks and 16 CPU-batch checks passed and matched the integrated
runs' hashes. Four-token configuration ratios were 1.043, 0.951, 0.983 and 1.002
(median 0.993). The one-token identical-code control had median 0.914 and
individual ratios 0.892–1.066 despite both labels running GCC. Thus the nominal
four-token improvement is smaller than observed scheduling noise; no useful
worker gain is established. The option should remain off there pending stronger
evidence. The TR1950X result is not a universal default recommendation.

These are loaded-worker partial CPU tests, not idle-system bandwidth, 90%-peak
utilization or full-model tok/s. Input quantization/setup are excluded. Existing
partial weights were reused, and no GPU or persistent workload setting changed.
Compact logs, hashes and comparison summaries are in
`results/20261020/worker-nvfp4-clang/`.

### Ryzen worker Q3 compiler check (2026-10-20)

The Q3 candidate was also checked on the Ryzen 9 3950X worker. A static paired
row probe passed locally, was hash-verified after transfer and ran pinned to
CPU 3. Four-token Clang/GCC process-CPU ratios were 0.761 for one weight bank
and 0.788 for 64 banks, winning all 24 pairs. One-token ratios were 1.096/0.994
and do not support changing that path. All 96 timed row batches passed.

The compiler-paired pool probe now allows its already-implemented complete-blob
branch on single-NUMA workers; two-node hosts retain their owned-shard path.
A static build passed locally and was hash-verified on the worker. Twenty pairs
per case, two synthetic H4096/F2048 experts, eight banks and 15 workers plus
host gave these complete-service median paired Clang/GCC ratios:

| Mode | Ordinary quantization | Canonical mode |
|---|---:|---:|
| Four-token split | 0.901 | 0.965 |
| Four-token dataflow | 1.008 | 0.905 |

Those four-token cases won 12/20, 12/20, 9/20 and 14/20 pairs respectively
(split ordinary, split canonical, flow ordinary, flow canonical). One-token
identical-code controls gave 1.004 split and 1.074 flow, showing scheduling noise.
All 240 timed complete-expert outputs and reductions matched their reference.
Only ordinary four-token Q3 down calls select Clang; Q2 LUT gate/up and all
one-token calls retain GCC. Expanded Q3 was not tested on this worker.

The worker row-kernel gain is promising, but service is mixed under the running
workloads. Keep the option experimental; these data do not justify a worker
default change or a model-throughput forecast. Q23 AVX2/Q2 LUT were enabled and
prefetch disabled. Synthetic weights, input quantization/setup exclusion and
uncontrolled host load limit the result. No GPU, miner setting or production
worker configuration changed. Logs and source/static-binary hashes are under
`results/20261020/worker-q3-clang/`.

### Expert-batch coverage audit (2026-10-20)

Worker inspection confirms that request tokens are grouped by expert before
`run_split_multi_native`: the native kernel sees the number of tokens routed
to that expert, not necessarily the request width. A four-token verification
request can therefore produce one-, two- or three-token expert jobs. Greedy
one-token decode cannot use the new exact-four-token compiler dispatches.

`tools/sim/expert_batches.py` audits existing routing-event JSONL without running
hardware or changing simulator defaults. It groups expert IDs only within each
published slot, reports job-width and request-width histograms by phase, and
counts exact-width job and routed-row fractions. It also reports a nonresident
proxy only when every event in a phase has resident metadata. That proxy is not
CPU placement: GPU staging, CPU assignment and remote partitioning are absent
from the trace. Downstream slicing can change widths, and route fractions are
not fractions of measured service time.

```sh
python3 -m tools.sim.expert_batches TRACE_JSONL --width 4
```

Four existing historical routing traces contained 102,438 published slots, all
of width one or two. Exact-four eligibility was zero throughout. Consequently
the recent batch-four microbench/service gains must not be applied to those
executions or to the current batch-one FP4 simulator report. These historical
traces are not a routing calibration for the current FP4 checkpoint.

Two-token groups accounted for 34.2–43.2% of verification routed rows across
the four traces, and 17.8–27.9% under the nonresident proxy. The remainder in
these slots is single-token expert work. This changes the optimization priority:
measure actual two-token and single-token service before projecting decoder
benefits from exact-four improvements. The four-token options remain useful
experimental paths for workloads that actually produce that expert width.

Five audit tests passed (grouping, phase/slot boundaries, exact-width semantics,
missing resident data and malformed inputs), and all 33 existing FP4 simulator
tests passed. No rates or defaults changed. Compact coverage reports and source
hashes, without raw expert IDs or machine-specific trace paths, are under
`results/20261020/expert-batch-coverage/`.


### Two-token compiler screen and complete Q3 service (2026-10-20)

The routing-width audit found no four-token expert jobs in the available
historical traces. This screen therefore checks two-token jobs with the same
GCC and Clang arithmetic candidates. On the Threadripper 1950X, pinned cores
3 and 11, one and 64 weight banks, Q3 down rows used about 20% less process CPU
time with Clang (median paired ratios 0.796–0.804, 47/48 pair wins). Q2 with its
lookup-table path enabled regressed (ratios 1.018–1.046); FP4 GU was roughly
unchanged (0.975–1.002). All 288 timed row batches passed their parity checks.
These are synthetic row workloads, with input preparation outside timing.

A scratch dispatcher then selected Clang only for Q3 down jobs of width two.
Complete synthetic Q2-GU/Q3-down service used H=4096, F=2048, two experts,
15 workers plus the host, and eight shared NUMA weight banks. Twenty alternating
pairs per case gave the following median candidate/baseline wall-time ratios:

| Expert tokens | Arithmetic | Split | Dataflow |
| --- | --- | ---: | ---: |
| 1 (identical-code control) | ordinary | 1.028 | 1.012 |
| 2 | ordinary | 0.870 | 0.849 |
| 2 | canonical | 0.892 | 0.890 |

All 240 timed calls retained bit-identical expert outputs and routed sums.
The two-token cases won 69/80 pairs, but every expert receives both tokens in
this geometry. Historical nonresident-route proxies put only about 18–28% of
routed rows in two-token jobs. A mixed-width service experiment is needed before
using these gains for a broader performance estimate. Host load was uncontrolled;
there is no full-model throughput or peak-utilization claim. No production
runtime flag or default was changed. Raw logs and paired summaries are under
`results/20261020/nt2-compiler-screen/` and
`results/20261020/q3-nt2-service/`.


### Mixed one/two-token Q3 compiler service (2026-10-20)

The paired pool probe now accepts an optional `mixed` argument after canonical
mode. It builds seven expert jobs: both tokens use expert 0, token 0 also uses
experts 1–3, and token 1 uses experts 4–6. Each token has four equal route weights;
unused dense result slots and weights are zero. Thus 25% of routed rows belong
to a two-token job. This approximates one reuse fraction from historical traces,
not a trace replay, actual CPU placement or a full-model workload.

On the Threadripper 1950X, the same width-two-only Q3 Clang dispatcher, 15 workers
plus host and eight NUMA banks gave these paired wall-time ratios in two runs:

| Arithmetic / scheduling | First run | Repeat |
| --- | ---: | ---: |
| Ordinary split | 0.995 | 0.954 |
| Ordinary dataflow | 0.945 | 0.975 |
| Canonical split | 0.986 | 0.974 |
| Canonical dataflow | 0.983 | 0.985 |

All 320 timed calls passed bitwise expert-output and routed-sum checks; candidate
won 114/160 pairs. The ordinary dataflow reduction was 2.5–5.5%; other modes
ranged from 0.5–4.6%. Host load was uncontrolled and weights were synthetic,
repeated banks. Input quantization is excluded. This supports further testing
of the width-two Q3 candidate, but does not establish a model speedup, greedy
single-token gain, or 90% peak utilization. Production flags/defaults remain
unchanged; the experiment uses a separately linked dispatch wrapper.

The first mixed attempt exposed a probe reference error: split reduction omitted
`f.canon`, while dataflow correctly used canonical reduction. The corrected probe
passes `f.canon`; canonical split and dataflow now have identical output hashes.
The failed run is preserved in `results/20261020/q3-nt2-mixed-service/` and is not
included in the table. Corrected logs, return codes, source/object/binary hashes
and paired statistics are in `results/20261020/q3-nt2-mixed-service-v2/`.
Only CPU experiment code changed in this follow-up; no GPU execution occurred.


### Optional two-token Q3 compiler dispatch (2026-10-20)

The mixed-job experiment now has a production dispatch path. Configure with
`-DSTRATA_Q3_CLANG_NT2=ON` and launch with `STRATA_Q3_CLANG_NT2=1` to select the
Clang-compiled original Q3_K down rows for exactly two-token expert jobs. Build
and runtime defaults are off; runtime value `2` is off. The existing four-token
option remains independent. Both options share one extra compilation of the
unchanged kernel arithmetic in `cmake/q3_clang_nt4.cmake`; neither selects Q2,
expanded Q3, gate/up rows or single-token jobs. Native Linux x86-64 GNU host
builds with Clang and native expert support are required.

The evidence for this option is the preceding synthetic mixed-route experiment,
not a full-model decode measurement. It does not improve greedy one-token jobs.
The source compiled with both build options disabled is byte-identical to the
saved pre-dispatch object (SHA256
`abea083f4be5ba6a16a8c42a1be7c1845ab22359e6eff2e72b2834b4507336e6`).
Integration validation is recorded in `results/20261020/q3-nt2-integration/`.


Both HIP- and CUDA-configured full decoders built successfully with the new
option, and both executable usage smoke checks passed. Each configuration passed
five CPU tests with NT2 enabled: Q23 rows, Q23 pool, dataflow pool, expanded Q3,
and canonical expert arithmetic. The Q23 row test covers widths 1–4 and also
passed separately with runtime values 0 and 2 on both builds. A production-archive
mixed-job smoke passed 60 timed bitwise checks across values 0/1/2 and ordinary/
canonical arithmetic; hashes matched across runtime settings and the preceding
scratch-dispatch experiment. Those timings overlapped builds and are not used
as performance evidence. No GPU execution, full-model decoding or SYCL validation
ran. Both build options remain off by default; only local experimental build
trees were configured on. The 90%-of-peak objective remains unverified.


### FP4 unpack and magnitude instruction screens (2026-10-20)

Two isolated GCC AVX2 candidates were screened beside the production CPU
archive on the Threadripper 1950X. The first combines low/high nibble indices
into a 256-bit register, performs a lane-local byte-table shuffle, then permutes
64-bit lanes into the original GGML order. It also preserves the explicit K32
order when requested. An independent scalar checker passed 8,192 cases covering
every byte value at every source position in both layouts. Fewer source-level
operations did not make it faster.

The second retains the original unpack and uses `abs_epi8(v)` instead of
`sign_epi8(v,v)` for the signed FP4 magnitudes. Activation sign transfer and the
FMA/reduction order are unchanged. Both candidates were compiled separately
with symbol-renamed exports; neither changed production code or defaults.

The existing paired GU probe tested K=4096, R=128, raw scale/code bytes and signed
Q8 bytes including -128, clamp=7 and global scales 0.001. Cores 3 and 11 were
pinned separately, using one or 64 weight banks and 12 alternating pairs of
64 calls each. Prefetch, narrow/offset dots and the NT4 Clang selector were off.
These are median candidate/baseline process-CPU-time ratio ranges across the
four core/bank configurations:

| Candidate | One token | Two tokens | Four tokens |
| --- | ---: | ---: | ---: |
| Wide unpack | 1.077–1.099 | 1.056–1.073 | 1.020–1.087 |
| Byte absolute magnitude | 1.017–1.026 | 1.011–1.025 | 0.987–0.992 |

All 576 timed batches passed bitwise output checks, in addition to initial
per-bank parity. Wide unpack won only 1/48 single-token pairs; byte absolute
magnitude won 5/48. Reject both for single-token decode. The small four-token
magnitude result (44/48 wins) has no complete-expert measurement and does not
justify production adoption. Host load was uncontrolled, and no hardware-counter
attribution, full-model throughput, GPU execution or peak-utilization result is
claimed. No backend production build is needed for these rejected scratch-only
candidates. Logs, source/binary hashes and compact diffs are in
`results/20261020/fp4-wide-unpack/` and `results/20261020/fp4-pabs/`.


### Balanced worker counts and pool sleep under load (2026-10-20)

`glm_fp4_native_pool_bench` accepts an optional final token count (1, 2 or 4)
after its layout argument. Omission retains the existing 1/4 sweep. This lets
single-token decode service be screened separately without changing kernels or
engine defaults. Both HIP- and CUDA-configured CPU benchmarks built. The CUDA
benchmark passed 108 timed checks and 12 final CPU-batch checks at width two;
explicit widths zero and three were rejected before loading weights.

The Threadripper 1950X worker-count screen used real layers 3/4, experts 0/1,
one token, eight banks, four tasks per participant, and disabled software
prefetch, narrow/offset dots, NT4 Clang and automatic small-expert unfusing.
Each process selected only physical cores, equally split across both NUMA nodes:
4 participants used cores 0–1/8–9, 8 used 0–3/8–11, 12 used 0–5/8–13,
and 16 used 0–15. Participant count includes the host. This avoids the earlier
seven-worker case that selected cores from only one node. Order was
16/8/4/12/12/4/8/16, with all other settings held fixed.

| Participants | Owned, unfused dataflow median-ms range across layers/runs |
| --- | ---: |
| 4 | 3.101–3.293 |
| 8 | 1.656–4.133 |
| 12 | 1.532–3.959 |
| 16 | 1.773–4.953 |

These ranges do not support a stable worker-count recommendation. All 864 timed
checks and 96 CPU-batch checks passed, and output hashes matched across counts.
A two-second host-level sample after all sweep processes finished found 78.08%
aggregate CPU busy, load averages 24.59/16.70/14.53 and CPU pressure some avg10
13.36. No background process was stopped. This is not an idle-system measurement
or a peak-bandwidth calibration. Evidence is in
`results/20261020/fp4-balanced-workers/`.

A follow-up kept 16 participants and compared the existing
`STRATA_POOL_SPIN_US=20000` (default window) against `0` in 20000/0/0/20000 order.
Owned dataflow, including both hidden-quantization modes and both layers, used
31.51–46.23 process CPU-ms per call with the default window versus 14.89–28.99
with immediate sleep. CPU time is summed across threads and includes pool work;
it is not elapsed token time or a hardware-utilization percentage. Wall-time
medians ranged 0.887–4.654 ms versus 1.122–4.395 ms, so latency improvement is
not established. All 432 timed and 48 CPU-batch checks passed with matching
hashes. This supports retaining immediate sleep as an available CPU-cost option
under load, not changing the default or claiming faster full-model decode.
Raw logs and summaries are in `results/20261020/fp4-pool-spin/`.

No GPU execution or production scheduling change ran. The 90%-theoretical target
and full-model throughput remain unverified; high apparent CPU activity here
must not be counted as useful FP4 compute utilization.


### Q2 scale-table construction and paired half conversion (2026-10-20)

Two isolated GCC candidates targeted Q2 scale work. A table-construction
candidate replaces the LUT's interleaves with byte-to-dword expansion and
multiplication by `0x03020100`: for a four-bit scale s this constructs exactly
`{0,s,2s,3s}` without carries. The paired row screen on Threadripper 1950X,
cores 3/11, one/64 banks, showed single-token CPU-time ratios 1.000–1.015 and
only 18/48 pair wins. Other widths were mixed. All 288 timed batches passed
bitwise parity, but this does not support adopting the table change. Evidence
is in `results/20261020/q2-table-multiply/`.

The second candidate converts the adjacent Q2 block scale and minimum scale
from FP16 together using one F16C conversion, then extracts the two floats.
The original performs the two conversions separately. Weight bytes, integer
operations, FMA sequence and reduction order stay unchanged. A standalone check
covers all 65,536 FP16 bit patterns in both positions, paired by xor `0xa55a`,
and matches separate baseline conversions bit-for-bit, including signed zeros,
subnormals, infinities and NaNs. This proves the conversion substitution for
those paired inputs, not arbitrary invalid-weight behavior of the whole kernel.

Twelve alternating paired batches of 512 row calls per case used K=4096, R=128,
cores 3/11, one/64 banks and prefetch off. The median candidate/baseline process
CPU-time ranges were:

| Q2 path | One token | Two tokens |
| --- | ---: | ---: |
| Direct (LUT off) | 0.914–0.936 | 0.969–0.987 |
| LUT on | 0.974–1.010 | 0.972–0.985 |

All 384 timed batches passed bitwise parity. Direct single-token candidate rows
won 43/48 pairs. The row results and conversion check are in
`results/20261020/q2-half-pair/`.

A scratch dispatcher selected this candidate only for Q2 gate/up jobs of width
one or two; Q3 down and expanded paths retained the GCC baseline. Complete
single-token expert service used synthetic H=4096/F=2048, two experts, eight
shared NUMA banks, 15 workers plus host, LUT off and prefetch off. The median
paired candidate/baseline wall-time ratios over 20 pairs per case were:

| Service | First run | Repeat |
| --- | ---: | ---: |
| Ordinary split | 0.944 | 0.983 |
| Ordinary dataflow | 0.954 | 0.947 |
| Canonical split | 0.857 | 0.935 |
| Canonical dataflow | 0.939 | 0.959 |

All 320 timed expert-output/routed-sum checks passed bit-for-bit; candidate won
116/160 pairs. Dataflow improves roughly 4–6% in these two runs. Host load was
uncontrolled; this remains synthetic partial service, with input quantization
excluded and repeatedly reused weight banks. It is evidence for further opt-in
integration testing, not a full-model throughput or 90%-peak result. No
production kernel/default changed, and no GPU execution ran. Full backend
integration builds remain outstanding for this scratch candidate. Service logs,
source/object/binary hashes and paired summaries are in
`results/20261020/q2-half-pair-service/`.


### Build-time opt-in paired Q2 half conversion (2026-10-20)

Configure with `-DSTRATA_Q2_HALF_PAIR_NT1=ON` to use paired FP16 conversion in
direct single-token AVX2 Q2 rows and gate/up processing. This is a build-only
option, off by default; it adds no per-row runtime selector. The existing AVX2
Q23 selection must already be active (for example `STRATA_Q23_AVX2=1`), and
`STRATA_Q2K_LUT=0` selects the measured direct path. The LUT path, wider-token
jobs and Q3 arithmetic remain unchanged. Native expert support is required.
Only the local experimental build trees were configured on.

The integrated helper is source-identical to the preceding candidate that passed
all FP16 bit patterns. Scope is deliberately limited to direct NT1, whose paired
row and complete-service measurements support the option. The earlier two-token
and LUT candidate measurements are not a recommendation to extend this switch.
With all experimental Q23 build options disabled, the CPU object is byte-identical
to the saved pre-dispatch baseline, SHA256
`abea083f4be5ba6a16a8c42a1be7c1845ab22359e6eff2e72b2834b4507336e6`.

Both HIP- and CUDA-configured full decoders built and passed executable usage
smokes. Each passed five CPU tests with LUT off: Q23 rows, Q23 pool, dataflow pool,
expanded Q3 and canonical expert arithmetic. Q23 rows and pool also passed with
LUT on in each build. The row tests compare widths 1–4, so the new NT1 conversion
is checked against the original wider-token arithmetic. An integrated CPU-archive
service probe passed 40 timed checks across split/dataflow and ordinary/canonical
modes; all output hashes match the frozen scratch comparison's baseline. These
integration timings overlapped builds and are not new performance evidence.

No GPU execution, full-model decode or SYCL validation ran. The preceding 4–6%
dataflow reduction is synthetic partial-service evidence on Threadripper 1950X;
it is not an end-to-end rate, a worker-host recommendation, or proof of the 90%
peak target. Source/binary hashes and validation summaries are in
`results/20261020/q2-half-integration/`.


### Paired Q2 half conversion on the worker (2026-10-20)

The Ryzen 9 3950X worker was tested with temporary static CPU probes, with
binary hashes verified after copying. Its initial load averages were
30.91/31.11/30.86 and available RAM was 59,127,972 KiB. Miner and persistent
runtime settings were unchanged. No model weights were transferred and no GPU
ran; these probes use small synthetic Q2/Q3 expert buffers.

The row probe linked the frozen original GCC object and paired-half candidate
side by side, avoiding the newly enabled local build option in the baseline.
With direct Q2 (LUT off), NT1, K=4096/R=128, prefetch off and 12 alternating
pairs of 512 calls, median candidate/baseline process CPU-time ratios were:

| Pinned CPU | One bank | 64 banks |
| --- | ---: | ---: |
| 3 | 0.927 | 0.945 |
| 11 | 0.931 | 0.945 |

All 48 pairs favored the candidate and all 96 timed batches passed bitwise
parity. This reproduces the row-kernel benefit on the worker CPU.

Complete synthetic service used two experts, H=4096/F=2048, eight banks,
15 workers plus host, complete blobs on this single-NUMA machine, and the
scratch dispatcher selecting paired half conversion only for Q2 gate/up at
width one or two. Q3 down remained original GCC. Twenty alternating NT1 pairs
per case gave these median paired wall-time ratios:

| Service | First run | Repeat |
| --- | ---: | ---: |
| Ordinary split | 1.110 | 0.844 |
| Ordinary dataflow | 0.978 | 1.066 |
| Canonical split | 0.953 | 1.113 |
| Canonical dataflow | 0.921 | 0.840 |

The candidate won only 82/160 NT1 service pairs. NT4 identical-code controls gave
1.011 split and 1.032 dataflow, with 16/40 pair wins. All 400 service checks
passed; output hashes also match the primary-host synthetic baseline. These
mixed wall-time results do not establish a worker service-speed recommendation,
even though row CPU time improves. Keep the worker's build/runtime configuration
unchanged. Input quantization is excluded, weight banks are repeatedly reused,
and host load is uncontrolled. No full-model throughput or peak-utilization
claim follows. Logs, binary/source hashes and paired summaries are under
`results/20261020/worker-q2-half-pair/`.


### Batched Q2 expert timing and CPU cost (2026-10-20)

The paired expert probe accepts `[dense|mixed [CALLS]]`, with CALLS 1–64 and
one call by default. Existing omitted-pattern and `mixed` invocations still
work. For CALLS>1 it times a batch and reports elapsed and process CPU ms per
call; split phase totals are divided by the same count. CPU time includes all
process threads and worker spinning, not just useful dot products. This is
batch-average service, not a distribution of individual token latency. Final
outputs are checked once after each batch, outside timing; intermediate outputs
are not individually checked. Each call advances the shared weight-bank ring.

Using the same frozen GCC baseline and Q2-only gate/up paired-half dispatcher,
16 calls per batch and 20 alternating pairs per case gave these median ratios.
Geometry remains two synthetic experts, H=4096/F=2048, eight banks, 15 workers
plus host, direct Q2 and prefetch off. The primary uses NUMA shards; the single-
NUMA Ryzen 3950X worker uses complete blobs. NT4 is an identical-code control.

| Host / mode | Wall-time ratio | Process CPU-time ratio |
| --- | ---: | ---: |
| Primary ordinary NT1 split | 0.968 | 0.997 |
| Primary ordinary NT1 flow | 0.943 | 0.994 |
| Primary canonical NT1 split | 1.003 | 1.003 |
| Primary canonical NT1 flow | 0.948 | 0.996 |
| Primary NT4 split control | 1.010 | 1.002 |
| Primary NT4 flow control | 1.003 | 1.000 |
| Worker ordinary NT1 flow, first / repeat | 0.894 / 0.973 | 0.891 / 0.984 |
| Worker canonical NT1 flow, first / repeat | 0.748 / 0.954 | 0.763 / 0.986 |
| Worker NT4 flow control, first / repeat | 0.987 / 0.971 | 0.994 / 1.009 |

The primary's 5–6% dataflow elapsed-time reduction is consistent with the earlier
partial-service screen, while its process CPU cost barely changes. Do not equate
the elapsed-time reduction with an equal reduction in total CPU work. The worker's
large first-run improvements were not reproduced: its ordinary repeat changed
about as much as the identical-code control. Keep the worker recommendation
inconclusive, despite its previously stable row-kernel CPU-time reduction.
Background load remained uncontrolled; no miner or persistent settings changed.

Across both hosts, 600 final-batch parity checks passed after 9,600 timed service
calls. Primary CLI smokes passed 80 additional single-call checks for the legacy
default and mixed patterns; batch counts zero and 65 were rejected. The static
CPU probe was built locally and its copied hash verified before remote execution.
Only benchmark code changed, with no new GPU/backend kernel change or full-model
run. Logs, snapshots, hashes and separate wall/CPU statistics are in
`results/20261020/q2-half-batched-primary/` and
`results/20261020/q2-half-batched-worker/`. No theoretical peak or full-model
throughput claim follows from these batched service results.


### Real FP4 scale distribution and normal-scale decode (2026-10-20)

A CPU-only histogram decoded the partial checkpoint's lossless scale streams
for layers 3/4, experts 0/1, and gate/up/down matrices. All 6,291,456 scale bytes
in these 12 matrices lie in the normal finite E4M3 range 8–126; all 1,572,864
four-scale groups are normal. The native GGML converter copies each group's
four scale bytes unchanged. This is a property of the sampled matrices, not
proof about every expert or the whole checkpoint. The histogram, source and
fixture hash are in `results/20261020/fp4-normal-scales/`.

The reusable paired GU probe now accepts an optional `raw|normal` scale pattern.
Default raw retains the prior random-byte sequence; normal generates bytes
8–126. Inputs still include arbitrary FP4 codes and signed Q8 bytes including
-128. A guarded candidate recognizes four nonzero-exponent scale bytes with
no exact 0x7f sentinel, constructs their FP32 bits using integer exponent
arithmetic, and falls back to the existing table otherwise. The high-bit
behavior is retained, including the distinct 0x7f and 0xff cases.

On Threadripper 1950X, pinned cores 3/11, one/64 banks, 12 alternating paired
batches of 64 GU calls, K=4096/R=128, prefetch/narrow/offset/Clang options off,
median candidate/baseline process CPU-time ratios ranged:

| Guarded candidate input | One token | Two tokens |
| --- | ---: | ---: |
| Normal scales | 1.102–1.134 | 1.030–1.064 |
| Raw scale bytes | 1.259–1.319 | 1.127–1.161 |

All 384 timed batches and initial per-bank checks matched baseline bits. The
candidate is rejected; ordinary-case specialization did not recover the earlier
arithmetic scale decoder's loss.

A second scratch-only candidate removed the guard, fallback and high-bit mask.
Its harness requires normal inputs and rejects raw mode before allocating or
calling the kernel; this is not a general-compatible kernel API. Even with
those restrictions, ratios were 1.039–1.104 at one token and 1.024–1.062 at two.
All 192 timed batches matched, but the single-token candidate won only 7/48
pairs. Removing the guard did not make arithmetic decode faster. The data do
not isolate an instruction-level cause; host load was uncontrolled. This
candidate is also rejected, with evidence in `results/20261020/fp4-normal-only/`.

Keep the production scalar scale-table implementation. No production kernel,
default, GPU path, complete-expert service or full-model throughput changed.
No theoretical utilization or universal scale-distribution claim follows from
this partial histogram and synthetic kernel screen.


### FP4 split-accumulator numerical screen (2026-10-20)

An isolated candidate separates the first and second 32-value halves of each
K64 block into two FP32 accumulators for NT1, then adds the accumulators before
the existing horizontal sum. This changes floating-point reduction order and
is not bit-identical. NT2 retains the original loop as an identical-arithmetic
control. Weight decoding, Q8 sign behavior, scale-table loads and post-dot
clamp/SiLU arithmetic remain unchanged. Production source/defaults were not
modified.

A scratch paired GU harness measures output differences instead of requiring
bit equality. Its `1e-4 + 1e-4*abs(reference)` bound is only a screen guard,
not a model-quality criterion. Maximum absolute and scaled errors are tracked
outside timing against the original kernel; the scaled metric divides by
`max(1,abs(reference))`. Initial per-bank outputs and the final output of each
timed batch are checked. Counts include baseline checks and should not be read
as a candidate-only changed-value fraction.

On Threadripper 1950X, cores 3/11, one/64 banks, K=4096/R=128, global scales
0.001, clamp=7 and 12 alternating pairs of 64 calls, single-token median CPU-time
ratios were 0.997–1.282 with normal scale bytes and 1.000–1.006 with raw bytes.
The large normal/core-3/64-bank outlier is retained. Across all eight NT1 cases,
only 41/96 pairs favored the candidate. Two-token controls ranged 0.992–1.029.
Prefetch, narrow/offset paths and the NT4 Clang selector were disabled; host load
was uncontrolled.

All 384 timed batches passed the stated numerical screen guard. NT1 outputs
changed, with maximum observed absolute difference 2.67029e-5; NT2 outputs were
bit-identical. This is not an independent FP64 error bound or a full-model
quality measurement. The candidate is rejected because it provides no consistent
speed benefit to justify further quality qualification. The experiment does
not prove a specific hardware bottleneck, but splitting this accumulation chain
did not help in the measured screen. No complete-expert or GPU execution ran.
Logs, source/binary hashes, probe snapshot and the rejected diff are under
`results/20261020/fp4-split-acc/`.


### Predecoded signed-byte FP4 cache experiment (2026-10-20)

A scratch-only cache representation replaces each K64 block's 32 packed code
bytes with 64 signed doubled-E2M1 bytes in element order, retaining the four
scale bytes. It uses 68 rather than 36 bytes per block (1.889x). The candidate
loads decoded bytes directly and retains the original sign transfer, integer
dots, scale-table lookup, FMA sequence and reduction order. It accepts only its
private expanded buffers; this is not a new checkpoint or native loader format.
Narrow/offset paths are disabled and rejected by the probe guard. No production
format, kernel or default changed.

On Threadripper 1950X, pinned cores 3/11, one/64 banks, K=4096/R=128, raw scale
bytes and Q8 inputs including -128, the paired GU screen found median process
CPU-time ratios 0.806–0.891 for NT1, 0.850–0.922 for NT2 and 0.891–0.931 for
NT4. All 288 timed batches and initial per-bank checks matched baseline bits;
NT1 won all 48 pairs. Expansion is outside kernel timing. One-time conversion
CPU costs are recorded separately; illustrative reuse estimates exclude
allocation and cache management and are not service predictions. The original
and expanded buffers coexist in the benchmark. Evidence is in
`results/20261020/fp4-byte-cache/`.

A complete-service prototype then loaded real partial layers 3/4, experts 0/1,
with H=4096/F=2048. Both gate/up and down use the selected representation;
activation/hidden quantization, global scales, SiLU and routed reduction retain
the original operations. Explicit scratch format fields track the larger row
strides and global-scale-tail offset. Both variants have NUMA-owned row shards
and full blobs; 15 workers plus host process NT1, with hidden fusion off. Each
variant rotates through its own one or four weight banks. Twenty alternating
pairs time four service calls per batch and check final outputs bit-for-bit.
Input quantization, expansion and allocation are excluded from service timing.

| Banks | First split ratios, layers 3/4 | Repeat split | First flow | Repeat flow |
| --- | --- | --- | --- | --- |
| 1 | 1.154 / 1.243 | 1.184 / 1.263 | 1.216 / 1.085 | 1.163 / 1.326 |
| 4 | 1.227 / 1.287 | 1.284 / 1.177 | 1.300 / 1.333 | 1.283 / 1.195 |

The two experts' representation grows from 28,311,584 to 53,477,408 bytes before
bank replication and NUMA copies. When replacing a representation this adds
12 MiB per expert at this geometry; a cache retaining the original also needs
the entire expanded allocation. The service candidate won only 32/320 pairs.
All 640 final-batch checks passed after 2,560 timed calls, with matching hashes
across variants and runs. Host load was uncontrolled, but all 16 case medians
regressed despite the isolated row gains.

Reject this cache representation for the measured CPU expert service. The
extra footprint/traffic is a plausible explanation for the reversal, not a
hardware-counter attribution. Faster unpack-free rows do not establish a faster
expert or model. Conversion optimization and a production cache policy were not
pursued after this service result. No GPU, full-model throughput or peak
utilization test ran. Service logs, snapshots and hashes are in
`results/20261020/fp4-byte-cache-service/`.


### Batched native FP4/K32 service (2026-10-20)

The real partial-weight pool benchmark now accepts a final CALLS argument (1–64)
after TOKENS. It times CALLS consecutive service invocations per latency sample
and reports milliseconds per call; omission preserves one call. The existing
32-call process-CPU batch remains separate. Final outputs are checked after
each batch, not after every intermediate invocation. Split phase deltas are
divided by CALLS before averaging. `ordered_samples` preserves the nine samples
in round order alongside the existing sorted `samples`, enabling corresponding
rounds to be paired across layouts instead of pairing sorted quantiles.

Both HIP- and CUDA-configured CPU benchmark targets built. CUDA smokes of the
legacy default and CALLS=2 passed 432 latency-batch checks and 48 CPU-batch
checks, with identical output hashes; CALLS zero and 65 were rejected. In the
main screen, split component means matched the mean batch-average latency to
printed precision, with maximum discrepancy 8.21e-6 ms.

A K32 reassessment used Threadripper 1950X, real layers 3/4, experts 0/1, NT1,
15 workers plus host, three tasks per participant and 16-call latency batches.
Software prefetch, narrow/offset paths, NT4 Clang and automatic small-expert
unfusing were off. One/four/four/one weight-bank runs retained the same bytes per
expert for original and K32 layouts. Each pair is adjacent, with order reversed
by round. Median K32/original paired batch-average ratios for owned dataflow were:

| Banks / run | Layer 3 unfused | Layer 4 unfused | Layer 3 fused | Layer 4 fused |
| --- | ---: | ---: | ---: | ---: |
| 1, first | 0.980 | 1.048 | 0.980 | 0.974 |
| 4, first | 0.972 | 1.031 | 0.927 | 1.020 |
| 4, repeat | 0.995 | 1.058 | 0.927 | 0.907 |
| 1, repeat | 1.000 | 1.014 | 1.001 | 0.957 |

All 864 latency-batch checks passed after 13,824 timed service calls; another
96 final CPU-batch checks passed. Hashes matched across layouts, modes and runs.
Some fused cases improve, but ordinary dataflow does not establish a consistent
service gain. Keep the existing explicit K32 option and unchanged defaults;
these results do not justify selecting it generally. Conversion, allocation and
input quantization are excluded, and host load was uncontrolled. No GPU or
full-model run occurred. Logs, chronological paired statistics, source/binary
hashes and normalization checks are in `results/20261020/fp4-k32-batched/`.

### FP4 half-wise gate/up pairing screen (2026-10-20)

A scratch AVX2 candidate processes gate and up together for each K32 half of a
K64 block. It reuses that half's Q8 activation vector and activation scale while
retaining the original packed weights and each output's FMA order. It applies
only to one-token ordinary activation handling; down and wider token paths are
unchanged. This differs from the earlier rejected full-K64 gate/up pair. Reduced
register pressure is a design hypothesis, not an assembly-verified explanation.
No production kernel, layout, default or simulator rate changed.

The primary Threadripper 1950X row screen used CPUs 3 and 11, one and 64
independently populated weight banks, raw scale bytes, and 12 alternating pairs
of 64 calls. One-token candidate/baseline median process-CPU ratios were
0.902–0.959; unchanged two-token controls were 0.994–1.003. All 192 final batch
checks matched bit for bit. Source, patch, hashes and logs are in
`results/20261020/fp4-gu-half-pair/`.

A real partial-weight service screen then used layers 3/4, experts 0/1, H4096,
F2048, one token, 15 workers plus host, three tasks per thread, and bank counts
1/4/4/1. It tested complete-blob and NUMA-owned split/dataflow paths, including
fused hidden quantization, with nine alternating pairs of eight-call batches.
All 864 latency-batch output/reduction checks after 6,912 timed calls and 96
separate CPU-batch checks passed. The candidate won 268 of 432 latency pairs.
NUMA-owned dataflow median paired ratios ranged 0.740–1.114 without hidden
quantization fusion and 0.774–1.043 with it. Repeated cases did not establish a
consistent complete-expert gain. CPU batch time includes pool spinning; host
load was uncontrolled. Source snapshots, settings, hashes, chronological samples
and summaries are in `results/20261020/fp4-gu-half-pair-service/`.

The Ryzen 9 3950X middle worker ran the same single-core row screen with its
miner left running. One-token CPU ratios were 0.836–0.930, winning all 48 pairs;
unchanged two-token controls were 0.976–0.990, indicating some control drift.
All 192 checks matched bit for bit. The static executable's checksum matched
before execution. Evidence is in `results/20261020/worker-fp4-gu-half-pair/`.
This warrants a worker complete-expert service test, not production enablement.
No GPU execution, full-model quality/throughput measurement, or proof of 90%
theoretical utilization was obtained.

### Worker complete-expert gate/up pairing (2026-10-20)

The middle Ryzen 9 3950X worker tested the half-wise gate/up candidate through
complete real partial experts (layers 3/4, experts 0/1, H4096, F2048). This used
15 workers plus host, three tasks per thread, complete-blob split and dataflow
paths on its single NUMA node, and unchanged down/hidden quantization. The miner
remained running; the pre-test load average was 30.38/30.31/30.46. No GPU was
used. Static binary checksums were verified before each experiment.

The first four runs used banks 1/4/4/1 and nine alternating pairs of eight-call
batches. All eight four-bank case medians improved: candidate/baseline ratios
were 0.924–0.974 for split and 0.931–0.968 for dataflow. Across all bank counts,
ratios were 0.905–1.022 and 0.883–1.012 respectively. All 288 final latency-batch
output/reduction checks after 2,304 timed calls, plus 32 separate CPU-batch
checks, passed bit for bit. A separate CPU-batch ratio reached 16.669 and is
preserved; this uncontrolled result is not a reliable CPU-cost improvement.

Two repeats with four banks increased latency batches to 16 calls and added
an unchanged two-token dispatch control. The one-token dataflow ratios were
0.975/0.951 and 0.911/0.967 for layers 3/4; split ratios were 0.955/1.001 and
0.931/0.913. Two-token dataflow controls ranged 0.965–1.065 and split controls
0.966–1.194. The direction of the one-token result is promising, but the control
variation limits confidence in the smaller gains. All 288 final latency-batch
checks after 4,608 timed calls and 32 separate CPU-batch checks passed. These
checks compare final expert outputs and routed sums, not every intermediate
call. Inputs and expert selections repeat; this is not a full-model routing
replay or an end-to-end throughput measurement.

Evidence, source hashes, the longer-batch source snapshot and raw outliers are
in `results/20261020/worker-fp4-gu-half-pair-service/`. Production kernels and
simulator rates remain unchanged. The next implementation gate is an explicit
opt-in with default-object identity and backend build checks, followed by
broader numerical coverage and repeated service measurements. No full-model
tok/s improvement or 90%-of-peak utilization is established.

### Opt-in FP4 gate/up half pairing integration (2026-10-20)

`STRATA_NVFP4_GU_HALF_PAIR_NT1` is now a CMake option, default OFF. It requires
native experts and pairs ordinary AVX2 one-token gate/up halves, including the
clamped/separate-up path. It has no runtime environment switch. Packed K32,
narrow/offset activation variants, down projection and other token widths keep
their existing implementations. The separately compiled Clang four-token object
also retains its existing compilation. Enable explicitly in an existing build:

```sh
cmake -S . -B BUILD_DIR -DSTRATA_NVFP4_GU_HALF_PAIR_NT1=ON
cmake --build BUILD_DIR --target strata-glm-decode -j 8
```

With this option disabled, the GCC O3 AVX2/FMA/F16C FP4 object is byte-identical
to the saved default object, SHA-256
`53ec0c87aa7c9de1d656ed3a9678c0e3c122353357d0c7e58511e080d2fc9f54`.
The local HIP and CUDA build trees were explicitly configured ON. Both complete
decoders built and reached their usage output. Five CPU tests per backend passed
(native ordinary/offset/narrow/combined variants plus packed K32 dot parity).
The native test now checks GU against GGML dots at four K sizes, all FP4 codes,
Q8 extremes including -128, seven finite half scales, distinct reversed up rows,
clamped/unclamped operation, both up-storage forms and nonzero partial-row bounds.
Eight real partial-weight test invocations (four variants per backend) also
passed, including clamp-boundary and separate-shard checks.

A probe linked against the integrated HIP CPU archive and frozen GCC baseline
passed all 96 raw-scale paired batch checks. One-token row CPU ratios were
0.961/0.897 for one/64 banks; two-token controls were 0.995/0.952. The latter
variation reinforces the timing limitations under uncontrolled load. These are
row screens, not an additional complete-expert qualification or model throughput
claim. The prior worker complete-expert results motivate keeping this option
available for further testing, not changing the default.

Build/test logs, source snapshots, hashes and validation status are in
`results/20261020/fp4-half-integration/`. No GPU kernels were executed. SYCL and
Windows builds were not available or validated. Full-model quality, delivered
tok/s and 90%-of-peak CPU utilization remain unverified.

### Q2 gate/up block-pair screen (2026-10-20)

A scratch Q2 direct-dot candidate evaluates gate then up for each K256 block,
retaining separate FP32 accumulators and the original per-output FMA order.
This tests block-local activation reuse rather than completing an entire gate
row before starting up. Both candidate and baseline include the already
integrated NT1 paired-half conversion. Q3 down, Q2 LUT, multi-token kernels and
canonical gate skipping retain their existing paths. In particular, positive
canonical skip thresholds bypass pairing so a skipped gate does not trigger an
unnecessary up read. Actual load reuse/register-pressure effects have not been
verified in assembly. No production source or default was changed by this test.

On the Threadripper 1950X, synthetic complete Q2/Q3 experts used H4096/F2048,
two experts, eight banks, 15 workers plus host, three tasks per thread, and 20
alternating pairs of eight-call batches. One-token candidate/baseline paired
latency ratios were 0.964/0.952 for ordinary/canonical dataflow and 0.983/0.979
for split. Process CPU ratios were 0.9949–0.9995, including pool spinning.
Unchanged two-token controls were 1.030 dataflow and 0.996 split; positive-skip
controls were 1.007/1.035. All 320 final batch output/reduction comparisons
passed after 2,560 timed calls. These synthetic fixed routes are not real model
routing or an end-to-end tok/s result; host load was uncontrolled.

A separate raw-Q2 GU row screen used random packed codes/scales, fixed finite
half scales, valid Q8_K block sums with signed byte extremes, K4096/R128,
CPUs 3/11, and one/64 independent weight banks. Twelve alternating pairs of
64-call batches per case gave one-token CPU ratios 0.954–1.011, while unchanged
two-token controls were 0.994–1.003. All 192 final batch checks matched bit for
bit. The row screen uses the current production CPU archive as baseline;
its Q2 half-conversion option is enabled. The inconsistent row improvement
and modest service effect do not justify integration yet.

Sources, patch, hashes and logs are in `results/20261020/q2-gu-block-pair/`.
The first link attempt had a mistyped library path and the first row-probe
compile used unavailable anonymous-union field names; both were corrected
before successful builds and execution. CPU-only scratch GCC validation was
performed; no CUDA/HIP/SYCL candidate build or GPU execution was needed or
performed. No model throughput or theoretical-peak utilization claim follows.

### Combined Q2/Q3 options under mixed routes (2026-10-20)

A static paired service probe tested the integrated Q2 single-token paired-half
conversion together with Q3 Clang selection at two and four tokens, against the
frozen GCC baseline from before those options. The candidate was compiled from
the current production Q23 source with the three build definitions and linked
to the current Clang object; its ordinary entry points were renamed so both
variants share one process, pool, inputs and weight banks. The wrapper switches
Q2 GU and Q3 down together; this is not an additive prediction from separate
microbenchmarks. No new production changes or simulator-rate changes were made.

Both machines used synthetic H4096/F2048 Q2 GU/Q3 down experts, eight banks,
15 workers plus host, three tasks per thread, 20 alternating pairs of 16-call
batches, Q2 LUT off and canonical gate skipping off. The mixed case has seven
experts: two tokens share expert 0 and each uses three additional distinct
experts, so 25% of routed expert slots use NT2 and 75% use NT1. It is a fixed
reuse proxy, not a model routing replay. Input quantization is excluded.

On the Threadripper 1950X with NUMA-owned rows, candidate/baseline paired
latency ratios for mixed routes were 0.9324/0.9317 (ordinary/canonical), then
0.9280/0.9331 when repeated in reverse canonical order. Process-CPU ratios were
0.9274–0.9321, including pool spinning. The candidate won 77/80 mixed-route
pairs. Dense one-token ratios were 0.936/0.940; all-shared two-token ratios
0.848/0.861; all-shared four-token ratios 0.911/0.892. The dense multi-token
results are not representative of mostly unique expert routes.

The same binary on the Ryzen 9 3950X middle worker, with its miner still
running and complete blobs on one NUMA node, did not reproduce the mixed-route
gain: ratios were 1.004/1.019. Dense one-token ratios were 1.004/0.971 and
four-token ratios 0.956/0.972. These observations do not support enabling the
combination on that worker. The primary mixed-route result supports further
TR16 testing with the existing opt-ins, not a global default change.

All 640 final batch output/reduction checks after 10,240 timed calls passed
bit for bit across ten primary and six worker cases. The worker executable
checksum matched before execution. Timing was under uncontrolled host load;
this is CPU expert service, not GPU execution, full-model quality validation,
delivered tok/s or proof of 90% theoretical utilization. Evidence and source
hashes are in `results/20261020/q23-combined/` and
`results/20261020/worker-q23-combined/`.

### V100 study follow-up and excluded CPU setup (2026-10-20)

A read-only follow-up of the worker's `docs/GLM_FP4_V100_EXPERIMENTS.md`
clarifies two different qualifications. Its current ordered-KDA/prepared-input
preset reports 32.10 decode and 520.89 prefill tok/s. An earlier GPU-reduction
qualification reports 32.67/454.29, versus a same-build control of 32.43/395.97.
The study does not claim that sub-percent decode difference as an improvement
with unlocked clocks. These remain speculative route-drop profiles, distinct
from the preserved 5.26 tok/s W4A16 reference; they are not local measurements.

CPU input Q8 reuse looked promising in the short 128-output screen but lost in
full 512-output qualification: 32.22 versus 32.67 tok/s. It is not selected.
Our native decoder already prepares one quantized input per token before
building expert groups in both the ordinary non-asynchronous and pipelined
paths. Porting that rejected optimization into these paths would duplicate
existing reuse. The partial-expert timing still excludes this quantization and
routing setup; no zero-cost assumption should follow from its omission.

The earlier qualified V100 median used 15.6427 seconds for 511 advances, with
13.5103 seconds (86.4%) in speculative verification. Eliminating every other
phase would yield only 37.82 tok/s. At fixed non-verification time, reaching 50
would require verification to be about 1.67 times faster. Verification is not a
CPU-only timer, so these numbers cannot be used to assign its entire cost to
FP4 CPU kernels. The separate 117.816 GB/s read probe and 347.16 GB original-code
traffic proxy give a loose 2.947-second read bound; that excludes scales,
unpacking, activation traffic, arithmetic and scheduling. It is not a measured
FP4 bandwidth-utilization denominator or a promise of attainable throughput.

Source fingerprint, selected reported measurements and independently calculated
bounds are in `results/20261020/v100-study-update/comparison.json`. This rejects
input-reuse porting as the next optimization and keeps the priority on measured
expert service on the limiting worker plus actual decoder phase timing. No
runtime settings, production code or simulator constants changed in this audit.

### Worker FP4 Zen 2 compiler scheduling screen (2026-10-20)

A scratch build compared generic GCC scheduling with `-mtune=znver2` on the
Ryzen 9 3950X worker. Both objects were compiled from the current FP4 source
with the integrated GU half-pair option enabled, O3, AVX2/FMA/F16C, and no fast
math. All six exported FP4 entry points were renamed and selected by a paired
wrapper; the ISA and arithmetic source were identical. Unlike a no-change
control, both NT1 and NT2 are scheduling candidates here. Object text sizes were
167,421 and 167,741 bytes; this small difference does not explain performance.

The real partial-weight test used layers 3/4, experts 0/1, H4096/F2048, 15 workers
plus host, three tasks per thread, complete-blob split/dataflow, one and two
tokens, and bank counts 1/4/4/1. Each variant had nine alternating pairs of
16-call latency batches and a separate 32-call CPU batch. The miner remained
running and host load was uncontrolled. The static executable checksum matched
before execution. All 576 final latency-batch output/reduction checks after
9,216 timed calls, plus 64 separate CPU-batch checks, passed bit for bit.

For four banks, NT1 dataflow candidate/baseline ratios were 0.996/1.115 for
layers 3/4, then 1.001/1.058 on repeat. NT2 ratios were 1.009/1.016, then
1.044/1.015. One-bank results included gains but were inconsistent across
layers and repeats. Generic scheduling is retained: this screen does not
justify a worker-specific compiler option. No production source, CMake default,
worker configuration or simulator rate changed. No GPU execution or full-model
throughput claim is made. Sources, exact compile arguments, hashes and raw
paired samples are in `results/20261020/worker-fp4-zen2-tune/`.

### Routed reduction dispatch screen (2026-10-20)

The split expert path launches a separate pool phase for routed-output reduction.
A CPU-only scratch probe compared this with host-side reduction: ordinary mode
uses eight-lane AVX2 multiply then add with `-ffp-contract=off`, preserving expert
order; canonical mode calls the existing canonical routine. Production code and
defaults were unchanged. The dataflow completion path already performs reduction
without this separate phase and was not modified.

On the Threadripper 1950X, H4096, one token and two experts, the microprobe's
ordinary medians were 7.035 us pooled versus 1.727 us serial; canonical medians
were 5.210 versus 1.302 us. With one token/eight experts the differences narrowed
to 8.300/6.782 and 6.266/5.612 us. Larger geometries had mixed results and severe
pool outliers, all retained in raw evidence. The probe repeatedly reuses buffers,
so it does not measure post-expert cache ownership or a general serial threshold.
It passed 432 final batch bitwise checks over NT1/2/4, K2/8/32, ordinary/canonical,
and randomized signed outputs/weights including a zero weight. Shapes were all
H4096; this is not arbitrary-length SIMD-tail validation.

A separate scratch complete-expert benchmark then changed only split reduction,
using the same production kernels on both sides: current Q2 paired-half build,
Q3 compiler selection off, two synthetic H4096/F2048 experts, eight banks,
15 workers plus host, three tasks per thread, and 20 alternating pairs of
16-call batches. One-token ordinary serial/pooled latency ratios were 0.989 then
0.971 on repeat; canonical ratios were 0.977 then 1.004. Two-token ratios were
0.996/0.996 then 0.991/0.996 for ordinary/canonical. All 320 final batch expert
output/reduction comparisons passed after 5,120 timed calls. This modest split
benefit does not justify a general threshold or replacing fused dataflow.

Sources, hashes, medians and frozen raw logs are in
`results/20261020/routed-reduce-screen/`. Timing was under uncontrolled host load.
No GPU, full-model decode, worker execution or theoretical-peak validation was
performed. This identifies a few microseconds of optional split-path overhead,
not an explanation for the full-model throughput gap.

### FP4 user-space hardware-counter screen (2026-10-20)

The middle worker's `perf_event_paranoid=4` denied cycles/instructions/cache-miss
profiling. No privilege or kernel-policy change was attempted. The primary
Threadripper 1950X permits user-space counters, so a bounded screen ran there on
CPU 3. These counters must not be presented as measurements of the worker.

The probe compares the integrated GU half-pair CPU archive with the frozen GCC
baseline. Each process fixes one variant, checks every bank against the baseline,
then runs eight 1,024-call batches. Four processes use baseline/candidate/
candidate/baseline order at one and 64 banks, NT1, K4096/R128, raw scales and
independent packed banks. `perf stat` collects user cycles, instructions and the
generic cache-misses event. Events report 100% running time. Whole-process counts
include initialization, warmup, parity and reporting, while printed CPU times
cover the batches; neither is a hardware DRAM-byte measurement.

Median whole-process retired-instruction ratios were 0.909 (one bank) and 0.922
(64 banks), with approximately 669.6 million fewer instructions in both cases.
Cycle ratios were 1.003 and 1.144, however, and timed process-CPU ratios were
1.019 and 1.128. Thus the long rotating-bank screen did not reproduce the earlier
short-run timing benefit. Generic cache-miss counts decreased but do not prove
less DRAM traffic or identify the limiting execution resource.

Four additional unchanged-NT2 control processes at 64 banks gave an instruction
ratio of 0.99982, cycle ratio 0.950 and timed CPU ratio 0.960. That control drift
limits timing conclusions under uncontrolled load. The evidence supports fewer
retired instructions for the NT1 candidate, not consistently lower latency or a
90%-of-peak claim. Its existing build option remains experimental and default
OFF; no default or worker configuration was changed.

All 96 final batch comparisons across the twelve completed processes passed bit
for bit after 98,304 timed calls. Source/binary hashes, raw counters, timings,
control results and settings are in `results/20261020/fp4-counter-profile/`.
No GPU or full-model decode ran. Next kernel work should account for execution
dependencies and resource pressure rather than assuming that removing loads or
instructions alone improves throughput.

### FP4 sampled hot loop and scheduling model (2026-10-20)

A primary CPU3 `perf record` capture sampled user cycles every 100,000 events
while the integrated half-pair probe ran NT1 with 64 independent packed banks.
The process completed successfully with eight final batch parity checks and no
lost samples. The ordinary clamped GU function accounted for 84.28% of samples;
main and random-bank generation accounted for substantial remaining samples.
This is whole-process sampling, not a region-gated kernel counter measurement.

The selected hot loop contains no vector-register spills to the stack. It does
reload a scalar bound from the stack. Many samples appear near scale multiply/
FMA and following instructions, but sampling skid prevents assigning a stall
to the instruction printed at that address. This does not demonstrate that
scale memory or FMA latency is the unique bottleneck.

The emitted loop was extracted for LLVM MCA 21's `znver1` model. Because the
run used prefetch distance zero, the untaken prefetch instructions were removed
and branch labels normalized; the test and branch remain. For 100 modeled
iterations, MCA reported 97 instructions and 122 micro-ops per iteration,
dispatch width four, reciprocal throughput 30.5 cycles and 3,135 total cycles.
It reported no additional resource or data-dependency bottleneck. The throughput
bound equals 122/4, supporting instruction-dispatch reduction as a next
hypothesis, rather than register-spill repair or changed accumulation order.

These are static scheduling estimates, not measured micro-op retirement,
cache behavior, FLOPS/BW peak or 90% utilization. They do not model contention
from other workloads and do not establish the worker's limiting resource.
Compact sampled disassembly, normalized loop, model output and source/capture
hashes are in `results/20261020/fp4-hotspot-model/`; the full raw capture remains
in the ignored build tree. The first annotation lookup used a mangled symbol
and returned no matching samples; using the demangled symbol produced the
33,373-sample function annotation. The first model input retained a numeric
branch target and failed parsing; it was corrected without dropping executed
instructions. No production code, defaults or simulator rates changed.

### Lossless K32 plus half-wise gate/up pairing (2026-10-20)

A scratch candidate extends the integrated NT1 ordinary-activation GU half-pair
helper to the existing lossless K32 weight layout. The production helper remains
restricted to original GGML bytes. The comparison is original layout plus
half-pair versus K32 layout plus half-pair, including the existing K32 down path.
No scales, activation quantization or accumulation order change. Both layouts
use 28,311,584 bytes for the two tested complete experts; conversion and extra
benchmark reference storage are outside timing. This directly tests whether
removing unpack instructions helps the newer paired kernel, rather than
assuming independent gains add together.

A primary real-weight smoke run covered layers 3/4, experts 0/1, NT1,
complete-blob and NUMA-owned split/dataflow including fused hidden quantization.
All 216 latency-batch and 24 separate CPU-batch checks passed. Its short two-call
batches were correctness checks, not a performance qualification.

The Ryzen 9 3950X worker then used the same real experts, 15 workers plus host,
three tasks per thread, complete-blob split/dataflow, NT1, banks 1/4/4/1 and nine
alternating pairs of 16-call batches. The miner remained running. Four-bank
K32/original dataflow ratios for layers 3/4 were 0.975/0.963, then 1.018/0.939 on
repeat. Split ratios were 1.022/0.969, then 0.986/0.987. One-bank results were
also mixed, including a 1.102 dataflow ratio. Separate CPU-cost batches did not
consistently agree with latency. No stable general service improvement follows
under this uncontrolled load.

All 288 worker latency-batch comparisons after 4,608 timed calls, plus 32 CPU
batch checks, passed bit for bit. The executable checksum matched before the
remote run. Production dispatch, layout defaults and simulator rates remain
unchanged. Evidence, exact source patch, benchmark snapshot and hashes are in
`results/20261020/fp4-k32-half-pair/`. This was a CPU-only scratch GCC build; no
GPU, full-model quality/throughput or theoretical-peak validation ran.

### Paired scale-vector lookup rejection (2026-10-20)

A scratch NT1 GU variant replaced each pair of scalar scale lookups and vector
assembly with a 16-bit pair index and one aligned 256-bit load. Its table stores
all 65,536 byte pairs as eight floats (2 MiB), preserving the existing decoded
scale values and FMA order. Packed weights do not expand, but the process gains
that table and its cache footprint. Construction occurs at startup outside the
timed interval. The baseline is the integrated half-wise GU path, not the older
unpaired kernel. Other token widths and down projection are unchanged.

On primary CPU3, K4096/R128, twelve alternating pairs of 64-call batches with
one/64 independent weight banks gave candidate/baseline CPU ratios 1.279/1.391
for raw byte scales and 1.113/1.188 for normal codes 8–126. Unchanged NT2 controls
were 0.9991/1.0000. All 144 final batch checks matched bit for bit. The clear
regression across scale patterns rejects this table before complete-expert
service testing. Cache pressure is a plausible explanation, not a measured
cache-stall attribution. Saving lookup/assembly instructions alone was not
sufficient to improve this kernel.

The experiment changed no production source, defaults or simulator rates.
Sources, binary hashes, settings and raw results are in
`results/20261020/fp4-scale-pair-table/`. This was a GCC CPU-only scratch build
under uncontrolled load; no worker, GPU or full-model execution was performed.

### Current FP4 counter budget input (2026-10-20)

`cpu_peak_budget.py` now accepts `--counter-summary` and
`--weights-per-probe`. It reads either the historical dot-counter `results`
schema or the current GU `runs` schema; the supplied denominator is the timed
scalar weight-product count, including both GU matrices and every token.
The default historical projections are unchanged. Counter metadata is reported
with a repository-relative label when possible, avoiding private absolute paths.
Empty/nonpositive cycle data and nonpositive denominators are rejected.

For the current NT1 GU counter probe, 8,192 calls times 128 rows times two
matrices times K4096 gives 8,589,934,592 timed weight products. Whole-process
cycles still include setup and checks; the tool does not subtract those costs.
Under the explicitly unverified four-channel DDR4-3200 scenario, 90% remains
92.16 GB/s and requires 163.84 Gweight-products/s for the 36-byte K64 format.
At sixteen cores and assumed 3.4 GHz, this allows 0.332 cycles/product.

Current whole-process samples span 0.334–0.547 cycles/product. Their idealized
sixteen-core projections at 3.4 GHz span 55.97–91.74 packed GB/s, all below the
scenario's 92.16 target. These numbers are not measured DRAM throughput, do not
account for other memory traffic, and assume perfect core scaling without
contention. They neither prove a physical ceiling nor establish a percentage of
FP32 peak. RAM speed/channel count and sustained frequency remain unverified;
configuration clarification was requested, with no hardware setting changed.

```sh
python3 experiment/glm_fp4_local/cpu_peak_budget.py \
  --memory-mts 3200 --channels 4 --cores 16 --ghz 3.0 3.4 \
  --counter-summary experiment/glm_fp4_local/results/20261020/fp4-counter-profile/summary.json \
  --weights-per-probe 8589934592
```

Known bandwidth arithmetic, the new schema, unchanged historical projections
and rejection of a zero denominator were checked. Output and checks are in
`results/20261020/cpu-current-peak-budget/`. No engine kernel, runtime default,
simulator calibration or measured-model throughput claim changed.

### Native dataflow setup phase profile (2026-10-20)

A scratch copy of the pool adds host timestamps around the existing wait-before,
queue/counter setup, publish-through-completion and wait-after regions. It keeps
the same kernels and scheduler; both benchmark variant labels select identical
code. This is an instrumented diagnostic, not a candidate speedup comparison.
The production pool already reuses counter/task allocations; this profile asks
whether caching more queue construction is worthwhile.

On the Threadripper 1950X, synthetic Q2/Q3 experts used H4096/F2048, eight weight
banks, 15 workers plus host, three tasks per thread and NUMA-owned dataflow.
NT1/NT4 used two dense experts; NT2 used seven mixed-route experts. Across 681
profiled calls per case (including warmups), mean setup costs were 2.123, 2.182
and 1.849 us, respectively. They were 0.200%, 0.103% and 0.099% of the profiled
interval. Mean wait-before was 0.158–0.176 us and wait-after 0.331–0.423 us.
Publish-through-completion dominated, at 1,056/2,109/1,872 us; that interval
includes scheduling, worker synchronization and kernels, not just arithmetic.

All 120 timed-batch output/reduction comparisons passed after 1,920 timed calls.
The phase totals include 123 additional warmup/initial flow calls. Input
quantization, format validation and grouping before this function are outside
the timed regions. Timestamp overhead, first-call allocation and uncontrolled
host load remain in the results. These cases do not justify adding a task-plan
cache: eliminating the measured setup entirely would save only about 0.1–0.2%
of this interval, with no evidence of a full-model benefit.

Source snapshots, binary hashes, settings and phase totals are in
`results/20261020/pool-setup-profile/`. The first scratch compile lacked the
source-local affinity-header include directory and failed; adding that include
path fixed the build before execution. No production code/default, GPU execution,
worker run, full-model measurement or peak-utilization claim changed.

### Dataflow down-task readiness waits (2026-10-20)

A further scratch pool profile times the existing wait for an expert's GU and
hidden quantization to finish after a down task is claimed. The original queue,
release/acquire protocol and stall guard remain intact. Timing starts only when
the initial ready check fails; after readiness, relaxed atomic counters record
wait duration/count. This instrumentation adds overhead and is diagnostic only.
The accumulated durations overlap across workers and include descheduling; they
are neither CPU-busy time nor additive critical-path latency.

On the primary Threadripper 1950X with 15 workers plus host, eight banks and the
same synthetic Q2/Q3 geometries as the setup profile, each case covered 681 flow
calls. NT1/two experts averaged 8.26 readiness waits and 897 us summed worker
wait per call, within a 566 us publish-through-completion wall interval. Dividing
by sixteen participants times wall gives 9.90% of nominal participant-time.
NT2/seven mixed experts averaged 1.13 waits and 227 us over 1,875 us (0.756% by
that normalization); NT4/two experts averaged 4.29 waits and 230 us over 1,521 us
(0.944%). Those are separate instrumented runs under uncontrolled load; their
wall timings must not be compared with the preceding setup run as a speedup.

All 120 timed-batch output/reduction comparisons passed after 1,920 timed calls.
Readiness waiting merits further investigation specifically for the NT1 small
expert case. This does not yet justify a ready-task queue: the profile does not
show whether useful unclaimed work for another expert was ready during a wait.
The next scheduling gate is to establish that opportunity before changing task
claim order. Source/hash evidence and counters are in
`results/20261020/pool-ready-profile/`. No production scheduler/default, GPU,
worker, full-model execution or peak-utilization claim changed.

### Ready work during dataflow dependency waits (2026-10-20)

The next scratch profile scans only the worker's permitted queue when its down
task initially finds GU not ready. It looks for an unclaimed down task belonging
to another ready expert. Epoch tags must match, and the queue cursor must remain
unchanged across the scan; moving-cursor observations are discarded. Queue
contents are immutable until all tasks finish. The probe never claims or
reorders work and retains the original readiness wait and synchronization.
This establishes an entry-time opportunity, not how long that work stays ready
or how much latency a different scheduler would save.

Using the prior synthetic H4096/F2048 Q2/Q3 setup on the primary, NT1/two experts
found alternatives in only 19/3,072 waits (0.62%), then 28/4,890 (0.57%). Dense
NT4 found none in 1,091 waits. In contrast, NT2/seven mixed-route experts found
alternatives in 1,532/1,884 waits (81.3%), then 3,363/7,086 (47.5%). The mixed
case has one two-token shared expert and six one-token experts. That suggests
testing down-task ordering by job width inside each permitted node/group stage,
while leaving GU scheduling, ownership and canonical reduction order intact.
It does not yet justify a production queue change.

All five runs completed with 681 profiled calls each and 200 timed-batch
output/reduction checks after 3,200 timed calls. The mixed repeat's instrumented
execution interval rose from 2.717 to 9.915 ms per call under uncontrolled load;
retain that variation rather than treating these as stable performance numbers.
Scan overhead is included in waits, and summing waits across workers includes
overlap and descheduling. Associated wait duration is not removable critical-path
time. Raw counters, snapshots, source/binary hashes and summary are in
`results/20261020/pool-ready-opportunity/`. Production scheduling and defaults
remain unchanged; no worker/GPU/full-model throughput or peak-utilization claim
was made.

### Static down-task width ordering rejection (2026-10-20)

A scratch paired pool build changes only NUMA-owned down-task queue construction:
within each node and completion-group stage, lower job-token widths precede
higher widths. Gate/up order, task IDs, ownership, release/acquire readiness,
completion counters and per-chunk routed reduction retain their existing code.
The selector changes only between synchronous layer calls. No profiling timers
are present in this comparison. For dense same-width jobs the task order is
unchanged, providing controls for queue construction overhead and timing drift.

The primary Threadripper 1950X used synthetic H4096/F2048 Q2/Q3 experts, eight
banks, 15 workers plus host, three tasks per thread and 20 alternating pairs of
16-call dataflow batches. The mixed case has one shared NT2 expert and six NT1
experts. Candidate/baseline median latency ratios were 1.047/1.036 for ordinary/
canonical routing, then 1.010/1.074 on repeat. CPU-time ratios also regressed,
1.019–1.055, including pool spinning. Dense NT1/NT4 controls were 0.986/1.012.
Only 30 of 80 mixed-route pairs favored the candidate. All 240 final batch
expert-output/reduction comparisons after 3,840 timed calls passed bit for bit.

Static width ordering is rejected. The earlier existence of ready alternatives
does not imply that permanently putting smaller jobs first is better; delaying
the shared expert's larger down work may lengthen the tail, but this test does
not isolate that cause. Any future ready-aware claim policy must retain the
original fallback and prove epoch ownership, exactly-once task completion and
completion-group behavior. Two-group callbacks were not exercised here; the
canonical tests checked arithmetic/reduction, not that callback protocol.

Sources, patch, hashes, settings and raw paired results are in
`results/20261020/pool-down-width/`. Timing was under uncontrolled host load.
Production scheduling/defaults remain unchanged. No worker, GPU, full-model
throughput or peak-utilization claim follows from this CPU-only scratch test.


### Ready-aware down-task claiming (2026-10-20)

A scratch scheduler scans the unclaimed suffix of its own NUMA queue when the
next down task's expert is not ready. It swaps the first ready task into that
slot, otherwise retains FIFO order. A per-node lock serializes down claims;
gate/up claims retain the existing epoch-tagged CAS. This applies only to owned
native dataflow without completion groups. Kernels and production scheduling
remain unchanged.

On the Threadripper 1950X, synthetic H4096/F2048 Q2/Q3 experts used eight banks,
15 workers plus host, three tasks per thread and 20 alternating pairs of
16-call batches. Mixed NT2 routing has one shared NT2 expert and six NT1 experts.
The first ordinary/canonical comparisons gave median paired candidate/baseline
latency ratios 0.967/1.008, then 0.991/1.011. Dense NT1/NT4 controls gave
0.982/1.094. Some initial runs overlapped compilation of a separate audit binary;
retain them as screening evidence rather than a clean timing qualification.

Repeats after compilation finished gave 1.023 ordinary and 0.992 canonical,
with 9/20 and 11/20 candidate wins. Process CPU-time ratios were 1.028/0.991,
including worker spinning. External host load remained uncontrolled. Across
all eight performance runs, 320 final batch output/reduction comparisons after
5,120 timed calls passed bit for bit. There is no consistent performance gain,
so this locked scanning implementation is rejected for integration.

A separate instrumented binary counted claims for every task, aborted on any
duplicate, and checked for missing claims after all workers parked. Five cases
covered NT1 dense ordinary, NT2 mixed ordinary/canonical/canonical gate-skip,
and NT4 dense canonical. Each passed 100 candidate batches with exactly one
claim per task, including warm calls: 500 audited candidate batches total.
Another 200 final batch output/reduction comparisons passed after 800 timed
calls. Instrumented timings are excluded from performance conclusions. These
checks do not qualify completion-group callbacks, concurrent pool instances,
or untested scheduling modes.

Sources, patch, binary/source hashes, return codes and raw logs are preserved in
`results/20261020/pool-ready-claim/`; `summary.json` uses within-round ratios,
not ratios of independently sorted samples. No GPU, worker or full-model run
was performed. This changes neither the decode throughput estimate nor the
unverified 90%-of-peak target. The next optimization should address useful
kernel work or measured memory traffic rather than extend this queue policy
without new evidence of a cheaper claim mechanism.


### Half-pair FP4 compiler comparison (2026-10-20)

The current NT1 gate/up half-pair implementation was compiled from identical
source with GCC 15.2 and Clang 21.1.8, O3, AVX2/FMA/F16C, no fast-math, and the
half-pair option enabled in both. Export renaming permits alternating calls
within one process. This isolates compiler choice for the newer half-pair loop;
it does not compare against the older non-paired baseline.

On the Threadripper 1950X, the synthetic K4096/R128 GU probe used random FP4
codes/scales and Q8 bytes including -128, 12 alternating pairs of 64 calls,
and cores 3 and 11 pinned separately. Prefetch, narrow/offset dots and the NT4
compiler selector were disabled. NT1 median paired Clang/GCC CPU-time ratios
were 1.028/1.075 on core 3 with 1/64 banks, and 1.108/1.071 on core 11.
Two-token 64-bank comparisons, which do not use half pairing, gave 0.999/0.997.
Clang is rejected for this NT1 path; keep the existing GCC implementation.

All 144 final batch bitwise comparisons after 9,216 timed calls passed, as did
initial parity for each bank. These checks use synthetic GU outputs with clamp
7 and global scales 0.001, not full decoder quality. Host load was uncontrolled.
No service-level, worker, GPU, full-model or peak-utilization result is claimed.
No production files or defaults changed. Source snapshots, exact build commands,
compiler versions, binary hashes, settings, return codes and raw paired logs
are in `results/20261020/fp4-half-compiler/`.


### Chronological FP4 streaming pairs under contention (2026-10-20)

The CPU streaming probe now emits each chronological read/dot pair before
sorting its existing summaries. Each pair includes order, wall time, process
CPU time and dot/read payload-rate ratio. The timing region and kernels are
unchanged. This preserves adjacent comparisons when external load changes;
it does not make sequential read and compute samples simultaneous.

The updated probe built against both HIP- and CUDA-configured CPU archives.
Both two-worker smoke runs passed independent GGML output checks. Four larger
HIP-configured runs used eight or sixteen pinned physical workers, 64 matrices
per worker, default pages, first touch by the owner, and prefetch/narrow/offset
selectors off. Every row was checked against GGML outside each timing interval.
Nine reported pairs per run followed one excluded warm pair. A parser verified
pair order, ratio arithmetic and agreement with both existing sorted medians.
No GPU ran; SYCL was not tested.

Eight-worker repeated runs gave plain-read payload rates 12.84/13.82 GB/s and
FP4 dot rates 6.32/5.92 GB/s. Sixteen-worker runs gave 20.14/20.15 GB/s read and
10.07/10.02 GB/s dot. Median adjacent dot/read payload ratios were 0.531/0.451
for eight workers and 0.524/0.473 for sixteen. These are ratios of logical input
bytes per wall time, not measured memory-controller traffic or theoretical
bandwidth utilization. In particular, a ratio of independently sorted medians
need not equal the median chronological ratio.

The sixteen-worker dot runs averaged only 4.26/4.40 process CPU seconds per
wall second; eight-worker runs averaged 2.63/2.49. An adjacent load-average
snapshot was about 61. Preserve these results as heavily contended observations,
not a replacement for earlier lower-load streaming rates or a kernel regression.
The visible PMU inventory has no memory-controller device. DIMM rate/channel
configuration remains unverified, so the 90% theoretical-bandwidth target is
still unqualified. No workloads were stopped or scheduling policy changed.

Source snapshot, build commands, hashes, PMU inventory, settings, return codes,
raw chronological pairs and summaries are in
`results/20261020/stream-paired-current/`. No model throughput estimate changes.


### FP4 compiler screening at verification widths (2026-10-20)

A renewed read of the V100 experiment identifies width-dependent CPU operand
selection as a separate optimization axis. Its full-model results still do not
establish a transferable local speedup: input reuse failed qualification, and
its selected 30-plus tok/s profile uses speculation and bounded route dropping.
The local compiler timing probe previously covered NT1/2/4. A scratch extension
accepts widths 1–8 and sizes its activation/output pointer arrays accordingly.

The same GCC 15.2 and Clang 21.1.8 objects from `fp4-half-compiler/` were compared
at widths 3–8. Both use identical source and O3 AVX2/FMA/F16C flags without
fast-math. Half pairing is enabled in both objects but only affects NT1, absent
from this screen. Each case used K4096/R128, 64 random weight banks, raw FP4
scales/codes, Q8 bytes including -128, clamp 7, and 12 alternating pairs of
64 calls. Core 3 ran widths 3/5/6/7/8/4; core 11 reversed that order. Prefetch,
narrow/offset paths and the existing NT4 selector were disabled.

| Token width | Clang/GCC CPU-time ratio, core 3 | Core 11 |
| --- | ---: | ---: |
| 3 | 1.025 | 0.984 |
| 4 | 0.971 | 0.949 |
| 5 | 0.905 | 0.887 |
| 6 | 0.922 | 0.920 |
| 7 | 0.873 | 0.859 |
| 8 | 0.832 | 0.863 |

Widths 5–8 favor Clang in 90 of 96 paired CPU-time comparisons. Width 3 is
inconclusive. All 288 final batch comparisons after 18,432 timed calls and all
initial bank comparisons passed bitwise. Host load was uncontrolled; CPU time
helps distinguish descheduling but does not control clock or shared-resource
variation. The probe checks clamped GU output, not all intermediate dot values.

These results justify a complete-expert test of widths 5–8, with GCC down,
quantization and reduction unchanged, before considering an opt-in dispatch
extension. They do not establish greedy NT1 speedup, full-model throughput,
worker benefit or 90% peak utilization. Production dispatch stays unchanged;
no GPU or backend source was modified. Scratch probe, object/binary hashes,
settings, raw logs and summaries are in `results/20261020/fp4-compiler-widths/`.


### Wider Clang GU complete-expert screen (2026-10-20)

A scratch dispatcher selects Clang only for ordinary FP4 GU widths 5–8. GCC
handles down, other widths and K32. Both compiler objects use the same current
half-pair-enabled source; no production dispatcher was modified. The real
partial-weight harness tests NT5 and NT8, layers 3/4, two experts, four rotating
banks, 15 workers plus host, three tasks per thread and NUMA-owned rows. Split
and dataflow modes include hidden quantization, down and routed reduction;
input quantization is outside timing. Hidden-quantization fusion is off.

Each configuration has nine adjacent alternating-order pairs of 16 calls plus
separate 32-call CPU-cost batches. Repeated candidate/baseline median paired
wall-time ratios were:

| Layer / width | Split, first / repeat | Dataflow, first / repeat |
| --- | ---: | ---: |
| 3 / 5 | 0.948 / 0.950 | 0.894 / 0.939 |
| 3 / 8 | 0.908 / 0.966 | 0.967 / 0.965 |
| 4 / 5 | 0.941 / 0.996 | 1.036 / 1.034 |
| 4 / 8 | 0.954 / 1.120 | 1.016 / 0.918 |

The first layer's improvements repeat, but the second layer does not establish
a consistent benefit. In particular its NT5 dataflow regression repeats. Keep
this as a candidate requiring broader service validation, not an unconditional
width 5–8 dispatch recommendation. External load remained uncontrolled; the
separate CPU-cost batches are not paired with the latency samples.

All 288 final latency-batch output/reduction comparisons after 4,608 calls and
32 CPU-batch comparisons after 1,024 calls passed bitwise against the sequential
GCC reference. Intermediate calls are not individually checked. This covers
ordinary arithmetic at NT5/8, not NT6/7 service, canonical arithmetic, fused
hidden quantization, worker hardware or full-model quality. No GPU was used.
No production code/default or simulator-rate change follows. Scratch sources,
commands, hashes, settings, return codes, raw ordered samples and summaries are
in `results/20261020/fp4-wide-service/`.


### Wider Clang GU worker follow-up (2026-10-20)

The preceding primary-node split-phase means favor Clang GU in seven of eight
cases (ratios 0.909–0.967, with one 1.044). Unchanged hidden-quantization and down
phases also vary substantially, so those means do not isolate the cause of the
whole-expert variation. `phase-summary.json` preserves all phase ratios.

A static CPU-only binary then tested the same GU-only selection on the Ryzen 9
3950X worker. Its single-NUMA topology uses complete expert blobs. Two runs used
the existing real partial weights, layers 3/4, two experts, widths 5/8, four
rotating banks, 15 workers plus host and nine alternating 16-call pairs. Both
kept GCC down, input/hidden quantization and reduction. The miner continued
running; an adjacent load-average snapshot was about 30. No GPU was used.

| Layer / width | Split ratio, first / repeat | Dataflow ratio, first / repeat |
| --- | ---: | ---: |
| 3 / 5 | 1.000 / 1.077 | 1.005 / 1.035 |
| 3 / 8 | 0.951 / 0.929 | 1.022 / 0.987 |
| 4 / 5 | 1.051 / 0.981 | 0.934 / 0.998 |
| 4 / 8 | 0.996 / 0.872 | 0.963 / 0.945 |

Width 8 remains a possible workload-specific optimization, but the worker does
not show a general dataflow gain and width 5 does not qualify. Do not broaden
production dispatch to widths 5–8 on this evidence. A width-8-only follow-up
would need additional route shapes, fused/canonical coverage and full-model
width-frequency measurements before selection; no speculative decode speedup
is established here.

Both runs exited zero. All 288 latency-batch output/reduction comparisons after
4,608 calls and 32 separate CPU-batch comparisons after 1,024 calls passed.
These compare final batch outputs with a sequential GCC reference, not every
intermediate invocation. Commands, scratch sources, static binary hash, build
output, raw logs and `worker-summary.json` are preserved with the preceding
`results/20261020/fp4-wide-service/` evidence. Production remains unchanged.


### Four-chain Q8_K maximum screening (2026-10-20)

A scratch copy of the native AVX2 Q8_K quantizer replaces its 32-operation
maximum dependency chain with four independent chains and a final maximum.
The original second pass still selects the first maximum's signed value;
scale division, rounding, byte packing and block sums are unchanged. Each
chain starts at zero and uses the existing operand order to skip NaNs. This
candidate targets input/hidden activation preparation for ordinary Q2/Q3 and
IQ experts; it does not change the separate canonical quantizer or FP4 Q8_0.

The existing parity suite was rebuilt with its direct AVX2 reference comparison
pointing at the candidate. Its 16 cases at four sizes passed, including ties,
zero blocks, extremes, NaNs, infinities and denormals. Its engine-entry/chunk
checks still exercise the production quantizer and are not candidate integration
coverage. A separate checker compared every output byte against GGML's scalar
reference for 2,048 position/pattern combinations: opposite-sign maximum ties,
NaNs, positive/negative infinity, otherwise-all-NaN blocks, signed zeros and
denormals. Unwritten zero-block sum fields were initialized alike. All passed.

On the Threadripper 1950X, four paired runs pinned to cores 3/11/11/3 rotated
four finite input rows, with 12 alternating pairs of 1,024 calls per size.
Candidate/baseline median CPU-time ratios were 0.952/0.917/0.934/0.972 at
2,048 values and 0.981/0.958/0.947/0.984 at 4,096. These are isolated quantizer
costs, not complete-expert gains. All 192 timed batches passed their final
output checks; there is no per-invocation comparison inside timing. Host load
was uncontrolled. The quantizer's share of complete service must bound any
expected benefit, and canonical execution remains outside this candidate.

The next gate is an ordinary Q2/Q3 complete-expert comparison with only the
quantizer switched. Production code/defaults remain unchanged, and no GPU,
worker, full-model speedup or peak-utilization result is claimed. Candidate,
probe and adversarial sources, source/binary hashes, return codes, raw logs and
summaries are in `results/20261020/q8k-max4/`.


### Four-chain Q8_K complete-expert check (2026-10-20)

A scratch copy of `native_expert.cpp` redirects only ordinary AVX2 Q8_K
activation/hidden calls through a synchronous candidate selector. Both variants
share the same dispatch overhead and all expert dot/activation/reduction code.
The selector changes between completed batches. Canonical quantization bypasses
it. Input quantization occurs before timing, so this measures the candidate's
hidden-quantization contribution rather than its entire possible decoder cost.

Synthetic Q2/Q3 H4096/F2048 experts used eight banks, 15 workers plus host,
NUMA-owned rows, three tasks per thread and 20 alternating pairs of 16 calls.
NT1 split gave a median paired candidate/baseline wall ratio of 0.944; NT1
flow gave 0.971 initially and 1.004 on repeat. NT2 mixed flow gave 1.040,
NT4 dense flow 1.008, and the unchanged canonical NT2 mixed control 1.017.
Mixed routing uses one NT2 expert and six NT1 experts. External host load was
uncontrolled, and the initial whole-service gain did not reproduce.

The split phase's paired quantization ratio was 0.958, but baseline hidden
quantization was only 0.00353 of total wall time. Holding all else fixed,
that isolated reduction would remove about 0.00015 of total service time
(0.015%), far below the observed whole-service movement. Products of these
medians are an approximate sensitivity calculation, not a measured causal
speedup. This is not evidence that quantization caused the larger initial win.
Do not integrate this candidate as a decode-throughput improvement based on
these results; retain it as an isolated quantizer lead of low current priority.

All six subprocesses exited zero. The 240 final batch expert-output/reduction
comparisons after 3,840 timed calls passed bitwise. Input preparation, canonical
implementation, production code, defaults and simulator rates are unchanged.
No worker, GPU or full-model run was made. Scratch sources, build commands,
hashes, ISA/environment settings, return codes and ordered logs are in
`results/20261020/q8k-max4-service/`; the original isolated and adversarial
checks remain separately preserved in `results/20261020/q8k-max4/`.


### Q2 integer accumulation chains (2026-10-20)

The preceding NT1 Q2/Q3 split baseline spends median fractions 0.561 in GU and
0.431 in down, compared with 0.00353 in hidden quantization. This motivated a
scratch change to the direct Q2 NT1 dot: accumulate even and odd 32-value chunks
separately, add their integer vectors, then retain the original scale conversion
and floating accumulation order. Q3 and multi-token direct dots remain unchanged.
Each Q2 lane's integer sum is bounded well below int32 range even at full Q8 and
scale extremes, so this rearranges exact integer additions. It does not split
or reorder floating-point sums. Both baseline and candidate enable the existing
Q2 half-pair conversion; lookup mode and prefetch are off.

A synthetic K4096/R128 paired GU probe used random packed Q2 codes/scales,
finite FP16 block scales, Q8 values including -128, and 12 alternating pairs
of 64 calls per case on the Threadripper 1950X. NT1 candidate/baseline median
CPU-time ratios with 1/64 banks were 0.998/0.981 on core 3 and 0.986/1.001 on
core 11. Unchanged NT2 64-bank comparisons gave 0.990/0.996. This small,
inconsistent difference overlaps control variation and does not justify a
complete-expert integration or claimed speedup. Retain the current kernel.

All 144 final batch output comparisons after 9,216 timed calls, plus initial
per-bank checks, passed bitwise. These check clamped GU outputs, not a full
scalar exhaustive dot oracle. Host load was uncontrolled. Scratch sources,
build commands, hashes, settings, return codes, raw logs and paired summaries
are in `results/20261020/q2-int-chains/`. Production code/defaults remain
unchanged; no worker, GPU, full-model or peak-utilization measurement follows.


### Vector decode of compact Q3 scales (2026-10-20)

A scratch Q3 direct-dot candidate decodes the existing twelve packed scale
bytes with an eight-byte load, nibble masks, a four-byte broadcast and per-dword
variable shifts 0/2/4/6. Subtracting 32 produces the same sixteen signed scales.
Both loads stay within the scale array; no padded or expanded representation
is introduced. Q2, expanded Q3, dot products, FMA order and horizontal reduction
are unchanged. This differs from the earlier expanded-scale layout: it saves
no persistent conversion and adds no weight bytes.

An independent scalar decode comparison passed 13,072 cases: every possible
byte value at each of twelve positions with a fixed background, plus 10,000
random complete scale arrays. This is targeted coverage, not exhaustive over
all 96 input bits. The paired row probe additionally checked an FP64 decoded
reference within its existing tolerance and required candidate/baseline bitwise
agreement for every bank and final timed output.

On the Threadripper 1950X, K4096/R128 quantized Q3 matrices used 64 replicated
physical banks (14,417,920 bytes), twelve alternating pairs of 128 calls per
case, and cores 3/11 pinned separately. Prefetch and existing Clang dispatch
selectors were off. Candidate/baseline median paired CPU-time ratios were
0.937/0.955 for NT1, 0.976/0.980 for NT2, and 0.991/0.994 for NT4. NT1 won all
24 pairs. All 144 timed batch checks after 18,432 calls passed. Replicated banks
exercise different addresses but are not independent random routing weights.
Host load was uncontrolled.

The NT1 result warrants complete-expert measurement with only compact Q3 down
changed, followed by wider raw-code/scale parity and worker testing if service
gains persist. It is not yet a production selection or full-model throughput
claim. Scratch sources, build commands, hashes, settings, return codes, scale
checks and raw paired timings are in `results/20261020/q3-vector-scales/`.
Production layout, code, defaults and simulator rates remain unchanged.


### Compact Q3 vector-scale service and raw parity (2026-10-20)

A scratch paired dispatcher changes only compact Q3 down projection. The GCC
baseline was freshly compiled from current source with the same half-pair flag;
Q2 GU, quantization, expanded Q3 and reduction are identical. Production code
and dispatch remain unchanged. Before timing, a separate parity probe passed
1,200 configurations covering K256/512/768/2048/4096, token widths 1–8,
arbitrary packed Q3 codes/scales, six Q8 patterns including -128/+127, finite
FP16 scales including negative/zero/denormal values, padded strides, guarded
partial rows, unclamped/clamped GU, canonical arithmetic and gate skipping.
Direct rows also passed the existing decoded FP64 tolerance check.

The Threadripper 1950X service screen used synthetic Q2/Q3 H4096/F2048 experts,
eight banks, 15 workers plus host, NUMA-owned rows and three tasks per thread.
Each case has twenty alternating pairs of sixteen calls; input preparation is
excluded. NT1 split candidate/baseline median paired latency was 0.988, with
GU/quant/down phase ratios 1.020/0.994/0.935. Ordinary NT1 dataflow gave
0.998 initially and 0.971 on repeat; mixed NT2 dataflow gave 0.972. Canonical
NT1 dataflow gave 0.921 initially but 1.062 on repeat, so its large initial gain
is not reproducible. CPU-time ratios for those canonical runs similarly swung
0.888 to 1.113. Mixed NT2 has one shared NT2 and six NT1 expert jobs.

All six runs exited zero and all 240 final batch output/reduction comparisons
after 3,840 timed calls passed bitwise. The split down improvement supports the
row-level finding, but whole-service gains remain small or inconsistent under
uncontrolled host load. Do not claim the initial canonical speedup, change a
default or update simulator rates from these results. Worker testing and more
stable service evidence remain necessary before any production selection.
No GPU or full-model execution was performed.

Sources, exact compile commands, hashes, settings, parity output, raw paired
service logs and summary are in `results/20261020/q3-vector-service/`.


### Compact Q3 vector-scale worker check (2026-10-20)

The same paired down-only comparison was statically linked and run on the
Ryzen 9 3950X worker. Its one-NUMA topology uses complete expert blobs; the
synthetic Q2/Q3 H4096/F2048 data, eight banks, 15 workers plus host, three tasks
per thread and twenty alternating pairs of sixteen calls match the preceding
primary-node protocol. The miner remained running and external load was not
controlled. No GPU was used.

NT1 split candidate/baseline median paired wall time was 1.013. Its GU/hidden
quant/down phase ratios were 0.980/1.001/1.049: the changed down phase regressed
in this run. Ordinary NT1 dataflow gave 1.014 then 1.083; canonical NT1 dataflow
gave 1.067 then 1.113. Mixed NT2 dataflow gave 0.990, with 11/20 candidate wins.
CPU-time ratios also regressed for both repeated NT1 modes. This does not
support the candidate on the worker. The earlier primary row win is not a
portable optimization recommendation, and inconsistent primary service results
do not yet justify a hardware-specific production switch either.

All six runs exited zero. All 240 final batch output/reduction comparisons
after 3,840 timed calls passed bitwise. The static build command, binary hash,
linker output, per-run return codes and raw logs are preserved alongside the
primary evidence in `results/20261020/q3-vector-service/`, with separate
`worker-metadata.json` and `worker-summary.json`. Production kernels, defaults,
weight layout and simulator rates remain unchanged. Do not extend this scale
decoder globally from the isolated Threadripper measurement.


### Q3 activation-correction preparation profile (2026-10-20)

A scratch single-thread profile times the existing Q3 per-call preparation of
`4 * activation pair sums`, then the full row-call scope. It changes no kernel
arithmetic. The diagnostic counters are deliberately single-thread-only; they
must not be linked into a worker-pool benchmark. Timer reads and counter work
are included in these instrumented wall durations, as is descheduling. Variant
latencies printed by the reused paired harness are not speedup measurements.

On Threadripper core 3, K4096 with eight physical weight banks and row ranges
1/8/32/128 gave preparation fractions 20.5%/3.83%/0.93%/0.257% for NT1 and
25.2%/3.96%/1.05%/0.261% for NT4. Average preparation durations were 114–186 ns
for NT1 and 453–508 ns for NT4. Each case profiles 1,556 candidate calls:
eight initial bank checks plus twelve warm calls and 1,536 timed calls. The
initial one-row attempt stopped before kernel entry because an unused packed
layout setup required 32-row tiles; the scratch harness now skips packing for
row-layout tests. The failed log is retained, not counted as a passing run.

All eight corrected cases exited zero and their 192 final timed batch output
checks passed, along with initial bank parity and the decoded FP64 check.
At the current synthetic service setting, 16 participants times three tasks
per participant, two experts and two NUMA sides give twelve down partitions
per side: 2048/12, or 170–171 output rows each. Its actual down K is 2048,
not this diagnostic's K4096, so the profile does not directly measure that
service fraction. It nevertheless gives no support for adding cross-task
preparation ownership to optimize these large row tasks. Tiny row tasks could
justify separate investigation if observed in real routing.

Do not interpret the preparation fraction as removable end-to-end time or a
90%-of-peak measurement. External load was uncontrolled. Sources, build
commands, hashes, settings, failure/corrected logs and summary are in
`results/20261020/q3-prepare-profile/`. Production code/defaults remain
unchanged; no GPU or full-model run was performed.


### Service-window Q2/Q3 cycle profile (2026-10-20)

A scratch no-op selector links the existing paired service harness against the
current production CPU archive. Both candidate labels execute identical code.
Linux perf samples user cycles at a fixed 200,000-cycle period. The first
whole-process capture includes synthetic quantization/initialization and worker
waiting during setup, so its symbol shares are not service costs. A second
probe adds CLOCK_MONOTONIC markers bracketing the measurement loop and records
with `--clockid mono`. Analysis includes only samples inside that interval.
The interval includes warm calls, output checks and printing, but excludes
weight creation, reference setup and shutdown.

The Threadripper 1950X NT1 dense dataflow case uses two synthetic Q2/Q3 experts,
H4096/F2048, eight weight banks, 15 workers plus host and three tasks per thread.
The marked interval contains 300,968 samples. Q2 direct dots account for 45.32%
and Q3 direct dots 33.28% of sampled user cycles. Pool worker and flow_task
functions account for 12.47% and 5.30%; wait_done is 0.46% and claim 0.25%.
These symbols include useful dispatch/work and potentially waits; their full
shares cannot be labeled idle time or removable scheduling cost. Sampling skid,
inline attribution and uncontrolled host load also limit interpretation. These
are CPU-cycle shares, not elapsed token latency or theoretical utilization.

The result favors further dot-kernel work over the previously measured small
quantization/preparation costs. It does not establish a new speedup. The marked
run exited zero and passed 40 final batch output/reduction comparisons after
2,560 timed calls. Initial and marked captures remain separate; the raw perf
files stay in the ignored build tree. Source, build command, hashes, settings,
logs and the compact marked symbol distribution are in
`results/20261020/q23-service-profile/`. Initial symbol extraction omitted the
instruction-pointer field required by perf script to resolve symbols and yielded
no classified samples; extraction was corrected without rerunning the workload.
Production code/defaults, simulator rates and full-model throughput claims
remain unchanged. No GPU was used.


### FP4 service-window cycle profile (2026-10-20)

The same timestamp-window method was applied to real partial FP4 weights:
layers 3/4, two experts, NT1/4, four rotating banks, 15 workers plus host,
NUMA-owned dataflow and three tasks per thread. Hidden quantization fusion,
prefetch, narrow/offset dots and NT4 Clang selection were off. The current CPU
archive includes the NT1 half-pair implementation. Both harness labels execute
the same kernels through a no-op selector. Monotonic markers bracket only the
nine alternating 16-call pairs, including warm calls and output checks, while
excluding setup and the separate CPU-cost batches.

Within those four intervals, pool worker/flow_task shares of sampled user
cycles were 47.63/19.28% and 52.04/19.13% for the two NT1 cases, and
34.21/28.54% and 31.04/28.56% for NT4. This differs markedly from the Q2/Q3
service profile. Symbol-offset extraction and disassembly place the hottest
worker offsets 0xba/0xbd immediately after the epoch-wait pause, and the
hottest flow_task offsets 0xda/0xdc immediately after the expert-readiness
pause. Other prominent samples fall on the same loops' checks. Sampling skid
prevents exact instruction-latency attribution, but these addresses support
waiting as a significant CPU-cycle cost rather than labeling all pool cycles
as useful dispatch. They do not establish removable elapsed time.

A fresh unprofiled waiting-policy comparison is justified by this evidence;
reduced spinning must be evaluated separately from token latency. The previous
scheduler scans and static ordering were unsuccessful and are not reselected
by this profile. Startup/check overhead, profiler overhead and uncontrolled
host load mean this capture is not a throughput benchmark or proof of peak
utilization. Real full-model routing can differ from these two-expert cases.

The capture exited zero. All 72 final timed batch output/reduction comparisons
after 1,152 calls passed, plus eight separate CPU-batch comparisons after 256
calls. Source, build command, hashes, settings, phase markers, per-window symbol
shares, pool symbol offsets and disassembly are in
`results/20261020/fp4-service-profile/`. The raw capture stays in the ignored
build tree. An unsupported perf script offset option failed during analysis;
the documented symoff field succeeded on the same capture. Production kernels,
settings and simulator rates remain unchanged; no GPU was used.


### Unprofiled intermediate spin-window check (2026-10-20)

Following the FP4 service-window cycle profile, the same binary ran without
perf under existing `STRATA_POOL_SPIN_US` settings 20000/50/0/0/50/20000.
This tests the parked-worker epoch wait only; it does not change the in-flight
expert-readiness loop. The workload retains real layers 3/4, two experts,
NT1/4, four rotating banks, 15 workers plus host, NUMA-owned unfused dataflow
and three tasks per thread. Both harness variant labels are no-ops. Their
18 chronological wall samples per geometry are pooled; process CPU cost comes
from separate 32-call batches, averaged across the two identical labels.

| Spin window, microseconds | NT1 wall ms range | NT1 CPU-ms range | NT4 wall ms range | NT4 CPU-ms range |
| --- | ---: | ---: | ---: | ---: |
| 20000 | 2.778–3.697 | 34.36–50.99 | 5.689–8.077 | 79.09–92.65 |
| 50 | 2.578–3.671 | 19.33–22.78 | 6.560–7.389 | 48.17–52.41 |
| 0 | 3.055–3.437 | 18.36–22.55 | 6.591–7.802 | 49.17–52.14 |

Ranges span both layers and process repeats, not confidence intervals. Both
short windows consistently lower process CPU cost under this load. There is
no consistent wall-time gain or clear 50-us advantage over zero, so the result
does not justify a latency claim or changing the default. CPU time includes
all threads' useful work and spinning; it is not a hardware-utilization metric.
The experiment confirms the earlier immediate-sleep CPU-cost finding and adds
an intermediate-window comparison. It leaves readiness waits as a distinct
possible target, requiring its own correctness and latency evaluation.

All six processes exited zero. All 432 final latency-batch comparisons after
6,912 calls and 48 CPU-batch comparisons after 1,536 calls passed. Output hashes
matched across policies. Background work/miners continued; load was uncontrolled.
No GPU or full-model run occurred. Binary hash, settings, return codes, raw logs
and summaries are in `results/20261020/fp4-spin-window/`; source/build evidence
is in the preceding `fp4-service-profile/`. No production setting or simulator
rate was changed, and the 90%-of-peak target remains unqualified.


### Periodic expert-readiness yield rejected (2026-10-20)

A scratch pool candidate adds `std::this_thread::yield()` after every 256
unsuccessful expert-readiness polls. The candidate flag is loaded once before
the wait and changes only between synchronous completed calls. Acquire loads,
ready tags, task claims, queue ordering, completion handling and the existing
stall guard are unchanged. This tests the in-flight down-task wait separately
from the parked-worker spin window; it does not sleep on a new notification
or cache a readiness decision.

Real partial FP4 layers 3/4 used two experts, NT1/4, four rotating banks,
15 workers plus host, owned unfused dataflow and three tasks per thread. Each
case has nine adjacent alternating-order pairs of sixteen calls plus separate
CPU-cost batches. Process park windows ran 50/20000/20000/50 microseconds.
With the 50-us park window, NT1 median paired yield/baseline latency ratios
were 1.332/1.307 across layers, then 1.210/1.396 on repeat. CPU-batch ratios
were 0.942/0.836 then 0.681/0.738. NT4 latency ratios were 1.041/0.946 then
0.952/0.953, with CPU-batch ratios 0.744–0.812. These CPU batches are separate
measurements, not paired latency-phase CPU attribution.

With the default 20-ms park window, NT1 latency ratios were 1.183/1.059 then
1.110/1.016. NT4 gave 1.064/0.909 then 1.190/1.014. CPU cost also rose in several
cases. The repeated NT1 latency regression rejects this candidate for decode;
reduced spin CPU cost with the short window does not compensate for slower
completion. Profiling a wait loop as hot is not proof that yielding accelerates
its dependency. Host load was uncontrolled and no background work was stopped.

All four subprocesses exited zero. The 288 final latency-batch comparisons
after 4,608 calls and 32 CPU-batch comparisons after 1,024 calls passed bitwise.
This is final-output coverage, not a race-detector proof or completion-group
qualification. No production scheduler/default changed and no GPU/full-model
run occurred. Scratch sources, patch, build commands, hashes, settings, raw
ordered samples and summaries are in `results/20261020/fp4-ready-yield/`.


### Paired split versus dataflow with short parking (2026-10-20)

A scratch harness now pairs the existing owned split and owned dataflow paths
adjacently, reversing their order each round. It changes no engine kernels or
scheduler. Candidate label zero means split, one means dataflow. Both use the
same production archive, real FP4 layers 3/4, two experts, widths 1/4, four
rotating banks, 15 workers plus host, three tasks per thread and unfused hidden
quantization. Split includes its routed reduction; outputs from both paths
are compared to the same sequential reference. Each geometry has nine paired
16-call batches plus separate 32-call CPU-cost measurements.

The process park windows ran 50/20000/20000/50 microseconds. With 50 us,
median paired dataflow/split latency ratios were:

| Layer / width | First | Repeat |
| --- | ---: | ---: |
| 3 / 1 | 1.232 | 1.428 |
| 3 / 4 | 1.189 | 1.232 |
| 4 / 1 | 1.302 | 1.349 |
| 4 / 4 | 1.180 | 1.158 |

Thus split used about 14–30% less wall time across those cases. Separate
CPU-batch dataflow/split ratios were 1.15–1.77, also favoring split. Under the
default 20-ms park window, latency ratios ranged 0.881–1.337; one NT4 repeat
favored dataflow. This supports short-parking plus split as a useful candidate
for this loaded two-expert partial workload, not a general model setting.
Single-/three-expert, fused/canonical and worker-node cases remain unqualified;
real route distributions and overlap can change the result. Host load was
uncontrolled, and CPU-batch ratios are not paired phase attribution.

All four processes exited zero. The 288 final latency-batch comparisons after
4,608 calls and 32 CPU-batch comparisons after 1,024 calls passed bitwise.
This is an execution-mode comparison, not a changed dot kernel or scheduler.
Production defaults, persistent settings and simulator rates remain unchanged.
No GPU/full-model run occurred. Scratch source, exact link command, binary hash,
settings, raw ordered samples and summary are in
`results/20261020/fp4-split-flow/`.


### Split/dataflow comparison across expert counts (2026-10-20)

The same paired execution-mode binary then ran one/three/three/one experts
using the existing three-expert partial fixture. The 50-us park window,
NT1/4, real layers 3/4, four banks, 15 workers plus host, three tasks per thread,
NUMA-owned rows and unfused hidden quantization remain fixed. Each configuration
has nine alternating pairs of sixteen calls plus separate CPU-cost batches.
This changes expert count, not kernels, task rules or weight format.

| Experts / width | Dataflow/split paired latency ratio range |
| --- | ---: |
| 1 / 1 | 1.522–1.826 |
| 1 / 4 | 1.350–1.435 |
| 3 / 1 | 1.323–1.413 |
| 3 / 4 | 1.016–1.084 |

Ranges span both layers and both process repeats. All geometry-level median
ratios favor split, although the small NT4 three-expert differences need caution
under uncontrolled host load. For NT1, split reduces elapsed service time about
34–45% at one expert and 24–29% at three experts in these tests. Separate
CPU-batch ratios also favor split (1.265–1.939 dataflow/split across all cases).
These results strengthen the small-workload split candidate; they do not imply
a fixed speedup for full-model routing, GPU overlap or other nodes. In particular,
the weaker advantage at NT4/E3 argues against treating execution mode as a
single hardware-wide throughput multiplier.

All four processes exited zero. The 288 final latency-batch comparisons after
4,608 calls and 32 CPU-batch comparisons after 1,024 calls passed bitwise. The
candidate remains an existing execution-path choice, not a production patch.
Worker topology, fused/canonical cases and end-to-end overlap still need
qualification before a general preset or dispatch policy. No default or
simulator-rate change follows. Logs, binary hash, settings, return codes and
summaries are in `results/20261020/fp4-split-flow-experts/`; the harness source
and link command remain in the preceding `fp4-split-flow/` evidence.


### Split/dataflow worker topology check (2026-10-20)

A static scratch harness permits both topologies: owned rows on two-NUMA hosts
and complete blobs on a single-NUMA host. It retains adjacent alternating split
and dataflow pairs with identical kernels. On the Ryzen 9 3950X worker, the
existing partial weights ran one/two/two/one experts, layers 3/4, NT1/4, four
banks, 15 workers plus host, three tasks per thread and the 50-us park window.
The miner continued running; no GPU was used or host load controlled.

| Experts / width | Dataflow/split paired latency ratio range |
| --- | ---: |
| 1 / 1 | 0.906–0.953 |
| 1 / 4 | 1.025–1.467 |
| 2 / 1 | 0.935–1.248 |
| 2 / 4 | 1.124–1.243 |

Dataflow wins all four geometry-level NT1/E1 medians (about 5–9% less time),
contrary to the primary two-NUMA result. NT4 favors split at both expert counts,
while NT1/E2 is inconsistent. Separate process CPU-batch ratios are all above
one (1.052–1.743), so lower CPU consumption does not necessarily imply lower
latency. This rules out applying the primary host's small-expert split result
as a cluster-wide policy. An eventual opt-in choice needs topology, expert count,
token width and overlap evidence; these partial tests do not establish a
complete policy or model-throughput multiplier.

All four worker processes exited zero. All 288 final latency-batch comparisons
after 4,608 calls and 32 CPU-batch comparisons after 1,024 calls passed bitwise.
The topology-aware scratch source, static build command/output, binary hash,
settings, return codes, raw ordered logs and summary are in
`results/20261020/fp4-split-flow-worker/`. No production code, persistent setting,
default or simulator rate changed. Full-model throughput and peak utilization
remain unverified.


### Fused hidden quantization and decoder-path audit (2026-10-20)

Decoder inspection found two constraints on applying the execution-mode result.
Ordinary native CPU execution already defaults to split: `STRATA_GLM_LAYER_FLOW`
selects dataflow only when exactly `1`. But when a second slot is present and
`STRATA_GLM_MOE_MERGE_GROUPS=early`, the early-completion branch calls grouped
dataflow regardless of that layer-flow switch, preserving mailbox completion
and GPU-overlap semantics. Disabling those groups to obtain a microbenchmark
win would need an end-to-end overlap test. Separately, assembled expert packs
normally enable hidden quantization fusion unless
`STRATA_GLM_PACK_FUSE_QUANT=0`; prior unfused measurements did not cover that
usual path. The distributed partial tree tool uses its separate W4A16 helper,
not this native ExpertPool path, so these settings are not automatically a
multi-node tree speedup.

The scratch paired harness therefore repeated the primary experiment with
fused hidden quantization in both split/dataflow paths. Explicit single/small
unfusing selectors were zero. One/three/three/one expert runs retained real
layers 3/4, NT1/4, four banks, 15 workers plus host, three tasks per thread,
NUMA-owned rows and the 50-us park window. Median paired flow/split ratios were:

| Experts / width | Range across layers/repeats |
| --- | ---: |
| 1 / 1 | 1.672–2.088 |
| 1 / 4 | 1.372–1.553 |
| 3 / 1 | 1.277–1.654 |
| 3 / 4 | 1.004–1.100 |

These support split for the tested small primary-node workloads with fusion
as well, while NT4/E3 remains close. They qualify an existing path choice,
not a newly accelerated decoder default. The comparison does not isolate
fusion itself: both modes enable it, including its existing task geometry.
External load was uncontrolled. No grouped-completion, GPU-overlap or full-model
throughput improvement is established.

All four runs exited zero. All 288 final latency-batch output/reduction checks
after 4,608 calls and 32 CPU-batch checks after 1,024 calls passed bitwise against
the sequential reference. Scratch source, compile command, hashes, settings,
return codes and paired results are in
`results/20261020/fp4-split-flow-fused/`. Production code/defaults, persistent
configuration and simulator rates remain unchanged.


### Fused split-path short-spin qualification screen (2026-10-20)

A read-only DIMM inventory attempt could not verify configured RAM speed:
noninteractive sudo required authentication, and readable EDAC exposed no memory
controller entries. No privilege policy changed. The theoretical-bandwidth
assumptions remain unverified; this does not block comparing actual CPU service.

The fused split/dataflow harness ran two real experts with park windows
20000/50/50/20000 microseconds, layers 3/4, NT1/4, four rotating banks,
15 workers plus host and three tasks per thread. No profiler ran. This directly
compares the ordinary fused split path's existing default park window with a
shorter setting; earlier tests primarily compared split against dataflow.

For split execution, 50-us/default-20-ms ratios of per-process wall medians were:

| Layer / width | Forward order | Reverse order |
| --- | ---: | ---: |
| 3 / 1 | 0.798 | 0.824 |
| 4 / 1 | 0.765 | 0.679 |
| 3 / 4 | 0.967 | 0.832 |
| 4 / 4 | 0.770 | 0.836 |

NT1 service was 18–32% shorter and NT4 3–23% shorter across these comparisons.
Separate CPU-batch ratios were 0.328–0.367 for NT1 and 0.386–0.448 for NT4,
so CPU cost also fell. This is a forward/reverse process comparison, not a
within-process randomized policy trial. Host load was uncontrolled. It supports
`STRATA_POOL_SPIN_US=50` as a candidate for further loaded primary-node fused
split tests; it does not justify a global default, a quiet-host recommendation,
a worker-node choice or a model-throughput multiplier. The source knob affects
parked-worker epochs, not expert-readiness waiting.

All four runs exited zero. The 288 final latency-batch comparisons after 4,608
calls and 32 CPU-batch comparisons after 1,024 calls passed bitwise. Hashes match
across paths and policies. Paired dataflow controls remain in the raw evidence
and summary. Binary hash, settings, return codes, hardware-inventory limitation,
ordered logs and forward/reverse ratios are in
`results/20261020/fp4-fused-spin-default/`. Production code/defaults, persistent
configuration and simulator rates remain unchanged. No GPU/full-model run or
90%-of-peak utilization claim follows.


### Same-process fused split park-window comparison (2026-10-20)

To reduce process-order drift, a scratch pool selector alternates the parked
worker threshold between the existing 20-ms default and 50 us within one
process. It uses an atomic flag read at the existing clock-check point, so
parked workers cannot race a write to the ordinary duration member. The sleep,
notification, epoch, claim and completion protocols are unchanged. Both
variants pay the selector overhead. Selection occurs before an untimed warm
call, followed by sixteen timed calls; the default and short-window variants
reverse order each round. Both use fused owned split execution throughout.
This selector is diagnostic only and is not added to the production API.

Real layers 3/4 ran one/three/three/one experts, NT1/4, four banks, 15 workers
plus host and three tasks per thread. Each geometry used nine alternating pairs
and separate CPU-cost batches. Median paired 50-us/20-ms wall-time ratios were:

| Experts / width | Range across layers/repeats |
| --- | ---: |
| 1 / 1 | 0.589–0.654 |
| 1 / 4 | 0.664–0.770 |
| 3 / 1 | 0.671–0.800 |
| 3 / 4 | 0.856–0.956 |

All sixteen geometry-level medians favor the shorter window. The result
supports 50 us for further loaded primary-node fused split experiments beyond
the earlier separate-process screen. CPU-batch ratios also favor short parking,
0.254–0.515, but are separate unpaired measurements. This establishes neither
quiet-host performance nor a full-model throughput gain. Host load was
uncontrolled, and a global default or automatic topology policy is not selected.

All four runs exited zero. All 288 final latency-batch comparisons after
4,608 calls and 32 CPU-batch comparisons after 1,024 calls passed bitwise.
No persistent setting changed. Scratch source/patch, exact build commands,
hashes, settings, ordered logs and summary are in
`results/20261020/fp4-spin-paired/`. Production scheduling and simulator rates
remain unchanged; GPU overlap and 90%-of-peak utilization remain unqualified.


### Q2/Q3 same-process park-window comparison (2026-10-20)

The shorter park window was tested on the existing Q2 GU / Q3 down service,
using the same atomic selector as the FP4 comparison. Baseline is 20 ms and
candidate is 50 us. Both variants use identical arithmetic kernels and pay
for the selector load. No production source or default changed.

On the primary Threadripper 1950X, 15 workers plus the host used owned NUMA
rows, three tasks per participant and eight rotating synthetic weight banks.
H=4096, F=2048; dense cases have two experts and one token, mixed cases have
seven experts and two tokens. Each run has 20 alternating pairs, one synchronous
warm call after selecting the threshold, then 16 timed calls per batch.
Cases ran split / flow / mixed flow / mixed flow / flow / split.

| Service | 50 us / 20 ms wall-time ratio, first / repeat | Process CPU-time ratio, first / repeat |
|---|---:|---:|
| One-token dense split | 0.673 / 0.668 | 0.391 / 0.382 |
| One-token dense dataflow | 0.737 / 0.767 | 0.437 / 0.422 |
| Two-token mixed dataflow | 0.925 / 0.973 | 0.523 / 0.538 |

Ratios are medians of within-round pairs. Dense split improves 33%, dense
flow 23–26%; mixed routing improves only 3–8%, with just 11/20 latency wins
in the second mixed run. Process CPU time decreases 46–62%; it includes
spinning and is not a measure of useful arithmetic efficiency.

All 240 batch output and routed-reduction comparisons were bit-exact after
3,840 timed service calls. These checks cover batch endpoints, not every
intermediate call. Q2 half pairing was compiled on, Q3 compiler dispatch
was disabled at runtime. The launch script sets an unused STRATA_Q2_LUT knob;
the actual STRATA_Q2K_LUT and prefetch knobs were independently confirmed
unset, so their default-off behavior applied.

Host load was uncontrolled (one sample: load average 40.28). Input quantization
is excluded, weights and inputs repeat, and RAM speed remains unverified.
These results support testing STRATA_POOL_SPIN_US=50 in end-to-end runs on
this host, but do not establish full-model tok/s, idle-host performance,
worker-node benefit or 90% theoretical CPU bandwidth/FLOPS. Evidence, source,
commands and hashes are in `results/20261020/q23-spin-paired/`.


### Worker-node Q2/Q3 park-window validation (2026-10-20)

The same park-window selector and Q2/Q3 probe were statically linked and run
on the Ryzen 9 3950X worker, with 15 workers plus host, eight weight banks and
three tasks per participant. The one-NUMA path uses complete expert blobs;
logs confirm owned=0. No GPU work or miner setting was changed. Runtime
Q2 LUT, Q3 compiler dispatch and Q23 prefetch were explicitly disabled.

The process order and within-process pairing match the primary-node test:
split / dataflow / mixed dataflow / mixed dataflow / dataflow / split,
20 alternating pairs per process and 16 timed calls after each warm call.

| Service | 50 us / 20 ms wall-time ratio, first / repeat | Process CPU-time ratio, first / repeat |
|---|---:|---:|
| One-token dense split | 0.995 / 0.991 | 0.744 / 0.644 |
| One-token dense dataflow | 0.903 / 0.857 | 0.624 / 0.646 |
| Two-token mixed dataflow | 1.456 / 1.324 | 0.998 / 0.836 |

Single-token split latency is essentially unchanged, and single-token flow
improves 10–14%. Mixed dataflow instead regresses 32–46%, despite unchanged
or lower process CPU use. Only 1/20 and 6/20 mixed-route latency pairs favor
the shorter window. The result rejects a cluster-wide 50 us recommendation:
reduced spinning is not sufficient evidence of lower service latency.
The primary's positive result remains specific to its tested configuration.
No default or persistent per-node setting changed.

All 240 batch endpoint output/reduction checks passed bit-exactly after
3,840 timed service calls. Host load was uncontrolled, inputs and weights
repeat, and input quantization is excluded. These are CPU service results,
not model throughput or peak-utilization measurements. The static build
command, binary hash, raw logs and paired summaries are preserved alongside
the primary evidence in `results/20261020/q23-spin-paired/worker-*`.


### Intermediate worker park threshold rejected (2026-10-20)

A same-process 500 us versus 20 ms comparison tested whether the worker's
mixed-route regression at 50 us could be avoided with a longer threshold.
Only the scratch pool selector constant changed; identical Q2/Q3 arithmetic,
15 workers plus host, eight banks, three tasks per participant, fixed inputs,
and complete-blob one-NUMA layout were retained. Process order was dense
split / dense flow / mixed flow / mixed flow / dense flow / dense split.
Each process ran 20 alternating pairs of 16 timed service calls, with one
warm call after selection.

| Service | 500 us / 20 ms wall ratio, first / repeat | Process CPU ratio, first / repeat |
|---|---:|---:|
| One-token dense split | 0.982 / 0.991 | 0.908 / 0.990 |
| One-token dense dataflow | 0.989 / 0.981 | 0.896 / 0.913 |
| Two-token mixed dataflow | 1.487 / 1.426 | 1.019 / 1.064 |

Mixed service regressed 43–49%, with just one of twenty latency pairs favoring
500 us in each repeat. Its process CPU time also increased 2–6%. Dense service
changes were small, at 1–2%. This rejects 500 us as a compromise for the worker's
mixed-route path. Retain the existing 20 ms setting for that path pending a
separately validated scheduling change. These measurements do not identify
the mechanism: wakeup overhead, scheduling and loaded-host interactions have
not been isolated. No production setting changed.

All 240 batch endpoint output/reduction checks passed after 3,840 timed calls.
The miner remained running. The synthetic weights repeat and input quantization
is excluded; no full-model tok/s or peak utilization is established. Source,
static build command/hash, raw logs and summaries are preserved under
`results/20261020/q23-spin-500us/`.

The external V100 experiment report was also re-read. Its resource-bound
section still attributes 86.4% of the selected decode request to verification;
the 37.82 tok/s zero-other-work bound and 1.67x verification speedup requirement
for 50 tok/s remain conditional. Its 173.4 tok/s compressed-code streaming
roof excludes metadata, unpack/dot, activation traffic and scheduling. Those
numbers provide no evidence that changing the worker park threshold can reach
the requested CPU utilization or model throughput target.


### Two-token FP4 gate/up half pairing screen (2026-10-20)

A scratch GCC candidate extends gate/up activation sharing to ordinary
two-token AVX2 calls. Each 32-value half unpacks gate and up once, builds their
scale vectors, then uses each token's activation load and activation scale for
both outputs. Four accumulators retain the original per-output FMA order.
NT1 keeps the existing half-pair implementation; other token widths, down,
K32 and narrow/offset paths retain their implementations. No weight expansion
or new activation layout is used. This is not integrated into production.

Both baseline and candidate use current source with NT1 half pairing enabled,
GCC O3, AVX2/FMA/F16C, without fast math. The screen uses K4096/R128, raw random
FP4 scale/code bytes and Q8 bytes including -128, finite fixed Q8 scales,
clamped GU with global scales 0.001, and twelve alternating pairs of 64 calls.
Every bank is checked before timing; each timed batch checks its final output
bitwise against the baseline. Prefetch and other optional dispatches are off.

| Core | Tokens | Weight banks | Candidate/baseline CPU-time ratio | CPU-time pair wins |
|---|---:|---:|---:|---:|
| 3 | 2 | 1 | 0.928 | 10/12 |
| 3 | 2 | 64 | 0.915 | 12/12 |
| 11 | 2 | 64 | 0.896 | 12/12 |
| 11 | 2 | 1 | 0.909 | 12/12 |
| 3 | 1, unchanged control | 64 | 1.003 | 3/12 |
| 11 | 4, unchanged control | 64 | 0.998 | 6/12 |

Two-token CPU time improves 7–10%, winning 46/48 pairs; wall-time ratios
are 0.896–0.927. Unchanged controls have CPU ratios within 0.3% of one.
All 144 timed batch checks pass after 9,216 timed calls, in addition to
initial bank checks. The 64 banks contain distinct random packed data.

The primary host remains loaded. This is a row-level screen, not expert-pool
service, full-model throughput, broad API coverage, or proof of theoretical
peak utilization. In particular, varied activation half scales, separate-up
buffers, partial row ranges, and unclamped calls need qualification before
integration. The next gate is complete CPU expert service at NT2 plus broader
parity checks. Sources, build commands, hashes and raw samples are in
`results/20261020/fp4-gu-pair2/`.


### Two-token FP4 pairing: real expert-service qualification (2026-10-20)

The two-token gate/up half-pair candidate was linked through an experimental
atomic dispatch wrapper, selecting it only for NT2 GU. Down and all other
widths remain GCC baseline. Freshly compiled baseline uses the current NT1
half-pair option; candidate is the object from the preceding row screen.
Real partial layers 3/4 ran one / three / three / one expert, four rotating
banks, 15 workers plus host, owned NUMA rows, three tasks per participant,
20 ms worker spin and hidden quantization fusion off. Both split and dataflow
were compared using nine alternating pairs of 16 calls, plus separate
32-call process CPU batches. Input quantization is excluded.

| Experts | Layer | Split candidate/baseline wall ratio, first / repeat | Dataflow ratio, first / repeat |
|---|---:|---:|---:|
| 1 | 3 | 0.938 / 0.911 | 0.890 / 1.008 |
| 1 | 4 | 0.894 / 0.922 | 1.064 / 1.054 |
| 3 | 3 | 0.864 / 1.068 | 0.959 / 0.953 |
| 3 | 4 | 0.946 / 0.922 | 1.114 / 1.023 |

One-expert split improves 6–11% in both repeats, but dataflow is not a general
win: layer 4 regresses in all four cases. Three-expert layer 3 split reverses
from improvement to regression. Split GU phase ratios span 0.875–1.117, while
unchanged down also varies, so whole-service differences cannot all be assigned
to GU arithmetic. The row-level 7–10% gain is insufficient for general rollout.
The candidate remains scratch-only; no dispatch or default changed.

All 288 latency batch checks and 32 separate CPU-batch checks passed bitwise
after 4,608 and 1,024 timed calls respectively. Additional API parity testing
passed 400 cases: K64/128/256/2048/4096, eight random seeds, raw packed bytes,
Q8 including -128, finite half scales including signed zeros/subnormals,
separate or contiguous up buffers, padded row strides, full or partial row
ranges, clamped and unclamped APIs. It compares the complete output buffers
against the current baseline, including untouched row regions; it is not an
independent scalar oracle or a proof for all floating-point inputs.

Host load was uncontrolled. No GPU, full-model quality/tok/s, worker-node
candidate service, or 90% peak-utilization validation was performed. The
same candidate needs a stable service win before production integration.
Evidence is in `results/20261020/fp4-pair2-service/`.


### Two-token pairing with fused hidden quantization (2026-10-20)

The NT2 gate/up candidate was compared again with hidden quantization fusion
on, matching the assembled-pack setting. The probe changes only the tested
mode selection from owned split/dataflow 2/3 to fused modes 4/5. Both variants
retain the same 20 ms park threshold, kernels and weight banks used in the
preceding unfused qualification. Fusion is disabled only for the sequential
reference; all timed variants report fuse_h_quant=1. Input quantization is
excluded. The primary host remains under uncontrolled load.

Real partial layers 3/4 ran E1 / E3 / E3 / E1, four rotating banks, 15 workers
plus host and three tasks per participant. Nine alternating pairs of 16 calls
per layer/mode compare baseline and NT2 pairing; separate 32-call CPU batches
remain outside the paired latency samples.

| Experts | Layer | Fused split candidate/baseline wall ratio, first / repeat | Fused dataflow ratio, first / repeat |
|---|---:|---:|---:|
| 1 | 3 | 0.990 / 0.873 | 0.913 / 0.856 |
| 1 | 4 | 0.830 / 0.875 | 0.963 / 0.817 |
| 3 | 3 | 1.033 / 0.925 | 0.946 / 0.914 |
| 3 | 4 | 1.038 / 0.930 | 1.034 / 1.000 |

All eight one-expert geometry-level medians favor the candidate, with 1–18%
less wall time. Three-expert layer 3 dataflow improves 5–9%, but split results
reverse between repeats and layer 4 dataflow does not improve. Separate CPU
batch ratios span 0.789–1.161 and are not paired latency attribution. Split GU
phase counters include fused quantization, so they cannot be read as isolated
GU kernel timing. Unchanged down-phase variation also remains visible.

All 288 latency batch checks and 32 CPU-batch checks passed bitwise after
4,608 and 1,024 calls. This adds evidence for a small-expert NT2 candidate but
not a universal service gain. No production dispatch/default, worker-node
behavior, simulator rate or full-model tok/s changed. A candidate selected for
production still needs worker-node qualification and applicable backend builds.
Source, commands, hashes and raw samples are preserved under
`results/20261020/fp4-pair2-fused/`.


### Two-token FP4 pairing on the single-NUMA worker (2026-10-20)

A statically linked NT2 candidate probe ran on the Ryzen 9 3950X worker with
the existing partial layers 3/4. Unlike the primary, this uses complete expert
blobs; every timed record confirms owned=0 and fuse_h_quant=0. Arithmetic
baseline and candidate objects match the primary service tests. The selector
changes only ordinary NT2 GU; down and other widths remain baseline.

The sequence E1 / E2 / E2 / E1 uses four banks, 15 workers plus host, three
tasks per participant and the unchanged 20 ms park threshold. Each layer/mode
has nine alternating baseline/candidate pairs of 16 timed calls, plus separate
32-call CPU batches. The miner was left running; input quantization is excluded.

| Experts | Layer | Split candidate/baseline wall ratio, first / repeat | Dataflow ratio, first / repeat |
|---|---:|---:|---:|
| 1 | 3 | 1.005 / 1.002 | 0.982 / 0.958 |
| 1 | 4 | 0.886 / 0.984 | 0.992 / 0.991 |
| 2 | 3 | 0.975 / 1.002 | 0.961 / 0.993 |
| 2 | 4 | 0.911 / 0.985 | 0.905 / 0.994 |

All eight dataflow geometry medians favor pairing, but each has only five or
six wins out of nine pairs. Most gains are small; the first E2 layer-4 gain of
9.5% shrinks to 0.6% on repeat. Split E1 layer 3 is essentially unchanged.
Unchanged down-phase and separate CPU-batch timings vary substantially, so
this is encouraging evidence, not a stable 7–10% service improvement derived
from the primary row screen. Do not select a default from these runs alone.

All 288 latency batch checks plus 32 CPU-batch checks passed bitwise after
4,608 plus 1,024 timed calls. No GPU, full-model output/tok/s or peak utilization
was measured. Production source/defaults remain unchanged; the candidate is
preserved for controlled service or model qualification. Raw logs, static
build/source hashes and paired summaries are in
`results/20261020/fp4-pair2-worker/`.


### Opt-in NT2 gate/up pairing integration (2026-10-20)

`STRATA_NVFP4_GU_HALF_PAIR_NT2` is now a CMake option, default OFF. Enable it
with `-DSTRATA_NVFP4_GU_HALF_PAIR_NT2=ON` in a native-expert CUDA or HIP build.
It selects half-paired GU only for ordinary two-token AVX2 NVFP4 rows. The NT1
option is independent; K32 and other token widths retain their previous paths.
The previously measured candidate reuses activation loads across gate/up and
reuses each weight unpack across both tokens without changing FMA order.

The option is experimental: primary row tests improved 7–10%, primary fused
one-expert service improved 1–18%, but three-expert and unfused service results
were mixed. Worker dataflow medians favored it in eight cases, mostly by small
amounts with limited paired wins. It is available for explicit evaluation,
not selected as a default or advertised as a full-model speedup.

Both existing HIP and CUDA build trees were configured with NT2 ON for this
validation (and retain that local setting). Full `strata-glm-decode` builds
passed. Each backend's five relevant FP4 native/offset/narrow/K32 CPU tests
passed. Each built CPU archive passed 400 broader baseline parity cases and
one fused real partial-weight run covering layers 3/4, NT2, E1, split/dataflow,
four banks and 15 workers plus host. The two partial runs passed 144 latency
batch checks plus 16 CPU-batch checks after 2,304 plus 512 calls; their two
variant labels intentionally select the same production implementation and
are correctness checks, not performance comparisons.

An additional NT2-only compile with NT1 OFF passed all 400 parity cases,
checking independence of the two options. With NT2 OFF, pre/post-change object
files are byte-identical for both NT1 OFF and NT1 ON. The first comparison used
different source basenames and failed due to object identity differences;
matching source basenames fixed the comparison. Final disabled-object hashes
and exact commands are preserved. `git diff --check` passes for changed source.

SYCL's separate build references this CPU source, but icpx is unavailable,
so SYCL was not built and is not qualified. No GPU execution, full-model token
quality/tok/s measurement or 90%-peak utilization proof was performed. Q2/Q3
kernels and runtime settings were not changed by this integration. Source
patch, hashes, build/test logs and validation commands are under
`results/20261020/fp4-pair2-integration/`.


### Integrated NT2 pairing hardware-counter screen (2026-10-20)

The integrated NT2 option was compared against the frozen pre-NT2 GCC object
on primary CPU 3. Both paths retain NT1 half pairing; only NT2 GU arithmetic
differs in the target runs. The probe checks every bank first, then fixes one
variant for eight batches of 1,024 calls, K4096/R128, raw packed scales/codes,
Q8 including -128, and clamped GU. Four processes per bank count run baseline /
candidate / candidate / baseline. Prefetch, narrow/offset and Clang dispatch
are disabled. Counters include initialization, parity, warmup and printing;
printed CPU/wall times cover the timed batches only. Host load is uncontrolled.

| Weight banks | Retired-instruction ratio | User-cycle ratio | Timed CPU ratio | Timed wall ratio |
|---|---:|---:|---:|---:|
| 1 | 0.873 | 0.989 | 0.979 | 0.986 |
| 64 | 0.891 | 0.916 | 0.916 | 0.816 |

Ratios compare the medians of two processes per variant. Both bank counts
retire about 1.278 billion fewer instructions for the candidate. The larger
setup contribution at 64 banks explains why the instruction percentage differs
while the absolute reduction stays similar. The 64-bank long run supports
about 8% lower cycles/CPU time, while the one-bank timing benefit is much smaller.
Whole-process generic cache-miss ratios are 0.828/0.801, but this event is not
a DRAM-byte counter and cannot establish memory-bandwidth utilization.

Four additional NT4 control processes at 64 banks retain the same arithmetic
on both paths. Their instruction ratio is 1.000003, cycle ratio 1.043, timed
CPU ratio 1.038 and wall ratio 0.99997. This exposes roughly 4% cycle/CPU drift
without an arithmetic change; timing results are not an idle-host calibration
or a precise attribution of the larger wall-time difference. NT4 production
includes the disabled compiler-dispatch check, so it is an arithmetic control,
not a claim of byte-identical code across the two objects.

All events reported 100% running time. Across twelve processes, 96 timed-batch
output checks passed bit-exactly after 98,304 calls, in addition to all-bank
initial checks. The result establishes fewer retired instructions for NT2,
with conditional timing improvement; it does not establish full-model tok/s,
90% floating-point peak or 90% theoretical memory bandwidth. No production
source/default changed during this screen. Raw logs, counter CSVs, commands,
binary/archive hashes and summaries are in `results/20261020/fp4-pair2-counters/`.


### Q2 NT2 paired-half conversion screen (2026-10-20)

A scratch candidate extends the existing direct-Q2 paired FP16 conversion from
NT1 to NT2. It changes only the compile-time condition selecting half_pair for
the adjacent d/dmin fields; dot accumulation, Q3, Q2 lookup mode, canonical
rules and other widths remain unchanged. The production NT1 option retains
its NT1-only meaning; this candidate was not integrated.

A primary-host K4096/R128 GU row screen used twelve alternating pairs of 64
calls, random packed scales/codes, fixed finite half scales and Q8_K bytes
including -128 with valid block sums. Candidate/baseline process CPU ratios:

| Core | Tokens | Banks | CPU ratio | Pair wins |
|---|---:|---:|---:|---:|
| 3 | 2 | 1 | 0.987 | 12/12 |
| 3 | 2 | 64 | 0.948 | 9/12 |
| 11 | 2 | 64 | 0.967 | 9/12 |
| 11 | 2 | 1 | 0.986 | 8/12 |
| 3 | 1, unchanged | 64 | 1.010 | 4/12 |
| 11 | 1, unchanged | 64 | 1.007 | 5/12 |

All 144 batch checks passed after 9,216 calls, plus initial bank parity checks.
The candidate's NT2 row CPU improvement is 1–5%, not a model speedup.

A fresh renamed baseline and candidate were then linked into the synthetic
Q2/Q3 expert-service probe, selecting the candidate only for Q2 NT2 GU. H4096,
F2048, seven experts with mixed two-token routes, eight banks, 15 workers plus
host, three tasks per participant and 20 ms spin were retained. One shared
expert has NT2 and six experts have NT1. Four processes tested ordinary /
canonical / canonical / ordinary dataflow, each with twenty alternating pairs
of sixteen calls and warmup after selection. Input quantization is excluded.

Ordinary dataflow wall ratios were 1.066 and 1.074 (only 6/20 candidate wins
each); CPU ratios were 1.041 and 1.077. Canonical dataflow wall ratios were
0.998 and 0.987, with CPU ratios 1.000 and 0.978. All 160 batch output/reduction
checks passed after 2,560 calls. These results reject general service rollout:
ordinary mixed routing repeatedly regressed despite the smaller row-level CPU
time. Only one expert uses the changed width, so its local gain is diluted;
the measurements do not isolate why the larger latency regression occurred.

Host load was uncontrolled. No worker-node, full-model or GPU test was run,
and no source/default or simulator rate changed. Source, exact commands,
object/archive hashes and paired raw samples are in `results/20261020/q2-half2/`.


### Three-token FP4 gate/up pairing screen (2026-10-20)

A scratch NT3 helper extends the same half-wise GU sharing used by the NT2
option to six accumulators. Each output preserves its FMA order. The existing
NT1/NT2 helpers remain unchanged. This experiment does not change the production
NT2 option's width or introduce a new production option.

The row screen compiles current GCC baseline and renamed candidate with NT1
and NT2 enabled on both sides. K4096/R128, raw packed bytes, fixed finite Q8
half scales and signed Q8 byte extremes, twelve alternating pairs of 64 calls,
clamp 7, global scales 0.001, and prefetch/other dispatches disabled are retained.

| Core | Tokens | Banks | Candidate/baseline CPU ratio | Pair wins |
|---|---:|---:|---:|---:|
| 3 | 3 | 1 | 0.975 | 10/12 |
| 3 | 3 | 64 | 0.968 | 10/12 |
| 11 | 3 | 64 | 0.945 | 8/12 |
| 11 | 3 | 1 | 0.974 | 12/12 |
| 3 | 2, unchanged | 64 | 1.003 | 5/12 |
| 11 | 4, unchanged | 64 | 1.005 | 2/12 |

NT3 CPU time decreases 2.5–5.5%, with 40/48 pair wins. All 144 timed-batch
checks passed after 9,216 calls, plus all-bank initial checks. This establishes
a row-level gain; no register-spill mechanism has been measured or asserted.

Real partial layers 3/4 were then tested with NT3, E1 / E3 / E3 / E1, four
banks, 15 workers plus host, owned NUMA rows, three tasks per participant,
fused hidden quantization and 20 ms spin. The selector changes only NT3 GU;
all down and other-width calls use baseline. Nine alternating pairs of sixteen
calls per geometry and separate 32-call CPU batches use the established protocol.

| Experts | Layer | Fused split wall ratio, first / repeat | Fused dataflow ratio, first / repeat |
|---|---:|---:|---:|
| 1 | 3 | 0.939 / 0.985 | 0.892 / 0.957 |
| 1 | 4 | 0.965 / 0.950 | 0.977 / 0.990 |
| 3 | 3 | 0.980 / 0.963 | 1.012 / 0.991 |
| 3 | 4 | 0.936 / 0.982 | 0.989 / 0.977 |

All eight split medians improve 1.5–6.4%. One-expert dataflow improves 1–11%,
while three-expert layer 3 is essentially unchanged. Fifteen of sixteen service
medians favor the candidate, but separate CPU-batch ratios span 0.702–1.278;
host load and unchanged down-phase variation prevent precise causal attribution.
Input quantization is excluded and the partial weights/inputs repeat.
All 288 latency batch checks plus 32 CPU-batch checks passed after 4,608 plus
1,024 calls. No production default, simulator rate or full-model throughput
claim changed. Broader API parity, worker qualification and backend integration
remain outstanding for NT3. Sources, commands, hashes and raw samples are in
`results/20261020/fp4-gu-pair3/` and `results/20261020/fp4-pair3-service/`.


### NT3 pairing: broader parity and worker qualification (2026-10-20)

The three-token candidate passed 400 broader baseline-parity cases on the
primary CPU. Coverage includes K64/128/256/2048/4096, eight random seeds,
raw packed FP4 codes/scales, Q8 including -128, finite positive/negative half
scales including signed zeros and subnormals, padded row strides, contiguous
or separate up buffers, full/partial row ranges and clamped/unclamped APIs.
Complete output buffers match, including untouched row regions. These are
comparisons with the existing kernel, not an independent scalar oracle or
coverage of all possible floating-point inputs.

A static probe then tested the same candidate on the Ryzen 9 3950X worker,
using complete expert blobs; all records confirm owned=0 and fusion off.
Real partial layers 3/4 ran E1 / E2 / E2 / E1 at NT3, four banks, 15 workers
plus host, three tasks per participant and 20 ms spin. The candidate changes
only NT3 GU. Each geometry has nine alternating pairs of sixteen calls plus
separate 32-call CPU batches. Input quantization is excluded, inputs/weights
repeat, and the miner was left running.

| Experts | Layer | Split wall ratio, first / repeat | Dataflow ratio, first / repeat |
|---|---:|---:|---:|
| 1 | 3 | 0.987 / 0.740 | 1.053 / 1.012 |
| 1 | 4 | 1.005 / 0.846 | 0.956 / 0.841 |
| 2 | 3 | 0.835 / 1.072 | 1.125 / 0.939 |
| 2 | 4 | 0.976 / 0.948 | 0.973 / 0.913 |

Layer-4 dataflow favors the candidate in all four cases. However, E1 layer-3
dataflow regresses 1–5%, with only 3/9 candidate wins in each repeat, and
E2 layer-3 results reverse. Large split and separate CPU-batch variation is
also present (CPU ratios 0.597–1.418). These loaded-host results do not establish
a stable general benefit, and large wins must not be attributed solely to the
3–5% primary row improvement. Keep NT3 scratch-only pending stronger evidence;
the production NT1/NT2 options and defaults are unchanged.

All 288 latency-batch and 32 CPU-batch checks passed bit-exactly after 4,608
and 1,024 calls. No GPU or full-model throughput/quality test ran; peak CPU
utilization remains unverified. Sources, hashes, commands and raw samples
are in `results/20261020/fp4-pair3-worker/`.


### Wider-first gate/up scheduling rejection (2026-10-20)

A scratch pool candidate stably sorts each owned-NUMA gate/up queue by descending
job token width. Down ordering, task IDs, node ownership, arithmetic, readiness
and reduction are unchanged. Sorting is disabled when grouped completion
callbacks are present. Both variants build the same temporary expert-index
vector; only candidate sorts it. This screen does not measure the cost of adding
that vector to production, and no production pool change was made.

The synthetic Q2/Q3 mixed-route probe was extended to move the sole NT2 expert
from the first job slot to the last. Stable expert IDs keep each job attached to
its original weights, activations and output slots, including bank rotation.
The same final output hash across all placements verifies that the workload's
numerical result is unchanged. The two-token expert remains one of seven;
the other six each receive one token. Existing production arithmetic is used
for both variants, with Q3 compiler dispatch and Q2 lookup disabled.

On the primary CPU, H4096/F2048, eight banks, 15 workers plus host, three tasks
per participant and 20 ms spin were held fixed. Four processes use shared expert
first / last / last / first. Each has twenty alternating pairs of sixteen calls,
with synchronous warmup after selection; input quantization is excluded.

| Shared-expert placement | Candidate/baseline wall ratio, first / repeat | Process CPU ratio, first / repeat |
|---|---:|---:|
| First, unchanged task order control | 1.0062 / 1.0002 | 1.0007 / 1.0003 |
| Last, wider-first candidate changes GU order | 1.0042 / 1.0069 | 1.0002 / 1.0004 |

The reordered cases won only 6/20 and 8/20 latency pairs. There is no observed
benefit to promote; retain the current ordering. The benchmark improvement is
coverage of expert-order sensitivity, which the old shared-first fixture alone
could not test. All 160 batch output/reduction checks passed bit-exactly after
2,560 calls, with the same output hash across both permutations.

Host load was uncontrolled and routes/inputs/weights repeat. No FP4, worker-node,
grouped-callback, GPU or full-model candidate run was performed; none is justified
as rollout qualification after this negative screen. CPU peak utilization is
not established. Sources, commands, hashes and raw samples are in
`results/20261020/pool-gu-width/`.


### Native-Q8 adapter for future partial distributed execution (2026-10-20)

The distributed CPU tree currently constructs `Fp4LocalCpuExpert` from
`src/kernels/cpu/fp4_local.cpp`. That is the W4A16/BF16 path, including its
explicit paired/fused/reuse selectors; it does not call the optimized native
NVFP4/Q8 kernels. Consequently, native row/service gains cannot be claimed as
speedups of the existing distributed CPU tree. The production native decoder
and that bounded tree runner are distinct execution paths.

`native_cpu_expert.hpp` now supplies an experimental `NativeFp4CpuExpert`
adapter for H4096/F2048 partial experts. It uses the shared C11 NVFP4 packing
rule and native format/global-scale tail, clamps GU at 10, quantizes inputs to
Q8 and executes the current split native pool path. A caller-provided persistent
pool is shared across experts; reference experts do not create additional
spinning teams. Calls must serialize access to the shared pool. One to sixteen
tokens are divided into balanced windows of at most eight. Invalid input shapes
and nonfinite activations are rejected. Output arithmetic is native FP4/Q8,
not the existing BF16 W4A16 reference.

Build the separate test from the repository root, providing the configured
native archive build and its GGML header directory:

```sh
make -f experiment/glm_fp4_local/Makefile cpu-native-adapter   NATIVE_BUILD=<native-build-directory> GGML_INCLUDE=<ggml-include-directory>
STRATA_NO_NVFP4_512=1 build-fp4-local/glm-fp4-native-adapter-test <partial-model-directory>
```

CPU-only tests linked both HIP- and CUDA-configured archives and passed on
real layer-3/4 expert-0 weights. Per backend, widths 1/2/3/4/7/8/9/15/16 passed
18 repeat checks and 130 per-token bitwise comparisons to native single-token
execution; six invalid-extent/nonfinite cases were rejected. This qualifies
windowing/repeat behavior, not an independent arithmetic oracle or W4A16 quality
equivalence. The Make target also built successfully. No GPU execution, SYCL
build, network exchange, performance measurement or full-model run was performed.

The adapter is not yet wired into the tree runner. That integration must be an
explicit arithmetic mode with a shared C11 handshake flag so W4A16 and native-Q8
peers cannot silently mix. The existing W4A16 backend, wire behavior and defaults
remain unchanged. Before claiming distributed improvement, compare quality
against W4A16 separately, exercise direct/relay transport with matching native
peers and measure complete service including input quantization. Evidence and
source/build hashes are in `results/20261020/native-adapter/`.


### Native-Q8 versus W4A16 quality and latency diagnostic (2026-10-20)

`native_cpu_quality.cpp` compares the new adapter with the existing fused,
weight-reusing W4A16 expert on real layer-3/4 expert-0 weights. Both receive
identical BF16-rounded synthetic inputs. Widths 1/2/4/8/16 use random inputs
with amplitudes 1, 0.001 and 64; a zero-input case runs at width 1. These are
32 arithmetic cases, not representative captured model activations or a
perplexity/quality evaluation. The W4A16 reference's normal width-1 case on
each layer additionally runs its complete FP64 per-projection bounds and
fused/separate equality check outside timing.

| Input pattern | Native versus W4A16 relative L2 error | Minimum cosine |
|---|---:|---:|
| Normal, amplitude 1 | 1.006–1.094% | 0.999940 |
| Clamp stress, amplitude 64 | 1.094–1.185% | 0.999930 |
| Tiny, amplitude 0.001 | 100% | 0 |
| All zero | 0 | 1, defined for two zero vectors |

The tiny-input native outputs are exactly zero in all ten cases, confirmed
by a separate norm-reporting run. W4A16 output norms range 1.608e-5–6.580e-5;
maximum absolute differences are 9.01e-7–1.192e-6. Thus the large relative error
has small absolute magnitude, but the adapter is not an exact W4A16 replacement.

A stage trace for one token identifies the immediate cause: all 128 input Q8
scale fields remain nonzero, and all 2,048 native hidden floats remain nonzero,
but all 64 hidden Q8_0 scale fields become zero. Maximum hidden magnitudes are
2.107e-6 and 2.694e-6 for layers 3/4. Normal and large-amplitude cases have no
zero hidden scales. The FP16 hidden-scale representation loses these tiny
blocks; this is not evidence of a new GU half-pair arithmetic bug. A scale-safe
hidden-quantization experiment or explicit fallback needs separate numerical
and performance qualification before claiming general replacement quality.
No production quantizer, scale rule or default was changed here.

Normal cases also collect nine alternating single-call pairs, with a warm call
before each measured call. Weight loading/packing and FP64 verification are
outside timing; input quantization/rounding, allocation and complete expert
execution are inside. Each backend has four participants. Native pool spin is
explicitly zero and OpenMP waiting passive to reduce idle-team interference;
thread placement is not matched, and host load is uncontrolled. Native/W4A16
paired median latency ratios for widths 1/2/4/8/16 are respectively
0.635/0.747/1.064/1.218/0.763 on layer 3 and
0.540/0.620/0.850/0.702/0.841 on layer 4. These are diagnostic hot-partial timings,
not a qualified backend speedup, full-model throughput or bandwidth-peak result.
All 180 timed calls matched their own backend's saved output bitwise.

Build with the explicit `cpu-native-quality` Make target, supplying NATIVE_BUILD
and GGML_INCLUDE as for the adapter test. Run the resulting
`build-fp4-local/glm-fp4-native-quality <partial-model-directory>`; optional
`quality-only` skips timing. The target built successfully. The tree runner,
wire protocol and W4A16 defaults remain unchanged. No GPU or network execution
ran. Raw quality/timing/norm/hidden-stage logs, source snapshots and hashes are
in `results/20261020/native-quality/`.


### Experimental scale-safe hidden Q8 conversion (2026-10-20)

A scratch native adapter uses the existing full-hidden transform callback to
rescale only small nonzero token vectors before Q8_0 quantization. If their
maximum absolute hidden value is below 2^-7, frexp determines a power-of-two
shift placing that maximum in [2^-7,2^-6). The original down kernel runs on
the rescaled hidden values; a per-token inverse shift is then applied to its
output. Zero vectors and vectors at/above the threshold are unchanged. The
format copy and callback live through the synchronous pool call, and shift
state resets for every balanced token window. No packed weights or global
scales are modified, and no production quantizer/default changed.

On the same real layer-3/4 expert-0 tiny-input cases, relative L2 error against
W4A16 falls from 100% to 0.969–1.031%; output norms are nonzero. Maximum absolute
error falls to 8.84e-9–1.25e-8. All 22 normal/large/zero cases remain bit-identical
to the unscaled native adapter. This recovers the tested whole-hidden underflow
case; it does not eliminate ordinary Q8 error, repair already-lost input Q8
scales, or guarantee preservation of arbitrarily tiny blocks within a token
whose overall peak exceeds the threshold. Full-model quality is still unknown.

The rescaled adapter also passed the nine-width 1/2/3/4/7/8/9/15/16 consistency
test on tiny inputs: 130 per-token bitwise comparisons with native single-token
execution, 18 repeat checks, and six invalid-input checks. These checks cover
window state reset and chunking; they are not an independent arithmetic oracle.

A nine-pair diagnostic compares rescaled and legacy native service, including
input quantization, scanning, output rescaling and allocations. Across layers
and widths, normal-case paired ratios are 0.985–1.031; tiny cases are 1.013–1.119.
All 360 timed calls match their own expected outputs. Tiny-case arithmetic is
deliberately different: this is the cost of recovering output lost by legacy
quantization, not an identical-output speed comparison. Four participants,
zero native park spin and passive OpenMP waiting are retained; host load and
thread placement are uncontrolled.

A second scratch version computes one power-of-two multiplier per token for
shifts up to 126 and uses per-value multiplication, retaining ldexp for larger
shifts. This avoids a library call per value in the common scaled case.
All 32 quality outputs are bit-identical to the first rescaled version, and
360 further timed-repeat checks pass. Nine of ten tiny-case latency medians
improve 2.4–6.6%; one width-16 case regresses 5.3%. Unchanged normal-case controls
range 0.974–1.061. Thus the multiplier version is promising, but loaded-host
measurements do not establish a general speedup. Extreme-exponent/fallback
coverage, worker testing and a qualified integration remain outstanding.

Neither version is connected to distributed execution or enabled in production.
GPU, SYCL, network and full-model tests were not run, and CPU peak utilization
is not established. Scratch source, commands, hashes, quality records, consistency
checks and paired timings are in `results/20261020/native-scale-safe/`.


### Optional hidden rescaling in the native adapter (2026-10-20)

The experimental native adapter now accepts `preserve_small_hidden`, false by
default. Selecting it uses `native_hidden_scale.hpp` with the multiplier path
and the large-shift ldexp fallback described above. This is confined to the
partial-model adapter: production quantizers, tree execution and defaults are
unchanged. The quality and consistency executables accept `preserve-small` to
select it. `quality-only` can be supplied together with that selector.

On the primary Threadripper 1950X, both HIP- and CUDA-configured CPU libraries
passed the adapter checks with the option off and on. Each combination checks
130 single-token comparisons, 18 repeated batches and six invalid inputs over
layers 3/4 expert 0 and token counts 1/2/3/4/7/8/9/15/16. A separate comparison
against the frozen pre-option adapter passed 36 bitwise batched comparisons
with the option disabled, covering both ordinary and tiny inputs.

Both CPU library builds passed 32 quality cases and explicit bitwise comparisons
for all 22 normal/large/zero cases against the unscaled adapter. Their printed
quality records agree exactly. Tiny-input relative L2 error versus W4A16 is
0.969–1.031%, compared with 100% from the all-zero unscaled output. Normal and
large input errors remain 1.006–1.094% and 1.094–1.185%, respectively. These are
synthetic activation tests with real partial weights, not model quality scores.

The standalone boundary test passes 4,809 cases, including 512 cases exercising
shifts above 126. It compares scaling against ldexp and checks bit-exact inverse
round trips, threshold neighbors, signed zero, state reset and nonfinite input
rejection. Floating-point FTZ/DAZ were disabled (MXCSR 0x1f80); the test rejects
other settings instead of changing the process floating-point mode. Reproduce:

```sh
make -f experiment/glm_fp4_local/Makefile cpu-native-scale-test
```

This resolves the earlier scratch experiment's extreme-exponent coverage gap.
Worker, GPU, SYCL, network and full-model checks remain outstanding. No new
speedup or CPU peak-utilization claim follows from these correctness checks.
The distributed runner still requires an explicit arithmetic-mode handshake
before native FP4/Q8 integration, and one shared CPU pool per process rather
than one spinning pool per cached expert. Logs, commands, source hashes and a
reproducible summary are in `results/20261020/native-scale-option/`.

### Native FP4/Q8 partial tree integration (2026-10-20)

The opt-in `tree-native-cpu` Make target builds a separate CPU-only tree runner
using the native adapter. It requires NATIVE_BUILD and GGML_INCLUDE as for the
native adapter tests. Each process owns one persistent pool shared by its local
and reference experts. `threads=N` includes the participating host and accepts
2..32 for this binary; the default is four. The native pool treats zero worker
count as automatic sizing, so one-thread requests are rejected explicitly.

All peers identify native Q8 arithmetic in the existing seven-word hello using
shared C11 feature constants. Optional `preserve-small` adds a second feature
bit and must agree across every edge. Native/W4A16 and rescaling mismatches were
both rejected before weight payloads. W4A16 paired/fused/reuse selectors are
rejected by the native binary. Existing W4A16 and GPU targets remain separate.

On Threadripper 1950X, three local CPU processes passed direct and relay UDP
execution using real layers 3/4, experts 0/1/2. Rescaling enabled passed 48 layer
checks per topology over eight scenarios and three repeats; disabled passed
16 per topology over four scenarios and two repeats. Every routed comparison
had zero relative L2 against the same native local reference. Repeated outputs
were bit-exact, and direct/relay dumps matched bytewise within each option state.
This reference validates distribution consistency, not independent numerical
accuracy; the adapter quality tests above compare against W4A16 separately.

```sh
make -f experiment/glm_fp4_local/Makefile tree-native-cpu \
  NATIVE_BUILD=CONFIGURED_CPU_BUILD GGML_INCLUDE=GGML_INCLUDE_DIRECTORY
STRATA_NO_NVFP4_512=1 STRATA_POOL_SPIN_US=0 \
  python3 tools/test_glm_fp4_cpu_tree.py PARTIAL_MODEL_DIR \
  --binary build-fp4-local/glm-fp4-tree-native-cpu \
  --output build-fp4-local/native-tree-local --extended --preserve-small
```

The native, W4A16, HIP and CUDA tree executables built. GPU execution and SYCL
were not tested. These loopback runs do not qualify cross-machine IB, full-model
quality, decode throughput or CPU peak utilization. Pool spin was zero and all
three processes shared one host, so these checks are not performance benchmarks.
The next integration check is the same partial workload over physical IB peers.
Logs, source hashes and compact evidence are in `results/20261020/native-tree/`;
raw output dumps remain in the ignored build directory.

The final default W4A16 tree object is byte-identical to a same-flags build of
the pre-integration source. Early comparisons differed from shared help text
and compiler scheduling of a new hello temporary; both were confined to the
native preprocessor branch before the final comparison.

### Native CPU tree over physical IB (2026-10-20)

The native FP4/Q8 runner now passes both physical topologies across the primary
Threadripper 1950X, Ryzen 3950X worker and Zen 5 worker. Each topology completed
48 layer checks using layers 3/4 and experts 0/1/2, eight routing scenarios,
three repeats, four CPU participants per process and hidden rescaling enabled.
Every distributed output matched the primary's native reference with zero
relative L2, and repeated residual outputs were bit-exact. The complete
3,293,184-byte output dumps from physical relay, physical direct and the prior
loopback test are identical (SHA256
`3e8790bfe14117cae5d5d266a61b93afce8ddbedd1b7e952d80be6e3ff242f29`).
All three processes reported IB transport, and both workers completed every
request without reloading weights. Direct mode used the second IB port on the
primary and third node for their peer edge. No GPU was used or miner changed.

The primary's default executable referenced sqrtf at GLIBC_2.43, unavailable
on the older middle worker (glibc 2.39). The tested portable executable links
C++/compiler support statically while keeping libc and IB loading dynamic;
a small link wrapper binds sqrtf to its existing GLIBC_2.2.5 symbol. Its highest
required glibc symbol version is 2.38. Both workers verified the same executable
SHA256 before running. The portable executable's distributed output matched
the original loopback executable bytewise. This compatibility shim is frozen
with the experiment; it does not alter production math or build defaults.

The launcher now accepts `--preserve-small` and passes it to every role.
Connection settings remain in ignored local configuration. The existing
`--direct`, `--extended` and `--dump-outputs` options select the tested modes.
All nodes selected AVX2, disabled experimental alternative NVFP4 selectors and
used zero pool spin; exact settings and binary/source hashes are recorded in
`results/20261020/native-tree-ib/`. Raw dumps remain in the ignored build tree.

These are distribution correctness results with partial weights. Timings include
reference work, startup and deliberate 250 ms sleeps per layer, so elapsed time
must not be converted to decode tok/s. Full-model quality, sustained service
latency, CPU/GPU overlap and 90% theoretical CPU utilization remain unverified.
The next performance measurement should time only the distributed critical path,
with reference checks outside that interval, then compare against same-day
W4A16 service under the same routing and CPU-thread settings.

### Distributed native/W4A16 service diagnostic (2026-10-20)

A scratch runner times dispatch through the final returned expert output on the
physical three-node direct topology. Routing-vector construction, reduction,
reference computation, residual updates and the deliberate layer sleeps are
outside the interval. Timed W4A16 cache experts disable FP64 verification, while
untimed reference experts retain it. W4A16 uses fused weight reuse. Native uses
the FP4/Q8 adapter with small-hidden protection. All nodes use four CPU
participants, AVX2 and passive/zero-spin waiting. Production defaults are
unchanged.

Four completed runs in native/W4A16/W4A16/native order passed 192 layer checks,
all with zero routed error against their own backend's local reference. Output
dumps are bit-identical across the two runs of each backend; native and W4A16
are different arithmetic and are not claimed bit-identical to each other.
The first repeat is excluded from timing medians, leaving four observations per
layer/scenario across two processes. Layer 3 uses identical starting inputs;
layer 4 receives backend-specific residual outputs.

Selected layer-3 median service times, in milliseconds:

| Expert token counts on the three nodes | Native FP4/Q8 | W4A16 reuse | Native/W4A16 |
| --- | ---: | ---: | ---: |
| 1 / 0 / 0 | 1.316 | 3.630 | 0.362 |
| 3 / 2 / 2 | 2.682 | 4.991 | 0.537 |
| 16 / 16 / 16 | 10.077 | 15.494 | 0.650 |
| 0 / 0 / 3 | 2.350 | 4.895 | 0.480 |
| 10 / 11 / 11 | 6.864 | 13.988 | 0.491 |
| 0 / 0 / 0 | 0.300 | 0.309 | 0.971 |

Across both layers, active-routing median ratios range 0.362–0.650. Separate
forward and reverse run comparisons also favor native in every active cell,
with ranges 0.248–0.787 and 0.431–0.604 respectively. These are diagnostics,
not a qualified general speedup: host load and placement are uncontrolled,
weights are a small reused set, and untimed W4A16 FP64 work consumes far more
CPU than the native reference, potentially changing thermal/cache conditions.
Idle medians are roughly 0.22–0.31 ms and do not establish an arithmetic gain.
No full-model tok/s or theoretical peak utilization follows from these numbers.

An initial W4A16 attempt was terminated by the launcher's 300-second deadline
during untimed verification. Its frozen logs are excluded. The same binary and
checks completed with a 600-second bound; no correctness checks were removed.
Raw samples, completed and interrupted logs, source, build commands, output
hashes and per-run medians are in `results/20261020/tree-service-ib/`.

### Persistent direct dispatch candidate (2026-10-20)

A scratch native runner replaces two per-request std::async thread launches on
the primary with two persistent sleeping service threads, one per IB context.
Each context still has exactly one owner. Workers, native kernels, weight
packing, wire format and routing are unchanged. Packaged tasks propagate
exceptions through their futures; shutdown drains an already queued task and
joins the thread before transport finalization. A standalone test passed 2,001
result/exception checks plus post-exception reuse and destruction with a queued
task. This helper remains experimental and is not in the production engine.

On the same three-node direct topology, baseline/candidate/candidate/baseline
runs passed 192 layer checks with zero routed error. All four full output dumps
are byte-identical. After excluding each first repeat, four samples per cell
produce candidate/baseline median ratios 0.733–0.979 for active routing. Selected
layer-3 baseline to candidate service medians are 1.365 to 1.140 ms for 1/0/0
tokens, 3.068 to 2.615 ms for 3/2/2, and 11.575 to 10.138 ms for 16/16/16.
Idle medians fall from 0.215 to 0.130 ms and 0.223 to 0.091 ms across the two
layers.

The separate-run comparisons are less consistent than those combined medians:
active-cell ratios range 0.558–1.194, including reversals. One idle forward
comparison regresses to 3.126 while its reverse comparison improves to 0.435.
Host load, placement and the small sample count remain limitations; pooled
medians alone do not qualify this candidate. No default is changed. A subsequent
same-process paired dispatch selection should test whether the benefit survives
closer interleaving, especially the low-work and idle controls. No full-model
tok/s, GPU overlap or theoretical peak-utilization claim is made. Frozen source,
helper checks, raw samples, per-run medians and output hashes are in
`results/20261020/tree-persistent-dispatch/`.

### Paired direct dispatch and opt-in runner flag (2026-10-20)

A same-process probe issues nine baseline/persistent pairs for each layer and
alternates order by pair, repeat, scenario and layer. Both arms use identical
inputs and the same native worker executables. Every pair compares all three
returned expert buffers bitwise before local reference work or residual updates.
Two physical three-node direct runs passed 864 pair checks (1,728 dispatches)
and 96 layer-reference checks. Both complete output dumps match the preceding
native runner. Worker logs confirm 864 requests per run without weight reloads.

Timing statistics exclude repeat zero and pair zero, leaving 32 pairs per
layer/scenario across the two processes. Median paired candidate/baseline ratios
range 0.899–0.994 on active routing, with 20–30 candidate wins out of 32 per cell.
For layer 3, single-primary-expert service medians fall from 0.785 to 0.695 ms
(paired ratio 0.906), 3/2/2 token service from 1.906 to 1.829 ms (0.950), and
16/16/16 from 11.118 to 10.436 ms (0.957). Ratios are medians of individual
pairs, so they need not equal the ratio of the displayed marginal medians.
Idle service falls from 0.118/0.121 ms to 0.028/0.030 ms across the two layers,
with paired ratios 0.233/0.246 and 32/32 wins in each. One active per-process
cell remains marginally slower (ratio 1.003); this does not establish a universal
gain. Full raw samples and per-order/per-run summaries are preserved.

The bounded runner now accepts opt-in `persistent-dispatch` on a direct-mode
primary. The Python launcher exposes it as `--persistent-dispatch` and passes
it only to the primary. Relay mode, worker use and repeated selectors are
rejected. Two persistent sleeping service threads each own one IB context;
shutdown joins them before transport finalization. Default per-request dispatch
and the production decoder are unchanged. The flag does not alter arithmetic
or the handshake and requires no worker rebuild.

The helper's standalone test covers 2,001 result/exception checks, reuse after
an exception, shutdown with a queued task and deterministic full-queue rejection:

```sh
make -f experiment/glm_fp4_local/Makefile tree-dispatch-test
```

CPU, HIP and CUDA tree binaries built successfully. GPU execution and SYCL
were not tested; the GPU tree kernels are unchanged. This remains a partial-
weight four-participant CPU result with uncontrolled host load. It does not
establish model tok/s, CPU/GPU overlap or 90% theoretical utilization. Source,
commands, raw paired samples and hashes are in `results/20261020/tree-dispatch-paired/`.

Integrated default and persistent modes each passed another 48 physical-IB layer
checks. Their complete output dumps are byte-identical to each other and to
the pre-integration native result. Three invalid selector checks also passed.

### CPU identity correction for the third node (2026-10-20)

A fresh CPU/boot audit confirms three distinct running hosts: the primary
reports Threadripper 1950X and both workers report Ryzen 9 3950X, each with 32
logical CPUs. Earlier references here to a "Zen 5 worker" used the supplied
machine alias rather than a verified CPU model. They must not be treated as
Zen 5 architecture qualification. The current generic hardware record is in
`results/20261020/tree-dispatch-primary16/hardware.json`; endpoints and boot IDs
are not retained. During the audit the primary's one-minute load was 3.82,
versus 30.82 and 28.44 on the workers. These loaded-worker measurements do not
qualify idle-host bandwidth or theoretical-peak utilization.

### Paired dispatch with sixteen primary CPU participants (2026-10-20)

The same paired binary ran twice with sixteen participants on the primary and
four on each worker. Worker teams were held fixed because their measured load
averages were near 30; no miner was stopped. Another 864 bit-exact pair checks
(1,728 dispatches) and 96 layer-reference checks passed. Complete output dumps
match the earlier four-participant native runs. Warmup exclusions and the 32
post-warmup pairs per layer/scenario are unchanged.

For the 1/0/0 token case, default to persistent service medians are 0.471 to
0.367 ms on layer 3 and 0.442 to 0.331 ms on layer 4. Median paired ratios are
0.781 and 0.775, with 30/32 and 32/32 wins. With 3/2/2 tokens, ratios are 0.919
and 0.887. With 16/16/16 tokens, ratios are 0.985 and 0.989: reduced primary
dispatch overhead does little when loaded remote workers dominate service.
Idle median service falls from 0.129/0.116 ms to 0.031/0.030 ms, with paired
ratios 0.247/0.246 and 32/32 wins in each layer. One large-batch per-run cell
still regresses (1.018), so these are measured case-specific improvements.

Comparisons between the four- and sixteen-participant runs are separate-process
observations, not paired thread-scaling measurements. These reused partial
weights can remain in cache; their service times do not establish DRAM bandwidth
or peak FLOPS efficiency. The result supports the opt-in dispatch flag for
small-request CPU runs, while pointing to worker computation and memory service
for the larger routed batches. No simulator default or full-model tok/s estimate
was changed. Raw pairs, generic CPU/load metadata, source and summaries are in
`results/20261020/tree-dispatch-primary16/`.

### Primary cache sensitivity and unstable bandwidth reference (2026-10-20)

A new primary CPU sweep uses real FP4 layers 3/4 at NT1, one or three experts,
fifteen workers plus the participating host, one versus sixteen independently
allocated copies of the packed weights, and zero park spin. The order is
E1/B1, E1/B16, E3/B1, E3/B16, E3/B16, E3/B1, E1/B16, E1/B1. Split, dataflow,
NUMA-owned rows and fused hidden-quantization modes retain the existing native
benchmark's correctness checks. The 96 output rows represent 864 timed-batch
checks plus 96 CPU-batch checks, covering 16,896 timed calls. Output hashes
agree across bank counts and execution modes for each expert-count/layer case.
Input quantization is outside these intervals.

For split execution, the ratio of the two-run median at sixteen banks to that
at one bank is 1.44–1.73 for a complete single-expert blob, 1.36–1.43 for its
NUMA-owned unfused rows, and 1.33–1.34 for owned fused rows. With three experts,
those ranges are 1.12–1.24, 1.08–1.36 and 1.01–1.12 respectively. These are
separate-process sensitivity comparisons, not paired optimization gains.
NUMA-owned rows generally outperform complete blobs here. Copies contain the
same model bytes, so bank rotation approximates a larger working set rather
than a real routing trace or distinct-expert population.

A separate same-process Q2/Q3 test changes only the bank selector: baseline
reuses bank zero and the comparison arm rotates sixteen physical copies, using
the same production kernels. Two dense NT1/E2 runs show median paired penalties
of 9.63% and 9.27%, with rotation slower in 19/20 and 17/20 pairs. Two mixed
NT2/E7 runs show only 2.05% and 0.44% penalties, with 12/20 and 13/20 slower
pairs. All 160 final sample checks pass across 2,560 timed calls, with matching
output hashes within each route pattern. Q2/Q3 weights and inputs are synthetic;
these are cache-sensitivity results, not a new kernel improvement.

The read-bandwidth reference is not stable enough to calibrate utilization.
Sixteen physical-core workers with first-touch buffers and five measured passes
report 33.96 then 82.95 GB/s before/after the FP4 sweep using 1 GiB per array.
Later 1 GiB / 4 GiB / 1 GiB runs report 53.95 / 37.43 / 52.83 GB/s. All validation
passes, but the cause of the drift is unresolved. These are algorithmic bytes
per second, not memory-controller counters, and installed memory speed remains
unverified. Dividing kernel packed-byte rates by one selected reference would
not prove 90% of theoretical memory bandwidth. No simulator rate or production
default is changed. Commands, source snapshots, raw logs, hashes and summaries
are in `results/20261020/cache-roof/`.

### Opt-in NUMA rows in the native adapter (2026-10-20)

`NativeFp4CpuExpert` now offers explicit `prepare_numa(pool)` and an optional
fourth `run` argument, `use_numa`, false by default. Preparation uses the existing
strictly bound NumaTensor allocator for gate, up and down rows and occurs outside
timed service. All three arenas are prepared before replacing prior state.
Repeated preparation for the same pool topology is idempotent; unsupported
single-node pools return false. Opt-in execution rejects missing preparation
or a different pool topology. Preparation and execution must be serialized.
The complete original blob remains available for global scales and baseline
comparison, so enabling this partial-model experiment approximately doubles its
packed-weight memory. This is not a memory-ready full-model representation.

HIP- and CUDA-configured CPU libraries each passed 156 scoped checks using real
layers 3/4 expert 0: normal and tiny activations, hidden rescaling off/on, widths
1/2/3/4/7/8/9/15/16, repeats, idempotent preparation and invalid pool/preparation
checks. A separate frozen-adapter comparison passed all 72 disabled-option
output comparisons bytewise. No native kernel or engine default changed.

On Threadripper 1950X with fifteen workers plus the host, two same-process paired
benchmarks compared complete blobs against prepared local rows. The second
process reversed layer order, bank-count order and first-arm order. Preparation
is excluded; input quantization, vector allocations and hidden rescaling remain
inside each eight-call service batch. All 1,600 timed-batch output checks pass
across 12,800 calls, plus 1,600 checked warmups. Both arms use copies of identical
weights, not a real expert-routing trace.

At NT1 with sixteen rotating banks, layer-3 marginal medians are 0.434 ms for
complete blobs and 0.270 ms for local rows; layer 4 gives 0.445 and 0.293 ms.
Median paired ratios are 0.638 and 0.663, with 40/40 local-row wins in each layer.
The ratios are medians of per-pair ratios, not ratios of the marginal medians.
One-bank NT1 gains are smaller: paired ratios 0.952/0.954, with 31/40 and 32/40
wins. NT2 rotating-bank ratios are 0.956/0.896. Larger batches are mostly near
parity and can regress, so there is no blanket selection rule or default change.

Build the opt-in check with `cpu-native-numa`, supplying NATIVE_BUILD and
GGML_INCLUDE as for the other native adapter targets. Run
`build-fp4-local/glm-fp4-native-numa-test PARTIAL_MODEL_DIR`; optional `bench`
adds the paired timing sweep. A two-node CPU pool is required. The option is
not yet exposed by the distributed runner. GPU, SYCL and full-model validation
were not performed; these results do not prove 90% theoretical utilization or
model tok/s. Evidence, legacy snapshots, source hashes and raw paired samples
are in `results/20261020/adapter-numa/`.

### Releasing the duplicate NUMA adapter blob (2026-10-20)

After explicit NUMA preparation, the experimental adapter can now call
`release_original_blob()`. The owned-row CPU paths obtain all weights from the
three supplied projection shards and read the original blob only for NVFP4
global scales. Release copies those four floats into a 32-byte scale buffer,
then swaps the packed vector with an empty vector. Each subsequent owned run
uses a local NativeFmt copy with a nonzero 16-byte scale-tail offset into that
buffer. The ordinary weight geometry and all global-scale values are unchanged;
no production kernel, shared format or serialized pack is modified.

Release is explicit and irreversible for that adapter instance. It requires
prepared rows, is idempotent, and makes complete-blob execution an error.
Preparation for the same pool topology remains idempotent after release;
changing topology is rejected because the source rows are no longer retained.
Prepare, release and run must be serialized by the caller. Instances that do
not call release retain the earlier default behavior and baseline-comparison
capability.

On the primary Threadripper 1950X, HIP- and CUDA-configured CPU libraries each
passed 432 bit-exact comparisons and eight invalid-transition checks. Inputs
cover zero, normal, tiny and large amplitudes, hidden rescaling off/on, both
real layers 3/4 expert 0, and widths 1/2/3/4/7/8/9/15/16. Complete-blob, original
NUMA and released NUMA outputs agree, including repeats and balanced windows.
A frozen pre-change adapter additionally matched 144 default-path outputs
bytewise. For each tested expert, retained packed-vector capacity falls from
14,155,792 bytes to zero, with 32 bytes of compact scale storage. This measures
owned allocation capacity, not whole-process RSS. Temporary preparation still
requires the old and new rows to coexist until release.

Build `cpu-native-compact` with NATIVE_BUILD and GGML_INCLUDE and run
`build-fp4-local/glm-fp4-native-compact-test PARTIAL_MODEL_DIR`. The Make target
and its test also pass. This removes the duplicate full blob from the partial
adapter, but does not qualify a full-model allocation plan: per-expert mappings,
other runtime buffers and distributed integration remain to be addressed.
No new throughput claim follows from these correctness/storage checks. GPU,
SYCL, network and full-model runs were not performed for this release option.
Evidence, frozen legacy source, build commands and hashes are in
`results/20261020/adapter-compact/`.

### Compact NUMA storage in the physical tree runner (2026-10-20)

The native CPU primary now accepts opt-in `numa-compact`; the Python launcher
exposes it as `--primary-numa-compact`. Before networking, it requires a primary
role and a pool with pinned workers on both NUMA nodes. Each cached and local
reference expert prepares NUMA rows, releases its original blob, and verifies
zero retained blob capacity before serving requests. The arithmetic handshake
and worker executables are unchanged. Complete-blob primary behavior remains
the default. The flag is independent of persistent direct dispatch and also
works in relay mode.

Three physical-IB runs passed 144 layer-reference checks: default direct,
compact direct, and compact relay. Each uses layers 3/4, experts 0/1/2, eight
routing scenarios and three repeats. The primary has sixteen CPU participants;
both workers retain four. The direct runs also use persistent dispatch. All
three 3,293,184-byte output dumps match the pre-integration native result
bytewise. Worker logs confirm unchanged CPU mode, complete request sequences
and no weight reloads. Four invalid configurations were rejected before
networking: a worker selector, a primary pool lacking both nodes, a repeated
selector, and a W4A16 binary.

For an already configured native CPU tree invocation, select
`--primary-cpu-threads 16 --primary-numa-compact`; keep the existing matching
`--preserve-small` setting on all peers. Add `--persistent-dispatch` only for
`--direct`. Actual connection settings remain in ignored local configuration.
The one-token rotating-weight gain measured above motivates this option; these
integration runs establish correctness, not an additional speedup. Larger-token
NUMA results were mixed, and whole-model memory usage remains unqualified.

Native CPU, W4A16 CPU, HIP and CUDA tree binaries built. No GPU execution, SYCL
validation, full-model throughput or peak-utilization qualification was performed.
The production decoder defaults are unchanged. Source hashes, build log,
invalid-selector checks and per-node IB evidence are in
`results/20261020/tree-numa-compact/`.

### Paired Q2/Q3 huge-page screen (2026-10-20)

A scratch Q2/Q3 pool probe compares ordinary 4-KiB arenas with huge-page-aligned,
MADV_HUGEPAGE arenas in the same process. Both arms use strict NUMA binding,
the same kernels, pool, input activations, bank sequence and sixteen rotating
copies of the same packed weights. There are fifteen workers plus host, three
tasks per participant and zero park spin. Q2 LUT, Q23 prefetch and alternate Q3
compiler selectors are disabled. Each process alternates arm order over twenty
pairs of sixteen calls. Dense NT1/E2 and mixed NT2/E7 routes run in both split
and dataflow modes, then repeat in reverse geometry order.

All 320 timed-batch output checks pass across 5,120 calls, including routed
sums. Process-wide AnonHugePages confirms nonzero candidate backing, but coverage
is partial and varies across processes: 282–384 MiB in dense runs and
806–1,052 MiB in mixed runs. These counts are not a per-buffer placement audit;
ordinary arenas explicitly request MADV_NOHUGEPAGE. Huge-page alignment also
rounds the arena extents, so this screen includes that allocation difference.

Median paired huge/ordinary latency ratios from the two runs are:

| Routes / execution | First run | Reverse-order run |
| --- | ---: | ---: |
| Dense NT1, split | 1.010 | 0.993 |
| Dense NT1, dataflow | 0.985 | 0.987 |
| Mixed NT2, split | 1.006 | 1.022 |
| Mixed NT2, dataflow | 1.007 | 1.007 |

The small dense-dataflow benefit does not support a general huge-page default.
Mixed routing regresses in both execution modes, and dense split reverses.
Phase counters show no consistent gate/up or down improvement across the split
runs. Retain the existing ordinary-page default. The decoder's existing
STRATA_GLM_PACK_HUGE option is unchanged; no production code was modified by
this screen. These synthetic partial CPU results do not qualify real model
routing, GPU overlap, full-model tok/s or 90% theoretical bandwidth/FLOPS.
Source, build commands, process page counts, raw samples, paired phase ratios
and binary hash are in `results/20261020/q23-huge-paired/`.

### Paired Q2/Q3 software prefetch with identical-work controls (2026-10-20)

On the primary Threadripper 1950X, a scratch kernel object replaces the cached
prefetch selector with one atomic selector read per row entry. Both arms execute
the same object: baseline selects zero, candidate selects 1,024 bytes. Arithmetic
is unchanged. The pool has fifteen workers plus host, three tasks per participant,
zero park spin, strict NUMA-owned ordinary pages and sixteen rotating synthetic
weight banks. Q2 LUT and alternate Q3 compiler selectors are off. Input
quantization is excluded. Each process alternates arm order over twenty pairs.

The initial screen uses sixteen calls per arm, repeats geometries in reverse
order, and passes 320 timed-batch output and routed-sum checks over 5,120 calls.
Median paired candidate/baseline ratios are:

| Routes / execution | First run | Reverse-order run |
| --- | ---: | ---: |
| Dense NT1, split | 1.006 | 0.960 |
| Dense NT1, dataflow | 0.978 | 0.980 |
| Mixed NT2, split | 0.973 | 0.970 |
| Mixed NT2, dataflow | 0.975 | 0.968 |

A second binary sets both selector arms to zero. Its dense-dataflow control
ratios are 0.972 and 1.003, so the apparent dense improvement is not qualified.
Mixed-dataflow controls are 0.9994 and 0.9998. A longer confirmation uses
sixty-four calls per arm, in candidate/control/control/candidate process order:

| Process | Actual candidate / baseline prefetch bytes | Median paired ratio | Candidate wins / 20 |
| --- | --- | ---: | ---: |
| First candidate | 1024 / 0 | 0.9771 | 15 |
| First control | 0 / 0 | 0.9947 | 13 |
| Second control | 0 / 0 | 1.0020 | 9 |
| Second candidate | 1024 / 0 | 0.9720 | 19 |

The controls and confirmation pass another 320 timed-batch checks over 12,800
calls, with identical output hashes within each process. The longer mixed-route
runs support a roughly 2–3% primary CPU latency reduction for this geometry.
Use the existing STRATA_Q23_PREFETCH=1024 option for further mixed-route testing;
retain the default of zero. This is not a worker-node recommendation or a
full-model speedup measurement. Host load was uncontrolled, synthetic copies
are not a real routing trace, and neither GPU execution nor theoretical peak
utilization was tested. No production source or simulator rates changed.
Sources, commands, binary hashes, raw samples and control ratios are retained
in `results/20261020/q23-prefetch-paired/`.

### Q2/Q3 prefetch on the loaded worker CPU (2026-10-20)

The paired 0/1024-byte prefetch probe and identical-work control were rebuilt
with three workers plus host for the Ryzen 9 3950X worker. Its initial load
average was 30.11/30.24/30.36; miners and other jobs remained running. Both
binaries use the same production CPU archive and arithmetic as the primary
screen, static C++ runtime and an older sqrtf ABI wrapper for worker libc
compatibility. Deployed binary hashes were verified before every process.

Mixed NT2/E7 dataflow uses sixteen rotating synthetic banks (about 1.02 GB),
sixty-four calls per arm and twenty alternating pairs. The single-NUMA worker
uses complete weight blobs. Input quantization is outside timing. Process order
is candidate/control/control/candidate; all 160 timed-batch output and routed-sum
checks pass over 10,240 calls.

| Process | Actual candidate / baseline prefetch bytes | Median paired ratio | Candidate wins / 20 |
| --- | --- | ---: | ---: |
| First candidate | 1024 / 0 | 0.8011 | 20 |
| First control | 0 / 0 | 1.0042 | 8 |
| Second control | 0 / 0 | 1.0003 | 10 |
| Second candidate | 1024 / 0 | 0.7973 | 20 |

The approximately 20% latency reduction supports STRATA_Q23_PREFETCH=1024 as
an opt-in setting for this loaded worker's mixed-route CPU service. It is larger
than the primary's 2–3% gain; those different CPU/team/NUMA conditions do not
isolate the cause. Controls are nearly neutral. Default settings remain zero,
and dense routes, the second worker, FP4 kernels and full-model decoding are
not qualified by this result. No GPU work or theoretical bandwidth/FLOPS
utilization measurement ran. Compact evidence and reproduction sources are in
`results/20261020/q23-prefetch-worker/`; connection settings stay local.

### Same-process FP4 prefetch screen with controls (2026-10-20)

A scratch copy of the current AVX2 FP4 kernel replaces its cached prefetch
selector with an atomic distance read. Both arms share the same kernel object,
including the current NT1/NT2 gate/up half-pair options. The baseline selects
zero; candidate selects 512, 2048 or zero bytes. The source default is 2048,
so comparison against zero is not a direct comparison against the default.
No production kernel or default was changed.

The primary Threadripper 1950X uses fifteen workers plus host, strict NUMA-owned
ordinary pages, fused split execution, three real partial experts from layers
3/4 and sixteen rotating copies of their weights. NT1 and NT2 each run twenty
alternating pairs of sixteen calls, followed by a separate thirty-two-call
CPU-time batch per arm. Input quantization is excluded. Spin is zero, tasks
per participant are three, and alternate Clang/narrow/offset paths are off.
Processes run 512/2048/0/0/2048/512 bytes in that order.

Median paired candidate/zero-prefetch latency ratios, first/repeat:

| Layer / tokens | 512 bytes | 2048 bytes | Identical-work controls |
| --- | --- | --- | --- |
| 3 / 1 | 0.9647 / 0.9837 | 1.0052 / 0.9815 | 1.0033 / 1.0020 |
| 3 / 2 | 0.9802 / 1.0039 | 0.9785 / 0.9791 | 0.9978 / 0.9968 |
| 4 / 1 | 0.9821 / 0.9806 | 0.9704 / 0.9796 | 0.9977 / 1.0144 |
| 4 / 2 | 0.9850 / 0.9812 | 0.9806 / 0.9950 | 1.0064 / 1.0036 |

512-byte prefetch improves all four single-token cells by about 1.6–3.5%,
winning 16–18 of twenty pairs per cell. This is a modest primary single-token
candidate, with control variation reaching 1.4% in one cell. NT2 layer 3 fails
to repeat the benefit. The 2048-byte result is also mixed. Neither distance
supports a general default change or an extrapolation of the worker Q2/Q3
20% improvement to FP4.

All 960 timed-batch and 48 separate CPU-batch output/routed-sum comparisons pass
across 16,896 calls. This is partial real-weight CPU evidence under uncontrolled
host load, with repeated expert contents rather than a full routing trace.
Worker CPUs, GPU execution, full-model quality/tok/s and theoretical peak
utilization were not tested here. Sources, exact build/settings, raw ordered
samples, controls and binary hash are in `results/20261020/fp4-prefetch-paired/`.

### FP4 prefetch directly against the current default (2026-10-20)

The preceding paired probe was rebuilt with baseline distance 2048 bytes.
Candidate distance is 512 for two processes; two intervening controls select
2048 in both arms. All geometry, pool settings, real partial weights, rotating
banks and validation remain as in the preceding screen. Process order is
512/2048/2048/512. This directly tests the proposed tuning change against the
current source default rather than against prefetch off.

Median paired candidate/baseline latency ratios, first/repeat:

| Layer / tokens | 512 versus 2048 | 2048 versus 2048 controls |
| --- | --- | --- |
| 3 / 1 | 0.9989 / 1.0024 | 0.9762 / 1.0037 |
| 3 / 2 | 0.9881 / 0.9987 | 0.9839 / 0.9877 |
| 4 / 1 | 0.9862 / 1.0021 | 0.9984 / 1.0051 |
| 4 / 2 | 0.9914 / 0.9975 | 0.9925 / 1.0147 |

No repeatable benefit exceeds identical-work variation. Retain the existing
2048-byte default; the earlier 512-versus-zero result does not justify a
512-byte recommendation over the default on this primary CPU. This decision
does not alter the separate Q2/Q3 worker recommendation, which uses a different
kernel and default. All 640 timed-batch plus 32 CPU-batch output/routed-sum
checks pass over 11,264 calls. Host load was uncontrolled. This adds no
worker/GPU/full-model throughput or theoretical peak qualification. Source,
commands, controls, ordered samples and binary hash are retained in
`results/20261020/fp4-prefetch-default/`.

### FP4 physical-core scaling with balanced NUMA placement (2026-10-20)

A primary-only diagnostic gives each team equal participants on the two NUMA
nodes. It validates the audited physical-core list before starting, reserves
CPU 0 for the host and orders workers 8,1,9,2,10,3,11,4,12,5,13,6,14,7,15.
Thus four participants use two physical cores per node, and so on. This avoids
conflating smaller teams with the default CPU order's initial single-node
placement. It changes only the diagnostic's explicit pool placement.

The real partial FP4 geometry is unchanged from the preceding screen: three
experts, layers 3/4, NT1/2, sixteen rotating weight banks, strict NUMA-owned
ordinary pages, fused split, three tasks per participant and zero spin. Both
paired labels use 2048-byte prefetch. Team sizes run 4/8/12/16/16/12/8/4 in
separate processes. Each process retains twenty alternating identical-work
pairs per layer/token geometry, sixteen calls per arm, and separate CPU batches.

Combined medians in ms, including both runs and both identical-work labels:

| Participants | Layer 3 NT1 | Layer 4 NT1 | Layer 3 NT2 | Layer 4 NT2 |
| --- | ---: | ---: | ---: | ---: |
| 4 | 2.021 | 2.030 | 3.169 | 3.269 |
| 8 | 1.100 | 1.100 | 1.740 | 1.795 |
| 12 | 0.788 | 0.788 | 1.193 | 1.231 |
| 16 | 0.737 | 0.754 | 1.150 | 1.140 |

Scaling from four to sixteen participants yields about 2.7x for NT1 and
2.8x for NT2, rather than the 4x core-count ratio. The last four participants
reduce combined median latency by only about 4–7%. One layer-4 NT1 sixteen-
participant process has median 1.218 ms while its repeat is 0.733 ms; retain
this anomaly and do not use the combined table as proof of a stable optimum.
Forward/reverse ordering reduces one source of drift but is not same-process
paired thread-count testing. Host load was uncontrolled.

At sixteen participants, logical packed bytes divided by NT1 latency give
56–58 GB/s. This counts each packed expert once, not actual memory-controller
traffic. The previously measured approximately 82 GB/s bound-read reference
is a different workload and was not rerun concurrently here; their ratio is
not theoretical utilization. The plateau motivates distinguishing coordination,
load balance and dot-kernel throughput rather than assuming more threads or
prefetch tuning will reach 90% peak. No production default changes follow.

All 1,280 timed-batch and 64 CPU-batch output/routed-sum checks pass across
22,528 calls. Input quantization is excluded. No worker, GPU, full-model tok/s
or theoretical-peak qualification was performed. Raw samples, per-process
medians, identical-work paired ratios, CPU placement, source and binary hash
are in `results/20261020/fp4-thread-scaling/`.

### Paired FP4 down-projection task granularity (2026-10-20)

Using the preceding balanced sixteen-participant placement, a scratch probe
changes the existing pool task-count setter only between completed calls.
Baseline is three tasks per participant. Candidate processes use 1/6/3/3/6/1,
with the middle processes serving as identical-work controls. Kernels, 2048-byte
prefetch, three real partial experts, layers 3/4, NT1/2, sixteen rotating banks,
NUMA-owned fused split and zero spin remain fixed. Each geometry has twenty
alternating pairs of sixteen calls and separate CPU batches.

Source inspection limits the interpretation: fused gate/up always has 24 tasks
for this geometry (256 rows per task), independent of this setting. The down
projection has 48 baseline tasks; rounding into expert/node partitions gives
18 tasks for candidate one and 96 for candidate six. This experiment therefore
changes down granularity, not gate/up granularity. Earlier 128-row fused GU
experiments are recorded above and were not rerun here.

Median paired candidate/baseline latency ratios, first/repeat:

| Layer / tokens | One versus three | Six versus three | Three versus three controls |
| --- | --- | --- | --- |
| 3 / 1 | 1.1242 / 1.1405 | 0.9692 / 0.9919 | 1.0176 / 1.0076 |
| 3 / 2 | 1.1814 / 1.1816 | 0.9573 / 0.9579 | 0.9796 / 1.0108 |
| 4 / 1 | 1.1439 / 1.1652 | 0.9654 / 0.9790 | 0.9966 / 0.9949 |
| 4 / 2 | 1.1126 / 1.1673 | 0.9603 / 0.9876 | 0.9992 / 1.0035 |

Larger down tasks are consistently worse by 11–18%. Smaller tasks improve
all eight candidate cells by 0.8–4.3%, with 13–20 wins per twenty pairs.
The layer-3 NT2 gain repeats at about 4.2%; several other gains approach the
control variation. Phase arithmetic-mean ratios put down improvement at
roughly 7–8% for all four NT2 candidate cells, but unchanged GU timings also
vary, so this is not precise causal attribution. One NT1 down-phase repeat
is effectively neutral. Retain three as the default; six is an opt-in tuning
candidate for this three-expert primary CPU geometry, not a worker or general
routing recommendation. Reducing task count does not resolve the observed
sixteen-core plateau.

All 960 timed-batch and 48 CPU-batch output/routed-sum checks pass across
16,896 calls. Input quantization is excluded; host load is uncontrolled.
No production source/default, simulator rate, GPU or full-model throughput
claim changed. Sources, commands, raw phase/ordered samples and binary hash
are in `results/20261020/fp4-task-paired/`.

### Paired 128-row FP4 fused gate/up chunks for three experts (2026-10-20)

A scratch pool object changes only fused split gate/up partitioning. Baseline
uses 256-row chunks (24 tasks for three experts); candidate uses 128-row chunks
(48 tasks). Both remain aligned to complete 32-value NVFP4 hidden Q8_0 blocks.
The selector excludes canonical arithmetic and non-NVFP4 projections. Down
projection remains at 48 tasks, with the existing three-tasks-per-participant
setting. The dataflow path is unchanged and not exercised by this probe.

The benchmark uses the preceding balanced sixteen-participant primary CPU
placement, real partial layers 3/4, three experts, NT1/2, sixteen rotating banks,
strict NUMA-owned ordinary pages, zero spin and 2048-byte prefetch. Process order
is candidate/control/control/candidate. Controls select 256 rows in both arms.
Each geometry has twenty alternating pairs of sixteen calls; separate CPU
batches and bitwise output/routed-sum checks remain enabled.

Median paired candidate/baseline latency ratios, first/repeat:

| Layer / tokens | 128 versus 256 rows | 256 versus 256 controls |
| --- | --- | --- |
| 3 / 1 | 0.9300 / 0.9045 | 0.9969 / 1.0173 |
| 3 / 2 | 0.9499 / 0.8272 | 0.9978 / 1.0061 |
| 4 / 1 | 0.9544 / 0.8970 | 1.0089 / 1.0008 |
| 4 / 2 | 0.8698 / 0.8516 | 0.9803 / 0.9925 |

All eight candidate cells improve, by 4.6–17.3%, while control median ratios
stay within about 2% of unity. Candidate wins range from 12 to 20 of twenty
pairs, so variability remains significant. The result supports further testing
of smaller fused GU chunks for three experts, where the current fixed task
count underfills the sixteen-core team. It does not isolate load balance from
cache/queue effects, nor prove this beats the existing unfused path. Earlier
one-expert prototypes above did not consistently beat unfused execution;
that comparison remains necessary before integrating a general selection rule.

All 640 timed-batch and 32 CPU-batch output/routed-sum comparisons pass over
11,264 calls. This scratch-only experiment changes no production defaults,
serialized layout or simulator rates. Input quantization is excluded, CPU load
is uncontrolled, and worker CPUs, other token widths, dataflow, GPU and
full-model throughput/quality are untested here. The 90% peak target remains
unverified. Source, commands, raw ordered samples, controls and binary hash
are in `results/20261020/fp4-gu-chunk-paired/`.

### 128-row FP4 fusion compared directly with unfused split (2026-10-20)

The scratch chunk prototype now compares against the existing owned unfused
split path. Candidate uses 128-row fused split; controls use unfused split in
both arms. Both use the same current kernels, 2048-byte prefetch, balanced
sixteen-physical-core placement, three tasks per participant, zero spin and
sixteen rotating copies of real partial weights. One- and three-expert cases
cover layers 3/4 and NT1/2/4/8. Candidate E1/E3, control E3/E1, control E1/E3,
and candidate E3/E1 process order repeats both conditions. Each geometry has
twenty alternating pairs of sixteen calls and separate CPU-cost batches.

Ranges of median paired candidate/baseline ratios across the two layers and
two processes (range endpoints are not paired speed comparisons):

| Experts / tokens | 128-row fused / unfused | Unfused / unfused controls |
| --- | --- | --- |
| 1 / 1 | 0.9865–1.0424 | 0.9913–1.0359 |
| 1 / 2 | 0.9703–1.0824 | 0.9854–1.0021 |
| 1 / 4 | 0.9637–0.9946 | 0.9831–1.0208 |
| 1 / 8 | 0.9711–1.0526 | 0.9837–1.0083 |
| 3 / 1 | 0.9853–0.9959 | 0.9864–1.0103 |
| 3 / 2 | 0.9871–0.9982 | 0.9911–1.0187 |
| 3 / 4 | 0.9799–0.9929 | 0.9710–1.0439 |
| 3 / 8 | 0.9636–0.9919 | 0.9959–1.0253 |

Three-expert cells consistently favor the candidate, but the 0.2–3.6% gains
are much smaller than the preceding comparison against 256-row fusion and
not clearly separated from control variation. One-expert cells include
4.2%, 8.2% and 5.3% regressions. Do not integrate a general smaller-chunk
selection rule on this evidence. The current unfused path remains a strong
baseline; avoiding its serial hidden quantization alone is not a large win
in these tests. Keep the prototype as a diagnostic and leave production
selection/defaults unchanged.

All 2,560 timed-batch and 128 CPU-batch output/routed-sum comparisons pass over
43,008 calls. This extends partial-weight numerical coverage through eight
tokens; it is not full-model quality validation. Input quantization is excluded,
host load is uncontrolled, and no worker CPU, dataflow, GPU or peak-utilization
qualification ran. Exact per-cell paired ratios, raw samples, sources,
commands and binary hash are in `results/20261020/fp4-fused-v-unfused/`.

### Native FP4 adapter activation allocation reduction (2026-10-20)

The experimental native CPU adapter now uses one contiguous call-local
activation buffer, sized for the largest balanced window in the request.
Previously it constructed eight separate buffers plus the outer vector and a
temporary fill buffer, even for one token. Window boundaries, quantization,
output allocation, finite checks, small-hidden rescaling and kernel selection
are unchanged. The buffer remains local to run(); no shared mutable workspace
or new caller lifetime requirement is introduced. Each active region is
quantized before the synchronous pool consumes it.

A frozen pre-change adapter was compared against the new implementation using
both HIP- and CUDA-configured CPU archives. Each build passes 720 bitwise
comparisons and eight rejection checks. Coverage includes layers 3/4, zero/
normal/tiny/large inputs, rescaling off/on, token counts 1/2/3/4/7/8/9/15/16,
complete blobs, owned NUMA rows and released original blobs with compact scale
tails. These are CPU executions; no GPU kernels ran. Both native tree runners
also rebuilt successfully against their respective CPU archives. SYCL is not
available and was not built; the changed adapter is an experimental CPU header.

Host-thread operator-new counts measured during run(), including its returned
output buffer, are eleven allocations before and two after for every sampled
width. Requested heap bytes for representative requests are:

| Tokens | Before bytes | After bytes |
| --- | ---: | ---: |
| 1 | 55,744 | 20,736 |
| 2 | 72,128 | 41,472 |
| 8 | 170,432 | 165,888 |
| 9 | 186,816 | 169,216 |
| 16 | 301,504 | 296,960 |

These are requested allocation bytes, not allocator-resident memory or a
hardware bandwidth measurement. The one-token request removes nine allocations
and 35,008 requested bytes; no service-latency or full-model tok/s gain is claimed
without timing. All sampled outputs match the frozen implementation exactly.
Source snapshots, allocation counters, build commands and validation logs are
in `results/20261020/adapter-act-storage/`. Model arithmetic, wire protocol and
production engine defaults remain unchanged.

### Whole-adapter timing after activation allocation reduction (2026-10-20)

A scratch class contains both the frozen old run method and the current run
method, sharing the exact same weight storage, pool and input. The candidate
therefore does not get a separate packed-weight cache placement. Both methods
include finite scans, activation/output allocation, input quantization, native
split execution, small-hidden protection and output validation. Small-hidden
protection is enabled; normal-amplitude real inputs are used. Both complete
blobs and owned NUMA rows run on sixteen primary physical participants, with
2048-byte prefetch and zero spin. Repeated one-expert requests use layers 3/4
and token widths 1/2/8/16.

Four processes run candidate/control/control/candidate; controls call the old
method in both arms. Each geometry alternates twenty pairs of sixteen complete
calls, with warmup outside the timed batch. Checks compare final batch output
bitwise with the frozen old method. Median paired new/old latency-ratio ranges
across both layers and repeats are:

| Weight access / tokens | New / old | Old / old controls |
| --- | --- | --- |
| Complete blob / 1 | 0.9736–0.9913 | 0.9830–1.0042 |
| Complete blob / 2 | 0.9762–1.0116 | 0.9908–1.0035 |
| Complete blob / 8 | 0.9744–0.9833 | 0.9852–1.0296 |
| Complete blob / 16 | 0.9953–1.0165 | 0.9821–1.0117 |
| Owned NUMA / 1 | 0.9706–0.9900 | 0.9878–1.0081 |
| Owned NUMA / 2 | 0.9761–0.9986 | 0.9918–1.0174 |
| Owned NUMA / 8 | 0.9860–0.9938 | 0.9867–1.0123 |
| Owned NUMA / 16 | 0.9789–1.0208 | 0.9920–1.0183 |

Single-token calls favor the change in all eight cells, by about 0.9–2.9%,
although some gains overlap identical-work variation. Larger windows do not
show a general improvement; sixteen-token results are mixed. Retain the
allocation reduction on its measured resource savings and exact-output checks,
without attributing a full-model throughput gain to these small timings.
These tests reuse one expert, exclude network service and do not establish
cold-weight bandwidth or theoretical utilization. Host load is uncontrolled.

All 2,560 timed-batch output comparisons pass over 40,960 timed calls. No GPU
execution or full-model quality/performance validation ran. The combined source,
raw pairs, commands, controls and binary hash are retained in
`results/20261020/adapter-act-timing/`.

### Q2/Q3 prefetch confirmation on the second worker (2026-10-20)

The second worker independently reports Ryzen 9 3950X, 32 logical CPUs and
initial load average 28.00/28.01/28.03. It runs the exact same hash-verified
four-participant candidate/control binaries as the first-worker screen. Miners
remain running. Mixed NT2/E7 dataflow uses sixteen rotating synthetic banks,
sixty-four calls per arm and twenty alternating pairs, in candidate/control/
control/candidate process order. The single-NUMA path uses complete blobs;
input quantization is excluded.

| Process | Actual candidate / baseline prefetch bytes | Median paired ratio | Candidate wins / 20 |
| --- | --- | ---: | ---: |
| First candidate | 1024 / 0 | 0.8348 | 20 |
| First control | 0 / 0 | 1.0012 | 9 |
| Second control | 0 / 0 | 1.0039 | 8 |
| Second candidate | 1024 / 0 | 0.7991 | 20 |

The 16.5–20.1% reduction supports STRATA_Q23_PREFETCH=1024 as an opt-in setting
for this loaded worker's tested mixed-route CPU service, alongside the prior
first-worker result. Both controls are nearly neutral. All 160 timed-batch
output/routed-sum checks pass over 10,240 calls. These results do not qualify
other routing patterns or an idle-worker configuration, and they do not imply
an equivalent improvement for FP4 kernels or full-model decode. Defaults are
unchanged. Generic hardware/load metadata, source, binary hashes and raw pairs
are in `results/20261020/q23-prefetch-worker2/`; connection details remain local.

A fresh primary DIMM-speed read still required interactive authentication;
configured memory speed and theoretical bandwidth remain unverified. No global
settings were changed to obtain it. The measured prefetch gains are service
latency results, not proof of the 90% theoretical-bandwidth/FLOPS target.

### Production-archive Q2/Q3 prefetch validation (2026-10-20)

The worker probe now links the unchanged production CPU archive, without the
scratch Q2/Q3 selector object. The harness selector is a no-op; prefetch is read
by the production cached STRATA_Q23_PREFETCH environment path. The executable
and archive hashes are recorded, and the worker executable hash is checked
before each process. Four CPU participants, mixed NT2/E7 routes, sixteen
rotating copied banks, sixty-four calls per batch and running miners match the
prior first-worker geometry. Input quantization remains excluded.

Prefetch settings run in separate processes as 0/1024/1024/0. Each process
contains twenty alternating identical-work label pairs, useful as noise
controls but not as within-process prefetch comparisons.

| Process | Prefetch bytes | Median service ms | Identical-work paired ratio |
| --- | ---: | ---: | ---: |
| First disabled | 0 | 6.2646 | 0.9979 |
| First enabled | 1024 | 4.9971 | 1.0024 |
| Second enabled | 1024 | 4.9972 | 1.0057 |
| Final disabled | 0 | 6.2370 | 1.0001 |

The enabled/disabled ratio of process medians is 0.7994 (about 20.1% lower
latency), confirming that the production option delivers the gain previously
seen with the scratch selector. Both disabled endpoints and both enabled
processes agree closely, while within-process controls remain nearly neutral.
This supports the existing opt-in worker setting, not a default change for
all CPUs or route patterns. No decoder speedup is inferred from partial service.

All 160 timed-batch output/routed-sum checks pass across 10,240 calls, with one
output hash across all four processes. Host load is uncontrolled; the synthetic
expert bank sequence is not a real model routing trace. No GPU execution,
full-model throughput or peak-utilization claim follows. Reproduction source,
archive/binary hashes, settings and raw results are in
`results/20261020/q23-production-prefetch/`. Connection settings remain local.

### FP4 nibble-index rearrangement rejected (2026-10-20)

A distinct scratch unpack candidate rearranges packed-byte indices before the
FP4 lookup. It broadcasts the sixteen input bytes, duplicates each eight-byte
half within its target lane, shifts/blends to select low/high nibbles, masks
indices and performs one lane-local 256-bit table shuffle. The result already
has GGML order, avoiding the prior candidate's final 64-bit lane permutation.
Packed K32 retains the existing implementation. Gate/up half pairing and all
arithmetic/FMA order remain those of the current production archive.

An independent scalar checker passes 8,192 cases covering every byte value at
every position in both layouts. The paired row probe uses K4096/R128, raw scale
and code bytes, signed Q8 including -128, clamp seven and global scales 0.001.
Primary CPU cores 3 and 11 run separately, with one and sixty-four weight banks,
twelve alternating pairs of sixty-four calls. Prefetch, narrow/offset and Clang
selection are off. Median candidate/baseline process-CPU ratios across core/bank
configurations are:

| Tokens | Ratio range | Candidate wins / 48 pairs |
| --- | --- | ---: |
| 1 | 1.0923–1.1108 | 0 |
| 2 | 1.0454–1.0519 | 2 |
| 4 | 1.0434–1.2248 | 2 |

Reject this rewrite: no single-token pair wins, and multi-token results also
regress. A lower source-level instruction count is not sufficient evidence of
faster execution; these timings do not attribute the loss to a specific port,
latency or register-pressure cause. Host load was uncontrolled, including one
larger four-token regression. All 288 timed batches pass bitwise checks after
18,432 calls, plus initial checks of every bank. No production source, default,
layout or simulator rate changes. No GPU, SYCL or full-model qualification ran.
Sources, build commands, scalar checker, raw pairs and binary hash are in
`results/20261020/fp4-index-unpack/`.

### Three-node IB validation of reduced adapter allocations (2026-10-20)

The updated native CPU runner was built with the existing older-libc sqrtf
wrapper and static C++ runtime, deployed under new temporary filenames, and
hash-verified on both workers. All three nodes execute the updated adapter.
The primary uses sixteen CPU participants; each worker uses four. Miners remain
running, and only partial layers 3/4 with experts 0/1/2 are loaded.

Three extended runs cover direct ordinary weights, direct compact NUMA weights,
and relay compact NUMA weights. Direct runs enable persistent request dispatch;
compact NUMA selection is primary-only. Small-hidden protection is enabled on
all peers. Every process reports IB transport, and all three runs pass 48 layer
checks each, including repeat-bit-exact validation. Their complete 3,293,184-byte
output dumps match the frozen pre-allocation-change SHA256:

`3e8790bfe14117cae5d5d266a61b93afce8ddbedd1b7e952d80be6e3ff242f29`

This closes distributed partial-weight regression coverage for the allocation
change. It does not measure sustained throughput: the validation runner includes
reference work and deliberate inter-layer waits. No full-model quality/tok/s,
GPU execution or theoretical utilization claim follows. No network protocol or
production decoder default changed. Build provenance, selected validation-log
excerpts and summaries are in `results/20261020/adapter-act-tree/`. Full raw logs,
output dumps and connection settings remain in the ignored local build tree.
The summary script was corrected for an already-present topology field and a
privacy-check self-match; the successful physical runs were not repeated.

### Vectorizable native-adapter finite validation (2026-10-20)

The experimental CPU adapter now classifies activation/output finiteness with
an integer reduction over IEEE binary32 exponent bits. Every exponent field of
255 is rejected, covering both infinities and all NaN payloads. Signed zeros,
subnormals and maximum finite values remain accepted. The helper asserts the
binary32 representation and uses std::bit_cast, avoiding aliasing violations.
No fast-math or additional ISA requirement is introduced. GCC's ordinary O3
x86-64 build vectorizes the integer loop with SSE2. Unlike floating-point
classification, this helper does not raise FP exceptions for signaling NaNs;
it still rejects those requests before quantization. That validation behavior
is deliberate and is not a claim of identical floating-point exception flags.

A first branchless std::isfinite reduction failed to vectorize and took about
1.40x the early-exit baseline time, so it was rejected. The integer candidate
was measured on primary CPU3 with finite vectors, twenty alternating pairs of
4,096 scans. Representative median times per scan:

| Elements | Early-exit baseline ns | Integer reduction ns |
| --- | ---: | ---: |
| 4,096 | 1,873 | 425 |
| 8,192 | 3,706 | 831 |
| 32,768 | 14,172 | 3,254 |
| 65,536 | 28,454 | 6,564 |

Median paired ratios are 0.225–0.231, about 4.3–4.5x faster for the scan alone.
These are repeated finite-buffer CPU measurements under uncontrolled host load,
not whole-adapter or model speedups. Invalid values near the beginning can favor
the old early-exit loop; the new loop intentionally scans the full range.

The permanent native_finite_test passes 68,545 classification checks spanning
all exponent values, both signs, selected zero/minimal/quiet-bit/maximal
mantissas, randomized bit patterns, unaligned starts, empty input and vector
tails. Run it with `make -f experiment/glm_fp4_local/Makefile cpu-native-finite-test`.
Both HIP- and CUDA-configured CPU archives pass 720 frozen-adapter bitwise
comparisons and eight compact-storage rejections each. Expanded adapter tests
pass 230 checks per backend/rescale setting (920 total), including positive/
negative infinities and quiet/signaling NaNs at the first/last element of
1/8/9/16-token requests. Both native tree runners rebuilt successfully. No GPU
execution, physical-IB rerun, SYCL build, full-model output or peak-utilization
qualification was performed for this finite-scan change.

The helper, adapter/test snapshots, raw scan timings, build commands and
validation logs are in `results/20261020/adapter-finite/`. Arithmetic for valid
inputs and the native wire protocol are unchanged; production decoder defaults
are unaffected.

### Whole-adapter timing of integer finite validation (2026-10-20)

The whole-call paired harness now compares the frozen adapter immediately before
the finite-scan change with the current adapter. Both already use contiguous
activation allocation, so this isolates finite classification. A combined class
shares exactly the same expert weight storage, input and persistent pool between
methods. Calls include finite scans, allocation, input quantization, native
split execution, small-hidden protection and output validation. Both complete
blobs and owned NUMA rows run on sixteen primary participants, with normal
inputs, 2048-byte prefetch and zero spin.

Layers 3/4 and NT1/2/8/16 each run twenty alternating pairs of sixteen calls.
Process order is candidate/control/control/candidate; controls use the old
method in both arms. Ranges of median paired latency ratios across the two
layers and repeats are:

| Weight access / tokens | Integer / scalar checks | Scalar / scalar controls |
| --- | --- | --- |
| Complete blob / 1 | 0.9700–0.9953 | 0.9932–1.0056 |
| Complete blob / 2 | 0.9755–0.9878 | 0.9920–1.0083 |
| Complete blob / 8 | 0.9636–0.9934 | 0.9868–1.0172 |
| Complete blob / 16 | 0.9645–0.9692 | 0.9959–1.0172 |
| Owned NUMA / 1 | 0.9712–0.9931 | 0.9997–1.0209 |
| Owned NUMA / 2 | 0.9821–1.0002 | 0.9939–1.0028 |
| Owned NUMA / 8 | 0.9704–0.9849 | 1.0043–1.0407 |
| Owned NUMA / 16 | 0.9585–0.9832 | 0.9954–1.0260 |

Thirty-one of thirty-two candidate cells favor the change; the remaining
owned NT2 cell is effectively tied (1.0002). Sixteen-token gains range from
1.7–4.2%. Smaller-request gains are modest and some overlap the observed control
variation. The result supports the integrated scan optimization but not a
precise universal speedup. Both implementations reuse one expert; these are
not cold-weight bandwidth or model-routing measurements. CPU load was
uncontrolled and network service is outside this harness.

All 2,560 timed-batch output comparisons pass across 40,960 timed calls. No GPU,
full-model quality/tok/s or theoretical peak-utilization qualification ran.
Combined source, helper/archive hashes, raw pairs, controls and build commands
are in `results/20261020/adapter-finite-timing/`.

### Batched FP4 activation-scale conversion screen (2026-10-20)

The AVX2 Acts constructor converts Q8_0 half scales once per row-chunk call.
A scratch candidate gathers eight strided half fields into one vector and
converts/stores eight floats at a time, retaining scalar tails. It changes no
weight layout, activation ownership, arithmetic/FMA order or public API. This
is a simpler alternative to adding a prepared-activation cache with new lifetime
requirements.

An independent scalar h2f comparison passes 1,048,626 values, covering every
16-bit half pattern (including zeros, subnormals, infinities and NaNs), token
widths 1/2/4/8 and block counts 1/3/4/64. This includes vector-loop tails and
checks converted float bits rather than approximate equality.

The paired GU screen uses primary CPU3/11, K4096/R128, one/sixty-four banks,
raw weight codes/scales, signed Q8 including -128, clamp seven, prefetch off
and twelve alternating pairs of sixty-four calls. Median candidate/baseline
process-CPU ratios span 0.9934–1.0047 at NT1, 0.9980–1.0083 at NT2 and
0.9973–1.0044 at NT4. Win counts range from two to eight of twelve pairs.
All 288 timed batches pass bitwise checks after 18,432 calls, plus initial
checks of every bank.

There is no consistent row-service improvement. Do not integrate this rewrite
or infer a useful service gain from eliminating repeated half conversions alone.
Host activity was uncontrolled; no complete-expert, worker, GPU, SYCL,
full-model or peak-utilization qualification ran. Production source/defaults
and simulator rates remain unchanged. Candidate source, scalar checker,
commands, raw pairs and binary hash are in
`results/20261020/fp4-act-scales/`.

### Compiler screen after NT2 FP4 half-pair integration (2026-10-20)

The current AVX2 source, including both NT1 and NT2 gate/up half-pair helpers,
was compiled with Clang 21.1.8 and symbol-renamed beside the GCC production CPU
archive. This revisits the NT2 path after its arithmetic restructuring; older
compiler screens had only the NT1 half-pair definition. No source algorithm or
production dispatch changed. Existing NT4 Clang selection is disabled for this
comparison, so NT4 results are not a new optimization over that opt-in path.

The paired row probe uses primary physical cores 3/11, K4096/R128, one and
sixty-four banks, raw scale/code bytes, signed Q8 including -128, clamp seven,
prefetch off, and twelve alternating pairs of sixty-four calls. Initial median
candidate/baseline process-CPU ratio ranges are:

| Tokens | Clang / GCC range | Candidate wins / 48 pairs |
| --- | --- | ---: |
| 1 | 1.0762–1.0873 | 0 |
| 2 | 0.9853–1.0165 | 39 |
| 4 | 0.9574–0.9702 | 47 |

The NT2 four-case screen was repeated in reverse bank/core order. Its ratios
are 0.9891 and 0.9908 for sixty-four banks on cores 11/3, and 0.9902 and 0.9849
for one bank on cores 11/3. Repeat win counts are 12/8/6/10 of twelve. Thus the
initial NT2 regression does not repeat, but the apparent benefit is only about
1–1.5% and some pair counts are marginal. Retain GCC NT1; treat NT2 Clang as a
small row-level candidate requiring whole-expert evidence before adding a
selector. Do not infer model throughput from these results.

All 384 timed batches pass bitwise output checks after 24,576 calls, in addition
to initial per-bank checks. Host activity was uncontrolled. No complete-expert,
worker CPU, GPU, SYCL or theoretical-utilization qualification ran. Current
production code/defaults and simulator rates remain unchanged. Exact compiler
versions, commands, source, repeated pairs and binary hash are in
`results/20261020/fp4-half2-compiler/`.

### Current NT2 FP4 Clang candidate in complete expert service (2026-10-20)

A scratch wrapper dispatches only NT2 gate/up calls to the current Clang object;
NT1, down projection and K32 stay with GCC. Both compiler objects use the same
current source and half-pair definitions. The wrapper selects variants only
between completed batches. Separate control processes force GCC in both arms.
Production dispatch and defaults remain unchanged.

Real partial layers 3/4 run one and three experts, NT1/2, sixteen rotating banks,
balanced sixteen-physical-core placement, NUMA-owned unfused dataflow, three
tasks per participant, zero spin and 2048-byte prefetch. Each geometry alternates
twenty pairs of sixteen calls, with separate CPU-cost batches. Process order is
candidate E1/E3, control E3/E1, control E1/E3, candidate E3/E1. Input quantization
is excluded. NT1 is an unchanged-kernel control in every process.

Median paired candidate/baseline ratio ranges across two layers and repeats:

| Experts / tokens | Candidate-labelled runs | Forced-GCC controls |
| --- | --- | --- |
| 1 / 1, unchanged | 0.9952–1.0056 | 0.9658–1.0147 |
| 1 / 2 | 0.9928–1.0150 | 0.9913–1.0187 |
| 3 / 1, unchanged | 0.9996–1.0058 | 0.9965–1.0021 |
| 3 / 2 | 0.9964–1.0080 | 0.9950–1.0039 |

Six of eight NT2 candidate cells regress slightly; the two improvements are
small and overlap control variation. The preceding approximately 1–1.5% row
benefit does not survive complete expert service in this geometry. Do not add
an NT2 compiler selector on this evidence. Retain the current GCC path and the
separately qualified existing NT4 option.

All 1,280 timed-batch plus 64 CPU-batch output/routed-sum checks pass across
22,528 calls. Host activity was uncontrolled. No worker CPU, GPU, SYCL,
full-model tok/s or theoretical-utilization qualification ran. Raw pairs,
source, build commands, control results and binary hash are in
`results/20261020/fp4-half2-service/`.

### Q2/Q3 prefetch cache-hint comparison (2026-10-20)

A scratch pair of current Q2/Q3 kernel objects compares T0 and T1 prefetch hints
at the same 1024-byte distance. Both ordinary Q2 gate/up and Q3 down use the
selected object; arithmetic and weight layout are unchanged. Explicit expanded
Q3 paths remain on baseline. The same-process wrapper switches only between
completed batches; controls force T0 in both arms.

Primary CPU dataflow uses fifteen workers plus host, strict NUMA-owned ordinary
pages, sixteen rotating synthetic banks, three tasks per participant and zero
spin. Q2 LUT and alternate Q3 compiler selectors are disabled. Dense NT1/E2 and
mixed NT2/E7 processes run candidate/control cases then repeat in reverse order.
Each has twenty alternating pairs of sixty-four calls. Input quantization is
excluded.

| Pattern | T1 / T0, first and repeat | T0 / T0 controls |
| --- | --- | --- |
| Dense NT1/E2 | 0.9934 / 0.9980 | 1.0021 / 0.9701 |
| Mixed NT2/E7 | 0.9982 / 0.9971 | 0.9993 / 1.0000 |

There is no useful demonstrated hint improvement: mixed routing differs by only
0.2–0.3%, and dense controls show more variation than the candidate gain. Keep
T0 and the previously measured opt-in distance; this does not justify another
production selector or extrapolation to worker CPUs. The architectural hint
name does not itself prove where hardware placed the prefetched lines.

All 320 timed-batch output/routed-sum checks pass across 20,480 calls. Host
activity was uncontrolled. No worker, GPU, full-model or theoretical-utilization
qualification ran. Production code/defaults and simulator rates are unchanged.
Sources, commands, raw pairs, controls and binary hash are in
`results/20261020/q23-prefetch-hint/`.

### Calibrated retired-FLOPS counters for FP4 GU (2026-10-20)

The local perf event catalog exposes fp_ret_sse_avx_ops.all as raw CPU event 3,
umask 0xff. A scratch diagnostic groups user-only cycles, instructions and
this event with perf_event_open. Counters are enabled only around the measured
work, excluding weight generation, initial output checks and printing. Each
case runs three times on primary physical CPU3, then CPU11 on the other NUMA
node. Event groups report equal enabled/running times (no multiplexing loss).

Before measuring the kernel, eight independent 256-bit float accumulators run
one million FMA iterations. Expected FMA work is 128,000,000 scalar operations,
followed by 64 scalar reduction adds. All six calibrations report exactly
128,000,064 retired FP operations. The reference sustains about 15.992/15.994
counted FLOPs per user cycle on CPUs 3/11. This is a measured saturated-FMA
reference, not certification of the processor's theoretical peak.

The production GCC GU kernel uses K4096/R128, current NT1/NT2 half pairing,
2048-byte prefetch, finite weight scales 8–126, signed Q8 including -128 and
clamp seven. A one-bank setup contains 576 KiB of packed weights; sixty-four
banks contain 36 MiB. Each counter window runs 1,024 calls. Retired-FP/cycle
ranges across both cores and bank counts, using per-case medians:

| Tokens | Retired FP operations / cycle | Fraction of measured pure-FMA reference |
| --- | ---: | ---: |
| 1 | 2.294–2.333 | 14.3–14.6% |
| 2 | 2.833–2.859 | 17.7–17.9% |
| 4 | 2.104–3.095 | 13.2–19.4% |

The NT1 and NT2 results reproduce closely across nodes and bank counts. Their
larger-bank cycle penalties are small in this single-core row test, supporting
instruction throughput rather than DRAM bandwidth alone as the next hypothesis
to investigate. It does not prove a particular execution-port bottleneck. NT4
has substantial outliers: CPU11 one-bank median is 445.93 ms versus 295.82 ms
at sixty-four banks, and CPU3 has one 446.39-ms larger-bank sample. Keep these
samples; they prevent a clean NT4 cache-scaling conclusion under current load.

The floating-point count excludes useful integer dot products, byte shuffles
and unpacking. It is also different from dense-equivalent mathematical FLOPs.
Thus 14–19% of the pure-FMA reference must not be called overall CPU utilization
or evidence that 81–86% of the CPU is idle. This does not establish the requested
90% theoretical target; it gives a calibrated measurement of one component.
No memory-controller byte count or theoretical DIMM bandwidth is established.

All 36 measured GU windows pass output-repeat checks after 36,864 calls, in
addition to initial per-bank output generation. The first run's enabled/running
times are cumulative because resetting event counts does not reset those totals;
its counts are per-window. The second build subtracts pre-window time totals.
Both raw logs and both source versions are retained. No production kernel,
default, simulator rate, GPU or full-model measurement changed. Sources, event
catalog description, commands, raw counts and binary hashes are in
`results/20261020/fp4-flop-counters/`.

### FP4 pipe and dispatch-counter diagnosis (2026-10-20)

The same production GU workload as the retired-FLOPS probe now runs five
three-event groups: user cycles plus retired micro-ops / total FPU assignments,
pipe assignments 0/1, pipe assignments 2/3, FP-scheduler-empty / retire-token
stalls, and AGSQ-token / ALU-token stalls. Raw encodings and local perf catalog
text are preserved in `results/20261020/fp4-pipe-counters/`. All groups count
only the calling user thread, excluding kernel and hypervisor execution. Each
window subtracts the enabled/running totals captured before reset and enable.
All 150 windows have equal enabled and running time; no scaling was needed.

CPU3 and CPU11 each run every group, with three repeats per case. The GU cases
use NT1/NT2, K4096/R128, one or 64 packed-weight banks (576 KiB / 36 MiB),
2048-byte prefetch and the production half-pair implementations. Each window
performs 1,024 calls. All 120 GU windows pass the final-output repeat check,
covering 122,880 timed calls. This checks the final bank output against its
untimed result; it is not an independent mathematical oracle for every call.
Thirty pure-FMA reference windows use the same eight independent accumulators
as the preceding experiment.

Across both cores and bank counts, medians of per-window ratios give:

| Event / user cycle | NT1 | NT2 |
| --- | ---: | ---: |
| FPU pipe 0 assignments | 0.767–0.775 | 1.148–1.173 |
| FPU pipe 1 assignments | 0.578–0.595 | 0.532–0.539 |
| FPU pipe 2 assignments | 0.653–0.668 | 0.182–0.184 |
| FPU pipe 3 assignments | 0.495–0.506 | 0.554–0.559 |
| FP scheduler empty | 0.0020–0.0054 | 0.0015–0.0034 |
| Retire-token stall | 0.0060–0.0089 | 0.0077–0.0124 |

AGSQ/ALU token-stall ratios are below 0.0002. Pure FMA assigns about one event
per cycle to each of pipes 0/1 and nearly none to 2/3 in the dedicated pipe
runs. Assignment counts are not busy-cycle percentages: the NT2 pipe-0 count
exceeds one per cycle, so treating it as utilization would be invalid. These
speculative events may include multi-pipe operations; events from separately
executed groups must not be summed into a simultaneous exclusive breakdown.
The selected token-stall masks do not cover every possible dispatch stall.

The FP scheduler is almost never empty during this workload, reinforcing that
low retired floating-point FLOPS are not evidence of an idle CPU. The strong
change in pipe distribution between NT1 and NT2 makes the integer/shuffle and
conversion instruction mix a useful next optimization target. This does not
identify a uniquely saturated port or prove an achievable speedup. The
one-bank CPU11 group-0 samples also show substantial slowdowns: NT1 retired
micro-ops/cycle is 2.171 versus 3.297 with 64 banks, and NT2 is 2.443 versus
2.937. Those outliers remain in the evidence; do not infer cache penalties from
that group or combine its counters with other groups as if sampled together.

No production source, defaults, GPU execution or simulator rates changed.
Full-model throughput and 90% theoretical FLOPS/bandwidth remain unverified.
The next candidate should reduce unpack/scale instruction cost and preserve
output order, then pass paired service timing against the existing unfused
baseline before integration.

### Compact FP4 scale-pair table screen (2026-10-20)

Following the pipe-counter diagnosis, a scratch candidate replaces the NT1/NT2
half-pair GU kernels' two scalar scale lookups and lane assembly with one packed
scale-pair lookup, 64-bit broadcast and variable permutation. All 65,536 input
byte pairs retain their exact existing decoding, including high-bit and 0x7f
behavior. The constexpr table stores two floats per pair (512 KiB), one quarter
of the earlier rejected eight-float table. It adds no startup initializer and
leaves the dot/FMA order unchanged. NT4 and other kernels are unmodified controls.

An exhaustive check compares all eight output lanes for every byte pair against
the original scale table: 524,288 bitwise checks pass. Paired GU tests link the
symbol-renamed candidate beside the current production archive, using GCC,
K4096/R128, prefetch 2048, one/64 banks (576 KiB / 36 MiB) and primary cores
3/11. Each case has 12 alternating-order pairs of 64 calls per arm. Initial
outputs are compared for every bank, and each timed arm checks its final output
against the baseline. These checks cover 480 timed arms and 30,720 calls.

| Scale bytes | Tokens | Candidate/baseline median wall-time ratio across cells |
| --- | ---: | ---: |
| All byte values | 1 | 1.321–1.400 |
| All byte values | 2 | 1.142–1.150 |
| All byte values, unchanged control | 4 | 0.998–1.000 |
| Finite normal range 8–126 | 1 | 1.120–1.137 |
| Finite normal range 8–126 | 2 | 1.063–1.072 |

The candidate loses every pair in all eight all-byte NT1/NT2 cells and all but
one pair across the eight normal-scale cells. Even reducing the table footprint
fourfold does not make pair lookup useful on this primary CPU. Reject the
candidate; retain the small scalar table and existing scale assembly. This is
not evidence that every table-free scale conversion will lose, but it rules out
this specific lookup/broadcast/permutation replacement at the tested shapes.

No production kernel, build option, default or simulator rate changed. GPU,
worker-node, full-model quality/throughput and 90%-peak qualification were not
run for this rejected candidate. Frozen source, commands, raw paired samples,
validation and hashes are in `results/20261020/fp4-compact-scale-pair/`.

### Primary Zen 1 scheduling and two-block unroll screens (2026-10-20)

Two separate scratch candidates test the existing FP4 arithmetic on the primary
Threadripper 1950X. The first compiles the current source with `-mtune=znver1`
while retaining AVX2/FMA/F16C, O3 and ordinary floating-point rules. The second
retains generic tuning and adds `#pragma GCC unroll 2` only to the NT1/NT2
half-pair GU loops over 64-value blocks. It preserves the ordered FMA chain for
each output and does not change the NT4 implementation. Neither changes weights,
activation precision, scale expansion or production source.

Both use the existing paired GU probe: K4096/R128, scale bytes 8–126, Q8 bytes
including -128, clamp seven, prefetch 2048, one/64 packed banks and primary
cores 3/11. NT1/2/4 each run 12 alternating-order pairs with 64 calls per arm.
Each candidate has 288 successful timed-arm final-output checks after 18,432
calls, plus initial candidate/baseline comparisons for every bank. The oracle
is bitwise agreement with the current implementation, not independent full-model
quality validation. Compiler-tuning NT4 is a candidate; unroll NT4 is unchanged.

| Candidate | Tokens | Median candidate/baseline wall-time ratios across four cells |
| --- | ---: | ---: |
| Zen 1 tuning | 1 | 0.9962–1.0003 |
| Zen 1 tuning | 2 | 1.0043–1.0095 |
| Zen 1 tuning | 4 | 1.0135–1.1786 |
| Two-block unroll | 1 | 0.9963–1.0064 |
| Two-block unroll | 2 | 0.9977–1.0022 |
| Two-block unroll, unchanged control | 4 | 0.9970–1.0042 |

Zen 1 tuning gives no useful general gain; the large NT4 CPU11 one-bank slowdown
is retained rather than discarded. Two-block unrolling produces small mixed
changes comparable with the unchanged NT4 control. Neither warrants a service
integration or a default change. The preceding worker Zen 2 tuning rejection
and this primary result apply to different CPUs and code snapshots; neither
proves that every compiler scheduling change is ineffective.

Evidence is under `results/20261020/fp4-zen1-tune/` and
`results/20261020/fp4-half-unroll2/`, including source snapshots, commands, hashes
and raw pairs. The latter summary was generated with the former directory's
`summarize.py fp4-half-unroll2`. GPU/worker execution, real routing, full-model
throughput and theoretical utilization were not measured in these screens.

### Q2/Q3 prefetch distance around the enabled baseline (2026-10-20)

After the earlier 0-versus-1024-byte prefetch qualification, a same-object
scratch selector compares 512, 2048 and 4096 bytes against 1024, with two
1024-versus-1024 controls. The selector is sampled at the same places in both
arms; arithmetic, weights and scheduling are shared. The inherited probe's
`set_q3_compiler` name and old 0/1024 comment describe its origin, not this
experiment: `q23.cpp` now selects only the prefetch distance, with 1024 as the
baseline and `STRATA_Q23_TEST_DISTANCE` as the candidate. Both Q2 GU and Q3 down
use the selected distance and retain their existing T0 hints.

The synthetic H4096/F2048 service has seven experts and two input tokens,
sharing one expert and otherwise selecting three distinct experts per token.
It rotates sixteen banks (1,020,264,448 packed bytes), uses native dataflow,
three tasks per participant, no idle spin, no canonical route dropping, Q2 LUT
disabled and Q3 Clang selectors disabled. Input quantization is outside timing.
The primary has 15 workers plus host and NUMA-owned rows; the Ryzen 9 3950X
worker has three workers plus host and complete blobs. Its miner remains
running. Host load and memory speed are uncontrolled; this is not an idle-worker
or theoretical-bandwidth qualification.

On each host the process order is 512, 2048, 4096, control, control, 4096, 2048,
512. Every process performs 20 alternating-order pairs of 64 synchronous calls
per arm. The portable worker binary is hash-checked before each process.
All 640 timed-arm final output and routed-reduction checks pass after 40,960
calls, with one shared output hash. These are final batch checks, not an
independent reference check after every call.

| Candidate bytes (baseline 1024) | Primary paired ratios, first/repeat | Worker paired ratios, first/repeat |
| --- | ---: | ---: |
| 512 | 1.0021 / 1.0029 | 1.0347 / 1.0327 |
| 2048 | 0.9988 / 0.9985 | 0.9924 / 0.9836 |
| 4096 | 0.9976 / 1.0039 | 0.9827 / 0.9853 |
| 1024, identical-work controls | 1.0017 / 0.9989 | 1.0049 / 1.0040 |

Primary differences do not support changing its enabled distance. On the
worker, 4096 wins 17/20 and 18/20 pairs, for a repeated 1.5–1.7% median paired
improvement. Corresponding process medians are 5.0012 to 4.9233 ms and 4.9875
to 4.9090 ms; ratios of process medians need not equal medians of paired ratios.
512 loses all forty worker pairs. 2048 helps but its gain varies more between
passes. Retain 4096 as a candidate for this worker/mixed-route configuration;
dense routes, other token widths and full-model service still need qualification.
The existing opt-in `STRATA_Q23_PREFETCH=4096` can express the candidate without
a production code change. Default prefetch remains off.

No production implementation, default, runtime deployment or simulator rate
changed. This extra gain is relative to enabled 1024-byte prefetch, and must not
be added directly to a previous gain relative to disabled prefetch. It does not
establish decode tok/s or 90% theoretical FLOPS/bandwidth. Sources, local commands,
worker argument/settings manifests, hashes and raw paired logs are preserved in
`results/20261020/q23-prefetch-distance/`; private transport configuration stays
in the ignored build tree.

### Worker dense-route prefetch qualification (2026-10-20)

The preceding mixed-route 4096-byte candidate is now screened at dense token
widths 1/2/4. The same deployed portable binary compares 4096 against 1024 bytes
in alternating-order pairs; separate 1024-versus-1024 processes are controls.
There are two dense experts, H4096/F2048, sixteen banks (291,504,128 packed
bytes), native dataflow and three workers plus host on the Ryzen 9 3950X worker.
Q2 LUT and Q3 Clang selectors remain disabled, three tasks per participant and
idle spin zero. The miner stays running. Input quantization is outside timing;
this is synthetic service, not a routing trace or full-model run.

The process sequence is candidates NT1/2/4, controls NT4/2/1, controls NT1/2/4,
then candidates NT4/2/1. Each process has 20 alternating pairs of 64 calls per
arm and checks both expert output and routed reduction against the initial
reference. All 480 timed-arm final checks pass after 30,720 calls. Each width
has one output hash across its four processes; each process verifies the
installed executable hash before work.

| Width | 4096/1024 paired ratios, first/repeat | Identical-work control ratios |
| --- | ---: | ---: |
| 1 | 0.9740 / 0.9818 | 0.9988 / 0.9969 |
| 2 | 1.0242 / 1.0139 | 0.9928 / 0.9939 |
| 4 | 1.0118 / 1.0056 | 1.0027 / 0.9929 |

NT1 wins 18/20 and 16/20 pairs: a repeated 1.8–2.6% paired-median improvement.
The corresponding process medians are 1.5050 to 1.4644 ms and 1.5041 to
1.4747 ms. NT2 regresses 1.4–2.4%; NT4 gives a small regression, with its
repeat within the observed control variation. Therefore do not recommend a
blanket 4096-byte distance for all worker token widths. The earlier mixed-route
benefit is consistent with its six single-token experts and one shared
NT2 expert, but this does not independently isolate the cause: expert count,
bank footprint and task distribution also differ between mixed and dense cases.

The next candidate is width-dependent prefetch: 4096 for single-token rows and
1024 for wider rows, compared against both fixed policies. It is not yet
implemented or qualified by this screen. The existing global option remains
unchanged, default off; no worker configuration, kernel source or simulator
rate was changed. The primary's preceding no-gain result remains applicable
only to its tested mixed case. Full-model throughput, quality and 90% peak
FLOPS/bandwidth are unverified.

Sources, build commands, portable-binary hashes, generic worker arguments,
settings and raw paired logs are in `results/20261020/q23-prefetch-dense/`.
Private transport settings stay in the ignored build tree. Reproduction uses
the `q23-prefetch-distance` build from the previous screen and the per-run
settings/arguments in `runs.json`; the copied build script additionally needs
its referenced previous probe source from that evidence set.

### Width-dependent Q2/Q3 prefetch screen (2026-10-20)

A scratch same-object policy now selects 4096-byte prefetch for NT1 rows and
1024 for wider rows. It samples the policy once per row-group call, before the
row loop, and keeps all arithmetic and weight layouts unchanged. The baseline
is selectable as fixed 1024 or 4096; fixed-1024-versus-fixed-1024 runs are
controls. In this scratch source only, policy -1 means the width-dependent
choice. It is not a new value accepted by the production environment option.

The Ryzen 9 3950X worker uses three workers plus host, native dataflow, sixteen
banks, H4096/F2048, tasks per participant three and idle spin zero. The miner
remains running. Dense cases have two experts and NT1/2/4, while mixed NT2 has
seven experts (six NT1 and one shared NT2). Q2 LUT and Q3 Clang paths remain
disabled. Input quantization is excluded. Twelve processes cover forward and
reverse candidate order, two mixed-route controls and both fixed mixed-route
baselines. Each process runs 20 alternating pairs of 64 calls per arm.

All 480 timed-arm final output/reduction checks pass after 30,720 calls, and
output hashes agree across processes for each geometry. Portable executable
hashes are verified before every process. The exact scratch source differs from
production only in the selector and its width argument; no kernel arithmetic
was edited.

| Case | Baseline | Candidate/baseline paired ratios, first/repeat |
| --- | ---: | ---: |
| Mixed NT2 | 1024 | 0.9852 / 0.9879 |
| Mixed NT2 | 4096 | 1.0074 / 1.0010 |
| Mixed NT2, fixed identical-work control | 1024 | 0.9995 / 0.9998 |
| Dense NT1 | 1024 | 0.9882 / 0.9685 |
| Dense NT2 | 1024 | 1.0021 / 1.0039 |
| Dense NT4 | 1024 | 1.0134 / 1.0092 |

The policy retains a 1.2–1.5% mixed-route improvement over fixed 1024, winning
16/20 and 18/20 pairs, but does not beat fixed 4096 there. Dense NT1 improves
1.2–3.1%; wider dense cases show small regressions despite selecting the same
1024-byte distance in both arms. These runs do not isolate whether the wider
variation comes from policy selection, code execution or external load, and
there are no additional width-specific identical-policy controls in this
screen. Do not claim the selector itself caused those regressions.

Do not integrate this policy as a general replacement on this evidence. The
fixed worker options retain their previously measured workload-specific
tradeoffs. No production source, environment semantics, defaults, runtime
configuration or simulator rate changed. This is not a full-model decode or
90%-peak utilization result. Source/build snapshots, portable hashes, settings,
arguments and raw pairs are in `results/20261020/q23-prefetch-width/`; transport
configuration remains in the ignored local build directory.

### FP4 offset half pairing: useful default-path screen, rejected integration (2026-10-20)

A scratch NT1 GU candidate shares the existing offset-dot activation correction
between gate and up. Shifted doubled E2M1 codes are 0..24; subtracting the
precomputed 12-times-activation pair sum replaces two sign operations with one
subtraction. Pair intermediates stay within int16, and integer reduction, scales,
FMA order and output reduction remain unchanged. Activations containing -128
fall back to the signed half-pair implementation to preserve its wrap behavior.
Preprocessing still builds the correction and detects that exceptional input.

The isolated paired GU probe compares the candidate with the current signed
half-pair implementation: K4096/R128, one/64 banks, cores 3/11, 2048-byte prefetch,
raw/all-byte or finite 8–126 scales. Safe Q8 bytes span -127..127; exceptional
cases additionally inject -128 into the first block. Twelve alternating pairs
of 64 calls per arm are used per case, including unchanged NT2 controls.

| Isolated case | Candidate/baseline paired ratios across four cells |
| --- | ---: |
| NT1 safe activations, raw scales | 0.9425–0.9516 |
| NT1 safe activations, normal scales | 0.9436–0.9529 |
| NT1 -128 fallback, raw scales | 1.0036–1.0112 |
| NT1 -128 fallback, normal scales | 1.0056–1.0122 |
| Unchanged NT2 control | 0.9954–1.0050 |

All 480 timed-arm final checks pass after 30,720 calls, plus initial per-bank
comparisons. Thus the isolated safe-input gain is about 5%, with a small fallback
preprocessing penalty. This does not itself qualify whole-expert service.

Real partial layers 3/4 then use owned unfused dataflow, E1/E3, NT1/2, sixteen
copied banks and sixteen balanced physical participants on the primary. Only
NT1 GU selects the candidate; down remains the baseline. Eight processes include
candidate/control/control/candidate orders for both expert counts. All 1,280
timed-batch plus 64 CPU-batch checks pass after 22,528 calls. E1 NT1 ratios vary
0.9264–1.0014, while identical-work controls vary 0.9763–1.0175; E3 NT1 ratios
are 0.9825–0.9955. The noisy short batches do not justify integration alone.

A four-process E3 repeat increases latency batches from 16 to 64 calls. NT1
candidate ratios are 0.9804/0.9809 and 0.9719/0.9744 for layers 3/4 in the two
passes, versus controls 1.0012/0.9974 and 1.0052/0.9984. NT2 unchanged controls
stay within about 0.6%. All 640 timed-batch plus 32 CPU-batch checks pass after
41,984 calls. This supports a 1.9–2.8% GU-candidate service gain against the
signed default in this test. The long probe retains old divisors in an unused
split-phase branch; these runs exclusively use dataflow and never execute or
report that branch. The subsequent integrated probe corrects those divisors.

An attempted integration added the paired overload behind the existing offset
option and half-pair build option, preserving default selection. CPU targets
built in both HIP- and CUDA-configured trees; each passed six targeted tests
covering native/default/offset/narrow variants, K32 and GGML CPU parity. Both
builds additionally passed real partial-weight native tests with offset disabled
and enabled. GPU execution was not performed; no SYCL compiler was available.

Crucially, a final paired service run compared that integration against the
**frozen previous offset implementation with offset enabled in both arms**.
The same long E3 protocol produced NT1 ratios 1.0071/1.0652, then 1.0055/1.0054;
identical-work controls were 0.9965/1.0008, then 0.9880/0.9991. It did not improve
this stronger baseline. All 640 timed-batch plus 32 CPU-batch checks still passed
after 41,984 calls. Do not discard the 6.5% regression or infer a transitive
speedup between the signed/default and previous-offset paths: those screens
also used different down-path settings.

**Reject the integration.** Production source was restored byte for byte to
the frozen pre-experiment source, preserving all earlier worktree changes. Both
CPU build trees were rebuilt and again passed all six targeted tests. No default,
existing offset implementation, runtime setting or simulator rate remains
changed by this experiment. This gate prevented replacing an existing option
with a candidate that only beat a weaker comparison path.

Evidence is in `results/20261020/fp4-offset-half-pair/`,
`fp4-offset-half-service/`, `fp4-offset-half-service-long/` and
`fp4-offset-half-integrated/`. The final directory preserves the rejected patch,
source snapshots, service results, candidate real-test commands, labeled test
excerpts and byte-identical restoration hashes. Full build/test logs remain in
the ignored build tree. These measurements do not establish full-model decode
speed, quality equivalence beyond the stated exact comparisons, or the 90%
theoretical FLOPS/bandwidth target.

### Current full offset option versus signed default (2026-10-20)

After rejecting offset half pairing against the existing offset implementation,
this screen directly compares the **current** offset option with the current
signed default. A scratch object replaces only the cached offset-enable getter
with a same-object atomic selector. It changes no arithmetic and selects the
existing offset path for every eligible call, including both GU and down, rather
than isolating a GU-only candidate. The selector is changed only between completed
synchronous pool calls. Identical-work control processes force it off in both
arms. Production source remains byte-identical to the pre-screen source.

The primary uses sixteen balanced physical participants, owned unfused native
dataflow, real partial layers 3/4, one/three experts, one/two tokens, sixteen
copied banks, prefetch 2048, tasks per participant three and idle spin zero.
Narrow and Clang selectors are disabled. Eight processes use candidates E1/E3,
controls E3/E1, controls E1/E3, then candidates E3/E1. Each layer/width has twenty
alternating pairs of 64-call latency batches plus a separate 32-call CPU batch.
Input quantization remains outside timing. All 1,280 latency-batch and 64
CPU-batch final output/reduction checks pass after 83,968 calls.

| Experts | Width | Offset/default paired ratios across layers and repeats | Identical-work control ratios |
| --- | ---: | ---: | ---: |
| 1 | 1 | 0.9595–0.9768 | 0.9924–1.0128 |
| 1 | 2, unchanged arithmetic | 0.9889–1.0279 | 0.9742–1.0091 |
| 3 | 1 | 0.9514–0.9663 | 0.9331–0.9982 |
| 3 | 2, unchanged arithmetic | 1.0003–1.0046 | 0.9972–1.0007 |

All eight NT1 candidate cells favor offset in paired medians, by 2.3–4.9%.
However, the first E3 layer-3 identical-work control reports an apparent 6.7%
gain, so the entire control range must be retained. E1 unchanged NT2 variation
is also material. In one E1 NT1 cell the candidate's process median is slightly
higher despite a lower median of paired ratios; these statistics are different
and must not be substituted for each other. This is conditional evidence for
using the existing offset path as an additional optimization baseline, not a
guaranteed latency gain or a default-change qualification.

Post-run load averages are recorded only as an observation. They include this
benchmark's own activity and are not evidence that the user was running another
CPU workload. No per-window external-load or memory-controller measurement was
made. The existing `STRATA_NVFP4_OFFSET_DOT=1` option remains opt-in, and its
raw -128 fallback remains unchanged. Future FP4 candidates should be compared
with both the signed and offset paths at the same complete-service scope; a
win over one must not be presented as a win over the other.

Sources, exact build/run commands, paired samples, controls and hashes are in
`results/20261020/fp4-current-offset-paired/`. No production kernel, default,
runtime configuration or simulator rate changed. No GPU, worker-node, full-model
quality/throughput or 90%-peak qualification was performed in this screen.

### Paired FP4 offset cycle, instruction and micro-op audit (2026-10-20)

The existing offset option and signed half-pair default are now compared inside
paired, region-gated performance-counter windows. The scratch same-object
selector from the preceding complete-service screen is reused. This probe is
single-thread GU only, includes activation preprocessing, uses K4096/R128,
2048-byte prefetch, finite scales 8–126, and one/64 weight banks (576 KiB / 36 MiB).
Both physical CPU3 and CPU11 run safe Q8 (-127..127) and exceptional Q8 (same
inputs with the first byte set to -128), at NT1/2. NT2 is unchanged arithmetic.
Every cell has eight alternating-order pairs of 256 calls per arm. All initial
per-bank references are generated with offset off, and each measured arm checks
its final output against that baseline.

The first event group measures user cycles, instructions and retired SSE/AVX
floating-point operations (raw 0xff03). It also runs three pure-FMA calibration
windows per process: all twelve report exactly 128,000,064 operations, matching
the eight-accumulator loop and its reduction. The second group replaces FP ops
with retired micro-ops (raw 0xc1, matched to the local perf event catalog). Its
pure-FMA loops are reference workloads, not a claim that micro-ops equal FLOPs.
Events are per calling thread, without inheritance, excluding kernel/hypervisor.
Enabled/running time deltas match in every window; no multiplexing scaling is
needed. These groups are separate runs, not a simultaneous multi-event trace.

| Safe NT1 measurement | Offset/default paired ratio across cores/bank counts |
| --- | ---: |
| Cycles, FP event group | 0.9331–0.9362 |
| Cycles, micro-op event group | 0.9344–0.9397 |
| Retired instructions, both groups | about 1.00567 |
| Retired FP operations | 1.00000 |
| Retired micro-ops | 0.98245–0.98255 |

Offset wins every safe NT1 cycle pair: 32/32 in each group. It retires about
0.57% more instructions but 1.75% fewer micro-ops, and spends about 6–7% fewer
cycles on the same counted FP work. Retired FP throughput rises from roughly
2.30–2.33 to 2.46–2.49 operations per cycle; measured retired micro-op throughput
rises from 3.32–3.38 to 3.49–3.53 per cycle. These are separate metrics and must
not be called a percentage of theoretical CPU FLOPS or overall utilization.
The data support a better instruction/execution mix, but do not uniquely identify
an execution-port or memory bottleneck. The cycle reduction is larger than the
micro-op-count reduction, so count alone does not explain it.

The -128 fallback retires about 9.49% more instructions and 9.72–9.74% more
micro-ops while preserving FP count and output. Most fallback cycle ratios are
1.007–1.030, but the first group's CPU3 one-bank median is 1.366. Preserve that
outlier. The second group's unchanged NT2 CPU11/64-bank cycle ratio is 0.9151,
with unchanged instructions and nearly unchanged micro-ops; this also demonstrates
substantial timing variability in an individual cell. Neither anomaly establishes
a cache or port mechanism. The original offset fallback runs separate signed
row dots; it is not the rejected paired fallback from the preceding prototype.

Each group passes 256 final-output windows after 65,536 timed GU calls, plus
initial per-bank reference checks. In total, 512 windows and 131,072 calls pass.
No production code, defaults, simulator rates, GPU state or worker configuration
changed. Full-model throughput, hardware DRAM traffic and the 90% theoretical
FLOPS/bandwidth target remain unverified. A next candidate is aligned activation
reuse for the existing offset path, charging its preparation and stack/cache
footprint in the same comparison rather than timing only the inner dot loop.

Raw counts, source snapshots, exact commands, settings, event scope and hashes
are in `results/20261020/fp4-offset-paired-counters/` and
`results/20261020/fp4-offset-uop-counters/`.

### Offset activation alignment and interleaving screens (2026-10-20)

Two scratch candidates test aligned activation reuse against the existing
**enabled offset** path. The first adds 128 aligned 32-byte activation vectors
to `OffsetActs`, filled during its existing correction-preparation loop. Its row
dot then loads copied Q8 vectors with a 32-byte stride instead of reading the
original 34-byte Q8_0 blocks. The second places each activation and correction
in a 64-byte-aligned record, replacing the separate arrays. Each increases the
maximum prepared-activation payload by 4 KiB relative to the original correction
array. Both preserve the old -128 fallback, scale calculations and FMA order.
This is not a weight-format change, and the preparation/copy cost is inside
every timed GU call.

Each candidate uses K4096/R128, NT1 with safe Q8 (-127..127) or an injected -128,
all-byte or finite 8–126 weight scales, one/64 banks and primary cores 3/11.
Unchanged NT2 cases are controls. The same process links a renamed candidate
beside the current production archive with offset enabled, prefetch 2048 and
narrow/Clang selectors disabled. Every case has twelve alternating pairs of
64 calls per arm, initial comparisons for all banks and final-output checks
after each measured arm. All 480 checks after 30,720 calls pass per candidate
(960 checks and 61,440 calls combined).

| Candidate | NT1 safe raw scales | NT1 safe normal scales | Unchanged NT2 controls |
| --- | ---: | ---: | ---: |
| Separate aligned Q8 array | 0.9929–1.0029 | 0.9957–1.0018 | 0.9920–0.9968 |
| Interleaved Q8/correction records | 0.9935–1.0008 | 0.9967–1.0028 | 0.9975–1.0378 |

Values are ranges of median paired wall-time ratios across cores and bank
counts. Neither candidate gives a consistent safe-input gain exceeding the
observed control variation. The first candidate's fallback ratios are
0.9822–0.9982; the interleaved fallback spans 0.9917–1.2407, including a CPU3,
64-bank, raw-scale cell that loses all twelve pairs. Preserve that regression.
Fallback still reads original activations, so these timings must not be
attributed solely to faster aligned Q8 loads. Preparation, object layout,
compiler placement and external variation are not independently isolated here.

Reject both candidates. The extra preparation and storage are not justified by
the tested safe-input performance. The interleaved source also adapts correction
references in the narrow template so it compiles; narrow execution was not
qualified. No production source, default, runtime setting or simulator rate
changed, and no GPU, worker-node or full-model throughput test ran. The 90%
theoretical FLOPS/bandwidth target remains unverified.

Frozen sources, build arguments, paired probes, settings, raw samples and hashes
are under `results/20261020/fp4-offset-aligned-acts/` and
`results/20261020/fp4-offset-interleaved-acts/`. The second candidate's exact
compile/link arguments are in `build.json`; it does not have a separate source
generation script.

### Compiler scheduling for the existing offset path (2026-10-20)

Earlier compiler screens used the signed half-pair implementation. This screen
instead enables the existing offset option for both baseline and candidate and
compiles the same current source with Clang 21.1.8 or GCC's `-mtune=znver1`.
The baseline is generic GCC. All use O3, AVX2/FMA/F16C, no fast math, prefetch
2048, narrow disabled and the optional NT4 Clang dispatcher disabled. Compiler
version lines and exact commands are preserved with the evidence.

The paired GU geometry is K4096/R128 on primary cores 3/11 with one/64 banks,
raw/all-byte or finite 8–126 scales, and safe Q8 or an injected -128. NT1 and
NT2 are compared; NT2 is also compiled by the candidate compiler/tuning and is
therefore not an identical-code control. Each candidate has twenty cells, twelve
alternating pairs per cell and sixty-four calls per arm. All 480 timed-arm final
checks pass after 30,720 calls per candidate, plus initial bank comparisons.

| Candidate | Safe NT1 raw scales | Safe NT1 normal scales | NT1 fallback, all scales | NT2 |
| --- | ---: | ---: | ---: | ---: |
| Clang / generic GCC | 1.0358–1.0445 | 1.0392–1.0480 | 1.0488–1.0621 | 0.9967–1.0058 |
| GCC Zen 1 / generic GCC | 0.9801–0.9850 | 0.9650–0.9834 | 0.9865–0.9987 | 1.0056–1.0119 |

These are ranges of median paired wall-time ratios across cores/banks. Clang
loses all ninety-six safe NT1 pairs and is rejected for this path. Zen 1 tuning
has a small isolated NT1 gain but regresses NT2, so a service wrapper next selects
tuned code only for NT1 GU and down. Wider widths and K32 remain the generic
baseline. Both arms keep the offset option enabled; no arithmetic changes.

The service screen uses real partial layers 3/4, three experts, NT1/2, sixteen
banks, owned unfused dataflow and sixteen balanced physical participants on the
primary. Four processes run candidate/control/control/candidate, each with twenty
pairs of 64-call latency batches and separate 32-call CPU batches. All 640
latency-batch and 32 CPU-batch checks pass after 41,984 calls. NT1 paired ratios
are 1.0078/0.9994, then 0.9945/0.9951 for layers 3/4. Identical-work controls are
1.0012/1.0024, then 1.0033/0.9797. The unchanged NT2 path stays within roughly
0.7%. Thus the isolated scheduling gain does not establish a complete-service
benefit; preserve the 2% control anomaly and retain generic GCC.

No compiler selector, build option, production source, default, runtime setting
or simulator rate changed. GPU/worker execution, full-model quality/throughput
and 90%-peak utilization were not qualified. The next broader candidate should
reduce packed-weight decoding itself on cache hits, while charging expansion,
extra host storage and increased streaming traffic. A cache-resident inner-loop
win alone would not justify that representation for cold experts.

Sources, commands, samples and hashes are in
`results/20261020/fp4-offset-clang/`, `fp4-offset-zen1/` and
`fp4-offset-zen1-service/`.

### Lossless K32 offset dots: experimental opt-in (2026-10-20)

The earlier byte-expanded cache already regressed complete expert service, so
that representation was not repeated. This change combines the existing
size-preserving K32 permutation with offset dots. It retains 36-byte K64 blocks,
four opaque scale bytes, Q8 activation processing, FMA/reduction order and global
scale handling. K32 removes the split-8 unpack shuffles; offset dots use shifted
E2M1 codes and the precomputed activation correction. Raw -128 activations take
a K32-aware signed fallback, never the original-layout decoder.

Production now exposes **`STRATA_NVFP4_K32_OFFSET=1`** for explicit K32 APIs when
the call has one token. The switch uses `strata_env_is_one`; unset, zero and other
values preserve the existing signed K32 path. Legacy `STRATA_NVFP4_OFFSET_DOT`
selection is unchanged and does not enable this K32 specialization. Wider K32
calls retain their existing implementation. No loader automatically selects the
K32 layout, no checkpoint format changes, and all defaults remain unchanged.
The existing C11 converter `strata_glm_nvfp4_repack_k32` supplies the bytes.

A CPU-only benchmark can exercise both representations after rebuilding:

```sh
cmake --build <native-build> --target glm_fp4_native_pool_bench
STRATA_NVFP4_K32_OFFSET=1 STRATA_NVFP4_OFFSET_DOT=1 \
  <native-build>/glm_fp4_native_pool_bench <partial-model> \
  3 15 fixed 16 paired 1 64
```

This command uses the standard benchmark's modes and repetition count; the
frozen paired probes below additionally fix owned unfused dataflow, balanced
primary placement and twenty pairs. Do not infer full-model speed from either.

**Isolated rows.** On primary cores 3/11, K4096/R128, one/64 banks, prefetch 2048,
raw or finite 8–126 scales, the prototype compares K32 offset with the original
layout's existing offset path. Safe NT1 ratios are 0.9179–0.9359; -128 fallback
ratios are 0.9530–0.9739. All 384 measured-arm final checks pass after 24,576
calls, plus initial per-bank comparisons. The representation uses the same
bytes per bank; the benchmark keeps both copies. Allocation and repacking are
timed once separately and excluded from steady-state GU timing. Those single
conversion observations are not a qualified amortization model.

**Representation plus arithmetic service.** Real partial layers 3/4, E1/E3,
NT1, sixteen banks per layout, sixteen balanced physical participants and owned
unfused dataflow use twenty alternating pairs of 64 calls. Both the original
layout and K32 use offset arithmetic. Eight processes include identical-original-
layout controls and reverse-order candidate repeats:

| Experts | Candidate ratios, first L3/L4 | Repeat L3/L4 | Control range |
| --- | ---: | ---: | ---: |
| 1 | 0.9788 / 0.9844 | 0.9365 / 0.9704 | 0.9941–1.0050 |
| 3 | 1.1638 / 1.0155 | 0.9783 / 0.9839 | 0.9985–1.0016 |

An additional E3 candidate/control/control/candidate run gives candidate ratios
0.9873/0.8969 and 0.9831/0.8682, with controls 0.9877–1.0121. Preserve the initial
16.4% regression and the later large gains. These results show substantial
process-to-process variation and do not justify a universal layout selection.
The two completed service sets pass 960 latency-batch and 48 CPU-batch final
checks after 62,976 calls. Input quantization, repacking and allocation are
outside service timing.

**Arithmetic-only baseline check.** A separate same-object selector keeps the
same K32 buffers and NUMA shards in both arms, toggling only K32 offset for
eligible GU/down calls. This prevents a win over the original layout from being
misreported as a win over the existing K32 implementation. E3, layers 3/4 and
NT1/2 use the same long paired protocol. NT1 ratios are 0.9760/0.9753, then
0.8802/0.9577; controls are 1.0016/0.9990, then 0.9967/0.9916. Unchanged NT2
ratios stay within about 0.4%, with controls within about 0.7%. All 640 latency-
batch and 32 CPU-batch checks pass after 41,984 calls. This supports the new
opt-in on the tested primary workload, with variable magnitude rather than a
fixed service-speedup guarantee.

**Correctness and integration.** The K32 API test passes 1,232 synthetic cases
covering widths 1..9/16/32, raw -128, padding, partial ranges, clamped GU,
separate up buffers and invalid geometry, plus 36 real projection/GU cases.
CPU targets and the native pool benchmark build in both HIP- and CUDA-configured
trees. Each tree passes seven targeted tests, including the new
`glm_nvfp4_k32_offset_dot_test`, original native/default/offset/narrow variants,
K32 default and GGML CPU parity. Both integrated builds also pass real K32 tests
with the new option set to 0, 1 and 2. Byte equality against GGML/original-layout
references validates the default and enabled outputs in these cases. Production
pool smoke runs pass all 24 reported cells per backend with paired layouts,
complete blobs/NUMA shards and fused/unfused modes; their short batches are
correctness checks, not performance evidence. GPU execution and SYCL builds
were not performed; no SYCL compiler was available.

Two failed scratch harness attempts are retained and excluded from timings.
The first reduced the benchmark to one mode but retained a modes==6 allocation
guard, leaving required NUMA vectors empty. The second same-buffer adaptation
still forced original-layout selection and swapped into absent alternate
storage. Both exited with signal 11, were corrected in the harness, and were
rerun successfully. Neither was a production kernel failure. Frozen failed
sources, logs and binary hashes are preserved beside the successful runs.

The kernel and explicit API documentation changed; the only CMake addition is
an opt-in test invocation. No default, GPU path, model routing, runtime deployment
or simulator rate changed. Full-model throughput, end-to-end quality and the
90% theoretical CPU FLOPS/bandwidth target remain unverified.

Evidence: `results/20261020/fp4-k32-offset/`, `fp4-k32-offset-service/` and
`fp4-k32-offset-arithmetic/`. The service directory includes the integrated
source/header snapshots, patch, build/test scope, real-test manifests and labeled
CTest excerpts. Full raw build/CTest logs stay in the ignored build tree.

### K32 offset worker validation (2026-10-20)

The same-buffer K32 arithmetic comparison now also ran on a Ryzen 9 3950X
worker with three pool workers plus the calling thread. Mining remained active;
recorded load averages were around 30–33, and affinity and clocks were not
controlled. This is a bounded four-participant CPU test, not idle-host capacity.
Real layers 3/4 use three experts, sixteen copied weight banks and complete-blob
unfused dataflow. Input quantization is excluded. Both arms use the same K32
buffers; the scratch selector changes only the eligible offset dot arithmetic.
Four processes run candidate, identical-work control, control, candidate, with
20 alternating pairs of 64 calls per cell and a separate 32-call CPU-cost batch.

| Tokens | Candidate/baseline paired median range | Identical-work controls |
|---|---:|---:|
| 1 | 0.9805–0.9958 | 0.9967–1.0108 |
| 2, unchanged arithmetic | 0.9937–1.0061 | 1.0005–1.0038 |

All four NT1 cells improve by 0.4–2.0%, with 60/80 individual pair wins.
Baseline service medians are 3.109–3.138 ms for these three-expert calls.
The modest gain overlaps the scale of control variability; it supports keeping
`STRATA_NVFP4_K32_OFFSET=1` experimental, not enabling it by default on workers.
NT2 wins 35/80 pairs and provides no evidence of a gain.
All four processes exited successfully: 640 timed-batch final-output checks and
32 CPU-batch final-output checks cover 41,984 timed calls. Outputs match the
reference; checks are after batches, not after every call. No GPU execution,
full-model throughput/quality or 90%-peak qualification follows from this test.
Source, build arguments, binary hash, logs and paired samples are preserved in
`results/20261020/fp4-k32-offset-worker/`. No runtime default changed.

### K32 vector unpack screens (2026-10-20)

Two scratch AVX2 candidates replace K32's two 128-bit nibble lookups and lane
assembly with a broadcast packed load and one 256-bit lookup. One uses variable
32-bit shifts (zero in the lower lane, four in the upper lane); the other uses
an immediate shift and blend. Neither changes bytes, scales, accumulation order
or the activation safety fallback. Both benchmark arms read the same K32 buffer
with K32 offset enabled, against the current integrated CPU library.

On Threadripper 1950X, separately pinned to cores 3 and 11, K4096/R128 GU tests
cover raw and normal-range scale bytes, safe Q8 and injected -128 activations,
and one or 64 copied banks. Each cell uses twelve alternating pairs of 64 calls.
NT2 safe normal-scale cases also test the signed-dot path. Mining is unchanged;
clocks and background load are uncontrolled. Candidate/baseline paired medians:

| Path | Variable shift | Immediate shift/blend |
|---|---:|---:|
| NT1 safe offset | 1.0833–1.0971 | 1.0723–1.0806 |
| NT1 -128 fallback | 1.0204–1.0292 | 1.0521–1.0650 |
| NT2 safe signed | 1.0058–1.0280 | 1.0403–1.0457 |

Both candidates are rejected. Wider lookup organization regresses this workload;
the result does not establish which microarchitectural resource causes it.
Each screen passes 480 timed-batch final-output comparisons covering 30,720
calls, plus initial per-bank comparisons. This is synthetic GU work, not full
expert service, model throughput or a peak-utilization measurement. Preparation
is outside timing; per-call offset correction construction remains included.
No production files or defaults changed. Sources, build commands, hashes and
samples are in `results/20261020/fp4-k32-vector-unpack/` and
`results/20261020/fp4-k32-blend-unpack/`.

### Offset correction after int32 reduction (2026-10-20)

A scratch K32 GU candidate precomputes the sum of adjacent int16 correction
lanes and subtracts it after `madd_epi16`, rather than subtracting the int16
correction before that reduction. It preserves integer sums: each biased pair
is bounded by 6,144, so there is no saturation or int16 overflow in either order.
Float operations and the -128 signed-dot fallback remain unchanged. This changes
instruction ordering, not the number of dot-loop arithmetic instructions.
The prototype retains the original correction for narrow-path compatibility and
adds a 4 KiB int32 array; preparation is included in each measured call.

Threadripper 1950X tests use K4096/R128, cores 3 and 11 separately, one or 64
copied banks, raw or normal-range scales, and safe or injected -128 Q8 inputs.
Both arms read the same K32 buffers with offset enabled. Twelve alternating
pairs of 64 calls per cell yield these candidate/baseline paired median ranges:

| Path | Ratio range |
|---|---:|
| NT1 safe offset | 0.9940–1.0049 |
| NT1 -128 fallback | 0.9989–1.0104 |
| NT2, unchanged arithmetic | 0.9888–1.0127 |

There is no consistent benefit; do not integrate this candidate. All 480 timed
batch-final comparisons passed, covering 30,720 calls, in addition to initial
per-bank comparisons. Mining remained active; clocks and background load were
uncontrolled. This is synthetic GU work, not full-model throughput or a
peak-utilization result. No production code or defaults changed. Evidence is in
`results/20261020/fp4-offset-correction32/`.

### Q2/Q3 prefetch cache hint (2026-10-20)

A same-object scratch selector compares T1 with T0 prefetch, both at 1,024 bytes,
on the existing mixed Q2 gate/up and Q3 down expert-service workload. Dot loops
are specialized by hint; selection happens outside each block loop. Both arms
retain identical arithmetic, weight banks, activations and scheduling. Controls
select T0 in both arms. This is a cache-hint request, not proof of actual cache
placement. No production setting was introduced.

The primary Threadripper 1950X uses sixteen participants with NUMA-owned rows;
the Ryzen 9 3950X worker uses three pool workers plus the caller and complete
blobs. Each uses NT2/E7 mixed routes, sixteen copied banks, twenty alternating
pairs of 64 service calls, and process order candidate/control/control/candidate.
Input quantization is excluded; dispatch and expert dataflow are included.
Mining remains active, and clocks/background load are uncontrolled.

| CPU | T1/T0 paired medians, two repeats | T0/T0 controls |
|---|---|---|
| Primary | 1.0023 / 0.9998 | 0.9956 / 0.9984 |
| Worker | 1.0066 / 0.9989 | 1.0055 / 1.0087 |

Neither host shows a repeatable gain. Keep T0 for the existing opt-in prefetch
path. All eight processes succeed, with 320 timed-batch final-output checks
covering 20,480 calls. This does not establish per-call checking, full-model
throughput or peak utilization. The first scratch build failed on the intrinsic's
hint enum type; it was corrected before any execution. The failed compiler log
is retained alongside successful builds and raw samples in
`results/20261020/q23-prefetch-l2/`. No GPU or SYCL execution ran.

### Q2/Q3 single-request prefetch screen (2026-10-20)

The next same-object candidate retains only the first of the two T0 prefetch
requests per block at 1,024 bytes. This halves requests but does not preserve
coverage. An address-only check over all 64 starting offsets shows that sixteen
Q2_K blocks touch 21–22 distinct requested lines with the baseline, versus 16
with the candidate; Q3_K touches 27–28 versus 16. These are address counts, not
observed cache misses or memory traffic. Block sizes are 84 and 110 bytes,
respectively, as enforced by the GGML layout assertions.

The service protocol matches the preceding hint screen: NT2/E7 mixed Q2 GU/Q3
down, sixteen banks, twenty alternating pairs of 64 calls, process order
candidate/control/control/candidate, primary sixteen CPU participants and worker
four. Controls issue two requests in both arms. Mining remains active; no clock
or background-load isolation is claimed. Candidate/baseline paired medians:

| CPU | One/two requests, two repeats | Two/two controls |
|---|---|---|
| Threadripper 1950X | 1.0139 / 1.0254 | 1.0015 / 0.9993 |
| Ryzen 9 3950X worker | 1.2368 / 1.2412 | 1.0014 / 0.9969 |

Reject the reduction: the worker loses 23.7–24.1%, with zero candidate wins in
40 pairs; the primary loses 1.4–2.5%. Fewer instructions do not compensate for
the omitted request coverage in this workload. This does not isolate cache
misses as the cause without hardware counters. Any future deduplication should
preserve requested-line coverage rather than simply remove every second request.

All eight processes succeed: 320 timed-batch final-output checks cover 20,480
calls. Input quantization is excluded; dispatch and dataflow are included. No
production change, GPU/SYCL execution, full-model rate or peak-utilization claim.
Raw samples, source, hashes and the address-coverage script/results are in
`results/20261020/q23-prefetch-single/`.

### Coverage-preserving Q2/Q3 prefetch deduplication (2026-10-20)

A follow-up scratch candidate skips the first prefetch of a block only when its
cache line was already requested by the preceding block's second prefetch.
The second request is always retained. An address-only check covers all 64 row
alignments and every prefix of sixteen blocks for the 84-byte Q2_K, 110-byte
Q3_K and 114-byte expanded-Q3 formats: all 3,072 prefix sets match the baseline.
Requests fall from 32 to 21–22, 27–28 and 28–29, respectively. This verifies
requested-line coverage, not cache residency or hardware behavior.

The same mixed NT2/E7 Q2 GU/Q3 down service protocol uses sixteen banks, twenty
alternating pairs of 64 calls and candidate/control/control/candidate process
order. Both arms use T0 at 1,024 bytes; controls retain the original requests in
both arms. Primary Threadripper 1950X has sixteen participants and NUMA-owned
rows; the Ryzen 9 3950X worker has four participants and complete blobs.
Mining remains active and clocks/background load are uncontrolled.

| CPU | Deduplicated/original paired medians | Original/original controls |
|---|---|---|
| Primary | 1.0010 / 1.0079 | 1.0004 / 0.9989 |
| Worker | 1.0205 / 1.0168 | 0.9968 / 1.0026 |

Reject this implementation. The primary is tied to 0.8% slower; the worker is
1.7–2.1% slower with only 5/40 candidate pair wins. Preserving coverage avoids
the preceding candidate's large worker regression, but added branching and
changed code layout remain possible costs rather than measured causes.
Keep the existing two-request schedule.

All eight processes succeeded, with 320 timed-batch final-output checks covering
20,480 calls. Input quantization is excluded; dispatch/dataflow are included.
The expanded layout is covered by address checks, not this service workload.
No production change, GPU/SYCL execution, full-model rate or peak-utilization
qualification. Evidence is in `results/20261020/q23-prefetch-dedup/`.

### K32 offset prefetch-distance service sweep (2026-10-20)

The integrated K32 offset path was screened at prefetch distances 0, 1,024 and
4,096 bytes against 2,048. A same-object atomic selector changes only distance;
both arms use the same K32 buffers and enable K32 offset for eligible NT1 calls.
Real layers 3/4, three experts, NT1/2 and sixteen copied banks run on the primary
Threadripper 1950X with sixteen balanced physical participants and NUMA-owned
unfused dataflow. Each cell uses twenty alternating pairs of 64 calls and a
separate 32-call CPU-cost batch. Process distance order is
0/1024/4096/2048/2048/4096/1024/0. The 2,048-byte runs are identical-work controls.

| Candidate distance | NT1 candidate/baseline paired median range | NT2 range |
|---|---:|---:|
| 0 | 1.0063–1.1003 | 1.0091–1.0202 |
| 1,024 | 0.9962–1.0808 | 0.9965–1.0029 |
| 2,048 control | 0.9968–0.9997 | 0.9952–1.0011 |
| 4,096 | 0.9982–1.0117 | 0.9994–1.0056 |

Keep 2,048. Disabling prefetch loses in all eight cells; neither alternate
nonzero distance demonstrates a repeatable improvement. The second 1,024-byte
run's layer-3 NT1 regression (8.1%) and the final no-prefetch run's layer-4 NT1
regression (10.0%) remain in the evidence. Clocks and background load were
uncontrolled; those outliers are not attributed to prefetch alone.

All eight processes succeeded. There are 1,280 timed-batch final-output checks
and 64 CPU-batch final-output checks covering 83,968 timed calls. Input
quantization is excluded; per-call activation preparation, dispatch and expert
dataflow remain included. Mining is unchanged. This does not establish worker
performance, full-model speed, GPU/SYCL behavior or peak utilization. No production
setting changed. Sources, hashes, raw samples and controls are in
`results/20261020/fp4-k32-prefetch-distance/`.

### Opt-in K32 loading for the native CPU tree (2026-10-20)

The partial-model adapter now accepts an explicit third constructor argument,
`packed_k32=true`. It uses the existing C11 K32 converter in place on each
projection after NVFP4 packing, before NUMA preparation, then marks the native
format as K32. Global scales and persistent weight size are unchanged. This
makes the previously tested kernel usable in the partial-model service without
adding a second persistent packed-weight copy. Conversion is load-time work;
its startup cost is not included in earlier kernel timings.

The native CPU tree binary accepts `native-k32`; the distributed launcher and
loopback test accept `--native-k32` and apply it to every role. Non-native binaries
reject the selector. It is independent of `preserve-small` and primary
`numa-compact`. K32 offset arithmetic remains separately opt-in through
`STRATA_NVFP4_K32_OFFSET=1` in each executing process; selecting the layout alone
does not enable offset arithmetic. Remote environment configuration remains the
operator's responsibility. No production decoder or default layout changed.

Example CPU-only loopback qualification (not a performance benchmark):

```sh
STRATA_NVFP4_K32_OFFSET=1 python3 tools/test_glm_fp4_cpu_tree.py "$PARTIAL_DIR" \
  --binary "$NATIVE_TREE_BINARY" --native-k32 --output "$NEW_OUTPUT_DIRECTORY"
```

Validation with real partial layers 3/4:

- HIP- and CUDA-linked native CPU adapter tests pass with offset off/on and
  preserve-small off/on: eight runs, 248 checks each. Added comparisons cover
  original versus K32 layout at token counts 1/2/3/4/7/8/9/15/16, including windows
  that split into multiple native calls. Existing rejection checks remain.
- Each library also passes compact-NUMA K32 comparisons against original-layout
  complete and owned rows: 432 comparisons and eight invalid-state rejections
  per library, with multiple input amplitudes and both hidden-scaling settings.
- Frozen pre-change, current default and K32 native tree binaries pass direct
  and relay loopback UDP tests, sixteen cases per topology. All six output
  artifacts are byte-identical. This verifies unchanged default output on this
  fixture, not a full-model release comparison.
- HIP and CUDA GPU tree binaries and the W4A16 CPU tree rebuild successfully
  without GPU execution. Python launcher tests pass (three tests). No SYCL toolchain qualification or
  physical IB rerun was performed.

The modest and variable prior service gains justify experimental availability,
not default enablement or a new full-model throughput claim. Reproduction scripts,
source snapshots and validation records are in `results/20261020/adapter-k32/`.

### Physical three-node K32 adapter validation (2026-10-20)

The opt-in loader now also passes the physical three-node IB partial-model suite.
A portable CPU-only native binary was built from the current tree, deployed to
both workers under a unique temporary name, and its SHA256 verified on each
worker before execution. Miners remained running. The primary used sixteen CPU
participants; each worker used four. K32 offset was enabled in all processes,
with prefetch 2,048, narrow/Clang paths disabled and pool spin disabled.

Three extended runs passed: direct original-layout baseline, direct K32 with
primary compact NUMA storage, and relay K32 with primary compact NUMA storage.
The direct runs also use persistent primary dispatch. All roles report physical
IB transport and CPU-only execution. All K32 roles report the layout selector;
only the primary reports compact NUMA storage. The cases use layers 3/4 and
experts 0/1/2, with hidden-value preservation enabled.

Each run passes 48 layer checks, for 144 total, with zero relative L2 error and
bit-exact repeats. All three 3,293,184-byte output dumps have SHA256
`3e8790bfe14117cae5d5d266a61b93afce8ddbedd1b7e952d80be6e3ff242f29`, matching the
frozen pre-K32 physical run. This also exercises the current finite-input check
and adapter through the real transport.

This is correctness validation, not throughput qualification. Process elapsed
time includes loading, startup, reference work and deliberate test sequencing.
The direct comparison also changes primary storage ownership, so it cannot
isolate K32 performance. No GPU execution, full-model inference or peak-utilization
claim follows. Logs, build arguments, deployment hash and generic environment
settings are in `results/20261020/adapter-k32-ib/`; connection settings and large
raw output dumps remain in ignored local storage.

### Complete adapter timing with K32 (2026-10-20)

The opt-in K32 loader was compared against the existing original-layout offset
path through full `NativeFp4CpuExpert::run` calls. Both offset flags are enabled,
with prefetch 2,048, narrow/Clang paths off and pool spin off. Timing includes
finite scans, allocation, input quantization, hidden rescaling, native execution
and output handling. Loading, packing/repacking and NUMA preparation are excluded.
The Threadripper 1950X pool uses fifteen workers plus the caller; mining is
unchanged and clocks/background load are uncontrolled.

Each layer (3/4, expert 0) has eight copied expert banks per arm, rotated by
independent equal-progress counters. Both representations and their NUMA shards
coexist. Controls also use separate banks, with the original layout in both arms,
so controls do not gain an artificial shared-buffer advantage. Input values and
routing are fixed. Each comparison has twenty alternating pairs; process order
is candidate/control/control/candidate. Complete-blob and owned-row modes are
measured separately, with hidden-value preservation enabled.

The initial 16-call batches cover NT1/2/8. All eight NT1 candidate cells favor K32
in paired medians: complete blobs 0.9678–0.9958, owned rows 0.9613–0.9943. Controls
span 0.9949–1.0039 and 0.9821–1.0240 respectively. NT2/8 results vary in direction;
there is no general wider-token improvement. All 1,920 timed-batch final-output
checks pass, covering 30,720 calls.

A separate NT1 confirmation uses 64 calls per batch:

| Storage | K32/original paired median range | Identical-layout controls |
|---|---:|---:|
| Complete blob | 0.9681–0.9944 | 0.9764–1.0016 |
| Owned NUMA rows | 0.9212–0.9810 | 0.9961–1.0008 |

Owned rows win 71/80 individual pairs, with 1.9–7.9% reductions in paired medians.
Keep the scope narrow: the last layer-4 owned run has higher absolute latency
than earlier cells, and whole-run medians need not agree with paired ratios.
For example, the first layer-3 owned cell has median call times 0.2678 ms baseline
and 0.2692 ms K32, despite a paired median ratio of 0.9509. Both statistics and
all raw samples are preserved; do not substitute the favorable one for an
unqualified throughput claim. Complete-blob controls include a 2.4% apparent
improvement, limiting confidence in similarly small candidate gains there.

All 640 confirmation batch-final checks pass, covering 40,960 calls. This supports
experimental use on owned rows, not default enablement. It does not measure
worker CPUs, physical IB service time, full-model throughput or peak utilization.
Evidence is in `results/20261020/adapter-k32-timing/` and
`results/20261020/adapter-k32-timing-long/`. Production settings remain unchanged.

### Complete K32 adapter calls on the worker (2026-10-20)

The full-call comparison now also ran on the Ryzen 9 3950X worker with three
pool workers plus the caller. The portable binary was hash-checked before each
of four processes, in candidate/control/control/candidate order. Mining remained
active. Each arm rotates eight copied real expert banks from layers 3/4;
complete blobs are used, with NT1/2 and hidden-value preservation. Both original
and K32 offset flags are enabled, prefetch is 2,048, and narrow/Clang paths and
pool spin are off. Controls use separate original-layout banks in both arms.

Twenty alternating pairs of 64 calls include finite scans, allocation, input
quantization, hidden rescaling and expert execution. Startup loading/repacking
is excluded. Candidate/baseline paired median ranges across layers and repeats:

| Tokens | K32/original | Identical-layout controls |
|---|---:|---:|
| 1 | 0.9865–1.0061 | 0.9823–1.0193 |
| 2 | 0.9997–1.0255 | 0.9913–1.0152 |

There is no reliable complete-adapter gain on this worker. Three of four NT1
cells favor K32, but all lie within the scale of control variation. NT2 is slower
in three of four cells, with overlap in control variation. Do not recommend
worker default enablement from the earlier isolated kernel gains. K32 remains
an explicit experimental option; the physical-network correctness result is
unaffected.

All four processes succeed: 640 timed-batch final-output checks cover 40,960
calls. Clocks/background load are uncontrolled; no full-model rate, GPU execution
or peak utilization is established. Sources, binary hash, generic settings and
raw paired samples are in `results/20261020/adapter-k32-worker/`. Private remote
connection settings are excluded. No production behavior changed.

### Caller-owned adapter workspace screen (2026-10-20)

A scratch `run_into` method retains caller-owned output and quantized activation
buffers between calls. It performs the same extent/finite checks, quantization,
hidden scaling and native work as `run`; it rejects input/output vector aliasing
before resizing or writing. This removes repeated buffer allocation and resizing
initialization when shape/capacity are reused. The allocating production method
is unchanged; no shared mutable workspace was added to the adapter.

The primary Threadripper 1950X uses sixteen CPU participants, real layer-3/4
expert-0 weights, eight copied banks per arm and both complete-blob and NUMA-owned
rows. Both arms use the original layout with offset enabled, prefetch 2,048 and
hidden-value preservation. Timing includes full calls; loading and NUMA
preparation are excluded. Twenty alternating pairs of sixteen calls per cell
run in candidate/control/control/candidate process order. Controls call the
allocating method in both arms with separate weight banks.

| Storage / tokens | Workspace/allocating paired median range | Controls |
|---|---:|---:|
| Complete / 1 | 0.9767–0.9917 | 0.9959–1.0024 |
| Complete / 2 | 0.9752–0.9832 | 0.9992–1.0044 |
| Complete / 8 | 0.9614–0.9723 | 0.9760–1.0157 |
| Owned / 1 | 0.9615–0.9906 | 0.9895–1.0163 |
| Owned / 2 | 0.9663–0.9774 | 0.9895–1.0047 |
| Owned / 8 | 0.9653–0.9854 | 0.9798–1.0152 |

All candidate cells favor reuse, by 0.8–3.9%; some wider-token and owned controls
are noisy. All 1,920 timed-batch final-output checks pass, covering 30,720 calls.
Separate correctness runs with hidden scaling off/on each pass 346 checks,
including bitwise workspace comparisons, growing/shrinking token windows,
nonfinite/invalid extents and alias rejection without changing input. The final
correctness runs enable both offset flags.

This is promising enough for worker validation, not yet integrated. Mining is
unchanged; clocks/background load are uncontrolled. No GPU, SYCL, network service,
full-model throughput or peak-utilization qualification is claimed. Sources,
commands, raw samples and quality logs are in `results/20261020/adapter-workspace/`.

### Workspace reuse: worker validation and opt-in API (2026-10-20)

The scratch workspace candidate was repeated on the Ryzen 9 3950X worker with
three pool workers plus the caller. NT1/2 complete-blob calls rotate eight copies
per arm, with layers 3/4, original-layout offset arithmetic and hidden-value
preservation. Twenty alternating pairs of 64 calls per cell run in
candidate/control/control/candidate process order. Binary hashes are checked
before every process; mining remains active and clocks/load are uncontrolled.

| Tokens | Workspace/allocating paired median range | Identical-work controls |
|---|---:|---:|
| 1 | 0.9515–0.9670 | 0.9940–1.0204 |
| 2 | 0.9701–0.9775 | 0.9915–1.0123 |

Every candidate cell improves: 3.3–4.9% at NT1 and 2.2–3.0% at NT2. All 640
batch-final checks pass, covering 40,960 timed calls. Timing includes complete
adapter processing but excludes startup packing and initial workspace growth.
These are partial-expert results, not model throughput. Evidence is in
`results/20261020/adapter-workspace-worker/`.

The adapter now exposes explicit caller-owned storage:

```cpp
std::vector<float> output;
std::vector<uint8_t> activation_scratch;
// Retain both buffers between serialized calls; select owned rows only after preparation.
expert.run_into(pool, input, tokens, output, activation_scratch, use_numa);
```

The existing `run()` API retains call-local buffers and delegates to the same
execution body, avoiding two copies of the algorithm. Neither the tree runner
nor production decoder automatically retains workspace. `run_into` rejects
input/output vector aliasing before writes; callers must serialize use of both
the pool and retained buffers. Shrinking shapes may retain capacity. Output
contents after an execution failure are not promised; invalid input is rejected
before resizing. No global mutable workspace or new environment default exists.

Integrated HIP- and CUDA-linked CPU adapter validation passes eight runs of
346 checks (2,768 total), with offset off/on and hidden scaling off/on, growing
and shrinking shapes, bitwise output checks and invalid-input/alias rejection.
Compact-NUMA validation passes 576 comparisons and eight invalid-state rejections
per library (1,152 comparisons total), including explicit reused workspace on
K32 owned rows. The pre-integration timings above qualify the candidate design;
they are not a new full-model speed result. Sources and integrated validation are
in `results/20261020/adapter-workspace-integration/`.

Frozen pre-change, current default and K32 native tree binaries also pass direct
and relay loopback UDP (six topology runs, sixteen cases each); all output bytes
match. This checks the refactored allocating API's default output on the partial
fixture. The native tree builds use both CPU libraries; no GPU source changed,
no GPU execution or SYCL qualification ran, and physical IB was not rerun for
this workspace API integration. The earlier physical K32 run remains separate.

### Opt-in workspace reuse in the partial-model tree (2026-10-20)

Native CPU tree binaries now accept `native-workspace`; the distributed launcher
and loopback test accept `--native-workspace` and apply it to all roles. Each
expert retains activation scratch lazily. Serialized worker requests retain
output capacity after sending results; primary local and reference computations
retain separate output buffers. Network tasks do not access these buffers or
the CPU expert pool. Idle worker routes clear previous output lengths before
sending, and an idle primary route leaves its local output empty. Active calls
resize through the validated `run_into` API.

The option is off by default, independent of `native-k32`, `preserve-small` and
primary compact-NUMA storage. Without it, `Expert::run_into` delegates to the
existing allocating path and retained buffers remain empty. Buffers may retain
peak capacity when enabled. Input receive buffers and network result allocations
are not part of this optimization. Non-native binaries reject the selector;
duplicate selectors are rejected before transport setup.

The extended loopback suite passes default, workspace, workspace+K32, and
CUDA-linked CPU workspace configurations in both direct and relay topologies:
eight topology runs, 48 layer checks each (384 total). Cases include changing
sizes, all-expert-idle routes, zero inputs and varying top-2 routes. Every output
dump matches the frozen pre-workspace physical-run hash
`3e8790bfe14117cae5d5d266a61b93afce8ddbedd1b7e952d80be6e3ff242f29`.
This proves default and opt-in output agreement on this partial fixture.

Native CPU tree builds linked against HIP and CUDA CPU libraries pass, as do
W4A16 CPU and HIP/CUDA GPU tree builds. The launcher tests pass (three tests),
plus two invalid-selector checks. No GPU execution, physical IB rerun, SYCL
qualification, tree-service performance or full-model rate is claimed here.
The preceding adapter timing results motivate the option but are not a measured
network-service speedup. Source snapshots, build logs and per-role test logs are
in `results/20261020/tree-workspace/`.

### Physical IB workspace validation and option-count fix (2026-10-20)

Physical testing exposed an argument-count guard left over from before the K32
and workspace selectors. Combining compact NUMA, K32, workspace reuse, persistent
dispatch and the test options made the primary reject the command before opening
transport. The first baseline had passed; the next process failed at CLI parsing,
not during computation. Its raw logs are retained separately. The native-only
limit now admits all ten optional selectors (argc up to 17), and usage lists
both new options. Per-option duplicate and compatibility checks remain intact.

The loopback test now supports primary thread count, compact NUMA and direct-only
persistent dispatch so that the maximal option combination can be exercised.
Defaults of the binary and test tool are unchanged.

After rebuilding and hash-verifying the portable binary on both workers, four
physical-IB runs passed with 48 layer checks each (192 total): K32 direct without
workspace reuse, K32 direct with reuse, K32 relay with reuse, and original-layout
direct with reuse. All use compact NUMA on the primary, sixteen primary CPU
participants and four per worker; direct runs use persistent dispatch. Every role
reports the expected workspace/layout selectors and physical IB transport.
Miners remain running and no GPU executes.

All four output dumps match the frozen SHA256
`3e8790bfe14117cae5d5d266a61b93afce8ddbedd1b7e952d80be6e3ff242f29`, with zero relative
L2 error and bit-exact repeats. The successful rerun uses one verified binary on
all nodes; the pre-fix attempt is not counted among these 192 checks. These are
partial-model correctness checks, not decode throughput: elapsed process time
includes loading, reference execution and deliberate fixture waits. Evidence is
in `results/20261020/tree-workspace-ib/`; private connection settings and raw output
blobs remain in ignored local storage.

The added maximum-option regression also passes on the CUDA-linked CPU binary:
48 direct plus 48 relay loopback layer checks, both output hashes identical to
the frozen physical result. The direct command exercises argc 17. Portable
HIP-linked CPU, CUDA-linked CPU, W4A16 CPU and HIP/CUDA GPU tree builds pass;
three launcher tests pass. GPU execution and SYCL qualification were not run.
No service timing or peak-utilization claim follows from this validation.

### Distributed workspace service timing (2026-10-20)

A scratch tree binary times from dispatch-header construction through receipt of
all expert results. Routing construction, reference execution, output comparison
and fixture waits are outside each window. All nodes use the same verified
portable binary and original-layout offset arithmetic. The primary uses sixteen
participants, compact NUMA and persistent direct dispatch; workers use four each.
Workspace is the only execution selector changed across allocating/reuse/reuse/
allocating process order. Miners remain active; clocks/background load are not
controlled. There are no identical-work duplicate process controls in this screen.

The first four processes retain the correctness fixture's 250 ms waits and three
repeats. Excluding repeat zero leaves only four samples per scenario/run. Results
are inconsistent across forward/reverse comparisons: for example all-expert M16
is 4.3% faster then 4.8% slower; primary-only M1 is 3.2% and 10.1% slower. A
zero-input worker route improves in both repeats, but this sparse-request screen
cannot establish a general service gain. All 192 layer checks pass. Evidence is
in `results/20261020/tree-workspace-service/`; the initial scratch compile's
missing include directory was corrected before execution and its log is retained.

A second scratch binary removes the deliberate waits and uses sixteen repeats.
Reference computation still separates service windows, so this is not a production
continuous-decoder trace. Excluding repeat zero gives thirty samples per scenario
per process. Full service medians in allocating/reuse/reuse/allocating order:

| Scenario | Four run medians (ms) | Reuse/allocating, forward / reverse |
|---|---|---|
| Primary-only M1 | 0.309 / 0.310 / 0.315 / 0.316 | 1.004 / 0.996 |
| Split M7 | 1.641 / 1.669 / 1.655 / 1.710 | 1.017 / 0.968 |
| All experts M16 | 8.112 / 8.220 / 8.139 / 8.227 | 1.013 / 0.989 |
| Leaf-only M3 | 1.951 / 1.952 / 1.913 / 1.954 | 1.000 / 0.979 |
| Zero-input worker M2 | 1.618 / 1.648 / 1.639 / 1.674 | 1.019 / 0.979 |
| All experts M15 | 7.668 / 7.656 / 8.012 / 7.626 | 0.998 / 1.051 |
| Varying top-2 M16 | 5.682 / 5.560 / 5.559 / 5.600 | 0.979 / 0.993 |
| All idle | 0.045 / 0.044 / 0.046 / 0.043 | 0.987 / 1.062 |

There is still no consistent distributed gain. The top-2 scenario favors reuse
in both comparisons, but most others change direction or are effectively tied;
one all-expert M15 repeat regresses 5.1%. Keep workspace opt-in and distinguish
its adapter-only gains from network-service latency. No model tok/s follows
from these partial scenarios.

All 1,024 no-wait layer checks and 1,024 measured service windows complete;
960 windows enter the post-warmup summaries. Every output repetition matches the
frozen original fixture, verified by its three-repeat prefix hash and by equality
of all sixteen repetitions. No GPU or SYCL execution or peak-utilization claim.
The production runner is unchanged. Source, generic settings, raw timing samples
and logs are in `results/20261020/tree-workspace-service-nowait/`; large output
blobs and private connection scripts remain in ignored local storage.

### Correlated CPU and transport service phases (2026-10-20)

Two scratch physical-IB direct runs correlate primary service, primary local
compute and both workers' local compute/receive/send durations by request
sequence. Workspace reuse, original-layout offset arithmetic, compact primary
NUMA and persistent dispatch are enabled. The primary has sixteen CPU
participants, workers four each; miners remain active. Sixteen fixture repeats
run without deliberate sleeps, with reference computation still between service
windows. Repeat zero is excluded from latency summaries.

All 512 layer checks and 512 request correlations pass; 480 post-warmup requests
enter summaries. Outputs match the frozen fixture in every repetition. Durations
are compared without assuming synchronized clocks across nodes. For each request,
the maximum of the three local compute durations provides a lower bound on
service latency; subtracting it does not isolate pure network cost.

| Scenario | Median longest-local / service, two runs | Median service minus longest-local (ms) |
|---|---|---|
| Primary-only M1 | 98.6% / 98.6% | 0.005 / 0.004 |
| Split M7 | 74.8% / 74.7% | 0.415 / 0.423 |
| All experts M16 | 91.1% / 91.6% | 0.738 / 0.694 |
| Leaf-only M3 | 75.4% / 75.7% | 0.480 / 0.481 |
| Zero-input worker M2 | 75.2% / 74.5% | 0.425 / 0.419 |
| All experts M15 | 90.6% / 91.1% | 0.717 / 0.680 |
| Varying top-2 M16 | 88.5% / 88.2% | 0.641 / 0.654 |

These percentages describe service-time composition, **not CPU FLOPS/bandwidth
utilization**. Worker one has the longest measured local compute duration in
59/60 all-expert M16 requests and 55/60 top-2 requests. Small worker-only routes
have about 0.42–0.48 ms beyond local computation. Across both runs, their active
worker receive/send medians are approximately 0.18/0.23 ms (zero-input M2) and
0.20/0.26 ms (leaf-only M3). No negative service-minus-longest-local durations
occur. Profiling/logging can perturb execution; there is no uninstrumented
speed comparison in this screen.

Source inspection identifies a concrete next transport candidate. The current
IB eager threshold is 8,152 bytes, below even one 4,096-float activation (16,384
bytes). Larger unregistered transfers call `get_mr`, register memory for that
transfer and later deregister it. The tree does not currently pre-register its
activation/result buffers. The existing public `ucomm_mr_reg` API supports
pre-registration. A bounded persistent staging buffer per context could avoid
that churn at the cost of copies; this is a hypothesis requiring measurement,
not an attribution of all observed receive/send time to registration.

Evidence is in `results/20261020/tree-cpu-critical-path/`. No production change,
GPU/SYCL execution, full-model rate or peak-utilization qualification occurred.

### Registered staging buffers over physical IB (2026-10-20)

A scratch tree prototype pre-registers one fixed send buffer and one fixed receive
buffer per context through the existing `ucomm_mr_reg` API. Each holds at most
16 × 4,096 floats: 512 KiB per endpoint, 2 MiB across the four endpoints of either
three-node topology. Activation/result sends copy into registered storage; receives
copy back into the existing vectors. Both copies are inside service timing.
Weights, control messages and the wire protocol are unchanged. Registration is
startup work and excluded from service timing.

The context registry is populated before service and read-only while requests run.
Each context retains one owner; direct primary lanes and relay forwarding use
different contexts. Receive/send methods block until their requests complete.
Registrations are released before context finalization. The prototype rejects
oversized payloads and cleans up the first registration if the second fails.
No production selector or default was changed in this screen.

Both topologies use the same verified portable CPU binary with explicit staging
0/1/1/0 process order. Workspace reuse, original-layout offset, primary compact
NUMA and sixteen/four/four CPU participants remain fixed; direct uses persistent
primary dispatch. Sixteen repeats have no deliberate fixture waits, but reference
calculations still separate service windows. Repeat zero is excluded, leaving
thirty samples per scenario/run. Miners remain active; clocks/load are uncontrolled.

| Scenario | Direct staging/baseline, forward / reverse | Relay staging/baseline, forward / reverse |
|---|---|---|
| Primary-only M1, no remote payload | 0.960 / 1.033 | 1.042 / 0.861 |
| Split M7 | 0.783 / 0.773 | 0.587 / 0.604 |
| All experts M16 | 0.955 / 0.958 | 0.889 / 0.899 |
| Leaf-only M3 | 0.828 / 0.820 | 0.755 / 0.721 |
| Zero-input worker M2 | 0.805 / 0.771 | 0.828 / 0.828 |
| All experts M15 | 0.951 / 0.945 | 0.892 / 0.883 |
| Varying top-2 M16 | 0.937 / 0.925 | 0.871 / 0.851 |
| All idle, no payload | 0.820 / 0.933 | 0.966 / 0.891 |

All payload-bearing remote scenarios improve in both comparisons. Direct split
M7 drops from 1.643/1.656 ms baseline to 1.287/1.280 ms with staging; relay drops
from 2.439/2.387 to 1.431/1.442 ms. Direct larger-batch gains are about 4–8%; relay
larger-batch gains about 10–15%. No-payload cases do not exercise staging and show
process-level variation; their changes must not be attributed to registration.
This is a stronger candidate than workspace reuse alone, but not a full-model
speed measurement or proof that all transfer overhead came from registration.

All eight processes pass: 2,048 layer checks and measured service windows, with
1,920 post-warmup windows summarized. All output repetitions match the frozen
fixture. Staging markers verify two buffers per active context and the expected
endpoint counts on all nodes. No GPU/SYCL execution or peak-utilization claim.
Evidence is in `results/20261020/tree-registered-staging/` and
`results/20261020/tree-registered-staging-relay/`. The candidate remains scratch-only
pending integration and focused lifetime/bounds validation; production is unchanged.

### Registered staging integration validation

The tree experiment now accepts opt-in `registered-staging`; both launchers
expose `--registered-staging`. Each transport context keeps separate registered
send and receive buffers, each bounded at 16 x 4096 floats. Registration occurs
before service; buffers are deregistered before transport finalization. The
wire format is unchanged, and the default registers no staging buffers.

On 2026-10-20, CPU-only loopback direct and relay runs passed 192 partial-layer
checks across enabled and default configurations. All four output dumps match
the frozen extended reference byte for byte. The enabled run also exercises
K32, workspace reuse, compact NUMA with 16 primary participants, persistent
direct dispatch, and the maximum native argument count. A fault-injection test
passes ASan/UBSan/LSan for bounds, registration failure cleanup and backing
storage lifetime; LeakSanitizer required execution outside the sandbox because
its initial sandbox run failed under ptrace. Three launcher tests pass.

HIP- and CUDA-linked native CPU trees, W4A16 CPU, and HIP/CUDA GPU trees compile
and link. GPU execution and SYCL validation were not performed. The integrated
option still needs a physical IB retest; the earlier staging prototype's speedup
is not a new integration measurement. No full-model throughput or peak CPU
utilization claim follows. Evidence: `results/20261020/registered-staging-integration/summary.json`.

The subsequent physical IB integration retest also passed: four bounded runs
(staging-enabled direct/relay with K32, then staging-disabled direct/relay with
original layout), all using workspace reuse, passed 192 further partial-layer
checks. All four output dumps match the same frozen reference exactly. Logs
verify IB transport on every role, 16 primary participants, four per worker,
and the expected registered-buffer counts only when enabled. GPUs were not
used. This is correctness evidence, not an isolated staging speed comparison:
the enabled and disabled configurations also differ in packed layout, and
fixture waits remain. The earlier prototype timings remain the performance
evidence. The same integration summary now includes binary and log hashes.

The remote Xeon/V100 experiment report was rechecked and its content hash is
unchanged (`7c5a6a46a166dc78ff41049e29403bd81128446065512585e1be8bbd39ada80d`).
Its selected 32.10 tok/s speculative, route-dropping profile does not establish
an equivalent exact-greedy local performance target. Full-model throughput,
quality and 90% theoretical CPU utilization remain unverified.

### Integrated registered staging: matched physical IB timing

The integrated option was subsequently measured with the same binary and
settings in off/on/on/off order, separately for direct and relay transport.
Both arms use native FP4/Q8, workspace reuse, original packed layout, the
NT1 offset path and prefetch distance 2048. The primary uses 16 participants
with compact NUMA; each worker uses four. Direct mode uses persistent dispatch.
GPUs remain unused and miners remain active. The scratch build adds service
timers, extends the fixture to 16 repeats and removes deliberate waits; it
uses the integrated registered-buffer implementation without modifications.

Each topology passes 1,024 partial-layer checks, giving 2,048 total. Every output
repeat is byte-identical and the first three repeats match the frozen reference.
Exclude repeat zero: each scenario has 30 warm samples per process. The ratios
below compare process medians in forward/reverse order; these are not within-
process paired samples.

| Scenario | Direct staging/baseline | Relay staging/baseline |
| --- | --- | --- |
| Split batch, 7 tokens | 0.768 / 0.807 | 0.614 / 0.597 |
| All experts, 16 tokens | 0.935 / 0.955 | 0.923 / 0.874 |
| Leaf only, 3 tokens | 0.823 / 0.818 | 0.765 / 0.735 |
| Zero-input worker, 2 tokens | 0.808 / 0.775 | 0.856 / 0.805 |
| All experts, 15 tokens | 0.939 / 0.948 | 0.906 / 0.879 |
| Varying top-2 routing, 16 tokens | 0.924 / 0.928 | 0.851 / 0.833 |

The split-batch direct medians change from 1.691/1.652 ms to 1.298/1.333 ms;
relay changes from 2.332/2.441 ms to 1.430/1.457 ms. The no-payload scenarios
are retained in evidence but their timing changes cannot be attributed to
registered activation buffers. Primary-only direct ratios are 0.975/1.200,
showing appreciable run noise even without payload transfers.

The timed window includes dispatch, local compute, activation/result transfers,
staging copies and result receipt. It excludes registration/startup, routing
construction, reference work and output mixing. Reference computation still
separates service windows, so this is not a sustained decoder benchmark. No
full-model tok/s or 90% CPU peak-utilization claim follows. The option remains
off by default. Evidence, raw timing samples, build/source hashes and logs are
in `results/20261020/staging-integrated-timing` and
`results/20261020/staging-integrated-timing-relay`.

### Q3 two-token integer accumulation chains

A scratch Q3 NT2 candidate accumulates even/odd chunks into separate int32
vectors, combines them, and retains the original float conversion, FMA order
and horizontal reduction. Q2 and other token widths remain unchanged. A
conservative per-lane bound is 32 x 4 x 128 x 32 = 524,288, so the integer
regrouping cannot overflow int32. It adds registers and a final integer add;
a shorter dependency chain alone does not guarantee better service time.

The same-object selector was tested with mixed Q2 GU/Q3 down CPU service,
16 rotating weight banks, 20 alternating pairs and 64 calls per sample.
Prefetch 1024, Q2 lookup off, and Clang paths off are identical in both arms.
The primary uses 16 participants; the worker uses four with its miner running.
Each host runs candidate/control/control/candidate processes. Controls compare
the original implementation with itself. Mixed routing uses seven experts,
only one of which has two tokens; a primary dense eight-expert screen makes
all experts exercise the changed path.

| Workload | Candidate/baseline paired median ratios | Unchanged-code controls |
| --- | --- | --- |
| Primary mixed | 1.0013 / 1.0031 | 1.0003 / 1.0034 |
| Worker mixed | 1.0022 / 1.0051 | 1.0015 / 0.9988 |
| Primary dense NT2 | 1.0059 / 0.9743 | 1.0087 / 1.0026 |

The mixed service does not improve. The dense result is inconsistent across
processes; its second candidate run has only 10/20 paired wins despite the
2.6% median improvement. This does not justify integration. Keep the current
kernel and defaults. All 480 final batch output checks after 30,720 timed calls
match the split reference bitwise; this does not check every timed call or
provide an exhaustive scalar oracle. Inputs/routing are synthetic, input
quantization is excluded, and clocks/background load were uncontrolled. No
GPU/SYCL, full-model throughput/quality, or peak-utilization validation follows.
Frozen sources, build settings, raw paired logs and summaries are under
`results/20261020/q3-nt2-int-chains`.

### Refreshed primary FP4 streaming limit

A fresh Threadripper 1950X screen compares the current native NT1 FP4 projection
with a checksum-producing AVX2 read loop over the same 301,989,888 packed bytes
(288 MiB). Each worker first-touches its own weights while pinned. Four/eight/
sixteen threads use 128/64/32 copies respectively, keeping total bytes fixed.
Nine timed passes per mode alternate read/dot order after one warmup each.
Every FP4 output is checked against GGML and each read checksum is verified,
outside timing. Fourteen process runs pass all 252 timed and 28 warmup passes.

| Configuration | Read payload GB/s | FP4 payload GB/s | FP4 active cores |
| --- | --- | --- | --- |
| Original, 16 physical threads | 82.96 / 83.13 | 76.74 / 77.90 | 15.06 / 15.34 |
| Offset + prefetch 2048, 16 threads | 82.54 / 83.40 | 78.16 / 70.57 | 15.03 / 13.94 |
| Original, 8 threads balanced across nodes | 77.29 / 76.14 | 45.02 / 43.26 | 7.92 / 7.84 |
| Original, 16 threads balanced across nodes | 83.52 / 83.33 | 76.36 / 77.12 | 15.22 / 15.18 |
| Original, 4 threads balanced across nodes | 76.80 / 69.36 | 22.39 / 13.89 | 3.94 / 3.10 |

The original sixteen-thread paired FP4/read payload ratios are 0.928/0.946.
This is much closer to the measured read rate than the older, low-CPU-use
screen. It is not proof of 90% theoretical memory bandwidth: these are payload
rates without DRAM counters, and memory speed/channel configuration remains
unverified (a noninteractive privileged DMI query failed). Active cores are
process CPU time divided by wall time, not instruction efficiency or FLOPS.
The late four-thread outlier and the slower second tuned sixteen-thread run
are retained; this screen does not establish a repeatable tuned-kernel gain.

Physical-prefix eight-thread runs stay on node zero: read is 42.35/42.49 GB/s
and tuned FP4 is 42.20/42.07 GB/s. Balancing eight threads across both nodes
nearly doubles available read throughput, while original FP4 remains around
43–45 GB/s. At sixteen threads FP4 approaches the read rate. This motivates
further execution-throughput work at low thread counts, especially the worker
four-participant budget, rather than inferring that all cases are limited by
DRAM bandwidth. The node-placement comparison also changes the tuned arithmetic
setting, so it is not an isolated NUMA speedup comparison.

This is synthetic row streaming, not full expert service or model throughput;
activation quantization is excluded. No production path or default changed.
No GPU was used and miners were left running. Evidence and source/build hashes:
`results/20261020/stream-refresh`.

### Four-thread FP4 arithmetic/prefetch screen

A follow-up isolates original/offset arithmetic and prefetch off/2048 at four
threads, first forward then reverse order. Both hosts stream 288 MiB with the
same GGML-checked benchmark. The Threadripper primary alternates NUMA nodes;
the Ryzen 3950X worker uses four physical cores and retains its running miner.
Only settings change within each host. Each process alternates nine timed read
and dot passes after warmup; setting comparisons are across processes.

| Worker configuration | FP4 payload GB/s, forward/reverse |
| --- | --- |
| Original, no prefetch | 12.827 / 13.025 |
| Offset, no prefetch | 13.494 / 13.466 |
| Original, prefetch 2048 | 15.076 / 14.977 |
| Offset, prefetch 2048 | 15.047 / 15.277 |

Prefetch 2048 improves original-arithmetic payload throughput by 17.5%/15.0%
against the corresponding baseline runs. Offset alone improves 5.2%/3.4%; its
increment beyond prefetch varies from -0.2% to +2.0%. These are projection-only
measurements and do not supersede complete-adapter/service evidence.

A second worker sweep keeps original arithmetic and compares prefetch distances
1024, 2048, 4096 and 8192, then reverses the order:

| Prefetch distance, bytes | FP4 payload GB/s, forward/reverse |
| --- | --- |
| 1024 | 14.259 / 14.238 |
| 2048 | 15.167 / 14.945 |
| 4096 | 15.205 / 15.000 |
| 8192 | 15.271 / 15.233 |

The 4096 difference is small; 8192 is 0.7%/1.9% above 2048. There are no
same-setting duplicate controls in this sweep, so this is only a candidate
for a paired complete-service test, not evidence to change the current 2048
experimental setting or any default. Dot/read paired payload ratios span
0.818–0.889 across those three distances; they are not theoretical bandwidth
utilization. The primary screen has large retained low-CPU-use outliers
(12.2–13.5 versus 22.4–24.2 GB/s), preventing a reliable setting recommendation.

All 24 processes pass 432 timed and 48 warmup passes with every FP4 output and
read checksum checked outside timing. No GPU, full-model or theoretical peak
qualification was performed. Production code/defaults remain unchanged.
Evidence: `results/20261020/stream-four-settings`.

### Worker prefetch 8192: complete-adapter confirmation

The small 8192-byte streaming advantage was tested against 2048 through current
`NativeFp4CpuExpert::run_into`, with a same-object runtime selector changing only
prefetch distance. Both arms retain original packed layout, NT1 offset arithmetic
and caller-owned activation/output buffers. The timed call includes finite
checks, activation quantization, native expert execution, small-hidden rescaling
and output; it excludes startup loading/packing. Eight independent weight copies
per arm rotate through real partial layers 3/4, expert zero. The worker uses four
CPU participants; its miner is unchanged and no GPU is used.

Candidate/control/control/candidate process order provides two candidate runs
and two unchanged-2048 controls. Each layer/token cell contains 20 alternating
pairs, each timing 64 calls. Candidate/baseline paired-median ratios are:

| Tokens | Candidate ratios, both layers/runs | Unchanged-2048 controls |
| --- | --- | --- |
| 1 | 1.0109, 1.0016, 0.9938, 0.9952 | 1.0122, 0.9802, 0.9944, 1.0149 |
| 2 | 0.9988, 1.0165, 1.0027, 0.9953 | 1.0004, 0.9840, 1.0044, 1.0175 |

There is no repeatable complete-adapter gain beyond control variation. Retain
2048 in the experimental worker configuration; do not promote 8192 from the
streaming-only result. All 640 final batch checks after 40,960 timed calls match
the baseline output bitwise. Checks occur after batches, not every timed call.
These repeated partial experts do not represent a model routing trace or full
model throughput. CPU peak utilization remains unqualified. No production
kernel/default changed; sources, commands, hashes and raw paired samples are in
`results/20261020/adapter-prefetch-worker`.

### Current worker adapter phase profile

A scratch instrumented copy of the current native adapter measures setup/input
validation, input quantization plus job construction, pool execution, output
restoration and final finite checks. Pool counters divide execution into GU,
hidden transform/quantization and down phases. Both alternating arms run the
same code and reuse caller-owned buffers; this is a profile, not a speedup test.
The worker uses four CPU participants, eight copied banks per arm, original
layout, NT1 offset arithmetic and prefetch 2048. Real partial layers 3/4 expert
zero run at NT1/2, with small-hidden preservation enabled. Miners are unchanged.

Two process runs give these ranges of per-cell median fractions:

| Work | Fraction of full adapter time |
| --- | --- |
| Gate/up pool phase | 64.05–64.71% |
| Down pool phase | 34.28–34.89% |
| Hidden transform and quantization | 0.22% at NT1, 0.36% at NT2 |
| All work outside pool | 0.39–0.41% at NT1, 0.63–0.64% at NT2 |

Full-call medians range 1.766–1.821 ms at NT1 and 1.941–1.995 ms at NT2.
Removing all measured adapter work outside the pool would save less than 0.7%
under these conditions. Further finite-scan/workspace cleanup is therefore a
lower priority than the GU/down phases. The latter include task dispatch and
completion waits, so the profile does not establish pure arithmetic cost or
rule out scheduling/imbalance. Independent medians need not sum exactly.

All 320 final batch checks after 20,480 timed calls match the reference bitwise.
Timers add overhead, correctness checks occur after batches, and fixed repeated
partial weights do not model real routing. No production change, GPU test,
full-model throughput or theoretical-utilization claim follows. Evidence,
profile source, logs and source/build hashes are under
`results/20261020/adapter-phase-worker`.

### Worker task granularity: one-token candidate

Using the current pool's existing diagnostic task setter, a paired complete-
adapter screen compares one and six native tasks per participant against three.
The setter runs only between completed calls. Both arms reuse caller-owned
buffers, original FP4 layout, NT1 offset arithmetic and prefetch 2048. The
worker retains four CPU participants and its running miner. Real partial layers
3/4 expert zero rotate through eight independent banks per arm at NT1/2.

An initial one/six/three/three/six/one process sweep uses ten alternating pairs
and 32 calls per sample. One task is slower in all eight cells (1.0–9.6%). Six
tasks improve all four NT1 paired medians, but unchanged-setting controls are
noisy. A separate six/three/three/six confirmation therefore uses twenty pairs
and 64 calls per sample:

| Tokens | Six/three paired-median ratios, both layers/runs | Three/three controls |
| --- | --- | --- |
| 1 | 0.9609, 0.9687, 0.9504, 0.9460 | 1.0175, 1.0420, 1.0328, 0.9918 |
| 2 | 1.0141, 0.9840, 1.0114, 0.9964 | 0.9881, 1.0114, 0.9877, 0.9985 |

For NT1, whole-run medians change from 1.730–1.782 ms to 1.682–1.697 ms.
Candidate paired wins are 11/20, 14/20, 17/20 and 13/20. This supports six tasks
as a one-token worker candidate under these conditions, but control variation
reaches 4.2%, individual pairs vary, and NT2 has no consistent benefit. Do not
change the global three-task default or apply six tasks to all tree batches.
Mixed-routing service validation is still needed before a token-dependent
scheduler policy is integrated. The Q2/Q3 paths were not retested at six tasks.

The initial and confirmation runs pass 1,120 final batch checks after 56,320
timed calls, all bitwise identical to reference. Input quantization, finite
checks, hidden rescaling and pool execution are timed; loading and packing are
excluded. This is repeated partial-expert service, not a full-model routing
trace or 90%-peak result. No production/default change or GPU execution.
Evidence: `results/20261020/adapter-task-worker` and
`results/20261020/adapter-task-worker-confirmation`.

### Token-dependent worker tasks over physical IB

The existing eight-scenario tree fixture never submits a one-token worker call;
its only one-token case is primary-only. A scratch extension adds scenario 8
(one token per node) and scenario 9 (one leaf token). The original eight outputs
in every repetition still match the frozen reference. The new fixture uses ten
scenarios, two layers and sixteen repeats, with 320 checked service windows per
process. All outputs match between baseline/candidate runs, including the new
cases. Worker counters verify exactly 32 middle-node and 64 leaf NT1 calls per
run, and six-task selection only in the enabled arm. Larger worker calls and all
primary calls retain three tasks per participant.

The scratch policy uses the current scheduling setter at call boundaries; no
arithmetic changes. Workspace reuse and registered staging are on in both arms.
The primary uses 16 participants and compact NUMA, workers use four. Direct
mode uses persistent dispatch. Separate off/on/on/off runs for direct and relay
exclude repeat zero, leaving 30 warm samples per scenario per process.

| Scenario | Direct candidate/baseline, forward/reverse | Relay candidate/baseline, forward/reverse |
| --- | --- | --- |
| One token per node | 0.9761 / 0.9842 | 1.0072 / 0.9651 |
| One leaf token | 0.9500 / 0.9556 | 0.9021 / 0.9486 |

Leaf-only direct medians change from 1.088/1.072 ms to 1.033/1.025 ms; relay
changes from 1.191/1.173 ms to 1.074/1.113 ms. The leaf case improves in both
orders on both topologies. Split NT1 improves in direct mode but is inconsistent
in relay mode. Unchanged scenarios retain timing variation, including direct
primary-only ratios 0.968/1.098. Treat these as process-median comparisons, not
paired individual-call measurements or proof that all NT1 routing improves.

Across eight processes, 2,560 layer checks and output repeat checks pass, with
2,400 warm timing windows. Timers include dispatch, local work, staging copies,
IB transfers and result receipt; routing construction, reference work, output
mixing and startup are excluded. Reference work separates requests. Miners and
clocks were uncontrolled; no GPU execution, full-model throughput/quality or
90%-peak qualification follows. The result supports an opt-in worker-NT1 policy
candidate, not a global task-default change. Production code remains unchanged;
Q2/Q3 task-policy behavior was not tested. Evidence:
`results/20261020/tree-nt1-tasks` and `results/20261020/tree-nt1-tasks-relay`.

### Opt-in worker NT1 scheduling integration

The native partial-model tree now accepts `worker-nt1-tasks` on four-participant
workers. Both Python launchers expose `--worker-nt1-tasks` and forward it only
to worker roles. Enabled workers select six native tasks per participant for
one-token calls and three otherwise, including when workspace reuse is off.
This explicit policy overrides the worker's task-count environment setting on
each call; it is not a new global default. Primary use, other worker thread
counts and repeated flags are rejected before connecting. GPU/W4A16 modes do
not accept the native-only option. Policy markers and final call counts allow
validation to check actual activation.

Initial local validation passes 336 layer checks: original direct/relay cases
match the frozen extended hash, and scratch ten-scenario fixtures exercise the
one-token branch both with allocating calls and workspace reuse. The middle
worker reports six NT1 calls and the leaf twelve per topology/run, matching
routing. New fixture outputs match the prior physical-IB baseline bytes. The
production fixture itself is unchanged. Five launcher tests pass, including
worker-only forwarding and rejection of missing policy evidence; three native
CLI rejection checks pass. HIP- and CUDA-linked native CPU, W4A16 CPU and
HIP/CUDA GPU tree builds compile/link. No GPU or SYCL execution was performed.

These checks validate integration, not a new performance measurement. The
integrated physical-IB retest remains pending; the preceding scratch timings
remain the performance evidence. No full-model throughput, quality or 90%-peak
claim follows. Evidence: `results/20261020/nt1-policy-integration`.

The option-disabled CUDA-linked CPU run subsequently passed 96 additional
original-fixture checks across direct/relay, bringing integration validation to
432 checks. Both default output dumps match the frozen extended reference and
contain no task-policy markers. The evidence summary includes these runs.

The integrated worker-NT1 option subsequently passed physical IB validation in
both direct and relay configurations, enabled and disabled. Four sixty-case
runs add 240 layer checks, bringing local plus physical integration checks to
672. The fixture differs from current tree source only by the two added NT1
routing cases; source equality was checked before reusing its portable binary.
The launcher differs only in fixture counts/output extent and uses the public
`--worker-nt1-tasks` forwarding and evidence checks.

Enabled worker logs report six NT1 calls on the middle node, twelve on the leaf,
and thirty larger calls on each; policy markers are absent on the primary and
all disabled runs. All four output dumps match the previous extended fixture
exactly. Every role confirms physical IB and CPU-only execution, with sixteen
primary/four worker participants. Workspace reuse and registered staging are on
throughout. These are correctness runs with fixture waits, not new performance
measurements. Earlier scratch timings remain the measured speed evidence.
No GPU or full-model/theoretical-utilization qualification ran. Evidence:
`results/20261020/nt1-policy-ib`.

### Q2/Q3 worker scheduling screen

The existing native task setter was tested separately for Q2 GU/Q3 down on the
four-participant Ryzen 3950X worker. Both arms use identical compiled arithmetic,
prefetch 1024, lookup off and Clang paths off. Three tasks per participant is
the baseline, six the candidate; setters run only between completed submissions.
Synthetic fixed routes rotate sixteen weight banks. Each configuration runs
candidate/control/control/candidate processes, with twenty alternating pairs of
64 calls. Controls retain three tasks in both arms. The miner remains active.

| Workload | Six/three paired median ratios | Three/three controls | Candidate paired wins |
| --- | --- | --- | --- |
| NT1, two experts, split service | 0.9719 / 0.9712 | 1.0014 / 0.9952 | 16/20, 14/20 |
| Mixed NT2, seven experts, dataflow | 0.9833 / 0.9928 | 1.0043 / 1.0002 | 15/20, 11/20 |

For NT1, whole-run medians change from 1.960/1.966 ms to 1.895/1.880 ms.
For mixed routing they change from 5.016/4.997 ms to 4.935/4.940 ms. NT1 GU
phase paired ratios are 0.9599/0.9498 and down ratios 0.9684/0.9902; these phase
timers include dispatch and completion waits. The smaller mixed-route gain is
less consistent across individual pairs. These results justify further task-
granularity validation for Q2/Q3, not automatically applying the FP4 tree's NT1
policy or changing global defaults. Other expert counts, widths and real-model
routing remain untested by this screen.

All 320 final batch output checks after 20,480 timed calls are bitwise identical
to the split reference. Input quantization is excluded; every final batch is
checked, not each timed call. No GPU, full-model throughput/quality, or peak-
utilization qualification ran. Source/default behavior is unchanged. The reused
harness's obsolete prefetch comment was corrected and a rebuild verified an
identical binary hash. Evidence, exact settings, source/library hashes and raw
paired samples: `results/20261020/q23-task-worker`.

### Q2/Q3 task scheduling with three experts

The four-participant worker task comparison was extended to three experts at
NT1 split service and dense NT2 dataflow. Same compiled Q2 GU/Q3 down kernels,
16 rotating banks, prefetch 1024, lookup/Clang off, twenty alternating pairs and
64 calls per sample are retained. Each geometry again runs candidate/control/
control/candidate processes; candidate selects six tasks versus three, controls
select three in both arms. The miner remains running; no GPU work is performed.

| Workload | Six/three paired median ratios | Three/three controls | Candidate paired wins |
| --- | --- | --- | --- |
| NT1, three experts, split | 0.9457 / 0.9529 | 1.0036 / 1.0027 | 19/20, 20/20 |
| NT2, three experts, dataflow | 0.9900 / 0.9905 | 1.0016 / 0.9972 | 13/20, 13/20 |

Whole-run NT1 medians change from 2.312/2.305 ms to 2.193/2.198 ms. Dense NT2
changes from 3.221/3.234 ms to 3.195/3.207 ms. The stronger NT1 improvement
survives both orders and is larger than controls; the roughly 1% NT2 result is
smaller and has fewer paired wins. Along with the preceding two-expert result,
this supports six tasks as a worker NT1 tuning candidate for Q2/Q3. It does not
establish a universal policy for all expert counts, larger widths, primary CPU
execution or real-model routing. Global defaults and the FP4-specific tree
option remain unchanged.

All 320 final batch checks after 20,480 timed calls match the split reference
bitwise. Inputs are prequantized, weights/routing are synthetic and background
load/clocks are uncontrolled. No full-model throughput/quality or theoretical-
utilization claim follows. Evidence and raw pairs:
`results/20261020/q23-task-shapes`.

### Second-worker Q2/Q3 task validation

The same portable executable was copied unchanged to the second worker, whose
CPU was reverified as a Ryzen 9 3950X (16 cores / 32 threads). Four participants
are used; its miner remains running. Three-expert NT1 split and dense NT2
flow repeat the first worker's three/six task comparison, prefetch 1024,
16 banks, twenty alternating pairs of 64 calls, and candidate/control/control/
candidate process order. No arithmetic, input or routing changes are made.

| Workload | Six/three paired median ratios | Three/three controls | Candidate paired wins |
| --- | --- | --- | --- |
| NT1, three experts, split | 0.9769 / 0.9707 | 1.0082 / 1.0173 | 15/20, 18/20 |
| NT2, three experts, dataflow | 0.9901 / 0.9921 | 1.0028 / 0.9977 | 15/20, 16/20 |

NT1 whole-run medians change from 2.258/2.257 ms to 2.209/2.187 ms. NT2 changes
from 3.218/3.218 ms to 3.189/3.199 ms. Thus the one-token improvement repeats
on the second worker but is smaller (2.3–2.9%) than the first worker's 4.7–5.4%;
the dense two-token gain is again below 1%. Controls show up to 1.7% variation,
so neither magnitude should be treated as a hardware-wide constant. This
supports retaining six tasks as a qualified worker candidate, not a universal
or primary-host default.

All 320 final batch checks after 20,480 timed calls pass. Every output hash also
matches the first worker for the same geometry. Inputs are prequantized and the
weights/routes are synthetic; correctness is checked at batch boundaries.
No GPU, full-model, quality or theoretical-utilization qualification ran, and
production/default settings remain unchanged. Evidence and identical-binary
provenance: `results/20261020/q23-task-second-worker`.

### Primary Q2/Q3 task-granularity control

The three/six-task test was repeated on the Threadripper 1950X primary with
fifteen workers plus the host, NUMA-owned rows, three experts and sixteen
rotating banks. Arithmetic, prefetch 1024, Q2 lookup/Clang settings, twenty
alternating pairs and 64 calls per sample match the worker screen. NT1 uses
split service and dense NT2 uses dataflow. Both geometries run candidate/
control/control/candidate process order; controls retain three tasks.

| Workload | Six/three paired median ratios | Three/three controls | Candidate paired wins |
| --- | --- | --- | --- |
| NT1, three experts, split | 0.9937 / 0.9814 | 0.9938 / 1.0090 | 15/20, 14/20 |
| NT2, three experts, dataflow | 1.0007 / 1.0006 | 1.0033 / 1.0008 | 8/20, 9/20 |

The first NT1 candidate is indistinguishable from one control. The second NT1
run has whole-run medians 0.483/0.455 ms, a larger apparent change than its
0.9814 paired ratio; retain both rather than treating the unpaired ratio as
confirmation. NT2 is effectively tied. This is insufficient evidence to change
the primary's three-task setting. The worker's stronger one-token result should
remain qualified by host, CPU budget and tested routing shapes.

All 320 final batch checks after 20,480 timed calls match reference bitwise.
Input quantization is excluded; these are synthetic repeated expert calls, not
model decode. No GPU, full-model quality/throughput or theoretical-utilization
qualification was performed. No production/default change. Evidence:
`results/20261020/q23-task-primary`.

### Q2/Q3 worker: twelve tasks versus six

A further four-participant worker screen uses six native tasks per participant
as baseline and twelve as candidate, with six/six controls. Three experts,
16 rotating banks, prefetch 1024, arithmetic settings, twenty alternating pairs
and 64 calls per sample remain fixed. Candidate/control/control/candidate order
is repeated for NT1 split service and dense NT2 dataflow; the miner remains on.

| Workload | Twelve/six paired median ratios | Six/six controls | Candidate paired wins |
| --- | --- | --- | --- |
| NT1, three experts, split | 0.9696 / 0.9735 | 1.0045 / 1.0070 | 18/20, 17/20 |
| NT2, three experts, dataflow | 0.9946 / 0.9941 | 1.0078 / 1.0005 | 12/20, 14/20 |

NT1 whole-run medians change from 2.228/2.240 ms to 2.169/2.182 ms. Twelve tasks
therefore remain a candidate for this worker's one-token Q2/Q3 workload rather
than assuming six is optimal. NT2 improves less than 0.6% by paired medians,
which is small relative to controls. Do not compound this screen's ratios with
previous three/six results into an unmeasured total speedup. A direct comparison
to the original setting and second-worker validation remain necessary before
recommending twelve beyond this fixture. The existing FP4 tree policy remains
six at NT1 and is not changed by Q2/Q3-only evidence.

All 320 final batch checks after 20,480 timed calls match reference bitwise.
Input quantization is excluded; weights/routes are synthetic; clocks and
background load are uncontrolled. No GPU, full-model throughput/quality or
90%-peak qualification ran. Production/default settings remain unchanged.
Evidence: `results/20261020/q23-task-twelve`.

The remote Xeon/V100 experiment report was also revalidated: its SHA-256 remains
`7c5a6a46a166dc78ff41049e29403bd81128446065512585e1be8bbd39ada80d`. No new result
changes the earlier interpretation of its 32.10 tok/s speculative/route-dropping
profile. The report revalidation is included with this screen's evidence.

### Direct twelve/three Q2/Q3 comparison on both workers

Twelve tasks were compared directly against the original three-task setting on
both Ryzen 3950X workers, using the same portable binary, four participants per
host, three experts and sixteen rotating banks. This avoids compounding ratios
from separate three/six and six/twelve experiments. Prefetch remains 1024 and
arithmetic/lookup/Clang settings are identical. Each geometry runs candidate/
control/control/candidate processes, twenty alternating pairs and 64 calls per
sample. Controls retain three tasks. The two hosts run independently at the
same time with miners unchanged.

| Worker/workload | Twelve/three paired median ratios | Three/three controls | Candidate paired wins |
| --- | --- | --- | --- |
| Worker one, NT1 split | 0.9316 / 0.9445 | 0.9927 / 0.9976 | 20/20, 20/20 |
| Worker two, NT1 split | 0.9518 / 0.9582 | 0.9980 / 1.0062 | 20/20, 20/20 |
| Worker one, dense NT2 flow | 0.9793 / 0.9875 | 0.9958 / 0.9990 | 17/20, 17/20 |
| Worker two, dense NT2 flow | 0.9837 / 0.9871 | 0.9964 / 1.0021 | 17/20, 14/20 |

NT1 whole-run medians on worker one change from 2.320/2.304 ms to 2.154/2.175 ms;
worker two changes from 2.263/2.281 ms to 2.165/2.178 ms. All eighty NT1 pairs
favor twelve tasks. Dense NT2 gains are smaller. Twelve is therefore a stronger
Q2/Q3 tuning candidate than six for these tested three-expert worker shapes,
using the existing `STRATA_NATIVE_TASKS_PER_THREAD=12` opt-in. This is not a
recommendation for the primary, other CPU budgets, all routing shapes or the
FP4 tree policy, which retains its separately validated settings. Defaults are
unchanged; the setting is not applied automatically to running services.

All 640 final batch checks after 40,960 timed calls pass, and output hashes match
across both workers for each geometry. Synthetic weights/routes, prequantized
inputs and uncontrolled background load limit applicability. No GPU execution,
full-model throughput/quality or theoretical-utilization claim follows.
Evidence: `results/20261020/q23-task-twelve-direct`.

### FP4 complete adapter: twelve tasks versus six

The four-participant worker FP4 adapter was tested separately with twelve tasks
per participant against the validated six-task candidate. Both arms reuse
caller-owned output/activation buffers, original packed layout, NT1 offset
arithmetic, prefetch 2048 and small-hidden preservation. Real layers 3/4 expert
zero rotate through eight independent banks per arm. Candidate/control/control/
candidate processes use twenty alternating pairs of 64 full adapter calls;
controls retain six tasks in both arms. The miner is unchanged and GPUs unused.

| Tokens | Twelve/six paired median ratios, both layers/runs | Six/six controls |
| --- | --- | --- |
| 1 | 0.7755, 0.7963, 0.7569, 0.7310 | 1.0189, 0.9906, 0.9947, 1.0012 |
| 2 | 0.9468, 0.9503, 0.9034, 0.9511 | 0.9978, 1.0126, 0.9807, 1.0065 |

Every one of eighty NT1 pairs favors twelve tasks. Whole-run NT1 medians change
from 1.647–1.698 ms to 1.242–1.330 ms. NT2 candidate wins are 19/20, 17/20,
17/20 and 18/20; whole-run medians change from 1.919–1.974 ms to 1.809–1.861 ms.
The observed effect is larger than the preceding Q2/Q3 task-granularity gain;
no causal claim about arithmetic, cache behavior or hardware utilization is
made from timing alone. Scheduling changes the row chunking and work sharing,
while arithmetic and output bytes remain identical.

All 640 final batch checks after 40,960 timed calls match reference bitwise.
Full calls include finite scans, input quantization, hidden rescaling and native
expert processing; startup packing is excluded. Checks are after batches, not
every timed call. This remains one worker's repeated partial-expert result;
second-worker and distributed IB confirmation are needed before changing the
integrated worker-NT1 policy. That policy remains six tasks and defaults remain
unchanged. No full-model throughput/quality or 90%-peak qualification follows.
Evidence: `results/20261020/adapter-task-twelve`.

### Second-worker FP4 twelve/six adapter confirmation

The identical portable adapter executable was tested on the second Ryzen 3950X
worker with four CPU participants, the same real layers 3/4 expert-zero inputs,
eight banks per arm, workspace reuse, original layout, offset arithmetic and
prefetch 2048. Candidate/control/control/candidate order again uses twenty
alternating pairs of 64 calls. Candidate selects twelve tasks versus six;
controls retain six tasks in both arms. The miner remains active and GPUs unused.

The initial attempt failed before timing because this worker had no local
partial-weight fixture: previous tree tests streamed weights from the primary.
Only the existing two-layer fixture was copied (80,637,192-byte weight file plus
12,912-byte index), and both SHA-256 hashes were verified remotely. The failed
attempt is preserved in ignored local scratch; fixture hashes and failure scope
are recorded in shared evidence. No full checkpoint was transferred.

| Tokens | Twelve/six paired median ratios, both layers/runs | Six/six controls |
| --- | --- | --- |
| 1 | 0.8593, 0.8368, 0.8719, 0.8498 | 0.9888, 1.0067, 0.9937, 1.0046 |
| 2 | 0.9368, 0.9498, 0.9404, 0.9358 | 0.9938, 0.9928, 0.9989, 1.0119 |

All eighty NT1 pairs favor twelve tasks. Whole-run NT1 medians change from
1.290–1.330 ms to 1.111–1.134 ms. NT2 wins are 18/20, 19/20, 20/20 and 18/20;
whole-run medians change from 1.705–1.725 ms to 1.600–1.618 ms. Thus the gain
repeats on this worker, though the 12.8–16.3% NT1 improvement is smaller than
the first worker's result. Host-specific magnitude must not be generalized.

All 640 final batch checks after 40,960 timed calls match reference bitwise.
Full adapter work is timed, startup packing is excluded, and correctness checks
follow batches rather than every timed call. This is not a routing trace or
full-model throughput/quality measurement. Distributed IB confirmation remains
necessary before changing the integrated six-task NT1 policy, which remains
unchanged. No theoretical-utilization claim follows. Evidence:
`results/20261020/adapter-task-twelve-second`.

### Twelve/six FP4 worker tasks over physical IB

The complete adapter gain was checked in the three-node CPU tree, using sixteen
primary participants and four per worker, registered staging, reusable workspace,
original packed weights, offset arithmetic and prefetch 2048. Miners remained
active; GPUs were unused. Baseline/candidate/candidate/baseline processes select
six/twelve tasks only for worker NT1 calls. Primary and larger worker calls retain
three tasks. Each process runs ten scenarios across two real partial layers for
sixteen repetitions; the first repetition is excluded from timing summaries.
Direct links use persistent dispatch; relay links use the normal service path.

| Topology and NT1 scenario | Twelve/six process-median ratios, forward / reverse |
| --- | --- |
| Direct, one token per node | 1.0121 / 0.9899 |
| Direct, one leaf token | 1.0077 / 1.0173 |
| Relay, one token per node | 0.9724 / 0.9837 |
| Relay, one leaf token | 0.9900 / 1.0105 |

The adapter's larger gain does not reproduce in direct distributed service.
Relay one-token-per-node service improves 1.6–2.8%, but leaf-only results are
mixed. Retain the integrated six-task opt-in policy: this evidence does not
justify replacing it with twelve. Cache residency, service scheduling and host
load differ from the adapter benchmark; these timings do not isolate a cause.
Unchanged scenarios and all run medians are preserved, including idle-service
variation. Ratios compare process medians, not paired individual calls.

All 2,560 partial-layer checks pass with zero relative difference; outputs match
across settings, repetitions and the frozen original eight-scenario prefix.
Worker counters confirm 32/64 NT1 calls on the middle/leaf nodes and 160 larger
calls on each per process; primary has no worker-policy marker. The 2,400 warm
service windows include IB, staging copies, compute and result receipt, but
exclude routing, reference calculation and final mixing. No full-model routing,
decode throughput, quality or theoretical-peak claim follows. Defaults remain
unchanged. Evidence: `results/20261020/tree-nt1-twelve` and
`results/20261020/tree-nt1-twelve-relay`.

### Worker compute timing explains the next measurement priority

A separate scratch build adds a steady-clock timer around each worker
`run_into`, buffering samples until shutdown to keep printing outside service.
The same six/twelve ABBA comparison, two layers, ten scenarios, sixteen repeats,
CPU budgets and physical direct/relay links were rerun. All 2,560 checks pass
with frozen-prefix and cross-setting output equality. There are 2,400 warm
service windows; sample sequence, layer and token counts are checked against
the primary log before worker and service times are associated.

| Worker compute scope | Twelve/six forward / reverse median ratios |
| --- | --- |
| Direct, middle, one token per node | 1.0258 / 1.0147 |
| Direct, leaf, one token per node | 0.9600 / 0.9955 |
| Direct, leaf-only token | 0.9793 / 0.9781 |
| Relay, middle, one token per node | 1.0106 / 1.0316 |
| Relay, leaf, one token per node | 0.9869 / 0.9748 |
| Relay, leaf-only token | 0.9892 / 1.0084 |

The missing standalone-adapter gain is not explained solely by network time:
middle-worker compute itself does not improve. Direct leaf-only compute changes
from about 0.925 ms to 0.904–0.906 ms. Its paired service-minus-compute median
is 0.080–0.091 ms across runs; relay leaf-only residuals are 0.174–0.183 ms.
These residuals include dispatch, transport, copies and scheduling; they are
not pure link latency. For concurrent scenarios a worker subtraction does not
isolate the critical path. The instrumented relay service improvements also
vary in sign, reinforcing the decision to retain six tasks.

The tree repeatedly uses two experts per worker, whereas the standalone adapter
rotates through eight banks per arm. Test working-set size under otherwise
matched conditions next; this is a hypothesis, not an established cache cause.
Timers, uncontrolled load and different service scheduling remain limitations.
No default change, GPU execution, full-model throughput or peak-utilization
claim. Source hashes and raw compute/service records are in
`results/20261020/tree-task-profile` and `results/20261020/tree-task-profile-relay`.

### Adapter working-set control for worker task granularity

The first worker's complete NT1 adapter comparison now varies one, two and eight
independent weight banks per arm in the same executable. Everything else stays
fixed: real layers 3/4 expert zero, four CPU participants, explicitly pinned host
participant, original layout, offset arithmetic, prefetch 2048, preserved small
hidden values and reused workspace. Each bank count runs candidate/control/
control/candidate processes, reversing bank order between groups. Each layer has
twenty alternating pairs of 64 calls; candidate selects twelve versus six tasks,
and controls select six in both arms. Miners remain active and GPUs unused.

| Banks per arm | Twelve/six paired median ratios, both layers and runs | Six/six controls |
| --- | --- | --- |
| 1 | 0.7417, 0.6975, 0.7195, 0.7342 | 1.0119, 1.0160, 1.0061, 1.0096 |
| 2 | 0.7721, 0.7431, 0.7392, 0.7417 | 0.9921, 1.0029, 0.9804, 1.0221 |
| 8 | 0.7648, 0.7312, 0.7511, 0.7630 | 1.0080, 1.0303, 1.0015, 0.9819 |

The gain persists with one bank: bank rotation alone does not explain the
standalone/tree discrepancy. One-bank whole-run medians are 1.665–1.701 ms for
six tasks and 1.186–1.259 ms for twelve; eight-bank medians are 1.717–1.726 ms
and 1.294–1.304 ms. These are repeated partial-expert measurements, not a
full-model stream. Both arms coexist in memory; no cache-counter causal claim
is made. Code inspection identifies a remaining difference: this adapter pins
its participating host thread, whereas the tree does not explicitly pin that
thread. A matched host-affinity comparison is the next test, not a proven
explanation or a reason to change tree defaults.

All 960 final batch checks after 61,440 timed calls match reference bitwise.
Checks follow batches, not every timed call. Startup packing is excluded, host
load is uncontrolled, and neither full-model quality/throughput nor theoretical
peak utilization was measured. Production settings and the integrated six-task
policy remain unchanged. Evidence: `results/20261020/adapter-bank-tasks`.

### Host-affinity control: twelve-task gain depends on placement

A matched adapter executable tests pinned and unpinned host participation on
the first Ryzen 3950X worker, with its miner unchanged. Both modes retain three
pinned pool workers plus one participating host thread: allowing host migration
does not add simultaneous participants. The probe checks the host affinity mask
at startup (one allowed CPU when pinned, 32 when unpinned) and fails if pinning
does not succeed. One weight bank per arm, real layers 3/4, NT1, arithmetic,
workspace reuse and prefetch settings match the preceding test. Each affinity
mode runs candidate/control/control/candidate processes with twenty alternating
pairs of 64 calls per layer; affinity order reverses between process groups.

| Host affinity | Twelve/six paired median ratios, both layers and runs | Six/six controls |
| --- | --- | --- |
| Pinned | 0.7149, 0.7064, 0.7294, 0.6996 | 1.0061, 0.9887, 0.9858, 0.9862 |
| Unpinned | 0.9572, 1.0247, 0.9930, 1.0159 | 1.0124, 1.0318, 0.9826, 0.9762 |

The large twelve-task gain repeats only with the pinned host. Pinned candidate
processes have six-task whole-run medians 1.645–1.678 ms and twelve-task medians
1.167–1.207 ms. Unpinned processes instead give 0.801–0.837 ms and
0.796–0.832 ms respectively. These cross-process absolute timings are not paired
affinity measurements, but their large repeated separation and the loss of the
twelve-task advantage show that placement materially affects this worker result.
The specific mechanism (core contention, placement or migration) is not isolated.
Do not attribute the pinned gain to universally faster FP4 arithmetic, transfer
it to an unpinned tree, or use it to increase a full-model throughput forecast.

This supports retaining the tree's current unpinned host and six-task worker
policy, rather than adding host pinning to reproduce the standalone benchmark.
It does not establish optimal affinity on the primary, an idle worker or other
CPU budgets. A placement-aware service table should distinguish these conditions;
earlier pinned Q2/Q3 task results also remain restricted to their measured setup.

All 640 final batch checks after 40,960 timed calls match reference bitwise.
Startup weight packing is excluded and correctness is checked after each batch.
No production/default change, GPU execution, full-model quality/throughput or
90%-theoretical-utilization claim. Evidence, affinity markers, source hashes and
all control results: `results/20261020/adapter-host-affinity`.

### Q2/Q3 host-affinity control

The first Ryzen 3950X worker was tested with the same binary in pinned and
unpinned host modes, retaining three pinned pool workers plus the host and
leaving the miner unchanged. Startup affinity checks confirm one allowed CPU
for the pinned host and 32 for the unpinned host. The workload is synthetic
NT1 split service, three experts, sixteen rotating banks (437,256,192 packed
bytes), prefetch 1024 and prequantized input. Twelve tasks are compared directly
against three, with three/three controls. Both affinity modes use candidate/
control/control/candidate process order, alternating affinity order between
groups, twenty alternating sample pairs and 64 calls per sample.

| Host affinity | Twelve/three paired median ratios | Three/three controls | Candidate wins |
| --- | --- | --- | --- |
| Pinned | 0.9246 / 0.9318 | 1.0115 / 0.9983 | 20/20, 20/20 |
| Unpinned | 0.9536 / 0.9684 | 1.0037 / 1.0031 | 19/20, 17/20 |

Unlike FP4, this Q2/Q3 task-granularity gain persists with an unpinned host:
3.2–4.6% by paired medians, compared with 6.8–7.5% when pinned. Whole-run
medians for unpinned candidate processes change from 2.123–2.145 ms to
2.025–2.037 ms; pinned processes change from 2.316–2.342 ms to 2.166–2.176 ms.
Absolute affinity comparisons are across processes, not paired samples.
Twelve remains an opt-in candidate for this three-expert NT1 Q2/Q3 shape;
the result does not justify applying it to FP4, every route, or the primary.
Host migration keeps the simultaneous participant count at four.

All 320 final batch checks after 20,480 timed calls match reference bitwise,
with identical output hashes across affinity and task settings. CPU-time/wall-
time ratios are retained in the summary, but include spinning and overhead
and do not measure useful FLOPS utilization. No GPU execution, full-model
quality/throughput or 90%-peak qualification. Production defaults are unchanged.
Evidence: `results/20261020/q23-host-affinity`.

### FP4 prefetch recheck with an unpinned host

After identifying the host-affinity effect, the first worker's complete adapter
prefetch comparison was repeated without pinning the host participant. Startup
checks require more than one allowed CPU and record 32. Three pool workers remain
pinned; simultaneous participation stays at four and the miner remains active.
Native tasks match the tree policy: six at NT1, three at NT2. Real layers 3/4
expert zero, original layout, offset arithmetic, small-hidden preservation,
eight independent banks per arm and reusable workspace remain fixed.

One scratch kernel object selects prefetch distance only between completed
batches: baseline 2048 bytes, candidate 8192 bytes, and controls 2048 in both
arms. Candidate/control/control/candidate processes each use twenty alternating
pairs of 64 complete adapter calls per layer and token width.

| Tokens | 8192/2048 paired median ratios, both layers and runs | 2048/2048 controls |
| --- | --- | --- |
| 1 | 1.0138, 0.9876, 1.0277, 0.9962 | 0.9870, 1.0202, 1.0310, 0.9744 |
| 2 | 1.0144, 1.0076, 1.0176, 1.0149 | 0.9966, 0.9945, 1.0047, 0.9952 |

NT1 changes are mixed and within control variation. All four NT2 candidate
medians are slower by 0.8–1.8%, while controls stay within about 0.6%. Retain
2048; there is no complete-adapter benefit supporting the larger distance under
this placement. The earlier isolated stream gain remains limited to that workload.

All 640 final batch checks after 40,960 timed calls match reference bitwise.
Startup packing is excluded, checking follows batches, and background load is
uncontrolled. No GPU execution, production/default change, full-model throughput
or theoretical-utilization claim. Evidence: `results/20261020/adapter-prefetch-unpinned`.

### Second-worker Q2/Q3 affinity confirmation

The identical Q2/Q3 affinity probe was run on the second Ryzen 3950X worker,
with its miner unchanged, three pinned pool workers and one participating host.
The workload remains NT1 split service with three synthetic experts, sixteen
rotating banks, prefetch 1024 and prequantized input. Affinity masks are checked
at startup. Candidate/control/control/candidate processes, reversed affinity
order between groups, twenty alternating pairs and 64 calls per sample match
the first worker's experiment.

| Host affinity | Twelve/three paired median ratios | Three/three controls | Candidate wins |
| --- | --- | --- | --- |
| Pinned | 0.9798 / 0.9576 | 0.9962 / 0.9957 | 17/20, 20/20 |
| Unpinned | 0.9275 / 0.9378 | 1.0062 / 1.0060 | 20/20, 19/20 |

Unpinned Q2/Q3 gains repeat on this worker: 6.2–7.3%, versus the first worker's
3.2–4.6%. Whole-run unpinned medians change from 2.049–2.073 ms to
1.922–1.929 ms. Pinned gains are smaller here (2.0–4.2%); do not infer a
universal relationship between affinity and task-count gain. Both workers
support twelve as a tuning candidate for this particular unpinned, four-
participant, three-expert NT1 Q2/Q3 shape. Other token widths, routing shapes,
the primary and FP4 remain outside that conclusion; defaults stay unchanged.

All 320 final batch checks after 20,480 timed calls pass. Binary SHA-256 and
output hashes match the first-worker experiment; affinity/task variants also
match one another. Synthetic routes, excluded input quantization, repeated
weights and uncontrolled host load limit extrapolation. CPU-time ratios include
spin and overhead. No GPU, full-model throughput/quality or theoretical-peak
qualification. Evidence: `results/20261020/q23-host-affinity-second`.

### Explicit CPU bandwidth roofline audit

`cpu_roofline.py` converts the frozen stream-refresh evidence into effective
packed-weight bandwidth and NT1 useful-operation rates, keeping theoretical
bandwidth unknown unless active memory channels and configured MT/s are supplied.
It changes no benchmark or simulator defaults:

```sh
python3 experiment/glm_fp4_local/cpu_roofline.py \
  experiment/glm_fp4_local/results/20261020/stream-refresh/summary.json
python3 experiment/glm_fp4_local/test_cpu_roofline.py
```

Optional `--channels` and `--memory-mts` must be supplied together. The memory
formula is `channels * MT/s * 8 / 1000` GB/s for 64-bit data channels; MT/s
already counts transfers, so DDR adds no further factor of two. These inputs
remain explicit assumptions until independently verified on the tested machine.
No CPU FLOPS peak is inferred.

The original-layout sixteen-participant primary stream runs deliver 76.7405
and 77.9024 GB/s against read controls of 82.9597 and 83.1298 GB/s. Ratios of
process medians are 92.5% and 93.7%; these differ from the earlier median of
paired pass ratios, and both statistics remain recorded. Native FP4 stores
64 weights in 36 bytes including scales. Counting one multiply and one add per
weight at NT1 gives 3.556 useful operations per packed byte, or 272.9–277.0
useful GOP/s for these runs. This is model arithmetic accounting, not measured
retired floating-point instructions: unpacking, integer operations, scaling and
quantization have additional costs.

For the measured payload rates to equal 90% of configured theoretical bandwidth,
that bandwidth would need to be at most 85.27–86.56 GB/s. The configured RAM
speed/channel denominator is still missing. Furthermore, effective weight bytes
are not physical DRAM traffic counters; other accesses and caches prevent this
ratio alone from certifying hardware utilization. Thus the 90% theoretical
target remains unverified despite exceeding 90% of the measured read control.
Do not turn the useful-operation rate into a percentage of an assumed FLOPS peak.

Three CPU-only unit tests cover transfer units, scale-byte arithmetic, unknown
denominators and invalid inputs. All fourteen frozen stream runs are processed
without changing raw evidence. The report records the input SHA-256 and every
assumption in `results/20261020/cpu-roofline-audit/report.json`; source hashes are
alongside it. These are partial-stream diagnostics, not full-model decode rates.

### Three-expert native FP4 adapter prototype

Inspection of the rejected offset/unpack candidates shifted the next prototype
to batching complete expert work. A scratch adapter submits three real experts
sharing one NT1 input to `run_split_multi_native` in one call. It quantizes the
input once and uses separate hidden-rescale state indexed by each job's expert
ID. This separation is necessary: the existing single-expert scaler tracks
tokens but ignores expert ID, so sharing it unchanged would overwrite shifts.
Outputs remain separate, with per-expert scale restoration and finite checks.
The prototype supports only three experts, original complete blobs and NT1;
it is not connected to the decoder or tree.

A first primary Threadripper 1950X screen uses three pinned workers plus an
unpinned participating host, six tasks per participant, offset arithmetic and
prefetch 2048. It compares three sequential workspace-reusing calls against one
batched call using experts 0/1/2 from real layers 3/4. Ordinary, 0.001-amplitude
and zero inputs each run eight alternating pairs of eight calls. All 288 final
expert comparisons across 96 timed windows match the sequential output bitwise,
after 768 timed three-expert invocations. Input quantization, finite checks and
hidden-scale restoration are included; startup packing is excluded.

| Layer | Input amplitude 1 batch/sequential | Amplitude 0.001 | Zero input |
| --- | --- | --- | --- |
| 3 | 0.9563 | 0.9487 | 0.9581 |
| 4 | 0.9963 | 0.9568 | 0.9578 |

These are medians of paired ratios. The single process has no separate
identical-work controls, and one normal-input case is nearly tied. This is an
exploratory candidate, not a qualified speedup or full-primary result. Next
validation needs controls, NUMA-owned batching at sixteen primary participants,
and worker measurements. Input sharing and grouped scheduling change together;
the screen does not isolate their separate contributions. No production API,
default, GPU or full-model behavior changed. Evidence and scratch source are in
`results/20261020/fp4-batch-prototype`.

### Sixteen-participant NUMA FP4 batching control

The scratch three-expert adapter now accepts prepared NUMA-owned rows, validating
that every expert was prepared for the current pool topology. Original blobs
are retained in both arms; this prototype does not support released compact
blobs. Each job receives its own two-node gate/up/down shards and dense expert
ID for independent hidden rescaling. Common NT1 input is quantized once in the
batched path; three sequential calls each quantize it independently.

The Threadripper 1950X uses fifteen pinned workers plus a pinned host, three
tasks per participant, original packed layout, offset arithmetic and prefetch
2048. Real experts 0/1/2 from layers 3/4 use normal, 0.001-amplitude and zero
inputs. Candidate/control/control/candidate processes each run sixteen
alternating pairs of 32 calls per cell. Controls execute the sequential path in
both arms. Quantization, finite scans, scale restoration and pool service are
timed; loading, packing and NUMA preparation are excluded.

| Layer / input amplitude | Batch/sequential paired medians, first / repeat | Sequential/sequential controls |
| --- | --- | --- |
| 3 / 1 | 0.8004 / 0.8076 | 0.9989 / 0.9594 |
| 4 / 1 | 0.8023 / 0.8139 | 1.0148 / 0.9987 |
| 3 / 0.001 | 0.8138 / 0.8102 | 0.9974 / 1.0038 |
| 4 / 0.001 | 0.8163 / 0.8445 | 0.9951 / 1.0407 |
| 3 / 0 | 0.8039 / 0.8104 | 0.9968 / 0.9995 |
| 4 / 0 | 0.8716 / 0.8134 | 0.9959 / 0.9995 |

Normal-input gains repeat at 18.6–20.0% by paired medians. First-process
normal-input whole-run medians change from 0.731/0.733 ms to 0.585/0.587 ms;
repeat medians change from 0.836/0.744 ms to 0.660/0.610 ms. Preserve the noisy
second control, including 0.9594 and 1.0407 ratios and larger absolute times:
clocks/background load are uncontrolled. These are partial three-expert service
measurements, not a full-model forecast. Input reuse and grouped pool scheduling
change together, so their contributions are not separately identified.

All 2,304 final expert-output comparisons across 768 timed windows match the
sequential reference bitwise after 24,576 timed three-expert calls. Checks occur
after each batch, not every call. No production API/default, GPU execution,
full-model throughput/quality or theoretical-peak qualification changed.
Worker confirmation and a bounded public API with invalid-input tests remain
necessary before integrating this batching candidate. Raw samples, controls,
source snapshots and hashes: `results/20261020/fp4-batch-numa`.

### Worker confirmation of three-expert FP4 batching

The same scratch batching implementation was tested on the first Ryzen 3950X
worker with three pinned pool workers, an unpinned host, six tasks per
participant and complete original blobs. Startup confirms 32 allowed host CPUs;
this still uses four simultaneous participants. The miner is unchanged. Real
experts 0/1/2, layers 3/4, common NT1 inputs, arithmetic, prefetch 2048 and
workspace reuse match the primary comparison. Candidate/control/control/
candidate processes each use sixteen alternating pairs of 32 calls per cell.
Controls execute the sequential path in both arms.

| Layer / input amplitude | Batch/sequential paired medians, first / repeat | Sequential/sequential controls |
| --- | --- | --- |
| 3 / 1 | 0.9643 / 0.9681 | 1.0234 / 1.0131 |
| 4 / 1 | 0.9509 / 0.9613 | 1.0044 / 1.0116 |
| 3 / 0.001 | 0.9658 / 0.9687 | 0.9928 / 1.0019 |
| 4 / 0.001 | 0.9493 / 0.9605 | 1.0159 / 1.0192 |
| 3 / 0 | 0.9758 / 0.9661 | 0.9949 / 1.0329 |
| 4 / 0 | 0.9626 / 0.9644 | 0.9984 / 1.0407 |

Normal-input improvements repeat at 3.2–4.9%, smaller than the primary's
18.6–20.0%. Worker whole-run normal-input medians change from 3.146–3.262 ms
to 3.055–3.088 ms across candidate processes. Ratios of these medians differ
from medians of paired ratios. Preserve all control variation: some zero-input
controls shift by 3.3–4.1%, comparable in size to the candidate gains. No broad
or workload-independent speedup follows from this partial-weight test.

All 2,304 final expert comparisons across 768 timed windows pass bitwise after
24,576 timed three-expert calls. Common-input quantization, finite scans, hidden
rescaling and pool submission are timed; startup packing is excluded. Input
reuse and grouped scheduling remain combined changes. No production API/default,
GPU execution, full-model throughput/quality or theoretical-utilization claim.
The next integration step needs bounded input/alias/topology validation and
explicit per-expert rescale tests. Evidence: `results/20261020/fp4-batch-worker`.

### Explicit three-expert adapter API and validation

`NativeFp4CpuExpert::run_three_into` exposes the measured batching strategy as
an explicit experimental API. It accepts exactly three experts, one shared
4096-element NT1 input, three caller-owned outputs and reusable activation
scratch. It supports complete original blobs and optionally prepared NUMA rows;
K32 and released compact blobs are rejected. No decoder or tree selects it
automatically. The caller must serialize pool and workspace use.

All pointer, extent, finite-input, input/output-alias, weight-layout, hidden-scale
policy and NUMA preparation/topology checks precede output or scratch writes.
Repeated expert pointers are permitted; each job has an independent dense ID.
`MultiExpertHiddenScale<3>` keeps expert/token shifts separate and validates
expert IDs. This fixes the potential state collision from reusing the original
token-only scaler across several jobs. Existing single-expert code is unchanged:
removing only the new API reproduces the frozen adapter header byte-for-byte.

Build the test with the same native-library parameters as `cpu-native-adapter`:

```sh
make -f experiment/glm_fp4_local/Makefile cpu-native-batch \
  BUILD=BUILD_DIR NATIVE_BUILD=NATIVE_CPU_BUILD GGML_INCLUDE=GGML_INCLUDE_DIR
BUILD_DIR/glm-fp4-native-batch-test PARTIAL_MODEL_DIR complete
BUILD_DIR/glm-fp4-native-batch-test PARTIAL_MODEL_DIR numa
```

The NUMA test requires the sixteen-participant two-node primary placement.
Both HIP-build and CUDA-build CPU libraries compiled/linked the test and passed
complete-blob and NUMA execution: 120 bit-exact expert-output comparisons and
60 invalid-argument checks total. Cases include small/zero/normal input,
workspace reuse, disabled hidden scaling, repeated expert pointers, nulls,
wrong input extent, infinity, aliasing, mixed scaling policies, K32, unprepared
NUMA rows and released blobs. Rejected calls preserve output and scratch bytes.
An independent scale check uses three different expert/token shifts, including
the smallest positive subnormal float, and restores them in reverse order.

The existing single-expert adapter passes 692 checks across normal and
preserve-small modes. Existing hidden-scale tests pass 4,809 cases and 512
fallback cases. GPU execution and SYCL validation were not performed; the SYCL
compiler remains unavailable. No runtime default, model throughput or quality
claim changed. Prior speed measurements concern the prototype; the validated
API still needs matched timing before deployment. Evidence:
`results/20261020/batch-api`.

### Matched timing of the validated batching API

Both timing probes now include the actual `native_cpu_expert.hpp` and call
`run_three_into`, including its argument checks and per-expert scale handling.
They no longer compile a private adapter implementation. The primary uses
sixteen participants, pinned host, prepared NUMA rows and three tasks per
participant; the first worker uses four participants, unpinned host, complete
blobs and six tasks. Offset arithmetic and prefetch 2048 are enabled. Miners
remain unchanged. Three real experts from each of layers 3/4 use normal, small
and zero shared NT1 inputs. Candidate/control/control/candidate processes each
run sixteen alternating pairs of 32 calls per cell; controls remain sequential
in both arms.

| Host / normal-input layer | API batch/sequential paired medians, first / repeat | Sequential/sequential controls |
| --- | --- | --- |
| Primary / 3 | 0.7949 / 0.8023 | 1.0338 / 0.9931 |
| Primary / 4 | 0.8263 / 0.8072 | 1.0020 / 0.9972 |
| Worker / 3 | 0.9772 / 0.9729 | 0.9817 / 0.9758 |
| Worker / 4 | 0.9911 / 0.9721 | 0.9966 / 0.9993 |

The validated API retains a 17.4–20.5% normal-input gain on the primary.
Primary normal-input candidate windows win 16/16, 13/16, 16/16 and 16/16
pairs. In the repeat process, sequential layer medians are 0.738/0.738 ms and
batch medians 0.595/0.596 ms. Preserve the first-process layer-4 outlier and
all controls: its ratio of whole-run medians differs materially from the median
paired ratio. Primary small/zero-input paired ratios span 0.7993–0.8215.

Worker normal-input gains are only 0.9–2.8%; layer-3 controls show a similar
1.8–2.4% favorable shift. Small-input ratios improve more (0.9391–0.9716),
while zero-input ratios are 0.9787–0.9958. These results support prioritizing
primary integration, not asserting the earlier prototype's worker gain carries
unchanged into the API. The runs do not isolate validation overhead from load,
placement and code-generation changes, so no such causal claim is made.

Across both hosts all 4,608 final expert comparisons pass bitwise, covering
1,536 timed windows and 49,152 three-expert calls. These checks follow batches;
they are not independent full-model quality checks. No decoder/tree routing or
default was changed. Full-model throughput, GPU execution and the theoretical-
utilization target remain unverified. Current headers, source hashes, controls
and samples: `results/20261020/batch-api-timing` and
`results/20261020/batch-api-worker`.

### Compact NUMA support for three-expert batches

`run_three_into` now accepts three experts whose original blobs have all been
released after NUMA preparation. It uses each expert's existing compact scale
buffer and the same nonzero scale-tail offset as single-expert compact calls;
weight rows continue to come from prepared NUMA shards. The batch must use one
storage state throughout: mixed compact/original experts are rejected before
output or activation-scratch writes. Compact calls require NUMA mode and matching
prepared topology. K32 remains unsupported. The existing single-expert methods
remain byte-for-byte identical to the frozen adapter source.

Tests with HIP-build and CUDA-build CPU libraries pass 168 expert cases and
64 rejection cases in total. The 48 compact expert cases compare bytes against
both pre-release batches and post-release single-expert calls, across normal,
small, zero and repeated normal inputs. Every released expert reports zero
retained original-blob capacity. Mixed storage and compact-without-NUMA calls
preserve caller output/scratch on rejection. GPU execution and SYCL validation
were not performed; the SYCL compiler remains unavailable. Validation snapshots
and logs: `results/20261020/batch-compact-api`.

Matched primary timing now releases all three original blobs before either arm
runs. Sixteen participants, pinned host, three tasks per participant, offset
arithmetic, prefetch 2048, real layers 3/4 and the candidate/control/control/
candidate protocol remain unchanged. Each process uses sixteen alternating pairs
of 32 calls per cell. Both arms use compact NUMA storage, so the comparison
measures batching, not the effect of freeing original copies.

| Layer / input amplitude | Batch/sequential paired medians, first / repeat | Sequential/sequential controls |
| --- | --- | --- |
| 3 / 1 | 0.8036 / 0.8421 | 1.0022 / 0.9933 |
| 4 / 1 | 0.8725 / 0.8409 | 0.9936 / 0.9895 |
| 3 / 0.001 | 0.7972 / 0.8260 | 1.0241 / 0.9968 |
| 4 / 0.001 | 0.8013 / 0.9058 | 1.0248 / 1.0069 |
| 3 / 0 | 0.8166 / 0.8188 | 0.9915 / 0.9976 |
| 4 / 0 | 0.8055 / 0.8349 | 1.0123 / 1.0221 |

Normal-input gains remain 12.8–19.6% by paired medians. Load-related variation
and differences between paired ratios and ratios of whole-run medians remain
visible in the raw data; do not infer a compact-storage slowdown by comparing
separate experiments. All 2,304 final expert comparisons across 768 windows pass
bitwise after 24,576 timed three-expert calls. Compact storage removes the
prototype's need to retain duplicate original weight blobs, but this remains an
explicit adapter API, not an enabled full-model route. No decode tok/s, GPU
execution or theoretical-utilization claim. Timing evidence:
`results/20261020/batch-compact-timing`.

### Routed native CPU partial-chain runner

`tools/glm_fp4_native_chain.cpp` now consumes the batching API in a local,
CPU-only two-layer MoE chain. `batch-three` selects batching when exactly three
experts share the NT1 input; one/two-expert routes retain sequential calls and
idle routes perform no expert work. Weighted outputs are reduced in route order,
then a fixed BF16-rounded residual update feeds the next partial layer. The
default is sequential. `numa-compact` prepares primary NUMA rows and releases
original blobs before execution. Native Q8 activations and preserved small hidden
values are explicit properties of this runner, not a W4A16 quality claim.

```sh
make -f experiment/glm_fp4_local/Makefile cpu-native-chain \
  BUILD=BUILD_DIR NATIVE_BUILD=NATIVE_CPU_BUILD GGML_INCLUDE=GGML_INCLUDE_DIR
BUILD_DIR/glm-fp4-native-chain PARTIAL_MODEL_DIR 16 numa-compact
BUILD_DIR/glm-fp4-native-chain PARTIAL_MODEL_DIR 16 numa-compact batch-three
```

The fixture uses real experts 0/1/2 from layers 3/4 with eight fixed scenarios:
one expert, two reordered experts, three weighted experts, three reordered
experts, idle, zero input, small input and zero route gains. Three repetitions
check that retained workspaces do not leak stale outputs into later routes.
Each layer compares both its weighted reduction and updated activation bytes
against sequential native expert calls. This validates routing mechanics and
reduction order, not a learned router or the model's full attention/mHC path.

Eight executions cover sequential/batched, complete four-participant/compact
sixteen-participant, and HIP-build/CUDA-build CPU libraries. All 384 layer checks
pass. Every 393,216-byte output dump is identical across all eight executions;
each batched run reports 30 batched, 12 sequential and six idle layer calls.
Compact runs retain zero original-blob bytes, versus 84,934,752 bytes in this
six-expert complete fixture. Five invalid option/thread/placement cases reject.

Per-layer service timestamps include expert execution, reduction and residual
construction, excluding the sequential reference. These short validation runs
are not a controlled performance qualification or full-model tok/s measurement.
No GPU execution, IB protocol, existing decoder default or model quality setting
changed. SYCL validation remains unavailable. Build commands, validation logs,
output hashes and source snapshot: `results/20261020/native-chain`. Raw output
dumps remain in ignored local scratch.

### Task granularity for compact three-expert batches

The primary's validated compact batching API was screened with one and two
native tasks per participant against its existing three-task setting. Both
arms now call `run_three_into`; this is not another batch-versus-sequential
comparison. Sixteen participants, pinned host, prepared NUMA rows, released
original blobs, real experts from layers 3/4, offset arithmetic and prefetch
2048 remain fixed. The existing pool setter changes task count only between
completed batches. Process order is 1, 2, 3, 3, 2, 1; three/three processes
are controls. Every layer/input cell has sixteen alternating pairs of 32 calls.

| Candidate tasks / baseline three | Normal-input paired ratios, layers 3/4 first then repeat |
| --- | --- |
| One | 1.5186, 1.4969, 1.4298, 1.5115 |
| Two | 1.1509, 1.1471, 1.0948, 1.1620 |
| Three/three controls | 0.9917, 1.0021, 1.0769, 1.0108 |

Retain three tasks for this primary batch shape. One is 43–52% slower for normal
inputs; two is 9–16% slower. Small/zero-input candidates are also slower, with
all paired medians above one. Preserve the 1.0769 control and absolute-time
outliers; those show background variation but do not overturn the repeated
direction. From the pool's NUMA partition formula, three experts with sixteen
participants produce 18, 36 and 48 phase tasks for settings one, two and three.
Task sizes and load balancing change along with dispatch count; timing alone
does not isolate the cause. This result does not apply to other expert counts,
token widths or worker CPU budgets.

All 3,456 final expert comparisons across 1,152 timed windows pass bitwise after
36,864 timed three-expert calls. Reference calls are sequential native execution;
checking occurs after batches. No production default, routed runner, GPU or
full-model behavior changed. Theoretical utilization remains unverified.
Evidence: `results/20261020/batch-task-granularity`.

### Bounded command validation for grouped native CPU work

The existing tree command encodes one resident expert per worker and cannot
represent a local three-expert batch. A separate nine-word native-Q8 command
format is now defined in `glm_fp4_protocol.h`, with validation in the shared
C11 `strata_host` implementation. It has its own magic/version, one common
4096-value input token, two layer slots, zero to three resident-expert IDs and
an exact sequence number. Stop and idle commands have no activation payload;
non-idle work returns one output vector per requested expert in request order.
Repeated IDs are valid; unused ID words must be zero. Native Q8 and preserved
small-hidden scaling are fixed semantics of this experimental version.

The validator checks all fields before a future transport consumer reads or
allocates payloads. The maximum input is 16,384 bytes and maximum output is
49,152 bytes. The existing seven-word tree validator and protocol are unchanged;
tests confirm the two formats do not cross-accept. This addition does not yet
wire batch requests into IB service or replace the existing tree protocol.

Seven old/new protocol test methods pass, including all 80 legal grouped-route
headers, malformed fields, unused IDs, stop/idle, null pointers, stale/zero
sequence values and the UINT32_MAX boundary. C11 compilation passes with
`-Wall -Wextra -Werror`; HIP and CUDA host archives rebuild successfully.
Freshly linked names/environment/expert-pack host tests pass in both builds.

The broader HIP `glm` label is not clean: 27 passed, one skipped and 17 failed
or did not run. Nine entries lacked executables, seven GPU binaries could not
load their shared libraries, and the sandboxed mixed-transport test disconnected.
That transport test passes when rerun with socket access. Missing artifacts and
GPU-library dependencies remain unresolved; no full-suite success is claimed.
GPU binaries failed before execution, and no new batch transport or GPU workload
was run. SYCL validation remains unavailable. Shared source/tests and structured
validation are in `results/20261020/batch-protocol`; frozen raw CTest logs remain
in ignored scratch because they contain private absolute paths, with hashes in
the shared record. Next is the bounded transport consumer and physical IB checks.

### Grouped native CPU requests over transport

The native partial-chain runner now has explicit `serve=ROOT` and `remote=ROOT`
modes. The server loads the existing six-expert partial fixture once; each
request identifies a layer slot and up to three local expert IDs. It validates
the shared C11 command before reading activations, selects the batching API only
for three-expert requests when `batch-three` is enabled, and returns separate
expert outputs in request order. The client retains weighted reduction, residual
updates and sequential-reference checks. Idle/stop requests carry no activation
payload, and every reply acknowledges the exact request sequence.

The link has one synchronous owner and registered transmit/receive staging
buffers. It exchanges a fixed magic/version/geometry handshake and enforces
little-endian IEEE float32. Registrations are destroyed before their connection;
payload reads require exact lengths. Each peer needs the same partial fixture;
this protocol transfers activations/results, not weights. The server selects
`batch-three`; clients reject that option rather than implying a local setting
controls remote execution. `ib-port=1|2` selects an IB port, and explicit `udp`
is available for loopback validation. Existing local execution remains default.

Example with an authorized private IB endpoint supplied by local configuration:

```sh
# Primary service, compact NUMA weights, sixteen participants:
BUILD_DIR/glm-fp4-native-chain PARTIAL_MODEL_DIR 16 \
  numa-compact batch-three serve=ROOT
# Client on a worker, four participants:
BUILD_DIR/glm-fp4-native-chain PARTIAL_MODEL_DIR 4 remote=ROOT
```

Both HIP-build and CUDA-build CPU executables pass local and loopback UDP tests.
Physical direct IB tests use a portable binary, a sixteen-participant compact
primary server and four-participant first-worker client, with miners unchanged.
Both sequential and batched servers pass. The batched server records 48 requests:
30 grouped, 12 smaller-route sequential and six idle calls, with zero original
blob bytes retained. Deployed binary and worker fixture hashes are checked before
physical execution. Six invalid transport option combinations reject before
connecting.

All 672 layer checks pass: 384 local, 192 loopback and 96 physical IB. Every one
of the fourteen 393,216-byte output dumps matches the frozen local reference
byte-for-byte. No GPU was used; SYCL validation remains unavailable. This is
two-node grouped execution, not yet relay/three-node grouped routing or a full
decoder. Short validation service times do not establish a transport speedup,
decode tok/s or theoretical utilization. Existing tree protocol/defaults remain
unchanged. Evidence: `results/20261020/native-batch-transport`; private connection
scripts and raw dumps stay in ignored scratch.

### Native three-expert batching over IB: matched service timing

The partial native chain was extended to sixteen repeats in a scratch timing
probe and run sequential/batched/batched/sequential over direct IB. The primary
Threadripper 1950X server uses sixteen participants, pinned host and compact NUMA
storage with original blobs released; the Ryzen 3950X client uses four
participants with an unpinned host. Miners remain running. Both arms use the same
portable binary, real partial weights, native Q8 activation semantics and
registered staging. Only the server's explicit `batch-three` option changes.

Each process checks 256 layers. All 1,024 layer checks pass bitwise; all sixteen
output repeats match each other and the earlier frozen three-repeat reference.
The first repeat is excluded from timing, leaving 960 service windows. The
following are ratios of process medians (fifteen samples per cell), not medians
of within-process paired ratios. Each candidate is compared to its adjacent
sequential run in the ABBA order.

| Normal three-expert route / layer | Batch/sequential first / repeat | Sequential repeat/first |
| --- | --- | --- |
| Ordered / 3 | 0.830 / 0.942 | 0.990 |
| Ordered / 4 | 0.850 / 0.917 | 0.925 |
| Permuted / 3 | 0.893 / 0.958 | 0.993 |
| Permuted / 4 | 0.852 / 0.828 | 1.012 |

Normal-route service medians range 1.110–1.223 ms sequential and 0.931–1.084 ms
batched. This is a 4.2–17.2% reduction across these comparisons. Service includes
remote request/result, weighted reduction and residual update, but excludes the
client's local sequential oracle. Unchanged one/two-expert routes show arm ratios
0.974–1.189, indicating background variation; the result is promising under load,
not an isolated compute or wire-latency measurement. Zero/small-input and
zero-gain routes are retained in the evidence, along with every warm sample.

The server confirms 160 batched, 64 sequential and 32 idle requests in each
batched run, versus 224 sequential and 32 idle in each baseline. The option
remains confined to the partial runner. No production/default dispatch changed,
and no GPU, full-model routing/quality/tok/s or theoretical-peak qualification
was performed. Frozen logs, source snapshots, fixture/output hashes and analysis:
`results/20261020/native-batch-ib-timing/`.

### Two-expert native FP4 batching

The production decoder already passes all available host expert jobs to
`run_split_multi_native` or `run_layer_native`. The partial runner's earlier
serial-to-batched gain therefore cannot be added to production decode estimates.
The experimental adapter, however, still exposed batching only for three
experts. It now provides `run_batch_into<Count>` for two or three common-input
NT1 experts, with the existing `run_three_into` wrapper preserved. All code
outside the batch method is identical to the preceding frozen header, including
the default single-expert implementation. No production dispatch/default changed.

Two-expert calls share input quantization and one pool submission, while keeping
independent hidden-scale transforms and outputs. Validation still precedes
workspace mutation and rejects mixed storage/rescale policy, K32 layout, invalid
inputs, missing NUMA preparation and input/output aliasing. Both HIP- and
CUDA-configured CPU builds pass 280 final expert comparisons and 128 invalid-call
checks across two/three-expert, complete/compact modes. Compact results match
both pre-release batches and post-release sequential references. SYCL was not
built because its compiler is unavailable; no GPU ran.

A matched primary test uses sixteen participants, pinned host, three tasks per
participant and compact NUMA storage with original blobs released. Two real
experts from layers 3/4 run in candidate/control/control/candidate processes,
with sixteen alternating pairs of 32 calls per cell. Normal-input paired ratios:

| Layer | Batch/sequential first / repeat | Sequential/sequential controls |
| --- | --- | --- |
| 3 | 0.8517 / 0.8493 | 0.9990 / 0.9917 |
| 4 | 0.8459 / 0.9481 | 0.9931 / 0.9900 |

Normal-input improvement is 5.2–15.4%, with visible background-load variation.
The first candidate process has sequential medians about 0.462 ms and batch
medians 0.392–0.393 ms. The repeated layer-4 candidate is slower in both arms
(0.777/0.708 ms), so it remains in the report. Small/zero-input paired ratios
range 0.8395–0.9005. One small-input control shifts by 5%; raw pairs and all
controls are preserved. Input quantization, finite checks, hidden restoration
and pool service are included; packing/NUMA preparation are excluded.

All 1,536 final expert checks after 24,576 timed two-expert calls pass bitwise.
This supports an explicit two-expert adapter candidate on the primary, not a
full-model speedup, worker recommendation or 90%-peak claim. Routed IB integration
of the two-expert option has not been tested. Evidence and source snapshots:
`results/20261020/batch-two-api/` and `results/20261020/batch-two-timing/`.

### Q2/Q3 task count depends on the number of routed experts

The existing `STRATA_NATIVE_TASKS_PER_THREAD=12` opt-in was tested against three
for NT1 routes with one, two, four and eight experts on both Ryzen 3950X workers.
Each worker uses four CPU participants with an unpinned host, Q2_K gate/up,
Q3_K down, H4096/F2048, prefetch 1024 and sixteen rotating banks. Packed rotating
footprints are 145,752,064 / 291,504,128 / 583,008,256 / 1,166,016,512 bytes,
respectively. Both arms share weights and arithmetic; input quantization is
excluded. Miners remain running. Each geometry has candidate/control/control/
candidate processes, with twenty alternating pairs of 64 calls per sample.
Geometry order reverses between processes.

| Worker / experts | Twelve/three paired ratios, first / repeat | Three/three controls |
| --- | --- | --- |
| One / 1 | 1.0055 / 0.9716 | 1.0203 / 1.0132 |
| One / 2 | 0.9500 / 0.9841 | 0.9979 / 1.0131 |
| One / 4 | 0.9471 / 0.9497 | 1.0021 / 1.0063 |
| One / 8 | 0.9377 / 0.9414 | 1.0063 / 0.9983 |
| Two / 1 | 1.0655 / 1.1008 | 0.9983 / 0.9999 |
| Two / 2 | 0.9845 / 0.9962 | 0.9931 / 0.9805 |
| Two / 4 | 0.9409 / 0.9311 | 0.9882 / 0.9920 |
| Two / 8 | 0.9315 / 0.9360 | 1.0029 / 1.0011 |

Four/eight-expert routes consistently improve 5.0–6.9%; all eighty eight-expert
candidate pairs favor twelve. One-expert results on worker one are mixed, while
worker two is 6.6–10.1% slower with twelve. Two-expert improvements are small or
comparable to control variation on worker two. This argues against setting twelve
globally for variable routing. Keep three for one-expert calls and retain the
baseline for two until stronger evidence exists; a future opt-in routing policy
can select twelve for the separately measured three/four/eight-expert NT1
shapes. Other counts, token counts, worker budgets and primary NUMA behavior
need separate validation. No running service or production/default policy was
changed by this screen.

Gate/up and down service dominate these samples. For example, worker two's first
eight-expert baseline has medians 5.471 ms total, 3.271 ms gate/up, 2.186 ms down
and 0.0157 ms hidden quantization. These are independently computed medians and
must not be summed as an exact decomposition. Raw phase counters and wall/CPU
timings are preserved; they do not establish physical memory bandwidth or
retired-FLOPS utilization.

All 1,280 final batch checks after 81,920 timed calls match their sequential
reference bitwise. Per-geometry output hashes agree across both workers and all
arms. This is synthetic route-count evidence, not a full-model routing trace,
decode quality/tok/s measurement or 90%-peak qualification. Source, binary/library
hashes, raw timings and analysis: `results/20261020/q23-task-route-count/`.

### Opt-in Q2/Q3 route-aware split tasks

`STRATA_Q23_ROUTE_TASKS=1` now enables a bounded policy in the CPU split scheduler.
The exact-one environment helper is used: unset, `0` and `2` do not enable it.
`ExpertPool::set_q23_route_tasks` exposes the same selection for controlled tests
between completed calls. Defaults remain disabled and running services were not
changed. The measured setup also uses `STRATA_Q23_AVX2=1` and an unpinned host
thread on a Ryzen 3950X; this is not a recommendation for other CPU placements.

The policy selects twelve tasks per participant only when all of these match:
three worker threads plus host participation, complete weights, Q2_K gate/up and
Q3_K down, H4096/F2048, three/four/eight expert jobs, exactly one token per job,
SwiGLU limit 10, and no canonical mode, gate skipping, expanded Q3 down storage,
fused hidden quantization, observer or hidden transform. NUMA-local execution
is excluded. The base per-participant count must be three and no fixed task
budget may be configured. Nondefault per-participant counts and fixed task
budgets take precedence. Every other call keeps its existing partitioning.
The decision is local to a call; it does not change the pool's base setting.
Direct layer-dataflow scheduling is unchanged.

The new test checks 62 selection decisions and 24 bitwise changing-route cases
per build, alternating counts 3/1/8/2/4/5/3/0 through the same pool. The existing
pool partition test passes 168 bitwise checks and the native pool test passes
384 batches per build. All three tests pass on both HIP- and CUDA-configured CPU
builds. SYCL could not be built because its compiler is unavailable. These are
CPU tests; no GPU execution or full decoder qualification ran.

The actual implemented policy was then compared with the disabled policy on both
four-participant workers. Sixteen rotating synthetic banks, NT1 split mode,
unchanged miners, candidate/control/control/candidate processes, twenty
alternating pairs and 64 calls per sample match the preceding route screen.

| Worker / expert count | Policy/baseline paired medians, first / repeat | Disabled/disabled controls |
| --- | --- | --- |
| One / 1 | 0.9613 / 1.0172 | 1.0025 / 0.9748 |
| One / 4 | 0.9297 / 0.9450 | 0.9978 / 1.0084 |
| Two / 1 | 1.0013 / 0.9973 | 0.9998 / 0.9967 |
| Two / 4 | 0.9352 / 0.9246 | 1.0077 / 1.0079 |

Four-expert calls improve 5.5–7.5%; 78 of 80 candidate pairs favor the policy.
One-expert calls use three tasks in both arms: worker two no longer shows the
preceding forced-twelve regression, while worker one's variation remains visible
and is not attributed to the no-op policy. All 640 final batch checks after
40,960 timed calls match reference bitwise, with output hashes agreeing across
workers and arms. Input quantization is excluded. There is no new full-model
tok/s, quality or theoretical-peak claim. Shared source snapshots, raw paired
measurements, hashes and validation summary: `results/20261020/q23-route-policy/`.

### FP4 batching with rotating compact weight banks

Three-expert NT1 batching was compared with sequential adapter calls using one
and sixteen banks on the primary Threadripper 1950X. Each bank holds independent
NUMA allocations of three real experts from layer 3 or 4. Original packed blobs
are released after preparation. The packed projection payload is 42,467,328
bytes per three-expert call; sixteen banks retain 679,477,248 projection bytes
per tested layer, excluding small metadata/allocation overhead. Contents repeat
across banks; this is a controlled streaming experiment, not a routing trace.

Both arms rotate through the same allocations with independent counters,
including untimed warmups. Each cell has sixteen alternating pairs of 32 calls.
Candidate/control/control/candidate processes interleave one/sixteen-bank tests,
reversing their order between processes. Sixteen CPU participants, pinned host,
three tasks per participant, offset arithmetic, prefetch 2048 and small-hidden
preservation match the preceding compact batch tests. Miners remain unchanged.

| Banks / layer, normal input | Batch/sequential paired medians, first / repeat | Sequential/sequential controls |
| --- | --- | --- |
| 1 / 3 | 0.8902 / 0.8075 | 1.0174 / 1.0015 |
| 1 / 4 | 0.8088 / 0.7999 | 0.9828 / 1.0116 |
| 16 / 3 | 0.8324 / 0.8860 | 1.0030 / 0.9953 |
| 16 / 4 | 0.8231 / 0.9178 | 0.9976 / 0.9964 |

The sixteen-bank normal-input paired medians improve 8.2–17.7%, so this screen
does not show the batching advantage disappearing with the larger working set.
However, it is load-sensitive: first-process sequential/batch medians are
0.761/0.625 ms and 0.759/0.619 ms, while repeated medians are 1.060/1.058 ms
and 1.176/1.067 ms. A median paired ratio is not the ratio of whole-run medians;
the repeated layer-3 result especially must not be described as an 11% gain in
whole-run median service. Raw pairs and control outliers are retained.

First-process batch requested-weight-payload rates are 68.0–68.6 GB/s; repeated
normal-input rates fall to about 40 GB/s. These are packed bytes requested per
call divided by service time, not memory-controller traffic, configured DRAM
peak utilization or retired FLOPS. No 90%-peak claim follows. Input quantization,
finite scans, hidden restoration and native pool service are included; packing
and NUMA preparation are excluded.

All 4,608 final expert comparisons after 49,152 timed three-expert calls pass
bitwise across 1,536 windows. Final checked banks cycle through all sixteen bank
indices in each arm. Production already batches host experts; these results
qualify the adapter optimization and must not be added to production full-model
throughput estimates. No GPU or full-model quality/tok/s measurement ran.
Source snapshots, fixture/library/binary hashes, raw timings and analysis:
`results/20261020/fp4-batch-streaming/`.

### Bracketed FP4 read/projection controls at the adapter footprint

A follow-up uses the streaming benchmark with 72 copies on each of sixteen
pinned threads, totaling 679,477,248 packed bytes, matching the preceding
sixteen-bank adapter footprint. Both K4096/N256 and K2048/N512 shapes are tested
so gate/up and down projection widths are represented. Each row process checks
GGML dot outputs and checksum read controls outside timing. K4096/K2048 runs
bracket each real compact adapter process in forward/reverse order; four adapter
processes retain candidate/control/control/candidate order. Settings remain
offset arithmetic, prefetch 2048 and no AVX512, with miners unchanged.

Read payload rates range 77.4–84.1 GB/s. K4096 FP4 rates range 42.6–78.5 GB/s,
and K2048 rates range 42.5–76.4 GB/s. Paired FP4/read ratios range approximately
0.54–0.94. This spread remains in the evidence; the fastest result is not a
stable hardware or theoretical-peak claim.

Weighting row time by two gate/up projections and one down projection gives
synthetic three-expert time proxies of 0.886/0.875/0.602/0.551 ms across the four
brackets. These are not measured adapter lower bounds. For example, the first
adapter candidate has normal-input medians 0.622/0.619 ms, below its independent
0.886 ms projection proxy. The row benchmark uses different weights, persistent
pinned threads and allocations; the adapter uses its NUMA pool and real partial
weights. Cross-process timing variation prevents attributing their difference
solely to adapter overhead or dequantization. Do not subtract these proxies to
claim a measured overhead budget or fit full-model tok/s from them.

All sixteen row processes pass 288 timed and 32 warmup read/dot checks. Four
adapter processes pass 2,304 final expert comparisons after 24,576 timed calls.
No kernel/default changed in this diagnostic. Physical DRAM traffic, configured
RAM peak and full-model throughput remain unmeasured. Frozen source, logs,
analysis and binary/library hashes: `results/20261020/fp4-service-roofline/`.

### Same-call FP4 adapter phase profile

To avoid subtracting independent benchmarks, the rotating sixteen-bank adapter
now has a frozen scratch probe that samples the pool's existing cumulative
phase counters around each 32-call timed window. No production instrumentation
or scheduling changed. It reports gate/up, host hidden transformation plus
quantization, down, and the residual of total wall time minus those three
intervals. Gate/up and down intervals include their pool dispatch/wait costs;
they are not pure arithmetic timers. The residual includes input quantization,
finite scans, output restoration, pool bookkeeping and timer overhead.

Normal-input candidate-process medians in the first run are:

| Layer / mode | Total ms | Gate/up ms | Hidden transform + quant ms | Down ms | Remainder ms |
| --- | --- | --- | --- | --- | --- |
| 3 / sequential | 0.7562 | 0.4552 | 0.0080 | 0.2737 | 0.0165 |
| 3 / batch | 0.6305 | 0.3989 | 0.0077 | 0.2170 | 0.0089 |
| 4 / sequential | 0.7593 | 0.4592 | 0.0083 | 0.2765 | 0.0164 |
| 4 / batch | 0.6257 | 0.3949 | 0.0080 | 0.2137 | 0.0090 |

These are independent medians and need not sum exactly. Per-window fractions
show gate/up at 63.0–63.9% and down at 34.0–34.2% of batched normal-input time
across both candidate processes. Hidden transformation/quantization takes
1.1–1.3%, and the residual 1.1–1.4%. Sequential residual fractions are 1.9–2.2%.
The late candidate again experiences elevated absolute times, including layer-3
1.135/1.058 ms sequential/batch, and is retained. Normal-input paired latency
ratios are 0.8470/0.9313 for layer 3 and 0.8255/0.8046 for layer 4; same-code
controls are 0.9960–1.0048.

This profile prioritizes matrix phases and their dispatch/placement costs over
further input-quantization or hidden-rescale micro-optimizations: eliminating
the measured hidden step alone could save only about one percent of these
batched calls. It does not identify the matrix phases as compute-bound versus
memory-bound. No theoretical utilization or full-model tok/s follows.

All 2,304 final expert comparisons pass after 24,576 timed calls across 768
windows. Sixteen primary participants, real partial weights, compact NUMA
storage, rotating copies and background miners match the preceding screen.
Frozen source, raw per-window counters and analysis:
`results/20261020/fp4-batch-phase-profile/`.

### Finer FP4 batch task partitioning with rotating weights

The primary three-expert NT1 compact batch was tested with six and twelve tasks
per participant against three. Both arms use the batch API, sixteen participants,
a pinned host and sixteen rotating NUMA weight banks (679,477,248 projection
bytes per layer). Input quantization and small-hidden preservation remain
included. This isolates partitioning from the previously measured benefit of
submitting experts together. Processes use task choices 6/12/3/3/12/6; each
contains sixteen alternating pairs of 32 calls per cell, across two real partial
layers and normal/small/zero input. Miners and defaults remain unchanged.

For three experts, the owned-row scheduler produces 48/96/192 tasks per matrix
phase at settings 3/6/12. This changes row chunk size as well as dispatch count.
Normal-input paired medians are:

| Task setting / layer | Candidate/three, first / repeat | Three/three controls |
| --- | --- | --- |
| 6 / 3 | 0.9865 / 0.9887 | 0.9970 / 0.9976 |
| 6 / 4 | 0.9629 / 0.9914 | 0.9938 / 0.9930 |
| 12 / 3 | 0.9699 / 0.9592 | 0.9970 / 0.9976 |
| 12 / 4 | 1.0006 / 0.9939 | 0.9938 / 0.9930 |

Six tasks has small paired improvements in all twelve cells (0.3–3.7%), but its
normal-input repeat improves only 0.9–1.1%. Twelve tasks is mixed, with four of
twelve cells slower. Across all inputs, same-code controls range 0.9708–1.0117.
Normal-input candidate wins are only 8–13 of sixteen pairs. These results do not
justify replacing the current three-task primary batch setting. Six remains a
minor candidate, not a broadly qualified improvement.

Some whole-run medians show much larger differences than paired ratios because
of outliers: e.g. first six-task layer 4 has 0.875/0.669 ms medians but a paired
ratio of 0.9629. Do not turn that into a 24% scheduling speedup. Per-window phase
counters, raw pairs and all controls are preserved. The useful next priority is
matrix execution and placement, rather than expecting a large gain from finer
partitioning alone.

All 3,456 final expert checks pass bitwise after 36,864 timed calls across 1,152
windows. There is no GPU execution, full-model throughput/quality or theoretical
peak-utilization result. Source, binary/library hashes and measured phase data:
`results/20261020/fp4-batch-finer-tasks/`.

### Precomputed FP4/activation scale products: rejected

A scratch offset-dot candidate moves FP4 scale × Q8 activation-scale products
out of the row loop. After the existing raw -128 safety scan, safe NT1 inputs
prepare a 128 × 256 float table (128 KiB at K4096), indexed by activation block
and FP4 scale byte. The row loop reads those products in place of its scale
broadcast/multiply. Preparation stays inside every timed GU call. Integer dots,
per-output FMA order and final reduction are unchanged. Unsafe raw -128 inputs
skip table filling and retain the original arithmetic fallback, but the larger
object/stack layout remains. No production source or runtime option changed.

The same-process paired harness uses K4096/R128, one/64 banks and primary cores
3/11. Weight scales are either all-byte or finite 8–126; positive finite Q8 half
scales vary by block (`0x1000` through `0x3fff`) instead of a single power of two,
so bitwise comparisons cover rounded scale products. Safe inputs use -127..127;
fallback cases inject -128. Unchanged NT2 cases are controls. Each case has
twelve alternating pairs of 64 calls, with initialization checks for every bank
and output checks after each timed arm. Prefetch 2048, offset enabled, narrow
and Clang selectors disabled match the existing offset baseline.

| Cases | Candidate/baseline median paired wall-time ratios |
| --- | --- |
| NT1 safe, all-byte weight scales | 0.9774–0.9935 |
| NT1 safe, finite weight scales | 1.0010–1.0166 |
| NT1 raw -128 fallback | 1.0260–1.0335 |
| Unchanged NT2 controls | 1.0008–1.0033 |

Reject this candidate: modest all-byte gains do not offset the finite-scale and
fallback regressions. The fallback regression cannot be attributed to table
construction because filling is skipped there; stack/object layout and compiler
placement effects are not isolated. The experiment does not justify removing
multiplications by expanding per-call scale storage on this primary CPU.

All 480 final timed-arm checks after 30,720 calls pass bitwise, as do 650 initial
bank comparisons. These are synthetic CPU GU tests; K32/narrow execution, worker
CPUs, GPU/SYCL, full-model quality/tok/s and theoretical utilization were not
qualified. Candidate source, exact build commands, paired raw logs and hashes:
`results/20261020/fp4-offset-scale-products/`.

### Benchmark metadata records CPU kernel choices

`glm_v100_bench.experiment_environment()` now records explicitly configured
Q2/Q3 and FP4 kernel choices, including route-aware scheduling, prefetch, narrow
and offset selection, compiler variants, lookup mode and ISA restrictions.
Exact environment strings are retained, including disabled values such as `0`
and `2`; no defaults are synthesized. Only named CPU switches were added to the
existing selection, and unrelated environment variables remain excluded.
`glm_v100_sweep` uses this same metadata helper. All 25 benchmark diagnostic
unit tests pass with Python 3.12 and the `regex` dependency in isolated temporary
storage; no inference or GPU execution was needed.

The remote TP launcher already supports explicit `WORKER_ENV` settings. Recording
local benchmark environment is not evidence of the remote process's environment;
worker settings still need to be supplied and recorded separately. Its partial
intermediate-width geometry also must not inherit full-width performance claims.
Validation and source hashes: `results/20261020/benchmark-cpu-settings/`.

### Q2/Q3 half-width worker task screen

Synthetic Q2_K gate/up and Q3_K down tests now cover H4096/F1024, a geometry
relevant to a half-intermediate tensor-parallel worker. This is a CPU kernel
screen, not a network-worker or full-decoder test. Each Ryzen 3950X worker uses
four participants and an unpinned host, NT1 split mode, sixteen rotating banks,
prefetch 1024 and unchanged miners. Manual twelve/three task comparisons retain
candidate/control/control/candidate processes, twenty alternating pairs and 64
calls per timed arm.

| Worker / experts | Twelve/three first / repeat | Three/three controls |
| --- | --- | --- |
| One / 3 | 0.9617 / 0.9695 | 1.0021 / 1.0211 |
| One / 8 | 0.9517 / 0.9411 | 0.9997 / 0.9939 |
| Two / 3 | 1.0390 / 1.0303 | 1.0034 / 1.0014 |
| Two / 8 | 0.9329 / 0.9324 | 1.0116 / 1.0027 |

Eight-expert calls improve 4.8–6.8%, with all eighty candidate pairs favoring
twelve. Three-expert results disagree across workers and regress 3.0–3.9% on
worker two. Therefore a half-width policy must exclude three-expert calls rather
than reuse the full-width count predicate. All 640 final batch checks after
40,960 timed calls match reference bitwise; per-geometry hashes agree across
workers and arms. Input quantization is excluded. Evidence:
`results/20261020/q23-half-width-tasks/`.

### Eight-expert half-width Q2/Q3 policy integration

The opt-in `STRATA_Q23_ROUTE_TASKS=1` split policy now additionally accepts
H4096/F1024 with exactly eight NT1 expert jobs. The existing full-width
H4096/F2048 three/four/eight cases and all other guards remain unchanged.
Half-width three-expert calls retain baseline scheduling; untested widths/counts
also retain it. The option remains disabled by default. This extension does not
change direct layer-dataflow scheduling or automatically configure any worker.

The actual policy, compared with disabled policy on both four-participant workers,
produces these paired medians under the same rotating-bank protocol:

| Worker / experts | Policy/baseline first / repeat | Disabled/disabled controls |
| --- | --- | --- |
| One / 3 | 0.9977 / 0.9909 | 1.0062 / 1.0124 |
| One / 8 | 0.9575 / 0.9282 | 1.0040 / 0.9997 |
| Two / 3 | 0.9944 / 0.9894 | 1.0133 / 1.0000 |
| Two / 8 | 0.9417 / 0.9331 | 1.0013 / 0.9990 |

Eight-expert service improves 4.2–7.2%, with 76/80 candidate pairs faster.
Three-expert calls use baseline scheduling in both arms; their small differences
are background variation, not a policy speedup. All 640 final batch checks after
40,960 timed calls match bitwise, including output hashes from the earlier manual
half-width screen. No network worker or full decoder ran in this qualification.

The expanded CPU test checks 72 policy decisions and 48 changing-route bitwise
comparisons across both intermediate widths per backend. It and the existing
168-check task partition and 384-batch native pool tests pass on HIP- and
CUDA-configured CPU builds. SYCL remains unbuilt because its compiler is
unavailable. No GPU, full-model quality/tok/s or 90%-peak result follows.
Frozen source, paired raw results, hashes and validation:
`results/20261020/q23-half-width-policy/`.

### Two-expert batching in the partial routed runner

`glm-fp4-native-chain` now accepts explicit `batch-two` in addition to
`batch-three`. They can be selected independently or together. The two-expert
path uses the validated native batch API with independent hidden-scale state and
persistent two-output storage, swapping completed vectors into the existing
routed output slots. One-expert/idle routes keep their existing execution. A
remote client cannot choose either option: the server selects compute policy.
The batch wire protocol and default sequential behavior are unchanged.

Both HIP- and CUDA-configured CPU builds pass all sixteen local combinations of
complete/compact storage and the two flags, totaling 768 layer checks. All output
dumps match the frozen pre-option reference byte for byte. Duplicate `batch-two`
and client-side selection are rejected before opening weights or a connection.
No GPU was used; SYCL remains unbuilt because its compiler is unavailable.
Local evidence: `results/20261020/native-two-route/`.

A first direct-IB ABBA comparison varies only two-expert batching; three-expert
batching stays enabled. Primary server: sixteen participants, pinned host,
compact NUMA weights with original blobs released. Worker client: four unpinned
participants. The four runs pass another 1,024 layer checks. Two-expert process
medians improve 2.1–11.4%, but unchanged routes vary by up to about 14%; this is
not a clean isolated gain. Frozen evidence:
`results/20261020/native-two-ib-timing/`.

A second scratch probe alternates sequential/batched two-expert execution within
one server process, with consecutive pairs reversing order. Thirty-two repeats
run all eight routing scenarios; the first pair is excluded from timing.
Candidate/control/control/candidate processes keep the same peer topology and
settings. Only the scratch probe uses the alternating policy; the runner's
public option is fixed for the process.

| Layer, two-expert route | Batch/sequential paired medians, first / repeat | Sequential/sequential controls |
| --- | --- | --- |
| 3 | 0.8708 / 0.8944 | 0.9387 / 0.9393 |
| 4 | 0.9215 / 0.8789 | 1.0453 / 1.0021 |

Candidate improvements are 7.9–12.9%, with 10–12 of fifteen pairs faster in each
cell. Controls retain a 6.1% downward shift for layer 3 and up to a 4.5% upward
shift for layer 4, so timing variation remains material. Candidate process
medians span 0.782–0.841 ms sequential and 0.676–0.764 ms batched. Service includes
request/reply, weighted reduction and residual update; local sequential oracle
work is excluded. Miners remain running and clocks/load are uncontrolled.

All 2,048 layer checks in the paired probe pass bitwise, with 1,920 warm windows
retained. Server counters confirm exactly 32 two-expert batch calls per candidate
process; control processes retain sequential two-expert computation. Output
repeats match the frozen reference. This establishes partial routed correctness
and a promising contention-conditioned service result, not full-model quality,
decode tok/s or 90%-peak utilization. Production already batches host experts,
so this adapter gain must not be added to production throughput estimates.
Source snapshots, raw paired samples, fixture/output hashes and analysis:
`results/20261020/native-two-ib-paired/`.

### Two-token FP4 batch API and verification-work screen

The worker's Xeon/V100 study was revalidated read-only: its report SHA-256 remains
`7c5a6a46a166dc78ff41049e29403bd81128446065512585e1be8bbd39ada80d`. Its selected
32.10 tok/s profile still uses speculation and route dropping, distinct from the
preserved 5.26 tok/s W4A16 reference. There is no new remote throughput result.
The previously reported verification dominance motivates measuring multi-token
CPU work, but it does not make verification a CPU-only timer or establish local
acceptance rates.

`NativeFp4CpuExpert::run_batch_into` and its three-expert wrapper now accept a
trailing `tokens` argument of one or two, defaulting to one. The caller supplies
token-major input/output vectors. Input quantization is shared across experts
once per token; each job points to the appropriate activation/output slice.
Hidden scaling remains independent for every expert/token pair and is restored
per output slice. The same original/compact storage and K32 rejection rules
apply. Unsupported token counts are rejected before buffer writes. The existing
single-expert implementation remains text-identical to its previous snapshot.
The partial network protocol and routed runner still carry NT1 requests only.

Both HIP- and CUDA-configured CPU builds pass 720 expert-vector comparisons and
320 argument-rejection checks across two/three experts, NT1/NT2 and complete/
compact storage. Tests include different token magnitudes, original-weight
release, disabled scaling, repeated expert pointers, small/zero inputs and
NT2 -> NT1 -> NT2 reuse of one pool and its workspaces. Each vector comparison
covers the entire active token extent. SYCL was not built because its compiler
is unavailable; no GPU ran. Evidence: `results/20261020/batch-nt2-api/`.

The primary service screen compares one three-expert NT2 batch with three
sequential NT2 expert calls. Sixteen participants, pinned host, compact NUMA
storage, sixteen rotating weight banks, real layers 3/4, prefetch 2048 and three
tasks per participant match the NT1 profile. The second token has an additional
0.001 input factor to exercise independent hidden scaling. Four processes run
candidate/control/control/candidate, each with sixteen alternating pairs of 32
calls per cell. Preparation is excluded; input quantization, finite scans,
hidden restoration and pool service are included.

| Layer / normal input | Batch/sequential paired medians, first / repeat | Sequential/sequential controls |
| --- | --- | --- |
| 3 | 0.8635 / 0.8640 | 0.9976 / 0.9802 |
| 4 | 0.8959 / 0.8616 | 0.9953 / 0.9984 |

Normal-input latency improves 10.4–13.8%, with 58/64 candidate pairs faster.
Candidate whole-run medians range 0.941–1.327 ms versus sequential 1.081–1.414 ms;
these ranges include different cells and are not themselves a paired speedup.
Across all inputs, controls range 0.9802–1.0156. Background-load outliers remain
in raw samples and phase counters. All 2,304 final expert-vector checks after
24,576 timed NT2 calls pass bitwise.

This establishes an opt-in partial two-token service primitive. Production
already batches native expert jobs, and no new speculative decoder policy,
acceptance rate, quality qualification, full-model tok/s or theoretical-peak
claim follows. Do not multiply its service improvement into the V100 report's
throughput. Source snapshots, timings, controls and hashes:
`results/20261020/batch-nt2-timing/`.

### NT2 FP4 batching on four-participant workers

The primary's two-token batching gain was checked on both Ryzen 3950X workers
with four CPU participants and unpinned host threads. Sixteen independent banks
retain complete original-layout weights (679,477,248 projection bytes per real
partial layer); no NUMA copies are created. Fixture and portable binary hashes
are verified before execution. Both arms use three tasks per participant and
native Q8 activations with small-hidden preservation. The NT2 kernel is the
existing signed two-token path; the recorded offset switch only selects NT1.
Miners and running services remain unchanged.

Candidate/control/control/candidate processes each run sixteen alternating pairs
of 32 calls per cell, across layers 3/4 and normal/small/zero inputs. Each call
processes three experts and two tokens; the second token has an additional
0.001 amplitude factor. Input quantization, finite scans, hidden restoration and
pool service are timed; startup packing is excluded.

| Worker / layer, normal input | Batch/sequential first / repeat | Sequential/sequential controls |
| --- | --- | --- |
| One / 3 | 0.9865 / 0.9814 | 0.9894 / 0.9960 |
| One / 4 | 0.9700 / 0.9839 | 1.0097 / 1.0112 |
| Two / 3 | 1.0302 / 1.0145 | 1.0043 / 0.9910 |
| Two / 4 | 1.0350 / 1.0209 | 0.9984 / 0.9950 |

Worker one's normal-input gain is only 1.4–3.0%, versus the primary's earlier
10.4–13.8%. Worker two regresses 1.4–3.5%, with just 2–4 of sixteen candidate
pairs faster per normal-input cell. Keep sequential adapter calls for this
worker-two workload; do not enable the NT2 batch API universally based on the
primary result. This does not recommend changing the production decoder's
already-batched native path, which has different surrounding work.

Same-call phase measurements show the regression is not explained by input
quantization or output restoration: in worker two's first layer-4 comparison,
the remainder drops from 0.0306 to 0.0225 ms while the down phase rises from
1.2042 to 1.3107 ms. These are independent medians, and matrix phases include
pool dispatch/waits. Placement, task partitioning and background contention
have not been separately isolated.

All 4,608 final expert-vector comparisons after 49,152 timed NT2 calls pass
bitwise across 1,536 windows. Every comparison covers both token outputs; all
banks also cycle through the checked final positions. No NT2 IB transport,
GPU, full-model speculative acceptance/quality/tok/s or 90%-peak qualification
ran. Source snapshots, raw phase timings, fixture/library/binary hashes and
analysis: `results/20261020/batch-nt2-workers/`.


### NT2 worker task partition screen

Worker two's earlier NT2 batching regression motivated a task-partition screen
on the same Ryzen 3950X, with four CPU participants, unpinned host threads and
miners unchanged. Both arms batch three experts with two tokens. The baseline
uses three tasks per participant; the candidate uses six or twelve. Six processes
use task settings 6/12/3/3/12/6, with the middle two serving as identical-policy
controls. Each process covers real partial layers 3/4 and normal/small/zero input,
with sixteen alternating pairs of 32 calls per cell. Sixteen original-layout
weight banks retain 679,477,248 projection bytes per layer. Input quantization,
finite checks and hidden restoration are included; startup packing is excluded.

| Tasks / layer, normal input | Candidate/baseline paired medians, first / repeat |
| --- | --- |
| 6 / 3 | 0.8975 / 0.9084 |
| 6 / 4 | 0.8980 / 0.9010 |
| 12 / 3 | 0.9050 / 0.8974 |
| 12 / 4 | 0.8863 / 0.9080 |
| 3 control / 3 | 1.0166 / 1.0080 |
| 3 control / 4 | 1.0067 / 0.9987 |

Six tasks improves normal-input batched latency 9.2–10.3%, winning 62/64 pairs;
twelve improves it 9.2–11.4%, winning 64/64. Across all amplitudes, six-task
ratios are 0.8936–0.9273 and twelve-task ratios 0.8863–0.9168; identical-policy
controls range 0.9848–1.0170. All 3,456 final expert-vector comparisons after
36,864 timed calls pass bitwise, covering both tokens. The raw log's `batch`
field selects candidate versus baseline; both arms are batched in this screen.
The summary's inherited `sequential_ms` field therefore means baseline batch
latency, not sequential calls.

This supports a direct six-task batch versus three-task sequential comparison;
it does not by itself establish that batching beats sequential calls. Do not
multiply this result by the earlier independent batching ratios. Production
policy and defaults remain unchanged. No GPU, full-model quality/tok/s or
90%-peak qualification follows. Source, settings, paired samples and analysis:
`results/20261020/batch-nt2-worker-tasks/`.


### Direct NT2 six-task batching confirmation on worker two

A separate paired experiment compares three-expert NT2 batching at six tasks
per participant directly against three sequential NT2 expert calls at three
tasks. It retains the preceding screen's four-participant Ryzen 3950X setup,
original-layout sixteen-bank partial weights, amplitudes, two layers, timing
boundaries and unchanged miners. Candidate/control/control/candidate processes
run sixteen alternating pairs of 32 calls per cell. Control processes use
sequential calls at three tasks in both arms. The six-task selection is made
through the existing per-call pool setter, outside the timed service interval.

| Layer / normal input | Batch/sequential paired medians, first / repeat | Sequential controls |
| --- | --- | --- |
| 3 | 0.9242 / 0.9501 | 1.0006 / 1.0050 |
| 4 | 0.9177 / 0.9187 | 0.9873 / 1.0098 |

Normal-input service latency improves 5.0–8.2%, with all 64 candidate pairs
faster. Across normal/small/zero inputs, candidate ratios range 0.9061–0.9501
with 191/192 wins; identical-policy controls range 0.9873–1.0106. Candidate
normal-input whole-run medians are 3.304–3.358 ms versus sequential
3.536–3.633 ms; these ranges span different cells and are not paired ratios.
All 2,304 final expert-vector checks after 24,576 timed calls pass bitwise.
Every output includes both tokens. Binary and fixture hashes are checked before
remote execution; the analyzer also checks build-time source/library hashes
against the current files before freezing source snapshots.

This supersedes the earlier sequential-adapter recommendation for this exact
worker-two three-expert NT2 workload: six-task batching now has a direct
measured benefit. It does not qualify twelve tasks, other expert counts, NT1,
other workers, or a global environment setting. In particular, a global
`STRATA_NATIVE_TASKS_PER_THREAD=6` also affects other routes and overrides the
measured Q2/Q3 adaptive policy; do not apply it globally from this result.
Production defaults remain unchanged. No GPU execution, NT2 IB transport,
full-model throughput/quality or 90%-peak utilization was measured. Frozen
source, build manifest, settings, paired timings and analysis:
`results/20261020/batch-nt2-worker-direct/`.


### NT2 six-task batching confirmation on worker one

The same portable probe and fixture hashes were checked on worker one, another
Ryzen 3950X, with four CPU participants and unpinned host threads. Miners remained
running. The protocol matches the preceding direct comparison: three-expert
NT2 batching at six tasks versus three sequential expert calls at three tasks,
sixteen rotating original-layout weight banks, layers 3/4, normal/small/zero
inputs, and candidate/control/control/candidate processes. Each cell contains
sixteen alternating pairs of 32 calls. The second token has an additional 0.001
input scale. Input quantization and hidden restoration are timed.

| Layer / normal input | Batch/sequential paired medians, first / repeat | Sequential controls |
| --- | --- | --- |
| 3 | 0.9060 / 0.9634 | 1.0073 / 0.9890 |
| 4 | 0.8898 / 0.9381 | 0.9902 / 0.9964 |

Normal-input gains range 3.7–11.0%, with 60/64 candidate pairs faster. Across all
amplitudes, candidate ratios range 0.8898–0.9634 with 184/192 wins; controls range
0.9840–1.0073. All 2,304 final expert-vector comparisons after 24,576 timed calls
pass bitwise. The smaller repeat gain remains in the record; background load
and placement are uncontrolled. Together with worker two's independent direct
comparison, this supports six-task batching for the tested three-expert NT2
adapter workload on both workers. It does not select a global task override or
qualify NT1, other route counts, IB NT2 execution or full-model decode.
Production defaults remain unchanged. Evidence, source snapshots and build-time
hash validation: `results/20261020/batch-nt2-worker-one-direct/`.

The requested V100 repository now links a newer `docs/GLM_DFLASH2_V100.md`
report. Its selected MTP result remains 32.10 decode / 520.89 prefill tok/s;
a later experimental MTP configuration reports 40.26 / 519.34 and passes eight
reused long and eight reused natural-length coding checks, with Fresh20 still
unscored. The 38.56 tok/s DFlash2 route-drop trial failed all eight reused coding
checks and was rejected. These are remote report observations, not new runs
performed here and not exact batch-one throughput comparisons.

The newer report also rejects its width-specific CPU policy after a repeated
36.43 versus 35.45 tok/s control/candidate comparison. An instrumented warmed
512-output request records 17,841 one-token groups out of 20,455 CPU expert
groups (87.2%), and only one eight-token group. Those counters include request
preparation and decode and do not establish this tree's route distribution.
The same instrumented request reports 271 ms setup, 4,147 ms fused CPU products
and 4,857 ms routing/input preparation/coordinator waiting. The last category
includes synchronization; phase totals overlap and must not be added to fit
a decode estimate. This shifts the next optimization priority toward NT1
service and surrounding routing work, rather than extrapolating the NT2 gain.
Report fingerprints and qualification notes are frozen alongside this result
in `v100-revalidation.json`. No remote GPU benchmark ran for this recheck.


### NT1 FP4 task partitioning on both workers

The newer V100 profile's predominance of one-token expert groups motivated an
NT1 scheduling check. Both Ryzen 3950X workers use four CPU participants with
unpinned host threads; miners remain running. Both arms call the validated
three-expert batching API. The candidate uses six tasks per participant versus
three in the baseline, isolating partitioning from the earlier batching versus
sequential comparisons. Offset arithmetic and prefetch 2048 remain enabled.
Real partial layers 3/4 each rotate sixteen original-layout banks, totaling
679,477,248 projection bytes per layer. Input quantization, finite checks,
hidden restoration and service are timed; startup preparation is excluded.

Candidate/control/control/candidate processes each cover normal/small/zero
input with sixteen alternating pairs of 32 calls per cell. Controls use three
tasks in both arms. Portable binary and fixture hashes match across workers.

| Worker / layer, normal input | Six/three-task paired medians, first / repeat | Three/three controls |
| --- | --- | --- |
| One / 3 | 0.9222 / 0.9596 | 1.0080 / 0.9797 |
| One / 4 | 0.9409 / 0.9269 | 0.9704 / 1.0043 |
| Two / 3 | 0.9158 / 0.9193 | 0.9843 / 0.9933 |
| Two / 4 | 0.9251 / 0.9033 | 0.9830 / 1.0129 |

Worker one's normal-input gain is 4.0–7.8%, winning 58/64 candidate pairs;
worker two gains 7.5–9.7%, winning 64/64. Across all amplitudes, worker-one
candidate ratios are 0.9221–0.9785 versus controls 0.9704–1.0092; worker-two
ratios are 0.9033–0.9415 versus controls 0.9830–1.0129. Worker one's smallest
normal-input gain is close to its largest control drift; retain that limit
rather than claiming a fixed gain. All 4,608 final expert-vector comparisons
after 49,152 timed NT1 calls pass bitwise across 1,536 timing windows.
Build-time source/library hashes are checked before evidence snapshots.

This supplies NT1 evidence for six-task scheduling on the tested three-expert
worker service, independently of the NT2 result. It does not qualify a global
six-task override: that would also affect other route counts and override the
Q2/Q3 adaptive policy. The next validation is a direct routed IB service test
with the worker doing expert computation; previous primary-server IB tests
cannot establish this worker gain survives communication and coordination.
No production/default change, GPU execution, full-model throughput/quality or
90%-peak utilization claim follows. Source, settings, paired samples and
analysis: `results/20261020/batch-nt1-worker-tasks/`.


### Worker-side NT1 task scheduling over IB

The worker-one Ryzen 3950X now serves the partial native FP4 requests over IB,
using four CPU participants and retained original-layout blobs. The primary
Threadripper client uses sixteen participants with compact NUMA weights for
its untimed sequential oracle. Miners remain unchanged. This reverses the
compute direction of earlier primary-server IB tests. Both server arms batch
three-expert requests; only those requests alternate six versus three tasks.
One/two-expert and idle routes retain their prior execution. The scratch probe
resets task count before each nonempty request, so later routes cannot inherit
six tasks accidentally. Production source and wire protocol are unchanged.

Thirty-two repeats cover eight scenarios on two real partial layers: one/two/
three experts, permuted routing, idle, zero input, small input and zero gains.
The server selects alternating task policies by repeat, reversing their order
across consecutive pairs. The first pair is excluded from timing analysis.
Candidate/control/control/candidate processes provide fifteen warm pairs per
cell. Controls use three tasks throughout. Client timing includes the IB
request/replies, weighted reduction and residual; the local oracle is excluded.
This uses one resident weight bank per layer, unlike the sixteen-bank CPU
service screen, and does not isolate communication overhead by subtraction.

All 2,048 layer checks and complete output dumps match the frozen native-Q8
reference bitwise. Each process serves 512 requests with 320 three-expert batch
calls, 128 sequential-route calls and 64 idle calls. Candidate processes select
six tasks for 160 calls; controls never do. Across normal ordered/permuted
three-expert cells, the first candidate's paired ratios are 0.9225–0.9669 but
the repeat gives 0.9718–1.0140. Matching controls span 0.9748–1.0204. The repeat
therefore does not establish a reliable routed improvement. Preserve both the
initial gain and the weak repeat. A longer paired run is needed before selecting
this setting for routed service; the CPU-only result is insufficient.
Source, settings, logs, output/fixture hashes and build-source hash checks:
`results/20261020/native-nt1-worker-ib-tasks/`. No full-model, GPU or
90%-peak qualification follows.


### Longer worker-side IB scheduling confirmation

The same worker-one IB comparison was repeated with 128 scenario repeats per
process. The first pair is again excluded, leaving 63 alternating pairs per
cell; all other settings, route patterns and timing boundaries match the short
screen. Candidate/control/control/candidate order is retained. The longer
probe is separately built and hashed; the short screen remains frozen.

| Scenario / layer | Six/three-task paired medians, first / repeat | Three/three controls |
| --- | --- | --- |
| Ordered three-expert / 3 | 0.9359 / 0.9404 | 0.9984 / 1.0011 |
| Ordered three-expert / 4 | 0.9327 / 0.9470 | 1.0194 / 1.0041 |
| Permuted three-expert / 3 | 0.9454 / 0.9350 | 0.9865 / 1.0176 |
| Permuted three-expert / 4 | 0.9366 / 0.9361 | 0.9957 / 1.0029 |

Normal-input routed latency improves 5.3–6.7% across both repeats, with 399/504
candidate pairs faster. Matching controls span 0.9865–1.0194. This longer run
supports the route-specific worker scheduling gain despite the inconclusive
short repeat. It does not erase that earlier variability or isolate the source
of the gain within CPU dispatch, queueing and communication.

All 8,192 layer checks and complete output dumps match the frozen native-Q8
reference bitwise. Each process serves 2,048 requests: 1,280 batched, 512
sequential-route and 256 idle calls. Candidate processes select six tasks for
640 three-expert calls; control processes retain three. The analysis includes
8,064 warm windows. Settings, source snapshots, hashes and raw samples are in
`results/20261020/native-nt1-worker-ib-tasks-long/`.

This qualifies the measured four-participant worker-one, three-expert NT1
service over IB, using resident partial weights. A production opt-in should
select six only for the measured route and preserve explicit task budgets,
other expert/token counts and Q2/Q3 policy. No global setting or production
default changed here. No full-model decode, GPU execution, full-model quality
or theoretical-peak qualification follows.


### Opt-in FP4 route task policy

`STRATA_NVFP4_ROUTE_TASKS=1` now selects six tasks per participant for the
measured three-expert FP4 split-service route. The constructor reads exact-one
semantics through the shared C11 environment helper; unset, `0` and `2` stay
off. The diagnostic `set_nvfp4_route_tasks` setter changes selection between
completed calls. This is independent of `STRATA_Q23_ROUTE_TASKS` and defaults
off. Benchmark environment capture now records the exact switch value.

Selection requires four CPU participants including the host, original-layout
non-NUMA-owned rows, H4096/F2048 NVFP4 gate/up and down, three jobs with matching
NT1 or NT2 widths, SwiGLU limit 10, and the base three tasks per participant.
Nondefault per-participant settings and fixed task budgets win. K32, canonical
arithmetic, gate skipping, fused hidden quantization, observers and native-local
execution retain their prior selection. Hidden transformation remains serial
between phases and is supported, as in the measured small-hidden adapter.
The decision is local to each split call; it never changes the configured base
or the layer-dataflow schedule. Q2/Q3's twelve-task choices remain intact.

For the measured worker configuration, retain offset-dot arithmetic, prefetch
2048 and disabled AVX512/local scheduling, then opt in with:

```sh
STRATA_NVFP4_ROUTE_TASKS=1
```

This is a worker route option, not a recommendation to set
`STRATA_NATIVE_TASKS_PER_THREAD=6` globally. The primary sixteen-participant
route and other expert counts remain outside its selection.

The longer worker-one IB probe was rebuilt against the integrated policy.
It now alternates the policy setter rather than manually setting the task
count; Q2/Q3 policy is enabled concurrently. Four processes retain the
128-repeat candidate/control/control/candidate protocol and 63 warm pairs per
cell. Normal ordered/permuted three-expert results are:

| Scenario / layer | Enabled/disabled paired medians, first / repeat | Disabled controls |
| --- | --- | --- |
| Ordered / 3 | 0.9453 / 0.9231 | 0.9937 / 1.0205 |
| Ordered / 4 | 0.9398 / 0.9398 | 0.9973 / 1.0008 |
| Permuted / 3 | 0.9595 / 0.9429 | 1.0193 / 0.9945 |
| Permuted / 4 | 0.9353 / 0.9353 | 0.9910 / 0.9884 |

The integrated policy improves these routed cells 4.0–7.7%, with 404/504 pairs
faster. Controls span 0.9884–1.0205. All 8,192 layer checks and complete dumps
match the pre-change frozen native-Q8 reference bitwise, including the disabled
path. This is partial-weight output preservation, not a full-model default
output or quality qualification. Timing includes IB service, reduction and
residual; the primary client's oracle is excluded. Miners remain running.
NT2 performance remains supported by the earlier CPU service tests, not by this
NT1-only transport protocol. Integrated timing evidence:
`results/20261020/native-nt1-worker-ib-policy/`.

Both HIP- and CUDA-configured CPU builds pass the four scoped tests
`pool_tasks_test`, `native_pool_test`, `q23_route_tasks_test` and
`nvfp4_route_tasks_test`. The new test checks 162 selection decisions and 60
changing-route comparisons per build, including mixed-width rejection,
nondefault/fixed budgets, format exclusions, Q2/Q3 coexistence and serial
hidden transforms. Outputs must be finite and nonzero for nonempty routes;
the enabled and disabled buffers match bitwise across route/token transitions.
The initial test compile lacked the CPU-dispatch declaration include; that was
corrected before these successful builds. No unresolved compile failure remains.

Real layers 3/4 also pass 2,160 expert-vector comparisons and 960 invalid-input
checks across both builds, complete/compact modes and switch values 0/1/2.
These exercise NT1/NT2 reuse, independent hidden scaling, compact-original
release and excluded two-expert routes. All 25 benchmark metadata tests pass.
Only CPU targets were built and executed; no GPU ran. SYCL compilers remain
unavailable. Validation and source snapshots:
`results/20261020/nvfp4-route-policy-validation/`. Full-model decode and
90%-peak utilization remain unverified.


### Two-expert NT1 worker task screen

The two-expert share in the three-node 3/3/2 assignment motivated a separate
screen; the existing FP4 route policy still selects only three-expert calls.
Both Ryzen 3950X workers use four CPU participants with unpinned host threads
and unchanged miners. Both arms call the validated two-expert NT1 batch API,
using six versus three tasks per participant. The FP4 route policy is explicitly
disabled so only the manual task setting differs. Offset arithmetic and
prefetch 2048 remain enabled. Sixteen original-layout banks retain 452,984,832
projection bytes per real partial layer. Quantization, finite checks and hidden
restoration are timed; startup preparation is excluded.

Candidate/control/control/candidate processes cover layers 3/4 and normal,
small and zero input with sixteen alternating pairs of 32 calls per cell.
Controls use three tasks in both arms. Binary and fixture hashes match on both
workers; build-source hashes are checked before freezing evidence.

| Worker / layer, normal input | Six/three-task paired medians, first / repeat | Three/three controls |
| --- | --- | --- |
| One / 3 | 0.9192 / 0.9487 | 1.0152 / 1.0059 |
| One / 4 | 0.9298 / 0.9275 | 0.9898 / 0.9893 |
| Two / 3 | 0.9011 / 0.9338 | 0.9902 / 0.9881 |
| Two / 4 | 0.9046 / 0.9419 | 1.0019 / 1.0082 |

Worker one gains 5.1–8.1% on normal input, winning 57/64 pairs; worker two gains
5.8–9.9%, winning 60/64. Across all amplitudes, candidate ratios are
0.9192–0.9672 and 0.9011–0.9419 respectively; controls span 0.9786–1.0152 and
0.9769–1.0218. All 3,072 final expert-vector comparisons after 49,152 timed
calls pass bitwise, covering 1,536 windows. This is a two-expert NT1 service
result, not an NT2, IB or full-model qualification. The three-expert opt-in is
not broadened from this CPU-only screen. Evidence:
`results/20261020/batch-two-nt1-worker-tasks/`.


### Two-expert NT1 service on the direct worker-two IB link

The worker-two Ryzen 3950X serves the two-expert batch over the direct IB link
with four CPU participants and original-layout partial weights. The primary
client uses sixteen compact NUMA participants for its untimed oracle. Both
peers select IB port 2. An initial attempt incorrectly selected worker port 1,
produced completion errors and timed out before qualification. Its raw logs and
hashes remain in `failed-port1/`; it contributes no performance number. The
client was terminated on its 90-second timeout, and absence of the bounded
remote server was checked before retrying the corrected mapping.

The corrected run uses 128 repeats, eight route scenarios and layers 3/4.
Both arms batch two/three-expert calls; only two-expert calls alternate six
versus three tasks. Other routes retain three tasks. Candidate/control/control/
candidate processes provide 63 warm pairs per cell after discarding the first
pair. The FP4 route policy remains disabled for this manual-setting comparison.
The service interval includes request/replies, reduction and residual. One
resident bank per layer differs from the rotating-bank CPU screen.

| Layer / normal two-expert route | Six/three-task paired medians, first / repeat | Three/three controls |
| --- | --- | --- |
| 3 | 0.9267 / 0.9551 | 1.0018 / 0.9904 |
| 4 | 0.9140 / 0.9266 | 1.0079 / 0.9901 |

The routed two-expert latency improves 4.5–8.6%, winning 211/252 pairs. All
8,192 layer checks and complete dumps match the frozen native-Q8 reference
bitwise. Each process serves 2,048 requests: 1,536 batched, 256 sequential and
256 idle. Candidate processes select six tasks for 128 two-expert calls;
controls do not. Miners remain running. This supports extending the opt-in to
two-expert NT1, but supplies no two-expert NT2 or full-model qualification.
Source, settings, hashes, failed-attempt evidence and corrected timings:
`results/20261020/native-two-nt1-worker2-ib-tasks/`.


### Integrated two-expert NT1 route selection

The existing `STRATA_NVFP4_ROUTE_TASKS=1` opt-in now also selects six tasks for
two-expert NT1 split calls. Its measured selection is now:

| Expert count | Token count per job | Selected tasks per participant |
| --- | --- | --- |
| 2 | 1 | 6 |
| 3 | 1 or 2, matching across jobs | 6 |
| Other combinations | Any | Existing configured value |

The previous four-participant, original-layout, H4096/F2048 and format guards
remain. Nondefault per-participant settings and fixed budgets still win; the
base is never mutated. Two-expert NT2 remains excluded because it has not been
measured. Exact-one environment semantics, disabled default, Q2/Q3 behavior and
the layer-dataflow schedule are unchanged.

The direct worker-two IB probe was rebuilt against this policy and now toggles
the policy setter, rather than manually changing the task count. Q2/Q3 policy
is enabled concurrently. The same 128-repeat paired protocol produced:

| Layer / normal two-expert route | Enabled/disabled paired medians, first / repeat | Disabled controls |
| --- | --- | --- |
| 3 | 0.9427 / 0.9362 | 0.9933 / 0.9952 |
| 4 | 0.9310 / 0.9512 | 1.0003 / 1.0082 |

Integrated routed latency improves 4.9–6.9%, winning 209/252 pairs. All 8,192
layer checks and complete dumps match the pre-policy frozen native-Q8 reference,
including disabled calls and unchanged routes. This is partial-weight native-Q8
parity, not full-model quality or exact BF16 W4A16 equivalence. Miners remain
running. No independent service gains are multiplied into full-model tok/s.

Both HIP- and CUDA-configured CPU libraries were rebuilt. All four scoped tests
pass in each: FP4 route policy, Q2/Q3 route policy, native pool and pool tasks.
The updated FP4 test retains 162 selection decisions and 60 changing-route
comparisons per build, now covering the two-expert NT1 selection and two-expert
NT2 fallback. The outputs are finite/nonzero for nonempty routes and bitwise
equal across policy changes. No GPU execution ran; SYCL remains unavailable.
Evidence and current source snapshots:
`results/20261020/native-two-nt1-worker2-ib-policy/`. Full-model throughput and
90%-peak utilization remain unverified.

### Concurrent eight-expert native CPU batch tree

Inspection found that the older tree runner owns one expert per node and groups
multiple tokens for that expert. Its calls cannot exercise the new two/three-
expert route policy. A separate bounded NT1 runner now uses eight distinct real
experts from layers 3/4 with a concurrent 3/3/2 assignment. It preserves the old
runner and its protocol. This new runner uses two direct links with the existing
three-expert batch protocol, so no global expert-ID wire extension is needed:
local IDs map to experts 0–2, 3–5 and 6–7 on the primary and workers respectively.

The completed local checkpoint supplied 336 tensors, 214,983,432 payload bytes.
All 126 tensors shared with the old three-expert fixture have identical recorded
ranges and hashes. The public extractor now accepts `--experts 8` and an optional
`--local-only` flag; its default remains two experts with the original local/
HTTP behavior. Local-only extraction rejects missing shards before any network
request. A fresh extraction through the public tool produced byte-identical
weights, index and provenance to the initial fixture; the duplicate validation
copy was removed. No weights or output dumps are included in shared evidence.

```sh
python3 tools/glm_fp4_fetch_partial.py --source MODEL_DIR \
  --output PARTIAL_EIGHT --experts 8 --local-only
make -f experiment/glm_fp4_local/Makefile cpu-native-batch-tree \
  BUILD=BUILD_DIR NATIVE_BUILD=CPU_BUILD GGML_INCLUDE=GGML_INCLUDE_DIR
```

`glm-fp4-native-batch-tree ROLE MODEL ROOT1 ROOT2 REPEATS POLICY DUMP` runs role
0 on the primary, 1 on worker one, and 2 on worker two. Each host needs the
bounded fixture. ROOT1 and ROOT2 identify the respective worker bootstrap
endpoints. The primary uses IB ports 1/2; worker one uses port 1 and worker two
port 2. DUMP is the primary output path and `-` on workers. Start the workers
before the primary. REPEATS is 2–128. POLICY=1 alternates the worker route policy
on/off by repeat for paired measurement; POLICY=0 is an identical-policy
control, not a different numerical mode. Use matching arguments across roles.
The measured configuration retains offset-dot arithmetic, prefetch 2048,
disabled AVX512/native-local scheduling and small-hidden preservation.

The primary has sixteen compute participants, compact NUMA rows and two
persistent communication lanes. Workers have four compute participants each
and original-layout rows. The primary also loads all eight experts for an
untimed sequential oracle. Ten fixed scenarios cover full/permuted 3/3/2,
each owner idle, all idle, small/zero inputs, sparse 1/1/1 and 2/2/2 routes.
Every expert output, weighted reduction and BF16 residual is compared bitwise;
complete repeats must also match. Service timing includes dispatch, local work,
both remote replies, reduction and residual, but excludes the oracle and startup.
These are fixed routes, not the model's learned router or a token-generation run.

The first smoke run passed 40 layer checks and the initial four-process timing
run passed 5,120. However, the first scratch harness created communication lanes
after pinning the primary host thread, causing both lanes to inherit one CPU.
Its normal full-route latencies were around 5–6 ms, and candidate/control timing
was inconsistent. This is a harness finding, not a claimed regression in the
existing production decoder. Initial evidence remains frozen at
`results/20261020/native-batch-tree/`.

### Communication-lane affinity measurement and corrected runner

A same-binary paired experiment isolated that inherited mask. Before each repeat,
outside timing, both communication threads selected either the primary's original
32-CPU allowed mask or its pinned one-CPU mask. Each thread read back and checked
its mask. Worker route policies remained off in both arms. Four processes used
candidate/control/control/candidate order, 64 repeats and 31 warm pairs per cell.
Controls kept the one-CPU mask. The primary's sixteen compute participants and
worker counts remained unchanged; miners stayed running.

| Normal full route / layer | Wide/narrow lane paired medians, first / repeat |
| --- | --- |
| Ordered / 3 | 0.6434 / 0.7214 |
| Ordered / 4 | 0.6402 / 0.6425 |
| Permuted / 3 | 0.6388 / 0.6566 |
| Permuted / 4 | 0.6482 / 0.6438 |

The wider communication mask reduces these partial-layer latencies 27.9–36.1%,
winning 245/248 pairs. Matching controls range 0.9515–1.0356. All 5,120 layer
checks and dumps match the initial eight-expert reference. Preserve the first
ordered-layer repeat's 0.7214 paired ratio even though its ratio of whole-run
medians differs. This experiment includes two communication threads in addition
to sixteen compute participants; it does not establish greater CPU arithmetic
or memory-bandwidth utilization. Evidence:
`results/20261020/native-batch-tree-affinity/`.

The repository runner `native_cpu_batch_tree.cpp` now creates its communication
threads before pinning the compute host thread, retaining their original allowed
mask without per-request affinity changes. Pending tasks borrow each iteration's
input and routing state; exception handling waits for both tasks before those
buffers leave scope. Communication lanes are destroyed before their link
contexts. None of this changes the older tree runner or decoder defaults.

The corrected runner then repeated the worker route-policy comparison, with
wide communication masks throughout. Normal full-route results were:

| Route / layer | Policy on/off paired medians, first / repeat | Off/off controls |
| --- | --- | --- |
| Ordered / 3 | 0.9833 / 0.9687 | 1.0305 / 1.0066 |
| Ordered / 4 | 0.9697 / 0.9715 | 0.9947 / 1.0022 |
| Permuted / 3 | 0.9495 / 0.9786 | 1.0057 / 1.0064 |
| Permuted / 4 | 0.9357 / 0.9316 | 1.0069 / 1.0084 |

Candidate whole-run medians were 3.213–3.283 ms per partial layer. Paired gains
span 1.7–6.8%, with 182/248 pairs faster; the smaller gains overlap control drift
and are not a universal speedup. Do not multiply these ratios by the independent
affinity experiment. All 5,120 layer checks and dumps match the initial reference.
Each worker serves 1,280 requests per process, including 896 batched and 256 idle
calls. Candidate processes select the route policy for 448 eligible calls per
worker; controls select none.

The new target builds against both HIP- and CUDA-configured CPU libraries. A
CUDA-configured CPU primary also passed a 40-layer three-node smoke run against
the same worker binaries and output reference. No GPU ran; SYCL remains
unavailable. The public extractor's missing-local-shard check made zero network
calls. Final source, build records, validation and timings:
`results/20261020/native-batch-tree-final/`.

These results qualify a two-layer CPU/IB experiment with eight real experts.
They omit attention, KV cache, learned routing, GPU caching and full-model
streaming. No new full-model decode tok/s or 90%-peak utilization claim follows.

### Concurrent completion profile and 4/2/2 assignment screen

The corrected eight-expert runner was instrumented with a single primary clock:
local completion, each worker reply completion, join completion and mixing time.
Each communication lane writes its own timestamp before publishing its future;
the primary reads them only after joining both futures. Completion timestamps
include queueing, transport and remote service, not isolated kernel durations.
Four 64-repeat processes retain the existing policy candidate/control order.
All 5,120 layer checks and dumps match the uninstrumented reference.

For normal ordered full routes with the worker policy enabled, local completion
medians are 0.906–0.984 ms, worker one 3.172–3.256 ms and worker two 2.137–2.183 ms.
Mixing medians are about 0.018–0.020 ms. Across ordered/permuted normal routes,
worker one is last in 247/248 enabled samples; the primary is last in one and
worker two in none. These overlapping completion times must not be summed.
This favors testing a different assignment before optimizing the small mixing
phase. Diagnostic source and samples:
`results/20261020/native-batch-tree-profile/`.

A separate same-binary paired prototype moves real expert 3 from worker one to
the primary, changing 3/3/2 to 4/2/2. The primary uses its existing three-expert
batch plus one sequential expert call; it does not introduce a four-expert kernel.
Worker one requests its local experts 1/2 (global 4/5), while worker two continues
with global experts 6/7. All eight experts still contribute. Both assignments
reduce in canonical global expert-ID order, so placement and request permutation
cannot change floating-point reduction order. This canonical-order prototype is
separate from the older runner's permuted-route reduction tests.

The primary remains at sixteen compute participants, workers at four each, and
communication lanes retain their wide masks. Eligible worker route policies stay
on in both arms. Four scenarios cover ordinary/permuted, small and zero inputs;
64 repeats with candidate/control/control/candidate processes provide 31 warm
pairs per cell. Assignment alternates by repeat in candidate processes; controls
retain 3/3/2. The same input/output buffers and experts are checked across repeats.

| Route / layer | 4/2/2 versus 3/3/2 paired medians, first / repeat | 3/3/2 controls |
| --- | --- | --- |
| Ordered / 3 | 0.7032 / 0.7106 | 1.0094 / 0.9974 |
| Ordered / 4 | 0.6854 / 0.6966 | 0.9950 / 0.9883 |
| Permuted / 3 | 0.6934 / 0.6930 | 1.0235 / 0.9751 |
| Permuted / 4 | 0.7030 / 0.6909 | 0.9812 / 0.9996 |

Normal partial-layer latency improves 28.9–31.5%, winning all 248 candidate pairs.
Candidate whole-run medians are 2.238–2.273 ms versus baseline 3.214–3.261 ms.
All 2,048 layer checks and complete repeated dumps pass bitwise. The ordinary
route's final output also matches the earlier eight-expert reference. The
prototype verifies each expert, full reduction and residual independently
against the sequential oracle; changing placement does not omit expert work.

This is a placement experiment, not free migration: the primary already holds
all eight experts for the oracle, and worker one retains the third expert while
its computation moves. No weight transfer or full-model residency cost is timed.
A full deployment must place the moved weights on the primary; actual router
loads and GPU cache misses may produce different active counts. Miners remain
running. No production assignment/default changed, and no full-model tok/s or
90%-peak claim follows. Evidence:
`results/20261020/native-batch-tree-rebalance/`.

### Checkpoint memory accounting for the placement candidate

A bounded header/index audit read the completed checkpoint's 295,041 tensor
entries without reading their weight payloads. It checked shard membership,
range bounds and non-overlap, and recorded header/index hashes. The checkpoint
contains 12,384 expert groups: 288 experts in each of layers 3–45. Configuration
records 45 main layers and one next-token prediction layer; thus this storage
account includes all 42 main MoE layers plus auxiliary expert layer 45.
All non-expert tensors, 10,820,476,072 bytes, are assigned to the primary in the
following accounting. Static owner selection uses expert ID modulo eight.

| Static owner fraction | Raw tensor GiB, primary / worker one / worker two | One native-packed expert copy GiB, same order |
| --- | --- | --- |
| 3/3/2 | 68.26 / 58.18 / 38.79 | 71.30 / 61.22 / 40.82 |
| 4/2/2 | 87.66 / 38.79 / 38.79 | 91.71 / 40.82 / 40.82 |

The native estimate verifies every expert's three U8 projection shapes, applies
36 bytes per 64 weights plus a 16-byte scale tail, and retains raw sizes for
non-expert tensors. A direct call to the current `native_fmt(40,40,4096,2048)`
confirms 14,155,792 bytes per expert group. This is a single-copy layout estimate,
not an allocation measurement. It excludes retained original expert copies,
NUMA replicas, other conversions/metadata, KV cache, staging, OS and miner memory.
The audit does not rehash the full checkpoint payload.

Against the user-stated 128/64/64 GiB capacities, one native copy leaves roughly
36.29/23.18/23.18 GiB before runtime costs under 4/2/2. Under 3/3/2, worker one
has only 2.78 GiB left by the same accounting. The 4/2/2 fraction means
144/72/72 resident experts per layer, not a guarantee that each token routes
4/2/2 active experts. These results make it a better placement candidate to
validate, without claiming current available RAM or full-model runtime fit.
Reproducible audit code, geometry, format check and accounting are alongside
the rebalance evidence in `memory-audit-packed.json`.

### Five-expert primary placement and completion profile

A further bounded placement experiment compares 4/2/2 against 5/1/2 active
experts using the same eight real experts in layers 3/4. The primary composes
its existing three-expert batch with a two-expert batch. Worker one executes
one expert; worker two retains two. All eight experts still contribute in
canonical global-ID reduction order. Compute participants remain 16/4/4,
communication lanes retain their wide masks, and miners remain running.

Four 64-repeat processes in candidate/control/control/candidate order pass
2,048 layer checks, including ordinary, permuted, small and zero inputs.
Repeated output dumps match bitwise, including the preceding placement
experiment. Normal-input paired latency improves 2.8–4.8%, with 184/248 wins;
control ratios span 0.9734–1.0158, so the smaller improvements are near observed
control drift. Candidate whole-run normal medians are 2.176–2.208 ms. Evidence
and the paired analyzer are in `results/20261020/native-batch-tree-five/`.

The same checkpoint header audit estimates one packed copy plus all non-expert
weights at 112.12/20.41/40.82 GiB for 5/1/2, versus 91.71/40.82/40.82 GiB for
4/2/2. Thus the modest speed improvement costs another 20.41 GiB on the primary,
leaving only 15.88 GiB of its stated 128 GiB before runtime and OS/miner costs.
Under the same assumptions, 6/1/1 requires 132.53 GiB on the primary, exceeding
capacity before those costs. This does not exclude a different offload or
storage plan. Static resident fractions do not guarantee active router counts.
Keep 4/2/2 as the safer planning baseline pending full runtime residency tests.

A separate instrumented candidate/control pair passes another 1,024 layer checks
with identical outputs. All completion timestamps use the primary's clock and
start at dispatch; remote times include queuing, transport and computation.
After excluding the first two repeats, ordinary/permuted cases provide 124
samples per assignment in the candidate process:

| Assignment | Primary completion ms | Worker one completion ms | Worker two completion ms | Last completion counts: primary / one / two |
| --- | --- | --- | --- | --- |
| 4/2/2 | 1.239–1.265 | 2.173–2.222 | 2.118–2.146 | 1 / 89 / 34 |
| 5/1/2 | 1.562–1.763 | 1.127–1.188 | 2.152–2.192 | 15 / 0 / 109 |

Ranges are the four scenario/layer cell medians, not uncertainty intervals.
These overlapping times must not be summed. Worker two becomes the usual
critical completion under 5/1/2, making its two-expert CPU service the next
optimization target. Frozen instrumentation, raw logs and a standalone analyzer
are in `results/20261020/native-batch-tree-five-profile/`.

These tests keep the selected experts resident and do not time weight migration,
full-checkpoint routing, attention, KV cache or GPU caching. They neither revise
full-model decode forecasts nor establish 90% theoretical CPU utilization.

### Worker-two eight-task scheduling trial

Following the completion profile, a bounded test fixes active placement at
5/1/2 and changes only worker two from six to eight native tasks per compute
participant. It retains four participants on each worker and sixteen on the
primary; worker one's eligible policy and primary computation are unchanged.
This is an explicit experimental task count, not a change to the route policy.
Workers select the paired arm from the request sequence, in agreement with the
primary's alternating repeat schedule. Both arms use experts 6/7 on worker two.

Four 64-repeat processes run candidate/control/control/candidate, with 31 warm
pairs per scenario/layer. All 2,048 layer checks pass and complete output dumps
match the preceding placement experiment bitwise. Miners remain running.
For ordinary/permuted inputs, the paired ratios are:

| Timed quantity | Eight versus six tasks | Six-task controls | Candidate wins |
| --- | --- | --- | --- |
| Whole partial-layer service | 0.9782–1.0201 | 0.9820–1.0008 | 129/248 |
| Worker-two completion from dispatch | 0.9775–1.0161 | 0.9806–1.0102 | 131/248 |

Ranges describe eight cell medians, not confidence intervals. Remote completion
includes queueing and IB transfer. The candidate does not consistently improve
either worker-two completion or layer latency relative to control variation.
Retain six tasks; do not enable eight in the opt-in policy. Further work should
inspect the two-expert compute phases and memory access rather than treating
finer task partitioning as a demonstrated gain. No GPU or full-decoder run was
performed, and no peak-utilization claim follows.

Frozen source, build command, raw samples, output hashes and a standalone paired
analyzer are in `results/20261020/native-batch-tree-worker2-eight/`.

### Worker-two compute phases inside the distributed route

Two further 64-repeat processes retain 5/1/2 placement and six worker-two tasks
per participant. They use the pool's existing gate/up, hidden-transform plus
quantization, and down-phase counters. Worker-local complete adapter time is
measured separately. Samples are buffered and printed after all requests, avoiding
per-request log I/O in the timed service. The extra timestamping and buffering
still make this a diagnostic run, not a new performance qualification.

All 1,024 layer checks pass, and complete output hashes equal the preceding
placement experiment. Four participants on each worker, sixteen on the primary,
and miners are unchanged. After excluding two warmup repeats, each of the four
ordinary/permuted layer cells contains 62 samples per process:

| Quantity | Range of eight normal-input cell medians, ms |
| --- | --- |
| Worker-two complete adapter compute | 2.0415–2.0731 |
| Gate/up phase | 1.3687–1.4019 |
| Hidden transform and activation quantization | 0.0062–0.0071 |
| Down phase | 0.6392–0.6826 |
| Adapter time outside those phases | 0.0076–0.0081 |
| Worker-two completion observed from primary dispatch | 2.1414–2.1803 |

The gate/up phase includes pool scheduling and SiLU as well as dot products;
it is not an isolated kernel timing. Its roughly two-thirds share of worker-local
compute identifies the next useful optimization target. Hidden rescaling and
quantization are a small fraction. Primary-observed completion includes dispatch,
queueing and IB transfers; subtracting separate medians would not isolate a
transport latency distribution. No clock synchronization is assumed: each
reported interval is measured entirely on its own host.

This repeated two-layer fixture does not establish full-model bandwidth, routing
behavior, throughput or theoretical utilization. Source/build hashes, raw samples,
output hashes and a standalone analyzer are in
`results/20261020/native-batch-tree-worker2-phases/`.

### Gate/up-only eight-task scheduling trial

The preceding eight-task trial changed both projections. A scratch pool object
now isolates gate/up scheduling: the candidate uses 32 gate/up tasks and 24 down
tasks for worker two's two-expert NT1 FP4 route, versus 24/24 in the baseline.
With 4,096 gate/up output rows, 32 tasks gives 128 rows per task. Arithmetic,
weights, four worker participants, 5/1/2 placement and miners are unchanged.
The scratch pool object links ahead of the existing CPU archive; production
pool source and build outputs remain unchanged.

Candidate/control/control/candidate processes provide 31 warm pairs per cell.
All 2,048 layer checks pass, and output hashes match prior placement tests.
Normal ordered/permuted inputs produce these paired-median ratio ranges:

| Quantity | Candidate/baseline | Unchanged controls | Candidate wins |
| --- | --- | --- | --- |
| Partial-layer service | 0.9824–1.0168 | 0.9939–1.0051 | 123/248 |
| Worker-two completion | 0.9789–1.0131 | 0.9935–1.0058 | 122/248 |
| Gate/up phase | 0.9800–1.0170 | 0.9928–1.0101 | 125/248 |
| Down phase | 0.9745–1.0291 | 0.9816–1.0270 | 135/248 |

No consistent gain justifies adopting this schedule. Retain six tasks for both
phases. The down arithmetic and task count are unchanged; its timing variation
illustrates why small whole-layer changes need controls. Results include the
same buffered phase instrumentation in both arms and exclude warmup only.

The initial scratch compile lacked the pool source's local include directory;
its failed log is preserved, and the successful build manifest adds it. A first
completed run had successful primary/worker exits but was rejected by an incorrect
launcher exit-status assertion. Its logs are preserved under `harness-assert-*`
and excluded from the qualified paired sequence. The assertion was corrected
before starting that sequence. All those processes had exited before restarting.

Frozen scratch source, build commands, raw logs and the standalone paired
analyzer are in `results/20261020/native-batch-tree-worker2-gu8/`. This CPU-only
experiment does not qualify GPU/SYCL execution, full-model tok/s or theoretical
utilization. No production scheduling policy changed.

### Worker-two Zen 2 tuning and inline-assembly dot trial

Two kernel-level trials retain six tasks per participant and 5/1/2 active
placement. Both select the candidate only for worker two's ordinary NT1 gate/up
entry points. Down projection, K32, wider widths, other nodes and arithmetic
remain on the generic baseline. A per-process atomic selector changes between
completed requests; both arms incur the same dispatch wrapper. Each trial runs
candidate/control/control/candidate processes with 64 repeats, giving 31 warm
pairs per scenario/layer. Worker compute participants remain four; miners run.

The first trial compiles the current AVX2 source with GCC `-mtune=znver2` versus
generic GCC. This targets the Ryzen 3950X worker; it does not change the ISA or
enable fast math. The second uses generic GCC for both arms and replaces the
NT1 offset dot's integer intrinsic chain with extended inline assembly:
`vpmaddubsw`, `vpsubw`, then `vpmaddwd`. Its output uses an early-clobber register
constraint; inputs are register operands, with no hidden memory accesses or
condition-code effects. Floating-point scale operations and FMA order stay
unchanged. Raw activations containing -128 still select the existing fallback.
The wrapper selects candidate gate/up only, even though the compiled source
also contains candidate versions of other entry points.

| Trial / quantity | Candidate/baseline paired ratios | Identical-work controls | Candidate wins |
| --- | --- | --- | --- |
| Zen 2 tuning, layer service | 0.9951–1.0162 | 0.9942–1.0183 | 101/248 |
| Zen 2 tuning, gate/up phase | 0.9836–1.0347 | 0.9838–1.0090 | 120/248 |
| Inline assembly, layer service | 0.9794–1.0184 | 0.9958–1.0434 | 123/248 |
| Inline assembly, gate/up phase | 0.9856–1.0251 | 0.9897–1.0167 | 120/248 |

Ranges cover the eight ordinary/permuted layer/run cell medians. Retain the
4.3% assembly control anomaly: these data do not establish a consistent gain
for either candidate. Each trial passes 2,048 layer checks, and all output hashes
match the preceding placement experiment. Neither candidate is integrated.

The assembly helper also passes 16,777,216 exhaustive doubled-E2M1 code-pair and
signed-Q8 activation-pair combinations against scalar integer arithmetic, checking
all eight result lanes. This covers the helper's integer arithmetic, including
-128, but does not replace full kernel scale/fallback tests or full-model quality
qualification. The test source embeds the exact helper from the candidate and
can be built with `g++ -O3 -mavx2 asm_test.cpp -o asm_test`.

Sources, build commands, hashes, raw timing samples and standalone analyzers are
in `results/20261020/native-batch-tree-worker2-zen2/` and
`results/20261020/native-batch-tree-worker2-asm/`. Tests were CPU-only; GPU/SYCL
execution, full-decoder throughput and 90% peak utilization remain unqualified.

### Interleaved integer-chain inline assembly

A second assembly candidate groups the two independent offset-dot halves of each
K64 block in one six-instruction extended-assembly statement: two `vpmaddubsw`,
two `vpsubw`, then two `vpmaddwd`. Both outputs have early-clobber register
constraints; seven inputs remain register operands. This tests enforced
interleaving rather than changing arithmetic or the ordered floating-point FMA
chain. Candidate dispatch remains restricted to worker two's ordinary NT1
gate/up; down, K32, other widths and other nodes use the baseline. There is no
production change. Object disassembly confirms the six-instruction order and
compact object-local excerpts are retained with the evidence.

The standalone integer test exhausts 16,777,216 code/activation-pair combinations
and checks both outputs, with reversed codes in the second chain, across all
eight lanes. It passes, including -128 integer operands. The production candidate
retains the original -128 fallback. Four 64-repeat distributed processes in
candidate/control/control/candidate order pass another 2,048 layer checks; their
complete output hashes match the preceding placement runs. Six scheduling tasks,
four worker participants and 5/1/2 placement remain fixed, with miners running.

| Normal-input quantity | Candidate/baseline ratios | Identical-work controls | Candidate wins |
| --- | --- | --- | --- |
| Partial-layer service | 0.9790–1.0192 | 0.9700–1.0246 | 132/248 |
| Gate/up phase | 0.9822–1.0138 | 0.9800–1.0118 | 117/248 |

Ranges cover eight paired cell medians after warmup; outliers are retained.
There is no consistent improvement beyond control variation. Retain the current
intrinsic implementation. Forcing these integer instructions into a chosen
order does not establish better complete-service performance. A larger assembly
change would need evidence about decoding, register pressure or memory stalls,
rather than assuming this local dependency-chain ordering is the bottleneck.

Sources, commands, hashes, raw samples, disassembly excerpts and the standalone
analyzer are in `results/20261020/native-batch-tree-worker2-asm-pair/`. No GPU,
full-decoder throughput or 90%-peak qualification follows from these CPU tests.

### V100 study refresh: overlap matters beyond the inner loop

A fresh read of `docs/GLM_DFLASH2_V100.md` from the separate experiment checkout
adds qualified results beyond the earlier CPU-only screens. These are reported
measurements on dual Xeon Gold 6240 plus two V100 PCIe 32 GB GPUs, with a 181 GiB
allocation; they were not rerun here.

One-token CPU scale decoding alone measured 36.41/36.84 decode tok/s, missing
the report's 3% gate. Combining it with batched KDA Q/K/V, with RAM execution
in both arms, later measured 36.11/37.46 tok/s with residency checks passed.
Earlier faulting attempts remain unqualified. The successful combined comparison
does not isolate a benefit from running the executable in RAM.

The more relevant scheduling result moves narrow routed input/coefficient
readback before the shared FFN. CPU experts wait on that copy's event while the
shared FFN runs; final summation remains after the shared FFN on its original
stream. With earlier CPU/KDA options held fixed, control/candidate measured
37.78/40.26 decode tok/s, a reported 6.54% gain. Each side used three full warmups
and three timed 512-output requests. All six streams matched the same mild-policy
reference, with zero timed major faults or swap, and both sides passed eight
reused long and eight reused natural-length coding checks. Fresh20 remain
unscored. Both sides used 134 verification rounds and 20,455 CPU expert groups
per timed request. The selected 32.10 tok/s preset and frozen 5.26 tok/s reference
remain unchanged; these results are not interchangeable with exact W4A16 decode.

Source inspection limits direct transfer. The remote scale-local routine uses
AVX-512 VNNI with sparse-CSF32 reconstruction; our current native AVX2 packs
already store decoded scale bytes. The local checkout lacks the remote
`src/program/glm_fp4_native.inc` implementation. Our three-node harness begins
with host-resident inputs and has no shared FFN, so it cannot demonstrate the
readback gain. The current FP4 simulator also explicitly excludes the shared
expert and router. No simulator rate, overlap default or local throughput claim
is changed by this study.

For full-decoder integration, retain this dependency order as a concrete target:
router completes, copy narrow routed input/coefficient buffers, record completion,
launch shared FFN, then let CPU/remote expert work begin as soon as the copy event
completes. Join both results before final summation, preserving buffer lifetime
and the original sum order. Test event ownership, reused buffers and rollback
before timing. Do not insert a whole-stream synchronization that also waits for
the shared FFN. This is an integration target, not an implemented local gain.

Source-report fingerprint, exact reported comparisons, qualification limits and
local applicability are saved in
`results/20261020/v100-overlap-revalidation/review.json`.

### Local decoder overlap audit after the V100 study

Inspection of the local GLM Q2/Q3-capable pipeline finds that the useful ordering
from the V100 early-readback experiment is already present here.
`enqueue_moe_publish` publishes routed IDs, coefficients and activation data
before calling `enqueue_moe_residents`, in both fused-router branches and the
separate mailbox branch. Resident experts and then the shared FFN run on the
GPU. `enqueue_moe_combine` waits for CPU completion afterward. On the host,
`service_pipeline` waits for the published generation and immediately calls
`cpu_moe_mailbox` in the ordinary path, without waiting for the shared FFN.

The mapped mailbox publisher uses system fences before publishing its generation;
the host waits with acquire loads. Source ordering is evidence of intended
overlap, not a fresh hardware concurrency or performance measurement. The core
layer path likewise rings the route doorbell before optional early shared-expert
execution; verification has a late shared-stream fork designed to prioritize
publication. A duplicate early-readback option is not warranted by this audit.

There is an intentional delay in merged speculative groups: group A waits for
group B's publication to batch shared expert-weight reads. Its tradeoff is start
latency versus weight reuse, not an accidental shared-FFN wait. Existing
`STRATA_GLM_STEP_TRACE=2` measures per-slot CPU/wait time for ordinary service,
but the merge guard explicitly rejects tracing. Do not prescribe that switch
as a working merged-group profile without adding compatible instrumentation.
Any such experiment must preserve group completion, generation and cancellation
semantics, and compare complete service rather than only saved weight reads.

The audit and source fingerprints are in
`results/20261020/local-overlap-audit/review.json`. No production code, environment
semantics, default output or simulator rates changed; no new runtime speed or
90%-peak claim follows.

### Primary streaming bandwidth refresh

A current CPU-only refresh uses the existing paired streaming probe on the
Threadripper 1950X, with pinned physical cores and owner-first-touched buffers.
It retains offset dots, 2048-byte prefetch, default pages and 128 packed copies
per participant. Four-core runs use 301,989,888 packed bytes; sixteen-core runs
use 1,207,959,552. Each process has a discarded warmup and nine alternating
read/FP4 samples, checking FP4 outputs and read checksums. Miners remain running.
This is synthetic projection streaming, not complete real-expert service.

| Process order | Physical cores | Read payload GB/s | FP4 payload GB/s | Median paired FP4/read payload-rate ratio |
| --- | --- | --- | --- | --- |
| 0 | 4 | 61.67 | 12.20 | 0.1998 |
| 1 | 16 | 74.72 | 46.57 | 0.6241 |
| 2 | 16 | 73.47 | 70.49 | 0.9594 |
| 3 | 4 | 75.04 | 24.44 | 0.3288 |
| 4, confirmation | 16 | 76.64 | 73.58 | 0.9600 |
| 5, confirmation | 16 | 70.61 | 68.79 | 0.9712 |

The first runs are materially slower; retain them rather than reporting only
later peaks. Two additional sixteen-core processes check the large change seen
between the initial sixteen-core runs. The later three processes reach 95.9–97.1%
of their paired same-buffer read rate, supporting near-read-rate throughput for
this warmed streaming workload at full core count. It does not establish the
same efficiency with four participants, routed experts, other representations,
full-model decoding or the theoretical DRAM interface. CPU use and raw samples
are retained in the logs; no causal attribution to clocks or contention is made.

The SMBIOS memory query requires interactive administrator authentication, and
EDAC exposes no memory controllers here. Current configured RAM rate and channel
population remain unverified, so a 90% theoretical bandwidth claim is still
unsupported. No credentials were requested or system settings changed.

Raw paired samples, settings, source/binary hashes and summaries are in
`results/20261020/primary-bandwidth-refresh/`. The existing probe/build is reused;
no production kernel or default changed.

### Q2/Q3 paired read baseline on the primary

A scratch extension of `q23_test` adds a same-buffer AVX2 checksum-read control
to its existing row-kernel benchmark. Primary CPU 3 is pinned before allocation
and execution. The geometry is K4096/R128, one token, 1,024 distinct weight copies:
176,160,768 bytes for Q2_K and 230,686,720 for Q3_K. Quantization, packing and
reference checks are untimed. Each process has ten alternating read/dot pairs
of 4,096 calls per arm; the first pair is excluded from summaries. Processes run
Q2/Q3/Q3/Q2. Miners remain running; no GPU benchmark executes.

| Format / process | Read payload GB/s | Dot payload GB/s | Median paired dot/read rate |
| --- | --- | --- | --- |
| Q2 / first | 26.93 | 6.18 | 0.2299 |
| Q3 / first | 26.38 | 5.19 | 0.1988 |
| Q3 / repeat | 27.29 | 5.43 | 0.2000 |
| Q2 / repeat | 24.08 | 5.89 | 0.2500 |

These are single-core projection results, not directly comparable to the
sixteen-core FP4 refresh. The 20–25% read-baseline ratios point to unpack/dot work
as a useful one-token Q2/Q3 target; they do not identify a specific instruction
bottleneck or measure theoretical utilization. Value-repeated weight rings do
not represent actual model routing. Both formats retain their original row
layout, with Q23 AVX2 enabled and prefetch disabled.

The launcher recorded an unused `STRATA_Q2_ROW_LUT=0` name. The actual switch is
`STRATA_Q2K_LUT`; its absence was verified in the parent environment inherited
by these processes, selecting the default-off path. The raw settings are
preserved with that clarification rather than silently rewritten. Future
launchers should set the actual switch explicitly.

Every physical copy passes the existing output comparison and initial FP64
reference check. All forty timed/warmup read checksums and final dot-output checks
pass. Source, build manifest, raw samples, effective-setting note and standalone
analyzer are in `results/20261020/q23-read-refresh/`. Production kernels and
settings remain unchanged.

### One-token Q3 integer-chain candidate

Following the single-core read comparison, a scratch Q3 NT1 kernel accumulates
even/odd 32-value chunks separately, then adds their int32 vectors before the
original float conversion and ordered FMA. Q2 and other token widths remain
unchanged. The conservative per-lane integer bound is 524,288, well within int32;
this rearranges exact integer sums rather than floating-point reductions.
Earlier chain screens covered Q2 NT1 and Q3 NT2, not this case.

The baseline/candidate projection screen pins primary CPU 3 before first touch,
uses K4096/R128, 1,024 distinct packed Q3 matrices (230,686,720 bytes), and ten
alternating pairs of 4,096 calls per arm. The first pair is excluded from timing
summaries. Both arms disable prefetch and Q2 LUT explicitly. Four NT1 processes
use candidate/control/control/candidate order; a fifth checks the unchanged NT2
path. Miners remain running.

| Process | Candidate/baseline median latency ratio | Warm wins |
| --- | --- | --- |
| NT1 candidate, first | 0.9719 | 9/9 |
| NT1 identical-code control | 0.9989 | 5/9 |
| NT1 identical-code control | 1.0034 | 4/9 |
| NT1 candidate, repeat | 0.9742 | 7/9 |
| Unchanged NT2 | 1.0013 | 4/9 |

The 2.6–2.8% NT1 projection gain exceeds these controls and warrants a complete
Q2-GU/Q3-down expert-service test. It is not yet an integrated optimization or
full-model gain. These are synthetic quantized weights and activations; no
claim covers all raw scale patterns or full-model quality. Every copy is checked
against the baseline before timing, the original FP64 reference check passes,
and all 100 timed/warmup arm-final output comparisons pass bitwise. This does
not independently check every timed call. The scratch candidate is linked under
renamed symbols; production code and defaults remain unchanged.

Reproducible source, build commands, hashes, raw samples and analyzer are in
`results/20261020/q3-nt1-chains/`. Complete service, worker CPUs, GPU/SYCL builds
and peak-utilization qualification remain outstanding for this candidate.

### Q3 NT1 chain candidate: complete CPU service gate

The promising projection candidate was screened in complete Q2-gate/up and
Q3-down expert service before integration. A same-object selector changes only
NT1 Q3 integer chunk accumulation to the even/odd chains, retaining the original
floating-point order. The primary uses sixteen participants, two dense experts,
sixteen rotating weight banks, owned NUMA dataflow, prefetch 1024 and Q2 LUT off.
Twenty alternating pairs time 64 service calls per arm. Input quantization is
outside timing; pool execution, hidden processing and dataflow are included.
Miners remain running. Inputs and routing are synthetic.

| Process | Candidate/baseline median latency ratio | Paired wins |
| --- | --- | --- |
| Candidate, first | 1.0000 | 10/20 |
| Identical-code control | 0.9969 | 12/20 |
| Identical-code control | 0.9954 | 11/20 |
| Candidate, repeat | 0.9993 | 10/20 |

All 160 final batch output comparisons after 10,240 timed service calls pass
bitwise. These checks occur after batches, not every timed call. The service
result is neutral, despite the isolated 2.6–2.8% projection gain; do not integrate
the candidate or increase model throughput estimates. This is a different
geometry, participant count and prefetch setting from the single-core projection
screen, so it does not identify exactly why the isolated gain disappears.

The existing Q3 kernel and production defaults remain unchanged. Worker/GPU/SYCL
execution, full-model quality/tok/s and peak utilization were not qualified.
Source, build commands, hashes, raw samples and standalone analyzer are in
`results/20261020/q3-nt1-chains-service/`.

### Q3 candidate dispatch coverage and split-phase follow-up

A separate, untimed-for-performance coverage build counts NT1 Q3 row calls at
the candidate selector. In complete two-expert dataflow service it records
344,064 baseline rows and 327,680 candidate rows. The forced-baseline process
records 671,744 baseline rows and zero candidate rows. Both processes pass
output checks. Their per-row atomics invalidate timing interpretation; this
establishes that the earlier neutral service test exercised the changed kernel,
not a bypassed implementation.

Four new split-phase processes use the original uninstrumented service binary,
the same two-expert/NT1/sixteen-bank geometry, sixteen participants and prefetch
1024. They run twenty alternating pairs of 64 calls per arm, in
candidate/control/control/candidate order. All 160 batch-final output checks
after 10,240 timed calls pass bitwise. Coverage processes had completed before
these timings began; their counters are absent from the timed binary.

| Process | Whole-service ratio | Gate/up ratio | Down ratio |
| --- | --- | --- | --- |
| Candidate, first | 0.9629 | 0.9984 | 0.9415 |
| Identical-code control | 0.9824 | 0.9918 | 0.9847 |
| Identical-code control | 1.0031 | 1.0057 | 0.9973 |
| Candidate, repeat | 0.9996 | 0.9926 | 1.0035 |

These are medians of paired candidate/baseline times. The first split-phase
candidate improves, but the repeat does not; the first control also drifts.
Together with the neutral owned-dataflow service screen, this does not justify
integration. Retain all observations rather than choosing the favorable down
phase. No causal memory-bandwidth or clock explanation is established.

Coverage source/logs and split-phase samples are in
`results/20261020/q3-nt1-chains-coverage/` and
`results/20261020/q3-nt1-chains-phases/`. Production code/defaults and full-model
throughput estimates remain unchanged.

### Q2/Q3 primary hardware-counter profile

The primary permits process-local hardware counters (`perf_event_paranoid=1`).
Bounded runs pin CPU 3 and use the existing `q23_test --bench TYPE 1 65536 row
1024` with Q23 AVX2 enabled, Q2 LUT off and prefetch off. Q3 Clang NT2/NT4
options are off. All benchmark reference/output checks pass. No GPU runs.

| Format | User cycles | User instructions | Instructions/cycle |
| --- | --- | --- | --- |
| Q2_K | 6,710,896,277 | 11,571,720,512 | 1.7243 |
| Q3_K | 9,897,097,613 | 19,241,807,795 | 1.9442 |

These paired events ran 100% of their enabled time, without multiplexing. They
cover the whole benchmark process, including initialization, quantization,
replication, validation and warmup, rather than only the internal timed loop.
An initial broad six-event Q3 run was multiplexed at 83%; preserve it as a
separate diagnostic, not an exact capacity measurement. Generic cache events
are not DRAM traffic, and IPC is not useful-FLOPS utilization.

A separate 499 Hz user-cycle sample attributes 96.29% to Q3's ordinary NT1 dot
routine, with zero lost samples. Annotated disassembly shows stack-backed shuffle
mask reloads in the hot loop, alongside packed-code decoding, integer products,
scaling and FMA. One such reload has 3.95% of routine-local sampled period, but
sampling skid and instruction dependencies prohibit a causal stall attribution.
A concrete next hypothesis is reducing live shuffle-mask register pressure with
an explicit memory-operand shuffle, tested against current arithmetic and whole
service. The profile does not prove that this change will help.

Raw counter text, benchmark outputs, source/binary hashes, symbol report and
annotated disassembly are in `results/20261020/q23-counter-profile/`. The raw
binary perf capture remains in ignored local scratch. No source/default or
simulator changes follow from this profile.

### Q3 NT1 shuffle-mask memory operand

The hardware-counter profile motivated a scratch assembly candidate that forces
Q3 NT1 scale shuffles to read their mask as a memory operand: `vpshufb` uses an
`m` input constraint instead of requiring another live vector input. The source
vector and result use vector-register constraints; this single instruction may
reuse its source register safely. There are no hidden memory effects or flag
clobbers. Q2 and other widths retain the intrinsic implementation. Arithmetic,
integer accumulation and floating-point order are unchanged. Object disassembly
confirms stack-addressed shuffle instructions; it does not itself prove a speedup.

On primary CPU 3, the K4096/R128/NT1 screen rotates 1,024 weight copies and times
ten alternating pairs of 4,096 calls, excluding the first pair from summaries.
Candidate/control/control/candidate processes produce NT1 paired latency ratios
0.9479/0.9483 (18/18 candidate wins), with identical-code controls 1.0103/1.0012.
The unchanged NT2 comparison is 0.9991. All 100 timed/warmup arm-final output
checks pass, as do the baseline FP64 and per-copy checks. This is a repeated
roughly 5.2% projection gain on synthetic inputs, not full-model throughput.

A same-object selector then tests complete Q2-GU/Q3-down CPU expert service:
two dense NT1 experts, sixteen rotating banks, sixteen primary participants,
owned NUMA dataflow, prefetch 1024 and Q2 LUT off. Each process has twenty
alternating pairs of 64 calls per arm. The new assembly path is selected only
for Q3 NT1; both arms share the selector overhead.

| Process | Complete-service paired latency ratio | Wins |
| --- | --- | --- |
| Candidate, first | 0.9880 | 17/20 |
| Identical-code control | 1.0013 | 10/20 |
| Identical-code control | 0.9938 | 11/20 |
| Candidate, repeat | 0.9953 | 12/20 |

All 160 batch-final comparisons after 10,240 timed calls pass bitwise. Input
quantization is excluded; pool scheduling, hidden processing and dataflow are
included. Miners remain running. The 0.5–1.2% service gain is modest relative to
the 0.6% control variation and does not justify a default change. Keep this as
an experimental candidate pending stronger service and worker evidence; do not
transfer the 5.2% projection gain into model forecasts.

Source, build commands, raw samples, hashes, disassembly excerpts and analyzers
are in `results/20261020/q3-nt1-mask-memory/` and `q3-nt1-mask-service/`.
Production kernels/defaults are unchanged. Worker/GPU/SYCL execution, full-model
quality/tok/s and theoretical utilization remain unqualified.

### Q3 shuffle-mask candidate: worker service gate

The same assembly candidate was tested on a Ryzen 3950X worker with four CPU
participants and its miner unchanged. Two dense NT1 experts use sixteen rotating
complete weight banks, prefetch 1024, Q2 LUT off, and the same native layer
service. Each candidate/control/control/candidate process runs twenty alternating
pairs of 64 calls per arm. The portable binary hash is checked remotely before
each run; each process has a 90-second deadline and all exit successfully.

| Process | Candidate/baseline service ratio | Wins |
| --- | --- | --- |
| Candidate, first | 1.0093 | 6/20 |
| Identical-code control | 0.9920 | 12/20 |
| Identical-code control | 1.0003 | 10/20 |
| Candidate, repeat | 1.0013 | 10/20 |

All 160 batch-final output checks after 10,240 timed service calls pass bitwise.
The candidate does not improve worker service, and its small primary-service
advantage was already near control variation. Retain the current intrinsic
kernel: a repeatable primary projection gain does not justify portable
integration or a changed default. Neither this screen nor the primary screen
measures full-model quality/tok/s or theoretical CPU utilization.

Worker source/build command, hashes, raw logs and standalone analyzer are added
to `results/20261020/q3-nt1-mask-service/`, alongside the preserved primary
results. No production source or default changed; GPU/SYCL execution was not
performed for this rejected scratch candidate.

### Real FP4 zero-group screen

Before adding sparse execution branches, a bounded scan inspects all 48 packed
matrices in the eight-expert, layers-3/4 partial fixture. It decodes CSF32 scales
through the current shared C11 decoder and interprets adjacent low/high E2M1
nibbles in the same order as the existing projection tests. Both signed zero
codes count as zero. Effective zeros also include zero local scales; global
scales are checked finite and nonzero. This is a weight inventory, not a timing
or model-quality test. Only the existing partial weights are read; no GPU runs.

| Projection | Matrices | Effective zero values | All-zero K16 / K64 groups | All-zero rows |
| --- | --- | --- | --- | --- |
| Gate | 16 | 6.8833% | 0 / 0 | 0 |
| Up | 16 | 6.8858% | 0 / 0 | 0 |
| Down | 16 | 6.8741% | 0 / 0 | 0 |

All decoded local scales are finite nonnegative E4M3 values and none is zero.
Thus whole-group or whole-row zero skipping would avoid no arithmetic on this
fixture, despite individual zero values. Do not add such checks or infer a
model-wide sparsity property from this two-layer subset. Per-value sparse
encoding would need separate storage/dispatch cost evidence and is not selected
by this result. Existing K32 complete-worker evidence was also rechecked: it
does not establish a reliable gain beyond control variation, so that rejected
screen was not repeated.

Reproducible NumPy/ctypes analysis, per-matrix counts, decoder build command and
source/fixture hashes are in `results/20261020/fp4-zero-groups/`. Run with
`python3.12 analyze.py PARTIAL_DIRECTORY DECODER_LIBRARY`. No weight payloads or
built libraries are stored with the shared evidence. Production code and defaults
remain unchanged.

### Real FP4 repeated-scale coverage

A follow-up scan reuses the shared C11 CSF32 decoder and the same 48 real
matrices (layers 3/4, experts 0..7). It counts equal adjacent K16 scale pairs,
equal aligned quartets of K16 scales, and rows whose scales are all identical.
No tensor values, scale bytes or global scales are changed.

| Projection | Equal adjacent pairs | Equal aligned quartets | Uniform-scale rows |
| --- | --- | --- | --- |
| Gate | 12.8259% | 0.4177% | 0 |
| Up | 12.7885% | 0.4150% | 0 |
| Down | 13.4982% | 0.4642% | 0 |

These counts supply potential branch coverage, not execution-time savings.
A uniform-K64 scale fast path would cover less than 0.5% of blocks in this
fixture; a pair path still needs a fallback for most pairs. This does not justify
adding branches to the existing scalar scale-table path without separate timing.
Per-matrix byte histograms are retained for future specialization decisions.
The earlier arithmetic scale-decoder trial was also reviewed: it lost 46/48
pairs under its own settings, so it was not repeated as though untested. That
older result does not prove all alternative scale decoders are slower.

Reproducible analysis, per-matrix counts/histograms, decoder/fixture hashes and
source fingerprints are in `results/20261020/fp4-scale-repeats/`. The decoder
build command is the one recorded with `fp4-zero-groups/`. This is a partial
weight inventory only: no new kernel timing, model-wide distribution claim,
production change or theoretical-utilization result follows.

### Complete FP4 CPU service cycle sampling

A fresh bounded harness links the current CPU library and runs three real experts
from layers 3/4, NT1, sixteen rotating banks and sixteen primary participants
with owned NUMA dataflow. Offset dots and prefetch 2048 are enabled; narrow/NT4
Clang paths and pool spin are off. The old compiler-selector hook is a no-op,
so both benchmark arms execute the current kernel. Input preparation remains
outside service timing. Output and reduction comparisons pass. Miners remain
running and no GPU executes.

User cycles are sampled at 199 Hz with the monotonic clock. The harness marks
service-window begin/end with that same clock; samples outside those windows
are excluded from the following period-weighted attribution. The capture reports
zero lost samples. Layer 3 contributes 5,588 samples and layer 4 contributes
4,603. These are sampled on-CPU cycle shares, not additive wall-time fractions.

| Symbol group | Layer 3 | Layer 4 |
| --- | --- | --- |
| NT1 offset gate/up row routine | 61.75% | 60.64% |
| NVFP4 down-row entry point | 32.50% | 35.13% |
| Pool dataflow task | 1.72% | 1.00% |
| Pool worker | 1.20% | 0.54% |
| Out-of-line exponential | 0.81% | 0.53% |

The two main arithmetic routines account for about 94–96% of sampled periods in
these windows. This supports prioritizing their execution over speculative
changes to pool coordination for this workload. It does not separate decoding,
loads, arithmetic or stalls inside those routines, and does not mean 94–96%
of theoretical FLOPS or bandwidth. Sleeping time is absent from user-cycle
sampling; hardware samples may skid across nearby instructions.

Compact symbol attribution, output checks, source/build hashes and analysis code
are in `results/20261020/fp4-service-counter-profile/`. Raw perf data and per-sample
addresses remain in ignored local scratch. Reproduce the sample extraction with
`perf script --ns -F time,period,ip,sym`; the `ip` field is needed here for symbol
resolution. Feed that output to `analyze.py SAMPLES_FILE`. No production kernel,
default or full-model throughput estimate changed.

### FP4 whole-K64 unpack on the limiting worker

A new scratch candidate decodes both 16-byte packed halves of a K64 block with
one 256-bit load, two lane-local byte shuffles, two 64-bit unpacks and two
128-bit lane selections. It restores the original GGML element order and retains
all scale, integer-dot, FMA and reduction operations. This differs from the
older rejected wide-unpack screen, which widened each half separately. K32 and
raw -128 fallback paths remain unchanged. Only worker two's ordinary NT1 gate/up
selects the candidate; down projection and other nodes retain the baseline.

An independent scalar mapping check passes 8,192 byte-position cases, verifying
all 64 output bytes per case with varied neighboring bytes. The first generation
attempt stopped at an assertion before compilation because the searched unpack
statement also appears in the signed path; the generator was corrected to scope
the change to the offset overload. No incorrect candidate was executed.

The real eight-expert, two-layer distributed test retains 5/1/2 placement,
16/4/4 compute participants and six tasks on worker two. Four 64-repeat processes
run candidate/control/control/candidate with miners unchanged. All 2,048 layer
checks pass, and complete output hashes match the preceding placement tests.

| Normal-input quantity | Candidate/baseline paired ratios | Identical-work controls | Candidate wins |
| --- | --- | --- | --- |
| Partial-layer service | 0.9667–1.0071 | 0.9754–1.0104 | 132/248 |
| Worker-two gate/up phase | 0.9650–1.0049 | 0.9837–1.0136 | 152/248 |

These ranges cover eight scenario/layer/run cell medians after warmup. Some
cells improve, but neither layer service nor gate/up improves consistently;
control variation overlaps much of the observed effect. Do not integrate this
candidate or revise full-model forecasts. The existing packed layout and kernel
remain selected. Source, commands, hashes, scalar checker, raw samples and paired
analyzer are in `results/20261020/native-batch-tree-worker2-block-unpack/`.
No GPU/SYCL execution, model-wide quality or 90%-peak qualification follows.

### Shared gate/up activation loads on worker two

The earlier paired offset gate/up helper, rejected against the primary's existing
offset path, was tested separately on the Ryzen 3950X worker. It shares each Q8
activation half, activation scale and offset correction between gate and up,
retaining separate accumulators and the original FMA sequence for each output.
The raw -128 case keeps the existing signed fallback. Only worker two's ordinary
NT1 gate/up path selects the candidate; down and other nodes are unchanged.

Four 64-repeat processes retain 5/1/2 assignment, 16/4/4 participants, six worker
tasks, prefetch 2048 and active miners. Candidate/control/control/candidate order
provides 248 normal-input warm pairs across ordinary/permuted routes and layers.
All 2,048 layer checks pass; complete output hashes match previous placement runs.

| Quantity | Candidate/baseline paired ratios | Identical-code controls | Candidate wins |
| --- | --- | --- | --- |
| Partial-layer service | 0.9921–1.0113 | 0.9893–1.0203 | 114/248 |
| Worker-two gate/up phase | 0.9704–1.0216 | 0.9886–1.0112 | 135/248 |

There is no consistent service improvement. Retain the existing offset kernel
on this worker as well as the primary. Do not turn shared activation loads into
an assumed service gain or multiply this screen by earlier placement gains.
The initial scratch generation inserted the helper before its type definitions;
compilation failed, the helper placement was corrected and the error log is
preserved. No failed binary was executed.

Sources, build commands, hashes, raw samples and analyzer are in
`results/20261020/native-batch-tree-worker2-gu-pair/`. Production code/defaults
remain unchanged. This bounded CPU test does not qualify GPU/SYCL execution,
full-model quality/tok/s or theoretical utilization.

### Worker-two L3 placement screen and reverse-order confirmation

On the Ryzen 3950X second worker, the default three pinned pool workers use
CPUs 1/2/3 in one L3 group. A scratch runner initializes the pool with the
allowed mask 0/4/8/12, yielding pinned workers 4/8/12 across three other L3
groups, then immediately restores the participating host thread's original
32-CPU allowed mask. It does not reserve those cores from other processes.
Both arms retain four compute participants on each worker, sixteen on the
primary, six worker-two tasks and fixed 5/1/2 active expert placement. Miners
continue running. Kernel arithmetic and production defaults are unchanged.

The first four-process order is spread/control/control/spread. A second run
uses control/spread/spread/control with the identical executable. Each process
runs 64 repeats across four input scenarios and two real layers. Each screen
passes 2,048 layer checks; all eight output dumps match bitwise, including the
prior five-expert-placement experiment. Ordinary/permuted timing excludes the
first two repeats, leaving 62 samples per scenario/layer/process.

| Screen | Control cell medians ms | Spread cell medians ms | Spread/control matched-cell ratio |
| --- | --- | --- | --- |
| Initial | 2.168–2.225 | 2.092–2.172 | 0.9571–0.9798 |
| Reverse order | 2.161–2.217 | 2.111–2.164 | 0.9674–0.9847 |

Ratios divide the mean of the two spread process medians by the mean of the
two control process medians for each matching scenario/layer. They are
process-level comparisons, not within-process paired measurements or confidence
intervals. Both screens favor spreading, but uncontrolled miner activity and
host migration remain confounders. This supports a topology-aware opt-in
candidate; it does not justify changing the default or imply full-model tok/s.
The active-route placement and resident partial weights also limit extrapolation.

Sources, topology, build fingerprints, raw text logs and relative-path analyzers
are frozen in `results/20261020/native-batch-tree-worker2-spread/` and
`results/20261020/native-batch-tree-worker2-spread-confirm/`. Connection settings,
executables and raw output arrays remain outside shared evidence. GPU execution,
full-model quality/throughput and 90% theoretical utilization remain unqualified.

### Spreading both workers under 4/2/2 regresses service

The next screen holds active expert assignment at 4/2/2 and applies the previous
L3-spreading initialization to both Ryzen 3950X workers. Both log pinned workers
4/8/12 in the candidate versus 1/2/3 in control, with participating host masks
restored to 32 CPUs. Primary participation stays sixteen, workers four each;
miners remain running. Worker one retains its existing route-task policy and
worker two retains its explicit six-task budget. Only affinity and the fixed
assignment differ from the preceding scratch harness.

Four 64-repeat processes in spread/control/control/spread order pass 2,048
layer checks. All output dumps match the prior placement experiments bitwise.
Excluding the first two repeats, and comparing means of matching process cell
medians as above:

| Completion | Spread/control ratio across four ordinary/permuted layer cells |
| --- | --- |
| Total layer service | 1.0542–1.0973 |
| Worker one | 1.0632–1.1114 |
| Worker two | 0.9793–0.9874 |

Worker two retains a small improvement, but worker one becomes slower and
controls total service. Reject applying this placement to both workers. Identical
CPU models do not establish identical optimal placement under different miner
loads; this test does not isolate cache effects from contention or host-thread
migration. The next useful affinity comparison should keep worker one unchanged,
and any deployment choice needs independent per-node evidence. This negative
result also limits the scope of the earlier 5/1/2 affinity recommendation.

Source, build fingerprints, raw text logs, output hashes and analysis are in
`results/20261020/native-batch-tree-422-spread/`. Production defaults and kernels
are unchanged; no GPU, full-model throughput or theoretical-peak claim follows.

### Worker-two-only affinity does not improve 4/2/2 total service

A follow-up isolates the affinity change to worker two, retaining worker one's
pinned CPUs 1/2/3 in all processes. Worker two uses 4/8/12 only in the candidate;
both participating host threads retain their original 32-CPU masks. Other
settings match the preceding 4/2/2 screen, including 16/4/4 participants and
running miners. Four processes in candidate/control/control/candidate order
pass another 2,048 layer checks with bitwise-identical output dumps versus
prior placement experiments. The same 62-sample cell medians and process-level
ratio calculation apply.

| Completion | Candidate/control matched-cell ratio |
| --- | --- |
| Total layer service | 0.9965–1.0063 |
| Unchanged worker one | 0.9962–1.0019 |
| Worker two | 0.9778–0.9899 |

Worker two becomes faster but remains off the critical completion path.
Worker-one matched process-average cell medians are about 2.198–2.212 ms,
versus worker two's candidate 2.098–2.113 ms. Total service is effectively flat,
so this does not qualify an affinity throughput improvement under 4/2/2.
Combined with the preceding regression when spreading worker one, it closes
the current simple-spread experiment: retain existing defaults and prioritize
worker-one service or a measured change to workload assignment. Do not multiply
the worker-two gain into a full-model throughput forecast.

Evidence is frozen at
`results/20261020/native-batch-tree-422-worker2-spread/`. The `spread` placement
log field echoes the process policy even on unchanged worker one; the explicit
`pinned` CPU list is the authoritative placement. Sources and output hashes are
included; connection settings and binaries are excluded. No full-model quality,
GPU execution or 90% theoretical utilization was qualified.

### Worker-one phase profile under 4/2/2

After worker-two-only affinity failed to reduce total service, two bounded
baseline processes measure worker one's local CPU phases. The fixed 4/2/2
assignment, default pinned worker CPUs 1/2/3, restored host masks, 16/4/4 compute
participants and running miners match the preceding control. Phase counters
are read around the two-expert adapter call; text is buffered until service ends.
Both 64-repeat processes pass 1,024 layer checks in total, with output hashes
identical to earlier placement experiments.

Across eight ordinary/permuted scenario/layer/process cells, excluding the
first two repeats (62 samples per cell):

| Metric | Cell median range, ms |
| --- | --- |
| Worker-one complete local adapter call | 2.122–2.168 |
| Gate/up phase, including pool execution | 1.406–1.451 |
| Hidden activation quantization | 0.00634–0.00672 |
| Down phase, including pool execution | 0.676–0.696 |
| Adapter remainder after measured pool phases | 0.00795–0.00906 |
| Worker-one completion observed by primary | 2.220–2.267 |

The gate/up phase is the main local target, accounting for roughly two thirds
of compute. Hidden quantization is too small to explain the bottleneck.
Phase timers include scheduling and synchronization; they are not isolated
instruction-throughput measurements. Primary completion includes dispatch,
transport and remote execution, with a different start point, so subtracting
these independent medians does not measure network latency. Ranges are cell
medians, not confidence intervals. Instrumentation is a diagnostic and does
not qualify an optimization or full-model decode rate.

Source, build fingerprints, phase logs and a relative-path analyzer are in
`results/20261020/native-batch-tree-422-worker1-phases/`. Both processes use
baseline settings; there is no candidate/control speedup claim. Production
kernels/defaults remain unchanged, and no GPU or theoretical-peak validation ran.

### Worker-one gate/up task refinement does not improve service

The worker-one phase profile motivates a scratch trial increasing only its
two-expert NT1 gate/up task count from 24 to 32. Down remains at 24 tasks in
both arms, as does worker two's task budget. The scratch pool object is linked
ahead of the existing CPU archive; production sources are unchanged. Assignment
is fixed at 4/2/2 with sixteen primary and four participants per worker,
original affinity, offset dots, prefetch 2048 and running miners.

Candidate processes alternate baseline/candidate repeat pairs; separate control
processes keep the baseline in both slots. Four 64-repeat processes in
candidate/control/control/candidate order pass 2,048 layer checks. All output
dumps match earlier placement experiments bitwise. After excluding the first
two repeats, normal/permuted inputs provide 248 pairs per process class.

| Metric | Candidate paired-median ratio range | Candidate wins | Unchanged-control ratio range |
| --- | --- | --- | --- |
| Total layer service | 0.9764–1.0110 | 123/248 | 0.9756–1.0100 |
| Worker-one completion | 0.9713–1.0183 | 117/248 | 0.9784–1.0141 |
| Gate/up phase | 0.9528–1.0146 | 121/248 | 0.9773–1.0038 |
| Down phase | 0.9690–1.0279 | 129/248 | 0.9622–1.0324 |

There is no repeatable total-service benefit beyond control variation. Retain
the existing task count. This extends the earlier negative worker-two trial
to the actual worker-one bottleneck under 4/2/2; it does not establish that all
possible partitions are equivalent. Gate/up remains the dominant phase, but
this particular finer split does not resolve it. No production change, GPU
execution, full-model throughput or theoretical-utilization claim follows.
Evidence, source snapshots, build fingerprints and the paired analyzer are in
`results/20261020/native-batch-tree-worker1-gu8/`.

### Worker-one interleaved inline-assembly dot screen

The earlier two-chain assembly candidate is also screened on worker one, the
measured 4/2/2 bottleneck. Its source matches the earlier worker-two candidate;
the baseline matches current `nvfp4_avx2.cpp`. The candidate interleaves two
independent `vpmaddubsw`/`vpsubw`/`vpmaddwd` chains while preserving floating-point
FMA order. Object inspection finds twenty instances of that six-instruction
sequence. The prior exhaustive integer-helper check remains applicable to the
identical source. Runtime dispatch enables the candidate only for worker-one
NT1 ordinary-layout gate/up; down and K32 paths use baseline, as do other nodes.
The compiler-control override is explicitly zero in the recorded environment.

Four 64-repeat processes use candidate/control/control/candidate order, with
alternating baseline/candidate repeat pairs inside candidate processes and
unchanged baseline slots in controls. Compute participation remains 16/4/4,
all worker phases retain 24 tasks, active assignment is 4/2/2, affinity stays
unchanged and miners continue. All 2,048 layer checks pass; output dumps match
prior placement runs bitwise. Normal/permuted inputs yield 248 pairs per class
after excluding the first two repeats.

| Metric | Candidate paired-median ratio range | Wins | Control ratio range |
| --- | --- | --- | --- |
| Total service | 0.9835–1.0298 | 118/248 | 1.0006–1.0312 |
| Worker-one completion | 0.9837–1.0192 | 120/248 | 1.0014–1.0259 |
| Gate/up phase | 0.9838–1.0195 | 115/248 | 0.9794–1.0196 |

This does not establish a repeatable improvement. Reject integration on this
worker as well; explicit instruction interleaving alone does not resolve the
measured gate/up bottleneck. Control variation precludes interpreting isolated
faster cells as a qualified gain. No production defaults or arithmetic change.
Evidence and the paired analyzer are frozen in
`results/20261020/native-batch-tree-worker1-asm-pair/`. GPU execution, full-model
quality/tok/s and theoretical-peak utilization remain unqualified.

### V100 refresh: component improvements failed the model gate

A later read of the Xeon/V100 report resolves the previously queued ordered-KDA
model test. Its ordered-four rollback component had measured 46.34 microseconds
versus serial-128's 116.93, with 180 bitwise component comparisons. The matched
model test now reports 40.51/40.48 decode and 520.45/518.71 prefill tok/s
(control/candidate), using three full warmups and three timed 512-output requests
per arm. All six output streams match the mild reference and timed major faults
and swap remain zero. Coding panels were skipped after the absence of a model
gain. The selected profiles remain unchanged. Static compiler reports of fewer
registers and eliminated spills do not establish a runtime throughput benefit.

The report also adds a private FP4 kernel prototype that omits reductions for
unused rows. It passes 96 bitwise cases, 36 graph replays, CUDA memcheck and
racecheck, but its geometric-mean speed ratio is 0.999901 across eight expert
shape/width cases. Each variant uses five alternating 128-call samples with
identical storage. It fails the report's 3% component gate; no model timing or
coding evaluation follows, and the original kernel remains selected.

Neither result justifies a new local port or forecast increase. The earlier
CPU/GPU overlap result remains more relevant to end-to-end performance, but the
local host-input partial-expert harness omits GPU shared FFN and cannot measure
that benefit. Existing local source inspection already found early CPU work
publication in the Q2/Q3 decoder. The next architectural investigation must
measure actual full-layer dependencies rather than multiply isolated component
gains. These are reported remote results, not independently rerun measurements.
A sanitized summary and report fingerprint are saved in
`results/20261020/v100-component-model-refresh/review.json`; the raw remote
report stays in ignored scratch storage. No simulator defaults or local
throughput/utilization claims change.

### Matched-request critical-path headroom under 4/2/2

A new offline analyzer joins each primary completion record to its worker-one
phase record by repeat/scenario/layer, rather than subtracting independent
medians. The existing two instrumented baseline processes supply 496
ordinary/permuted samples after warmup. Worker one finishes last in 386,
worker two in 109, and the primary in one. This analysis runs no new workload.

Across eight scenario/layer/process cells, median worker-one elapsed time
outside its measured local adapter call is 0.0879–0.0959 ms. This is the
primary-observed worker completion duration minus that same request's remote
compute duration. It includes dispatch, request transfer, response transfer
and other overhead; it is not an isolated network latency measurement. Local
join time after the last recorded completion is 0.0180–0.0200 ms, and post-join
reduction/residual work is 0.0186–0.0198 ms. Logging precision limits these
small differences. Durations are compared; remote absolute clocks are not.

A counterfactual reduces only worker-one compute, holding worker-two and
primary completion plus all later work fixed. Reductions of 10%, 25% and 100%
all yield the same cell-median critical-path saving of 0.0722–0.1526 ms: worker
two becomes the limiter. These are diagnostic bounds under fixed-other-work
assumptions, not predicted hardware speedups. Real changes can alter contention,
scheduling and transfer timing. They do not model full-checkpoint routing or
the omitted GPU work.

This limits the value of further worker-one-only microtuning: a substantial
service improvement needs a kernel gain on both workers or a different workload
assignment, while respecting the primary RAM constraint. The earlier 5/1/2
tradeoff remains relevant but costs about 20.41 GiB more primary packed-weight
storage. No scheduling, kernel or throughput default is changed by this audit.
The reproducible analyzer and computed cells are in
`results/20261020/critical-path-sensitivity/`, referencing the frozen worker-one
phase logs directly. Full-model decode and 90% theoretical utilization remain
unverified.

### Both-worker FP4 NTA gate/up prefetch improves 4/2/2 service

A new scratch candidate changes only the offset NT1 dot loop's weight-prefetch
hint from T0 to NTA, retaining the 2,048-byte distance and every arithmetic
operation. Runtime dispatch selects that candidate only for ordinary-layout
NT1 gate/up on both workers; down, K32 and primary kernels remain baseline.
This is a hardware hint, not evidence of a particular cache placement.
Object inspection confirms `prefetchnta` in the candidate. No production
source/default change is included in this screen.

Both workers keep four compute participants, 24 tasks per phase and original
affinity. Primary participation remains sixteen, assignment 4/2/2, and miners
continue running. The first four-process order is candidate/control/control/
candidate; a second run reverses it using the identical binary. Candidate
processes alternate T0/NTA repeat pairs, while controls select T0 in both slots.
Each screen passes 2,048 layer checks, 4,096 combined. Output dumps match all
preceding placement experiments bitwise. After warmup, ordinary/permuted cases
supply 248 pairs per process class per screen.

| Screen/metric | Candidate paired-median ratio range | Wins/248 | Control ratio range |
| --- | --- | --- | --- |
| Initial total service | 0.7768–0.8083 | 247 | 0.9819–1.0031 |
| Initial worker-one gate/up | 0.6827–0.7570 | 247 | 0.9727–1.0062 |
| Initial worker-two gate/up | 0.6813–0.7068 | 248 | 0.9689–1.0255 |
| Reverse-order total service | 0.7799–0.8170 | 238 | 0.9726–1.0156 |
| Reverse-order worker-one gate/up | 0.7047–0.7596 | 246 | 0.9628–1.0139 |
| Reverse-order worker-two gate/up | 0.6789–0.6929 | 248 | 0.9790–1.0185 |

Reverse-order candidate service cell medians are 1.821–1.855 ms. The gain on
both workers reduces the critical path, unlike earlier worker-one-only changes.
This supports a narrowly opt-in candidate for further validation. It does not
justify replacing all prefetch hints: primary Zen1, Q2/Q3, other distances,
verification widths, down projection, larger resident expert sets and full-model
quality/throughput have not been qualified by this experiment. Miners and
background clocks remain uncontrolled, although the effect exceeds the control
variation in both runs. No theoretical FLOPS/BW utilization claim follows.

Sources, fingerprints, raw logs and paired analyzers are frozen at
`results/20261020/native-batch-tree-workers-nta/` and
`results/20261020/native-batch-tree-workers-nta-confirm/`. Confirmation reuses the
initial binary hash and source/build record. The next integration gate is an
exact-one opt-in for this measured path, default-output parity and backend
build/test coverage, followed by a larger-working-set service check.

### Opt-in NTA integration and parity checks

`STRATA_NVFP4_GU_NTA=1` now selects an NTA-specialized offset dot loop for
ordinary-layout NT1 gate/up. It uses the existing C11 exact-one environment
helper; unset, zero and two retain T0. `STRATA_NVFP4_OFFSET_DOT=1` is also
required. Narrow mode takes precedence; signed/raw-minus-128 fallback, K32,
down and other token widths retain their previous paths. The existing
`STRATA_NVFP4_PREFETCH` controls distance, including disabling prefetch at zero.
The switch is per process, not automatic CPU detection: current performance
evidence supports the two Ryzen worker processes at distance 2048 only. Do not
enable it globally on unmeasured primary CPUs from this result.

The kernel chooses the hint specialization outside the block loop and leaves
arithmetic unchanged. The expanded dot test checks ordinary combined and
separate gate/up weights against independent GGML-dot/activation references,
clamped/unclamped paths, partial row guards, NT1 and wider calls, several K
sizes and activation scale patterns, including raw -128 fallback. CTest runs
NTA values 0/1/2 with offset mode enabled. Both HIP and CUDA configurations build
the touched CPU library/test targets and pass all six targeted tests each.
These are CPU executions in those configurations, not GPU kernel tests. The
SYCL compiler is unavailable, so that backend remains unverified.

An integrated-library 4/2/2 test then runs four baseline/NTA/NTA/baseline
processes with unchanged 16/4/4 participation and miners running. All 2,048
layer checks pass and output hashes match the pre-integration experiment,
including baseline runs. Ordinary/permuted service cell medians are
2.254–2.303 ms for T0 and 1.791–1.852 ms for NTA. This is a process-level
integration confirmation, not another within-process paired experiment.
Original release output has not been independently rebuilt here; parity is
against the same-day pre-integration fixture evidence and GGML references.

The default remains T0. Evidence and source/build hashes are in
`results/20261020/nta-integration/`. Larger working sets, primary CPU performance,
full-model quality/throughput and 90% theoretical utilization still require
validation. No Q2/Q3 prefetch policy changes in this integration.

### NTA benefit survives eight rotating worker weight banks

A larger-working-set integration test retains the real layers-3/4 fixture but
constructs eight independent packed copies of each worker expert. Each pair of
layer requests advances to the next bank. All banks contain identical numerical
weights at different addresses; the primary and oracle remain unchanged. The
two active experts per layer occupy 452,985,344 bytes (about 432 MiB) across
banks on each worker. Worker one also loads its unused third expert as in the
previous harness, for 679,478,016 total packed bytes; worker two loads
452,985,344. Preparation is outside timed service. This tests reuse distance,
not varied full-model routes or different weight distributions.

Four baseline/NTA/NTA/baseline processes keep the same 4/2/2 assignment,
16/4/4 compute participation, original affinity and running miners. All 2,048
layer checks pass and output hashes match the integrated single-bank evidence.
For ordinary/permuted inputs, excluding the first two repeats leaves 62 samples
per scenario/layer/process. Comparing the means of the two process medians
within each matching scenario/layer gives:

| Metric | T0 matched cell average ms | NTA matched cell average ms | NTA/T0 range |
| --- | --- | --- | --- |
| Total service | 2.314–2.351 | 1.873–1.896 | 0.7965–0.8195 |
| Worker one completion | 2.270–2.296 | 1.803–1.826 | 0.7873–0.8033 |
| Worker two completion | 2.133–2.162 | 1.647–1.682 | 0.7687–0.7854 |

The roughly 18–20% service reduction survives this larger rotating set. These
are process-level comparisons, not within-process pairs or confidence intervals.
The fixture remains a bounded CPU test with host-resident activations and a
fixed selected-expert set. It does not establish full-model decode throughput,
quality, GPU overlap, primary-CPU benefit or theoretical utilization. The
production NTA switch stays opt-in; Q2/Q3 and down-projection policies remain
unchanged. Evidence, source/build hashes and a reproducible analyzer are in
`results/20261020/native-batch-tree-nta-banks8/`.

### Down-projection NTA adds a further partial-service gain

A separate scratch candidate tests NTA for both workers' ordinary-layout NT1
down projection, keeping integrated gate/up NTA enabled in both arms. Baseline
and candidate use current source; only the offset row-loop hint differs. The
runtime dispatcher always uses baseline gate/up (which selects integrated GU
NTA) and selects the candidate only for down. Primary, K32 and other widths stay
unchanged. Distance is 2048 bytes, assignment 4/2/2, participation 16/4/4,
original worker affinity and miners remain active.

Two four-process screens run candidate/control/control/candidate and then
control/candidate/candidate/control with the identical binary. Candidate
processes alternate down T0/NTA repeat pairs; controls keep T0 in both slots.
Each screen passes 2,048 layer checks, 4,096 total, with output dumps matching
the integrated GU-only experiment bitwise. Ordinary/permuted inputs provide
248 pairs per process class after excluding the first two repeats.

| Screen/metric | Candidate paired-median ratio range | Wins/248 | Control range |
| --- | --- | --- | --- |
| Initial service | 0.9013–0.9458 | 208 | 0.9603–1.0342 |
| Initial worker-one down | 0.7588–0.8176 | 246 | 0.9683–1.0417 |
| Initial worker-two down | 0.7893–0.8049 | 248 | 0.9658–1.0191 |
| Reverse-order service | 0.8860–0.9566 | 193 | 0.9616–1.0077 |
| Reverse-order worker-one down | 0.7722–0.8067 | 245 | 0.9674–1.0016 |
| Reverse-order worker-two down | 0.7811–0.8274 | 247 | 0.9801–1.0265 |

Reverse-order candidate service cell medians are 1.634–1.744 ms. Down-phase
improvements repeat strongly; total-service gains vary because other work can
become limiting. The smaller service gains approach control variation and should
not be presented as a uniform speedup. These comparisons already include GU
NTA in both arms; do not multiply them with independent GU screen medians.
This supports a separate down opt-in candidate, followed by integrated parity,
backend builds and larger-working-set checks. Production down behavior is not
changed in this screen. No full-model throughput or theoretical-utilization
claim follows.

Evidence and paired analyzers are frozen in
`results/20261020/native-batch-tree-down-nta/` and
`results/20261020/native-batch-tree-down-nta-confirm/`. Confirmation uses the
initial binary and source/build hashes. GPU execution, primary-CPU benefit and
Q2/Q3 cache-policy changes remain outside scope.

### Separate down NTA opt-in and rotating-bank integration

`STRATA_NVFP4_DOWN_NTA=1` now selects the NTA specialization in the ordinary
standalone row entry point used by down projection. It uses the existing C11
exact-one helper independently of `STRATA_NVFP4_GU_NTA`; unset/0/2 retain T0.
It requires offset mode and NT1, and narrow mode still takes precedence. K32,
wider token calls and raw-minus-128 fallback retain previous implementations.
Because the switch controls the standalone row API, a direct caller of that API
also receives the opted-in hint; it is not restricted by tensor names.

Ordinary full/partial row tests now compare against GGML's independent dot
reference, including output guards, several K sizes, token widths, activation
scales and raw -128 inputs. New CTest cases exercise down values 0/1/2 with GU
NTA enabled. The touched CPU library/test targets build in HIP and CUDA
configurations; all nine targeted tests pass in each. These are CPU executions,
not GPU tests. SYCL remains unverified because its compiler is unavailable.

The integrated three-node runner then uses eight rotating banks (about 432 MiB
of active packed data per worker) with GU NTA enabled in both arms. Down is T0
in baseline and NTA in candidate. Four baseline/candidate/candidate/baseline
processes retain 4/2/2 assignment, 16/4/4 participation, original affinity and
running miners. All 2,048 checks pass and outputs match the earlier rotating-bank
experiment bitwise. Matched ordinary/permuted scenario/layer averages of process
medians give 1.782–1.812 ms baseline and 1.640–1.670 ms candidate, ratios
0.9146–0.9270. This is a further roughly 7–9% reduction beyond GU NTA; it is a
process-level comparison rather than within-process paired timing.

For the measured Ryzen worker processes only, the tested opt-ins are:

```sh
STRATA_NVFP4_OFFSET_DOT=1
STRATA_NVFP4_PREFETCH=2048
STRATA_NVFP4_GU_NTA=1
STRATA_NVFP4_DOWN_NTA=1
```

T0 remains the default. Primary-CPU performance, GPU execution, full-model
quality/tok/s and theoretical utilization are not established by these results.
The Q2/Q3 path is unchanged. Evidence and source/build hashes are in
`results/20261020/native-batch-tree-down-nta-banks8/`. Default-output parity is
against same-day pre-integration fixture hashes and GGML references; this does
not claim a separately rebuilt last-release comparison.

### Q2/Q3 NT1 service NTA prefetch screen

Following the FP4 result, a scratch same-object Q2/Q3 hint selector compares NTA
with T0 at 1,024 bytes. It reuses the earlier T1/T0 experiment's specialization
outside dot loops, changing only the candidate hint. Arithmetic, activation
buffers and task scheduling remain identical; controls select T0 in both slots.
This is not an inference that the CPU honors a particular cache placement.

The workload is complete synthetic NT1/E2 service, Q2 gate/up and Q3 down,
H4096/F2048, with sixteen rotating banks totaling 291,504,128 packed bytes.
Input quantization is excluded, while dispatch/dataflow are included. The
Threadripper primary uses sixteen participants and owned NUMA rows; the Ryzen
3950X worker uses four participants and complete blobs. Q2 LUT and alternate
Clang Q3 kernels are disabled. Miners continue; clocks/load are uncontrolled.
Each host runs candidate/control/control/candidate processes with twenty
alternating pairs of 64 calls per arm.

| Host | Candidate NTA/T0 paired median ratios | Wins | T0/T0 controls |
| --- | --- | --- | --- |
| Primary | 0.9758 / 0.9554 | 11/20, 18/20 | 1.0059 / 1.0073 |
| Worker one | 0.7853 / 0.7571 | 20/20, 20/20 | 0.9917 / 0.9847 |

The worker result is strong; the primary result is smaller and less consistent.
All eight processes pass, including 320 timed-batch final-output checks covering
20,480 calls. Checks compare batch-final hashes, not every timed call. This
supports further NTA qualification on the other worker and verification widths;
it does not yet justify a Q2/Q3 production or default change. It is not a
network-service or full-model throughput measurement and provides no 90%-peak
qualification. Sources, build fingerprints, raw logs and a relative-path
analyzer are in `results/20261020/q23-prefetch-nta/`. No GPU/SYCL execution ran.

### Q2/Q3 NTA confirmation on both workers and mixed verification routes

The same scratch binary is next tested on worker two for NT1/E2 and on both
workers for NT2/E7 mixed routes. Each host retains four participants, sixteen
rotating weight banks, 1,024-byte prefetch distance and active mining. NT2
routes four experts per token: expert zero is shared by both tokens, while
three additional experts are unique to each token. This exercises NT1 and NT2
jobs in one call. Its packed rotating set is 1,020,264,448 bytes; the NT1 set
remains 291,504,128 bytes. Input quantization stays outside timing.

Each workload runs candidate/control/control/candidate processes, twenty
alternating pairs of 64 calls per arm. Candidate NTA and baseline T0 share one
object and identical dataflow; controls select T0 in both slots.

| Workload | NTA/T0 paired median ratios | Candidate wins | T0/T0 controls |
| --- | --- | --- | --- |
| Worker two NT1/E2 | 0.7563 / 0.7625 | 40/40 | 1.0150 / 0.9936 |
| Worker one NT2/E7 mixed | 0.8481 / 0.8144 | 40/40 | 1.0121 / 1.0004 |
| Worker two NT2/E7 mixed | 0.8596 / 0.8608 | 40/40 | 1.0011 / 1.0008 |

All twelve processes pass, adding 480 timed-batch final-output checks covering
30,720 calls. This confirms worker NTA gains beyond the original NT1 screen,
including the roughly 1 GB mixed-route working set. It supports a restricted
opt-in for ordinary Q2/Q3 NT1/NT2 dot paths, followed by integrated parity and
backend validation. LUT, expanded Q3, wider token calls and alternate compiler
kernels have not been performance-qualified here. Production Q2/Q3 behavior is
still unchanged. These synthetic service measurements do not establish network
service, full-model throughput, quality or 90% theoretical utilization.

Raw logs, matched-pair analyzers and original-binary references are in
`results/20261020/q23-nta-worker2-nt1/`,
`results/20261020/q23-nta-worker1-nt2/` and
`results/20261020/q23-nta-worker2-nt2/`. Source/build snapshots remain at
`results/20261020/q23-prefetch-nta/`. No GPU/SYCL execution ran.

### Restricted Q2/Q3 NTA opt-in integrated

`STRATA_Q23_NTA=1` now enables NTA for ordinary NT1/NT2 Q2/Q3 dot kernels when
`STRATA_Q23_PREFETCH` is positive. It uses the shared C11 exact-one helper;
unset/0/2 retain T0. Selection happens outside block loops, with separate
compile-time hint specializations and unchanged arithmetic. Q2 LUT, expanded
Q3, wider token calls and alternate Clang objects retain their existing policy.
The measured worker distance is 1024 bytes; no default distance or hint changes.

Both HIP and CUDA configurations build the touched CPU library, including its
alternate Clang object, and pass six targeted Q2/Q3 tests each. Eight additional
`q23_test --dump` processes show identical output hashes across unset/0/1/2 in
both configurations. The test compares scalar-reference dots, row bounds and
extreme/zero activations at widths 1–4. The first build rejected a GCC-specific
hint enum cast under Clang; a compile-time boolean fixed that portability error
before successful builds and execution. SYCL is unavailable; no GPU tests ran.

An integrated-library worker service confirmation retains NT1/E2, sixteen banks,
four participants and active mining. Its old per-pair selector is deliberately
a no-op: NTA is selected once per process in candidate/control/control/candidate
order. Median service is 1.167/1.173 ms for NTA and 1.481/1.488 ms for T0; the
ratio of process-median means is 0.7881. All 160 timed-batch final checks across
10,240 calls match the pre-integration output hash. This is a process-level
confirmation, not within-process paired timing.

The opt-in is now available for further runtime testing, but remains off by
default. Primary performance is less strongly established, and full-model
quality/tok/s or 90% theoretical utilization is not inferred from these service
measurements. Default-output parity is against same-day pre-integration evidence
and independent references, not a separately rebuilt last release. Evidence,
source/build hashes and the standalone service analyzer are in
`results/20261020/q23-nta-integration/`.

### Primary FP4 NTA process-level screen is inconclusive

The integrated GU and down switches are tested together on the Threadripper
1950X using three real experts from layers 3/4, NT1, sixteen rotating banks,
sixteen participants and NUMA-owned flow. The audited balanced physical-core
placement is retained. Each layer's bank set holds 679,478,016 packed bytes;
input quantization is outside timing. The CPU-service harness's legacy pair
selector is a no-op, so both within-process slots share the selected environment.
The four-process order is NTA/T0/T0/NTA at distance 2048. Mining is unchanged;
background clocks/load are not controlled.

| Layer | First NTA median ms | T0 medians ms | Second NTA median ms | Ratio of process-median means |
| --- | --- | --- | --- | --- |
| 3 | 0.5557 | 0.5719 / 0.5896 | 0.6524 | 1.0402 |
| 4 | 0.5670 | 0.5714 / 0.5731 | 0.5588 | 0.9837 |

The slower second candidate on layer 3 is retained. This does not establish a
consistent primary benefit and does not extend the worker recommendation to
Zen1. A same-process paired screen would be needed to distinguish a small hint
effect from process variation. No primary default changes. All 336 timed-batch
final checks pass across 20,992 timed calls; each layer's output hash is identical
across processes. This is not per-call validation, GPU execution, full-model
throughput or a theoretical-bandwidth measurement.

Sources, exact settings, raw logs, build fingerprints and the relative-path
analyzer are in `results/20261020/fp4-primary-nta/`. The existing opt-ins remain
available but their demonstrated performance benefit is on the Ryzen workers.

### Paired primary FP4 NTA follow-up

A same-process T0/NTA comparison follows the inconclusive integrated primary
screen. The harness now dispatches between separately compiled baseline and
NTA dot objects, with identical arithmetic, selection overhead, three real
experts, sixteen rotating banks and NUMA-owned sixteen-participant flow.
Gate/up and down switch together at distance 2048. The existing frozen
`native-batch-tree-down-nta` objects supply the kernels; production NTA switches
are explicitly zero so they do not override the comparison. T0/T0 processes
measure control drift. Process order is candidate/control/control/candidate.

| Layer | Candidate paired median ratios | Wins | T0/T0 controls |
| --- | --- | --- | --- |
| 3 | 1.0135 / 0.9644 | 9/20, 14/20 | 1.0175 / 0.9973 |
| 4 | 0.9702 / 0.9676 | 17/20, 19/20 | 0.9993 / 0.9997 |

Layer four shows a small repeated improvement; layer three remains uneven.
This is much smaller than the Ryzen-worker result and does not justify changing
the primary default. All 336 timed-batch final checks across 20,992 calls pass;
layer hashes match the prior integrated primary screen. Neither per-call
correctness, full-model throughput nor theoretical utilization is established.
No new production change is made. Sources, object provenance, raw logs and the
paired analyzer are frozen in `results/20261020/fp4-primary-nta-paired/`.

### FP4 NTA distance screen retains 2048 bytes

With both integrated GU/down NTA switches enabled on the Ryzen workers, two
process-level screens compare 1024 and 4096 byte prefetch distances against
2048. The primary retains T0 at 2048. Each screen uses eight rotating worker
banks, fixed 4/2/2 assignment, 16/4/4 participation, original affinity and active
miners. Worker distance is explicitly set by the scratch runner before any
kernel call; the launcher's recorded global distance remains the primary value.

Each four-process order is baseline/candidate/candidate/baseline. All 2,048
layer checks per screen pass (4,096 combined), with output hashes matching the
previous integrated bank test. Ordinary/permuted cell medians exclude the first
two repeats. Ratios below compare means of matching process medians, not paired
requests or confidence intervals.

| Candidate distance | Candidate/2048 total-service ratio range |
| --- | --- |
| 1024 bytes | 1.0122–1.0372 |
| 4096 bytes | 0.9983–1.0127 |

Neither distance supplies a service gain. Retain 2048 for the measured worker
NTA configuration; this limited screen does not prove a global optimum across
all distances or workloads. Both workers generally finish later at 1024, while
4096 is effectively flat to slightly worse. No production/default setting
changes. Evidence and reproducible analyzers are in
`results/20261020/native-batch-tree-nta-distance1024/` and
`results/20261020/native-batch-tree-nta-distance4096/`. Full-model throughput,
GPU execution and theoretical utilization remain unqualified.

### Launcher support for worker-only FP4 NTA

`tools/glm_fp4_tree.py --worker-nvfp4-nta` now applies the tested CPU settings
only to the two remote worker commands. It selects AVX2 ordinary offset dots,
disables the narrow path, uses prefetch 2048 and enables GU/down NTA. It leaves
the primary process environment unchanged and records the requested worker
environment in the run summary. The option is for updated native CPU binaries;
it does not make a GPU or legacy binary use this CPU path. `--native-k32` is
rejected with this option because that layout is outside its measured scope.

The default command construction and summary fields remain unchanged when the
option is absent. Seven launcher tests pass without SSH, GPUs or model data,
including remote environment quoting/placement, preservation of the primary's
inherited setting, absence of new default environment fields, request evidence
checks and failure cleanup. These are launcher control-flow tests, not an
additional live three-node benchmark; measured performance remains the native
partial-layer evidence above. No system-wide environment or saved deployment
configuration was changed.

### Live direct-launcher NTA validation

The new worker-only option is exercised through the real public launcher with
freshly linked native CPU tree binaries, copied to temporary worker paths with
SHA-256 verification. Baseline and NTA runs both use direct IB links, sixteen
primary/four worker participants, persistent dispatch, registered staging,
reused native workspaces, compact primary NUMA rows, small-value preservation
and extended cases. Miners remain running. The primary NTA environment is
explicitly zero in both runs. A temporary worker wrapper logs effective GU/down
NTA values: both workers report 0/0 for baseline and 1/1 for the opted-in run.

Each arm passes all 48 extended cases, with zero reported relative error,
bit-exact repeats and zero exits from all three nodes. The two 3,293,184-byte
output dumps are byte-identical, hash
`3e8790bfe14117cae5d5d266a61b93afce8ddbedd1b7e952d80be6e3ff242f29`.
This covers ordinary and wider-token cases through actual process launch and
transport, adding 96 case checks. It is a correctness/environment-propagation
qualification, not a timed decode benchmark; setup and oracle work are included
in process elapsed time. No GPU execution or full-model quality test ran.

Compact raw text logs, summaries and build/source fingerprints are frozen at
`results/20261020/tree-nta-launcher/`. Connection configuration, executables and
raw output arrays remain in ignored scratch storage. No persistent worker
configuration or primary default was changed.

### FP4 NT2 signed-path NTA screen

The existing integrated FP4 NTA switches affect NT1 offset dots only. A new
scratch candidate tests NTA in ordinary NT2 signed-dot gate/up and down paths,
including the existing paired gate/up helper where selected. Baseline and
candidate retain identical arithmetic and differ only in prefetch hints; runtime
dispatch selects the candidate only when `nt==2`. K32 and other widths remain
baseline. Both arms use batched execution, six tasks per participant, four CPU
participants, sixteen rotating banks of three real experts and distance 2048.
The host thread remains unpinned, and miners continue on both Ryzen workers.

Each worker runs candidate/control/control/candidate processes over layers 3/4
and normal, small and zero input amplitudes. Each process has eight alternating
pairs of sixteen calls per cell; the first pair is excluded from timing ratios.
All 768 timed-batch checks pass across 12,288 calls. Every timed batch's final
three outputs match a separately evaluated baseline reference. This checks
batch-final outputs, not every timed call.

| Worker, normal inputs | Candidate paired-median ratio range | Wins | T0/T0 controls |
| --- | --- | --- | --- |
| One | 0.8618–0.8940 | 28/28 | 0.9702–1.0081 |
| Two | 0.9191–0.9560 | 28/28 | 0.9873–1.0085 |

This supports a distinct NT2 candidate rather than silently broadening the
NT1 switches. The effect is smaller on worker two but consistent across the
retained normal-input pairs. A wider-path opt-in needs integrated parity,
backend builds and validation before recommendation. No production behavior,
full-model throughput or theoretical-utilization claim changes here. The test
is a CPU adapter service, not network service or a speculative acceptance test.
Sources, fingerprints, settings, raw logs and a relative-path analyzer are in
`results/20261020/fp4-nt2-nta/`. No GPU or SYCL execution ran.

### Separate exact-NT2 FP4 NTA opt-in integrated

`STRATA_NVFP4_NT2_NTA=1` now enables NTA for ordinary calls with exactly two
tokens, independently of the existing NT1 GU/down switches. It uses the shared
C11 exact-one helper; unset/0/2 retain T0. Separate compile-time specializations
cover signed row dots and the existing two-token paired gate/up helper without
changing arithmetic order. Selection occurs at ordinary public entry points,
so K32 and two-token slices of wider requests retain existing behavior. The
normal prefetch distance setting still applies; measured distance is 2048.

HIP and CUDA CPU configurations build the touched library/tests and pass all
12 targeted tests each. The new values 0/1/2 are checked with the existing
GGML-reference suite covering ordinary full/partial rows, combined/separate GU,
clamping, token widths 1–32, multiple K values and activation edge patterns.
These are CPU tests; GPU execution and SYCL remain unverified (no SYCL compiler).

A separate integrated-library confirmation runs both Ryzen workers with three
real experts, NT2, sixteen banks, four participants and six tasks per
participant. Candidate/control/control/candidate processes select the new
environment flag; the old within-process selector is a no-op. All 768
batch-final checks across 12,288 calls pass against baseline references,
including normal, small and zero inputs. Normal-input matched averages of
process medians are:

| Worker/layer | NTA ms | T0 ms | Ratio |
| --- | --- | --- | --- |
| One / 3 | 3.139 | 3.483 | 0.9013 |
| One / 4 | 3.213 | 3.487 | 0.9213 |
| Two / 3 | 3.066 | 3.371 | 0.9097 |
| Two / 4 | 3.084 | 3.357 | 0.9187 |

This confirms roughly 8–10% lower local service latency in the integrated
path, with process-level rather than within-process pairing. The switch remains
off by default and is not silently added to `--worker-nvfp4-nta`, whose current
profile is NT1 only. Wider-model speculative throughput/quality, primary benefit
and 90% theoretical utilization are not established. Evidence and build/source
hashes are in `results/20261020/fp4-nt2-nta-integrated/`.

### Post-NTA effective worker projection rates

An offline analyzer revisits both paired down-NTA screens, selecting only
candidate processes and the NTA-active requests after warmup. It joins no
independent medians: each request's recorded phase durations determine its
rates and fractions before cell medians are taken. Two workers, four candidate
processes and four normal/permuted layer cells yield 32 cells of 31 samples.
The geometry is E2/H4096/F2048/NT1 with both GU and down NTA active.

| Metric | Worker one cell-median range | Worker two cell-median range |
| --- | --- | --- |
| Gate/up phase | 0.964–1.099 ms | 0.923–0.979 ms |
| Down phase | 0.477–0.500 ms | 0.467–0.484 ms |
| Gate/up fraction of local compute | 66.1–67.5% | 65.5–67.2% |
| Combined projection effective rate | 31.39–34.47 Gweight evaluations/s | 34.34–35.98 Gweight evaluations/s |
| Logical packed payload per projection time | 17.66–19.39 GB/s | 19.31–20.24 GB/s |
| Hidden quantization | 0.00434–0.00478 ms | 0.00423–0.00450 ms |
| Remaining local adapter time | 0.00543–0.00649 ms | 0.00487–0.00544 ms |

The rate numerator counts two gate/up matrices and one down matrix for each
selected expert. The payload proxy uses 36 bytes per 64 weights and omits
scale tails and activation traffic. It is not measured DRAM bandwidth: caches,
prefetch overfetch and memory-controller transactions are not counted. Phase
times include scheduling and GU activation work. Neither rate establishes a
fraction of theoretical FLOPS or RAM peak. Input quantization is in adapter
remainder; the separate quantization phase is hidden-only.

This preserves gate/up as the main compute target after the cache-hint gains;
quantization/remainder are too small to explain the remaining millisecond-scale
cost. No simulator default is updated, since its separate cost terms and worker
scheduling do not match this instrumented service directly. The calculation,
per-cell results and source-log hashes are in
`results/20261020/fp4-nta-phase-rates/`. No new hardware run or production change
is made by this analysis.

### Shared NT1 gate/up loads revisited with NTA

The earlier T0 shared-load candidate is revisited after NTA integration. A
scratch overload for `NtaOffsetActs` shares each activation half, activation
scale and offset correction between gate and up, retaining separate accumulators
and each output's original FMA order. Both arms use NTA on GU/down; only the
candidate GU implementation changes. Raw-minus-128 inputs retain the previous
signed helper fallback. Dispatch limits the change to ordinary NT1 GU on both
workers; primary, down and other widths/layouts remain baseline.

The fixture uses eight rotating banks, 4/2/2 active placement, 16/4/4 compute
participation, original affinity, distance 2048 and active miners. Two screens
run candidate/control/control/candidate and the reverse order with the same
binary. Both pass 2,048 layer checks (4,096 combined); output dumps match the
integrated NTA evidence bitwise. Normal/permuted cells supply 248 paired requests
per class per screen after warmup.

| Screen/metric | Candidate paired-median range | Wins/248 | Control range |
| --- | --- | --- | --- |
| Initial service | 0.9414–1.0004 | 153 | 0.9774–1.0263 |
| Initial worker-one GU | 0.8943–0.9818 | 167 | 0.9487–1.0302 |
| Initial worker-two GU | 0.9358–0.9652 | 165 | 0.9777–1.0124 |
| Reverse-order service | 0.9490–0.9870 | 165 | 0.9589–1.0317 |
| Reverse-order worker-one GU | 0.9257–0.9769 | 176 | 0.9809–1.0469 |
| Reverse-order worker-two GU | 0.9383–0.9909 | 172 | 0.9639–1.0089 |

Unlike the earlier T0 trial, both screens now favor the candidate in most pairs,
and GU cell medians improve on both workers. However, much of the 1–5% repeated
service effect overlaps control variation. Retain this as a candidate for a
more isolated confirmation, not a demonstrated uniform throughput gain or an
immediate default change. No production kernel was changed by this screen.
Independent NTA and sharing gains must not be multiplied into model tok/s.
Evidence and analyzers are in `results/20261020/native-batch-tree-nta-gu-pair/`
and `results/20261020/native-batch-tree-nta-gu-pair-confirm/`. Full-model quality,
GPU execution and theoretical utilization remain unqualified.

### Isolated shared-load confirmation limits the recommendation

A local service follow-up removes IB and primary scheduling from the shared-load
comparison. It reuses the exact candidate/baseline kernel objects from the
NTA GU-pair screen, with three real experts, NT1, sixteen rotating banks,
four participants and six tasks per participant on each Ryzen worker. Both
arms execute the same batched adapter and NTA policy. Candidate/control/control/
candidate processes run normal, small and zero inputs on layers 3/4, with
sixteen alternating pairs of sixteen calls. The first pair is omitted from
ratios. Miners remain active.

| Worker/metric, normal inputs | Candidate paired-median range | Wins/60 | Control range |
| --- | --- | --- | --- |
| One service | 0.9696–1.0187 | 33 | 0.9871–1.0205 |
| One gate/up | 0.9678–1.0134 | 35 | 0.9618–1.0481 |
| Two service | 0.9490–0.9913 | 41 | 0.9812–1.0141 |
| Two gate/up | 0.8772–0.9930 | 41 | 0.9619–1.0100 |

Worker one does not show a repeatable benefit beyond control variation. Worker
two remains somewhat favorable, but that does not qualify a common worker
policy, especially when worker one limits 4/2/2 completion. Do not integrate
this sharing candidate on both workers from the existing evidence. The earlier
network results remain frozen and are qualified by this weaker local result.
All 1,536 timed-batch checks pass across 24,576 calls; each batch-final output
is compared against the baseline expert reference. No production/default change,
GPU execution, full-model throughput or theoretical-utilization claim follows.
Evidence and kernel-object provenance are in
`results/20261020/fp4-nta-gu-pair-service/`.

### Halving NT1 NTA GU prefetch requests does not help

A scratch candidate issues the offset NTA GU prefetch only on even block
indices, instead of every 36-byte block. Both workers retain NTA GU/down at
2048 bytes; down, primary and arithmetic are unchanged. The trial uses eight
rotating banks, 4/2/2 assignment, 16/4/4 participants, original affinity and
active miners. Candidate/control/control/candidate processes alternate repeat
pairs in candidate runs and retain baseline in both control slots.

This is not pure duplicate removal. For K4096, the loop has 64 blocks; the
candidate issues 32 requests with a 72-byte stride. At representative
16-byte-aligned row offsets, the baseline addresses 36–37 distinct cache lines
and the candidate 32, omitting 4–5 lines. This is address geometry, not measured
cache traffic or proof that prefetch requests are honored.

All 2,048 layer checks pass with output hashes matching integrated NTA evidence.
Normal/permuted cases provide 248 pairs per class after warmup:

| Metric | Candidate paired-median range | Wins/248 | Control range |
| --- | --- | --- | --- |
| Total service | 0.9682–1.0318 | 125 | 0.9824–1.0356 |
| Worker-one GU | 0.9588–1.0247 | 123 | 0.9868–1.0727 |
| Worker-two GU | 1.0053–1.0695 | 89 | 0.9720–1.0311 |

Reject this frequency reduction: service gains do not separate from control
variation and worker-two GU regresses. Retain current request frequency. No
production/default, model-throughput or theoretical-utilization change follows.
Source snapshots, coverage geometry, logs and paired analysis are in
`results/20261020/native-batch-tree-nta-prefetch-half/`.

### Cache-line-deduplicated NT1 NTA GU prefetch does not help

The `20261020/native-batch-tree-nta-prefetch-lines` experiment follows the
half-frequency test with a coverage-preserving candidate. It issues a prefetch
only when the target enters a different 64-byte cache line, retaining the first
request. For K4096 and 36-byte blocks this reduces 64 requests to 36–37;
`coverage.json` verifies identical target-line sets for all 64 byte alignments.
This is an address-coverage check, not measured hardware memory traffic.

The existing three-node CPU harness uses two real layers, eight experts, fixed
4/2/2 placement, 16/4/4 participants, eight rotating worker weight banks,
2048-byte lookahead, and NTA GU/down in both arms. Only worker NT1 GU request
selection changes. Miners continue running. Four 64-repeat processes use
candidate/control/control/candidate ordering and within-process alternation.
All 2,048 layer checks pass, with output hashes matching the preceding NTA test.
Normal and permuted scenarios give 248 pairs per comparison after warmup.

| Candidate / baseline | Median ratio range | Candidate wins | Control ratio range |
|---|---:|---:|---:|
| Distributed service | 1.0035–1.0503 | 86/248 | 0.9830–1.0322 |
| Worker one GU | 1.0441–1.0885 | 59/248 | 0.9967–1.0344 |
| Worker two GU | 1.0137–1.0746 | 67/248 | 0.9654–1.0199 |

Reject this candidate and retain the existing prefetch frequency. The added
address comparison and conditional request do not improve this service test;
the timings do not isolate which instruction or cache effect causes the loss.
Production code and defaults are unchanged. This is CPU partial-weight evidence,
not GPU validation, full-model decode throughput, or theoretical peak efficiency.

### Removing worker GU NTA prefetch regresses service

`20261020/native-batch-tree-nta-gu-no-prefetch` isolates the contribution of
software prefetch in NT1 offset gate/up. The candidate removes NTA requests in
that loop; the dispatcher selects it only for ordinary worker NT1 GU calls.
Down retains baseline NTA, and primary execution is unchanged. The compiler can
eliminate the candidate prefetch condition because its NTA template argument is
constant. This is not a hardware-prefetch disable test.

The preceding coverage-preserving experiment's harness is retained: two real
layers, eight experts, 4/2/2 placement, 16/4/4 CPU participants, eight worker
weight banks, 2048-byte baseline lookahead, miners running. Four 64-repeat
processes use candidate/control/control/candidate ordering with within-process
alternation. All 2,048 layer checks pass and output hashes match the preceding
NTA test. Normal and permuted scenarios provide 248 pairs after warmup.

| No GU prefetch / baseline NTA | Median ratio range | Candidate wins | Control ratio range |
|---|---:|---:|---:|
| Distributed service | 1.3204–1.4043 | 10/248 | 0.9781–1.0289 |
| Worker one GU | 1.4666–1.5622 | 1/248 | 0.9741–1.0457 |
| Worker two GU | 1.4723–1.5556 | 0/248 | 0.9490–1.0133 |

Retain NTA software prefetch for this workload. Together with the half-frequency
and cache-line-deduplication tests, this result argues against further request
removal as the next optimization. Arithmetic and scheduling remain candidates;
these measurements do not isolate DRAM traffic or prove theoretical efficiency.
No production defaults changed, and no GPU or full-model decode was measured.

### Moving offset correction after integer reduction gives no service gain

`20261020/native-batch-tree-nta-correction32` tests precomputing 32-bit activation
corrections, then subtracting after `vpmaddwd` rather than before it. The original
16-bit correction remains for other helpers, so candidate activation storage
increases by 4096 bytes. Preparation is included in service timing. Integer bounds
in `arithmetic.json` justify the distributive transformation without saturation;
the raw -128 fallback and floating-point FMA order are unchanged.

Only worker ordinary NT1 GU uses the candidate. Down and primary retain baseline
kernels. The three-node test retains two real layers, eight experts, 4/2/2
placement, 16/4/4 participants, eight rotating worker banks, 2048-byte NTA GU/down
prefetch, and running miners. Four 64-repeat processes alternate candidate and
baseline within candidate/control/control/candidate runs. All 2048 layer checks
pass; output hashes match the preceding NTA test. Normal and permuted scenarios
supply 248 pairs after warmup.

| Candidate / baseline | Median ratio range | Candidate wins | Control ratio range |
|---|---:|---:|---:|
| Distributed service | 0.9773–1.0281 | 114/248 | 0.9671–1.0153 |
| Worker one GU | 0.9797–1.0188 | 110/248 | 0.9655–1.0189 |
| Worker two GU | 0.9877–1.0060 | 121/248 | 0.9804–1.0058 |

Do not integrate: the candidate does not show a repeatable service benefit over
control variation. This test does not isolate preparation cost from inner-loop
scheduling. Production and defaults remain unchanged. No GPU execution,
full-model throughput, or theoretical efficiency claim follows from this test.

### Q2/Q3 NTA six-task scheduling screen

`20261020/q23-nta-tasks6` revisits task size with integrated NTA enabled in both
arms. Synthetic Q2 gate/up and Q3 down use H4096/F2048, one token, two experts,
16 rotating weight banks, four CPU participants on each worker, and 1024-byte
prefetch. Complete expert service includes hidden quantization and reduction;
input quantization is excluded. Miners continue running. Three tasks per
participant is the baseline; six is the candidate. No kernel arithmetic changes.

Each worker runs candidate/control/control/candidate processes. Each process
has twenty alternating pairs with 64 timed calls per arm and a warm call before
each batch. Controls keep three tasks in both slots. All 320 batch checks pass
across 20,480 timed calls, with identical output hashes across both workers.

| Worker | Candidate process median ratios | Candidate pair wins | Control ratios |
|---|---:|---:|---:|
| One | 0.9930 / 0.9766 | 25/40 | 0.9863 / 1.0038 |
| Two | 0.9890 / 0.9933 | 24/40 | 1.0051 / 0.9956 |

Retain the existing task policy. The small improvement and modest pair win
counts do not establish a reliable benefit over timing variation. Production
and defaults are unchanged. This is synthetic CPU service evidence, not a
full-model or GPU measurement and not proof of theoretical efficiency.

### V100 grouped-expert report refresh

`20261020/v100-grouping-model-refresh/review.json` records a fresh read of the
remote `docs/GLM_DFLASH2_V100.md`, identified by its content hash. These are
reported remote results, not independently rerun local measurements.

Resident GPU expert grouping reached 1.535068x component speed and a matched
39.96/41.33 decode tok/s comparison. The 3.43% paired improvement did not reach
3% above the historical 40.26 reference: that threshold is 41.4678 tok/s.
Coding panels were skipped and the prototype was not selected. Coalesced FP16
operand stores reached 1.665286x component speed but only 39.53/39.82 decode
and 519.19/520.63 prefill tok/s. The prefill gate failed. Both model comparisons
used three full warmups and three timed 512-output requests per arm, matching
all six output streams with zero timed major faults or swap. The main checkout
source search did not establish the private prototype implementation.

The local CPU adapter already shares activation quantization and submits two
or three experts together. Inspection found a remaining grouping boundary:
the four-expert primary branch in the distributed harness executes three plus
one. A single four-expert submission is a concrete next candidate, but it is
not yet measured, and faster primary work may not shorten the worker-limited
critical path. No GPU component ratio is applied to local CPU throughput,
simulator capacity, or the theoretical-efficiency target.

### One four-expert primary batch improves local completion only

`20261020/native-batch-tree-nta-primary4` and its `-confirm` follow-up test one
four-expert primary CPU submission against three plus one. A scratch adapter
extends the batch count to four, sharing input quantization and a pool submission.
Expert output order, per-expert hidden scaling, arithmetic, and worker kernels
are unchanged. The primary uses sixteen participants and compact NUMA weights;
each worker uses four participants, integrated NTA, and eight rotating banks.
Placement stays 4/2/2, with two real layers and eight experts; miners run throughout.

Each experiment has four 64-repeat processes and within-process alternation.
The initial process order is candidate/control/control/candidate; confirmation
reverses it using the identical binary. Normal and permuted scenarios yield
248 paired samples per comparison after warmup. All 4096 layer checks across
both experiments pass, with bitwise output hashes matching preceding tests.

| Run / metric | Candidate ratio range | Candidate wins | Control ratio range |
|---|---:|---:|---:|
| Initial primary completion | 0.9247–0.9840 | 167/248 | 0.9843–1.0153 |
| Confirm primary completion | 0.9294–0.9768 | 181/248 | 0.9915–1.0301 |
| Initial distributed service | 0.9584–1.0353 | 130/248 | 0.9710–1.0267 |
| Confirm distributed service | 0.9870–1.0142 | 125/248 | 0.9942–1.0413 |

Retain the scratch candidate for broader validation: primary completion improves,
but the workers still determine service latency, so this is not a demonstrated
distributed or model throughput gain. The primary has one weight bank and runs
an untimed oracle between requests; this does not establish a cold full-model
working-set benefit. Four-expert wider-token validation and production integration
remain outstanding. No defaults changed; no GPU execution or peak-efficiency
claim is supported.

### Four-expert primary batching with eight rotating banks

`20261020/native-batch-tree-nta-primary4-banks8` extends the preceding primary
batch comparison to eight primary banks, retaining eight worker banks and the
same 4/2/2 placement, 16/4/4 participants, NTA worker settings, and running miners.
The primary retains eight experts in each of two real layers for the oracle;
service executes four. Loaded logical projection payload is 1,811,941,376 bytes;
active service payload across banks is 905,970,688 bytes. These are logical
payload sizes, not measured DRAM traffic. Banks contain identical copied weights.
The untimed oracle now indexes the selected bank, and candidate/baseline
alternation balances each bank across arms over four repeats.

Four 64-repeat processes use candidate/control/control/candidate order and
within-process alternation. All 2048 layer checks pass with bitwise output hashes
matching the single-bank experiment. Normal and permuted scenarios supply 248
pairs after warmup.

| Metric | Candidate ratio range | Candidate wins | Control ratio range |
|---|---:|---:|---:|
| Primary completion | 0.9341–0.9724 | 169/248 | 0.9834–1.0099 |
| Distributed service | 0.9662–1.0032 | 138/248 | 0.9724–1.0224 |

The primary completion benefit survives bank rotation, supporting continued
validation of the four-expert adapter. Distributed latency still overlaps control
variation. This is not a full-model working set or routing trace; the untimed
oracle still runs between requests. Wider-token validation and production
integration remain outstanding. No production defaults or throughput estimates
were changed.

### Four-expert adapter correctness validation and explicit support

`20261020/fp4-four-validation` validates the four-expert adapter on the primary
with sixteen CPU participants. The experimental adapter now accepts explicit
`run_batch_into` arrays of four experts; existing two/three-expert callers and
all defaults remain unchanged. The sole implementation change extends its
compile-time count constraint from two/three to two/three/four.

The probe compares every four-expert output bitwise with individual expert calls.
It covers two real layers, original and compact NUMA storage, one and two tokens,
normal/small/zero input, and reversed expert order: 48 cases per run. It passes
with NT2 NTA disabled and enabled against both HIP and CUDA CPU-library builds,
for 192 cases total. Input token two has a smaller magnitude than token one.
No GPU execution occurred; a SYCL compiler was unavailable.

Separate old/new compilation of existing two- and three-expert probes produces
byte-identical object files, recorded in `existing-instantiations.json`. These
checks establish unchanged generated code for those instantiations, not a full
release-binary comparison. The previously measured NT1 primary completion gain
remains local: no full-model or distributed throughput improvement is claimed,
and no two-token performance improvement was measured in this correctness test.

### Six worker tasks still help with NTA

`20261020/native-batch-tree-nta-workers3` compares three tasks per participant
against six on both workers, with integrated NTA gate/up and down retained.
Both arms disable automatic route-policy selection and set their task count
explicitly. The inherited `policy_on_calls` counter reflects an old local
boolean rather than the actual route-policy switch; use the source setters and
`tuned` field to identify treatment. Primary execution stays three plus one.

The test retains two real layers, eight experts, 4/2/2 placement, 16/4/4
participants, eight worker banks, 2048-byte prefetch, and running miners. Four
64-repeat processes use candidate/control/control/candidate order with paired
alternation. All 2048 checks pass, with output hashes matching preceding runs.
Normal and permuted scenarios give 248 pairs after warmup.

| Three / six tasks | Median ratio range | Candidate wins | Control ratio range |
|---|---:|---:|---:|
| Distributed service | 1.0214–1.0820 | 74/248 | 0.9914–1.0159 |
| Worker one GU | 1.0703–1.1354 | 44/248 | 0.9608–1.0223 |
| Worker one down | 0.9919–1.0442 | 85/248 | 0.9682–1.0073 |
| Worker two GU | 1.0768–1.1414 | 35/248 | 0.9684–1.0072 |

Retain six tasks for this NTA workload. Reduced task count does not improve
complete service, despite the opportunity to reduce repeated setup. The test
does not separately measure scheduling imbalance or activation setup overhead.
Production and defaults remain unchanged; no full-model or GPU claim is made.

### Eight worker tasks split GU and down effects under NTA

`20261020/native-batch-tree-nta-workers8` compares eight tasks per participant
against six on both workers. It retains the preceding three-task test's NTA
settings, two real layers, 4/2/2 expert placement, 16/4/4 participants, eight
worker banks, running miners, and fixed primary 3+1 execution. Both arms disable
automatic route policy and explicitly set task counts. The inherited
`policy_on_calls` counter does not indicate the actual policy switch.

Eight tasks produce exactly 128 GU rows and 256 down rows per task; six produces
170/171 and 341/342 respectively. Four 64-repeat processes use candidate/control/
control/candidate order with within-process alternation. All 2048 checks pass;
output hashes match preceding runs. Normal and permuted scenarios provide 248
pairs after warmup.

| Eight / six tasks | Median ratio range | Candidate wins | Control ratio range |
|---|---:|---:|---:|
| Distributed service | 0.9869–1.0112 | 136/248 | 0.9864–1.0170 |
| Worker one GU | 0.9990–1.0433 | 99/248 | 0.9847–1.0232 |
| Worker one down | 0.9245–0.9914 | 173/248 | 0.9977–1.0248 |
| Worker two GU | 0.9943–1.0380 | 101/248 | 0.9726–1.0086 |

Retain six tasks for complete service. The down-phase improvement merits a
separate six-GU/eight-down candidate, but is not proof that such a combination
will improve service: changing GU can affect subsequent cache state. Production
and defaults are unchanged; no full-model, GPU, or efficiency claim follows.

### Six GU tasks and eight down tasks improve the down phase

`20261020/native-batch-tree-nta-down8` and `-confirm` isolate eight down tasks
per participant while keeping GU at six. A scratch pool selects the override
only for the two-expert, one-token, four-participant, unsharded FP4 worker shape.
The primary and arithmetic are unchanged. Both workers retain integrated NTA,
eight rotating banks, and the existing 4/2/2 placement; miners continue running.
The inherited `policy_on_calls` field does not report the actual policy state.

Each experiment has four 64-repeat processes with within-process alternation.
Confirmation reverses candidate/control/control/candidate order using the same
binary. All 4096 layer checks pass with output hashes matching prior tests.
Normal and permuted scenarios give 248 pairs per comparison after warmup.

| Run / metric | Candidate ratio range | Candidate wins | Control ratio range |
|---|---:|---:|---:|
| Initial service | 0.9573–1.0184 | 142/248 | 0.9890–1.0501 |
| Confirm service | 0.9589–1.0171 | 142/248 | 0.9671–1.0397 |
| Initial worker one down | 0.9353–0.9760 | 187/248 | 0.9775–1.0076 |
| Confirm worker one down | 0.9408–0.9822 | 176/248 | 0.9846–1.0171 |
| Initial worker two down | 0.9258–0.9491 | 225/248 | 0.9866–1.0098 |
| Confirm worker two down | 0.9234–0.9574 | 206/248 | 0.9879–1.0099 |

Down-phase gains repeat on both workers, but total service remains within broad
control variation. Retain the scratch candidate without changing production
policy. The component improvement is not a measured model-throughput gain;
no GPU execution, full-model run, or theoretical-efficiency claim is made.

### Down-task service uncertainty analysis

`20261020/down8-service-analysis` reanalyzes the two preceding experiments without
new hardware runs. Four related normal/permuted layer cells remain grouped by
repeat pair. It computes equal-weight process means of log timing ratios,
subtracts the control-slot log ratio, and uses 4000 seeded circular moving-block
resamples within each process. Block lengths of two, four, and eight repeat-pair
groups check sensitivity to temporal correlation. The analysis is exploratory,
not a predeclared selection gate; intervals condition on these captured processes
and do not cover day-to-day, miner-load, or process-selection uncertainty.

| Metric | Pooled control-adjusted ratio | Conditional 95% interval, block four |
|---|---:|---:|
| Distributed service | 0.9804 | 0.9561–1.0049 |
| Worker one GU | 0.9971 | 0.9813–1.0131 |
| Worker one down | 0.9568 | 0.9477–0.9658 |
| Worker two GU | 0.9905 | 0.9780–1.0021 |
| Worker two down | 0.9395 | 0.9345–0.9448 |

Both down-phase intervals remain below one for every tested block length.
Service intervals include one at block lengths two and four, although the
block-eight upper endpoint is 0.9993. Initial service ratio is 0.9649 versus
0.9962 in confirmation; confirmation intervals include one at every block
length. Do not choose the favorable block length as proof of a service gain.
Retain the down-only candidate for further work without promoting its policy or
raising full-model throughput estimates. Source summary hashes and the analysis
script are recorded for reproduction; all timing outliers remain included.

### Isolated two-expert service does not establish a common down8 benefit

`20261020/fp4-down8-service` follows the distributed down8 experiments with local
worker service: two real experts, two layers, sixteen rotating weight banks,
four CPU participants, and NTA in both arms. The same scratch pool object keeps
GU at six tasks and toggles down between six and eight. Timing includes input
quantization, hidden scaling/quantization, expert calls, and an equal-weight
output reduction. IB and primary execution are absent; miners remain running.

Each worker uses four candidate/control/control/candidate processes, sixteen
alternating pairs per layer/amplitude, and sixteen calls per timed batch. Normal,
small, and zero inputs pass independent expert-output and reduction checks:
1536 batch checks across 24,576 timed calls. The first pair is excluded from
analysis. Normal-input comparisons give 60 pairs per worker/treatment class.

| Metric | Candidate ratio range | Candidate wins | Control ratio range |
|---|---:|---:|---:|
| Worker one service | 0.9472–1.0267 | 28/60 | 0.9634–1.0231 |
| Worker one down | 0.9585–1.0214 | 30/60 | 0.9499–1.0684 |
| Worker two service | 0.9364–1.0064 | 36/60 | 0.9858–1.0445 |
| Worker two down | 0.9592–0.9795 | 49/60 | 0.9913–1.0172 |

Do not promote a common worker policy. Worker two retains a modest down-phase
benefit, but worker one overlaps control variation and complete service lacks a
consistent improvement on both. This workload differs from the distributed test
in expert IDs, bank count, input sequence, and timing cadence, so it does not
invalidate that test's captured phase timings. It does limit generalization.
Production defaults and model-throughput estimates remain unchanged.

### Primary memory baseline refresh does not reproduce the older ratio

`20261020/primary-memory-roof-refresh` rebuilds the NUMA-balanced streaming
probe against the current CPU library on the Threadripper 1950X, using sixteen
pinned physical threads and first-touch placement. Miners are untouched. The
same-buffer read and FP4 modes alternate; the probe checks exact numerical
parity. A noninteractive DIMM metadata read requires administrator authentication,
so configured memory speed and theoretical bandwidth remain unverified.

With offset arithmetic, 2048-byte T0 prefetch, and NTA disabled, the two 1.21 GB
runs read at 81.29–81.41 GB/s and process FP4 payload at 45.37–45.77 GB/s. Median
paired payload ratios are 0.5590–0.5621. The two 302 MB runs give 39.87–46.28 GB/s
FP4 and ratios 0.5418–0.5781, with substantial read variation in the last run.
These are logical payload rates, not memory-controller traffic counters.

A forward/reverse settings screen at 1.21 GB gives:

| Offset arithmetic / prefetch bytes | FP4 payload GB/s | Median paired read ratio |
|---|---:|---:|
| Off / 0 | 41.31–42.86 | 0.5182–0.5397 |
| On / 0 | 39.91–41.23 | 0.5339–0.5505 |
| Off / 2048 | 35.39–39.76 | 0.4907–0.5225 |

No tested setting reproduces the older near-read-baseline result. The initial
1.21 GB FP4 runs report approximately 9.68–9.69 active CPU cores from process
CPU-time divided by wall-time, versus about 15.37 for read. This is not an
instruction-retirement utilization measurement and does not identify the cause
of reduced throughput. The settings screen is not a same-process kernel A/B
comparison and supplies no basis for a policy change. Do not infer a kernel
regression, theoretical efficiency, or full-model tok/s from these observations;
retain raw measurements and investigate execution conditions before using a
peak-performance denominator. Production and simulator defaults are unchanged.

### Per-thread streaming timing isolates unequal execution time

`20261020/primary-stream-thread-timing` adds thread CPU-clock and wall-clock
measurements around each worker's compute section, excluding its rendezvous.
Two 1.21 GB runs retain sixteen pinned physical threads, balanced NUMA first
touch, offset arithmetic, 2048-byte T0 prefetch, and the same exact parity checks.
Each run records nine measured samples per mode/thread, 288 records total.
Instrumentation adds two thread-clock and two steady-clock reads per section;
these runs should not be treated as an uninstrumented performance A/B comparison.

In the first run, most FP4 workers take approximately 14–15 ms, but thread IDs
2, 5, and 8 take median wall times around 24.6, 27.7, and 26.6 ms respectively.
Their CPU times are nearly equal to wall times. The second run is slower across
nearly all FP4 workers, around 23–28 ms. Read work stays near 14–15 ms per worker.
Median thread CPU/wall fractions grouped by node exceed 99% in both modes/runs.

The earlier low aggregate active-core count therefore cannot be interpreted as
simple worker descheduling. Faster threads can finish and wait for slow peers at
the barrier, reducing aggregate CPU-time/wall-time. On-CPU time includes stalls,
SMT contention, and other execution costs; it does not identify their cause or
prove useful-instruction utilization. Unequal execution time and its changing
pattern need further isolation before attributing a kernel regression or using
this benchmark to claim theoretical peak efficiency. No defaults changed.

### Slow streaming threads correlate with busy SMT siblings

`20261020/primary-stream-placement` extends the per-thread timing probe with
buffer offsets modulo 4096 and logical-CPU busy ticks from `/proc/stat`. Three
runs retain sixteen physical threads, NUMA-balanced first touch, a 1.21 GB
working set, and current offset/T0 settings. Miners are untouched. Only address
offsets are recorded, not raw pointers. CPU-stat reads and logging occur outside
the measured interval but can affect subsequent execution conditions.

Weight, activation, and output alignment for each worker is identical across
all three runs. Every weight buffer begins at offset 16 modulo 4096. FP4 worker
wall time correlates with its SMT sibling's busy fraction: the descriptive
Pearson coefficient across 48 worker/run observations is 0.823. This is not an
independence-adjusted statistical test. CPU 2 takes about 26–28 ms with its
sibling busy 80–100% of sampled intervals. In the third run, most workers with
quiet siblings finish around 14–16 ms, while that worker remains slow.

SMT contention is therefore a plausible contributor to the variability, despite
workers remaining on-CPU. Fixed observed alignment cannot explain changing
per-worker times across these runs, although this does not prove alignment is
irrelevant to absolute performance. CPU-stat ticks are coarse; fractions sum
nine intervals and identify activity, not the responsible process or instruction
mix. No miner, system affinity, or production setting was changed. Future peak
calibration must account for sibling activity rather than interpreting process
CPU-time as useful work or attributing the entire slowdown to kernel code.

### NUMA-local stream work sharing removes much of the straggler penalty

`20261020/primary-stream-node-queue` tests dynamic FP4 copy assignment within
each NUMA node. Buffers retain first-touch placement, and every task uses its
owner's activation and output slice. An atomic node queue distributes independent
256-row copies among that node's eight compute threads. Read control remains
statically assigned. A rendezvous completes all tasks before exact per-buffer
verification. No worker accesses another node's weight jobs and no miner settings
change. The synthetic 1.21 GB test retains sixteen compute threads.

Initial process-level runs vary widely, so `primary-stream-node-queue-paired`
alternates static and queued FP4 work within a process. Four processes use
candidate/control/control/candidate order; controls retain static assignment in
both slots. After discarding the first measured pair member, sixteen adjacent
pairs per process alternate treatment order. All read checksums and FP4 bitwise
parity checks pass, including output slices computed by a different thread.

| Process | Median queued/static ratio | Wins | Queued-slot / static-slot median ms |
|---|---:|---:|---:|
| Candidate first | 0.5832 | 16/16 | 16.531 / 28.207 |
| Control first | 0.9991 | 8/16 | 28.495 / 28.314 |
| Control second | 1.0056 | 7/16 | 29.128 / 28.872 |
| Candidate second | 0.6058 | 16/16 | 16.751 / 28.024 |

Queued candidate median paired payload/read ratios are 0.9105 and 0.8912.
These compare logical payload rates with the measured same-buffer read control,
not theoretical memory bandwidth or memory-controller counters. Dynamic work
sharing demonstrably reduces this benchmark's static-assignment penalty under
the observed conditions. It does not prove SMT contention is the sole cause,
and it is not a 39–42% full-model or expert-service gain: the existing expert pool
already distributes tasks dynamically. No production or simulator defaults
changed. The result improves interpretation of the streaming calibration rather
than satisfying the unverified theoretical-efficiency objective.

### Finer primary NUMA tasks improve four-expert completion

Source inspection confirms that NUMA-owned expert rows already use separate
dynamic node queues, with cross-node stealing disabled for these owned rows.
`20261020/native-batch-tree-nta-primary4-tasks6` and `-confirm` therefore test
task size rather than adding another queue. Both arms execute one four-expert
primary batch, with eight primary/worker banks, two real layers, 4/2/2 placement,
16/4/4 participants, and unchanged NTA worker kernels. Primary task count changes
from three to six per participant; this also affects the untimed oracle after
each request. Miners continue running.

Four 64-repeat processes per experiment alternate within-process treatments.
Confirmation reverses process order with the same binary. All 4096 layer checks
pass and output hashes match previous experiments. Normal/permuted scenarios
supply 248 pairs per comparison after warmup.

| Run / metric | Six / three ratio range | Candidate wins | Control ratio range |
|---|---:|---:|---:|
| Initial primary completion | 0.9325–0.9655 | 169/248 | 0.9694–1.0251 |
| Confirm primary completion | 0.9153–0.9700 | 190/248 | 0.9823–1.0315 |
| Initial service | 0.9741–1.0147 | 126/248 | 0.9553–1.0090 |
| Confirm service | 0.9430–1.0109 | 143/248 | 0.9910–1.0086 |

Six tasks repeatably improve primary completion for this exact four-expert
NUMA shape. Complete service remains limited by workers and does not establish
a consistent gain. The existing task-count setter supports this configuration;
no global default or automatic policy is changed. Do not multiply this local gain
by earlier batching gains to predict model throughput. No GPU execution or
full-model measurement occurred.

### Pinning worker host threads regresses NTA service

`20261020/native-batch-tree-nta-host-pin` compares each worker's participating
host thread pinned to its reserved core versus its original allowed mask. The
reserved core is captured once before affinity changes. Three background compute
threads retain their existing pins; both arms use six tasks, integrated NTA,
eight worker banks, two real layers, and 4/2/2 placement. Primary execution stays
3+1. Only benchmark threads change affinity; miners remain untouched.

The affinity syscall is issued in both slots after receiving a command and
before input receipt. Its cost is included in distributed service, but excluded
from worker compute-phase timing. Four 64-repeat processes use candidate/control/
control/candidate order with within-process alternation. All 2048 checks pass;
output hashes match preceding runs. Normal/permuted scenarios give 248 pairs.

| Pinned / original mask | Median ratio range | Candidate wins | Control ratio range |
|---|---:|---:|---:|
| Distributed service | 1.0241–1.0744 | 89/248 | 0.9876–1.0189 |
| Worker one completion | 0.9731–1.0171 | 120/248 | 0.9878–1.0195 |
| Worker two completion | 1.0296–1.3263 | 55/248 | 0.9763–1.0351 |
| Worker one GU | 0.9695–1.0809 | 122/248 | 0.9898–1.0158 |
| Worker two GU | 1.0726–1.3554 | 62/248 | 0.9735–1.0394 |

Retain the original host-thread policy. Fixed affinity does not guarantee a
better core under background load, and worker two regresses in this test. The
experiment does not identify the responsible background process or isolate
migration cost from core contention. No production default or full-model
throughput estimate changes.

### Shared Q2 scale expansion does not improve NTA service

`20261020/q2-wide-scale-nta` tests one unsigned byte-to-word expansion for Q2
scales, followed by word masking and shifting to derive scale/minimum vectors.
This replaces separate expansions in the baseline. A compile-time specialization
is selected outside the block loop only for NT1 Q2 with NTA; Q3, wider tokens,
lookup mode, and floating-point accumulation order remain unchanged. The
same-object selector changes only this specialization.

Two workers each run candidate/control/control/candidate processes with twenty
alternating pairs and 64 timed calls per arm. The synthetic complete service
uses Q2 gate/up, Q3 down, H4096/F2048, two experts, one token, sixteen rotating
banks, four participants, and NTA in both arms. Input quantization is excluded;
hidden quantization and routed reduction are included. Miners continue running.
All 320 batch checks pass across 20,480 timed calls with identical output hashes.

| Worker | Candidate process ratios | Candidate wins | Control ratios |
|---|---:|---:|---:|
| One | 1.0027 / 0.9938 | 19/40 | 0.9864 / 0.9834 |
| Two | 1.0081 / 0.9921 | 21/40 | 0.9793 / 0.9790 |

Reject this candidate for service optimization. Fewer source-level expansions
do not establish a faster complete workload; results overlap or trail control
variation. Production and defaults remain unchanged. Broader raw-scale parity
was not pursued after the performance screen failed, and no GPU or full-model
throughput claim is made.

### Q2 split integer chains do not help Ryzen NTA service

`20261020/q2-split-chain-nta` tests even/odd integer accumulation chains in
NT1 Q2 under NTA. The earlier `q2-int-chains` screen covered Threadripper row
calls with prefetch off and found no reliable gain; this follow-up addresses
the previously untested Ryzen worker complete-service configuration. Separate
integer sums combine before the original scale and floating-point accumulation.
Q3 and other token widths retain baseline arithmetic. A compile-time candidate
is selected outside the block loop; controls use baseline in both slots.

The synthetic H4096/F2048 service uses two experts, one token, sixteen rotating
banks, four CPU participants per worker, and integrated NTA. Each worker runs
candidate/control/control/candidate processes, twenty alternating pairs per
process, and 64 timed calls per arm. Miners continue running. All 320 batch
checks pass over 20,480 calls with identical output hashes. Hidden quantization
and routed reduction are timed; input quantization is excluded.

| Worker | Candidate process ratios | Candidate wins | Control ratios |
|---|---:|---:|---:|
| One | 1.0021 / 1.0040 | 19/40 | 1.0016 / 0.9897 |
| Two | 1.0189 / 1.0113 | 18/40 | 0.9847 / 1.0141 |

Retain the existing accumulation loop. The candidate gives no service gain
on either worker, so it is not integrated. No production policy, simulator
capacity, or full-model throughput estimate changes; no GPU execution occurred.

### Zen 2 tuning does not establish a current NTA service gain

`20261020/native-batch-tree-nta-znver2` recompiles the current identical FP4
source with an extra `-mtune=znver2`, using GCC 15.2.0. The earlier worker
compiler screen predated the integrated NTA changes. The dispatcher selects
the candidate only for ordinary NT1 GU on both Ryzen workers; down and primary
remain baseline. Object text sizes differ (162417 versus 161950 bytes), but that
static size difference is not evidence of faster execution.

The test retains two real layers, eight experts, 4/2/2 placement, 16/4/4
participants, eight worker banks, current NTA, and running miners. Four
64-repeat processes use candidate/control/control/candidate ordering and paired
alternation. All 2048 checks pass with bitwise output hashes matching preceding
tests. Normal/permuted scenarios supply 248 pairs after warmup.

| Tuned / baseline | Median ratio range | Candidate wins | Control ratio range |
|---|---:|---:|---:|
| Distributed service | 0.9805–1.0092 | 136/248 | 0.9787–1.0323 |
| Worker one GU | 0.9842–1.0023 | 143/248 | 0.9790–1.0256 |
| Worker two GU | 0.9901–1.0312 | 123/248 | 0.9900–1.0115 |

Retain the current compiler settings. Results do not establish a repeatable
service benefit over control variation. No production, simulator, GPU, or
full-model throughput claim follows from this screen.

### V100 cache-admission refresh: fewer CPU groups did not improve decode

`20261020/v100-cache-admission-refresh/review.json` records newly reported
remote results from `docs/GLM_DFLASH2_V100.md`, identified by content hash.
These are read-only report findings, not locally reproduced measurements.

Counting all suitable GPU arena holes for cache admission, while retaining
scratch reserves, reduced CPU expert groups per request from 20455 to 20138
(1.55%). Complete MTP decode nevertheless fell from 39.68 to 37.59 tok/s (5.27%),
and prefill from 519.55 to 495.80. Each side used three full warmups and three
timed 512-output requests with zero timed major faults or swap. Speculative
rounds rose from 134 to 137. Outputs changed under the existing affinity/drop
policy, though they were repeatable and passed length/EOS/repetition checks.
Coding checks were skipped after regression; the option remains disabled. These
counters do not establish the cause of the slowdown.

A separate grouped-scatter prototype preserves expert order and separate FP32
multiply/add rounding, improving a twelve-case component benchmark by 1.053437x.
It passed 88 bitwise and 88 graph checks plus CUDA memcheck/racecheck, original
128-output checks, and width/history/rollback checks. CUDA and HIP builds and
six host checks per backend passed; HIP GPU execution was not tested. Full-model
timing was deferred before startup, so no additional model throughput is known.

Local `tools/sim/fp4.py` uses fixed expert counts and explicit hit rates and
already excludes MTP, quality, routing traces, and acceptance prediction. Do not
map fewer CPU groups or the scatter component multiplier directly into a full
model speedup. Cache-placement trials must measure complete decode and report
any changed output/acceptance policy. No simulator rates or production defaults
were changed based on these remote findings.

### NTA GU preparation occupies a small measured budget

`20261020/fp4-nta-prepare-profile` times the `NtaOffsetActs` constructor in the
ordinary clamped NT1 GU entry. A compiler barrier keeps construction between
steady-clock readings; counters aggregate after the measured constructor and
before row evaluation. The constructor duration includes timer overhead and
possible interruption. Atomic aggregation is outside that duration but inside
the instrumented GU phase, so these timings are not an uninstrumented A/B test.

Two 64-repeat processes retain the current NTA kernels, six worker tasks,
eight banks, two real layers, 4/2/2 placement, and 16/4/4 participants. Miners
remain running. All 1024 layer checks pass with prior output hashes. Each worker
process records exactly 12288 preparations (24 per request), totaling 49152.
All input amplitudes and warmup requests are included in aggregate statistics.

| Worker / run | Mean constructor ns | Summed preparation per request, microseconds | Summed preparation / GU wall |
|---|---:|---:|---:|
| One / 0 | 563.7 | 13.53 | 1.31% |
| Two / 0 | 582.7 | 13.98 | 1.36% |
| One / 1 | 558.3 | 13.40 | 1.31% |
| Two / 1 | 542.0 | 13.01 | 1.32% |

Summed preparation counts parallel workers and is not a critical-path fraction.
It supplies no guaranteed recoverable latency or speedup. The small budget does
not justify introducing a shared prepared-activation API at this point; focus
on row evaluation instead. Production, defaults, and throughput estimates remain
unchanged. No GPU or full-model execution occurred.

### Two-block unrolling does not improve current offset NTA worker service

`20261020/native-batch-tree-nta-unroll2` adds `#pragma GCC unroll 2` only to
the current offset-dot block loop. Earlier unroll screens targeted signed or
half-wise loops and different workloads. Both current objects use matching
`-fPIC`, O3, AVX2/FMA/F16C, and feature definitions. Dispatch selects the candidate
only for worker ordinary NT1 GU; down and primary retain baseline. FMA order
is unchanged. Object text grows from 162427 to 168984 bytes; that is not a
measurement of hot-loop footprint or instruction-cache traffic.

Four 64-repeat processes use candidate/control/control/candidate order and
within-process alternation. Two real layers, eight experts, 4/2/2 placement,
16/4/4 participants, eight worker banks, integrated NTA, and running miners
match the current service configuration. All 2048 checks pass with prior output
hashes. Normal/permuted scenarios give 248 pairs after warmup.

| Unrolled / baseline | Median ratio range | Candidate wins | Control ratio range |
|---|---:|---:|---:|
| Distributed service | 0.9632–1.0250 | 128/248 | 0.9426–1.0267 |
| Worker one GU | 0.9674–1.0107 | 124/248 | 0.9622–1.0364 |
| Worker two GU | 0.9983–1.0208 | 112/248 | 0.9783–1.0247 |

Retain the existing loop. The candidate does not establish a repeatable gain
and is not integrated. Production defaults, simulator rates, and full-model
throughput estimates remain unchanged; no GPU execution occurred.

### Current worker NTA streaming is near its measured read workload

`20261020/worker-nta-stream-roof` refreshes the old worker streaming baseline
with the current offset/NTA kernel, four pinned physical threads, a 301989888-byte
working set, and NUMA-local dynamic FP4 copy assignment. Two runs per worker
pass exact GGML parity and read checksum checks for each pass. Miners remain
running. Initial FP4/read ratios of 1.005–1.076 are not peak-utilization evidence:
that version still assigns read work statically.

`20261020/worker-nta-stream-fair` gives read work the same per-node copy queue.
Read checksums accumulate per owner with an atomic add per copy, then verify
after all tasks complete; FP4 output slices still compare bitwise with GGML.
Both modes use the same buffers and four pinned cores. Two fresh runs per worker
also pass all checks.

| Worker / run | Read GB/s | FP4 payload GB/s | Median paired FP4/read payload ratio |
|---|---:|---:|---:|
| One / 0 | 17.309 | 17.854 | 1.0288 |
| One / 1 | 17.658 | 17.892 | 1.0146 |
| Two / 0 | 17.853 | 18.066 | 1.0152 |
| Two / 1 | 17.647 | 18.024 | 1.0213 |

The read loop is a reference workload, not a hard memory-bandwidth ceiling.
Its instruction sequence, checksum reduction, and lack of the FP4 software
prefetch pattern differ, even with matching scheduling. Slightly greater than
one ratios do not mean hardware utilization exceeds 100%. These are logical
payload rates, not memory-controller measurements. This synthetic row call also
omits GU activation and the expert service's preparation/reduction, and pins the
host thread unlike the selected service configuration. Do not substitute its
rates into a full-model estimate or claim the theoretical 90% target is met.
Production and simulator defaults remain unchanged.

### Software prefetch does not strengthen the worker read reference

`20261020/worker-nta-read-prefetch` keeps FP4 NTA and dynamic scheduling fixed
while changing only the read loop's software prefetch: none, T0, or NTA, with a
2048-byte lookahead and requests for both cache lines per 128-byte iteration.
Each worker runs T0/none/NTA/NTA/none/T0 processes with four pinned threads and
the same 302 MB synthetic buffers; miners are untouched. All read checksum and
FP4 GGML parity checks pass. An initial compile rejected an integer template
hint where GCC requires an enum; explicit enum selection fixed it, and the
failed build log is retained.

| Worker | No-prefetch read GB/s | T0 read GB/s | NTA read GB/s |
|---|---:|---:|---:|
| One | 17.600–17.658 | 16.030–16.092 | 16.549–16.853 |
| Two | 17.806–17.967 | 16.589–16.590 | 16.962–16.984 |

FP4 throughput stays within 17.85–18.22 GB/s across these runs. Retain the
unprefetched read reference: adding either hint weakens it for this workload.
The screen does not prove that the reference reaches hardware peak, nor explain
all slightly-above-one payload ratios. Rates remain logical payload divided by
wall time; no memory-controller or theoretical bandwidth claim follows.
Production kernels, simulator rates, and full-model estimates are unchanged.

### Explicit paired integer assembly with current NTA worker kernels

The earlier inline-assembly pair was retested against the current offset-dot
NTA path. Both objects use matching PIC build flags. The candidate interleaves
independent `vpmaddubsw`, `vpsubw` and `vpmaddwd` chains with early-clobber output
constraints; floating accumulation and prefetch instructions remain unchanged.
The dispatcher selects it only for ordinary NT1 gate/up calls on both Ryzen
3950X workers. Primary and down kernels retain the baseline implementation.

The three-node partial-weight test retains 4/2/2 expert assignment, 16/4/4 CPU
participants, eight worker weight banks, direct IB and running miners. Four
processes use candidate/control/control/candidate ordering, with alternating
paired selection inside candidate processes. After warmup, normal/permuted
inputs provide 248 pairs per measurement class. All 2,048 layer checks pass
with bitwise matching output.

| Measurement | Candidate/baseline median-ratio range | Faster pairs |
|---|---:|---:|
| Distributed service | 1.0050–1.0753 | 88/248 |
| Worker 1 gate/up | 0.9895–1.0651 | 86/248 |
| Worker 2 gate/up | 1.0376–1.0956 | 50/248 |
| Control-process service | 0.9790–1.0351 | 105/248 |

This scheduling constraint provides no repeatable service improvement and
slows worker 2's gate/up phase. Retain the intrinsic implementation and existing
opt-in NTA settings. This tests the old assembly arithmetic in the newer NTA
configuration; it does not rerun the earlier exhaustive integer-domain test.
No production kernel changed, and no full-model decode or theoretical-peak
claim follows. Source snapshots, logs, build arguments and hashes are in
`results/20261020/native-batch-tree-nta-asm-pair/`.

### V100 grouped-scatter model result and local CPU coverage

A later read-only refresh of the remote report adds the previously pending
full-model grouped-scatter comparison. With grouped products in both arms,
MTP decode measured 39.85/41.24 tok/s and prefill 519.94/519.34. Each arm used
three full warmups and three timed 512-output requests. All six streams matched
the mild reference with zero timed major faults or swap. The 3.50% paired decode
gain missed the predeclared 41.4678 absolute threshold (3% above 40.26), so
conditional coding panels were skipped. A wrapper failure before model startup
was corrected without changing the frozen binary/settings.

Increasing depth and soft acceptance cap from three to four on the same grouped
core measured 41.79/40.47 decode and 519.59/520.57 prefill, using the same repeated
timing protocol. Outputs changed but remained repeatable; this slower setting
was rejected. The 41.79 control is a separate pair, not a newly qualified preset.
The best reported result with both reused coding panels remains 40.26 tok/s;
the fresh twenty-case holdout remains unscored. These are remote MTP results,
not local exact-decode measurements.

The local CPU implementation already has a related scheduling optimization:
`STRATA_GLM_LAYER_FLOW=1` selects `run_layer_native`, whose mode-eight tasks
perform expert work and dependency-scheduled routed reduction in one pool
dispatch. The ordinary phased path uses separate `reduce_routed` dispatch.
Observers, hidden transforms, native-local scheduling and other unsupported
cases fall back to phased work even when the option is selected. Therefore a
new CPU scatter fusion should first measure fallback incidence and reduction
cost in the intended configuration; merely duplicating the existing flow path
would not port a missing optimization. No local kernel, simulator capacity or
default changed. Report hash and structured findings are in
`results/20261020/v100-scatter-model-refresh/review.json`.

### Current NTA Q2/Q3: combined flow versus separate phases

The V100 grouped-scatter study motivates checking local scheduling coverage.
A same-process comparison now holds the current Q2 gate/up and Q3 down kernels,
NTA prefetch at 1024 bytes and three tasks per participant fixed, changing only
combined `run_layer_native` versus split products plus `reduce_routed`.

Both Ryzen 3950X workers use four pinned CPU participants with miners running.
The synthetic geometry is NT1, two experts, H4096/F2048 and sixteen rotating
weight banks. Input quantization is excluded; hidden quantization and routed
reduction are included. Four processes per worker use experiment/control/
control/experiment order, twenty alternating-order pairs per process and
64 calls per timed batch. Controls execute split phases on both labels.
The probe asserts that `ms_layer_flow` increases exactly for selected combined
calls, ruling out an unnoticed fallback in this geometry.

| Worker | Flow/split median paired ratio, first / last process | Faster pairs | Split / flow median ms, first process |
|---|---:|---:|---:|
| 1 | 0.6507 / 0.6541 | 40/40 | 1.6219 / 1.0475 |
| 2 | 0.7642 / 0.7732 | 40/40 | 1.4307 / 1.1039 |

Control ratios are 0.9882/1.0208 on worker 1 and 0.9837/0.9750 on worker 2.
All 320 batch output/reduction comparisons pass bitwise against the initial
split reference, across 20,480 timed calls. These measurements qualify the
existing combined scheduler for this bounded geometry, not a new kernel gain.
Earlier current-NTA expert-service tests already used combined flow, so these
reductions cannot be multiplied into their timings or a full-model forecast.
No production/default changed. NUMA-local fallback incidence, other routed
widths and end-to-end decode remain unmeasured by this test. Sources, logs,
settings and hashes are in `results/20261020/q23-nta-flow-paired/`.

### FP4 rescaling-aware combined flow: scratch service comparison

Unlike the preceding Q2/Q3 probe, the real-weight FP4 adapter uses split phases
and a small-hidden-value transform. The production combined scheduler rejects
that callback. A scratch-only pool variant invokes it in the last gate/up task
for each expert before hidden quantization, after the existing acquire/release
row-completion barrier. The tested transform has independent storage per expert;
this does not establish thread safety or exception handling for arbitrary
callbacks. Its output scale restoration remains on the caller after completion.

The scratch adapter calls combined flow with a retained contiguous output
workspace and zero reduction weights, then copies expert outputs back and runs
the original restoration and equal-weight reduction. This deliberately includes
an unused reduction and output-copy overhead; it is not a final fused adapter.
An assertion verifies combined scheduling rather than hidden fallback.

Two Ryzen 3950X workers each run four CPU participants with miners active, two
real experts from each of layers 3/4, NT1, sixteen weight banks, six tasks per
participant and integrated GU/down NTA. Four processes per worker alternate
experiment/control/control/experiment; controls use split for both labels.
Normal, tiny and zero inputs cover sixteen alternating pairs per layer, sixteen
calls per timed batch. All 1,536 batch checks match independent expert outputs
bitwise, including the original routed reduction, over 24,576 timed calls.

| Worker | Normal-input flow/split median-ratio range | Faster pairs | Control ratio range |
|---|---:|---:|---:|
| 1 | 0.9794–1.0484 | 32/60 | 1.0005–1.0584 |
| 2 | 1.0087–1.0675 | 23/60 | 0.9870–1.0136 |

There is no repeatable normal-input gain. Keep this implementation in scratch;
production retains split execution and its callback contract. The result does
not rule out a combined adapter without the extra copy and unused reduction.
Logged split phase counters do not cover combined work and must not be used to
infer that its GU/down time disappeared. No GPU, SYCL, sanitizer or full-model
qualification follows. Evidence is in `results/20261020/fp4-transform-flow/`.

### FP4 combined products without copy or unused reduction

A follow-up scratch variant removes the preceding prototype's extra output copy
and zero-weight reduction. All-null routed buffers select a products-only flow
in the scratch pool; down tasks write the original expert output buffers and
return before routed reduction. Completion still waits for every task. The
caller retains original per-expert scale restoration and equal-weight reduction.
This null-buffer convention is not added to the production API.

The same two real experts, layers 3/4, NT1, sixteen banks, four participants per
Ryzen worker, six tasks per participant and running miners are retained. The
same four-process paired/control protocol yields 1,536 bitwise batch checks and
24,576 timed calls across normal, tiny and zero inputs, all passing. The probe
again asserts actual flow execution rather than fallback.

| Worker | Normal-input flow/split median-ratio range | Faster pairs | Control ratio range |
|---|---:|---:|---:|
| 1 | 0.9339–1.0116 | 39/60 | 0.9841–1.0163 |
| 2 | 0.9982–1.0413 | 26/60 | 0.9873–1.0101 |

Worker 1 shows a possible small benefit, while worker 2 remains flat to slower.
Tiny and zero cases likewise show no consistent common-worker improvement.
Removing the copy and unused reduction does not establish a broadly faster
FP4 scheduler. Retain the production split path and keep this prototype scoped
to further experiments; generic transform concurrency/exception handling,
NUMA execution, sanitizers, GPU/SYCL and full-model performance are unqualified.
Do not derive a speedup from zero split-phase counters during combined calls.
Sources and results are in `results/20261020/fp4-transform-flow-direct/`.

### Three-task FP4 combined flow does not beat the split baseline

The copy-free, products-only scratch flow was screened with three tasks per
participant, against the existing six-task split baseline. This tests a combined
scheduler/task-count setting, not an isolated task-count effect; comparisons
with the preceding six-task flow run are cross-run observations only.

The prior real-weight geometry and protocol remain: layers 3/4, two experts,
NT1, sixteen banks, four CPU participants per Ryzen worker, NTA and active
miners. Controls keep six-task split execution on both labels. All 1,536
bitwise batch checks pass over 24,576 timed calls, covering ordinary, tiny and
zero input. Actual combined scheduling is asserted for candidate calls.

| Worker | Normal-input candidate/baseline median-ratio range | Faster pairs | Control range |
|---|---:|---:|---:|
| 1 | 0.9373–1.0346 | 29/60 | 0.9849–0.9972 |
| 2 | 1.0119–1.0847 | 12/60 | 0.9892–1.0216 |

There is no reliable common-worker gain; retain the existing split setting.
The three-task configuration does not solve worker 2's combined-flow slowdown.
Production code and defaults remain unchanged, and this scratch test does not
qualify general transform concurrency, NUMA execution, GPU/SYCL, sanitizers or
full-model throughput. Evidence is in `results/20261020/fp4-transform-flow-tasks3/`.

### Instrumented FP4 combined-flow dependency waits

The six-task copy-free scratch flow now has a diagnostic variant. A down task
whose initial acquire-load finds gate/up unfinished records elapsed wall time
until readiness; relaxed atomic counters collect nanoseconds and waiting-task
counts. The probe resets counters after warmup and before each sixteen-call
batch. Split calls assert zero counters. This adds an acquire-load and timing/
counter overhead and is not an uninstrumented performance comparison.

Two processes per worker retain two real experts from layers 3/4, NT1, sixteen
banks, four participants and active miners. Normal/tiny/zero inputs pass all
768 bitwise batch checks across 12,288 timed calls. Normal-input cells, excluding
the first pair, show the following medians across the sixteen-call batches:

| Worker | Summed wait per call, range across process/layer cells | Waiting tasks per call | Summed wait/service ratio |
|---|---:|---:|---:|
| 1 | 0–0.224 ms | 0–0.1875 | 0–0.152 |
| 2 | 0.240–0.374 ms | 0.1875–0.3125 | 0.161–0.221 |

The fractional counts are batch averages. Infrequent waits can be long, and
worker 2 warrants a readiness-handling investigation. Summed waits overlap
across workers, include possible descheduling and do not equal critical-path
latency or active CPU time. They do not prove a corresponding removable
service fraction or explain all previous differences. Profiled service itself
is 1.43–1.58 ms across these normal cells; it must not replace prior
uninstrumented measurements. No production change or model-rate inference
follows. Sources, logs and hashes are in
`results/20261020/fp4-flow-wait-profile/`.

### Bounded yielding does not improve FP4 dependency waits

Following the wait profile, a scratch comparison keeps six-task, copy-free,
products-only flow active in both timing arms. Only the wait policy changes:
the candidate calls `std::this_thread::yield()` after every 256 unsuccessful
readiness polls; baseline retains `_mm_pause()` alone. Both retain acquire/
release readiness publication and existing stall handling. The selector is
read once before a wait, and the earlier timing instrumentation is absent.
Controls use pause-only flow for both labels. This isolates wait policy rather
than comparing flow with split scheduling.

The same two real experts, layers 3/4, NT1, sixteen banks, four participants per
Ryzen worker and active miners produce 1,536 passing bitwise checks over 24,576
timed calls across ordinary, tiny and zero inputs.

| Worker | Normal-input yield/pause median-ratio range | Faster pairs | Control range |
|---|---:|---:|---:|
| 1 | 0.9972–1.0136 | 29/60 | 0.9795–1.0176 |
| 2 | 0.9634–1.0442 | 29/60 | 0.9767–1.0045 |

No repeatable gain separates from control variation. Retain pause-only waiting;
the preceding summed wait measurements do not imply that yielding can recover
them. Production remains unchanged. These results do not qualify general
callback concurrency, NUMA, sanitizers, GPU/SYCL or full-model throughput.
Evidence is in `results/20261020/fp4-flow-yield/`.

### NT2 Q2/Q3 combined-flow gain is much smaller than NT1

The preceding NT1 combined-versus-split probe was repeated with two routed
tokens, retaining two synthetic Q2 gate/up and Q3 down experts, H4096/F2048,
sixteen banks, NTA1024 and three tasks per participant. Each Ryzen 3950X worker
uses four pinned participants with mining active. Input quantization is outside
timing; hidden quantization and reduction are included. Both tokens route densely
to the same experts, which is not a mixed-route speculative verification trace.

Four processes per worker retain experiment/control/control/experiment ordering,
twenty alternating pairs and sixty-four calls per timed batch. All 320 batch
checks match the initial split reference bitwise over 20,480 timed calls, and
counter assertions confirm combined calls do not fall back.

| Worker | Flow/split median paired ratio, first / last experiment | Faster pairs | Split / flow median ms, first experiment |
|---|---:|---:|---:|
| 1 | 0.9925 / 0.9947 | 27/40 | 2.0935 / 2.0743 |
| 2 | 0.9661 / 0.9684 | 37/40 | 2.1787 / 2.0844 |

Control ratios are 1.0005/0.9975 and 1.0026/0.9980 respectively. The combined
scheduler yields only a small NT2 reduction here, unlike the earlier 23–35%
NT1 result. Token width must therefore remain explicit in scheduling evidence;
the NT1 multiplier must not be applied to speculative verification or model
throughput. No production/default changes follow. Evidence is in
`results/20261020/q23-nta-flow-nt2/`.

### Mixed-route NT2 Q2/Q3 combined scheduling

The two-token comparison now uses seven distinct synthetic experts: token zero
routes to 0/1/2/3 and token one to 0/4/5/6. Expert zero processes both tokens;
the others process one. Unused dense output slots remain zero with zero routing
weight. Four rotating banks total 255,066,112 packed bytes per worker. This
keeps the working set bounded while increasing expert count; comparisons with
the dense two-expert/sixteen-bank geometry cannot isolate routing alone.

Both Ryzen 3950X workers retain four pinned participants, active miners,
Q2 gate/up and Q3 down, NTA1024 and three tasks per participant. Combined and
split arms share kernels, weights and geometry. Input quantization is excluded;
hidden quantization and reduction are included. Four processes per worker use
experiment/control/control/experiment order with twenty alternating pairs and
sixty-four calls per batch. All 320 bitwise checks pass over 20,480 timed calls;
flow counter assertions exclude unnoticed fallback.

| Worker | Flow/split median paired ratio, first / last experiment | Faster pairs | Split / flow median ms, first experiment |
|---|---:|---:|---:|
| 1 | 0.9006 / 0.9179 | 40/40 | 4.5809 / 4.1372 |
| 2 | 0.9416 / 0.9354 | 40/40 | 4.3617 / 4.1152 |

Control ratios are 0.9963/1.0087 and 0.9983/0.9991 respectively. The existing
combined scheduler improves this mixed-route service by roughly 8–10% and
6–6.5%; token width alone does not describe scheduling performance. Retain the
existing opt-in without changing production defaults. This synthetic routing
pattern does not establish full-model speculative acceptance, quality or decode
throughput. Evidence is in `results/20261020/q23-nta-flow-mixed/`.

### V100 compact CPU lookup and later grouped-core coding validation

A new remote report snapshot adds separate functional validation of both sides
of the earlier 39.85/41.24 MTP pair: all eight reused long and eight reused
natural-length coding checks passed. This supersedes the earlier statement
that 40.26 was the highest result with both panels passed, but does not change
the original failed absolute speed-selection gate or select a new preset.
The experimental reference remains 40.26 and Fresh20 remain unscored.

A compact CPU scale-lookup prototype reports 1,920 bitwise cases on each of
AVX512 VNNI, AVX2 fallback and scalar fallback. At 32 workers its scalar lookup
component ratio is 1.052376x across gate/up and down shapes; single-worker
ratios are 1.024051x/1.026193x. Five cyclic samples per shape use sixteen rotating
4 MiB code buffers and shared activation storage. A four-lane gather variant
regressed to 0.531222x and was rejected. The report requires preserving the zero
sentinel and accounting for padded lookup tables against the existing CPU cache
budget. No full-model lookup result is reported in this snapshot.

The inspected remote CPU source uses CSF32 compressed scale metadata and a
1,032-float rounded weight-value table. That inspection does not establish the
new private prototype's exact implementation. Locally, ordinary AVX2 NVFP4
already reads a compile-time 256-float UE4M3 scale table (1 KiB) directly from
the four scale bytes in each 36-byte block. These representations and rounding
contracts differ; neither a duplicate compact table nor a 5.2% model multiplier
is justified. A compressed-scale port would need independent correctness,
storage-accounting and complete-service measurements. No local kernel/default
or simulator rate changed. Source/report hashes and findings are in
`results/20261020/v100-compact-scale-refresh/review.json`.

### Primary NUMA-owned Q2/Q3 mixed-route scheduling

The mixed-route NT2 comparison now runs on the Threadripper 1950X with sixteen
CPU participants and NUMA-owned rows. The route pattern, seven synthetic experts,
H4096/F2048 and four banks (255,066,112 packed bytes) match the worker probe.
Both arms use T0 prefetch1024 and three tasks per participant; native-local
scheduling is explicitly disabled. Miners remain active. This separates the
primary scheduling comparison from worker NTA selection.

Four processes use experiment/control/control/experiment ordering, twenty
alternating pairs and sixty-four calls per timed batch. Combined execution
is asserted through the pool timing counter, while log headers confirm owned
rows. All 160 bitwise checks pass over 10,240 timed calls. The output/reduction
hash matches the earlier worker mixed-route probe. Input quantization is
excluded; hidden quantization and reduction are included.

| Experiment | Flow/split median paired ratio | Faster pairs | Split / flow median ms |
|---|---:|---:|---:|
| First | 0.8948 | 20/20 | 1.3750 / 1.2259 |
| Last | 0.8799 | 20/20 | 1.3040 / 1.1438 |

Control ratios are 1.0073 and 1.0005. The existing combined scheduler reduces
latency by roughly 10.5–12% for this primary geometry, extending the worker
result to NUMA-owned execution. It does not qualify the separate native-local
option, which causes fallback, or full-model speculation/quality/throughput.
Retain the opt-in and unchanged defaults. Evidence and a local runner are in
`results/20261020/q23-primary-flow-mixed/`.

### Allowing owned combined flow with native-local enabled: scratch candidate

Source inspection finds that combined scheduling constructs its own node-local
queues for explicitly owned weights. The broad `native_local_enabled_` fallback
therefore has a candidate narrower form: retain it only when weights are
unowned. A scratch pool changes that one guard to
`(native_local_enabled_ && !owned)`. Other unsupported cases retain their guards;
production source is unchanged. Owned claims still do not steal from the other
NUMA node, and execution waits for all tasks before reusing queue storage.

The primary probe enables `STRATA_NATIVE_NUMA_LOCAL=1` in both arms. The
candidate executes combined flow, while baseline/controls use split work.
Geometry and protocol match the preceding primary mixed-route test: sixteen
participants, seven Q2/Q3 experts, NT2, four banks, T0 prefetch1024 and three
tasks per participant, with miners active. All 160 bitwise checks pass over
10,240 timed calls; combined execution is asserted and the output hash matches
the earlier probes.

| Experiment | Combined/split median paired ratio | Faster pairs | Split / combined median ms |
|---|---:|---:|---:|
| First | 0.8824 | 20/20 | 1.4023 / 1.2377 |
| Last | 0.8962 | 14/20 | 1.8244 / 1.4527 |

Controls give 1.0000/1.0006 ratios. The last experiment is noisier and paired
ratio medians need not equal the ratio of separate medians. Treat the narrow
exception as promising, pending repeat timing and broader correctness coverage
for unowned, fused, canonical and grouped-callback paths before integration.
This does not qualify GPU/SYCL or full-model throughput. Sources and evidence
are in `results/20261020/q23-primary-owned-flow/`.

### Owned/native-local exception: broader correctness and reversed ordering

The same scratch binary was rerun with control/experiment/experiment/control
process order. Candidate median paired ratios were 0.9023/0.8984, winning
16/20 and 18/20 pairs; controls were 0.9988/0.9938. All 160 bitwise batch checks
passed over 10,240 timed calls. A validation compilation overlapped measurement,
so retain these results as supporting observations, not a clean independent
performance confirmation. The binary hash matches the first experiment.

After timing completed, the existing flow-pool test was linked against the
scratch pool. Native-local was enabled, AVX512 disabled, Q2/Q3 AVX2 enabled,
and the pool spin window zero. Two extra assertions check that nonempty owned
jobs actually execute combined flow and unowned jobs retain fallback.
All 588 cases pass bitwise against whole-row phased references. Coverage
includes IQ, Q2/Q3, plain and owned rows, fused hidden quantization, shared
experts, canonical reductions, grouped callbacks with resident-token patterns,
viewed shards and uneven geometry. Group callbacks verify completed sums and
exactly-once publication. Validation execution did not overlap performance.

Production remains unchanged. Clean confirmation and backend integration checks
remain before promoting the narrow exception; these CPU results do not qualify
GPU/SYCL or full-model throughput. Logs, strengthened test source and hashes are
in `results/20261020/q23-primary-owned-flow-confirm/`.

### Owned native-local combined scheduling: clean confirmation and integration

A clean repeat used the identical scratch binary in control/experiment/
experiment/control process order, with no concurrent agent builds or tests.
Miners remained active. Candidate median paired ratios were 0.8699/0.8830,
winning 19/20 and 20/20 pairs; controls were 0.9993/1.0031. All 160 bitwise batch
checks passed over 10,240 timed calls. This confirms roughly 11.7–13.0% less
service time for the measured Threadripper 1950X mixed NT2 geometry: sixteen
participants, seven Q2/Q3 experts, four banks, owned NUMA rows, T0 prefetch1024,
three tasks per participant and native-local enabled. It is not model decode.

The narrow fallback exception is now integrated in `pool.cpp`. Explicitly owned
weights may use combined flow even with native-local enabled because flow
constructs its own owner-specific queues. Unowned weights retain the phased
placement-query path. Observers, transforms and other existing fallback guards
are unchanged. No default setting changes: this behavior requires selecting
combined layer flow and native-local scheduling together. With native-local off,
the guard follows the same branch as before.

`flow_pool_test` now asserts actual flow for nonempty owned cases and fallback
for unowned native-local cases, in addition to its bitwise whole-row references.
The new `flow_pool_native_local_test` CTest entry runs the same coverage with
native-local enabled. Single-NUMA systems retain the existing skip behavior.
The test covers IQ/Q2/Q3, fused quantization, shared experts, canonical sums,
grouped callbacks, viewed shards and uneven geometry. Default-path verification
is CPU bitwise oracle coverage, not a full release-server output comparison.

Clean timing, source snapshots and validation records are in
`results/20261020/q23-primary-owned-flow-clean/`. Published build/test logs replace
the repository's absolute path with `<repo>`; original logs remain in the local
build directory and their hashes are recorded. No GPU execution, full-model
throughput or 90% theoretical efficiency claim follows.

Both HIP and CUDA configurations rebuilt the touched CPU targets successfully.
In each, `q23_pool_test`, `flow_pool_test` and `flow_pool_native_local_test`
passed (three tests per configuration; each flow execution checks 588 cases).
These are CPU executions from the two build configurations. SYCL compilation
remains unavailable; GPU execution was not tested.

### Integrated owned/native-local flow: real FP4 coverage

The integrated pool exception also affects NVFP4 formats without hidden
transforms, so a real partial-weight probe now validates those cases. It uses
layers 3/4, one and three experts, token widths 1/2/4/8, owned NUMA rows,
split/combined scheduling and fused/unfused hidden quantization. Native-local
is enabled throughout. The primary uses sixteen CPU participants; miners remain
active. CPU-only binaries link against the rebuilt HIP and CUDA CPU archives.

All 384 batch output-and-sum comparisons pass bitwise against sequential
complete-blob GU, hidden quantization, down and ordinary routed reduction.
Output hashes also agree across schedules and both build configurations.
An assertion on every service call verifies that combined cases actually use
flow while split cases do not. This checks the intended branch, not just
numerically equivalent fallback behavior.

This is correctness coverage only: one weight bank and shortened timings do not
establish a performance gain. The small-hidden transform is absent; its
production fallback remains unchanged. GPU/SYCL execution, full-model decode
and theoretical efficiency are not tested. Source, build arguments, fixture
hash and results are in `results/20261020/fp4-owned-flow-validation/`.

### Small per-scale broadcast table: current NTA service screen

The earlier rejected FP4 pair tables occupied 512 KiB and 2 MiB. A distinct
scratch candidate instead adds a constexpr 256-by-four-float table (4 KiB),
aligned to sixteen bytes, so an individual scale can be loaded directly into
an SSE lane before assembling the AVX vector. The ordinary 1 KiB scalar table
remains for other paths. Values are copied exactly from the existing constexpr
decoder, including sentinel/high-bit behavior; packed weights and FMA order
remain unchanged. Object text/readonly size is 162,427 bytes baseline versus
167,422 candidate; symbol inspection confirms the new 4 KiB table is emitted.

The dispatcher changes only ordinary NT1 offset gate/up on both workers.
The current three-node probe retains NTA, 4/2/2 assignment, 16/4/4 participants,
eight worker banks, direct IB and active miners. Four experiment/control/
control/experiment processes provide 248 normal/permuted pairs per measurement
class after warmup. All 2,048 checks pass with bitwise matching outputs.

| Measurement | Candidate/baseline median-ratio range | Faster pairs |
|---|---:|---:|
| Distributed service | 0.9170–1.1358 | 132/248 |
| Worker 1 gate/up | 1.0069–1.0528 | 97/248 |
| Worker 2 gate/up | 0.9703–1.0320 | 116/248 |
| Control service | 0.9793–1.0309 | 124/248 |

Service variation is broad, while worker 1's gate/up cells consistently slow
and worker 2 has no clear gain. Reject this candidate and retain scalar loads;
a small table footprint alone does not establish faster service. No production
source/default or model-rate change follows. Evidence is in
`results/20261020/native-batch-tree-scale-single/`.

### Primary FP4 rescaling-aware flow with NUMA-owned rows

The earlier copy-free rescaling-aware scratch flow is now tested on the
Threadripper 1950X with sixteen participants and owned rows. It retains two
real experts from layers 3/4, NT1 and sixteen banks; both arms use three tasks
per participant, T0 prefetch2048 and native-local disabled. Miners remain
active. The independent per-expert small-hidden transform runs before hidden
quantization in flow; output restoration and equal-weight reduction remain
unchanged after pool completion. The scratch API now permits owned NT1 inputs.

Four experiment/control/control/experiment processes produce 768 passing
bitwise batch checks over 12,288 timed calls. Controls use split execution
for both labels, and selected combined calls assert actual flow execution.

| Input amplitude | Flow/split median-ratio range | Faster pairs | Control range |
|---|---:|---:|---:|
| 1 | 0.9222–1.0091 | 38/60 | 0.9803–1.0556 |
| 0.001 | 0.8654–0.9598 | 47/60 | 0.9434–1.0127 |
| 0 | 0.8722–1.0365 | 39/60 | 0.9857–1.0136 |

Normal-input timing remains mixed; tiny inputs show a provisional benefit that
warrants a repeat. This is not a general FP4 speed qualification or evidence
about real activation frequency. Production's hidden-transform fallback is
unchanged; generic callback concurrency and exception handling still need a
separate design before integration. No GPU/SYCL, full-model or theoretical-peak
claim follows. Evidence is in `results/20261020/fp4-primary-transform-flow/`.

### Primary rescaling-aware flow: initial tiny-input gain does not repeat

The identical scratch binary/settings were rerun in reversed control/
experiment/experiment/control order, with no concurrent agent build/test and
miners still active. All 768 bitwise checks pass across 12,288 timed calls.

| Input amplitude | Flow/split median-ratio range | Faster pairs | Control range |
|---|---:|---:|---:|
| 1 | 1.0042–1.2105 | 24/60 | 0.9223–1.1391 |
| 0.001 | 0.9894–1.1107 | 25/60 | 0.9601–1.0294 |
| 0 | 0.9385–1.6952 | 17/60 | 0.9937–1.0589 |

The preceding provisional tiny-input improvement does not survive this repeat.
Variation is substantial, with no attribution to a particular background
process. Retain the production hidden-transform fallback rather than changing
the callback concurrency contract for an unqualified performance benefit.
Across the two screens, 1,536 checks and 24,576 timed calls preserve outputs,
but correctness alone does not justify integration. Evidence is in
`results/20261020/fp4-primary-transform-flow-confirm/`.

### Mixed-route job order: shared expert first versus last

A primary CPU experiment holds combined scheduling fixed and changes only job
order. Seven experts serve two tokens: one shared expert has two routed rows
and six have one each. Candidate puts the shared expert first, baseline last;
controls put it last for both labels. Job output pointers and reduction order
stay fixed. Both arms copy the job array; baseline rotation overhead is included.
This is not a comparison against an unchanged production route-order policy.

The Threadripper uses sixteen participants, owned NUMA rows, four banks,
T0 prefetch1024 and three tasks per participant with miners active. Four
experiment/control/control/experiment processes retain twenty alternating pairs
and sixty-four calls per batch. All 160 bitwise checks pass over 10,240 timed
calls; all calls assert combined scheduling.

| Experiment | Shared-first/shared-last median paired ratio | Faster pairs | Last / first median ms |
|---|---:|---:|---:|
| First | 0.9471 | 14/20 | 1.9058 / 1.7015 |
| Last | 0.9082 | 13/20 | 1.5437 / 1.4128 |

Control ratios are 0.9744/0.9907. Process variation is substantial; paired ratio
medians differ from ratios of separate latency medians. Treat shared-first as
a provisional ordering candidate requiring repeat and broader route patterns,
not a qualified production optimization. No kernel/default or model estimate
changes. Evidence is in `results/20261020/q23-primary-route-order/`.

### Shared-first ordering: repeated and broader overlap patterns

The identical one-shared-expert binary was repeated with control/experiment/
experiment/control process order and no concurrent agent build/test. Candidate
median paired ratios were 0.8930/0.8923 with 17/20 and 19/20 wins; controls were
0.9862/0.9841. All 160 bitwise checks pass over 10,240 timed calls. This supports
the ordering effect seen in the initial noisier screen.

A separate probe then varies the number of shared experts while retaining four
routes per token: two shared gives six distinct experts, three gives five, and
four gives four. Shared experts process two tokens, the others one. Candidate
keeps the shared block first; baseline rotates that entire block to the end.
With four shared experts this rotation is a no-op, providing a control. Fixed
output pointers preserve the original reduction order. Each process uses twenty
alternating pairs and sixty-four calls per batch; shapes run 2/3/4 then 4/3/2.
The primary retains sixteen participants, owned rows, four banks, T0prefetch1024,
three tasks per participant and running miners.

| Shared experts | First/last median paired ratio, first / repeat | Faster pairs |
|---|---:|---:|
| 2 | 0.8748 / 0.9685 | 29/40 |
| 3 | 0.9232 / 0.9543 | 33/40 |
| 4, no-op order control | 1.0035 / 0.9892 | 21/40 |

All 240 broader-pattern checks pass bitwise over 15,360 timed calls. The trend
merits further study, but gain size varies. The baseline deliberately puts
shared experts last, rather than preserving a captured production order;
copying and rotation overhead are included. Do not promote a production sort
or extrapolate model speed until it beats realistic incoming order with its
own sorting cost included. Evidence is in
`results/20261020/q23-primary-route-order-confirm/` and
`results/20261020/q23-primary-route-overlap/`.

### Stable token-count sort versus first-seen job order

Executor inspection shows `host_jobs` is appended when a selected expert first
appears while walking token routes. Later occurrences extend that job. Thus a
shared expert appears among the first token's routes; placing it last among all
seven jobs was an artificial unfavorable baseline in the preceding screens.

A new synthetic probe follows that first-seen rule: twenty seeded permutations
of token zero's four experts, followed by a shuffled order of token one's three
new experts. Every pair uses the same incoming order. Candidate performs
`std::stable_sort` by descending job token count; baseline leaves it unchanged.
Both copy job descriptors, and sorting cost is inside timing. Output pointers
and reduction order remain fixed. This models the ordering rule, not a captured
routing trace or nonuniform route weights.

The primary uses sixteen participants, owned NUMA rows, seven Q2/Q3 experts,
four banks, T0prefetch1024 and three tasks per participant, with miners active.
All 160 bitwise checks pass across 10,240 timed calls.

| Experiment | Sorted/first-seen median paired ratio | Faster pairs | First-seen / sorted median ms |
|---|---:|---:|---:|
| First | 0.9598 | 12/20 | 3.2092 / 3.2737 |
| Last | 0.9931 | 11/20 | 5.3880 / 5.4866 |

Control ratios are 1.0873/1.0545 with only 4/20 and 8/20 label wins. Timing
variation is substantial; paired medians are not ratios of the separate latency
medians. The apparent gain is smaller than the shared-last comparison and is
not qualified against the observed controls. Retain production first-seen order.
Do not infer a model speedup or blame a particular background process from this
screen. Evidence is in `results/20261020/q23-primary-first-seen/`.

### Primary timing environment: current unrelated CPU load confirmed

After the first-seen ordering screen, read-only sampling with no benchmark from
this task active found all 32 logical CPUs about 85–99% busy. A separate host
process sample over 5.13 seconds found the largest benchmark process averaging
9.6 CPU cores, Python workloads in the top-ten sample about 11.5 cores, browser
work about 3.1 and compilers about 1.9. These are aggregate CPU seconds divided
by elapsed wall time, not clock frequencies or useful-work utilization.

The sandbox exposed only three process stat files while host inspection exposed
871; the sandbox's empty active-process list did not establish an idle host.
Host and per-CPU samples cover sequential intervals. Published evidence groups
processes by workload category; private raw snapshots are hash-identified.
No process was stopped, reprioritized or reconfigured.

The primary cannot currently be treated as free for small performance
comparisons. Defer further primary timing qualification while high unrelated
load persists, continuing low-impact implementation and correctness work.
This current snapshot does not establish the cause of each earlier fluctuation
or retrospectively identify when the load began. Existing frozen measurements
retain their controls and limitations; neither new model rates nor peak
utilization claims follow. Summary is in
`results/20261020/cpu-load-diagnostic/summary.json`.

### CPU input-preparation audit under primary contention

Read-only comparison with the V100 executor finds that local Q2/Q3 already
shares each token's quantized input across expert jobs. The asynchronous path
also skips quantization if no CPU job needs it. Reimplementing shared input
would therefore duplicate existing work rather than remove a bottleneck.

The pipelined path has a narrower candidate: it quantizes all token inputs
before route filtering constructs `host_jobs`. An opt-in per-token lazy path
could defer quantization until the first CPU job uses that token. It must reset
readiness every layer and preserve buffer pointer stability and exact quantizer
rounding. Route filtering includes pruned, GPU-resident, masked and mixed-worker
routes. Two consumers prevent unconditional skipping: remote TP receives the
full quantized token array, and resident-row parity checking computes CPU
references for resident routes using those buffers. An initial safe scope
would retain eager preparation whenever either consumer is enabled.

No executor change or speed claim is made by this audit. Required correctness
coverage includes all-GPU/all-CPU/mixed token demand, changed inputs across
layers, transport/diagnostic fallbacks and partial-model parity. Primary timing
remains deferred under the observed unrelated CPU load. Findings and the exact
local source hash are in `results/20261020/cpu-input-preparation-audit/audit.json`.

### Opt-in deferred pipelined CPU input quantization

`STRATA_GLM_LAZY_CPU_QUANT=1` now enables deferred per-token input quantization
in the pipelined executor. Buffers are sized before job construction; a token
is quantized when its first accepted local CPU job stores an activation pointer.
Subsequent jobs reuse that buffer. Readiness is local to one layer invocation
and is published only after the quantizer succeeds. The quantizer and its
rounding are unchanged. Pruned, GPU-resident, masked and mixed-worker routes
that do not create a local job do not trigger quantization.

The option defaults off and uses the shared C11 exact-one environment helper.
Remote TP or presence of `STRATA_GLM_CHECK_RESIDENT_ROWS` forces eager
preparation, preserving the complete transport token array and resident CPU
reference inputs. Presence retains the diagnostic's existing semantics even
if its value is `0`. The ordinary/asynchronous path is unchanged; it already
shares token inputs and conditionally prepares them.

`input_quant_ready_test` checks all token-demand subsets for widths 1/2/8 on
Q2/Q3 and FP4, two changed-input layers per subset, repeated requests, untouched
unused buffers, bounds and successful retry after a quantization exception.
Each configuration passes 1,048 demand/layer cases with bitwise quantizer
reference comparison. The policy gate also covers all request/remote/diagnostic
boolean combinations. This is CPU helper coverage, not end-to-end GPU-resident
routing or remote-transport execution. Full model parity and performance remain
unqualified; do not enable this option as a recommended preset yet. Primary
performance timing remains deferred under the recorded external CPU load.

Both CUDA and HIP decoder targets build successfully with this change. The
focused CTest passes in each configuration; these test executions use the CPU.
SYCL was not built because its compiler is unavailable. Sanitized build/test
logs, source hashes and the helper/test snapshots are preserved in
`results/20261020/lazy-input-quant/`. The evidence records the log transformations
and hashes of the original local logs. No full-model throughput improvement
is attributed to this change.

### Format-specific conditional 90% bandwidth budgets

`cpu_peak_budget.py --format` now covers native FP4, Q2_K, Q3_K and the
Q2/Q3 expert mixture, including expanded Q3 down rows. Payload counts include
scales and minima: 36/64, 84/256 and 110/256 bytes per weight respectively.
The mixed expert has two Q2 matrices and one equally sized Q3 matrix; expanded
Q3 uses 114 bytes per block. These are NT1 packed-weight traffic budgets, not
advertised floating-point arithmetic utilization.

For illustration only, assuming DDR4-2400, four populated channels and 16 cores
at 3 GHz gives a 69.12 GB/s target at 90% of theoretical payload bandwidth:

| Format | Required billion weight products/s | Maximum cycles/weight per core under ideal scaling |
| --- | ---: | ---: |
| Native FP4 | 122.880 | 0.3906 |
| Q2_K | 210.651 | 0.2279 |
| Q3_K | 160.861 | 0.2984 |
| Q2/Q3 expert | 190.950 | 0.2514 |
| Q2/Q3 with expanded down | 188.242 | 0.2550 |

These memory and frequency assumptions are not verified machine settings.
Lower bytes per weight require more weight products per second to saturate the
same memory interface. Extra activation/output traffic and dispatch consume
resources too; the table is a necessary packed-throughput budget, not a speed
forecast. Expanded weights can raise reported GB/s without increasing useful
work, so their byte ratio must not be mistaken for a decode improvement.

Q2/Q3 modes do not inherit the existing FP4 counter measurements. An explicit
counter file requires an explicit weight-product count; otherwise no measured
projection is shown. All three new budget tests and three existing roofline
tests pass. Evidence is in `results/20261020/format-peak-budget/`. A new three-second
host snapshot finds 19.4–97.3% CPU busy fractions, median 55.1%; this remains an
uneven concurrent workload rather than a clean all-core timing window. No
primary performance benchmark or GPU work was started for this calculation.

### Compact activation-scale products: rejected

A scratch FP4 offset-dot candidate compresses the earlier 256-entry product
lookup to 129 entries per activation block. Codes differing only in their high
bit share a product, except `0x7f`, which maps to a separate zero entry;
`0xff` retains its finite value. K4096 product storage falls from 128 KiB to
64.5 KiB. Preparation remains inside each call, and a raw -128 activation keeps
the existing fallback. Weight bytes, dot products and FMA order are unchanged.

The same-process synthetic GU screen compares this candidate against the current
integrated CPU library on Threadripper 1950X, one thread pinned separately to
cores 3 and 11. K4096/R128, one/64 banks, all-byte/finite weight scales and varied
finite half activation scales match the preceding scale-product experiment.
Twenty cells each run twelve alternating pairs of 64 calls. Miners and other
host work remain running; this is a diagnostic screen under concurrent load,
not an uncontended all-core performance qualification.

| Cases | Candidate/baseline median wall-time ratio |
| --- | ---: |
| NT1 safe, all-byte scales | 1.2068–1.2113 |
| NT1 safe, finite scales | 1.2049–1.2149 |
| NT1 raw -128 fallback | 1.0215–1.0592 |
| Unchanged NT2 control | 0.9990–1.0020 |

All 480 timed-arm output checks after 30,720 calls pass bitwise, along with
650 initial per-bank comparisons. Every safe-input candidate pair loses.
Reject the candidate: smaller product storage does not produce a useful gain
with this index mapping. The timings do not isolate mapping, stride or compiler
layout costs. The older 256-entry table's timings are not a same-day paired
comparator. Production kernels and defaults remain unchanged. Sources, build
commands, settings, raw samples and hashes are preserved in
`results/20261020/fp4-compact-scale-products/`. Full expert service, K32/narrow
runtime, GPU execution, model throughput and theoretical utilization were not
qualified.

### Prepared scale codes with a 64 KiB product table

The follow-up scratch candidate normalizes duplicate weight-scale codes once:
original `0x7f` becomes zero, and other codes retain their low seven bits. In this
prepared representation code 127 means finite 240, preserving original `0xff`.
The candidate's scalar scale table and signed fallback use that interpretation;
these bytes must never be passed to the ordinary GGML decoder. All 256 mappings
pass an independent scalar binary32 comparison. No production format changed.

The per-activation product table now has 128 entries and a 512-byte row stride,
for 64 KiB at K4096. Direct scale indexing removes the preceding compact
candidate's conditional mapping. Weight normalization and its extra copy are
outside timing; table construction remains inside each call. The first screen
uses separate original/candidate weight buffers. Its 20 cells cover the same
cores, banks, safe/-128 activations, varied half scales and raw/finite weight
codes as the preceding experiment. All 480 timed-arm checks after 30,720 calls
and 650 initial bank checks pass. Safe NT1 ratios range from 0.9633 to 1.0441
for raw scales and 0.9692 to 1.1146 for finite scales: no consistent gain.

A follow-up reverses the case order and uses the identical original weight
buffer for both kernels in finite-scale cases (codes 8–126 already satisfy
both representations). This removes the different-buffer cache effect for
those cases. All 288 timed-arm checks after 18,432 calls and 390 initial bank
checks pass. Safe NT1 candidate/baseline medians are 0.9870/0.9916 on core 11
but 1.0644/1.0420 on core 3 for 64/one bank. Unchanged NT2 controls range from
0.9989 to 1.0030. The prototype is not selected: gains fail to repeat across
cores and bank sizes, and concurrent host load prevents clean all-core
qualification. The experiment does not isolate stride, code placement or
cache effects and does not establish full expert or model throughput.

Evidence, sources, commands and binary/library hashes are in
`results/20261020/fp4-normalized-scale-products/` and
`results/20261020/fp4-normalized-scale-products-confirm/`. The confirmation links
the original candidate object; its hash is recorded in both summaries.

### Q3 signed-code dot: correction-buffer removal rejected

A scratch NT1 Q3 candidate converts unsigned three-bit codes to signed -4..3,
then uses absolute-code `maddubs` with sign-adjusted activations before the
existing scale multiply. It avoids constructing and reading the 4×activation
pair-sum correction buffer when all activation bytes are above -128. A once-per-
row-range activation scan selects the original correction path if any -128 is
present; this preserves the signed-byte negation edge case. Expanded Q3, NT2+
and Q2 retain their original arithmetic. Integer lane sums, block FMA order
and final reduction remain unchanged. Only a scratch object selects this path.

The CPU screen uses synthetic quantized Q3 weights, K4096/R128, one/64 identical
weight banks, and threads separately pinned to cores 3 and 11 on Threadripper
1950X. Safe fixtures replace any quantized -128 with -127 before constructing
the independent dequantized FP64 reference; fallback fixtures then inject -128.
The baseline reads those same activation bytes. Both arms use T0 prefetch 1024;
NTA and alternate compiler selectors are disabled. Each of sixteen cells runs
twelve alternating pairs of 64 calls. Other host workloads and miners remain
running; this is a single-thread diagnostic screen, not all-core qualification.

| Cases | Candidate/baseline median wall ratio | Candidate wins |
| --- | ---: | ---: |
| Q3 NT1 safe | 1.1575–1.1784 | 0/48 |
| Q3 NT1 -128 fallback | 0.9915–1.0047 | 18/48 |
| Unchanged Q3 NT2 | 1.0008–1.0062 | 19/48 |
| Unchanged Q2 NT1 | 0.9943–1.0082 | 27/48 |

All 384 timed-arm checks after 24,576 calls pass bitwise, along with initial
replicated-bank comparisons and independent FP64 reference checks. Reject the
candidate: avoiding correction-buffer traffic does not compensate for its new
signed-code operations in this screen. The measurements do not isolate a
particular instruction or stall. Production remains unchanged. Sources, exact
build commands, settings, samples and hashes are in
`results/20261020/q3-signed-code/`. Complete expert service, GPU execution,
full-model throughput and theoretical peak utilization remain unqualified.

### Q3 direct scale-word loads with inline assembly

Inspection of the current ordinary Q3 NT1 object finds a 12-byte scale copy
through the stack before three scalar words are consumed. A scratch candidate
uses three explicit `movl` loads into early-clobber general registers. Its
nonvolatile inline assembly describes all twelve bytes as a memory input and
never reads beyond the scale array. The scale reconstruction, integer dot,
correction buffers and floating-point order remain unchanged. NT2+ and Q2 keep
the original source path. Disassembly confirms direct loads in the candidate.

The same synthetic K4096/R128 screen, cores 3/11, one/64 banks, T0 prefetch 1024
and sixteen safe/-128/control cells as the preceding Q3 experiment passes all
384 timed-arm checks after 24,576 calls. Q3 NT1 candidate/baseline median ratios
range from 0.9773 to 1.0175. Unchanged Q3 NT2 controls range from 0.9961 to 1.0029
and Q2 NT1 from 0.9956 to 1.0002. Host load remains uncontrolled. Small mixed
changes do not qualify a complete-service gain, so this candidate is not selected.

LLVM MCA 21's `znver1` model of the emitted prefetch loop reports 138/135
instructions and 216/213 micro-ops per K256 iteration, baseline/candidate.
Modeled reciprocal throughput changes from 54.0 to 53.3 cycles. Without prefetch,
it reports 135/133 instructions, 213/211 micro-ops and 53.3/52.8 cycles. The
removed stack copy therefore changes only a small part of the modeled loop.
These static estimates do not measure hardware retirement, cache behavior,
contention, FLOPS/BW utilization or decode throughput. Baseline is the current
integrated library object; the candidate is a standalone experimental build,
with both objects' hashes and candidate flags recorded. Neither static counts
nor mixed timings establish causality for individual loads.

Sources, bounded-load assembly, normalized loop bodies, model outputs, raw
samples and hashes are in `results/20261020/q3-direct-scale-loads/`. Production
remains unchanged; no GPU, complete-service or full-model run was performed.

### Q3 scale-load comparison with matched build flags

The direct-load inline assembly was retested with baseline and candidate built
from the current Q3 source using identical GCC flags, including PIC, and linked
into one probe. Only NT1's twelve-byte scale load differs; both baseline and
candidate object hashes are retained. The independent GGML quantizer/reference
and packing helpers still come from the integrated CPU library. This addresses
the preceding screen's standalone-versus-library build difference.

The same sixteen K4096/R128 cells, T0 prefetch 1024, cores 3/11 and one/64 banks
pass 384 timed-arm comparisons after 24,576 calls. Q3 NT1 candidate/baseline
median ratios are 0.9935–0.9991 (0.1–0.7% lower latency). Unchanged Q3 NT2 controls
range from 0.9953 to 1.0126 and unchanged Q2 NT1 from 0.9888 to 1.0083. Host load
remains uncontrolled. The NT1 movement is smaller than the control variation;
this does not qualify a useful service gain or justify production assembly.
The broader CPU optimization effort should move beyond this scale-load change.
Evidence is in `results/20261020/q3-direct-scale-loads-matched/`. No full expert
service, GPU or full-model throughput test ran, and no peak claim follows.

### V100 combined scheduling result and local overlap audit

The refreshed worker report records an external DFlash2 pair at 35.30/37.89
model decode tok/s and 513.15/513.47 prefill tok/s. Early input readback, route
upload batching, resident expert grouping and ordered scatter are enabled
together. Each side uses three full warmups and three timed 512-output requests;
all six output streams match, with no timed major faults or observed swap in
the target/draft workers. Both sides pass the two reused eight-case coding
panels; the fresh twenty-case holdout remains unscored. This 7.34% decode gain
belongs to the combined target-engine changes, not an individual CPU kernel.

Both median requests use 147 verification rounds. Verification takes 12.06/11.12
seconds and drafting 2.11/2.06 seconds. Deleting all nonverification work for
the candidate's same proposal stream gives 45.95 tok/s; reaching 50 with other
work fixed requires 1.42 times faster verification. These are conditional phase
calculations, not forecasts or hardware ceilings. The report now freezes a
41.24 tok/s grouped MTP working profile and redirects its effort to prefill;
the earlier absolute speed-selection gate remains failed, and its experimental
reference remains 40.26. Compact CPU lookup still has no new qualified model
result in the inspected text.

Local source inspection confirms that the pipelined executor publishes its
mailbox before resident/shared-expert GPU work, optionally fuses batched route
publication, and already has mailbox wait/add plus resident-aware combination.
It does not expose the same obvious scheduling gap as the remote earlier
readback change. `pipeline_ready` disables this path in several configurations;
source inspection is not proof that a given model run selects it. Do not
multiply the remote combined gain into local forecasts or duplicate these
existing scheduling mechanisms. Further local scheduling work needs a trace
from the selected execution path. No GPU or model run was performed here.
Source snapshot hashes and the audit are in
`results/20261020/v100-combined-scheduling-refresh/audit.json`.

### All-core qualification prerequisites refreshed

Three new two-second CPU samples, with no task benchmark or build running,
observe 13.19, 13.20 and 17.08 busy logical-core equivalents out of 32. Median
per-logical-CPU busy fractions are 28.0%, 25.6% and 47.2%; at least one CPU is
fully busy in each sample. CPU pressure is low, so this records existing
activity rather than proving scheduler saturation. No unrelated process was
stopped or reprioritized. Accessible EDAC exposes no DIMM configuration.

All-core throughput qualification remains deferred pending a quiet CPU window
and actual DDR data rate/populated-channel information. GPU miners can stay
running. Existing Q3 correction-preparation and FP4 NTA constructor profiles
already show small preparation budgets; a shared prepared-activation ownership
API is not justified by revisiting that idea alone. The current load evidence
is in `results/20261020/cpu-qualification-refresh/summary.json`. No new speed,
full-model throughput or theoretical-utilization claim follows.
