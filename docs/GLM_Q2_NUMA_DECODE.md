# Q2 decode on two NUMA nodes

Measurements on 2026-10-05 use a Threadripper 1950X / X399, four 32 GB DDR4
DIMMs (125.8 GiB usable RAM), two NUMA nodes, and an RTX 5060 Ti 16 GB.
There are 15 pinned physical-core workers plus the host. Context allocation is
8192, GPU prefill batch 256, dense cache 4096 MiB, and GPU budget 12288 MiB.
The desktop GPU memory guards remain enabled. All weights retain their original
GGUF quantization.

The 90 GB/s streaming result does not predict 15 tokens/s automatically.
Each ordinary token reads 2.751 GB of routed packed weights. IQ2_XS and
IQ3_XXS execution also performs codebook lookups, sign expansion, scaled integer
dots, and float accumulation. The real-weight expert benchmark measured about
30–35 GB/s before the final task-size sweep, including its output checksum.
Whole-model decode additionally runs attention, fixed GPU projections, routing,
CPU/GPU transfers, and sampling.

## Coding results

Each fixture used three restored-state repetitions and a 512-token cap. All
answers ended on a stop token. Rates count decode transitions, excluding the
first output selected from prefill logits.

| Fixture | Input / output including stop | Ordinary median tok/s | GPU MTP depth 2 median tok/s |
| --- | --- | ---: | ---: |
| Primality | 53 / 133 | 9.744 | 11.323 |
| JSON escaping | 82 / 210 | 9.744 | 11.607 |
| CSV record parser | 101 / 490 | 9.759 | 10.248 |

MTP uses 12 row tasks per worker (`STRATA_NATIVE_TASKS_PER_THREAD=12`), combined
layer graphs and verification graphs. Ordinary decode retains its existing
three-task setting. MTP accepted 87/92, 137/144 and 300/380 proposals respectively,
with no prefix replay. All generated IDs match across modes and repetitions.
Peak sampled host RSS was 102,555 MiB ordinary and 104,861 MiB MTP; every
100 ms process-swap sample was zero. Peak engine GPU allocations were 8223.18
and 10299.70 MiB respectively.

Both sets of generated C++ compiled with warnings treated as errors and passed
400,538 primality cases, 2052 JSON cases and 2058 CSV cases each. The
[answers and reproducible checks](fixtures/glm53_q2_numa_decode/README.md) are saved. These fixtures
check this workload; they do not establish general coding quality. The
**15 tokens/s target remains unmet**.

The same-allocation mode sweep, before task-size tuning, measured 9.703 ordinary,
9.377 / 9.187 / 8.779 lookup at depths 1 / 2 / 3, and 10.519 / 10.665 / 10.273
GPU MTP at depths 1 / 2 / 3 on the prime fixture. The final 12-task MTP prime
rate is 6.2% above the three-task MTP depth-2 result. The three-token expert
microbenchmark improved from 198.48 ms in the reverse control to 185.14 ms
with 12 tasks. Prefetch distances 0 and 4096 did not beat the 2048-byte default.

See [the measurements](glm53_flash_q2_numa_decode_measurement.json) for individual
trials, acceptance, expert timings, memory, and commands.

The balanced physical-core sweep retained 15 workers plus the host. With 12 row
tasks, widths 1 / 3 took 122.69 / 319.17 ms on eight physical cores,
89.75 / 224.65 ms on twelve, and 78.59 / 182.63 ms on sixteen. The smaller
sets were evenly split between NUMA nodes. On ordinary CSV decode, the CPU
expert phases alone averaged about 72 ms/token, or 38 GB/s of packed traffic;
roughly 30 ms/token remained outside those phases.

## Implementation

- GPU input quantization can be shared by projections for verification batches
  up to eight tokens, using the existing `STRATA_GLM_QUANT_ONCE=1` switch.
- `STRATA_GLM_LAYER_GRAPHS=1` captures the KDA, mHC, normalization, router,
  host-transfer and shared-expert prefix with CPU routed experts.
- `STRATA_GLM_VERIFY_GRAPHS=1` enables separate graph entries for widths 1–4
  and retained-history state. Main buffer addresses remain stable across widths.
  History graphs are destroyed before their history allocation is released.
- `--check-verify-graphs` compares direct batches with graph warmup, capture and
  replay, mixed widths, two history allocations, accepted prefixes and rollback.
- `--decode-mode-sweep --decode-bench=3` compares ordinary decode, prompt lookup
  depths 1–3 and GPU MTP depths 1–3 from restored prompt state. All seven modes
  keep the same MTP allocation; this is distinct from an ordinary-only run.
- The NUMA pool uses epoch-tagged node queues so a late waking worker cannot
  claim work from the next phase. Its stall timer also survives spin-counter wrap.

## Further IQ2 tuning

The table-sign AVX2 path now reads the four grid/sign indices needed for
each half and expands the eight IQ2_XS scale bytes once per 256-weight block. Each half shuffles its two scales from that vector. The
codebook, signs, integer sums and float accumulation order remain unchanged.
No weights or activations are converted.

Forward and reverse controls used real routed weights, synthetic activations,
15 workers plus the host, 12 row tasks per worker and 20 timed rounds. Width 3
improved from 192.00 ms to 185.74 ms, about 3.3%; width 1 improved from 77.00 ms
to 75.59 ms, about 1.8%. Every output byte had the same hash as the old kernel.
These are expert-only measurements, not whole-model throughput.

A completed three-repetition primality sweep with scale expansion measured
9.853 ordinary, 11.250 MTP depth 1, 11.723 depth 2 and 11.375 depth 3 tokens/s.
All four modes retained the MTP allocation and produced the same 133 IDs,
including the stop token. They also matched the earlier ordinary control and
passed 400,538 generated-code cases. Peak sampled RSS was 104,455 MiB, with
zero process swap. This sweep supports retaining depth 2; it does not establish
a matched whole-model speedup from the kernel change. That sweep retained the
original array index reader; the final expert-only candidate uses direct half
loads as well. The final kernel matched original output hashes at widths 1–4.

Further matched coding comparisons were interrupted by concurrent compilation
and CPU test executables (`XModelDiff`, then `XBinUsdDiff`). Their throughput
numbers are discarded. CSV generated during that interference still matched
all 490 earlier control IDs and passed 2058 correctness cases. The original
coding table above remains the completed matched comparison. The 15 tokens/s
target remains unmet.

Other prototypes were removed: compact accumulators gave only about 1% in the
width-3 benchmark; a larger combined grid/sign table was slower in reverse
controls; direct 16-bit index loads and Zen 1 compiler tuning gave no repeatable
gain; sorting expensive tasks first was slower. SMT also lost: width 3 took
261.90 ms with 31 pinned workers plus the host, versus 189.31 ms on 16 physical
cores. All of these experiments preserved output hashes.

The initial follow-up model run encountered 229 MiB of process swap while
prefaulting all tensors and was stopped. The completed primality retry used
GPU prefill to visit expert weights and omitted the separate prefault pass.
`tools/glm_q2_coding_bench.py --no-warm-weights` selects that sequence.
`--mtp-sweep` compares ordinary and MTP depths 1–3 without lookup modes.
`--reject-competitors` stops only the benchmark when it samples a compiler or
sustained external CPU work. Any sampled process swap also stops the run;
rejected memory and competitor samples are saved beside its logs. These
controls change benchmarking, not the server configuration.

## Owned packed rows

`STRATA_GLM_Q2_NUMA_WEIGHTS=1` is an optional Linux experiment. Every expert's
original gate, up and down rows are split evenly into node-bound anonymous
arenas. Both projections run only on their owning node; there is no work stealing
between nodes. GPU prefill staging reconstructs the original packed expert bytes
from the two halves. There is no numerical conversion.

The main routed weights require 99,033,808,896 bytes: 47,223 MiB per node.
The default owned-weight cap is 98,304 MiB; change it with
`STRATA_GLM_Q2_NUMA_WEIGHT_MIB`. Preparation requires an additional 16 GiB of
host capacity and rejects incompatible prepacking, direct upload, TP and legacy
expert-cache settings. It stops if process swap exceeds 1 MiB during preparation.

Copied source pages are released with mapping and file-cache advice, covering
only full pages within the tensor. GGUF files remain unchanged. An earlier trial
that released mappings alone swapped about 6.2 GiB and was discarded. The
corrected trial completed preparation in 167.98 seconds, with 102,965 MiB peak
host RSS. Its three ordinary decode rates were 9.563, 9.017 and 8.335 tokens/s,
versus a fresh control median of 8.713. The small median gain does not justify
enabling this layout by default. Preparation now occurs before the first prefill
to avoid a second full source read during startup. The final startup check took
166.48 seconds to prepare, then 15.08 seconds for prefill. Its one decode trial
measured 9.740 tokens/s, matched all 133 control IDs and had zero sampled swap.
This verifies staging from owned rows; it does not establish a decode gain.

The selected config also completed a 4801-token API prompt in the 8192-token
context allocation. A streamed request was cancelled after a generated delta;
the following complete answer matched the first request. The engine reset and
continued serving. These are correctness checks, separate from the decode rates.

## Reproduction

The regular configuration remains [glm53f-q2.json](../configs/glm53f-q2.json).
The optional MTP configuration is
[glm53f-q2-mtp.json](../configs/glm53f-q2-mtp.json). To use it on this Linux host:

```sh
numactl --interleave=all python3 -m serve.server --engine glm \
  --config configs/glm53f-q2-mtp.json --host 127.0.0.1 --port 8095
```

Release the GPU from an existing model service before starting this config.
Keep ordinary throughput
separate from accepted-output throughput under speculation.

```sh
export GLM_Q2_MODEL=/mnt/nvme01/models/glm53f/q2/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
export STRATA_NATIVE_NUMA_LOCAL=0 STRATA_POOL_SPIN_US=20000
export STRATA_GLM_QUANT_ONCE=1 STRATA_IQ2_TABLE_SIGNS=1
export STRATA_GLM_LAYER_GRAPHS=1 STRATA_GLM_VERIFY_GRAPHS=1
numactl --interleave=all python3 tools/glm_q2_coding_bench.py "$GLM_Q2_MODEL" \
  --sweep --fixtures prime --tokens 256 --output build-q2-sweep
python3 tools/glm_q2_coding_check.py build-q2-sweep --fixtures prime
```

`numactl` controls new page placement; it does not relocate existing GGUF file
cache pages. Loading, prefill, MTP prime and graph qualification are excluded
from decode timing. The helper monitors process RSS and swap, respects EOS,
rejects divergent IDs across trials/modes, and writes decoded answers and JSON
measurements. The three fixtures are primality, JSON escaping and CSV parsing;
the code checker uses a sieve/boundary oracle, Python's JSON decoder, and CSV
writer/reader with explicit malformed-record checks.

The real-weight CPU benchmark uses actual routing and packed model slices,
with deterministic synthetic activations:

```sh
cmake --build build-glm --target strata-glm-q2-kernel-bench numa_pool_test
build-glm/strata-glm-q2-kernel-bench "$GLM_Q2_MODEL" build-q2-fast/baseline.routes mmap 3 15 20
STRATA_POOL_SPIN_US=0 ctest --test-dir build-glm \
  -R '^(native_pool_test|numa_pool_test)$' --output-on-failure
```

The kernel benchmark's final timer excludes output validation and covers only
expert execution. It supports `mmap` and `numa` placement and widths 1–4.
It excludes attention, drafting and GPU work, so its rate is not model throughput.
AVX2 codebook-gather and paired-row prototypes passed arithmetic parity but
were slower in the reverse-order control; they were removed.
