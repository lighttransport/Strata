# RAM bandwidth

Small Linux/x86 AVX2 benchmark. Requires a C++17 compiler and OpenMP;
no libnuma development package is needed.

```sh
make -C tools/membw
tools/membw/membw
tools/membw/membw --threads 16 --mib 2048 --repeats 10
```

Workers are pinned to CPUs from the process's allowed affinity mask.
By default, one worker runs on each physical core. Extra threads use SMT
siblings after physical cores. Each worker allocates and initializes its
own buffers after pinning, so Linux's default first-touch memory policy
places pages on that worker's node. Arrays are split equally between workers
and rounded down to whole pages. `--mib` is the size of each of two arrays,
not the total allocation. Use buffers much larger than the last-level cache.

Select a node with `--node N`. This filters worker CPUs; it does not override
an inherited memory policy. For strict node binding, use numactl:

```sh
numactl --cpunodebind=0 --membind=0 tools/membw/membw --node 0
```

On multiple nodes, the default run uses local first-touch buffers across all
eligible nodes. Check `numactl --hardware` to see what Linux exposes. A system
exposing only one node cannot measure separate die-local and remote placement.
The tool prints selected CPUs and nodes. Automatic NUMA balancing and other
system activity can affect measurements.

Read sums every double with 4, 8, or 16 AVX2 accumulators (`--unroll`, default 4). Copy uses AVX2
non-temporal stores and a store fence, avoiding destination write allocation.
Bandwidth is decimal GB/s: read counts one array, copy counts source reads
plus destination writes (twice the array size). These are algorithmic traffic
counts, not hardware memory-controller counters. Allocation, initialization,
and full result validation are outside timing. A persistent OpenMP worker team runs the measured passes. Each mode discards one warm-up
pass, then reports median, best, and minimum across the measured passes.

## Measurement on this machine

Measured 2026-10-05 on an AMD Threadripper 1950X (16 cores, 32 threads),
X399, 4 x 32 GB DDR4, Linux 6.8.0-142-generic, GCC 13.3.0, with one NUMA
node exposed. Memory clock was not checked. Two 2048 MiB arrays, 10 measured
passes per mode; all results passed validation:

| Threads | Read median GB/s | Copy median GB/s |
| ---: | ---: | ---: |
| 1 | 19.83 | 13.57 |
| 4 | 30.91 | 36.56 |
| 8 | 41.33 | 47.53 |
| 16 | 70.90 | 77.02 |
| 32 | 68.13 | 72.46 |

The 16-thread best results were 71.42 GB/s read and 77.57 GB/s copy.
CPU selection follows sysfs CPU order, so smaller thread counts need not
spread evenly across dies. A separate 1024 MiB run at 16 threads measured
71.50 GB/s median read and 78.91 GB/s median copy.

### Two NUMA nodes

Remeasured 2026-10-05 after the system exposed two nodes, each with eight
physical cores and about 64 GiB RAM, on the same CPU and Linux kernel.
Two 2048 MiB arrays, 20 measured passes per mode. The both-node runs used
`numactl --localalloc` with pinned workers and first-touch initialization;
the single-node runs bound both CPUs and memory to the selected node.

| Placement | Threads | Read median GB/s | Copy median GB/s |
| --- | ---: | ---: | ---: |
| Both nodes, local allocation | 16 | 85.26 | 80.80 |
| Node 0, local memory | 8 | 42.50 | 40.37 |
| Node 1, local memory | 8 | 43.41 | 40.84 |
| Both nodes, local allocation, SMT | 32 | 80.68 | 75.66 |

The 16-thread best results were 86.38 GB/s read and 82.24 GB/s copy.
All data validation passed. Sampling `/proc/PID/numa_maps` confirmed
524288 buffer pages on each node in the both-node runs (2 GiB per node),
and 1048576 pages on the bound node in each single-node run (4 GiB).
These counts cover large anonymous mappings, with 4 KiB pages.

Reproduce the 16-core run:

```sh
numactl --localalloc tools/membw/membw --threads 16 --mib 2048 --repeats 20
```

Full output and commands: [results-2nodes.txt](results-2nodes.txt).


### Tuning and decode-like access

Added `--spread` to select physical cores round-robin across L3 cache groups
before selecting SMT siblings. On this 1950X, eight spread workers use CPUs
`0 4 8 12 1 5 9 13`: two cores per L3 group and four per NUMA node.
The earlier default eight-worker selection uses CPUs 0-7, all on node 0.
`--spread` changes CPU selection; local memory still comes from pinning and
first-touch initialization under `numactl --localalloc`.

Other controls:

- `--kernel read`: double-precision streaming sum; `--unroll 4|8|16` and
  `--prefetch BYTES` select loop shape and T0 prefetch distance.
- `--kernel read-int` / `read-sse`: eight accumulators of 64-bit integer sums,
  using AVX2 / 128-bit SSE respectively. Unroll and prefetch controls do not
  apply to these kernels. Every initialized word participates in a checked sum.
- `--kernel copy`: streaming copy with non-temporal stores.
- `--kernel decode-stream` / `decode`: synthetic batch-one packed int4 x int8
  matrix-vector dot products, with sequential / shuffled segment order.
  `--block-kib` sets segment size per worker (default 256 KiB). Each 2048-byte
  weight row produces one output from 4096 weights and a reused 4096-byte
  activation vector. The nibbles and activations vary; every output is checked
  against an independent scalar reference.
- `--layers N`: split the decode scan into N stages, with a worker barrier
  after each stage. Default 1; streaming read/copy kernels ignore this option.
- `--huge`: align buffers to 2 MiB and request transparent huge pages. This is
  advice; backing depends on the OS. The default explicitly disables THP for
  buffers. Per-worker sizes round down to 2 MiB with this option.
- `--kernel all` (default): read, copy, then shuffled decode.

The decode surrogate scans the whole allocated weight array once per pass,
without reusing small subsets that fit in cache. Shuffled segments and layer
barriers approximate selected expert/matrix segments and serial layers.
Workers own local row shards. It excludes real model routing, quantization
scales, IQ lookup tables, attention/KV work, activation quantization, PCIe,
GPU work, and file faults. Its bandwidth counts packed weight bytes only;
activation reads and output writes are not included. These results are not
engine tokens/s measurements.

Measured 2026-10-05 on the same two-node 1950X/X399 system above. All runs were
sequential, used local allocation unless stated, and passed data validation.
The initial 2048 MiB, 12-pass core/loop sweep found 92.03 GB/s median integer
reads with eight spread cores. Larger 4096 MiB arrays and 40 measured passes
confirmed the result:

| Streaming configuration | Median GB/s | Best GB/s |
| --- | ---: | ---: |
| 8 spread cores, AVX2 integer, first confirmation | 91.17 | 91.93 |
| 16 cores, AVX2 integer | 86.87 | 87.52 |
| 8 spread cores, AVX2 integer, repeated confirmation | 91.78 | 92.69 |
| 8 spread cores, AVX2 double, unroll 16 | 91.75 | 92.55 |
| 8 spread cores, SSE integer | 91.75 | 92.54 |

On this workload, eight spread cores outperformed 16 or SMT workers. In the
16-core double-read sweep, larger unrolling gave a small improvement, while
software prefetch and THP did not improve the best median. The roughly
92 GB/s measured stream is about 92% of the user's approximate 100 GB/s
channel-rate ceiling; the memory clock has not been checked.

The decode sweep used 2048 MiB arrays and 20 measured passes:

| Decode pattern | Threads | Page advice | Prefetch bytes | Median GB/s |
| --- | ---: | --- | ---: | ---: |
| Sequential segments, 1 stage | 8 | 4 KiB | 0 | 79.98 |
| Sequential segments, 1 stage | 16 | 4 KiB | 0 | 85.24 |
| Shuffled 256 KiB segments, 64 stages | 8 | 4 KiB | 0 | 76.76 |
| Shuffled 256 KiB segments, 64 stages | 8 | THP | 2048 | 83.02 |
| Shuffled 256 KiB segments, 64 stages | 16 | 4 KiB | 0 | 82.08 |
| Shuffled 256 KiB segments, 64 stages | 16 | 4 KiB | 2048 | 83.26 |
| Shuffled 4 KiB segments, 1 stage | 8 | 4 KiB | 0 | 46.32 |
| Shuffled 4 KiB segments, 1 stage | 16 | 4 KiB | 0 | 58.80 |

For the shuffled 256 KiB, 64-stage, 2048-byte-prefetch surrogate, node-0
workers with memory bound to node 0 reached 41.65 GB/s; binding their memory
to node 1 gave 21.05 GB/s. Sixteen workers with page interleaving over both
nodes reached 64.24 GB/s, compared with 83.26 GB/s with worker-local shards.
This suggests keeping CPU decode row shards local and using contiguous
weight segments on this machine. Eight workers maximize the measured plain
stream, while 16 workers help the quantized arithmetic and smaller-segment
workloads. Actual engine tuning still needs per-layer and end-to-end timing.
No inference-engine settings or system-wide memory policies were changed.

Separate placement probes confirmed equal buffer page counts on both nodes.
The THP probe backed all 4 GiB of buffers with anonymous huge pages; the
4 KiB probe reported zero anonymous huge pages. Performance confirmations
ran without the placement sampler.

Reproduce the streaming and decode configurations:

```sh
make -C tools/membw
numactl --localalloc tools/membw/membw --spread --threads 8 \
  --mib 4096 --repeats 40 --kernel read-int
numactl --localalloc tools/membw/membw --spread --threads 16 \
  --mib 2048 --repeats 20 --kernel decode --block-kib 256 \
  --layers 64 --prefetch 2048
python3 tools/membw/tune.py --suite read
python3 tools/membw/tune.py --suite confirm --mib 4096 --repeats 40
python3 tools/membw/tune.py --suite decode --repeats 20
```

Raw commands and measurements:
[initial loop sweep](tuning-results.json),
[core and instruction sweep](core-tuning-results.json),
[larger-buffer confirmations](confirm-tuning-results.json),
[decode sweep](decode-tuning-results.json),
[local/remote/interleaved comparison](locality-results.json), and
[placement probes](placement-results.json).
