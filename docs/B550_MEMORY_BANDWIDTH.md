# B550 RAM bandwidth — 2026-10-09

Ryzen 9 3950X, 16 physical cores / 32 threads, one NUMA node, 62.69 GiB RAM.
Measured with inference stopped; the authenticated LAN service was restarted
afterward. Kernel 7.0.0-38-generic, GCC 13.3.0, `amd-pstate-epp`, `powersave`
governor with boost enabled. No clock or firmware settings were changed.
Configured DIMM speed was unavailable without sudo authentication; these results
do not establish the memory clock or whether XMP is enabled.

## Results

Median of seven timed iterations after one warmup, three 1 GiB arrays (3 GiB
total), much larger than CPU cache. Rates are decimal **GB/s of logical bytes**.
Copy counts both reading and writing: 28 GB/s here means 14 GB/s of copied
payload. Triad counts two reads and one write; memory-controller traffic can
include additional write allocation and is not measured by this benchmark.

| Threads | Read AVX2, 4 KiB | memcpy, 4 KiB | Triad, 4 KiB | Read AVX2, huge pages |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16.58 | 23.01 | 13.76 | 21.54 |
| 2 | 26.79 | 28.03 | 19.18 | 29.27 |
| 4 | 28.79 | 28.55 | 21.17 | 28.61 |
| 8 | 27.63 | 27.76 | 20.26 | 27.66 |
| 12 | 29.27 | 27.73 | 20.78 | 29.34 |
| 16 | 29.90 | 26.85 | 20.80 | 29.92 |
| 32 | 28.90 | 26.09 | 19.97 | 28.86 |

An explicit AVX2 non-temporal copy peaked at 29.15 GB/s with four threads and
4 KiB pages. The reverse-order repeat of 12/16/32 threads confirmed read rates
of 29.27/29.89/28.90 GB/s for 4 KiB pages and 29.31/29.91/28.92 for huge pages.
Huge-page allocation was confirmed by `/proc/self/smaps_rollup`: 3020 MiB for
the first single-thread case, 3072 MiB for the others; 4 KiB cases had zero.

The main finding is **about 30 GB/s sustained read bandwidth**. More threads
past a few cores do not substantially increase it, and SMT does not help.
Huge pages improved single-thread reads but gave almost no additional read
bandwidth at the GLM configuration's 12 threads. This does not rule out a
benefit for GLM's different access pattern; it rules out assuming a large
multithreaded sequential-bandwidth gain from huge pages alone.

## Relation to GLM

The previous Q3 exact-cache warm run processed 1,086,242,160,640 logical CPU
expert bytes over 383 decode steps: about 2.836 GB/token. Dividing by this
12-thread read rate gives roughly **97 ms/token for weight reads alone**,
or 10.3 tok/s before dequantization, arithmetic, synchronization, and other
traffic. This is a simplified bandwidth estimate, not a decode prediction.
The measured CPU expert phase already processed about 22.1 logical GB/s;
end-to-end decode was 6.63 tok/s. More GPU expert residency or fewer CPU weight
bytes are more promising than simply increasing CPU threads. Actual decode
must validate any change.

## Reproduce

[Source](../tools/glm_memory_bandwidth.cpp),
[all measurements and metadata](benchmarks/b550_memory_bandwidth_20261009.json).
The program checks the read reduction exactly and samples every 4096th output
plus consumes the last element for copy/triad. It is a custom RAM microbenchmark,
not an official STREAM score. No GPU or model is used.

```sh
tools/glm_b550_60g.sh stop
mkdir -p build-memory-bandwidth-20261009
g++ -O3 -march=znver2 -std=c++20 -fopenmp tools/glm_memory_bandwidth.cpp \
  -o build-memory-bandwidth-20261009/bench
export OMP_PROC_BIND=close OMP_WAIT_POLICY=PASSIVE
export OMP_PLACES="$(python3 -c 'print(",".join("{"+str(i)+"}" for i in range(32)))')"
# Each invocation allocates 3 GiB and runs read, memcpy, triad and streaming copy.
# Run inside a MemoryMax=6G / MemorySwapMax=0 cgroup, as in the measurement.
for pages in 4k thp; do
  for threads in 1 2 4 8 12 16 32; do
    build-memory-bandwidth-20261009/bench "$threads" "$pages"
  done
done
tools/glm_b550_60g.sh start-lan
```

Binding uses CPU IDs 0–15 for distinct physical cores, then their SMT siblings
16–31. Check `lscpu -e=CPU,CORE,SOCKET,NODE` before reusing this on another host.
