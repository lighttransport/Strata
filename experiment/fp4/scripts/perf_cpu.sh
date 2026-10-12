#!/usr/bin/env bash
# perf stat of the CPU kernels: IPC and cache-miss rates per kernel family at 16 and 32 threads.
# perf_event_paranoid <= 1 is enough for user-space counters.
# Usage: scripts/perf_cpu.sh [build-dir]
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
build=${1:-$here/build}
out=$here/results/perf
mkdir -p "$out"
export OMP_PROC_BIND=spread OMP_PLACES=cores
events=cycles,instructions,cache-references,cache-misses,stalled-cycles-frontend,stalled-cycles-backend
for t in 16 32; do
  perf stat -e $events -o "$out/cpu_t$t.txt" "$build/bench_cpu" --threads $t --sets 4 --iters 40 > "$out/cpu_t$t.log"
  perf stat -e $events -o "$out/cpu_moe_t$t.txt" "$build/bench_cpu" --threads $t --sets 4 --iters 200 --moe-only > "$out/cpu_moe_t$t.log"
done
grep -H -E "instructions|cycles|misses" "$out"/*.txt
