#!/usr/bin/env bash
# Full measurement pass: selftests, CPU sweep, GPU peak/decode/MoE/prefill, dummy and real weights.
# Usage: scripts/run_all.sh [build-dir] [model-dir]
# Results: results/<date>/{cpu,gpu}_{dummy,real}.csv plus the console logs next to them.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
build=${1:-$here/build}
model=${2:-/mnt/nvme01/models/m26f}
out=$here/results/$(date +%Y%m%d-%H%M)
mkdir -p "$out"

{
  echo "date: $(date -Is)"
  uname -a
  lscpu | grep -E "Model name|NUMA node|MHz"
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit --format=csv
  uptime
} > "$out/machine.txt"

export OMP_PROC_BIND=spread OMP_PLACES=cores

"$build/bench_cpu" --selftest --sets 1 | tee "$out/cpu_selftest.log"
"$build/bench_cpu" --selftest --sets 1 --weights real --model "$model" | tee -a "$out/cpu_selftest.log"
"$build/bench_gpu" --mode selftest | tee "$out/gpu_selftest.log"
"$build/bench_gpu" --mode selftest --weights real --model "$model" | tee -a "$out/gpu_selftest.log"

"$build/bench_cpu" --threads 1,2,4,8,16,32 --sets 4 --csv "$out/cpu_dummy.csv" | tee "$out/cpu_dummy.log"
"$build/bench_cpu" --threads 16,32 --sets 4 --weights real --model "$model" --layer 1 --csv "$out/cpu_real.csv" | tee "$out/cpu_real.log"

"$build/bench_gpu" --mode all --tokens 512,2048,8192 --csv "$out/gpu_dummy.csv" | tee "$out/gpu_dummy.log"
for layer in 1 47; do
  "$build/bench_gpu" --mode decode --weights real --model "$model" --layer $layer --ncols 1,4 --csv "$out/gpu_real.csv" | tee -a "$out/gpu_real.log"
  "$build/bench_gpu" --mode moe --weights real --model "$model" --layer $layer --csv "$out/gpu_real.csv" | tee -a "$out/gpu_real.log"
done
"$build/bench_gpu" --mode prefill --weights real --model "$model" --layer 1 --tokens 2048 --csv "$out/gpu_real.csv" | tee -a "$out/gpu_real.log"
echo "results in $out"
