#!/usr/bin/env bash
# Nsight Compute profile of the main GPU kernels (one launch each).
# Needs GPU performance-counter access. Without it ncu prints ERR_NVGPUCTRPERM; enable once (root):
#   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-prof.conf
#   then reboot (or reload the nvidia module); or run this script with sudo.
# Usage: scripts/ncu.sh [build-dir]
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
build=${1:-$here/build}
out=$here/results/ncu
mkdir -p "$out"
ncu=${NCU:-/usr/local/cuda/bin/ncu}
sections="--section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats --section InstructionStats --section LaunchStats"

# decode GEMV and fused MoE (batch 1)
"$ncu" $sections --kernel-name regex:'k_g1_dp4a|k_g2_f16|k_moe_gu|k_moe_down' --launch-skip 4 --launch-count 6 \
  -o "$out/decode" -f "$build/bench_gpu" --mode decode --ncols 1 --kinds 99 --reps 1 --trials 1
"$ncu" $sections --kernel-name regex:'k_moe' --launch-skip 4 --launch-count 2 \
  -o "$out/moe" -f "$build/bench_gpu" --mode moe --reps 1 --trials 1
# prefill grouped GEMM, one launch per kind at T=2048
for k in 0 1 2 3 4 5; do
  "$ncu" $sections --kernel-name regex:k_gemm --launch-skip 2 --launch-count 1 \
    -o "$out/prefill_k$k" -f "$build/bench_gpu" --mode prefill --tokens 2048 --kinds $k --reps 1 --trials 1
done
echo "open with: ncu-ui $out/*.ncu-rep   (or: ncu --import FILE --page details)"
