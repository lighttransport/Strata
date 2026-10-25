#!/bin/bash
# tools/a64fx_fp4/build_runner.sh - build src/cpu_runner/glm_cpu_runner.cpp natively on an A64FX compute node.
#
#   STRATA=<synced tree> OUT=<dir> tools/a64fx_fp4/build_runner.sh
#
# clang 21 from the job shell's LLVM module with libc++ (the system libstdc++ is GCC 8: no C++20 headers),
# -mcpu=a64fx -msve-vector-bits=512, OpenMP, and the MPI wrappers that module ships (mpiclang / mpiclang++).
# MPI=0 builds the one-rank binary without MPI. UTOFU_DIR=<dir with tp_allreduce.h> adds the deterministic uTofu
# all-reduce (src/cpu_runner/comm_utofu.c, -ltofucom; STRATA_RUNNER_COMM=mpi at run time keeps MPI). Prints the binary
# path and the sources' sha256.
set -euo pipefail
STRATA=${STRATA:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT=${OUT:-/local/strata-cpu-runner-${PJM_JOBID:-nojob}}
MPI=${MPI:-1}
mkdir -p "$OUT"
CXX=clang++; CC=clang; DEF=""
if [ "$MPI" = 1 ]; then CXX=mpiclang++; CC=mpiclang; DEF=-DSTRATA_CPU_RUNNER_MPI; fi
FLAGS="-O3 -mcpu=a64fx -msve-vector-bits=512 -fopenmp"
cd "$STRATA"
C_SRC="src/kernels/cpu/glm_fp4_sve.c src/host/glm_fp4.c src/host/glm_names.c src/host/glm_env.c src/host/glm_expert_pack.c"
OBJS=""
for c in $C_SRC; do
  o="$OUT/$(basename "$c" .c).o"
  $CC $FLAGS -Iinclude -c "$c" -o "$o"
  OBJS="$OBJS $o"
done
LIBS=""
if [ "$MPI" = 1 ] && [ -n "${UTOFU_DIR:-}" ]; then
  $CC $FLAGS -I"$UTOFU_DIR" -c src/cpu_runner/comm_utofu.c -o "$OUT/comm_utofu.o"
  OBJS="$OBJS $OUT/comm_utofu.o"; DEF="$DEF -DSTRATA_CPU_RUNNER_UTOFU"; LIBS="-ltofucom"
fi
$CXX $FLAGS -std=c++20 -stdlib=libc++ $DEF -Iinclude src/cpu_runner/glm_cpu_runner.cpp $OBJS $LIBS -o "$OUT/glm_cpu_runner"
echo "RUNNER_BUILD $OUT/glm_cpu_runner mpi=$MPI utofu=${UTOFU_DIR:+1} $(sha256sum src/cpu_runner/glm_cpu_runner.cpp src/kernels/cpu/glm_fp4_sve.c | awk '{printf "%s:%s ", $2, substr($1,1,12)}')"
