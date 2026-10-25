#!/bin/bash
# tools/a64fx_probe/run_fp4.sh - build and run the FP4-Local kernel probe (src/kernels/cpu/glm_fp4_sve.c) on one A64FX node.
#
#   STRATA=<synced tree> OUT=<shared dir> tools/a64fx_probe/run_fp4.sh
#
# Builds glm_fp4_test and glm_fp4_bench with clang (the LLVM module of the job shell, -mcpu=a64fx
# -msve-vector-bits=512) into a node-local dir, runs the parity test (stops on a failure), then the bandwidth sweep:
# expert shapes (full 2048 x 4096 / 4096 x 2048 and the 4-way slices the 12-node layout streams) and the trunk shapes
# of one rank at TP12, 1 / 12 / 48 threads, 1..4 tokens, stream and phase modes. Result lines start with A64FX_FP4.
# Stages: STAGES="test experts trunk threads gemm" (default all); gemm = prefill shapes, M tokens per expert call.
set -uo pipefail
STRATA=${STRATA:-$(cd "$(dirname "$0")/../.." && pwd)}
ROOT=${ROOT:-/local/strata-fp4-probe-${PJM_JOBID:-nojob}}
OUT=${OUT:-$STRATA/tmp/a64fx-fp4-${PJM_JOBID:-nojob}}
STAGES=${STAGES:-"test experts trunk threads gemm"}
CC=${FP4_CC:-clang}
FLAGS=${FP4_FLAGS:-"-O3 -mcpu=a64fx -msve-vector-bits=512 -fopenmp"}
has() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }
mkdir -p "$ROOT" "$OUT"
P=$STRATA/tools/a64fx_probe
CCDIR=$(dirname "$(dirname "$(readlink -f "$(command -v $CC)")")")
export LD_LIBRARY_PATH=$CCDIR/lib:$CCDIR/lib64:$CCDIR/lib64/aarch64-unknown-linux-gnu:${LD_LIBRARY_PATH:-}
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active XOS_MMM_L_PAGING_POLICY=demand:demand:demand
echo "A64FX_FP4_PROBE host=$(hostname) job=${PJM_JOBID:-none} cc=$($CC --version | head -1) flags=$FLAGS out=$OUT"
K=$STRATA/src/kernels/cpu/glm_fp4_sve.c
echo "A64FX_FP4_PROBE sources $(cd "$STRATA" && sha256sum src/kernels/cpu/glm_fp4_sve.c include/strata/kernels/cpu/glm_fp4_sve.h tools/a64fx_probe/glm_fp4_bench.c tools/a64fx_probe/glm_fp4_test.c | awk '{printf "%s:%s ", $2, substr($1,1,12)}')"
$CC $FLAGS -Wall -I"$STRATA/include" "$P/glm_fp4_test.c" "$K" -lm -o "$ROOT/glm_fp4_test" > "$OUT/build.log" 2>&1 \
  && $CC $FLAGS -Wall -I"$STRATA/include" "$P/glm_fp4_bench.c" "$K" -lm -o "$ROOT/glm_fp4_bench" >> "$OUT/build.log" 2>&1 \
  || { tail -30 "$OUT/build.log"; echo "A64FX_FP4_PROBE FAIL build"; exit 1; }
B="$ROOT/glm_fp4_bench"

if has test; then
  "$ROOT/glm_fp4_test" | tee "$OUT/test.log"
  grep -q "A64FX_FP4_TEST PASS" "$OUT/test.log" || { echo "A64FX_FP4_PROBE FAIL parity"; exit 1; }
fi
NV=w4a8,w4a16q,w4a16p
if has experts; then
  for mode in stream phase; do
    for shape in "2048 4096" "4096 2048" "512 4096" "4096 512"; do
      set -- $shape
      "$B" --kernels $NV --rows $1 --cols $2 --threads 48 --mb 2048 --mode $mode | tee -a "$OUT/experts.log"
    done
  done
fi
if has trunk; then
  # one rank at TP12: KDA q/k/v 6 heads (768 x 4096), o_proj (4096 x 768), MLA o (4096 x 1536), dense FFN (1024 x 4096,
  # 4096 x 1024), shared expert slice (176 x 4096), head slice (12912 rows rounded to 12928), BF16 kv_b (6 heads x 512 rows)
  for shape in "768 4096" "4096 768" "4096 1536" "1024 4096" "4096 1024" "176 4096" "12928 4096"; do
    set -- $shape
    for mode in stream phase; do
      "$B" --kernels mxfp8,bfp8,q8 --rows $1 --cols $2 --threads 48 --mb 1024 --mode $mode | tee -a "$OUT/trunk.log"
    done
  done
  for shape in "3072 512" "768 4096" "4096 1536"; do
    set -- $shape
    "$B" --kernels bf16 --rows $1 --cols $2 --threads 48 --mb 1024 --mode stream | tee -a "$OUT/trunk.log"
  done
fi
if has threads; then
  for th in 1 12; do
    "$B" --kernels $NV,mxfp8,q8,bf16 --rows 2048 --cols 4096 --threads $th --mb $((th == 1 ? 128 : 512)) --tokens 1,4 | tee -a "$OUT/threads.log"
  done
fi
if has gemm; then
  for M in 16 32 64 128; do
    "$B" --kernels w4a8,w4a16p,q8,mxfp8 --rows 2048 --cols 4096 --threads 48 --mb 1024 --gemm $M | tee -a "$OUT/gemm.log"
  done
fi
echo "A64FX_FP4_PROBE DONE out=$OUT"
