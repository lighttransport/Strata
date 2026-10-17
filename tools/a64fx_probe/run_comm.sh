#!/bin/bash
# tools/a64fx_probe/run_comm.sh - collective costs at 6, 8 and 12 ranks for the A64FX simulator (tools/sim/a64fx.py),
# measured with gemm's pure-uTofu MoE dispatch bench (utofu-tests/moe_dispatch_bench.c: multi-TNI dispatch + combine
# and the tree all-reduce) on GLM-5.3-Flash geometry: 288 experts, top-8, hidden 4096, 42 MoE layers.
# Run inside a 12-node 2x3x2 allocation, from the interactive shell (node (0,0,0) becomes rank 0):
#   GEMM_UTOFU=<gemm checkout>/a64fx/utofu-tests tools/a64fx_probe/run_comm.sh
# Every sweep point prints one A64FX_COMM line; the full report per point is kept as moe_np<N>_ab<A>_b<B>.log. Sequential mpiexec runs only (one MPI job at a time per allocation).
set -uo pipefail
STRATA=${STRATA:-$HOME/work/Strata}
GEMM_UTOFU=${GEMM_UTOFU:?set GEMM_UTOFU to gemm a64fx/utofu-tests}
NPS=${NPS:-"12 8 6"}; ABYTES=${ABYTES:-"2 4"}; BATCHES=${BATCHES:-"1 3"}; ITERS=${ITERS:-2000}
W=$STRATA/tmp/a64fx-comm-${PJM_JOBID:-nojob}    # shared storage: every rank reads tofu_topo.txt from here
mkdir -p "$W" && cd "$W" || exit 1
cp "$GEMM_UTOFU"/tofu_demo.h "$GEMM_UTOFU"/tofu_topo_helper.c "$GEMM_UTOFU"/moe_dispatch_bench.c .
module unload LLVM/llvmorg-21.1.0 >/dev/null 2>&1; unset OPAL_PREFIX
export PATH=/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH
mpifcc -Nclang -O3 -march=armv8.2-a+sve -o tofu_topo_helper tofu_topo_helper.c -ltofucom > build.log 2>&1 \
  && fcc -Nclang -O3 -march=armv8.2-a+sve -o moe_dispatch_bench moe_dispatch_bench.c -ltofucom >> build.log 2>&1 \
  || { tail -20 build.log; echo "A64FX_COMM FAIL build"; exit 1; }
SX=${PJM_MPI_SHAPE_X:-${PJM_NODE_X:-2}}; SY=${PJM_MPI_SHAPE_Y:-${PJM_NODE_Y:-3}}; SZ=${PJM_MPI_SHAPE_Z:-${PJM_NODE_Z:-2}}
echo "A64FX_COMM shape=${SX}x${SY}x${SZ} job=${PJM_JOBID:-none} dir=$W"
for NP in $NPS; do
  V=vcoord_$NP.txt; : > $V
  for ((x=0;x<SX;x++)); do for ((y=0;y<SY;y++)); do for ((z=0;z<SZ;z++)); do echo "($x,$y,$z)" >> $V; done; done; done
  head -n $NP $V > $V.tmp && mv $V.tmp $V
  [ "$(wc -l < $V)" -eq "$NP" ] || { echo "A64FX_COMM SKIP np=$NP (shape too small)"; continue; }
  ok=0; for try in 1 2 3; do rm -f tofu_topo.txt; mpiexec -np $NP -vcoordfile $V ./tofu_topo_helper > topo_$NP.log 2>&1 && ok=1 && break; done
  [ $ok = 1 ] || { echo "A64FX_COMM FAIL topo np=$NP"; tail -5 topo_$NP.log; continue; }
  for AB in $ABYTES; do for B in $BATCHES; do
    L=moe_np${NP}_ab${AB}_b${B}.log
    mpiexec -np $NP -vcoordfile $V -x MOE_E=288 -x MOE_K=8 -x MOE_HID=4096 -x MOE_ABYTES=$AB -x MOE_BATCH=$B \
      -x MOE_LAYERS=42 -x MOE_ITERS=$ITERS -x MOE_RELAY=0 -x MOE_BRUCK=0 ./moe_dispatch_bench > $L.stdout 2>&1
    rc=$?
    # the bench writes its report to moe_log_<rank-0 coords>.txt (stdout stays with the launcher): keep a copy per point
    r0=$(awk '$1 == "0" {printf "moe_log_%s_%s_%s_%s_%s_%s.txt", $2, $3, $4, $5, $6, $7}' tofu_topo.txt)
    cp "$r0" $L 2>/dev/null || cp $L.stdout $L
    dc=$(grep -o "dispatch+combine  naive=[0-9.]*  multiTNI=[0-9.]*  best=[0-9.]*" $L | tail -1)
    tr=$(grep -o "tree all-reduce (decode_estimate assumption)=[0-9.]*" $L | tail -1 | sed 's/.*=//')
    echo "A64FX_COMM np=$NP abytes=$AB batch=$B rc=$rc tree_us=${tr:-NA} ${dc:-dispatch+combine NA}"
  done; done
done
echo "A64FX_COMM DONE dir=$W"
