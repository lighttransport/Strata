#!/usr/bin/env bash
# cfg16.sh NAME S [ENV=VAL ...]: the current best batched-session config (ADAPT tier, dense MMQ for K-quants, wide canonical tier), 24 steps, timing on
cd "$(dirname "$0")"
n=$1 S=$2; shift 2
ADAPT=3000 ./sess.sh $n $S ${LEN:-120} ${STEPS:-24} STRATA_DS41_ADAPT_GAP_MS=${GAPMS:-30} STRATA_DS41_HELPER_CPUS=14,15,30,31 STRATA_DS41_ADAPT_OWNED_WEIGHT=2 \
  STRATA_DS41_DECODE_TIMING=1 STRATA_DS41_DENSE_MMQ=1 STRATA_MMQ_KQ=1 STRATA_DS41_CANON_TOKENS=$S "$@" | head -1 | cut -c1-300
grep -hE "decode_gpu_ms" logs/$n.log | cut -c1-400
