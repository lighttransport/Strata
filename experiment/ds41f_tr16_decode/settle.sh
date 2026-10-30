#!/usr/bin/env bash
# settle.sh MODE NAME S [ENV=VAL ...]: MODE single|dual, S sessions on a tier filled once from the recorded traffic (STRATA_DS41_CANON_SETTLE) and then frozen
cd "$(dirname "$0")"
mode=$1 n=$2 S=$3; shift 3
T=/dev/shm/ds41rev-aux/traffic48.i32
COMMON="STRATA_DS41_ADAPT_GAP_MS=5 STRATA_DS41_ADAPT_OWNED_WEIGHT=2 STRATA_DS41_CANON_SETTLE=$T STRATA_DS41_CANON_TRAFFIC=$T STRATA_DS41_REMOTE_OWN_SKIP_MIB=34000 STRATA_DS41_DENSE_MMQ=1 STRATA_MMQ_KQ=1"
if [ "$mode" = dual ]; then
  H=$((S / 2))
  ADAPT=3000 W_ARGS_OVR="--numa-place --helper-cpus=14,15,30,31 --root2=10.10.10.1:29581 --engram-port2=29582" D_THREADS=${D_THREADS:-8} \
    ./sess.sh $n $S ${LEN:-120} ${STEPS:-24} $COMMON STRATA_DS41_HELPER_CPUS=30,31 STRATA_DS41_DUAL=1 STRATA_DS41_DECODE_TIMING=${DT:-0} STRATA_DS41_CANON_TOKENS=$H "$@" | cut -c1-300
else
  ADAPT=3000 ./sess.sh $n $S ${LEN:-120} ${STEPS:-24} $COMMON STRATA_DS41_HELPER_CPUS=14,15,30,31 STRATA_DS41_DECODE_TIMING=${DT:-0} STRATA_DS41_CANON_TOKENS=$S "$@" | cut -c1-300
fi
grep -hE "DS41 sessions=|canon_settle|canon_traffic|error|ds41:" logs/$n.log | cut -c1-260 | head -8
