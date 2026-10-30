#!/usr/bin/env bash
# dual.sh NAME S [ENV=VAL ...]: STRATA_DS41_DUAL=1: two decoders in one process (S/2 sessions each, shared GPU weights and tier, ports +10),
# one worker serving both (--root2/--engram-port2), static tier (no ADAPT). Aggregate tok/s = the two printed decode_tok_s lines added.
cd "$(dirname "$0")"
n=$1 S=$2; shift 2
H=$((S / 2))
W_ARGS_OVR="--numa-place --helper-cpus=14,15,30,31 --root2=10.10.10.1:29581 --engram-port2=29582" \
ADAPT=${ADAPT:-} D_THREADS=${D_THREADS:-8} ./sess.sh $n $S ${LEN:-120} ${STEPS:-24} STRATA_DS41_HELPER_CPUS=30,31 STRATA_DS41_DUAL=1 \
  STRATA_DS41_DECODE_TIMING=${DT:-0} STRATA_DS41_DENSE_MMQ=1 STRATA_MMQ_KQ=1 STRATA_DS41_CANON_TOKENS=$H STRATA_DS41_CANON_TRAFFIC=/dev/shm/ds41rev-aux/traffic48.i32 STRATA_DS41_REMOTE_OWN_SKIP_MIB=34000 "$@" | cut -c1-300
grep -hE "DS41 sessions=|dual|error|ds41" logs/$n.log | cut -c1-260 | head -12
