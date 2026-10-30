#!/usr/bin/env bash
# static48.sh NAME S [ENV=VAL ...]: single decoder, static traffic-ranked tier (no ADAPT): the baseline for dual.sh with the same tier
cd "$(dirname "$0")"
n=$1 S=$2; shift 2
./sess.sh $n $S ${LEN:-120} ${STEPS:-24} STRATA_DS41_HELPER_CPUS=14,15,30,31 STRATA_DS41_DENSE_MMQ=1 STRATA_MMQ_KQ=1 STRATA_DS41_CANON_TOKENS=$S \
  STRATA_DS41_CANON_TRAFFIC=/dev/shm/ds41rev-aux/traffic48.i32 STRATA_DS41_REMOTE_OWN_SKIP_MIB=34000 "$@" | cut -c1-300
grep -hE "DS41 sessions=|canon_traffic|error|ds41:" logs/$n.log | cut -c1-260 | head -6
