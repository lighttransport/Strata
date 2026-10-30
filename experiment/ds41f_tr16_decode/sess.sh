#!/usr/bin/env bash
# sess.sh NAME S LEN STEPS [ENV=VAL ...]: batched plain decode of S sessions with a fresh worker per run (a reused persistent worker slows ~30x after its first run); prints the sessions line.
cd "$(dirname "$0")"
name=$1 S=$2 L=$3 N=$4; shift 4
export W_THREADS=${W_THREADS:-14} W_ARGS="${W_ARGS_OVR:---numa-place --helper-cpus=14,15,30,31}" W_ENV="STRATA_IQ1S_GRID8=1 STRATA_IQ1S_PAIR=1 STRATA_DS41_LANE_GROUP_EXACT=1 STRATA_DS41_LANE_SKIP_UNUSED=1 STRATA_DS41_LANE_SLEEP=0 STRATA_DS41_LANE_NUMA=1 ${W_EXTRA:-}"
PROMPT=${PROMPT:-c_osd} STEPS=$N ./rev_node.sh run $name STRATA_DS41_CANON_MIB=${CANON:-5600} STRATA_DS41_DESKTOP_MIB=512 STRATA_DS41_DSPARK= \
  STRATA_DS41_ATTN_MULTI=0 STRATA_DS41_VERIFY_FAST_TRUNK=0 STRATA_DS41_SESSIONS=$S STRATA_DS41_SESSION_LEN=$L "$@" > logs/$name.out 2>&1
grep -hE "DS41 sessions=.*pass_ms|error|ds41:" logs/$name.log | tail -4 | cut -c1-400
