#!/usr/bin/env bash
# pf1.sh NAME [ENV=VAL ...]: a real prefill on the reversed layout: the B550 decoder + a fresh TR16 worker (rev_node.sh run), prompt PF_PROMPT (name of a .txt in the B550's aux dir,
# default p4k), STEPS generated tokens. STREAM=1 adds the worker's --stream-port and STRATA_DS41_OWN_STREAM (owned experts over IB into the streamed prefill).
cd "$(dirname "$0")"
source ./connection.sh
name=$1; shift
S=/mnt/nvme01/tmp/claude-1000/-mnt-nvme02-work-Strata/808a845c-499a-49b7-86f6-4b9195dd0fa8/scratchpad
pr=${PF_PROMPT:-p4k}
scp -q $S/$pr.txt "$GPU_HOST":/dev/shm/ds41rev-aux/$pr.txt
W_ARGS_X="--numa-place --helper-cpus=${HELPERS:-14,15,30,31}"
ENVX=""
if [ "${STREAM:-1}" = 1 ]; then W_ARGS_X="$W_ARGS_X --stream-port=29574 ${STREAM_CPU:+--stream-cpu=$STREAM_CPU}"; ENVX="STRATA_DS41_OWN_STREAM=${GPU_ADDR}:29574"; fi
export W_THREADS=14 W_ARGS="$W_ARGS_X" W_ENV="STRATA_DS41_STREAM_RING=${SRING:-24} STRATA_DS41_STREAM_COPIERS=${SCOPY:-4} STRATA_DS41_WORKER_THP=${THP:-0} STRATA_IQ1S_GRID8=1 STRATA_IQ1S_PAIR=1 STRATA_IQ1S_VIDX=1 STRATA_DS41_LANE_GROUP_EXACT=1 STRATA_DS41_LANE_SKIP_UNUSED=1 STRATA_DS41_LANE_SLEEP=${LANE_SLEEP:-0} STRATA_DS41_LANE_NUMA=1"
PROMPT=$pr STEPS=${STEPS:-16} CTX=${CTX:-4700} KEEP_MIB=34000 ./rev_node.sh run $name STRATA_DS41_CANON_MIB=${CANON:-0} STRATA_DS41_DESKTOP_MIB=512 STRATA_DS41_DSPARK= STRATA_DS41_PIN_GB=0 STRATA_DS41_PIN_EXPERTS=0 STRATA_DS41_POPULATE=0 \
   STRATA_DS41_DECODE_TIMING=1 STRATA_DS41_PREFILL_TIMING=1 $ENVX "$@" > logs/$name.out 2>&1
grep -hE "prompt_tok_s|^tokens:|own_stream|chunk_gpu_ms|chunk_dma|error|ds41:" logs/$name.log | cut -c1-260 | head -12
grep -h "stream experts" logs/$name.worker
