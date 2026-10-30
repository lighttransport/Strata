#!/usr/bin/env bash
# pf1.sh NAME [ENV=VAL ...]: a real prefill on the reversed layout: the B550 decoder + a fresh TR16 worker (rev_node.sh run), then STEPS greedy tokens.
#   STREAM=1 (default) starts the worker with --stream-port and the decoder with STRATA_DS41_OWN_STREAM: the worker's experts arrive over IB into the GPU-streamed prefill
#   (add STRATA_DS41_PROMPT_BATCH / STRATA_DS41_STREAM_MIN / STRATA_DS41_PREFILL_HALVES... as ENV=VAL); STREAM=0 is the CPU-lane prefill (the reference).
#   PF_PROMPT_FILE a token-id list, or PF_FILES the names of eval/*.txt lists to join (default c_osd c_ff c_imgui c_llama t_doc2 t_mujoco, ~6.7K tokens)   CTX (7200)   STEPS (16)   SRING / SCOPY: the worker's staging ring and copier threads
#   HELPERS / STREAM_CPU: the worker's helper CPUs and the stream thread's own CPU   THP=1: transparent huge pages for the worker's rows   LANE_SLEEP
#   e.g. PF_PROMPT_FILE=long.txt CTX=12400 ./pf1.sh L12k STRATA_DS41_PROMPT_BATCH=12288 STRATA_DS41_PREFILL_HALVES=1 STRATA_DS41_PREFILL_TILE=2048 STRATA_DS41_ATTN_MMA=1
cd "$(dirname "$0")"
source ./connection.sh
name=$1; shift
pr=pf_prompt
if [ -n "${PF_PROMPT_FILE:-}" ]; then scp -q "$PF_PROMPT_FILE" "$GPU_HOST":/dev/shm/ds41rev-aux/$pr.txt
else for f in ${PF_FILES:-c_osd c_ff c_imgui c_llama t_doc2 t_mujoco}; do cat eval/$f.txt; echo; done | ssh -o BatchMode=yes "$GPU_HOST" "cat > /dev/shm/ds41rev-aux/$pr.txt"; fi
W_ARGS_X="--numa-place --helper-cpus=${HELPERS:-14,15,30,31}"
ENVX=""
if [ "${STREAM:-1}" = 1 ]; then W_ARGS_X="$W_ARGS_X --stream-port=29574 ${STREAM_CPU:+--stream-cpu=$STREAM_CPU}"; ENVX="STRATA_DS41_OWN_STREAM=${GPU_ADDR}:29574"; fi
export W_THREADS=14 W_ARGS="$W_ARGS_X" W_ENV="STRATA_DS41_STREAM_RING=${SRING:-24} STRATA_DS41_STREAM_COPIERS=${SCOPY:-4} STRATA_DS41_WORKER_THP=${THP:-0} STRATA_IQ1S_GRID8=1 STRATA_IQ1S_PAIR=1 STRATA_IQ1S_VIDX=1 STRATA_DS41_LANE_GROUP_EXACT=1 STRATA_DS41_LANE_SKIP_UNUSED=1 STRATA_DS41_LANE_SLEEP=${LANE_SLEEP:-0} STRATA_DS41_LANE_NUMA=1"
PROMPT=$pr STEPS=${STEPS:-16} CTX=${CTX:-7200} KEEP_MIB=34000 ./rev_node.sh run $name STRATA_DS41_CANON_MIB=${CANON:-0} STRATA_DS41_DESKTOP_MIB=512 STRATA_DS41_DSPARK= STRATA_DS41_PIN_GB=0 STRATA_DS41_PIN_EXPERTS=0 STRATA_DS41_POPULATE=0 \
   STRATA_DS41_DECODE_TIMING=1 STRATA_DS41_PREFILL_TIMING=1 $ENVX "$@" > logs/$name.out 2>&1
grep -hE "prompt_tok_s|^tokens:|own_stream|chunk_gpu_ms|chunk_dma|error|ds41:" logs/$name.log | cut -c1-260 | head -12
grep -h "stream experts" logs/$name.worker
