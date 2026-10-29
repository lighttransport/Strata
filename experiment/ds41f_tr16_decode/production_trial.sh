#!/usr/bin/env bash
# trial.sh NAME [ENV=VAL ...]: one two-node c_osd 128-token decode on best2n.env (overridable), B550 rows kept in
# /dev/shm by fingerprint (W_ARGS / W_ENV / PROMPT override); one summary line appended to logs/trials.txt
set -u
E=$(cd "$(dirname "$0")" && pwd); cd "$E"
name=$1; shift
p=${PROMPT:-c_osd}
W=${W_ARGS:-'--threads=16 --weights-dir=/dev/shm --reserve-gib=3 --gpu-reserve-mib=1024'}
export DS41_DECODE_THREADS=${DS41_DECODE_THREADS:-14}
# shared machines: wait (up to 30 min) until neither node runs compilers and both 1-minute loads are below 3
quiet() { local h=$1; local cmd="awk '{print \$1}' /proc/loadavg; pgrep -c -x 'cc1plus|clang++|cc1|nvcc|ptxas|ld' || true"
  if [ "$h" = local ]; then out=$(bash -c "$cmd"); else out=$(ssh -o BatchMode=yes $h "$cmd"); fi
  set -- $out; awk -v l="$1" -v c="${2:-0}" 'BEGIN { exit !(l < 3 && c == 0) }'; }
for i in $(seq 1 180); do
  if [[ -n ${DS41_EXPERIMENT_DEADLINE:-} ]] && (( $(date +%s) + 180 >= DS41_EXPERIMENT_DEADLINE )); then
    echo 'experiment deadline reached before trial' >&2; exit 2
  fi
  quiet local && quiet b550 && break; [ $i = 1 ] && echo "waiting for quiet nodes..." >&2; sleep 10; done
t0=$(date +%s)
MODE=own W_BIN="${DS41_WORKER_BIN:-$(cat logs/worker-bin.local)}" W_MAX=56G ./two_node.sh $name eval/$p.txt 128 "$W" -- $(tr '\n' ' ' < "${DS41_TRIAL_ENV_FILE:-best2n.env}") "$@" > logs/$name.run 2>&1
ref=$( [ $p = c_osd ] && echo base_greedy_osd || echo base_greedy_llama )
same=$( [ "$(grep '^tokens:' logs/$name.log | head -1)" = "$(grep '^tokens:' logs/$ref.log | head -1)" ] && echo same || echo DIFF )
pass=$(grep -E 'decode_gpu_ms' logs/$name.log | grep -oE 'experts=[0-9.]+|lane_wait=[0-9.]+|layers=[0-9.]+|enqueue [0-9.]+|lane [0-9.]+' | tr '\n' ' ')
rem=$(grep -E 'remote_own_decode' logs/$name.log | grep -oE '\(per layer call ms [0-9./ ]+\)')
wk=$(grep -E 'DS41_WORKER decode_requests' logs/$name.run | grep -oE 'total=[0-9.]+|cpu_lane=[0-9.]+|gpu_share=[0-9.]+' | tr '\n' ' ')
ds=$(grep -E 'dspark steps' logs/$name.log | grep -oE 'tokens_per_step=[0-9.]+|draft_s=[0-9.]+|verify_s=[0-9.]+' | tr '\n' ' ')
line="$name $(grep -oE 'decode_tok_s=[0-9.]+' logs/$name.log | head -1) ids=$same $ds| $pass| remote $rem | worker $wk| $(grep -oE 'guard=[a-z_]+' logs/$name.guard | tail -1) wall=$(( $(date +%s) - t0 ))s args: $*"
echo "$line" | tee -a logs/trials.txt
