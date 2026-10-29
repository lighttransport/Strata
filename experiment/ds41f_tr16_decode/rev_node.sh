#!/usr/bin/env bash
# rev_node.sh feed|run|stop NAME [ENV=VAL ...]: the decoder on the B550 (RTX 5060 Ti), the CPU worker on this node (no GPU).
#   feed: strata-ds41-feed shards the model into the B550's /dev/shm over ucomm (dense tensors + the hottest KEEP_MIB of experts)
#   run : strata-ds41-cpu-worker here (owns every other expert, rows read from the local model) + strata-glm-decode on the B550
# KEEP_MIB (default 26000) must be the same in both steps; PROMPT, STEPS, CTX, B_BIN override the defaults. No node's other GPU is used.
set -u
E=$(cd "$(dirname "$0")" && pwd); R=$(cd "$E/../.." && pwd); cd "$E"
step=$1 name=${2:-rev}; shift 2 || true
B=10.10.10.1 PORT_FEED=${PORT_FEED:-29601} PORT_MAIN=${PORT_MAIN:-29571} PORT_ENGRAM=${PORT_ENGRAM:-29572} PORT_FETCH=${PORT_FETCH:-29573}
KEEP_MIB=${KEEP_MIB:-26000}; DIR=/dev/shm/ds41rev; AUX=/dev/shm/ds41rev-aux
M=$(cat logs/model.local); PROFILE=$R/experiment/ds41f_tr16_decode/prof/routes_code4k.i32
DSPARK=$(grep '^STRATA_DS41_DSPARK=' best2n.env | cut -d= -f2-)
BBIN=${B_BIN:-'~/work/strata-own-build'}; TBIN=${T_BIN:-$R/build-glm}
case $step in
feed)
  ssh -o BatchMode=yes b550 "pgrep -x strata-ds41-fee >/dev/null" && { echo "a feeder is already running on b550" >&2; exit 1; }
  ssh -o BatchMode=yes b550 "mkdir -p $AUX; awk '/MemAvailable/ {printf \"b550 MemAvailable %.1f GiB\\n\", \$2/1048576}' /proc/meminfo; df -h /dev/shm | tail -1"
  scp -q "$PROFILE" "eval/${PROMPT:-c_osd}.txt" b550:$AUX/
  ssh -n -f -o BatchMode=yes b550 "XDG_RUNTIME_DIR=/run/user/\$(id -u) setsid nohup systemd-run --user --scope -q -u ds41feed-$name -p MemoryMax=52G -p MemorySwapMax=0 $BBIN/strata-ds41-feed recv --listen $B:$PORT_FEED --dir $DIR --backend=ib < /dev/null > /tmp/ds41feed-$name.log 2>&1 &"
  sleep 3
  extra=(); [ -n "$DSPARK" ] && extra=(--extra "$DSPARK")
  "$TBIN/strata-ds41-feed" serve --root $B:$PORT_FEED --model "$M" --profile "$PROFILE" --keep-mib $KEEP_MIB "${extra[@]}" --backend=ib 2>&1 | sed 's#/[^ ]*/##g'
  sleep 2; ssh -o BatchMode=yes b550 "tail -3 /tmp/ds41feed-$name.log; du -sh $DIR | tail -1; free -g | sed -n 2p"
  ;;
run)
  prompt=${PROMPT:-c_osd}; steps=${STEPS:-128}
  ssh -o BatchMode=yes b550 "pgrep -x strata-glm-deco >/dev/null" && { echo "a decoder is already running on b550" >&2; exit 1; }
  pgrep -x strata-ds41-cpu >/dev/null && { echo "a cpu worker is already running here" >&2; exit 1; }
  first=$(basename "$M"); dsp=$(basename "$DSPARK")
  envs=$(sed -e "s#^STRATA_DS41_CANON_PROFILE=.*#STRATA_DS41_CANON_PROFILE=$AUX/routes_code4k.i32#" -e '/^STRATA_DS41_REMOTE_OWN/d' -e '/^STRATA_DS41_DSPARK=/d' -e '/^STRATA_DS41_DECODE_TIMING/d' best2n.env | tr '\n' ' ')
  envs="$envs STRATA_DS41_DSPARK=$DIR/$dsp STRATA_DS41_REMOTE_OWN=$B:$PORT_MAIN STRATA_DS41_REMOTE_OWN_MIB=200000 STRATA_DS41_REMOTE_OWN_SKIP_MIB=$KEEP_MIB"
  [ -n "${PROMOTE:-}" ] && envs="$envs STRATA_DS41_OWN_PROMOTE=$PROMOTE STRATA_DS41_OWN_FETCH=$B:$PORT_FETCH"
  envs="$envs STRATA_DS41_ENGRAM_REMOTE=$B:$PORT_ENGRAM STRATA_DS41_ENGRAM_DIRECT=0 STRATA_DS41_VERIFY_EXACT=1 STRATA_DS41_TIER_PROMOTE=0 STRATA_DS41_GPU_EXPERT_MIB=0 STRATA_DS41_POPULATE=0 STRATA_DS41_PIN_GB=0 STRATA_DS41_RAM_COPY=0 STRATA_DS41_DSPARK_STREAM_EXPERTS=1 STRATA_DS41_DSPARK_PIN_EXPERTS=0"
  envs="$envs STRATA_DS41_LANE_SKIP_UNUSED=1 STRATA_DS41_LANE_GROUP_EXACT=1 STRATA_DS41_LANE_SLEEP=0 STRATA_IQ1S_PAIR=1 STRATA_IQ1S_GRID8=1 UCOMM_BACKEND=ib UCOMM_TIMEOUT_MS=600000 $*"
  # the worker first (it opens its Engram context, then waits for the decoder's contexts)
  XDG_RUNTIME_DIR=/run/user/$(id -u) systemd-run --user --scope -q -u ds41cpuw-$name -p MemoryMax=${W_MAX:-100G} -p MemorySwapMax=0 \
    ${W_NUMA:+numactl --interleave=all} ${W_ENV:+env $W_ENV} "$TBIN/strata-ds41-cpu-worker" --root=$B:$PORT_MAIN --model="$M" --threads=${W_THREADS:-14} --backend=ib --engram-port=$PORT_ENGRAM ${PROMOTE:+--fetch-port=$PORT_FETCH} ${W_ARGS:-} > logs/$name.worker 2>&1 &
  wpid=$!
  ssh -o BatchMode=yes b550 "cd /tmp && XDG_RUNTIME_DIR=/run/user/\$(id -u) systemd-run --user --scope -q -u ds41rev-$name -p MemoryMax=${D_MAX:-14G} -p MemorySwapMax=0 env $envs $BBIN/strata-glm-decode $DIR/$first @$AUX/$prompt.txt $steps --context=${CTX:-1400} --threads=${D_THREADS:-16} > /tmp/ds41rev-$name.log 2>&1"
  rc=$?
  ssh -o BatchMode=yes b550 "cat /tmp/ds41rev-$name.log" > logs/$name.log
  for i in $(seq 1 30); do pgrep -x strata-ds41-cpu >/dev/null || break; sleep 1; done
  pgrep -x strata-ds41-cpu >/dev/null && { echo "cpu worker still running: stopping it" >&2; pkill -x strata-ds41-cpu; }
  echo "== $name rc=$rc"; grep -hoE 'prompt_tok_s=[0-9.]+|decode_tok_s=[0-9.]+|tokens_per_step=[0-9.]+' logs/$name.log | tr '\n' ' '; echo; tail -3 logs/$name.worker
  ;;
stop)
  ssh -o BatchMode=yes b550 "pkill -x strata-glm-deco; pkill -x strata-ds41-fee" ; pkill -x strata-ds41-cpu; true
  ;;
*) echo "usage: $0 feed|run|stop NAME [ENV=VAL ...]" >&2; exit 2;;
esac
