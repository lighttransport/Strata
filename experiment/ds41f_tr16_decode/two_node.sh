#!/usr/bin/env bash
# two_node.sh NAME PROMPT STEPS WORKER_ARGS -- [ENV=VAL ...]: B550 worker in a memory-capped user scope, TR16 decoder
# through run.sh (the guard). WORKER_ARGS is one quoted string of worker flags; W_ENV: worker environment; W_MAX:
# the worker scope's MemoryMax (default 40G); W_BIN (default ~/work/strata-bin/strata-ds41-worker); PORT (29571).
set -u
E=$(cd "$(dirname "$0")" && pwd)
name=$1 prompt=$2 steps=$3 wargs=$4; shift 4; [ "${1:-}" = "--" ] && shift
port=${PORT:-29571} addr=10.10.10.2 wbin=${W_BIN:-'~/work/strata-bin/strata-ds41-worker'}
ssh -o BatchMode=yes b550 'pgrep -x strata-ds41-wor >/dev/null' && { echo "a worker is already running on b550" >&2; exit 1; }
ssh -o BatchMode=yes b550 "awk '/MemAvailable/ {printf \"b550 MemAvailable %.1f GiB\\n\", \$2/1048576}' /proc/meminfo; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; pgrep -c -x xmrig >/dev/null && echo 'b550: xmrig running'"
ssh -n -f -o BatchMode=yes b550 "cd /tmp && XDG_RUNTIME_DIR=/run/user/\$(id -u) setsid nohup systemd-run --user --scope -q -u ds41w-$name-\$\$ -p MemoryMax=${W_MAX:-40G} -p MemorySwapMax=0 \
  env UCOMM_BACKEND=ib UCOMM_TIMEOUT_MS=600000 ${W_ENV:-} $wbin --root=$addr:$port --backend=ib $wargs < /dev/null > /tmp/ds41w-$name.log 2>&1 &" < /dev/null
sleep 2
if [ "${MODE:-tp}" = own ]; then rvar=STRATA_DS41_REMOTE_OWN; else rvar=STRATA_DS41_REMOTE_TP; fi   # MODE=own: STD2 whole experts
"$E/run.sh" "$name" "$prompt" "$steps" $rvar=$addr:$port UCOMM_BACKEND=ib UCOMM_TIMEOUT_MS=600000 "$@"
rc=$?
for i in $(seq 1 30); do ssh -o BatchMode=yes b550 'pgrep -x strata-ds41-wor >/dev/null' || break; sleep 1; done
wpid=$(ssh -o BatchMode=yes b550 'pgrep -x strata-ds41-wor' || true)
if [ -n "$wpid" ]; then echo "worker still running (pid $wpid): stopping it" >&2; ssh -o BatchMode=yes b550 "kill $wpid"; fi
echo "-- worker log:"; ssh -o BatchMode=yes b550 "tail -5 /tmp/ds41w-$name.log"
exit $rc
