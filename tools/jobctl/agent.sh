#!/bin/bash
# Node-side agent of jobctl, started by job.sh on node 0 of the allocation.
#
# Every CTL_INTERVAL seconds it:
#   - keeps the bash-over-HTTP server and the reverse tunnel up (re-picks the frontend port if one is taken),
#   - writes state/info.json (heartbeat, port, phase) and state/port.<jobid>,
#   - honours flags/STOP, flags/PAUSE and flags/DRAIN,
#   - runs inbox/<seq>.cmd files one at a time and writes outbox/<seq>.out and .rc,
#   - asks for a successor allocation (chain/next.req) when walltime runs low or DRAIN is set,
#   - sends SIGTERM to the workload shortly before the walltime ends so it can checkpoint.
# The workload is $CTL_RUNDIR/work.sh (optional). It resumes from $CTL_WORK. Exit codes:
#   0 = finished (chain stops), 75 = not finished (continue in the next allocation), other = failed (chain stops).
#
# Several allocations of one run may be queued at once (racing variants, successors queued ahead). Only the
# holder of chain/active runs the workload:
#   - a racer (CTL_RACE=1) that finds the lock taken exits at once; the first racer to start wins;
#   - a successor waits until the holder hands over (the holder sees chain/up.<seq>, sends SIGTERM to the
#     workload, waits for its checkpoint, releases the lock and exits) or until the holder's heartbeat is stale.
#
# Test switches: CTL_NO_SERVER=1, CTL_NO_TUNNEL=1, CTL_MAX_TICKS=N. CTL_TUNNEL=0 turns the HTTP bridge off.
set -u
umask 077
RUN=${CTL_RUNDIR:?CTL_RUNDIR not set}
SEQ=${CTL_SEQ:-1}
JOB=${PJM_JOBID:-nojob}
INTERVAL=${CTL_INTERVAL:-30}
LEAD=$(( ${CTL_LEAD_MIN:-60} * 60 ))
GRACE=${CTL_GRACE:-300}
ELAPSE=${CTL_ELAPSE_SEC:-3600}
CHAIN=${CTL_CHAIN:-1}
CMD_TIMEOUT=${CTL_CMD_TIMEOUT:-0}          # 0 = no limit per inbox command
HEALTH_EVERY=${CTL_HEALTH_EVERY:-4}
SERVER_PORT=${SERVER_PORT:-21264}
FLO=${CTL_FPORT_LO:-33000}
FHI=${CTL_FPORT_HI:-33599}
TARGET=${FRONTEND_SSH_TARGET:-}
BUNDLE=${CTL_BUNDLE:-}
MAX_TICKS=${CTL_MAX_TICKS:-0}
STALE=${CTL_STALE_SEC:-600}     # heartbeat age after which a holder counts as dead (must exceed the file-system lag)
RACE=${CTL_RACE:-0}
VARIANT=${CTL_VARIANT:-0}
START=$(date +%s)
[ "${CTL_TUNNEL:-1}" = 0 ] && { CTL_NO_SERVER=1; CTL_NO_TUNNEL=1; }
export CTL_WORK=$RUN/work CTL_SEQ=$SEQ CTL_NODES=${CTL_NODES:-1}
mkdir -p "$RUN"/{state,flags,inbox,outbox,chain,work,logs}
LOG=$RUN/logs/agent.$SEQ.log
exec >>"$LOG" 2>&1
ts() { date -u +%FT%TZ; }
say() { echo "$(ts) $*"; }

SSH_OPTS=(-n -o BatchMode=yes -o ConnectTimeout=5 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30
          -o ServerAliveCountMax=3 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
          -o UserKnownHostsFile="$RUN/state/known_hosts")

PHASE=starting; SERVER_PID=; TUNNEL_PID=; WORK_PID=; CMD_PID=; REQUESTED=0; TERMED=0; FPORT=0; TICK=0

latest_port() {   # newest allocation's port file wins
  local f; f=$(ls "$RUN"/state/port.* 2>/dev/null | sort -V | tail -1)
  [ -n "$f" ] && cat "$f" || echo "$FLO"
}

# On the shared file system a file created by another node can stay invisible to `test -e` / `cat` for minutes
# (measured: a STOP flag written from the login node was seen after about 2.5 minutes). A directory listing is fresh.
# So every cross-node check below lists the directory, and heartbeats are encoded in new file NAMES.
has() { ls "$1" 2>/dev/null | grep -qx -- "$2"; }          # has DIR NAME
newest() { ls "$1" 2>/dev/null | grep -- "^$2\." | sort -t. -k3 -n | tail -1; }   # newest DIR PREFIX  (prefix.<seq>.<epoch>)

atomic() { # atomic FILE  (content on stdin)
  local t="$1.tmp$$"; cat >"$t" && mv "$t" "$1"
}

write_info() {
  local now rem; now=$(date +%s); rem=$(( ELAPSE - (now - START) ))
  printf '{"name":"%s","jobid":"%s","seq":%s,"host":"%s","phase":"%s","hb":%s,"remaining":%s,"frontend_port":%s,"server_port":%s,"target":"%s","workload":"%s"}\n' \
    "${CTL_NAME:-}" "$JOB" "$SEQ" "$(hostname)" "$PHASE" "$now" "$rem" "$FPORT" "$SERVER_PORT" "$TARGET" \
    "$([ -n "$WORK_PID" ] && kill -0 "$WORK_PID" 2>/dev/null && echo running || echo none)" | atomic "$RUN/state/info.json"
  echo "$FPORT" | atomic "$RUN/state/port.$JOB"
}

health_local() {
  python3 - "$SERVER_PORT" >/dev/null 2>&1 <<'PY'
import json, sys, urllib.request
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%d/health" % int(sys.argv[1])), timeout=2)
sys.exit(0 if json.loads(r.read().decode()).get("ok") else 1)
PY
}

health_tunnel() {   # from the login node, through the forwarded port
  ssh "${SSH_OPTS[@]}" "$TARGET" "python3 - $FPORT" >/dev/null 2>&1 <<'PY'
import json, sys, urllib.request
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%d/health" % int(sys.argv[1])), timeout=5)
sys.exit(0 if json.loads(r.read().decode()).get("ok") else 1)
PY
}

start_server() {
  [ -n "${CTL_NO_SERVER:-}" ] && return 0
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  python3 "$BUNDLE/bash_http_server.py" --host 127.0.0.1 --port "$SERVER_PORT" >>"$RUN/logs/server.$SEQ.log" 2>&1 &
  SERVER_PID=$!
  local i; for i in $(seq 1 50); do health_local && return 0; sleep 0.2; done
  say "server did not come up"; return 1
}

open_tunnel() {   # try the current port, then the next ones in the range
  [ -n "${CTL_NO_TUNNEL:-}" ] && return 0
  local size=$(( FHI - FLO + 1 )) i p
  [ -n "$TUNNEL_PID" ] && { kill "$TUNNEL_PID" 2>/dev/null; wait "$TUNNEL_PID" 2>/dev/null; }
  TUNNEL_PID=
  for i in $(seq 0 $(( size - 1 ))); do
    p=$(( FLO + ( FPORT - FLO + i ) % size ))
    ssh -N "${SSH_OPTS[@]}" -R "127.0.0.1:${p}:127.0.0.1:${SERVER_PORT}" "$TARGET" >>"$RUN/logs/tunnel.$SEQ.log" 2>&1 &
    TUNNEL_PID=$!
    sleep 2
    if kill -0 "$TUNNEL_PID" 2>/dev/null; then
      [ "$p" != "$FPORT" ] && say "frontend port $FPORT -> $p"
      FPORT=$p; return 0
    fi
    wait "$TUNNEL_PID" 2>/dev/null; TUNNEL_PID=
    [ "$i" -ge 20 ] && break
  done
  say "no frontend port could be opened"; return 1
}

run_inbox() {   # start the oldest unanswered command if none is running
  [ -n "$CMD_PID" ] && kill -0 "$CMD_PID" 2>/dev/null && return
  CMD_PID=
  local f s
  for f in $(ls "$RUN"/inbox/*.cmd 2>/dev/null | sort); do
    s=$(basename "$f" .cmd)
    [ -e "$RUN/outbox/$s.rc" ] && continue
    say "inbox $s"
    (
      cd "$RUN/work" || exit 1
      if [ "$CMD_TIMEOUT" -gt 0 ]; then timeout "$CMD_TIMEOUT" bash "$f" >"$RUN/outbox/$s.out.run" 2>&1
      else bash "$f" >"$RUN/outbox/$s.out.run" 2>&1; fi
      rc=$?
      mv "$RUN/outbox/$s.out.run" "$RUN/outbox/$s.out"
      echo "$rc" >"$RUN/outbox/$s.rc.run" && mv "$RUN/outbox/$s.rc.run" "$RUN/outbox/$s.rc"
    ) &
    CMD_PID=$!
    return
  done
}

start_workload() {
  [ -f "$RUN/work.sh" ] || return 0
  has "$RUN/work" DONE && return 0
  say "workload start (seq $SEQ)"
  bash "$RUN/work.sh" >>"$RUN/logs/work.$SEQ.log" 2>&1 &
  WORK_PID=$!
}

release_active() { rm -f "$RUN"/chain/active.$SEQ.* 2>/dev/null; }

touch_active() {   # new name each tick: active.<seq>.<epoch>; the old one is removed afterwards
  local old; old=$(ls "$RUN/chain" 2>/dev/null | grep "^active\.$SEQ\.")
  : >"$RUN/chain/active.$SEQ.$(date +%s)"
  for f in $old; do rm -f "$RUN/chain/$f"; done
}

active_stale() {   # true when there is no holder, or its newest heartbeat name is old
  local n t; n=$(newest "$RUN/chain" active)
  [ -z "$n" ] && return 0
  t=${n##*.}
  [ $(( $(date +%s) - t )) -gt $STALE ]
}

try_take() {   # short critical section under a mkdir lock (mkdir is decided by the file server)
  mkdir "$RUN/chain/take" 2>/dev/null || return 1
  if active_stale; then
    touch_active
    if [ "$RACE" = 1 ] || ! has "$RUN/chain" winner; then echo "$SEQ $VARIANT" | atomic "$RUN/chain/winner"; fi
    rmdir "$RUN/chain/take"; return 0
  fi
  rmdir "$RUN/chain/take"; return 1
}

handoff() {   # a successor is up: stop the workload at a checkpoint, release the lock, leave
  say "handoff to a successor"
  if [ -n "$WORK_PID" ] && kill -0 "$WORK_PID" 2>/dev/null; then
    kill -TERM "$WORK_PID" 2>/dev/null
    local i; for i in $(seq 1 "$GRACE"); do kill -0 "$WORK_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$WORK_PID" 2>/dev/null
    wait "$WORK_PID" 2>/dev/null; WORK_PID=
  fi
  PHASE=handed-off; write_info; release_active; exit 0
}

cleanup() {
  release_active
  [ -n "$WORK_PID" ] && kill "$WORK_PID" 2>/dev/null
  [ -n "$TUNNEL_PID" ] && kill "$TUNNEL_PID" 2>/dev/null
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

request_successor() {
  [ "$REQUESTED" = 1 ] && return
  [ "$CHAIN" = 1 ] || return
  echo "$(date +%s) seq=$SEQ job=$JOB" | atomic "$RUN/chain/next.req"
  REQUESTED=1; say "successor requested"
}

while ! try_take; do
  if has "$RUN/flags" STOP || has "$RUN/work" DONE; then say "run stopped or done; leaving"; exit 0; fi
  if [ "$RACE" = 1 ]; then say "lost the race; leaving"; exit 0; fi
  : >"$RUN/chain/up.$SEQ.$(date +%s)"; for f in $(ls "$RUN/chain" | grep "^up\.$SEQ\." | sort -t. -k3 -n | head -n -1); do rm -f "$RUN/chain/$f"; done
  [ "${CTL_MAX_TICKS:-0}" -gt 0 ] && [ -n "${CTL_WAIT_LIMIT:-}" ] && { WAITED=$(( ${WAITED:-0} + 1 )); [ "$WAITED" -ge "$CTL_WAIT_LIMIT" ] && exit 0; }
  sleep "${CTL_WAIT_POLL:-10}"
done
rm -f "$RUN"/chain/up.$SEQ.*
FPORT=$(latest_port)
say "agent start seq=$SEQ job=$JOB target=$TARGET port=$FPORT elapse=$ELAPSE"
start_server || PHASE=server-failed
open_tunnel || PHASE=tunnel-failed
[ "$PHASE" = starting ] && PHASE=running
start_workload
write_info

while :; do
  sleep "$INTERVAL"
  TICK=$(( TICK + 1 ))
  now=$(date +%s); rem=$(( ELAPSE - (now - START) ))

  if has "$RUN/flags" STOP; then PHASE=stopped; write_info; say "STOP flag"; exit 0; fi
  touch_active

  # server and tunnel
  if [ -z "${CTL_NO_SERVER:-}" ] && ! health_local; then say "server unhealthy, restarting"; start_server; fi
  if [ -z "${CTL_NO_TUNNEL:-}" ]; then
    down=0
    kill -0 "$TUNNEL_PID" 2>/dev/null || down=1
    [ "$down" = 0 ] && [ $(( TICK % HEALTH_EVERY )) = 0 ] && ! health_tunnel && { say "tunnel probe failed"; down=1; }
    [ "$down" = 1 ] && { say "tunnel down"; open_tunnel; }
  fi

  # workload state
  if [ -n "$WORK_PID" ] && ! kill -0 "$WORK_PID" 2>/dev/null; then
    wait "$WORK_PID"; rc=$?; WORK_PID=
    say "workload exit rc=$rc"
    case "$rc" in
      0)  touch "$RUN/work/DONE"; PHASE=done; write_info; exit 0 ;;
      75) request_successor; PHASE=suspended; write_info; release_active
          say "workload saved a checkpoint (rc 75); leaving so the nodes are freed"; exit 0 ;;
      *)  PHASE=failed; touch "$RUN/flags/STOP"; write_info; exit 1 ;;
    esac
  fi

  # a successor is up: hand over (after the workload-exit check above, so a finished run is not handed on)
  for u in $(ls "$RUN/chain" 2>/dev/null | grep '^up\.'); do
    n=${u#up.}; n=${n%%.*}; t=${u##*.}
    [ "$n" -gt "$SEQ" ] 2>/dev/null || continue
    [ $(( $(date +%s) - t )) -le $STALE ] && handoff
  done

  # lead time, drain, grace
  if [ "$rem" -lt "$LEAD" ] || has "$RUN/flags" DRAIN; then request_successor; fi
  if [ "$rem" -lt "$GRACE" ] && [ "$TERMED" = 0 ] && [ -n "$WORK_PID" ]; then
    say "grace: SIGTERM to workload"; kill -TERM "$WORK_PID" 2>/dev/null; TERMED=1
  fi

  # inbox
  if ! has "$RUN/flags" PAUSE && ! has "$RUN/flags" DRAIN; then run_inbox; fi
  has "$RUN/flags" PAUSE && PHASE=paused || PHASE=running
  write_info
  [ "$MAX_TICKS" -gt 0 ] && [ "$TICK" -ge "$MAX_TICKS" ] && { say "max ticks"; exit 0; }
done
