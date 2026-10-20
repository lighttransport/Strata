#!/usr/bin/env bash
# Two-node DeepSeek-V4.1 decode (docs/DS41F_PLAN.md section 9): strata-glm-tp-worker in DS4.1 mode on WORKER_HOST
# holds FFN rows [fl, ff) of every routed expert in RAM (a /dev/shm file by default, kept for the next run), and
# strata-glm-decode here computes the rest. Checks the things that failed earlier two-node runs first: locked-memory
# limits on both nodes (IB registration), the worker host's free RAM and /dev/shm space, and a live worker left over.
#
#   tools/ds41_remote_tp.sh WORKER_HOST WORKER_BIN ADDR:PORT -- DECODER MODEL.gguf PROMPT [STEPS] [decoder args...]
#
# ADDR is this host's IPoIB address. Environment: DS41_SHARE (worker rows, default 2/9), WORKER_THREADS (15),
# WORKER_DIR (/dev/shm; empty: the worker's anonymous memory, sent again every run), WORKER_RESERVE_GIB (16),
# UCOMM_TIMEOUT_MS (600000); everything else as tools/glm_remote_tp.sh.
set -euo pipefail
if [ $# -lt 5 ]; then sed -n 2,11p "$0"; exit 2; fi
host=$1 bin=$2 addr=$3
shift 3
if [ "$1" != "--" ]; then echo "ds41_remote_tp.sh: expected -- before the decoder command" >&2; exit 2; fi
shift
here=$(cd "$(dirname "$0")" && pwd)

fail() { echo "ds41_remote_tp.sh: $*" >&2; exit 1; }
[ "$(ulimit -l)" = unlimited ] || [ "$(ulimit -l)" -ge 4194304 ] ||
    fail "locked-memory limit here is $(ulimit -l) KiB; IB registration needs more (prlimit or limits.d)"
remote_lock=$(ssh "$host" 'ulimit -l')
[ "$remote_lock" = unlimited ] || [ "$remote_lock" -ge 4194304 ] || fail "locked-memory limit on $host is $remote_lock KiB"
if ssh "$host" "ps -eo comm | grep -qx strata-glm-tp-w"; then
    fail "a strata-glm-tp-worker is already running on $host (stop it by its PID)"
fi
avail=$(ssh "$host" "awk '/MemAvailable/ {print int(\$2 / 1048576)}' /proc/meminfo")
echo "ds41_remote_tp.sh: $host MemAvailable ${avail} GiB, share ${DS41_SHARE:-2/9}" >&2

dir_flag=()
if [ -n "${WORKER_DIR-/dev/shm}" ]; then dir_flag=(--weights-dir="${WORKER_DIR:-/dev/shm}"); fi
export UCOMM_TIMEOUT_MS=${UCOMM_TIMEOUT_MS:-600000}
export WORKER_THREADS=${WORKER_THREADS:-15}
exec "$here/glm_remote_tp.sh" "$host" "$bin" "$addr" "${dir_flag[@]}" --reserve-gib="${WORKER_RESERVE_GIB:-16}" -- \
    "$@" --remote-tp-share="${DS41_SHARE:-2/9}"
