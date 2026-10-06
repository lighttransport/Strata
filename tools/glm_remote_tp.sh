#!/usr/bin/env bash
# Two-node GLM decode with cross-node expert tensor parallelism (docs/GLM_REMOTE_TP_COMM.md).
# Starts strata-glm-tp-worker on WORKER_HOST over ssh, then runs the decoder command here with --remote-tp. The
# decoder streams the worker's rows of every expert at startup, so the worker host needs no model file. Worker flag
# --weights-dir=/dev/shm keeps the rows in RAM between runs (the next run skips the transfer); SHIP_WORKER=PATH
# copies a local worker binary to WORKER_BIN first (e.g. /dev/shm/strata-glm-tp-worker), so the worker host needs
# nothing on persistent disk.
#
#   tools/glm_remote_tp.sh WORKER_HOST WORKER_BIN ADDR:PORT [worker flags...] -- DECODER [decoder args...]
#
# ADDR is this host's address on the link both nodes use (the IPoIB address for UCOMM_BACKEND=ib); the decoder
# listens there and the worker connects. Environment: UCOMM_BACKEND (default ib), WORKER_THREADS (default 16),
# WORKER_LOG (default remote-tp-worker.log), SHIP_WORKER, WORKER_ENV (extra VAR=value settings for the worker,
# e.g. STRATA_POOL_SPIN_US=20000 so its pool threads do not park between requests).
set -euo pipefail
if [ $# -lt 5 ]; then sed -n 2,14p "$0"; exit 2; fi
host=$1 bin=$2 addr=$3
shift 3
worker_flags=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do worker_flags+=("$1"); shift; done
if [ "${1:-}" != "--" ]; then echo "glm_remote_tp.sh: missing -- before the decoder command" >&2; exit 2; fi
shift
backend=${UCOMM_BACKEND:-ib}
log=${WORKER_LOG:-remote-tp-worker.log}
if [ -n "${SHIP_WORKER:-}" ]; then
    scp -q "$SHIP_WORKER" "$host:$bin.tmp" && ssh "$host" "chmod +x $bin.tmp && mv $bin.tmp $bin"
fi
ssh "$host" "UCOMM_BACKEND=$backend ${WORKER_ENV:-} exec $bin --root=$addr --threads=${WORKER_THREADS:-16} --backend=$backend ${worker_flags[*]:-}" \
    > "$log" 2>&1 &
worker=$!
# A decoder that exits normally sends the worker its shutdown; this covers the decoder failing first.
trap 'kill "$worker" 2>/dev/null || true' EXIT
status=0
UCOMM_BACKEND=$backend "$@" --remote-tp="$addr" || status=$?
wait "$worker" || true
tail -3 "$log" >&2
exit "$status"
