#!/usr/bin/env bash
# Run only when no throughput trial is active. Uses the guarded two-node path.
set -euo pipefail
E=$(cd "$(dirname "$0")/.." && pwd)
name=${1:?usage: run.sh NAME [ENV=VAL ...]}; shift
cd "$E"
[[ "$name" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[[ ! -e "logs/$name.log" && ! -e "cpp_gate/$name.grade.json" ]] || {
  echo 'refusing to overwrite evidence' >&2; exit 2;
}
export DS41_DECODE_THREADS=${DS41_DECODE_THREADS:-14}
extra=()
[[ -z ${DS41_GATE_CANON_MIB:-} ]] || extra+=("STRATA_DS41_CANON_MIB=$DS41_GATE_CANON_MIB")
MODE=own W_BIN="${DS41_WORKER_BIN:-$(cat logs/worker-bin.local)}" W_MAX=56G \
  ./two_node.sh "$name" cpp_gate/prompt_ids.txt 512 \
  "${DS41_GATE_WORKER_ARGS:---threads=16 --weights-dir=/dev/shm --reserve-gib=3 --gpu-reserve-mib=1024}" \
  -- $(cat "${DS41_GATE_ENV_FILE:-best2n.env}") "$@" "${extra[@]}" > "logs/$name.run" 2>&1
grep -q 'guard=none' "logs/$name.guard" || { echo 'guard did not qualify run' >&2; exit 1; }
# Specify the Python environment that supplies tokenizers and jinja2.
"${DS41_GATE_PYTHON:-python3}" cpp_gate/grade.py "logs/$name.log" "cpp_gate/$name.grade.json"
