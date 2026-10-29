#!/usr/bin/env bash
# Report six-prompt quality metrics for a two-node candidate. Metrics require review; this script does not sign off.
set -euo pipefail
cd "$(dirname "$0")"
name=${1:?usage: quality_2n.sh NAME ENV=VALUE ...}; shift
[[ "$name" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
report=logs/quality_$name.txt
[[ ! -e "$report" ]] || { echo 'refusing to overwrite evidence'; exit 2; }
for prompt in c_osd c_imgui c_llama c_ff t_mujoco t_doc2; do
  if [[ -n ${DS41_EXPERIMENT_DEADLINE:-} ]] && (( $(date +%s) + 180 >= DS41_EXPERIMENT_DEADLINE )); then
    echo 'deadline: quality assessment incomplete' | tee -a "$report"; exit 1
  fi
  trial=q_${name}_$prompt
  [[ ! -e logs/$trial.log ]] || exit 2
  reference_root=${DS41_REFERENCE_DIR:-$(cat logs/reference-root.local)}
  reference=$reference_root/ref_$prompt.f32
  [[ -s "$reference" ]] || { echo 'missing frozen reference'; exit 2; }
  MODE=own W_BIN="${DS41_WORKER_BIN:-$(cat logs/worker-bin.local)}" W_MAX=56G \
    ./two_node.sh "$trial" "eval/$prompt.txt" 1 \
    "${DS41_GATE_WORKER_ARGS:---threads=16 --weights-dir=/dev/shm --reserve-gib=3 --gpu-reserve-mib=1024}" \
    -- $(sed '/^STRATA_DS41_DSPARK/d' best2n.env) "$@" \
    DS41_DECODE_SAVE_LOGITS=1 DS41_DECODE_QUALITY_CHUNK=5 > "logs/$trial.run" 2>&1
  grep -q 'guard=none' "logs/$trial.guard"
  candidate=$(sed -n 's/.*teacher_forced_logits=\([^ ]*\).*/\1/p' "logs/$trial.guard" | tail -1)
  [[ -s "$candidate" && $(stat -c %s "$candidate") == $(stat -c %s "$reference") ]] || exit 1
  echo "$prompt $(./cmp_gate "$reference" "$candidate" "eval/$prompt.txt")" | tee -a "$report"
done
