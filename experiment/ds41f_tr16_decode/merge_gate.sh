#!/usr/bin/env bash
# Full-model teacher-forced equality for the alternate direct merge executable.
set -euo pipefail
cd "$(dirname "$0")"
: "${STRATA_DS41_DECODE_BIN:?set the alternate executable}"
extra=()
[[ -z ${DS41_GATE_CANON_MIB:-} ]] || extra+=("STRATA_DS41_CANON_MIB=$DS41_GATE_CANON_MIB")
for prompt in c_osd c_llama; do
  name=${DS41_MERGE_GATE_PREFIX:-t31_merge_gate_v2}_$prompt
  [[ ! -e logs/$name.log ]] || { echo 'refusing to overwrite evidence'; exit 2; }
  reference_root=${DS41_REFERENCE_DIR:-$(cat logs/reference-root.local)}
  reference=$reference_root/ref_$prompt.f32
  [[ -s "$reference" ]] || { echo 'missing frozen reference'; exit 2; }
  MODE=own W_BIN="${DS41_WORKER_BIN:-$(cat logs/worker-bin.local)}" W_MAX=56G \
    ./two_node.sh "$name" "eval/$prompt.txt" 1 \
    "${DS41_GATE_WORKER_ARGS:---threads=16 --weights-dir=/dev/shm --reserve-gib=3 --gpu-reserve-mib=1024}" \
    -- $(sed '/^STRATA_DS41_DSPARK/d' best2n.env) STRATA_DS41_MERGE_DIRECT=1 "${extra[@]}" DS41_DECODE_SAVE_LOGITS=1 \
    DS41_DECODE_QUALITY_CHUNK=5 > "logs/$name.run" 2>&1
  grep -q 'guard=none' "logs/$name.guard"
  candidate=$(sed -n 's/.*teacher_forced_logits=\([^ ]*\).*/\1/p' "logs/$name.guard" | tail -1)
  [[ -s "$candidate" ]] || exit 1
  cmp "$reference" "$candidate"
  echo "$prompt logits byte-identical" | tee "logs/$name.parity"
done
