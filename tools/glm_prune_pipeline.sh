#!/usr/bin/env bash
# REAP-style pruning of GLM-5.3-Flash routed experts 288 -> 224 on TR16, then held-out evaluation.
# Steps (each skipped if its output already exists, so the script can be re-run after a failure):
#   1. assemble Q22 single file (UD-Q2_K_XL base + calibrated experts-q22.gguf)
#   2. saliency calibration on the UD base (new: routing gate sums -> gates.json)
#   3. rank experts (tools/glm_reap_rank.py) -> plan
#   4. prune Q23 and Q22 assembled files
#   5. held-out evaluation (perplexity, KL, top-1 vs the original model's logits) for each pruned file
# Every location can be overridden from the environment; the defaults are the TR16 layout of 2026-10-11.
set -euo pipefail
cd "$(dirname "$0")/.."
B=${GLM_BUILD:-build-glm-release}                     # strata-glm-decode, -assemble, -prune-experts
PY=${GLM_PY:-build-exl3-venv/bin/python}             # runs tools/glm_q2_run_guard.py
GPY=${GLM_GGUF_PY_PYTHON:-python3}                   # runs tools/glm_reap_rank.py (numpy + gguf-py, see STRATA_GGUF_PY)
G=${GLM_DIR:-/mnt/nvme01/models/glm53f}
W=${GLM_PRUNE_DIR:-$G/prune}
MODEL=${GLM_MODEL:-$G/q2/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf}
Q23=${GLM_Q23:-$G/assembled/GLM-5.3-Flash-Q23-assembled.gguf}
Q22PACK=${GLM_Q22_PACK:-$G/q2-cpu-experiment/experts-q22.gguf}
Q22=$W/GLM-5.3-Flash-Q22-assembled.gguf
CORPUS=${GLM_CORPUS:-build-q2-redesign/corpus}       # tools/glm_q2_corpus.py output
BASELINE=${GLM_BASELINE_LOGITS:-$G/q2-cpu-experiment/eval-baseline.logits}
KEEP=${KEEP:-224}
LOG=${GLM_PRUNE_LOG:-build-prune}
mkdir -p "$W" "$LOG"
DECODE_ARGS=(0 1 4096 15 --context=8192 --prefill-batch=256 --gpu-budget-mib=12288 --cpu-affinity=numa --decode-graphs)

step() { echo "=== $(date -Is) $*"; }

if [ ! -f "$Q22" ]; then
  step "assemble Q22"
  $B/strata-glm-assemble "$MODEL" "$Q22PACK" "$Q22" 2>&1 | tee "$LOG/assemble-q22.log"
fi

if [ ! -f "$W/calibration/gates.json" ]; then
  step "saliency calibration"
  STRATA_GLM_CALIBRATION_DIR="$W/calibration" STRATA_GLM_LOCK_RUNTIME=1 \
    $PY tools/glm_q2_run_guard.py --output "$LOG/calibration-run" -- \
    numactl --interleave=all $B/strata-glm-decode "$MODEL" "${DECODE_ARGS[@]}" \
    --eval-corpus="$CORPUS/calibration.ids" > "$LOG/calibration.log" 2>&1
  test -f "$W/calibration/gates.json"
fi

PLAN=$W/plan-keep$KEEP.txt
if [ ! -f "$PLAN" ]; then
  step "rank"
  $GPY -I tools/glm_reap_rank.py "$Q23" "$W/calibration" --keep "$KEEP" --out "$PLAN" --report "$W/plan-keep$KEEP.report.json" --jobs 16
fi

for q in Q23 Q22; do
  src=${!q}
  out=$W/GLM-5.3-Flash-$q-REAP$KEEP-assembled.gguf
  if [ ! -f "$out" ]; then
    step "prune $q"
    $B/strata-glm-prune-experts "$src" "$PLAN" "$out" 2>&1 | tee "$LOG/prune-$q.log"
  fi
  if [ ! -f "$LOG/eval-$q-REAP$KEEP.log" ]; then
    step "evaluate $q-REAP$KEEP"
    $PY tools/glm_q2_run_guard.py --output "$LOG/eval-$q-REAP$KEEP-run" -- \
      numactl --interleave=all $B/strata-glm-decode "$out" "${DECODE_ARGS[@]}" \
      --eval-corpus="$CORPUS/evaluation.ids" --eval-reference="$BASELINE" > "$LOG/eval-$q-REAP$KEEP.log" 2>&1
  fi
done
step "done"
grep -hE "perplexity|KL|top-1|ppl" "$LOG"/eval-*.log | tail -20
