#!/usr/bin/env bash
# The full-model Hy4-Flash check, for an idle machine: llama.cpp reference dump, strata-hy4f end to end and per block
# (--inject), the comparisons, then decode speed over a small matrix. Every step checks the kernel log for amdgpu
# svm_range work and stops at a kernel BUG / list corruption (TR16 hung once after such work).
#
#   tools/hy4f_full_check.sh MODEL.gguf LLAMA_DUMP OUT_DIR [PROMPT_FILE]
#
# LLAMA_DUMP: tools/ds41f_llama_dump.cpp built against the llama.cpp tree with the hy_v4_flash port (CPU build).
# Environment: RUNNER (default build-hy4f/strata-hy4f), THREADS (15), CTX (4096), TIER_GIB (4), PROMPT_BYTES (12500),
# STEPS=ref,e2e,inject,speed (a subset to run), GEN (64 tokens per speed run).
set -euo pipefail
MODEL=$1; DUMP=$2; OUT=$3; PROMPT_FILE=${4:-docs/DETAILS.md}
RUNNER=${RUNNER:-build-hy4f/strata-hy4f}; THREADS=${THREADS:-15}; CTX=${CTX:-4096}; TIER_GIB=${TIER_GIB:-4}
PROMPT_BYTES=${PROMPT_BYTES:-12500}; STEPS=${STEPS:-ref,e2e,inject,speed}; GEN=${GEN:-64}
export LD_LIBRARY_PATH=${HY4F_ROCM_LIB:-/opt/rocm/core-10.1/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/summary.txt"; }
want() { [[ ",$STEPS," == *",$1,"* ]]; }
kcheck() {   # $1: start time
    local n bad
    n=$(journalctl -k --since "$1" --no-pager 2>/dev/null | grep -c svm_range || true)
    bad=$(journalctl -k --since "$1" --no-pager 2>/dev/null | grep -ciE 'list_del|BUG:|soft lockup' || true)
    log "  kernel log: $n svm_range lines, $bad BUG/list_del/lockup"
    if [ "$bad" != 0 ]; then log "STOP: kernel trouble, see journalctl -k"; exit 3; fi
    if [ "$n" != 0 ]; then log "  WARNING: svm_range work during this step (another GPU job?): repeat it before trusting it"; fi
}
compare() { uv run --no-project --with numpy python -I tools/hy4f_compare.py "$@"; }

if want ref; then
    S=$(date '+%Y-%m-%d %H:%M:%S'); log "llama.cpp reference ($PROMPT_BYTES bytes of $PROMPT_FILE)"
    "$DUMP" "$MODEL" "$OUT/ref" "$(head -c "$PROMPT_BYTES" "$PROMPT_FILE")" --names l_out,result_output > "$OUT/ref.log" 2>&1
    log "  $(wc -l < "$OUT/ref/tokens.txt") tokens"; kcheck "$S"
fi
IDS=$(paste -sd, "$OUT/ref/tokens.txt")
if want e2e; then
    S=$(date '+%Y-%m-%d %H:%M:%S'); log "strata end to end (CPU experts)"
    "$RUNNER" "$MODEL" --ids "$IDS" --dump "$OUT/e2e" --threads "$THREADS" --ctx "$CTX" --n 1 --tier-gib 0 2> "$OUT/e2e.log" > /dev/null
    grep prompt "$OUT/e2e.log" | tee -a "$OUT/summary.txt"; kcheck "$S"
    compare "$OUT/ref" "$OUT/e2e" | tee "$OUT/e2e.cmp" | tail -4 | tee -a "$OUT/summary.txt"
fi
if want inject; then
    S=$(date '+%Y-%m-%d %H:%M:%S'); log "strata per block (--inject)"
    "$RUNNER" "$MODEL" --ids "$IDS" --dump "$OUT/inject" --inject "$OUT/ref" --threads "$THREADS" --ctx "$CTX" --n 1 --tier-gib 0 \
        2> "$OUT/inject.log" > /dev/null
    kcheck "$S"
    compare "$OUT/ref" "$OUT/inject" | tee "$OUT/inject.cmp" | tee -a "$OUT/summary.txt"
fi
if want speed; then
    SHORT=$(head -64 "$OUT/ref/tokens.txt" | paste -sd,)
    run() {   # label, env..., -- args...
        local label=$1; shift; local envs=(); while [ "$1" != -- ]; do envs+=("$1"); shift; done; shift
        S=$(date '+%Y-%m-%d %H:%M:%S')
        env "${envs[@]}" "$RUNNER" "$MODEL" --ids "$SHORT" --n "$GEN" --ignore-eos --ctx 1024 "$@" 2> "$OUT/speed-$label.log" > /dev/null || true
        log "  $label: $(grep 'decode' "$OUT/speed-$label.log" | sed 's/^hy4f: //')"; kcheck "$S"
    }
    log "decode speed, $GEN tokens after a 64-token prompt"
    for rep in 1 2; do
        run "tier$TIER_GIB-t$THREADS-r$rep" X=1 -- --threads "$THREADS" --tier-gib "$TIER_GIB"
        run "notier-t$THREADS-r$rep" X=1 -- --threads "$THREADS" --tier-gib 0
        run "tier$TIER_GIB-t$THREADS-bands-r$rep" STRATA_DS41_LANE_BANDS=1 -- --threads "$THREADS" --tier-gib "$TIER_GIB"
        run "tier$TIER_GIB-t16-r$rep" X=1 -- --threads 16 --tier-gib "$TIER_GIB"
        run "tier$TIER_GIB-t$THREADS-eager-r$rep" STRATA_HY4F_GRAPH=0 -- --threads "$THREADS" --tier-gib "$TIER_GIB"
    done
fi
log "done: $OUT/summary.txt"
