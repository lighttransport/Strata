#!/bin/bash
#PJM -L "node=12"
#PJM -L "elapse=06:00:00"
#PJM -L "freq=2000,eco_state=0"
#PJM --mpi "proc=12"
#PJM --llio localtmp-size=87Gi
#PJM -j
# tools/ds41f_pjm_pack.sh - DeepSeek V4.1 Flash expert packs (and the trunk) on 12 A64FX nodes in one PJM batch job.
#
# Submit from a published build directory (tools/a64fx_probe/build_ds41f_pack.sh), site values on the command line:
#   pjsub -g GROUP -L rscgrp=RSCGRP -x PJM_LLIO_GFSCACHE=VOL \
#         -x DS41F_CKPT=<checkpoint dir> -x DS41F_OUT=<output dir> -x DS41F_SNAP=<published build dir> \
#         -x DS41F_PACK_SHA256=<binary sha256> -x DS41F_PACKS=q23:q2_k/q2_k/q3_k -x DS41F_TRUNK=q4_k/q8_0 \
#         [-x DS41F_IMATRIX=<imatrix.gguf>] [-x DS41F_THREADS=47] [-x DS41F_LAYER_SECONDS=600] [-x DS41F_CANARY=<file>] \
#         <published build dir>/ds41f_pjm_pack.sh
#
# DS41F_PACKS: name:gate/up/down, several joined with '+' (pjsub -x already uses commas). DS41F_TRUNK: trunk/head
# formats or "none". IQ formats need DS41F_IMATRIX. Each pack goes to $DS41F_OUT/experts-<name>.d, one chunk directory
# shared by all ranks (each layer has exactly one owner); the trunk to $DS41F_OUT/trunk-<trunk>-<head>.gguf (rank 11).
# Resumable: resubmit the same command; finished layers are skipped, an interrupted layer is redone.
# Status: $DS41F_OUT/status/<job>/rankNN.status and SENTINEL lines on stdout; the job ends with
# `SENTINEL DS41F_JOB_END state=complete|resubmit|failed`.
set -uo pipefail
T0=$(date +%s)
JOB=${PJM_JOBID:-manual$T0}
CKPT=${DS41F_CKPT:?DS41F_CKPT}; OUT=${DS41F_OUT:?DS41F_OUT}; SNAP=${DS41F_SNAP:?DS41F_SNAP}
BIN=$SNAP/strata-ds41f-pack; PACKS=${DS41F_PACKS:?DS41F_PACKS}; TRUNK=${DS41F_TRUNK:-none}
IMATRIX=${DS41F_IMATRIX:-}; THREADS=${DS41F_THREADS:-47}; LAYER_S=${DS41F_LAYER_SECONDS:-600}
ELAPSE=${DS41F_ELAPSE_S:-21600}; MARGIN=${DS41F_MARGIN_S:-900}; DEADLINE=$((T0 + ELAPSE - MARGIN))
LOG=$OUT/status/$JOB; mkdir -p "$LOG"
say() { echo "$*"; echo "$*" >> "$LOG/head.log"; }
end() { say "SENTINEL DS41F_JOB_END state=$1 job=$JOB elapsed_s=$(( $(date +%s) - T0 ))"; exit 0; }
say "SENTINEL DS41F_JOB_BEGIN job=$JOB host=$(hostname) packs=$PACKS trunk=$TRUNK threads=$THREADS deadline=$DEADLINE"
# 1. the binary is the published one
if [ -n "${DS41F_PACK_SHA256:-}" ] && [ "$(sha256sum "$BIN" | cut -d' ' -f1)" != "$DS41F_PACK_SHA256" ]; then say "SENTINEL DS41F_STALE_BINARY $BIN"; end failed; fi
# 2. one job at a time per output directory
LOCK=$OUT/.lock
if [ -f "$LOCK" ]; then
    read -r lj le < <(sed -e 's/job=//' -e 's/end=//' "$LOCK")
    if [ "${lj:-}" != "$JOB" ] && [ "${le:-0}" -gt "$(date +%s)" ] && [ "${DS41F_FORCE:-0}" != 1 ]; then say "SENTINEL DS41F_LOCKED by job=$lj until=$le"; end failed; fi
fi
echo "job=$JOB end=$((T0 + ELAPSE))" > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
# 3. the probe's canary (persistence of shared-FS writes across jobs)
if [ -n "${DS41F_CANARY:-}" ] && [ ! -s "$DS41F_CANARY" ]; then say "SENTINEL DS41F_CANARY_MISSING $DS41F_CANARY"; end failed; fi
# 4. plans, created once here so the ranks only validate them
calib=(--uncalibrated); [ -n "$IMATRIX" ] && calib=("--imatrix=$IMATRIX")
pack_args() { local f=${1#*:}; local g=${f%%/*} r=${f#*/}; local u=${r%%/*} d=${r#*/}; echo "--gate=$g --up=$u --down=$d"; }
complete=1
for spec in ${PACKS//+/ }; do
    name=${spec%%:*}; dir=$OUT/experts-$name.d
    if ! "$BIN" experts "$CKPT" --chunk-dir="$dir" $(pack_args "$spec") "${calib[@]}" --layers=0-39 --plan-only >> "$LOG/head.log" 2>&1; then
        say "SENTINEL DS41F_PLAN_FAIL pack=$name"; tail -5 "$LOG/head.log"; end failed; fi
    "$BIN" assemble "$dir" --check --layers=0-39 > /dev/null 2>&1 || complete=0
done
TFILE=""
if [ "$TRUNK" != none ]; then TFILE=$OUT/trunk-${TRUNK%%/*}-${TRUNK#*/}.gguf; [ -f "$TFILE.verified" ] || complete=0; fi
[ $complete = 1 ] && { say "SENTINEL DS41F_ALREADY_COMPLETE"; end complete; }
# 5. the ranks
say "DS41F_MPIEXEC start $(date +%T)"
mpiexec -np 12 -of-proc "$LOG/rank" "$SNAP/ds41f_pjm_pack_rank.sh" "$DEADLINE" "$JOB" "$CKPT" "$OUT" "$BIN" "$PACKS" "$TRUNK" "$THREADS" "${IMATRIX:-none}" "$LAYER_S" "${DS41F_LD_LIBRARY_PATH:-none}"
say "DS41F_MPIEXEC rc=$? end $(date +%T)"
# 6. what got done
ends=$(cat "$LOG"/rank*.status 2>/dev/null | grep -c "SENTINEL DS41F_RANK_END")
[ "$ends" = 12 ] || say "SENTINEL DS41F_RANK_LOST ends=$ends"
failed=0; grep -h "PACK_RESULT\|TRUNK_RESULT" "$LOG"/rank*.status 2>/dev/null | grep -q "rc=1\b\|verify=fail" && failed=1
complete=1
for spec in ${PACKS//+/ }; do
    name=${spec%%:*}; dir=$OUT/experts-$name.d
    if "$BIN" assemble "$dir" --check --layers=0-39 > "$LOG/check-$name.txt" 2>&1; then
        say "SENTINEL DS41F_PACK_COMPLETE pack=$name dir=$dir"; date -Is > "$dir/COMPLETE"
    else complete=0; say "SENTINEL DS41F_PACK_INCOMPLETE pack=$name missing=$(grep -c 'missing layer' "$LOG/check-$name.txt")"; fi
    python3 - "$dir" <<'PY' | while read -r line; do say "$line"; done
import json, pathlib, statistics, sys
d = pathlib.Path(sys.argv[1]); rows = []
for f in sorted(d.glob("*.chunk.json")):
    j = json.loads(f.read_text()); rows.append((j["normalized_mse"], j["name"]))
if rows:
    med = statistics.median(r[0] for r in rows); out = [n for v, n in rows if v > 2 * med]
    print(f"DS41F_NMSE dir={d.name} chunks={len(rows)} median={med:.5f} max={max(r[0] for r in rows):.5f} outliers={','.join(out) or 'none'}")
PY
done
if [ -n "$TFILE" ]; then
    if [ -f "$TFILE.verified" ]; then say "SENTINEL DS41F_TRUNK_COMPLETE file=$TFILE"; else complete=0; say "SENTINEL DS41F_TRUNK_INCOMPLETE"; fi
fi
[ $failed = 1 ] && end failed
[ $complete = 1 ] && end complete
end resubmit
