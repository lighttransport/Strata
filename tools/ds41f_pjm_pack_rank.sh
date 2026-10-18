#!/bin/bash
# tools/ds41f_pjm_pack_rank.sh - one rank of tools/ds41f_pjm_pack.sh (started by mpiexec, one per node).
# Arguments (mpiexec does not forward plain environment variables):
#   DEADLINE JOB CKPT OUT BIN PACKS TRUNK THREADS IMATRIX|none LAYER_SECONDS LD_LIBRARY_PATH|none
# Layers: ranks 0-3 own 4 layers each (0-15), ranks 4-11 own 3 each (16-39); rank 11 also packs the trunk with 8 of its
# threads. Always exits 0 (exit 3 = "more to do" is normal); the real return codes go to the status file.
set -u
DEADLINE=$1 JOB=$2 CKPT=$3 OUT=$4 BIN=$5 PACKS=$6 TRUNK=$7 THREADS=$8 IMATRIX=$9 LAYER_S=${10} LIBS=${11}
[ "$LIBS" != none ] && export LD_LIBRARY_PATH=$LIBS:${LD_LIBRARY_PATH:-}
R=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PJM_RANK:-}}}}
[ -n "$R" ] || { echo "no rank variable"; exit 0; }
if [ "$R" -lt 4 ]; then A=$((4 * R)); B=$((A + 3)); else A=$((16 + 3 * (R - 4))); B=$((A + 2)); fi
ST=$OUT/status/$JOB/rank$(printf %02d "$R").status
LOGP=$OUT/status/$JOB/rank$(printf %02d "$R")
has_trunk=0; [ "$R" = 11 ] && [ "$TRUNK" != none ] && has_trunk=1
st() { echo "$*" >> "$ST"; }
st "SENTINEL DS41F_RANK_BEGIN rank=$R host=$(hostname) layers=$A-$B trunk=$has_trunk bin=$(sha256sum "$BIN" | cut -c1-12) deadline=$DEADLINE"
LOCALDIR=/local/ds41f-pack-$JOB; mkdir -p "$LOCALDIR" 2>/dev/null || LOCALDIR=$OUT/status/$JOB/tmp-$R; mkdir -p "$LOCALDIR"; export TMPDIR=$LOCALDIR
calib=(--uncalibrated)
if [ "$IMATRIX" != none ]; then cp "$IMATRIX" "$LOCALDIR/imatrix.gguf" && calib=("--imatrix=$LOCALDIR/imatrix.gguf") || calib=("--imatrix=$IMATRIX"); fi
pack_args() { local f=${1#*:}; local g=${f%%/*} r=${f#*/}; local u=${r%%/*} d=${r#*/}; echo "--gate=$g --up=$u --down=$d"; }
maxrc=0; tpid=""
if [ $has_trunk = 1 ]; then
    TF=$OUT/trunk-${TRUNK%%/*}-${TRUNK#*/}.gguf
    if [ ! -f "$TF.verified" ]; then
        rm -f "$TF.partial"
        ( t0=$(date +%s)
          if [ -f "$TF" ] || "$BIN" trunk "$CKPT" "$TF" --trunk="${TRUNK%%/*}" --head="${TRUNK#*/}" --threads=8 > "$LOGP.trunk.log" 2>&1; then
              if "$BIN" verify "$CKPT" "$TF" >> "$LOGP.trunk.log" 2>&1; then date -Is > "$TF.verified"; echo "TRUNK_RESULT rc=0 verify=ok seconds=$(( $(date +%s) - t0 ))" >> "$ST"
              else echo "TRUNK_RESULT rc=0 verify=fail" >> "$ST"; fi
          else echo "TRUNK_RESULT rc=1 seconds=$(( $(date +%s) - t0 ))" >> "$ST"; fi ) &
        tpid=$!
    else st "TRUNK_RESULT rc=0 verify=ok already"; fi
fi
T=$THREADS; [ $has_trunk = 1 ] && [ -n "$tpid" ] && T=$((THREADS - 8))
for spec in ${PACKS//+/ }; do
    name=${spec%%:*}; dir=$OUT/experts-$name.d
    for i in $(seq 1 60); do [ -f "$dir/plan.json" ] && break; sleep 5; done
    if [ "$(date +%s)" -gt "$(python3 -c "print(int($DEADLINE - 1.3 * $LAYER_S))")" ]; then st "PACK_RESULT pack=$name rc=skipped reason=deadline"; continue; fi
    t0=$(date +%s)
    "$BIN" experts "$CKPT" --chunk-dir="$dir" $(pack_args "$spec") "${calib[@]}" --layers="$A-$B" --threads="$T" \
        --deadline="$DEADLINE" --layer-seconds="$LAYER_S" > "$LOGP.$name.log" 2>&1
    rc=$?; st "PACK_RESULT pack=$name rc=$rc layers=$A-$B seconds=$(( $(date +%s) - t0 ))"
    grep "PACK layer=.*seconds=" "$LOGP.$name.log" | sed 's/^/  /' >> "$ST"
    [ $rc = 1 ] && maxrc=1; [ $rc = 3 ] && [ $maxrc = 0 ] && maxrc=3
done
[ -n "$tpid" ] && wait "$tpid"
rm -rf "/local/ds41f-pack-$JOB" 2>/dev/null
st "SENTINEL DS41F_RANK_END rank=$R rc=$maxrc"
exit 0
