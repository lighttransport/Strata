#!/bin/bash
# tools/ds41f_pjm_pack_rank.sh - one rank of tools/ds41f_pjm_pack.sh (started by mpiexec, one per node).
# Arguments (mpiexec does not forward plain environment variables):
#   DEADLINE JOB CKPT OUT BIN PACKS TRUNK THREADS IMATRIX|none LAYER_SECONDS LD_LIBRARY_PATH|none [LOCAL_ROOT|none PUT_STREAMS]
# Layers: ranks 0-3 own 4 layers each (0-15), ranks 4-11 own 3 each (16-39); rank 11 also packs the trunk with 8 of its
# threads. Always exits 0 (exit 3 = "more to do" is normal); the real return codes go to the status file.
# With LOCAL_ROOT (the strata-pfetch mirror of ds41f_pjm_pack.sh: LOCAL_ROOT/rR/{ckpt,coverage,imatrix.gguf}) the rank
# reads only node-local bytes and packs one layer per call into LOCAL_ROOT/rR/<pack>.d; strata-pput publishes each finished
# layer (chunks, then jsons) to the shared chunk directory while the next layer packs. Layers complete there are skipped.
set -u
DEADLINE=$1 JOB=$2 CKPT=$3 OUT=$4 BIN=$5 PACKS=$6 TRUNK=$7 THREADS=$8 IMATRIX=$9 LAYER_S=${10} LIBS=${11} LROOT=${12:-none} PUT_STREAMS=${13:-8}
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
SRC=$CKPT; cov=(); MY=""
if [ "$LROOT" != none ] && [ -s "$LROOT/r$R/coverage" ]; then MY=$LROOT/r$R; SRC=$MY/ckpt; cov=("--coverage=$MY/coverage"); fi
PUT=$(dirname "$BIN")/strata-pput
st "DS41F_RANK_SOURCE rank=$R source=${MY:+mirror}${MY:-shared}"
calib=(--uncalibrated)
if [ "$IMATRIX" != none ] && [ -n "$MY" ] && [ -f "$MY/imatrix.gguf" ]; then calib=("--imatrix=$MY/imatrix.gguf")
elif [ "$IMATRIX" != none ]; then cp "$IMATRIX" "$LOCALDIR/imatrix.gguf" && calib=("--imatrix=$LOCALDIR/imatrix.gguf") || calib=("--imatrix=$IMATRIX"); fi
# layer_done DIR L: the pack tool's own test (3 chunk jsons whose chunk files have the recorded size)
layer_done() { python3 -c "
import json, os, sys
for p in ('gate', 'up', 'down'):
    b = os.path.join(sys.argv[1], 'blk.%s.ffn_%s_exps.weight.chunk' % (sys.argv[2], p))
    try:
        if os.path.getsize(b) != json.load(open(b + '.json'))['bytes']: sys.exit(1)
    except (OSError, ValueError, KeyError): sys.exit(1)" "$1" "$2"; }
pack_args() { local f=${1#*:}; local g=${f%%/*} r=${f#*/}; local u=${r%%/*} d=${r#*/}; echo "--gate=$g --up=$u --down=$d"; }
maxrc=0; tpid=""
if [ $has_trunk = 1 ]; then
    TF=$OUT/trunk-${TRUNK%%/*}-${TRUNK#*/}.gguf
    if [ ! -f "$TF.verified" ]; then
        rm -f "$TF.partial"
        ( t0=$(date +%s); TL=$TF; [ -n "$MY" ] && [ ! -f "$TF" ] && TL=$MY/trunk.gguf   # built on /local, then published
          if [ -f "$TF" ] || "$BIN" trunk "$SRC" "$TL" --trunk="${TRUNK%%/*}" --head="${TRUNK#*/}" --threads=8 "${cov[@]}" > "$LOGP.trunk.log" 2>&1; then
              if "$BIN" verify "$SRC" "$TL" >> "$LOGP.trunk.log" 2>&1 &&
                 { [ "$TL" = "$TF" ] || { echo "W $TL $TF" > "$MY/put-trunk.txt" && "$PUT" --write="$MY/put-trunk.txt" --streams="$PUT_STREAMS" >> "$LOGP.trunk.log" 2>&1; }; }; then
                  date -Is > "$TF.verified"; echo "TRUNK_RESULT rc=0 verify=ok seconds=$(( $(date +%s) - t0 ))" >> "$ST"
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
    if [ -z "$MY" ]; then
        "$BIN" experts "$CKPT" --chunk-dir="$dir" $(pack_args "$spec") "${calib[@]}" --layers="$A-$B" --threads="$T" \
            --deadline="$DEADLINE" --layer-seconds="$LAYER_S" > "$LOGP.$name.log" 2>&1
        rc=$?
    else   # one layer per call into /local; the previous layer is published while this one packs
        LD=$MY/$name.d; mkdir -p "$LD"; cp "$dir/plan.json" "$LD/plan.json"; last=$LAYER_S; mover=""; prc=0; rc=0; : > "$LOGP.$name.log"
        for L in $(seq "$A" "$B"); do
            layer_done "$dir" "$L" && { echo "PACK layer=$L already published" >> "$LOGP.$name.log"; continue; }
            [ $(( $(date +%s) + last * 12 / 10 )) -gt "$DEADLINE" ] && { rc=3; echo "PACK deadline: stopping before layer $L" >> "$LOGP.$name.log"; break; }
            l0=$(date +%s)
            "$BIN" experts "$SRC" --chunk-dir="$LD" $(pack_args "$spec") "${calib[@]}" --layers="$L-$L" --threads="$T" \
                --deadline="$DEADLINE" --layer-seconds="$last" "${cov[@]}" >> "$LOGP.$name.log" 2>&1
            rc=$?; [ $rc = 0 ] || break
            last=$(( $(date +%s) - l0 ))
            [ -n "$mover" ] && { wait "$mover" || prc=1; }
            list=$MY/put-$name-$L.txt; : > "$list"
            for p in gate up down; do echo "W $LD/blk.$L.ffn_${p}_exps.weight.chunk $dir/blk.$L.ffn_${p}_exps.weight.chunk" >> "$list"; done
            i=0; for p in gate up down; do echo "W $LD/blk.$L.ffn_${p}_exps.weight.chunk.json $dir/blk.$L.ffn_${p}_exps.weight.chunk.json after=$i" >> "$list"; i=$((i + 1)); done
            ( "$PUT" --write="$list" --streams="$PUT_STREAMS" >> "$LOGP.$name.put.log" 2>&1 && rm -f "$LD"/blk."$L".ffn_*_exps.weight.chunk ) & mover=$!
        done
        [ -n "$mover" ] && { wait "$mover" || prc=1; }
        [ $prc = 1 ] && { rc=1; echo "PACK publish failed (see $LOGP.$name.put.log)" >> "$LOGP.$name.log"; }
    fi
    st "PACK_RESULT pack=$name rc=$rc layers=$A-$B seconds=$(( $(date +%s) - t0 ))"
    grep "PACK layer=.*seconds=" "$LOGP.$name.log" | sed 's/^/  /' >> "$ST"
    [ $rc = 1 ] && maxrc=1; [ $rc = 3 ] && [ $maxrc = 0 ] && maxrc=3
done
[ -n "$tpid" ] && wait "$tpid"
rm -rf "/local/ds41f-pack-$JOB" 2>/dev/null; [ -n "$MY" ] && rm -rf "$MY"
st "SENTINEL DS41F_RANK_END rank=$R rc=$maxrc"
exit 0
