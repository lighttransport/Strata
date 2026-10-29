#!/usr/bin/env bash
# qual.sh NAME PROMPT [ENV=VAL ...]: a teacher-forced chunk-5 logits capture of the prompt in the reversed layout; NAME ref_* keeps the file as the
# exact reference of PROMPT, any other NAME is compared with it (cmp_gate) and the capture removed. Uses the exported W_* / ADAPT / D_THREADS settings.
set -u
cd "$(dirname "$0")"
name=$1 prompt=$2; shift 2
R=/mnt/nvme01/tmp/ref_${prompt}.f32
rm -f logs/$name.*
SAVE_LOGITS=1 PROMPT=$prompt STEPS=1 ./rev_node.sh run $name STRATA_DS41_CANON_MIB=${CANON:-5600} STRATA_DS41_DESKTOP_MIB=512 "$@" > logs/$name.run 2>&1
f=/mnt/nvme01/tmp/lg_$name.f32
if [ ! -s "$f" ]; then echo "$name: capture failed"; exit 1; fi
if [[ $name == ref_* ]]; then mv "$f" "$R"; echo "$name: reference kept ($(stat -c %s $R) bytes)"; exit 0; fi
[ -s "$R" ] || { echo "$name: no reference for $prompt"; rm -f "$f"; exit 1; }
echo "$name $prompt $(./cmp_gate "$R" "$f" eval/$prompt.txt) args: $*" | tee -a logs/qual.txt
rm -f "$f"
