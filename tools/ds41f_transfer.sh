#!/bin/bash
# Pull a DeepSeek V4.1 Flash chunk directory from the GPU cluster's login node to this machine, resumably, then assemble it.
#
#   tools/ds41f_transfer.sh LOGIN:/shared/.../experts-iq1s.d /models/ds41f/experts-iq1s.d [experts-iq1s.gguf] [--streams 4]   (LOGIN = ssh alias of the login node)
#
# Every chunk is one file (~0.5-1.3 GB), so rsync --partial --inplace resumes mid-file and the whole run restarts
# safely; `--streams N` runs N rsync processes over disjoint file lists (the site link measured 11 MB/s per stream for
# HF uploads; the GPU node -> here rate is unmeasured and is logged by this script). After the copy every chunk's size
# and FNV hash are checked by `strata-ds41f-pack assemble`, which also writes the single sidecar GGUF when a name is
# given. Rate and timing go to <dest>.transfer.log.
set -u
SRC=${1:?source host:dir}; DST=${2:?destination dir}; OUT=${3:-}
STREAMS=4
for a in "$@"; do case $a in --streams=*) STREAMS=${a#--streams=};; esac; done
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PACK=${DS41F_PACK:-$here/build-ds41f/strata-ds41f-pack}
mkdir -p "$DST"; L="$DST.transfer.log"
echo "START $(date +%FT%T) src=$SRC dst=$DST streams=$STREAMS" >> "$L"
# the file list: plan.json first, then chunks and their jsons
host=${SRC%%:*}; path=${SRC#*:}
until ssh -o BatchMode=yes "$host" "cd '$path' && ls -1 plan.json *.chunk.json *.chunk 2>/dev/null" > "$DST/.files"; do echo "listing failed, retry in 60s $(date +%T)" >> "$L"; sleep 60; done
total=$(ssh -o BatchMode=yes "$host" "cd '$path' && du -sb . | cut -f1")
echo "files=$(wc -l < "$DST/.files") bytes=$total" >> "$L"
split -n l/$STREAMS -d "$DST/.files" "$DST/.files."
t0=$(date +%s)
for f in "$DST"/.files.*; do
    ( until nice -n 10 rsync -a --partial --inplace --files-from="$f" "$host:$path/" "$DST/" >> "$L.$(basename "$f")" 2>&1; do echo "rsync $(basename "$f") failed, retry in 60s $(date +%T)" >> "$L"; sleep 60; done ) &
done
wait
t1=$(date +%s); el=$((t1 - t0)); [ $el -gt 0 ] && echo "RSYNC_DONE seconds=$el MB/s=$((total / 1000000 / el)) $(date +%FT%T)" >> "$L"
rm -f "$DST"/.files.*
"$PACK" assemble "$DST" --check >> "$L" 2>&1 || { echo "INCOMPLETE: chunks missing, rerun later" >> "$L"; exit 3; }
if [ -n "$OUT" ]; then
    "$PACK" assemble "$DST" "$OUT" >> "$L" 2>&1 && "$PACK" verify "$OUT" >> "$L" 2>&1 && echo "ASSEMBLED $OUT $(date +%FT%T)" >> "$L" || { echo "ASSEMBLE_FAILED" >> "$L"; exit 1; }
fi
tail -3 "$L"
