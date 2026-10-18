#!/bin/bash
# tools/ds41f_pjm_probe_rank.sh PHASE JOB CKPT PROBE_DIR BIN IMATRIX|none - one rank of tools/ds41f_pjm_probe.sh.
# PHASE pack: rank r packs real layer r into /local with the format/thread setting of its row below;
#       write: copies its chunks to PROBE_DIR/copy-rNN with fsync and publishes their sha256 list;
#       check: verifies rank (r+1)'s copies against that rank's list (cross-node coherence of the shared FS);
#       write1: one 2 GiB fsync'd write from a single node (uncontended rate).
set -u
PHASE=$1 JOB=$2 CKPT=$3 P=$4 BIN=$5 IM=$6
R=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PJM_RANK:-0}}}; RR=$(printf %02d "$R")
ST=$P/rank$RR.status; L=/local/ds41f-probe-$JOB/r$RR; mkdir -p "$L" 2>/dev/null || L=$P/local-r$RR; mkdir -p "$L"
CONF=("q2_k q2_k q3_k 47" "q2_k q2_k q3_k 48" "q2_k q2_k q3_k 24" "iq1_s iq1_s iq1_s 47" "iq1_s iq1_s iq1_s 48" "iq1_s iq1_s iq1_s 24"
      "iq2_xxs iq2_xxs q2_k 47" "iq2_xxs iq2_xxs q2_k 48" "iq2_xxs iq2_xxs q2_k 24" "iq1_m iq1_m iq1_m 47" "iq1_m iq1_m iq1_m 48" "q4_k q4_k q4_k 47")
case "$PHASE" in
pack)
    read -r g u d t <<< "${CONF[$R]}"
    calib=(--uncalibrated); case "$g" in iq*) [ "$IM" != none ] && calib=("--imatrix=$IM") || { echo "PROBE_PACK rank=$R formats=$g/$u/$d threads=$t rc=noimatrix" >> "$ST"; exit 0; };; esac
    ( mn=999999999; while :; do m=$(awk '/MemAvailable/ {print $2}' /proc/meminfo); [ "$m" -lt "$mn" ] && mn=$m; echo "$mn" > "$L/memmin"; sleep 10; done ) & smp=$!
    t0=$(date +%s.%N)
    timeout 1500 "$BIN" experts "$CKPT" --chunk-dir="$L/d" --gate="$g" --up="$u" --down="$d" "${calib[@]}" --layers="$R-$R" --threads="$t" > "$L/pack.log" 2>&1
    rc=$?; kill $smp 2>/dev/null
    secs=$(python3 -c "import time; print(round(time.time() - $t0, 1))")
    line=$(grep "PACK layer=$R seconds=" "$L/pack.log" | tail -1)
    cpu=$(echo "$line" | sed -n 's/.*cpu_user_s=\([0-9.e+]*\).*/\1/p'); rss=$(echo "$line" | sed -n 's/.*maxrss_kb=\([0-9]*\).*/\1/p')
    echo "PROBE_PACK rank=$R host=$(hostname) formats=$g/$u/$d threads=$t rc=$rc seconds=$secs cpu_user_s=${cpu:-0} maxrss_kb=${rss:-0} min_memavail_kb=$(cat "$L/memmin")" >> "$ST"
    cp "$L/pack.log" "$P/pack-r$RR.log" ;;
write)
    C=$P/copy-r$RR; mkdir -p "$C"; bytes=0; t0=$(date +%s.%N)
    for f in "$L"/d/*.chunk; do [ -f "$f" ] || continue; dd if="$f" of="$C/$(basename "$f")" bs=8M conv=fsync status=none; bytes=$((bytes + $(stat -c %s "$f"))); done
    secs=$(python3 -c "import time; print(round(time.time() - $t0, 2))")
    (cd "$C" && sha256sum *.chunk > "$P/sha-r$RR.txt.partial" && mv "$P/sha-r$RR.txt.partial" "$P/sha-r$RR.txt")
    echo "PROBE_WRITE rank=$R bytes=$bytes seconds=$secs MBps=$(python3 -c "print(round($bytes / max($secs, 1e-3) / 1e6, 1))")" >> "$ST" ;;
check)
    N=$(printf %02d $(( (R + 1) % 12 )))
    for i in $(seq 1 30); do [ -f "$P/sha-r$N.txt" ] && break; sleep 2; done
    if (cd "$P/copy-r$N" 2>/dev/null && sha256sum -c --quiet "$P/sha-r$N.txt" > /dev/null 2>&1); then s=ok; else s=mismatch; fi
    echo "PROBE_COHERENCE rank=$R peer=$N state=$s" >> "$ST" ;;
write1)
    t0=$(date +%s.%N); dd if=/dev/zero of="$P/write1.bin" bs=8M count=256 conv=fsync status=none
    secs=$(python3 -c "import time; print(round(time.time() - $t0, 2))"); rm -f "$P/write1.bin"
    echo "PROBE_WRITE1 rank=$R bytes=2147483648 seconds=$secs MBps=$(python3 -c "print(round(2147483648 / max($secs, 1e-3) / 1e6, 1))")" >> "$ST" ;;
esac
exit 0
