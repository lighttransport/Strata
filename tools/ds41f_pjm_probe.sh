#!/bin/bash
#PJM -L "node=12"
#PJM -L "elapse=01:30:00"
#PJM -L "freq=2000,eco_state=0"
#PJM --mpi "proc=12"
#PJM --llio localtmp-size=87Gi
#PJM -j
# tools/ds41f_pjm_probe.sh - measurements that size the DS4.1F pack jobs (tools/ds41f_pjm_pack.sh) on 12 A64FX nodes.
#
#   pjsub -g GROUP -L rscgrp=RSCGRP -x PJM_LLIO_GFSCACHE=VOL -x DS41F_STRATA=<Strata snapshot on the shared FS> \
#         -x GGML_SRC=<dir with ggml/> -x DS41F_BIN_ROOT=<publish dir> -x DS41F_CKPT=<checkpoint> -x DS41F_OUT=<output dir> \
#         -x DS41F_PROBE_IMATRIX=<synthetic imatrix, layers 0-11> [-x DS41F_GGML_QUANTS_SHA256=<hex>] [-x DS41F_PROBE_TESTS=1] \
#         <Strata snapshot>/tools/ds41f_pjm_probe.sh
#
# 1. builds and publishes strata-ds41f-pack (tools/a64fx_probe/build_ds41f_pack.sh) and runs its tests if numpy exists;
# 2. rank r packs real layer r into node-local storage with one format/thread setting per rank (q23, IQ1_S, IQ2_XXS/Q2_K at
#    47/48/24 threads, IQ1_M at 47/48, Q4_K x3 as the trunk stand-in), sampling MemAvailable;
# 3. every rank copies its chunks to the shared FS with fsync (per-node and aggregate write rate), then checks the next
#    rank's copies by sha256 (cross-node coherence); one single-node write for the uncontended rate;
# 4. prints PROBE_DECISION lines (threads, packs per job, per-layer seconds seeds, trunk threading, chunk-dir mode) and
#    leaves a canary file that the pack jobs check (DS41F_CANARY).
set -uo pipefail
T0=$(date +%s); JOB=${PJM_JOBID:-manual$T0}
S=${DS41F_STRATA:?DS41F_STRATA}; OUT=${DS41F_OUT:?DS41F_OUT}; CKPT=${DS41F_CKPT:?DS41F_CKPT}
P=$OUT/probe/$JOB; mkdir -p "$P"
say() { echo "$*"; echo "$*" >> "$P/head.log"; }
say "SENTINEL DS41F_PROBE_BEGIN job=$JOB host=$(hostname)"
if [ -n "${DS41F_SNAP:-}" ]; then SNAP=$DS41F_SNAP; say "PROBE_BUILD skipped, using $SNAP"   # an already published build
else
    build=$(GGML_SRC=${GGML_SRC:?GGML_SRC} DS41F_BIN_ROOT=${DS41F_BIN_ROOT:?DS41F_BIN_ROOT} bash "$S/tools/a64fx_probe/build_ds41f_pack.sh" 2>&1 | tee "$P/build.log" | grep "SENTINEL DS41F_PACK_BUILD")
    say "$build"
    SNAP=$(echo "$build" | sed -n 's/.*dir=\([^ ]*\).*/\1/p')
fi
BIN=$SNAP/strata-ds41f-pack
[ -x "$BIN" ] || { say "SENTINEL DS41F_PROBE_END state=failed reason=build"; exit 0; }
if [ "${DS41F_PROBE_TESTS:-1}" = 1 ] && python3 -c "import numpy" 2>/dev/null; then
    if python3 "$S/tools/test_ds41f_pack.py" --converter "$BIN" > "$P/tests.log" 2>&1; then say "PROBE_TESTS PASS $(tail -1 "$P/tests.log" | cut -c1-200)"
    else say "PROBE_TESTS FAIL (see $P/tests.log)"; fi
else say "PROBE_TESTS SKIP (numpy unavailable or disabled)"; fi
R="$S/tools/ds41f_pjm_probe_rank.sh"
mpiexec -np 12 -of-proc "$P/pack" "$R" pack "$JOB" "$CKPT" "$P" "$BIN" "${DS41F_PROBE_IMATRIX:-none}"
mpiexec -np 12 -of-proc "$P/write" "$R" write "$JOB" "$CKPT" "$P" "$BIN" none
mpiexec -np 12 -of-proc "$P/check" "$R" check "$JOB" "$CKPT" "$P" "$BIN" none
mpiexec -np 1 -of-proc "$P/write1" "$R" write1 "$JOB" "$CKPT" "$P" "$BIN" none
cat "$P"/rank*.status > "$P/all.status" 2>/dev/null
grep -h "^PROBE_" "$P/all.status" | while read -r l; do say "$l"; done
python3 - "$P/all.status" <<'PY' | while read -r l; do say "$l"; done
import re, statistics, sys
rows = [dict(re.findall(r"(\w+)=(\S+)", l), kind=l.split()[0]) for l in open(sys.argv[1]) if l.startswith("PROBE_")]   # python 3.6 safe
pack = [r for r in rows if r["kind"] == "PROBE_PACK" and r.get("rc") == "0"]
W = 384 * 3 * 2304 * 5120          # weights per layer
best = {}
for r in pack:
    f, t, s = r["formats"], int(r["threads"]), float(r["seconds"])
    print(f"PROBE_RATE formats={f} threads={t} layer_s={s:.0f} cpu_rate_Mw_per_core_s={W / float(r['cpu_user_s']) / 1e6:.2f} min_memavail_gib={int(r['min_memavail_kb']) / 2**20:.1f} maxrss_gib={int(r['maxrss_kb']) / 2**20:.1f}")
    if t != 24 and (f not in best or s < best[f][1]): best[f] = (t, s)
threads = 47
for f, (t, s) in best.items():
    s47 = next((float(r["seconds"]) for r in pack if r["formats"] == f and r["threads"] == "47"), s)
    if t == 48 and s < 0.95 * s47: threads = 48
memmin = min((int(r["min_memavail_kb"]) for r in pack), default=0) / 2**20
if memmin and memmin < 4: threads = 32
print(f"PROBE_DECISION threads={threads} min_memavail_gib={memmin:.1f}")
for f, (t, s) in best.items():
    print(f"PROBE_DECISION layer_seconds formats={f} seed={1.2 * s:.0f} critical_path_min={4 * s / 60:.0f}")
total = sum(4 * s for f, (t, s) in best.items() if not f.startswith("q4_k"))
print(f"PROBE_DECISION all_listed_packs_critical_path_min={total / 60:.0f} fits_one_job={'yes' if total + 600 < 0.8 * (21600 - 900) else 'no'}")
q4 = next((r for r in pack if r["formats"].startswith("q4_k")), None)
if q4:
    trunk_s = 9.8e9 / (W / float(q4["cpu_user_s"]))
    print(f"PROBE_DECISION trunk_single_thread_h={trunk_s / 3600:.2f} trunk_threads={'8' if trunk_s > 7200 else '1'}")
wr = [r for r in rows if r["kind"] == "PROBE_WRITE"]
if wr:
    agg = sum(float(r["bytes"]) for r in wr) / max(float(r["seconds"]) for r in wr) / 1e6
    print(f"PROBE_DECISION write_aggregate_MBps={agg:.0f} per_node_MBps_median={statistics.median(float(r['MBps']) for r in wr):.0f} one_pack_per_job={'yes' if agg < 20 else 'no'}")
coh = [r for r in rows if r["kind"] == "PROBE_COHERENCE"]
bad = [r for r in coh if r.get("state") != "ok"]
print(f"PROBE_DECISION chunk_dir={'shared' if coh and not bad else 'per-rank'} coherence_checked={len(coh)} mismatches={len(bad)}")
PY
date -Is > "$P/canary" && sync
say "PROBE_CANARY file=$P/canary"
rm -rf "$P"/copy-r*
say "SENTINEL DS41F_PROBE_END state=complete job=$JOB elapsed_s=$(( $(date +%s) - T0 )) snap=$SNAP"
