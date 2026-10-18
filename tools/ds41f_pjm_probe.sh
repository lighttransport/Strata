#!/bin/bash
#PJM -L "node=12"
#PJM -L "elapse=02:30:00"
#PJM -L "freq=2000,eco_state=0"
#PJM --mpi "proc=12"
#PJM --llio localtmp-size=87Gi
#PJM -j
# tools/ds41f_pjm_probe.sh - measurements that size the DS4.1F pack jobs (tools/ds41f_pjm_pack.sh) on 12 A64FX nodes.
#
#   pjsub -g GROUP -L rscgrp=RSCGRP -x PJM_LLIO_GFSCACHE=VOL -x DS41F_STRATA=<Strata snapshot on the shared FS> \
#         -x GGML_SRC=<dir with ggml/> -x DS41F_BIN_ROOT=<publish dir> -x DS41F_CKPT=<checkpoint> -x DS41F_OUT=<output dir> \
#         -x DS41F_PROBE_IMATRIX=<synthetic imatrix, layers 0-11> [-x DS41F_GGML_QUANTS_SHA256=<hex>] [-x DS41F_PROBE_TESTS=1] \
#         [-x DS41F_PROBE_POINT_S=45] <Strata snapshot>/tools/ds41f_pjm_probe.sh
# Every measurement line carries gfscache=$PJM_LLIO_GFSCACHE, so runs with different LLIO cache settings compare.
#
# 1. builds and publishes strata-ds41f-pack + strata-pfetch/pput (tools/a64fx_probe/build_ds41f_pack.sh) and runs the
#    pack tests if numpy exists;
# 1b. strata-pfetch --measure: shared-FS read MB/s per node and aggregate vs streams per node (1/4/16/32) for each rank on
#    its own file, all ranks on one file (per-target limit) and one file per stream; request size, one node alone, a
#    re-read (cache); shared-FS writes vs streams; MPI ring / fan-in / broadcast; /local rates and sparse files;
#    then strata-pfetch fills a node-local mirror of each rank's probe layer (+ imatrix), all nodes reading cooperatively;
# 2. rank r packs real layer r from that mirror (pure compute; the shared checkpoint if the fetch failed) into node-local
#    storage with one format/thread setting per rank (q23, IQ1_S, IQ2_XXS/Q2_K at 47/48/24 threads, IQ1_M at 47/48, Q4_K x3
#    as the trunk stand-in), sampling MemAvailable; an MPI program started from the rank wrapper (child MPI) is tried;
# 3. every rank copies its chunks to the shared FS with fsync (per-node and aggregate write rate), then checks the next
#    rank's copies by sha256 (cross-node coherence); one single-node write for the uncontended rate;
# 4. prints PROBE_DECISION lines (threads, packs per job, per-layer seconds seeds, trunk threading, chunk-dir mode,
#    pfetch streams/assignment, put streams, I/O- or compute-bound packing, overlap mode) and leaves a canary file that
#    the pack jobs check (DS41F_CANARY).
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
PF=$SNAP/strata-pfetch; LR=${DS41F_LOCAL_ROOT:-/local}/ds41f-probe-$JOB; MIRROR=none; GFS=${PJM_LLIO_GFSCACHE:-none}
if [ -x "$PF" ]; then
    t1=$(date +%s)
    mpiexec -np 12 -of-proc "$P/measure" "$PF" --measure="${DS41F_PROBE_MEASURE:-read,write,mpi,local}" --src-dir="$CKPT" \
        --scratch-shared="$P/pfm" --local-root="$LR/pfm" --point-s="${DS41F_PROBE_POINT_S:-45}" \
        --region-mib="${DS41F_PROBE_REGION_MIB:-512}" --min-file-gib="${DS41F_PROBE_MIN_FILE_GIB:-4}"
    cat "$P"/measure.* 2>/dev/null | grep -h "^PFETCH_MEASURE" | sed "s/\$/ gfscache=$GFS/" > "$P/measure.txt"
    while read -r l; do say "$l"; done < "$P/measure.txt"
    say "PROBE_MEASURE_DONE seconds=$(( $(date +%s) - t1 )) lines=$(wc -l < "$P/measure.txt")"
    t1=$(date +%s)
    if python3 "$SNAP/ds41f_pfetch_plan.py" --bin "$BIN" --ckpt "$CKPT" --rank-layers "$(seq -s, 0 11 | sed 's/\([0-9]*\)/\1:\1-\1/g')" \
           --imatrix "${DS41F_PROBE_IMATRIX:-none}" --imatrix-local '@L/r{rank}/imatrix.gguf' --mirror '@L/r{rank}/ckpt' > "$P/fetch.manifest" 2>> "$P/head.log"; then
        mpiexec -np 12 -of-proc "$P/pfetch" "$PF" --manifest="$P/fetch.manifest" --local-root="$LR" --streams="${DS41F_PFETCH_STREAMS:-16}" \
            --coverage-out='@L/r{rank}/coverage' --status-dir="$P/pfetch" --progress-s=30
        ok=$(cat "$P"/pfetch/pfetch.rank*.status 2>/dev/null | grep -c "state=ok")
        say "PROBE_FETCH ranks_ok=$ok seconds=$(( $(date +%s) - t1 )) gfscache=$GFS $(cat "$P"/pfetch.* 2>/dev/null | grep -o 'PFETCH_END.*' | head -1)"
        [ "$ok" = 12 ] && MIRROR=$LR
    fi
fi
mpiexec -np 12 -of-proc "$P/pack" "$R" pack "$JOB" "$CKPT" "$P" "$BIN" "${DS41F_PROBE_IMATRIX:-none}" "$MIRROR"
[ -x "$PF" ] && mpiexec -np 12 -of-proc "$P/childmpi" "$R" childmpi "$JOB" "$CKPT" "$P" "$BIN" none
mpiexec -np 12 -of-proc "$P/write" "$R" write "$JOB" "$CKPT" "$P" "$BIN" none
mpiexec -np 12 -of-proc "$P/check" "$R" check "$JOB" "$CKPT" "$P" "$BIN" none
mpiexec -np 1 -of-proc "$P/write1" "$R" write1 "$JOB" "$CKPT" "$P" "$BIN" none
cat "$P"/rank*.status "$P/measure.txt" > "$P/all.status" 2>/dev/null
grep -h "^PROBE_FETCH" "$P/head.log" >> "$P/all.status" 2>/dev/null
grep -h "^PROBE_" "$P"/rank*.status | while read -r l; do say "$l"; done
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
# strata-pfetch measurements
meas = [dict(re.findall(r"(\w+)=(\S+)", l)) for l in open(sys.argv[1]) if l.startswith("PFETCH_MEASURE")]
def curve(kind, layout, req="16"):
    return {int(m["streams"]): float(m["agg_MBps"]) for m in meas
            if m.get("kind") == kind and m.get("layout") == layout and m.get("req_mib") == req and "alone" not in m and "reread" not in m}
def knee(c):
    if not c: return 0, 0.0
    best = max(c.values()); return min(s for s, v in c.items() if v >= 0.9 * best), best
gfs = meas[0].get("gfscache", "none") if meas else "none"
own, one, per = curve("read", "own-file"), curve("read", "all-one-file"), curve("read", "per-stream-file")
if own:
    s_own, b_own = knee(own); _, b_one = knee(one); _, b_per = knee(per)
    print(f"PROBE_DECISION pfetch_streams={s_own} read_agg_MBps={b_own:.0f} own_file_curve={','.join(f'{k}:{v:.0f}' for k, v in sorted(own.items()))} gfscache={gfs}")
    print(f"PROBE_DECISION pfetch_assign={'local' if b_own >= 0.9 * b_per else 'spread'} own_file_MBps={b_own:.0f} per_stream_file_MBps={b_per:.0f}")
    print(f"PROBE_DECISION per_target_limited={'yes' if b_one < 0.5 * b_own else 'no'} all_one_file_MBps={b_one:.0f}")
    alone = {int(m["streams"]): float(m["agg_MBps"]) for m in meas if m.get("kind") == "read" and "alone" in m}
    rr = [float(m["agg_MBps"]) for m in meas if m.get("kind") == "read" and "reread" in m]
    if alone: print(f"PROBE_DECISION one_node_MBps={','.join(f'{k}:{v:.0f}' for k, v in sorted(alone.items()))} per_client_limited={'yes' if max(alone.values()) * 12 < 0.9 * b_own else 'no'}")
    if rr and own.get(1): print(f"PROBE_DECISION reread_MBps={rr[0]:.0f} first_read_MBps={own[1]:.0f} cache_effect={'yes' if rr[0] > 3 * own[1] else 'no'}")
wcur = curve("write", "file-per-stream")
if wcur:
    s_w, b_w = knee(wcur)
    print(f"PROBE_DECISION put_streams={s_w} write_agg_MBps={b_w:.0f} write_curve={','.join(f'{k}:{v:.0f}' for k, v in sorted(wcur.items()))}")
sp = next((m for m in meas if m.get("kind") == "sparse"), None)
if sp: print(f"PROBE_DECISION local_sparse={sp.get('local_sparse')} local_free_gib={sp.get('rank0_free_GiB')}")
lw, lr = curve("local-write", "file-per-stream"), curve("local-read", "file-per-stream")
if lw: print(f"PROBE_DECISION local_write_agg_MBps={max(lw.values()):.0f} local_read_agg_MBps={max(lr.values()) if lr else 0:.0f}")
ring = next((m for m in meas if m.get("kind") == "mpi" and m.get("layout") == "ring"), None)
if ring: print(f"PROBE_DECISION mpi_ring_rank_MBps={ring.get('rank_MBps_median')}")
fetch = next((r for r in rows if r["kind"] == "PROBE_FETCH"), None)
if fetch and pack:
    fs, med = float(fetch["seconds"]), statistics.median(float(r["seconds"]) for r in pack)
    src = {r.get("source", "shared") for r in pack}
    print(f"PROBE_DECISION fetch_layer_s={fs:.0f} pack_layer_s_median={med:.0f} pack_io_bound={'yes' if fs > med else 'no'} pack_source={','.join(sorted(src))} "
          f"mirror_ok={fetch.get('ranks_ok')} all_40_layers_fetch_min={fs * 40 / 12 / 60:.0f}")
ch = [r for r in rows if r["kind"] == "PROBE_CHILDMPI"]
if ch:
    good = sum(1 for r in ch if r.get("ranks") == "12")
    print(f"PROBE_DECISION child_mpi={'ok' if good == 12 else 'no'} ranks_joined={good} overlap_mode={'B' if good == 12 else 'A'}")
PY
date -Is > "$P/canary" && sync
say "PROBE_CANARY file=$P/canary"
rm -rf "$P"/copy-r*
say "SENTINEL DS41F_PROBE_END state=complete job=$JOB elapsed_s=$(( $(date +%s) - T0 )) snap=$SNAP"
