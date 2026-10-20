#!/usr/bin/env python3
"""tools/ds41f_pjm_pack.sh + ds41f_pjm_pack_rank.sh on this machine: a synthetic 40-layer x 1-expert checkpoint and a
fake `mpiexec` that starts the 12 rank wrappers as local processes (PMIX_RANK set). Checks:
  1. a full job: plans created on the head, every rank packs its slice, the trunk is packed and verified on rank 11,
     every pack checks complete, `SENTINEL DS41F_JOB_END state=complete`;
  2. resubmitting the same job exits early (`DS41F_ALREADY_COMPLETE`);
  3. a job whose deadline leaves no time packs nothing and ends with state=resubmit; a stale binary hash fails;
  4. with strata-pfetch built (tools/pfetch/build) the first job reads through the node-local mirror (fetched by
     strata-pfetch under the real mpirun, published by strata-pput) and its chunks and trunk equal those of jobs
     reading the shared checkpoint directly (DS41F_MIRROR=0).

    python3 tools/test_ds41f_pjm.py --converter build-ds41f/strata-ds41f-pack
"""
import argparse, os, pathlib, shutil, stat, subprocess, sys, tempfile
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_ds41f_pack import make_checkpoint  # noqa: E402

FAKE_MPIEXEC = """#!/bin/bash
# fake mpiexec: -np N [-of-proc PREFIX] CMD ARGS... -> N local processes with PMIX_RANK
np=1; prefix=/dev/null
while [ $# -gt 0 ]; do case "$1" in -np|-n) np=$2; shift 2;; -of-proc) prefix=$2; shift 2;; *) break;; esac; done
# an MPI program (strata-pfetch) runs under the real mpirun
case "$(basename "$1")" in strata-pfetch) exec mpirun --oversubscribe -np "$np" "$@" > "$prefix.out" 2>&1;; esac
pids=()
for ((r=0; r<np; r++)); do PMIX_RANK=$r "$@" > "$prefix.$r.out" 2>&1 & pids+=($!); done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done; exit $rc
"""


def job(repo, snap, env_extra, out):
    env = dict(os.environ, DS41F_SNAP=str(snap), DS41F_OUT=str(out), PJM_JOBID=env_extra.pop("PJM_JOBID"), **env_extra)
    env["PATH"] = str(snap / "fakebin") + os.pathsep + env["PATH"]
    r = subprocess.run(["bash", str(snap / "ds41f_pjm_pack.sh")], env=env, capture_output=True, text=True, timeout=3600)
    return r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--converter", default="build-ds41f/strata-ds41f-pack")
    ap.add_argument("--keep-dir")
    args = ap.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="strata-ds41f-pjm-") as tmp:
        root = pathlib.Path(args.keep_dir or tmp); root.mkdir(parents=True, exist_ok=True)
        ckpt = root / "ckpt"; ckpt.mkdir(exist_ok=True)
        make_checkpoint(ckpt, np.random.default_rng(11), layers=40, n_experts=1)
        snap = root / "snap"; (snap / "fakebin").mkdir(parents=True, exist_ok=True)
        shutil.copy(args.converter, snap / "strata-ds41f-pack")
        for s in ("ds41f_pjm_pack.sh", "ds41f_pjm_pack_rank.sh", "ds41f_pfetch_plan.py"):
            shutil.copy(repo / "tools" / s, snap / s)
        pf = repo / "tools" / "pfetch" / "build"
        mirror = (pf / "strata-pfetch").exists() and shutil.which("mpirun") is not None
        if mirror:
            for b in ("strata-pfetch", "strata-pput"):
                shutil.copy(pf / b, snap / b)
        fake = snap / "fakebin" / "mpiexec"; fake.write_text(FAKE_MPIEXEC); fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        sha = subprocess.run(["sha256sum", str(snap / "strata-ds41f-pack")], capture_output=True, text=True).stdout.split()[0]
        out = root / "out"
        base = dict(DS41F_CKPT=str(ckpt), DS41F_PACK_SHA256=sha, DS41F_PACKS="q23:q2_k/q2_k/q3_k+q2:q2_k/q2_k/q2_k",
                    DS41F_TRUNK="q4_k/q8_0", DS41F_THREADS="10", DS41F_LAYER_SECONDS="5", DS41F_LOCAL_ROOT=str(root / "local"))
        log = job(repo, snap, dict(base, PJM_JOBID="1001"), out)
        assert "SENTINEL DS41F_JOB_END state=complete" in log, log
        if mirror:
            assert "SENTINEL DS41F_PFETCH ranks_ok=12" in log and "mirror=" + str(root / "local") in log, log
            src = "".join(p.read_text() for p in (out / "status" / "1001").glob("rank*.status"))
            assert src.count("source=mirror") == 12, src
            assert not list((out / "experts-q23.d").glob("*.pfpart")), "unpublished partial left behind"
        for name in ("q23", "q2"):
            assert f"DS41F_PACK_COMPLETE pack={name}" in log and (out / f"experts-{name}.d" / "COMPLETE").exists(), log
        assert "DS41F_TRUNK_COMPLETE" in log and (out / "trunk-q4_k-q8_0.gguf.verified").exists(), log
        statuses = sorted((out / "status" / "1001").glob("rank*.status"))
        assert len(statuses) == 12 and all("DS41F_RANK_END" in s.read_text() for s in statuses), [s.name for s in statuses]
        assert "DS41F_NMSE dir=experts-q23.d chunks=120" in log, log
        subprocess.run([args.converter, "assemble", str(out / "experts-q23.d"), str(root / "q23.gguf")], check=True, capture_output=True)
        subprocess.run([args.converter, "verify", str(ckpt), str(root / "q23.gguf")], check=True, capture_output=True)
        log2 = job(repo, snap, dict(base, PJM_JOBID="1002"), out)
        assert "DS41F_ALREADY_COMPLETE" in log2 and "state=complete" in log2, log2
        out3 = root / "out3"
        log3 = job(repo, snap, dict(base, PJM_JOBID="1003", DS41F_ELAPSE_S="60", DS41F_MARGIN_S="30", DS41F_LAYER_SECONDS="1000"), out3)
        assert "state=resubmit" in log3 and "rc=skipped" in "".join(p.read_text() for p in (out3 / "status" / "1003").glob("rank*.status")), log3
        log4 = job(repo, snap, dict(base, PJM_JOBID="1004", DS41F_PACK_SHA256="0" * 64), root / "out4")
        assert "DS41F_STALE_BINARY" in log4 and "state=failed" in log4, log4
        # two jobs on different packs of one output root run in parallel; a third on a held pack is refused
        out5 = root / "out5"
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(2) as ex:
            fa = ex.submit(job, repo, snap, dict(base, PJM_JOBID="1005", DS41F_PACKS="q23:q2_k/q2_k/q3_k", DS41F_TRUNK="none", DS41F_MIRROR="0"), out5)
            fb = ex.submit(job, repo, snap, dict(base, PJM_JOBID="1006", DS41F_PACKS="q2:q2_k/q2_k/q2_k", DS41F_TRUNK="q4_k/q8_0", DS41F_MIRROR="0"), out5)
            la, lb = fa.result(), fb.result()
        if mirror:   # mirror-fed packs == shared-read packs, byte for byte
            for name in ("q23", "q2"):
                a = sorted(p.name for p in (out / f"experts-{name}.d").glob("*.chunk*"))
                assert a and a == sorted(p.name for p in (out5 / f"experts-{name}.d").glob("*.chunk*")), name
                for f in a:
                    assert (out / f"experts-{name}.d" / f).read_bytes() == (out5 / f"experts-{name}.d" / f).read_bytes(), f
            assert (out / "trunk-q4_k-q8_0.gguf").read_bytes() == (out5 / "trunk-q4_k-q8_0.gguf").read_bytes()
        assert "state=complete" in la and "DS41F_PACK_COMPLETE pack=q23" in la, la
        assert "state=complete" in lb and "DS41F_PACK_COMPLETE pack=q2" in lb and "DS41F_TRUNK_COMPLETE" in lb, lb
        import time as _t
        (out5 / "experts-q23.d.lock").write_text(f"job=9999 end={int(_t.time()) + 3600}\n")
        lc = job(repo, snap, dict(base, PJM_JOBID="1007", DS41F_PACKS="q23:q2_k/q2_k/q3_k", DS41F_TRUNK="none"), out5)
        assert "DS41F_LOCKED experts-q23.d.lock by job=9999" in lc and "state=failed" in lc, lc
        (out5 / "experts-q23.d.lock").write_text(f"job=9999 end={int(_t.time()) - 10}\n")      # stale: taken over
        ld = job(repo, snap, dict(base, PJM_JOBID="1008", DS41F_PACKS="q23:q2_k/q2_k/q3_k", DS41F_TRUNK="none"), out5)
        assert "DS41F_ALREADY_COMPLETE" in ld and not (out5 / "experts-q23.d.lock").exists(), ld
        # the probe job: every rank packs its layer with its setting, writes/checks copies, decisions are printed
        im = root / "probe-imatrix.gguf"
        subprocess.run([sys.executable, str(repo / "tools" / "ds41f_synth_imatrix.py"), str(im), "--layers", "0-11", "--experts", "1"], check=True, capture_output=True)
        env = dict(os.environ, DS41F_STRATA=str(repo), DS41F_SNAP=str(snap), DS41F_CKPT=str(ckpt), DS41F_OUT=str(root / "probe-out"),
                   DS41F_PROBE_IMATRIX=str(im), DS41F_PROBE_TESTS="0", PJM_JOBID="2001", PATH=str(snap / "fakebin") + os.pathsep + os.environ["PATH"],
                   DS41F_LOCAL_ROOT=str(root / "probe-local"), DS41F_PROBE_POINT_S="1", DS41F_PROBE_REGION_MIB="2", DS41F_PROBE_MIN_FILE_GIB="0")
        r = subprocess.run(["bash", str(repo / "tools" / "ds41f_pjm_probe.sh")], env=env, capture_output=True, text=True, timeout=3600)
        plog = r.stdout + r.stderr
        assert "DS41F_PROBE_END state=complete" in plog, plog
        assert plog.count("PROBE_PACK ") == 12 and "rc=0" in plog and " rc=1 " not in plog, plog
        for want in ("PROBE_DECISION threads=", "PROBE_DECISION layer_seconds formats=iq1_s/iq1_s/iq1_s", "trunk_threads=",
                     "write_aggregate_MBps=", "chunk_dir=shared coherence_checked=12 mismatches=0", "PROBE_WRITE1 ", "PROBE_CANARY"):
            assert want in plog, (want, plog)
        if mirror:
            for want in ("PROBE_FETCH ranks_ok=12", "PROBE_DECISION pfetch_streams=", "PROBE_DECISION pfetch_assign=", "PROBE_DECISION per_target_limited=",
                         "PROBE_DECISION put_streams=", "PROBE_DECISION local_sparse=", "PROBE_DECISION mpi_ring_rank_MBps=",
                         "pack_io_bound=", "pack_source=mirror", "PROBE_DECISION child_mpi="):
                assert want in plog, (want, plog)
        print("ds41f pjm PASS: 12 fake ranks pack 40 layers x 2 packs + trunk (complete, nmse summary, assembled pack verifies), "
              + ("node-local mirror via strata-pfetch == shared reads, " if mirror else "mirror SKIPPED (no strata-pfetch), ") +
              "resubmission exits early, a deadline-bound job asks for resubmission, a stale binary is refused, "
              "two jobs on different packs run in parallel, a held pack lock refuses and a stale one is taken over; "
              "probe: pfetch rates + mirror fetch, 12 settings packed, write/coherence/single-node write measured, decisions printed")


if __name__ == "__main__":
    main()
