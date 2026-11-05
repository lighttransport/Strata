#!/usr/bin/env python3
"""Tests for jobctl with no cluster: the 'login node' is the local shell, and pjsub/pjstat/pjdel are stubs.

Run:  python3 tools/jobctl/test_jobctl.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jobctl as C  # noqa: E402

PJSUB = """#!/bin/bash
echo "$@" >> "$STUB_DIR/pjsub.log"
n=$(( $(cat "$STUB_DIR/counter" 2>/dev/null || echo 1000) + 1 )); echo $n > "$STUB_DIR/counter"
echo "[INFO] PJM 0000 pjsub Job $n submitted."
"""
PJSTAT = """#!/bin/bash
cat "$STUB_DIR/pjstat.txt" 2>/dev/null
"""
PJDEL = """#!/bin/bash
echo "$@" >> "$STUB_DIR/pjdel.log"
"""


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        self.stub = os.path.join(t, "stub")
        os.makedirs(self.stub)
        for n, body in (("pjsub", PJSUB), ("pjstat", PJSTAT), ("pjdel", PJDEL)):
            p = os.path.join(self.stub, n)
            open(p, "w").write(body)
            os.chmod(p, 0o755)
        self.env = dict(os.environ)
        os.environ["STUB_DIR"] = self.stub
        os.environ["PATH"] = self.stub + os.pathsep + os.environ["PATH"]
        self.forwards = []
        self.cfg = dict(C.DEFAULTS)
        self.cfg.update({
            "ssh_cmd": ["bash", "-c"], "frontend_target": "login-stub", "bundle": os.path.join(t, "bundle"),
            "scheduler_opts": {"project": "grp", "rscgrp": "small", "node_spec": {"12": "2x3x2"}, "resource": {"freq": 2000}},
            "registry": os.path.join(t, "reg"), "rundir": os.path.join(t, "runs"), "submit_interval": 0,
            "local_ports": [45000, 45299], "frontend_ports": [35000, 35299], "interval": 1,
            "_forward_hook": lambda op, l, f: self.forwards.append((op, l, f)) or 0,
        })

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env)
        self.tmp.cleanup()

    def launch(self, name, nodes=1, time_="01:00:00", script=None, extra=None):
        class A:
            pass
        a = A()
        a.name, a.nodes, a.time, a.script, a.dry_run = name, nodes, time_, script, False
        a.variants = a.lead_min = a.queue_ahead = a.env = a.sched_arg = None
        for k, v in (extra or {}).items():
            setattr(a, k, v)
        C.cmd_launch(self.cfg, a)

    def pjsub_log(self):
        p = os.path.join(self.stub, "pjsub.log")
        return open(p).read().splitlines() if os.path.exists(p) else []


class PortsAndLaunch(Base):
    def test_ports_unique_for_256_jobs(self):
        seen = set()
        for i in range(256):
            lp, fp = C.alloc_ports(self.cfg)
            self.assertNotIn(lp, seen)
            seen.add(lp)
            C.save_job(self.cfg, {"name": "j%d" % i, "local_port": lp, "frontend_port": fp, "state": "running",
                                  "nodes": 1, "elapse": "01:00:00", "jobids": ["1"], "seq": 1, "run": "x", "hb": 0})
        self.assertEqual(len(seen), 256)
        with self.assertRaises(SystemExit):   # only 300 ports configured; fill the rest, then one more
            for i in range(256, 400):
                lp, fp = C.alloc_ports(self.cfg)
                C.save_job(self.cfg, {"name": "j%d" % i, "local_port": lp, "frontend_port": fp, "state": "running",
                                      "nodes": 1, "elapse": "01:00:00", "jobids": ["1"], "seq": 1, "run": "x", "hb": 0})

    def test_launch_writes_everything(self):
        self.launch("alpha", nodes=12)
        run = os.path.join(self.cfg["rundir"], "alpha")
        for d in ("state", "flags", "inbox", "outbox", "chain", "work", "logs"):
            self.assertTrue(os.path.isdir(os.path.join(run, d)), d)
        spec = json.load(open(run + "/spec.json"))
        joined = " ".join(spec["variants"][0])
        self.assertIn("rscgrp=small,node=2x3x2,elapse=01:00:00,freq=2000", joined)
        self.assertIn("{SEQ}", joined)
        self.assertEqual(open(run + "/chain/seq").read().strip(), "1")
        self.assertTrue(open(run + "/chain/jobs.log").read().startswith("1 1001 "))
        self.assertIn("CTL_NODES=12", joined)
        self.assertEqual(len(self.pjsub_log()), 1)
        job = C.load_job(self.cfg, "alpha")
        self.assertEqual(job["jobids"], ["1001"])
        files = os.listdir(os.path.join(self.cfg["rundir"], "_bin"))
        self.assertEqual(len([f for f in files if f.startswith("agent.")]), 1)
        self.assertEqual(len([f for f in files if f.startswith("jobctl.")]), 1)

    def test_node_and_name_limits(self):
        self.launch("big", nodes=384)
        self.assertIn("node=384", self.pjsub_log()[0])
        with self.assertRaises(SystemExit):
            self.launch("toobig", nodes=385)
        with self.assertRaises(SystemExit):
            self.launch("bad name")
        with self.assertRaises(SystemExit):
            self.launch("t", time_="1h")
        with self.assertRaises(SystemExit):
            self.launch("big")   # exists

    def test_variants_race_launch(self):
        self.launch("race", extra={"variants": "4@02:00:00,2@01:00:00"})
        log = self.pjsub_log()
        self.assertEqual(len(log), 2)
        self.assertIn("node=4,elapse=02:00:00", log[0])
        self.assertIn("node=2,elapse=01:00:00", log[1])
        self.assertIn("CTL_RACE=1", log[0])
        self.assertIn("CTL_NODES=2", log[1])
        self.assertIn("CTL_VARIANT=1", log[1])
        run = os.path.join(self.cfg["rundir"], "race")
        rows = open(run + "/chain/jobs.log").read().splitlines()
        self.assertEqual([r.split()[3] for r in rows], ["R", "R"])
        self.assertEqual(open(run + "/chain/seq").read().strip(), "2")
        self.assertEqual(len(C.load_job(self.cfg, "race")["jobids"]), 2)

    def test_dry_run_submits_nothing(self):
        self.launch("dry", extra={"dry_run": True, "variants": "3@01:00:00,1@00:30:00"})
        self.assertEqual(self.pjsub_log(), [])
        self.assertFalse(os.path.exists(C.job_path(self.cfg, "dry")))

    def test_no_tunnel_mode_uses_no_ports(self):
        self.cfg["tunnel"] = False
        self.launch("nt")
        job = C.load_job(self.cfg, "nt")
        self.assertEqual((job["local_port"], job["frontend_port"]), (0, 0))
        self.assertIn("CTL_TUNNEL=0", self.pjsub_log()[0])
        self.assertNotIn("FRONTEND_SSH_TARGET", self.pjsub_log()[0])

    def test_live_cap(self):
        self.cfg["max_live"] = 2
        self.launch("a")
        self.launch("b")
        with self.assertRaises(SystemExit):
            self.launch("c")

    def test_many_launches_get_distinct_ports(self):
        for i in range(24):
            self.launch("m%02d" % i)
        ports = [j["local_port"] for j in C.all_jobs(self.cfg)]
        self.assertEqual(len(set(ports)), 24)

    def test_rejects_exit_in_command(self):
        self.launch("x")
        class A:
            name, command, timeout, inbox = "x", "echo hi; exit", 5, True
        with self.assertRaises(SystemExit):
            C.cmd_run(self.cfg, A)


class Agent(Base):
    def run_agent(self, run, ticks, extra=None, name="t"):
        env = dict(os.environ)
        env.update({"CTL_RUNDIR": run, "CTL_NAME": name, "CTL_INTERVAL": "1", "CTL_MAX_TICKS": str(ticks),
                    "CTL_NO_SERVER": "1", "CTL_NO_TUNNEL": "1", "CTL_ELAPSE_SEC": "100000", "PJM_JOBID": "77"})
        env.update(extra or {})
        return subprocess.run(["bash", os.path.join(HERE, "agent.sh")], env=env, timeout=60,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_inbox_heartbeat_and_pause(self):
        run = os.path.join(self.tmp.name, "r1")
        os.makedirs(run + "/inbox")
        open(run + "/inbox/0001.cmd", "w").write("echo hello; echo err >&2; false\n")
        self.run_agent(run, 3)
        self.assertEqual(open(run + "/outbox/0001.rc").read().strip(), "1")
        out = open(run + "/outbox/0001.out").read()
        self.assertIn("hello", out)
        self.assertIn("err", out)
        info = json.load(open(run + "/state/info.json"))
        self.assertEqual(info["jobid"], "77")
        self.assertAlmostEqual(info["hb"], time.time(), delta=20)
        self.assertTrue(os.path.exists(run + "/state/port.77"))
        # PAUSE holds new commands
        os.makedirs(run + "/flags", exist_ok=True)
        open(run + "/flags/PAUSE", "w").close()
        open(run + "/inbox/0002.cmd", "w").write("echo later\n")
        self.run_agent(run, 2)
        self.assertFalse(os.path.exists(run + "/outbox/0002.rc"))
        self.assertEqual(json.load(open(run + "/state/info.json"))["phase"], "paused")
        os.remove(run + "/flags/PAUSE")
        self.run_agent(run, 3)
        self.assertEqual(open(run + "/outbox/0002.rc").read().strip(), "0")

    def test_commands_run_in_order_one_at_a_time(self):
        run = os.path.join(self.tmp.name, "r2")
        os.makedirs(run + "/inbox")
        for i in (1, 2, 3):
            open(run + "/inbox/%04d.cmd" % i, "w").write("date +%%s%%N >> %s/order.txt\nsleep 0.3\n" % run)
        self.run_agent(run, 8)
        rcs = sorted(f for f in os.listdir(run + "/outbox") if f.endswith(".rc"))
        self.assertEqual(rcs, ["0001.rc", "0002.rc", "0003.rc"])
        t = [int(x) for x in open(run + "/order.txt").read().split()]
        self.assertEqual(t, sorted(t))

    def test_stop_flag_exits(self):
        run = os.path.join(self.tmp.name, "r3")
        os.makedirs(run + "/flags")
        open(run + "/flags/STOP", "w").close()
        p = self.run_agent(run, 50)
        self.assertEqual(p.returncode, 0)
        self.assertEqual(json.load(open(run + "/state/info.json"))["phase"], "stopped")

    def test_drain_and_low_walltime_request_successor_once(self):
        run = os.path.join(self.tmp.name, "r4")
        os.makedirs(run + "/flags")
        open(run + "/flags/DRAIN", "w").close()
        self.run_agent(run, 2)
        self.assertTrue(os.path.exists(run + "/chain/next.req"))
        run2 = os.path.join(self.tmp.name, "r5")
        self.run_agent(run2, 2, {"CTL_ELAPSE_SEC": "100", "CTL_LEAD_MIN": "10"})
        self.assertTrue(os.path.exists(run2 + "/chain/next.req"))
        run3 = os.path.join(self.tmp.name, "r6")
        self.run_agent(run3, 2, {"CTL_ELAPSE_SEC": "100", "CTL_LEAD_MIN": "10", "CTL_CHAIN": "0"})
        self.assertFalse(os.path.exists(run3 + "/chain/next.req"))

    def test_workload_exit_codes(self):
        for rc, want_req, want_done in ((0, False, True), (75, True, False), (3, False, False)):
            run = os.path.join(self.tmp.name, "w%d" % rc)
            os.makedirs(run)
            open(run + "/work.sh", "w").write("echo step; exit %d\n" % rc)
            self.run_agent(run, 4)
            self.assertEqual(os.path.exists(run + "/chain/next.req"), want_req, rc)
            self.assertEqual(os.path.exists(run + "/work/DONE"), want_done, rc)
            if rc == 3:
                self.assertTrue(os.path.exists(run + "/flags/STOP"))
            if rc == 75:   # the agent leaves at once and frees the lock
                self.assertEqual([f for f in os.listdir(run + "/chain") if f.startswith("active.")], [])
                self.assertEqual(json.load(open(run + "/state/info.json"))["phase"], "suspended")


class Chain(Base):
    def prepare(self, name="c"):
        self.launch(name)
        return os.path.join(self.cfg["rundir"], name)

    def test_next_req_submits_successor_and_advances_seq(self):
        run = self.prepare()
        open(run + "/chain/next.req", "w").write("x\n")
        msgs = C.chain_tick(self.cfg["rundir"])
        self.assertEqual(len(msgs), 1)
        self.assertEqual(open(run + "/chain/seq").read().strip(), "2")
        self.assertFalse(os.path.exists(run + "/chain/next.req"))
        self.assertEqual(len(self.pjsub_log()), 2)
        self.assertIn("job.2.out", self.pjsub_log()[1])
        self.assertIn("CTL_SEQ=2", self.pjsub_log()[1])
        self.assertEqual(len(open(run + "/chain/jobs.log").read().splitlines()), 2)
        self.assertEqual(C.chain_tick(self.cfg["rundir"]), [])   # nothing new wanted

    def test_stop_and_done_block_chain(self):
        run = self.prepare()
        open(run + "/chain/next.req", "w").write("x\n")
        open(run + "/flags/STOP", "w").close()
        self.assertEqual(C.chain_tick(self.cfg["rundir"]), [])
        os.remove(run + "/flags/STOP")
        os.makedirs(run + "/work", exist_ok=True)
        open(run + "/work/DONE", "w").close()
        self.assertEqual(C.chain_tick(self.cfg["rundir"]), [])
        self.assertEqual(len(self.pjsub_log()), 1)

    def test_lock_prevents_double_submit(self):
        run = self.prepare()
        open(run + "/chain/next.req", "w").write("x\n")
        os.mkdir(run + "/chain/lock")
        self.assertEqual(C.chain_tick(self.cfg["rundir"]), [])
        self.assertEqual(len(self.pjsub_log()), 1)

    def test_stale_heartbeat_resubmits_only_when_nothing_queued(self):
        run = self.prepare()
        old = {"hb": time.time() - 600, "phase": "running", "frontend_port": 35000}
        json.dump(old, open(run + "/state/info.json", "w"))
        open(os.path.join(self.stub, "pjstat.txt"), "w").write("1001 c ST RUN u g\n")   # still active
        self.assertEqual(C.chain_tick(self.cfg["rundir"]), [])
        open(os.path.join(self.stub, "pjstat.txt"), "w").write("")                       # gone
        msgs = C.chain_tick(self.cfg["rundir"])
        self.assertEqual(len(msgs), 1)
        self.assertIn("stale", msgs[0])

    def test_max_chain(self):
        run = self.prepare()
        spec = json.load(open(run + "/spec.json"))
        spec["max_chain"] = 1
        json.dump(spec, open(run + "/spec.json", "w"))
        open(run + "/chain/next.req", "w").write("x\n")
        self.assertEqual(C.chain_tick(self.cfg["rundir"]), [])
        self.assertEqual(len(self.pjsub_log()), 1)


class FakeBridge(Base):
    """A tiny server speaking the bridge protocol (POST /session, POST /run NDJSON, GET /health)."""

    def serve(self, drop_first_session=False):
        import http.server, threading
        state = {"sessions": set(), "n": 0}

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send(200, b'{"ok": true}')

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/session":
                    state["n"] += 1
                    sid = "s%d" % state["n"]
                    state["sessions"].add(sid)
                    return self._send(200, json.dumps({"session": sid}).encode())
                if body.get("session") not in state["sessions"]:
                    return self._send(404, b"{}")
                out = ("ran:" + body["command"] + "\n").encode()
                lines = [{"type": "stdout", "data": out.decode()}, {"type": "exit", "code": 4 if "fail" in body["command"] else 0}]
                self._send(200, "".join(json.dumps(l) + "\n" for l in lines).encode(), "application/x-ndjson")

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        return srv.server_address[1], state

    def test_run_over_http_and_session_recovery(self):
        import io, contextlib
        port, state = self.serve()
        self.launch("h")
        job = C.load_job(self.cfg, "h")
        job["local_port"] = port
        C.save_job(self.cfg, job)
        class A:
            name, command, timeout, inbox = "h", "echo hi", 5, False
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(C.cmd_run(self.cfg, A), 0)
        self.assertIn("ran:echo hi", buf.getvalue())
        self.assertEqual(C.load_job(self.cfg, "h")["session"], "s1")
        state["sessions"].clear()                      # idle sweeper removed it
        with contextlib.redirect_stdout(buf):
            self.assertEqual(C.cmd_run(self.cfg, A), 0)
        self.assertEqual(C.load_job(self.cfg, "h")["session"], "s2")
        A.command = "do fail"
        with contextlib.redirect_stdout(buf):
            self.assertEqual(C.cmd_run(self.cfg, A), 4)


class Control(Base):
    def test_flags_and_kill(self):
        self.launch("f")
        run = os.path.join(self.cfg["rundir"], "f")
        class A:
            name = "f"
        C.cmd_flag(self.cfg, A, "PAUSE")
        self.assertTrue(os.path.exists(run + "/flags/PAUSE"))
        C.cmd_flag(self.cfg, A, "PAUSE", False)
        self.assertFalse(os.path.exists(run + "/flags/PAUSE"))
        C.cmd_kill(self.cfg, A)
        self.assertTrue(os.path.exists(run + "/flags/STOP"))
        self.assertIn("1001", open(os.path.join(self.stub, "pjdel.log")).read())
        self.assertEqual(C.load_job(self.cfg, "f")["state"], "stopped")

    def test_inbox_roundtrip_with_agent(self):
        self.launch("i")
        run = os.path.join(self.cfg["rundir"], "i")
        env = dict(os.environ)
        env.update({"CTL_RUNDIR": run, "CTL_NAME": "i", "CTL_INTERVAL": "1", "CTL_MAX_TICKS": "12",
                    "CTL_NO_SERVER": "1", "CTL_NO_TUNNEL": "1", "CTL_ELAPSE_SEC": "100000"})
        agent = subprocess.Popen(["bash", os.path.join(HERE, "agent.sh")], env=env)
        try:
            job = C.load_job(self.cfg, "i")
            self.cfg["interval"] = 1
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = C.run_inbox(self.cfg, job, "echo from-inbox; exit_code=3; (exit $exit_code)", 20)
            self.assertEqual(rc, 3)
            self.assertIn("from-inbox", buf.getvalue())
        finally:
            agent.terminate()
            agent.wait()

    def test_refresh_states_and_port_change(self):
        self.launch("r")
        run = os.path.join(self.cfg["rundir"], "r")
        jobs = C.all_jobs(self.cfg)
        remote = C.Remote(self.cfg)
        info = {"hb": time.time(), "phase": "running", "frontend_port": jobs[0]["frontend_port"]}
        json.dump(info, open(run + "/state/info.json", "w"))
        open(os.path.join(self.stub, "pjstat.txt"), "w").write(
            "JOB_ID JOB_NAME MD ST USER GROUP\n1001 r NM RUN u g\n")
        C.refresh(self.cfg, remote, jobs)
        job = C.load_job(self.cfg, "r")
        self.assertEqual(job["state"], "running")
        self.assertIn(("forward", job["local_port"], job["frontend_port"]), self.forwards)
        # the agent moved to a new frontend port: cancel the old forward, add the new one
        old = job["frontend_port"]
        info["frontend_port"] = old + 7
        json.dump(info, open(run + "/state/info.json", "w"))
        C.refresh(self.cfg, remote, [job])
        job = C.load_job(self.cfg, "r")
        self.assertEqual(job["frontend_port"], old + 7)
        self.assertIn(("cancel", job["local_port"], old), self.forwards)
        self.assertIn(("forward", job["local_port"], old + 7), self.forwards)
        # queued
        open(os.path.join(self.stub, "pjstat.txt"), "w").write("1001 r NM QUE u g\n")
        job["state"] = "submitted"
        C.save_job(self.cfg, job)
        C.refresh(self.cfg, remote, [job])
        self.assertEqual(C.load_job(self.cfg, "r")["state"], "queued")

    def test_report_lists_runs(self):
        self.launch("p1")
        self.launch("p2")
        rep = C.report(self.cfg["rundir"])
        self.assertEqual(sorted(rep), ["p1", "p2"])
        self.assertEqual(rep["p1"]["seq"], 1)
        self.assertEqual(rep["p1"]["jobs"], ["1001"])


class Coexist(Base):
    def test_listening_ports_parser(self):
        sample = ("  sl  local_address rem_address   st tx_queue rx_queue\n"
                  "   0: 0100007F:80E8 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1000\n"
                  "   1: 0100007F:8001 0100007F:C000 01 00000000:00000000 00:00000000 00000000 1000\n"
                  "   2: 00000000000000000000000001000000:5208 00000000000000000000000000000000:0000 0A 0:0 00:0 0 1000\n")
        self.assertEqual(C.parse_proc_net_tcp(sample), {0x80E8, 0x5208})

    def slot(self, name, lport, fport, nodes=12):
        d = os.path.join(self.tmp.name, "slots", name)
        os.makedirs(d)
        open(os.path.join(d, "slot.env"), "w").write("LOCAL_PORT=%d\nFRONTEND_PORT=%d\nNODES=%d\nNODE_SPEC=2x3x2\n" % (lport, fport, nodes))

    def test_default_ranges_are_disjoint_from_interactive_ones(self):
        C.check_ranges(dict(C.DEFAULTS))
        bad = dict(C.DEFAULTS, local_ports=[42900, 43100])
        with self.assertRaises(SystemExit):
            C.check_ranges(bad)
        bad = dict(C.DEFAULTS, frontend_ports=[31000, 32000])
        with self.assertRaises(SystemExit):
            C.check_ranges(bad)

    def test_external_slot_ports_are_never_handed_out(self):
        # an interactive slot that sits inside the range jobctl uses (a misconfiguration the allocator must survive)
        self.slot("a", 45000, 35001)
        self.slot("b", 45001, 35000)
        self.cfg["external_slots"] = [os.path.join(self.tmp.name, "slots", "*", "slot.env")]
        lp, fp = C.alloc_ports(self.cfg)
        self.assertEqual((lp, fp), (45002, 35002))
        self.assertEqual(sorted(e["name"] for e in C.external_slots(self.cfg)), ["ext:a", "ext:b"])

    def test_ports_command_flags_conflicts(self):
        self.slot("x", 42474, 32474)
        self.cfg["external_slots"] = [os.path.join(self.tmp.name, "slots", "*", "slot.env")]
        self.launch("j1")
        class A:
            check = None
        self.assertEqual(C.cmd_ports(self.cfg, A), 0)
        job = C.load_job(self.cfg, "j1")
        job["local_port"], job["frontend_port"] = 42474, 32474     # force a clash with the slot
        C.save_job(self.cfg, job)
        self.assertEqual(C.cmd_ports(self.cfg, A), 1)

    def test_check_pair_for_an_interactive_launch(self):
        class A:
            check = ["42500", "32500"]
        self.assertEqual(C.cmd_ports(self.cfg, A), 0)
        A.check = ["45010", "32500"]          # local port inside jobctl's range
        self.assertEqual(C.cmd_ports(self.cfg, A), 1)

    def test_agent_repick_stays_inside_the_range(self):
        # the agent only walks FLO..FHI; the range comes from jobctl's config, which check_ranges keeps clear of the interactive one
        src = open(os.path.join(HERE, "agent.sh")).read()
        self.assertIn("FLO + ( FPORT - FLO + i ) % size", src)


class Adapters(unittest.TestCase):
    def test_slurm_command_and_parsing(self):
        sl = C.scheduler("slurm")
        req = {"name": "n", "nodes": 4, "elapse": "30:15:00", "env": {"A": "1"}, "script": "job.sh", "out": "o", "err": "e"}
        argv = sl.submit_argv({"account": "acct", "partition": "p", "extra": ["--exclusive"]}, req)
        self.assertEqual(argv[:2], ["sbatch", "--parsable"])
        self.assertIn("1-06:15:00", argv)
        self.assertIn("ALL,A=1", argv)
        self.assertEqual(argv[-1], "job.sh")
        self.assertEqual(sl.parse_jobid("123456\n"), "123456")
        self.assertEqual(sl.parse_states("11 RUNNING\n12 PENDING\n"), {"11": "RUN", "12": "QUE"})
        self.assertEqual(sl.cancel_cmd("12"), "scancel 12")

    def test_pjm_states(self):
        pj = C.scheduler("pjm")
        text = "JOB_ID JOB_NAME MD ST USER\n11 a NM RUN u\n12 b NM QUE u\n13 c NM RNE u\n"
        self.assertEqual(pj.parse_states(text), {"11": "RUN", "12": "QUE", "13": "RUN"})

    def test_unknown_scheduler(self):
        with self.assertRaises(SystemExit):
            C.scheduler("nope")


class SpeculativeChain(Base):
    def test_winner_cancels_queued_racers_and_successor_queued_ahead(self):
        self.launch("sp", extra={"variants": "4@01:00:00,2@01:00:00,1@01:00:00"})
        run = os.path.join(self.cfg["rundir"], "sp")
        # variant 1 (job 1002) started and won; the others still wait
        open(run + "/chain/winner", "w").write("2 1\n")
        open(os.path.join(self.stub, "pjstat.txt"), "w").write("1001 sp NM QUE u g\n1002 sp NM RUN u g\n1003 sp NM QUE u g\n")
        msgs = C.chain_tick(self.cfg["rundir"])
        self.assertEqual(len([m for m in msgs if "cancelled" in m]), 2)
        dels = open(os.path.join(self.stub, "pjdel.log")).read().split()
        self.assertEqual(sorted(dels), ["1001", "1003"])
        # a successor was queued ahead, using the winning variant (2 nodes)
        self.assertEqual(len([m for m in msgs if "queued ahead" in m]), 1)
        last = self.pjsub_log()[-1]
        self.assertIn("node=2,elapse=01:00:00", last)
        self.assertIn("CTL_SEQ=4", last)
        rows = open(run + "/chain/jobs.log").read().splitlines()
        self.assertEqual(rows[-1].split()[3], "S")
        # once that successor is queued, no second one is added (queue_ahead = 1)
        open(os.path.join(self.stub, "pjstat.txt"), "w").write("1002 sp NM RUN u g\n1004 sp NM QUE u g\n")
        self.assertEqual([m for m in C.chain_tick(self.cfg["rundir"]) if "queued ahead" in m], [])

    def test_successors_are_not_racers(self):
        self.launch("sr", extra={"variants": "2@01:00:00,1@01:00:00"})
        run = os.path.join(self.cfg["rundir"], "sr")
        open(run + "/chain/winner", "w").write("1 0\n")
        open(run + "/chain/next.req", "w").write("x\n")
        C.chain_tick(self.cfg["rundir"])
        self.assertIn("CTL_RACE=1", self.pjsub_log()[0])
        self.assertIn("CTL_RACE=0", self.pjsub_log()[-1])

    def test_no_early_successor_before_a_winner(self):
        self.launch("nw")
        open(os.path.join(self.stub, "pjstat.txt"), "w").write("1001 nw NM RUN u g\n")
        self.assertEqual(C.chain_tick(self.cfg["rundir"]), [])

    def test_queue_ahead_zero_waits_for_request(self):
        self.launch("q0", extra={"queue_ahead": 0})
        run = os.path.join(self.cfg["rundir"], "q0")
        open(run + "/chain/winner", "w").write("1 0\n")
        open(os.path.join(self.stub, "pjstat.txt"), "w").write("1001 q0 NM RUN u g\n")
        self.assertEqual(C.chain_tick(self.cfg["rundir"]), [])


class AgentChain(Base):
    def agent_env(self, run, seq, **kw):
        env = dict(os.environ)
        env.update({"CTL_RUNDIR": run, "CTL_NAME": "t", "CTL_INTERVAL": "1", "CTL_NO_SERVER": "1", "CTL_NO_TUNNEL": "1",
                    "CTL_ELAPSE_SEC": "100000", "CTL_SEQ": str(seq), "PJM_JOBID": str(900 + seq), "CTL_WAIT_POLL": "1",
                    "CTL_GRACE": "10", "CTL_STALE_SEC": "20"})
        env.update({k: str(v) for k, v in kw.items()})
        return env

    def start(self, run, seq, **kw):
        return subprocess.Popen(["bash", os.path.join(HERE, "agent.sh")], env=self.agent_env(run, seq, **kw))

    def wait_for(self, pred, timeout=30):
        end = time.time() + timeout
        while time.time() < end:
            if pred():
                return True
            time.sleep(0.2)
        return False

    def test_second_racer_leaves_at_once(self):
        run = os.path.join(self.tmp.name, "race")
        a = self.start(run, 1, CTL_RACE=1)
        try:
            self.assertTrue(self.wait_for(lambda: any(f.startswith("active.") for f in os.listdir(run + "/chain")) if os.path.isdir(run + "/chain") else False))
            b = self.start(run, 2, CTL_RACE=1, CTL_VARIANT=1)
            self.assertEqual(b.wait(timeout=20), 0)
            self.assertIn("lost the race", open(run + "/logs/agent.2.log").read())
            self.assertEqual(open(run + "/chain/winner").read().split(), ["1", "0"])
            self.assertEqual(json.load(open(run + "/state/info.json"))["seq"], 1)
        finally:
            a.terminate()
            a.wait()
        self.assertEqual([f for f in os.listdir(run + "/chain") if f.startswith("active.")], [])   # released on exit

    def test_handoff_resumes_workload_from_checkpoint(self):
        run = os.path.join(self.tmp.name, "hand")
        os.makedirs(run)
        open(run + "/work.sh", "w").write(
            "f=$CTL_WORK/c; n=$(cat $f 2>/dev/null || echo 0)\n"
            "echo resume-at-$n seq=$CTL_SEQ\n"
            "trap 'exit 75' TERM\n"
            "while [ $n -lt 12 ]; do sleep 1 & wait $!; n=$((n+1)); echo $n > $f.t && mv $f.t $f; done\n"
            "exit 0\n")
        a = self.start(run, 1)
        b = None
        try:
            self.assertTrue(self.wait_for(lambda: os.path.exists(run + "/work/c") and int(open(run + "/work/c").read() or 0) >= 3))
            b = self.start(run, 2)        # the successor starts while the first allocation still runs
            self.assertEqual(a.wait(timeout=40), 0)      # the holder hands over and leaves
            mid = int(open(run + "/work/c").read())
            self.assertLess(mid, 12)
            self.assertEqual(b.wait(timeout=60), 0)      # the successor finishes the run
        finally:
            for p in (a, b):
                if p and p.poll() is None:
                    p.terminate()
                    p.wait()
        self.assertEqual(open(run + "/work/c").read().strip(), "12")
        self.assertTrue(os.path.exists(run + "/work/DONE"))
        w2 = open(run + "/logs/work.2.log").read()
        self.assertRegex(w2, r"resume-at-([3-9]|1[01]) seq=2")
        w1 = open(run + "/logs/work.1.log").read()
        self.assertIn("resume-at-0 seq=1", w1)

    def test_finished_workload_is_done_not_handed_over(self):
        run = os.path.join(self.tmp.name, "fin")
        os.makedirs(run + "/chain")
        open(run + "/work.sh", "w").write("echo done\nexit 0\n")
        open(run + "/chain/up.2.%d" % (time.time() + 5), "w").close()     # a successor is waiting
        p = self.start(run, 1, CTL_MAX_TICKS=5)
        self.assertEqual(p.wait(timeout=30), 0)
        self.assertTrue(os.path.exists(run + "/work/DONE"))
        self.assertNotIn("handoff", open(run + "/logs/agent.1.log").read())

    def test_stale_holder_is_taken_over(self):
        run = os.path.join(self.tmp.name, "stale")
        os.makedirs(run + "/chain")
        open(run + "/chain/active.1.%d" % (time.time() - 3600), "w").close()
        p = self.start(run, 2, CTL_MAX_TICKS=2)
        self.assertEqual(p.wait(timeout=30), 0)
        self.assertIn("agent start seq=2", open(run + "/logs/agent.2.log").read())

    def test_stale_successor_announcement_is_ignored(self):
        run = os.path.join(self.tmp.name, "ign")
        os.makedirs(run + "/chain")
        open(run + "/chain/up.5.%d" % (time.time() - 3600), "w").close()
        p = self.start(run, 1, CTL_MAX_TICKS=3)
        self.assertEqual(p.wait(timeout=30), 0)
        self.assertNotIn("handoff", open(run + "/logs/agent.1.log").read())


if __name__ == "__main__":
    unittest.main(verbosity=2)
