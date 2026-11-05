#!/usr/bin/env python3
"""Tests for the jobctl web board and API with no cluster (same stubs as test_jobctl.py).

Run:  python3 tools/jobctl/test_webui.py
"""
import hashlib
import hmac
import http.client
import http.server
import json
import os
import sys
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jobctl as C  # noqa: E402
import webui as W  # noqa: E402
from test_jobctl import Base  # noqa: E402

TOKEN = "t0k3n-for-tests"


class Receiver(object):
    """A local webhook receiver. Answers the first `fail` requests with 500."""

    def __init__(self, fail=0):
        self.got, self.fail = [], fail
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                outer.got.append((self.rfile.read(n), dict(self.headers)))
                code = 500 if outer.fail > 0 else 200
                outer.fail -= 1
                self.send_response(code)
                self.send_header("Content-Length", "0")
                self.end_headers()

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d/hook" % self.srv.server_address[1]

    def stop(self):
        self.srv.shutdown()


class WebBase(Base):
    def setUp(self):
        Base.setUp(self)
        self.cfg["webui"] = {}
        self.start()

    def start(self, hooks=None, confirm=True):
        self.cfg["webui"] = {"require_confirm": confirm}
        self.svc = W.Service(self.cfg, TOKEN, hooks=hooks or W.Webhooks([]))
        self.srv = W.make_server(self.svc, "127.0.0.1", 0)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(self.srv.shutdown)
        self.addCleanup(lambda: setattr(self.svc, "stopping", True))

    def req(self, method, path, body=None, token=TOKEN, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        h = dict(headers or {})
        if token:
            h["Authorization"] = "Bearer " + token
        data = json.dumps(body).encode() if body is not None else None
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        raw = r.read()
        c.close()
        try:
            return r.status, json.loads(raw.decode())
        except ValueError:
            return r.status, raw

    def set_queue(self, text):
        open(os.path.join(self.stub, "pjstat.txt"), "w").write(text)

    def info(self, name, **kw):
        run = os.path.join(self.cfg["rundir"], name)
        d = {"hb": time.time(), "phase": "running", "frontend_port": C.load_job(self.cfg, name)["frontend_port"]}
        d.update(kw)
        json.dump(d, open(run + "/state/info.json", "w"))

    def types(self, name=None):
        return [e["type"] for e in self.svc.events if name is None or e["job"] == name]


class Security(WebBase):
    def test_token_host_origin(self):
        self.assertEqual(self.req("GET", "/api/v1/jobs", token=None)[0], 401)
        self.assertEqual(self.req("GET", "/api/v1/jobs", token="wrong")[0], 401)
        self.assertEqual(self.req("GET", "/api/v1/spec", token=None)[0], 200)
        self.assertEqual(self.req("GET", "/api/v1/jobs")[0], 200)
        self.assertEqual(self.req("GET", "/api/v1/jobs", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.req("GET", "/api/v1/jobs", headers={"Origin": "http://evil.example"})[0], 403)
        self.assertEqual(self.req("POST", "/api/v1/refresh", {}, headers={"Origin": "null"})[0], 403)
        self.assertEqual(self.req("GET", "/api/v1/nothing")[0], 404)

    def test_page_and_csp(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port)
        c.request("GET", "/")
        r = c.getresponse()
        body = r.read().decode()
        self.assertEqual(r.status, 200)
        self.assertIn("jobctl board", body)
        self.assertIn("default-src 'self'", r.getheader("Content-Security-Policy"))
        self.assertNotIn(TOKEN, body)

    def test_spec_lists_every_route(self):
        st, spec = self.req("GET", "/api/v1/spec", token=None)
        listed = set((r["method"], r["path"]) for r in spec["routes"])
        self.assertEqual(listed, set((r[0], r[6]) for r in W.ROUTES))
        self.assertEqual(len(listed), len(W.ROUTES))
        for m, rx, fn, *_ in W.ROUTES:
            self.assertTrue(fn == "api_stream" or hasattr(self.svc, fn), fn)


class Actions(WebBase):
    def test_list_detail_and_flags(self):
        self.launch("a")
        st, d = self.req("GET", "/api/v1/jobs")
        self.assertEqual([j["name"] for j in d["jobs"]], ["a"])
        self.assertNotIn("run", d["jobs"][0])
        self.assertEqual(self.req("GET", "/api/v1/jobs/zzz")[0], 404)
        st, r = self.req("POST", "/api/v1/jobs/a/pause", {})
        self.assertEqual(st, 200)
        self.assertTrue(os.path.exists(os.path.join(self.cfg["rundir"], "a", "flags", "PAUSE")))
        self.req("POST", "/api/v1/jobs/a/resume", {})
        self.assertFalse(os.path.exists(os.path.join(self.cfg["rundir"], "a", "flags", "PAUSE")))
        self.assertEqual(self.types("a").count("job.flag"), 2)
        self.assertEqual(self.req("POST", "/api/v1/jobs/zzz/stop", {})[0], 404)

    def test_run_refuses_exit(self):
        self.launch("a")
        st, r = self.req("POST", "/api/v1/jobs/a/run", {"command": "exit 1"})
        self.assertEqual(st, 200)
        self.assertFalse(r["ok"])
        self.assertIn("refusing", r["error"])
        self.assertEqual(self.req("POST", "/api/v1/jobs/a/run", {})[0], 400)
        self.assertEqual(self.req("POST", "/api/v1/jobs/a/run", {"command": "echo hi", "timeout": 0})[0], 400)

    def test_bad_input(self):
        self.assertEqual(self.req("POST", "/api/v1/jobs", {"name": "bad name"})[0], 400)
        self.assertEqual(self.req("POST", "/api/v1/jobs", {"name": "x", "nodes": 9999})[0], 400)
        self.assertEqual(self.req("POST", "/api/v1/jobs", {"name": "x", "time": "soon"})[0], 400)
        self.assertEqual(self.req("POST", "/api/v1/jobs", {"name": "x", "env": {"A B": "1"}})[0], 400)
        self.assertEqual(self.req("POST", "/api/v1/jobs", {"name": "x", "env": {"A": "1\nrm"}})[0], 400)
        self.assertEqual(self.pjsub_log(), [])


class TwoStep(WebBase):
    def test_launch_needs_the_code(self):
        body = {"name": "L1", "nodes": 2, "time": "00:30:00"}
        st, p = self.req("POST", "/api/v1/jobs", body)
        self.assertEqual(st, 200)
        self.assertTrue(p["needs_confirm"])
        self.assertTrue(p["submit"][0].startswith("pjsub"))
        self.assertEqual(self.pjsub_log(), [])                       # a preview submits nothing
        self.assertEqual(C.all_jobs(self.cfg), [])
        # wrong code, and a valid code with a changed body, are refused
        self.assertEqual(self.req("POST", "/api/v1/jobs", dict(body, confirm="000000000000"))[0], 409)
        self.assertEqual(self.req("POST", "/api/v1/jobs", dict(body, nodes=3, confirm=p["confirm"]))[0], 409)
        self.assertEqual(self.pjsub_log(), [])
        st, p = self.req("POST", "/api/v1/jobs", body)
        st, r = self.req("POST", "/api/v1/jobs", dict(body, confirm=p["confirm"]))
        self.assertEqual(st, 200, r)
        self.assertEqual(len(self.pjsub_log()), 1)
        self.assertIn("job.launched", self.types("L1"))
        # a used code does not work twice
        self.assertEqual(self.req("POST", "/api/v1/jobs", dict(body, confirm=p["confirm"]))[0], 409)
        self.assertEqual(len(self.pjsub_log()), 1)

    def test_preview_reports_errors_and_script(self):
        self.launch("dup")
        st, r = self.req("POST", "/api/v1/jobs", {"name": "dup"})
        self.assertEqual(st, 400)
        self.assertIn("job exists", r["error"])
        st, p = self.req("POST", "/api/v1/jobs", {"name": "S1", "script": "#!/bin/bash\necho hi\n"})
        st, r = self.req("POST", "/api/v1/jobs", {"name": "S1", "script": "#!/bin/bash\necho hi\n", "confirm": p["confirm"]})
        self.assertEqual(st, 200, r)
        self.assertIn("echo hi", open(os.path.join(self.cfg["rundir"], "S1", "work.sh")).read())

    def test_kill_two_step(self):
        self.launch("k")
        st, p = self.req("POST", "/api/v1/jobs/k/kill", {})
        self.assertTrue(p["needs_confirm"])
        self.assertEqual(p["would_cancel"], C.load_job(self.cfg, "k")["jobids"])
        self.assertFalse(os.path.exists(os.path.join(self.stub, "pjdel.log")))
        self.assertEqual(self.req("POST", "/api/v1/jobs/k/kill", {"confirm": "nope"})[0], 409)
        st, r = self.req("POST", "/api/v1/jobs/k/kill", {"confirm": p["confirm"]})
        self.assertEqual(st, 200, r)
        self.assertIn("1001", open(os.path.join(self.stub, "pjdel.log")).read())
        self.assertIn("job.stopped", self.types("k"))

    def test_confirm_can_be_switched_off(self):
        self.srv.shutdown()
        self.start(confirm=False)
        st, r = self.req("POST", "/api/v1/jobs", {"name": "direct"})
        self.assertEqual(st, 200, r)
        self.assertEqual(len(self.pjsub_log()), 1)


class Events(WebBase):
    def test_state_events_and_chain_alert(self):
        self.launch("e")
        self.info("e")
        self.set_queue("JOB_ID JOB_NAME MD ST USER GROUP\n1001 e NM RUN u g\n")
        self.svc.refresh_once()
        self.assertEqual(C.load_job(self.cfg, "e")["state"], "running")
        self.assertIn("job.started", self.types("e"))
        self.assertEqual(self.types().count("service.chain_down"), 1)
        self.svc.refresh_once()
        self.assertEqual(self.types().count("service.chain_down"), 1)         # told once
        # a successor was queued
        run = os.path.join(self.cfg["rundir"], "e")
        open(run + "/chain/seq", "w").write("2\n")
        open(run + "/chain/jobs.log", "w").write("1 1001 1 S 0\n2 1002 2 S 0\n")
        self.svc.refresh_once()
        self.assertIn("job.successor", self.types("e"))
        # the workload finished
        self.info("e", phase="done")
        self.set_queue("")
        self.svc.refresh_once()
        self.assertIn("job.finished", self.types("e"))
        st, d = self.req("GET", "/api/v1/events?since=0")
        self.assertEqual([e["id"] for e in d["events"]], sorted(e["id"] for e in d["events"]))

    def test_lost_and_stale(self):
        self.launch("l")
        job = C.load_job(self.cfg, "l")
        job["created"] = time.time() - 1000
        C.save_job(self.cfg, job)
        self.info("l", hb=time.time() - 500)
        self.set_queue("JOB_ID JOB_NAME MD ST USER GROUP\n1001 l NM RUN u g\n")
        self.svc.refresh_once()
        self.assertIn("job.stale", self.types("l"))
        self.set_queue("")
        self.info("l", hb=time.time() - 500, phase="running")
        self.svc.refresh_once()
        self.assertIn("job.lost", self.types("l"))

    def test_long_poll(self):
        last = self.svc.last_id()
        threading.Timer(0.4, lambda: self.svc.emit("job.flag", None, {"flag": "X", "on": True})).start()
        t = time.time()
        st, d = self.req("GET", "/api/v1/events?since=%d&wait=10" % last)
        self.assertLess(time.time() - t, 5)
        self.assertEqual([e["type"] for e in d["events"]], ["job.flag"])
        t = time.time()
        st, d = self.req("GET", "/api/v1/events?since=%d&wait=1" % d["last_id"])
        self.assertEqual(d["events"], [])
        self.assertGreaterEqual(time.time() - t, 0.9)
        self.assertEqual(self.req("GET", "/api/v1/events?since=x")[0], 400)

    def test_stream(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("GET", "/api/v1/stream", headers={"Authorization": "Bearer " + TOKEN})
        r = c.getresponse()
        self.assertEqual(r.status, 200)
        self.assertEqual(r.fp.readline(), b": open\n")
        r.fp.readline()
        threading.Timer(0.3, lambda: self.svc.emit("job.flag", None, {"flag": "S", "on": False})).start()
        seen = ""
        while "data:" not in seen:
            seen += r.fp.readline().decode()
        self.assertIn('"job.flag"', seen)
        c.close()

    def test_persisted_ids_continue(self):
        self.svc.emit("job.flag", None, {})
        n = self.svc.last_id()
        again = W.Service(self.cfg, TOKEN, hooks=W.Webhooks([]))
        self.assertEqual(again.last_id(), n)


class Hooks(WebBase):
    def test_delivery_signature_retry_and_filter(self):
        rcv, txt = Receiver(fail=1), Receiver()
        self.addCleanup(rcv.stop)
        self.addCleanup(txt.stop)
        hooks = W.Webhooks([{"name": "main", "url": rcv.url, "secret": "s3cret", "events": ["job.started"]},
                            {"name": "chat", "url": txt.url, "format": "text", "events": ["job.*"]}], delays=(0, 0.1, 0.1))
        self.srv.shutdown()
        self.start(hooks=hooks)
        self.launch("h")
        self.info("h")
        self.set_queue("JOB_ID JOB_NAME MD ST USER GROUP\n1001 h NM RUN u g\n")
        self.svc.refresh_once()
        hooks.wait()
        # filtered: only job.started reaches the first hook; it failed once, then arrived
        self.assertEqual(len(rcv.got), 2)
        body, headers = rcv.got[-1]
        want = "sha256=" + hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()
        self.assertEqual(headers["X-Jobctl-Signature"], want)
        payload = json.loads(body.decode())
        self.assertEqual((payload["type"], payload["job"], payload["state"], payload["nodes"]), ("job.started", "h", "running", 1))
        self.assertNotIn(self.tmp.name, body.decode())                  # no private paths
        # the chat hook got text for both job.state and job.started
        texts = [json.loads(b.decode())["text"] for b, _ in txt.got]
        self.assertTrue(any("submitted -> running" in t for t in texts), texts)
        log = [d for d in hooks.log if d["hook"] == "main"]
        self.assertEqual((log[0]["ok"], log[0]["tries"]), (True, 2))
        st, d = self.req("GET", "/api/v1/webhooks")
        self.assertNotIn("url", json.dumps(d))
        self.assertNotIn("s3cret", json.dumps(d))
        self.assertEqual(self.req("POST", "/api/v1/webhooks/test", {})[0], 200)

    def test_test_endpoint_needs_a_hook(self):
        self.assertEqual(self.req("POST", "/api/v1/webhooks/test", {})[0], 400)


class Ports(WebBase):
    def test_ports(self):
        self.launch("p1")
        self.launch("p2")
        st, d = self.req("GET", "/api/v1/ports")
        self.assertEqual(len(d["ports"]), 2)
        self.assertTrue(all(p["conflict"] is None for p in d["ports"]))
        st, d = self.req("GET", "/api/v1/services")
        self.assertEqual(d["live_jobs"], 2)


class Mcp(WebBase):
    def setUp(self):
        WebBase.setUp(self)
        import mcp_server as M
        self.M = M
        os.environ["JOBCTL_WEBUI_URL"] = "http://127.0.0.1:%d" % self.port
        os.environ["JOBCTL_WEBUI_TOKEN"] = TOKEN

    def tool(self, tool_name, **args):
        r = self.M.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool_name, "arguments": args}})["result"]
        return r["isError"], json.loads(r["content"][0]["text"])

    def test_handshake_and_tools(self):
        init = self.M.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}})
        self.assertEqual(init["result"]["serverInfo"]["name"], "jobctl")
        self.assertIsNone(self.M.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        names = [t["name"] for t in self.M.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]]
        self.assertIn("preview_launch", names)
        self.assertNotIn("_call", json.dumps(self.M.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})))
        self.assertEqual(self.M.handle({"jsonrpc": "2.0", "id": 3, "method": "nope"})["error"]["code"], -32601)

    def test_launch_through_mcp_needs_the_code(self):
        err, p = self.tool("preview_launch", name="m1", nodes=2, time="00:10:00")
        self.assertFalse(err)
        self.assertTrue(p["needs_confirm"])
        self.assertEqual(self.pjsub_log(), [])
        err, r = self.tool("confirm_launch", name="m1", nodes=2, time="00:10:00", confirm="bad")
        self.assertTrue(err)
        err, r = self.tool("confirm_launch", name="m1", nodes=2, time="00:10:00", confirm=p["confirm"])
        self.assertFalse(err, r)
        self.assertEqual(len(self.pjsub_log()), 1)
        err, d = self.tool("list_jobs")
        self.assertEqual([j["name"] for j in d["jobs"]], ["m1"])
        err, d = self.tool("stop_job", name="m1")
        self.assertFalse(err)
        err, d = self.tool("job_events", since=0)
        self.assertIn("job.launched", [e["type"] for e in d["events"]])
        err, d = self.tool("get_job")
        self.assertTrue(err)

    def test_board_down_is_an_error_not_a_crash(self):
        os.environ["JOBCTL_WEBUI_URL"] = "http://127.0.0.1:1"
        err, d = self.tool("list_jobs")
        self.assertTrue(err)
        self.assertIn("cannot reach", d["error"])


if __name__ == "__main__":
    unittest.main()
