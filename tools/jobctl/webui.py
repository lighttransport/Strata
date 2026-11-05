#!/usr/bin/env python3
"""A local web board, JSON API and event feed for jobctl. Standard library only; run it on the laptop.

    python3 webui.py [--port 8765] [--open]          (or:  jobctl webui)

It imports jobctl as a library, so a launch, stop or run here is the same code as on the command line.
Open the printed URL for the board. Agents use /api/v1 (see /api/v1/spec) or mcp_server.py.

Safety: the board binds 127.0.0.1, every /api/v1 call except /spec needs the bearer token, the Host and Origin
headers must be local, and launch and kill are two steps (first a preview with the exact submit line and a one-shot
code, then the same request again with the code).
"""
import argparse
import collections
import contextlib
import fnmatch
import hashlib
import hmac
import io
import json
import os
import queue
import re
import secrets
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jobctl as C  # noqa: E402

MAX_BODY = 2 * 1024 * 1024
MAX_OUTPUT = 1000000
CONFIRM_TTL = 600
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
LOOPBACK = ("127.0.0.1", "localhost", "::1")
INFO_KEYS = ("phase", "hb", "jobid", "seq", "nodes", "frontend_port", "server_port", "variant")


class ApiError(Exception):
    def __init__(self, status, message, **extra):
        Exception.__init__(self, message)
        self.status, self.message, self.extra = status, message, extra


# ---------------------------------------------------------------- webhooks
def event_text(ev):
    d = ev.get("data") or {}
    t = ev["type"]
    if t == "job.state":
        what = "%s -> %s" % (d.get("from"), d.get("to"))
    elif t == "job.successor":
        what = "next allocation queued (chain seq %s)" % d.get("seq")
    elif t == "job.flag":
        what = "%s %s" % (d.get("flag"), "set" if d.get("on") else "cleared")
    elif t == "job.stale":
        what = "no heartbeat for %ss" % d.get("age")
    else:
        what = t
    return "jobctl %s: %s" % (ev.get("job") or "service", what if ev.get("job") else (d.get("message") or t))


class Webhooks(object):
    """Posts events to configured URLs on a worker thread: 3 tries, HMAC signature when a secret is set."""

    def __init__(self, hooks, delays=(0, 2, 6), timeout=10):
        self.hooks = [dict(h, name=h.get("name") or "hook%d" % i) for i, h in enumerate(hooks or [])]
        self.delays, self.timeout = delays, timeout
        self.log = collections.deque(maxlen=200)
        self.q = queue.Queue()
        self.t = None
        if self.hooks:
            self.t = threading.Thread(target=self._worker, daemon=True)
            self.t.start()

    def describe(self):
        # URLs and secrets can carry credentials: never shown
        return [{"name": h["name"], "events": h.get("events", ["*"]), "format": h.get("format", "json"),
                 "signed": bool(h.get("secret"))} for h in self.hooks]

    def matches(self, hook, etype):
        return any(fnmatch.fnmatchcase(etype, pat) for pat in hook.get("events", ["*"]))

    def payload(self, hook, ev, job):
        if hook.get("format") == "text":
            return json.dumps({"text": event_text(ev)}).encode()
        body = {"id": ev["id"], "ts": ev["ts"], "type": ev["type"], "job": ev.get("job"), "data": ev.get("data") or {}}
        if job:
            body.update({"state": job.get("state"), "nodes": job.get("nodes"), "seq": job.get("seq"), "jobids": job.get("jobids")})
        return json.dumps(body, sort_keys=True).encode()

    def dispatch(self, ev, job=None):
        for h in self.hooks:
            if self.matches(h, ev["type"]):
                self.q.put((h, ev, self.payload(h, ev, job)))

    def post(self, hook, data):
        req = urllib.request.Request(hook["url"], data=data, headers={"Content-Type": "application/json", "User-Agent": "jobctl-webui"})
        if hook.get("secret"):
            req.add_header("X-Jobctl-Signature", "sha256=" + hmac.new(hook["secret"].encode(), data, hashlib.sha256).hexdigest())
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.status

    def _worker(self):
        while True:
            hook, ev, data = self.q.get()
            status, err, tries = None, "", 0
            for delay in self.delays:
                if delay:
                    time.sleep(delay)
                tries += 1
                try:
                    status = self.post(hook, data)
                    err = ""
                    break
                except urllib.error.HTTPError as e:
                    status, err = e.code, "HTTP %d" % e.code
                except Exception as e:
                    status, err = None, type(e).__name__
            self.log.append({"ts": time.time(), "hook": hook["name"], "event": ev["id"], "type": ev["type"],
                             "ok": status is not None and 200 <= status < 300, "status": status, "tries": tries, "error": err})
            self.q.task_done()

    def wait(self):
        self.q.join()


# ---------------------------------------------------------------- the service
class Service(object):
    def __init__(self, cfg, token, hooks=None):
        self.cfg = cfg
        self.token = token
        ui = cfg.get("webui", {})
        self.require_confirm = ui.get("require_confirm", True)
        self.interval = max(1, int(cfg.get("interval", 30)))
        self.lock = threading.RLock()           # jobctl calls print to stdout, which is process-wide: one at a time
        self.cond = threading.Condition()
        self.events = collections.deque(maxlen=1000)
        self.next_id = 1
        self.pending = {}
        self.report = {}
        self.tunnel = {}
        self.ssh_ok = None
        self.ssh_error = ""
        self.last_refresh = 0.0
        self.chain_up = None
        self.chain_alerted = False
        self.stale = set()
        self.plock = threading.RLock()
        self.wake = threading.Event()
        self.stopping = False
        os.makedirs(cfg["registry"], exist_ok=True)
        self.log_path = os.path.join(cfg["registry"], "events.jsonl")
        self.hooks = hooks if hooks is not None else Webhooks(ui.get("webhooks"))
        self._load_events()
        self.prev = dict((j["name"], self.snap(j)) for j in C.all_jobs(cfg))

    # ---- events
    def _load_events(self):
        try:
            with open(self.log_path) as f:
                for line in f.readlines()[-1000:]:
                    try:
                        self.events.append(json.loads(line))
                    except ValueError:
                        pass
        except OSError:
            pass
        if self.events:
            self.next_id = self.events[-1]["id"] + 1

    def emit(self, etype, job=None, data=None):
        with self.cond:
            ev = {"id": self.next_id, "ts": round(time.time(), 3), "type": etype, "job": job, "data": data or {}}
            self.next_id += 1
            self.events.append(ev)
            self.cond.notify_all()
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(ev, sort_keys=True) + "\n")
        except OSError:
            pass
        view = None
        if job:
            try:
                view = self.view(C.load_job(self.cfg, job), time.time())
            except SystemExit:
                view = None
        self.hooks.dispatch(ev, view)
        return ev

    def events_since(self, since, wait=0.0, limit=200):
        end = time.time() + wait
        with self.cond:
            while True:
                out = [e for e in self.events if e["id"] > since]
                if out or wait <= 0 or time.time() >= end:
                    return out[-limit:] if since == 0 else out[:limit]
                self.cond.wait(min(1.0, max(0.05, end - time.time())))

    def last_id(self):
        with self.cond:
            return self.next_id - 1

    # ---- jobctl calls
    def call(self, fn, **kw):
        buf = io.StringIO()
        with self.lock:
            try:
                with contextlib.redirect_stdout(buf):
                    rc = fn(self.cfg, argparse.Namespace(**kw))
                rc = rc or 0
                return {"ok": rc == 0, "rc": rc, "output": buf.getvalue()[:MAX_OUTPUT]}
            except SystemExit as e:
                return {"ok": False, "rc": 1, "error": str(e.code) if isinstance(e.code, str) else "exit %s" % e.code,
                        "output": buf.getvalue()[:MAX_OUTPUT]}
            except Exception as e:
                return {"ok": False, "rc": 1, "error": "%s: %s" % (type(e).__name__, e), "output": buf.getvalue()[:MAX_OUTPUT]}

    @staticmethod
    def snap(job):
        return {"state": job.get("state"), "seq": job.get("seq", 0)}

    def view(self, job, now):
        hb = job.get("hb") or 0
        return {"name": job["name"], "state": job["state"], "nodes": job["nodes"], "elapse": job["elapse"],
                "jobids": job["jobids"], "latest": job["jobids"][-1] if job["jobids"] else None, "seq": job.get("seq", 0),
                "hb_age": round(now - hb) if hb else None, "tunnel_ok": self.tunnel.get(job["name"]),
                "created": job.get("created"), "variants": job.get("variants")}

    def mark(self, name):
        try:
            snap = self.snap(C.load_job(self.cfg, name))
        except SystemExit:
            return
        with self.plock:
            self.prev[name] = snap

    # ---- refresh and event diff
    def refresh_once(self):
        cfg = self.cfg
        self._diff(time.time())         # changes made by the command line since the last look
        live = [j for j in C.all_jobs(cfg) if j["state"] in C.LIVE_STATES]
        with self.lock:
            try:
                remote = C.Remote(cfg)
                if live:
                    self.report = C.refresh(cfg, remote, live) or {}
                self.ssh_ok, self.ssh_error = True, ""
                self.chain_up = self._chain_up(remote) if live and cfg.get("chain", True) else self.chain_up
            except (Exception, SystemExit) as e:
                self.ssh_ok, self.ssh_error = False, str(e.code if isinstance(e, SystemExit) else e)[:200]
        now = time.time()
        self.last_refresh = now
        for j in C.all_jobs(cfg):
            if j.get("tunnel", True) and j["state"] in ("running", "draining", "paused"):
                self.tunnel[j["name"]] = C.healthy(j["local_port"])
            else:
                self.tunnel.pop(j["name"], None)
        self._diff(now)

    def _chain_up(self, remote):
        pid = "%s/_chain.pid" % self.cfg["rundir"]
        rc, out, _ = remote.run("test -f %s && kill -0 $(cat %s) 2>/dev/null && echo up" % (pid, pid))
        return "up" in out

    def _diff(self, now):
        with self.plock:
            self._diff_locked(now)

    def _diff_locked(self, now):
        cur = {}
        for j in C.all_jobs(self.cfg):
            cur[j["name"]] = j
            old = self.prev.get(j["name"])
            new = self.snap(j)
            if old is None:
                self.emit("job.launched", j["name"], {"by": "command line", "nodes": j["nodes"], "jobids": j["jobids"]})
            else:
                if old["state"] != new["state"]:
                    self.emit("job.state", j["name"], {"from": old["state"], "to": new["state"]})
                    specific = {"running": "job.started", "done": "job.finished", "lost": "job.lost", "stopped": "job.stopped"}
                    if new["state"] in specific and not (new["state"] == "running" and old["state"] in ("draining", "paused")):
                        self.emit(specific[new["state"]], j["name"], {})
                if new["seq"] > old["seq"]:
                    self.emit("job.successor", j["name"], {"seq": new["seq"]})
            self.prev[j["name"]] = new
            hb = j.get("hb") or 0
            if j["state"] == "running" and hb and now - hb > 3 * self.interval:
                if j["name"] not in self.stale:
                    self.stale.add(j["name"])
                    self.emit("job.stale", j["name"], {"age": round(now - hb)})
            else:
                self.stale.discard(j["name"])
        live = any(j["state"] in C.LIVE_STATES for j in cur.values())
        if live and self.chain_up is False and self.cfg.get("chain", True):
            if not self.chain_alerted:
                self.chain_alerted = True
                self.emit("service.chain_down", None, {"message": "jobs are live but no chain loop runs on the login node"})
        elif self.chain_up:
            self.chain_alerted = False

    def refresher(self):
        while not self.stopping:
            try:
                self.refresh_once()
            except Exception as e:      # keep the board alive through anything odd
                sys.stderr.write("refresh: %s\n" % e)
            self.wake.wait(self.interval)
            self.wake.clear()

    # ---- confirm codes
    def _key(self, action, body):
        return json.dumps([action, body], sort_keys=True)

    def issue(self, key):
        now = time.time()
        for c in [c for c, (_, exp) in self.pending.items() if exp < now]:
            del self.pending[c]
        code = secrets.token_hex(6)
        self.pending[code] = (key, now + CONFIRM_TTL)
        return code

    def redeem(self, code, key):
        entry = self.pending.pop(code or "", None)
        return bool(entry) and entry[0] == key and entry[1] >= time.time()

    # ---- API handlers: (status, object)
    def api_spec(self, **_):
        return 200, {"api": "jobctl webui", "version": 1, "auth": "Authorization: Bearer <token> (token file, see README)",
                     "routes": [{"method": m, "path": path, "auth": a, "description": d, "params": prm} for m, _, _, a, d, prm, path in ROUTES]}

    def api_jobs(self, **_):
        now = time.time()
        return 200, {"jobs": [self.view(j, now) for j in C.all_jobs(self.cfg)], "ssh_ok": self.ssh_ok,
                     "refresh_age": round(now - self.last_refresh) if self.last_refresh else None}

    def api_job(self, name, **_):
        try:
            job = C.load_job(self.cfg, name)
        except SystemExit:
            raise ApiError(404, "no such job: %s" % name)
        rep = self.report.get(name, {})
        info = dict((k, v) for k, v in (rep.get("info") or {}).items() if k in INFO_KEYS)
        v = self.view(job, time.time())
        v.update({"flags": rep.get("flags", []), "chain_jobs": rep.get("jobs", []), "winner": rep.get("winner"),
                  "next_req": rep.get("next_req", False), "info": info})
        return 200, v

    def api_flag(self, name, flag, on, **_):
        try:
            C.load_job(self.cfg, name)
        except SystemExit:
            raise ApiError(404, "no such job: %s" % name)
        r = self.call(lambda cfg, a: C.cmd_flag(cfg, a, flag, on), name=name)
        if r["ok"]:
            self.emit("job.flag", name, {"flag": flag, "on": on})
            self.wake.set()
        return (200 if r["ok"] else 400), r

    def api_resubmit(self, name, **_):
        try:
            C.load_job(self.cfg, name)
        except SystemExit:
            raise ApiError(404, "no such job: %s" % name)
        r = self.call(C.cmd_resubmit, name=name)
        if r["ok"]:
            self.emit("job.resubmit", name, {})
            self.wake.set()
        return (200 if r["ok"] else 400), r

    def api_run(self, name, body, **_):
        cmd = body.get("command")
        if not isinstance(cmd, str) or not cmd.strip():
            raise ApiError(400, "command (string) is required")
        timeout = self._int(body, "timeout", 50, 1, 600)
        try:
            C.load_job(self.cfg, name)
        except SystemExit:
            raise ApiError(404, "no such job: %s" % name)
        r = self.call(C.cmd_run, name=name, command=cmd, timeout=timeout, inbox=bool(body.get("inbox")))
        return 200, r        # rc is the command's exit code (124 = timed out); the call itself worked

    @staticmethod
    def _int(body, key, default, lo, hi):
        v = body.get(key, default)
        if isinstance(v, bool) or not isinstance(v, (int, str)):
            raise ApiError(400, "%s must be an integer" % key)
        try:
            v = int(v)
        except ValueError:
            raise ApiError(400, "%s must be an integer" % key)
        if not lo <= v <= hi:
            raise ApiError(400, "%s must be %d..%d" % (key, lo, hi))
        return v

    def api_launch(self, body, **_):
        body = dict(body)
        code = body.pop("confirm", None)
        name = body.get("name")
        if not isinstance(name, str) or not C.NAME_RE.match(name):
            raise ApiError(400, "name: letters, digits, . _ - ; max 40")
        nodes = self._int(body, "nodes", 1, 1, self.cfg["max_nodes"])
        tm = body.get("time", "01:00:00")
        if not isinstance(tm, str) or not C.TIME_RE.match(tm):
            raise ApiError(400, "time must be HH:MM:SS")
        variants = body.get("variants") or None
        if variants is not None and not isinstance(variants, str):
            raise ApiError(400, "variants must be a string like '24@72:00:00,12@12:00:00'")
        env = body.get("env") or {}
        if not isinstance(env, dict) or not all(isinstance(k, str) and ENV_NAME.match(k) and isinstance(v, (str, int))
                                                and "\n" not in str(v) for k, v in env.items()):
            raise ApiError(400, "env must be an object of NAME: value")
        script = body.get("script")
        if script is not None and (not isinstance(script, str) or len(script) > MAX_BODY):
            raise ApiError(400, "script must be text")
        key = self._key("launch", body)

        def args(dry):
            path = None
            if script:
                fd, path = tempfile.mkstemp(prefix="jobctl-work-", suffix=".sh")
                with os.fdopen(fd, "w") as f:
                    f.write(script)
            ns = dict(name=name, nodes=nodes, time=tm, variants=variants, script=path, dry_run=dry,
                      lead_min=None, queue_ahead=None, env=["%s=%s" % kv for kv in env.items()], sched_arg=None)
            return ns, path

        if self.require_confirm and not code:
            ns, path = args(True)
            try:
                r = self.call(C.cmd_launch, **ns)
            finally:
                if path:
                    os.unlink(path)
            if not r["ok"]:
                return 400, r
            return 200, {"ok": True, "needs_confirm": True, "submit": r["output"].strip().splitlines(),
                         "confirm": self.issue(key), "expires_in": CONFIRM_TTL}
        if self.require_confirm and not self.redeem(code, key):
            raise ApiError(409, "confirm code is wrong, used, expired, or the request changed; preview again")
        ns, path = args(False)
        try:
            r = self.call(C.cmd_launch, **ns)
        finally:
            if path:
                os.unlink(path)
        if r["ok"]:
            try:
                job = C.load_job(self.cfg, name)
                self.emit("job.launched", name, {"nodes": job["nodes"], "elapse": job["elapse"], "jobids": job["jobids"]})
                self.mark(name)
            except SystemExit:
                pass
            self.wake.set()
        return (200 if r["ok"] else 400), r

    def api_kill(self, name, body, **_):
        body = dict(body)
        code = body.pop("confirm", None)
        try:
            job = C.load_job(self.cfg, name)
        except SystemExit:
            raise ApiError(404, "no such job: %s" % name)
        key = self._key("kill:" + name, body)
        if self.require_confirm and not code:
            return 200, {"ok": True, "needs_confirm": True, "would_cancel": job["jobids"], "state": job["state"],
                         "confirm": self.issue(key), "expires_in": CONFIRM_TTL}
        if self.require_confirm and not self.redeem(code, key):
            raise ApiError(409, "confirm code is wrong, used, expired, or the request changed; preview again")
        r = self.call(C.cmd_kill, name=name)
        if r["ok"]:
            self.emit("job.stopped", name, {"by": "kill"})
            self.mark(name)
            self.wake.set()
        return (200 if r["ok"] else 400), r

    def api_events(self, query, **_):
        since = self._qint(query, "since", 0, 0, 10 ** 12)
        wait = self._qint(query, "wait", 0, 0, 55)
        evs = self.events_since(since, float(wait))
        return 200, {"events": evs, "last_id": self.last_id()}

    @staticmethod
    def _qint(query, key, default, lo, hi):
        try:
            v = int(query.get(key, [default])[0])
        except ValueError:
            raise ApiError(400, "%s must be an integer" % key)
        if not lo <= v <= hi:
            raise ApiError(400, "%s must be %d..%d" % (key, lo, hi))
        return v

    def api_ports(self, **_):
        rows, seen = [], {}
        for kind, items in (("jobctl", C.all_jobs(self.cfg)), ("slot", C.external_slots(self.cfg))):
            for j in items:
                row = {"name": j["name"], "kind": kind, "local": j.get("local_port"), "frontend": j.get("frontend_port"),
                       "state": j.get("state"), "conflict": None}
                for side in ("local", "frontend"):
                    p = row[side]
                    if p:
                        other = seen.get((side, p))
                        if other:
                            row["conflict"] = "%s port %d also used by %s" % (side, p, other)
                        seen[(side, p)] = other or j["name"]
                rows.append(row)
        return 200, {"ports": rows, "ranges": {"local": self.cfg["local_ports"], "frontend": self.cfg["frontend_ports"]}}

    def api_services(self, **_):
        live = len([j for j in C.all_jobs(self.cfg) if j["state"] in C.LIVE_STATES])
        return 200, {"ssh_ok": self.ssh_ok, "ssh_error": self.ssh_error, "chain_loop": {None: "unknown", True: "up", False: "down"}[self.chain_up],
                     "live_jobs": live, "refresh_age": round(time.time() - self.last_refresh) if self.last_refresh else None,
                     "interval": self.interval, "require_confirm": self.require_confirm}

    def api_chain_start(self, **_):
        r = self.call(C.cmd_chain_start)
        if r["ok"]:
            self.chain_up = None
            self.wake.set()
        return (200 if r["ok"] else 400), r

    def api_refresh(self, **_):
        self.refresh_once()
        return self.api_jobs()

    def api_webhooks(self, **_):
        return 200, {"hooks": self.hooks.describe(), "deliveries": list(self.hooks.log)[-50:]}

    def api_webhook_test(self, **_):
        if not self.hooks.hooks:
            raise ApiError(400, "no webhooks configured (config: webui.webhooks)")
        ev = self.emit("webhook.test", None, {"message": "test event from the jobctl board"})
        return 200, {"ok": True, "event": ev["id"]}

    def dispatch(self, method, path, query, body):
        for m, rx, fn, _auth, _d, _p, _path in ROUTES:
            mt = rx.match(path) if m == method else None
            if mt:
                return getattr(self, fn)(query=query, body=body, **mt.groupdict())
        raise ApiError(404, "no such route: %s %s" % (method, path))


def _route(method, path, fn, auth, desc, params=None):
    rx = re.compile("^" + re.sub(r"<(\w+)>", r"(?P<\1>[A-Za-z0-9._-]{1,40})", path) + "$")
    return (method, rx, fn, auth, desc, params or {}, path)


P = "/api/v1"
ROUTES = [
    _route("GET", P + "/spec", "api_spec", False, "This list."),
    _route("GET", P + "/jobs", "api_jobs", True, "All jobs: state, nodes, elapse, heartbeat age, chain seq, job ids, tunnel_ok."),
    _route("POST", P + "/jobs", "api_launch", True,
           "Launch. First call returns the exact submit lines and a one-shot 'confirm' code; repeat the same body with 'confirm' to submit.",
           {"name": "str", "nodes": "int", "time": "HH:MM:SS", "variants": "'NODES@HH:MM:SS,...' (optional)",
            "script": "workload script text (optional)", "env": "{NAME: value} (optional)", "confirm": "code from the preview"}),
    _route("GET", P + "/jobs/<name>", "api_job", True, "One job, with flags, chain job ids and the latest heartbeat info."),
    _route("POST", P + "/jobs/<name>/stop", "api_stop", True, "Set STOP (the job ends after its current step)."),
    _route("POST", P + "/jobs/<name>/pause", "api_pause", True, "Set PAUSE."),
    _route("POST", P + "/jobs/<name>/resume", "api_resume", True, "Clear PAUSE."),
    _route("POST", P + "/jobs/<name>/drain", "api_drain", True, "Set DRAIN (finish the current command, then hand over)."),
    _route("POST", P + "/jobs/<name>/resubmit", "api_resubmit", True, "Clear STOP and queue the next allocation."),
    _route("POST", P + "/jobs/<name>/kill", "api_kill", True,
           "Cancel every allocation of the chain. First call returns the job ids and a 'confirm' code; repeat with it.",
           {"confirm": "code from the preview"}),
    _route("POST", P + "/jobs/<name>/run", "api_run", True, "Run a shell command on node 0. Refuses 'exit' and 'set -e'.",
           {"command": "str", "timeout": "seconds 1..600 (default 50)", "inbox": "bool: use the shared-file inbox"}),
    _route("GET", P + "/events", "api_events", True, "Events with id > since. wait=N (max 55) holds the call until one arrives.",
           {"since": "int", "wait": "seconds"}),
    _route("GET", P + "/stream", "api_stream", True, "Server-sent events (used by the page)."),
    _route("GET", P + "/ports", "api_ports", True, "Ports of jobs and interactive slots, with conflicts."),
    _route("GET", P + "/services", "api_services", True, "ssh status, chain-loop status, refresh age."),
    _route("POST", P + "/services/chain-start", "api_chain_start", True, "Start the chain loop on the login node."),
    _route("POST", P + "/refresh", "api_refresh", True, "Refresh from the cluster now."),
    _route("GET", P + "/webhooks", "api_webhooks", True, "Configured hooks (no URLs or secrets) and recent deliveries."),
    _route("POST", P + "/webhooks/test", "api_webhook_test", True, "Send a test event to every hook."),
]


def _flag_handler(flag, on):
    def h(self, name, **_):
        return self.api_flag(name, flag, on)
    return h


for _fn, _flag, _on in (("api_stop", "STOP", True), ("api_pause", "PAUSE", True), ("api_resume", "PAUSE", False), ("api_drain", "DRAIN", True)):
    setattr(Service, _fn, _flag_handler(_flag, _on))


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "jobctl-webui"

    def setup(self):
        BaseHTTPRequestHandler.setup(self)
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, fmt, *args):
        if getattr(self.server, "verbose", False):
            sys.stderr.write("%s\n" % (fmt % args))

    def _send(self, status, obj, ctype="application/json", extra=None):
        data = obj if isinstance(obj, bytes) else json.dumps(obj, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _local_request(self):
        host = self.headers.get("Host", "")
        name = host[1:host.index("]")] if host.startswith("[") and "]" in host else host.rsplit(":", 1)[0] if host.count(":") == 1 else host
        allowed = set(LOOPBACK) | set(self.server.allow_hosts)
        if name not in allowed:
            return False
        origin = self.headers.get("Origin")
        if origin and origin != "null" and urllib.parse.urlparse(origin).netloc != host:
            return False
        if origin == "null":
            return False
        return True

    def _authorized(self):
        got = self.headers.get("Authorization", "")
        return got.startswith("Bearer ") and hmac.compare_digest(got[7:].strip().encode(), self.server.svc.token.encode())

    def _handle(self, method):
        svc = self.server.svc
        u = urllib.parse.urlparse(self.path)
        try:
            if not self._local_request():
                raise ApiError(403, "Host or Origin is not local")
            if method == "GET" and u.path in ("/", "/index.html"):
                return self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8",
                                  {"Content-Security-Policy": "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'"})
            if not u.path.startswith(P + "/"):
                raise ApiError(404, "not found")
            route = None
            for m, rx, fn, auth, _d, _p, _path in ROUTES:
                if m == method and rx.match(u.path):
                    route = (fn, auth)
            if route is None:
                raise ApiError(404, "no such route: %s %s" % (method, u.path))
            if route[1] and not self._authorized():
                raise ApiError(401, "missing or wrong bearer token")
            body = {}
            if method == "POST":
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    raise ApiError(413, "body too large")
                raw = self.rfile.read(n) if n else b""
                if raw:
                    try:
                        body = json.loads(raw.decode())
                    except ValueError:
                        raise ApiError(400, "body must be JSON")
                    if not isinstance(body, dict):
                        raise ApiError(400, "body must be a JSON object")
            if route[0] == "api_stream":
                return self._stream(urllib.parse.parse_qs(u.query))
            status, obj = svc.dispatch(method, u.path, urllib.parse.parse_qs(u.query), body)
            self._send(status, obj)
        except ApiError as e:
            self._send(e.status, dict({"ok": False, "error": e.message}, **e.extra))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            self._send(500, {"ok": False, "error": "%s: %s" % (type(e).__name__, e)})

    def _stream(self, query):
        svc = self.server.svc
        since = svc.last_id()
        try:
            since = int(query.get("since", [since])[0])
        except ValueError:
            pass
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b": open\n\n")
        self.wfile.flush()
        try:
            while not svc.stopping:
                evs = svc.events_since(since, 15.0)
                if not evs:
                    self.wfile.write(b": keep-alive\n\n")
                else:
                    for e in evs:
                        self.wfile.write(("id: %d\ndata: %s\n\n" % (e["id"], json.dumps(e, sort_keys=True))).encode())
                        since = e["id"]
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")


def make_server(svc, host="127.0.0.1", port=8765, allow_hosts=()):
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    srv.svc = svc
    srv.allow_hosts = list(allow_hosts)
    srv.verbose = False
    return srv


# ---------------------------------------------------------------- token and entry point
def load_token(cfg):
    path = os.path.expanduser(cfg.get("webui", {}).get("token_file") or "~/.config/jobctl/webui_token")
    try:
        with open(path) as f:
            tok = f.read().strip()
        if tok:
            return tok, path
    except OSError:
        pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tok = secrets.token_urlsafe(24)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(tok + "\n")
    return tok, path


def main(argv=None, config=None):
    p = argparse.ArgumentParser(prog="jobctl webui", description="Local web board and API for jobctl.")
    p.add_argument("--config", default=config)
    p.add_argument("--host", help="bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, help="default 8765")
    p.add_argument("--allow-host", action="append", default=[], help="extra Host header names accepted (needed when --host is not loopback)")
    p.add_argument("--open", action="store_true", help="open the board in a browser")
    p.add_argument("--no-refresh", action="store_true", help="do not poll the cluster (read the registry only)")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    cfg = C.load_config(a.config)
    ui = cfg.get("webui", {})
    host, port = a.host or ui.get("host", "127.0.0.1"), a.port or ui.get("port", 8765)
    if host not in LOOPBACK:
        sys.stderr.write("warning: bound to %s. Anyone who can reach this port and has the token can launch and kill jobs.\n" % host)
    token, tpath = load_token(cfg)
    svc = Service(cfg, token)
    try:
        srv = make_server(svc, host, port, a.allow_host)
    except OSError as e:
        raise SystemExit("cannot listen on %s:%d: %s" % (host, port, e))
    srv.verbose = a.verbose
    url = "http://%s:%d/" % ("127.0.0.1" if host in ("0.0.0.0", "") else host, port)
    print("jobctl board: %s#token=%s" % (url, token), flush=True)
    print("token file: %s   (API: %sapi/v1/spec)" % (tpath, url), flush=True)
    if not a.no_refresh:
        threading.Thread(target=svc.refresher, daemon=True).start()
    if a.open:
        webbrowser.open("%s#token=%s" % (url, token))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        svc.stopping = True
        srv.server_close()
    return 0


with open(os.path.join(HERE, "webui_page.html"), encoding="utf-8") as _f:
    INDEX_HTML = _f.read()

if __name__ == "__main__":
    sys.exit(main())
