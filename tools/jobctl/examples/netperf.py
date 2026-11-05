#!/usr/bin/env python3
"""Measure latency and bandwidth of bash-over-HTTP bridges, and how a manager process
can fan tasks out over several of them.

Run it where the manager would run (on the login node: Python 3.6, stdlib only), against the
reverse-tunnel ports of one or more jobs:

    python3 netperf.py --target a=127.0.0.1:32477 --target b=127.0.0.1:32478 --out result.json

Tests (each per target, then all targets at once where it makes sense):
  latency   GET /health, POST /session, and a no-op command on a warm session
  down      bytes streamed node -> manager through the /run NDJSON stream
  up        bytes sent manager -> node inside a /run request body
  fanout    one command that starts one process per node (--fanout-cmd)
  conc      K sessions per target, no-op commands for a fixed time: ops/s and latency
  dispatch  M short tasks over K sessions per target: tasks/s against the ideal rate

`--selftest` starts a local stand-in server and runs everything against it.
"""
import argparse
import http.client
import json
import socketserver
import http.server
import subprocess
import sys
import threading
import time

PAYLOAD_LINE = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+/" * 8   # 512 chars


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))]


def lat_stats(xs):
    ms = [x * 1000.0 for x in xs]
    return {"n": len(ms), "min_ms": round(min(ms), 2) if ms else None, "p50_ms": round(pct(ms, 50), 2) if ms else None,
            "p95_ms": round(pct(ms, 95), 2) if ms else None, "p99_ms": round(pct(ms, 99), 2) if ms else None,
            "max_ms": round(max(ms), 2) if ms else None}


class Conn:
    """One keep-alive HTTP connection plus one bridge session."""

    def __init__(self, host, port, timeout=600):
        self.host, self.port, self.timeout = host, port, timeout
        self.c = http.client.HTTPConnection(host, port, timeout=timeout)
        self.sid = None

    def get(self, path):
        t = time.time()
        self.c.request("GET", path)
        r = self.c.getresponse()
        r.read()
        return time.time() - t, r.status

    def post(self, path, obj):
        data = json.dumps(obj).encode()
        self.c.request("POST", path, body=data, headers={"Content-Type": "application/json"})
        return self.c.getresponse()

    def session(self):
        r = self.post("/session", {})
        self.sid = json.loads(r.read().decode())["session"]
        return self.sid

    def run(self, command, timeout=300, max_bytes=2000000):
        """-> (seconds_total, seconds_first_byte, stdout_bytes, exit_code)"""
        if self.sid is None:
            self.session()
        t0 = time.time()
        r = self.post("/run", {"session": self.sid, "command": command, "timeout": timeout, "max_bytes": max_bytes})
        if r.status != 200:
            r.read()
            raise RuntimeError("run status %d" % r.status)
        first, nbytes, code = None, 0, None
        buf = b""
        while True:
            chunk = r.read(65536)
            if not chunk:
                break
            if first is None:
                first = time.time() - t0
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line:
                    continue
                try:
                    ev = json.loads(line.decode())
                except ValueError:
                    continue
                if ev.get("type") == "stdout":
                    nbytes += len(ev.get("data", ""))
                elif ev.get("type") == "exit":
                    code = ev.get("code")
        return time.time() - t0, first, nbytes, code

    def close(self):
        try:
            if self.sid:
                self.post("/close", {"session": self.sid}).read()
                self.sid = None
            self.c.close()
        except Exception:
            pass


def t_latency(host, port, n):
    out = {}
    c = Conn(host, port)
    out["health"] = lat_stats([c.get("/health")[0] for _ in range(n)])
    ss = []
    for _ in range(max(5, n // 10)):
        d = Conn(host, port)
        t = time.time()
        d.session()
        ss.append(time.time() - t)
        d.close()
    out["new_connection_and_session"] = lat_stats(ss)
    c.session()
    c.run(":")
    out["noop_command"] = lat_stats([c.run(":")[0] for _ in range(n)])
    c.close()
    return out


def t_down(host, port, sizes, reps):
    res = {}
    c = Conn(host, port)
    for size in sizes:
        rows = []
        for _ in range(reps):
            # printable data from a cheap source; no per-byte work on the node
            cmd = "yes %s | head -c %d" % (PAYLOAD_LINE, size)
            tot, first, nb, code = c.run(cmd, max_bytes=size + 1000000)
            rows.append({"s": round(tot, 4), "ttfb_s": round(first or 0, 4), "bytes": nb, "rc": code,
                         "MB_s": round(nb / 1e6 / tot, 3) if tot > 0 else None})
        res[str(size)] = rows
    c.close()
    return res


def t_up(host, port, sizes, reps):
    res = {}
    c = Conn(host, port)
    for size in sizes:
        lines = (PAYLOAD_LINE + "\n") * (size // 513)
        payload = lines[:size]
        cmd = "wc -c <<'NETPERF_EOF'\n%sNETPERF_EOF" % payload
        rows = []
        for _ in range(reps):
            tot, first, nb, code = c.run(cmd, max_bytes=1000)
            rows.append({"s": round(tot, 4), "bytes_sent": len(payload), "rc": code,
                         "MB_s": round(len(payload) / 1e6 / tot, 3) if tot > 0 else None})
        res[str(size)] = rows
    c.close()
    return res


def t_fanout(host, port, cmd, reps):
    c = Conn(host, port)
    rows = []
    for _ in range(reps):
        tot, first, nb, code = c.run(cmd, timeout=120)
        rows.append({"s": round(tot, 4), "rc": code, "stdout_bytes": nb})
    c.close()
    return rows


def t_conc(host, port, k, secs, results, key, down_bytes=0):
    """K sessions on K connections; each loops until the deadline."""
    lat, nbytes, errs = [], [0], [0]
    lock = threading.Lock()
    stop = time.time() + secs

    def worker():
        c = Conn(host, port)
        try:
            c.session()
            while time.time() < stop:
                cmd = ":" if not down_bytes else "yes %s | head -c %d" % (PAYLOAD_LINE, down_bytes)
                tot, first, nb, code = c.run(cmd, max_bytes=down_bytes + 1000000)
                with lock:
                    lat.append(tot)
                    nbytes[0] += nb
        except Exception:
            with lock:
                errs[0] += 1
        finally:
            c.close()

    t0 = time.time()
    ths = [threading.Thread(target=worker) for _ in range(k)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    el = time.time() - t0
    results[key] = {"sessions": k, "seconds": round(el, 2), "ops": len(lat), "ops_per_s": round(len(lat) / el, 1),
                    "MB_s": round(nbytes[0] / 1e6 / el, 2), "errors": errs[0], "latency": lat_stats(lat)}


def t_dispatch(host, port, k, m, task, results, key):
    q = list(range(m))
    lock = threading.Lock()
    done, errs, lat = [0], [0], []

    def worker():
        c = Conn(host, port)
        try:
            c.session()
            while True:
                with lock:
                    if not q:
                        return
                    q.pop()
                tot, first, nb, code = c.run(task)
                with lock:
                    lat.append(tot)
                    done[0] += 1
                    if code != 0:
                        errs[0] += 1
        except Exception:
            with lock:
                errs[0] += 1
        finally:
            c.close()

    t0 = time.time()
    ths = [threading.Thread(target=worker) for _ in range(k)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    el = time.time() - t0
    results[key] = {"sessions": k, "tasks": done[0], "task": task, "seconds": round(el, 2),
                    "tasks_per_s": round(done[0] / el, 1), "errors": errs[0], "latency": lat_stats(lat)}


def parallel(targets, fn):
    """Run fn(name, host, port, out) for every target at the same time."""
    out = {}
    ths = [threading.Thread(target=fn, args=(n, h, p, out)) for n, h, p in targets]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return out


# ---- stand-in server for --selftest (runs the command with bash, returns NDJSON) ----------------

class _Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def start_selftest_server():
    sessions = set()
    lock = threading.Lock()
    n = [0]

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, body, code=200):
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(b'{"ok": true}')

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path == "/session":
                with lock:
                    n[0] += 1
                    sid = "s%d" % n[0]
                    sessions.add(sid)
                return self._send(json.dumps({"session": sid}).encode())
            if body.get("session") not in sessions:
                return self._send(b"{}", 404)
            p = subprocess.run(["bash"], input=body["command"].encode(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            ev = [{"type": "stdout", "data": p.stdout.decode("utf-8", "replace")}, {"type": "exit", "code": p.returncode}]
            self._send("".join(json.dumps(e) + "\n" for e in ev).encode())

    srv = _Srv(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", default=[], help="NAME=HOST:PORT (repeatable)")
    ap.add_argument("--out", help="write the result as JSON")
    ap.add_argument("--n", type=int, default=100, help="latency samples")
    ap.add_argument("--down", default="65536,1048576,8388608,33554432", help="bytes, node -> manager")
    ap.add_argument("--up", default="65536,1048576,4194304", help="bytes, manager -> node (limit 8 MiB)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--fanout-cmd", default="", help="command that starts one process per node, e.g. 'mpiexec -np 12 hostname'")
    ap.add_argument("--sessions", default="1,4,16", help="K values for conc/dispatch")
    ap.add_argument("--secs", type=float, default=10.0, help="seconds per conc point")
    ap.add_argument("--tasks", type=int, default=200)
    ap.add_argument("--task", default="sleep 0.1; echo ok")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    srv = None
    if a.selftest:
        srv = start_selftest_server()
        a.target = ["self=127.0.0.1:%d" % srv.server_address[1]]
        a.n, a.reps, a.secs, a.tasks, a.down, a.up, a.sessions = 20, 2, 1.0, 20, "65536,1048576", "65536,1048576", "1,4"
    if not a.target:
        ap.error("give --target NAME=HOST:PORT (or --selftest)")
    targets = []
    for t in a.target:
        name, hp = t.split("=", 1)
        host, port = hp.rsplit(":", 1)
        targets.append((name, host, int(port)))
    down = [int(x) for x in a.down.split(",") if x]
    up = [int(x) for x in a.up.split(",") if x]
    ks = [int(x) for x in a.sessions.split(",") if x]

    R = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0], "targets": {}, "all_at_once": {}}
    for name, host, port in targets:
        r = R["targets"].setdefault(name, {})
        sys.stderr.write("[%s] latency\n" % name)
        r["latency"] = t_latency(host, port, a.n)
        sys.stderr.write("[%s] down\n" % name)
        r["down"] = t_down(host, port, down, a.reps)
        sys.stderr.write("[%s] up\n" % name)
        r["up"] = t_up(host, port, up, a.reps)
        if a.fanout_cmd:
            sys.stderr.write("[%s] fanout\n" % name)
            r["fanout"] = t_fanout(host, port, a.fanout_cmd, a.reps)
        r["conc"], r["dispatch"] = {}, {}
        for k in ks:
            sys.stderr.write("[%s] conc/dispatch k=%d\n" % (name, k))
            t_conc(host, port, k, a.secs, r["conc"], "k%d" % k)
            t_dispatch(host, port, k, a.tasks, a.task, r["dispatch"], "k%d" % k)

    # the manager drives every target at the same time
    for k in ks:
        sys.stderr.write("[all] conc k=%d per target\n" % k)
        R["all_at_once"]["conc_k%d" % k] = parallel(targets, lambda n, h, p, o, k=k: t_conc(h, p, k, a.secs, o, n))
        R["all_at_once"]["conc_down8M_k%d" % k] = parallel(targets, lambda n, h, p, o, k=k: t_conc(h, p, k, a.secs, o, n, down_bytes=8388608))
        R["all_at_once"]["dispatch_k%d" % k] = parallel(targets, lambda n, h, p, o, k=k: t_dispatch(h, p, k, a.tasks, a.task, o, n))

    text = json.dumps(R, indent=1, sort_keys=True)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(text)
    if srv:
        srv.shutdown()


if __name__ == "__main__":
    main()
