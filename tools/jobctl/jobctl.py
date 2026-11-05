#!/usr/bin/env python3
"""jobctl: launch and manage many non-interactive batch jobs on a remote cluster.

Generic parts (not tied to one site or one scheduler):
  - a scheduler adapter (PJM and Slurm are built in; add one with a few lines),
  - a transport (ssh to a login node, or any command prefix),
  - a shared-file control channel (state, flags, inbox/outbox, chain) under a run directory,
  - an optional HTTP bridge to the job through a reverse tunnel (needed only when the compute
    nodes cannot be reached any other way),
  - speculative launch (several differently sized requests race; the first to start wins),
  - a chain of allocations for runs longer than one job (the next allocation is queued while the
    current one runs, then takes over from a checkpoint).

One file, Python standard library only (it also runs on the login node).

  laptop:  jobctl.py launch|ls|status|run|stop|pause|resume|drain|resubmit|kill|watch|install|chain-start
  login:   jobctl.py chain-loop RUNDIR | report RUNDIR

Host names, accounts and paths belong in the config file, never in this code.
Config: ~/.config/jobctl/config.json or $JOBCTL_CONFIG. See README.md.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,40}$")
TIME_RE = re.compile(r"^\d{1,3}:[0-5]\d:[0-5]\d$")
LIVE_STATES = ("submitted", "queued", "running", "draining", "paused")

DEFAULTS = {
    "ssh": "",                      # ssh alias of the login node
    "ssh_cmd": None,                # full argv prefix instead of ssh; the remote command is appended (tests: ["bash", "-c"])
    "ssh_control": "",              # ControlPath of the one shared ssh connection (default: <registry>/master.sock)
    "scheduler": "pjm",             # pjm | slurm
    "scheduler_opts": {},           # adapter options, see the adapters below
    "max_nodes": 384,
    "tunnel": True,                 # false: shared-file control only (no HTTP bridge)
    "frontend_target": "",          # login node as seen from the compute nodes (tunnel only)
    "bundle": "",                   # directory holding bash_http_server.py on the shared file system (tunnel only)
    "extra_env": {},                # variables passed to every job
    "rundir": "$HOME/jobctl/runs",  # shared run directories (login node view)
    "registry": "~/.jobctl",        # laptop side state
    "local_ports": [43000, 43599],
    "frontend_ports": [33000, 33599],
    "server_port": 21264,
    "reserved_ports": {"local": [[42000, 42999]], "frontend": [[32000, 32999]]},   # ranges other tools (interactive slots) use
    "external_slots": [],           # globs of slot.env files of interactive bridges; their ports are never handed out
    "max_live": 256,
    "submit_interval": 2.0,         # seconds between submissions
    "interval": 30,
    "lead_min": 60,                 # minutes before the end at which a successor is requested (queue_ahead = 0 mode)
    "chain": True,
    "queue_ahead": 1,               # successors queued while a job runs (0 = request only near the end)
    "max_chain": 50,
}


def load_config(path=None):
    path = path or os.environ.get("JOBCTL_CONFIG") or os.path.expanduser("~/.config/jobctl/config.json")
    cfg = dict(DEFAULTS)
    if os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f))
    cfg["registry"] = os.path.expanduser(cfg["registry"])
    check_ranges(cfg)
    os.makedirs(cfg["registry"], mode=0o700, exist_ok=True)
    return cfg


def overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def check_ranges(cfg):
    """The ranges jobctl hands out must not touch ranges reserved for other tools (interactive bridges)."""
    for kind, key in (("local", "local_ports"), ("frontend", "frontend_ports")):
        mine = cfg[key]
        if mine[0] > mine[1]:
            raise SystemExit("config: %s is an empty range" % key)
        for r in cfg.get("reserved_ports", {}).get(kind, []):
            if overlap(mine, r):
                raise SystemExit("config: %s %s overlaps the reserved range %s (used by interactive bridges); "
                                 "move one of them" % (key, mine, r))


def read_env_file(path):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    out[k.strip()] = v.strip().strip("\"'")
    except OSError:
        pass
    return out


def external_slots(cfg):
    """Interactive bridges described by slot.env files: ports we must not use, and something to show in `ls --all`."""
    out = []
    for pat in cfg.get("external_slots", []):
        for path in sorted(glob.glob(os.path.expanduser(pat))):
            e = read_env_file(path)
            try:
                out.append({"name": "ext:" + os.path.basename(os.path.dirname(path)), "local_port": int(e["LOCAL_PORT"]),
                            "frontend_port": int(e.get("FRONTEND_PORT") or 0), "nodes": int(e.get("NODES") or 0),
                            "path": path})
            except (KeyError, ValueError):
                pass
    return out


# ---------------------------------------------------------------- scheduler adapters
class Scheduler(object):
    """Turns a request into a submit command and reads queue states. 'RUN' and 'QUE' are the only states used."""
    name = ""

    def submit_argv(self, opts, req):
        raise NotImplementedError

    def jobid_re(self):
        raise NotImplementedError

    def states_cmd(self):
        raise NotImplementedError

    def parse_states(self, text):
        raise NotImplementedError

    def cancel_cmd(self, jobid):
        raise NotImplementedError

    def parse_jobid(self, text):
        m = self.jobid_re().search(text)
        return m.group(1) if m else None


class PJM(Scheduler):
    """Fujitsu PJM (pjsub / pjstat / pjdel).
    Options: project (-g), rscgrp, node_spec {nodes: "2x3x2"}, resource {"freq": 2000, ...} added to -L, extra [args]."""
    name = "pjm"

    def submit_argv(self, opts, req):
        spec = str(opts.get("node_spec", {}).get(str(req["nodes"]), req["nodes"]))
        lres = ["rscgrp=%s" % opts.get("rscgrp", "small"), "node=%s" % spec, "elapse=%s" % req["elapse"]]
        lres += ["%s=%s" % kv for kv in sorted(opts.get("resource", {}).items())]
        argv = ["pjsub", "--no-check-directory", "-L", ",".join(lres), "-N", req["name"][:30]]
        if opts.get("project"):
            argv += ["-g", opts["project"]]
        argv += list(opts.get("extra", []))
        argv += ["-x", ",".join("%s=%s" % kv for kv in sorted(req["env"].items()))]
        argv += ["-o", req["out"], "-e", req["err"], req["script"]]
        return argv

    def jobid_re(self):
        return re.compile(r"Job (\d+) submitted")

    def states_cmd(self):
        return "pjstat 2>/dev/null"

    def parse_states(self, text):
        st = {}
        for line in text.splitlines():
            t = line.split()
            if len(t) > 3 and t[0].isdigit():
                st[t[0]] = "RUN" if t[3] in ("RUN", "RNO", "RNE") else "QUE"
        return st

    def cancel_cmd(self, jobid):
        return "pjdel %s" % shlex.quote(jobid)


class Slurm(Scheduler):
    """Slurm (sbatch / squeue / scancel). Not tested on a real Slurm system.
    Options: account, partition, extra [args]."""
    name = "slurm"

    def submit_argv(self, opts, req):
        h, m, s = [int(x) for x in req["elapse"].split(":")]
        argv = ["sbatch", "--parsable", "-N", str(req["nodes"]), "-t", "%d-%02d:%02d:%02d" % (h // 24, h % 24, m, s),
                "-J", req["name"][:30], "-o", req["out"], "-e", req["err"],
                "--export", "ALL," + ",".join("%s=%s" % kv for kv in sorted(req["env"].items()))]
        if opts.get("account"):
            argv += ["-A", opts["account"]]
        if opts.get("partition"):
            argv += ["-p", opts["partition"]]
        argv += list(opts.get("extra", []))
        return argv + [req["script"]]

    def jobid_re(self):
        return re.compile(r"^(\d+)", re.M)

    def states_cmd(self):
        return 'squeue -h -u "$USER" -o "%i %T" 2>/dev/null'

    def parse_states(self, text):
        st = {}
        for line in text.splitlines():
            t = line.split()
            if len(t) == 2 and t[0].isdigit():
                st[t[0]] = "RUN" if t[1] in ("RUNNING", "COMPLETING") else "QUE"
        return st

    def cancel_cmd(self, jobid):
        return "scancel %s" % shlex.quote(jobid)


SCHEDULERS = {"pjm": PJM, "slurm": Slurm}


def scheduler(name):
    try:
        return SCHEDULERS[name]()
    except KeyError:
        raise SystemExit("unknown scheduler %r (have: %s)" % (name, ", ".join(sorted(SCHEDULERS))))


# ---------------------------------------------------------------- transport
class Remote(object):
    """Runs a shell command on the login node over one shared ssh connection."""

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.get("ssh_cmd"):
            self.prefix = list(cfg["ssh_cmd"])
        else:
            if not cfg.get("ssh"):
                raise SystemExit("config: set 'ssh' (login node alias)")
            self.prefix = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ControlMaster=auto",
                           "-o", "ControlPersist=yes", "-S", self.control_path(), cfg["ssh"]]

    def control_path(self):
        return self.cfg.get("ssh_control") or os.path.join(self.cfg["registry"], "master.sock")

    def run(self, cmd, data=None, timeout=120):
        if isinstance(data, str):
            data = data.encode()
        p = subprocess.run(self.prefix + [cmd], input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")

    def put(self, path, data, mode="644"):
        """Write-once file: temp name, then rename. A reader never sees half a file."""
        d = os.path.dirname(path)
        cmd = "mkdir -p %s && cat > %s.tmp$$ && chmod %s %s.tmp$$ && mv %s.tmp$$ %s" % (d, path, mode, path, path, path)
        rc, _, err = self.run(cmd, data)
        if rc != 0:
            raise RuntimeError("put %s failed: %s" % (path, err.strip()))

    def get(self, path):
        rc, out, _ = self.run("cat %s 2>/dev/null" % path)
        return out if rc == 0 else None

    def forward(self, op, local, front):
        """Add ('forward') or remove ('cancel') a local port forward on the shared connection."""
        if self.cfg.get("ssh_cmd"):
            return self.cfg.get("_forward_hook", lambda *a: 0)(op, local, front)
        spec = "127.0.0.1:%d:127.0.0.1:%d" % (local, front)
        p = subprocess.run(["ssh", "-S", self.control_path(), "-O", op, "-L", spec, self.cfg["ssh"]],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        return p.returncode


class Local(object):
    """Same interface as Remote, for code that runs on the login node itself."""

    def run(self, cmd, data=None, timeout=120):
        p = subprocess.run(["bash", "-c", cmd], input=data.encode() if isinstance(data, str) else data,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")

    def put(self, path, data, mode="644"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp%d" % os.getpid()
        with open(tmp, "wb") as f:
            f.write(data.encode() if isinstance(data, str) else data)
        os.chmod(tmp, int(mode, 8))
        os.replace(tmp, path)

    def get(self, path):
        try:
            with open(path) as f:
                return f.read()
        except OSError:
            return None


# ---------------------------------------------------------------- registry (laptop)
def job_path(cfg, name):
    return os.path.join(cfg["registry"], "jobs", name, "job.json")


def load_job(cfg, name):
    try:
        with open(job_path(cfg, name)) as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit("no such job: %s" % name)


def save_job(cfg, job):
    p = job_path(cfg, job["name"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(job, f, indent=1, sort_keys=True)
    os.replace(tmp, p)


def all_jobs(cfg):
    out = []
    d = os.path.join(cfg["registry"], "jobs")
    for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        try:
            with open(os.path.join(d, n, "job.json")) as f:
                out.append(json.load(f))
        except (OSError, ValueError):
            pass
    return out


def port_free_local(port):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def parse_proc_net_tcp(text):
    """Listening ports from /proc/net/tcp and /proc/net/tcp6 (state 0A). Works where `ss` and `netstat` are missing."""
    ports = set()
    for line in text.splitlines():
        t = line.split()
        if len(t) > 3 and t[3] == "0A" and ":" in t[1]:
            try:
                ports.add(int(t[1].rsplit(":", 1)[1], 16))
            except ValueError:
                pass
    return ports


def listening_ports(remote):
    rc, out, _ = remote.run("cat /proc/net/tcp /proc/net/tcp6 2>/dev/null")
    return parse_proc_net_tcp(out) if rc == 0 else set()


def alloc_ports(cfg, remote=None):
    """A (local, frontend) pair no registered job uses, free here and (if asked) on the login node."""
    jobs = all_jobs(cfg) + external_slots(cfg)
    used_l = set(j["local_port"] for j in jobs)
    used_f = set(j["frontend_port"] for j in jobs)
    busy_f = listening_ports(remote) if remote is not None else set()
    (llo, lhi), (flo, fhi) = cfg["local_ports"], cfg["frontend_ports"]
    for i in range(min(lhi - llo, fhi - flo) + 1):
        lp, fp = llo + i, flo + i
        if lp in used_l or fp in used_f or fp in busy_f or not port_free_local(lp):
            continue
        return lp, fp
    raise SystemExit("no free port pair in the configured ranges")


def seconds(hms):
    h, m, s = [int(x) for x in hms.split(":")]
    return h * 3600 + m * 60 + s


# ---------------------------------------------------------------- install + launch
def install_files(cfg, remote):
    """Copy tool files to the run directory under content-hashed names (a running script is never edited)."""
    out = {}
    for fn in ("jobctl.py", "agent.sh", "job.sh"):
        with open(os.path.join(HERE, fn), "rb") as f:
            data = f.read()
        stem, ext = os.path.splitext(fn)
        dst = "%s/_bin/%s.%s%s" % (cfg["rundir"], stem, hashlib.sha256(data).hexdigest()[:10], ext)
        if remote.run("test -f %s" % dst)[0] != 0:
            remote.put(dst, data, "755")
        out[fn] = dst
    return out


def resolve(remote, text):
    """Expand $HOME and ~ on the login node so stored submit arguments are absolute."""
    if "$" not in text and not text.startswith("~"):
        return text
    rc, out, _ = remote.run("echo %s" % text)
    return out.strip() if rc == 0 and out.strip() else text


def parse_variants(text, default_nodes, default_time):
    """'384@12:00:00,192@12:00:00' -> [(384, '12:00:00'), (192, '12:00:00')]."""
    out = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        n, _, t = part.partition("@")
        out.append((int(n), t or default_time))
    return out or [(default_nodes, default_time)]


def build_request(cfg, name, run, files, nodes, elapse, variant, race):
    env = {
        "CTL_NAME": name, "CTL_RUNDIR": run, "CTL_SEQ": "{SEQ}", "CTL_AGENT": files["agent.sh"],
        "CTL_INTERVAL": str(cfg["interval"]), "CTL_LEAD_MIN": str(cfg["lead_min"]),
        "CTL_ELAPSE_SEC": str(seconds(elapse)), "CTL_CHAIN": "1" if cfg["chain"] else "0",
        "CTL_NODES": str(nodes), "CTL_VARIANT": str(variant), "CTL_RACE": "1" if race else "0",
    }
    if cfg["tunnel"]:
        env.update({
            "CTL_TUNNEL": "1", "CTL_BUNDLE": cfg["bundle"], "FRONTEND_SSH_TARGET": cfg["frontend_target"],
            "SERVER_PORT": str(cfg["server_port"]), "CTL_FPORT_LO": str(cfg["frontend_ports"][0]),
            "CTL_FPORT_HI": str(cfg["frontend_ports"][1]),
        })
    else:
        env["CTL_TUNNEL"] = "0"
    env.update(cfg.get("extra_env", {}))
    return {"name": name, "nodes": nodes, "elapse": elapse, "env": env, "script": files["job.sh"],
            "out": "%s/logs/job.{SEQ}.out" % run, "err": "%s/logs/job.{SEQ}.err" % run}


def fill(argv, seq):
    return [a.replace("{SEQ}", str(seq)) for a in argv]


def submit(remote, sched, argv, seq):
    cmd = " ".join(shlex.quote(a) for a in fill(argv, seq))
    rc, out, err = remote.run(cmd)
    jid = sched.parse_jobid(out + err)
    if rc != 0 or not jid:
        raise RuntimeError("submit failed rc=%s: %s" % (rc, (out + err).strip()))
    return jid


def cmd_launch(cfg, a):
    cfg = dict(cfg)
    if not NAME_RE.match(a.name):
        raise SystemExit("bad job name (letters, digits, . _ - ; max 40)")
    variants = parse_variants(a.variants, a.nodes, a.time)
    for n, t in variants:
        if not 1 <= n <= cfg["max_nodes"]:
            raise SystemExit("nodes must be 1..%d" % cfg["max_nodes"])
        if not TIME_RE.match(t):
            raise SystemExit("time must be HH:MM:SS")
    if a.lead_min is not None:
        cfg["lead_min"] = a.lead_min
    if a.queue_ahead is not None:
        cfg["queue_ahead"] = a.queue_ahead
    cfg["extra_env"] = dict(cfg.get("extra_env", {}), **dict(kv.split("=", 1) for kv in (a.env or [])))
    cfg["scheduler_opts"] = dict(cfg.get("scheduler_opts", {}))
    cfg["scheduler_opts"]["extra"] = list(cfg["scheduler_opts"].get("extra", [])) + list(a.sched_arg or [])
    if os.path.exists(job_path(cfg, a.name)):
        raise SystemExit("job exists: %s" % a.name)
    if len([j for j in all_jobs(cfg) if j["state"] in LIVE_STATES]) >= cfg["max_live"]:
        raise SystemExit("live job cap reached (%d)" % cfg["max_live"])
    sched = scheduler(cfg["scheduler"])
    remote = Remote(cfg)
    cfg["rundir"] = resolve(remote, cfg["rundir"])
    cfg["bundle"] = resolve(remote, cfg["bundle"])
    run = "%s/%s" % (cfg["rundir"], a.name)
    fake = dict((k, "<rundir>/_bin/%s" % k) for k in ("jobctl.py", "agent.sh", "job.sh"))
    race = len(variants) > 1
    if a.dry_run:
        for i, (n, t) in enumerate(variants):
            argv = sched.submit_argv(cfg["scheduler_opts"], build_request(cfg, a.name, run, fake, n, t, i, race))
            print(" ".join(shlex.quote(x) for x in fill(argv, i + 1)))
        return
    files = install_files(cfg, remote)
    lp, fp = alloc_ports(cfg, remote) if cfg["tunnel"] else (0, 0)
    if remote.run("mkdir -p %s/{state,flags,inbox,outbox,chain,work,logs}" % run)[0] != 0:
        raise SystemExit("cannot create the run directory")
    if a.script:
        with open(a.script, "rb") as f:
            remote.put("%s/work.sh" % run, f.read(), "755")
    reqs = [build_request(cfg, a.name, run, files, n, t, i, race) for i, (n, t) in enumerate(variants)]
    argvs = [sched.submit_argv(cfg["scheduler_opts"], r) for r in reqs]
    # successors are never racers: a racer that finds the lock taken leaves, a successor waits for the handoff
    succ = [sched.submit_argv(cfg["scheduler_opts"], build_request(cfg, a.name, run, files, n, t, i, False))
            for i, (n, t) in enumerate(variants)]
    spec = {"name": a.name, "scheduler": cfg["scheduler"], "variants": argvs, "successors": succ, "variant_nodes": [v[0] for v in variants],
            "interval": cfg["interval"], "max_chain": cfg["max_chain"], "queue_ahead": cfg["queue_ahead"],
            "chain": cfg["chain"], "frontend_port": fp}
    remote.put("%s/spec.json" % run, json.dumps(spec, indent=1))
    remote.put("%s/state/port.0" % run, "%d\n" % fp)
    ids, log = [], ""
    for i, argv in enumerate(argvs):
        jid = submit(remote, sched, argv, i + 1)
        ids.append(jid)
        log += "%d %s %d %s %d\n" % (i + 1, jid, time.time(), "R" if race else "S", i)
        time.sleep(cfg["submit_interval"])
    remote.put("%s/chain/seq" % run, "%d\n" % len(argvs))
    remote.put("%s/chain/jobs.log" % run, log)
    save_job(cfg, {"name": a.name, "nodes": max(v[0] for v in variants), "elapse": variants[0][1], "jobids": ids,
                   "seq": len(ids), "local_port": lp, "frontend_port": fp, "state": "submitted", "created": time.time(),
                   "run": run, "session": "", "hb": 0, "scheduler": cfg["scheduler"], "tunnel": cfg["tunnel"],
                   "variants": [list(v) for v in variants]})
    print("launched %s jobs=%s%s%s" % (a.name, ",".join(ids), " (race: first to start wins)" if race else "",
                                      " local=%d frontend=%d" % (lp, fp) if cfg["tunnel"] else ""))


# ---------------------------------------------------------------- control files
def flag(remote, job, name, on=True):
    rc, _, err = remote.run(("touch %s/flags/%s" if on else "rm -f %s/flags/%s") % (job["run"], name))
    if rc != 0:
        raise SystemExit(err)


def cmd_flag(cfg, a, name, on=True):
    flag(Remote(cfg), load_job(cfg, a.name), name, on)
    print("%s %s" % (name, "set" if on else "cleared"))


def cmd_kill(cfg, a):
    job = load_job(cfg, a.name)
    remote, sched = Remote(cfg), scheduler(job.get("scheduler", cfg["scheduler"]))
    flag(remote, job, "STOP")
    for jid in job["jobids"]:       # every allocation of the chain, queued or running
        remote.run(sched.cancel_cmd(jid))
    job["state"] = "stopped"
    save_job(cfg, job)
    print("stopped %s" % a.name)


def cmd_resubmit(cfg, a):
    job = load_job(cfg, a.name)
    remote = Remote(cfg)
    flag(remote, job, "STOP", False)
    print(chain_submit(remote, job["run"]) or "nothing submitted")


# ---------------------------------------------------------------- HTTP (tunnel up) and inbox (tunnel down)
def http(port, method, path, body=None, timeout=60):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path),
                                 data=json.dumps(body).encode() if body is not None else None,
                                 method=method, headers={"Content-Type": "application/json"})
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout)


def healthy(port):
    try:
        return bool(json.loads(http(port, "GET", "/health", timeout=3).read().decode()).get("ok"))
    except Exception:
        return False


def run_http(cfg, job, command, timeout):
    """Bridge wire format: POST /session -> {"session"}, POST /run -> NDJSON stdout / exit / error events."""
    port = job["local_port"]
    for attempt in (0, 1):
        sid = job.get("session")
        if not sid:
            sid = json.loads(http(port, "POST", "/session", {}).read().decode())["session"]
            job["session"] = sid
            save_job(cfg, job)
        try:
            resp = http(port, "POST", "/run", {"session": sid, "command": command, "timeout": timeout,
                                               "max_bytes": 1000000, "normalize": True}, timeout + 10)
        except urllib.error.HTTPError as e:
            if e.code == 404 and attempt == 0:      # the idle sweeper removed the session
                job["session"] = ""
                continue
            if e.code == 409:
                sys.stderr.write("session busy (one command at a time); retry or use --inbox\n")
                return 125
            raise
        rc = None
        for line in resp:
            try:
                ev = json.loads(line.decode())
            except ValueError:
                continue
            t = ev.get("type")
            if t == "stdout":
                sys.stdout.write(ev.get("data", ""))
            elif t == "exit":
                rc = ev.get("code")
            elif t == "error" and ev.get("reason") == "timeout":
                rc = 124
        return 124 if rc is None else int(rc)
    return 1


def run_inbox(cfg, job, command, timeout):
    remote = Remote(cfg)
    seq = "%013d" % int(time.time() * 1000)
    remote.put("%s/inbox/%s.cmd" % (job["run"], seq), command + "\n")
    end = time.time() + timeout
    while time.time() < end:
        rc_txt = remote.get("%s/outbox/%s.rc" % (job["run"], seq))
        if rc_txt is not None:
            sys.stdout.write(remote.get("%s/outbox/%s.out" % (job["run"], seq)) or "")
            return int(rc_txt.strip() or 1)
        time.sleep(min(5, cfg["interval"] / 3.0))
    sys.stderr.write("timed out waiting for the node agent (the command stays queued: %s)\n" % seq)
    return 124


def cmd_run(cfg, a):
    job = load_job(cfg, a.name)
    if "exit" in a.command.split() or "set -e" in a.command:
        raise SystemExit("refusing a command with 'exit' or 'set -e' (it would end the shared shell); wrap it in ( ... )")
    if job.get("tunnel", True) and healthy(job["local_port"]) and not a.inbox:
        return run_http(cfg, job, a.command, a.timeout)
    return run_inbox(cfg, job, a.command, a.timeout)


# ---------------------------------------------------------------- status, report, watch
def report(rundir):
    """Runs on the login node: every job's info, flags and chain state as one JSON object."""
    out = {}
    rundir = os.path.expandvars(rundir)
    for name in sorted(os.listdir(rundir)) if os.path.isdir(rundir) else []:
        base = os.path.join(rundir, name)
        if name.startswith("_") or not os.path.isdir(os.path.join(base, "state")):
            continue
        rec = {"info": None, "flags": [], "seq": 0, "jobs": [], "winner": None}
        try:
            with open(os.path.join(base, "state", "info.json")) as f:
                rec["info"] = json.load(f)
        except (OSError, ValueError):
            pass
        fl = os.path.join(base, "flags")
        rec["flags"] = sorted(os.listdir(fl)) if os.path.isdir(fl) else []
        try:
            with open(os.path.join(base, "chain", "seq")) as f:
                rec["seq"] = int(f.read().strip() or 0)
            with open(os.path.join(base, "chain", "jobs.log")) as f:
                rec["jobs"] = [l.split()[1] for l in f if l.strip()]
        except OSError:
            pass
        try:
            with open(os.path.join(base, "chain", "winner")) as f:
                rec["winner"] = f.read().split()
        except OSError:
            pass
        rec["next_req"] = os.path.exists(os.path.join(base, "chain", "next.req"))
        out[name] = rec
    return out


def queue_states(runner, sched):
    return sched.parse_states(runner.run(sched.states_cmd())[1])


def refresh(cfg, remote, jobs):
    """One ssh call for every job's report, one for the queue. Updates the registry and the local forwards."""
    py = install_files(cfg, remote)["jobctl.py"]
    rc, out, err = remote.run("python3 %s report %s" % (py, cfg["rundir"]))
    rep = json.loads(out) if rc == 0 and out.strip() else {}
    sched = scheduler(cfg["scheduler"])
    states = queue_states(remote, sched)
    now = time.time()
    for job in jobs:
        r = rep.get(job["name"], {})
        info = r.get("info") or {}
        ids = r.get("jobs") or job["jobids"]
        job["jobids"], job["seq"] = ids, r.get("seq", job["seq"])
        job["hb"] = info.get("hb", 0)
        live = [states[i] for i in ids if i in states]
        if "STOP" in r.get("flags", []) and not live:
            job["state"] = "stopped"
        elif info.get("phase") == "done":
            job["state"] = "done"
        elif "RUN" in live:
            job["state"] = "draining" if "DRAIN" in r.get("flags", []) else ("paused" if "PAUSE" in r.get("flags", []) else "running")
        elif live:
            job["state"] = "queued"
        elif job["state"] in LIVE_STATES and r and now - job.get("created", now) > 3 * cfg["interval"]:
            job["state"] = "lost"
        fp = info.get("frontend_port")
        if job.get("tunnel", True):
            if fp and fp != job["frontend_port"] and job["state"] in LIVE_STATES:
                remote.forward("cancel", job["local_port"], job["frontend_port"])
                job["frontend_port"], job["forwarded"] = fp, False
            if job["state"] in ("running", "draining", "paused") and (not job.get("forwarded") or not healthy(job["local_port"])):
                remote.forward("forward", job["local_port"], job["frontend_port"])
                job["forwarded"] = True
        save_job(cfg, job)
    return rep


def cmd_ls(cfg, a):
    jobs = all_jobs(cfg)
    if a.refresh and jobs:
        refresh(cfg, Remote(cfg), jobs)
        jobs = all_jobs(cfg)
    now = time.time()
    print("%-20s %-8s %5s %-9s %8s %5s %s" % ("NAME", "STATE", "NODES", "ELAPSE", "HB_AGE", "SEQ", "JOB"))
    for j in jobs:
        print("%-20s %-8s %5d %-9s %8s %5d %s" % (j["name"], j["state"], j["nodes"], j["elapse"],
                                                   "%ds" % (now - j["hb"]) if j.get("hb") else "-", j["seq"], j["jobids"][-1]))
    if getattr(a, "all", False):
        for e in external_slots(cfg):
            print("%-20s %-8s %5d %-9s %8s %5s %s" % (e["name"], "up" if healthy(e["local_port"]) else "down", e["nodes"],
                                                       "interact", "-", "-", "port %d" % e["local_port"]))


def cmd_ports(cfg, a):
    """Who holds which port: jobctl jobs, interactive slots, and (login node) listeners. Exit 1 on a conflict."""
    remote = Remote(cfg)
    listening = listening_ports(remote)
    rows, seen, bad = [], {}, 0
    for kind, items in (("jobctl", all_jobs(cfg)), ("slot", external_slots(cfg))):
        for j in items:
            for side, port in (("local", j["local_port"]), ("frontend", j["frontend_port"])):
                if not port:
                    continue
                key = (side, port)
                other = seen.get(key)
                bad += other is not None
                seen[key] = other or j["name"]
                rows.append((j["name"], kind, side, port, "CONFLICT with " + other if other else "ok",
                             "listening" if side == "frontend" and port in listening else ""))
    print("%-22s %-7s %-9s %6s  %-10s %s" % ("NAME", "KIND", "SIDE", "PORT", "CHECK", ""))
    for r in rows:
        print("%-22s %-7s %-9s %6d  %-10s %s" % r)
    if a.check:
        lp, fp = [int(x) for x in a.check]
        for side, port in (("local", lp), ("frontend", fp)):
            rng = cfg["local_ports" if side == "local" else "frontend_ports"]
            if rng[0] <= port <= rng[1] or (side, port) in seen:
                print("CONFLICT: %s port %d is in jobctl's range or already registered" % (side, port))
                bad += 1
        if not bad:
            print("ports %d/%d are free of jobctl" % (lp, fp))
    return 1 if bad else 0


def cmd_status(cfg, a):
    job = load_job(cfg, a.name)
    refresh(cfg, Remote(cfg), [job])
    job = load_job(cfg, a.name)
    if job.get("tunnel", True):
        job["tunnel_ok"] = healthy(job["local_port"])
    print(json.dumps(job, indent=1, sort_keys=True))


def cmd_watch(cfg, a):
    remote = Remote(cfg)
    while True:
        jobs = [j for j in all_jobs(cfg) if j["state"] in LIVE_STATES]
        if jobs:
            try:
                refresh(cfg, remote, jobs)
            except Exception as e:      # keep watching through ssh drops
                sys.stderr.write("watch: %s\n" % e)
        if a.once:
            return
        time.sleep(cfg["interval"])


# ---------------------------------------------------------------- chain (runs on the login node)
def read_jobs_log(run, runner):
    rows = []
    for line in (runner.get(run + "/chain/jobs.log") or "").splitlines():
        t = line.split()
        if len(t) >= 2:
            rows.append({"seq": int(t[0]), "id": t[1], "time": float(t[2]) if len(t) > 2 else 0,
                         "tag": t[3] if len(t) > 3 else "S", "variant": int(t[4]) if len(t) > 4 else 0})
    return rows


def chain_submit(runner, run, variant=None):
    """Submit one successor, from the winning variant's request. Lock by mkdir; seq advances only on success."""
    r = runner or Local()
    spec = json.loads(r.get(run + "/spec.json"))
    sched = scheduler(spec.get("scheduler", "pjm"))
    if r.run("mkdir %s/chain/lock 2>/dev/null" % run)[0] != 0:
        return None
    try:
        seq = int((r.get(run + "/chain/seq") or "0").strip()) + 1
        if seq > spec.get("max_chain", 50):
            return None
        if variant is None:
            w = (r.get(run + "/chain/winner") or "").split()
            variant = int(w[1]) if len(w) > 1 else 0
        jid = submit(r, sched, spec.get("successors", spec["variants"])[variant], seq)
        r.put(run + "/chain/seq", "%d\n" % seq)
        r.run("echo '%d %s %d S %d' >> %s/chain/jobs.log; rm -f %s/chain/next.req" % (seq, jid, time.time(), variant, run, run))
        return "submitted seq=%d job=%s" % (seq, jid)
    finally:
        r.run("rmdir %s/chain/lock" % run)


def chain_tick(rundir, now=None):
    """One pass over every run directory. Returns messages."""
    now = now or time.time()
    msgs = []
    loc = Local()
    rep = report(rundir)
    cache = {}
    for name, r in rep.items():
        run = os.path.join(os.path.expandvars(rundir), name)
        if "STOP" in r["flags"] or os.path.exists(run + "/work/DONE") or (r["info"] or {}).get("phase") == "done":
            continue
        spec = json.loads(loc.get(run + "/spec.json"))
        sched = cache.setdefault(spec.get("scheduler", "pjm"), scheduler(spec.get("scheduler", "pjm")))
        states = queue_states(loc, sched)
        rows = read_jobs_log(run, loc)
        running = [x for x in rows if states.get(x["id"]) == "RUN"]
        queued_succ = [x for x in rows if states.get(x["id"]) == "QUE" and x["tag"] == "S"]
        if r["winner"]:         # the race is decided: drop the racers still waiting in the queue
            for x in rows:
                if x["tag"] == "R" and states.get(x["id"]) == "QUE":
                    loc.run(sched.cancel_cmd(x["id"]))
                    msgs.append("%s: cancelled racer %s" % (name, x["id"]))
        hb = (r["info"] or {}).get("hb", 0)
        active = [x for x in rows if x["id"] in states]
        stale = (not active) and hb and now - hb > 3 * spec.get("interval", 30)
        ahead = spec.get("queue_ahead", 0)
        want = r["next_req"] and not queued_succ
        early = bool(running) and r["winner"] is not None and len(queued_succ) < ahead and spec.get("chain", True)
        if want or stale or early:
            res = chain_submit(None, run)
            if res:
                msgs.append("%s: %s (%s)" % (name, res, "requested" if want else "stale heartbeat" if stale else "queued ahead"))
    return msgs


def cmd_chain_loop(cfg_unused, a):
    while True:
        try:
            for m in chain_tick(a.rundir):
                print(time.strftime("%FT%T"), m, flush=True)
        except Exception as e:
            print("chain: %s" % e, file=sys.stderr, flush=True)
        if a.once:
            return
        time.sleep(a.interval)


def cmd_install(cfg, a):
    print(json.dumps(install_files(cfg, Remote(cfg)), indent=1))


def cmd_chain_start(cfg, a):
    remote = Remote(cfg)
    py = install_files(cfg, remote)["jobctl.py"]
    pid = "%s/_chain.pid" % cfg["rundir"]
    if "up" in remote.run("test -f %s && kill -0 $(cat %s) 2>/dev/null && echo up" % (pid, pid))[1]:
        print("chain loop already running")
        return
    remote.run("nohup python3 %s chain-loop %s --interval %d > %s/_chain.log 2>&1 < /dev/null & echo $! > %s" % (
        py, cfg["rundir"], cfg["interval"], cfg["rundir"], pid))
    print("chain loop started")


# ---------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(prog="jobctl")
    p.add_argument("--config")
    sub = p.add_subparsers(dest="cmd")
    sub.required = True      # keyword form needs Python 3.7; the login node has 3.6
    s = sub.add_parser("launch")
    s.add_argument("name")
    s.add_argument("--nodes", type=int, default=1)
    s.add_argument("--time", default="01:00:00", help="HH:MM:SS")
    s.add_argument("--variants", help="speculative launch: 'NODES@HH:MM:SS,...'; all are queued, the first to start wins")
    s.add_argument("--script", help="workload script run on node 0; it resumes from $CTL_WORK (exit 0 done, 75 continue)")
    s.add_argument("--dry-run", action="store_true", help="print the submit command(s) and exit")
    s.add_argument("--lead-min", type=int)
    s.add_argument("--queue-ahead", type=int, help="successors kept queued while a job runs (default from config)")
    s.add_argument("--env", action="append", metavar="K=V")
    s.add_argument("--sched-arg", action="append", help="extra scheduler argument (repeatable)")
    s = sub.add_parser("ls")
    s.add_argument("--refresh", action="store_true")
    s.add_argument("--all", action="store_true", help="also list interactive bridges from external_slots")
    s = sub.add_parser("ports", help="show port owners; --check LOCAL FRONTEND tests a pair an interactive launch wants")
    s.add_argument("--check", nargs=2, metavar=("LOCAL", "FRONTEND"))
    for n in ("status", "stop", "pause", "resume", "drain", "resubmit", "kill"):
        sub.add_parser(n).add_argument("name")
    s = sub.add_parser("run")
    s.add_argument("name")
    s.add_argument("command")
    s.add_argument("--timeout", type=int, default=50)
    s.add_argument("--inbox", action="store_true", help="use the shared-file inbox even if the tunnel is up")
    s = sub.add_parser("watch")
    s.add_argument("--once", action="store_true")
    sub.add_parser("install")
    sub.add_parser("chain-start")
    s = sub.add_parser("chain-loop")
    s.add_argument("rundir")
    s.add_argument("--interval", type=int, default=30)
    s.add_argument("--once", action="store_true")
    s = sub.add_parser("report")
    s.add_argument("rundir")
    s = sub.add_parser("webui", help="local web board and JSON API (see webui.py)")
    s.add_argument("rest", nargs=argparse.REMAINDER)
    args = list(sys.argv[1:] if argv is None else argv)
    if "webui" in args[:3]:     # everything after the word belongs to the board, including --help
        i = args.index("webui")
        import webui          # imported only here: nothing else needs it
        return webui.main(args[i + 1:], config=args[1] if i == 2 and args[0] == "--config" else None)
    a = p.parse_args(args)
    if a.cmd == "report":
        print(json.dumps(report(a.rundir)))
        return 0
    if a.cmd == "chain-loop":
        return cmd_chain_loop(None, a) or 0
    cfg = load_config(a.config)
    table = {
        "launch": cmd_launch, "ls": cmd_ls, "ports": cmd_ports, "status": cmd_status, "run": cmd_run, "watch": cmd_watch,
        "install": cmd_install, "chain-start": cmd_chain_start, "resubmit": cmd_resubmit, "kill": cmd_kill,
        "stop": lambda c, x: cmd_flag(c, x, "STOP"), "pause": lambda c, x: cmd_flag(c, x, "PAUSE"),
        "resume": lambda c, x: cmd_flag(c, x, "PAUSE", False), "drain": lambda c, x: cmd_flag(c, x, "DRAIN"),
    }
    return table[a.cmd](cfg, a) or 0


if __name__ == "__main__":
    sys.exit(main())
