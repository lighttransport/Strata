#!/usr/bin/env python3
"""Repeat three-process relay measurements using an external machine config.

Config keys: roots (two host:port strings), left/relay/right (each with exe and
optional host, ib_dev). Endpoints use port; relay uses left_port/right_port.
Omit host for local execution. Actual hostnames and paths belong in ignored local
configuration, not source. Remote argv is shell-quoted; no shell is used locally.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import time


def on_host(spec, argv):
    host = spec.get("host")
    if host:
        if host.startswith("-"):
            raise ValueError("host cannot start with '-'")
        ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        if spec.get("host_key_alias"):
            ssh += ["-o", "HostKeyAlias=" + spec["host_key_alias"]]
        return ssh + [host, shlex.join(argv)]
    return argv


def metadata(spec):
    code = '''import hashlib,json,pathlib,platform,resource,sys
p=pathlib.Path
devices=sorted(p('/sys/class/infiniband').glob('*'))
d=p('/sys/class/infiniband')/sys.argv[2] if sys.argv[2] else (devices[0] if devices else None)
def read(path):
 try:return path.read_text().strip()
 except OSError:return None
cpu=next((s.split(':',1)[1].strip() for s in p('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),None)
print(json.dumps(dict(binary_sha256=hashlib.sha256(p(sys.argv[1]).read_bytes()).hexdigest(),kernel=platform.release(),cpu=cpu,memlock=resource.getrlimit(resource.RLIMIT_MEMLOCK),pci_width=read(d/'device/current_link_width') if d else None,pci_speed=read(d/'device/current_link_speed') if d else None,ib_rates={port.name:read(port/'rate') for port in (d/'ports').glob('*')} if d else {})))'''
    result = subprocess.run(on_host(spec, ["python3", "-c", code, spec["exe"], spec.get("ib_dev", "")]),
                            capture_output=True, text=True, check=True, timeout=20)
    return json.loads(result.stdout)


def command(role, spec, roots, args, reverse):
    argv = ["env", f"UCOMM_TIMEOUT_MS={args.timeout * 1000}", "UCOMM_VERBOSE=0"]
    argv.append(f"UCOMM_IB_INLINE={args.ib_inline}")
    if spec.get("ib_dev"):
        argv.append("UCOMM_IB_DEV=" + spec["ib_dev"])
    argv += ["timeout", str(args.timeout + 5) + "s"]
    if spec.get("cpu") is not None:
        cpu = spec["cpu"]
        if not isinstance(cpu, int) or cpu < 0:
            raise ValueError("cpu must be a nonnegative integer")
        argv += ["taskset", "-c", str(cpu)]
    argv += [spec["exe"], "-r", role,
             "-a", roots[0], "-z", roots[1], "-b", args.backend,
             "-s", str(args.size), "-n", str(args.messages), "-d", str(args.depth),
             "-l", str(args.latency_iters), "-J"]
    if role == "relay":
        argv += ["-p", str(spec.get("left_port", 1)), "-q", str(spec.get("right_port", 2))]
    else:
        argv += ["-p", str(spec.get("port", 1))]
    if args.verify:
        argv.append("-V")
    if args.paired:
        argv.append("-P")
    if reverse:
        argv.append("-R")
    return on_host(spec, argv)


def run(config, args, reverse, directory):
    directory.mkdir()
    procs, logs = [], []
    try:
        for role in ("relay", "left", "right"):
            stdout = open(directory / f"{role}.jsonl", "w")
            stderr = open(directory / f"{role}.stderr", "w")
            logs += [stdout, stderr]
            proc = subprocess.Popen(command(role, config[role], config["roots"], args, reverse),
                                    stdout=stdout, stderr=stderr, start_new_session=True)
            procs.append((role, proc))
        deadline = time.monotonic() + args.timeout + 15
        while True:
            status = [(role, proc.poll()) for role, proc in procs]
            for role, code in status:
                if code not in (None, 0):
                    raise RuntimeError(f"{role} exited {code}; see {directory}")
            if all(code is not None for _, code in status):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"session deadline exceeded; see {directory}")
            time.sleep(0.05)
    finally:
        # Also terminate local descendants. GNU timeout bounds remote descendants
        # if an SSH connection is lost before cleanup reaches the remote process.
        for _, proc in procs:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for _, proc in procs:
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for log in logs:
            log.close()
    events = []
    for role in ("relay", "left", "right"):
        for line in (directory / f"{role}.jsonl").read_text().splitlines():
            event = json.loads(line)
            for value in event.values():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("non-finite benchmark result")
            events.append(event)
    latency = [e for e in events if e["event"] == "latency"]
    bandwidth = [e for e in events if e["event"] == "bandwidth"]
    forwarded = [e for e in events if e["event"] == "relay" and e["tag"] == 20]
    if len(latency) != 1 or len(bandwidth) != 1 or len(forwarded) != 1:
        raise ValueError("missing or duplicate benchmark result")
    bw = bandwidth[0]
    if (bw["backend"] != args.backend or bw["size"] != args.size or
            bw["depth"] != args.depth or bw["messages"] != args.messages or
            bw["reverse"] != int(reverse) or bw["verify"] != int(args.verify) or
            latency[0]["iterations"] != args.latency_iters or
            forwarded[0]["forward_bytes"] != args.size * args.messages or
            forwarded[0]["reverse_bytes"] != 8 or bw["seconds"] <= 0):
        raise ValueError("benchmark result does not match requested configuration")
    return {"latency": latency[0], "bandwidth": bw}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory for results and raw logs")
    parser.add_argument("--source-root", type=Path, help="ucomm source snapshot matching the tested binaries")
    parser.add_argument("--backend", choices=("ib", "udp"), default="ib")
    parser.add_argument("--size", type=int, default=1048576)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--messages", type=int, default=512)
    parser.add_argument("--latency-iters", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60, help="seconds per session")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--paired", action="store_true", help="opt-in paired duplex polling")
    parser.add_argument("--ib-inline", type=int, default=0, help="opt-in inline send limit, 0..256 bytes")
    parser.add_argument("--direction", choices=("forward", "reverse", "both"), default="both")
    args = parser.parse_args()
    if not (1 <= args.size <= 268435456 and 1 <= args.depth <= 64 and
            1 <= args.messages <= 10000000 and 1 <= args.latency_iters <= 1000000 and
            1 <= args.repeats <= 100 and 1 <= args.timeout <= 3600 and 0 <= args.ib_inline <= 256):
        parser.error("size/depth/count/repeats/timeout out of range")
    raw_config = args.config.read_bytes()
    config = json.loads(raw_config)
    if len(config["roots"]) != 2 or any(not isinstance(root, str) for root in config["roots"]):
        parser.error("config roots must contain two host:port strings")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "config_sha256": hashlib.sha256(raw_config).hexdigest(),
              "settings": {k: v for k, v in vars(args).items() if k not in ("config", "output", "source_root")},
              "runs": []}
    # No connection settings are embedded in the summary. Keep private configs
    # and raw error logs local when sharing the summary.
    try:
        root = args.source_root or Path(__file__).resolve().parents[1]
        sources = [path for directory in ("include", "src", "tools", "python")
                   for path in sorted((root / directory).glob("*")) if path.suffix in (".c", ".h", ".py")]
        sources += [root / "Makefile", root / "CMakeLists.txt"]
        report["source_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        report["hosts"] = {role: metadata(config[role]) for role in ("left", "relay", "right")}
        for role, host in report["hosts"].items():
            host["configured_cpu"] = config[role].get("cpu")
        for i in range(args.repeats):
            directions = [False, True] if args.direction == "both" else [args.direction == "reverse"]
            if i % 2:
                directions.reverse()
            for reverse in directions:
                name = f"{i + 1:02d}-{'reverse' if reverse else 'forward'}"
                record = run(config, args, reverse, args.output / name)
                record.update(repeat=i + 1, direction="reverse" if reverse else "forward")
                report["runs"].append(record)
                print(f"{name}: {record['bandwidth']['MBps']:.1f} MB/s, "
                      f"RTT p50/p99 {record['latency']['rtt_p50_us']:.2f}/"
                      f"{record['latency']['rtt_p99_us']:.2f} us", flush=True)
        report["summary"] = {}
        for direction in ("forward", "reverse"):
            rates = [r["bandwidth"]["MBps"] for r in report["runs"] if r["direction"] == direction]
            if rates:
                report["summary"][direction] = {"MBps_min": min(rates), "MBps_median": statistics.median(rates),
                                                "MBps_max": max(rates), "repeats": len(rates)}
    finally:
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
