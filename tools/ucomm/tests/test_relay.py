#!/usr/bin/env python3
"""Three-process integration: duplex ping-pong, bounded bulk pipeline, verification.

Pass the executable as argv[1]; UDP makes this runnable without RDMA hardware.
"""
import os
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile


def run(binary, size, depth, count, reverse=False, loss=False):
    # Reserve two distinct bootstrap ports until immediately before launch.
    sockets = [socket.socket(), socket.socket()]
    for sock in sockets:
        sock.bind(("127.0.0.1", 0))
    ports = [sock.getsockname()[1] for sock in sockets]
    for sock in sockets:
        sock.close()
    args = [str(binary), "-a", f"127.0.0.1:{ports[0]}", "-z", f"127.0.0.1:{ports[1]}",
            "-b", "udp", "-s", str(size), "-d", str(depth), "-n", str(count), "-l", "16", "-V", "-J"]
    if reverse:
        args.append("-R")
    env = dict(os.environ, UCOMM_TIMEOUT_MS="15000", UCOMM_UDP_DROP="0.02" if loss else "0")
    procs = []
    with tempfile.TemporaryDirectory() as tmp:
        logs = []
        try:
            for role in ("relay", "left", "right"):
                log = open(Path(tmp) / role, "w+")
                logs.append(log)
                procs.append(subprocess.Popen(args + ["-r", role], stdout=log,
                                               stderr=subprocess.STDOUT, env=env))
            for proc in procs:
                assert proc.wait(timeout=25) == 0, "relay role failed"
            events = []
            for log in logs:
                log.flush()
                log.seek(0)
                events.extend(json.loads(line) for line in log if line.startswith("{"))
            latency = [event for event in events if event["event"] == "latency"]
            bw = [event for event in events if event["event"] == "bandwidth"]
            relay = [event for event in events if event["event"] == "relay" and event["tag"] == 20]
            assert len(latency) == len(bw) == len(relay) == 1
            assert latency[0]["iterations"] == 16
            assert 0 <= latency[0]["rtt_min_us"] <= latency[0]["rtt_p50_us"] <= latency[0]["rtt_p99_us"] <= latency[0]["rtt_max_us"]
            assert bw[0]["size"] == size and bw[0]["messages"] == count and bw[0]["depth"] == depth
            assert bw[0]["backend"] == "udp" and bw[0]["reverse"] == int(reverse) and bw[0]["verify"] == 1
            assert bw[0]["seconds"] > 0 and bw[0]["MBps"] > 0
            assert relay[0]["forward_bytes"] == size * count and relay[0]["reverse_bytes"] == 8
            print(f"PASS size={size} depth={depth} count={count} reverse={reverse} loss={loss}")
        except BaseException:
            for log in logs:
                log.flush()
                log.seek(0)
                print(log.read(), file=sys.stderr)
            raise
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
            for log in logs:
                log.close()


if __name__ == "__main__":
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else "build/ucomm_relay").resolve()
    for case in [(1, 1, 9), (8152, 3, 17), (8153, 4, 19), (1048579, 3, 11), (17, 8, 2)]:
        run(binary, *case)
    run(binary, 65539, 3, 13, reverse=True)
    run(binary, 32771, 3, 11, loss=True)
    for invalid in [("-s", "0"), ("-n", "-1"), ("-d", "65"), ("-l", "0"), ("-b", "auto")]:
        result = subprocess.run([str(binary), *invalid], capture_output=True, timeout=3)
        assert result.returncode != 0, invalid
