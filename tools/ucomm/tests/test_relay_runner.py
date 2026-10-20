#!/usr/bin/env python3
"""Exercise the repeat runner locally, including metadata and persisted results."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

binary = Path(sys.argv[1] if len(sys.argv) > 1 else "build/ucomm_relay").resolve()
runner = Path(__file__).resolve().parents[1] / "python" / "relay_bench.py"
with tempfile.TemporaryDirectory() as tmp:
    directory = Path(tmp)
    sockets = [socket.socket(), socket.socket()]
    for sock in sockets:
        sock.bind(("127.0.0.1", 0))
    config = {"roots": [f"127.0.0.1:{sock.getsockname()[1]}" for sock in sockets]}
    config.update({role: {"exe": str(binary)} for role in ("left", "relay", "right")})
    path = directory / "config.json"
    path.write_text(json.dumps(config))
    for sock in sockets:
        sock.close()
    subprocess.run([sys.executable, str(runner), "--config", str(path), "--output", str(directory / "results"),
                    "--backend", "udp", "--size", "8153", "--messages", "7", "--depth", "3",
                    "--latency-iters", "16", "--repeats", "2", "--timeout", "10", "--verify",
                    "--paired", "--ib-inline", "64"],
                   check=True, timeout=45)
    report = json.loads((directory / "results" / "results.json").read_text())
    assert len(report["runs"]) == 4
    assert [r["direction"] for r in report["runs"]] == ["forward", "reverse", "reverse", "forward"]
    expected_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    assert all(host["binary_sha256"] == expected_hash for host in report["hosts"].values())
    assert all(summary["repeats"] == 2 for summary in report["summary"].values())
    assert report["config_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "src/ucomm_relay.c" in report["source_sha256"]
    assert report["settings"]["paired"] and report["settings"]["ib_inline"] == 64
    assert str(path) not in json.dumps(report)
print("relay repeat runner: PASS")
