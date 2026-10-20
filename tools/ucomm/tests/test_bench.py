#!/usr/bin/env python3
"""Validate asymmetric benchmark arguments and the small-buffer round-trip path."""
import os
from pathlib import Path
import socket
import subprocess
import sys

binary = Path(sys.argv[1] if len(sys.argv) > 1 else "build/ucomm_bench").resolve()
for value in ("8", "0:8", "8:-1", "8:", "8:8garbage", "999999999999999999999999:8"):
    result = subprocess.run([str(binary), "-p", value], capture_output=True, timeout=3)
    assert result.returncode == 2, value
for value in ("0", "-1"):
    result = subprocess.run([str(binary), "-n", value], capture_output=True, timeout=3)
    assert result.returncode == 2, value
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    root = f"127.0.0.1:{sock.getsockname()[1]}"
args = [str(binary), "-w", "2", "-b", "udp", "-a", root, "-p", "8:17", "-n", "16"]
env = dict(os.environ, UCOMM_TIMEOUT_MS="3000")
server = subprocess.Popen(args + ["-r", "0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
try:
    peer = subprocess.run(args + ["-r", "1"], capture_output=True, timeout=10, env=env)
    out, err = server.communicate(timeout=10)
    assert peer.returncode == server.returncode == 0, (out, err, peer.stderr)
    assert "roundtrip 8 B -> 17 B:" in out
finally:
    if server.poll() is None:
        server.kill()
    server.wait()
print("asymmetric benchmark: PASS")
