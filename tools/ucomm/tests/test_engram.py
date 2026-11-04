"""Exercise the read-only service and client in separate processes."""
import os
import hashlib
import json
import pathlib
import socket
import subprocess
import sys
import tempfile

build = pathlib.Path(sys.argv[1])
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    address = '127.0.0.1:%d' % sock.getsockname()[1]
with tempfile.NamedTemporaryFile() as fixture:
    fixture.write(b'GGUF0123456789ab')
    fixture.flush()
    env = dict(os.environ, UCOMM_BACKEND='udp', UCOMM_TIMEOUT_MS='5000', STRATA_ENGRAM_ONCE='1')
    service = subprocess.Popen([str(build / 'strata-dsv41-engram'), 'serve', fixture.name, address], env=env)
    try:
        subprocess.run([str(build / 'ucomm_test_engram'), address], env=env, check=True, timeout=15)
        assert service.wait(timeout=10) == 0
    finally:
        if service.poll() is None:
            service.kill()
            service.wait()

    # A fresh session tests the actual production reader, including its split batches.
    env.pop('STRATA_ENGRAM_ONCE', None)
    env['UCOMM_TIMEOUT_MS'] = '500'
    service = subprocess.Popen([str(build / 'strata-dsv41-engram'), 'serve', fixture.name, address], env=env, start_new_session=True)
    try:
        described = subprocess.check_output([str(build / 'strata-dsv41-engram'), 'describe', address], env=env, timeout=15)
        descriptor = json.loads(described)
        digest = hashlib.sha256(b'GGUF0123456789ab').hexdigest()
        assert descriptor['sha256'] == digest
        assert bytes.fromhex(descriptor['header_hex']) == b'GGUF0123456789ab'
        subprocess.run([str(build / 'ucomm_test_engram_native'), address, digest], env=env, check=True, timeout=15)
        mismatch = subprocess.run([str(build / 'ucomm_test_engram_native'), address, '0'*64], env=env, timeout=15)
        assert mismatch.returncode == 3
        subprocess.run([str(build / 'ucomm_test_engram_native'), address, digest], env=env, check=True, timeout=15)
    finally:
        import signal
        os.killpg(service.pid, signal.SIGTERM)
        service.wait(timeout=10)
