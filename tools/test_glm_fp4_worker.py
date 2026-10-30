#!/usr/bin/env python3
"""Real CPU worker over loopback UDP; creates ~570 MiB of synthetic weights temporarily."""
import argparse
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile


def fixture(root):
    header, index, payloads, offset = {}, {}, [], 0
    for layer in range(3, 45):
        for projection, rows, cols in (('gate_proj', 2048, 4096), ('up_proj', 2048, 4096), ('down_proj', 4096, 2048)):
            prefix = f'model.language_model.layers.{layer}.mlp.experts.0.{projection}.weight'
            for name, dtype, shape, size, value in (
                (prefix, 'U8', [rows, cols // 2], rows * cols // 2, b'\x11'),
                (prefix + '_scale', 'F8_E4M3', [rows, cols // 16], rows * cols // 16, b'\x38'),
                (prefix + '_scale_2', 'F32', [], 4, struct.pack('<f', 1.0)),
            ):
                header[name] = dict(dtype=dtype, shape=shape, data_offsets=[offset, offset + size])
                index[name] = 'part.safetensors'; payloads.append((size, value)); offset += size
    raw = json.dumps(header, separators=(',', ':')).encode()
    with (root / 'part.safetensors').open('wb') as stream:
        stream.write(struct.pack('<Q', len(raw))); stream.write(raw)
        for size, value in payloads:
            chunk = value * ((1 << 20) // len(value))
            while size:
                n = min(size, len(chunk)); stream.write(chunk[:n]); size -= n
    (root / 'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=index)))
    (root / 'config.json').write_text(json.dumps(dict(text_config=dict(swiglu_limit=10), quantization_config=dict(quantized_layers={}))))
    (root / 'partition.json').write_text(json.dumps(dict(schema='strata.glm.fp4.partition.v1', owner='worker',
        remote_experts_per_layer=1, payload_bytes=offset, reserve_bytes=6 << 30, layout_sha256='a' * 64, index_sha256='b' * 64, shards={})))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='fp4-worker-loopback-') as temporary:
        root = Path(temporary); fixture(root)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0)); address = f'127.0.0.1:{listener.getsockname()[1]}'
        env = dict(os.environ, UCOMM_BACKEND='udp', UCOMM_TIMEOUT_MS='15000', STRATA_FORCE_ISA='avx2')
        worker = subprocess.Popen([str(args.build.resolve() / 'strata-glm-fp4-cpu-worker'), '--model=' + str(root),
                                   '--root=' + address, '--threads=3', '--sessions=2'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            for session in range(2):
                client = subprocess.run([str(args.build.resolve() / 'glm_fp4_worker_client_test'), address], env=env,
                                        capture_output=True, text=True, timeout=60)
                if client.returncode: raise AssertionError(client.stderr)
            stdout, stderr = worker.communicate(timeout=20)
            if client.returncode or worker.returncode:
                raise AssertionError(f'client={client.returncode} worker={worker.returncode}\n{client.stderr}\n{stderr}')
            print(client.stdout.strip() + '; two consecutive sessions')
        finally:
            if worker.poll() is None:
                worker.kill(); worker.wait()


if __name__ == '__main__': main()
