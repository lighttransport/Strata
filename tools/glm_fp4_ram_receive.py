#!/usr/bin/env python3
"""Receive a compact partition into sealed RAM, then run one explicitly supplied command."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import shutil
from glm_fp4_partition import (available_memory, create_memfd, F_ADD_SEALS,
                               F_SEAL_SEAL, F_SEAL_SHRINK, F_SEAL_GROW, F_SEAL_WRITE)
from glm_fp4_layout_rules import rules


def receive(stream, root):
    if root.exists():
        raise FileExistsError('RAM receiver destination already exists')
    root.mkdir(mode=0o700, parents=True)
    descriptors, seen, report = {}, set(), None
    weight_bytes = metadata_bytes = 0
    try:
        with tarfile.open(fileobj=stream, mode='r|') as archive:
            for member in archive:
                name = member.name
                if not member.isfile() or Path(name).name != name or name in seen:
                    raise ValueError('invalid or repeated archive member')
                seen.add(name)
                source = archive.extractfile(member)
                if report is None:
                    if name != 'partition.json' or member.size > (1 << 20):
                        raise ValueError('partition report must arrive first')
                    report = json.loads(source.read())
                    if report.get('schema') != 'strata.glm.fp4.partition.v1' or report.get('owner') != 'decoder':
                        raise ValueError('expected decoder partition')
                    # No weights are accepted before current RAM admission succeeds.
                    for key in ('payload_bytes', 'reserve_bytes'):
                        if type(report[key]) is not int or not 0 <= report[key] <= (1 << 40):
                            raise ValueError('invalid memory budget')
                    if not rules().strata_glm_fp4_layout_admit(report.get('resident_bytes', report['payload_bytes']), report['reserve_bytes'], available_memory()):
                        raise MemoryError('decoder partition plus reserve exceeds currently available RAM')
                    gpu_bytes = report['payload_bytes'] - report.get('resident_bytes', report['payload_bytes'])
                    if gpu_bytes < 0 or shutil.disk_usage(root).free < gpu_bytes + (256 << 20):
                        raise OSError('insufficient disk for GPU-fixed sources and partition metadata')
                    (root / name).write_text(json.dumps(report))
                    continue
                if name in report['shards']:
                    weight_bytes += member.size
                    if weight_bytes > report['payload_bytes'] + (64 << 20):
                        raise ValueError('oversized compact shard')
                    gpu_file = name in report.get('gpu_fixed_shards', [])
                    if gpu_file and subprocess.check_output(['stat', '-f', '-c', '%T', str(root)], text=True).strip() == 'tmpfs':
                        raise ValueError('GPU-fixed backing must be on disk; choose a disk-backed remote_parent')
                    fd = os.open(root / name, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600) if gpu_file else create_memfd('glm-fp4-received')
                    descriptors[name] = fd
                    digest = hashlib.sha256()
                    left = member.size
                    while left:
                        chunk = source.read(min(left, 4 << 20))
                        if not chunk:
                            raise ValueError('truncated weight stream')
                        digest.update(chunk); left -= len(chunk)
                        view = memoryview(chunk)
                        while view:
                            count = os.write(fd, view)
                            if count <= 0:
                                raise OSError('short RAM write')
                            view = view[count:]
                    if digest.hexdigest() != report['shards'][name]:
                        raise ValueError('received shard checksum mismatch')
                    if not gpu_file:
                        fcntl.fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL)
                    else:
                        os.fsync(fd)
                    # Read back the sealed copy without a second weight allocation.
                    check = hashlib.sha256()
                    for offset in range(0, member.size, 4 << 20):
                        check.update(os.pread(fd, min(4 << 20, member.size - offset), offset))
                    if check.digest() != digest.digest():
                        raise ValueError('sealed RAM readback checksum mismatch')
                    if not gpu_file:
                        (root / name).symlink_to(f'/proc/{os.getpid()}/fd/{fd}')
                    else:
                        os.posix_fadvise(fd, 0, member.size, os.POSIX_FADV_DONTNEED)
                else:
                    allowed = {'config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json',
                               'chat_template.jinja', 'generation_config.json', 'glm-fp4-layout.json', *report['remote_catalogs']}
                    metadata_bytes += member.size
                    if name not in allowed or member.size > (64 << 20) or metadata_bytes > (256 << 20):
                        raise ValueError('unexpected or oversized partition metadata')
                    with (root / name).open('xb') as target:
                        shutil.copyfileobj(source, target, 1 << 20)
        if report is None or set(descriptors) != set(report['shards']):
            raise ValueError('incomplete compact partition')
        required = {'config.json', 'model.safetensors.index.json', 'glm-fp4-layout.json', *report['remote_catalogs']}
        if not required.issubset(seen):
            raise ValueError('missing partition metadata')
        return descriptors
    except BaseException:
        for fd in descriptors.values(): os.close(fd)
        shutil.rmtree(root)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--wait-seconds', type=int, default=900)
    parser.add_argument('--run-seconds', type=int, default=3600)
    args = parser.parse_args()
    descriptors, child = {}, None
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(sig, interrupted)
    try:
        descriptors = receive(sys.stdin.buffer, args.root)
        print('FP4_RAM_READY', flush=True, file=sys.stderr)
        deadline = time.monotonic() + args.wait_seconds
        while not (args.root / 'run.json').exists():
            if (args.root / 'cancel').exists() or time.monotonic() >= deadline:
                raise TimeoutError('decoder command did not arrive')
            time.sleep(.2)
        request = json.loads((args.root / 'run.json').read_text())
        requests = request.get('runs', [request])
        if not isinstance(requests, list) or not 1 <= len(requests) <= 16:
            raise ValueError('expected 1..16 decoder runs')
        for index, request in enumerate(requests):
            print(f'FP4_RUN index={index} begin', file=sys.stderr, flush=True)
            argv = request['argv']
            if not argv or not all(isinstance(arg, str) for arg in argv):
                raise ValueError('invalid decoder command')
            environment = {key: value for key, value in os.environ.items() if not key.startswith(('STRATA_', 'UCOMM_'))}
            environment['STRATA_GLM_FP4_RAM_FDS'] = '1'
            for key, value in request.get('environment', {}).items():
                if not key.startswith(('STRATA_', 'UCOMM_')) and key != 'CUDA_VISIBLE_DEVICES':
                    raise ValueError('unsupported runtime environment setting')
                environment[key] = str(value)
            argv = [arg.replace('{model}', str(args.root)).replace('{layout}', str(args.root / 'glm-fp4-layout.json')) for arg in argv]
            child = subprocess.Popen(argv, env=environment, start_new_session=True)
            deadline = time.monotonic() + args.run_seconds
            while child.poll() is None:
                if (args.root / 'cancel').exists() or time.monotonic() >= deadline:
                    raise TimeoutError('decoder run canceled or timed out')
                time.sleep(.2)
            if child.returncode: return child.returncode
            print(f'FP4_RUN index={index} end', file=sys.stderr, flush=True)
        return 0
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        for fd in descriptors.values(): os.close(fd)
        if descriptors and args.root.exists(): shutil.rmtree(args.root)


if __name__ == '__main__':
    raise SystemExit(main())
