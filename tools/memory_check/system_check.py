#!/usr/bin/env python3
"""Standalone Linux x86-64 CPU/RAM/PCIe/GPU performance pre-check; no model needed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from memory_check import available_mib

FAULT = re.compile(r'device lost from bus|GPU reset|ring .*timeout|\bVM\b.*\bfault\b|'
                   r'page fault|AER:.*(?:error|fatal)|device inaccessible', re.I)


def gpu_pci_devices():
    devices = []
    for path in sorted(Path('/sys/bus/pci/devices').glob('*')):
        if (path / 'vendor').read_text().strip() in ('0x1002', '0x10de') and \
                (path / 'class').read_text().strip().startswith('0x03'):
            devices.append(path)
    return devices


def pci_links(devices):
    links = {}
    for device in devices:
        for path in (device.resolve(), *device.resolve().parents):
            if (path/'current_link_speed').exists():
                links[path.name] = {
                    'speed': (path/'current_link_speed').read_text().strip(),
                    'width': (path/'current_link_width').read_text().strip()}
    return links


def pci_identity(path):
    with (path / 'config').open('rb') as stream:
        identity = stream.read(4)
    if len(identity) != 4 or identity[:2] in (b'\xff\xff', b'\x00\x00'):
        raise RuntimeError(f'GPU PCI identity unavailable: {path.name}')
    return identity.hex()


def kernel_log():
    run = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=5)
    return run.stdout if run.returncode == 0 else None


def run_gpu(command, output, timeout, devices):
    """Monitor only this child; never reset the GPU or stop other workloads."""
    identities = {p.name: pci_identity(p) for p in devices}
    before = kernel_log()
    if before is not None and FAULT.search(before):
        raise RuntimeError('GPU fault already logged in this boot; refusing GPU launch')
    started = time.monotonic()
    reason = None
    log = output / 'gpu.jsonl'
    with log.open('w') as stdout, (output / 'gpu.stderr.log').open('w') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            while process.poll() is None:
                current = kernel_log() if before is not None else None
                delta = current[len(before):] if current is not None and current.startswith(before) else current
                if delta and FAULT.search(delta):
                    reason = 'kernel GPU/PCI fault'
                    break
                if any(pci_identity(p) != identities[p.name] for p in devices):
                    reason = 'GPU PCI identity changed'
                    break
                if time.monotonic() - started > timeout:
                    reason = 'GPU timeout'
                    break
                time.sleep(0.25)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    reason = reason or 'GPU process did not exit after termination'
    after = kernel_log() if before is not None else None
    delta = after[len(before):] if after is not None and after.startswith(before) else after
    (output / 'gpu.kernel.log').write_text(delta or '')
    if delta and FAULT.search(delta):
        reason = reason or 'kernel GPU/PCI fault'
    for path in devices:
        if pci_identity(path) != identities[path.name]:
            reason = reason or 'GPU PCI identity changed'
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    return {'passed': process.returncode == 0 and reason is None and bool(rows) and
            rows[-1].get('passed') is True, 'returncode': process.returncode,
            'termination_reason': reason, 'kernel_log_checked': before is not None,
            'pci_before': identities, 'pci_links_after': pci_links(devices), 'measurements': rows,
            'wall_seconds': time.monotonic() - started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', nargs=argparse.REMAINDER, help='Launch an app only after requested checks pass')
    parser.add_argument('--backend', choices=['none', 'hip', 'cuda'], default='none',
                        help='GPU checks are opt-in; none runs CPU/RAM only')
    parser.add_argument('--output', type=Path, default=Path('system-check-results'))
    parser.add_argument('--disk-file', type=Path, help='Existing regular file for opt-in direct-I/O reads')
    parser.add_argument('--disk-mib', type=int, default=1024)
    parser.add_argument('--disk-repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, nargs='+')
    parser.add_argument('--array-mib', type=int, default=1024)
    parser.add_argument('--pages', choices=['4k', 'thp'], default='4k')
    parser.add_argument('--gpu-buffer-mib', type=int, default=256)
    parser.add_argument('--gemm-n', type=int, default=2048)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--ram-mt-s', type=int, help='User-supplied configured RAM speed, recorded only')
    parser.add_argument('--device', type=int, default=0, help='CUDA/HIP runtime ordinal')
    parser.add_argument('--timeout', type=int, default=120, help='GPU execution time limit in seconds')
    args = parser.parse_args(argv)
    if args.run is not None and not args.run:
        parser.error('--run requires a command')
    if args.ram_mt_s is not None and args.ram_mt_s <= 0:
        parser.error('--ram-mt-s must be positive')
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else []
    threads = args.threads or sorted(set([1, min(4, len(cpus)), min(16, len(cpus))]))
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        parser.error('Supports Linux x86-64 only')
    if any(t < 1 or t > len(cpus) for t in threads) or args.array_mib < 16:
        parser.error('Invalid thread count or array size (minimum 16 MiB)')
    if not 16 <= args.gpu_buffer_mib <= 1024 or not 256 <= args.gemm_n <= 4096 or args.gemm_n % 256:
        parser.error('GPU buffer 16..1024 MiB; GEMM N 256..4096 and multiple of 256')
    if not 3 <= args.repeats <= 31 or not 1 <= args.timeout <= 600 or args.device < 0:
        parser.error('Repeats 3..31; timeout 1..600 seconds; nonnegative device')
    if not 1 <= args.disk_mib <= 65536 or not 2 <= args.disk_repeats <= 15:
        parser.error('Disk size 1..65536 MiB; repeats 2..15')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    result = {'schema_version': 1, 'passed': False, 'kind': 'performance pre-check',
              'stability_certified': False, 'platform': platform.platform(),
              'cpu_model': next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), 'unknown'),
              'board': Path('/sys/class/dmi/id/board_name').read_text().strip() if Path('/sys/class/dmi/id/board_name').exists() else 'unknown',
              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              'cpu_affinity': cpus, 'settings': vars(args) | {'output': str(args.output), 'disk_file': str(args.disk_file) if args.disk_file else None},
              'source_sha256': {n: hashlib.sha256((source/n).read_bytes()).hexdigest()
                                for n in ['system_check.py', 'memory_check.py', 'bandwidth.cpp',
                                          'cpu_compute.cpp', 'gpu_check.cu', 'disk_read.py']},
              'cpu_compute': [], 'gpu': None, 'disk': None}
    code = 2
    try:
        flags = Path('/proc/cpuinfo').read_text()
        if not re.search(r'\bavx2\b', flags) or not re.search(r'\bfma\b', flags):
            raise RuntimeError('CPU compute checks require AVX2 and FMA')
        env = os.environ.copy()
        env.update(OMP_PROC_BIND='close', OMP_PLACES=','.join('{%d}' % c for c in cpus),
                   OMP_WAIT_POLICY='PASSIVE')
        print('Measuring CPU RAM bandwidth', file=sys.stderr, flush=True)
        ram = subprocess.run([sys.executable, str(source/'memory_check.py'), '--mode', 'bandwidth',
                              '--output', str(args.output/'ram'), '--threads', *map(str, threads),
                              '--array-mib', str(args.array_mib), '--pages', args.pages],
                             capture_output=True, text=True, timeout=1200)
        (args.output/'ram.stderr.log').write_text(ram.stderr)
        result['ram'] = json.loads((args.output/'ram/result.json').read_text())
        if ram.returncode or not result['ram']['passed']:
            raise RuntimeError('CPU bandwidth check failed; see ram/result.json')
        with tempfile.TemporaryDirectory(prefix='system-check-') as directory:
            binary = str(Path(directory)/'cpu-compute')
            command = ['g++', '-O3', '-mavx2', '-mfma', '-std=c++20', '-fopenmp',
                       str(source/'cpu_compute.cpp'), '-o', binary]
            subprocess.run(command, check=True, timeout=120)
            result['cpu_compiler'] = command
            for count in threads:
                print(f'Measuring CPU compute: {count} threads', file=sys.stderr, flush=True)
                run = subprocess.run([binary, str(count)], env=env, capture_output=True,
                                     text=True, check=True, timeout=120)
                rows = [json.loads(line) for line in run.stdout.splitlines()]
                if len(rows) != 2 or not all(row.get('verified') for row in rows):
                    raise RuntimeError('CPU compute correctness check failed')
                result['cpu_compute'].extend(rows)
            if args.backend != 'none':
                if 2*args.gpu_buffer_mib + 3*args.gemm_n**2*4//1048576 + 256 > available_mib():
                    raise RuntimeError('Insufficient available host RAM for GPU buffers with reserve')
                devices = gpu_pci_devices()
                if not devices:
                    raise RuntimeError('No enumerated AMD/NVIDIA display GPU')
                compiler = shutil.which('hipcc' if args.backend == 'hip' else 'nvcc')
                if not compiler:
                    raise RuntimeError(f'{args.backend} compiler missing')
                gpu_binary = str(Path(directory)/'gpu-check')
                command = [compiler, '-O3', '-std=c++17', str(source/'gpu_check.cu'), '-o', gpu_binary]
                command += ['-lhipblas', '-Wl,--disable-new-dtags,-rpath,/opt/rocm/lib'] if args.backend == 'hip' else ['-lcublas']
                print(f'Building {args.backend} GPU check', file=sys.stderr, flush=True)
                build = subprocess.run(command, capture_output=True, text=True, timeout=300)
                (args.output/'gpu.build.log').write_text(build.stdout+build.stderr)
                if build.returncode:
                    raise RuntimeError('GPU compilation failed; see gpu.build.log')
                result['gpu_compiler'] = command
                print('Measuring PCIe, GPU bandwidth and FP32 GEMM', file=sys.stderr, flush=True)
                result['gpu'] = run_gpu([gpu_binary, str(args.device), str(args.gpu_buffer_mib),
                                         str(args.gemm_n), str(args.repeats)], args.output, args.timeout, devices)
                if not result['gpu']['passed']:
                    raise RuntimeError('GPU check failed; see gpu logs')
        if args.disk_file:
            print('Measuring sequential direct disk reads', file=sys.stderr, flush=True)
            disk = subprocess.run([sys.executable, str(source/'disk_read.py'), str(args.disk_file),
                                   '--mib', str(args.disk_mib), '--repeats', str(args.disk_repeats),
                                   '--output', str(args.output/'disk.json')],
                                  capture_output=True, text=True, timeout=130)
            (args.output/'disk.stderr.log').write_text(disk.stderr)
            result['disk'] = json.loads((args.output/'disk.json').read_text())
            if disk.returncode or not result['disk']['passed']:
                raise RuntimeError('Disk check failed; see disk.json')
        result['passed'] = True
        code = 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        result['error'] = str(error)
    except KeyboardInterrupt:
        result['error'] = 'Interrupted'
        code = 130
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    if code == 0 and args.run:
        return subprocess.call(args.run)
    return code


if __name__ == '__main__':
    sys.exit(main())
