#!/usr/bin/env python3
"""Standalone Linux x86-64 RAM benchmark and verified stress-test preflight."""
import argparse
import json
import math
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


def cgroup_available_mib(root, relative):
    """Minimum remaining allowance across the visible cgroup v2 hierarchy."""
    remaining = []
    root = root.resolve()
    group = (root / relative.lstrip('/')).resolve()
    if group != root and root not in group.parents:
        raise RuntimeError('Cgroup path escapes the visible hierarchy')
    while True:
        try:
            limit = (group / 'memory.max').read_text().strip()
            if limit != 'max':
                used = int((group / 'memory.current').read_text())
                remaining.append(max(0, (int(limit) - used) // 1048576))
        except FileNotFoundError:
            pass
        if group == root:
            break
        if root not in group.parents:
            raise RuntimeError('Cgroup path escapes the visible hierarchy')
        group = group.parent
    return min(remaining, default=None)


def available_mib():
    info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(info['MemAvailable'].split()[0]) // 1024
    # Honor the process's cgroup v2 memory budget as well as host MemAvailable.
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            remaining = cgroup_available_mib(Path('/sys/fs/cgroup'), line[3:])
            if remaining is not None:
                available = min(available, remaining)
    return available


def stress_status(text, returncode):
    incidents = [int(x) for x in re.findall(r'Found (\d+) hardware incidents', text)]
    bad = bool(re.search(r'Hardware Error:|Report Error:|Status: FAIL|CRC mismatch', text))
    passed = returncode == 0 and 'Status: PASS' in text and not bad and not any(incidents)
    return {'passed': passed, 'hardware_incidents': max(incidents, default=0),
            'returncode': returncode, 'data_error_detected': bad}


def stop(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_stress(executable, mib, seconds, threads, output):
    log = output / 'stress.log'
    command = [executable, '-M', str(mib), '-s', str(seconds), '-m', str(threads),
               '-i', str(min(4, threads)), '-W', '--stop_on_errors', '--printsec', '30']
    print(f'Stress: {mib} MiB, {seconds}s; log: {log}', file=sys.stderr, flush=True)
    started = time.monotonic()
    with log.open('w') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                text = log.read_text(errors='replace')
                if re.search(r'Hardware Error:|Report Error:|Status: FAIL|CRC mismatch', text):
                    stop(process)
                    break
                if time.monotonic() - started > seconds + 120:
                    stop(process)
                    break
                time.sleep(0.5)
        finally:
            stop(process)
    result = stress_status(log.read_text(errors='replace'), process.returncode)
    result.update(memory_mib=mib, requested_seconds=seconds,
                  elapsed_seconds=round(time.monotonic() - started, 2), log=str(log), command=command)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['bandwidth', 'stress', 'check'], default='check')
    parser.add_argument('--output', type=Path, default=Path('memory-check-results'))
    parser.add_argument('--threads', type=int, nargs='+', default=None)
    parser.add_argument('--array-mib', type=int, default=1024, help='Size per array; three arrays allocated')
    parser.add_argument('--pages', choices=['4k', 'thp'], default='4k')
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--memory-mib', type=int, help='Stress allocation; default 80%% of available RAM')
    parser.add_argument('--reserve-mib', type=int, default=4096)
    parser.add_argument('--stressapptest', default='stressapptest', help='Executable name or path')
    parser.add_argument('--min-read-gbps', type=float, help='Optional application-specific bandwidth floor')
    parser.add_argument('--run', nargs=argparse.REMAINDER, help='Start command only after both checks pass')
    args = parser.parse_args(argv)
    if args.run is not None and (not args.run or args.mode != 'check'):
        parser.error('--run requires --mode check and a launch command')
    if args.min_read_gbps is not None and (not math.isfinite(args.min_read_gbps) or args.min_read_gbps <= 0):
        parser.error('--min-read-gbps must be finite and positive')
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        parser.error('This version supports Linux x86-64 only')
    cpus = sorted(os.sched_getaffinity(0))
    threads = args.threads or sorted(set([1, min(4, len(cpus)), min(16, len(cpus))]))
    if any(t < 1 or t > len(cpus) for t in threads):
        parser.error('Thread count exceeds available CPU affinity')
    if args.array_mib < 16 or args.seconds < 1 or args.reserve_mib < 0:
        parser.error('Use array-mib >= 16, seconds >= 1 and reserve-mib >= 0')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'schema_version': 1, 'mode': args.mode, 'passed': False,
              'platform': platform.platform(), 'cpu_affinity': cpus,
              'units': 'decimal GB/s; logical bytes, not memory-controller traffic',
              'ram_speed': 'not probed; verify with sudo dmidecode --type 17',
              'bandwidth': [], 'stress': None}
    code = 2
    try:
        executable = None
        if args.mode in ('stress', 'check'):
            executable = shutil.which(args.stressapptest)
            if not executable:
                raise RuntimeError('stressapptest missing; install it or pass --stressapptest /path/to/binary')
        available = available_mib()
        budget = max(0, available - args.reserve_mib)
        mib = args.memory_mib if args.memory_mib is not None else min(int(available * 0.8), budget)
        if executable and (mib < 256 or mib > budget):
            raise RuntimeError(f'Stress memory must be between 256 and {budget} MiB after reserve')
        if args.mode in ('bandwidth', 'check'):
            if 3 * args.array_mib > budget:
                raise RuntimeError('Bandwidth arrays exceed available RAM after reserve')
            flags = Path('/proc/cpuinfo').read_text()
            if not re.search(r'\bavx2\b', flags):
                raise RuntimeError('Bandwidth benchmark requires AVX2')
            env = os.environ.copy()
            env.update(OMP_PROC_BIND='close', OMP_PLACES=','.join('{%d}' % c for c in cpus),
                       OMP_WAIT_POLICY='PASSIVE')
            with tempfile.TemporaryDirectory(prefix='memory-check-') as directory:
                binary = str(Path(directory) / 'bandwidth')
                compiler = ['g++', '-O3', '-mavx2', '-std=c++20', '-fopenmp',
                            str(Path(__file__).with_name('bandwidth.cpp')), '-o', binary]
                subprocess.run(compiler, check=True, stdout=sys.stderr, stderr=sys.stderr)
                result['compiler_command'] = compiler
                for count in threads:
                    print(f'Bandwidth: {count} threads', file=sys.stderr, flush=True)
                    run = subprocess.run([binary, str(count), args.pages, str(args.array_mib)],
                                         env=env, check=True, capture_output=True, text=True, timeout=300)
                    result['bandwidth'].extend(json.loads(line) for line in run.stdout.splitlines())
            peak = max(row['median_GB_s'] for row in result['bandwidth'] if row['kernel'] == 'read_avx2')
            result['peak_read_gbps'] = peak
            if args.min_read_gbps is not None and peak < args.min_read_gbps:
                result['error'] = 'Read bandwidth below requested floor; stress skipped'
                code = 1
            else:
                code = 0
        else:
            code = 0
        if code == 0 and executable:
            if mib > max(0, available_mib() - args.reserve_mib):
                raise RuntimeError('Available RAM decreased; stress allocation no longer fits after reserve')
            result['stress'] = run_stress(executable, mib, args.seconds, max(threads), args.output)
            code = 0 if result['stress']['passed'] else 1
        result['passed'] = code == 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        result['error'] = str(error)
        code = 2
    except KeyboardInterrupt:
        result['error'] = 'Interrupted'
        code = 130
    payload = json.dumps(result, indent=2)
    (args.output / 'result.json').write_text(payload + '\n')
    print(payload, flush=True)
    if code == 0 and args.run:
        return subprocess.call(args.run)
    return code


if __name__ == '__main__':
    sys.exit(main())
