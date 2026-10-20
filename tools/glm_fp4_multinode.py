#!/usr/bin/env python3
"""Run the bounded FP4 MoE test on a HIP primary and an existing SSH CUDA worker."""
import argparse
import pathlib
import shlex
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--ssh', required=True, help='SSH alias from local configuration')
p.add_argument('--root', required=True, help='Primary private IB address:port')
p.add_argument('--model', required=True, type=pathlib.Path)
p.add_argument('--output', required=True, type=pathlib.Path)
p.add_argument('--primary-binary', default='build-fp4-local/glm-fp4-moe-node-hip')
p.add_argument('--worker-binary', required=True, help='Already-installed worker executable')
p.add_argument('--tiled', action='store_true')
p.add_argument('--timeout', default=300, type=int)
a = p.parse_args()
if not 30 <= a.timeout <= 1800 or a.ssh.startswith('-') or a.worker_binary.startswith('-'):
    p.error('Invalid timeout, SSH alias or worker path')
if not (a.model / 'model.safetensors.index.json').is_file():
    p.error('Missing partial checkpoint index')
a.output.mkdir(parents=True, exist_ok=True)
# Never overwrite frozen evidence.
if any((a.output / n).exists() for n in ('primary.log', 'worker.log')):
    p.error('Output logs already exist; use a new output directory')
local = [a.primary_binary, '0', a.root, str(a.model)] + (['tiled'] if a.tiled else [])
remote = ['timeout', '--signal=TERM', '--kill-after=5s', str(a.timeout)+'s', a.worker_binary, '1', a.root, 'unused']
processes = []
try:
    with (a.output / 'primary.log').open('w') as primary, (a.output / 'worker.log').open('w') as worker:
        processes.append(subprocess.Popen(local, stdout=primary, stderr=subprocess.STDOUT))
        processes.append(subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', a.ssh, shlex.join(remote)], stdout=worker, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + a.timeout
        while any(proc.poll() is None for proc in processes):
            if any(proc.poll() not in (None, 0) for proc in processes):
                raise RuntimeError('A node failed; inspect output logs')
            if time.monotonic() >= deadline:
                raise TimeoutError('Multi-node test deadline exceeded')
            time.sleep(0.25)
        if any(proc.returncode != 0 for proc in processes):
            raise RuntimeError('A node failed; inspect output logs')
    for name in ('primary', 'worker'):
        log = (a.output / (name+'.log')).read_text()
        if 'TRANSPORT ib' not in log or 'PASS persistent two-node partial-weight MoE' not in log:
            raise RuntimeError('Missing IB completion evidence: '+name)
    print('PASS both nodes over IB; logs:', a.output)
finally:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    for proc in processes:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
