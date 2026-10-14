#!/usr/bin/env python3
"""Run the TR16 redesign matrix against separately frozen CUDA/HIP binaries.

Each binary root contains cuda/ and hip/ decoder and worker executables. No
builds, model copies or configuration changes are made by this driver. Existing
complete batches may be resumed; partial batches are retained for inspection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def completed(path, expected):
    result = path/'results.json'
    if not result.exists():
        return False
    data = json.loads(result.read_text())
    rows = data.get('results', [])
    return len(rows) == expected and all(r.get('returncode') == 0 for r in rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'after', 'reap224-config', 'full-config', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--python', default=sys.executable, help='Python environment with runner dependencies')
    p.add_argument('--idle-seconds', type=int, default=3600)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    a.output.mkdir(parents=True, exist_ok=True)
    prompts = root/'docs/fixtures/glm_tr16_20261009/prompts'
    cases = ['cuda', 'hip', 'hip-cuda']
    configs = {'reap224': a.reap224_config.resolve(), 'full': a.full_config.resolve()}
    matrix = [(s, n) for s in ('8k', '2k') for n in ('train-code', 'holdout-chat')]
    batches = [('before-reap224-trials', 'reap224', a.before, [('8k', 'train-code')], False)]
    batches += [('after-'+pack, pack, a.after, matrix, False) for pack in configs]
    batches += [('trace-'+pack, pack, a.after, [('8k', 'train-code')], True) for pack in configs]
    for tag, pack, build, workload, diagnostic in batches:
        out = a.output.resolve()/tag
        receipt = a.output/(tag+'.inputs.json')
        inputs = dict(config_sha256=hashlib.sha256(configs[pack].read_bytes()).hexdigest(),
                      executables={f'{backend}/{name}': hashlib.sha256((build/backend/name).read_bytes()).hexdigest()
                                   for backend in ('cuda', 'hip')
                                   for name in ('strata-glm-decode', 'strata-glm-gpu-worker')},
                      prompts={f'{s}/{n}': hashlib.sha256((prompts/s/(n+'.ids')).read_bytes()).hexdigest()
                               for s, n in workload}, diagnostic=diagnostic)
        if out.exists():
            if a.resume and receipt.exists() and json.loads(receipt.read_text()) == inputs and completed(out, len(workload)*len(cases)):
                print('RETAIN', tag, flush=True)
                continue
            raise SystemExit(f'{out}: partial/existing batch; use a new output root or inspect and preserve it before retrying')
        receipt.write_text(json.dumps(inputs, indent=2)+'\n')
        command = [a.python, str(root/'tools/glm_mixed_bench.py'), '--config', str(configs[pack]),
                   '--cuda-build', str(build.resolve()/'cuda'), '--hip-build', str(build.resolve()/'hip'),
                   '--output', str(out), '--allow-desktop-gpu', '--wait-idle-seconds', str(a.idle_seconds),
                   '--tokens', '128' if diagnostic else '512', '--trials', '1' if diagnostic else '3',
                   '--repetitions', '1', '--combine', 'device', '--fractions', '0.5',
                   '--capture-prefill-logits', '--cases', *cases, '--prompts',
                   *[str(prompts/s/(n+'.ids')) for s, n in workload]]
        if diagnostic:
            command.append('--capture-execution-trace')
        print('START', tag, flush=True)
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = '/opt/rocm/core-10.1/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        with (a.output/(tag+'.driver.log')).open('w') as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        print('DONE', tag, result.returncode, flush=True)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
