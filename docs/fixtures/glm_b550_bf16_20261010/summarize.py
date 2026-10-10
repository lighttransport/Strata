"""Keep completed quality, incomplete probes and clean timing separate."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
OLD = ROOT.parent / 'glm_b550_prefill_20261010'


def quality(name, fresh=False):
    folder = ROOT / name
    path = folder / 'measurement.json'
    if not path.exists():
        return {'requested_tasks_complete': False, 'record': str(path.relative_to(REPO))}
    data = json.loads(path.read_text())
    rows = data['results']
    malformed = []
    for row in rows:
        source = folder / (row['task_id'].replace('/', '_') + '.py')
        try:
            ast.parse(source.read_text())
        except SyntaxError:
            malformed.append(row['task_id'])
    memory = ROOT / (name + '.memory.json')
    guard = json.loads(memory.read_text()) if memory.exists() else {}
    return {'record': str(path.relative_to(REPO)), 'tasks': len(rows),
            'passed': sum(row['passed'] for row in rows), 'malformed': malformed,
            'task_ids': [row['task_id'] for row in rows],
            'requested_tasks_complete': data.get('requested_tasks_complete', False),
            'cap_hits': sum(row['generated_tokens'] == 1024 for row in rows),
            'fresh_process': fresh, 'guard_exit_code': guard.get('exit_code'),
            'guard_rejected': guard.get('rejected'), 'peak_swap_kib': guard.get('peak_swap_kib'),
            'completed_matched_protocol': not fresh and data.get('requested_tasks_complete', False)
                and guard.get('exit_code') == 0 and guard.get('rejected') is None}


def main():
    perf = {}
    for path in sorted(ROOT.glob('*.result.json')):
        d = json.loads(path.read_text())
        if 'measurements' not in d:
            continue
        perf[path.stem] = {'record': path.name, 'complete': d['complete'], 'clean': d['clean'],
                          'rejected': d.get('rejected'), 'prefill': d['measurements'].get('prefill'),
                          'decode': d['measurements'].get('decode'),
                          'peak_ram_gib': d.get('cgroup_after', {}).get('memory.peak', 0) / 2**30,
                          'minimum_gpu_free_mib': d.get('minimum_gpu_free_mib'),
                          'interference': d.get('interference', [])}
    parity = None
    control = ROOT / 'mmq8k-control.result.json'
    if control.exists() and json.loads(control.read_text())['complete']:
        parity = {'all_128_tokens_equal_previous_default':
                  (ROOT / 'mmq8k-control.stdout').read_text().split() == (OLD / 'base8k-02.stdout').read_text().split(),
                  'reference': str((OLD / 'base8k-02.stdout').relative_to(REPO))}
    summary = {'hardware': {'cpu': 'Ryzen 9 3950X', 'gpu': 'RX 9070 XT gfx1201',
                            'hip_version': '7.14.60850 (local packaging; headers/share/hip/version)',
                            'ddr_mt_s': 2133, 'clock_changes': False, 'ram_limit_gib': 60,
                            'inference_swap_limit_bytes': 0, 'gpu_physical_mib': 16304},
               'microbench': {f'v{i}': f'wmma-v{i}.parity-bench.txt' for i in (1,2,3)},
               'quality_control': str((OLD / 'he-mmq-long-targets/measurement.json').relative_to(REPO)),
               'quality': {name: quality(name, 'fresh' in name) for name in
                           ('he-bf16-blas-long','he-bf16-wmma-long','he-bf16-wmma-fresh-17','he-bf16-wmma-fresh-25')},
               'performance': perf, 'default_parity': parity,
               'final_wrapper_parity': json.loads((ROOT / 'final-wrapper-parity.json').read_text()),
               'incomplete_timing': {'mmq8k-optimized-control': {'reason': 'service/benchmark timeout, changed file-page accounting and low GPU activity', 'progress': 'mmq8k-optimized-control.progress.json', 'launcher': 'mmq8k-optimized-control.runner.txt'}},
               'selected': False,
               'selection_reason': 'Library BF16 introduced long coding failures; WMMA resident run incomplete and fresh probes failed. No BF16 preset or default change.'}
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    sources = ['CMakeLists.txt','src/prefill/wmma_gfx12.cu','src/prefill/gemm.cu',
               'src/kernels/cuda/glm_prefill.cu','src/kernels/cuda/dequant_bf16.cu',
               'src/kernels/cuda/iq_kernels.cu','src/program/glm_decode.cpp','build-hip/strata-glm-decode']
    (ROOT / 'build-manifest.json').write_text(json.dumps({f: hashlib.sha256((REPO / f).read_bytes()).hexdigest()
                                                        for f in sources}, indent=2) + '\n')
    print('Updated', ROOT / 'summary.json')


if __name__ == '__main__':
    main()
