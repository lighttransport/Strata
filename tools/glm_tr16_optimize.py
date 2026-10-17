#!/usr/bin/env python3
"""Screen opt-in TR16 runner changes against immutable CUDA/HIP binaries.

No build, model download, default change or quality qualification is performed.
Use a new output root for every run; raw results and failed attempts are retained.
"""
import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
KNOBS = ('STRATA_GLM_MIXED_REGISTER', 'STRATA_GLM_MIXED_EXCHANGE', 'STRATA_GLM_MIXED_PROTOCOL',
         'STRATA_HIPBLASLT_TUNING', 'STRATA_GLM_MTP_GPU_CACHE', 'STRATA_GLM_STEP_TRACE')
VARIANTS = {
    'parent': {},
    'register': {'STRATA_GLM_MIXED_REGISTER': '1'},
    'exchange': {'STRATA_GLM_MIXED_REGISTER': '1', 'STRATA_GLM_MIXED_PROTOCOL': '5',
                 'STRATA_GLM_MIXED_EXCHANGE': 'f16'},
    'gpu-draft': {'STRATA_GLM_MTP_GPU_CACHE': '1'},
    'flow': {'STRATA_NATIVE_FLOW_SCHEDULE': 'span'},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def no_swap_scope(cgroup_text=None, cgroup_root=Path('/sys/fs/cgroup')):
    """Record the cgroup-v2 constraint for this controller and its children."""
    cgroup_text = Path('/proc/self/cgroup').read_text() if cgroup_text is None else cgroup_text
    root = cgroup_root.resolve()
    for line in cgroup_text.splitlines():
        if line.startswith('0::'):
            relative = line.split('::', 1)[1].lstrip('/')
            path = (root/relative).resolve()
            if not path.is_relative_to(root): continue
            while True:
                setting = path/'memory.swap.max'
                if setting.exists() and setting.read_text().strip() == '0':
                    return str(path)
                if path == root: break
                path = path.parent
    return None


def profile(base, parent, variant, case, table=None):
    c = copy.deepcopy(base)
    env = c.setdefault('env', {})
    for name in KNOBS:
        env.pop(name, None)
    # BATCHED_RESIDENT uses presence, so setting it to "0" would enable it.
    env.pop('STRATA_GLM_BATCHED_RESIDENT', None)
    env.pop('STRATA_GLM_MLA_GRAPHS', None)
    if parent != 'base':
        env['STRATA_GLM_BATCHED_RESIDENT'] = '1'
    if parent == 'batched-mla':
        env['STRATA_GLM_MLA_GRAPHS'] = '1'
    env.update(VARIANTS[variant])
    c['mtp_experts'] = 'gpu' if variant == 'gpu-draft' else 'cpu'
    if table is not None:
        env['STRATA_HIPBLASLT_TUNING'] = str(table)
    if case == 'cuda-hip':
        c['decode_cache_mib'] = min(c['decode_cache_mib'], 10240)
    c['model_name'] = 'tr16-' + parent + '-' + variant + '-' + case
    c['experimental_role'] = 'Speed screen only; full corpus and HumanEval qualification required.'
    return c


def results_summary(path, trials, tokens=512):
    data = json.loads(path.read_text())
    rows = data.get('results', [])
    eligible = bool(rows) and all(r.get('returncode') == 0 and
        r.get('measurement', {}).get('complete') and r['measurement'].get('clean') for r in rows)
    summary = data.get('summary', [])
    lengths = {r['prompt']: len([t for t in re.split(r'[,\s]+',Path(r['prompt']).read_text().strip()) if t]) for r in summary}
    # Each prompt and fraction remains a separate result; never pool code/chat.
    gates = [dict(prompt=r['prompt'], fraction=r['fraction'],
                  decode_tok_s=r['median_decode_tok_s'], prefill_tok_s=r['median_prefill_tok_s'],
                  speed_gate=eligible and trials >= 3 and tokens >= 512 and lengths[r['prompt']] == 8192 and
                             r['median_decode_tok_s'] >= 50 and r['median_prefill_tok_s'] >= 900,
                  quality_qualified=False) for r in summary]
    return dict(eligible=eligible, rows=gates, quality_qualified=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--model', type=Path)
    ap.add_argument('--prior', type=Path)
    ap.add_argument('--cuda-build', type=Path, default=ROOT/'build-glm-release')
    ap.add_argument('--hip-build', type=Path, default=ROOT/'build-hip-rocm101-tr16')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--parent', choices=('base', 'batched', 'batched-mla'), default='batched')
    ap.add_argument('--variants', nargs='+', choices=tuple(VARIANTS), default=['resident', 'register', 'exchange', 'resident-exchange', 'parent'])
    ap.add_argument('--cases', nargs='+', choices=('hip-cuda', 'cuda-hip'), default=['hip-cuda', 'cuda-hip'])
    ap.add_argument('--prompts', type=Path, nargs='+', default=[ROOT/'docs/fixtures/glm_tr16_20261009/prompts/8k/holdout-code.ids',
                                                             ROOT/'docs/fixtures/glm_tr16_20261009/prompts/8k/holdout-chat.ids'])
    ap.add_argument('--fractions', nargs='+', default=['0.5'])
    ap.add_argument('--hipblaslt-table', type=Path)
    ap.add_argument('--trials', type=int, choices=range(1, 10), default=3)
    ap.add_argument('--tokens', type=int, default=512)
    ap.add_argument('--idle-seconds', type=int, default=120)
    ap.add_argument('--allow-desktop-gpu', action='store_true')
    a = ap.parse_args()
    if a.output.exists() or a.tokens < 1:
        ap.error('use a new output root and a positive token count')
    if any(v in ('hiplt', 'joint') for v in a.variants) and not a.hipblaslt_table:
        ap.error('hiplt and joint require --hipblaslt-table')
    scope = no_swap_scope()
    if scope is None:
        ap.error('run in a task-only cgroup with MemorySwapMax=0 (systemd-run --user --scope -p MemorySwapMax=0)')
    a.output.mkdir(parents=True)
    output = a.output.resolve()
    base = json.loads(a.config.read_text())
    if a.model: base['model'] = str(a.model.resolve())
    if a.prior: base.setdefault('env', {})['STRATA_GLM_EXPERT_PRIOR'] = str(a.prior.resolve())
    frozen = output/'frozen'
    identities = {}
    for backend, build in (('cuda', a.cuda_build), ('hip', a.hip_build)):
        (frozen/backend).mkdir(parents=True)
        for name in ('strata-glm-decode', 'strata-glm-gpu-worker'):
            source = build/name; before = sha(source); dest = frozen/backend/name
            shutil.copy2(source, dest)
            if sha(source) != before or sha(dest) != before:
                raise RuntimeError('binary changed while freezing: ' + str(source))
            identities[backend+'/'+name] = before
    table = None
    if a.hipblaslt_table:
        table = frozen/'hipblaslt.txt'; shutil.copy2(a.hipblaslt_table, table)
        identities['hipblaslt_table'] = sha(table)
    (frozen/'manifest.json').write_text(json.dumps(dict(binaries=identities, source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), config_sha256=sha(a.config),
        no_swap_cgroup=scope,
        source_diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff'], cwd=ROOT)).hexdigest()), indent=2)+'\n')
    env = {k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}
    records = []; failed = False
    for variant in a.variants:
        for case in a.cases:
            tag = variant+'-'+case
            config = output/(tag+'.json'); config.write_text(json.dumps(profile(base,a.parent,variant,case,table),indent=2)+'\n')
            out = output/('out-'+tag)
            command = [a.python, str(ROOT/'tools/glm_mixed_bench.py'), '--config', str(config),
                       '--cuda-build', str(frozen/'cuda'), '--hip-build', str(frozen/'hip'), '--combine', 'device',
                       '--cases', case, '--fractions', *a.fractions, '--tokens', str(a.tokens), '--trials', str(a.trials),
                       '--repetitions', '1', '--wait-idle-seconds', str(a.idle_seconds), '--capture-prefill-logits',
                       '--prompts', *map(str,a.prompts), '--output', str(out)]
            if a.allow_desktop_gpu: command.append('--allow-desktop-gpu')
            print('START', tag, flush=True)
            with (output/(tag+'.driver.log')).open('w') as log:
                r = subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            item = dict(variant=variant,case=case,returncode=r.returncode)
            if (out/'results.json').exists(): item.update(results_summary(out/'results.json',a.trials,a.tokens))
            failed |= r.returncode != 0 or not item.get('eligible',False)
            records.append(item)
            (output/'summary.json').write_text(json.dumps(dict(records=records,quality_qualified=False),indent=2)+'\n')
            print('DONE',json.dumps(item),flush=True)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
