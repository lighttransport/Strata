"""Stage 2b/2c: for chosen Stage 2 case names, run train-prompt 512x3, then held-out 2K and 8K 512x3.

  python finalize.py --cases x4-aff016-reuse x1-aff010-reuse [--skip-train] [--only-8k NAME ...]

Held-out control: the archived soft4 runs (same binary/config) measured 31.2495 (2K) and 34.3397 (8K) medians.
Held-out prompts are used only here. Nothing here qualifies quality.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
ROOT = Path('/mnt/nvme02/work/Strata'); R = ROOT / 'build-tr16-35-20261010'; OLD = ROOT / 'build-tr16-goal-20261010'; PY = str(ROOT / 'build-exl3-venv/bin/python')
ap = argparse.ArgumentParser(); ap.add_argument('--cases', nargs='+', required=True); ap.add_argument('--skip-train', action='store_true')
ap.add_argument('--base', default='base-soft4.json'); ap.add_argument('--skip-2k', action='store_true'); ap.add_argument('--skip-8k', action='store_true'); ap.add_argument('--tag', default='')
args = ap.parse_args()
s2 = {c['name']: c for c in json.loads((R / 'cases/s2.json').read_text())}
extra = {c['name']: c for p in R.glob('cases/s*.json') for c in json.loads(p.read_text())}
chosen = [dict(extra[n], control=False) for n in args.cases]
(R / f'cases/final{args.tag}.json').write_text(json.dumps(chosen, indent=1) + '\n')
HOLD_CONTROL = dict(k2=31.2495, k8=34.3397)
def run(name, prompt, trials, rate, out_log, control_stdout):
    cmd = [PY, '-u', str(R / 'screen.py'), '--name', name, '--base', str(R / args.base), '--cases', str(R / f'cases/final{args.tag}.json'),
           '--tokens', '512', '--trials', str(trials), '--prompt', prompt]
    if rate: cmd += ['--control-rate', str(rate), '--control-stdout', str(control_stdout)]
    print('RUN', name, time.strftime('%H:%M:%S'), flush=True)
    with open(out_log, 'a') as f: code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    print('EXIT', code, name, flush=True)
log = R / 'stage2b.out'
x0 = json.loads((R / 's2/summary.json').read_text())[0]
if not args.skip_train: run('s2b-train512x3' + args.tag, 'build-tr16-20261009/prompts2k/train-code.ids', 3, x0['decode_tok_s'], log, R / 's2/x0-control/000-baseline.stdout')
if not args.skip_2k: run('s2c-holdout2k' + args.tag, 'build-tr16-20261009/prompts2k/holdout-code.ids', 3, HOLD_CONTROL['k2'], log, OLD / 'long-finalists/q22-soft4-diagnostic/2k/000-baseline.stdout')
if not args.skip_8k: run('s2c-holdout8k' + args.tag, 'build-tr16-20261009/prompts8k/holdout-code.ids', 3, HOLD_CONTROL['k8'], log, OLD / 'long-finalists/q22-soft4-diagnostic/8k/000-baseline.stdout')
print('STAGE2B_DONE', flush=True)
