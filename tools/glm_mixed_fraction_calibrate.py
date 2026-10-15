#!/usr/bin/env python3
"""Find the mixed-runner primary fraction (expert groups on the primary GPU) for a GPU pair.

The fraction only takes effect in whole 16-expert groups, so the search runs over group counts
1..groups-1. It starts from a guess (single-card prefill speeds if given, else half), measures
each candidate with tools/glm_mixed_bench.py in a fresh process, and climbs one group at a time
while the objective improves. Every measurement is kept in the report; nothing is cached.

Example (REAP-224: 14 groups):
  python3 tools/glm_mixed_fraction_calibrate.py --config cfg.json --case hip-cuda --groups 14 \\
      --model M.gguf --prior P.json --prompt 8k.ids --output out/ --objective prefill \\
      --primary-tok-s 611 --worker-tok-s 406
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm_mixed_bench import performance_summary  # noqa: E402


def fraction_for(groups_on_primary, groups):
    """A fraction that the engines round (lround(f*groups)) back to exactly this group count."""
    return round(groups_on_primary / groups, 4)


def initial_groups(groups, primary_tok_s=None, worker_tok_s=None):
    if primary_tok_s and worker_tok_s:
        share = primary_tok_s / (primary_tok_s + worker_tok_s)
    else:
        share = 0.5
    return min(groups - 1, max(1, round(share * groups)))


def climb(evaluate, groups, start, max_steps=None):
    """Hill-climb over group counts; evaluate(g) -> score (higher is better) or None on failure."""
    scores = {}
    def score(g):
        if g not in scores:
            scores[g] = evaluate(g)
        return scores[g]
    best = start
    if score(best) is None:
        raise RuntimeError(f'starting point {start} groups failed')
    steps = 0
    while max_steps is None or steps < max_steps:
        moved = False
        for g in (best + 1, best - 1):
            if 1 <= g <= groups - 1 and (score(g) or 0) > scores[best]:
                best, moved = g, True
                break
        steps += 1
        if not moved:
            break
    return best, scores


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--case', required=True, choices=('hip-cuda', 'cuda-hip'))
    p.add_argument('--groups', type=int, required=True, help='16-expert groups per layer (experts / 16)')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--prompt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--objective', choices=('prefill', 'decode'), default='prefill')
    p.add_argument('--tokens', type=int, default=8)
    p.add_argument('--trials', type=int, default=2)
    p.add_argument('--combine', default='device')
    p.add_argument('--primary-tok-s', type=float, help='primary card alone, same objective (optional start guess)')
    p.add_argument('--worker-tok-s', type=float, help='worker card alone, same objective (optional start guess)')
    p.add_argument('--start-groups', type=int)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--bench-arg', action='append', default=[], help='extra argument passed to glm_mixed_bench.py')
    a = p.parse_args()
    if a.groups < 2:
        p.error('--groups must be at least 2')
    if a.output.exists():
        p.error('output directory exists; choose a fresh one')
    a.output.mkdir(parents=True)
    bench = Path(__file__).resolve().parent / 'glm_mixed_bench.py'
    key = 'median_prefill_tok_s' if a.objective == 'prefill' else 'median_decode_tok_s'
    rows = []

    def evaluate(g):
        fraction = fraction_for(g, a.groups)
        out = a.output / f'groups-{g}'
        cmd = [a.python, str(bench), '--config', str(a.config), '--model', str(a.model), '--prior', str(a.prior),
               '--combine', a.combine, '--cases', a.case, '--fractions', str(fraction), '--tokens', str(a.tokens),
               '--trials', str(a.trials), '--repetitions', '1', '--allow-desktop-gpu', '--prompts', str(a.prompt),
               '--output', str(out), *a.bench_arg]
        with open(a.output / f'groups-{g}.log', 'w') as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy()).returncode
        value = None
        results = out / 'results.json'
        if code == 0 and results.exists():
            summary = performance_summary(json.loads(results.read_text())['results'], a.tokens)
            values = [x[key] for x in summary if x['case'] == a.case]
            value = values[0] if values else None
        rows.append(dict(groups=g, fraction=fraction, returncode=code, **{key: value}))
        print(f'groups={g} fraction={fraction} {key}={value}', flush=True)
        return value

    start = a.start_groups or initial_groups(a.groups, a.primary_tok_s, a.worker_tok_s)
    best, scores = climb(evaluate, a.groups, start)
    report = dict(case=a.case, objective=a.objective, groups=a.groups, start_groups=start, best_groups=best,
                  best_fraction=fraction_for(best, a.groups), best_score=scores[best], measurements=rows,
                  config=str(a.config), prompt=str(a.prompt))
    (a.output / 'fraction.json').write_text(json.dumps(report, indent=1) + '\n')
    print(f'best: STRATA_GLM_MIXED_PRIMARY_FRACTION={report["best_fraction"]} ({best}/{a.groups} groups) '
          f'{key}={scores[best]}')


if __name__ == '__main__':
    main()
