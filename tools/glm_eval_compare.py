#!/usr/bin/env python3
"""Paired comparison of two coding-evaluation result sets (candidate against reference).

Each input is a results.json written by an adapter (tools/glm_eval_effibench.py, glm_eval_aider.py, glm_eval_omnicode.py):
  {"suite": ..., "profile": ..., "tasks": [{"id": ..., "passed": bool, "metrics": {...}}, ...]}
Only tasks present in both sets are compared. Reports per-task flips, an exact McNemar test on the discordant pairs,
and a bootstrap confidence interval for the pass-rate difference. With --margin, exits 1 when the candidate is
significantly worse (McNemar p below --alpha with more losses than wins) or when the lower bound of the CI is below
-margin.

  python3 tools/glm_eval_compare.py reference/results.json candidate/results.json --margin 0.03
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path


def load(path):
    data = json.loads(Path(path).read_text())
    tasks = {}
    for row in data.get('tasks', []):
        if not isinstance(row.get('passed'), bool):
            raise ValueError(f'{path}: task {row.get("id")!r} has no boolean "passed"')
        if row['id'] in tasks:
            raise ValueError(f'{path}: duplicate task id {row["id"]!r}')
        tasks[row['id']] = row
    return data, tasks


def mcnemar_exact(wins, losses):
    """Two-sided exact binomial test on the discordant pairs (wins = candidate-only passes)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_diff(pairs, samples=10000, seed=20261016, level=0.95):
    """Percentile CI of mean(candidate - reference) over paired 0/1 outcomes."""
    if not pairs:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(pairs)
    diffs = [c - r for r, c in pairs]
    stats = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    lo = stats[int((1 - level) / 2 * samples)]
    hi = stats[min(samples - 1, int((1 + level) / 2 * samples))]
    return sum(diffs) / n, lo, hi


def compare(reference, candidate, alpha=0.05, margin=None):
    common = sorted(set(reference) & set(candidate))
    pairs = [(int(reference[t]['passed']), int(candidate[t]['passed'])) for t in common]
    wins = [t for t in common if candidate[t]['passed'] and not reference[t]['passed']]
    losses = [t for t in common if reference[t]['passed'] and not candidate[t]['passed']]
    diff, lo, hi = bootstrap_diff(pairs)
    p = mcnemar_exact(len(wins), len(losses))
    failures = []
    if p < alpha and len(losses) > len(wins):
        failures.append(f'significant regression: McNemar p={p:.4f} ({len(losses)} losses, {len(wins)} wins)')
    if margin is not None and lo < -margin:
        failures.append(f'pass-rate difference CI lower bound {lo:+.3f} is below -{margin:.3f}')
    return dict(tasks=len(common), only_reference=len(set(reference) - set(candidate)),
                only_candidate=len(set(candidate) - set(reference)),
                reference_passed=sum(r for r, _ in pairs), candidate_passed=sum(c for _, c in pairs),
                wins=wins, losses=losses, mcnemar_p=p, diff=diff, ci95=[lo, hi], failures=failures)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('reference')
    ap.add_argument('candidate')
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--margin', type=float, help='largest acceptable drop in pass rate (e.g. 0.03)')
    ap.add_argument('--json', action='store_true', help='print the result as JSON')
    a = ap.parse_args()
    ref_meta, ref = load(a.reference)
    cand_meta, cand = load(a.candidate)
    if ref_meta.get('suite') != cand_meta.get('suite'):
        raise SystemExit(f'different suites: {ref_meta.get("suite")} vs {cand_meta.get("suite")}')
    r = compare(ref, cand, a.alpha, a.margin)
    if a.json:
        print(json.dumps(r, indent=1))
    else:
        n = r['tasks'] or 1
        print(f'suite {ref_meta.get("suite")}: {r["tasks"]} paired tasks')
        print(f'  reference {ref_meta.get("profile")}: {r["reference_passed"]} ({r["reference_passed"] / n:.1%})')
        print(f'  candidate {cand_meta.get("profile")}: {r["candidate_passed"]} ({r["candidate_passed"] / n:.1%})')
        print(f'  difference {r["diff"]:+.3f}, 95% CI [{r["ci95"][0]:+.3f}, {r["ci95"][1]:+.3f}], '
              f'McNemar p={r["mcnemar_p"]:.4f}')
        print(f'  candidate-only passes: {len(r["wins"])}; reference-only passes: {len(r["losses"])}')
        for f in r['failures']:
            print('  FAIL:', f)
    return 1 if r['failures'] else 0


if __name__ == '__main__':
    sys.exit(main())
