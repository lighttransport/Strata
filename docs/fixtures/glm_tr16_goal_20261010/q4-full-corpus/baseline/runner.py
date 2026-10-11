"""Evaluate a TR16 candidate against unchanged Q23, without changing regression tolerances.

Run once with the unchanged config, then with --baseline pointing at that result.
Corpus quality alone never qualifies an approximate candidate: a paired complete
164-task HumanEval run is also required before promotion.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from glm_regression import parse_eval, check_quality, TRAILING
from glm_tr16_calibrate import ROOT, digest, wait_idle, input_identity

TOLERANCE = dict(ppl_rel=.03, kl_rel=0, kl_abs=.03, top1_abs=.01)


def evaluation_width(config):
    env = config.get('env', {})
    width = int(env.get('STRATA_GLM_EVAL_WIDTH', '4'))
    if not 1 <= width <= 8:
        raise ValueError('STRATA_GLM_EVAL_WIDTH must be 1..8')
    if int(env.get('STRATA_GLM_DEFER_EXPERTS', '0')) and width != 1:
        raise ValueError('expert deferral quality requires STRATA_GLM_EVAL_WIDTH=1')
    return width


def compare_humaneval(candidate, baseline):
    """Apply the one-pass budget only to complete, paired 164-task evaluations."""
    expected = {f'HumanEval/{i}' for i in range(164)}
    for item in (candidate, baseline):
        rows = item.get('results', [])
        if (not item.get('complete') or item.get('tasks') != 164 or len(rows) != 164
                or {row.get('task_id') for row in rows} != expected
                or any(not isinstance(row.get('passed'), bool) for row in rows)):
            return ['incomplete or invalid HumanEval task set']
    candidate_passes = sum(row['passed'] for row in candidate['results'])
    baseline_passes = sum(row['passed'] for row in baseline['results'])
    return ([f'HumanEval {candidate_passes}/164 loses more than one pass versus {baseline_passes}/164']
            if candidate_passes < baseline_passes - 1 else [])


def compare(result, baseline):
    failures = []
    try:
        widths = [evaluation_width(item.get('config', {})) for item in (result, baseline)]
        if widths[0] != widths[1]:
            failures.append('incomparable evaluation width')
    except (ValueError, TypeError) as exc:
        failures.append('invalid evaluation width: ' + str(exc))
    for key in ('corpus_sha256', 'reference', 'reference_size', 'reference_mtime_ns', 'sequences', 'inputs'):
        if result[key] != baseline[key]:
            failures.append('incomparable ' + key)
    if not result['complete'] or not baseline['complete']:
        failures.append('incomplete evaluation')
    for item in (result, baseline):
        if any(not math.isfinite(item.get('metrics', {}).get(key, float('nan')))
               for key in ('perplexity', 'kl', 'top1_agreement', 'tokens')):
            failures.append('missing or nonfinite quality metrics')
    if failures:
        return failures
    return check_quality(result['metrics'], baseline['metrics'], TOLERANCE)


def quality_guard(output, command, record_cpu_interference=False):
    """CPU contention may be recorded for scoring, never for timing claims."""
    return [sys.executable, str(ROOT / 'tools/glm_q2_run_guard.py'), '--output', str(output / 'eval'),
            '--min-gpu-free-mib', '512', '--require-gpu-idle',
            '--record-interference' if record_cpu_interference else '--require-idle', '--', *command]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--baseline', type=Path)
    ap.add_argument('--regression-config', type=Path, default=ROOT / 'tests/glm/regression.json')
    ap.add_argument('--wait-idle-seconds', type=int, default=1800)
    ap.add_argument('--record-cpu-interference', action='store_true',
                    help='record CPU contention without aborting quality scoring; no timing comparison')
    args = ap.parse_args()
    c = json.loads(args.config.read_text())
    width = evaluation_width(c)
    if width != 4 and b'STRATA_GLM_EVAL_WIDTH' not in Path(c['exe']).read_bytes():
        raise ValueError('decoder does not support STRATA_GLM_EVAL_WIDTH')
    q = json.loads(args.regression_config.read_text())['quality']
    corpus = ROOT / q['corpus']
    reference = Path(q['reference']).resolve()
    refstat = reference.stat()
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    (output / 'corpus.ids').write_bytes(corpus.read_bytes())
    lines = [s for s in corpus.read_text().splitlines() if s.strip()]
    sequences = len(lines)
    expected_tokens = sum(len(re.split(r'[,\s]+', line.strip())) - 1 for line in lines)
    cmd = ['numactl', '--interleave=all', str(Path(c['exe']).resolve()), c['model'], '0', '1',
           str(c['dense_cache_mib']), str(c['threads']), '--context=' + str(c['context']),
           '--prefill-batch=256', '--gpu-budget-mib=' + str(c['gpu_budget_mib']),
           '--cpu-affinity=' + c['cpu_affinity'], '--decode-graphs',
           '--decode-cache-mib=' + str(c['decode_cache_mib']), '--expert-pack=' + c['expert_pack'],
           '--eval-corpus=' + str(output / 'corpus.ids'), '--eval-reference=' + str(reference)]
    result = dict(config=c, command=cmd, decoder_sha256=digest(c['exe']), runner_sha256=digest(__file__),
                  inputs=input_identity(c),
                  corpus_sha256=digest(corpus), reference=str(reference), reference_size=refstat.st_size,
                  reference_mtime_ns=refstat.st_mtime_ns, sequences=sequences, expected_tokens=expected_tokens,
                  evaluation_width=width,
                  quality_tolerance=TOLERANCE, complete=False, qualified=False,
                  cpu_interference_policy='record' if args.record_cpu_interference else 'reject',
                  performance_comparable=False)
    if args.baseline:
        result['baseline_result'] = str(args.baseline.resolve())
        result['baseline_result_sha256'] = digest(args.baseline)
    (output / 'runner.py').write_bytes(Path(__file__).read_bytes())
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    wait_idle(args.wait_idle_seconds)
    env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
    env.update(c['env'])
    guard = quality_guard(output, cmd, args.record_cpu_interference)
    with (output / 'guard.log').open('w') as log:
        proc = subprocess.run(guard, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    stdout = (output / 'eval.stdout').read_text() if (output / 'eval.stdout').exists() else ''
    stderr = (output / 'eval.log').read_text() if (output / 'eval.log').exists() else ''
    rows = parse_eval(stdout)
    memory = json.loads((output / 'eval.memory.json').read_text()) if (output / 'eval.memory.json').exists() else {}
    finalstat = reference.stat()
    reference_unchanged = (refstat.st_size, refstat.st_mtime_ns) == (finalstat.st_size, finalstat.st_mtime_ns)
    result.update(rows=rows, memory=memory, guard_exit_code=proc.returncode,
                  complete=bool([row['sequence'] for row in rows] == list(range(1, sequences + 1))
                                and rows and rows[-1]['tokens'] == expected_tokens
                                and not memory.get('rejected') and memory
                                and reference_unchanged and input_identity(c) == result['inputs']
                                and (proc.returncode == 0 or TRAILING in stderr)))
    if rows:
        result['metrics'] = rows[-1]
    if args.baseline:
        result['failures'] = compare(result, json.loads(args.baseline.read_text()))
        result['corpus_gate_passed'] = not result['failures']
    result['promotion_note'] = 'Full paired HumanEval and held-out timing still required.'
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('complete', 'metrics', 'failures', 'corpus_gate_passed') if k in result}))
    return 0 if result['complete'] and not result.get('failures') else 1


if __name__ == '__main__':
    raise SystemExit(main())
