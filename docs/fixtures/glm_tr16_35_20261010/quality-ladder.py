"""Quality ladder for one finalist, cheap to expensive. PREPARED; not run in the 4-hour window.

  python quality-ladder.py --config <finalist.json> --name <slug> --track 35|45 --steps task31 smoke20 pilot2048 full8192 full164 [--deadline EPOCH]

The finalist config must use the merged binary (merged-build-complete.json) with STRATA_GLM_MTP_REPEAT_GUARD=4.
Corpus steps score at margin 0 and evaluation width 1 (precision/routing loss only); generated-task steps use the
pinned HumanEval protocol (2048-token cap, 12K context, greedy, isolated tests). Each step stops the ladder on failure.
Gates come from goal.json; passing them here does not promote anything by itself (held-out speed is separate).
"""
import argparse, copy, gzip, hashlib, json, math, os, signal, subprocess, sys, time
from pathlib import Path
ROOT = Path('/mnt/nvme02/work/Strata'); os.chdir(ROOT)
R = ROOT / 'build-tr16-35-20261010'; OLD = ROOT / 'build-tr16-goal-20261010'; sys.path.insert(0, str(ROOT / 'tools'))
from glm_q2_run_guard import descendants
from glm_tr16_calibrate import digest, input_identity, wait_idle
from glm_tr16_quality import evaluation_width
PY = str(ROOT / 'build-exl3-venv/bin/python')
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--config', type=Path, required=True); ap.add_argument('--name', required=True)
ap.add_argument('--track', choices=['35', '45'], required=True)
ap.add_argument('--steps', nargs='+', default=['task31', 'smoke20', 'pilot2048', 'full8192', 'full164'])
ap.add_argument('--deadline', type=int); ap.add_argument('--continue-on-fail', action='store_true', help='record gate failures but run every requested step')
args = ap.parse_args()
goal = json.loads((R / 'goal.json').read_text()); end = args.deadline or goal['deadline']; gates = goal['gates']['track' + args.track]
c = json.loads(args.config.read_text())
marker = R / 'merged-build-complete.json' if Path(c['exe']).parent.name == 'merged-cuda' else OLD / 'draft-route-build-complete.json'
built = json.loads(marker.read_text())
for path, expected in built['sha256'].items(): assert digest(path) == expected, path
assert str(Path(c['exe'])) in built['sha256'] and built['cuda_parities_passed'], 'finalist must use a validated guard binary'
assert c['env'].get('STRATA_GLM_MTP_REPEAT_GUARD') == '4', 'generation quality requires the repeat guard'
inputs = input_identity(c)
out = R / ('q-' + args.name); out.mkdir(exist_ok=False)
(out / 'finalist.json').write_text(json.dumps(c, indent=2) + '\n')


def identity(pid):
    try:
        f = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split(); return f[19] if f[0] != 'Z' else None
    except (FileNotFoundError, ProcessLookupError): return None


def run(command, log, seconds, env=None):
    if time.time() + 300 >= end: raise SystemExit('Deadline before ' + log.name)
    with log.open('x') as f:
        p = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        try: return p.wait(timeout=min(seconds, end - time.time() - 30))
        except subprocess.TimeoutExpired:
            owned = {ch: identity(ch) for ch in descendants(p.pid)}
            for sig in (signal.SIGINT, signal.SIGKILL):
                for ch, stamp in owned.items():
                    if stamp and identity(ch) == stamp:
                        try: os.kill(ch, sig)
                        except ProcessLookupError: pass
                if sig == signal.SIGINT: time.sleep(10)
            p.wait(); raise


dataset = ROOT / 'build-q2-redesign/HumanEval.jsonl.gz'
tasks = [json.loads(line) for line in gzip.decompress(dataset.read_bytes()).splitlines()]; assert len(tasks) == 164
baseline_folder = ROOT / 'build-tr16-20261009/humaneval-baseline'
hb = json.loads((baseline_folder / 'measurement.json').read_text()); hbm = json.loads((baseline_folder / 'manifest.json').read_text())
assert hb['complete'] and hb['tasks'] == 164 and digest(dataset) == hbm['dataset_sha256']
baseline_pass = {v['task_id']: v['passed'] for v in hb['results']}
ladder = []


def humaneval(step, limit, task_id=None, seconds=5400):
    env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}; env.update(c['env'])
    tool = str(OLD / 'humaneval-one-task.py') if task_id else 'tools/glm_q2_humaneval.py'
    cmd = ['numactl', '--interleave=all', PY, tool, c['model'], str(dataset), '--output', str(out / step), '--decoder', c['exe'],
           '--expert-pack', c['expert_pack'], '--tokens', '2048', '--limit', str(limit), '--context', str(c['context']),
           '--prefill-batch', str(c['prefill_batch']), '--gpu-budget-mib', str(c['gpu_budget_mib']), '--decode-cache-mib', str(c['decode_cache_mib']),
           '--speculative', 'mtp', '--draft-depth', str(c['draft_depth']), '--route-affinity', c['env'].get('STRATA_GLM_ROUTE_AFFINITY', '0')]
    if task_id: cmd += ['--task-id', task_id]
    if c.get('mtp_experts') == 'gpu': raise SystemExit('humaneval tool has no --mtp-experts option; extend before using a GPU-draft finalist')
    guard = [PY, 'tools/glm_q2_run_guard.py', '--output', str(out / (step + '-guard')), '--min-gpu-free-mib', '512', '--require-gpu-idle', '--record-interference', '--', *cmd]
    wait_idle(1800); code = run(guard, out / (step + '-driver.log'), seconds, env)
    m = out / step / 'measurement.json'; x = json.loads(m.read_text()) if m.exists() else {}
    mem = out / (step + '-guard.memory.json'); memory = json.loads(mem.read_text()) if mem.exists() else {}
    results = x.get('results', []); ids = [v.get('task_id') for v in results]
    expected = [task_id] if task_id else [t['task_id'] for t in tasks[:limit]]
    complete = code == 0 and ids == expected and all(isinstance(v.get('passed'), bool) for v in results) and bool(memory) and not memory.get('rejected') and input_identity(c) == inputs
    if complete:
        man = json.loads((out / step / 'manifest.json').read_text())
        for key in ('dataset_revision', 'dataset_sha256', 'protocol', 'maximum_generated_tokens'): assert man[key] == hbm[key], key
    passes = sum(v['passed'] is True for v in results); base = sum(baseline_pass[t] for t in expected)
    capped = [v['task_id'] for v in results if v.get('generated_tokens', 0) >= 2048]
    lost = [v['task_id'] for v in results if not v['passed'] and baseline_pass[v['task_id']]]
    gained = [v['task_id'] for v in results if v['passed'] and not baseline_pass[v['task_id']]]
    return dict(step=step, exit_code=code, complete=complete, tasks=len(results), passes=passes, baseline_passes=base, net=passes - base,
                lost=lost, gained=gained, capped_2048=capped, guard_rejected=memory.get('rejected'))


def corpus(step, regression, expected_tokens, baseline, seconds):
    q = copy.deepcopy(c); q['env'].update(STRATA_GLM_EVAL_WIDTH='1', STRATA_GLM_MTP_ACCEPT_MARGIN='0')
    cfg = out / (step + '.json'); cfg.write_text(json.dumps(q, indent=2) + '\n')
    code = run([PY, 'tools/glm_tr16_quality.py', '--config', str(cfg), '--output', str(out / step), '--regression-config', str(regression),
                '--wait-idle-seconds', '1800', '--record-cpu-interference'], out / (step + '-driver.log'), seconds)
    x = json.loads((out / step / 'result.json').read_text()); b = json.loads(baseline.read_text())
    assert x['complete'] and x['expected_tokens'] == expected_tokens and x['metrics']['tokens'] == expected_tokens
    assert evaluation_width(x['config']) == evaluation_width(b['config'])
    for key in ('corpus_sha256', 'reference', 'reference_size', 'reference_mtime_ns', 'sequences'): assert x[key] == b[key], key
    assert all(math.isfinite(x['metrics'][k]) for k in ('perplexity', 'kl', 'top1_agreement'))
    ppl = x['metrics']['perplexity'] / b['metrics']['perplexity'] - 1; kl = x['metrics']['kl'] - b['metrics']['kl']
    top1 = 100 * (b['metrics']['top1_agreement'] - x['metrics']['top1_agreement'])
    scale = 1.5 if step == 'pilot2048' else 1.0
    ok = ppl <= gates['ppl_rel_max'] * scale and kl <= gates['kl_abs_max'] * scale and top1 <= gates['top1_drop_pp_max'] * scale
    return dict(step=step, exit_code=code, complete=True, metrics=x['metrics'], baseline=str(baseline), ppl_increase_percent=100 * ppl,
                kl_increase=kl, top1_drop_pp=top1, gate_scale=scale, passes_gate=ok, note='baseline is unchanged Q23; the Q22 pack change is intentional')


for step in args.steps:
    print('START', step, time.strftime('%H:%M:%S'), flush=True)
    if step == 'task31':
        row = humaneval(step, 1, 'HumanEval/31', 900); row['passes_gate'] = row['complete'] and row['passes'] == 1 and not row['capped_2048']
    elif step == 'smoke20':
        row = humaneval(step, 20, None, 1500); row['passes_gate'] = row['complete'] and row['net'] >= -1 and not row['capped_2048']
    elif step == 'pilot2048':
        row = corpus(step, OLD / 'renorm-quality-pilot/regression.json', 2048, OLD / 'renorm-quality-pilot/00-renorm-default/result.json', 1800)
    elif step == 'full8192':
        row = corpus(step, OLD / 'q4-full-corpus/regression.json', 8192, OLD / 'q4-full-corpus/baseline/result.json', 2400)
    elif step == 'full164':
        row = humaneval(step, 164, None, 5400); row['passes_gate'] = row['complete'] and row['passes'] >= gates['humaneval_min'] and not row['capped_2048']
    else: raise SystemExit('unknown step ' + step)
    row['quality_qualified'] = False; ladder.append(row)
    (out / 'ladder.json').write_text(json.dumps(dict(track=args.track, gates=gates, steps=ladder), indent=2) + '\n')
    print('DONE', json.dumps(row), flush=True)
    if not row['passes_gate']:
        print('GATE FAILED at', step, flush=True)
        if not args.continue_on_fail: print('STOP', flush=True); break
(out / 'complete.json').write_text(json.dumps(dict(steps=[r['step'] for r in ladder], all_passed=all(r['passes_gate'] for r in ladder), quality_qualified=False)) + '\n')
