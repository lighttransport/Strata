"""TR16 35/45 tok/s follow-up: guarded short decode screens on isolated research binaries.

Each case is a full calibrator config derived from --base with env/set overrides. Speed
numbers come from tools/glm_tr16_calibrate.py under the quiet-run guard. Nothing here
qualifies quality: the loop proxy and acceptance statistics only select candidates.
"""
import argparse, copy, hashlib, json, os, re, signal, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path('/mnt/nvme02/work/Strata'); os.chdir(ROOT)
R = ROOT / 'build-tr16-35-20261010'; OLD = ROOT / 'build-tr16-goal-20261010'
sys.path.insert(0, str(ROOT / 'tools'))
from glm_q2_run_guard import descendants
from glm_tr16_calibrate import digest, input_identity, missing_decoder_flags

PYTHON = str(ROOT / 'build-exl3-venv/bin/python')
MARKERS = {'gpu-cache-v2-cuda': OLD / 'gpu-cache-v2-build-complete.json',
           'draft-route-cuda': OLD / 'draft-route-build-complete.json',
           'cold-prune-cuda': OLD / 'cold-prune-build-complete.json',
           'merged-cuda': R / 'merged-build-complete.json', 'stream-cuda': R / 'stream-build-complete.json'}


def identity(pid):
    try:
        f = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return f[19] if f[0] != 'Z' else None
    except (FileNotFoundError, ProcessLookupError):
        return None


def run(command, log, end, seconds):
    if time.time() + 180 >= end:
        raise SystemExit('Deadline before ' + log.name)
    with log.open('x') as f:
        p = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return p.wait(timeout=min(seconds, end - time.time() - 30))
        except subprocess.TimeoutExpired:
            owned = {child: identity(child) for child in descendants(p.pid)}
            for sig in (signal.SIGINT, signal.SIGKILL):
                for child, stamp in owned.items():
                    if stamp and identity(child) == stamp:
                        try: os.kill(child, sig)
                        except ProcessLookupError: pass
                if sig == signal.SIGINT: time.sleep(10)
            p.wait(); raise


def distinct_ngram_fraction(seq, n=4):
    grams = [tuple(seq[j:j + n]) for j in range(len(seq) - n + 1)]
    return len(set(grams)) / len(grams) if grams else None


def has_repeated_ngram(seq, n=32):
    seen = set()
    for j in range(len(seq) - n + 1):
        g = tuple(seq[j:j + n])
        if g in seen: return True
        seen.add(g)
    return False


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y: return i
    return None if len(a) == len(b) else min(len(a), len(b))


def case_summary(x, log, tokens, trials, depth):
    m = x['measurements']
    spec = m['speculation'][-trials:]; timing = m['mtp_timing'][-trials:]; cpu = m['cpu_expert'][-trials:]
    generated = sum(s['generated'] for s in spec); rounds = sum(s['rounds'] for s in spec)
    accepted = sum(s['accepted'] for s in spec); proposed = sum(s['proposed'] for s in spec)
    pos = {}
    for rec in m['mtp_positions'][-trials * depth:] if depth else []:
        p = pos.setdefault(rec['index'], dict(proposed=0, accepted=0))
        p['proposed'] += rec['proposed']; p['accepted'] += rec['accepted']
    approx = m.get('mtp_approx', [])[-trials:]
    out = dict(
        decode_tok_s=m['median_decode_tok_s'], decode_trials=[d['tokens_per_second'] for d in m['decode']],
        decode_warmup_tok_s=m['decode_warmup'][0]['tokens_per_second'] if m.get('decode_warmup') else None,
        prefill_tok_s=m['median_prefill_tok_s'],
        rounds=rounds, generated=generated, tokens_per_round=generated / rounds if rounds else None,
        accepted_per_round=accepted / rounds if rounds else None, proposed_per_round=proposed / rounds if rounds else None,
        position_accept_rate={k: (v['accepted'] / v['proposed'] if v['proposed'] else None) for k, v in sorted(pos.items())},
        soft_accepted=sum(a.get('soft_accepted', 0) for a in approx) if approx else None,
        logit_regret_per_token=(sum(a.get('sum_logit_regret', 0) for a in approx) / generated) if approx and generated else None,
        draft_ms_per_round=sum(t['draft_ms'] for t in timing) / rounds if rounds and timing else None,
        verify_ms_per_round=sum(t['verify_ms'] for t in timing) / rounds if rounds and timing else None,
        resync_ms_per_round=sum(t['resync_ms'] for t in timing) / rounds if rounds and timing else None,
        expert_gb_per_token=sum(c['bytes'] for c in cpu) / 1e9 / generated if cpu and generated else None,
        flow_ms_per_token=sum(c['layer_flow_ms'] for c in cpu) / generated if cpu and generated else None,
        ms_per_token=sum(d['milliseconds'] for d in m['decode']) / generated if generated else None,
        peak_gpu_mib=m.get('peak_allocated_gpu_mib'), gpu_live=m['gpu_live'][-1]['tags'] if m.get('gpu_live') else None,
        repeat_guard=m.get('mtp_repeat_guard', [])[-trials:], sync_state_only=m.get('mtp_sync_state', [])[-trials:],
        lookup=re.findall(r'MTP_LOOKUP .*', log)[-trials:], mtp_depth=re.findall(r'MTP_DEPTH .*', log)[-1:],
    )
    cache = re.findall(r'DECODE_CACHE prefix_trained=1 slots=(\d+) MiB=([\d.]+) fingerprint=(\d+)', log)
    out.update(actual_cache_slots=int(cache[-1][0]) if cache else None, actual_cache_mib=float(cache[-1][1]) if cache else None,
               cache_fingerprint=cache[-1][2] if cache else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name', required=True); ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--tokens', type=int, default=128); ap.add_argument('--trials', type=int, default=1)
    ap.add_argument('--prompt', default='build-tr16-20261009/prompts2k/train-code.ids')
    ap.add_argument('--prefill-batch', type=int, choices=[2048, 4096, 8192])
    ap.add_argument('--control-stdout', type=Path, help='token stream to compare against; default: goal.json reference for 128x1, else the first case')
    ap.add_argument('--control-rate', type=float, help='control decode tok/s; default from goal.json reference or first case')
    ap.add_argument('--deadline', type=int); ap.add_argument('--wait-idle-seconds', type=int, default=3600)
    ap.add_argument('--out-root', type=Path, default=R)
    args = ap.parse_args()
    goal = json.loads((R / 'goal.json').read_text()); end = args.deadline or goal['deadline']
    base = json.loads(args.base.read_text()); cases = json.loads(args.cases.read_text())
    out = args.out_root / args.name; out.mkdir(exist_ok=False)
    rows = []; control_tokens = None; control_rate = args.control_rate
    ref = goal['reference']
    if args.control_stdout:
        control_tokens = args.control_stdout.read_text().split()
    elif args.tokens == ref['tokens'] and args.trials == ref['trials'] and args.prompt == ref['prompt']:
        control_tokens = Path(ref['stdout']).read_text().split(); control_rate = control_rate or ref['decode_tok_s']
    for case in cases:
        name = case['name']; c = copy.deepcopy(base)
        for k, v in case.get('env', {}).items():
            if v is None: c['env'].pop(k, None)
            else: c['env'][k] = str(v)
        for k, v in case.get('set', {}).items():
            c[k] = v
        c['model_name'] = 'tr35-' + name
        c['experimental_role'] = '35/45 follow-up screen; routing and outputs can change; no quality qualification'
        exe = Path(c['exe']); marker = MARKERS[exe.parent.name]
        built = json.loads(marker.read_text()); assert built['cuda_parities_passed'], marker
        assert digest(exe) == built['sha256'][str(exe)], 'binary hash changed ' + str(exe)
        missing = missing_decoder_flags([c], exe.read_bytes())
        if missing: raise SystemExit(f'{name}: {missing} missing from {exe}')
        cfg = out / (name + '.json'); cfg.write_text(json.dumps(c, indent=2) + '\n'); dest = out / name
        cmd = [PYTHON, 'tools/glm_tr16_calibrate.py', '--config', str(cfg), '--prompt', args.prompt, '--output', str(dest),
               '--tokens', str(args.tokens), '--trials', str(args.trials), '--cases', 'baseline',
               '--wait-idle-seconds', str(args.wait_idle_seconds), '--timeout', str(600 + int(args.trials * args.tokens * 0.5) + (300 if args.prefill_batch == 8192 else 0))]
        if args.prefill_batch: cmd += ['--prefill-batch', str(args.prefill_batch)]
        print('START', args.name, name, time.strftime('%H:%M:%S'), flush=True)
        code = run(cmd, out / (name + '-driver.log'), end, 600 + args.trials * args.tokens + 600)
        result = dest / '000-baseline.result.json'
        row = dict(name=name, note=case.get('note'), exact_expected=bool(case.get('exact')), exit_code=code, config=str(cfg),
                   env_overrides=case.get('env', {}), set_overrides=case.get('set', {}), exe=str(exe), quality_qualified=False)
        if not result.exists():
            row.update(clean=False, decision='FAILED'); rows.append(row)
            (out / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n'); print('FAILED', name, flush=True); continue
        x = json.loads(result.read_text()); log = (dest / '000-baseline.log').read_text()
        row.update(clean=x['clean'], memory={k: x['memory'].get(k) for k in ('peak_rss_kib', 'peak_swap_kib', 'minimum_gpu_free_mib', 'rejected', 'wall_seconds')})
        row.update(case_summary(x, log, args.tokens, args.trials, int(c.get('draft_depth', 0)) if c.get('speculative') == 'mtp' else 0))
        tokens = (dest / '000-baseline.stdout').read_text().split()
        row['token_count'] = len(tokens); row['token_count_expected'] = (args.trials + 1) * args.tokens
        streams = [list(map(int, tokens[i * args.tokens:(i + 1) * args.tokens])) for i in range(args.trials + 1)]
        row['distinct_4gram_fraction'] = [distinct_ngram_fraction(s) for s in streams]
        row['repeated_32gram'] = [has_repeated_ngram(s) for s in streams]
        row['trial_token_sha256'] = x['trial_token_sha256']
        if control_tokens is None or case.get('control'):
            row['is_control'] = True
            if control_tokens is not None:
                row['reference_tokens_identical'] = tokens == control_tokens
                row['reference_cache_identical'] = (row['actual_cache_slots'], row['actual_cache_mib'], row['cache_fingerprint']) == (ref['slots'], ref['mib'], ref['fingerprint'])
                assert row['reference_tokens_identical'] and row['reference_cache_identical'], 'control does not reproduce the reference'
            control_tokens = tokens
            if not args.control_rate: control_rate = row['decode_tok_s']
        row['tokens_identical_to_control'] = tokens == control_tokens
        row['first_difference_index_by_trial'] = [first_difference(s, list(map(int, control_tokens[i * args.tokens:(i + 1) * args.tokens]))) for i, s in enumerate(streams)]
        row['control_rate'] = control_rate
        row['speed_change_percent'] = (row['decode_tok_s'] / control_rate - 1) * 100 if control_rate and row['decode_tok_s'] else None
        ctrl_rows = [r for r in rows if r.get('is_control') and r.get('distinct_4gram_fraction')]
        ctrl_d4 = min(ctrl_rows[0]['distinct_4gram_fraction'][1:]) if ctrl_rows else 0.99
        loop_ok = row['clean'] and all(v is not None and v >= 0.85 for v in row['distinct_4gram_fraction'][1:]) and not any(row['repeated_32gram'][1:])  # v2 rule: absolute floor; quoted code signatures legitimately repeat 4-grams
        gain = row['speed_change_percent'] or 0
        if row.get('is_control'): decision = 'CONTROL'
        elif not row['clean']: decision = 'DROP:unclean'
        elif not loop_ok: decision = 'DROP:loop-proxy'
        elif row['exact_expected'] and not row['tokens_identical_to_control']: decision = ('NOTE' if gain >= 1 else 'DROP') + ':reclassified-lossy'
        elif gain >= 3: decision = 'KEEP'
        elif gain >= 1: decision = 'NOTE'
        else: decision = 'DROP:no-gain'
        row['decision'] = decision; row['input_identity_matches_base'] = input_identity(c) == input_identity(base)
        rows.append(row); (out / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
        print('DONE', name, json.dumps(dict(tok_s=row['decode_tok_s'], change=row['speed_change_percent'], tpr=row['tokens_per_round'],
              gb=row['expert_gb_per_token'], flow=row['flow_ms_per_token'], cache=row['actual_cache_mib'], same=row['tokens_identical_to_control'],
              d4=row['distinct_4gram_fraction'], decision=decision)), flush=True)
        if code and row.get('is_control'): raise SystemExit('Control failed; stopping')
    (out / 'complete.json').write_text(json.dumps(dict(cases=len(rows), decisions={r['name']: r.get('decision') for r in rows}, quality_qualified=False)) + '\n')


if __name__ == '__main__':
    main()
