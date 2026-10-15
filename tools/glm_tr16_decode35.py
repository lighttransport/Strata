#!/usr/bin/env python3
"""Strict fresh-process TR16 decode comparisons; speed gates do not imply quality qualification.

Example: --config control.json --config candidate.json --prompt-2k code2k.ids
         --prompt-8k code8k.ids --output build-tr16-screen
Each fresh decoder performs one warmup and one timed continuation. Candidate order
reverses on alternating repetitions. Outputs are never reused or overwritten.
"""
import argparse
import copy
import json
import statistics
import subprocess
import sys
from pathlib import Path

from glm_tr16_calibrate import ROOT, digest, input_identity


def checked_speed(result, tokens):
    if not all(result.get(k) for k in ('complete', 'clean', 'quiet', 'inputs_unchanged')):
        raise ValueError('incomplete, unguarded or changed-input measurement')
    manifest, memory = result['manifest'], result['memory']
    if (result.get('guard_exit_code') != 0 or manifest.get('allow_interference') or
            memory.get('exit_code') != 0 or memory.get('rejected') or memory.get('peak_swap_kib') != 0):
        raise ValueError('measurement failed strict guard')
    if manifest.get('timed_trials') != 1 or manifest.get('warmup_decode_trials') != 1:
        raise ValueError('expected one warmup and one timed trial per process')
    rows = result['measurements']['decode']
    if len(rows) != 1 or manifest.get('decode_tokens') != tokens:
        raise ValueError('incomplete continuation')
    # Ordinary decode times the steps after the first token supplied by prefill.
    expected = tokens - 1 if rows[0].get('kind') == 'single' else tokens
    if rows[0]['tokens'] != expected:
        raise ValueError('incomplete continuation')
    speed = rows[0]['tokens_per_second']
    if not 0 < speed < float('inf'):
        raise ValueError('invalid speed')
    return speed


def summary(rows, names, trials):
    result = {}
    for name in names:
        arms = {}
        for width in (2048, 8192):
            values = [r['tok_s'] for r in rows if r['name'] == name and r['width'] == width]
            arms[str(width)] = dict(trials=values, median=statistics.median(values) if values else None)
        complete = all(len(v['trials']) == trials for v in arms.values())
        result[name] = dict(contexts=arms, complete=complete,
                            screen_above_target=complete and all(v['median'] >= 35 for v in arms.values()),
                            speed_gate=complete and trials >= 3 and all(v['median'] >= 35 for v in arms.values()),
                            quality_qualified=False)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, action='append', required=True)
    ap.add_argument('--prompt-2k', type=Path, required=True)
    ap.add_argument('--prompt-8k', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--tokens', type=int, default=512)
    ap.add_argument('--wait-idle-seconds', type=int, default=600)
    args = ap.parse_args()
    if not 1 <= args.trials <= 9 or not 512 <= args.tokens <= 4096:
        ap.error('trials must be 1..9 and tokens 512..4096')
    names = [p.stem for p in args.config]
    if len(set(names)) != len(names):
        ap.error('config file stems must be unique')
    prompts = {2048: args.prompt_2k.resolve(), 8192: args.prompt_8k.resolve()}
    prompt_hashes = {width: digest(prompt) for width, prompt in prompts.items()}
    for width, prompt in prompts.items():
        if len(prompt.read_text().replace(',', ' ').split()) != width:
            ap.error(f'expected {width} prompt tokens: {prompt}')
    configs = [json.loads(p.read_text()) for p in args.config]
    identities = [(digest(c['exe']), input_identity(c)) for c in configs]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for trial in range(args.trials):
        order = range(len(configs)) if trial % 2 == 0 else reversed(range(len(configs)))
        for index in order:
            for width, prompt in prompts.items():
                c = copy.deepcopy(configs[index]); c['prefill_batch'] = width
                if digest(prompt) != prompt_hashes[width]:
                    raise RuntimeError('prompt changed during comparison')
                if (digest(c['exe']), input_identity(c)) != identities[index]:
                    raise RuntimeError('decoder/model/prior changed during comparison')
                name = f'{names[index]}-{width}-fresh{trial}'
                config = args.output / (name + '.json')
                config.write_text(json.dumps(c, indent=2) + '\n')
                output = args.output / name
                command = [sys.executable, str(ROOT / 'tools/glm_tr16_calibrate.py'), '--config', str(config),
                           '--prompt', str(prompt), '--output', str(output), '--cases', 'baseline',
                           '--tokens', str(args.tokens), '--trials', '1', '--max-swap-mib', '0',
                           '--wait-idle-seconds', str(args.wait_idle_seconds)]
                with (args.output / (name + '.driver.log')).open('w') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                if digest(prompt) != prompt_hashes[width]:
                    raise RuntimeError('prompt changed while decoder was running')
                if (digest(c['exe']), input_identity(c)) != identities[index]:
                    raise RuntimeError('decoder/model/prior changed while decoder was running')
                path = output / '000-baseline.result.json'
                r = json.loads(path.read_text())
                speed = checked_speed(r, args.tokens)
                rows.append(dict(name=names[index], width=width, trial=trial, tok_s=speed,
                                 result=str(path.relative_to(args.output)),
                                 tokens_sha256=r['trial_token_sha256'][-1]))
                (args.output / 'summary.json').write_text(json.dumps(summary(rows, names, args.trials), indent=2) + '\n')
                (args.output / 'trials.json').write_text(json.dumps(rows, indent=2) + '\n')
                print(json.dumps(rows[-1]), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
