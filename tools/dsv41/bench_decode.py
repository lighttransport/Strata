#!/usr/bin/env python3
"""Sequential decode screens against an immutable token reference (engine must be idle).

Outputs raw stdout/stderr, exact argv/env overrides, and measured decode counters.
No service management, model downloads, or changes to the supplied run config.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

BOOL_FLAGS = {'--serve', '--no-expert-cache', '--profile-readonly', '--verbose', '--vision-resident', '--lookup-check'}


def token_ids(raw):
    value = raw.strip().rstrip(b',')
    if not value:
        raise ValueError('empty token output')
    ids = tuple(int(x) for x in value.split(b','))
    if any(x < 0 for x in ids):
        raise ValueError('negative token output')
    return ids


def flags(base, changes):
    out = list(base)
    for key, value in changes.items():
        if not key.startswith('--'):
            raise ValueError(f'invalid flag: {key}')
        while key in out:
            i = out.index(key)
            del out[i:i + (1 if key in BOOL_FLAGS else 2)]
        if value is not None and value is not False:
            out.append(key)
            if value is not True:
                out.extend(map(str, value if isinstance(value, list) else [value]))
    return out


def counters(stderr):
    result = {}
    match = re.search(r'placement hash ([0-9a-f]+)', stderr)
    if match:
        result['placement_hash'] = match[1]
    match = re.search(r'target placement hash ([0-9a-f]+)', stderr)
    if match:
        result['target_placement_hash'] = match[1]
    match = re.search(r'decode: (\d+) tokens in ([\d.]+) ms \(([\d.]+) tok/s\)', stderr)
    if match:
        result.update(tokens=int(match[1]), decode_ms=float(match[2]), tok_s=float(match[3]))
    match = re.search(r'CPU lane ([\d.]+).*?head wait ([\d.]+), Engram reads ([\d.]+)', stderr)
    if match:
        result.update(cpu_ms_per_token=float(match[1]), head_wait_ms_per_token=float(match[2]), engram_wait_ms_per_token=float(match[3]))
    match = re.search(r'Engram exposed wait ms per committed token \(round-amortized\): p50 ([\d.]+) p95 ([\d.]+) max ([\d.]+)', stderr)
    if match:
        result.update(engram_wait_p50_ms=float(match[1]), engram_wait_p95_ms=float(match[2]), engram_wait_max_ms=float(match[3]))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--variants', required=True, type=Path)
    ap.add_argument('--prompt-ids', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--tokens', type=int, default=128)
    ap.add_argument('--repetitions', type=int, default=1)
    ap.add_argument('--timeout', type=int, default=1800)
    ap.add_argument('--reference-ids', type=Path, help='frozen engine token output; compare its first --tokens IDs')
    a = ap.parse_args()
    if a.tokens <= 0 or a.repetitions <= 0 or a.timeout <= 0:
        ap.error('tokens, repetitions, and timeout must be positive')
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_text())
    variants = json.loads(a.variants.read_text())
    if not isinstance(variants, list) or not variants:
        ap.error('variants must be a nonempty JSON list')
    names = [v['name'] for v in variants]
    if len(names) != len(set(names)) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', x) for x in names):
        ap.error('variant names must be unique and contain only letters, digits, underscores, or hyphens')
    reference = None
    if a.reference_ids:
        reference = token_ids(a.reference_ids.read_bytes())[:a.tokens]
        if len(reference) != a.tokens:
            ap.error('frozen reference contains fewer than --tokens IDs')
    results = []
    meta = {'config_sha256': hashlib.sha256(a.config.read_bytes()).hexdigest(),
            'prompt_sha256': hashlib.sha256(a.prompt_ids.read_bytes()).hexdigest(),
            'binary_sha256': hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest(),
            'tokens_requested': a.tokens, 'sampling': 'greedy', 'results': results}
    if a.reference_ids:
        meta['reference_sha256'] = hashlib.sha256(a.reference_ids.read_bytes()).hexdigest()
    for rep in range(a.repetitions):
        for variant in variants:
            name = variant['name']
            if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
                raise ValueError('invalid variant name')
            args = flags(cfg['args'], {**variant.get('flags', {}), '--serve': None,
                                      '--ids': '@'+str(a.prompt_ids.resolve()), '--n': a.tokens,
                                      '--temperature': 0, '--no-expert-cache': True})
            env = dict(os.environ, **cfg.get('env', {}), **variant.get('env', {}))
            env['DS4_PLACE_HASH'] = '1'
            prefix = a.output / f'{rep:02d}-{name}'
            t0 = time.monotonic()
            with prefix.with_suffix('.stdout').open('wb') as out, prefix.with_suffix('.stderr').open('wb') as err:
                try:
                    code = subprocess.run([cfg['exe'], *args], env=env, stdout=out, stderr=err, timeout=a.timeout).returncode
                except subprocess.TimeoutExpired:
                    code = 'timeout'
            raw = prefix.with_suffix('.stdout').read_bytes()
            stderr = prefix.with_suffix('.stderr').read_text(errors='replace')
            measured = counters(stderr)
            if code == 0 and 'tokens' not in measured:
                code = 'missing_decode_measurement'
            ids = None
            if code == 0:
                try:
                    ids = token_ids(raw)
                except ValueError:
                    code = 'invalid_token_output'
                if ids is not None and len(ids) != measured['tokens']:
                    code = 'token_count_mismatch'
            if reference is None and code == 0:
                reference = ids
            result = {'name': name, 'repeat': rep, 'exit_code': code, 'elapsed_seconds': time.monotonic()-t0,
                      'argv': [cfg['exe'], *args], 'env_overrides': variant.get('env', {}),
                      'token_sha256': hashlib.sha256(raw).hexdigest(), 'same_tokens': code == 0 and ids == reference,
                      'completed_requested_tokens': measured.get('tokens') == a.tokens, **measured}
            results.append(result)
            (a.output/'results.json').write_text(json.dumps(meta, indent=2))
            print(json.dumps({k: v for k, v in result.items() if k not in ('argv', 'env_overrides')}), flush=True)
            if code != 0:
                raise RuntimeError(f'{name} failed: {code}; inspect {prefix}.stderr')


if __name__ == '__main__':
    main()
