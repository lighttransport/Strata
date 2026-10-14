#!/usr/bin/env python3
"""Reproduce scoped redesign sensitivities from existing diagnostic records.

Inputs are synchronized aggregate phases, not independent operator calibration.
This deliberately produces conditional rates rather than an achievable maximum.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from sim.heterogeneous import Event, schedule, partition_prefill


def read(path):
    data = path.read_bytes()
    return json.loads(data), dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())


def analyze(prefill, decode_trace, decode_benchmark):
    p, pi = read(prefill)
    t, ti = read(decode_trace)
    b, bi = read(decode_benchmark)
    measured = b['results'][0]['measurement']['measurements']['decode'][-1]
    # The engine excludes the first, prefill-produced token from decode tok/s.
    timed_tokens = round(measured['tokens_per_second']*measured['milliseconds']/1000)
    if timed_tokens <= 0:
        raise ValueError('invalid measured decode numerator')
    scopes = t['reports'][-1]['raw']
    fixed = measured['milliseconds'] - sum(scopes[k] for k in ('head_ms', 'cpu_ms', 'between_ms', 'tail_ms'))
    if fixed < 0:
        raise ValueError('nested or mismatched timing scopes')
    decode_rows = []
    for service, wait in itertools.product((1, 1.25, 1.5, 2), repeat=2):
        parts = [('other', fixed), ('head', scopes['head_ms']),
                 ('host_service', scopes['cpu_ms']/service),
                 ('between_wait', scopes['between_ms']/wait), ('tail', scopes['tail_ms'])]
        events = [Event(name, {'serial': ms}, (parts[i-1][0],) if i else ())
                  for i, (name, ms) in enumerate(parts)]
        result = schedule(events, {'serial': 1})
        decode_rows.append(dict(host_service_speedup=service, between_wait_speedup=wait,
                                conditional_tok_s=timed_tokens*1000/result['milliseconds']))
    prefill_rows = []
    overhead = p['mixed_calibration']['equivalent_overhead_ms_per_moe']
    for primary, mixer, expert in itertools.product(('cuda', 'hip'), (1, 1.25, 1.5, 2), (1, 1.25, 1.5, 2)):
        worker = 'hip' if primary == 'cuda' else 'cuda'
        candidates = []
        for groups in range(1, 14):
            result = partition_prefill(p['profiles'][primary], p['profiles'][worker], groups/14,
                                       overhead, mixer, expert)
            candidates.append((result['tok_s'], groups))
        rate, groups = max(candidates)
        prefill_rows.append(dict(primary=primary, mixer_speedup=mixer, expert_speedup=expert,
                                 primary_groups=groups, conditional_tok_s=rate))
    return dict(sources=[pi, ti, bi], quality_qualified=False, calibrated=False,
                scope='Historical REAP224 aggregate-phase sensitivity, fixed acceptance; future speedups assumed',
                decode=decode_rows, prefill=prefill_rows,
                decode_timed_tokens=timed_tokens,
                prefill_residual_ms_per_moe=overhead,
                limitations=['Host service includes worker waits; CPU flow must not be added to it.',
                             'Serial phase scopes reproduce a diagnostic, not a scheduling improvement.',
                             'Proportional FFN scaling includes shared work and is optimistic.',
                             'The CUDA-primary residual is applied to AMD-primary without validation.',
                             'No new kernel implementation or quality result is implied.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefill', type=Path, required=True)
    p.add_argument('--decode-trace', type=Path, required=True)
    p.add_argument('--decode-benchmark', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.prefill, a.decode_trace, a.decode_benchmark)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(decode=result['decode'][-1], prefill=max(result['prefill'], key=lambda x:x['conditional_tok_s']))))


if __name__ == '__main__':
    main()
