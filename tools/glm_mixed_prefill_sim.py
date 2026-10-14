#!/usr/bin/env python3
"""Sweep a dependency-preserving mixed prefill model from synchronized traces."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics

from glm_mixed_sim import layer_partition, read_prefill_timeline


def mean_batch(batches):
    first = batches[0]
    inventory = [x['layer'] for x in first['layers']]
    if any(x['tokens'] != first['tokens'] or [z['layer'] for z in x['layers']] != inventory for x in batches):
        raise ValueError('inconsistent trace batches')
    return dict(tokens=first['tokens'],
                wall_ms=statistics.mean(x['wall_ms'] for x in batches),
                unassigned_ms=statistics.mean(x['unassigned_ms'] for x in batches),
                layers=[dict(layer=l, mixer_ms=statistics.mean(x['layers'][i]['mixer_ms'] for x in batches),
                             ffn_ms=statistics.mean(x['layers'][i]['ffn_ms'] for x in batches))
                        for i, l in enumerate(inventory)])


def analyze(timeline, benchmark):
    trace = read_prefill_timeline(timeline)
    data = json.loads(benchmark.read_text())
    if data['config']['env'].get('STRATA_GLM_TASK_TRACE_SYNC') != '1':
        raise ValueError('synchronized phase trace is required')
    profiles, offset = {}, 0
    for record in data['results']:
        if record['returncode'] != 0:
            raise ValueError('failed measurement')
        measured = record['measurement']['measurements']['prefill']
        # The engine runs one unreported prefill warmup when --bench > 1.
        # Match the explicit command, then discard that trace batch.
        repetitions = [int(x.split('=',1)[1]) for x in record['command'] if x.startswith('--bench=')]
        if repetitions != [len(measured)]:
            raise ValueError('unrecognized prefill repetition protocol')
        if len(measured) > 1:
            offset += 1
        batches = trace[offset:offset+len(measured)]
        if not batches or len(batches) != len(measured):
            raise ValueError('missing trace batches')
        for b, m in zip(batches, measured):
            if b['tokens'] != m['tokens'] or b['tokens'] != 8192 or len(b['layers']) != 45:
                raise ValueError('requires complete 8K/45-layer traces')
            if abs(b['wall_ms'] - m['milliseconds']) > max(5, .01*m['milliseconds']):
                raise ValueError('trace and measured batch timing do not match')
        case = record['case']
        if case in profiles:
            raise ValueError('one process per case expected')
        profiles[case] = mean_batch(batches)
        offset += len(measured)
    if offset != len(trace) or not {'cuda', 'hip', 'cuda-hip'} <= profiles.keys():
        raise ValueError('requires complete CUDA, HIP, mixed profiles and no extra batches')
    mixed_record = next(r for r in data['results'] if r['case'] == 'cuda-hip')
    uncorrected = layer_partition(profiles['cuda'], profiles['hip'], float(mixed_record['fraction']))
    missing_ms = profiles['cuda-hip']['wall_ms'] - uncorrected['milliseconds']
    moe_layers = sum(x['layer'] >= 3 for x in profiles['cuda']['layers'])
    equivalent_overhead = max(0, missing_ms / moe_layers)
    rows = []
    for primary, mixer, expert, comm in itertools.product(
            ['cuda', 'hip'], [1, 1.25, 1.5, 2], [1, 1.25, 1.5, 2], [0, 10, 20, equivalent_overhead]):
        worker = 'hip' if primary == 'cuda' else 'cuda'
        candidates = []
        # Current REAP224 worker partitions complete groups of 16 experts.
        for groups in range(1, 14):
            result = layer_partition(profiles[primary], profiles[worker], groups/14,
                                     communication_ms=comm, mixer_speedup=mixer, expert_speedup=expert)
            candidates.append((result['tok_s'], groups))
        rate, groups = max(candidates)
        rows.append(dict(primary=primary, primary_groups=groups, mixer_speedup=mixer,
                         expert_speedup=expert, communication_ms_per_moe_layer=comm,
                         conditional_tok_s=rate, reaches_1000=rate >= 1000))
    return dict(scope='Conditional per-layer expert partition; assumes linear FFN cost scaling.',
                sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [timeline, benchmark]],
                mixed_calibration=dict(measured_ms=profiles['cuda-hip']['wall_ms'],
                                       uncorrected_ms=uncorrected['milliseconds'], residual_ms=missing_ms,
                                       equivalent_overhead_ms_per_moe=equivalent_overhead,
                                       scope='Residual includes non-scaling FFN work, transport, contention and model error; not measured transfer time. Applying it to HIP-primary is unvalidated.'),
                profiles=profiles, scenarios=rows,
                limitations=['Instrumented single-process traces, not qualified throughput.',
                             'Full FFN cost includes shared experts and staging; proportional scaling is optimistic.',
                             'Communication cost is swept, not fitted.',
                             'Per-layer joins preserve serial mixer dependencies; no quality prediction.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timeline', type=Path, required=True)
    p.add_argument('--benchmark', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.timeline, a.benchmark)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['mixed_calibration']))
    for r in result['scenarios']:
        if r['communication_ms_per_moe_layer'] == result['mixed_calibration']['equivalent_overhead_ms_per_moe'] and (r['mixer_speedup'], r['expert_speedup']) in [(1,1),(1,2),(2,1),(1.5,1.5),(2,2)]:
            print(json.dumps(r))


if __name__ == '__main__':
    main()
