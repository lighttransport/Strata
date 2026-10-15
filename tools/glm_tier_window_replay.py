#!/usr/bin/env python3
"""Compare route and service-window tier scores on a causal routing-event trace.

Rank on the training prefix; evaluate later service calls without looking ahead.
Charge the initial replacement uploads separately. This is a fixed-route byte
screen, not a speed or quality prediction: affinity changes routes with residency.
"""
import argparse
import collections
import json
import re
from pathlib import Path
from gguf_reader import GGUFFile


def compare(train, test, sizes, eligible, budget):
    if not train or not test:
        raise ValueError('nonempty disjoint training and evaluation prefixes required')
    results = {}
    actual_bytes = sum(sizes[row['layer']] * len(set(e for token in row['routes'] for e in token) - set(row['resident']))
                       for row in test)
    for policy in ('routes', 'window'):
        scores = collections.Counter()
        for row in train:
            routed = [e for token in row['routes'] for e in token]
            if policy == 'window': routed = set(routed)
            scores.update((row['layer'], e) for e in routed if row['layer'] in eligible)
        resident = set(); used = 0
        for (layer, expert), _ in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
            if used + sizes[layer] <= budget:
                resident.add((layer, expert)); used += sizes[layer]
        cpu_bytes = sum(sizes[row['layer']] for row in test
                        for e in set(e for token in row['routes'] for e in token)
                        if (row['layer'], e) not in resident)
        # Compare against the residency observed at the boundary, not its union across the prefix.
        boundary = {}
        for row in train: boundary[row['layer']] = set(row['resident'])
        upload = sum(sizes[layer] for layer, e in resident if e not in boundary.get(layer, set()))
        results[policy] = dict(cpu_bytes=cpu_bytes, initial_upload_bytes=upload, slots=len(resident),
                               resident_bytes=used, bytes_with_initial_upload=cpu_bytes + upload)
    baseline = results['routes']['bytes_with_initial_upload']
    results['window_reduction_after_upload'] = 1 - results['window']['bytes_with_initial_upload'] / baseline if baseline else 0
    results['observed_cpu_bytes'] = actual_bytes
    results['implementation_gate'] = results['window_reduction_after_upload'] >= .05
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model', type=Path); ap.add_argument('events', type=Path)
    ap.add_argument('--train-rounds', type=int, default=32)
    ap.add_argument('--cache-mib', type=int, default=6656)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    parts = collections.defaultdict(list); eligible = set()
    for tensor in GGUFFile(a.model).tensors:
        match = re.fullmatch(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight', tensor.name)
        if match and 3 <= int(match[1]) <= 44:
            parts[int(match[1])].append((tensor.expected_bytes() // tensor.shape[2], tensor.type_id))
    if set(parts) != set(range(3, 45)) or any(len(p) != 3 for p in parts.values()):
        raise ValueError('incomplete model expert inventory')
    sizes = {layer: sum(size for size, _ in items) for layer, items in parts.items()}
    eligible = {layer for layer, items in parts.items() if all(type_id in (10, 11) for _, type_id in items)}
    rows = [r for line in a.events.read_text().splitlines() if (r := json.loads(line)).get('phase') == 'verify']
    rounds = sorted({(r['request'], r['round']) for r in rows})
    if a.train_rounds < 1 or a.train_rounds >= len(rounds) or a.cache_mib < 1:
        ap.error('need separate training/evaluation rounds and a positive budget')
    prefix = set(rounds[:a.train_rounds])
    train = [r for r in rows if (r['request'], r['round']) in prefix]
    test = [r for r in rows if (r['request'], r['round']) not in prefix]
    result = compare(train, test, sizes, eligible, a.cache_mib * 1024**2)
    result.update(train_rounds=a.train_rounds, evaluation_rounds=len(rounds) - a.train_rounds,
                  scope='fixed-route static byte screen; upload bytes charged once; no latency or quality prediction')
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
