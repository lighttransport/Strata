"""Audit published routing slots for possible exact-width expert-kernel coverage.

This does not infer CPU placement, accepted speculative tokens, or model speed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from .glm_traces import read_routing_events


def _routes(event):
    routes = event.get('routes')
    if not isinstance(routes, list) or not routes:
        raise ValueError('routes must contain at least one token list')
    for ids in routes:
        if not isinstance(ids, list) or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError('expert IDs must be nonnegative integers')
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate expert within one token')
    return routes


def _summary(histogram, width):
    jobs = sum(histogram.values())
    routed_rows = sum(n * count for n, count in histogram.items())
    eligible = histogram[width]
    return dict(job_width_histogram=dict(sorted(histogram.items())), jobs=jobs,
                routed_token_expert_rows=routed_rows, exact_width_jobs=eligible,
                exact_width_job_fraction=eligible/jobs if jobs else 0.0,
                exact_width_route_fraction=width*eligible/routed_rows if routed_rows else 0.0)


def analyze(events, width=4):
    if type(width) is not int or width < 1:
        raise ValueError('width must be a positive integer')
    phases = {}
    for event in events:
        routes = _routes(event)
        phase = event.get('phase', 'unspecified')
        if not isinstance(phase, str):
            raise ValueError('phase must be a string')
        p = phases.setdefault(phase, dict(events=0, widths=Counter(), all=Counter(),
                                         nonresident=Counter(), resident_events=0))
        p['events'] += 1
        p['widths'][len(routes)] += 1
        # Group only inside one published slot, never across layers/rounds/trials.
        counts = Counter(expert for ids in routes for expert in ids)
        p['all'].update(counts.values())
        if 'resident' in event:
            resident = event['resident']
            if not isinstance(resident, list) or any(type(x) is not int or x < 0 for x in resident):
                raise ValueError('resident must be a list of nonnegative expert IDs')
            resident = set(resident)
            p['resident_events'] += 1
            p['nonresident'].update(n for expert, n in counts.items() if expert not in resident)
    result = {}
    for phase, p in phases.items():
        result[phase] = dict(events=p['events'], event_width_histogram=dict(sorted(p['widths'].items())),
                             all_routes=_summary(p['all'], width),
                             resident_metadata_events=p['resident_events'],
                             nonresident_proxy=_summary(p['nonresident'], width)
                             if p['resident_events'] == p['events'] else None)
    return dict(exact_kernel_width=width, phases=result, limitations=[
        'Counts group routes within each published slot; downstream slicing may change job widths',
        'Nonresident routes are only a proxy: GPU staging, CPU assignment and remote partitioning are not recorded here',
        'Route fractions are not fractions of measured service time and cannot establish a model speedup',
        'Target and draft phases are separate; repeated trials and rejected speculative work are retained'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', type=Path, nargs='+')
    parser.add_argument('--width', type=int, default=4)
    args = parser.parse_args()
    reports = []
    for path in args.traces:
        reports.append(dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            coverage=analyze(read_routing_events(path), args.width)))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
