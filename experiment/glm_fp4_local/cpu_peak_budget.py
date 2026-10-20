#!/usr/bin/env python3
"""Conditional CPU bandwidth budget; never reports measured peak utilization."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def packed_bytes_per_weight(fmt):
    # Stored scales/minima are included. Q2/Q3 experts have equal-sized
    # gate, up and down matrices: two Q2 matrices for every Q3 matrix.
    return {
        'nvfp4': 36 / 64,
        'q2_k': 84 / 256,
        'q3_k': 110 / 256,
        'q2q3': (2 * 84 + 110) / (3 * 256),
        'q2q3-expanded': (2 * 84 + 114) / (3 * 256),
    }[fmt]


def positive(value):
    x = float(value)
    if not math.isfinite(x) or x <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--memory-mts', required=True, type=positive)
    p.add_argument('--channels', required=True, type=int)
    p.add_argument('--cores', required=True, type=int)
    p.add_argument('--ghz', required=True, type=positive, nargs='+')
    p.add_argument('--target', default=.9, type=positive)
    p.add_argument('--format', choices=['nvfp4', 'q2_k', 'q3_k', 'q2q3', 'q2q3-expanded'],
                   default='nvfp4')
    p.add_argument('--counter-summary', type=Path,
                   help='optional matching-format counter evidence; NVFP4 retains its existing default probe')
    p.add_argument('--weights-per-probe', type=int,
                   help='timed scalar weight products per counter process, including all GU matrices/tokens')
    a = p.parse_args()
    if a.format == 'nvfp4':
        if a.counter_summary is None:
            a.counter_summary = ROOT / 'experiment/glm_fp4_local/results/20261020/fp4-counter-load-audit/summary.json'
        if a.weights_per_probe is None:
            a.weights_per_probe = 2097152 * 4096
    elif a.counter_summary is not None and a.weights_per_probe is None:
        p.error('non-NVFP4 counter evidence requires --weights-per-probe')
    if a.channels < 1 or a.cores < 1 or a.target > 1 or (a.weights_per_probe is not None and a.weights_per_probe < 1):
        p.error('positive channel/core/weight counts and target <= 1 required')
    # A memory channel carries 64 payload bits; decimal GB/s, not GiB/s.
    peak = a.memory_mts * a.channels * 8 / 1000
    target = peak * a.target
    bytes_per_weight = packed_bytes_per_weight(a.format)
    needed_weights = target / bytes_per_weight
    data = json.loads(a.counter_summary.read_text()) if a.counter_summary else {}
    if a.counter_summary is None:
        counters = []
    elif 'results' in data:
        counters = [(row['log'], row['cycles']) for row in data['results']]
    elif 'runs' in data:
        counters = [(row['name'], row['cycles:u']) for row in data['runs']]
    else:
        p.error('counter summary needs results or runs')
    if a.counter_summary is not None and (not counters or any(not math.isfinite(cycles) or cycles <= 0 for _, cycles in counters)):
        p.error('counter summary must contain positive finite cycle counts')
    try:
        evidence_label = str(a.counter_summary.resolve().relative_to(ROOT)) if a.counter_summary else None
    except ValueError:
        evidence_label = a.counter_summary.name
    total_weights = a.weights_per_probe
    samples = []
    for source, cycles in counters:
        c = cycles / total_weights
        samples.append(dict(source=source, whole_probe_cycles_per_weight=c,
                            idealized_packed_gb_s=[dict(ghz=ghz, gb_s=a.cores*ghz/c*bytes_per_weight)
                                                  for ghz in a.ghz]))
    print(json.dumps(dict(
        basis='CONDITIONAL BUDGET, NOT PEAK UTILIZATION OR FULL-MODEL THROUGHPUT',
        assumptions={**vars(a), 'counter_summary': evidence_label}, channel_payload_bits=64,
        theoretical_payload_gb_s=peak, target_payload_gb_s=target,
        packed_bytes_per_weight=bytes_per_weight,
        target_gweights_s=needed_weights,
        max_cycles_per_weight=[dict(ghz=ghz, cycles=a.cores*ghz/needed_weights) for ghz in a.ghz],
        observed_counter_projection=samples,
        limitations=[
            'Q2/Q3 expert mixtures assume equal gate/up/down matrix element counts and NT1; phase throughput can differ',
            'Counter evidence must match the selected format; absent evidence means no measured throughput projection',
            'Memory data rate, populated channels and all-core frequency are explicit assumptions, not verified here',
            'Probe counts include initialization, checking and scalar overhead; no timing subtraction was made',
            'Idealized scaling assumes equal per-core throughput with no shared-resource or scheduling penalty',
            'Only packed weights/scales are counted; activations, outputs, routing and other traffic consume bandwidth too',
            'Integer-dot/unpack instruction throughput is not comparable to advertised FP32 FMA FLOPS',
            'Concurrent host load and the observed counter drift prevent inferring a hardware ceiling']), indent=2))


if __name__ == '__main__':
    main()
