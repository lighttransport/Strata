#!/usr/bin/env python3
"""Audit frozen FP4 stream evidence; no machine-peak assumptions or decoder forecast."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('rates and counts must be finite and positive')
    return value


def assess(payload_gbs, read_gbs, *, channels=None, memory_mts=None):
    payload_gbs, read_gbs = positive(payload_gbs), positive(read_gbs)
    if (channels is None) != (memory_mts is None):
        raise ValueError('supply both active memory channels and configured MT/s')
    # Native block: 64 weights, 32 code bytes + 4 scale bytes. NT1 only.
    intensity = 2 * 64 / 36
    peak = None
    if channels is not None:
        channels = positive(channels)
        if not channels.is_integer():
            raise ValueError('active channel count must be an integer')
        # MT/s already counts transfers: do not multiply by two for DDR.
        peak = channels * positive(memory_mts) * 8 / 1000
    return dict(payload_gb_s=payload_gbs, read_control_gb_s=read_gbs,
                payload_over_read_control=payload_gbs/read_gbs,
                nt1_useful_operations_per_packed_byte=intensity,
                nt1_useful_gops_s=payload_gbs*intensity,
                theoretical_memory_gb_s=peak,
                payload_over_theoretical_memory=None if peak is None else payload_gbs/peak,
                # This inverse bound helps identify what RAM specification is needed.
                maximum_theoretical_memory_gb_s_for_90pct_payload=payload_gbs/.9,
                qualification='unverified: payload rates are not physical DRAM counters or retired FLOPS')


def report(source, channels=None, memory_mts=None):
    raw=source.read_bytes()
    evidence=json.loads(raw)
    rows=[]
    for run in evidence['runs']:
        metrics={m['mode']:m for m in run['metrics']}
        if not {'read','nvfp4'} <= metrics.keys():
            raise ValueError('missing paired stream modes')
        read,dot=metrics['read'],metrics['nvfp4']
        if (read['threads'],read['packed_bytes']) != (dot['threads'],dot['packed_bytes']):
            raise ValueError('read and dot geometry differ')
        rows.append(dict(run=run['name'],threads=int(dot['threads']),
                         **assess(dot['payload_GB_s'],read['payload_GB_s'],
                                  channels=channels,memory_mts=memory_mts)))
    return dict(source_sha256=hashlib.sha256(raw).hexdigest(),
                hardware=evidence['hardware'],rows=rows,
                assumptions=dict(active_channels=channels,configured_memory_mts=memory_mts),
                limitations=[
                    'Ratios use process medians, not the stored median of paired pass ratios.',
                    'Useful operations count one multiply and one add per weight at NT1; unpack, quantization, scales and integer instructions are not counted as retired FLOPS.',
                    'Physical DRAM traffic also includes other accesses; cache hits and read controls do not establish theoretical utilization.',
                    'RAM inputs, if supplied, are explicit assumptions unless independently verified. No CPU FLOPS peak is inferred.',
                    'Partial synthetic stream, not complete expert service or full-model decode. No production settings change.'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--channels',type=int)
    parser.add_argument('--memory-mts',type=float)
    args=parser.parse_args()
    try:
        print(json.dumps(report(args.source,args.channels,args.memory_mts),indent=2,allow_nan=False))
    except (ValueError,KeyError,OSError) as error:
        parser.error(str(error))
