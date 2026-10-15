#!/usr/bin/env python3
"""Per-slot GPU decomposition of a decode step from a kernel trace (rocprofv3 CSV or nsys cuda_gpu_trace CSV).

A decode slot is delimited by consecutive mailbox wait-add kernels on the primary stream. For every
slot the kernels in between are summed by category (mixer, router/publish, resident experts, wait-add,
draft/head, other) and the wall period between wait-add starts is reported, so the gap (period minus
kernel time) is the GPU idle time per slot. Prefill kernels (before the first wait-add) are ignored.
Output is JSON with per-category means per slot and the slot period distribution; it is a kernel-time
decomposition, not a throughput measurement.
"""
import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CATEGORIES = [
    ('wait_add', re.compile(r'wait_add')),
    ('resident', re.compile(r'gu_kernel|down_kernel|Q2KTraits|Q3KTraits|expert_grouped|q23|canon_expert|post_rows', re.I)),
    ('publish', re.compile(r'router|publish|mailbox|route_|topk|top_k|selection|sigmoid|moe_prefix', re.I)),
    ('mixer', re.compile(r'kda_|mla|kv_|attn|rope|indexer|norm|mhc|hc_|collapse|gdn|conv|Q4KTraits|Q80Block|Q8|mmvq|^Cijk|gemm|wmma|cutlass|sm[0-9]+_', re.I)),
    ('copy', re.compile(r'fillBuffer|copyBuffer|memcpy|memset|CopyBuffer', re.I)),
]


def category(name):
    for cat, rx in CATEGORIES:
        if rx.search(name):
            return cat
    return 'other'


def read_rocprof(path):
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('Kind', 'KERNEL_DISPATCH') != 'KERNEL_DISPATCH':
                continue
            yield row['Kernel_Name'], int(row['Start_Timestamp']), int(row['End_Timestamp'])


def nsys_lanes(path):
    with open(path, newline='') as f:
        return {(row.get('Device'), row.get('Stream')) for row in csv.DictReader(f)}


def read_nsys(path, device=None, stream=None):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if device is not None and 'Device' not in reader.fieldnames or stream is not None and 'Stream' not in reader.fieldnames:
            raise ValueError('trace has no requested device/stream identity column')
        for row in reader:
            if device is not None and row.get('Device') != str(device): continue
            if stream is not None and row.get('Stream') != str(stream): continue
            start = row.get('Start (ns)') or row.get('Start(ns)') or row.get('Start')
            dur = row.get('Duration (ns)') or row.get('Duration(ns)') or row.get('Duration')
            name = row.get('Name') or row.get('Kernel Name') or ''
            if not start or not dur or not name:
                continue
            s = int(float(start)); yield name, s, s + int(float(dur))


def analyze(kernels, min_slot_us=50):
    kernels = sorted(kernels, key=lambda k: k[1])
    waits = [i for i, k in enumerate(kernels) if 'wait_add' in k[0]]
    if len(waits) < 10:
        raise SystemExit('fewer than 10 wait-add kernels: not a decode trace')
    slots = []
    for a, b in zip(waits, waits[1:]):
        period_ns = kernels[b][1] - kernels[a][1]
        if period_ns < min_slot_us * 1000 or period_ns > 50_000_000:
            continue  # draft/head/tail boundaries and trial gaps are not slots
        sums = defaultdict(float); counts = defaultdict(int); names = defaultdict(float)
        for name, s, e in kernels[a:b]:
            c = category(name); sums[c] += (e - s) / 1e6; counts[c] += 1; names[name] += (e - s) / 1e6
        slots.append(dict(period_ms=period_ns / 1e6, kernel_ms=sum(sums.values()), by_category={k: round(v, 4) for k, v in sums.items()},
                          launches=sum(counts.values()), names=names))
    if not slots:
        raise SystemExit('no slots found')
    cats = sorted({c for s in slots for c in s['by_category']})
    mean = lambda key: statistics.fmean(s[key] for s in slots)
    per_cat = {c: statistics.fmean(s['by_category'].get(c, 0.0) for s in slots) for c in cats}
    top = defaultdict(float)
    for s in slots:
        for n, v in s['names'].items():
            top[n] += v
    top_list = sorted(top.items(), key=lambda x: -x[1])[:25]
    periods = sorted(s['period_ms'] for s in slots)
    return dict(slots=len(slots), period_ms=dict(mean=mean('period_ms'), p50=periods[len(periods) // 2], p90=periods[int(0.9 * len(periods))]),
                kernel_ms_mean=mean('kernel_ms'), gap_ms_mean=mean('period_ms') - mean('kernel_ms'),
                launches_mean=mean('launches'), by_category_ms=per_cat,
                top_kernels=[dict(name=n[:120], total_ms_per_slot=round(v / len(slots), 4)) for n, v in top_list],
                scope='Kernel time per decode slot between consecutive wait-add kernels on the primary stream; the gap is GPU idle or host-blocked time. Not a throughput rate.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('trace'); ap.add_argument('--format', choices=('rocprof', 'nsys'), default='rocprof'); ap.add_argument('--output')
    ap.add_argument('--device'); ap.add_argument('--stream')
    ap.add_argument('--all-streams', action='store_true', help='report each Nsight device/stream independently')
    a = ap.parse_args()
    if a.format!='nsys' and (a.device is not None or a.stream is not None or a.all_streams):
        ap.error('device/stream selection currently applies to Nsight CSV')
    if a.format=='nsys':
        lanes=sorted(nsys_lanes(a.trace),key=lambda x:tuple(str(v) for v in x))
        lanes=[(d,s) for d,s in lanes if (a.device is None or d==a.device) and (a.stream is None or s==a.stream)]
        if not lanes:ap.error('no matching device/stream')
        if a.all_streams:
            reports=[]
            for device,stream in lanes:
                try:report=analyze(list(read_nsys(a.trace,device,stream)))
                except SystemExit as exc:report=dict(not_decode_trace=str(exc))
                reports.append(dict(device=device,stream=stream,**report))
            result=dict(source=a.trace,streams=reports,scope='Independent device/stream reports; no cross-GPU time sum.')
            if a.output:Path(a.output).write_text(json.dumps(result,indent=1)+'\n')
            print(json.dumps({**result,'streams':[{k:v for k,v in row.items() if k!='top_kernels'} for row in reports]},indent=1))
            return
        if len(lanes)>1:ap.error('trace contains multiple device/stream lanes; select --device and --stream or use --all-streams')
        a.device,a.stream=lanes[0]
    kernels = list(read_rocprof(a.trace) if a.format == 'rocprof' else read_nsys(a.trace,a.device,a.stream))
    result = analyze(kernels)
    result['source'] = a.trace
    text = json.dumps(result, indent=1)
    if a.output:
        Path(a.output).write_text(text + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'top_kernels'}, indent=1))
    for t in result['top_kernels'][:14]:
        print(f"{t['total_ms_per_slot']:8.4f}  {t['name'][:110]}")


if __name__ == '__main__':
    sys.exit(main())
