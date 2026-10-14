#!/usr/bin/env python3
"""Summarize mixed decode host timelines without calling worker waits CPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def summarize(path):
    raw = path.read_bytes()
    reports = []
    for line in raw.decode(errors='replace').splitlines():
        if line.startswith('STEP_TRACE '):
            values = {k: float(v) for k, v in re.findall(r'(\w+)=([\d.eE+-]+)', line)}
            # These four scopes partition the traced host step. Enqueue, sync,
            # planning and readback are nested and must not be added again.
            total = sum(values[k] for k in ('head_ms', 'cpu_ms', 'between_ms', 'tail_ms'))
            reports.append(dict(raw=values, host_step_total_ms=total,
                                host_service_share=values['cpu_ms']/total if total else 0,
                                mixer_wait_share=values['between_ms']/total if total else 0,
                                layers=[]))
        elif line.startswith('STEP_TRACE_SLOTS ') and reports:
            group = int(re.search(r'group=(\d+)', line)[1])
            for layer, kind, wait, service in re.findall(r'(\d+)([km]):(\d+)/(\d+)', line):
                reports[-1]['layers'].append(dict(group=group, layer=int(layer),
                    mixer='kda' if kind == 'k' else 'mla',
                    wait_us_per_step=int(wait), service_us_per_step=int(service)))
    for r in reports:
        r['largest_service_slots'] = sorted(r['layers'], key=lambda x:x['service_us_per_step'], reverse=True)[:10]
        r['largest_wait_slots'] = sorted(r['layers'], key=lambda x:x['wait_us_per_step'], reverse=True)[:10]
    return dict(source=str(path), sha256=hashlib.sha256(raw).hexdigest(), reports=reports,
                scope='Host service includes CPU expert work, worker ownership RPCs and worker completion waits. It is not pure CPU execution.',
                limitations=['Instrumented timing, not a qualified performance rate.',
                             'Between-slot waits are exposed GPU/dependency waits, not total GPU kernel time.',
                             'Reports may include warmup; pair their order with the benchmark records.'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('log', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    result = summarize(a.log)
    if not result['reports']:
        ap.error('no completed STEP_TRACE reports in this log')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result['reports'][-1].items() if k != 'layers'}, indent=2))


if __name__ == '__main__':
    main()
