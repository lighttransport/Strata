#!/usr/bin/env python3
"""Build a disjoint resident-cache candidate from an explicit training request.

The output is simulator input, not a deployable runner configuration. Kernel
costs must be supplied independently. No held-out request is used for planning.
"""
import argparse
import hashlib
import json
from pathlib import Path
from sim.heterogeneous import plan_resident_cache


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-routes',type=Path,required=True)
    p.add_argument('--request',type=int,required=True)
    p.add_argument('--costs',type=Path,required=True)
    p.add_argument('--hardware',type=Path,required=True)
    p.add_argument('--allocations',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    groups=[];seen=set()
    for line in a.training_routes.read_text().splitlines():
        row=json.loads(line)
        if row.get('request')!=a.request or 'routes' not in row or row.get('phase')=='cache_probe':
            continue
        if row['event'] in seen:
            raise ValueError('duplicate routing event')
        seen.add(row['event']);groups.append(row)
    if not groups:
        p.error('no execution groups in the selected training request')
    result=plan_resident_cache(groups,json.loads(a.hardware.read_text()),
                               json.loads(a.costs.read_text()),json.loads(a.allocations.read_text()))
    result['training_request']=a.request
    result['training_groups']=len(groups)
    result['sources']=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                       for path in (a.training_routes,a.costs,a.hardware,a.allocations)]
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(training_groups=len(groups),resident_experts={g:len(v) for g,v in result['initial'].items()})))


if __name__=='__main__':
    main()
