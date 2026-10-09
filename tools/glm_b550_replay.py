"""Summarize measured residency and optimistic policy replay on one routing request.

Policy replay admits immediately and omits the engine's three-step delay and
staging limits. It is a screening bound, not an implemented-policy speed claim.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent/'sim'))
import events
import model
from glm_low_memory_bench import parse_log


def summarize(path, request, capacity_mib, dram_gbps=28.0, h2d_gbps=27.21, engine_log=None):
    pack = model.pack('reap50_q23')
    rows = [e for e in events.read_events(path) if e['request'] == request and e['layer'] in pack.formats]
    if not rows:
        raise ValueError('no main-model executions for this request')
    initial = events.read_initial(path, request)
    total = hit = 0
    for e in rows:
        wanted = {x for row in e['routes'] for x in row}
        resident = set(e.get('resident', []))
        size = pack.expert_bytes(e['layer'])
        total += len(wanted)*size
        hit += len(wanted & resident)*size
    policies = []
    for policy in ('static','lru','protected'):
        r = events.replay(rows, pack, capacity_mib, policy, initial)
        cpu_bytes = r['requested_bytes']-r['hit_bytes']
        # Pinned staging reads the original, writes staging, then DMA reads it.
        host_bytes = cpu_bytes + 3*r['upload_bytes']
        r['staged_host_traffic_estimate_bytes'] = host_bytes
        r['standalone_upload_time_s'] = r['upload_bytes']/(h2d_gbps*1e9)
        r['staged_memory_time_estimate_s'] = max(host_bytes/(dram_gbps*1e9),r['upload_bytes']/(h2d_gbps*1e9))
        policies.append(r)
    observed=dict(requested_bytes=total,hit_bytes=hit,cpu_bytes=total-hit,hit_share=hit/total)
    if engine_log:
        metrics=parse_log(engine_log.read_text())
        traces=metrics.get('step_trace',[])
        sizes={pack.expert_bytes(layer) for layer in pack.formats}
        if len(traces)!=1 or len(sizes)!=1:
            raise ValueError('upload-count conversion requires one trace and uniform expert bytes')
        uploads=int(traces[0]['tier_swaps'])*next(iter(sizes))
        counted=sum(x.get('bytes',0) for x in metrics.get('cpu_expert',[]))
        if counted!=total-hit:raise ValueError('routing and CPU byte counters disagree')
        host_bytes=total-hit+3*uploads
        observed.update(upload_bytes=uploads,cpu_counters_match=True,staged_host_traffic_estimate_bytes=host_bytes,
                        staged_memory_time_estimate_s=max(host_bytes/(dram_gbps*1e9),uploads/(h2d_gbps*1e9)))
    return dict(request=request,executions=len(rows),observed=observed,policies=policies,
                caveat='Instant admission; staging assumes source read + staging write + DMA read reach DRAM (cache reuse can reduce traffic). No promotion delay, staging capacity, GPU compute or launch costs. Not a throughput forecast.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('events',type=Path);p.add_argument('--request',type=int,required=True)
    p.add_argument('--capacity-mib',type=float,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--engine-log',type=Path)
    a=p.parse_args();result=summarize(a.events,a.request,a.capacity_mib,engine_log=a.engine_log)
    text=json.dumps(result,indent=2)+'\n'
    if a.output:a.output.write_text(text)
    print(text,end='')
