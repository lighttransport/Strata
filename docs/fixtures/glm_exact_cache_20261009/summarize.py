import hashlib,json,re,statistics
from pathlib import Path
root=Path(__file__).resolve().parent

def analyze(name):
    r=json.loads((root/f'{name}.result.json').read_text()); log=(root/f'{name}.log').read_text()
    steps=[]
    for part in log.split('DECODE_TRIAL index=')[1:]:
        ms=[float(v) for v in re.findall(r'DECODE_STEP index=\d+ ms=([\d.]+)',part)]
        cpu=re.findall(r'CPU_EXPERT .*?layer_flow_ms=([\d.]+) bytes=(\d+)',part)
        steps.append(dict(steps=len(ms),median_ms=statistics.median(ms),p95_ms=sorted(ms)[int(.95*len(ms))],last128_tok_s=128000/sum(ms[-128:]) if len(ms)>=128 else None,cpu_layer_flow_ms=float(cpu[-1][0]) if cpu else None))
    tail=[s for s in r['samples'] if s['seconds']>r['wall_seconds']-50]
    ids=(root/f'{name}.stdout').read_text().split()
    return dict(measurements=r['measurements'],decode_latency=steps,complete=r['complete'],clean=r['clean'],rejected=r['rejected'],wall_seconds=r['wall_seconds'],peak_ram_gib=r['cgroup_after']['memory.peak']/2**30,minimum_gpu_free_mib=r['minimum_gpu_free_mib'],memory_events=r['cgroup_after']['memory.events'],max_swap_bytes=max(s['swap_bytes'] for s in r['samples']),last50s_peak_ram_gib=max(s['memory_current'] for s in tail)/2**30,last50s_process_read_bytes=tail[-1]['process_io']['read_bytes']-tail[0]['process_io']['read_bytes'],gpu_cache=re.findall(r'DECODE_CACHE prefix_trained[^\n]+',log),cache_stats=re.findall(r'EXACT_CACHE_STATS[^\n]+',log),output_count=len(ids),repeat_output_equal=ids[:384]==ids[384:] if len(ids)==768 else None,interference=r['interference'])

report={n:analyze(n) for n in ('check','code','baseline','eviction-final')}
check_rows=(root/'check.routes.csv').read_text().splitlines()
report['parity']={'eight_steps_all_logits_bitwise_equal':'EXACT_CACHE_CHECK steps=8 all_logits=bitwise_equal' in (root/'check.log').read_text(),'eight_steps_route_rows_equal':check_rows[:336]==check_rows[336:672]}
report['comparison']={'output_ids_equal':(root/'code.stdout').read_bytes()==(root/'baseline.stdout').read_bytes(),'route_rows_equal':(root/'code.routes.csv').read_bytes()==(root/'baseline.routes.csv').read_bytes(),'gpu_cache_equal':report['code']['gpu_cache']==report['baseline']['gpu_cache']}
eviction_rows=(root/'eviction-check.routes.csv').read_text().splitlines()
report['parity']['final_forced_eviction_logits_equal']='EXACT_CACHE_CHECK steps=8 all_logits=bitwise_equal' in (root/'eviction-final.log').read_text()
report['parity']['final_forced_eviction_routes_equal']=eviction_rows[:336]==eviction_rows[336:672]
report['final_binary_sha256']='6f4951b463803750e6f77c8dda51cf5eab368c9314b516567aac0bc7fd91958b'
report['binary_sha256']='f57c3cd3f69023c3a63686f39186e17632d0e6d35c05979bb1c47a4a14810d02'
report['notes']=['Linux B550 Ryzen 9 3950X / RX 9070 XT; REAP50 Q3_K_M; 60 GiB cgroup, swap disabled.', 'Code and baseline each run one untimed prefill, two timed prefills, then two 383-step decode passes from the same checkpoint.', 'Exact cache starts empty for decode pass 0 and retains entries for repeated pass 1. Prefill is unchanged.', 'One process per arm; repeated identical output is a warm best case, not a new conversation or long-context qualification.']
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['comparison']))
