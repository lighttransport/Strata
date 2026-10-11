"""Compare completed protected-route screens; never promote a speed/quality profile."""
import hashlib,json,re
from pathlib import Path
r=Path(__file__).resolve().parent;out=r/'cold-prune-screen'
source=out/'summary.json'
if not source.exists():raise SystemExit('No completed protected-route screens yet')
references={'00-default-control':'q22-screen/02-q22-aff08','01-prune08-control':'q22-screen/04-q22-aff08-prune08','02-cold-zero-control':'q22-screen/02-q22-aff08'}
rows=[];controls=[]
for row in json.loads(source.read_text()):
 if not row['clean']:continue
 folder=out/row['name'];result=folder/'000-baseline.result.json';log=folder/'000-baseline.log';stdout=folder/'000-baseline.stdout'
 x=json.loads(result.read_text());assert x['complete']
 ids=stdout.read_text().split();assert len(ids)==256
 m=x['measurements'];sp=m['speculation'][-1];timing=m['mtp_timing'][-1];assert sp['generated']==128
 text=log.read_text();cache=re.findall(r'DECODE_CACHE prefix_trained=1 slots=(\d+) MiB=([\d.]+) fingerprint=(\d+)',text);assert cache
 flow=re.findall(r'CPU_EXPERT gu_ms=[\d.]+ quant_ms=[\d.]+ down_ms=[\d.]+ layer_flow_ms=([\d.]+) bytes=(\d+)',text);assert flow
 routes=re.findall(r'resident expert entries=(\d+)/(\d+)',text);assert routes
 pruning=re.findall(r'ROUTE_PRUNE pruned=(\d+) routes=(\d+)',text)
 value=dict(name=row['name'],decode_tok_s=row['decode_tok_s'],prefill_tok_s=row['prefill_tok_s'],cache_slots=int(cache[-1][0]),cache_mib=float(cache[-1][1]),cache_fingerprint=cache[-1][2],timed_rounds=sp['rounds'],timed_accepted=sp['accepted'],draft_ms_per_round=timing['draft_ms']/sp['rounds'],verify_ms_per_round=timing['verify_ms']/sp['rounds'],timed_cpu_expert_bytes=int(flow[-1][1]),timed_layer_flow_ms=float(flow[-1][0]),all_trials_resident_routes=int(routes[-1][0]),all_trials_routes=int(routes[-1][1]),all_trials_pruned_routes=int(pruning[-1][0]) if pruning else 0,quality_qualified=False,sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (result,log,stdout)})
 rows.append(value)
 if row['name'] in references:
  reference=r/references[row['name']];before=(reference/'000-baseline.stdout').read_text().split();assert len(before)==256
  refcache=re.findall(r'DECODE_CACHE prefix_trained=1 slots=(\d+) MiB=([\d.]+) fingerprint=(\d+)',(reference/'000-baseline.log').read_text());assert refcache
  controls.append(dict(name=row['name'],reference=str(reference),all_256_tokens_identical=ids==before,cache_identity_equal=cache[-1]==refcache[-1],first_difference_by_trial=[next((i for i,(a,b) in enumerate(zip(before[start:start+128],ids[start:start+128])) if a!=b),None) for start in (0,128)]))
passed=len(controls)==3 and all(v['all_256_tokens_identical'] and v['cache_identity_equal'] for v in controls)
report=dict(scope='Completed 128-token training-prompt screens only. CPU bytes are work counters; route counts cover warm-up plus timed trials. Changed routing changes the generated workload. Short controls do not qualify long generation.',model_controls_passed=passed,quality_qualified=False,controls=controls,rows=rows)
(out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'model_controls_passed':passed,'controls':controls,'rows':[{k:v for k,v in x.items() if k!='sha256'} for x in rows]},indent=2))
