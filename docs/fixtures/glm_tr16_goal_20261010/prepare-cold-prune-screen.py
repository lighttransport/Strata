from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'screen-q22.py').read_text()
a=s.index('state=json.loads');b=s.index('rows=[]',a)
preamble='''state=json.loads((r/'pipeline-v10-state.json').read_text())
assert state['current'] is None and len(state['completed'])==3 and all(v['exit_code']==0 for v in state['completed'])
assert time.time()+3600<end, 'Reserve final hour for validation'
out.mkdir(exist_ok=False)
cases=[]
def add(name,source,env=None,reference=None):
 c=json.loads(source.read_text());c['exe']=str(r/'cold-prune-cuda/strata-glm-decode');c['source_patch']=str(r/'cold-prune.patch')
 c['env'].update(env or {});c['model_name']=name;c['experimental_role']='Positive-bonus route protection; fixed-cache Q22 research only; no quality qualification'
 assert c['expert_pack'].endswith('/experts-q22.gguf') and c['decode_cache_mib']==6656
 assert c['env']['STRATA_GLM_ROUTE_AFFINITY']=='0.08' and not c['env'].get('STRATA_GLM_ROUTE_AFFINITY_PROFILE')
 cases.append((name,c,reference))
base=r/'q22-screen/02-q22-aff08.json'
add('00-default-control',base,reference=r/'q22-screen/02-q22-aff08')
add('01-prune08-control',r/'q22-screen/04-q22-aff08-prune08.json',reference=r/'q22-screen/04-q22-aff08-prune08')
# With no threshold, turning on protection must preserve all target tokens.
add('02-cold-zero-control',base,{'STRATA_GLM_ROUTE_PRUNE_COLD_ONLY':'1'},reference=r/'q22-screen/02-q22-aff08')
for share in (.08,.12):
 for renorm in (0,1):
  name='cold'+str(share)+'-renorm'+str(renorm)
  add(name,base,{'STRATA_GLM_ROUTE_PRUNE_COLD_ONLY':'1','STRATA_GLM_ROUTE_MIN_SHARE':str(share),'STRATA_GLM_ROUTE_RENORMALIZE':str(renorm)})
'''
s=s[:a]+preamble+s[b:]
a=s.index(" if name=='01-q22-aff0':");b=s.index(' cfg=out/',a);s=s[:a]+s[b:]
s=s.replace("'gpu-cache-v2-build-complete.json'","'cold-prune-build-complete.json'").replace("out=r/'q22-screen'","out=r/'cold-prune-screen'")
s=s.replace('START_Q22_SCREEN','START_COLD_PRUNE_SCREEN').replace('DONE_Q22_SCREEN','DONE_COLD_PRUNE_SCREEN')
s=s.replace("row['weight_variant']='Q22 down projections; differs from Q23 baseline'", "row['weight_variant']='Q22 unchanged from same-day Q22 controls';row['input_unchanged_from_control']=input_identity(c)==input_identity(cases[0][1]);assert row['input_unchanged_from_control']")
s=s.replace('Speed screening only; Q22 changes model input identity','Speed screening only; Q22 precision unchanged from Q22 controls; routing changes require quality evaluation')
s=s.replace("assert built['cuda_parities_passed'] and built['hip_build_passed']", "assert built['cuda_parities_passed'] and built['hip_build_passed']\nassert built['route_oracles_passed']\nassert '100% tests passed, 0 tests failed out of 20' in (r/'cold-prune-routing-oracles.log').read_text(), 'Require all 20 route tests, not an empty CTest run'")
s=s.replace("log=(dest/'000-baseline.log').read_text();tokens=", "log=(dest/'000-baseline.log').read_text()\n if name.startswith('cold'):\n  pruned=re.findall(r'ROUTE_PRUNE pruned=(\\d+) routes=(\\d+)',log);assert pruned and int(pruned[-1][0])>0, 'Pruning branch not exercised'\n  row['pruned_routes']=int(pruned[-1][0]);row['total_routes']=int(pruned[-1][1])\n tokens=")
p=r/'screen-cold-prune.py';p.write_text(s);compile(s,str(p),'exec')
print('Prepared seven cold-prune screens, not queued')
