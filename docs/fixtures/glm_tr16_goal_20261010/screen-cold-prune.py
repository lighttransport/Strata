"""Q3 dense model controls, timing screens and component-quality pilots."""
import copy,hashlib,json,os,re,signal,subprocess,sys,time
from pathlib import Path
root=Path('/mnt/nvme02/work/Strata');os.chdir(root)
r=root/'build-tr16-goal-20261010';sys.path.insert(0,str(root/'tools'))
from glm_q2_run_guard import descendants
from glm_tr16_calibrate import digest,input_identity
end=json.loads((r/'goal.json').read_text())['deadline'];python=str(root/'build-exl3-venv/bin/python')
built=json.loads((r/'cold-prune-build-complete.json').read_text())
assert built['cuda_parities_passed'] and built['hip_build_passed']
assert built['route_oracles_passed']
assert '100% tests passed, 0 tests failed out of 20' in (r/'cold-prune-routing-oracles.log').read_text(), 'Require all 20 route tests, not an empty CTest run'
for path,expected in built['sha256'].items():assert digest(path)==expected,path
out=r/'cold-prune-screen'
def identity(pid):
 try:
  f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return f[19] if f[0]!='Z' else None
 except (FileNotFoundError,ProcessLookupError):return None
def run(command,log,seconds=1200):
 if time.time()+180>=end:raise SystemExit('Deadline before '+log.name)
 with log.open('x') as f:
  p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:return p.wait(timeout=min(seconds,end-time.time()-30))
  except subprocess.TimeoutExpired:
   owned={child:identity(child) for child in descendants(p.pid)}
   for child,stamp in owned.items():
    if stamp and identity(child)==stamp:
     try:os.kill(child,signal.SIGINT)
     except ProcessLookupError:pass
   time.sleep(10)
   for child,stamp in owned.items():
    if stamp and identity(child)==stamp:
     try:os.kill(child,signal.SIGKILL)
     except ProcessLookupError:pass
   p.wait();raise
state=json.loads((r/'pipeline-v10-state.json').read_text())
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
rows=[]
for name,c,reference in cases:
 cfg=out/(name+'.json');cfg.write_text(json.dumps(c,indent=2)+'\n');dest=out/name
 print('START_COLD_PRUNE_SCREEN',name,flush=True)
 code=run([python,'tools/glm_tr16_calibrate.py','--config',str(cfg),'--prompt','build-tr16-20261009/prompts2k/train-code.ids','--output',str(dest),'--tokens','128','--trials','1','--cases','baseline','--wait-idle-seconds','900','--timeout','900'],out/(name+'-driver.log'))
 result=dest/'000-baseline.result.json'
 if not result.exists():raise RuntimeError('Missing Q22 screen '+name)
 x=json.loads(result.read_text());row={'name':name,'exit_code':code,'clean':x['clean'],'decode_tok_s':x['measurements']['median_decode_tok_s'],'prefill_tok_s':x['measurements']['median_prefill_tok_s'],'quality_qualified':False,'config':str(cfg),'memory':x['memory']}
 if not row['clean']:
  rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');raise RuntimeError('Unclean Q22 screen '+name)
 log=(dest/'000-baseline.log').read_text()
 if name.startswith('cold'):
  pruned=re.findall(r'ROUTE_PRUNE pruned=(\d+) routes=(\d+)',log);assert pruned and int(pruned[-1][0])>0, 'Pruning branch not exercised'
  row['pruned_routes']=int(pruned[-1][0]);row['total_routes']=int(pruned[-1][1])
 tokens=(dest/'000-baseline.stdout').read_text().split()
 if len(tokens)!=256:raise RuntimeError('Incomplete Q22 tokens '+name)
 if reference:
  assert tokens==(reference/'000-baseline.stdout').read_text().split(),'Control changed outputs '+name
  row['control_tokens_identical']=256
 else:
  row['weight_variant']='Q22 unchanged from same-day Q22 controls';row['input_unchanged_from_control']=input_identity(c)==input_identity(cases[0][1]);assert row['input_unchanged_from_control']
  row['input_identity']=input_identity(c)
  if c.get('route_affinity_profile_sha256'):
   assert digest(c['env']['STRATA_GLM_ROUTE_AFFINITY_PROFILE'])==c['route_affinity_profile_sha256']
   row['route_affinity_profile_sha256']=c['route_affinity_profile_sha256']
  assert 'EXPERT_PACK source='+c['expert_pack'] in log, 'Wrong expert pack'
 rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print('DONE_COLD_PRUNE_SCREEN',json.dumps({k:v for k,v in row.items() if k!='memory'}),flush=True)
(out/'complete.json').write_text(json.dumps({'screen_cases':len(rows),'quality_qualified':False,'scope':'Speed screening only; Q22 precision unchanged from Q22 controls; routing changes require quality evaluation'},indent=2)+'\n')
