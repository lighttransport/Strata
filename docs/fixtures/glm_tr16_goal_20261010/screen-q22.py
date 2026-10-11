"""Q3 dense model controls, timing screens and component-quality pilots."""
import copy,hashlib,json,os,re,signal,subprocess,sys,time
from pathlib import Path
root=Path('/mnt/nvme02/work/Strata');os.chdir(root)
r=root/'build-tr16-goal-20261010';sys.path.insert(0,str(root/'tools'))
from glm_q2_run_guard import descendants
from glm_tr16_calibrate import digest,input_identity
end=json.loads((r/'goal.json').read_text())['deadline'];python=str(root/'build-exl3-venv/bin/python')
built=json.loads((r/'gpu-cache-v2-build-complete.json').read_text())
assert built['cuda_parities_passed'] and built['hip_build_passed']
for path,expected in built['sha256'].items():assert digest(path)==expected,path
out=r/'q22-screen'
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
state=json.loads((r/'pipeline-v9-state.json').read_text())
assert state['current'] is None and len(state['completed'])==3 and all(v['exit_code']==0 for v in state['completed']), 'Wait for all GPU/Q3 work to finish'
if time.time()+3600>=end:raise SystemExit('Reserve final hour for validation')
out.mkdir(exist_ok=False)
cases=[]
def add(name,source,env=None,cache=None,reference=None):
 c=json.loads(source.read_text());c['exe']=str(r/'gpu-cache-v2-cuda/strata-glm-decode');c['source_patch']=str(r/'gpu-cache-v2.patch')
 c['env'].update(env or {});c['model_name']=name;c['experimental_role']='Q22 expert precision experiment; quality and speed not qualified'
 if cache is not None:c['decode_cache_mib']=cache
 cases.append((name,c,reference))
q4=r/'dense-q3-screen/01-q4-control.json'
add('00-q23-q4-control',q4,reference=r/'dense-q3-screen/01-q4-control')
for name,affinity,share,depth,margin in [('01-q22-aff0',0,0,1,0),('02-q22-aff08',.08,0,1,0),('03-q22-aff08-prune04',.08,.04,1,0),('04-q22-aff08-prune08',.08,.08,1,0),('05-q22-aff08-depth3-soft2',.08,0,3,2)]:
 add(name,q4,{'STRATA_GLM_ROUTE_AFFINITY':str(affinity),'STRATA_GLM_ROUTE_MIN_SHARE':str(share),'STRATA_GLM_ROUTE_RENORMALIZE':'1' if share else '0','STRATA_GLM_MTP_ACCEPT_MARGIN':str(margin),'STRATA_GLM_MTP_GPU_CACHE':'0'})
 cases[-1][1]['expert_pack']='/mnt/nvme01/models/glm53f/q2-cpu-experiment/experts-q22.gguf'
 cases[-1][1]['draft_depth']=depth;cases[-1][1]['mtp_experts']='cpu'
assert Path(cases[-1][1]['expert_pack']).is_file()
# The preceding stage supplies measured profiles and their quality pilots.
profile_quality=json.loads((r/'shared-projection-screen/quality-summary.json').read_text())
feasible={v['name'] for v in profile_quality if v['complete'] and v['ppl_increase_percent']<=10}
profile_speed=json.loads((r/'shared-projection-screen/summary.json').read_text())
control=next(v for v in profile_speed if v['name']=='00-q4-control')
choices=[v for v in profile_speed if v['name'] in feasible and v['decode_tok_s']>control['decode_tok_s']]
if choices:
 chosen=max(choices,key=lambda v:v['decode_tok_s'])
 c=json.loads(Path(chosen['config']).read_text())
 c['expert_pack']=cases[-1][1]['expert_pack'];c['model_name']='06-q22-layer-profile'
 c['precision_routing_reference']=chosen['name']
 c['experimental_role']='Q22 combined with measured layer affinity; no quality qualification'
 assert digest(c['env']['STRATA_GLM_ROUTE_AFFINITY_PROFILE'])==c['route_affinity_profile_sha256']
 cases.append(('06-q22-layer-profile',c,None))
rows=[]
for name,c,reference in cases:
 if name=='01-q22-aff0':
  from glm_tr16_calibrate import wait_idle
  from glm_low_memory_bench import clear_model_cache
  wait_idle(min(900,max(1,end-time.time()-30)))
  # Release only the unused old overlay, never global or unrelated cache.
  released=clear_model_cache({'model':cases[0][1]['expert_pack']})
  (out/'released-q23-cache.json').write_text(json.dumps(released,indent=2)+'\n')
 cfg=out/(name+'.json');cfg.write_text(json.dumps(c,indent=2)+'\n');dest=out/name
 print('START_Q22_SCREEN',name,flush=True)
 code=run([python,'tools/glm_tr16_calibrate.py','--config',str(cfg),'--prompt','build-tr16-20261009/prompts2k/train-code.ids','--output',str(dest),'--tokens','128','--trials','1','--cases','baseline','--wait-idle-seconds','900','--timeout','900'],out/(name+'-driver.log'))
 result=dest/'000-baseline.result.json'
 if not result.exists():raise RuntimeError('Missing Q22 screen '+name)
 x=json.loads(result.read_text());row={'name':name,'exit_code':code,'clean':x['clean'],'decode_tok_s':x['measurements']['median_decode_tok_s'],'prefill_tok_s':x['measurements']['median_prefill_tok_s'],'quality_qualified':False,'config':str(cfg),'memory':x['memory']}
 if not row['clean']:
  rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');raise RuntimeError('Unclean Q22 screen '+name)
 log=(dest/'000-baseline.log').read_text();tokens=(dest/'000-baseline.stdout').read_text().split()
 if len(tokens)!=256:raise RuntimeError('Incomplete Q22 tokens '+name)
 if reference:
  assert tokens==(reference/'000-baseline.stdout').read_text().split(),'Control changed outputs '+name
  row['control_tokens_identical']=256
 else:
  row['weight_variant']='Q22 down projections; differs from Q23 baseline'
  row['input_identity']=input_identity(c)
  if c.get('route_affinity_profile_sha256'):
   assert digest(c['env']['STRATA_GLM_ROUTE_AFFINITY_PROFILE'])==c['route_affinity_profile_sha256']
   row['route_affinity_profile_sha256']=c['route_affinity_profile_sha256']
  assert 'EXPERT_PACK source='+c['expert_pack'] in log, 'Wrong expert pack'
 rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print('DONE_Q22_SCREEN',json.dumps({k:v for k,v in row.items() if k!='memory'}),flush=True)
(out/'complete.json').write_text(json.dumps({'screen_cases':len(rows),'quality_qualified':False,'scope':'Speed screening only; Q22 changes model input identity'},indent=2)+'\n')
