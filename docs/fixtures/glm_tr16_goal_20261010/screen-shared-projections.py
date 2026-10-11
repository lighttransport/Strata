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
out=r/'shared-projection-screen'
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
for name,source,flag,reference in [
 ('00-q4-control',r/'dense-q3-screen/01-q4-control.json','0',None),
 ('01-q4-shared',r/'dense-q3-screen/01-q4-control.json','1',out/'00-q4-control'),
 ('02-q3-depth3-shared',r/'q3-gpu-v2-configs/q3-depth3-soft2-cpu.json','1',r/'q3-gpu-v2/q3-depth3-soft2-cpu')]:
 c=json.loads(source.read_text());c.update(exe=str(r/'gpu-cache-v2-cuda/strata-glm-decode'),source_patch=str(r/'gpu-cache-v2.patch'),model_name=name)
 c['env']['STRATA_GLM_SPLIT_PROJECTIONS']=flag
 c['experimental_role']='Shared projection screen; output and quality unqualified'
 cases.append((name,c,reference))
assert b'STRATA_GLM_ROUTE_AFFINITY_PROFILE' in (r/'gpu-cache-v2-cuda/strata-glm-decode').read_bytes()
profiles=json.loads((r/'affinity-profiles-prepared.json').read_text())['profiles']
for i,profile in enumerate(profiles,3):
 assert digest(profile['path'])==profile['sha256']
 c=copy.deepcopy(cases[0][1]);name=f"{i:02d}-q4-profile-{profile['name']}"
 c['model_name']=name;c['env']['STRATA_GLM_ROUTE_AFFINITY_PROFILE']=profile['path']
 c['route_affinity_profile_sha256']=profile['sha256']
 c['experimental_role']='Layer-profiled affinity on full TR16 Q23; B550 speed/quality not transferable'
 cases.append((name,c,None))
rows=[]
for name,c,reference in cases:
 cfg=out/(name+'.json');cfg.write_text(json.dumps(c,indent=2)+'\n');dest=out/name
 print('START_SHARED_PROJECTION',name,flush=True)
 code=run([python,'tools/glm_tr16_calibrate.py','--config',str(cfg),'--prompt','build-tr16-20261009/prompts2k/train-code.ids','--output',str(dest),'--tokens','128','--trials','1','--cases','baseline','--wait-idle-seconds','900','--timeout','900'],out/(name+'-driver.log'))
 x=json.loads((dest/'000-baseline.result.json').read_text());assert code==0 and x['clean'],'Unclean shared projection run'
 tokens=(dest/'000-baseline.stdout').read_text().split();assert len(tokens)==256
 row=dict(name=name,config=str(cfg),decode_tok_s=x['measurements']['median_decode_tok_s'],clean=True,quality_qualified=False)
 if c.get('route_affinity_profile_sha256'):
  assert digest(c['env']['STRATA_GLM_ROUTE_AFFINITY_PROFILE'])==c['route_affinity_profile_sha256']
  row['route_affinity_profile_sha256']=c['route_affinity_profile_sha256']
 if reference:
  expected=(reference/'000-baseline.stdout').read_text().split();assert len(expected)==256
  row['all_tokens_identical']=tokens==expected
  old=json.loads((reference/'000-baseline.result.json').read_text())['measurements']['median_decode_tok_s']
  row['reference_decode_tok_s']=old;row['nominal_rate_change_percent']=100*(row['decode_tok_s']/old-1)
 rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
 print('DONE_SHARED_PROJECTION',json.dumps(row),flush=True)
# Score the fastest layer profile and the gentler profile before switching packs.
profile_rows=[v for v in rows if 'route_affinity_profile_sha256' in v]
selected={v['name']:v for v in [max(profile_rows,key=lambda v:v['decode_tok_s']),next(v for v in profile_rows if v['name'].endswith('-gentle'))]}
baseline=r/'renorm-quality-pilot/00-renorm-default/result.json';b=json.loads(baseline.read_text());quality=[]
assert b['complete'] and b['expected_tokens']==2048
for name,row in selected.items():
 if time.time()+3600>=end:raise SystemExit('Reserve final hour before affinity quality pilot')
 c=json.loads(Path(row['config']).read_text());c['env'].update(STRATA_GLM_EVAL_WIDTH='1',STRATA_GLM_MTP_ACCEPT_MARGIN='0')
 assert input_identity(c)==b['inputs']
 assert digest(c['env']['STRATA_GLM_ROUTE_AFFINITY_PROFILE'])==c['route_affinity_profile_sha256']
 cfg=out/(name+'-quality.json');cfg.write_text(json.dumps(c,indent=2)+'\n');dest=out/(name+'-quality')
 print('START_AFFINITY_PROFILE_QUALITY',name,flush=True)
 code=run([python,'tools/glm_tr16_quality.py','--config',str(cfg),'--output',str(dest),'--regression-config',str(r/'renorm-quality-pilot/regression.json'),'--baseline',str(baseline),'--wait-idle-seconds','900','--record-cpu-interference'],out/(name+'-quality-driver.log'),1800)
 x=json.loads((dest/'result.json').read_text());assert x['complete'] and x['metrics']['tokens']==2048
 assert digest(c['env']['STRATA_GLM_ROUTE_AFFINITY_PROFILE'])==c['route_affinity_profile_sha256']
 quality.append(dict(name=name,complete=True,exit_code=code,metrics=x['metrics'],failures=x.get('failures',[]),ppl_increase_percent=100*(x['metrics']['perplexity']/b['metrics']['perplexity']-1),top1_drop_percentage_points=100*(b['metrics']['top1_agreement']-x['metrics']['top1_agreement']),profile_sha256=c['route_affinity_profile_sha256'],quality_qualified=False))
 (out/'quality-summary.json').write_text(json.dumps(quality,indent=2)+'\n')
 print('DONE_AFFINITY_PROFILE_QUALITY',json.dumps(quality[-1]),flush=True)
