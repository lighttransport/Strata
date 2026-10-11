"""Q22 cache-capacity follow-up after precision quality pilots; opt-in screens only."""
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
out=r/'q22-cold-combination'
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
coordinators=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:args=(proc/'cmdline').read_bytes().split(b'\0')
 except (OSError,ProcessLookupError):continue
 if any(arg in (b'build-tr16-goal-20261010/pipeline-v11.py',str(r/'pipeline-v11.py').encode()) for arg in args):
  stamp=identity(int(proc.name))
  if stamp:coordinators.append((int(proc.name),stamp))
state=json.loads((r/'pipeline-v11-state.json').read_text())
finished=lambda v:v['current'] is None and len(v['completed'])==3 and all(row['exit_code']==0 for row in v['completed'])
assert finished(state) or len(coordinators)==1, 'Cannot identify the live V11 coordinator'
print('WAIT_FOR_V11_COMBINATION',flush=True)
while not finished(state):
 if any(row['exit_code'] for row in state['completed']):raise RuntimeError('V11 failed; inspect before combination')
 if not coordinators or identity(coordinators[0][0])!=coordinators[0][1]:
  state=json.loads((r/'pipeline-v11-state.json').read_text())
  if finished(state):break
  raise RuntimeError('V11 coordinator ended without successful completion')
 if time.time()+3600>=end:raise SystemExit('Validation reserve before combination')
 time.sleep(5);state=json.loads((r/'pipeline-v11-state.json').read_text())
print('V11_SUCCESS_FOR_COMBINATION',flush=True)
assert time.time()+3600<end, 'Reserve final hour for validation'
quality=json.loads((r/'q22-quality/summary.json').read_text())
qualified_for_screen={v['name'] for v in quality if v['complete'] and v['ppl_increase_percent']<=10}
assert '02-q22-aff08' in qualified_for_screen, 'Do not extend Q22 if the unpruned precision pilot exceeds the moderate PPL budget'
base=json.loads((r/'q22-screen/02-q22-aff08.json').read_text())
assert base['decode_cache_mib']==6656
out.mkdir(exist_ok=False)
rows=[]
cases=[]
# The preceding stage has measured both speed and a same-reference pilot.
# This PPL-only eligibility rule permits exploration, never qualification.
cold_quality=json.loads((r/'cold-prune-quality/summary.json').read_text())
feasible={v['name'] for v in cold_quality if v['complete'] and v['ppl_increase_percent']<=10}
cold_speed=json.loads((r/'cold-prune-screen/summary.json').read_text())
control=next(v for v in cold_speed if v['name']=='00-default-control')
choices=[v for v in cold_speed if v['name'] in feasible and v['clean'] and v['decode_tok_s']>control['decode_tok_s']]
if choices:
 selected=max(choices,key=lambda v:v['decode_tok_s'])
 combined=json.loads(Path(selected['config']).read_text())
 cold_build=json.loads((r/'cold-prune-build-complete.json').read_text())
 assert cold_build['cuda_parities_passed'] and cold_build['hip_build_passed'] and cold_build['route_oracles_passed']
 for path,expected in cold_build['sha256'].items():assert digest(path)==expected,path
 assert str(Path(combined['exe'])) in cold_build['sha256']
 assert json.loads((r/'cold-prune-screen/analysis.json').read_text())['model_controls_passed']
 assert combined['env']['STRATA_GLM_ROUTE_PRUNE_COLD_ONLY']=='1'
 combined['draft_depth']=3;combined['env']['STRATA_GLM_MTP_ACCEPT_MARGIN']='2'
 combined['pruning_pilot_reference']=selected['name']
 cases.append(('cold-pilot-depth3-soft2-cache7424',combined,7424))
assert len(cases)==1, 'No eligible protected-route combination'
for name,source,cache in cases:
 c=copy.deepcopy(source);c['decode_cache_mib']=cache;c['model_name']='q22-'+name
 c['experimental_role']='Q22 cache/drafting follow-up; routing and outputs can change; no quality qualification'
 cfg=out/(name+'.json');cfg.write_text(json.dumps(c,indent=2)+'\n');dest=out/name
 print('START_Q22_CACHE_SCREEN',name,flush=True)
 code=run([python,'tools/glm_tr16_calibrate.py','--config',str(cfg),'--prompt','build-tr16-20261009/prompts2k/train-code.ids','--output',str(dest),'--tokens','128','--trials','1','--cases','baseline','--wait-idle-seconds','900','--timeout','900'],out/(name+'-driver.log'))
 result=dest/'000-baseline.result.json'
 if not result.exists():raise RuntimeError('Missing cache screen result '+name)
 x=json.loads(result.read_text());row=dict(name=name,exit_code=code,clean=x['clean'],decode_tok_s=x['measurements']['median_decode_tok_s'],prefill_tok_s=x['measurements']['median_prefill_tok_s'],config=str(cfg),memory=x['memory'],quality_qualified=False)
 rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
 if code or not row['clean']:raise RuntimeError('Unclean cache screen '+name)
 tokens=(dest/'000-baseline.stdout').read_text().split();assert len(tokens)==256
 reference=r/('q22-screen/05-q22-aff08-depth3-soft2' if c['draft_depth']==3 else 'q22-screen/02-q22-aff08')
 before=(reference/'000-baseline.stdout').read_text().split();assert len(before)==256
 row['token_reference']=str(reference);row['all_256_tokens_identical_to_reference']=tokens==before
 log=(dest/'000-baseline.log').read_text()
 pruned=re.findall(r'ROUTE_PRUNE pruned=(\d+) routes=(\d+)',log);assert pruned and int(pruned[-1][0])>0
 row['pruned_routes']=int(pruned[-1][0]);row['total_routes']=int(pruned[-1][1])
 cache_match=re.findall(r'DECODE_CACHE prefix_trained=1 slots=(\d+) MiB=([\d.]+) fingerprint=(\d+)',log);assert cache_match
 row['speculation']=x['measurements']['speculation'];row['mtp_approx']=x['measurements'].get('mtp_approx',[]);row['actual_cache_slots']=int(cache_match[-1][0]);row['actual_cache_mib']=float(cache_match[-1][1]);row['cache_fingerprint']=cache_match[-1][2]
 assert input_identity(c)==input_identity(base)
 (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
 print('DONE_Q22_CACHE_SCREEN',json.dumps({k:v for k,v in row.items() if k!='memory'}),flush=True)
(out/'complete.json').write_text(json.dumps(dict(screen_cases=len(rows),quality_qualified=False))+'\n')
