"""Full8192-token Q22 component-quality follow-up; prepared, not automatically queued."""
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
out=r/'q22-full-corpus'
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
from glm_tr16_quality import check_quality,TOLERANCE,evaluation_width
assert (r/'q22-cache-screen/complete.json').exists()
rows=json.loads((r/'q22-cache-screen/summary.json').read_text())
selected={v['name']:v for v in rows if v['name']=='depth3-soft4-cache6656' and v['clean']}
assert len(selected)==1
baseline=r/'q4-full-corpus/baseline/result.json';b=json.loads(baseline.read_text())
assert b['complete'] and b['expected_tokens']==8192
out.mkdir(exist_ok=False)
results=[]
for name,row in selected.items():
 c=json.loads(Path(row['config']).read_text());c['env'].update(STRATA_GLM_EVAL_WIDTH='1',STRATA_GLM_MTP_ACCEPT_MARGIN='0')
 if c.get('route_affinity_profile_sha256'):assert digest(c['env']['STRATA_GLM_ROUTE_AFFINITY_PROFILE'])==c['route_affinity_profile_sha256']
 cfg=out/(name+'.json');cfg.write_text(json.dumps(c,indent=2)+'\n');dest=out/name
 print('START_Q22_QUALITY',name,flush=True)
 # No same-input --baseline comparison: the expert pack intentionally differs.
 code=run([python,'tools/glm_tr16_quality.py','--config',str(cfg),'--output',str(dest),'--regression-config',str(r/'q4-full-corpus/regression.json'),'--wait-idle-seconds','900','--record-cpu-interference'],out/(name+'-driver.log'),2400)
 x=json.loads((dest/'result.json').read_text())
 if c.get('route_affinity_profile_sha256'):assert digest(c['env']['STRATA_GLM_ROUTE_AFFINITY_PROFILE'])==c['route_affinity_profile_sha256']
 assert x['complete'] and x['expected_tokens']==8192 and x['metrics']['tokens']==8192, 'Incomplete Q22 evaluation'
 assert evaluation_width(x['config'])==evaluation_width(b['config'])
 for key in ('corpus_sha256','reference','reference_size','reference_mtime_ns','sequences'):assert x[key]==b[key],key
 candidate_pack=str(Path(c['expert_pack']).resolve())
 baseline_pack=str(Path(b['config']['expert_pack']).resolve())
 assert Path(candidate_pack).name=='experts-q22.gguf' and Path(baseline_pack).name=='experts-q23.gguf'
 left=copy.deepcopy(x['inputs']);right=copy.deepcopy(b['inputs'])
 changed=[]
 for identity,path in ((left,candidate_pack),(right,baseline_pack)):
  pack=[v for v in identity['weight_files'] if v['path']==path];assert len(pack)==1
  changed.append(pack[0]);identity['weight_files']=[v for v in identity['weight_files'] if v['path']!=path]
 assert left==right,'Unexpected additional input identity change'
 import math
 assert all(math.isfinite(x['metrics'][key]) for key in ('perplexity','kl','top1_agreement')), 'Nonfinite quality metrics'
 failures=check_quality(x['metrics'],b['metrics'],TOLERANCE)
 result=dict(name=name,exit_code=code,complete=True,metrics=x['metrics'],baseline=str(baseline),baseline_sha256=digest(baseline),intentional_weight_change={'candidate':changed[0],'baseline':changed[1]},all_other_inputs_equal=True,failures=failures,ppl_increase_percent=100*(x['metrics']['perplexity']/b['metrics']['perplexity']-1),top1_drop_percentage_points=100*(b['metrics']['top1_agreement']-x['metrics']['top1_agreement']),quality_qualified=False,scope='Full8192-target precision/routing evaluation; relaxed acceptance disabled; generated-task quality remains required')
 results.append(result);(out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
 print('DONE_Q22_QUALITY',json.dumps(result),flush=True)

(out/'complete.json').write_text(json.dumps(dict(full_corpus_complete=True,tokens=8192,quality_qualified=False,generated_task_quality_pending=True))+'\n')
