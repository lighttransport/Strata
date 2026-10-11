"""Prepared diagnostic only: try synchronous Nsight range export after the active pipeline.

The earlier deferred-export capture lost its agent. This changes export mode,
not model arithmetic. Report order/timeline still needs inspection; no speed claim.
"""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path('/mnt/nvme02/work/Strata');os.chdir(root)
r=root/'build-tr16-goal-20261010';sys.path.insert(0,str(root/'tools'))
from glm_tr16_calibrate import command,wait_idle,input_identity,digest
from glm_q2_run_guard import descendants
end=json.loads((r/'goal.json').read_text())['deadline']
assert time.time()+5400<end,'Reserve 90 minutes for any resulting fix and qualification'
state=json.loads((r/'pipeline-v11-state.json').read_text())
assert state['current'] is None and len(state['completed'])==3 and all(v['exit_code']==0 for v in state['completed']), 'Finish serial benchmark pipeline first'
built=json.loads((r/'cold-prune-build-complete.json').read_text())
for path,h in built['sha256'].items():assert digest(path)==h
assert json.loads((r/'cold-prune-screen/analysis.json').read_text())['model_controls_passed']
c=json.loads((r/'cold-prune-screen/00-default-control.json').read_text())
assert c['env'].get('STRATA_GLM_ROUTE_PRUNE_COLD_ONLY','0')=='0'
wait_idle(min(900,end-time.time()-30))
out=r/'nsys-q22-q4';out.mkdir(exist_ok=False)
cmd=command(c,root/'build-tr16-20261009/prompts2k/train-code.ids',128,1)+['--profile-decode']
trace=['nsys','profile','--trace=cuda','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=repeat:2:sync','--cuda-graph-trace=node','--output='+str(out/'trace'),*cmd]
manifest=dict(diagnostic=True,speed_comparable=False,config=c,command=trace,inputs=input_identity(c),decoder_sha256=digest(c['exe']),reason='Attempt synchronous per-range export after prior deferred-export agent loss; success is not assumed',range_interpretation='Inspect report timelines before identifying a timed range')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
python=str(root/'build-exl3-venv/bin/python')
guard=[python,'tools/glm_q2_run_guard.py','--output',str(out/'profile'),'--record-interference','--require-gpu-idle','--min-gpu-free-mib','512','--',*trace]
env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')};env.update(c['env'])
def identity(pid):
 try:
  f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return f[19] if f[0]!='Z' else None
 except (FileNotFoundError,ProcessLookupError):return None
print('START_Q22_PROFILE_DIAGNOSTIC',flush=True)
with (out/'driver.log').open('x') as log:
 p=subprocess.Popen(guard,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:code=p.wait(timeout=min(900,end-time.time()-30))
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
memory=json.loads((out/'profile.memory.json').read_text())
log=(out/'profile.log').read_text();reports=sorted(out.glob('*.nsys-rep'))
valid=code==0 and memory['rejected'] is None and 'Connection to Agent lost' not in log and len(reports)==2
(out/'status.json').write_text(json.dumps(dict(capture_complete=valid,reports=[str(p) for p in reports],exit_code=code,speed_comparable=False,quality_qualified=False,timeline_review_pending=True),indent=2)+'\n')
assert valid, 'Incomplete profile; preserve artifacts and inspect, do not infer success from exit code alone'
assert input_identity(c)==manifest['inputs'] and digest(c['exe'])==manifest['decoder_sha256']
for report in reports:
 with report.with_suffix('.kernel-stats.log').open('x') as f:
  subprocess.run(['nsys','stats','--report','cuda_gpu_kern_sum','--format','csv',str(report)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=min(180,end-time.time()-30))
print('Q22_PROFILE_CAPTURE_COMPLETE_TIMELINE_REVIEW_PENDING',flush=True)
