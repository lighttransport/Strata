import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path('/mnt/nvme02/work/Strata');os.chdir(root)
r=root/'build-tr16-goal-20261010';sys.path.insert(0,str(root/'tools'))
from glm_q2_run_guard import descendants
end=json.loads((r/'goal.json').read_text())['deadline']
def identity(pid):
 try:
  f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return f[19] if f[0]!='Z' else None
 except (FileNotFoundError,ProcessLookupError):return None
jobs=[('cold-prune-speed','screen-cold-prune.py'),('cold-prune-quality','quality-cold-prune.py'),('q22-cache-drafting','screen-q22-cache.py')]
rows=[]
build_pid=3222418
build_start=identity(build_pid)
marker=r/'cold-prune-build-complete.json'
assert build_start or marker.exists(), 'Queued build process missing'
if build_start:
 assert 'build-cold-prune.py' in Path(f'/proc/{build_pid}/cmdline').read_text(), 'Wrong build process'
(r/'pipeline-v11-state.json').write_text(json.dumps({'current':'waiting-for-cold-build','build_pid':build_pid,'build_start':build_start,'completed':[]},indent=2)+'\n')
print('WAIT_FOR_COLD_BUILD',flush=True)
while not marker.exists() or (build_start and identity(build_pid)==build_start):
 state=json.loads((r/'pipeline-v10-state.json').read_text())
 if any(v['exit_code'] for v in state['completed']):raise RuntimeError('V10 failed')
 if build_start and identity(build_pid)!=build_start and not marker.exists():raise RuntimeError('Cold build terminated without success marker; inspect build session')
 if time.time()+3600>=end:raise SystemExit('Validation reserve before cold build completion')
 time.sleep(5)
assert '100% tests passed, 0 tests failed out of 20' in (r/'cold-prune-routing-oracles.log').read_text()
print('COLD_BUILD_SUCCESS_CONFIRMED',flush=True)
for name,script in jobs:
 if time.time()+120>=end:raise SystemExit('Deadline before '+name)
 print('START_STAGE',name,flush=True)
 with (r/('pipeline-v11-'+name+'.log')).open('x') as log:
  p=subprocess.Popen([str(root/'build-exl3-venv/bin/python'),'-u',str(r/script)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (r/'pipeline-v11-state.json').write_text(json.dumps({'current':name,'pid':p.pid,'process_start':identity(p.pid),'completed':rows},indent=2)+'\n')
  try:code=p.wait(timeout=min(7200,end-time.time()-30))
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
 rows.append({'name':name,'exit_code':code,'time':time.time()})
 (r/'pipeline-v11-state.json').write_text(json.dumps({'current':None,'completed':rows},indent=2)+'\n')
 print('DONE_STAGE',name,code,flush=True)
 if code:raise RuntimeError('Stage failed '+name)
print('PIPELINE_V11_COMPLETE',flush=True)
