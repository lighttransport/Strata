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
assert (r/'dense-q3-build-complete.json').exists(), 'Q3 build not finished'
jobs=[('shared-projections','screen-shared-projections.py'),('q22-speed','screen-q22.py'),('q22-quality','quality-q22.py')]
rows=[]
(r/'pipeline-v10-state.json').write_text(json.dumps({'current':'waiting-for-v9','completed':[]},indent=2)+'\n')
print('WAIT_FOR_V9_SUCCESS',flush=True)
while True:
 state=json.loads((r/'pipeline-v9-state.json').read_text())
 if any(v['exit_code']!=0 for v in state['completed']):raise RuntimeError('V9 failed; inspect before follow-up')
 if state['current'] is None and len(state['completed'])==3:break
 if time.time()+3600>=end:raise SystemExit('Validation reserve reached before V9 completion')
 time.sleep(5)
print('V9_SUCCESS_CONFIRMED',flush=True)
for name,script in jobs:
 if time.time()+120>=end:raise SystemExit('Deadline before '+name)
 print('START_STAGE',name,flush=True)
 with (r/('pipeline-v10-'+name+'.log')).open('x') as log:
  p=subprocess.Popen([str(root/'build-exl3-venv/bin/python'),'-u',str(r/script)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (r/'pipeline-v10-state.json').write_text(json.dumps({'current':name,'pid':p.pid,'process_start':identity(p.pid),'completed':rows},indent=2)+'\n')
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
 (r/'pipeline-v10-state.json').write_text(json.dumps({'current':None,'completed':rows},indent=2)+'\n')
 print('DONE_STAGE',name,code,flush=True)
 if code:raise RuntimeError('Stage failed '+name)
print('PIPELINE_V10_COMPLETE',flush=True)
