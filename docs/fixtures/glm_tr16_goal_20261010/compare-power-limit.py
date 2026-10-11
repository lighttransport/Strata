"""Requires explicit user approval: compare factory180W against current150W, restoring150W."""
import argparse,copy,json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path('/mnt/nvme02/work/Strata');os.chdir(root)
r=root/'build-tr16-goal-20261010';sys.path.insert(0,str(root/'tools'))
from glm_tr16_calibrate import digest,wait_idle
from glm_q2_run_guard import descendants
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--approved',action='store_true');args=ap.parse_args()
if not args.approved:raise SystemExit('Explicit user approval required before changing the existing GPU power cap')
end=json.loads((r/'goal.json').read_text())['deadline']
if time.time()+900>=end:raise SystemExit('Insufficient time before goal deadline')
stage=json.loads((r/'pipeline-v5-state.json').read_text());assert stage['current'] is None and len(stage['completed'])==4 and all(v['exit_code']==0 for v in stage['completed'])
built=json.loads((r/'renorm-build-complete.json').read_text())
for path,expected in built['sha256'].items():assert digest(path)==expected,path
out=r/'power-comparison';out.mkdir(exist_ok=False)
def query():
 fields=['uuid','name','power.limit','power.default_limit','power.min_limit','power.max_limit']
 lines=subprocess.check_output(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits'],text=True).strip().splitlines()
 assert len(lines)==1,'Expected one GPU'
 values=[v.strip() for v in lines[0].split(',')];return dict(zip(fields,values))
def cap():return float(query()['power.limit'])
def set_cap(value):
 subprocess.run(['sudo','-n','nvidia-smi','-i',initial['uuid'],'--power-limit='+str(value)],check=True)
 assert cap()==float(value),'Power cap was not applied'
def identity(pid):
 try:
  f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return f[19] if f[0]!='Z' else None
 except (FileNotFoundError,ProcessLookupError):return None
def stop(proc):
 if not proc or proc.poll() is not None:return
 owned={pid:identity(pid) for pid in [proc.pid,*descendants(proc.pid)]}
 for pid,stamp in owned.items():
  if stamp and identity(pid)==stamp:
   try:os.kill(pid,signal.SIGINT)
   except ProcessLookupError:pass
 time.sleep(10)
 for pid,stamp in owned.items():
  if stamp and identity(pid)==stamp:
   try:os.kill(pid,signal.SIGKILL)
   except ProcessLookupError:pass
 proc.wait(timeout=10)
def interrupt(signum,frame):raise KeyboardInterrupt
signal.signal(signal.SIGTERM,interrupt)
initial=query();assert '5060 Ti' in initial['name'] and float(initial['power.limit'])==150 and float(initial['power.default_limit'])==180,initial
(out/'initial-gpu.json').write_text(json.dumps(initial,indent=2)+'\n')
rows=[];changed=False;proc=None
try:
 for watts in (150,180):
  wait_idle(min(900,max(1,end-time.time()-30)))
  assert query()['uuid']==initial['uuid'],'GPU changed'
  if watts==180:
   changed=True  # Restore even if the set operation succeeds but verification fails.
   set_cap(180)
  assert cap()==watts
  c=json.loads((r/'split-q4-configs/00-q4-aff08-depth1-split.json').read_text())
  c['model_name']='q4-aff08-'+str(watts)+'w';c['experimental_role']='Approved factory power comparison, paired held-out2K512x3; no quality promotion'
  cfg=out/(str(watts)+'w.json');cfg.write_text(json.dumps(c,indent=2)+'\n');dest=out/(str(watts)+'w')
  command=[str(root/'build-exl3-venv/bin/python'),'tools/glm_tr16_calibrate.py','--config',str(cfg),'--prompt','build-tr16-20261009/prompts2k/holdout-code.ids','--output',str(dest),'--tokens','512','--trials','3','--cases','baseline','--wait-idle-seconds','900','--timeout','1200']
  print('START_POWER_COMPARISON',watts,flush=True)
  with (out/(str(watts)+'w-driver.log')).open('x') as log:
   proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   code=proc.wait(timeout=min(1500,end-time.time()-30))
  assert cap()==watts,'Power cap changed during run'
  x=json.loads((dest/'000-baseline.result.json').read_text());assert code==0 and x['clean'],'Unclean power comparison'
  tokens=(dest/'000-baseline.stdout').read_text().split();assert len(tokens)==2048
  row={'power_limit_w':watts,'decode_tok_s':x['measurements']['median_decode_tok_s'],'prefill_tok_s':x['measurements']['median_prefill_tok_s'],'gpu_after':query(),'quality_qualified':False}
  if watts==180:
   row['tokens_identical_to_150w']=tokens==(out/'150w/000-baseline.stdout').read_text().split()
   row['decode_change_percent']=100*(row['decode_tok_s']/rows[0]['decode_tok_s']-1)
  rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print('DONE_POWER_COMPARISON',json.dumps(row),flush=True)
finally:
 try:
  stop(proc)
 finally:
  try:
   if changed:set_cap(150)
  finally:
   final_gpu=query()
   (out/'restoration.json').write_text(json.dumps({'final_gpu':final_gpu,'restored_150w':float(final_gpu['power.limit'])==150,'power_change_attempted':changed},indent=2)+'\n')
print('POWER_COMPARISON_COMPLETE restored150W',flush=True)
