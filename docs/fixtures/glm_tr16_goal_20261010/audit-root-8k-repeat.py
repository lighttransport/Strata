"""Held-out same-day comparisons for the assembled opt-in state-only change."""
import copy,json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path('/mnt/nvme02/work/Strata');os.chdir(root)
r=root/'build-tr16-goal-20261010';sys.path.insert(0,str(root/'tools'))
from glm_tr16_calibrate import digest
from glm_q2_run_guard import descendants
end=json.loads((r/'goal.json').read_text())['deadline'];python=str(root/'build-exl3-venv/bin/python')
built=json.loads((r/'root-state-build-complete.json').read_text());assert built['cuda_parities_passed'] and built['hip_build_passed']
for path,expected in built['sha256'].items():assert digest(path)==expected,path
out=r/'root-state-8k-repeat';out.mkdir(exist_ok=False);rows=[]
def identity(pid):
 try:
  f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return f[19] if f[0]!='Z' else None
 except (FileNotFoundError,ProcessLookupError):return None
for size in ('8k',):
 old=root/'build-tr16-20261009'/('final-aff08-'+size)
 prior=json.loads((old/'000-baseline.manifest.json').read_text())
 prompt=root/'build-tr16-20261009'/('prompts'+size)/'holdout-code.ids'
 assert digest(prompt)==prior['prompt_sha256']
 c=copy.deepcopy(prior['config']);c.update(exe=str(r/'root-state-cuda/strata-glm-decode'),source_patch=str(r/'root-state.patch'),source_commit='e5232ca3f2e926ffb41f9d3ff686efd4c8cc91de')
 c['stop_token_ids']=json.loads((r/'fifth-configs/00-renorm-default.json').read_text())['stop_token_ids']
 for variant in ('root-control-repeat',):
  enabled=variant=='state-only'
  if time.time()+600>=end:raise SystemExit('Deadline before held-out pair')
  name=size+'-'+variant;c['model_name']=name
  c['exe']=str(r/'root-state-cuda/strata-glm-decode');c['source_patch']=str(r/'root-state.patch')
  c['env']['STRATA_GLM_MTP_SYNC_STATE_ONLY']='1' if enabled else '0'
  c['experimental_role']='Held-out state-only paired test; no full generated-task qualification'
  cfg=out/(name+'.json');cfg.write_text(json.dumps(c,indent=2)+'\n');dest=out/name
  print('START_ROOT_STATE',name,flush=True)
  command=[python,'tools/glm_tr16_calibrate.py','--config',str(cfg),'--prompt',str(prompt),'--output',str(dest),'--tokens','512','--trials','3','--cases','baseline','--wait-idle-seconds','900','--timeout','1200']
  with (out/(name+'-driver.log')).open('x') as f:
   p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=p.wait(timeout=min(1500,end-time.time()-30))
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
  x=json.loads((dest/'000-baseline.result.json').read_text())
  assert code==0 and x['clean'],'Unclean held-out test '+name
  reference=r/'root-state-heldout/8k-control'
  a=(reference/'000-baseline.stdout').read_text().split();b=(dest/'000-baseline.stdout').read_text().split()
  assert len(a)==len(b)==2048,'Incomplete token output '+name
  counts=x['measurements'].get('mtp_sync_state',[])
  if enabled:assert sum(v.get('omitted',0) for v in counts)>0,'State-only not exercised'
  row={'name':name,'decode_tok_s':x['measurements']['median_decode_tok_s'],'prefill_tok_s':x['measurements']['median_prefill_tok_s'],'tokens_identical':sum(u==v for u,v in zip(a,b)),'all_tokens_identical':a==b,'first_difference_by_trial':[next((i for i in range(512) if a[k*512+i]!=b[k*512+i]),None) for k in range(4)],'reference':str(reference),'sync_state':counts,'quality_qualified':False,'config':str(cfg)}
  archived=(old/'000-baseline.stdout').read_text().split()
  row['archived_tokens_identical']=b==archived
  if enabled:
   row['standalone_off_tokens_identical']=b==(out/'8k-standalone-off/000-baseline.stdout').read_text().split()
   control=json.loads((reference/'000-baseline.result.json').read_text())['measurements']['median_decode_tok_s']
   row['versus_root_control_percent']=100*(row['decode_tok_s']/control-1)
   row['versus_standalone_off_percent']=100*(row['decode_tok_s']/rows[0]['decode_tok_s']-1)
  rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print('DONE_ROOT_STATE',json.dumps(row),flush=True)
(out/'complete.json').write_text(json.dumps({'audit_complete':True,'tokens_per_case':2048,'repeat_matches_archive':rows[0]['archived_tokens_identical'],'repeat_matches_first_root_control':rows[0]['all_tokens_identical'],'quality_qualified':False},indent=2)+'\n')
print('ROOT_8K_REPEAT_COMPLETE',flush=True)
