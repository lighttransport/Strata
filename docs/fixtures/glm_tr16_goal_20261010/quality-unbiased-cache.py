"""Test whether unbiased cache learning reduces affinity-induced distribution loss."""
import copy,json,os,signal,subprocess,sys,time
from pathlib import Path
# Pending stage: complete the standalone audit before quality pilots, serially.
with open('/mnt/nvme02/work/Strata/build-tr16-goal-20261010/standalone-8k-audit-controller.log','x') as audit_log:
 subprocess.run([sys.executable,'-u','/mnt/nvme02/work/Strata/build-tr16-goal-20261010/audit-standalone-8k.py'],stdout=audit_log,stderr=subprocess.STDOUT,check=True)
root=Path('/mnt/nvme02/work/Strata');os.chdir(root)
r=root/'build-tr16-goal-20261010';sys.path.insert(0,str(root/'tools'))
from glm_tr16_calibrate import digest,input_identity
from glm_q2_run_guard import descendants
end=json.loads((r/'goal.json').read_text())['deadline'];python=str(root/'build-exl3-venv/bin/python')
stage=json.loads((r/'pipeline-v5-state.json').read_text());assert stage['current'] is None and len(stage['completed'])==4 and all(v['exit_code']==0 for v in stage['completed'])
built=json.loads((r/'renorm-build-complete.json').read_text())
for path,expected in built['sha256'].items():assert digest(path)==expected,path
out=r/'unbiased-cache-quality';out.mkdir(exist_ok=False)
baseline=r/'renorm-quality-pilot/00-renorm-default/result.json';b=json.loads(baseline.read_text());assert b['complete'] and b['expected_tokens']==2048
cases=[('aff20-unbiased','aff20-only-retry1'),('q4-aff08-unbiased','q4-aff08')];rows=[]
def identity(pid):
 try:
  f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return f[19] if f[0]!='Z' else None
 except (FileNotFoundError,ProcessLookupError):return None
for name,prior_name in cases:
 if time.time()+600>=end:raise SystemExit('Deadline before '+name)
 prior_root=r/'component-quality-pilot';c=json.loads((prior_root/(prior_name+'.json')).read_text());prior=json.loads((prior_root/prior_name/'result.json').read_text());assert prior['complete']
 c['env'].update(STRATA_GLM_TIER_LEARN_UNBIASED='1',STRATA_GLM_EVAL_WIDTH='1',STRATA_GLM_MTP_ACCEPT_MARGIN='0')
 c['model_name']=name;c['experimental_role']='Unbiased cache-learning teacher-forcing pilot; no speed or generated-task qualification'
 assert input_identity(c)==b['inputs']==prior['inputs']
 cfg=out/(name+'.json');cfg.write_text(json.dumps(c,indent=2)+'\n');dest=out/name
 command=[python,'tools/glm_tr16_quality.py','--config',str(cfg),'--output',str(dest),'--regression-config',str(r/'renorm-quality-pilot/regression.json'),'--baseline',str(baseline),'--wait-idle-seconds','900','--record-cpu-interference']
 print('START_UNBIASED_QUALITY',name,flush=True)
 with (out/(name+'-driver.log')).open('x') as log:
  p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=min(1800,end-time.time()-30))
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
 x=json.loads((dest/'result.json').read_text());assert x['complete'],'Incomplete quality '+name
 assert 'DECODE_CACHE_LEARN source=unbiased_router' in (dest/'eval.log').read_text(),'Unbiased learning branch not exercised'
 assert all(x[k]==prior[k] for k in ('corpus_sha256','reference','reference_size','reference_mtime_ns','sequences','inputs')),'Incomparable prior pilot'
 row={'name':name,'exit_code':code,'complete':True,'metrics':x['metrics'],'failures':x.get('failures',[]),'quality_qualified':False,'performance_comparable':False,'prior_result':str(prior_root/prior_name/'result.json'),'ppl_change_vs_prior_percent':100*(x['metrics']['perplexity']/prior['metrics']['perplexity']-1),'ppl_increase_vs_baseline_percent':100*(x['metrics']['perplexity']/b['metrics']['perplexity']-1),'top1_drop_vs_baseline_percentage_points':100*(b['metrics']['top1_agreement']-x['metrics']['top1_agreement'])}
 rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print('DONE_UNBIASED_QUALITY',json.dumps(row),flush=True)
(out/'complete.json').write_text(json.dumps({'complete':True,'pilot_cases':2,'targets_per_case':2048,'quality_qualified':False},indent=2)+'\n')
print('UNBIASED_CACHE_QUALITY_COMPLETE',flush=True)
