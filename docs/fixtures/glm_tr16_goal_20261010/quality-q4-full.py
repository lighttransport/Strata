"""Resolve the borderline Q4+affinity pilot with a fresh paired full-corpus evaluation."""
import copy,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path('/mnt/nvme02/work/Strata');os.chdir(root)
r=root/'build-tr16-goal-20261010';sys.path.insert(0,str(root/'tools'))
from glm_q2_run_guard import descendants
from glm_tr16_calibrate import digest,input_identity
end=json.loads((r/'goal.json').read_text())['deadline']
def identity(pid):
 try:
  f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return f[19] if f[0]!='Z' else None
 except (FileNotFoundError,ProcessLookupError):return None
built=json.loads((r/'renorm-build-complete.json').read_text())
for path,expected in built['sha256'].items():
 if digest(path)!=expected:raise RuntimeError('Changed build artifact '+path)
out=r/'q4-full-corpus';out.mkdir(exist_ok=False)
regression=json.loads((root/'tests/glm/regression.json').read_text())
corpus=root/regression['quality']['corpus']
prior=json.loads((root/'build-tr16-20261009/quality-baseline/result.json').read_text())
if digest(corpus)!=prior['corpus_sha256']:raise RuntimeError('Full corpus changed')
frozen=out/'evaluation16.ids';frozen.write_bytes(corpus.read_bytes())
regression['quality']['corpus']=str(frozen)
regpath=out/'regression.json';regpath.write_text(json.dumps(regression,indent=2)+'\n')
base=json.loads((r/'fifth-configs/00-renorm-default.json').read_text())
candidate=json.loads((r/'split-q4-configs/00-q4-aff08-depth1-split.json').read_text())
inputs=input_identity(base);rows=[]
for name,c in [('baseline',base),('q4-aff08',candidate)]:
 if time.time()+1200>=end:raise SystemExit('Deadline before full quality '+name)
 if input_identity(c)!=inputs:raise RuntimeError('Model identity mismatch')
 c['env']['STRATA_GLM_EVAL_WIDTH']='1';c['env']['STRATA_GLM_MTP_ACCEPT_MARGIN']='0'
 c['model_name']=name;c['experimental_role']='Full8192-token component quality; no generation/speed qualification'
 cfg=out/(name+'.json');cfg.write_text(json.dumps(c,indent=2)+'\n')
 command=[str(root/'build-exl3-venv/bin/python'),'tools/glm_tr16_quality.py','--config',str(cfg),'--output',str(out/name),'--regression-config',str(regpath),'--wait-idle-seconds','900','--record-cpu-interference']
 if name!='baseline':command += ['--baseline',str(out/'baseline/result.json')]
 print('START_FULL_CORPUS',name,flush=True)
 with (out/(name+'-driver.log')).open('x') as log:
  p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=min(2400,end-time.time()-30))
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
 x=json.loads((out/name/'result.json').read_text())
 if not x['complete'] or x['expected_tokens']!=8192 or x['metrics']['tokens']!=8192 or x['sequences']!=16:raise RuntimeError('Incomplete full corpus '+name)
 row={'name':name,'exit_code':code,'complete':True,'metrics':x['metrics'],'failures':x.get('failures',[]),'corpus_gate_passed':x.get('corpus_gate_passed'),'quality_qualified':False,'performance_comparable':False,'interference':x['memory'].get('interference',[])}
 if name=='baseline':row['metrics_match_prior_full']=x['metrics']==prior['metrics']
 else:
  b=rows[0]['metrics']
  row['ppl_increase_percent']=100*(x['metrics']['perplexity']/b['perplexity']-1)
  row['top1_drop_percentage_points']=100*(b['top1_agreement']-x['metrics']['top1_agreement'])
 rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print('DONE_FULL_CORPUS',json.dumps(row),flush=True)
(out/'complete.json').write_text(json.dumps({'paired_full_corpus_complete':True,'tokens_each':8192,'corpus_gate_passed':rows[-1]['corpus_gate_passed'],'quality_qualified':False,'note':'Full paired HumanEval and held-out repeated timing still required.'},indent=2)+'\n')
