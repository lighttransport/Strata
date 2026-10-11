"""Held-out 512x3 decode checks for a candidate that actually reached its short-screen target.

This script never qualifies model quality. Full corpus, full HumanEval and manual
continuation review remain required. It is prepared for explicit selection, not queued.
"""
import argparse,copy,hashlib,json,os,re,signal,subprocess,sys,time
from pathlib import Path
root=Path('/mnt/nvme02/work/Strata');os.chdir(root)
r=root/'build-tr16-goal-20261010';sys.path.insert(0,str(root/'tools'))
from glm_tr16_calibrate import digest,input_identity
from glm_q2_run_guard import descendants
from strata_tokenizer import Tokenizer
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--screen-result',type=Path,required=True)
ap.add_argument('--track',choices=('low','moderate'),required=True)
ap.add_argument('--name',required=True)
ap.add_argument('--near-threshold-diagnostic',action='store_true',help='Permit a short screen within5% of target; held-out target remains unchanged')
a=ap.parse_args();assert re.fullmatch(r'[a-z0-9][a-z0-9-]*',a.name)
end=json.loads((r/'goal.json').read_text())['deadline']
assert time.time()+5100<end, 'Reserve 85 minutes for long tests, full corpus and generated-task evaluation'
state=json.loads((r/'pipeline-v11-state.json').read_text())
assert state['current'] is None and len(state['completed'])==3 and all(v['exit_code']==0 for v in state['completed']), 'Finish the queued pipeline first'
x=json.loads(a.screen_result.read_text());assert x['clean'] and x['complete']
target=30 if a.track=='low' else 50
minimum=target*.95 if a.near_threshold_diagnostic else target
assert x['measurements']['median_decode_tok_s']>=minimum, 'Short screen is below diagnostic eligibility; held-out target remains unchanged'
c=copy.deepcopy(x['config']);executable=Path(c['exe'])
markers=[r/'gpu-cache-v2-build-complete.json',r/'cold-prune-build-complete.json']
built=[json.loads(p.read_text()) for p in markers if p.exists()]
built=[b for b in built if str(executable) in b['sha256']];assert len(built)==1
assert built[0]['cuda_parities_passed'] and built[0]['hip_build_passed']
for path,h in built[0]['sha256'].items():assert digest(path)==h,path
if executable.parent.name=='cold-prune-cuda':
 assert json.loads((r/'cold-prune-screen/analysis.json').read_text())['model_controls_passed']
inputs=input_identity(c);tokenizer=Tokenizer.from_gguf(Path(c['model']))
out=r/'long-finalists'/a.name;out.mkdir(parents=True,exist_ok=False)
(out/'selection.json').write_text(json.dumps(dict(screen_result=str(a.screen_result.resolve()),screen_result_sha256=digest(a.screen_result),short_decode_tok_s=x['measurements']['median_decode_tok_s'],track=a.track,target_tok_s=target,quality_qualified=False,scope='Held-out speed and continuation diagnostics only; not a quality gate'),indent=2)+'\n')
python=str(root/'build-exl3-venv/bin/python');rows=[]
def identity(pid):
 try:
  f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return f[19] if f[0]!='Z' else None
 except (FileNotFoundError,ProcessLookupError):return None
for size in ('2k','8k'):
 assert time.time()+4200<end, 'Preserve generated-task evaluation time'
 prompt=root/'build-tr16-20261009'/('prompts'+size)/'holdout-code.ids'
 prior=json.loads((root/'build-tr16-20261009'/('final-aff08-'+size)/'000-baseline.manifest.json').read_text())
 assert digest(prompt)==prior['prompt_sha256']
 assert input_identity(c)==inputs
 cfg=out/(size+'.json');cfg.write_text(json.dumps(c,indent=2)+'\n');dest=out/size
 command=[python,'tools/glm_tr16_calibrate.py','--config',str(cfg),'--prompt',str(prompt),'--output',str(dest),'--tokens','512','--trials','3','--cases','baseline','--wait-idle-seconds','900','--timeout','1200']
 print('START_LONG_FINALIST',size,flush=True)
 with (out/(size+'-driver.log')).open('x') as log:
  p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
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
 result=json.loads((dest/'000-baseline.result.json').read_text());assert code==0 and result['clean'] and result['complete']
 ids=list(map(int,(dest/'000-baseline.stdout').read_text().split()));assert len(ids)==2048
 assert result['manifest']['timed_trials']==3 and result['manifest']['decode_tokens']==512
 samples=[]
 for trial in range(4):
  seq=ids[trial*512:(trial+1)*512];grams=[tuple(seq[j:j+4]) for j in range(509)]
  samples.append(dict(trial=trial,warmup=trial==0,text=tokenizer.decode(seq),distinct_fourgram_fraction=len(set(grams))/len(grams)))
 (dest/'text-samples.json').write_text(json.dumps(dict(scope='Manual review required; distinct four-grams alone cannot establish task quality',quality_qualified=False,trials=samples),indent=2)+'\n')
 row=dict(prompt=size,decode_tok_s=result['measurements']['median_decode_tok_s'],prefill_tok_s=result['measurements']['median_prefill_tok_s'],timed_trials=3,generated_per_trial=512,speed_target_met=result['measurements']['median_decode_tok_s']>=target,quality_qualified=False,manual_review_pending=True)
 rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print('DONE_LONG_FINALIST',json.dumps(row),flush=True)
 assert input_identity(c)==inputs
(out/'complete.json').write_text(json.dumps(dict(heldout_2k_8k_complete=True,speed_target_met=all(row['speed_target_met'] for row in rows),quality_qualified=False,full_corpus_pending=True,full_humaneval_pending=True,manual_review_pending=True),indent=2)+'\n')
