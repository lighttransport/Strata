"""Prepared full164-task evaluation of Q22 soft4; never auto-promotes a profile."""
import copy,gzip,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path('/mnt/nvme02/work/Strata');os.chdir(root)
r=root/'build-tr16-goal-20261010';sys.path.insert(0,str(root/'tools'))
from glm_q2_run_guard import descendants
from glm_tr16_calibrate import input_identity,wait_idle,digest
end=json.loads((r/'goal.json').read_text())['deadline']
def identity(pid):
 try:
  f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  return f[19] if f[0]!='Z' else None
 except (FileNotFoundError,ProcessLookupError):return None
assert time.time()+3600<end, 'Reserve60min for bounded full task evaluation'
screens=json.loads((r/'q22-repeat-screen/summary.json').read_text())
assert len(screens)==2 and all(v['clean'] for v in screens)
assert screens[0]['all_256_tokens_identical_to_reference']
focused=json.loads((r/'q22-repeat-task31/summary.json').read_text())
assert len(focused)==2 and all(v['focused_complete'] for v in focused)
assert focused[1]['name']=='q22-repeat4' and focused[1]['passes']==1
# Prior component result admits research only: the executable differs.
q=json.loads((r/'q22-full-corpus/summary.json').read_text())
assert len(q)==1 and q[0]['complete'] and q[0]['ppl_increase_percent']<=10
built=json.loads((r/'draft-route-build-complete.json').read_text())
for path,expected in built['sha256'].items():assert digest(path)==expected
c=json.loads((r/'q22-repeat-screen/01-repeat4.json').read_text())
assert c['draft_depth']==3 and c['env']['STRATA_GLM_MTP_ACCEPT_MARGIN']=='4' and c['env']['STRATA_GLM_MTP_REPEAT_GUARD']=='4'
cases=[('q22-repeat4',c)];inputs=input_identity(c)
out=r/'q22-repeat-humaneval-full';out.mkdir(exist_ok=False)
dataset=root/'build-q2-redesign/HumanEval.jsonl.gz'
tasks=[json.loads(line) for line in gzip.decompress(dataset.read_bytes()).splitlines()]
expected_ids=[x['task_id'] for x in tasks];assert len(expected_ids)==164
baseline_folder=root/'build-tr16-20261009/humaneval-baseline'
baseline=json.loads((baseline_folder/'measurement.json').read_text())
baseline_manifest=json.loads((baseline_folder/'manifest.json').read_text())
assert baseline['complete'] and baseline['tasks']==164
assert [v['task_id'] for v in baseline['results']]==expected_ids
assert digest(dataset)==baseline_manifest['dataset_sha256']
rows=[];python=str(root/'build-exl3-venv/bin/python')
(out/'selection.json').write_text(json.dumps(dict(quality_qualified=False,baseline=str(baseline_folder),baseline_measurement_sha256=digest(baseline_folder/'measurement.json'),scope='Same pinned164 tasks and generation protocol. Intentional precision, cache, routing and drafting changes. This is task-quality research only. Guarded held-out speed and corpus validation remain separate; the earlier corpus uses a different executable. No target qualification follows automatically.'),indent=2)+'\n')
for name,c in cases:
 if time.time()+600>=end:raise SystemExit('Deadline before quality '+name)
 if input_identity(c)!=inputs:raise RuntimeError('Input identity mismatch')
 wait_idle(min(1800,max(1,end-time.time()-60)))
 env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')};env.update(c['env'])
 (out/(name+'.config.json')).write_text(json.dumps(c,indent=2)+'\n')
 command=['numactl','--interleave=all',python,'tools/glm_q2_humaneval.py',c['model'],str(dataset),'--output',str(out/name),'--decoder',c['exe'],'--expert-pack',c['expert_pack'],'--tokens','2048','--limit','164','--context',str(c['context']),'--prefill-batch',str(c['prefill_batch']),'--gpu-budget-mib',str(c['gpu_budget_mib']),'--decode-cache-mib',str(c['decode_cache_mib']),'--speculative','mtp','--draft-depth',str(c['draft_depth']),'--route-affinity',c['env'].get('STRATA_GLM_ROUTE_AFFINITY','0')]
 guard=[python,'tools/glm_q2_run_guard.py','--output',str(out/(name+'-guard')),'--min-gpu-free-mib','512','--require-gpu-idle','--record-interference','--',*command]
 print('START_HE_FULL',name,flush=True)
 with (out/(name+'-driver.log')).open('x') as log:
  p=subprocess.Popen(guard,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=min(5400,end-time.time()-30))
  except subprocess.TimeoutExpired:
   owned={pid:identity(pid) for pid in descendants(p.pid)}
   for pid,token in owned.items():
    if token and identity(pid)==token:
     try:os.kill(pid,signal.SIGINT)
     except ProcessLookupError:pass
   time.sleep(10)
   for pid,token in owned.items():
    if token and identity(pid)==token:
     try:os.kill(pid,signal.SIGKILL)
     except ProcessLookupError:pass
   p.wait();raise
 measure=out/name/'measurement.json';x=json.loads(measure.read_text()) if measure.exists() else {}
 memory_path=out/(name+'-guard.memory.json');memory=json.loads(memory_path.read_text()) if memory_path.exists() else {}
 results=x.get('results',[])
 complete=code==0 and x.get('tasks')==164 and [v.get('task_id') for v in results]==expected_ids and all(isinstance(v.get('passed'),bool) for v in results) and bool(memory) and not memory.get('rejected') and input_identity(c)==inputs
 row={'name':name,'exit_code':code,'full_complete':complete,'tasks':x.get('tasks',0),'passes':sum(v.get('passed') is True for v in results),'quality_qualified':False,'guard_rejected':memory.get('rejected'),'decoder_sha256':digest(c['exe'])}
 if complete:
  (out/name/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
  row['passes_minus_baseline']=row['passes']-sum(v['passed'] for v in baseline['results'])
  manifest=json.loads((out/name/'manifest.json').read_text())
  for key in ('dataset_revision','dataset_sha256','protocol','maximum_generated_tokens'):assert manifest[key]==baseline_manifest[key],key
  row['protocol_matches_baseline']=True
 rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print('DONE_HE_FULL',json.dumps(row),flush=True)
 if not complete:raise RuntimeError('Full HumanEval incomplete')
subprocess.run([python,str(r/'analyze-q22-repeat-humaneval.py')],check=True)
print('HE_FULL_COMPLETE',flush=True)

(out/'complete.json').write_text(json.dumps(dict(full164_complete=True,quality_qualified=False))+'\n')
