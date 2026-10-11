import re,copy,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path('/mnt/nvme02/work/Strata');os.chdir(root)
r=root/'build-tr16-goal-20261010';sys.path.insert(0,str(root/'tools'))
from glm_q2_run_guard import descendants
end=json.loads((r/'goal.json').read_text())['deadline']
def identity(pid):
 try:
  f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  return f[19] if f[0]!='Z' else None
 except (FileNotFoundError,ProcessLookupError):return None
marker=r/'gpu-cache-v2-build-complete.json'
if not marker.exists():raise RuntimeError('IL build marker absent')
built=json.loads(marker.read_text())
assert built['cuda_parities_passed'] and built['hip_build_passed']
for path,expected in built['sha256'].items():
 if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError('Changed build artifact '+path)
base=json.loads((r/'baseline.json').read_text());base['exe']=str(r/'gpu-cache-v2-cuda/strata-glm-decode');base['source_patch']=str(r/'gpu-cache-v2.patch')
from gguf_reader import GGUFFile
meta=GGUFFile(Path(base['model'])).metadata
base['stop_token_ids']=sorted({int(meta[k]) for k in ('tokenizer.ggml.eos_token_id','tokenizer.ggml.eot_token_id','tokenizer.ggml.eom_token_id') if k in meta})
if not base['stop_token_ids']:raise RuntimeError('No model stop IDs found')
print('BUILD_VERIFIED stop_ids='+str(base['stop_token_ids']),flush=True)
cases=[]
table=json.loads((r/'glm-mmvq-il-tr16-table.json').read_text())['env']
common=json.loads((r/'fifth-configs/05-renorm-soft.json').read_text())
common['decode_cache_mib']=6656
common['env'].update(table)
common['env'].update(STRATA_GLM_DENSE_Q4='all', STRATA_GLM_ROUTE_AFFINITY='0.08',
 STRATA_GLM_ROUTE_MIN_SHARE='0', STRATA_GLM_ROUTE_RENORMALIZE='0',
 STRATA_GLM_MTP_ACCEPT_MARGIN='0', STRATA_GLM_MTP_ACCEPT_TOP_K='4',
 STRATA_GLM_MTP_ACCEPT_MAX_SOFT='3', STRATA_GLM_MTP_SYNC_STATE_ONLY='1',
 STRATA_GLM_MMVQ_IL='1', STRATA_GLM_SPLIT_VERIFY='0')
def add(name,depth,margin=0,share=0):
 c=copy.deepcopy(common);c.update(exe=base['exe'],source_patch=base['source_patch']);c['model_name']=name;c['draft_depth']=depth
 c['experimental_role']='Lower-loss candidate screen; quality not qualified'
 c['env']['STRATA_GLM_MTP_ACCEPT_MARGIN']=str(margin)
 c['env']['STRATA_GLM_ROUTE_MIN_SHARE']=str(share)
 c['env']['STRATA_GLM_ROUTE_RENORMALIZE']='1' if share else '0'
 cases.append((name,c))
common['decode_cache_mib']=3584
common['env']['STRATA_GLM_SPLIT_VERIFY']='1'
for label,affinity,depth,margin in [('aff0-depth1',0,1,0),('aff08-depth1',.08,1,0),('aff08-depth3-soft1',.08,3,1)]:
 for backend in ('cpu','gpu'):
  add(label+'-'+backend,depth,margin)
  cases[-1][1]['mtp_experts']=backend
  cases[-1][1]['env']['STRATA_GLM_MTP_GPU_CACHE']='1' if backend=='gpu' else '0'
  cases[-1][1]['env']['STRATA_GLM_ROUTE_AFFINITY']=str(affinity)
  cases[-1][1]['experimental_role']='Matched CPU/GPU draft screen at3584MiB target cache; no quality qualification'
# Reject incompatible opt-in requests before any model weights are loaded.
from glm_tr16_calibrate import command, wait_idle
wait_idle(min(900,max(1,end-time.time()-30)))
checks=[]
for label,backend,flag,owned,expected in [
 ('default-gpu-cache','gpu','0','1','decode-cache-mib requires CPU target experts'),
 ('cpu-with-gpu-flag','cpu','1','1','GPU MTP target cache requires GPU draft'),
 ('missing-owned-reserve','gpu','1','0','GPU MTP target cache requires GPU draft'),
 ('invalid-flag','gpu','2','1','GPU MTP target cache must be 0 or 1')]:
 c=copy.deepcopy(cases[0][1]);c['mtp_experts']=backend
 c['env'].update(STRATA_GLM_MTP_GPU_CACHE=flag,STRATA_GLM_TIER_OWNED_RESERVE=owned)
 env=os.environ.copy();env.update(c['env'])
 cmd=command(c,root/'build-tr16-20261009/prompts2k/train-code.ids',1,1)
 x=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=30)
 assert x.returncode!=0 and expected in x.stderr and 'EXPERT_PACK source=' not in x.stderr,label
 checks.append({'case':label,'exit_code':x.returncode,'stderr':x.stderr,'rejected_before_loading':True})
(r/'gpu-draft-v2-cli-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
(r/'gpu-draft-v2-configs').mkdir(exist_ok=True);summary=[]
references={}
for name,c in cases:
 if time.time()+180>=end:raise SystemExit('deadline before '+name)
 cfg=r/'gpu-draft-v2-configs'/(name+'.json');cfg.write_text(json.dumps(c,indent=2)+'\n');out=r/'gpu-draft-v2'/name
 if out.exists():raise RuntimeError('Refuse overwrite '+str(out))
 nt,trials=128,1
 cmd=[str(root/'build-exl3-venv/bin/python'),'tools/glm_tr16_calibrate.py','--config',str(cfg),'--prompt','build-tr16-20261009/prompts2k/train-code.ids','--output',str(out),'--tokens',str(nt),'--trials',str(trials),'--cases','baseline','--wait-idle-seconds','900','--timeout','900']
 print('START',name,flush=True)
 with (r/(name+'-gpu-draft-v2-driver.log')).open('x') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=min(1800,end-time.time()-30))
  except subprocess.TimeoutExpired:
   owned={pid:identity(pid) for pid in descendants(p.pid)}
   for pid,stamp in owned.items():
    if stamp and identity(pid)==stamp:
     try:os.kill(pid,signal.SIGINT)
     except ProcessLookupError:pass
   time.sleep(10)
   for pid,stamp in owned.items():
    if stamp and identity(pid)==stamp:
     try:os.kill(pid,signal.SIGKILL)
     except ProcessLookupError:pass
   p.wait();raise
 row={'name':name,'exit_code':code,'config':str(cfg)};path=out/'000-baseline.result.json'
 if path.exists():
  x=json.loads(path.read_text());row.update(clean=x['clean'],decode=x['measurements']['median_decode_tok_s'],prefill=x['measurements']['median_prefill_tok_s'],repeat_equal=x['repeated_tokens_equal'],rejected=x['memory'].get('rejected'),approx=x['measurements'].get('mtp_approx',[]),first_margin=x['measurements'].get('mtp_first_margin',[]),sync_state=x['measurements'].get('mtp_sync_state',[]),decode_trials=x['measurements']['decode'])
 if not row.get('clean'):
  summary.append(row);(r/'gpu-draft-v2-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  raise RuntimeError('Invalid GPU draft comparison '+name)
 if row.get('clean'):
  raw=(out/'000-baseline.log').read_text()
  assert len((out/'000-baseline.stdout').read_text().split())==256, 'Incomplete token output '+name
  if 'MTP expert_backend='+c['mtp_experts'] not in raw:raise RuntimeError('Wrong draft backend '+name)
  if name=='aff0-depth1-cpu':
   prior=r/'gpu-draft-retry/aff0-depth1-cpu/000-baseline.stdout'
   row['previous_binary_tokens_equal']=prior.read_text().split()==(out/'000-baseline.stdout').read_text().split()
   assert row['previous_binary_tokens_equal'], 'GPU-cache guard fix changed CPU control'
  if c['mtp_experts']=='gpu':
   reserve=re.search(r'GPU_MTP_CACHE reserved_before_target=1 resident_MiB=([0-9.]+)',raw)
   assert reserve and float(reserve[1])>0 and reserve.start()<raw.index('DECODE_CACHE prefix_trained'), 'Draft not reserved before target cache'
   row['draft_resident_mib']=float(reserve[1])
   ref=r/'gpu-draft-v2'/(name.removesuffix('-gpu')+'-cpu')
   oldraw=(ref/'000-baseline.log').read_text()
   cache=r'DECODE_CACHE prefix_trained=1 slots=(\d+) MiB=([0-9.]+)'
   assert re.findall(cache,raw)==re.findall(cache,oldraw), 'Target cache capacity differs'
   row['cache_capacity_equal']=True
   old=ref/'000-baseline.stdout'
   if old.exists():
    left=old.read_text().split();right=(out/'000-baseline.stdout').read_text().split()
    row['cpu_draft_tokens_equal']=left==right
    row['token_count_pair']=[len(left),len(right)]
 summary.append(row);(r/'gpu-draft-v2-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('DONE',json.dumps(row),flush=True)
 if name in references:
  if not row.get('clean'):raise RuntimeError('Invalid parity run '+name)
  ref=references[name]/'000-baseline.stdout'
  if ref.read_text().split()!=(out/'000-baseline.stdout').read_text().split():raise RuntimeError('Token parity failed '+name)
  if name in ('01-first100','02-first100-resync') and not any(a.get('skipped',0)>0 for a in row['first_margin']):raise RuntimeError('Skip branch not exercised')
  print('TOKEN_PARITY_PASS',name,flush=True)
 elif name in ('04-depth2-control','06-depth3-control') and not row.get('clean'):raise RuntimeError('Invalid IL control '+name)
print('GPU_DRAFT_COMPLETE',flush=True)
