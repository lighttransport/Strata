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
marker=r/'dense-q3-build-complete.json'
if not marker.exists():raise RuntimeError('IL build marker absent')
built=json.loads(marker.read_text())
assert built['cuda_parities_passed'] and built['hip_build_passed']
for path,expected in built['sha256'].items():
 if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError('Changed build artifact '+path)
base=json.loads((r/'baseline.json').read_text());base['exe']=str(r/'dense-q3-cuda/strata-glm-decode');base['source_patch']=str(r/'dense-q3.patch')
from gguf_reader import GGUFFile
meta=GGUFFile(Path(base['model'])).metadata
base['stop_token_ids']=sorted({int(meta[k]) for k in ('tokenizer.ggml.eos_token_id','tokenizer.ggml.eot_token_id','tokenizer.ggml.eom_token_id') if k in meta})
if not base['stop_token_ids']:raise RuntimeError('No model stop IDs found')
print('BUILD_VERIFIED stop_ids='+str(base['stop_token_ids']),flush=True)
# Requires the preceding Q4 CPU/GPU cache comparisons to have succeeded.
prior_gpu=json.loads((r/'gpu-draft-retry-summary.json').read_text())
assert len(prior_gpu)==6 and all(v['clean'] for v in prior_gpu), 'GPU draft prerequisites incomplete'
assert (r/'dense-q3-screen/complete.json').exists()
quality=json.loads((r/'dense-q3-screen/quality-summary.json').read_text())
feasible={v['name'] for v in quality if v['complete'] and v['ppl_increase_percent']<=10}
rows=json.loads((r/'dense-q3-screen/summary.json').read_text())
table=json.loads((r/'q3-il-measured-table.json').read_text())
if not feasible or not any(v for row in table['rows'] for v in row):
 (r/'q3-gpu-combined-skipped.json').write_text(json.dumps({'skipped':True,'reason':'No Q3 pilot within moderate PPL budget or no measured IL rows','quality_qualified':False},indent=2)+'\n')
 print('Q3_GPU_COMBINED_SKIPPED',flush=True);sys.exit(0)
chosen=max((v for v in rows if v['name'] in feasible),key=lambda v:v['decode_tok_s'])
common=json.loads(Path(chosen['config']).read_text())
common['decode_cache_mib']=4352 if common['env']['STRATA_GLM_DENSE_Q3']=='all' else 4096
common['env']['STRATA_MMVQ_IL_ROWS']=common['env'].get('STRATA_MMVQ_IL_ROWS','')+';'+table['override']
common['env']['STRATA_GLM_MMVQ_IL']='1'
common['precision_pilot_reference']=chosen['name']
common['experimental_role']='Q3 IL plus GPU draft screen; cache differs from pilot; no quality qualification'
cases=[]
def add(name,depth,margin=0,share=0):
 c=copy.deepcopy(common);c.update(exe=base['exe'],source_patch=base['source_patch']);c['model_name']=name;c['draft_depth']=depth
 c['experimental_role']='Lower-loss candidate screen; quality not qualified'
 c['env']['STRATA_GLM_MTP_ACCEPT_MARGIN']=str(margin)
 c['env']['STRATA_GLM_ROUTE_MIN_SHARE']=str(share)
 c['env']['STRATA_GLM_ROUTE_RENORMALIZE']='1' if share else '0'
 cases.append((name,c))
common['env']['STRATA_GLM_SPLIT_VERIFY']='1'
for label,depth,margin in [('q3-depth1',1,0),('q3-depth3-soft2',3,2),('q3-depth3-soft4',3,4)]:
 for backend in ('cpu','gpu'):
  add(label+'-'+backend,depth,margin)
  cases[-1][1]['mtp_experts']=backend
  cases[-1][1]['env']['STRATA_GLM_MTP_GPU_CACHE']='1' if backend=='gpu' else '0'
  cases[-1][1]['experimental_role']=f"Matched Q3 IL CPU/GPU draft at{common['decode_cache_mib']}MiB requested target cache; no quality qualification"
# Depth-one split verification uses one-token groups, so its IL comparison
# does not exercise the new multi-token Q3 kernel. Add a matched depth-three
# native-Q3 control before the existing two-token-group IL CPU case.
il_case=next(c for name,c in cases if name=='q3-depth3-soft2-cpu')
native=copy.deepcopy(il_case);native['model_name']='q3-depth3-soft2-native-cpu'
native['env']['STRATA_MMVQ_IL_ROWS']=';'.join(v for v in native['env']['STRATA_MMVQ_IL_ROWS'].split(';') if not v.startswith('11:'))
index=next(i for i,(name,_) in enumerate(cases) if name=='q3-depth3-soft2-cpu')
cases.insert(index,('q3-depth3-soft2-native-cpu',native))
(r/'q3-gpu-combined-configs').mkdir(exist_ok=True);summary=[]
references={}
for name,c in cases:
 if time.time()+180>=end:raise SystemExit('deadline before '+name)
 cfg=r/'q3-gpu-combined-configs'/(name+'.json');cfg.write_text(json.dumps(c,indent=2)+'\n');out=r/'q3-gpu-combined'/name
 if out.exists():raise RuntimeError('Refuse overwrite '+str(out))
 nt,trials=128,1
 cmd=[str(root/'build-exl3-venv/bin/python'),'tools/glm_tr16_calibrate.py','--config',str(cfg),'--prompt','build-tr16-20261009/prompts2k/train-code.ids','--output',str(out),'--tokens',str(nt),'--trials',str(trials),'--cases','baseline','--wait-idle-seconds','900','--timeout','900']
 print('START',name,flush=True)
 with (r/(name+'-q3-gpu-combined-driver.log')).open('x') as f:
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
 row={'name':name,'exit_code':code,'config':str(cfg),'precision_pilot':chosen['name'],'quality_qualified':False};path=out/'000-baseline.result.json'
 if path.exists():
  x=json.loads(path.read_text());row.update(clean=x['clean'],decode=x['measurements']['median_decode_tok_s'],prefill=x['measurements']['median_prefill_tok_s'],repeat_equal=x['repeated_tokens_equal'],rejected=x['memory'].get('rejected'),approx=x['measurements'].get('mtp_approx',[]),first_margin=x['measurements'].get('mtp_first_margin',[]),sync_state=x['measurements'].get('mtp_sync_state',[]),decode_trials=x['measurements']['decode'])
 if not row.get('clean'):
  summary.append(row);(r/'q3-gpu-combined-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  raise RuntimeError('Invalid GPU draft comparison '+name)
 if row.get('clean'):
  raw=(out/'000-baseline.log').read_text()
  assert len((out/'000-baseline.stdout').read_text().split())==256, 'Incomplete token output '+name
  if 'MTP expert_backend='+c['mtp_experts'] not in raw:raise RuntimeError('Wrong draft backend '+name)
  if name=='q3-depth3-soft2-cpu':
   ref=r/'q3-gpu-combined/q3-depth3-soft2-native-cpu/000-baseline.stdout'
   row['q3_il_native_tokens_equal']=ref.read_text().split()==(out/'000-baseline.stdout').read_text().split()
   assert row['q3_il_native_tokens_equal'], 'Multi-token Q3 IL output parity failed'
  if c['mtp_experts']=='gpu':
   reserve=re.search(r'GPU_MTP_CACHE reserved_before_target=1 resident_MiB=([0-9.]+)',raw)
   assert reserve and float(reserve[1])>0 and reserve.start()<raw.index('DECODE_CACHE prefix_trained'), 'Draft not reserved before target cache'
   row['draft_resident_mib']=float(reserve[1])
   ref=r/'q3-gpu-combined'/(name.removesuffix('-gpu')+'-cpu')
   oldraw=(ref/'000-baseline.log').read_text()
   cache=r'DECODE_CACHE prefix_trained=1 slots=(\d+) MiB=([0-9.]+)'
   assert re.findall(cache,raw)==re.findall(cache,oldraw), 'Target cache capacity differs'
   row['cache_capacity_equal']=True
   old=ref/'000-baseline.stdout'
   if old.exists():
    left=old.read_text().split();right=(out/'000-baseline.stdout').read_text().split()
    row['cpu_draft_tokens_equal']=left==right
    row['token_count_pair']=[len(left),len(right)]
 summary.append(row);(r/'q3-gpu-combined-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('DONE',json.dumps(row),flush=True)
 if name in references:
  if not row.get('clean'):raise RuntimeError('Invalid parity run '+name)
  ref=references[name]/'000-baseline.stdout'
  if ref.read_text().split()!=(out/'000-baseline.stdout').read_text().split():raise RuntimeError('Token parity failed '+name)
  if name in ('01-first100','02-first100-resync') and not any(a.get('skipped',0)>0 for a in row['first_margin']):raise RuntimeError('Skip branch not exercised')
  print('TOKEN_PARITY_PASS',name,flush=True)
 elif name in ('04-depth2-control','06-depth3-control') and not row.get('clean'):raise RuntimeError('Invalid IL control '+name)
print('GPU_DRAFT_COMPLETE',flush=True)
