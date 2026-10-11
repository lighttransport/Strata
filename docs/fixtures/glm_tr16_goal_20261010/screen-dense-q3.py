"""Q3 dense model controls, timing screens and component-quality pilots."""
import copy,hashlib,json,os,re,signal,subprocess,sys,time
from pathlib import Path
root=Path('/mnt/nvme02/work/Strata');os.chdir(root)
r=root/'build-tr16-goal-20261010';sys.path.insert(0,str(root/'tools'))
from glm_q2_run_guard import descendants
from glm_tr16_calibrate import digest,input_identity
end=json.loads((r/'goal.json').read_text())['deadline'];python=str(root/'build-exl3-venv/bin/python')
built=json.loads((r/'dense-q3-build-complete.json').read_text())
assert built['cuda_parities_passed'] and built['hip_build_passed']
for path,expected in built['sha256'].items():assert digest(path)==expected,path
out=r/'dense-q3-screen';out.mkdir(exist_ok=False)
def identity(pid):
 try:
  f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return f[19] if f[0]!='Z' else None
 except (FileNotFoundError,ProcessLookupError):return None
def run(command,log,seconds=1200):
 if time.time()+180>=end:raise SystemExit('Deadline before '+log.name)
 with log.open('x') as f:
  p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:return p.wait(timeout=min(seconds,end-time.time()-30))
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
cases=[]
def add(name,source,env=None,cache=None,reference=None):
 c=json.loads(source.read_text());c['exe']=str(r/'dense-q3-cuda/strata-glm-decode');c['source_patch']=str(r/'dense-q3.patch')
 c['env'].update(env or {});c['model_name']=name;c['experimental_role']='Q3 dense experiment; quality and speed not qualified'
 if cache is not None:c['decode_cache_mib']=cache
 cases.append((name,c,reference))
add('00-default-control',r/'fifth-configs/00-renorm-default.json',reference=r/'fifth/00-renorm-default')
q4=r/'split-q4-configs/00-q4-aff08-depth1-split.json'
add('01-q4-control',q4,reference=r/'split-q4/00-q4-aff08-depth1-split')
add('02-q3-aff0',q4,{'STRATA_GLM_DENSE_Q4':'0','STRATA_GLM_DENSE_Q3':'all','STRATA_GLM_ROUTE_AFFINITY':'0'})
add('03-q3-aff08',q4,{'STRATA_GLM_DENSE_Q4':'0','STRATA_GLM_DENSE_Q3':'all'})
add('04-q3-aff08-cache7424',q4,{'STRATA_GLM_DENSE_Q4':'0','STRATA_GLM_DENSE_Q3':'all'},7424)
add('05-q3-mixers-aff08',q4,{'STRATA_GLM_DENSE_Q4':'0','STRATA_GLM_DENSE_Q3':'mixers'})
rows=[]
for name,c,reference in cases:
 cfg=out/(name+'.json');cfg.write_text(json.dumps(c,indent=2)+'\n');dest=out/name
 print('START_Q3_SCREEN',name,flush=True)
 code=run([python,'tools/glm_tr16_calibrate.py','--config',str(cfg),'--prompt','build-tr16-20261009/prompts2k/train-code.ids','--output',str(dest),'--tokens','128','--trials','1','--cases','baseline','--wait-idle-seconds','900','--timeout','900'],out/(name+'-driver.log'))
 result=dest/'000-baseline.result.json'
 if not result.exists():raise RuntimeError('Missing Q3 screen '+name)
 x=json.loads(result.read_text());row={'name':name,'exit_code':code,'clean':x['clean'],'decode_tok_s':x['measurements']['median_decode_tok_s'],'prefill_tok_s':x['measurements']['median_prefill_tok_s'],'quality_qualified':False,'config':str(cfg),'memory':x['memory']}
 if not row['clean']:
  rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');raise RuntimeError('Unclean Q3 screen '+name)
 log=(dest/'000-baseline.log').read_text();tokens=(dest/'000-baseline.stdout').read_text().split()
 if len(tokens)!=256:raise RuntimeError('Incomplete Q3 tokens '+name)
 if reference:
  assert tokens==(reference/'000-baseline.stdout').read_text().split(),'Control changed outputs '+name
  row['control_tokens_identical']=256
 else:
  converted=re.findall(r'^DENSE_Q3 tensor=(\S+) source_type=(\d+) MiB=([\d.]+)',log,re.M)
  assert converted and not re.search(r'^DENSE_Q4 ',log,re.M),'Wrong conversion path'
  row['converted_tensors']=len(converted);row['converted_mib']=sum(float(v[2]) for v in converted)
  assert all(v[1] in ('13','14') and not v[0].startswith('blk.45.') and not v[0].endswith('_exps.weight') for v in converted),'Unexpected Q3 tensor scope'
 rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print('DONE_Q3_SCREEN',json.dumps({k:v for k,v in row.items() if k!='memory'}),flush=True)
# Teacher forcing separates the precision/cache changes from relaxed acceptance.
baseline=r/'renorm-quality-pilot/00-renorm-default/result.json';b=json.loads(baseline.read_text());assert b['complete'] and b['expected_tokens']==2048
quality=[]
for name,c,_ in cases:
 if name not in ('02-q3-aff0','04-q3-aff08-cache7424','05-q3-mixers-aff08'):continue
 c=copy.deepcopy(c);c['env'].update(STRATA_GLM_EVAL_WIDTH='1',STRATA_GLM_MTP_ACCEPT_MARGIN='0')
 assert input_identity(c)==b['inputs']
 cfg=out/(name+'-quality.json');cfg.write_text(json.dumps(c,indent=2)+'\n');dest=out/(name+'-quality')
 print('START_Q3_QUALITY',name,flush=True)
 code=run([python,'tools/glm_tr16_quality.py','--config',str(cfg),'--output',str(dest),'--regression-config',str(r/'renorm-quality-pilot/regression.json'),'--baseline',str(baseline),'--wait-idle-seconds','900','--record-cpu-interference'],out/(name+'-quality-driver.log'),1800)
 x=json.loads((dest/'result.json').read_text());assert x['complete'],'Incomplete Q3 quality'
 row={'name':name,'exit_code':code,'complete':True,'metrics':x['metrics'],'failures':x.get('failures',[]),'quality_qualified':False,'performance_comparable':False,'ppl_increase_percent':100*(x['metrics']['perplexity']/b['metrics']['perplexity']-1),'top1_drop_percentage_points':100*(b['metrics']['top1_agreement']-x['metrics']['top1_agreement'])}
 quality.append(row);(out/'quality-summary.json').write_text(json.dumps(quality,indent=2)+'\n');print('DONE_Q3_QUALITY',json.dumps(row),flush=True)
# Only test IL on a Q3 precision candidate whose pilot stays within the moderate PPL budget.
bench=r/'dense-q3-il-parity-bench.log'
entries=[];selected_rows=[]
for line in bench.read_text().splitlines():
 match=re.match(r'\s*\{11, \{\{([^}]*)\}, \{([^}]*)\}, \{([^}]*)\}\}\}',line)
 if not match:continue
 for nc,group in enumerate(match.groups(),2):
  values=[int(v.strip()) for v in group.split(',')]
  assert len(values)==5 and all(v in (0,1,2,4) for v in values)
  selected_rows.append(values);entries.append('11:'+str(nc)+':'+','.join(map(str,values)))
assert len(selected_rows)==3,'Missing measured Q3 table'
(r/'q3-il-measured-table.json').write_text(json.dumps({'rows':selected_rows,'override':';'.join(entries),'bench_sha256':digest(bench),'default_table_unchanged':True},indent=2)+'\n')
feasible={v['name'] for v in quality if v['ppl_increase_percent']<=10}
il_result={'tested':False,'quality_qualified':False,'reason':'No Q3 pilot within moderate PPL budget, or no measured IL rows'}
if feasible and any(selected_rows[0]):
 chosen=max((row for row in rows if row['name'] in feasible),key=lambda row:row['decode_tok_s'])
 c=json.loads(Path(chosen['config']).read_text());name='06-q3-il-selected'
 assert c['draft_depth']==1 and c['env'].get('STRATA_GLM_MMVQ_IL')=='1'
 c['model_name']=name;c['env']['STRATA_MMVQ_IL_ROWS']=c['env'].get('STRATA_MMVQ_IL_ROWS','')+';'+(';'.join(entries))
 cfg=out/(name+'.json');cfg.write_text(json.dumps(c,indent=2)+'\n');dest=out/name
 print('START_Q3_IL_SCREEN',chosen['name'],flush=True)
 code=run([python,'tools/glm_tr16_calibrate.py','--config',str(cfg),'--prompt','build-tr16-20261009/prompts2k/train-code.ids','--output',str(dest),'--tokens','128','--trials','1','--cases','baseline','--wait-idle-seconds','900','--timeout','900'],out/(name+'-driver.log'))
 x=json.loads((dest/'000-baseline.result.json').read_text());assert code==0 and x['clean'],'Unclean Q3 IL screen'
 left=(out/chosen['name']/'000-baseline.stdout').read_text().split();right=(dest/'000-baseline.stdout').read_text().split()
 il_result={'tested':True,'reference':chosen['name'],'config':str(cfg),'all_tokens_identical':len(left)==len(right)==256 and left==right,'decode_tok_s':x['measurements']['median_decode_tok_s'],'reference_decode_tok_s':chosen['decode_tok_s'],'quality_qualified':False}
 il_result['nominal_rate_change_percent']=100*(il_result['decode_tok_s']/chosen['decode_tok_s']-1)
 (out/'il-summary.json').write_text(json.dumps(il_result,indent=2)+'\n')
 assert il_result['all_tokens_identical'],'Q3 IL model output parity failed'
 print('DONE_Q3_IL_SCREEN',json.dumps(il_result),flush=True)
else:print('SKIP_Q3_IL_SCREEN',json.dumps(il_result),flush=True)
(out/'il-summary.json').write_text(json.dumps(il_result,indent=2)+'\n')
(out/'complete.json').write_text(json.dumps({'controls_passed':True,'screen_cases':6,'quality_pilots':3,'q3_il_screen':il_result,'quality_qualified':False},indent=2)+'\n')
print('DENSE_Q3_SCREEN_COMPLETE',flush=True)
