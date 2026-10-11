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
out=r/'dense-q3-screen';assert out.is_dir(), 'Missing native Q3 screen evidence'
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

state=json.loads((r/'pipeline-v7-state.json').read_text())
assert state['current'] is None and state['completed'][-1]['name']=='dense-q3-screen' and state['completed'][-1]['exit_code']!=0, 'Original screen must be terminal'
assert 'Missing measured Q3 table' in (r/'pipeline-v7-dense-q3-screen.log').read_text(), 'Inspect unexpected screen failure'
rows=json.loads((out/'summary.json').read_text());quality=json.loads((out/'quality-summary.json').read_text())
assert len(rows)==6 and all(x['clean'] for x in rows) and len(quality)==3 and all(x['complete'] for x in quality)
from glm_tr16_calibrate import wait_idle
wait_idle(min(900,max(1,end-time.time()-30)))
code=run([str(r/'dense-q3-cuda/mmvq_il_parity'),'--bench','--emit-table','--glm-shapes'],r/'dense-q3-il-timing.log',900)
assert code==0,'Q3 IL corrected benchmark failed'
# Only test IL on a Q3 precision candidate whose pilot stays within the moderate PPL budget.
bench=r/'dense-q3-il-timing.log'
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
