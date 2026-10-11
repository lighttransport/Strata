"""Full-model controls for the independent state-only patch and proposal-only changes."""
import copy,hashlib,json,os,re,signal,subprocess,sys,time
from pathlib import Path
root=Path('/mnt/nvme02/work/Strata');os.chdir(root)
r=root/'build-tr16-goal-20261010';sys.path.insert(0,str(root/'tools'))
from glm_q2_run_guard import descendants
from glm_tr16_calibrate import command,digest,input_identity,wait_idle
end=json.loads((r/'goal.json').read_text())['deadline']
def identity(pid):
 try:
  f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return f[19] if f[0]!='Z' else None
 except (FileNotFoundError,ProcessLookupError):return None
def run(cmd,log,env=None,timeout=1200):
 if time.time()+120>=end:raise SystemExit('Deadline before model control')
 with log.open('x') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:return p.wait(timeout=min(timeout,end-time.time()-30))
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
out=r/'proposal-model-controls';out.mkdir(exist_ok=False)
base=json.loads((r/'fifth-configs/00-renorm-default.json').read_text());rows=[]
python=str(root/'build-exl3-venv/bin/python')
def config(standalone,depth,env=None):
 c=copy.deepcopy(base);c['draft_depth']=depth
 c['exe']=str(r/('state-only-cuda' if standalone else 'draft-route-cuda')/'strata-glm-decode')
 c['source_patch']=str(r/('state-only-standalone.patch' if standalone else 'draft-route.patch'))
 c['env'].update(env or {});c['experimental_role']='Full-model correctness/proposal screen; no quality qualification'
 return c
def check(name,c,reference):
 print('START_MODEL_CONTROL',name,flush=True)
 c['model_name']=name
 cfg=out/(name+'.json');cfg.write_text(json.dumps(c,indent=2)+'\n');dest=out/name
 cmd=[python,'tools/glm_tr16_calibrate.py','--config',str(cfg),'--prompt','build-tr16-20261009/prompts2k/train-code.ids','--output',str(dest),'--tokens','128','--trials','1','--cases','baseline','--wait-idle-seconds','900','--timeout','900']
 code=run(cmd,out/(name+'-driver.log'))
 result=dest/'000-baseline.result.json'
 if code or not result.exists():raise RuntimeError('Model control failed '+name)
 x=json.loads(result.read_text())
 if not x['clean']:raise RuntimeError('Unclean model control '+name)
 tokens=(dest/'000-baseline.stdout').read_text().split();expected=(reference/'000-baseline.stdout').read_text().split()
 if len(tokens)!=256 or tokens!=expected:raise RuntimeError('Token parity failed '+name)
 if c['env'].get('STRATA_GLM_MTP_SYNC_STATE_ONLY')=='1' and not any(z.get('omitted',0)>0 for z in x['measurements'].get('mtp_sync_state',[])):raise RuntimeError('State-only branch not exercised')
 raw=(dest/'000-baseline.log').read_text()
 if c['env'].get('STRATA_GLM_MTP_UNPRUNED')=='1' and 'MTP_ROUTER unpruned=1' not in raw:raise RuntimeError('Unpruned router not exercised')
 if 'STRATA_GLM_MTP_RECURSIVE_BLEND' in c['env'] and 'MTP_RECURSIVE_BLEND factor=' not in raw:raise RuntimeError('Recursive blend not exercised')
 if c['env'].get('STRATA_GLM_MTP_REPEAT_GUARD')=='4' and 'MTP_REPEAT_GUARD ngram=4 history=128 blocked=0' not in raw:raise RuntimeError('Exact repeat guard changed acceptance')
 row={'name':name,'tokens_identical':256,'reference':str(reference),'decode_tok_s':x['measurements']['median_decode_tok_s'],'quality_qualified':False,'binary_sha256':digest(c['exe']),'position_counters':re.findall(r'^MTP_POSITION.*$',raw,re.M)}
 rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print('MODEL_TOKEN_PARITY_PASS',json.dumps(row),flush=True)
 return dest
standalone=check('00-standalone-default',config(True,1),r/'fifth/00-renorm-default')
check('01-standalone-state-only',config(True,1,{'STRATA_GLM_MTP_SYNC_STATE_ONLY':'1'}),standalone)
# Exercise the complete state comparison at8K, outside timing comparisons.
c=config(True,3,{'STRATA_GLM_MTP_SYNC_STATE_ONLY':'1','STRATA_GLM_CHECK_MTP_SYNC_STATE':'1','STRATA_GLM_MTP_SKIP_ANCHOR':'0'})
prompt=root/'build-tr16-20261009/prompts8k/train-code.ids'
if not prompt.is_file():raise RuntimeError('Missing frozen8K prompt')
prefix=out/'02-state-diagnostic-8k';wait_idle(min(900,max(1,end-time.time()-30)))
env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')};env.update(c['env'])
cmd=command(c,prompt,32,1)
prefix.with_suffix('.manifest.json').write_text(json.dumps({'config':c,'command':cmd,'binary_sha256':digest(c['exe']),'inputs':input_identity(c),'diagnostic':True,'speed_comparable':False},indent=2)+'\n')
code=run([python,'tools/glm_q2_run_guard.py','--output',str(prefix),'--require-idle','--require-gpu-idle','--min-gpu-free-mib','512','--',*cmd],out/'02-state-diagnostic-driver.log',env)
if code:raise RuntimeError('8K state diagnostic failed')
counts=[tuple(map(int,m)) for m in re.findall(r'MTP_SYNC_STATE_ONLY omitted=(\d+) checked=(\d+)',prefix.with_suffix('.log').read_text())]
checked=[pair for pair in counts if pair[1]>0]
if len(checked)!=2 or any(a<b or b!=16 for a,b in checked):raise RuntimeError('8K state diagnostic not fully exercised')
(out/'state-diagnostic-8k.json').write_text(json.dumps({'passed':True,'checked':checked,'speed_comparable':False},indent=2)+'\n');print('STATE_DIAGNOSTIC_8K_PASS',checked,flush=True)
control=check('03-proposal-default',config(False,3),r/'third/06-depth3-control')
check('04-unpruned-no-prune',config(False,3,{'STRATA_GLM_MTP_UNPRUNED':'1'}),control)
for name,factor in [('05-blend0','0'),('06-blend05','0.5'),('07-blend075','0.75')]:
 check(name,config(False,3,{'STRATA_GLM_MTP_RECURSIVE_BLEND':factor}),control)
check('08-repeat-guard-exact',config(False,3,{'STRATA_GLM_MTP_REPEAT_GUARD':'4'}),control)
(out/'complete.json').write_text(json.dumps({'model_controls_passed':True,'target_token_pairs':8,'tokens_compared_each':256,'state_diagnostic_8k':True,'quality_qualified':False},indent=2)+'\n')
print('PROPOSAL_MODEL_CONTROLS_COMPLETE',flush=True)
