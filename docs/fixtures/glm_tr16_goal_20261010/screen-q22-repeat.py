"""Q22 cache-capacity follow-up after precision quality pilots; opt-in screens only."""
import copy,hashlib,json,os,re,signal,subprocess,sys,time
from pathlib import Path
root=Path('/mnt/nvme02/work/Strata');os.chdir(root)
r=root/'build-tr16-goal-20261010';sys.path.insert(0,str(root/'tools'))
from glm_q2_run_guard import descendants
from glm_tr16_calibrate import digest,input_identity
end=json.loads((r/'goal.json').read_text())['deadline'];python=str(root/'build-exl3-venv/bin/python')
built=json.loads((r/'draft-route-build-complete.json').read_text())
assert built['cuda_parities_passed'] and built['hip_build_passed']
for path,expected in built['sha256'].items():assert digest(path)==expected,path
out=r/'q22-repeat-screen'
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
# Wait for the one known held-out run; never overlap or restart it.
coordinators=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:args=(proc/'cmdline').read_bytes().split(b'\0')
 except OSError:continue
 if any(a.endswith(b'/screen-finalist-long.py') or a==b'build-tr16-goal-20261010/screen-finalist-long.py' for a in args) and b'q22-soft2-diagnostic' in args:
  stamp=identity(int(proc.name))
  if stamp:coordinators.append((int(proc.name),stamp))
marker=r/'long-finalists/q22-soft2-diagnostic/complete.json'
assert marker.exists() or len(coordinators)==1
print('WAIT_SOFT2_LONG_BEFORE_REPEAT_SCREEN',flush=True)
while not marker.exists():
 assert identity(coordinators[0][0])==coordinators[0][1], 'Long run stopped without completion'
 assert time.time()+3600<end, 'Reserve final hour'
 time.sleep(5)
assert time.time()+3600<end
base=json.loads((r/'q22-cache-screen/depth3-soft4-cache6656.json').read_text())
base['exe']=str(r/'draft-route-cuda/strata-glm-decode');base['source_patch']=str(r/'draft-route.patch')
assert str(Path(base['exe'])) in built['sha256']
base['env']['STRATA_GLM_MTP_REPEAT_GUARD']='0'
guarded=copy.deepcopy(base);guarded['env']['STRATA_GLM_MTP_REPEAT_GUARD']='4'
out.mkdir(exist_ok=False);rows=[]
cases=[('00-control',base,6656),('01-repeat4',guarded,6656)]
for name,source,cache in cases:
 c=copy.deepcopy(source);c['decode_cache_mib']=cache;c['model_name']='q22-'+name
 c['experimental_role']='Q22 cache/drafting follow-up; routing and outputs can change; no quality qualification'
 cfg=out/(name+'.json');cfg.write_text(json.dumps(c,indent=2)+'\n');dest=out/name
 print('START_Q22_CACHE_SCREEN',name,flush=True)
 code=run([python,'tools/glm_tr16_calibrate.py','--config',str(cfg),'--prompt','build-tr16-20261009/prompts2k/train-code.ids','--output',str(dest),'--tokens','128','--trials','1','--cases','baseline','--wait-idle-seconds','900','--timeout','900'],out/(name+'-driver.log'))
 result=dest/'000-baseline.result.json'
 if not result.exists():raise RuntimeError('Missing cache screen result '+name)
 x=json.loads(result.read_text());row=dict(name=name,exit_code=code,clean=x['clean'],decode_tok_s=x['measurements']['median_decode_tok_s'],prefill_tok_s=x['measurements']['median_prefill_tok_s'],config=str(cfg),memory=x['memory'],quality_qualified=False)
 rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
 if code or not row['clean']:raise RuntimeError('Unclean cache screen '+name)
 tokens=(dest/'000-baseline.stdout').read_text().split();assert len(tokens)==256
 reference=r/'q22-cache-screen/depth3-soft4-cache6656'
 before=(reference/'000-baseline.stdout').read_text().split();assert len(before)==256
 row['token_reference']=str(reference);row['all_256_tokens_identical_to_reference']=tokens==before
 if name=='00-control':assert tokens==before, 'Frozen-build control changed tokens; do not run guarded case'
 log=(dest/'000-baseline.log').read_text()
 cache_match=re.findall(r'DECODE_CACHE prefix_trained=1 slots=(\d+) MiB=([\d.]+) fingerprint=(\d+)',log);assert cache_match
 row['speculation']=x['measurements']['speculation'];row['mtp_approx']=x['measurements'].get('mtp_approx',[]);row['actual_cache_slots']=int(cache_match[-1][0]);row['actual_cache_mib']=float(cache_match[-1][1]);row['cache_fingerprint']=cache_match[-1][2]
 if name=='00-control':
  prior=json.loads((r/'q22-cache-screen/summary.json').read_text())
  prior=next(v for v in prior if v['name']=='depth3-soft4-cache6656')
  for key in ('actual_cache_slots','actual_cache_mib','cache_fingerprint'):assert row[key]==prior[key],key
 else:
  match=re.findall(r'MTP_REPEAT_GUARD ngram=(\d+) history=128 blocked=(\d+)',log);assert match and all(int(v[0])==4 for v in match)
  row['repeat_guard_reports']=match
 assert input_identity(c)==input_identity(base)
 (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
 print('DONE_Q22_CACHE_SCREEN',json.dumps({k:v for k,v in row.items() if k!='memory'}),flush=True)
(out/'complete.json').write_text(json.dumps(dict(screen_cases=len(rows),quality_qualified=False))+'\n')
