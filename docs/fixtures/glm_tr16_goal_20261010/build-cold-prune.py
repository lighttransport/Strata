"""Build an isolated Q3 dense experiment, only after the active pipeline finishes."""
import difflib,hashlib,json,os,shutil,signal,subprocess,sys,time,tempfile
from pathlib import Path
root=Path('/mnt/nvme02/work/Strata');os.chdir(root)
r=root/'build-tr16-goal-20261010';sys.path.insert(0,str(root/'tools'))
from glm_tr16_calibrate import wait_idle
from glm_q2_run_guard import descendants
end=json.loads((r/'goal.json').read_text())['deadline']
print('WAIT_FOR_V10_SUCCESS',flush=True)
while True:
 state=json.loads((r/'pipeline-v10-state.json').read_text())
 if any(v['exit_code'] for v in state['completed']):raise RuntimeError('V10 failed; review required')
 if state['current'] is None and len(state['completed'])==3:break
 if time.time()+3600>=end:raise SystemExit('Validation reserve before build')
 time.sleep(5)
wait_idle(min(900,max(1,end-time.time()-30)))
s=r/'cold-prune-source'
prepared=json.loads((r/'cold-prune-prepared.json').read_text())
for f,h in prepared['sha256'].items():assert hashlib.sha256((s/f).read_bytes()).hexdigest()==h
files=json.loads((r/'renorm-patch-check.json').read_text())['files'];files=list(dict.fromkeys(files+['src/kernels/cuda/native_mmvq.cu']));patch='';original={}
for name in files:
 old=subprocess.run(['git','show','e5232ca3:'+name],capture_output=True,text=True)
 original[name]=old.stdout if old.returncode==0 else None
 patch+=''.join(difflib.unified_diff(old.stdout.splitlines(True) if old.returncode==0 else [],(s/name).read_text().splitlines(True),fromfile='a/'+name if old.returncode==0 else '/dev/null',tofile='b/'+name))
(r/'cold-prune.patch').write_text(patch)
with tempfile.TemporaryDirectory(prefix='strata-q3-patch-') as tmp:
 for name,value in original.items():
  p=Path(tmp)/name;p.parent.mkdir(parents=True,exist_ok=True)
  if value is not None:p.write_text(value)
 subprocess.run(['patch','-p1','--batch','-i',str(r/'cold-prune.patch')],cwd=tmp,check=True,stdout=subprocess.DEVNULL)
 for name in files:assert (Path(tmp)/name).read_bytes()==(s/name).read_bytes(),name
(r/'cold-prune-patch-check.json').write_text(json.dumps({'applies_to':'e5232ca3','reproduces_all_changed_files':True,'patch_sha256':hashlib.sha256(patch.encode()).hexdigest(),'files':files},indent=2)+'\n')
def identity(pid):
 try:
  f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return f[19] if f[0]!='Z' else None
 except (FileNotFoundError,ProcessLookupError):return None
def run(name,command,seconds=1800):
 if time.time()+120>=end:raise SystemExit('Deadline before '+name)
 print('START',name,flush=True)
 with (r/(name+'.log')).open('x') as log:
  p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=min(seconds,end-time.time()-30))
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
 if code:raise RuntimeError(name+' failed '+str(code))
 print('PASS',name,flush=True)
cu=r/'cold-prune-cuda';hip=r/'cold-prune-hip';gg=r/'ggml-reference'
for name,b,args in [('cold-prune-cuda',cu,['-DSTRATA_ENABLE_CUDA=ON','-DSTRATA_ENABLE_HIP=OFF','-DCMAKE_CUDA_ARCHITECTURES=120','-DSTRATA_BUILD_TESTS=ON']),('cold-prune-hip',hip,['-DSTRATA_ENABLE_CUDA=OFF','-DSTRATA_ENABLE_HIP=ON','-DCMAKE_HIP_ARCHITECTURES=gfx1201','-DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0','-DSTRATA_BUILD_TESTS=OFF'])]:
 run(name+'-configure',['cmake','-S',str(s),'-B',str(b),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DSTRATA_GGML_DIR='+str(gg),*args])
 targets=['strata-glm-decode']
 if b==cu:targets+=['glm_route_prune_test','glm_parity','glm_q23_parity','glm_q8_test','glm_mailbox_test','glm_prefill_parity']
 run(name+'-build',['cmake','--build',str(b),'--target',*targets,'-j','12'])
wait_idle(min(900,max(1,end-time.time()-30)))
for name in ('glm_parity','glm_q23_parity','glm_q8_test','glm_mailbox_test','glm_prefill_parity'):
 run('cold-prune-'+name,[str(cu/name)],300)
run('cold-prune-routing-oracles',['ctest','--test-dir',str(cu),'-R','^glm_route_prune_','--output-on-failure'],300)
files=[cu/'strata-glm-decode',hip/'strata-glm-decode',r/'cold-prune.patch',r/'cold-prune-routing-oracles.log']
(r/'cold-prune-build-complete.json').write_text(json.dumps({'cuda_parities_passed':True,'hip_build_passed':True,'route_oracles_passed':True,'default_off':True,'model_controls_passed':False,'quality_qualified':False,'time':time.time(),'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
print('COLD_PRUNE_BUILD_COMPLETE',flush=True)
