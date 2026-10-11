"""Build an isolated Q3 dense experiment, only after the active pipeline finishes."""
import difflib,hashlib,json,os,shutil,signal,subprocess,sys,time,tempfile
from pathlib import Path
# Repeat the anomalous assembled control once, serially before compilation.
with open('/mnt/nvme02/work/Strata/build-tr16-goal-20261010/root-8k-repeat-controller.log','x') as audit_log:
 subprocess.run([sys.executable,'-u','/mnt/nvme02/work/Strata/build-tr16-goal-20261010/audit-root-8k-repeat.py'],stdout=audit_log,stderr=subprocess.STDOUT,check=True)
root=Path('/mnt/nvme02/work/Strata');os.chdir(root)
r=root/'build-tr16-goal-20261010';sys.path.insert(0,str(root/'tools'))
from glm_tr16_calibrate import wait_idle
from glm_q2_run_guard import descendants
end=json.loads((r/'goal.json').read_text())['deadline']
state=json.loads((r/'pipeline-v5-state.json').read_text())
assert state['current'] is None and len(state['completed'])==4 and all(row['exit_code']==0 for row in state['completed']), 'Existing pipeline not complete'
wait_idle(min(900,max(1,end-time.time()-30)))
prepared=json.loads((r/'dense-q3-prepared.json').read_text())
assert hashlib.sha256((r/'renorm-source/src/program/glm_decode.cpp').read_bytes()).hexdigest()==prepared['base_source_sha256']
assert hashlib.sha256((r/'dense-q3-decoder.cpp').read_bytes()).hexdigest()==prepared['decoder_sha256']
s=r/'dense-q3-source'
if s.exists():raise RuntimeError('Refuse to overwrite Q3 source')
shutil.copytree(r/'renorm-source',s)
shutil.copyfile(r/'dense-q3-decoder.cpp',s/'src/program/glm_decode.cpp')
il_prepared=json.loads((r/'q3-il-prepared.json').read_text())
for name,expected in il_prepared['sha256'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==expected,name
for source,target in [('dense-q3-native-mmvq.cu','src/kernels/cuda/native_mmvq.cu'),('dense-q3-mmvq-il-parity.cpp','src/kernels/mmvq_il_parity.cpp')]:
 shutil.copyfile(r/source,s/target)

files=json.loads((r/'renorm-patch-check.json').read_text())['files'];files=list(dict.fromkeys(files+['src/kernels/cuda/native_mmvq.cu']));patch='';original={}
for name in files:
 old=subprocess.run(['git','show','e5232ca3:'+name],capture_output=True,text=True)
 original[name]=old.stdout if old.returncode==0 else None
 patch+=''.join(difflib.unified_diff(old.stdout.splitlines(True) if old.returncode==0 else [],(s/name).read_text().splitlines(True),fromfile='a/'+name if old.returncode==0 else '/dev/null',tofile='b/'+name))
(r/'dense-q3.patch').write_text(patch)
with tempfile.TemporaryDirectory(prefix='strata-q3-patch-') as tmp:
 for name,value in original.items():
  p=Path(tmp)/name;p.parent.mkdir(parents=True,exist_ok=True)
  if value is not None:p.write_text(value)
 subprocess.run(['patch','-p1','--batch','-i',str(r/'dense-q3.patch')],cwd=tmp,check=True,stdout=subprocess.DEVNULL)
 for name in files:assert (Path(tmp)/name).read_bytes()==(s/name).read_bytes(),name
(r/'dense-q3-patch-check.json').write_text(json.dumps({'applies_to':'e5232ca3','reproduces_all_changed_files':True,'patch_sha256':hashlib.sha256(patch.encode()).hexdigest(),'files':files},indent=2)+'\n')
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
cu=r/'dense-q3-cuda';hip=r/'dense-q3-hip';gg=r/'ggml-reference'
for name,b,args in [('dense-q3-cuda',cu,['-DSTRATA_ENABLE_CUDA=ON','-DSTRATA_ENABLE_HIP=OFF','-DCMAKE_CUDA_ARCHITECTURES=120','-DSTRATA_BUILD_TESTS=ON']),('dense-q3-hip',hip,['-DSTRATA_ENABLE_CUDA=OFF','-DSTRATA_ENABLE_HIP=ON','-DCMAKE_HIP_ARCHITECTURES=gfx1201','-DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0','-DSTRATA_BUILD_TESTS=OFF'])]:
 run(name+'-configure',['cmake','-S',str(s),'-B',str(b),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DSTRATA_GGML_DIR='+str(gg),*args])
 targets=['strata-glm-decode']
 if b==cu:targets+=['mmvq_il_parity','glm_parity','glm_q23_parity','glm_q8_test','glm_mailbox_test','glm_prefill_parity']
 run(name+'-build',['cmake','--build',str(b),'--target',*targets,'-j','12'])
wait_idle(min(900,max(1,end-time.time()-30)))
for name in ('glm_parity','glm_q23_parity','glm_q8_test','glm_mailbox_test','glm_prefill_parity'):
 run('dense-q3-'+name,[str(cu/name)],300)
run('dense-q3-il-parity-bench',[str(cu/'mmvq_il_parity'),'--bench','--emit-table','--glm-shapes'],600)
files=[cu/'strata-glm-decode',hip/'strata-glm-decode',r/'dense-q3.patch',r/'ggml-reference-manifest.json',r/'q3-il-prepared.json',r/'dense-q3-il-parity-bench.log']
(r/'dense-q3-build-complete.json').write_text(json.dumps({'cuda_parities_passed':True,'hip_build_passed':True,'q3_il_parity_passed':True,'q3_il_default_inactive':True,'model_controls_passed':False,'quality_qualified':False,'time':time.time(),'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
print('DENSE_Q3_BUILD_COMPLETE',flush=True)
