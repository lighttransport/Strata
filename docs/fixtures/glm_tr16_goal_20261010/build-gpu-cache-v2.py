"""Build an isolated Q3 dense experiment, only after the active pipeline finishes."""
import difflib,hashlib,json,os,shutil,signal,subprocess,sys,time,tempfile
from pathlib import Path
root=Path('/mnt/nvme02/work/Strata');os.chdir(root)
r=root/'build-tr16-goal-20261010';sys.path.insert(0,str(root/'tools'))
from glm_tr16_calibrate import wait_idle
from glm_q2_run_guard import descendants
end=json.loads((r/'goal.json').read_text())['deadline']
state=json.loads((r/'pipeline-v8-state.json').read_text())
assert state['current'] is None and state['completed'][-1]['name']=='gpu-draft-retry' and state['completed'][-1]['exit_code']==1
assert 'prefix-trained decode cache requires unpacked CPU target experts' in (r/'gpu-draft-retry/aff0-depth1-gpu/000-baseline.log').read_text()
wait_idle(min(900,max(1,end-time.time()-30)))
s=r/'gpu-cache-v2-source'
if s.exists():raise RuntimeError('Refuse overwrite GPU cache v2 source')
shutil.copytree(r/'dense-q3-source',s)
p=s/'src/program/glm_decode.cpp';text=p.read_text()
old='        if (!prepared_experts.empty() || mtp_experts || gpu_decode_experts)'
new='        // The opt-in draft owns its weights and rollback history before cache\n        // admission. It is separate from the unpacked CPU target experts.\n        const bool reserved_gpu_draft = cached_gpu_mtp_enabled() && tier_owned_reserve &&\n            capture_hidden && !mtp_cpu_experts && mtp_ready && verify_history &&\n            !secondary && !remote_tp && !decode_cache_auto;\n        if (!prepared_experts.empty() || (mtp_experts && !reserved_gpu_draft) || gpu_decode_experts)'
assert text.count(old)==1;text=text.replace(old,new);p.write_text(text)
files=json.loads((r/'renorm-patch-check.json').read_text())['files'];files=list(dict.fromkeys(files+['src/kernels/cuda/native_mmvq.cu']));patch='';original={}
for name in files:
 old=subprocess.run(['git','show','e5232ca3:'+name],capture_output=True,text=True)
 original[name]=old.stdout if old.returncode==0 else None
 patch+=''.join(difflib.unified_diff(old.stdout.splitlines(True) if old.returncode==0 else [],(s/name).read_text().splitlines(True),fromfile='a/'+name if old.returncode==0 else '/dev/null',tofile='b/'+name))
(r/'gpu-cache-v2.patch').write_text(patch)
with tempfile.TemporaryDirectory(prefix='strata-q3-patch-') as tmp:
 for name,value in original.items():
  p=Path(tmp)/name;p.parent.mkdir(parents=True,exist_ok=True)
  if value is not None:p.write_text(value)
 subprocess.run(['patch','-p1','--batch','-i',str(r/'gpu-cache-v2.patch')],cwd=tmp,check=True,stdout=subprocess.DEVNULL)
 for name in files:assert (Path(tmp)/name).read_bytes()==(s/name).read_bytes(),name
(r/'gpu-cache-v2-patch-check.json').write_text(json.dumps({'applies_to':'e5232ca3','reproduces_all_changed_files':True,'patch_sha256':hashlib.sha256(patch.encode()).hexdigest(),'files':files},indent=2)+'\n')
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
cu=r/'gpu-cache-v2-cuda';hip=r/'gpu-cache-v2-hip';gg=r/'ggml-reference'
for name,b,args in [('gpu-cache-v2-cuda',cu,['-DSTRATA_ENABLE_CUDA=ON','-DSTRATA_ENABLE_HIP=OFF','-DCMAKE_CUDA_ARCHITECTURES=120','-DSTRATA_BUILD_TESTS=ON']),('gpu-cache-v2-hip',hip,['-DSTRATA_ENABLE_CUDA=OFF','-DSTRATA_ENABLE_HIP=ON','-DCMAKE_HIP_ARCHITECTURES=gfx1201','-DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0','-DSTRATA_BUILD_TESTS=OFF'])]:
 run(name+'-configure',['cmake','-S',str(s),'-B',str(b),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DSTRATA_GGML_DIR='+str(gg),*args])
 targets=['strata-glm-decode']
 if b==cu:targets+=['mmvq_il_parity','glm_parity','glm_q23_parity','glm_q8_test','glm_mailbox_test','glm_prefill_parity']
 run(name+'-build',['cmake','--build',str(b),'--target',*targets,'-j','12'])
wait_idle(min(900,max(1,end-time.time()-30)))
for name in ('glm_parity','glm_q23_parity','glm_q8_test','glm_mailbox_test','glm_prefill_parity'):
 run('gpu-cache-v2-'+name,[str(cu/name)],300)
run('gpu-cache-v2-il-parity-bench',[str(cu/'mmvq_il_parity'),'--bench','--emit-table','--glm-shapes'],600)
files=[cu/'strata-glm-decode',hip/'strata-glm-decode',r/'gpu-cache-v2.patch',r/'ggml-reference-manifest.json',r/'q3-il-prepared.json',r/'gpu-cache-v2-il-parity-bench.log']
(r/'gpu-cache-v2-build-complete.json').write_text(json.dumps({'cuda_parities_passed':True,'hip_build_passed':True,'q3_il_parity_passed':True,'q3_il_default_inactive':True,'model_controls_passed':False,'quality_qualified':False,'time':time.time(),'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
print('DENSE_Q3_BUILD_COMPLETE',flush=True)
