import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path('/mnt/nvme02/work/Strata');os.chdir(root)
r=root/'build-tr16-goal-20261010';sys.path.insert(0,str(root/'tools'))
from glm_tr16_calibrate import wait_idle
end=json.loads((r/'goal.json').read_text())['deadline']
def identity(pid):
 try:
  f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  return f[19] if f[0]!='Z' else None
 except (FileNotFoundError,ProcessLookupError):return None
def run(name,command,seconds=1800):
 if time.time()+120>=end:raise SystemExit('deadline before '+name)
 print('START',name,flush=True)
 with (r/(name+'.log')).open('x') as log:
  p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=min(seconds,end-time.time()-30))
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise
 if code:raise RuntimeError(f'{name} failed {code}')
 print('PASS',name,flush=True)
s=r/'state-only-source';cu=r/'state-only-cuda';hip=r/'state-only-hip';gg=r/'ggml-reference'
for name,b,args in [('state-only-cuda',cu,['-DSTRATA_ENABLE_CUDA=ON','-DSTRATA_ENABLE_HIP=OFF','-DCMAKE_CUDA_ARCHITECTURES=120','-DSTRATA_BUILD_TESTS=ON']),('state-only-hip',hip,['-DSTRATA_ENABLE_CUDA=OFF','-DSTRATA_ENABLE_HIP=ON','-DCMAKE_HIP_ARCHITECTURES=gfx1201','-DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0','-DSTRATA_BUILD_TESTS=OFF'])]:
 run(name+'-configure',['cmake','-S',str(s),'-B',str(b),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DSTRATA_GGML_DIR='+str(gg),*args])
 targets=['strata-glm-decode']
 if b==cu:targets+=['mmvq_il_parity','glm_parity','glm_q23_parity','glm_q8_test','glm_mailbox_test','glm_prefill_parity']
 run(name+'-build',['cmake','--build',str(b),'--target',*targets,'-j','12'])
wait_idle(min(900,max(1,end-time.time()-30)))
for name in ('glm_parity','glm_q23_parity','glm_q8_test','glm_mailbox_test','glm_prefill_parity'):
 run('state-only-'+name,[str(cu/name)],300)
files=[cu/'strata-glm-decode',hip/'strata-glm-decode',r/'state-only-standalone.patch',r/'ggml-reference-manifest.json']
(r/'state-only-build-complete.json').write_text(json.dumps({'cuda_parities_passed':True,'hip_build_passed':True,'time':time.time(),'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
print('STATE_ONLY_BUILD_COMPLETE',flush=True)
