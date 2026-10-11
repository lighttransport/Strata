import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path('/mnt/nvme02/work/Strata');os.chdir(root)
r=root/'build-tr16-goal-20261010';sys.path.insert(0,str(root/'tools'))
from glm_tr16_calibrate import wait_idle
from glm_q2_run_guard import descendants
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
   p.wait()
   raise
 if code:raise RuntimeError(f'{name} failed {code}')
 print('PASS',name,flush=True)
s=r/'draft-route-source';cu=r/'draft-route-cuda';hip=r/'draft-route-hip';gg=r/'ggml-reference'
unit=json.loads((r/'draft-policy-unit-tests.json').read_text())
assert unit['bounded_accept_passed'] and unit['repeat_guard_passed']
for path,expected in unit['sha256'].items():
 if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError('Policy unit-test artifact changed '+path)
print('POLICY_UNIT_TESTS_VERIFIED',flush=True)
for name,b,args in [('draft-route-cuda',cu,['-DSTRATA_ENABLE_CUDA=ON','-DSTRATA_ENABLE_HIP=OFF','-DCMAKE_CUDA_ARCHITECTURES=120','-DSTRATA_BUILD_TESTS=ON']),('draft-route-hip',hip,['-DSTRATA_ENABLE_CUDA=OFF','-DSTRATA_ENABLE_HIP=ON','-DCMAKE_HIP_ARCHITECTURES=gfx1201','-DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0','-DSTRATA_BUILD_TESTS=OFF'])]:
 run(name+'-configure',['cmake','-S',str(s),'-B',str(b),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DSTRATA_GGML_DIR='+str(gg),*args])
 targets=['strata-glm-decode']
 if b==cu:targets+=['mmvq_il_parity','glm_route_prune_test','glm_parity','glm_q23_parity','glm_q8_test','glm_mailbox_test','glm_prefill_parity']
 run(name+'-build',['cmake','--build',str(b),'--target',*targets,'-j','12'])
wait_idle(min(900,max(1,end-time.time()-30)))
for name in ('glm_parity','glm_q23_parity','glm_q8_test','glm_mailbox_test','glm_prefill_parity'):
 run('draft-route-'+name,[str(cu/name)],300)
run('draft-route-tests',['ctest','--test-dir',str(cu),'-R','^glm_route_prune_','--output-on-failure'],300)
run('draft-route-glm-mmvq-parity-bench',[str(cu/'mmvq_il_parity'),'--bench','--emit-table','--glm-shapes'],900)
run('draft-route-model-controls',[str(root/'build-exl3-venv/bin/python'),'-u',str(r/'proposal-model-controls.py')],2400)
files=[cu/'strata-glm-decode',hip/'strata-glm-decode',r/'draft-route.patch',r/'ggml-reference-manifest.json',r/'proposal-model-controls/complete.json',r/'draft-policy-unit-tests.json']
(r/'draft-route-build-complete.json').write_text(json.dumps({'cuda_parities_passed':True,'hip_build_passed':True,'route_prune_oracle_passed':True,'glm_mmvq_parity_passed':True,'model_controls_passed':True,'policy_unit_tests_passed':True,'time':time.time(),'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
print('DRAFT_ROUTE_BUILD_COMPLETE',flush=True)

run('repeat-guard-screen',[str(root/'build-exl3-venv/bin/python'),'-u',str(r/'screen-repeat-guard.py')],2400)
run('gpu-draft-screen',[str(root/'build-exl3-venv/bin/python'),'-u',str(r/'screen-gpu-draft.py')],2400)
