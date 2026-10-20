#!/usr/bin/env python3
"""Launch the bounded three-node FP4 tree test over two IB edges."""
import argparse
import contextlib
import pathlib
import shlex
import subprocess
import time
import json
import re
import hashlib

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--relay','--worker-one',dest='relay',required=True,help='SSH alias of worker one (relay only in relay mode)')
p.add_argument('--leaf','--worker-two',dest='leaf',required=True,help='SSH alias of worker two')
p.add_argument('--up-root',required=True,help='Primary IB address:port')
p.add_argument('--down-root',required=True,help='Second bootstrap endpoint: middle node in relay mode, primary in direct mode')
p.add_argument('--model',required=True,type=pathlib.Path)
p.add_argument('--output',required=True,type=pathlib.Path)
p.add_argument('--primary-binary',default='build-fp4-local/glm-fp4-tree-hip')
p.add_argument('--worker-binary',required=True,help='Installed node executable path on both workers')
p.add_argument('--primary-port',type=int,choices=(1,2),default=1)
p.add_argument('--relay-up-port',type=int,choices=(1,2),default=1)
p.add_argument('--relay-down-port',type=int,choices=(1,2),default=2)
p.add_argument('--leaf-port',type=int,choices=(1,2),default=1)
p.add_argument('--timeout',type=int,default=300)
p.add_argument('--ssh-config',type=pathlib.Path,help='Optional local SSH config file')
p.add_argument('--primary-numa-compact',action='store_true',help='Native CPU primary only: prepare local rows and release duplicate blobs; requires a two-node CPU pool')
p.add_argument('--persistent-dispatch',action='store_true',help='Opt-in persistent primary threads; requires --direct')
p.add_argument('--tiled',action='store_true')
p.add_argument('--worker-nvfp4-nta',action='store_true',help='Opt-in native AVX2 FP4 workers only: offset dots, GU/down NTA, prefetch 2048; primary environment unchanged')
p.add_argument('--worker-nt1-tasks',action='store_true',help='Native four-participant workers only: six tasks for one token, three otherwise')
p.add_argument('--registered-staging',action='store_true',help='Opt-in fixed registered activation/result buffers on all roles')
p.add_argument('--native-workspace',action='store_true',help='Opt-in native CPU activation/output buffer reuse; native CPU binaries only')
p.add_argument('--native-k32',action='store_true',help='Opt-in native CPU K32 packed layout; native CPU binaries only')
p.add_argument('--preserve-small',action='store_true',help='Native CPU hidden rescaling; all peers must use native CPU binaries')
cpu_mode=p.add_mutually_exclusive_group()
cpu_mode.add_argument('--primary-cpu-paired',action='store_true',help='Opt-in paired gate/up on CPU primary only')
cpu_mode.add_argument('--primary-cpu-reuse',action='store_true',help='Opt-in fused CPU expert with weight reuse on primary only')
cpu_mode.add_argument('--primary-cpu-fused',action='store_true',help='Opt-in persistent-team CPU expert on primary only')
p.add_argument('--primary-cpu-threads',type=int,choices=range(1,33),help='CPU binary only; worker settings unchanged')
p.add_argument('--extended',action='store_true',help='48 layer cases with additional routing and numerical edge cases')
p.add_argument('--dump-outputs',action='store_true',help='Save bounded primary output bytes for exact cross-mode comparison')
p.add_argument('--direct',action='store_true',help='Primary connects directly to both workers; down-root must be on primary')
p.add_argument('--primary-leaf-port',type=int,choices=(1,2),default=2)
a=p.parse_args()
if not 30<=a.timeout<=1800 or any(x.startswith('-') for x in (a.relay,a.leaf,a.worker_binary)):
    p.error('Invalid timeout, SSH alias or executable path')
if a.persistent_dispatch and not a.direct:p.error('--persistent-dispatch requires --direct')
if a.worker_nvfp4_nta and a.native_k32:p.error('--worker-nvfp4-nta requires the ordinary layout, not --native-k32')
if a.direct and a.up_root==a.down_root:p.error('Direct mode needs two distinct bootstrap endpoints')
if not (a.model/'model.safetensors.index.json').is_file():p.error('Missing partial checkpoint index')
a.output.mkdir(parents=True,exist_ok=True)
names=('primary','relay','leaf')
if any((a.output/f).exists() for f in ('primary.log','relay.log','leaf.log','summary.json','outputs.f32')):p.error('Use a new output directory')
processes=[]
started=time.monotonic()
summary={'status':'running','topology':'direct' if a.direct else 'relay','extended':a.extended}
worker_env={'STRATA_NO_NVFP4_512':'1','STRATA_NVFP4_128':'0','STRATA_NVFP4_OFFSET_DOT':'1','STRATA_NVFP4_PREFETCH':'2048','STRATA_NVFP4_GU_NTA':'1','STRATA_NVFP4_DOWN_NTA':'1'} if a.worker_nvfp4_nta else {}
if worker_env:summary['worker_environment']=worker_env
try:
    with contextlib.ExitStack() as stack:
        logs=[stack.enter_context((a.output/(n+'.log')).open('w')) for n in names]
        args=[a.primary_binary,'0',a.up_root,a.down_root,str(a.model),str(a.primary_port),str(a.primary_leaf_port if a.direct else 1)]
        if a.primary_numa_compact:args+=['numa-compact']
        if a.persistent_dispatch:args+=['persistent-dispatch']
        if a.tiled:args+=['tiled']
        if a.registered_staging:args+=['registered-staging']
        if a.native_workspace:args+=['native-workspace']
        if a.native_k32:args+=['native-k32']
        if a.preserve_small:args+=['preserve-small']
        if a.primary_cpu_threads is not None:args+=['threads='+str(a.primary_cpu_threads)]
        if a.primary_cpu_paired:args+=['paired']
        if a.primary_cpu_fused:args+=['fused']
        if a.primary_cpu_reuse:args+=['reuse']
        if a.direct:args+=['direct']
        if a.extended:args+=['extended']
        if a.dump_outputs:args+=['dump='+str(a.output/'outputs.f32')]
        processes.append(subprocess.Popen(args,stdout=logs[0],stderr=subprocess.STDOUT))
        for role,alias,up,down in ((1,a.relay,a.relay_up_port,a.relay_down_port),(2,a.leaf,1,a.leaf_port)):
            cmd=['timeout','--signal=TERM','--kill-after=5s',str(a.timeout)+'s',a.worker_binary,str(role),a.up_root,a.down_root,'unused',str(up),str(down)]
            if worker_env:cmd[4:4]=['env',*[key+'='+value for key,value in worker_env.items()]]
            if a.worker_nt1_tasks:cmd+=['worker-nt1-tasks']
            if a.registered_staging:cmd+=['registered-staging']
            if a.native_workspace:cmd+=['native-workspace']
            if a.native_k32:cmd+=['native-k32']
            if a.preserve_small:cmd+=['preserve-small']
            if a.direct:cmd+=['direct']
            if a.extended:cmd+=['extended']
            processes.append(subprocess.Popen(['ssh',*(['-F',str(a.ssh_config)] if a.ssh_config else []),'-o','BatchMode=yes','-o','ConnectTimeout=15',alias,shlex.join(cmd)],stdout=logs[role],stderr=subprocess.STDOUT))
        deadline=time.monotonic()+a.timeout
        while any(x.poll() is None for x in processes):
            if any(x.poll() not in (None,0) for x in processes):raise RuntimeError('A node failed; inspect logs')
            if time.monotonic()>deadline:raise TimeoutError('Tree test deadline exceeded')
            time.sleep(.25)
        if any(x.returncode!=0 for x in processes):raise RuntimeError('A node failed; inspect logs')
    for n in names:
        log=(a.output/(n+'.log')).read_text()
        if 'TRANSPORT ib' not in log or ('PASS three-node direct partial-weight MoE' if a.direct else 'PASS three-node tree partial-weight MoE') not in log:raise RuntimeError('Missing IB completion: '+n)
    primary=(a.output/'primary.log').read_text()
    expected_cases=48 if a.extended else 16
    cases=re.findall(r'TREE PASS .*rel_l2=([0-9.eE+-]+)',primary)
    if len(cases)!=expected_cases or f'PRIMARY PASS cases={expected_cases} repeat_bit_exact=1' not in primary:
        raise RuntimeError('Incomplete primary correctness evidence')
    for role,name in ((1,'relay'),(2,'leaf')):
        log=(a.output/(name+'.log')).read_text()
        if a.worker_nt1_tasks and ('CPU_TASK_POLICY worker_nt1=6 other=3' not in log or 'CPU_TASK_COUNTS nt1=' not in log):
            raise RuntimeError('Missing worker task-policy evidence: '+name)
        if f'WORKER PASS role={role} commands={expected_cases} weights_reloaded=0' not in log:
            raise RuntimeError('Incomplete worker request evidence: '+name)
        if f'TOPOLOGY {summary["topology"]} role={role}' not in log:
            raise RuntimeError('Worker topology mismatch: '+name)
    summary.update(status='passed',cases=expected_cases,max_relative_l2=max(map(float,cases)),repeat_bit_exact=True)
    if a.dump_outputs:
        raw=(a.output/'outputs.f32').read_bytes()
        expected_bytes=(sum((1,7,16,3,2,15,16,7))*3 if a.extended else sum((1,7,16,3))*2)*4096*4
        if len(raw)!=expected_bytes:raise RuntimeError('Output dump length mismatch')
        summary.update(output_bytes=len(raw),output_sha256=hashlib.sha256(raw).hexdigest())
    print('PASS all three nodes over two IB edges; topology:', 'direct' if a.direct else 'relay', 'logs:',a.output)
except BaseException as error:
    summary.update(status='failed',error=type(error).__name__+': '+str(error))
    raise
finally:
    for x in processes:
        if x.poll() is None:x.terminate()
    for x in processes:
        try:x.wait(timeout=5)
        except subprocess.TimeoutExpired:x.kill();x.wait()

    summary.update(elapsed_seconds=round(time.monotonic()-started,3),exit_codes=[x.returncode for x in processes])
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
