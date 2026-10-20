"""CPU-only partial FP4 tree smoke test over loopback UDP, not an IB benchmark.
Run from repository root; requires the tree-cpu build and partial-model directory.
"""
import argparse,os,socket,subprocess,time
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('model_dir')
parser.add_argument('--binary',default='build-fp4-local/glm-fp4-tree-cpu')
parser.add_argument('--output',default='build-fp4-local/cpu-tree-local')
parser.add_argument('--primary-mode',choices=['separate','paired','fused','reuse'],default='separate')
parser.add_argument('--primary-numa-compact',action='store_true',help='Native CPU primary only; requires a two-node pool')
parser.add_argument('--primary-cpu-threads',type=int,choices=range(1,33))
parser.add_argument('--persistent-dispatch',action='store_true',help='Enable only for the direct-mode primary')
parser.add_argument('--worker-nt1-tasks',action='store_true',help='Native four-participant workers only: six tasks for one token, three otherwise')
parser.add_argument('--registered-staging',action='store_true',help='Opt-in fixed registered activation/result buffers on all roles')
parser.add_argument('--native-workspace',action='store_true',help='Opt-in native CPU activation/output buffer reuse; native CPU binaries only')
parser.add_argument('--native-k32',action='store_true',help='Opt-in native CPU K32 packed layout; native CPU binaries only')
parser.add_argument('--preserve-small',action='store_true',help='native CPU binary only; applies to all roles')
parser.add_argument('--extended',action='store_true')
args=parser.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
for topology in ['direct','relay']:
 sockets=[socket.socket() for _ in range(2)]
 for s in sockets:s.bind(('127.0.0.1',0))
 roots=['127.0.0.1:'+str(s.getsockname()[1]) for s in sockets]
 for s in sockets:s.close()
 children=[];logs=[]
 try:
  for role in [0,1,2]:
   log=(out/f'{topology}-{role}.log').open('w');logs.append(log)
   cmd=[args.binary,str(role),*roots,args.model_dir,'1','1','udp']
   if topology=='direct':cmd.append('direct')
   if args.worker_nt1_tasks and role!=0:cmd.append('worker-nt1-tasks')
   if args.registered_staging:cmd.append('registered-staging')
   if args.native_workspace:cmd.append('native-workspace')
   if args.native_k32:cmd.append('native-k32')
   if args.preserve_small:cmd.append('preserve-small')
   if args.extended:cmd.append('extended')
   if role==0:
    if args.primary_numa_compact:cmd.append('numa-compact')
    if args.primary_cpu_threads is not None:cmd.append('threads='+str(args.primary_cpu_threads))
    if args.persistent_dispatch and topology=='direct':cmd.append('persistent-dispatch')
    cmd.append('dump='+str(out/(topology+'.bin')))
    if args.primary_mode!='separate':cmd.append(args.primary_mode)
   children.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_PROC_BIND='spread',OMP_PLACES='cores')))
  deadline=time.monotonic()+600
  while any(p.poll() is None for p in children):
   failed=[p.returncode for p in children if p.poll() is not None and p.returncode!=0]
   if failed:raise RuntimeError(f'{topology} child failed {failed}')
   if time.monotonic()>deadline:raise RuntimeError('local tree test timeout')
   time.sleep(.5)
  assert all(p.returncode==0 for p in children)
  for f in logs:f.flush()
  primary_log=(out/f'{topology}-0.log').read_text()
  assert f'PRIMARY PASS cases={48 if args.extended else 16} repeat_bit_exact=1' in primary_log
  if args.primary_numa_compact:assert 'CPU_NUMA compact=1' in primary_log
  if args.persistent_dispatch and topology=='direct':assert 'DISPATCH persistent=1' in primary_log
  markers={'paired':'CPU_GATE_UP paired=1','fused':'CPU_EXPERT fused=1','reuse':'CPU_EXPERT fused=1 reuse_weights=1'}
  if args.primary_mode in markers:assert markers[args.primary_mode] in primary_log
  if args.registered_staging:
   for role in [0,1,2]:
    contexts=2 if role==(0 if topology=='direct' else 1) else 1
    assert f'REGISTERED_STAGING contexts={contexts} bytes={contexts*2*16*4096*4}' in (out/f'{topology}-{role}.log').read_text()
  if args.worker_nt1_tasks:
   assert 'CPU_TASK_POLICY' not in primary_log
   for role in [1,2]:
    log=(out/f'{topology}-{role}.log').read_text()
    assert 'CPU_TASK_POLICY worker_nt1=6 other=3' in log and 'CPU_TASK_COUNTS nt1=' in log
  if args.native_workspace:
   for role in [0,1,2]:assert 'CPU_WORKSPACE reuse=1' in (out/f'{topology}-{role}.log').read_text()
  if args.native_k32:
   for role in [0,1,2]:assert 'CPU_LAYOUT native_k32=1' in (out/f'{topology}-{role}.log').read_text()
  for role in [1,2]:assert f'WORKER PASS role={role}' in (out/f'{topology}-{role}.log').read_text()
  print(topology,'PASS',flush=True)
 finally:
  for p in children:
   if p.poll() is None:p.terminate()
  for p in children:
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  for f in logs:f.close()
assert (out/'direct.bin').read_bytes()==(out/'relay.bin').read_bytes()
print('PASS direct/relay output dumps are byte-identical',flush=True)
