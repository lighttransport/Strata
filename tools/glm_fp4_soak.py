#!/usr/bin/env python3
"""Bounded low-duty endurance tests; connection settings stay in local JSON."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
p.add_argument('--interval',type=int,default=600)
p.add_argument('--max-runs',type=int,default=24)
a=p.parse_args()
if not 60<=a.interval<=3600 or not 1<=a.max_runs<=48:p.error('interval 60..3600 seconds; max-runs 1..48')
cfg=json.loads(a.config.read_text())
required={'worker_one','worker_two','model','worker_binary','download_dir','direct_roots','relay_roots','expected_output_sha256'}
if set(cfg)!=required:p.error('Config keys must match the documented local template')
for key in ('direct_roots','relay_roots'):
    if not isinstance(cfg[key],list) or len(cfg[key])!=2:p.error('Each topology requires two bootstrap endpoints')
a.output.mkdir(parents=True,exist_ok=True)
lock=(a.output/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (a.output/'status.json').exists():p.error('Use a new output directory; prior evidence is preserved')
(a.output/'pid').write_text(str(os.getpid()))
status={'state':'waiting','completed_runs':0,'interval_seconds':a.interval,'max_runs':a.max_runs}
def save():
    tmp=a.output/'status.tmp'
    tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(a.output/'status.json')
save()
try:
    for run in range(a.max_runs):
        deadline=time.monotonic()+a.interval
        while time.monotonic()<deadline:
            if (a.output/'STOP').exists():status['state']='stopped_by_request';save();sys.exit(0)
            if (Path(cfg['download_dir'])/'download-complete.json').exists():
                status['state']='download_complete';save();sys.exit(0)
            time.sleep(min(5,max(0,deadline-time.monotonic())))
        direct=run%2==0;mode='direct' if direct else 'relay';roots=cfg[mode+'_roots']
        result=a.output/f'run-{run+1:03d}-{mode}'
        cmd=[sys.executable,str(Path(__file__).with_name('glm_fp4_tree.py')),
             '--worker-one',cfg['worker_one'],'--worker-two',cfg['worker_two'],
             '--up-root',roots[0],'--down-root',roots[1],'--model',cfg['model'],
             '--worker-binary',cfg['worker_binary'],'--output',str(result),
             '--tiled','--extended','--dump-outputs','--timeout','300']
        if direct:cmd+=['--direct','--leaf-port','2']
        status.update(state='running',active_run=run+1,topology=mode);save()
        completed=subprocess.run(cmd,timeout=330)
        if completed.returncode:raise RuntimeError('Partial-weight runner failed; inspect run logs')
        summary=json.loads((result/'summary.json').read_text())
        if summary.get('status')!='passed' or summary.get('output_sha256')!=cfg['expected_output_sha256']:
            raise RuntimeError('Endurance output differs from the qualified baseline')
        # Raw output is only 3.14 MiB/run; retain it locally for byte comparisons.
        status.update(state='waiting',completed_runs=run+1,last_output_sha256=summary['output_sha256']);save()
        print('PASS endurance run',run+1,mode,flush=True)
    status['state']='run_limit_reached';save()
except BaseException as error:
    if isinstance(error,SystemExit):raise
    status.update(state='failed',error=type(error).__name__+': '+str(error));save();raise
