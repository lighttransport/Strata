#!/usr/bin/env python3
"""Exercise launcher failure cleanup and evidence checks without SSH or GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

LAUNCHER=Path(__file__).with_name('glm_fp4_tree.py').resolve()
STUB=r'''
import os,sys,time,shlex
from pathlib import Path
primary=Path(sys.argv[0]).name=='primary'
mode=os.environ['FP4_TEST_MODE']
if primary and mode=='worker-fails':
    Path(os.environ['FP4_TEST_PID']).write_text(str(os.getpid()))
    time.sleep(15)
if not primary and mode=='worker-fails':
    time.sleep(.2)
    sys.exit(7)
role=0 if primary else (1 if 'worker-one' in sys.argv else 2)
if mode=='nta':
    if primary:
        if os.environ.get('STRATA_NVFP4_GU_NTA')!='2':sys.exit(10)
    else:
        remote=shlex.split(sys.argv[-1])
        expected=['STRATA_NO_NVFP4_512=1','STRATA_NVFP4_128=0','STRATA_NVFP4_OFFSET_DOT=1','STRATA_NVFP4_PREFETCH=2048','STRATA_NVFP4_GU_NTA=1','STRATA_NVFP4_DOWN_NTA=1']
        if remote[4]!='env' or any(x not in remote[5:11] for x in expected) or remote[11]!='unused':sys.exit(11)
if mode.startswith('tasks'):
    selected=any('worker-nt1-tasks' in arg for arg in sys.argv[1:])
    if selected==primary:sys.exit(9)
    if not primary and mode!='tasks-missing':
        print('CPU_TASK_POLICY worker_nt1=6 other=3')
        print('CPU_TASK_COUNTS nt1=0 other=16')
print('TRANSPORT ib')
print(f'TOPOLOGY relay role={role}')
if primary:
    for i in range(16): print('TREE PASS case='+str(i)+' rel_l2=0')
    print('PRIMARY PASS cases=16 repeat_bit_exact=1')
else:
    print(f'WORKER PASS role={role} commands={15 if mode=="incomplete" else 16} weights_reloaded=0')
print('PASS three-node tree partial-weight MoE')
'''
class Launcher(unittest.TestCase):
    def run_stub(self,mode):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model=root/'model';model.mkdir();(model/'model.safetensors.index.json').write_text('{}')
            for name in ('primary','ssh'):
                f=root/name;f.write_text('#!'+sys.executable+'\n'+STUB);f.chmod(0o700)
            env=dict(os.environ,PATH=str(root)+os.pathsep+os.environ['PATH'],FP4_TEST_MODE=mode,STRATA_NVFP4_GU_NTA='2',FP4_TEST_PID=str(root/'pid'))
            output=root/'results'
            result=subprocess.run([sys.executable,str(LAUNCHER),'--worker-one','worker-one','--worker-two','worker-two',
                '--up-root','test:1','--down-root','test:2','--model',str(model),'--output',str(output),
                '--primary-binary',str(root/'primary'),'--worker-binary','unused','--timeout','30',*(['--worker-nt1-tasks'] if mode.startswith('tasks') else []),*(['--worker-nvfp4-nta'] if mode=='nta' else [])],
                env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=8)
            summary=json.loads((output/'summary.json').read_text())
            if (root/'pid').exists():
                with self.assertRaises(ProcessLookupError):os.kill(int((root/'pid').read_text()),0)
            return result.returncode,summary
    def test_worker_nta_environment(self):
        code,summary=self.run_stub('nta')
        self.assertEqual(code,0);self.assertEqual(summary['status'],'passed')
        self.assertEqual(summary['worker_environment']['STRATA_NVFP4_GU_NTA'],'1')
        self.assertEqual(summary['worker_environment']['STRATA_NVFP4_DOWN_NTA'],'1')
    def test_default_has_no_worker_environment(self):
        code,summary=self.run_stub('success')
        self.assertEqual(code,0);self.assertNotIn('worker_environment',summary)
    def test_worker_task_policy_forwarding(self):
        code,summary=self.run_stub('tasks')
        self.assertEqual(code,0);self.assertEqual(summary['status'],'passed')
    def test_missing_task_policy_evidence(self):
        code,summary=self.run_stub('tasks-missing')
        self.assertNotEqual(code,0);self.assertIn('task-policy evidence',summary['error'])
    def test_success(self):
        code,summary=self.run_stub('success')
        self.assertEqual(code,0);self.assertEqual(summary['status'],'passed');self.assertEqual(summary['cases'],16)
    def test_missing_request_evidence(self):
        code,summary=self.run_stub('incomplete')
        self.assertNotEqual(code,0);self.assertEqual(summary['status'],'failed')
        self.assertIn('request evidence',summary['error'])
    def test_worker_failure_reaps_primary(self):
        code,summary=self.run_stub('worker-fails')
        self.assertNotEqual(code,0);self.assertEqual(summary['status'],'failed')
        self.assertTrue(all(c is not None for c in summary['exit_codes']))
unittest.main()
