"""Process isolation and cleanup tests; no GPU/model required. Needs Unix sockets."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT=Path(__file__).resolve().parents[1]


class MixedLauncherTest(unittest.TestCase):
    def test_launch_and_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            worker=directory/'worker.py';primary=directory/'primary.py'
            worker.write_text('#!'+sys.executable+'\n'+textwrap.dedent('''
                import json,os,socket,struct,sys,time
                if sys.argv[1]=='--backend': print('hip');raise SystemExit(0)
                if sys.argv[1]=='--protocol-size': print(75549016);raise SystemExit(0)
                if sys.argv[1]=='--protocol-layout': print(json.dumps(dict(version=int(os.environ.get('STRATA_GLM_MIXED_PROTOCOL','4')),bytes=75549016)));raise SystemExit(0)
                open(os.environ['FAKE_WORKER_PID'],'w').write(str(os.getpid()))
                channel=socket.socket(fileno=int(sys.argv[3]))
                channel.send(struct.pack('QII',0x474d5831,0,0))
                if os.environ.get('FAKE_FAIL'):
                    time.sleep(.2);raise SystemExit(7)
                while True:
                    value=channel.recv(16)
                    if not value or struct.unpack('QII',value)[0]==0: break
                    channel.send(value)
                '''))
            primary.write_text('#!'+sys.executable+'\n'+textwrap.dedent('''
                import json,mmap,os,socket,struct,sys,time
                if sys.argv[1]=='--mixed-backend': print('cuda');raise SystemExit(0)
                if sys.argv[1]=='--protocol-layout': print(json.dumps(dict(version=int(os.environ.get('STRATA_GLM_MIXED_PROTOCOL','4')),bytes=75549016)));raise SystemExit(0)
                open(os.environ['FAKE_PRIMARY_PID'],'w').write(str(os.getpid()))
                open(os.environ['FAKE_ARGV'],'w').write(json.dumps(sys.argv[1:]))
                channel=socket.socket(fileno=int(os.environ['STRATA_GLM_MIXED_SOCKET_FD']))
                assert struct.unpack('QII',channel.recv(16))[0]==0x474d5831
                if os.environ.get('FAKE_FAIL'): time.sleep(60)
                channel.send(struct.pack('QII',1,0,0));assert struct.unpack('QII',channel.recv(16))[0]==1
                mapping=mmap.mmap(int(os.environ['STRATA_GLM_MIXED_MEMORY_FD']),64)
                struct.pack_into('I',mapping,28,1);channel.send(struct.pack('QII',0,0,0))
                print(42)
                '''))
            worker.chmod(0o700);primary.chmod(0o700)
            for failed,entrypoint,version in ((failed,entrypoint,version) for version in ('4','5')
                    for failed,entrypoint in ((False,'launcher'),(True,'launcher'),(False,'serve'),(False,'worker'))):
                env=os.environ.copy()
                env['STRATA_GLM_MIXED_PROTOCOL']=version
                env.pop('STRATA_GLM_MIXED_EXCHANGE',None)
                env.update(FAKE_WORKER_PID=str(directory/'worker.pid'),FAKE_PRIMARY_PID=str(directory/'primary.pid'),FAKE_ARGV=str(directory/'argv.json'))
                env.update(STRATA_GLM_MIXED_PRIMARY=str(primary),STRATA_GLM_MIXED_WORKER=str(worker))
                if failed: env['FAKE_FAIL']='1'
                report=directory/(entrypoint+'-v'+version+('-failed.json' if failed else '-success.json'))
                cmd=[sys.executable,str(ROOT/'tools/glm_mixed_run.py'),'--primary',str(primary),'--worker',str(worker),
                     '--model','mock.gguf','--primary-worker-mode','--report',str(report),'--','test']
                if entrypoint!='launcher':
                    cmd=[sys.executable,str(ROOT/'tools/glm_mixed_decode.py'),'--'+entrypoint,'mock.gguf','test']
                result=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=15)
                self.assertEqual(result.returncode==0,not failed,result.stderr)
                if not failed:
                    self.assertEqual(result.stdout.strip(),'42')
                    self.assertEqual(json.loads((directory/'argv.json').read_text()),['--serve','mock.gguf','test'])
                    if entrypoint=='launcher': self.assertEqual(json.loads(report.read_text())['returncode'],0)
                for pidfile in ('primary.pid','worker.pid'):
                    pid=int((directory/pidfile).read_text())
                    with self.assertRaises(ProcessLookupError): os.kill(pid,0)


if __name__=='__main__': unittest.main()
