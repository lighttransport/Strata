import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from glm_external_draft_worker import bridge

class TinyTokenizer:
    def decode(self, ids, errors='strict'):
        return ''.join(['a','b','ab','c'][i] for i in ids)
    def encode(self, text, parse_special=True):
        out=[]
        while text:
            if text.startswith('ab'):out.append(2);text=text[2:]
            else:out.append({'a':0,'b':1,'c':3}[text[0]]);text=text[1:]
        return out

class DraftBridgeTest(unittest.TestCase):
    def test_stable_prefix_and_bound(self):
        self.assertEqual(bridge(TinyTokenizer(),[0],'ccc',2),[3,3])
    def test_merging_across_committed_boundary_is_rejected(self):
        self.assertEqual(bridge(TinyTokenizer(),[0],'b',2),[])
    def test_noncanonical_committed_ids_remain_unchanged(self):
        context=[0,1]
        self.assertEqual(bridge(TinyTokenizer(),context,'c',2),[3])
        self.assertEqual(context,[0,1])

@unittest.skipUnless(shutil.which('c++'),'C++ compiler unavailable')
class DraftTransportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        code=cls.root/'driver.cpp';code.write_text('''#include "strata/spec/external_worker.hpp"
int main(int argc,char**argv){strata::spec::ExternalWorker w(argv[1]);
if(argc>2){if(!w.propose({1,2},2,10).empty())return 4;
return w.propose({1,2},2,10).empty()?0:5;}
auto a=w.propose({1,2},2,10);if(a!=std::vector<int>{3,4})return 1;
auto b=w.propose({1,2,3},2,10);if(!b.empty())return 2;
if(!w.propose({1,2,3},2,10).empty())return 3;}
''')
        cls.exe=cls.root/'driver';subprocess.run(['c++','-std=c++20','-Wall','-Wextra','-Werror','-I'+str(Path(__file__).resolve().parents[1]/'include'),str(code),'-o',str(cls.exe)],check=True)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_invalid_reply_disables_worker_without_corrupting_proposals(self):
        worker=self.root/'fake.py';worker.write_text('''import json,sys
print('{"protocol":1,"ready":true,"vram_mib":0}',flush=True)
for line in sys.stdin:
 r=json.loads(line);print(json.dumps({'protocol':1,'request_id':r['request_id'],'tokens':[3,4] if r['request_id']==1 else [100]}),flush=True)
''')
        config=self.root/'config.json';config.write_text(json.dumps({'argv':['python3',str(worker)],'timeout_ms':100}))
        p=subprocess.run([str(self.exe),str(config)],capture_output=True,text=True,timeout=5)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('EXTERNAL_DRAFT_DISABLED',p.stderr)

    def test_protocol_faults_and_timeout_fall_back(self):
        faults = {
            'wrong_id': "print(json.dumps({'protocol':1,'request_id':999,'tokens':[3]}),flush=True)",
            'oversized': "print(' ' * 8193 + '{}',flush=True)",
            'eof': "sys.exit(0)",
            'timeout': "time.sleep(10)",
            'too_many': "print(json.dumps({'protocol':1,'request_id':r['request_id'],'tokens':[3,4,5]}),flush=True)",
        }
        for name, reply in faults.items():
            with self.subTest(name=name):
                worker=self.root/(name+'.py')
                worker.write_text('import json,sys,time\nprint(\'{"protocol":1,"ready":true,"vram_mib":0}\',flush=True)\nr=json.loads(sys.stdin.readline())\n'+reply+'\n')
                config=self.root/(name+'.json')
                config.write_text(json.dumps({'argv':['python3',str(worker)],'timeout_ms':25}))
                p=subprocess.run([str(self.exe),str(config),'failure'],capture_output=True,text=True,timeout=3)
                self.assertEqual(p.returncode,0,p.stderr)
                self.assertIn('EXTERNAL_DRAFT_DISABLED',p.stderr)

if __name__=='__main__':unittest.main()
