"""Remote table descriptors are headers, never sparse local weight files."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import os
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('remote_pack',ROOT/'tools/dsv41/pack_gguf.py')
PG=importlib.util.module_from_spec(spec)
spec.loader.exec_module(PG)

class RemoteDescriptor(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        header=(ROOT/'tests/fixtures/dsv41/engram-converted-27oct-header.gguf').read_bytes()
        self.descriptor={'version':1,'size':110595290816,'sha256':'a'*64,'header_hex':header.hex()}
    def load(self):
        manifest=self.root/'descriptor.json'
        manifest.write_text(json.dumps(self.descriptor))
        return PG.remote_engram_manifest(manifest,self.root)
    def test_real_header_and_table_splits_without_local_weights(self):
        d,g=self.load()
        self.assertEqual(g.data_start,416)
        self.assertFalse(g.path.exists())
        kv={'general.architecture':'deepseek41','deepseek41.engram.layers':[1,14],
            'deepseek41.engram.num_embeddings':[384006168,384016682]}
        PG.split_ple(g,kv)
        self.assertEqual([t[0] for t in g.tensors],['blk.1.engram_embd.weight','blk.14.engram_embd.weight'])
        self.assertEqual(g.tensors[1][3],384006168*144)
    def test_short_or_long_source_rejected(self):
        for delta in (-1,1):
            self.descriptor['size']=110595290816+delta
            with self.assertRaises(SystemExit):self.load()
    def test_invalid_digest_rejected(self):
        self.descriptor['sha256']='not-a-digest'
        with self.assertRaises(SystemExit):self.load()
    def test_wrong_table_format_rejected(self):
        self.descriptor['header_hex']=(ROOT/'tests/fixtures/dsv41/engram-final-header.gguf').read_bytes().hex()
        with self.assertRaises(SystemExit):self.load()
    def test_missing_header_data_rejected(self):
        self.descriptor['header_hex']=''
        with self.assertRaises((SystemExit, ValueError, EOFError, __import__("struct").error)):self.load()
    def test_bad_version_rejected(self):
        self.descriptor['version']=2
        with self.assertRaises(SystemExit):self.load()

class LaunchBackend(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith('linux'), 'owned process identity uses Linux procfs')
    def test_reader_override_is_recorded_and_live_setting_cannot_silently_change(self):
        spec=importlib.util.spec_from_file_location('remote_launcher',ROOT/'tools/dsv41/remote_engram.py')
        launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
        original_check_output=subprocess.check_output
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); observed=root/'readers.txt'; helper=root/'helper'
            helper.write_text('#!'+sys.executable+'\nimport os,pathlib,time\n'
                              "pathlib.Path(os.environ['READER_TEST_FILE']).write_text(os.environ.get('STRATA_ENGRAM_READERS','8'))\n"
                              'time.sleep(60)\n')
            helper.chmod(0o700)
            cfg={'ssh_host':'unused','remote_state_dir':str(root/'state'),'remote_exe':str(helper),
                 'remote_file':'unused','engram_endpoint':'127.0.0.1:1','remote_readers':24}
            def local_remote(args, **kwargs):
                command=shlex.split(args[-1])
                self.assertEqual(command[:2],['python3','-c'])
                return original_check_output([sys.executable,*command[1:]],text=True,timeout=10,stderr=subprocess.PIPE)
            with mock.patch.dict(os.environ,{'READER_TEST_FILE':str(observed),'STRATA_ENGRAM_READERS':'13'}), \
                 mock.patch.object(launcher.subprocess,'check_output',side_effect=local_remote):
                try:
                    started=launcher.remote(cfg,'start')
                    self.assertTrue(started['running']); self.assertEqual(started['readers'],24)
                    deadline=time.monotonic()+3
                    while not observed.exists() and time.monotonic()<deadline:time.sleep(.01)
                    self.assertEqual(observed.read_text(),'24')
                    with self.assertRaises(subprocess.CalledProcessError):
                        launcher.remote({**cfg,'remote_readers':32},'start')
                    status=launcher.remote(cfg,'status')
                    self.assertEqual(status['pid'],started['pid']); self.assertEqual(status['readers'],24)
                finally:
                    stopped=launcher.remote(cfg,'stop')
                self.assertFalse(stopped['running'])
                self.assertFalse((root/'state/process.json').exists())

    def test_remote_reader_count_is_bounded_and_defaults_to_existing_pool(self):
        spec=importlib.util.spec_from_file_location('remote_launcher',ROOT/'tools/dsv41/remote_engram.py')
        launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
        self.assertEqual(launcher.remote_reader_count({}),8)
        for count in (1,8,24,64):
            self.assertEqual(launcher.remote_reader_count({'remote_readers':count}),count)
        for count in (0,-1,65,True,8.0,'8',None):
            with self.assertRaises(ValueError):
                launcher.remote_reader_count({'remote_readers':count})

    def test_api_selects_real_engine_and_loopback_explicitly(self):
        spec=importlib.util.spec_from_file_location('remote_launcher',ROOT/'tools/dsv41/remote_engram.py')
        launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
        args=launcher.api_command('run.json',8080)
        self.assertEqual(args[args.index('--engine')+1],'strata')
        self.assertEqual(args[args.index('--host')+1],'127.0.0.1')
        self.assertEqual(args[args.index('--config')+1],'run.json')

if __name__=='__main__':unittest.main()
