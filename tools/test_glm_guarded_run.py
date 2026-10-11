import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import glm_guarded_run as guard


class FakeProcess:
    pid=123
    def __init__(self):self.returncode=None
    def poll(self):return self.returncode
    def wait(self,timeout=None):self.returncode=-9;return self.returncode


class OwnedGuardTest(unittest.TestCase):
    def test_task_event_records_order_boot_and_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tasks.jsonl'
            with patch.object(guard.os,'fsync') as sync:
                guard.task_event(path,'prefill-2k','begin',tokens=2048)
                guard.task_event(path,'prefill-2k','passed')
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row['state'] for row in rows],['begin','passed'])
            self.assertEqual(rows[0]['tokens'],2048)
            self.assertTrue(rows[0]['boot_id'])
            self.assertTrue(rows[0]['timestamp'].endswith('+00:00'))
            self.assertEqual(sync.call_count,2)

    def test_fault_cleanup_cannot_follow_an_untrusted_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);exe=root/'exe';exe.write_bytes(b'fake')
            config=root/'config.json';config.write_text(json.dumps({'exe':str(exe)}))
            prompt=root/'prompt';prompt.write_text('1,2');output=root/'run'
            output.with_suffix('.manifest.json').write_text(json.dumps({'command':['--unit=unrelated']}))
            process=FakeProcess()
            with patch.object(guard,'gpu_devices',return_value=[Path('GPU')]), \
                    patch.object(guard,'identities',return_value={'GPU':'02105075'}), \
                    patch.object(guard,'kernel_log',side_effect=['','GPU reset begin','GPU reset begin']), \
                    patch('glm_low_memory_bench.clear_model_cache',return_value=[]), \
                    patch.object(guard.uuid,'uuid4',return_value=SimpleNamespace(hex='a'*32)), \
                    patch.object(guard.subprocess,'Popen',return_value=process) as launch, \
                    patch.object(guard.subprocess,'run') as cleanup, \
                    patch.object(guard.os,'killpg') as kill:
                result=guard.run(config,prompt,output,tokens=2,timeout=10)
            self.assertFalse(result['passed'])
            self.assertEqual(result['termination_reason'],'hardware/GPU fault')
            self.assertEqual(cleanup.call_args.args[0][-1],'strata-glm-opt-'+'a'*32+'.service')
            self.assertIn('--property=MemoryMax=60G',launch.call_args.args[0])
            self.assertIn('--property=MemorySwapMax=0',launch.call_args.args[0])
            kill.assert_called_once()

    def test_already_failed_boot_does_not_launch(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(guard,'gpu_devices',return_value=[Path('GPU')]), \
                patch.object(guard,'kernel_log',return_value='amdgpu: VM protection fault'), \
                patch.object(guard.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'already contains'):
                guard.run(Path(directory)/'cfg',Path(directory)/'ids',Path(directory)/'out')
            launch.assert_not_called()

    def test_previous_run_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'run';path=output.with_suffix('.guard.json')
            path.write_text('frozen evidence')
            with patch.object(guard.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(FileExistsError,'fresh output'):
                    guard.run(Path(directory)/'cfg',Path(directory)/'ids',output)
            self.assertEqual(path.read_text(),'frozen evidence')
            launch.assert_not_called()

    def test_short_or_disappeared_pci_configuration_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for value in [b'',b'\xff'*4,b'\0'*4]:
                (root/'config').write_bytes(value)
                with self.assertRaisesRegex(RuntimeError,'identity lost'):guard.identities([root])


if __name__=='__main__':unittest.main()
