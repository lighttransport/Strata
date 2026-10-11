import contextlib
import io
import json
from types import SimpleNamespace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import system_check as check


class SystemCheckTests(unittest.TestCase):
    def test_bad_pci_identity_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'config').write_bytes(b'\xff'*4)
            with self.assertRaisesRegex(RuntimeError, 'PCI identity unavailable'):
                check.pci_identity(path)

    def test_existing_fault_refuses_launch(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(check, 'kernel_log', return_value='amdgpu: VM page fault'), \
                patch.object(check.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'refusing GPU launch'):
                check.run_gpu(['unused'], Path(directory), 1, [])
            launch.assert_not_called()

    def test_zero_exit_without_final_verification_fails(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(check, 'kernel_log', return_value=''):
            result = check.run_gpu([sys.executable, '-c', 'print("{}")'], Path(directory), 2, [])
            self.assertFalse(result['passed'])
            self.assertEqual(result['returncode'], 0)

    def test_timeout_fails_and_stops_only_child(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(check, 'kernel_log', return_value=''):
            result = check.run_gpu([sys.executable, '-c', 'import time; time.sleep(30)'],
                                   Path(directory), 0.05, [])
            self.assertFalse(result['passed'])
            self.assertEqual(result['termination_reason'], 'GPU timeout')
            self.assertIsNotNone(result['returncode'])

    def test_launch_gate_uses_verification_and_propagates_app_exit(self):
        for verified in (False, True):
            with self.subTest(verified=verified), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)

                def run(command, **kwargs):
                    if '--mode' in command:
                        (output/'ram').mkdir()
                        (output/'ram/result.json').write_text(json.dumps({'passed': True}))
                        return SimpleNamespace(returncode=0, stderr='')
                    if command[0] == 'g++':
                        return SimpleNamespace(returncode=0)
                    return SimpleNamespace(stdout=json.dumps({'verified': verified})+'\n'+
                                           json.dumps({'verified': verified}), returncode=0)

                with patch.object(check.subprocess, 'run', side_effect=run), \
                        patch.object(check.subprocess, 'call', return_value=7) as launch, \
                        contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    code = check.main(['--threads', '1', '--array-mib', '16', '--output', directory,
                                       '--run', 'app'])
                self.assertEqual(code, 7 if verified else 2)
                self.assertEqual(launch.call_count, int(verified))
                self.assertEqual(json.loads((output/'result.json').read_text())['passed'], verified)

    def test_boot_default_text_not_fault(self):
        self.assertIsNone(check.FAULT.search('BOOT_IMAGE=vmlinuz root=UUID=abc ro default_hugepagesz=1G'))
        self.assertIsNotNone(check.FAULT.search('amdgpu VM protection fault'))


if __name__ == '__main__':
    unittest.main()
