import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import memory_check as check


class MemoryCheckTests(unittest.TestCase):
    def test_parent_cgroup_budget_applies_to_unlimited_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = root / 'parent' / 'child'
            child.mkdir(parents=True)
            (root / 'memory.max').write_text('max')
            (child / 'memory.max').write_text('max')
            (child.parent / 'memory.max').write_text(str(4096 * 1048576))
            (child.parent / 'memory.current').write_text(str(1024 * 1048576))
            self.assertEqual(check.cgroup_available_mib(root, '/parent/child'), 3072)
            # A cgroup namespace may expose only its root, with a different
            # process-relative path; still honor that visible root's budget.
            (root / 'memory.max').write_text(str(2048 * 1048576))
            (root / 'memory.current').write_text(str(1024 * 1048576))
            self.assertEqual(check.cgroup_available_mib(root, '/unmounted/path'), 1024)

    def test_changed_memory_budget_prevents_stress_and_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(check, 'available_mib', side_effect=[8192, 4096]), \
                 patch.object(check.shutil, 'which', return_value='/fake/stress'), \
                 patch.object(check, 'run_stress') as stress, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(check.main(['--mode', 'stress', '--output', tmp]), 2)
                stress.assert_not_called()

    def test_pass_requires_clean_exit_and_explicit_pass(self):
        summary = 'Completed: 1.00M in 300.03s 100MB/s, with 0 hardware incidents, 0 errors\n'
        def status(text, code, **kwargs):
            return check.stress_status(text, code, requested_seconds=300, elapsed_seconds=305, **kwargs)
        self.assertTrue(status(summary + 'Found 0 hardware incidents\nStatus: PASS', 0)['passed'])
        for text, code in [('Status: PASS', 1), ('', 0),
                           ('Found 5 hardware incidents\nStatus: PASS', 0),
                           ('Hardware Error: miscompare\nStatus: PASS', 0),
                           ('CRC mismatch\nStatus: PASS', 0)]:
            self.assertFalse(status(summary + text, code)['passed'])
        self.assertFalse(status('Status: PASS', 0)['passed'])
        self.assertFalse(status('Completed: 1.00M in 0.01s\nStatus: PASS', 0)['passed'])
        self.assertFalse(status(summary + 'Status: PASS', 0, termination_reason='timeout')['passed'])
        self.assertFalse(status(summary + 'Status: PASS', 0, termination_reason='interrupted')['passed'])
        self.assertFalse(check.stress_status(summary + 'Status: PASS', 0,
                         requested_seconds=300, elapsed_seconds=.5)['passed'])

    def test_early_pass_cannot_launch_application(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / 'stress'
            fake.write_text('#!/usr/bin/env python3\n'
                            'print("Completed: 1.00M in 0.01s, with 0 hardware incidents, 0 errors")\n'
                            'print("Status: PASS")\n')
            fake.chmod(0o755)
            benchmark = type('Result', (), {'stdout': '{"kernel":"read_avx2","median_GB_s":40}\n'})()
            with patch.object(check, 'available_mib', return_value=8192), \
                 patch.object(check.subprocess, 'run', return_value=benchmark), \
                 patch.object(check.subprocess, 'call') as launch, \
                 contextlib.redirect_stdout(io.StringIO()):
                code = check.main(['--output', tmp, '--stressapptest', str(fake), '--threads', '1',
                                   '--seconds', '300', '--run', 'application'])
            self.assertEqual(code, 1)
            launch.assert_not_called()
            result = json.loads((root / 'result.json').read_text())
            self.assertFalse(result['passed'])
            self.assertFalse(result['stress']['duration_completed'])
            self.assertEqual(result['stress']['completed_seconds'], .01)

    def test_detected_mismatch_stops_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / 'stress'
            fake.write_text('#!/usr/bin/env python3\nimport time\n'
                            'print("Hardware Error: miscompare", flush=True)\ntime.sleep(60)\n')
            fake.chmod(0o755)
            result = check.run_stress(str(fake), 256, 30, 1, root)
            self.assertFalse(result['passed'])
            self.assertTrue(result['data_error_detected'])
            self.assertLess(result['elapsed_seconds'], 15)

    def test_failed_preflight_never_launches_application(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(check, 'available_mib', return_value=8192), \
                 patch.object(check.shutil, 'which', return_value='/fake/stress'), \
                 patch.object(check.subprocess, 'run', side_effect=OSError('compiler unavailable')), \
                 patch.object(check.subprocess, 'call') as launch, \
                 contextlib.redirect_stdout(io.StringIO()):
                code = check.main(['--output', tmp, '--run', 'application'])
                self.assertEqual(code, 2)
                launch.assert_not_called()
                self.assertFalse(json.loads((Path(tmp) / 'result.json').read_text())['passed'])

    def test_memory_budget_rejects_excess_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(check, 'available_mib', return_value=8192), \
                 patch.object(check.shutil, 'which', return_value='/fake/stress'), \
                 patch.object(check, 'run_stress') as stress, \
                 contextlib.redirect_stdout(io.StringIO()):
                code = check.main(['--mode', 'stress', '--memory-mib', '8000', '--output', tmp])
                self.assertEqual(code, 2)
                stress.assert_not_called()

    def test_launch_requires_stress_pass(self):
        for passed in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                benchmark = type('Result', (), {'stdout': '{"kernel":"read_avx2","median_GB_s":40}\n'})()
                with patch.object(check, 'available_mib', return_value=8192), \
                     patch.object(check.shutil, 'which', return_value='/fake/stress'), \
                     patch.object(check.subprocess, 'run', return_value=benchmark), \
                     patch.object(check, 'run_stress', return_value={'passed': passed}), \
                     patch.object(check.subprocess, 'call', return_value=7) as launch, \
                     contextlib.redirect_stdout(io.StringIO()):
                    code = check.main(['--output', tmp, '--run', 'application', '--flag'])
                    self.assertEqual(code, 7 if passed else 1)
                    if passed:
                        launch.assert_called_once_with(['application', '--flag'])
                    else:
                        launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
