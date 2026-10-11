import json
from pathlib import Path
import tempfile
import unittest
from glm_diagnostic_status import inspect


class DiagnosticStatusTest(unittest.TestCase):
    def test_unfinished_old_boot_cannot_be_reported_as_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/'run.guard.json').write_text(json.dumps(dict(boot_id='old',passed=False)))
            (p/'run.engine.jsonl').write_text('{"unix_ms":1000,"phase":"layer_begin","layer":7}\n{"partial":')
            report=inspect(p,boot_id='new');row=report['runs'][0]
            self.assertEqual(row['state'],'unfinished across reboot')
            self.assertEqual(row['last_engine_phase']['layer'],7)
            self.assertEqual(row['malformed_log_lines'],1)
            self.assertEqual(row['last_phase_utc'],'1970-01-01T00:00:01+00:00')

    def test_completed_software_failure_is_distinct_from_reboot(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/'run.guard.json').write_text(json.dumps(dict(boot_id='old',passed=False,exit_code=1,faults=[])))
            row=inspect(p,boot_id='new')['runs'][0]
            self.assertEqual(row['state'],'failed')
            self.assertEqual(row['faults'],[])


if __name__=='__main__':
    unittest.main()
