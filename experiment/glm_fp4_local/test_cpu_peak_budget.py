"""Check payload accounting and prevent reusing FP4 counters for Q2/Q3."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

SCRIPT = Path(__file__).with_name('cpu_peak_budget.py')


class PeakBudgetTest(unittest.TestCase):
    def run_budget(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), '--memory-mts', '2400',
             '--channels', '4', '--cores', '16', '--ghz', '3', *args],
            capture_output=True, text=True)

    def test_q23_payload_and_no_inherited_counter(self):
        for fmt, byte_count in [('q2_k', 84), ('q3_k', 110),
                                ('q2q3', 278 / 3),
                                ('q2q3-expanded', 282 / 3)]:
            r = self.run_budget('--format', fmt)
            self.assertEqual(r.returncode, 0, r.stderr)
            data = json.loads(r.stdout)
            self.assertAlmostEqual(data['packed_bytes_per_weight'], byte_count / 256)
            self.assertAlmostEqual(data['target_payload_gb_s'], 69.12)
            self.assertAlmostEqual(data['target_gweights_s'],
                                   69.12 * 256 / byte_count)
            self.assertAlmostEqual(data['max_cycles_per_weight'][0]['cycles'],
                                   48 / data['target_gweights_s'])
            self.assertEqual(data['observed_counter_projection'], [])
            self.assertIsNone(data['assumptions']['counter_summary'])

    def test_non_fp4_evidence_needs_explicit_count(self):
        r = self.run_budget('--format', 'q2q3', '--counter-summary', 'missing.json')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('requires --weights-per-probe', r.stderr)

    def test_existing_fp4_default(self):
        r = self.run_budget()
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)
        self.assertEqual(data['packed_bytes_per_weight'], 36 / 64)
        self.assertTrue(data['observed_counter_projection'])
        self.assertAlmostEqual(data['target_gweights_s'], 122.88)


if __name__ == '__main__':
    unittest.main()
