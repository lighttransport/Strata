import json
import tempfile
import unittest
from pathlib import Path

from glm_eval_compare import bootstrap_diff, compare, load, mcnemar_exact


def results(passed):
    return {f't{i}': {'id': f't{i}', 'passed': p} for i, p in enumerate(passed)}


class EvalCompareTest(unittest.TestCase):
    def test_mcnemar_exact_values(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(0, 5), 2 / 32)          # one-sided 1/32, doubled
        self.assertAlmostEqual(mcnemar_exact(1, 9), 2 * 11 / 1024)    # (C(10,0)+C(10,1))/2^10, doubled
        self.assertEqual(mcnemar_exact(5, 5), 1.0)

    def test_identical_results_pass(self):
        ref = results([True] * 80 + [False] * 20)
        r = compare(ref, ref, margin=0.03)
        self.assertEqual((r['wins'], r['losses'], r['failures']), ([], [], []))
        self.assertEqual(r['ci95'], [0.0, 0.0])

    def test_clear_regression_fails(self):
        ref = results([True] * 100)
        cand = results([False] * 12 + [True] * 88)
        r = compare(ref, cand, margin=0.03)
        self.assertEqual(len(r['losses']), 12)
        self.assertLess(r['mcnemar_p'], 0.001)
        self.assertEqual(len(r['failures']), 2)

    def test_balanced_flips_do_not_fail_significance(self):
        ref = results([True] * 5 + [False] * 5 + [True] * 90)
        cand = results([False] * 5 + [True] * 5 + [True] * 90)
        r = compare(ref, cand)
        self.assertEqual(r['diff'], 0.0)
        self.assertEqual(r['failures'], [])

    def test_only_common_tasks_are_paired(self):
        ref = results([True, True, False])
        cand = {'t0': {'id': 't0', 'passed': True}, 'x': {'id': 'x', 'passed': False}}
        r = compare(ref, cand)
        self.assertEqual((r['tasks'], r['only_reference'], r['only_candidate']), (1, 2, 1))

    def test_bootstrap_is_deterministic(self):
        pairs = [(1, 0)] * 3 + [(1, 1)] * 47
        self.assertEqual(bootstrap_diff(pairs), bootstrap_diff(pairs))
        diff, lo, hi = bootstrap_diff(pairs)
        self.assertAlmostEqual(diff, -0.06)
        self.assertLessEqual(lo, diff)
        self.assertLessEqual(diff, hi)

    def test_load_rejects_bad_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'r.json'
            p.write_text(json.dumps({'tasks': [{'id': 'a', 'passed': 'yes'}]}))
            with self.assertRaises(ValueError):
                load(p)
            p.write_text(json.dumps({'tasks': [{'id': 'a', 'passed': True}, {'id': 'a', 'passed': False}]}))
            with self.assertRaises(ValueError):
                load(p)


if __name__ == '__main__':
    unittest.main()
