"""Check that routing cache estimates do not leak future routes into training."""
import collections
import pathlib
import tempfile
import unittest

from glm_routing_cache import coverage, read_trace


class RoutingCacheTest(unittest.TestCase):
    def test_training_and_oracle_are_distinct(self):
        training = collections.Counter({(3, 0): 3, (3, 1): 1})
        evaluation = collections.Counter({(3, 1): 5})
        self.assertEqual(coverage(training, evaluation, {3: 10}, 10)["byte_hit_fraction"], 0)
        self.assertEqual(coverage(evaluation, evaluation, {3: 10}, 10)["byte_hit_fraction"], 1)

    def test_slots_respect_byte_budget(self):
        counts = collections.Counter({(3, 0): 5, (4, 0): 3})
        result = coverage(counts, counts, {3: 20, 4: 10}, 25)
        self.assertEqual(result["allocated_bytes"], 20)
        self.assertEqual(result["slots"], 1)
        self.assertAlmostEqual(result["byte_hit_fraction"], 100 / 130)

    def test_incomplete_or_repeated_trace_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "trace.csv"
            row = "7,3,0,1,2,3,4,5,6,7\n"
            path.write_text(row)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                read_trace(path, {3: 10, 4: 10})
            path.write_text(row * 2)
            with self.assertRaisesRegex(ValueError, "repeated"):
                read_trace(path, {3: 10})


if __name__ == "__main__":
    unittest.main()
