"""Hy4-Flash simulator: census-derived geometry and decode arithmetic."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "sim"))

import hy4f  # noqa: E402

GIB = 2 ** 30


class GeometryTest(unittest.TestCase):
    def test_census(self):
        g = hy4f.GEOMETRY
        self.assertEqual((g.trunk_layers, g.moe_layers, g.experts, g.top_k), (42, 41, 512, 16))
        self.assertEqual(g.expert_bytes, 3 * 1536 * 1536 * 9 // 16)          # NVFP4: 36 B per 64 values
        self.assertAlmostEqual(g.expert_total_bytes() / GIB, 77.84, delta=0.01)
        self.assertAlmostEqual(g.routed_bytes_per_token() / 1e9, 2.61, delta=0.01)
        self.assertAlmostEqual(g.engram_table_bytes / GIB, 62.70, delta=0.01)
        self.assertEqual(g.engram_layers, (1, 11, 21, 31))


class DecodeTest(unittest.TestCase):
    def test_fit(self):
        m = hy4f.memory(hy4f.Plan())
        self.assertFalse(m["notes"])
        self.assertGreater(m["tier_gib"], 6)

    def test_cpu_only_bound(self):
        r = hy4f.decode(hy4f.Plan(hit_model="fixed", hit=0.0))
        h = hy4f.PROFILES["now"]
        self.assertAlmostEqual(r["experts_ms"], 2611.74 / h.cpu_lane_gbps + h.layer_ms * 41, delta=0.1)

    def test_monotone(self):
        lo = hy4f.decode(hy4f.Plan(hit_model="fixed", hit=0.3))["tok_s"]
        hi = hy4f.decode(hy4f.Plan(hit_model="fixed", hit=0.6))["tok_s"]
        split = hy4f.decode(hy4f.Plan(hit_model="fixed", hit=0.3, split_misses=True))["tok_s"]
        self.assertLess(lo, hi)
        self.assertLess(lo, split)


if __name__ == "__main__":
    unittest.main()
