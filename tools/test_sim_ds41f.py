"""DeepSeek V4.1 Flash model: census, memory fit, Engram, kernel bounds, validation rows, machine ordering."""
import contextlib
import dataclasses
import io
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "sim"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import ds41f  # noqa: E402
import a64fx  # noqa: E402
import decode  # noqa: E402
import hw as hwmod  # noqa: E402
import kernels  # noqa: E402
from config import RunConfig  # noqa: E402

GIB = 2 ** 30


class CensusTest(unittest.TestCase):
    def test_experts(self):
        g = ds41f.GEOMETRY
        self.assertEqual(g.expert_params(), 35_389_440)
        self.assertAlmostEqual(ds41f.pack("mxfp4").total_bytes() / 1e9, 288.78, delta=0.05)      # ds41f.md:2519
        self.assertAlmostEqual(ds41f.pack("mxfp4").bytes_per_token() / 1e9, 4.51, delta=0.01)
        self.assertAlmostEqual(ds41f.pack("iq2xxs_q2k").total_bytes() / 1e9, 152.9, delta=0.2)    # the public MixedQ2 GGUF
        self.assertAlmostEqual(ds41f.pack("iq2xxs_q2k").bpw(), 2.25, delta=0.01)
        self.assertAlmostEqual(ds41f.pack("iq1s").total_bytes() / 1e9, 106.2, delta=0.2)

    def test_engram_and_trunk(self):
        g = ds41f.GEOMETRY
        self.assertEqual(g.engram_bytes(), 202_758_032_400)
        self.assertEqual(g.engram_rows_per_token(), 48)
        self.assertAlmostEqual(g.trunk_fp8_bytes / 1e9, 7.20, delta=0.01)
        self.assertAlmostEqual(g.kv_bytes_per_token * 1_048_576 / 1e9, 0.93, delta=0.01)


class MemoryFitTest(unittest.TestCase):
    def test_tr16(self):
        self.assertTrue(ds41f.memory_fit(ds41f.Plan(machine="tr16", pack="iq1s"))["fits"])
        self.assertTrue(ds41f.memory_fit(ds41f.Plan(machine="tr16", pack="iq2xxs_q2k_reap256"))["fits"])
        fit = ds41f.memory_fit(ds41f.Plan(machine="tr16", pack="iq2xxs_q2k"))
        self.assertFalse(fit["fits"])
        self.assertIn("RAM", fit["notes"][0])

    def test_two_node_and_xeon(self):
        self.assertTrue(ds41f.memory_fit(ds41f.Plan(machine="tr16+b550", pack="iq2xxs_q2k"))["fits"])
        self.assertTrue(ds41f.memory_fit(ds41f.Plan(machine="xeon_v100", pack="iq2xxs_q2k", trunk="q8", head="q8"))["fits"])
        self.assertFalse(ds41f.memory_fit(ds41f.Plan(machine="xeon_v100", pack="q23", trunk="q8", head="q8"))["fits"])

    def test_a64fx(self):
        self.assertTrue(ds41f.a64fx_memory_fit(ds41f.A64FXPlan(nodes=12, pack="mxfp4"))["fits"])
        self.assertFalse(ds41f.a64fx_memory_fit(ds41f.A64FXPlan(nodes=8, pack="mxfp4"))["fits"])
        self.assertFalse(ds41f.a64fx_memory_fit(ds41f.A64FXPlan(nodes=6, pack="mxfp4"))["fits"])
        self.assertTrue(ds41f.a64fx_memory_fit(ds41f.A64FXPlan(nodes=6, pack="iq2xxs_q2k"))["fits"])

    def test_ssd_budget(self):
        fit = ds41f.memory_fit(ds41f.Plan(machine="tr16", pack="iq1s"))
        self.assertGreater(fit["ssd_total_gib"], 280)       # Engram 189 GiB + pack 99 GiB + trunk


class EngramTest(unittest.TestCase):
    def test_exposed_under_a_millisecond_with_prefetch(self):
        d = ds41f.decode(ds41f.Plan(machine="tr16", pack="iq1s"))
        self.assertLess(d.engram_exposed_ms, 1.0)
        self.assertGreater(d.engram_ms, 0.1)
        slow = ds41f.decode(ds41f.Plan(machine="tr16", pack="iq1s", engram_prefetch=False))
        self.assertGreater(slow.engram_exposed_ms, d.engram_exposed_ms)


class KernelBoundTest(unittest.TestCase):
    def test_iq1s_is_compute_bound_on_tr16(self):
        d = ds41f.decode(ds41f.Plan(machine="tr16", pack="iq1s"))
        self.assertEqual(d.cpu_bound, "compute")
        fast = ds41f.decode(ds41f.Plan(machine="tr16", pack="iq1s", cpu_rate_override_gbps=60.0))
        self.assertGreater(fast.tok_s, d.tok_s)
        self.assertIn("IQ1", " ".join(ds41f.decode(ds41f.Plan(machine="tr16", pack="iq1m")).hypotheses))

    def test_more_bits_more_bytes(self):
        bytes_ = [ds41f.pack(p).bytes_per_token() for p in ("iq1s", "iq1m", "iq2xxs", "iq2xxs_q2k", "q23", "iq3xxs", "mxfp4")]
        self.assertEqual(bytes_, sorted(bytes_))


class ValidationTest(unittest.TestCase):
    def test_rows(self):
        rows = ds41f.validate()
        self.assertEqual(len(rows), 5)
        for r in rows:
            self.assertTrue(r["within"], r["name"])
        fitted = [r for r in rows if r["fit"]]
        self.assertEqual(len(fitted), 2)
        for r in fitted:
            self.assertLessEqual(abs(r["error"]), 0.05, r["name"])


class OrderingTest(unittest.TestCase):
    def test_xeon_faster_than_tr16_two_node_between(self):
        a = ds41f.decode(ds41f.Plan(machine="tr16+b550", pack="iq1s")).tok_s
        b = ds41f.decode(ds41f.Plan(machine="tr16", pack="iq1s")).tok_s
        c = ds41f.decode(ds41f.Plan(machine="xeon_v100", pack="iq1s", trunk="q8", head="q8")).tok_s
        self.assertGreater(a, b)
        self.assertGreater(c, a)

    def test_a64fx_redesign_and_nodes(self):
        cur = ds41f.a64fx_decode(ds41f.A64FXPlan(nodes=12, pack="mxfp4", profile="current"))
        new = ds41f.a64fx_decode(ds41f.A64FXPlan(nodes=12, pack="mxfp4", profile="redesign"))
        self.assertGreater(new["tok_s"], 2 * cur["tok_s"])
        six = ds41f.a64fx_decode(ds41f.A64FXPlan(nodes=6, pack="iq2xxs_q2k", profile="redesign"))
        twelve = ds41f.a64fx_decode(ds41f.A64FXPlan(nodes=12, pack="iq2xxs_q2k", profile="redesign"))
        self.assertLess(six["tok_s"], twelve["tok_s"])
        self.assertLess(ds41f.a64fx_prefill(ds41f.A64FXPlan(nodes=6, pack="iq2xxs_q2k", profile="redesign"))["tok_s"],
                        ds41f.a64fx_prefill(ds41f.A64FXPlan(nodes=12, pack="iq2xxs_q2k", profile="redesign"))["tok_s"])

    def test_dspark_needs_acceptance(self):
        plain = ds41f.decode(ds41f.Plan(machine="xeon_v100", pack="iq2xxs_q2k", trunk="q8", head="q8"))
        best = ds41f.decode(ds41f.Plan(machine="xeon_v100", pack="iq2xxs_q2k", trunk="q8", head="q8", width=3, acceptance="a64fx_best512"))
        self.assertGreater(best.tok_s, plain.tok_s)
        self.assertAlmostEqual(best.tokens_per_round, 2.93, delta=0.01)


class UntouchedTest(unittest.TestCase):
    def test_existing_simulators_unchanged(self):
        hw = hwmod.PRESETS["tr16+b550-ib"]()
        cfg = RunConfig(pack="reap50_q23", speculation="none", mtp_depth=0, decode_cache_mib=3584, tier_policy="static", remote_share=0.25)
        self.assertAlmostEqual(decode.simulate(hw, cfg, kernels.Params()).tok_s, 16.05, delta=0.02)
        self.assertAlmostEqual(a64fx.decode(a64fx.Layout(nodes=12, pack="ud_q4kxl", context=8192, ranks_per_node=1, trunk_tp=12,
                                                         expert_layout="ep_parts", parts=4)).tok_s, 34.2, delta=1.0)

    def test_cli(self):
        import glm_ds41f_sim
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = glm_ds41f_sim.main(["validate"])
        self.assertEqual(code, 0)
        self.assertIn("5 of 5 rows within tolerance", out.getvalue())


if __name__ == "__main__":
    unittest.main()
