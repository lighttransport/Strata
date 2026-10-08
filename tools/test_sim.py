"""Tests for tools/sim: byte census, VRAM plan, calibrated predictions, monotonic sanity, CLI smoke.

Run: python3 -m unittest tools.test_sim   (system python3 with numpy; the trace tests skip without traces)
"""
import io
import json
import pathlib
import sys
import unittest
from contextlib import redirect_stdout

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "sim"))

import calibration  # noqa: E402
import decode  # noqa: E402
import glm_sim  # noqa: E402
import hw  # noqa: E402
import kernels  # noqa: E402
import model  # noqa: E402
import prefill  # noqa: E402
import routing  # noqa: E402
import vram  # noqa: E402
from config import RunConfig  # noqa: E402


class ModelTest(unittest.TestCase):
    def test_pack_census_matches_gguf_headers(self):
        self.assertEqual(model.pack("q2_orig").routed_total_bytes(), 99_033_808_896)
        self.assertEqual(model.pack("q23").routed_total_bytes(), 111_188_901_888)
        self.assertEqual(model.pack("reap50_q23").routed_total_bytes(), 55_094_280_192)
        self.assertEqual(model.pack("q23").expert_bytes(3), 9_109_504)
        self.assertEqual(model.pack("q2_orig").expert_bytes(3), 8_060_928)
        self.assertEqual(model.pack("q8").expert_bytes(3), 26_738_688)
        self.assertAlmostEqual(model.pack("q23").routed_bytes_per_token() / 1e9, 3.09, delta=0.01)

    def test_dense_bytes_and_macs(self):
        self.assertAlmostEqual(model.dense_gpu_bytes_per_token() / 1e9, 6.45, delta=0.05)
        self.assertEqual(model.GEOMETRY.routed_macs_per_token(), 8 * 42 * 3 * 4096 * 2048)


class RoutingTest(unittest.TestCase):
    def test_tokens_per_round(self):
        self.assertAlmostEqual(routing.tokens_per_round(2, "prime"), 2.89, delta=0.05)
        self.assertAlmostEqual(routing.tokens_per_round(3, "prime"), 3.74, delta=0.1)
        self.assertEqual(routing.tokens_per_round(0), 1.0)

    def test_union_ratio(self):
        self.assertEqual(routing.union_ratio(1), 1.0)
        self.assertAlmostEqual(routing.union_ratio(3), 2.46)
        self.assertAlmostEqual(routing.union_ratio(8), 5.3, delta=0.25)    # traces: 5.24-5.42
        self.assertAlmostEqual(routing.union_ratio(16), 8.7, delta=0.5)    # traces: 8.41-8.94
        self.assertLess(routing.union_ratio(8), routing.union_ratio(8, consecutive=False))

    def test_tier_curve_matches_calibration_points(self):
        curve = routing.TierCurve()
        for slots, share in ((400, 0.181), (500, 0.205), (750, 0.255)):
            self.assertAlmostEqual(curve.hit(slots / 12096, policy="static_prior"), share, delta=0.01)
        self.assertAlmostEqual(curve.hit(565 / 12096, policy="adaptive", affinity=0.10), 0.528, delta=0.03)
        self.assertLess(curve.hit(565 / 12096, policy="static"), curve.hit(565 / 12096, policy="adaptive"))

    @unittest.skipUnless(routing.default_trace_paths() and routing.default_prior_path().exists(), "no traces")
    def test_trace_replay(self):
        trace = routing.read_routes(routing.default_trace_paths()[0])
        self.assertAlmostEqual(routing.trace_union_ratio(trace, 3), 2.45, delta=0.25)
        adaptive = routing.trace_tier_hit(trace, routing.default_prior_path(), 565, True)
        static = routing.trace_tier_hit(trace, routing.default_prior_path(), 565, False)
        self.assertGreater(adaptive, static)
        self.assertAlmostEqual(adaptive, 0.22, delta=0.06)


class VramTest(unittest.TestCase):
    def test_v5_tier(self):
        plan = vram.plan(hw.tr16(), model.pack("q23"), 4096, mtp_depth=2, budget_mib=14336, reserve_mib=512,
                         decode_cache_mib=5632)
        self.assertAlmostEqual(plan.tier_mib, 4908, delta=60)
        self.assertAlmostEqual(plan.slots, 565, delta=8)

    def test_tier_shrinks_with_reserve(self):
        a = vram.plan(hw.tr16(), model.pack("q23"), 4096, 2, budget_mib=14336, reserve_mib=512)
        b = vram.plan(hw.tr16(), model.pack("q23"), 4096, 2, budget_mib=14336, reserve_mib=2048)
        self.assertGreater(a.tier_mib, b.tier_mib)


class PredictionTest(unittest.TestCase):
    """Calibrated predictions stay inside the record tolerances."""

    def test_fit_records_within_tolerance(self):
        params = calibration.load_params()
        rows = calibration.evaluate(calibration.load_records(), params, only_fit=True)
        misses = [(r["name"], r["error"]) for r in rows if not r["within"]]
        self.assertEqual(misses, [])
        mean = sum(abs(r["error"]) for r in rows) / len(rows)
        self.assertLess(mean, 0.10)

    def test_headline_numbers(self):
        params = calibration.load_params()
        H = hw.tr16()
        d = decode.simulate(H, RunConfig(mtp_depth=2, decode_cache_mib=4800, acceptance="prime"), params)
        self.assertAlmostEqual(d.tok_s, 21.0, delta=3.0)
        self.assertEqual(d.bottleneck, "cpu")
        p = prefill.simulate(H, RunConfig(prompt=8192, prefill_chunk=8192, prefetch_groups=12, context=12288), params)
        self.assertAlmostEqual(p.tok_s, 395, delta=60)
        q2 = decode.simulate(H, RunConfig(pack="q2_orig", mtp_depth=0, decode_cache_mib=0, split_verify=False), params)
        self.assertAlmostEqual(q2.tok_s, 9.74, delta=1.5)
        self.assertEqual(q2.step.cpu_bound, "compute")


class MonotonicTest(unittest.TestCase):
    def setUp(self):
        self.params = kernels.Params()
        self.cfg = RunConfig(mtp_depth=2, decode_cache_mib=4800, acceptance="prime")

    def speed(self, hw_):
        return decode.simulate(hw_, self.cfg, self.params).tok_s

    def test_more_dram_bandwidth_is_faster(self):
        a, b = hw.tr16(), hw.tr16()
        b.memory.dram_gbps *= 2
        b.memory.per_core_gbps *= 2
        self.assertGreater(self.speed(b), self.speed(a))

    def test_more_vram_is_faster(self):
        a, b = hw.tr16(), hw.tr16()
        b.gpu.vram_mib += 8000
        self.assertGreater(self.speed(b), self.speed(a))

    def test_affinity_raises_hits_and_speed_until_gpu_bound(self):
        H = hw.tr16()
        base = decode.simulate(H, self.cfg, self.params)
        lossy = decode.simulate(H, self.cfg.copy(affinity=0.1), self.params)
        self.assertGreater(lossy.hit_bytes_share, base.hit_bytes_share)
        self.assertGreater(lossy.tok_s, base.tok_s)
        self.assertFalse(lossy.lossless)

    def test_pcie_bandwidth_speeds_up_prefill_and_streaming(self):
        a, b = hw.tr16(), hw.tr16()
        b.pcie.h2d_gbps = 25.0
        cfg = RunConfig(prompt=1024, prefill_chunk=1024)
        self.assertGreater(prefill.simulate(b, cfg).tok_s, prefill.simulate(a, cfg).tok_s)
        stream = RunConfig(placement="gpu_stream", mtp_depth=0, split_verify=False)
        self.assertGreater(decode.simulate(b, stream).tok_s, decode.simulate(a, stream).tok_s)

    def test_split_verify_beats_unsplit(self):
        H = hw.tr16()
        split = decode.simulate(H, self.cfg).tok_s
        unsplit = decode.simulate(H, self.cfg.copy(split_verify=False)).tok_s
        self.assertGreater(split, unsplit)

    def test_remote_share_helps_on_ib(self):
        H = hw.tr16_plus_b550_ib()
        cfg = RunConfig(mtp_depth=0, decode_cache_mib=3584, tier_policy="static", pack="reap50_q23")
        alone = decode.simulate(H, cfg).tok_s
        shared = decode.simulate(H, cfg.copy(remote_share=0.25)).tok_s
        self.assertGreater(shared, alone)

    def test_ram_tier_modes(self):
        H = hw.tr16()
        H.memory.gib = 60
        exact = decode.simulate(H, self.cfg.copy(placement="ram_tier", ram_mode="exact"))
        frozen = decode.simulate(H, self.cfg.copy(placement="ram_tier", ram_mode="frozen"))
        self.assertGreater(exact.disk_gb_per_token, 0.1)
        self.assertLess(exact.tok_s, 6)
        self.assertGreater(frozen.tok_s, 15)
        self.assertFalse(frozen.lossless)


class SpeculationTest(unittest.TestCase):
    def setUp(self):
        self.params = kernels.Params()
        self.H = hw.tr16()

    def test_dflash_acceptance_profiles_match_published_lengths(self):
        self.assertAlmostEqual(routing.tokens_per_round(7, "dflash_chat") - 1, 4.10, delta=0.15)
        self.assertAlmostEqual(routing.tokens_per_round(7, "dflash_code") - 1, 4.39, delta=0.15)
        self.assertAlmostEqual(routing.tokens_per_round(7, "dflash_math") - 1, 5.46, delta=0.15)

    def test_dflash_round_shape(self):
        cfg = RunConfig(speculation="dflash", draft_block=8, decode_cache_mib=-1)
        d = decode.simulate(self.H, cfg, self.params)
        self.assertEqual(d.step.width, 8)
        self.assertEqual(d.resync_ms, 0.0)
        self.assertGreater(d.draft_ms, 1.0)
        self.assertLess(d.draft_ms, 15.0)
        self.assertAlmostEqual(d.tokens_per_round, 5.42, delta=0.1)
        plan = decode.make_plan(self.H, cfg, model.pack("q23"))
        self.assertIn("draft_model", plan.items)
        self.assertNotIn("mtp_dense", plan.items)

    def test_verify_width_cap(self):
        capped = decode.simulate(self.H, RunConfig(speculation="dflash", draft_block=16), self.params)
        lifted = decode.simulate(self.H, RunConfig(speculation="dflash", draft_block=16, max_verify_width=16), self.params)
        self.assertEqual(capped.step.width, 8)
        self.assertEqual(lifted.step.width, 16)
        self.assertTrue(any("capped" in n for n in capped.notes))

    def test_levers_direction_and_lossless_flag(self):
        base = RunConfig(mtp_depth=2, acceptance="prime")
        b = decode.simulate(self.H, base, self.params)
        skip = decode.simulate(self.H, base.copy(expert_skip=0.1), self.params)
        q4 = decode.simulate(self.H, base.copy(dense_format="q4"), self.params)
        self.assertGreater(skip.tok_s, b.tok_s)
        self.assertFalse(skip.lossless)
        self.assertGreater(q4.tier_mib, b.tier_mib)
        self.assertGreaterEqual(q4.tok_s, b.tok_s)
        self.assertFalse(q4.lossless)
        self.assertTrue(b.lossless)

    def test_none_speculation_is_single_token(self):
        d = decode.simulate(self.H, RunConfig(speculation="none", mtp_depth=2), self.params)
        self.assertEqual(d.step.width, 1)
        self.assertEqual(d.tokens_per_round, 1.0)


class ArchitectureKnobTest(unittest.TestCase):
    def setUp(self):
        self.params = kernels.Params()
        self.H = hw.tr16()
        self.cfg = RunConfig(pack="q22", mtp_depth=3, acceptance="prime")

    def test_tier_compress_adds_slots_and_is_lossy(self):
        a = decode.simulate(self.H, self.cfg, self.params)
        b = decode.simulate(self.H, self.cfg.copy(tier_compress=1.4), self.params)
        self.assertAlmostEqual(b.tier_slots / a.tier_slots, 1.4, delta=0.02)
        self.assertGreater(b.hit_bytes_share, a.hit_bytes_share)
        self.assertLess(b.gpu_gb_per_token / b.hit_bytes_share, a.gpu_gb_per_token / a.hit_bytes_share)
        self.assertFalse(b.lossless)
        self.assertTrue(a.lossless)

    def test_pcie_share_moves_bytes_and_hurts_on_gen3(self):
        a = decode.simulate(self.H, self.cfg, self.params)
        b = decode.simulate(self.H, self.cfg.copy(pcie_share=0.1), self.params)
        self.assertGreater(b.step.pcie_bytes, 0)
        self.assertLess(b.cpu_gb_per_token, a.cpu_gb_per_token)
        self.assertGreater(b.cpu_ceiling_tok_s, a.cpu_ceiling_tok_s)
        self.assertLess(b.tok_s, a.tok_s)      # 7 GB/s of PCIe cannot keep up with the alternation
        self.assertTrue(b.lossless)

    def test_ceilings_bracket_the_prediction(self):
        d = decode.simulate(self.H, self.cfg, self.params)
        self.assertLessEqual(d.tok_s, d.cpu_ceiling_tok_s + 1e-6)
        self.assertLessEqual(d.tok_s, d.gpu_ceiling_tok_s + 1e-6)

    def test_param_override_cli(self):
        out = io.StringIO()
        with redirect_stdout(out):
            glm_sim.main(["estimate", "--hw", "tr16", "--pack", "q22", "--mtp", "3", "--param", "gpu_kernels_per_layer=10"])
        self.assertIn("GPU-bound", out.getvalue())
        with self.assertRaises(SystemExit):
            with redirect_stdout(io.StringIO()):
                glm_sim.main(["estimate", "--hw", "tr16", "--param", "no_such_param=1"])


class CliTest(unittest.TestCase):
    def run_cli(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = glm_sim.main(list(argv))
        return code, out.getvalue()

    def test_estimate(self):
        code, out = self.run_cli("estimate", "--hw", "tr16", "--mtp", "2", "--decode-cache-mib", "4800")
        self.assertEqual(code, 0)
        self.assertIn("DECODE", out)
        self.assertIn("PREFILL", out)

    def test_estimate_with_overrides_and_output(self):
        out_path = ROOT / "tools" / "sim" / "data" / ".test_estimate.json"
        try:
            code, _ = self.run_cli("estimate", "--hw", "tr16", "--set", "pcie.h2d_gbps=25", "--set", "memory.dram_gbps=180",
                                   "--output", str(out_path))
            self.assertEqual(code, 0)
            data = json.loads(out_path.read_text())
            self.assertEqual(data["hardware"]["pcie"]["h2d_gbps"], 25.0)
            self.assertIn("tok_s", data["decode"])
        finally:
            out_path.unlink(missing_ok=True)

    def test_sweep(self):
        code, out = self.run_cli("sweep", "--hw", "tr16", "--affinity", "0", "0.05", "--decode-cache-mib", "3000", "4800")
        self.assertEqual(code, 0)
        self.assertEqual(out.count("\n"), 2 + 4)

    def test_validate(self):
        code, out = self.run_cli("validate")
        self.assertEqual(code, 0)
        self.assertIn("records", out)

    def test_plan(self):
        code, out = self.run_cli("plan", "--hw", "tr16", "--target-decode", "30", "--top", "5", "--packs", "q23", "--lossless")
        self.assertEqual(code, 0)
        self.assertIn("best:", out)

    def test_estimate_dflash(self):
        code, out = self.run_cli("estimate", "--hw", "tr16", "--speculation", "dflash", "--draft-block", "8")
        self.assertEqual(code, 0)
        self.assertIn("dflash block 8", out)

    def test_hw(self):
        code, out = self.run_cli("hw", "--hw", "b550")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["cpu"]["cores"], 16)


if __name__ == "__main__":
    unittest.main()
