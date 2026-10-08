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


class ResearchKnobTest(unittest.TestCase):
    """Algorithms taken from the literature survey (docs/GLM_DECODE_RESEARCH.md)."""

    def setUp(self):
        self.params = kernels.Params()
        self.H = hw.tr16()
        self.cfg = RunConfig(pack="q22", mtp_depth=3, acceptance="prime")

    def test_lru_policy_hits_more_and_stays_lossless(self):
        a = decode.simulate(self.H, self.cfg, self.params)
        b = decode.simulate(self.H, self.cfg.copy(tier_policy="lru"), self.params)
        self.assertGreater(b.hit_bytes_share, a.hit_bytes_share)
        self.assertLess(b.hit_bytes_share, a.hit_bytes_share * 1.3)   # traces: 27 vs 24 % at this size
        self.assertTrue(b.lossless)

    def test_pcie_prefetch_scales_with_link_speed(self):
        a = decode.simulate(self.H, self.cfg, self.params)
        b = decode.simulate(self.H, self.cfg.copy(pcie_prefetch=True), self.params)
        fast = hw.tr16(); fast.pcie.h2d_gbps = 20.0
        c = decode.simulate(fast, self.cfg.copy(pcie_prefetch=True), self.params)
        self.assertGreater(b.tok_s, a.tok_s)
        self.assertGreater(c.tok_s, b.tok_s)
        self.assertGreater(b.step.pcie_bytes, 0)
        self.assertTrue(b.lossless)

    def test_expert_deferral_helps_ordinary_decode_most(self):
        plain = RunConfig(pack="q22", speculation="none")
        a = decode.simulate(self.H, plain, self.params)
        b = decode.simulate(self.H, plain.copy(expert_deferral=True), self.params)          # default share 0.3
        c = decode.simulate(self.H, plain.copy(expert_deferral=True, deferral_share=1.0), self.params)
        self.assertGreater(b.tok_s, a.tok_s * 1.05)
        self.assertGreater(c.tok_s, a.tok_s * 1.3)
        self.assertFalse(b.lossless)
        split = decode.simulate(self.H, self.cfg.copy(expert_deferral=True), self.params)
        self.assertLess(split.tok_s / decode.simulate(self.H, self.cfg, self.params).tok_s, 1.05)

    def test_verify_tapering_cuts_bytes(self):
        a = decode.simulate(self.H, self.cfg, self.params)
        b = decode.simulate(self.H, self.cfg.copy(spec_tail_topk=4), self.params)
        self.assertLess(b.step.cpu_bytes, a.step.cpu_bytes * 0.9)
        self.assertLess(b.tokens_per_round, a.tokens_per_round)
        self.assertFalse(b.lossless)


class PolishTest(unittest.TestCase):
    def setUp(self):
        self.params = kernels.Params()
        self.H = hw.tr16()

    def test_graded_deferral(self):
        plain = RunConfig(pack="q22", speculation="none")
        base = decode.simulate(self.H, plain, self.params).tok_s
        small = decode.simulate(self.H, plain.copy(expert_deferral=True, deferral_share=0.3), self.params).tok_s
        full = decode.simulate(self.H, plain.copy(expert_deferral=True, deferral_share=1.0), self.params).tok_s
        self.assertGreater(small, base)
        self.assertGreater(full, small)

    def test_adaptive_window_shortens_verify(self):
        cfg = RunConfig(pack="q22", speculation="dflash", draft_block=8, draft_acceptance="dflash_chat")
        a = decode.simulate(self.H, cfg, self.params)
        b = decode.simulate(self.H, cfg.copy(adaptive_window=True), self.params)
        self.assertLess(b.step.width, a.step.width)
        self.assertGreater(b.tok_s, a.tok_s)
        self.assertTrue(b.lossless)

    def test_prediction_hides_disk_wait(self):
        H = hw.tr16(); H.memory.gib = 60
        cfg = RunConfig(pack="q22", mtp_depth=2, placement="ram_tier", ram_mode="exact")
        a = decode.simulate(H, cfg, self.params)
        b = decode.simulate(H, cfg.copy(pcie_prefetch=True), self.params)
        self.assertGreater(b.tok_s, a.tok_s)
        self.assertGreater(b.disk_gb_per_token, 0)

    def test_presets_load_and_run(self):
        for name in hw.PRESETS:
            H = hw.load(name)
            d = decode.simulate(H, RunConfig(pack="reap50_q23", mtp_depth=1), self.params)
            self.assertGreater(d.tok_s, 0)

    def test_xeon_out_of_sample_rows_within_tolerance(self):
        rows = [r for r in calibration.evaluate(calibration.load_records(), self.params) if r["name"].startswith("xeon_v100")]
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(r["within"] for r in rows), [(r["name"], round(r["error"], 2)) for r in rows])


class OwnIdeaTest(unittest.TestCase):
    def test_tail_affinity_between_none_and_full(self):
        H = hw.tr16(); params = kernels.Params()
        cfg = RunConfig(pack="q22", mtp_depth=3, acceptance="prime")
        none = decode.simulate(H, cfg, params).tok_s
        tail = decode.simulate(H, cfg.copy(tail_affinity=0.1), params)
        full = decode.simulate(H, cfg.copy(affinity=0.1), params).tok_s
        self.assertGreater(tail.tok_s, none)
        self.assertLess(tail.tok_s, full)
        self.assertFalse(tail.lossless)


class PrefillKnobTest(unittest.TestCase):
    def setUp(self):
        self.params = kernels.Params()
        self.H = hw.tr16()
        self.cfg = RunConfig(pack="q23", prompt=8192, prefill_chunk=8192, prefetch_groups=12, context=12288, prefill_scratch_mib=2048)

    def test_p_series_ladder_is_monotonic(self):
        steps = [dict(prefetch_groups=0, prefill_mla_f16=False, prefill_kda_parts=False, prefill_legacy=True),
                 dict(prefetch_groups=0, prefill_mla_f16=False, prefill_kda_parts=False),
                 dict(prefetch_groups=12, prefill_mla_f16=False, prefill_kda_parts=False),
                 dict(prefetch_groups=12, prefill_mla_f16=True, prefill_kda_parts=False),
                 dict(prefetch_groups=12, prefill_mla_f16=True, prefill_kda_parts=True)]
        speeds = [prefill.simulate(self.H, self.cfg.copy(**s), self.params).tok_s for s in steps]
        self.assertEqual(speeds, sorted(speeds))
        self.assertAlmostEqual(speeds[-1], 403, delta=25)

    def test_more_prefetch_never_hurts(self):
        a = prefill.simulate(self.H, self.cfg.copy(prefetch_groups=12), self.params).tok_s
        b = prefill.simulate(self.H, self.cfg.copy(prefetch_groups=18), self.params).tok_s
        self.assertGreaterEqual(b, a - 1e-6)

    def test_continuous_streaming_and_gemm_scale(self):
        base = prefill.simulate(self.H, self.cfg, self.params)
        stream = prefill.simulate(self.H, self.cfg.copy(prefill_stream_depth=18), self.params)
        fast = prefill.simulate(self.H, self.cfg.copy(prefill_stream_depth=18, prefill_gemm_scale=2), self.params)
        self.assertGreater(stream.tok_s, base.tok_s)
        self.assertGreater(fast.tok_s, stream.tok_s)
        self.assertEqual(fast.bottleneck, "pcie")   # PCIe floor: 110 GB at 7.2 GB/s
        self.assertAlmostEqual(fast.total_s, 15.3, delta=0.5)

    def test_cpu_experts_win_short_prompts_only(self):
        short = self.cfg.copy(prompt=53)
        gpu = prefill.simulate(self.H, short, self.params).total_s
        cpu = prefill.simulate(self.H, short.copy(prefill_experts="cpu"), self.params).total_s
        self.assertLess(cpu, gpu / 3)
        auto_long = prefill.simulate(self.H, self.cfg.copy(prompt=2048, prefill_experts="auto"), self.params)
        self.assertEqual(auto_long.experts, "gpu")
        auto_short = prefill.simulate(self.H, short.copy(prefill_experts="auto"), self.params)
        self.assertEqual(auto_short.experts, "cpu")

    def test_cpu_assist_helps_a_little(self):
        a = prefill.simulate(self.H, self.cfg.copy(prefill_stream_depth=18, prefill_gemm_scale=2), self.params).tok_s
        b = prefill.simulate(self.H, self.cfg.copy(prefill_stream_depth=18, prefill_gemm_scale=2, prefill_cpu_assist=True), self.params).tok_s
        self.assertGreaterEqual(b, a * 0.98)


class QualityAndReplayTest(unittest.TestCase):
    def test_kl_anchors(self):
        import quality
        self.assertAlmostEqual(quality.estimate_kl(RunConfig(pack="q23"))[0], 0.137, delta=1e-6)
        self.assertAlmostEqual(quality.estimate_kl(RunConfig(pack="q23", affinity=0.10))[0], 0.165, delta=1e-3)
        self.assertAlmostEqual(quality.estimate_kl(RunConfig(pack="q23", affinity=0.15))[0], 0.198, delta=1e-3)
        self.assertAlmostEqual(quality.estimate_kl(RunConfig(pack="q22"))[0], 0.164, delta=1e-6)

    def test_kl_budget_filters_plan(self):
        out = io.StringIO()
        with redirect_stdout(out):
            glm_sim.main(["plan", "--hw", "tr16", "--packs", "q23", "--kl-budget", "0.15", "--top", "30"])
        for line in out.getvalue().splitlines():
            parts = line.split()
            if parts and parts[0] == "q23":
                self.assertLessEqual(float(parts[-2]), 0.15)

    def test_replay_curve_continuity_and_saturation(self):
        c = routing.TierCurve()
        below = c.hit(c.knee - 1e-6, "static_prior")
        above = c.hit(c.knee + 1e-6, "static_prior")
        self.assertAlmostEqual(below, above, delta=0.002)
        self.assertAlmostEqual(c.hit(1.0, "static_prior"), 1.0, delta=1e-6)
        mids = [c.hit(f, "static_prior") for f in (0.1, 0.2, 0.4, 0.8)]
        self.assertEqual(mids, sorted(mids))

    def test_xeon_two_card_hit_near_measured(self):
        H = hw.xeon_v100()
        d = decode.simulate(H, RunConfig(pack="q2_orig", speculation="none", mtp_depth=0, gpus=2, threads=35,
                                         context=8192, split_verify=False, prefetch_groups=0), kernels.Params())
        self.assertAlmostEqual(d.hit_bytes_share, 0.83, delta=0.08)


class RemainingItemsTest(unittest.TestCase):
    def setUp(self):
        self.H = hw.tr16(); self.params = kernels.Params()
        self.cfg = RunConfig(pack="q22", mtp_depth=3, acceptance="prime")

    def test_cost_aware_drafts_lossless_gain(self):
        a = decode.simulate(self.H, self.cfg, self.params)
        b = decode.simulate(self.H, self.cfg.copy(cost_aware_drafts=True), self.params)
        self.assertGreater(b.tok_s, a.tok_s)
        self.assertTrue(b.lossless)

    def test_selfspec_runs_and_is_slower_than_mtp_here(self):
        a = decode.simulate(self.H, self.cfg, self.params).tok_s
        b = decode.simulate(self.H, self.cfg.copy(speculation="selfspec"), self.params)
        self.assertGreater(b.tok_s, 0)
        self.assertLess(b.tok_s, a)

    def test_cold_experts_fit_64gb(self):
        H = hw.tr16(); H.memory.gib = 60
        base = RunConfig(pack="q23", mtp_depth=2, placement="ram_tier", pcie_prefetch=True, draft_prefetch=True)
        a = decode.simulate(H, base, self.params)
        b = decode.simulate(H, base.copy(cold_share=0.7, cold_scale=0.55), self.params)
        self.assertGreater(b.tok_s, 3 * a.tok_s)
        self.assertFalse(b.lossless)

    def test_draft_prefetch_helps_disk_bound(self):
        H = hw.tr16(); H.memory.gib = 60
        base = RunConfig(pack="q23", mtp_depth=2, placement="ram_tier", pcie_prefetch=True)
        self.assertGreater(decode.simulate(H, base.copy(draft_prefetch=True), self.params).tok_s,
                           decode.simulate(H, base, self.params).tok_s)

    def test_estimate_mc_and_request(self):
        out = io.StringIO()
        with redirect_stdout(out):
            glm_sim.main(["estimate", "--hw", "tr16", "--mc", "5"])
        self.assertIn("UNCERTAINTY", out.getvalue())
        self.assertIn("REQUEST", out.getvalue())

    def test_pareto(self):
        out = io.StringIO()
        with redirect_stdout(out):
            glm_sim.main(["plan", "--hw", "tr16", "--packs", "q23", "--pareto", "--top", "20"])
        self.assertIn("pack", out.getvalue())


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
