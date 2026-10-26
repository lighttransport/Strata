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


class ReleaseTest(unittest.TestCase):
    def test_1017_config_facts(self):
        g = ds41f.GEOMETRY
        self.assertEqual(g.mtp_layers, 3)
        self.assertEqual(len(g.compress_ratios), 43)
        self.assertEqual(g.compress_ratios[:2] + g.compress_ratios[20:22] + g.compress_ratios[-3:], (0, 0, 1, 1, 0, 0, 0))
        self.assertEqual(g.candidate_blocks * g.candidate_block, 16384)
        self.assertIn("1017", g.release)

    def test_xeon_tesla_engram_slower_than_tr16(self):
        slow = ds41f.decode(ds41f.Plan(machine="xeon_tesla", pack="iq2xxs_q2k", trunk="q8", head="q8"))
        fast = ds41f.decode(ds41f.Plan(machine="tr16", pack="iq1s"))
        self.assertGreater(slow.engram_ms, fast.engram_ms)
        self.assertGreater(slow.engram_exposed_ms, 0.0)


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
        self.assertEqual(len(rows), 22)
        for r in rows:
            self.assertTrue(r["within"], r["name"])
        fitted = [r for r in rows if r["fit"]]
        self.assertEqual(len(fitted), 9)
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
        self.assertIn("22 of 22 rows within tolerance", out.getvalue())

class RunnerTest(unittest.TestCase):
    def test_target_beats_now(self):
        now = ds41f.runner_decode(ds41f.RunnerPlan())
        target = ds41f.runner_decode(ds41f.RunnerPlan(profile="target"))
        self.assertGreater(target["tok_s"], now["tok_s"])
        self.assertGreater(target["hit"], now["hit"])

    def test_sweep_and_enc_skip(self):
        one = ds41f.runner_prefill(ds41f.RunnerPlan(profile="target", mode="stream", chunk=8192), 8192)
        four = ds41f.runner_prefill(ds41f.RunnerPlan(profile="target", mode="stream", chunk=2048), 8192)
        skip = ds41f.runner_prefill(ds41f.RunnerPlan(profile="target", mode="stream", chunk=8192, enc_skip=2560), 8192)
        self.assertGreater(one["tok_s"], four["tok_s"])      # one pass moves each expert once
        self.assertGreaterEqual(skip["tok_s"], one["tok_s"])
        warm = ds41f.runner_prefill(ds41f.RunnerPlan(profile="target", mode="stream", chunk=2048), 2048)
        cold = ds41f.runner_prefill(ds41f.RunnerPlan(profile="target", mode="stream", chunk=2048, uncached_gb=15), 2048)
        self.assertLess(cold["tok_s"], warm["tok_s"])   # short prompts are upload-bound: SSD reads show


class V100Test(unittest.TestCase):
    def test_calibration_points(self):
        rows = [r for r in ds41f.validate() if r["kind"] == "v100_decode"]
        self.assertEqual(len(rows), 12)
        for r in rows:
            self.assertTrue(r["within"], r["name"])
            if r["fit"]:
                self.assertLess(abs(r["error"]), 0.10, r["name"])

    def test_levers(self):
        base = ds41f.v100_decode(ds41f.V100Plan())
        e3 = ds41f.v100_decode(ds41f.V100Plan(pack="IQ2_XXSE3", engram_gib=79))
        self.assertGreater(e3["tok_s"], base["tok_s"])                     # Engram reads are the IQ2_XXSE4 gap
        self.assertAlmostEqual(e3["engram_ms"], 0.0)
        lane = ds41f.v100_decode(ds41f.V100Plan(set=dict(cpu_expert_ms=0.03)))
        self.assertGreater(lane["tok_s"], base["tok_s"])
        more = ds41f.v100_decode(ds41f.V100Plan(extra_gpu_gib=6.3))
        self.assertGreater(more["hit"], base["hit"])
        plain = ds41f.v100_decode(ds41f.V100Plan(spec=False, drafter_on_gpu=False))
        self.assertGreater(plain["hit"], base["hit"])                     # the drafter's VRAM costs resident experts

    def test_gsq_pack(self):
        pk = ds41f.V100_PACKS["GSQ-IQ2_XXS"]
        self.assertEqual(len(pk.layers), 40)
        self.assertAlmostEqual(pk.expert_gib, 123.0, delta=0.2)            # 132.07 GB of IQ1_M / IQ2_XXS / IQ2_XS experts
        self.assertAlmostEqual(pk.expert_mb, 8.60, delta=0.02)
        self.assertGreater(pk.lane_mb_eff({"IQ1_M": 0.6}), pk.lane_mb_eff())
        e4 = ds41f.V100_PACKS["IQ2_XXSE4"]
        self.assertAlmostEqual(e4.lane_mb_eff(), 9.95)                      # the fit's reference expert: no scaling
        base = ds41f.v100_decode(ds41f.V100Plan())
        gsq = ds41f.v100_decode(ds41f.V100Plan(pack="GSQ-IQ2_XXS"))
        self.assertGreater(gsq["hit"], base["hit"])                        # fewer expert bytes: more of them resident
        self.assertGreater(gsq["tok_s"], base["tok_s"])
        slow = ds41f.v100_decode(ds41f.V100Plan(pack="GSQ-IQ2_XXS", lane_eff={"IQ1_M": 0.3}))
        self.assertLess(slow["tok_s"], gsq["tok_s"])
        p4 = ds41f.v100_prefill(ds41f.V100Plan(), 4096)["tok_s"]
        p8 = ds41f.v100_prefill(ds41f.V100Plan(), 8192)["tok_s"]
        self.assertAlmostEqual(p4, 389, delta=20)                           # the two measured IQ2_XXSE4 points
        self.assertAlmostEqual(p8, 412, delta=25)
        self.assertEqual(len(ds41f.v100_packs_compare()), 5)

    def test_research_levers(self):
        # the phase model against the 8j measurements (Engram RAM 56 GiB), within 10 %
        for kw, meas in ((dict(kernel="mmq"), 34.9), (dict(), 36.0), (dict(kernel="mmq", workload="wiki"), 29.9),
                         (dict(workload="wiki"), 32.4), (dict(kernel="mmq", pack="IQ2_XXSE4"), 24.2)):
            r = ds41f.lever_decode(ds41f.LeverPlan(**kw))
            self.assertLess(abs(r["tok_s"] / meas - 1), 0.10, kw)
        base = ds41f.lever_decode(ds41f.LeverPlan())
        self.assertGreater(ds41f.lever_decode(ds41f.LeverPlan(defer_rank=3))["tok_s"], base["tok_s"])
        self.assertGreaterEqual(ds41f.lever_decode(ds41f.LeverPlan(sched=True))["tok_s"], base["tok_s"] - 1e-9)
        self.assertLess(ds41f.lever_decode(ds41f.LeverPlan(decode_stream=2))["tok_s"], base["tok_s"])   # the link is too slow
        self.assertAlmostEqual(ds41f.lever_prefill(4096, batch=4096)["tok_s"], 389, delta=25)
        self.assertAlmostEqual(ds41f.lever_prefill(106000, batch=4096)["tok_s"], 377, delta=30)
        rows, info = ds41f.tr16_levers()
        self.assertAlmostEqual(rows[0][1], 12.5, delta=1.0)          # the measured TR16 decode
        self.assertLess(info["vram_for_experts_and_drafter_gib"], info["drafter_gib"])

    def test_sessions_and_context(self):
        one = ds41f.v100_decode(ds41f.V100Plan(spec=False))
        four = ds41f.v100_decode(ds41f.V100Plan(spec=False, sessions=4))
        self.assertGreater(four["tok_s"], one["tok_s"])                   # batched sessions share the weights' reads
        self.assertLess(four["per_session_tok_s"], one["per_session_tok_s"])
        short = ds41f.v100_decode(ds41f.V100Plan())
        long32 = ds41f.v100_decode(ds41f.V100Plan(context=1048576, kv="fp32"))
        long8 = ds41f.v100_decode(ds41f.V100Plan(context=1048576, kv="q8", index_tc=True))
        self.assertLess(long32["tok_s"], short["tok_s"])
        self.assertGreater(long8["tok_s"], long32["tok_s"])
        self.assertAlmostEqual(long32["kv_gib"], 6400 * 1048576 / GIB)


class A64FXSdotTest(unittest.TestCase):
    def test_width_cost_interpolates_the_measured_points(self):
        self.assertEqual(ds41f.a64fx_sdot_cost("IQ2_XXS", 1), 1.0)
        c = ds41f.A64FX_SDOT_WIDTH_COST["Q2_K"]
        self.assertAlmostEqual(ds41f.a64fx_sdot_cost("Q2_K", 2), c[1])
        self.assertAlmostEqual(ds41f.a64fx_sdot_cost("Q2_K", 2.5), (c[1] + c[2]) / 2)
        self.assertGreater(ds41f.a64fx_sdot_cost("Q2_K", 5), c[3])
        self.assertTrue(all(b > a for a, b in zip(c, c[1:])))

    def test_iq_packs_use_the_measured_kernels(self):
        # IQ2_XXS/IQ1_S stream 3x slower per byte than Q2_K on the SDOT kernels: their expert time is no longer the
        # 250 GB/s hypothesis and exceeds q23's at the same node count
        iq = ds41f.a64fx_decode(ds41f.A64FXPlan(nodes=8, pack="iq2xxs_q2k", profile="redesign"))
        q23 = ds41f.a64fx_decode(ds41f.A64FXPlan(nodes=8, pack="q23", profile="redesign"))
        self.assertGreater(iq["experts_ms"], q23["experts_ms"])


if __name__ == "__main__":
    unittest.main()
