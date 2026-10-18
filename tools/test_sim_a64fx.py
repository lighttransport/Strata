"""A64FX cluster model: census, memory fit, collectives, load imbalance, calibration rows, scaling classes."""
import contextlib
import dataclasses
import io
import pathlib
import sys
import json
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "sim"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import a64fx  # noqa: E402
import decode  # noqa: E402
import hw as hwmod  # noqa: E402
import kernels  # noqa: E402
import routing  # noqa: E402
from config import RunConfig  # noqa: E402

GIB = 2 ** 30


def layout(nodes=12, **kw):
    base = dict(nodes=nodes, trunk_tp=nodes, expert_layout="ep_parts", parts=4, pack="q23_fp8", trunk_format="q8", context=8192)
    base.update(kw)
    base.setdefault("trunk_tp", base["nodes"] * base.get("ranks_per_node", 1))
    return a64fx.Layout(**base)


class CensusTest(unittest.TestCase):
    def test_q23_fp8_per_node(self):
        pk = a64fx.pack("q23_fp8")
        self.assertAlmostEqual(pk.routed_total_bytes() / 1e9, 110.2, delta=0.1)
        self.assertAlmostEqual(pk.routed_total_bytes() / 12 / GIB, 8.55, delta=0.02)
        self.assertAlmostEqual(pk.routed_total_bytes() / 6 / GIB, 17.1, delta=0.05)

    def test_ud_q4kxl_matches_the_runner_census(self):
        pk = a64fx.pack("ud_q4kxl")
        self.assertAlmostEqual(pk.routed_total_bytes() / 12 / 1e9, 15.33, delta=0.15)       # 15 GB per rank measured
        self.assertAlmostEqual(pk.routed_bytes_per_token() / 12 / 1e6, 426, delta=5)       # 429 MB per rank per token


class MemoryFitTest(unittest.TestCase):
    def test_q4_does_not_fit_six_nodes(self):
        fit = a64fx.memory_fit(layout(6, pack="ud_q4kxl"))
        self.assertFalse(fit["fits"])
        self.assertIn("exceed", fit["notes"][0])

    def test_q23_fits(self):
        six = a64fx.memory_fit(layout(6))
        self.assertTrue(six["fits"])
        self.assertGreater(six["margin_gib"], 5)
        self.assertTrue(a64fx.memory_fit(layout(12))["fits"])
        self.assertTrue(a64fx.memory_fit(layout(6, ranks_per_node=4))["fits"])


class CollectiveTest(unittest.TestCase):
    def setUp(self):
        self.tofu = a64fx.TofuD()

    def test_ar_steps(self):
        self.assertEqual([a64fx.ar_steps(n) for n in (12, 6, 8, 48)], [5, 4, 3, 7])

    def test_tree_allreduce_matches_measurement(self):
        fp32 = a64fx.tree_allreduce_us(self.tofu, 16384, 12)
        bf16 = a64fx.tree_allreduce_us(self.tofu, 16384, 12, bf16=True)
        self.assertAlmostEqual(fp32, 23.47, delta=2.4)
        self.assertAlmostEqual(bf16, 13.61, delta=2.0)
        self.assertLess(a64fx.tree_allreduce_us(self.tofu, 16384, 6), fp32)

    def test_dispatch_combine_b1(self):
        # measured 2026-10-18 on GLM geometry (E=288, K=8, 8 KiB bf16 per token)
        self.assertAlmostEqual(a64fx.dispatch_combine_us(self.tofu, 8192, 1, 12), 25.92, delta=0.01)
        self.assertAlmostEqual(a64fx.dispatch_combine_us(self.tofu, 8192, 1, 8), 21.14, delta=0.01)
        self.assertAlmostEqual(a64fx.dispatch_combine_us(self.tofu, 8192, 1, 6), 18.31, delta=0.01)
        self.assertAlmostEqual(a64fx.dispatch_combine_us(self.tofu, 8192, 3, 12), 61.04, delta=0.01)
        # the fallback model still reproduces gemm's original 12-rank anchor (34.6 us, E=256, hidden 6144)
        bare = a64fx.TofuD(tree_measured_us={}, dc_measured_us={})
        self.assertAlmostEqual(a64fx.dispatch_combine_us(bare, 8192, 1, 12), 34.6, delta=34.6 * 0.15)

    def test_tree_measured_points(self):
        for ranks, fp32, bf16 in ((6, 18.48, 11.70), (8, 12.50, 8.32), (12, 22.61, 14.27)):
            self.assertAlmostEqual(a64fx.tree_allreduce_us(self.tofu, 16384, ranks), fp32, delta=0.01)
            self.assertAlmostEqual(a64fx.tree_allreduce_us(self.tofu, 16384, ranks, bf16=True), bf16, delta=0.01)
        self.assertAlmostEqual(a64fx.tree_allreduce_us(self.tofu, 3 * 16384, 12), 55.34, delta=0.01)
        # the step model agrees with the measured 6- and 12-rank points within 5 %
        bare = a64fx.TofuD(tree_measured_us={}, dc_measured_us={})
        for ranks, meas in ((6, 18.48), (12, 22.61)):
            self.assertAlmostEqual(a64fx.tree_allreduce_us(bare, 16384, ranks), meas, delta=meas * 0.05)

    def test_reduce_scatter_allgather(self):
        self.assertAlmostEqual(a64fx.reduce_scatter_allgather_ms(self.tofu, 512, 12), 5.7, delta=0.01)
        self.assertAlmostEqual(a64fx.reduce_scatter_allgather_ms(self.tofu, 4096, 12), 45.6, delta=0.1)
        six = a64fx.reduce_scatter_allgather_ms(self.tofu, 512, 6)
        self.assertAlmostEqual(six, 5.7, delta=0.57)


class LoadImbalanceTest(unittest.TestCase):
    def test_matches_the_shard_simulator(self):
        # glm53f_expert_shard_sim.py, seed 53: mean busiest-rank units at 12 ranks for parts 1 / 2 / 4
        for parts, expected in ((1, 2.23), (2, 2.89), (4, 4.06)):
            mean_max, _ = a64fx.ep_max_load(12, 8, parts, trials=40000, seed=53)
            self.assertAlmostEqual(mean_max, expected, delta=0.03, msg=f"parts {parts}")
        ratios = [a64fx.ep_max_load(6, 8, p, trials=20000)[1] for p in (1, 2, 3)]
        self.assertAlmostEqual(ratios[0], 2.17, delta=0.03)
        self.assertEqual(ratios, sorted(ratios, reverse=True))


class ValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {r["name"]: r for r in a64fx.validate()}

    def rows_of(self, kind, needle):
        return [r for r in self.rows.values() if r["kind"] == kind and needle in r["name"]]

    def test_plain_decode_rows(self):
        rows = self.rows_of("decode", "plain decode")
        self.assertEqual(len(rows), 4)
        for r in rows:
            self.assertLessEqual(abs(r["error"]), 0.10, r["name"])

    def test_prefill_rows(self):
        rows = self.rows_of("prefill", "")
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertLessEqual(abs(r["error"]), 0.15, r["name"])

    def test_mtp_rows(self):
        rows = self.rows_of("decode", "MTP")
        self.assertEqual(len(rows), 6)
        for r in rows:
            self.assertLessEqual(abs(r["error"]), 0.10, r["name"])

    def test_stage_split(self):
        r = [r for r in self.rows.values() if r["measured_stages"]][0]
        for stage, measured in r["measured_stages"].items():
            self.assertAlmostEqual(r["stages"][stage], measured, delta=0.2 * measured, msg=stage)


class ScalingTest(unittest.TestCase):
    def setUp(self):
        self.node, self.tofu = a64fx.A64FXNode(), a64fx.TofuD()
        self.current, self.redesign = a64fx.current(), a64fx.redesign()

    def dec(self, L, profile=None):
        return a64fx.decode(L, self.node, self.tofu, profile or self.current)

    def test_classes_under_current(self):
        six, twelve = self.dec(layout(6)), self.dec(layout(12))
        self.assertAlmostEqual(six.mhc_ms, twelve.mhc_ms)                       # C: replicated, no MPI
        self.assertLess(six.comm_ms, twelve.comm_ms)                             # L: 4 steps instead of 5
        self.assertGreater(six.trunk_ms, twelve.trunk_ms)                       # S: twice the bytes per rank
        self.assertLess(six.trunk_ms, 2 * twelve.trunk_ms)                      # ... minus the fixed parts
        self.assertAlmostEqual(six.ctx_ms, twelve.ctx_ms)                       # X: context only

    def test_six_nodes_slower_than_twelve(self):
        for profile in (self.current, self.redesign):
            for el, parts in (("ep_parts", 4), ("tp_parts", 0), ("ep", 1)):
                for rpn in (1, 4):
                    L6, L12 = layout(6, expert_layout=el, parts=parts, ranks_per_node=rpn), layout(12, expert_layout=el, parts=parts, ranks_per_node=rpn)
                    self.assertLess(self.dec(L6, profile).tok_s, self.dec(L12, profile).tok_s, (profile.name, el, rpn))
                    self.assertLess(a64fx.prefill(L6, self.node, self.tofu, profile).tok_s,
                                    a64fx.prefill(L12, self.node, self.tofu, profile).tok_s, (profile.name, el, rpn))

    def test_redesign_halves_the_step(self):
        cur, new = self.dec(layout(12)), self.dec(layout(12), self.redesign)
        self.assertLess(new.step_ms, 0.5 * cur.step_ms)
        # slicing experts finer is not the lever: every-rank slices and 4-rank slices are within 5 %
        self.assertAlmostEqual(self.dec(layout(12, expert_layout="tp_parts"), self.redesign).step_ms, new.step_ms, delta=0.05 * new.step_ms)
        # whole-expert EP at 12 ranks pays the busiest-rank tail and loses
        self.assertGreater(self.dec(layout(12, expert_layout="ep"), self.redesign).step_ms, new.step_ms)

    def test_cmg_ranks_pay_collectives(self):
        one, four = self.dec(layout(12)), self.dec(layout(12, ranks_per_node=4))
        self.assertGreater(four.comm_ms, one.comm_ms)
        self.assertLess(four.tok_s, one.tok_s)

    def test_replicated_trunk_flagged(self):
        d = self.dec(layout(12, trunk_tp=1))
        self.assertTrue(any("bandwidth" in n for n in d.notes))
        self.assertGreater(d.trunk_ms, 12.5)

    def test_mtp_uses_both_fixtures(self):
        best = self.dec(layout(12, mtp_depth=2, acceptance="a64fx_best512"))
        sweep = self.dec(layout(12, mtp_depth=2, acceptance="a64fx_sweep768"))
        self.assertGreater(best.tok_s, sweep.tok_s)
        self.assertAlmostEqual(best.tokens_per_round, 2.93, delta=0.01)


class UntouchedTest(unittest.TestCase):
    def test_existing_simulators_unchanged(self):
        hw = hwmod.PRESETS["tr16+b550-ib"]()
        cfg = RunConfig(pack="reap50_q23", speculation="none", mtp_depth=0, decode_cache_mib=3584, tier_policy="static", remote_share=0.25)
        self.assertAlmostEqual(decode.simulate(hw, cfg, kernels.Params()).tok_s, 16.05, delta=0.02)
        self.assertAlmostEqual(routing.tokens_per_round(2, "prime"), 1 + 0.97 + 0.97 * 0.95, places=6)
        self.assertEqual(a64fx.A64FXNode().set("stream_gbps", "850").stream_gbps, 850.0)

    def test_cli_validate_runs(self):
        import glm_a64fx_sim
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = glm_a64fx_sim.main(["validate"])
        self.assertEqual(code, 0)
        self.assertIn("13 of 13 rows within tolerance", out.getvalue())

class MeasuredKernelTest(unittest.TestCase):
    def test_redesign_q2_rate_is_the_measured_sdot_mix(self):
        d = json.loads((a64fx.DATA / "measured_a64fx_kernels.json").read_text())["sdot_kernel_v3"]["chosen_llvm_best_per_width"]
        q2, q3 = d["q2_K_2048x4096"][0], d["q3_K_4096x2048"][0]
        mix = (2 * 84 + 110) / (2 * 84 / q2 + 110 / q3)        # gate + up Q2_K, down Q3_K, bytes per 256 weights
        self.assertAlmostEqual(a64fx.redesign().expert_q2k_gbps, mix, delta=0.2)

    def test_sdot_beats_ggml_for_every_format(self):
        d = json.loads((a64fx.DATA / "measured_a64fx_kernels.json").read_text())["sdot_kernel_v2"]["node48_gbps_tokens_1_to_4"]
        best = {"q2_K_2048x4096": "v2", "q3_K_4096x2048": "v2", "iq2_xxs_2048x4096": "gather", "iq1_s_2048x4096": "iq"}
        for shape, k in best.items():
            self.assertGreater(d[shape][k][0], 1.9 * d[shape]["ggml"][0], shape)

    def test_width_cost_grows_with_tokens(self):
        prof, pk = a64fx.redesign(), a64fx.pack("q23_fp8")
        costs = [a64fx._width_cost(prof, pk, w) for w in (1, 2, 3, 4)]
        self.assertEqual(costs[0], 1.0)
        self.assertTrue(all(b > a for a, b in zip(costs, costs[1:])))
        self.assertEqual(a64fx._width_cost(prof, a64fx.pack("ud_q4kxl"), 3), 1.0)   # Q4 path: bandwidth-bound

    def test_node_constants_measured(self):
        node = a64fx.A64FXNode()
        self.assertEqual(node.stream_gbps, 915.0)
        self.assertAlmostEqual(node.stream_gbps * node.chain_efficiency, 504, delta=2)   # absolute chain rate unchanged
        self.assertEqual(node.hw_barrier_us, 1.38)


if __name__ == "__main__":
    unittest.main()
