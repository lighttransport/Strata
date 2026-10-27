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
        self.assertEqual(len(rows), 6)
        for r in rows:
            self.assertLessEqual(abs(r["error"]), 0.10, r["name"])

    def test_mhc_ab_ratio(self):
        # the in-model A/B (52449126): local-gram over legacy measured 1.082; the model must predict the gain itself
        legacy, = self.rows_of("decode", "legacy mHC (in-model A/B")
        gram, = self.rows_of("decode", "local-gram mHC (in-model A/B")
        measured = gram["measured"] / legacy["measured"]
        predicted = gram["predicted"] / legacy["predicted"]
        self.assertAlmostEqual(measured, 1.082, places=3)
        self.assertLess(abs(predicted / measured - 1), 0.01)

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
        self.assertIn("15 of 15 rows within tolerance", out.getvalue())

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

    def test_redesign_mhc_is_the_inmodel_measurement(self):
        d = json.loads((a64fx.DATA / "measured_a64fx_mhc.json").read_text())["per_site_us_mean_over_ranks"]
        self.assertEqual(a64fx.redesign().mhc_site_us, (d["local_gram"]["end_site"],) * 2)
        self.assertLess(d["local_gram"]["end_site"], d["legacy"]["end_site"])
        # today's calibrated profile stays within 5 % of the uninstrumented runner's legacy sites
        for got, meas in zip(a64fx.current().mhc_site_us, (d["legacy"]["ffn_site_incl_router"], d["legacy"]["end_site"])):
            self.assertAlmostEqual(got, meas, delta=0.05 * meas)

    def test_node_constants_measured(self):
        node = a64fx.A64FXNode()
        self.assertEqual(node.stream_gbps, 915.0)
        self.assertAlmostEqual(node.stream_gbps * node.chain_efficiency, 504, delta=2)   # absolute chain rate unchanged
        self.assertEqual(node.hw_barrier_us, 1.38)


def fp4(nodes=12, **kw):
    return layout(nodes, pack="fp4_local", trunk_format="fp4l", **kw)


class Fp4LocalTest(unittest.TestCase):
    """FP4-Local: census bytes, memory fit, kernel-rate sensitivity, numerics, measured-table hook."""

    def test_census_matches_the_v100_census(self):
        ref = json.loads((a64fx.DATA / "glm_fp4_v100.json").read_text())["census"]
        c = a64fx._FP4L
        self.assertEqual(a64fx.pack("fp4_local").routed_total_bytes(), ref["main_experts"]["gpu_bytes"])   # 171,228,266,496
        self.assertEqual(c["main_fixed_matrix_bytes"], ref["main_fixed"]["checkpoint_bytes"])
        self.assertEqual(c["head"]["bytes"], ref["head"]["checkpoint_bytes"])
        self.assertAlmostEqual(a64fx.pack("fp4_local").routed_bytes_per_token() / 1e9, 4.756, delta=0.001)
        per_layer = {k: sum(m["bytes"] for m in v["sublayers"].values()) / 1e6 for k, v in c["kinds"].items()}
        self.assertAlmostEqual(per_layer["dense"], 302.5, delta=0.1)
        self.assertAlmostEqual(per_layer["kda"], 174.3, delta=0.1)
        self.assertAlmostEqual(per_layer["mla"], 177.2, delta=0.1)

    def test_fits_eight_and_twelve_nodes_not_six(self):
        self.assertTrue(a64fx.memory_fit(fp4(12, context=32768))["fits"])
        self.assertTrue(a64fx.memory_fit(fp4(8, context=32768))["fits"])
        self.assertFalse(a64fx.memory_fit(fp4(6, context=32768))["fits"])

    def test_head_split_share(self):
        m = a64fx._FP4L["kinds"]["kda"]["sublayers"]["q_proj"]
        self.assertEqual(a64fx.fp4l_share(m, 12), 6 / 64)
        self.assertEqual(a64fx.fp4l_share(m, 8), 8 / 64)
        self.assertEqual(a64fx.fp4l_share(a64fx._FP4L["kinds"]["kda"]["sublayers"]["router"], 12), 1.0)

    def test_decode_monotone_in_kernel_rate_and_exact_slower(self):
        rates = (120, 160, 250, 330, 504)
        tok = [a64fx.decode(fp4(), profile=dataclasses.replace(a64fx.redesign(), expert_nvfp4_w4a8_gbps=float(r))).tok_s for r in rates]
        self.assertTrue(all(b > a for a, b in zip(tok, tok[1:])))
        fast = a64fx.decode(fp4(), profile=a64fx.redesign()).tok_s
        exact = a64fx.decode(fp4(numerics="exact"), profile=a64fx.redesign()).tok_s
        self.assertLess(exact, fast)
        self.assertGreater(fast, 62)
        self.assertLess(fast, 85)

    def test_gguf_paths_ignore_numerics(self):
        prof = a64fx.redesign()
        self.assertEqual(a64fx.decode(layout(12), profile=prof).tok_s, a64fx.decode(layout(12, numerics="exact"), profile=prof).tok_s)

    def redesign_with(self, kern):
        saved = a64fx._KERN
        try:
            a64fx._KERN = kern
            return a64fx.redesign()
        finally:
            a64fx._KERN = saved

    def test_measured_table_replaces_hypothesis(self):
        bare = {k: v for k, v in a64fx._KERN.items() if k not in ("nvfp4_kernel_v1", "trunk_kernel_v1", "nvfp4_gemm_v1")}
        kern = dict(bare, nvfp4_kernel_v1=dict(chosen_llvm_best_per_width=dict(
            w4a8=dict(gate_up=[300.0, 200.0, 150.0, 120.0], down=[250.0, 170.0, 130.0, 100.0]))))
        t = a64fx.fp4_tables(kern)
        self.assertAlmostEqual(t["w4a8"][0], 3 / (2 / 300 + 1 / 250), places=6)
        self.assertEqual(t["w4a8"][1][0], 1.0)
        prof = self.redesign_with(kern)
        self.assertAlmostEqual(prof.expert_nvfp4_w4a8_gbps, round(t["w4a8"][0], 1))
        self.assertFalse(any(h.startswith("NVFP4 W4A8") for h in prof.fp4l_hypotheses))
        self.assertTrue(any(h.startswith("NVFP4 W4A8") for h in self.redesign_with(bare).fp4l_hypotheses))

    def test_redesign_uses_the_measured_fp4_kernels(self):
        d = json.loads((a64fx.DATA / "measured_a64fx_kernels.json").read_text())
        w = d["nvfp4_kernel_v1"]["chosen_llvm_best_per_width"]["w4a8"]
        mix = 3 / (2 / w["gate_up"][0] + 1 / w["down"][0])
        prof = a64fx.redesign()
        self.assertAlmostEqual(prof.expert_nvfp4_w4a8_gbps, mix, delta=0.1)
        self.assertGreater(prof.expert_nvfp4_w4a8_gbps, 168)          # above gemm's fused MXFP4 SDOT
        self.assertEqual(prof.trunk_q8_gbps, d["trunk_kernel_v1"]["chosen_llvm_node48_gbps"]["q8_0"])

    def test_prefill_fast_near_q23_and_exact_slowest(self):
        prof = a64fx.redesign()
        q23 = a64fx.prefill(layout(12), profile=prof).tok_s
        fast = a64fx.prefill(fp4(), profile=prof).tok_s
        exact = a64fx.prefill(fp4(numerics="exact"), profile=prof).tok_s
        self.assertAlmostEqual(fast / q23, 1.0, delta=0.1)      # W4A8 GEMM measured at the int8 SDOT GEMM rate
        self.assertLess(exact, 0.6 * fast)

    def test_cli_report_and_sweep(self):
        import glm_a64fx_sim
        for argv, want in ((["report", "--pack", "fp4_local"], "W4A8 GB/s"),
                           (["sweep", "--pack", "fp4_local", "--profile", "redesign", "--grid", "nodes=8,12"], "prefill")):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(glm_a64fx_sim.main(argv), 0)
            self.assertIn(want, out.getvalue())


if __name__ == "__main__":
    unittest.main()
