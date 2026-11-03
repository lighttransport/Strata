"""MiniMax M3.1 Flash in tools/sim: the census-backed model entry and the invariants of the decode projection.

No GPU, no model files: the census JSON in tools/sim/data was produced from the GGUF headers by
tools/minimax_sim_census.py. Run: python3 tools/test_sim_minimax.py
"""
import pathlib
import sys
import unittest

TOOLS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "sim"))
import minimax_m31_sim as proj  # noqa: E402
import minimax_step as mstep  # noqa: E402
import kernels  # noqa: E402
import model  # noqa: E402

KEY, PACK = "minimax_m31", "minimax_m31_ud_q2_k_xl"


class MiniMaxModelTest(unittest.TestCase):
    def tearDown(self):
        model.use_model("flash")

    def test_bytes_match_the_gguf_census(self):
        model.use_model(KEY)
        p = model.PACKS[PACK]
        g = model.GEOMETRY
        self.assertEqual((g.layers, g.dense_layers, g.experts, g.top_k, g.expert_ff, g.hidden), (60, 3, 128, 4, 3072, 6144))
        self.assertEqual(p.routed_total_bytes(), 133_857_017_856)           # sum of the *_exps tensors in the header
        self.assertAlmostEqual(p.routed_bytes_per_token() / 1e9, 4.183, places=3)
        self.assertAlmostEqual(p.mean_expert_bytes() / 1e6, 18.35, places=2)
        # the first MoE layer's expert: IQ2_XS gate/up + IQ3_XXS down = 18.14 MB; layer 59 keeps IQ3_XXS/IQ4_XS = 24.48 MB
        self.assertAlmostEqual(p.expert_bytes(3) / 1e6, 18.14, places=2)
        self.assertAlmostEqual(p.expert_bytes(59) / 1e6, 24.48, places=2)

    def test_dense_weights_match_the_engine_load(self):
        model.use_model(KEY)
        # strata-m3 loaded 9.35 GiB of dense weights onto the GPU on the V100 node
        self.assertAlmostEqual(model.VRAM["dense"] / 1024, 9.35, delta=0.01)
        self.assertAlmostEqual(model.dense_gpu_bytes_per_token() / 1e9, 9.200, places=3)   # no embedding rows on the GPU path
        self.assertEqual(model.layer_fixed_bytes(0), model.FIXED_BYTES["L0"])
        self.assertGreater(model.layer_shared_bytes(3), 0)

    def test_other_models_are_unchanged(self):
        model.use_model("flash")
        self.assertAlmostEqual(model.dense_gpu_bytes_per_token() / 1e9, 6.452, places=3)
        self.assertEqual(model.layer_shared_bytes(5), model.FIXED_BYTES["shared_expert"])
        self.assertNotIn(PACK, model.packs_for("flash"))
        self.assertEqual(model.packs_for(KEY), [PACK])

    def test_iq1_m_pack_is_smaller(self):
        model.use_model("minimax_m31_ud_iq1_m")
        small = model.PACKS["minimax_m31_ud_iq1_m"].routed_bytes_per_token()
        model.use_model(KEY)
        self.assertLess(small, model.PACKS[PACK].routed_bytes_per_token())


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.hw = proj.load_hw()
        self.params = kernels.Params()

    def tearDown(self):
        model.use_model("flash")

    def tok_s(self, hit, hw=None, **kw):
        return proj.run(KEY, PACK, hw or self.hw, self.params, hit=hit, pcie_prefetch=True, **kw)[0].tok_s

    def test_more_hits_never_slower(self):
        values = [self.tok_s(h) for h in (0.1, 0.3, 0.5, 0.7, 0.9)]
        self.assertEqual(values, sorted(values))

    def test_faster_cpu_never_slower(self):
        values = []
        for gbps in (50, 80, 120):
            hw = self.hw.copy(); hw.set("memory.dram_gbps", gbps)
            values.append(self.tok_s(0.4, hw))
        self.assertEqual(values, sorted(values))

    def test_speculation_needs_acceptance(self):
        plain = self.tok_s(0.5)
        spec = lambda p: self.tok_s(0.5, speculation="external", mtp_depth=2, acceptance_probabilities=[p, p],
                                    external_draft_ms=5.0, proposal_probability=1.0)
        self.assertLess(spec(0.05), plain)    # a draft that is never accepted only costs time
        self.assertGreater(spec(0.95), plain)

    def test_measured_anchor_is_derived_from_the_counts(self):
        m = proj.MEASURED
        self.assertAlmostEqual(m["decode_hit"], 1 - (13493 - 5083) / (63 * 57 * 4), places=9)
        self.assertGreater(m["decode_hit"], 0.4)
        self.assertLess(m["decode_hit"], 0.43)


class MeasuredFitTest(unittest.TestCase):
    """tools/sim/minimax_step.py: the model fitted on the measured strata-m3 path (tools/sim/data/minimax_m31_measured.json)."""

    @classmethod
    def setUpClass(cls):
        cls.p, _ = mstep.fit(mstep.Params())

    def test_reproduces_the_six_single_stream_speeds(self):
        for gpus, mode, meas, tok, _ in mstep.check_anchors(self.p):
            self.assertLess(abs(tok / meas - 1), 0.05, (gpus, mode, meas, tok))

    def test_more_cache_and_a_second_gpu_never_hurt(self):
        for mode in ("exact", "fast", "max"):
            one = mstep.decode_tok_s(mode, 1, 1, 4.1, self.p)[0]
            two = mstep.decode_tok_s(mode, 2, 1, 4.1, self.p)[0]
            self.assertGreater(two, one)
        self.assertGreater(mstep.slots_for(1, 8192, 2), mstep.slots_for(1, 8192, 1))
        self.assertGreater(mstep.slots_for(1, 2048, 1), mstep.slots_for(8, 2048, 1))   # more sequences leave fewer slots

    def test_batching_raises_the_aggregate_and_lowers_the_per_sequence_speed(self):
        a1 = mstep.decode_tok_s("max", 1, 1, 4.1, self.p, ctx=2048)[0]
        a4 = mstep.decode_tok_s("max", 1, 4, 4.1, self.p, ctx=2048)[0]
        self.assertGreater(a4, a1)
        self.assertLess(a4 / 4, a1)

    def test_union_law_and_hit_curves(self):
        self.assertAlmostEqual(mstep.rho(4), 0.74, places=2)
        self.assertLessEqual(mstep.cross_ratio(4), mstep.cross_ratio(2))
        self.assertLessEqual(mstep.hit_degradation(8, 1100), mstep.hit_degradation(2, 1100))
        self.assertGreater(mstep.hit_for_slots("exact", 2900), mstep.hit_for_slots("exact", 1100))

    def test_prefill_reproduces_the_measured_points(self):
        pp, _ = mstep.fit_prefill(mstep.PrefillParams())
        for n, _, meas, mod in mstep.prefill_anchor_errors(pp):
            self.assertLess(abs(mod / meas - 1), 0.10, (n, meas, mod))
        # many short prompts in one chunk share one pass over the pack
        self.assertGreater(mstep.prefill_s(210, 512, 1, pp) / 210, 4 * mstep.prefill_s(32 * 210, 7168, 1, pp) / (32 * 210))


if __name__ == "__main__":
    unittest.main()
