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


if __name__ == "__main__":
    unittest.main()
