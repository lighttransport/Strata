"""Muse Spark Open two-node model: census numbers, placement and RAM checks, decode/prefill accounting, the CLI."""
import contextlib
import dataclasses
import io
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "sim"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import model  # noqa: E402
import muse  # noqa: E402
import muse_sim  # noqa: E402

IB = muse.LINKS["ib-qdr"]


def nodes(name="split-gpus"):
    return list(muse.scenarios()[name])


class CensusTest(unittest.TestCase):
    def test_architecture_from_headers(self):
        a = muse.arch()
        self.assertEqual((a.layers, a.hidden, a.latent, a.experts, a.top_k), (62, 8192, 4096, 256, 8))
        self.assertEqual((a.q_heads, a.kv_heads, a.head_dim, a.window), (128, 16, 64, 2048))
        self.assertEqual(sorted(a.full_layers), list(range(1, 62, 4)))
        self.assertEqual(a.total_bytes, 175_626_404_864)
        # IQ1_S: 3 x 4096 x 4096 weights at 50 bytes per 256
        self.assertEqual(a.expert_bytes, 3 * 4096 * 4096 // 256 * 50)
        self.assertEqual(a.experts_total_bytes + a.layer_fixed_bytes * 62 + a.lm_head_bytes + a.token_embd_bytes
                         + 8192 * 4, a.total_bytes)          # + output_norm (f32)

    def test_model_registry(self):
        pack = model.MUSE_PACKS["muse_ud_iq1s"]
        self.assertEqual(pack.expert_bytes(0), muse.arch().expert_bytes)
        self.assertEqual(pack.routed_bytes_per_token(), 8 * 62 * muse.arch().expert_bytes)
        self.assertIn("muse", model.MODELS)
        # GLM-5.3-Flash sizes are unchanged by the expert_hidden field
        self.assertEqual(model.PACKS["q23"].expert_bytes(5), 9_109_504)
        self.assertEqual(model.GEOMETRY.expert_macs(), 3 * 4096 * 2048)

    def test_q4_trunk_scale(self):
        s = muse.fixed_scale("q4")
        self.assertTrue(0.55 < s < 0.65, s)


class PlacementTest(unittest.TestCase):
    def test_model_needs_both_nodes(self):
        a, b = nodes()
        place = muse.placement(muse.Plan(split="row", b_share=0.0), [a, b])
        self.assertFalse(place["feasible"])
        self.assertIn("tr16", place["notes"][0])

    def test_b550_ram_cap(self):
        a, b = nodes()
        self.assertTrue(muse.placement(muse.Plan(b_share=0.375), [a, b])["feasible"])
        self.assertFalse(muse.placement(muse.Plan(b_share=0.40), [a, b])["feasible"])

    def test_layer_split_without_gpu_on_b_is_infeasible(self):
        place = muse.placement(muse.Plan(split="layer"), nodes("gpus-on-tr16"))
        self.assertFalse(place["feasible"])

    def test_vram_split_and_caches(self):
        place = muse.placement(muse.Plan(gpu_layers=(32, 30)), nodes())
        g0, g1 = place["gpus"]
        self.assertEqual((len(g0["layers"]), len(g1["layers"])), (32, 30))
        self.assertGreater(g0["slots"], 0)
        self.assertGreater(g1["slots"], 0)
        self.assertGreater(g1["fixed_gib"], g0["fixed_gib"])      # the tail GPU holds the lm_head
        with self.assertRaises(ValueError):
            muse.placement(muse.Plan(gpu_layers=(32, 31)), nodes())


class DecodeTest(unittest.TestCase):
    def test_step_is_the_sum_of_its_parts(self):
        r = muse.decode(muse.Plan(gpu_layers=(32, 30)), nodes(), IB)
        parts = r["gpu_ms"] + r["cpu_ms"] + r["mailbox_ms"] + r["hops_ms"] + r["head_tail_ms"] + r["host_ms"]
        self.assertAlmostEqual(r["step_ms"], parts, places=6)
        self.assertLessEqual(r["remote_wait_ms"], r["cpu_ms"])

    def test_kernels_order(self):
        speeds = {k: muse.decode(muse.Plan(cpu_kernel=k, gpu_layers=(32, 30)), nodes(), IB)["tok_s"]
                  for k in ("ggml", "iq256", "dram")}
        self.assertLess(speeds["ggml"], 1.0)
        self.assertLess(speeds["ggml"], speeds["iq256"])
        self.assertLess(speeds["iq256"], speeds["dram"])

    def test_cpu_lane_hand_check(self):
        """One layer of the row split, by hand: TR16 lane = cold experts x (1 - share) x MACs / GMAC/s."""
        a, b = nodes()
        plan = muse.Plan(gpu_layers=(32, 30), hit_scale=1e-9, gpu_experts=False)
        r = muse.decode(plan, [a, b], IB)
        cold_macs = 8 * muse.arch().expert_macs
        t_a = cold_macs * (1 - plan.b_share) / (a.cpu_gmacs["iq256"][1] * 1e9) * 1e3   # one token per expert
        t_b = cold_macs * plan.b_share / (b.cpu_gmacs["iq256"][1] * 1e9) * 1e3
        trip = muse.net_ms(IB, 4096 * 4) + muse.net_ms(IB, 8 * 4096 * 4)   # f32 latent out, 8 f32 partials back
        per_layer_a = max(t_a, t_b + trip + b.request_ms)   # 9070 XT layers: the B550 lane is remote
        per_layer_b = max(t_b, t_a + trip + a.request_ms)   # 5060 Ti layers: the TR16 lane is remote
        self.assertAlmostEqual(r["cpu_ms"], 32 * per_layer_a + 30 * per_layer_b, places=6)

    def test_lane_rate_by_width(self):
        a, _ = nodes()
        self.assertAlmostEqual(muse.gmacs(a, "iq256", 1.0), 92.2)
        self.assertAlmostEqual(muse.gmacs(a, "iq256", 1.5), (92.2 + 156.9) / 2)
        self.assertAlmostEqual(muse.gmacs(a, "iq256", 8.0), 209.2)
        muse.set_lane_rate(a, "iq256", 46.1)
        self.assertAlmostEqual(muse.gmacs(a, "iq256", 4.0), 104.6)

    def test_link_and_contention(self):
        base = muse.decode(muse.Plan(gpu_layers=(32, 30)), nodes(), IB)["tok_s"]
        gbe = muse.decode(muse.Plan(gpu_layers=(32, 30)), nodes(), muse.LINKS["gbe"])["tok_s"]
        a, b = nodes()
        muse.set_lane_rate(b, "iq256", 20.0)
        busy = muse.decode(muse.Plan(gpu_layers=(32, 30)), [a, b], IB)["tok_s"]
        self.assertLess(gbe, base)
        self.assertLess(busy, base)

    def test_expert_split_probabilities(self):
        total = sum(p for _, p in muse.multinomial(8, (0.3, 0.5, 0.2)))
        self.assertAlmostEqual(total, 1.0, places=12)
        r = muse.decode(muse.Plan(split="expert", b_share=0.3, gpu_layers=(32, 30)), nodes(), IB)
        self.assertGreater(r["tok_s"], 0)

    def test_search_prefers_row_split(self):
        best = muse.search(nodes(), IB, muse.Plan(), what="decode")
        self.assertEqual(max(best, key=lambda s: best[s][0]), "row")


class PrefillTest(unittest.TestCase):
    def test_chunks_and_layers(self):
        plan = muse.Plan(gpu_layers=(46, 16))
        small = muse.prefill(plan, nodes(), IB, tokens=8192, chunk=1024)
        big = muse.prefill(plan, nodes(), IB, tokens=8192, chunk=8192)
        self.assertEqual(small["chunks"], 8)
        self.assertGreater(big["tok_s"], small["tok_s"])           # experts are streamed once per chunk

    def test_layer_pipeline_overlaps_chunks(self):
        plan = muse.Plan(split="layer", gpu_layers=(40, 22))
        one = muse.prefill(plan, nodes(), IB, tokens=4096, chunk=4096)
        two = muse.prefill(plan, nodes(), IB, tokens=8192, chunk=4096)
        self.assertLess(two["seconds"], 2 * one["seconds"] * 1.2)


class CliTest(unittest.TestCase):
    def test_decode_and_prefill_commands(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            muse_sim.main(["decode", "--gpu-layers", "32,30", "--set", "b550.cpu_gmacs.iq256=20"])
            muse_sim.main(["prefill", "--gpu-layers", "46,16", "--tokens", "2048", "--chunk", "2048"])
        text = out.getvalue()
        self.assertIn("decode", text)
        self.assertIn("prefill 2048 tokens", text)


if __name__ == "__main__":
    unittest.main()
