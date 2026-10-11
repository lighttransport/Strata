"""Multinode split model: legacy single-worker path unchanged, worker rooflines, links, RAM fit and the CLI."""
import contextlib
import io
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "sim"))

import decode  # noqa: E402
import glm_sim  # noqa: E402
import hw as hwmod  # noqa: E402
import kernels  # noqa: E402
import multinode  # noqa: E402
from config import RunConfig  # noqa: E402


def reap_cfg(**kw):
    return RunConfig(pack="reap50_q23", speculation="none", mtp_depth=0, decode_cache_mib=3584, tier_policy="static", **kw)


class LegacyPathTest(unittest.TestCase):
    def test_single_worker_preset_numbers_unchanged(self):
        """The calibrated one-worker preset keeps the numbers recorded before the multinode model was added."""
        hw = hwmod.PRESETS["tr16+b550-ib"]()
        params = kernels.Params()
        for share, expected in ((0.0, 14.14), (0.25, 16.05), (0.5, 12.18)):
            got = decode.simulate(hw, reap_cfg(remote_share=share), params).tok_s
            self.assertAlmostEqual(got, expected, delta=0.02, msg=f"share {share}")


class MultinodeModelTest(unittest.TestCase):
    def setUp(self):
        self.tr16 = hwmod.PRESETS["tr16"]()
        self.b550 = hwmod.PRESETS["b550"]()
        self.params = kernels.Params()

    def workers(self, link="ib", count=1, node=None):
        node = node or self.b550
        return [hwmod.worker_from_hardware(node, hwmod.parse_link(link)) for _ in range(count)]

    def run_share(self, workers, shares, mode="row"):
        hw = self.tr16.copy()
        hw.workers = workers
        cfg = reap_cfg(remote_share=sum(shares), remote_shares=",".join(map(str, shares)), split_mode=mode)
        return decode.simulate(hw, cfg, self.params)

    def test_ib_worker_quarter_share_helps_ordinary_decode(self):
        single = decode.simulate(self.tr16, reap_cfg(), self.params).tok_s
        two = self.run_share(self.workers(), [0.25]).tok_s
        self.assertGreater(two, single)
        self.assertGreater(two, single * 1.05)

    def test_half_share_on_the_slower_node_is_worse_than_a_quarter(self):
        # B550 streams experts at ~29 GB/s against TR16's 72: giving it half the rows makes it the long pole.
        self.assertGreater(self.run_share(self.workers(), [0.25]).tok_s, self.run_share(self.workers(), [0.5]).tok_s)

    def test_gbe_is_slower_than_ib(self):
        self.assertGreater(self.run_share(self.workers("ib"), [0.25]).tok_s, self.run_share(self.workers("gbe"), [0.25]).tok_s)

    def test_two_workers_share_the_remote_work(self):
        one = self.run_share(self.workers(count=1), [0.5]).tok_s
        two = self.run_share(self.workers(count=2), [0.25, 0.25]).tok_s
        self.assertGreater(two, one)

    def test_expert_split_pays_an_imbalance_penalty(self):
        row = self.run_share(self.workers(), [0.25], "row").tok_s
        expert = self.run_share(self.workers(), [0.25], "expert").tok_s
        self.assertLess(expert, row)

    def test_ram_overflow_is_reported(self):
        small = self.b550.copy()
        small.memory.gib = 8.0
        result = self.run_share(self.workers(node=small), [0.5])
        self.assertTrue(any("exceed its RAM" in n for n in result.notes))

    def test_share_grid(self):
        rows = list(multinode.share_grid(2, "row"))
        self.assertIn((0.25, 0.25), rows)
        self.assertNotIn((0.5, 0.5), rows)       # local share must stay >= 0.25
        self.assertTrue(all(0 < sum(r) <= 0.75 + 1e-9 for r in rows))
        self.assertTrue(all(0 < sum(r) <= 0.8 + 1e-9 for r in multinode.share_grid(3, "expert")))

    def test_search_ranks_and_reports(self):
        rows, base = multinode.search(self.tr16, self.workers(), reap_cfg(), self.params, "row", top=5)
        self.assertTrue(rows)
        self.assertEqual(rows, sorted(rows, key=lambda r: -r["decode"]))
        text = multinode.describe(rows, base, self.tr16, self.workers(), "row")
        self.assertIn("best:", text)
        self.assertIn("single node:", text)


class CliTest(unittest.TestCase):
    def run_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = glm_sim.main(list(argv))
        return code, out.getvalue()

    def test_multinode_command(self):
        code, text = self.run_cli("multinode", "--node", "tr16", "--node", "b550", "--link", "ib", "--pack", "reap50_q23",
                                  "--speculation", "none", "--decode-cache-mib", "3584", "--tier-policy", "static", "--top", "3")
        self.assertEqual(code, 0)
        self.assertIn("best: shares", text)
        self.assertIn("unvalidated", text)

    def test_multinode_both_splits_and_custom_link(self):
        code, text = self.run_cli("multinode", "--node", "tr16", "--node", "b550", "--node", "b550", "--link", "ib", "--link", "401,0.117,80",
                                  "--split", "both", "--pack", "reap50_q23", "--speculation", "none", "--top", "2")
        self.assertEqual(code, 0)
        self.assertIn("split: row", text)
        self.assertIn("split: expert", text)


if __name__ == "__main__":
    unittest.main()


class Glm53ModelTest(unittest.TestCase):
    def tearDown(self):
        import model
        model.use_model("flash")

    def test_geometry_and_sizes(self):
        import model
        model.use_model("glm53")
        self.assertEqual(model.GEOMETRY.moe_layers, 75)
        self.assertEqual(decode.GEOMETRY.hidden, 6144)
        full = model.pack("glm53_q23")
        # 724.8B routed parameters at Q2_K/Q2_K/Q3_K (2.896 bits per weight)
        self.assertAlmostEqual(full.routed_total_bytes() / 1e9, 262.4, delta=0.5)
        self.assertAlmostEqual(model.pack("glm53_reap50_q23").routed_total_bytes() / full.routed_total_bytes(), 0.5, places=3)

    def test_switching_back_restores_flash_numbers(self):
        hw = hwmod.PRESETS["tr16+b550-ib"]()
        before = decode.simulate(hw, reap_cfg(remote_share=0.25), kernels.Params()).tok_s
        import model
        model.use_model("glm53")
        model.use_model("flash")
        self.assertAlmostEqual(decode.simulate(hw, reap_cfg(remote_share=0.25), kernels.Params()).tok_s, before, places=6)

    def test_cli_model_switch(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = glm_sim.main(["--model", "glm53", "estimate", "--hw", "tr16", "--pack", "glm53_reap50_q22", "--dense-format", "q4",
                                 "--speculation", "mtp", "--mtp", "1", "--decode-cache-mib", "3000"])
        self.assertEqual(code, 0)
        self.assertIn("DECODE", out.getvalue())


class MimoModelTest(unittest.TestCase):
    def tearDown(self):
        import model
        model.use_model("flash")

    def test_sizes_from_config(self):
        import model
        model.use_model("mimo_flash")
        self.assertEqual(model.GEOMETRY.moe_layers, 47)
        self.assertAlmostEqual(model.pack("mimof_q23").routed_total_bytes() / 1e9, 109.6, delta=0.5)   # 302.8B routed
        model.use_model("mimo_pro")
        self.assertEqual(model.GEOMETRY.moe_layers, 69)
        self.assertAlmostEqual(model.pack("mimop_q23").routed_total_bytes() / 1e9, 362.0, delta=1.0)  # 1,000B routed

    def test_flash_runs_on_one_node_and_pro_needs_two(self):
        import model
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(glm_sim.main(["--model", "mimo_flash", "estimate", "--hw", "tr16", "--pack", "mimof_q23",
                                           "--dense-format", "q4", "--speculation", "mtp", "--mtp", "1", "--decode-cache-mib", "6000"]), 0)
            self.assertEqual(glm_sim.main(["--model", "mimo_pro", "multinode", "--node", "tr16", "--node", "b550", "--pack",
                                           "mimop_reap62_q23", "--dense-format", "q4", "--speculation", "none", "--top", "1"]), 0)
        text = out.getvalue()
        self.assertIn("DECODE", text)
        self.assertIn("best: shares", text)
