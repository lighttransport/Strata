"""Unit tests for tools/glm_regression.py (no model or GPU: the decoder is mocked)."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import glm_regression as gr  # noqa: E402


def completed(code=0, out="", err=""):
    return subprocess.CompletedProcess([], code, out, err)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        d = pathlib.Path(self.dir.name)
        for name in ("decoder", "model.gguf", "pack.gguf", "ref.logits"):
            (d / name).write_text("x")
        self.decoder = d / "decoder"
        self.corpus = d / "corpus.ids"
        self.corpus.write_text("1,2,3\n4,5,6\n")
        self.cfg = gr.load_config()
        self.cfg["model"], self.cfg["pack"] = str(d / "model.gguf"), str(d / "pack.gguf")
        self.cfg["quality"]["reference"] = str(d / "ref.logits")
        self.cfg["quality"]["corpus"] = str(self.corpus)          # absolute: ROOT / abs = abs
        self.cfg["quality"]["baseline"] = {"tokens": 4, "kl": 0.137, "top1_agreement": 0.877, "perplexity": 4.8}
        self.log = []

    def tearDown(self):
        self.dir.cleanup()

    def logf(self, msg):
        self.log.append(msg)


def eval_rows(kl=0.137, top1=0.877, ppl=4.8, n=2):
    return "".join(json.dumps({"sequence": i + 1, "tokens": 2 * (i + 1), "nll": 1.5, "perplexity": ppl, "kl": kl,
                               "top1_agreement": top1}) + "\n" for i in range(n))


class ConfigTest(unittest.TestCase):
    def test_config_is_consistent(self):
        cfg = gr.load_config()
        sc = cfg["short_context"]
        self.assertEqual(len(sc["golden"]), sc["steps"])
        self.assertGreaterEqual(len(sc["prompt"]), 16)
        for name in ("decode", "check_verify", "mtp_split_verify"):
            self.assertIn(name, sc["cases"])
        self.assertIn("glm_quant_parity", cfg["parity"])

    def test_env_overrides_paths(self):
        with mock.patch.dict("os.environ", {"STRATA_GLM_TEST_MODEL": "/m", "STRATA_GLM_TEST_PACK": "/p"}):
            cfg = gr.load_config()
        self.assertEqual((cfg["model"], cfg["pack"]), ("/m", "/p"))


class CaseTest(Fixture):
    def test_pass_on_golden(self):
        out = "\n".join(map(str, self.cfg["short_context"]["golden"]))
        with mock.patch.object(gr.subprocess, "run", return_value=completed(0, out)) as run:
            self.assertEqual(gr.run_case(self.cfg, "check_verify", self.decoder, self.logf), 0)
        cmd = run.call_args[0][0]
        self.assertIn("--check-verify", cmd)
        self.assertTrue(any(a.startswith("--expert-pack=") for a in cmd))
        self.assertEqual(run.call_args[1]["env"]["STRATA_GLM_CANON"], "1")

    def test_case_env_applied(self):
        out = " ".join(map(str, self.cfg["short_context"]["golden"]))
        with mock.patch.object(gr.subprocess, "run", return_value=completed(0, out)) as run:
            gr.run_case(self.cfg, "mtp_split_verify", self.decoder, self.logf)
        self.assertEqual(run.call_args[1]["env"]["STRATA_GLM_SPLIT_VERIFY"], "1")

    def test_fail_on_token_change(self):
        golden = list(self.cfg["short_context"]["golden"])
        golden[-1] += 1
        with mock.patch.object(gr.subprocess, "run", return_value=completed(0, " ".join(map(str, golden)))):
            self.assertEqual(gr.run_case(self.cfg, "decode", self.decoder, self.logf), 1)
        self.assertIn("differ", self.log[-1])

    def test_fail_on_decoder_error(self):
        with mock.patch.object(gr.subprocess, "run", return_value=completed(1, "", "GLM: mismatch")):
            self.assertEqual(gr.run_case(self.cfg, "check_replay", self.decoder, self.logf), 1)

    def test_skip_when_model_missing(self):
        self.cfg["model"] = "/nonexistent.gguf"
        self.assertEqual(gr.run_case(self.cfg, "decode", self.decoder, self.logf), gr.SKIP)


class ParityTest(Fixture):
    def test_parity_exit_code(self):
        bindir = self.decoder.parent
        (bindir / "glm_quant_parity").write_text("x")
        with mock.patch.object(gr.subprocess, "run", return_value=completed(0, "ok")):
            self.assertEqual(gr.run_parity(self.cfg, "glm_quant_parity", bindir, self.logf), 0)
        with mock.patch.object(gr.subprocess, "run", return_value=completed(1, "bad")):
            self.assertEqual(gr.run_parity(self.cfg, "glm_quant_parity", bindir, self.logf), 1)
        self.assertEqual(gr.run_parity(self.cfg, "glm_cache_parity", bindir, self.logf), gr.SKIP)


class QualityTest(Fixture):
    def run_q(self, proc, **kw):
        with mock.patch.object(gr.subprocess, "run", return_value=proc):
            return gr.run_quality(self.cfg, self.decoder, log=self.logf, **kw)

    def test_pass_with_expected_trailing_exit(self):
        self.assertEqual(self.run_q(completed(1, eval_rows(), "x\n" + gr.TRAILING + "\n")), 0)
        self.assertIn("PASS", self.log[-1])

    def test_fail_on_kl_regression(self):
        self.assertEqual(self.run_q(completed(0, eval_rows(kl=0.15))), 1)
        self.assertTrue(any("KL" in m for m in self.log))

    def test_fail_on_top1_and_ppl(self):
        self.assertEqual(self.run_q(completed(0, eval_rows(top1=0.86, ppl=5.0))), 1)
        joined = "\n".join(self.log)
        self.assertIn("top-1", joined)
        self.assertIn("perplexity", joined)

    def test_other_errors_fail(self):
        self.assertEqual(self.run_q(completed(1, eval_rows(n=1), "GLM: nonfinite evaluation logits")), 1)

    def test_subset_reports_without_thresholds(self):
        self.assertEqual(self.run_q(completed(1, eval_rows(kl=0.5, n=1), gr.TRAILING), sequences=1), 0)

    def test_no_baseline_fails(self):
        self.cfg["quality"]["baseline"] = None
        self.assertEqual(self.run_q(completed(0, eval_rows())), 1)

    def test_update_baseline(self):
        cfg_path = pathlib.Path(self.dir.name) / "cfg.json"
        cfg_path.write_text(json.dumps(json.loads(gr.CONFIG.read_text())))
        self.assertEqual(self.run_q(completed(0, eval_rows()), update_baseline=True, config_path=cfg_path), 0)
        stored = json.loads(cfg_path.read_text())["quality"]["baseline"]
        self.assertAlmostEqual(stored["kl"], 0.137)
        self.assertEqual(stored["tokens"], 4)

    def test_configured_tolerance_catches_affinity_005(self):
        cfg = gr.load_config()
        base, tol = cfg["quality"]["baseline"], cfg["quality"]["tolerance"]
        same = {k: base[k] for k in ("tokens", "kl", "top1_agreement", "perplexity")}
        self.assertEqual(gr.check_quality(same, base, tol), [])
        affinity = dict(same, kl=0.143)        # measured KL at route affinity 0.05
        self.assertTrue(gr.check_quality(affinity, base, tol))
        eight_tokens = dict(same, top1_agreement=base["top1_agreement"] - 9 / 8192)
        self.assertTrue(gr.check_quality(eight_tokens, base, tol))

    def test_tolerance_math(self):
        base = {"tokens": 10, "kl": 0.1, "top1_agreement": 0.9, "perplexity": 5.0}
        tol = {"kl_abs": 0.003, "kl_rel": 0.03, "top1_abs": 0.005, "ppl_rel": 0.01}
        ok = {"tokens": 10, "kl": 0.105, "top1_agreement": 0.896, "perplexity": 5.04}
        self.assertEqual(gr.check_quality(ok, base, tol), [])
        bad = dict(ok, kl=0.107)
        self.assertEqual(len(gr.check_quality(bad, base, tol)), 1)


if __name__ == "__main__":
    unittest.main()
