import json
import gzip
import hashlib
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import glm_q2_humaneval as humaneval
from glm_q2_humaneval import load_engine_arguments, result_summary


class ArgumentIdentityTest(unittest.TestCase):
    def test_config_is_bound_to_resumable_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            decoder = root / "decoder"
            decoder.write_bytes(b"fake decoder")
            dataset = root / "tasks.gz"
            dataset.write_bytes(gzip.compress(b"\n".join(json.dumps(dict(
                task_id=f"HumanEval/{i}", prompt="def candidate():\n", entry_point="candidate"
            )).encode() for i in range(164))))
            config = root / "config.json"
            config.write_text(json.dumps(dict(context=256, threads=3, prefill_batch=16,
                                             env={"STRATA_TEST_CONFIG": "1"})))
            output = root / "output"
            argv = ["humaneval", str(model), str(dataset), "--output", str(output),
                    "--decoder", str(decoder), "--config", str(config),
                    "--generate-only", "--tokens", "4", "--limit", "1"]
            tokenizer = Mock()
            tokenizer.encode.return_value = [1, 2]
            tokenizer.decode.return_value = "def candidate(): return 1"
            engine = Mock()
            engine.generate.return_value = [3]
            factory = Mock(return_value=engine)
            metadata = {"tokenizer.chat_template": "{{ messages[0].content }}",
                        "tokenizer.ggml.eos_token_id": 99,
                        "tokenizer.ggml.tokens": ["<|user|>", "<|observation|>"]}
            with patch.object(humaneval.sys, "argv", argv), \
                 patch.object(humaneval, "DATA_SHA256", hashlib.sha256(dataset.read_bytes()).hexdigest()), \
                 patch.object(humaneval, "GGUFFile", return_value=types.SimpleNamespace(metadata=metadata)), \
                 patch.object(humaneval.Tokenizer, "from_gguf", return_value=tokenizer), \
                 patch.dict(humaneval.sys.modules, server=types.SimpleNamespace(GlmEngine=factory)), \
                 patch("sys.stdout", new_callable=io.StringIO):
                humaneval.main()
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["engine_args"][1:4], ["256", "4096", "3"])
                self.assertEqual(manifest["engine_args"][6], "0,1,99")
                self.assertEqual(manifest["environment"]["STRATA_TEST_CONFIG"], "1")
                factory.reset_mock()
                argv.append("--resume")
                humaneval.main()
                factory.assert_not_called()
                config.write_text(json.dumps(dict(context=256, env={"STRATA_TEST_CONFIG": "2"})))
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    humaneval.main()
                factory.assert_not_called()

    def test_ungreaded_and_partial_runs_cannot_qualify(self):
        ungraded = [dict(task_id=f"HumanEval/{i}", passed=None) for i in range(164)]
        self.assertTrue(result_summary(ungraded)["generation_complete"])
        self.assertFalse(result_summary(ungraded)["complete"])
        self.assertIsNone(result_summary(ungraded)["pass_at_1"])
        graded = [dict(task_id=f"HumanEval/{i}", passed=True) for i in range(164)]
        self.assertTrue(result_summary(graded)["complete"])
        self.assertFalse(result_summary(graded[:163])["complete"])

    def test_profile_binds_model_context_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "full-q23.gguf"
            path = Path(directory) / "arguments.json"
            values = [
                str(model),
                "32768",
                "4096",
                "32",
                "0",
                "4096",
                "154820,154827",
            ] + ["0"] * 16
            path.write_text(json.dumps(values))
            self.assertEqual(
                load_engine_arguments(path, model, 32768, {154820, 154827}), values
            )
            for index, value in [
                (0, str(model.with_name("other.gguf"))),
                (1, "8192"),
                (6, "154820"),
            ]:
                broken = values.copy()
                broken[index] = value
                path.write_text(json.dumps(broken))
                with self.assertRaises(ValueError):
                    load_engine_arguments(path, model, 32768, {154820, 154827})


if __name__ == "__main__":
    unittest.main()
