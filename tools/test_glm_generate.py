"""Long-prompt transport and context admission regressions without a GPU."""
import contextlib
import io
import pathlib
import tempfile
import unittest
from unittest import mock

import glm_generate


class GenerateTransport(unittest.TestCase):
    def run_frontend(self, arguments, ids, popen):
        metadata = {"general.architecture": "glm5next", "glm5next.context_length": 1048576,
                    "tokenizer.chat_template": "{{ messages[0].content }}"}
        tokenizer = mock.Mock()
        tokenizer.encode.return_value = ids
        tokenizer.decode.side_effect = lambda generated, **kw: "x" * len(generated)
        with mock.patch("sys.argv", ["glm_generate", "model.gguf", *arguments]), \
             mock.patch.object(glm_generate, "GGUFFile", return_value=mock.Mock(metadata=metadata)), \
             mock.patch.object(glm_generate.Tokenizer, "from_gguf", return_value=tokenizer), \
             mock.patch.object(glm_generate.subprocess, "Popen", side_effect=popen), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            glm_generate.main()

    def test_large_ids_use_live_file_and_cleanup(self):
        ids = [154880] * 20000  # More than Linux's single-argument limit.
        observed = []
        with tempfile.TemporaryDirectory() as temporary:
            prompt = pathlib.Path(temporary) / "prompt.txt"
            prompt.write_text("source code", encoding="utf-8")

            def process(command, **kwargs):
                self.assertTrue(command[2].startswith("@"))
                tokens = pathlib.Path(command[2][1:])
                self.assertEqual(list(map(int, tokens.read_text().split(","))), ids)
                self.assertIn("--context=131072", command)
                self.assertIn("--gpu-devices=0,1", command)
                self.assertIn("--prefill-expert-cache-mib=auto", command)
                self.assertIn("--lock-weights", command)
                self.assertIn("--decode-prefill-cache", command)
                self.assertIn("--decode-cache-mib=auto", command)
                self.assertIn("--decode-cache-adapt", command)
                self.assertIn("--decode-graphs", command)
                self.assertIn("--decode-cache-window=128", command)
                self.assertLess(max(map(len, command)), 131072)
                observed.append(tokens)
                result = mock.MagicMock()
                result.__enter__.return_value = result
                result.stdout = io.StringIO("7\n9\n")
                result.wait.return_value = 0
                return result

            self.run_frontend(["--prompt-file", str(prompt), "--context", "131072", "--gpu-devices", "0,1",
                               "--prefill-expert-cache-mib", "auto", "--lock-weights", "--decode-prefill-cache",
                               "--decode-cache-mib", "auto", "--decode-cache-adapt", "--decode-graphs",
                               "--decode-cache-window", "128"], ids, process)
        self.assertEqual(len(observed), 1)
        self.assertFalse(observed[0].exists())

    def test_overflow_rejected_before_launch(self):
        launch = mock.Mock()
        with self.assertRaises(SystemExit) as result:
            self.run_frontend(["hello", "--context", "32", "--tokens", "16"], [1] * 17, launch)
        self.assertEqual(result.exception.code, 2)
        launch.assert_not_called()

    def test_ambiguous_prompt_rejected(self):
        with self.assertRaises(SystemExit) as result:
            self.run_frontend(["hello", "--prompt-file", "ignored.txt"], [1], mock.Mock())
        self.assertEqual(result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
