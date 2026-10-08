"""GLM API boundary regressions; no weights or GPU required."""
import threading
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock
import unittest

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, GlmEngine, Service


class GlmBoundary(unittest.TestCase):
    def test_reap_alias_server_startup(self):
        import serve.server as server
        import gguf_reader
        import strata_tokenizer

        class FakeGlmEngine(GlmEngine):
            def __init__(self, *args, **kwargs):
                self.max_context = 128
                self.launch_args = args
                self.info, self.last = {}, {}
                self.ended, self.unloaded = False, False

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "glm.json"
            config.write_text(json.dumps({"exe": "unused", "model": "reap.gguf", "prefill_experts": "f16-batched"}))
            metadata = {"general.architecture": "glm5-next", "tokenizer.ggml.eos_token_id": 100,
                        "tokenizer.ggml.tokens": ["ordinary", "<|user|>", "<|observation|>", "<|assistant|>"],
                        "tokenizer.chat_template": "{% for m in messages %}{{m.content}}{% endfor %}"}
            with mock.patch.object(sys, "argv", ["server", "--engine", "glm", "--config", str(config), "--port", "0"]), \
                 mock.patch.object(gguf_reader, "GGUFFile", return_value=mock.Mock(metadata=metadata)), \
                 mock.patch.object(strata_tokenizer.Tokenizer, "from_gguf", return_value=ByteTokenizer()), \
                 mock.patch.object(server, "GlmEngine", FakeGlmEngine), \
                 mock.patch.object(server, "Server"), \
                 mock.patch.object(server, "serve") as http, \
                 mock.patch.object(server.time, "sleep", side_effect=KeyboardInterrupt):
                self.assertEqual(server.main(), 0)
            service = http.call_args.args[0]
            self.assertFalse(service.effort_end)
            self.assertEqual(service.stop_ids, {1, 2, 100})
            self.assertIn("1,2,100", service.engine.launch_args[1])
            self.assertEqual(service.engine.launch_args[1][22], "f16-batched")

    def test_sampling_rejected_before_engine_write(self):
        engine = GlmEngine.__new__(GlmEngine)
        # There is deliberately no process: invalid settings must fail before IO.
        with self.assertRaisesRegex(ValueError, "greedy"):
            list(engine.generate([1], 2, {"temperature": 0.7}, threading.Event()))
        with self.assertRaisesRegex(ValueError, "text"):
            list(engine.generate([1], 2, {}, threading.Event(), embeddings="image"))
        self.assertEqual(GlmEngine.sampling_keys({"temperature": 0, "top_p": 1}), "")

    def test_parallel_glm_admits_directly_without_solo_reprefill(self):
        import serve.server as server
        from serve.test_parallel import FAKE_BATCH
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "fake.py"
            script.write_text(FAKE_BATCH)
            log = Path(directory) / "requests.log"
            real = server.subprocess.Popen
            with mock.patch.object(server.subprocess, "Popen",
                                   lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
                engine = GlmEngine("glm", ["--batch", "2", "--log", str(log)])
            try:
                output = [t for t in engine.generate([10, 11], 4, {}, threading.Event()) if t is not None]
                self.assertEqual(output, list(b"ok, "))
                self.assertTrue(log.read_text().startswith("BGEN "))
                self.assertNotIn("\nGEN ", log.read_text())
            finally:
                engine.close()

    def test_chat_request_metadata_is_not_sampling(self):
        request = {"model": "glm-5.3-flash-q2", "messages": [{"role": "user", "content": "hello"}],
                   "stream": True, "max_tokens": 512, "reasoning_effort": "low", "temperature": 0}
        self.assertEqual(GlmEngine.sampling_keys(request), "")
        for field, value in (("top_p", 0.9), ("seed", 42), ("frequency_penalty", 0.5)):
            with self.assertRaisesRegex(ValueError, "greedy"):
                GlmEngine.sampling_keys({**request, field: value})

    def test_gguf_template_and_stop_ids(self):
        engine = GlmEngine.__new__(GlmEngine)
        engine.max_context = 128
        tok = ByteTokenizer()
        template = ChatTemplate(source="{% for m in messages %}{{ m.content }}{% endfor %}")
        svc = Service(engine, tok, template, stop_ids={154820, 154827, 154829})
        ids, _, _ = svc.prepare([{"role": "user", "content": "ordinary text"}], None, {}, 2)
        self.assertEqual(tok.decode(ids), "ordinary text")
        self.assertEqual(svc.stop_ids, {154820, 154827, 154829})
        with self.assertRaisesRegex(ValueError, "tool-call"):
            svc.prepare([{"role": "user", "content": "hello"}], [{"name": "test"}], {}, 2)


if __name__ == "__main__":
    unittest.main()
