"""GLM API boundary regressions; no weights or GPU required."""
import threading
import unittest

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, GlmEngine, Service


class GlmBoundary(unittest.TestCase):
    def test_sampling_rejected_before_engine_write(self):
        engine = GlmEngine.__new__(GlmEngine)
        # There is deliberately no process: invalid settings must fail before IO.
        with self.assertRaisesRegex(ValueError, "greedy"):
            list(engine.generate([1], 2, {"temperature": 0.7}, threading.Event()))
        with self.assertRaisesRegex(ValueError, "text"):
            list(engine.generate([1], 2, {}, threading.Event(), embeddings="image"))
        self.assertEqual(GlmEngine.sampling_keys({"temperature": 0, "top_p": 1}), "")

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
