"""tools/strata_agent_config.py: the settings it gives each coding agent, and pi's models.json merge.  No server needed.

    python tools/test_strata_agent_config.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_agent_config as A  # noqa: E402

GLM = {"model": "glm-5.3-flash", "context": 131072, "images": False, "key_required": False}
QWEN = {"model": "qwen3.8-flash-next", "context": 65536, "images": True, "key_required": True}
BASE = "http://127.0.0.1:8080"


class Configs(unittest.TestCase):
    def test_pi(self):
        p = A.configs(GLM, BASE, "none")["pi"]["content"]["providers"]["strata"]
        self.assertEqual((p["baseUrl"], p["api"], p["apiKey"]), (BASE + "/v1", "openai-completions", "none"))
        self.assertEqual(p["compat"]["thinkingFormat"], "zai")
        self.assertEqual(p["compat"]["maxTokensField"], "max_tokens")
        m = p["models"][0]
        self.assertEqual((m["id"], m["contextWindow"], m["maxTokens"], m["input"], m["reasoning"]),
                         ("glm-5.3-flash", 131072, 32768, ["text"], True))
        q = A.configs(QWEN, BASE, "$STRATA_API_KEY")["pi"]["content"]["providers"]["strata"]
        self.assertNotIn("thinkingFormat", q["compat"])        # Qwen: OpenAI's reasoning_effort
        self.assertEqual(q["models"][0]["input"], ["text", "image"])
        self.assertEqual(q["models"][0]["maxTokens"], 16384)   # a quarter of a 64K context

    def test_codex_uses_responses(self):
        toml = A.configs(GLM, BASE, "none")["codex"]["content"]
        self.assertIn('wire_api = "responses"', toml)
        self.assertNotIn("chat", toml)
        self.assertIn("model_context_window = 131072", toml)
        self.assertIn("model_auto_compact_token_limit = 104857", toml)
        self.assertNotIn("env_key", toml)
        self.assertIn('env_key = "STRATA_API_KEY"', A.configs(QWEN, BASE, "x")["codex"]["content"])

    def test_claude_code(self):
        env = A.configs(GLM, BASE, "none")["claude-code"]["env"]
        self.assertEqual(env["ANTHROPIC_BASE_URL"], BASE)                  # no /v1: Claude Code adds it
        self.assertEqual({env[k] for k in env if k.startswith("ANTHROPIC_DEFAULT_")}, {"glm-5.3-flash"})
        self.assertEqual(env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "131072")

    def test_server_keys(self):
        self.assertEqual(A.configs(GLM, BASE, "none")["server"]["content"],
                         {"responses_overflow_event": True, "glm_sampling": "ignore"})
        self.assertNotIn("glm_sampling", A.configs(QWEN, BASE, "x")["server"]["content"])

    def test_omp(self):
        y = A.configs(GLM, BASE, "none")["omp"]["content"]
        self.assertIn("    auth: none\n", y)
        self.assertIn("        contextWindow: 131072\n", y)
        self.assertNotIn("auth: none", A.configs(QWEN, BASE, "$STRATA_API_KEY")["omp"]["content"])

    def test_max_output_bounds(self):
        self.assertEqual([A.max_output(c) for c in (2048, 65536, 1048576)], [1024, 16384, 32768])


class WritePi(unittest.TestCase):
    def test_merge_keeps_other_providers_and_a_backup(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "models.json"
            path.write_text(json.dumps({"providers": {"ollama": {"baseUrl": "x"}, "strata": {"old": 1}}}))
            backup = A.write_pi({"baseUrl": "new"}, path)
            data = json.loads(path.read_text())
            self.assertEqual(data["providers"], {"ollama": {"baseUrl": "x"}, "strata": {"baseUrl": "new"}})
            self.assertEqual(json.loads(backup.read_text())["providers"]["strata"], {"old": 1})
            fresh = Path(d) / "sub" / "models.json"
            self.assertIsNone(A.write_pi({"baseUrl": "n"}, fresh))
            self.assertEqual(json.loads(fresh.read_text()), {"providers": {"strata": {"baseUrl": "n"}}})


if __name__ == "__main__":
    unittest.main()
