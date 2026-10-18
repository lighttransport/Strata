"""GLM-5.x tool calls, effort and thinking on the server: the real chat template (serve/fixtures, from the GGUF's
tokenizer.chat_template), its call syntax through the output parser, and the request mapping.  No weights or GPU."""
import json
import unittest
from pathlib import Path

from serve.frontend import (ChatTemplate, anthropic_to_messages, forced_call, make_parser, openai_to_messages,
                            parse_glm_tool_call)
from serve.server import ByteTokenizer, GlmEngine, Service

TEMPLATE = Path(__file__).parent / "fixtures" / "glm53f_chat_template.jinja"
TOOLS = [{"name": "Bash", "description": "run a command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"},
                                                          "opts": {"type": "object"}}}},
         {"name": "Read", "description": "read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
         {"name": "Now", "description": "the time", "parameters": {"type": "object", "properties": {}}}]


def run(text, chunks=None, thinking=True, tools=TOOLS):
    parser = make_parser("glm", thinking=thinking, tools=tools, stream_tools=True)
    events = []
    for piece in (chunks if chunks is not None else [text]):
        events += parser.feed(piece)
    return events + parser.finish()


def summary(events):
    """(merged text and call events, the streamed argument JSON per call name)"""
    out, args = [], {}
    for e in events:
        if e.kind == "tool_call":
            out.append(("call", e.call.name, e.call.arguments))
        elif e.kind in ("reasoning", "content"):
            if out and out[-1][0] == e.kind:
                out[-1] = (e.kind, out[-1][1] + e.text)
            else:
                out.append((e.kind, e.text))
        elif e.kind == "tool_args":
            args.setdefault(e.call.name, []).append(e.text)
    return out, {k: "".join(v) for k, v in args.items()}


class GlmToolCalls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = ChatTemplate(TEMPLATE)

    def test_template_dialect_and_caps(self):
        self.assertEqual(self.template.dialect, "glm")
        self.assertTrue(self.template.uses_tool_ids)
        self.assertTrue(all(self.template.caps.values()), self.template.caps)

    def test_parallel_calls_with_typed_arguments(self):
        text = ('Let me look.</think>I will run it.<tool_call>Bash<arg_key>command</arg_key><arg_value>ls -la\n'
                ' echo "<a>"\n</arg_value><arg_key>timeout</arg_key><arg_value>30</arg_value><arg_key>opts</arg_key>'
                '<arg_value>{"x": [1, 2]}</arg_value></tool_call><tool_call>Read<arg_key>path</arg_key>'
                '<arg_value>/tmp/a b.txt</arg_value></tool_call>')
        whole = summary(run(text))
        self.assertEqual(whole[0], [
            ("reasoning", "Let me look."), ("content", "I will run it."),
            ("call", "Bash", {"command": 'ls -la\n echo "<a>"\n', "timeout": 30, "opts": {"x": [1, 2]}}),
            ("call", "Read", {"path": "/tmp/a b.txt"})])
        self.assertEqual(summary(run(text, list(text))), whole)            # streamed one character at a time
        self.assertEqual({k: json.loads(v) for k, v in whole[1].items()},  # the streamed JSON is the arguments
                         {"Bash": whole[0][2][2], "Read": whole[0][3][2]})

    def test_call_without_arguments_and_unclosed_call(self):
        self.assertEqual(summary(run("</think><tool_call>Now</tool_call>"))[0], [("call", "Now", {})])
        # a reply cut inside a value: the announced call is not completed, nothing is invented
        out = summary(run("</think><tool_call>Read<arg_key>path</arg_key><arg_value>/tm"))[0]
        self.assertFalse([e for e in out if e[0] == "call"])

    def test_call_written_inside_the_reasoning(self):
        text = "Check it.\n<tool_call>Read<arg_key>path</arg_key><arg_value>/a</arg_value></tool_call></think>"
        out = summary(run(text))[0]
        self.assertIn(("call", "Read", {"path": "/a"}), out)
        self.assertEqual(summary(run(text, list(text)))[0], out)

    def test_prose_mentioning_the_tag_is_text(self):
        self.assertEqual(summary(run("</think>Use <tool_call> blocks.\nThen go on."))[0],
                         [("content", "Use <tool_call> blocks.\nThen go on.")])

    def test_render_and_parse_round_trip(self):
        calls = [{"id": "c1", "function": {"name": "Bash", "arguments": {"command": "ls\nx", "timeout": 5}}},
                 {"id": "c2", "function": {"name": "Read", "arguments": {"path": "/a"}}}]
        prompt = self.template.render([{"role": "user", "content": "hi"},
                                       {"role": "assistant", "content": "ok", "tool_calls": calls}],
                                      tools=TOOLS, add_generation_prompt=False)
        body = prompt[prompt.index("<think></think>") + len("<think></think>"):]
        out = summary(run(body, thinking=False))[0]
        self.assertEqual(out, [("content", "ok")] + [("call", c["function"]["name"], c["function"]["arguments"])
                                                     for c in calls])
        self.assertEqual(parse_glm_tool_call("Bash<arg_key>timeout</arg_key><arg_value>5</arg_value>", TOOLS[0]).arguments,
                         {"timeout": 5})

    def test_forced_openers(self):
        self.assertEqual(forced_call("required", TOOLS, "glm"), "<tool_call>")
        self.assertEqual(forced_call({"type": "tool", "name": "Read"}, TOOLS, "glm"), "<tool_call>Read")
        self.assertEqual(forced_call("required", TOOLS[:1], "glm"), "<tool_call>Bash")   # one tool: named
        self.assertEqual(forced_call("required", TOOLS), "<tool_call>\n<function=")      # Qwen unchanged
        # the forced opening followed by the model's text parses as one call
        self.assertEqual(summary(run("<tool_call>Read" + "<arg_key>path</arg_key><arg_value>/b</arg_value></tool_call>",
                                     thinking=False))[0], [("call", "Read", {"path": "/b"})])

    def test_results_are_sorted_by_call_id(self):
        req = {"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "Bash", "arguments": "{\"command\": \"a\"}"}},
                {"id": "c2", "type": "function", "function": {"name": "Read", "arguments": "{\"path\": \"b\"}"}}]},
            {"role": "tool", "tool_call_id": "c2", "content": "SECOND"},
            {"role": "tool", "tool_call_id": "c1", "content": "FIRST"}]}
        messages, _, _ = openai_to_messages(req, keep_ids=True)
        prompt = self.template.render(messages, tools=TOOLS)
        self.assertLess(prompt.index("FIRST"), prompt.index("SECOND"))
        messages, _, _ = openai_to_messages(req)                 # without ids: the order the client sent
        self.assertNotIn("tool_call_id", messages[2])
        self.assertLess(self.template.render(messages, tools=TOOLS).index("SECOND"),
                        self.template.render(messages, tools=TOOLS).index("FIRST"))
        anth = {"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "u1", "name": "Bash", "input": {"command": "a"}},
                                              {"type": "tool_use", "id": "u2", "name": "Read", "input": {"path": "b"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "u2", "content": "SECOND"},
                                         {"type": "tool_result", "tool_use_id": "u1", "content": "FIRST"}]}]}
        messages, _, _ = anthropic_to_messages(anth, keep_ids=True)
        prompt = self.template.render(messages, tools=TOOLS)
        self.assertLess(prompt.index("FIRST"), prompt.index("SECOND"))


class GlmRequestMapping(unittest.TestCase):
    def service(self):
        engine = GlmEngine.__new__(GlmEngine)
        engine.max_context = 4096
        return Service(engine, ByteTokenizer(), ChatTemplate(TEMPLATE))

    def test_effort_levels(self):
        svc = self.service()
        user = [{"role": "user", "content": "hi"}]
        line = lambda kw: svc.render_prompt(user, None, kw).split("<|user|>")[0]   # noqa: E731
        self.assertIn("Reasoning Effort: Low", line({"reasoning_effort": "low"}))
        self.assertIn("Reasoning Effort: High", line({"reasoning_effort": "medium"}))
        self.assertIn("Reasoning Effort: Max", line({"reasoning_effort": "xhigh"}))
        self.assertIn("Reasoning Effort: Max", line({}))
        svc.reasoning_effort_lock = "low"
        self.assertIn("Reasoning Effort: Low", line({"reasoning_effort": "xhigh"}))
        svc.reasoning_effort_lock = "max"
        self.assertIn("Reasoning Effort: Max", line({"reasoning_effort": "low"}))

    def test_thinking_off_closes_the_block(self):
        svc = self.service()
        prompt = svc.render_prompt([{"role": "user", "content": "hi"}], None, {"enable_thinking": False})
        self.assertTrue(prompt.endswith("<|assistant|><think></think>"))
        self.assertTrue(svc.render_prompt([{"role": "user", "content": "hi"}], None, {}).endswith("<|assistant|><think>"))

    def test_quoted_argument_tags_stay_text(self):
        svc = self.service()
        self.assertEqual(svc.dialect, "glm")
        self.assertIn("<arg_key>", svc.literals)
        ids = svc.encode_prompt([{"role": "user", "content": "the tag <arg_key> in a file"}], None, {})
        self.assertIn("<arg_key>", ByteTokenizer().decode(ids))

    def test_sampling_modes(self):
        with self.assertRaisesRegex(ValueError, "greedy"):
            GlmEngine.sampling_keys({"temperature": 1.0})
        engine = GlmEngine.__new__(GlmEngine)
        engine.set_sampling_mode("ignore")
        self.assertEqual(engine.sampling_keys({"temperature": 1.0, "top_p": 0.95}), "")
        with self.assertRaises(ValueError):
            engine.set_sampling_mode("sometimes")

    def test_state_cache_args(self):
        import tempfile
        from serve.server import glm_state_args
        meta = {"tokenizer.ggml.tokens": ["a", "<|user|>", "b", "<|observation|>", "<|assistant|>", "<think>"]}
        self.assertEqual(glm_state_args({}, meta), [])
        with tempfile.TemporaryDirectory() as d:
            args = glm_state_args({"state_cache": {"ram_mib": 512, "disk_dir": d + "/states", "disk_gib": 8}}, meta)
            self.assertIn("--state-boundary=1,3,4,5", args)        # <|user|> first: the root is before it
            self.assertIn("--state-ram-mib=512", args)
            self.assertIn(f"--state-dir={d}/states", args)
            self.assertIn("--state-disk-gib=8.0", args)
        for bad in ({"state_cache": 1}, {"state_cache": {"ram_mib": -1}}, {"state_cache": {"disk_dir": "/tmp/x"}},
                    {"state_cache": {}}):
            with self.assertRaises(SystemExit):
                glm_state_args(bad, meta)
        with self.assertRaises(SystemExit):                          # a model without the role tokens
            glm_state_args({"state_cache": {"ram_mib": 512}}, {"tokenizer.ggml.tokens": ["a"]})

    def test_tool_orders_survive_a_restart(self):
        import tempfile
        import serve.frontend as fe
        tools = [{"name": "Zeta"}, {"name": "Alpha"}, {"name": "Mid"}]
        with tempfile.TemporaryDirectory() as d:
            path = d + "/tool-orders.json"
            saved = dict(fe._TOOL_ORDERS)
            try:
                fe._TOOL_ORDERS.clear(); fe.keep_tool_orders(path)
                fe._stable_tool_order(tools, lambda t: t["name"])           # first sight: Zeta, Alpha, Mid
                fe._TOOL_ORDERS.clear(); fe.keep_tool_orders(path)          # a restart
                again = fe._stable_tool_order(list(reversed(tools)), lambda t: t["name"])
                self.assertEqual([t["name"] for t in again], ["Zeta", "Alpha", "Mid"])
            finally:
                fe.keep_tool_orders(None); fe._TOOL_ORDERS.clear(); fe._TOOL_ORDERS.update(saved)

    def test_session_file_needs_prefix_cache(self):
        from serve.server import SessionRefused
        engine = GlmEngine.__new__(GlmEngine)
        engine.set_sampling_mode("reject")
        with self.assertRaisesRegex(SessionRefused, "prefix_cache"):
            engine.session_file("save", "/tmp/x")

    def test_honored_sampling(self):
        engine = GlmEngine.__new__(GlmEngine)
        engine.set_sampling_mode("honor")
        self.assertEqual(engine.sampling_keys({"temperature": 1.0, "top_p": 0.95, "top_k": 200, "seed": 7,
                                               "model": "x", "messages": []}),
                         " temperature=1.0 top_p=0.95 top_k=64 seed=7")
        self.assertEqual(engine.sampling_keys({"temperature": 0}), "")              # greedy stays greedy
        with self.assertRaisesRegex(ValueError, "greedy"):                          # penalties are not sampled here
            engine.sampling_keys({"temperature": 1.0, "frequency_penalty": 0.5})

    def test_prefix_cache_keys(self):
        engine = GlmEngine.__new__(GlmEngine)
        engine.set_sampling_mode("reject", prefix_cache=True)
        self.assertEqual(engine.sampling_keys({"strata_prefix": {"tokens": 1200}}), " pin=1200")
        self.assertEqual(engine.sampling_keys({"strata_checkpoint": False}), " ckpt=0")
        with self.assertRaisesRegex(ValueError, "greedy"):          # still strict about sampling
            engine.sampling_keys({"temperature": 1.0})
        plain = GlmEngine.__new__(GlmEngine)
        plain.set_sampling_mode("reject")                             # no prefix cache: no keys, as before
        self.assertEqual(plain.sampling_keys({"strata_prefix": {"tokens": 1200}}), "")


if __name__ == "__main__":
    unittest.main()
