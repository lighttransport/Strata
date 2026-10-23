"""Coding-agent compatibility (pi, oh-my-pi, Claude Code, Codex, OpenAI SDKs) over HTTP, without a GPU or model pack.

    python -m unittest serve.test_agent_compat -v
"""
from __future__ import annotations

import contextlib
import http.client
import io
import json
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate, openai_to_messages
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, agent_hints, serve, thinking_signature

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ChatTemplate(ROOT / "serve/chat_template.jinja")

THOUGHT = "Look first."
ANSWER = f"{THOUGHT}\n</think>\n\nDone."
CALL = "<tool_call>\n<function=exec_command>\n<parameter=cmd>\n{}\n</parameter>\n</function>\n</tool_call>"
TWO_CALLS = f"{THOUGHT}\n</think>\n\n" + CALL.format("ls") + "\n" + CALL.format("pwd")
CHAT_TOOLS = [{"type": "function", "function": {"name": "exec_command", "description": "Runs a command.",
                                                "parameters": {"type": "object", "properties": {
                                                    "cmd": {"type": "string"}}, "required": ["cmd"]}}}]
ANTHROPIC_TOOLS = [{"name": "exec_command", "description": "Runs a command.",
                    "input_schema": CHAT_TOOLS[0]["function"]["parameters"]}]
RESPONSES_TOOLS = [{"type": "function", "name": "exec_command", "description": "Runs a command.",
                    "parameters": CHAT_TOOLS[0]["function"]["parameters"]}]
SDK = {"anthropic-version": "2023-06-01"}


class Agent(unittest.TestCase):
    script = ANSWER

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, self.script, max_context=16384)
        self.svc = Service(self.engine, self.tok, TEMPLATE)
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body, headers=None):
        """(status, headers, raw body text)."""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                c.request("POST", path, body=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json", **(headers or {})})
                r = c.getresponse()
                return r.status, r, r.read().decode()
        finally:
            c.close()

    @staticmethod
    def sse(raw):
        """[(event name or None, data or None for a comment)] of an SSE body."""
        out = []
        for block in raw.split("\n\n"):
            lines = block.strip().splitlines()
            if not lines:
                continue
            if lines[0].startswith(":"):
                out.append((None, None))
                continue
            name = lines[0][len("event: "):] if lines[0].startswith("event: ") else None
            data = lines[-1][len("data: "):]
            out.append((name, data if data == "[DONE]" else json.loads(data)))
        return out


class AnthropicSdk(Agent):
    def test_thinking_signature_and_cache_usage_for_an_sdk_client(self):
        body = {"model": "m", "max_tokens": 200, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        status, _, raw = self.post("/v1/messages", body, SDK)
        self.assertEqual(status, 200)
        events = self.sse(raw)
        start = events[0][1]["message"]["usage"]
        self.assertEqual((start["cache_creation_input_tokens"], start["cache_read_input_tokens"]), (0, 0))
        # the thinking block ends with its signature, then content_block_stop
        sig = [d for n, d in events if n == "content_block_delta" and d["delta"]["type"] == "signature_delta"]
        self.assertEqual(len(sig), 1)
        self.assertEqual(sig[0]["delta"]["signature"], thinking_signature(THOUGHT + "\n"))
        after = events[[d for _, d in events].index(sig[0]) + 1]
        self.assertEqual(after[0], "content_block_stop")
        delta = [d for n, d in events if n == "message_delta"][0]
        self.assertEqual(delta["usage"]["cache_creation_input_tokens"], 0)
        # collected (not streamed): the block carries the same signature
        status, _, raw = self.post("/v1/messages", {**body, "stream": False}, SDK)
        thinking = json.loads(raw)["content"][0]
        self.assertEqual((thinking["type"], thinking["signature"]), ("thinking", thinking_signature(THOUGHT + "\n")))

    def test_a_client_without_anthropic_version_gets_the_old_stream(self):
        body = {"model": "m", "max_tokens": 200, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        _, _, raw = self.post("/v1/messages", body)
        self.assertNotIn("signature_delta", raw)
        self.assertNotIn("cache_creation_input_tokens", raw)
        self.assertIn('"signature": ""', raw)

    def test_heartbeats_are_ping_events_for_an_sdk_client_and_comments_otherwise(self):
        def run(*args, **kwargs):
            yield "ping", None
            yield "done", {"finish": "stop", "completion_tokens": 0, "reused": 0, "timings": None}
        body = {"model": "m", "max_tokens": 20, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        with mock.patch.object(self.svc, "run", run):
            _, _, sdk = self.post("/v1/messages", body, SDK)
            _, _, plain = self.post("/v1/messages", body)
        self.assertIn(("ping", {"type": "ping"}), self.sse(sdk))
        self.assertNotIn(": keep-alive", sdk)
        self.assertIn(": keep-alive", plain)
        self.assertNotIn("event: ping", plain)

    def test_errors_use_anthropics_envelope(self):
        for path in ("/v1/messages", "/v1/messages/count_tokens"):
            with self.subTest(path=path):
                status, _, raw = self.post(path, {"model": "m", "max_tokens": 5, "messages": 5})
                self.assertEqual(status, 400)
                body = json.loads(raw)
                self.assertEqual(body["type"], "error")
                self.assertEqual(body["error"]["type"], "invalid_request_error")
                self.assertTrue(body["request_id"].startswith("req_"))
        # the other routes keep their own error shape
        status, _, raw = self.post("/v1/chat/completions", {"model": "m", "messages": 5})
        self.assertEqual(status, 400)
        self.assertNotIn("request_id", json.loads(raw))


class ModelList(Agent):
    def get(self, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            c.request("GET", "/v1/models", headers=headers or {})
            r = c.getresponse()
            return r.status, json.loads(r.read())
        finally:
            c.close()

    def test_context_under_every_name_agents_read(self):
        status, body = self.get()
        self.assertEqual(status, 200)
        model = body["data"][0]
        self.assertEqual(body["object"], "list")
        self.assertEqual((model["context_length"], model["max_model_len"], model["meta"]["n_ctx"]),
                         (16384, 16384, 16384))
        self.assertEqual(model["owned_by"], "strata")
        self.assertIsInstance(model["created"], int)

    def test_anthropic_shape_for_an_anthropic_sdk(self):
        status, body = self.get(SDK)
        self.assertEqual(status, 200)
        self.assertEqual(set(body), {"data", "has_more", "first_id", "last_id"})
        model = body["data"][0]
        self.assertEqual((model["type"], model["id"]), ("model", self.svc.model))
        self.assertTrue(model["created_at"].endswith("Z"))
        self.assertEqual(body["first_id"], self.svc.model)


class ChatCompat(Agent):
    BODY = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}

    def chunks(self, body):
        status, _, raw = self.post("/v1/chat/completions", body)
        self.assertEqual(status, 200)
        return [d for _, d in self.sse(raw) if d is not None]

    def test_include_usage_adds_a_last_chunk_without_choices(self):
        chunks = self.chunks({**self.BODY, "stream_options": {"include_usage": True}})
        self.assertEqual(chunks[-1], "[DONE]")
        tail, finish = chunks[-2], chunks[-3]
        self.assertEqual(tail["choices"], [])
        self.assertEqual(tail["usage"], finish["usage"])            # the finish chunk keeps its usage too
        self.assertEqual(finish["choices"][0]["finish_reason"], "stop")
        self.assertEqual(tail["id"], finish["id"])

    def test_without_include_usage_the_stream_is_unchanged(self):
        chunks = self.chunks(self.BODY)
        self.assertTrue(all(c == "[DONE]" or c["choices"] for c in chunks))
        self.assertNotIn("reasoning", json.dumps(chunks).replace("reasoning_content", ""))

    def test_reasoning_field_alias(self):
        self.svc.reasoning_field = "reasoning"
        chunks = self.chunks(self.BODY)
        deltas = [c["choices"][0]["delta"] for c in chunks if c != "[DONE]" and c["choices"]]
        thought = [d for d in deltas if "reasoning_content" in d]
        self.assertTrue(thought)
        self.assertTrue(all(d["reasoning"] == d["reasoning_content"] for d in thought))
        _, _, raw = self.post("/v1/chat/completions", {**self.BODY, "stream": False})
        msg = json.loads(raw)["choices"][0]["message"]
        self.assertEqual(msg["reasoning"], msg["reasoning_content"])


class ResponsesOverflow(Agent):
    BODY = {"model": "m", "input": "hi", "stream": True}

    def test_streamed_overflow_is_a_failed_event_when_configured(self):
        self.svc.responses_overflow_event = True
        with mock.patch.object(self.svc, "encode_prompt", return_value=[ord("x")] * 20000):
            status, r, raw = self.post("/v1/responses", self.BODY)
            self.assertEqual((status, r.getheader("Content-Type")), (200, "text/event-stream"))
            events = [d for _, d in self.sse(raw)]
            self.assertEqual([e["type"] for e in events],
                             ["response.created", "response.in_progress", "response.failed"])
            error = events[-1]["response"]["error"]
            self.assertEqual(error["code"], "context_length_exceeded")
            self.assertIn("exceeds the context window", error["message"])
            # not streamed: the 400 as before
            status, _, raw = self.post("/v1/responses", {**self.BODY, "stream": False})
            self.assertEqual((status, json.loads(raw)["error"]["code"]), (400, "context_length_exceeded"))

    def test_off_by_default(self):
        with mock.patch.object(self.svc, "encode_prompt", return_value=[ord("x")] * 20000):
            status, _, raw = self.post("/v1/responses", self.BODY)
        self.assertEqual((status, json.loads(raw)["error"]["code"]), (400, "context_length_exceeded"))


class ClientShapedRequests(Agent):
    """Request bodies shaped as the agents send them (from their sources): each is accepted and answered."""
    script = f"{THOUGHT}\n</think>\n\n" + CALL.format("ls")

    def test_pi_openai_completions_with_zai_thinking(self):
        body = {"model": "m", "stream": True, "stream_options": {"include_usage": True}, "store": False,
                "max_tokens": 512, "reasoning_effort": "medium", "thinking": {"type": "enabled", "clear_thinking": False},
                "tools": CHAT_TOOLS, "messages": [
                    {"role": "developer", "content": "You are pi."},
                    {"role": "user", "content": "list"},
                    {"role": "assistant", "content": "", "reasoning_content": "plan",
                     "tool_calls": [{"id": "c1", "type": "function",
                                     "function": {"name": "exec_command", "arguments": "{\"cmd\": \"ls\"}"}}]},
                    {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
                    {"role": "user", "content": "again"}]}
        status, _, raw = self.post("/v1/chat/completions", body, {"session_id": "s1", "x-session-affinity": "s1"})
        self.assertEqual(status, 200, raw)
        chunks = [d for _, d in self.sse(raw) if d not in (None, "[DONE]")]
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "tool_calls")

    def test_claude_code_messages(self):
        body = {"model": "glm-5.3-flash", "max_tokens": 4096, "stream": True,
                "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"},
                "context_management": {"edits": []}, "tools": ANTHROPIC_TOOLS,
                "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1; cch=0;"},
                           {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": [{"type": "text", "text": "list",
                                                           "cache_control": {"type": "ephemeral"}}]}]}
        headers = {**SDK, "anthropic-beta": "interleaved-thinking-2025-05-14,context-management-2025-06-27",
                   "x-claude-code-session-id": "abc"}
        status, _, raw = self.post("/v1/messages?beta=true", body, headers)
        self.assertEqual(status, 200, raw)
        events = self.sse(raw)
        self.assertEqual(events[-1][0], "message_stop")
        delta = [d for n, d in events if n == "message_delta"][0]
        self.assertEqual(delta["delta"]["stop_reason"], "tool_use")

    def test_codex_responses(self):
        body = {"model": "m", "stream": True, "store": False, "parallel_tool_calls": False, "tool_choice": "auto",
                "prompt_cache_key": "019a-thread", "include": ["reasoning.encrypted_content"],
                "reasoning": {"effort": "medium", "summary": "auto"}, "tools": RESPONSES_TOOLS,
                "input": [{"type": "message", "role": "developer",
                           "content": [{"type": "input_text", "text": "You are Codex."}]},
                          {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "list"}]}]}
        status, _, raw = self.post("/v1/responses", body, {"session-id": "019a", "originator": "codex_cli_rs"})
        self.assertEqual(status, 200, raw)
        events = [d for _, d in self.sse(raw)]
        self.assertEqual(events[-1]["type"], "response.completed")
        done = [e["item"] for e in events if e["type"] == "response.output_item.done"]
        self.assertEqual([i["type"] for i in done if i["type"] == "function_call"], ["function_call"])


class MaxQueue(Agent):
    def test_a_full_queue_is_429_or_529_with_retry_after(self):
        bodies = {"/v1/chat/completions": {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                  "/v1/messages": {"model": "m", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]},
                  "/v1/responses": {"model": "m", "input": "hi"}}
        self.svc.max_queue = 1
        with self.svc.status_lock:
            self.svc.status.update(busy=True, queued=1)
        try:
            for path, body in bodies.items():
                with self.subTest(path=path):
                    status, r, raw = self.post(path, body)
                    err = json.loads(raw)
                    self.assertEqual((r.getheader("Retry-After"), r.getheader("x-should-retry")), ("5", "true"))
                    if path == "/v1/messages":
                        self.assertEqual((status, err["type"], err["error"]["type"]), (529, "error", "overloaded_error"))
                    else:
                        self.assertEqual((status, err["error"]["code"]), (429, "rate_limit_exceeded"))
        finally:
            with self.svc.status_lock:
                self.svc.status.update(busy=False, queued=0)
        for path, body in bodies.items():                   # room again: answered
            self.assertEqual(self.post(path, body)[0], 200)

    def test_unset_means_no_limit(self):
        with self.svc.status_lock:
            self.svc.status.update(busy=True, queued=50)
        try:
            self.svc.check_queue()                            # no exception
        finally:
            with self.svc.status_lock:
                self.svc.status.update(busy=False, queued=0)


class Hints(unittest.TestCase):
    """Agents' session and request-class headers, as the request's own strata_ fields."""

    def hinted(self, headers, req=None):
        req = dict(req or {})
        agent_hints(headers, req)
        return req

    def test_session_headers(self):
        for h in ("x-claude-code-session-id", "session-id", "session_id", "x-session-affinity", "thread-id"):
            with self.subTest(header=h):
                self.assertEqual(self.hinted({h: " s1 "}), {"strata_session": "s1"})
        self.assertEqual(self.hinted({"session_id": "s1"}, {"strata_session": "mine"}), {"strata_session": "mine"})
        self.assertEqual(self.hinted({}), {})

    def test_side_requests_save_no_checkpoint(self):
        for headers in ({"x-claude-code-request-class": "auxiliary", "x-claude-code-session-id": "s"},
                        {"x-claude-code-request-class": "Compaction"}, {"x-openai-subagent": "compact"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.hinted(headers), {"strata_checkpoint": False})
        # a main request, a Codex review sub-agent: as before, with their session
        self.assertEqual(self.hinted({"x-claude-code-request-class": "main", "x-claude-code-session-id": "s"}),
                         {"strata_session": "s"})
        self.assertEqual(self.hinted({"x-openai-subagent": "review"}), {})
        self.assertEqual(self.hinted({"x-openai-subagent": "compact"}, {"strata_checkpoint": True}),
                         {"strata_checkpoint": True})

    def test_a_session_returns_to_its_own_slot_when_no_slot_holds_its_start(self):
        e = StrataEngine("missing-executable", [], lazy=True)
        e.batch, e.slot_groups, e.slot_gs = 3, 1, 0
        e.slot_order, e.slot_group = [0, 1, 2], [0, 0, 0]
        e.slot_busy = [False] * 3
        e.slot_held = [[1, 2, 3], [4, 5, 6], []]
        e.slot_used = [3.0, 1.0, 2.0]
        e.slot_session = {"b": 1}
        self.assertEqual(e.pick_slot([1, 2, 3, 4]), 0)            # a held start still wins
        self.assertEqual(e.pick_slot([9, 9], "b"), 1)             # its own slot, not the empty one
        self.assertEqual(e.pick_slot([9, 9], "c"), 2)             # another session: as before (empty first)
        self.assertEqual(e.pick_slot([9, 9]), 2)
        e.slot_busy[1] = True
        self.assertEqual(e.pick_slot([9, 9], "b"), 2)             # its slot is busy: as before


class ResponsesForcedCall(unittest.TestCase):
    """Responses tool_choice "required" / a named function opens the call, as on the chat route."""
    TOOLS = [{"type": "function", "name": "search", "parameters": {"type": "object", "properties": {
        "q": {"type": "string"}}}}, {"type": "function", "name": "done", "parameters": {"type": "object"}}]

    def setUp(self):
        from serve.test_server import CallingEngine
        tok = ByteTokenizer()
        self.svc = Service(CallingEngine(tok), tok, TEMPLATE)
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def respond(self, **extra):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                c.request("POST", "/v1/responses", body=json.dumps({"model": "m", "input": "2+2?", "tools": self.TOOLS,
                                                                    "max_output_tokens": 400, **extra}),
                          headers={"Content-Type": "application/json"})
                r = c.getresponse()
                return json.loads(r.read())
        finally:
            c.close()

    def test_required_and_named(self):
        for choice in ("required", {"type": "function", "name": "search"}):
            with self.subTest(choice=choice):
                out = self.respond(tool_choice=choice)
                calls = [(o["name"], json.loads(o["arguments"])) for o in out["output"] if o["type"] == "function_call"]
                self.assertEqual(calls, [("search", {"q": "2+2"})])
        out = self.respond(tool_choice="auto")                 # the model decides: it answers in text
        self.assertFalse([o for o in out["output"] if o["type"] == "function_call"])


class ChatRequestMapping(unittest.TestCase):
    def test_reasoning_given_back_under_other_names(self):
        for name in ("reasoning", "reasoning_text"):
            with self.subTest(name=name):
                req = {"messages": [{"role": "user", "content": "q"},
                                    {"role": "assistant", "content": "a", name: "thought"},
                                    {"role": "user", "content": "q2"}]}
                messages, _, _ = openai_to_messages(req)
                self.assertEqual(messages[1]["reasoning_content"], "thought")
        # reasoning_content wins, and a non-string is not thinking
        req = {"messages": [{"role": "assistant", "content": "a", "reasoning_content": "x", "reasoning": "y"},
                            {"role": "assistant", "content": "b", "reasoning": {"effort": "low"}}]}
        messages, _, _ = openai_to_messages(req)
        self.assertEqual(messages[0]["reasoning_content"], "x")
        self.assertNotIn("reasoning_content", messages[1])

    def test_zai_thinking_field(self):
        base = {"messages": [{"role": "user", "content": "q"}]}
        self.assertEqual(openai_to_messages({**base, "thinking": {"type": "disabled"}})[2], {"enable_thinking": False})
        kw = openai_to_messages({**base, "reasoning_effort": "low",
                                 "thinking": {"type": "enabled", "clear_thinking": False}})[2]
        self.assertEqual(kw, {"reasoning_effort": "low", "clear_thinking": False})
        self.assertEqual(openai_to_messages(base)[2], {})


class OneToolCall(Agent):
    script = TWO_CALLS

    def chat(self, **extra):
        _, _, raw = self.post("/v1/chat/completions", {"model": "m", "tools": CHAT_TOOLS, "messages": [
            {"role": "user", "content": "go"}], **extra})
        return json.loads(raw)["choices"][0]

    def test_parallel_tool_calls_false_keeps_the_first_call(self):
        both = self.chat()
        self.assertEqual(len(both["message"]["tool_calls"]), 2)
        one = self.chat(parallel_tool_calls=False)
        self.assertEqual(one["finish_reason"], "tool_calls")
        self.assertEqual([json.loads(c["function"]["arguments"]) for c in one["message"]["tool_calls"]],
                         [{"cmd": "ls"}])
        # streamed: the same single call, and the stream ends as a tool-call turn
        _, _, raw = self.post("/v1/chat/completions", {"model": "m", "tools": CHAT_TOOLS, "stream": True,
                                                       "parallel_tool_calls": False,
                                                       "messages": [{"role": "user", "content": "go"}]})
        chunks = [d for _, d in self.sse(raw) if d not in (None, "[DONE]")]
        indexes = {tc["index"] for c in chunks for tc in c["choices"][0]["delta"].get("tool_calls") or []}
        self.assertEqual(indexes, {0})
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_disable_parallel_tool_use(self):
        body = {"model": "m", "max_tokens": 500, "tools": ANTHROPIC_TOOLS,
                "messages": [{"role": "user", "content": "go"}]}
        _, _, raw = self.post("/v1/messages", body)
        self.assertEqual([b["type"] for b in json.loads(raw)["content"]].count("tool_use"), 2)
        _, _, raw = self.post("/v1/messages", {**body, "tool_choice": {"type": "auto",
                                                                       "disable_parallel_tool_use": True}})
        msg = json.loads(raw)
        self.assertEqual(msg["stop_reason"], "tool_use")
        self.assertEqual([b["input"] for b in msg["content"] if b["type"] == "tool_use"], [{"cmd": "ls"}])

    def test_responses_parallel_tool_calls_false(self):
        body = {"model": "m", "input": "go", "tools": RESPONSES_TOOLS}
        _, _, raw = self.post("/v1/responses", body)
        self.assertEqual([o["type"] for o in json.loads(raw)["output"]].count("function_call"), 2)
        _, _, raw = self.post("/v1/responses", {**body, "parallel_tool_calls": False})
        out = json.loads(raw)
        self.assertEqual(out["status"], "completed")
        self.assertEqual([json.loads(o["arguments"]) for o in out["output"] if o["type"] == "function_call"],
                         [{"cmd": "ls"}])


if __name__ == "__main__":
    unittest.main()
