#!/usr/bin/env python3
"""tools/strata_agent_config.py - settings for coding agents pointed at a Strata server: pi, oh-my-pi, Claude Code,
Codex, and the server's own config keys for agent use.

    python tools/strata_agent_config.py                      # every client, from the server on 127.0.0.1:8080
    python tools/strata_agent_config.py --client pi --write  # add Strata to ~/.pi/agent/models.json (a backup is kept)
    python tools/strata_agent_config.py --url http://127.0.0.1:8081 --client codex

The model name, context size and image support come from the running server (/health and /v1/models). When it is
not running, give --model and --context.  Only pi's models.json is written (JSON, merged by provider); the others
are printed, since their files hold the user's other settings.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

CLIENTS = ("pi", "omp", "claude-code", "codex", "server")


def server_facts(url: str, api_key: str | None = None, timeout: float = 3.0) -> dict | None:
    """The running server's model id, context size, image support and whether it wants a key; None if none answers."""
    def get(path):
        req = urllib.request.Request(url.rstrip("/") + path)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    try:
        health = get("/health")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if not isinstance(health, dict) or health.get("service") != "strata":
        return None
    facts = {"model": health.get("model") or "strata", "context": int(health.get("max_context") or 0),
             "images": bool(health.get("images")), "key_required": bool(health.get("api_key"))}
    try:                                                   # the context the model list reports, when it says one
        models = get("/v1/models").get("data") or []
        if models and models[0].get("context_length"):
            facts["context"] = int(models[0]["context_length"])
    except (urllib.error.URLError, OSError, ValueError, AttributeError):
        pass
    return facts


def max_output(context: int) -> int:
    """The output cap to give agents: a quarter of the context, at most 32K (Claude Code's own default for a model it
    does not know), at least 1K.  Strata answers a larger request with a 400 unless fit_max_tokens is set."""
    return max(1024, min(32768, context // 4))


def is_glm(model: str) -> bool:
    return "glm" in model.lower()


def pi_provider(facts: dict, base: str, key: str) -> dict:
    """pi's models.json provider (openai-completions).  GLM gets Z.ai's thinking format, which Strata reads (thinking
    {type, clear_thinking} plus reasoning_effort), so pi's "off" level turns thinking off; max_tokens is the field
    Strata's 400 names; store is not sent; pi's session id comes as x-session-affinity (slot affinity)."""
    compat = {"maxTokensField": "max_tokens", "supportsStore": False, "supportsReasoningEffort": True,
              "sendSessionAffinityHeaders": True}
    if is_glm(facts["model"]):
        compat["thinkingFormat"] = "zai"
    model = {"id": facts["model"], "name": f"{facts['model']} (Strata)", "reasoning": True,
             "input": ["text", "image"] if facts["images"] else ["text"],
             "contextWindow": facts["context"], "maxTokens": max_output(facts["context"]),
             "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}
    return {"baseUrl": f"{base}/v1", "api": "openai-completions", "apiKey": key, "compat": compat, "models": [model]}


def omp_yaml(facts: dict, base: str, key: str) -> str:
    """oh-my-pi's models.yml provider: the core fields only (its local-endpoint defaults suit Strata)."""
    lines = ["providers:", "  strata:", f"    baseUrl: {base}/v1", "    api: openai-completions",
             f"    apiKey: {json.dumps(key)}"]
    if key == "none":
        lines.append("    auth: none")
    lines += ["    models:", f"      - id: {json.dumps(facts['model'])}",
              f"        name: {json.dumps(facts['model'] + ' (Strata)')}", "        reasoning: true",
              "        input: [" + ", ".join(["text", "image"] if facts["images"] else ["text"]) + "]",
              f"        contextWindow: {facts['context']}", f"        maxTokens: {max_output(facts['context'])}"]
    return "\n".join(lines) + "\n"


def claude_code_env(facts: dict, base: str, key: str) -> dict:
    """Claude Code's environment: Strata's own model id for every role (titles and summaries too), the real context
    (it assumes 200K for a model it does not know) and output cap, and the request-class hint headers Strata reads to
    keep side requests out of the conversation's cache."""
    m = facts["model"]
    return {"ANTHROPIC_BASE_URL": base, "ANTHROPIC_AUTH_TOKEN": key, "ANTHROPIC_MODEL": m,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": m, "ANTHROPIC_DEFAULT_SONNET_MODEL": m, "ANTHROPIC_DEFAULT_HAIKU_MODEL": m,
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(facts["context"]),
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(max_output(facts["context"])),
            "CLAUDE_CODE_GATEWAY_HINT_HEADERS": "1"}


def codex_toml(facts: dict, base: str, key_required: bool) -> str:
    """Codex's ~/.codex/config.toml: the Responses wire API (Codex dropped "chat"), the real context with compaction
    at 80 % of it, and a stream idle timeout that outlasts a long prompt read."""
    lines = [f'model = "{facts["model"]}"', 'model_provider = "strata"',
             f'model_context_window = {facts["context"]}',
             f'model_auto_compact_token_limit = {facts["context"] * 4 // 5}', "",
             "[model_providers.strata]", 'name = "Strata"', f'base_url = "{base}/v1"', 'wire_api = "responses"',
             "stream_idle_timeout_ms = 600000"]
    if key_required:
        lines.append('env_key = "STRATA_API_KEY"')
    return "\n".join(lines) + "\n"


def server_keys(facts: dict) -> dict:
    """The strata-<model>.json keys for agent use.  GLM refuses sampling settings by default (a 400); agents send a
    temperature, so ignore it.  Codex's overflow then arrives in its stream, where Codex compacts."""
    keys = {"responses_overflow_event": True}
    if is_glm(facts["model"]):
        keys["glm_sampling"] = "ignore"
    return keys


def configs(facts: dict, base: str, key: str) -> dict:
    """Every client's settings, as {client: {"file": where it goes, "content" or "env": ...}}."""
    return {
        "pi": {"file": "~/.pi/agent/models.json",
               "content": {"providers": {"strata": pi_provider(facts, base, key)}}},
        "omp": {"file": "~/.omp/agent/models.yml", "content": omp_yaml(facts, base, key)},
        "claude-code": {"file": "environment (or ~/.claude/settings.json \"env\")",
                        "env": claude_code_env(facts, base, key)},
        "codex": {"file": "~/.codex/config.toml", "content": codex_toml(facts, base, facts["key_required"])},
        "server": {"file": "strata-<model>.json (the server's config; restart to apply)",
                   "content": server_keys(facts)},
    }


def write_pi(provider: dict, path: Path) -> Path | None:
    """Merge the provider into pi's models.json; the old file is kept as models.json.bak.  Returns the backup."""
    data, backup = {}, None
    if path.exists():
        data = json.loads(path.read_text() or "{}")
        backup = path.with_suffix(".json.bak")
        shutil.copy2(path, backup)
    data.setdefault("providers", {})["strata"] = provider
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return backup


def show(name: str, cfg: dict) -> str:
    head = f"# {name}: {cfg['file']}"
    if "env" in cfg:
        body = "\n".join(f"export {k}={json.dumps(v)}" for k, v in cfg["env"].items())
    elif isinstance(cfg["content"], str):
        body = cfg["content"].rstrip()
    else:
        body = json.dumps(cfg["content"], indent=2)
    return f"{head}\n{body}\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", default="http://127.0.0.1:8080", help="the Strata server (default: %(default)s)")
    p.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY"), help="its API key, if it has one")
    p.add_argument("--client", choices=CLIENTS + ("all",), default="all")
    p.add_argument("--model", help="the model id, when the server is not running")
    p.add_argument("--context", type=int, help="the context size, when the server is not running")
    p.add_argument("--write", action="store_true", help="with --client pi: merge into ~/.pi/agent/models.json")
    a = p.parse_args(argv)
    facts = server_facts(a.url, a.api_key)
    if facts is None:
        if not (a.model and a.context):
            print(f"no Strata server answers at {a.url}: start it, or give --model and --context", file=sys.stderr)
            return 1
        facts = {"model": a.model, "context": a.context, "images": False, "key_required": bool(a.api_key)}
    if a.model:
        facts["model"] = a.model
    if a.context:
        facts["context"] = a.context
    key = a.api_key or ("$STRATA_API_KEY" if facts["key_required"] else "none")
    all_cfgs = configs(facts, a.url.rstrip("/"), key)
    if a.write:
        if a.client != "pi":
            print("--write is for --client pi only; the other files are printed", file=sys.stderr)
            return 2
        path = Path.home() / ".pi/agent/models.json"
        backup = write_pi(all_cfgs["pi"]["content"]["providers"]["strata"], path)
        print(f"wrote the \"strata\" provider to {path}" + (f" (the old file: {backup})" if backup else ""))
        return 0
    for name in CLIENTS if a.client == "all" else (a.client,):
        print(show(name, all_cfgs[name]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
