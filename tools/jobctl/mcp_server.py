#!/usr/bin/env python3
"""MCP server (stdio) for the jobctl web board. Standard library only.

It does not talk to the cluster itself: every tool calls the JSON API of a running board (python3 webui.py), so
the token check and the two-step confirm for launch and kill apply to agents exactly as they do to people.

Settings (environment):
  JOBCTL_WEBUI_URL         default http://127.0.0.1:8765
  JOBCTL_WEBUI_TOKEN       the bearer token, or
  JOBCTL_WEBUI_TOKEN_FILE  default ~/.config/jobctl/webui_token

Example MCP client entry:  {"command": "python3", "args": ["/path/to/tools/jobctl/mcp_server.py"]}
"""
import json
import os
import sys
import urllib.error
import urllib.request

PROTOCOL = "2024-11-05"


def base():
    return os.environ.get("JOBCTL_WEBUI_URL", "http://127.0.0.1:8765").rstrip("/") + "/api/v1"


def token():
    t = os.environ.get("JOBCTL_WEBUI_TOKEN")
    if t:
        return t.strip()
    path = os.path.expanduser(os.environ.get("JOBCTL_WEBUI_TOKEN_FILE", "~/.config/jobctl/webui_token"))
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def api(method, path, body=None, timeout=700):
    req = urllib.request.Request(base() + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": "Bearer " + token(), "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except ValueError:
            return e.code, {"error": "HTTP %d" % e.code}


S = lambda desc: {"type": "string", "description": desc}      # noqa: E731
I = lambda desc: {"type": "integer", "description": desc}     # noqa: E731
NAME = {"name": S("job name")}
LAUNCH = {"name": S("job name: letters, digits, . _ - (max 40)"), "nodes": I("node count"), "time": S("elapse HH:MM:SS"),
          "variants": S("optional speculative launch 'NODES@HH:MM:SS,...': all are queued, the first to start wins"),
          "script": S("optional workload script text; it must resume from $CTL_WORK (exit 0 done, 75 continue in the next allocation)"),
          "env": {"type": "object", "description": "optional environment, NAME: value", "additionalProperties": {"type": "string"}}}


def tool(name, desc, props, required, method, path, fixed=None, query=()):
    return {"name": name, "description": desc, "inputSchema": {"type": "object", "properties": props, "required": required},
            "_call": (method, path, fixed or {}, query)}


TOOLS = [
    tool("list_jobs", "List all jobs: state, nodes, elapse, heartbeat age, chain sequence, job ids, tunnel status.", {}, [], "GET", "/jobs"),
    tool("get_job", "One job with its flags, chain job ids and latest heartbeat info.", NAME, ["name"], "GET", "/jobs/{name}"),
    tool("job_events", "Events (state changes, successors queued, stale heartbeat, chain loop down). Pass the last_id you saw as since; wait holds the call up to 55 s for a new one.",
         {"since": I("return events with id greater than this (0 = recent history)"), "wait": I("seconds to wait for a new event, 0..55")}, [],
         "GET", "/events", query=("since", "wait")),
    tool("preview_launch", "Step 1 of a launch. Submits nothing. Returns the exact submit lines and a one-shot confirm code. Show the lines to the user and get their approval before confirm_launch.",
         LAUNCH, ["name"], "POST", "/jobs"),
    tool("confirm_launch", "Step 2 of a launch: repeat the same arguments as preview_launch plus the confirm code. This submits real cluster jobs; only call it after the user approved the submit lines.",
         dict(LAUNCH, confirm=S("code from preview_launch")), ["name", "confirm"], "POST", "/jobs"),
    tool("stop_job", "Set STOP: the job ends after its current step.", NAME, ["name"], "POST", "/jobs/{name}/stop"),
    tool("pause_job", "Set PAUSE: hold the command inbox.", NAME, ["name"], "POST", "/jobs/{name}/pause"),
    tool("resume_job", "Clear PAUSE.", NAME, ["name"], "POST", "/jobs/{name}/resume"),
    tool("drain_job", "Set DRAIN: finish the current command, then hand over to the next allocation.", NAME, ["name"], "POST", "/jobs/{name}/drain"),
    tool("resubmit_job", "Clear STOP and queue the next allocation.", NAME, ["name"], "POST", "/jobs/{name}/resubmit"),
    tool("preview_kill", "Step 1 of a kill. Cancels nothing. Returns the job ids that would be cancelled and a confirm code.", NAME, ["name"], "POST", "/jobs/{name}/kill"),
    tool("confirm_kill", "Step 2 of a kill: cancels every allocation of the chain. Only after the user approved.",
         dict(NAME, confirm=S("code from preview_kill")), ["name", "confirm"], "POST", "/jobs/{name}/kill"),
    tool("run_command", "Run a shell command on node 0 of a running job. Do not use exit or set -e. Returns the output and the exit code (124 = timed out).",
         dict(NAME, command=S("shell command"), timeout=I("seconds, 1..600 (default 50)")), ["name", "command"], "POST", "/jobs/{name}/run"),
    tool("ports", "Ports used by jobs and interactive slots, with conflicts.", {}, [], "GET", "/ports"),
    tool("services", "ssh status, chain-loop status, refresh age.", {}, [], "GET", "/services"),
    tool("start_chain_loop", "Start the chain loop on the login node (queues successors for running jobs).", {}, [], "POST", "/services/chain-start"),
]
BY_NAME = dict((t["name"], t) for t in TOOLS)


def call_tool(name, args):
    t = BY_NAME.get(name)
    if t is None:
        return True, {"error": "unknown tool: %s" % name}
    method, path, fixed, query = t["_call"]
    args = dict(args or {})
    for req in t["inputSchema"]["required"]:
        if req not in args:
            return True, {"error": "missing argument: %s" % req}
    path = path.replace("{name}", str(args.pop("name", ""))) if "{name}" in path else path
    if method == "GET" and query:
        q = "&".join("%s=%s" % (k, int(args[k])) for k in query if k in args)
        path += "?" + q if q else ""
        body = None
    elif method == "GET":
        body = None
    else:
        body = args
    try:
        status, data = api(method, path, body)
    except Exception as e:
        return True, {"error": "cannot reach the board at %s: %s" % (base(), e)}
    return status >= 400 or data.get("ok") is False and "needs_confirm" not in data, dict(data, http_status=status)


def handle(msg):
    """One JSON-RPC message in, the reply out (None for notifications)."""
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if method is None or (mid is None and method.startswith("notifications/")):
        return None
    if method == "initialize":
        result = {"protocolVersion": params.get("protocolVersion") or PROTOCOL, "capabilities": {"tools": {}},
                  "serverInfo": {"name": "jobctl", "version": "1"}}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": [dict((k, v) for k, v in t.items() if k != "_call") for t in TOOLS]}
    elif method == "tools/call":
        is_err, data = call_tool(params.get("name"), params.get("arguments"))
        result = {"content": [{"type": "text", "text": json.dumps(data, indent=1, sort_keys=True)}], "isError": bool(is_err)}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "method not found: %s" % method}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            reply = handle(json.loads(line))
        except ValueError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
