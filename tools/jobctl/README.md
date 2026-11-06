# jobctl

Launch and manage many batch jobs on a cluster from your laptop. It is written for clusters where you cannot
ssh to the compute nodes, but it also works where you can.

What it does:

- **Many jobs.** Each job has a name. `jobctl ls` shows all of them. Up to 256 live jobs is the default cap
  (`max_live`), and a job can ask for up to 384 nodes (`max_nodes`). Both are config values.
- **Speculative launch.** Queues can take days for a big job. `launch --variants 384@12:00:00,192@12:00:00,96@12:00:00`
  queues all three. The first one to start wins and the others are cancelled (and any that start anyway leave
  at once).
- **Long runs.** One job is limited by the walltime. While a job runs, the next one is already queued
  (`queue_ahead`, default 1). When it starts, the old one stops its workload at a checkpoint and hands over.
  The workload resumes from `$CTL_WORK`.
- **Control without ssh to the node.** The run directory on the shared file system carries flags (`STOP`, `PAUSE`,
  `DRAIN`), an inbox for commands, an outbox for results and the job's status. Files are written to a temporary
  name and renamed, so a reader never sees half a file.
- **Optional HTTP bridge.** With `"tunnel": true` the job starts a small bash-over-HTTP server and a reverse
  ssh tunnel to the login node. `jobctl run NAME "cmd"` uses it when it is up and falls back to the inbox when
  it is not. The agent checks the tunnel every `interval` seconds (30 by default) and, if the port is taken,
  picks another. `jobctl watch` repoints the local port forward to follow it.

## Running next to interactive bridges

Interactive allocations (a pinned `slot.env` with `LOCAL_PORT` / `FRONTEND_PORT`) and jobctl jobs use the same
login node, so their reverse-tunnel ports must never meet. Three rules keep them apart:

1. **Separate ranges.** jobctl only hands out ports from `local_ports` and `frontend_ports` (default 43000-43599 and
   33000-33599). `reserved_ports` (default 42000-42999 and 32000-32999) lists the ranges the interactive tools use.
   jobctl refuses to start if its own ranges overlap a reserved one. The node agent re-picks a busy
   frontend port only inside jobctl's range.
2. **Known slots are skipped.** List the `slot.env` files of your interactive bridges in `external_slots`
   (globs). Their ports are never handed out, and `jobctl ls --all` shows them.
3. **A check for the other side.** Before pinning a pair for an interactive launch, run
   `jobctl ports --check LOCAL FRONTEND`. It exits 1 if the pair is in jobctl's range or already registered.
   `jobctl ports` alone lists every owner and flags clashes.

The login node here has no `ss` or `netstat`, so listening ports are read from `/proc/net/tcp`.
Each side keeps its own ssh connection socket, session files and known-hosts file. The compute-node server port
(21264) is not shared either, because an interactive and a batch allocation never share a node.

## Parts

| File | Where it runs | Job |
|---|---|---|
| `jobctl.py` | laptop and login node | the command line, the registry, the chain loop |
| `agent.sh` | node 0 of each job | server, tunnel, heartbeat, inbox, flags, handoff |
| `job.sh` | the scheduler's script | starts `agent.sh` |
| `examples/counter_work.sh` | node 0 | a small resumable workload for tests |
| `webui.py`, `webui_page.html` | laptop | the local web board, JSON API, event feed and webhooks (opt-in) |
| `mcp_server.py` | laptop | MCP tools for agents, a thin client of the board (opt-in) |
| `examples/netperf.py` | login node | measures latency and bandwidth of the HTTP bridges |

Files are copied to `<rundir>/_bin/` under names that carry a hash of their content. A running script is never
overwritten.

When the bridge server has a `PUT /put` route, `netperf.py --put` checks raw 64 KiB and 1 MiB uploads and verifies each
stored file by SHA-256. The existing `/run` upload measurement remains the default.

## Config

`~/.config/jobctl/config.json` (or `$JOBCTL_CONFIG`). Keep host names and accounts here, not in the code.

```json
{
  "ssh": "login-alias",
  "scheduler": "pjm",
  "scheduler_opts": {"project": "my-group", "rscgrp": "small",
                     "node_spec": {"12": "2x3x2"}, "resource": {"freq": 2000}},
  "tunnel": true,
  "frontend_target": "login-host-as-seen-from-the-nodes",
  "bundle": "$HOME/path/to/dir-with-bash_http_server.py",
  "rundir": "$HOME/jobctl/runs"
}
```

Schedulers: `pjm` (the first target; the stub tests pass and a real-system test is under way, see the results file when it is done) and `slurm` (command lines and output parsing are unit
tested only; not tested on a real Slurm system). To add another, subclass `Scheduler` in `jobctl.py`: how to build
the submit command, how to read the job id, how to read the queue, how to cancel.

## Use

```
jobctl launch run1 --nodes 12 --time 12:00:00 --script my_work.sh
jobctl launch big  --variants 384@12:00:00,192@12:00:00 --script my_work.sh
jobctl chain-start        # once: runs the chain loop on the login node
jobctl watch &            # keeps local port forwards right
jobctl ls --refresh
jobctl run run1 "uptime"
jobctl pause run1 ; jobctl resume run1 ; jobctl drain run1 ; jobctl stop run1 ; jobctl kill run1
```

Workload exit codes: 0 finished (the chain stops), 75 not finished (continue in the next allocation), any other
value is a failure (the chain stops). The workload gets SIGTERM about five minutes before the walltime ends
(`CTL_GRACE`) and when a successor takes over, so it can save a checkpoint.

## Web board and API

An optional local board for people, and a JSON API with an event feed and webhooks for scripts and LLM agents. It
reads and changes jobs through the same `jobctl` code as the command line, and it replaces `jobctl watch` while it
runs (it refreshes every `interval` seconds and repoints the forwards).

```
python3 tools/jobctl/webui.py --open        # or: jobctl webui --open
```

It prints a URL with a token in the `#fragment`. The token lives in `~/.config/jobctl/webui_token` (mode 0600; change the
path with `webui.token_file`). The board shows each job's state, nodes, heartbeat age and chain position, buttons for
Stop, Pause, Resume, Drain, Resubmit and Kill, a command box, a launch form, a live event list and the ports. It warns
when jobs are live but no chain loop runs on the login node.
Active scheduler jobs from `pjstat` appear in a separate list, including jobs not launched through jobctl. Refresh
the scheduler list on demand; queued jobs can be canceled through a two-step confirmation. Canceling a jobctl-managed
allocation stops its whole chain.
The board can post job events to a Slack incoming webhook. Paste the webhook URL in the board, select events and
send a test message. The default events are `job.started` and `job.finished`; canceled, lost and stale-heartbeat
alerts are optional. The URL is stored in the private jobctl registry. By default the API redacts it; selecting
`Remember URL in this browser` opts into also loading a copy into this browser's local storage. For
scheduler jobs outside jobctl, the board detects queued-to-running transitions and jobs leaving the active list.

To create the Slack URL, create or select a Slack app, turn on **Incoming Webhooks**, choose **Add New Webhook to
Workspace**, select the destination channel and authorize. Paste the resulting URL into **Slack notifications**,
save it, then send a test message. Slack treats the URL as a secret; do not post it in a channel or commit it.

**Safety.** It listens on `127.0.0.1` only (another address needs `--host` and `--allow-host`, and prints a warning). Every
`/api/v1` call except `/spec` needs `Authorization: Bearer <token>`. The `Host` and `Origin` headers must be local.
Launch and kill take two steps: the first call returns the exact submit line (or the job ids to cancel) and a code that
works once for ten minutes; only the same request with that code submits or cancels. Set `"require_confirm": false`
to switch this off. Events and webhooks carry job names, states, node counts and job ids, never paths, hosts or
command output.

**API.** `GET /api/v1/spec` lists every route with its parameters. Examples (with `T` the token):

```
curl -H "Authorization: Bearer $T" localhost:8765/api/v1/jobs
curl -H "Authorization: Bearer $T" -d '{"name":"run1","nodes":24,"time":"72:00:00"}' localhost:8765/api/v1/jobs      # preview
curl -H "Authorization: Bearer $T" "localhost:8765/api/v1/events?since=0&wait=50"                                 # long poll
```

**Webhooks.** In `~/.config/jobctl/config.json`:

```json
"webui": {"port": 8765, "require_confirm": true,
          "webhooks": [{"name": "ops", "url": "https://example.invalid/hook", "secret": "shared-secret",
                        "events": ["job.started", "job.finished", "job.lost", "job.stale", "service.chain_down"]},
                       {"name": "chat", "url": "https://example.invalid/chat", "format": "text", "events": ["job.*"]}]}
```

Events: `job.launched`, `job.state`, `job.started`, `job.finished`, `job.lost`, `job.stopped`, `job.successor`,
`job.stale`, `job.flag`, `job.resubmit`, `service.chain_down`. `events` takes names or patterns like `job.*`. A JSON
body is signed with `X-Jobctl-Signature: sha256=<hmac of the body>` when a secret is set. `format: "text"` sends
`{"text": "..."}`. Each delivery is tried three times; `GET /api/v1/webhooks` shows the results (no URLs or secrets).

**Agents (MCP).** `mcp_server.py` speaks MCP over stdio and calls a running board, so launch and kill keep their two
steps: `preview_launch` returns the submit line, `confirm_launch` needs the code. Add it to an MCP client as
`{"command": "python3", "args": ["/path/to/tools/jobctl/mcp_server.py"]}`; it reads `JOBCTL_WEBUI_URL`
(default `http://127.0.0.1:8765`) and the token file.

Not measured: refresh time with 256 live jobs, and delivery to a real chat service (tests use a local receiver).

## Limits to know

- On the shared file system a file made by another node can stay invisible to `test -e` for minutes (measured: about
  2.5 minutes for a `STOP` flag). The agent lists directories instead, which is fresh. Flags still act on the next
  tick (30 s by default), and a command sent within seconds of a flag may run before the flag is seen.

- Only one command runs at a time per allocation through the inbox, and a command sent while the workload is
  running a parallel launch can collide with it. Many MPI launchers allow one launch at a time.
- A command text containing `exit` or `set -e` is refused. It would end the shared shell.
- The tool measures nothing about your workload. Tests with a fake scheduler cover the logic; see the test file.

## Tests

`python3 tools/jobctl/test_jobctl.py` and `python3 tools/jobctl/test_webui.py` need no cluster. The login node is the local shell and the scheduler
commands are stubs. It covers port allocation for 256 jobs, launch, racing variants, chain submission, handoff
between two agents, flags, the inbox and the HTTP path. `test_webui.py` covers the token, Host and Origin
checks, the two-step launch and kill, event detection, long poll, the event stream, signed and retried webhooks, and the MCP
tools.
