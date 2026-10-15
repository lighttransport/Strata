"""Shared pieces of the coding-evaluation suite (docs/GLM53F_EVAL.md).

- chat(): one greedy request to an OpenAI-compatible /v1/chat/completions endpoint (our serve.server or any other),
  returning the answer, the reasoning text, token usage and wall time.
- served(): a context manager that starts `python -m serve.server --engine glm --config <profile>` (with
  $STRATA_SERVE_PYTHON, else build-exl3-venv, else this interpreter) and waits until /health reports the model loaded,
  or uses an endpoint that is already running.
- write_results(): the results.json format read by tools/glm_eval_compare.py.
"""
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _post(url, body, timeout, api_key=None):
    data = json.dumps(body).encode()
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(base_url, messages, max_tokens=4096, reasoning_effort='low', timeout=3600, model='glm', api_key=None,
         retries=2):
    """Greedy chat completion. Returns dict(content, reasoning, prompt_tokens, completion_tokens, finish, seconds)."""
    body = {'model': model, 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0, 'stream': False}
    if reasoning_effort:
        body['reasoning_effort'] = reasoning_effort
    url = base_url.rstrip('/') + '/v1/chat/completions'
    for attempt in range(retries + 1):
        t0 = time.monotonic()
        try:
            r = _post(url, body, timeout, api_key)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries:
                raise RuntimeError(f'chat request failed after {retries + 1} attempts: {e}') from e
            time.sleep(10 * (attempt + 1))
    choice = r['choices'][0]
    msg = choice.get('message') or {}
    usage = r.get('usage') or {}
    return dict(content=msg.get('content') or '', reasoning=msg.get('reasoning_content') or '',
                prompt_tokens=usage.get('prompt_tokens'), completion_tokens=usage.get('completion_tokens'),
                finish=choice.get('finish_reason'), seconds=round(time.monotonic() - t0, 3))


def health(base_url, timeout=5):
    try:
        with urllib.request.urlopen(base_url.rstrip('/') + '/health', timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
        return None


@contextlib.contextmanager
def served(config=None, base_url=None, port=8095, log=None, ready_timeout=3600, extra_env=None):
    """Yield (base_url, info). With config: start serve.server for it and stop it afterwards."""
    if base_url:
        if not health(base_url):
            raise SystemExit(f'no server answering at {base_url}/health')
        yield base_url, dict(server='external', base_url=base_url)
        return
    if not config:
        raise SystemExit('give --config (a GLM serve profile) or --base-url')
    url = f'http://127.0.0.1:{port}'
    if health(url):
        raise SystemExit(f'port {port} is already in use by another server; stop it or pick --port')
    env = dict(os.environ, **(extra_env or {}))
    python = os.environ.get('STRATA_SERVE_PYTHON') or next(
        (str(p) for p in (REPO / 'build-exl3-venv/bin/python',) if p.exists()), sys.executable)
    cmd = [python, '-m', 'serve.server', '--engine', 'glm', '--config', str(config), '--host', '127.0.0.1',
           '--port', str(port), '--fit-max-tokens']    # clamp the output cap to the context instead of a 400
    out = open(log, 'ab') if log else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + ready_timeout
        while True:
            if proc.poll() is not None:
                raise SystemExit(f'serve.server exited with {proc.returncode} before it was ready (log: {log})')
            h = health(url)
            if h and h.get('loaded', True):
                break
            if time.monotonic() > deadline:
                raise SystemExit(f'serve.server not ready after {ready_timeout} s (log: {log})')
            time.sleep(5)
        yield url, dict(server='serve.server', config=str(config), config_sha256=sha256_file(config), port=port)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        if log:
            out.close()


def git_head():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def write_results(path, suite, profile, tasks, meta):
    """tasks: list of dict(id, passed, metrics). Written atomically."""
    ids = [t['id'] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate task ids')
    path = Path(path)
    data = dict(suite=suite, profile=profile, created=time.strftime('%Y-%m-%dT%H:%M:%S'), strata_commit=git_head(),
                passed=sum(1 for t in tasks if t['passed']), total=len(tasks), meta=meta, tasks=tasks)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=1) + '\n')
    tmp.replace(path)
    return data
