#!/usr/bin/env python3
"""Upload release files to a Hugging Face repo with verification and automatic retries.

    python3 tools/hf_release_upload.py --repo lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts \
        --manifest release/manifest-tr16.json --log release/upload-tr16.log \
        [--hf ~/.local/bin/hf] [--wait-marker FILE:STRING] [--max-attempts 0] [--nice 19]

The manifest is a JSON list of {"local": path, "remote": path_in_repo, "sha256": optional}. For every item the
script checks the repo tree first (LFS sha256 against the manifest value, or against a local hash computed once
and cached next to the file as <name>.sha256), uploads with `hf upload` when the remote copy is missing or
different, and verifies again. A failed or interrupted upload is retried with exponential backoff (60 s to
15 min) until it verifies or --max-attempts is reached (0 = unlimited). Progress goes to the log and to a
<log>.state.json file so a watcher can be restarted at any time; already verified items are skipped.
Hashing a 100 GB file takes minutes of one core; --nice lowers its priority. The script never deletes anything.
"""
import argparse, hashlib, json, os, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

API = 'https://huggingface.co/api/models/{repo}/tree/main/{folder}?recursive=false&expand=true'


def log(path, msg):
    line = time.strftime('%Y-%m-%dT%H:%M:%S%z') + ' ' + msg
    print(line, flush=True)
    with open(path, 'a') as f: f.write(line + '\n')


def remote_entries(repo, folder, token):
    url = API.format(repo=repo, folder=folder)
    req = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + token} if token else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return {e['path']: e for e in json.load(r)}


def remote_sha(repo, remote, token):
    folder = str(Path(remote).parent) if '/' in remote else ''
    folder = '' if folder == '.' else folder
    try:
        e = remote_entries(repo, folder, token).get(remote)
    except urllib.error.HTTPError as err:
        if err.code == 404: return None, 'absent (folder missing)'
        return None, f'tree query failed: {err}'
    except Exception as err:  # noqa: BLE001 - network errors are retried by the caller
        return None, f'tree query failed: {err}'
    if not e: return None, 'absent'
    if e.get('lfs'): return e['lfs'].get('sha256') or e['lfs'].get('oid'), 'lfs'  # the tree API reports the LFS sha256 as `oid`
    return e.get('oid'), 'git-blob'


def local_sha(path, nice):
    cache = Path(str(path) + '.sha256')
    st = path.stat()
    if cache.exists():
        try:
            c = json.loads(cache.read_text())
            if c.get('size') == st.st_size and c.get('mtime_ns') == st.st_mtime_ns: return c['sha256']
        except (ValueError, KeyError): pass
    os.nice(nice) if nice and os.nice(0) < nice else None
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(64 << 20), b''): h.update(block)
    cache.write_text(json.dumps(dict(sha256=h.hexdigest(), size=st.st_size, mtime_ns=st.st_mtime_ns)) + '\n')
    return h.hexdigest()


def io_progress(pid):
    """Bytes written by the process (sockets included), from /proc; None when gone."""
    try:
        for line in Path(f'/proc/{pid}/io').read_text().splitlines():
            if line.startswith('wchar:'): return int(line.split()[1])
    except OSError: return None
    return None


def run_upload(args, cmd, env, log_fn):
    """Run one `hf upload`; kill it when its write counter stops advancing for --stall-seconds."""
    out = open(str(args.log) + '.hf.log', 'a')
    try:
        p = subprocess.Popen(['nice', '-n', str(args.nice), *cmd], stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    except OSError as err:
        log_fn(args.log, f'hf upload could not start: {err}'); out.close(); return
    start = time.time(); last_change = start; last = io_progress(p.pid) or 0; window = [(start, last)]
    while p.poll() is None:
        time.sleep(30)
        now = io_progress(p.pid)
        if now is not None and now - last >= args.stall_bytes * 30: last, last_change = now, time.time()   # real progress: above the floor rate
        elif now is not None: last = now
        if time.time() - last_change > args.stall_seconds:
            log_fn(args.log, f'STALL: write rate below {args.stall_bytes / 1e6:.1f} MB/s for {args.stall_seconds}s ({last / 1e9:.2f} GB written); killing attempt')
            os.killpg(p.pid, 15); time.sleep(10)
            if p.poll() is None: os.killpg(p.pid, 9)
            p.wait(); out.close(); return
        if time.time() - start > args.timeout:
            log_fn(args.log, f'hf upload timed out after {args.timeout}s'); os.killpg(p.pid, 9); p.wait(); out.close(); return
    out.close()
    tail = Path(str(args.log) + '.hf.log').read_text().strip().splitlines()[-2:]
    log_fn(args.log, f'hf upload exit {p.returncode} after {time.time() - start:.0f}s ({last / 1e9:.2f} GB written): ' + ' | '.join(t[:160] for t in tail))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stall-seconds', type=int, default=600, help='kill an upload attempt whose write rate stays below --stall-bytes per second this long')
    ap.add_argument('--stall-bytes', type=float, default=500_000, help='bytes per second counted as progress (default 0.5 MB/s)')
    ap.add_argument('--alternate-lfs', action='store_true', help='retry every other attempt over classic LFS (loses xet resume)')
    ap.add_argument('--repo', required=True); ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--log', type=Path, required=True); ap.add_argument('--hf', default='hf')
    ap.add_argument('--wait-marker', help='FILE:STRING; wait until FILE contains STRING before uploading')
    ap.add_argument('--max-attempts', type=int, default=0); ap.add_argument('--nice', type=int, default=19)
    ap.add_argument('--timeout', type=int, default=6 * 3600, help='seconds per upload attempt')
    args = ap.parse_args()
    token = None
    for p in (Path.home() / '.cache/huggingface/token',):
        if p.exists(): token = p.read_text().strip()
    items = json.loads(args.manifest.read_text())
    state_path = Path(str(args.log) + '.state.json')
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if args.wait_marker:
        file, string = args.wait_marker.split(':', 1)
        log(args.log, f'waiting for {string!r} in {file}')
        while not (Path(file).exists() and string in Path(file).read_text()): time.sleep(30)
        log(args.log, 'marker found; starting')
    for it in items:
        local = Path(it['local']).expanduser(); remote = it['remote']
        if not local.is_file(): log(args.log, f'MISSING local file {local}'); state[remote] = dict(status='missing'); continue
        want = it.get('sha256')
        if not want:
            log(args.log, f'hashing {local} ({local.stat().st_size / 1e9:.1f} GB)'); want = local_sha(local, args.nice)
        state[remote] = dict(status='pending', local=str(local), sha256=want, size=local.stat().st_size, attempts=0)
        state_path.write_text(json.dumps(state, indent=1))
        attempt, delay = 0, 60
        while True:
            have, how = remote_sha(args.repo, remote, token)
            if have == want:
                log(args.log, f'VERIFIED {remote} ({how}) sha256={want[:16]}'); state[remote]['status'] = 'verified'; break
            if local.stat().st_size < 1 << 20 and how == 'git-blob' and have:
                # small text files are stored as git blobs; compare content instead of sha256
                try:
                    req = urllib.request.Request(f'https://huggingface.co/{args.repo}/raw/main/{remote}', headers={'Authorization': 'Bearer ' + token} if token else {})
                    with urllib.request.urlopen(req, timeout=60) as r: same = r.read() == local.read_bytes()
                except Exception: same = False  # noqa: BLE001
                if same: log(args.log, f'VERIFIED {remote} (git blob content)'); state[remote]['status'] = 'verified'; break
            attempt += 1; state[remote]['attempts'] = attempt; state[remote]['last_remote'] = how
            state_path.write_text(json.dumps(state, indent=1))
            if args.max_attempts and attempt > args.max_attempts:
                log(args.log, f'GAVE UP {remote} after {attempt - 1} attempts'); state[remote]['status'] = 'failed'; break
            transport = 'xet' if (attempt % 2 or not args.alternate_lfs) else 'lfs'  # xet resumes uploaded chunks; LFS only on request
            log(args.log, f'UPLOAD attempt {attempt} ({transport}): {local} -> {remote} (remote: {how})')
            cmd = [args.hf, 'upload', args.repo, str(local), remote, '--repo-type', 'model',
                   '--commit-message', f'Add {remote}']
            env = dict(os.environ); env['HF_HUB_DISABLE_XET'] = '0' if transport == 'xet' else '1'
            if transport == 'xet': env['HF_XET_HIGH_PERFORMANCE'] = '1'
            run_upload(args, cmd, env, log)
            time.sleep(10)
            have, how = remote_sha(args.repo, remote, token)
            if have == want: continue  # verified on the next loop iteration
            log(args.log, f'not verified yet (remote {how}); retry in {delay}s'); time.sleep(delay); delay = min(delay * 2, 900)
        state_path.write_text(json.dumps(state, indent=1))
    done = all(v.get('status') == 'verified' for v in state.values())
    log(args.log, 'ALL VERIFIED' if done else 'INCOMPLETE: ' + ', '.join(k for k, v in state.items() if v.get('status') != 'verified'))
    return 0 if done else 1


if __name__ == '__main__':
    sys.exit(main())
