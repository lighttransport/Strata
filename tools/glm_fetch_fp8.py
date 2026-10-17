#!/usr/bin/env python3
"""Download the official GLM-5.3-Flash FP8 release (zai-org/GLM-5.3-Flash, ~328 GB) with resume and verification.

- Resumes: huggingface_hub keeps partial files and continues them; the whole run is retried on errors, so killing
  and restarting the script continues where it stopped.
- Refuses to start when the target filesystem lacks the remaining bytes plus a margin.
- Verifies every file against the repository's LFS sha256 (or size for small non-LFS files) at the end and writes
  verify.json next to the files.

  build-exl3-venv/bin/python tools/glm_fetch_fp8.py --out models/glm53f/fp8 [--workers 4] [--verify-only]
"""
import argparse
import os

# Plain HTTP range downloads: the Xet backend stalled at one connection on this link (2026-10-16).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import hashlib
import json
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = 'zai-org/GLM-5.3-Flash'


def remote_files(api, revision):
    info = api.model_info(REPO, revision=revision, files_metadata=True)
    files = {}
    for s in info.siblings:
        lfs = s.lfs or {}
        sha = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
        files[s.rfilename] = dict(size=s.size or 0, sha256=sha)
    return info.sha, files


def sha256(path, chunk=16 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--revision', default='main')
    ap.add_argument('--margin-gb', type=float, default=10.0)
    ap.add_argument('--retries', type=int, default=50)
    ap.add_argument('--verify-only', action='store_true')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    commit, files = remote_files(api, a.revision)
    total = sum(f['size'] for f in files.values())
    have = sum((a.out / n).stat().st_size for n in files if (a.out / n).exists())
    need = total - have
    free = shutil.disk_usage(a.out).free
    print(f'{REPO}@{commit}: {len(files)} files, {total / 1e9:.1f} GB; present {have / 1e9:.1f} GB; '
          f'free {free / 1e9:.1f} GB', flush=True)
    if not a.verify_only:
        if free < need + a.margin_gb * 1e9:
            raise SystemExit(f'not enough space: need {need / 1e9:.1f} GB + {a.margin_gb:.0f} GB margin')
        for attempt in range(1, a.retries + 1):
            try:
                snapshot_download(REPO, revision=commit, local_dir=a.out, max_workers=a.workers)
                break
            except Exception as e:  # network errors: partial files stay and are resumed
                print(f'attempt {attempt} failed: {e!r}; retrying in 30 s', flush=True)
                time.sleep(30)
        else:
            raise SystemExit('download did not finish')
    bad, report = [], {}
    for name, meta in sorted(files.items()):
        p = a.out / name
        if not p.exists() or p.stat().st_size != meta['size']:
            bad.append(name)
            report[name] = 'missing or wrong size'
            continue
        if meta['sha256']:
            ok = sha256(p) == meta['sha256']
            report[name] = 'ok' if ok else 'sha256 mismatch'
            if not ok:
                bad.append(name)
        else:
            report[name] = 'ok (size)'
    (a.out / 'verify.json').write_text(json.dumps(dict(repo=REPO, commit=commit, files=report, bad=bad), indent=1) + '\n')
    print(f'verified {len(files) - len(bad)}/{len(files)} files' + (f'; bad: {bad[:5]}' if bad else ''))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
