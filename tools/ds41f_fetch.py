#!/usr/bin/env python3
"""Fetch the DeepSeek V4.1 Flash (1017) checkpoint shards that quantization needs, resumably, skipping the Engram tables.

The release is 48 safetensors shards (510 GB). Two of them hold only the Engram tables (layers 1 and 14, ~98 GB each
plus scales), which the pack tool never reads; this script peeks at every shard's header over HTTP, groups shards by
the layers they hold, and downloads only what `--layers` needs (default: every non-Engram shard, ~307 GB). Downloads
go through huggingface_hub with plain HTTP range resume (the Xet backend stalled on this site), one retry loop per
shard, and are verified by size and by the header SHA-256 recorded in `fetch-manifest.json`.

    build-exl3-venv/bin/python tools/ds41f_fetch.py --out /path/ckpt [--repo deepseek-ai/DeepSeek-V4.1-Flash-1017]
        [--layers 0-39] [--workers 4] [--plan-only] [--delete-after CHUNK_DIR]

--plan-only prints the shard -> layers table and the bytes to fetch without downloading. --delete-after removes a
shard once every layer it holds has its three *.chunk.json in CHUNK_DIR (the streaming mode for a space-limited node).
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

EXPERT = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.")
LAYER = re.compile(r"^layers\.(\d+)\.")
ENGRAM = re.compile(r"^layers\.(\d+)\.engram\.embed\.")


def parse_layers(text):
    if "-" in text:
        a, b = text.split("-")
        return set(range(int(a), int(b) + 1))
    return {int(text)}


def hf():
    try:
        from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
    except ImportError:
        sys.exit("huggingface_hub is not importable; use build-exl3-venv/bin/python or pip install huggingface_hub")
    return HfApi(), hf_hub_download, hf_hub_url


def peek_header(url, token=None):
    """The safetensors header (JSON) of a remote shard via two HTTP range requests; returns (header_dict, header_sha256)."""
    headers = {"Range": "bytes=0-7"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        n = struct.unpack("<Q", r.read(8))[0]
    headers["Range"] = f"bytes=8-{8 + n - 1}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
        raw = r.read(n)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def classify(header):
    """Layers whose routed experts live in this shard, layers with other (trunk) tensors, and whether it is Engram-only."""
    expert_layers, other_layers, engram, other = set(), set(), False, False
    for name in header:
        if name == "__metadata__":
            continue
        if ENGRAM.match(name):
            engram = True
            continue
        m = EXPERT.match(name)
        if m:
            expert_layers.add(int(m.group(1)))
            continue
        other = True
        m = LAYER.match(name)
        if m:
            other_layers.add(int(m.group(1)))
    return expert_layers, other_layers, engram and not other and not expert_layers


def local_header_sha(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return hashlib.sha256(f.read(n)).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", default="deepseek-ai/DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--layers", default="0-39", help="layers whose expert shards to fetch (trunk shards are always fetched)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=100)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--include-engram", action="store_true", help="also fetch the Engram-only shards (the runner needs them; the pack tool does not)")
    ap.add_argument("--delete-after", type=Path, help="chunk directory: delete a shard once all its layers have their chunk jsons")
    a = ap.parse_args()
    api, hf_hub_download, hf_hub_url = hf()
    token = os.environ.get("HF_TOKEN")
    want = parse_layers(a.layers)
    a.out.mkdir(parents=True, exist_ok=True)
    manifest_path = a.out / "fetch-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("repo") != a.repo or "shards" not in manifest:
        info = api.model_info(a.repo, revision=a.revision, files_metadata=True)
        shards = {}
        for s in info.siblings:
            if not s.rfilename.endswith(".safetensors"):
                continue
            header, sha = peek_header(hf_hub_url(a.repo, s.rfilename, revision=info.sha), token)
            expert_layers, other_layers, engram_only = classify(header)
            lfs = s.lfs or {}
            shards[s.rfilename] = dict(size=s.size or 0, header_sha256=sha, expert_layers=sorted(expert_layers), other_layers=sorted(other_layers),
                                       engram_only=engram_only, lfs_sha256=(lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)))
            print(f"{s.rfilename}: {(s.size or 0) / 1e9:6.1f} GB experts {sorted(expert_layers)} trunk-layers {len(other_layers)}{' ENGRAM-ONLY' if engram_only else ''}", flush=True)
        manifest = dict(repo=a.repo, commit=info.sha, shards=shards)
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    shards = manifest["shards"]
    selected = []
    for name, s in sorted(shards.items()):
        if s["engram_only"] and not a.include_engram:
            continue
        if s["expert_layers"] and not (set(s["expert_layers"]) & want) and not s["other_layers"]:
            continue
        selected.append(name)
    total = sum(shards[n]["size"] for n in selected)
    have = sum((a.out / n).stat().st_size for n in selected if (a.out / n).exists())
    print(f"{a.repo}@{manifest['commit']}: {len(selected)} of {len(shards)} shards selected, {total / 1e9:.1f} GB, present {have / 1e9:.1f} GB, "
          f"engram-only shards skipped: {sum(1 for s in shards.values() if s['engram_only'])} ({sum(s['size'] for s in shards.values() if s['engram_only']) / 1e9:.0f} GB)", flush=True)
    if a.plan_only:
        return 0
    for name in selected:
        s = shards[name]
        dst = a.out / name
        if dst.exists() and dst.stat().st_size == s["size"] and local_header_sha(dst) == s["header_sha256"]:
            continue
        for attempt in range(1, a.retries + 1):
            try:
                hf_hub_download(a.repo, name, revision=manifest["commit"], local_dir=a.out, force_download=False)
                if dst.stat().st_size != s["size"] or local_header_sha(dst) != s["header_sha256"]:
                    raise RuntimeError(f"{name}: size or header mismatch after download")
                print(f"fetched {name} ({s['size'] / 1e9:.1f} GB)", flush=True)
                break
            except Exception as e:  # network errors: the partial file stays and resumes
                print(f"{name}: attempt {attempt} failed: {e!r}; retrying in 30 s", flush=True)
                time.sleep(30)
        else:
            sys.exit(f"{name}: download did not finish")
        if a.delete_after:
            for other in selected:
                so = shards[other]
                if not so["expert_layers"] or so["other_layers"] or not (a.out / other).exists():
                    continue
                done = all((a.delete_after / f"blk.{l}.ffn_{p}_exps.weight.chunk.json").exists() for l in so["expert_layers"] for p in ("gate", "up", "down"))
                if done:
                    (a.out / other).unlink()
                    print(f"deleted {other}: its layers {so['expert_layers']} are packed", flush=True)
    print("fetch complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
