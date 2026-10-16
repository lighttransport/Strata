#!/usr/bin/env python3
"""A runnable few-layer Muse Spark Open GGUF cut out of the published release by HTTP range requests.

    python3 tools/muse_layer_slice.py --headers s1.gguf h2.gguf h3.gguf h4.gguf h5.gguf \
        --url https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-GGUF/resolve/main/UD-IQ1_S/Muse-Spark-Open-816B-A42B-UD-IQ1_S-{n:05d}-of-00005.gguf \
        --layers 0,1 --cache parts/ --output muse-2layer.gguf

The headers are the release's shard headers (tools/muse_gguf_census.py says how to fetch them; shard 1 whole). The
output keeps shard 1's metadata byte for byte except block_count and attention.sliding_window_pattern (the chosen
layers' entries) and drops the split.* keys; it holds token_embd, output_norm, output and the chosen layers' tensors
renumbered from 0. llama.cpp loads it as a smaller muse-spark model, which is what the reference check needs
(tools/muse_reference_check.py): the same block arithmetic on the same weights, without the 175 GB download. Layer 0
is a sliding-window (RoPE) layer and layer 1 a full-attention (NoPE) one. Downloads go to --cache in 64 MiB parts and
are resumed by size.
"""
import argparse
import concurrent.futures
import os
import pathlib
import re
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import BLOCK_GEOMETRY, GGUFFile  # noqa: E402

PART = 64 << 20
ARCH = "muse-spark"


def raw_kv(path):
    """[(key, type_id, raw value bytes)] of a GGUF's metadata, in file order."""
    out = []
    with open(path, "rb") as fh:
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", fh.read(24))
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def skip_value(t):
            if t == 8:
                (n,) = struct.unpack("<Q", fh.read(8))
                fh.seek(n, 1)
            elif t == 9:
                et, count = struct.unpack("<IQ", fh.read(12))
                if et == 8:
                    for _ in range(count):
                        skip_value(8)
                else:
                    fh.seek(sizes[et] * count, 1)
            else:
                fh.seek(sizes[t], 1)

        for _ in range(n_kv):
            (n,) = struct.unpack("<Q", fh.read(8))
            key = fh.read(n).decode()
            (t,) = struct.unpack("<I", fh.read(4))
            start = fh.tell()
            skip_value(t)
            end = fh.tell()
            fh.seek(start)
            out.append((key, t, fh.read(end - start)))
    return out


def gguf_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def nbytes(t):
    block, size = BLOCK_GEOMETRY[t.type_name]
    elements = 1
    for d in t.shape:
        elements *= d
    return elements // block * size


def fetch(url, start, length, path):
    if path.exists() and path.stat().st_size == length:
        return length
    tmp = path.with_name(path.name + ".tmp")     # not with_suffix: the parts of a tensor differ only in the suffix
    piece = path.with_name(path.name + ".piece")
    stalled = 0
    while stalled < 40:
        # curl, not urllib: the release redirects to a CDN, and urllib's retried requests came back as whole-file 200s
        # (the Range header ignored). Each attempt asks for exactly the missing bytes and appends only a reply of
        # at most that size; the CDN also closes some connections early.
        have = tmp.stat().st_size if tmp.exists() else 0
        if have >= length:
            break
        piece.unlink(missing_ok=True)
        r = subprocess.run(["curl", "-sfL", "--max-time", "600", "--speed-limit", "200000", "--speed-time", "20", "-r", f"{start + have}-{start + length - 1}", "-o", str(piece), url])
        got = piece.stat().st_size if piece.exists() else 0
        if 0 < got <= length - have:
            with open(tmp, "ab") as f:
                f.write(piece.read_bytes())
            stalled = 0
        else:
            stalled += 1
            if got:
                print(f"\n  {path.name}: a {got}-byte reply to a {length - have}-byte range, discarded", file=sys.stderr)
        if r.returncode:
            time.sleep(min(30, 2 ** min(stalled, 5)))
    piece.unlink(missing_ok=True)
    if not tmp.exists() or tmp.stat().st_size != length:
        raise IOError(f"{path.name}: got {tmp.stat().st_size if tmp.exists() else 0} of {length} bytes")
    os.replace(tmp, path)
    return length


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headers", nargs="+", required=True, help="every shard's header, shard 1 first")
    ap.add_argument("--url", required=True, help="shard URL with {n} for the 1-based shard number")
    ap.add_argument("--layers", default="0,1")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",")]
    shards = [GGUFFile(pathlib.Path(p)) for p in args.headers]
    meta = shards[0].metadata
    if meta.get("general.architecture") != ARCH:
        raise SystemExit("not a muse-spark release")
    pattern = meta[f"{ARCH}.attention.sliding_window_pattern"]
    where = {t.name: (i, t) for i, s in enumerate(shards) for t in s.tensors}
    want = [("token_embd.weight", "token_embd.weight"), ("output_norm.weight", "output_norm.weight"),
            ("output.weight", "output.weight")]
    for new, old in enumerate(layers):
        for name in sorted(n for n in where if n.startswith(f"blk.{old}.")):
            want.append((re.sub(r"^blk\.\d+\.", f"blk.{new}.", name), name))
    cache = pathlib.Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    jobs = []
    for _, old in want:
        si, t = where[old]
        start, size = shards[si].data_start + t.offset, nbytes(t)
        for k, off in enumerate(range(0, size, PART)):
            jobs.append((args.url.format(n=si + 1), start + off, min(PART, size - off), cache / f"{old}.{k:04d}"))
    total = sum(j[2] for j in jobs)
    print(f"{len(want)} tensors, {total / 1e9:.2f} GB in {len(jobs)} parts", file=sys.stderr)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        for n in pool.map(lambda j: fetch(*j), jobs):
            done += n
            print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", file=sys.stderr)
    print(file=sys.stderr)

    kv = []
    for key, t, raw in raw_kv(args.headers[0]):
        if key.startswith("split."):
            continue
        if key == f"{ARCH}.block_count":
            raw = struct.pack("<I", len(layers)) if t == 4 else raw
        if key == f"{ARCH}.attention.sliding_window_pattern":
            raw = struct.pack("<IQ", 7, len(layers)) + bytes(1 if pattern[l] else 0 for l in layers)
        kv.append((key, t, raw))
    align = 32
    infos, offset = [], 0
    for new, old in want:
        _, t = where[old]
        infos.append((new, t, offset))
        offset += (nbytes(t) + align - 1) // align * align
    with open(args.output, "wb") as out:
        out.write(struct.pack("<IIQQ", 0x46554747, 3, len(infos), len(kv)))
        for key, t, raw in kv:
            out.write(gguf_str(key) + struct.pack("<I", t) + raw)
        for new, t, off in infos:
            out.write(gguf_str(new) + struct.pack("<I", len(t.shape)) + struct.pack(f"<{len(t.shape)}Q", *t.shape)
                      + struct.pack("<IQ", t.type_id, off))
        out.write(b"\0" * ((-out.tell()) % align))
        for (new, t, off), (_, old) in zip(infos, want):
            size = nbytes(t)
            for k in range((size + PART - 1) // PART):
                out.write((cache / f"{old}.{k:04d}").read_bytes())
            out.write(b"\0" * ((-size) % align))
    print(f"wrote {args.output}: {len(layers)} layers {layers}, {len(infos)} tensors", file=sys.stderr)


if __name__ == "__main__":
    main()
