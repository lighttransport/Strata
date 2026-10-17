#!/usr/bin/env python3
"""Fetch selected routed experts of one layer from the original BF16 checkpoint with HTTP range requests.

Reads each needed shard's safetensors header (8-byte length + JSON) and then only the byte ranges of the wanted
tensors; writes one local safetensors file plus a manifest with sha256 of every tensor. Used by the quantization
study (docs/GLM_QUANT_LEECH_GSQ_STUDY.md) as ground truth; no model is downloaded in full.

  python3 tools/quant_study/fetch_bf16_experts.py --layer 20 --experts 21,29,... --out DIR
"""
import argparse
import hashlib
import json
import struct
import time
import urllib.request
from pathlib import Path

REPO = 'zai-org/GLM-5.3-Flash-BF16'
BASE = f'https://huggingface.co/{REPO}/resolve/main/'


def set_repo(repo):
    global REPO, BASE
    REPO, BASE = repo, f'https://huggingface.co/{repo}/resolve/main/'


def get_range(url, start, end, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f'short read {len(data)} of {end - start + 1}')
            return data
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def header(shard):
    url = BASE + shard
    n = struct.unpack('<Q', get_range(url, 0, 7))[0]
    return json.loads(get_range(url, 8, 8 + n - 1)), 8 + n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--experts', required=True, help='comma-separated expert ids')
    ap.add_argument('--shared', action='store_true', help='also fetch the shared expert')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--repo', default=REPO, help='also zai-org/GLM-5.3-Flash (the FP8 release: adds weight_scale_inv)')
    ap.add_argument('--tag', default='bf16')
    a = ap.parse_args()
    set_repo(a.repo)
    a.out.mkdir(parents=True, exist_ok=True)
    index = json.loads(urllib.request.urlopen(BASE + 'model.safetensors.index.json', timeout=120).read())['weight_map']
    prefix = f'model.language_model.layers.{a.layer}.mlp.'
    names = [f'{prefix}experts.{e}.{p}_proj.{s}' for e in a.experts.split(',') for p in ('gate', 'up', 'down')
             for s in ('weight', 'weight_scale_inv') if f'{prefix}experts.{e}.{p}_proj.{s}' in index]
    if a.shared:
        names += [f'{prefix}shared_experts.{p}_proj.weight' for p in ('gate', 'up', 'down')]
    by_shard = {}
    for n in names:
        by_shard.setdefault(index[n], []).append(n)
    tensors, blobs = {}, []
    offset = 0
    for shard, wanted in sorted(by_shard.items()):
        h, data_start = header(shard)
        for n in wanted:
            meta = h[n]
            s, e = meta['data_offsets']
            data = get_range(BASE + shard, data_start + s, data_start + e - 1)
            tensors[n] = dict(dtype=meta['dtype'], shape=meta['shape'], data_offsets=[offset, offset + len(data)],
                              source=shard, sha256=hashlib.sha256(data).hexdigest())
            blobs.append(data)
            offset += len(data)
            print(f'{n} {meta["dtype"]} {meta["shape"]} from {shard}', flush=True)
    head = json.dumps({n: {k: t[k] for k in ('dtype', 'shape', 'data_offsets')} for n, t in tensors.items()}).encode()
    head += b' ' * (-len(head) % 8)
    path = a.out / f'{a.tag}-layer{a.layer}.safetensors'
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(head)))
        f.write(head)
        for b in blobs:
            f.write(b)
    (a.out / f'{a.tag}-layer{a.layer}.manifest.json').write_text(json.dumps(dict(repo=REPO, layer=a.layer, tensors=tensors),
                                                                           indent=1) + '\n')
    print(f'wrote {path} ({offset / 2**30:.2f} GiB)')


if __name__ == '__main__':
    main()
