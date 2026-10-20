#!/usr/bin/env python3
"""Fetch a bounded expert fixture from two layers, using local shards or strict HTTP byte ranges."""
import argparse
import hashlib
import json
import pathlib
import struct
import time
import urllib.request

REPO = 'nvidia/GLM-5.3-Flash-FP4-Local'
REVISION = '76348ddb2be2e35e39ecfc00e75dc07703f4f3d5'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=pathlib.Path, required=True, help='In-progress checkpoint directory (read only)')
p.add_argument('--output', type=pathlib.Path, required=True)
p.add_argument('--experts', type=int, choices=(1,2,3,8), default=2)
p.add_argument('--local-only', action='store_true', help='Require local shards; never request HTTP ranges')
p.add_argument('--reuse', type=pathlib.Path, help='Previously fetched fixture from this pinned revision')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
if any(a.output.iterdir()):
    raise SystemExit('Output directory must be empty')
manifest = a.source / 'download-manifest.json'
if manifest.exists() and json.loads(manifest.read_text()).get('sha') != REVISION:
    raise SystemExit('Source checkpoint revision differs from the pinned fixture revision')
index = json.loads((a.source / 'model.safetensors.index.json').read_text())['weight_map']
prefixes = [f'model.language_model.layers.{layer}.mlp.experts.{expert}.'
            for layer in (3, 4) for expert in range(a.experts)]
names = sorted(n for n in index if any(n.startswith(prefix) for prefix in prefixes)
               and ('.weight' in n))
if len(names) != 42*a.experts:
    raise SystemExit(f'Expected {42*a.experts} weight/scale tensors, got {len(names)}')

def read_range(shard, start, count):
    if count < 1 or count > 16 * 2**20:
        raise ValueError('Range outside bounded read budget')
    local = a.source / shard
    if local.is_file():
        with local.open('rb') as f:
            f.seek(start)
            data = f.read(count)
        if len(data) != count:
            raise ValueError('Truncated local shard')
        return data
    if a.local_only:
        raise ValueError('Local-only extraction requires every source shard')
    end = start + count - 1
    url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{shard}?slice={start}-{end}'
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
            with urllib.request.urlopen(req, timeout=90) as response:
                if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                    raise ValueError('Server did not honor exact range')
                data = response.read(count + 1)
            if len(data) != count:
                raise ValueError('Range length mismatch')
            return data
        except (OSError, ValueError):
            if attempt == 4:
                raise
            time.sleep(attempt + 1)

reuse_header, reuse_records, reuse_base = {}, {}, 0
if a.reuse:
    provenance=json.loads((a.reuse/'source-ranges.json').read_text())
    if provenance.get('revision') != REVISION:
        raise SystemExit('Reuse fixture revision mismatch')
    reuse_records={x['name']:x for x in provenance['tensors']}
    with (a.reuse/'experts.safetensors').open('rb') as f:
        hsize=struct.unpack('<Q',f.read(8))[0]
        if hsize>16*2**20: raise ValueError('Oversized reuse header')
        reuse_header=json.loads(f.read(hsize));reuse_base=8+hsize
headers = {}
for shard in sorted({index[n] for n in names}):
    size = struct.unpack('<Q', read_range(shard, 0, 8))[0]
    headers[shard] = (8 + size, json.loads(read_range(shard, 8, size)))
entries = []
for name in names:
    shard = index[name]
    base, header = headers[shard]
    tensor = header[name]
    start, end = tensor['data_offsets']
    entries.append((name, shard, base + start, end - start, tensor))
if sum(e[3] for e in entries) > a.experts * 32 * 2**20:
    raise SystemExit('Fixture exceeds 32 MiB per expert budget')
out_header, offset = {}, 0
for name, _, _, size, tensor in entries:
    out_header[name] = dict(dtype=tensor['dtype'], shape=tensor['shape'], data_offsets=[offset, offset+size])
    offset += size
encoded = json.dumps(out_header, separators=(',', ':')).encode()
encoded += b' ' * (-len(encoded) % 8)
records = []
part = a.output / 'experts.safetensors.part'
with part.open('wb') as f:
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)
    for name, shard, start, size, _ in entries:
        if name in reuse_header:
            old=reuse_header[name]
            if old['dtype']!=out_header[name]['dtype'] or old['shape']!=out_header[name]['shape'] or old['data_offsets'][1]-old['data_offsets'][0]!=size:
                raise ValueError('Reuse tensor metadata mismatch')
            with (a.reuse/'experts.safetensors').open('rb') as rf:
                rf.seek(reuse_base+old['data_offsets'][0]);data=rf.read(size)
            if len(data)!=size or hashlib.sha256(data).hexdigest()!=reuse_records[name]['sha256']:
                raise ValueError('Reuse tensor checksum mismatch')
        else:
            data = read_range(shard, start, size)
        f.write(data)
        records.append(dict(name=name, shard=shard, offset=start, bytes=size, sha256=hashlib.sha256(data).hexdigest()))
        print('FETCH', name, size, flush=True)
part.rename(a.output / 'experts.safetensors')
(a.output / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {'total_size': offset}, 'weight_map': {n: 'experts.safetensors' for n in names}}))
(a.output / 'source-ranges.json').write_text(json.dumps({'repo': REPO, 'revision': REVISION, 'tensors': records}, indent=2)+'\n')
print('COMPLETE', len(entries), offset, flush=True)
