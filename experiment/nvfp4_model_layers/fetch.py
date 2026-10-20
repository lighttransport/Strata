#!/usr/bin/env python3
"""Fetch only two dense MLPs via bounded HTTP ranges; never download shards."""
import argparse, hashlib, json, math, pathlib, struct, urllib.request

REPO = 'nvidia/GLM-5.3-Flash-NVFP4'
REVISION = 'da920bb0b9f4a06727223a349e55468e38352348'
BASE = f'https://huggingface.co/{REPO}/resolve/{REVISION}/'
LIMIT = 256 * 1024 * 1024

def get(name, start=None, count=None):
    headers = {'Accept-Encoding': 'identity'}
    url = BASE + name
    if start is not None:
        end = start + count - 1
        headers['Range'] = f'bytes={start}-{end}'
        url += f'?range={start}-{end}'  # distinct cache key for each range
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
        if start is not None:
            if r.status != 206 or not r.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                raise RuntimeError('Server did not honor exact byte range; refusing shard download')
        cap = count if count is not None else 24 * 1024 * 1024
        data = r.read(cap + 1)
        if len(data) > cap or (count is not None and len(data) != count):
            raise RuntimeError('Unexpected response size')
        return data

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', default='build/model'); args = ap.parse_args()
    out = pathlib.Path(args.output); out.mkdir(parents=True, exist_ok=True)
    def metadata(name):
        path = out / name
        data = get(name); path.write_bytes(data); return json.loads(data)
    index = metadata('model.safetensors.index.json')['weight_map']
    metadata('config.json'); metadata('hf_quant_config.json')
    keys = [f'model.language_model.layers.{layer}.mlp.{proj}.{suffix}'
            for layer in [0, 1] for proj in ['gate_proj', 'up_proj', 'down_proj']
            for suffix in ['weight', 'weight_scale', 'weight_scale_2', 'input_scale']]
    headers = {}; selected = []; total = 0
    for key in keys:
        shard = index[key]
        if shard not in headers:
            length = struct.unpack('<Q', get(shard, 0, 8))[0]
            if length > 16 * 1024 * 1024: raise RuntimeError('Oversized header')
            headers[shard] = (8 + length, json.loads(get(shard, 8, length)))
        offset, head = headers[shard]; info = head[key]; begin, end = info['data_offsets']
        size = end - begin
        item_size = {'U8': 1, 'F8_E4M3': 1, 'F32': 4}.get(info['dtype'])
        if item_size is None or math.prod(info['shape']) * item_size != size:
            raise RuntimeError(f'Unexpected tensor type/size: {key} {info}')
        total += size
        selected.append(dict(name=key, shard=shard, offset=offset + begin, size=size,
                             dtype=info['dtype'], shape=info['shape'], file=key + '.bin'))
    if total > LIMIT: raise RuntimeError(f'Tensor budget exceeded: {total}')
    print(f'Selected {len(selected)} tensors: {total} bytes', flush=True)
    for item in selected:
        path = out / item['file']
        data = bytearray()
        for pos in range(0, item['size'], 4 * 1024 * 1024):
            data.extend(get(item['shard'], item['offset'] + pos, min(4 * 1024 * 1024, item['size'] - pos)))
        item['sha256'] = hashlib.sha256(data).hexdigest()
        path.write_bytes(data)
        print(item['name'], item['dtype'], item['shape'], flush=True)
    manifest = dict(repo=REPO, revision=REVISION, tensor_bytes=total, tensors=selected)
    (out / 'selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Compact binary input with explicit shapes and scales for the standalone HIP runner.
    with (out / 'layers.bin').open('wb') as f:
        f.write(b'NVLAYER1'); f.write(struct.pack('<I', 6))
        for layer in [0, 1]:
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                prefix = f'model.language_model.layers.{layer}.mlp.{proj}.'
                items = {t['name'][len(prefix):]: t for t in selected if t['name'].startswith(prefix)}
                w, s, g = (items[x] for x in ['weight', 'weight_scale', 'weight_scale_2'])
                n, halfk = w['shape']; k = halfk * 2
                assert w['dtype'] == 'U8' and s['dtype'] == 'F8_E4M3' and s['shape'] == [n, k // 16]
                assert g['dtype'] == 'F32' and g['size'] == 4
                alpha = struct.unpack('<f', (out / g['file']).read_bytes())[0]
                assert math.isfinite(alpha) and alpha > 0
                f.write(struct.pack('<IIIf', layer, n, k, alpha))
                f.write((out / w['file']).read_bytes()); f.write((out / s['file']).read_bytes())
    print('Ready: layers.bin (six projections; no complete shards)', flush=True)

if __name__ == '__main__': main()
