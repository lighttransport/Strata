#!/usr/bin/env python3
"""Inspect native GLM weights and plan whole-expert ownership without reading payloads."""
import argparse
import hashlib
import json
import pathlib
import re
import struct
from glm_fp4_layout_rules import rules

GIB = 1 << 30
DTYPES = dict(U8=1, I8=1, BOOL=1, F8_E4M3=1, U16=2, I16=2, F16=2, BF16=2, U32=4, I32=4, F32=4, I64=8)
EXPERT = re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.')


def text_tensor(name):
    # The vocabulary head is a root tensor, outside model.language_model.
    return name.startswith('model.language_model.') or name == 'lm_head.weight' or name.startswith('lm_head.weight_scale')


def inspect(root, expected=None):
    root = pathlib.Path(root).resolve()
    raw = (root / 'model.safetensors.index.json').read_bytes()
    index = json.loads(raw)['weight_map']
    config = json.loads((root / 'config.json').read_text())
    text = config.get('text_config', config)
    records = {} if expected is None else set()
    for filename in sorted(set(index.values())):
        relative = pathlib.Path(filename)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe shard path')
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('shard escapes checkpoint')
        with path.open('rb') as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError('truncated shard')
            size = struct.unpack('<Q', prefix)[0]
            if size > 128 * 1024 * 1024 or size > path.stat().st_size - 8:
                raise ValueError('invalid header size')
            header = json.loads(stream.read(size))
        spans = []
        for name, value in header.items():
            if name == '__metadata__':
                continue
            if index.get(name) != filename:
                raise ValueError('tensor/index disagreement')
            count = DTYPES[value['dtype']]
            for dimension in value['shape']:
                if type(dimension) is not int or dimension < 0 or dimension > (1 << 64) - 1:
                    raise ValueError('invalid tensor shape')
                count *= dimension
                if count > (1 << 64) - 1:
                    raise ValueError('tensor shape overflow')
            begin, end = value['data_offsets']
            if any(type(v) is not int or not 0 <= v < (1 << 64) for v in (begin, end)):
                raise ValueError('tensor offset exceeds unsigned 64-bit extent')
            if count != end - begin:
                raise ValueError('tensor shape/size disagreement')
            if not 0 <= begin <= end or not rules().strata_glm_fp4_layout_span(begin, end - begin, path.stat().st_size - 8 - size):
                raise ValueError('invalid tensor bounds')
            if name in records:
                raise ValueError('duplicate tensor')
            spans.append((begin, end))
            row = dict(file=filename, offset=8 + size + begin,
                       bytes=end - begin, dtype=value['dtype'], shape=value['shape'])
            if expected is None:
                records[name] = row
            else:
                if name not in expected or any(expected[name].get(key) != value for key, value in row.items()):
                    raise ValueError('layout disagrees with checkpoint metadata')
                records.add(name)
        spans.sort()
        nonempty = [(a, b) for a, b in spans if a != b]
        if nonempty and (nonempty[0][0] != 0 or any(a[1] != b[0] for a, b in zip(nonempty, nonempty[1:]))):
            raise ValueError('overlapping tensors or payload holes')
        if (nonempty[-1][1] if nonempty else 0) != path.stat().st_size - 8 - size:
            raise ValueError('trailing payload bytes')
    if set(records) != set(index):
        raise ValueError('missing indexed tensors')
    if expected is not None and records != set(expected):
        raise ValueError('layout tensor coverage mismatch')
    return text, records if expected is None else None, hashlib.sha256(raw).hexdigest()


def plan(text, records, fingerprint, worker_limit, decoder_limit, reserve=6 * GIB, gpu_fixed=False):
    if any(not isinstance(v, int) or v < 0 or v > (1 << 64) - 1 for v in (worker_limit, decoder_limit, reserve)):
        raise ValueError('invalid memory allowance')
    if not 0 <= text['first_k_dense_replace'] < text['num_hidden_layers'] <= 128 or not 1 <= text['n_routed_experts'] <= 288:
        raise ValueError('unsupported expert geometry')
    if any(type(row['bytes']) is not int or row['bytes'] < 0 for row in records.values()) or sum(row['bytes'] for row in records.values()) >= (1 << 64):
        raise ValueError('checkpoint payload exceeds unsigned 64-bit extent')
    layers = range(text['first_k_dense_replace'], text['num_hidden_layers'])
    experts = text['n_routed_experts']
    sizes = {(layer, expert): 0 for layer in layers for expert in range(experts)}
    fixed = 0
    excluded = 0
    for name, record in records.items():
        match = EXPERT.match(name)
        key = tuple(map(int, match.groups())) if match else None
        if key in sizes:
            sizes[key] += record['bytes']
        elif text_tensor(name):
            fixed += record['bytes']
        else:
            excluded += record['bytes']
    if not sizes or any(size <= 0 for size in sizes.values()):
        raise ValueError('incomplete expert coverage')
    gpu_matrices = {name for name, row in records.items() if gpu_fixed and text_tensor(name)
                    and '.mlp.experts.' not in name and name != 'model.language_model.embed_tokens.weight'
                    and name.endswith('.weight') and len(row['shape']) == 2}
    gpu_names = set()
    for name in records:
        base = name.split('_scale', 1)[0]
        if base in gpu_matrices:
            gpu_names.add(name)
    gpu_bytes = sum(records[name]['bytes'] for name in gpu_names)
    candidates = []
    for count in range(experts + 1):
        worker = sum(size for (layer, expert), size in sizes.items() if expert < count)
        decoder = fixed + sum(sizes.values()) - worker
        if rules().strata_glm_fp4_layout_admit(worker, reserve, worker_limit) and rules().strata_glm_fp4_layout_admit(decoder - gpu_bytes, reserve, decoder_limit):
            candidates.append((count, worker, decoder))
    if not candidates:
        raise ValueError(f'weights plus reserves do not fit both nodes: worker_allowance={worker_limit} decoder_allowance={decoder_limit} reserve={reserve}')
    count, worker, decoder = candidates[-1]
    assignments = {}
    for name, record in records.items():
        match = EXPERT.match(name)
        key = tuple(map(int, match.groups())) if match else None
        remote = key in sizes and rules().strata_glm_fp4_layout_role(key[0], key[1], text['first_k_dense_replace'], text['num_hidden_layers'], experts, count) == 1
        owner = 'worker' if remote else 'decoder' if text_tensor(name) else 'excluded'
        assignments[name] = dict(record, owner=owner, residency='gpu' if name in gpu_names else 'ram')
    result = dict(schema='strata.glm.fp4.layout.v1', index_sha256=fingerprint,
                  remote_experts_per_layer=count, worker_bytes=worker, decoder_bytes=decoder,
                  excluded_bytes=excluded, reserve_bytes=reserve, tensors=assignments,
                  gpu_fixed_bytes=gpu_bytes, decoder_resident_bytes=decoder - gpu_bytes)
    result['layout_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=pathlib.Path)
    parser.add_argument('--worker-gib', type=float, default=118)
    parser.add_argument('--decoder-gib', type=float, default=60)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--gpu-fixed', action='store_true', help='keep fixed-matrix source files on decoder disk after their GPU upload')
    args = parser.parse_args()
    text, records, fingerprint = inspect(args.model)
    result = plan(text, records, fingerprint, int(args.worker_gib * GIB), int(args.decoder_gib * GIB), gpu_fixed=args.gpu_fixed)
    with args.output.open('x') as stream:
        json.dump(result, stream, sort_keys=True)
    print(json.dumps({key: value for key, value in result.items() if key != 'tensors'}, indent=2))


if __name__ == '__main__':
    main()
