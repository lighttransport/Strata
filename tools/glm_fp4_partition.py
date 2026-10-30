#!/usr/bin/env python3
"""Write a compact exact-byte partition; source checkpoint is never modified.

Outputs are model artifacts, not repository files. This tool does not load weights
into inference RAM or claim that distributed generation is available.
"""
import argparse
import fcntl
import subprocess
import sys
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from glm_fp4_layout import inspect
from glm_fp4_layout_rules import rules




# Linux UAPI constants may be omitted by standalone CPython builds.
F_ADD_SEALS = getattr(fcntl, 'F_ADD_SEALS', 1033)
F_GET_SEALS = getattr(fcntl, 'F_GET_SEALS', 1034)
F_SEAL_SEAL = getattr(fcntl, 'F_SEAL_SEAL', 1)
F_SEAL_SHRINK = getattr(fcntl, 'F_SEAL_SHRINK', 2)
F_SEAL_GROW = getattr(fcntl, 'F_SEAL_GROW', 4)
F_SEAL_WRITE = getattr(fcntl, 'F_SEAL_WRITE', 8)

def create_memfd(name):
    """Create sealable Linux RAM storage on Python builds without os.memfd_create."""
    if hasattr(os, 'memfd_create'):
        return os.memfd_create(name, 3)  # MFD_CLOEXEC | MFD_ALLOW_SEALING
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    create = libc.memfd_create
    create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    create.restype = ctypes.c_int
    fd = create(name.encode(), 3)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def available_memory():
    meminfo = dict((line.split()[0].rstrip(':'), int(line.split()[1]) * 1024)
                   for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split()) >= 2 and line.split()[1].isdigit())
    available = meminfo['MemAvailable']
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        if controllers:
            continue
        base = Path('/sys/fs/cgroup')
        current = base / relative.lstrip('/')
        while current.is_relative_to(base):
            limit = current / 'memory.max'
            usage = current / 'memory.current'
            if limit.is_file() and usage.is_file():
                value = limit.read_text().strip()
                if value.isdigit():
                    available = min(available, max(0, int(value) - int(usage.read_text())))
            if current == base:
                break
            current = current.parent
    return available


def partition(source, destination, layout, owner, chunk_bytes=4 << 20, ram_fds=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if owner not in ('worker', 'decoder') or chunk_bytes < 1:
        raise ValueError('invalid partition options')
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError('destination must be new and outside the source checkpoint')
    _, _, fingerprint = inspect(source, expected=layout['tensors'])
    if fingerprint != layout['index_sha256']:
        raise ValueError('checkpoint index fingerprint mismatch')
    canonical = dict(layout)
    expected = canonical.pop('layout_sha256')
    if hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != expected:
        raise ValueError('layout fingerprint mismatch')
    if ram_fds is not None:
        available = available_memory()
        if not rules().strata_glm_fp4_layout_admit(layout[owner + '_bytes'], layout['reserve_bytes'], available):
            raise MemoryError(f'partition plus runtime reserve exceeds currently available RAM: weights={layout[owner + "_bytes"]} reserve={layout["reserve_bytes"]} available={available}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if ram_fds is None and shutil.disk_usage(destination.parent).free < layout[owner + '_bytes'] + (256 << 20):
        raise OSError('insufficient disk for compact partition; use --ram for the dedicated run')
    with tempfile.TemporaryDirectory(prefix='.fp4-partition-', dir=destination.parent) as temporary:
        output = Path(temporary)
        selected = {}
        gpu_shards = []
        for name, row in layout['tensors'].items():
            if row['owner'] == owner:
                selected.setdefault((row['file'], row.get('residency', 'ram')), []).append((name, row))
        index, hashes, total = {}, {}, 0
        for number, entries in enumerate(sorted(selected.values(), key=lambda items: items[0][1]['file'])):
            filename = f'part-{number:05d}.safetensors'
            if entries[0][1].get('residency') == 'gpu':
                gpu_shards.append(filename)
            header, offset = {}, 0
            for name, row in sorted(entries):
                header[name] = dict(dtype=row['dtype'], shape=row['shape'], data_offsets=[offset, offset + row['bytes']])
                offset += row['bytes']
                index[name] = filename
            raw = json.dumps(header, sort_keys=True, separators=(',', ':')).encode()
            raw += b' ' * (-len(raw) % 8)
            digest = hashlib.sha256()
            if ram_fds is None:
                target_stream = (output / filename).open('xb')
            else:
                descriptor = create_memfd('glm-fp4-part')
                ram_fds[filename] = descriptor
                target_stream = os.fdopen(os.dup(descriptor), 'wb')
            with target_stream as target:
                prefix = struct.pack('<Q', len(raw)) + raw
                target.write(prefix); digest.update(prefix)
                for name, row in sorted(entries):
                    with (source / row['file']).open('rb') as incoming:
                        incoming.seek(row['offset'])
                        left = row['bytes']
                        while left:
                            data = incoming.read(min(left, chunk_bytes))
                            if not data:
                                raise ValueError('truncated tensor during copy')
                            target.write(data); digest.update(data); left -= len(data)
                        if hasattr(os, 'posix_fadvise'):
                            os.posix_fadvise(incoming.fileno(), row['offset'], row['bytes'], os.POSIX_FADV_DONTNEED)
                target.flush(); os.fsync(target.fileno())
            if ram_fds is not None:
                seals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL
                fcntl.fcntl(descriptor, F_ADD_SEALS, seals)
                (output / filename).symlink_to(f'/proc/{os.getpid()}/fd/{descriptor}')
            # Read back the complete compact artifact independently before publishing.
            verified = hashlib.sha256()
            with (output / filename).open('rb') as incoming:
                while data := incoming.read(chunk_bytes):
                    verified.update(data)
            if verified.digest() != digest.digest():
                raise ValueError('partition readback checksum mismatch')
            hashes[filename] = digest.hexdigest(); total += offset
        (output / 'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=index), sort_keys=True))
        for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json'):
            if (source / name).is_file():
                shutil.copyfile(source / name, output / name)
        catalogs = []
        if owner == 'decoder':
            remote = sorted((name, row) for name, row in layout['tensors'].items() if row['owner'] == 'worker')
            for start in range(0, len(remote), 8192):
                catalog = {}
                for tensor_id, (name, row) in enumerate(remote[start:start + 8192], start):
                    catalog[name] = dict(dtype=row['dtype'], shape=row['shape'], bytes=row['bytes'], tensor_id=tensor_id)
                filename = f'remote-{len(catalogs):05d}.json'
                (output / filename).write_text(json.dumps(catalog, sort_keys=True))
                catalogs.append(filename)
        (output / 'glm-fp4-layout.json').write_text(json.dumps(layout, sort_keys=True, separators=(',', ':')))
        report = dict(schema='strata.glm.fp4.partition.v1', owner=owner, payload_bytes=total,
                      index_sha256=fingerprint, layout_sha256=expected, shards=hashes,
                      remote_experts_per_layer=layout['remote_experts_per_layer'], reserve_bytes=layout['reserve_bytes'],
                      remote_catalogs=catalogs, gpu_fixed_shards=gpu_shards,
                      resident_bytes=total - (layout.get('gpu_fixed_bytes', 0) if owner == 'decoder' else 0))
        (output / 'partition.json').write_text(json.dumps(report, sort_keys=True))
        os.rename(output, destination)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--layout', type=Path, required=True)
    parser.add_argument('--owner', choices=('worker', 'decoder'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ram', action='store_true', help='hold compact shards in sealed RAM files while running COMMAND')
    argv = sys.argv[1:]
    split = argv.index('--') if '--' in argv else len(argv)
    args = parser.parse_args(argv[:split])
    args.command = argv[split + 1:]
    if args.ram and not args.command:
        parser.error('--ram requires a child command after --')
    if args.command and not args.ram:
        parser.error('a child command requires --ram')
    descriptors = {} if args.ram else None
    created = False
    try:
        result = partition(args.model, args.output, json.loads(args.layout.read_text()), args.owner, ram_fds=descriptors)
        created = True
        print(json.dumps(result, indent=2), flush=True)
        if args.command:
            command = args.command[1:] if args.command[0] == '--' else args.command
            if not command:
                parser.error('empty child command')
            environment = dict(os.environ, STRATA_GLM_FP4_RAM_FDS='1')
            raise SystemExit(subprocess.run(command, env=environment).returncode)
    finally:
        if descriptors is not None:
            for descriptor in descriptors.values():
                os.close(descriptor)
            # This directory was created by this invocation and its links expire now.
            if created and args.output.exists():
                shutil.rmtree(args.output)



if __name__ == '__main__':
    main()
