#!/usr/bin/env python3
"""strata-pfetch manifest for a DS4.1F pack job: a sparse node-local mirror of the checkpoint on every rank.

Each rank gets, under MIRROR (e.g. @L/ckpt), a full-size sparse copy of every shard it reads -- the header bytes plus
exactly the ranges `strata-ds41f-pack ranges` reports for its layers (and the trunk on the trunk rank) -- and a
symlink to the shared shard for every other shard, so `strata-ds41f-pack experts MIRROR ...` opens the same checkpoint
identity (basenames, sizes, header bytes) and reads only local bytes. Layers already complete in every chunk directory
of the job (3 chunk jsons whose chunk files have the recorded size: the tool's own test) are not fetched. With
--imatrix, IMATRIX_LOCAL on every rank receives a copy. python3 stdlib only (runs on the compute head node).

    ds41f_pfetch_plan.py --bin strata-ds41f-pack --ckpt DIR --rank-layers 0:0-3,1:4-7,... [--trunk-rank 11]
        [--chunk-dirs DIR1,DIR2] [--imatrix FILE --imatrix-local @L/imatrix.gguf] [--mirror @L/ckpt] > manifest
"""
import argparse
import json
import os
import re
import subprocess
import sys

PROJS = ('gate', 'up', 'down')


def ranges(binary, ckpt, extra):
    out = subprocess.run([binary, 'ranges', ckpt] + extra, stdout=subprocess.PIPE, check=True, universal_newlines=True).stdout
    shards, rs = {}, []
    for line in out.splitlines():
        f = line.split()
        if f[0] == 'SHARD':
            shards[f[1]] = (int(f[2]), int(f[3]))
        elif f[0] == 'R':
            rs.append((f[1], int(f[2]), int(f[3]), f[4]))
    return shards, rs


def layer_complete(chunk_dir, layer):
    for proj in PROJS:
        base = os.path.join(chunk_dir, 'blk.{}.ffn_{}_exps.weight.chunk'.format(layer, proj))
        try:
            with open(base + '.json') as f:
                want = json.load(f)['bytes']
            if os.path.getsize(base) != want:
                return False
        except (OSError, ValueError, KeyError):
            return False
    return True


def disjoint(spans):
    """[(off, n, group)] -> disjoint [(off, n, group)]: every byte once, in the lowest group that wants it; adjacent
    ranges of one group merged."""
    events = sorted([(off, 1, g) for off, n, g in spans] + [(off + n, -1, g) for off, n, g in spans])
    active, out, pos = {}, [], None
    for x, kind, g in events:
        if pos is not None and x > pos and active:
            low = min(active)
            if out and out[-1][2] == low and out[-1][0] + out[-1][1] == pos:
                out[-1][1] += x - pos
            else:
                out.append([pos, x - pos, low])
        pos = x
        active[g] = active.get(g, 0) + kind
        if active[g] == 0:
            del active[g]
    return [tuple(o) for o in out]


def parse_rank_layers(text):
    out = {}
    for part in text.split(','):
        r, span = part.split(':')
        a, _, b = span.partition('-')
        out[int(r)] = list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', required=True, help='strata-ds41f-pack')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--rank-layers', required=True, help='rank:first-last,... (ranks without layers get only symlinks)')
    ap.add_argument('--ranks', type=int, default=12)
    ap.add_argument('--trunk-rank', default='none')
    ap.add_argument('--chunk-dirs', default='', help='comma-separated shared chunk directories of this job')
    ap.add_argument('--keep', default='')
    ap.add_argument('--imatrix', default='none')
    ap.add_argument('--imatrix-local', default='@L/imatrix.gguf')
    ap.add_argument('--mirror', default='@L/ckpt')
    ap.add_argument('--block-mib', type=int, default=16)
    args = ap.parse_args()
    ckpt = os.path.abspath(args.ckpt)
    keep = ['--keep=' + args.keep] if args.keep else []
    rank_layers = parse_rank_layers(args.rank_layers)
    every = sorted({l for ls in rank_layers.values() for l in ls})
    span = ['--layers={}-{}'.format(every[0], every[-1])] if every else ['--no-experts']
    shards, expert_ranges = ranges(args.bin, ckpt, keep + span)
    trunk_rank = None if args.trunk_rank == 'none' else int(args.trunk_rank)
    trunk_ranges = ranges(args.bin, ckpt, ['--no-experts', '--trunk'])[1] if trunk_rank is not None else []
    by_layer = {}
    for shard, off, n, name in expert_ranges:
        m = re.match(r'layers\.(\d+)\.', name)
        by_layer.setdefault(int(m.group(1)), []).append((shard, off, n))
    dirs = [d for d in args.chunk_dirs.split(',') if d]
    for name in list(shards) + [ckpt, args.mirror]:
        if any(c.isspace() for c in name):
            sys.exit('paths with spaces are not supported: ' + name)

    lines = ['PFETCH 1 block={}'.format(args.block_mib << 20), 'G 0 headers 0']
    groups = {}
    did = 0
    summary = []
    for r in range(args.ranks):
        todo = [l for l in rank_layers.get(r, []) if not dirs or not all(layer_complete(d, l) for d in dirs)]
        want = {}                                    # shard -> [(off, n, group)]
        for wave, layer in enumerate(todo):
            g = groups.setdefault('layer{}'.format(layer), (len(groups) + 1, wave + 1))[0]   # priority: the wave
            for shard, off, n in by_layer.get(layer, []):
                want.setdefault(shard, []).append((off, n, g))
        if r == trunk_rank:
            for shard, off, n, _ in trunk_ranges:
                want.setdefault(shard, []).append((off, n, 0))
        nbytes = 0
        for shard in sorted(shards):
            size, header = shards[shard]
            path = '{}/{}'.format(args.mirror, shard)
            if shard not in want:
                lines.append('S {} {} {}/{}'.format(r, path, ckpt, shard))
                continue
            lines.append('D {} {} {} {} sparse'.format(did, r, path, size))
            for off, n, g in disjoint([(0, header, 0)] + want[shard]):
                lines.append('C {} {} {}/{} {} {} {}'.format(did, off, ckpt, shard, off, n, g))
                nbytes += n
            did += 1
        summary.append('{}:{}L/{:.1f}GB'.format(r, len(todo), nbytes / 1e9))
    for name, (g, prio) in sorted(groups.items(), key=lambda kv: kv[1]):
        lines.insert(1 + g, 'G {} {} {}'.format(g, name, prio))
    if args.imatrix != 'none':
        size = os.path.getsize(args.imatrix)
        lines.append('D {} * {} {} atomic'.format(did, args.imatrix_local, size))
        lines.append('C {} 0 {} 0 {} 0'.format(did, os.path.abspath(args.imatrix), size))
    sys.stdout.write('\n'.join(lines) + '\n')
    print('PFETCH_PLAN ranks={} {}'.format(args.ranks, ' '.join(summary)), file=sys.stderr)


if __name__ == '__main__':
    main()
