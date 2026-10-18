#!/usr/bin/env python3
"""Tests for strata-pfetch / strata-pput (python3 stdlib; needs mpirun for the multi-rank cases).

    make -C tools/pfetch test                 # or: python3 tools/pfetch/test_pfetch.py
    PFETCH_BUILD=dir MPIRUN="mpirun --oversubscribe" python3 test_pfetch.py

Planner invariants come from `strata-pput --plan-dump --ranks=R`; copies run under `mpirun -np 4` and are compared
byte for byte with a Python reference of the manifest."""
import hashlib
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.environ.get('PFETCH_BUILD', os.path.join(HERE, 'build'))
FETCH = os.path.join(BUILD, 'strata-pfetch')
PUT = os.path.join(BUILD, 'strata-pput')
MPIRUN = shlex.split(os.environ.get('MPIRUN', 'mpirun --oversubscribe'))
MIB = 1 << 20


def run(cmd, ok=True, timeout=300):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
    if ok and p.returncode != 0:
        raise AssertionError('command failed ({}): {}\n{}'.format(p.returncode, ' '.join(cmd), p.stdout[-4000:]))
    return p


def fetch(manifest, root, ranks=4, extra=(), ok=True):
    return run(MPIRUN + ['-np', str(ranks), FETCH, '--manifest=' + manifest, '--local-root=' + root, '--streams=3',
                         '--progress-s=1'] + list(extra), ok=ok)


def plan_dump(manifest, ranks, extra=()):
    out = run([PUT, '--plan-dump', '--ranks={}'.format(ranks), '--manifest=' + manifest] + list(extra)).stdout
    loads = {int(m.group(1)): (int(m.group(2)), m.group(3)) for m in re.finditer(r'^LOAD rank=(\d+) read=(\d+) blocks=\d+ first=(\S*)', out, re.M)}
    head = dict(kv.split('=') for kv in re.search(r'^PLAN (.*)$', out, re.M).group(1).split())
    blocks = re.findall(r'^BLOCK \d+ src=(\d+) index=(\d+) reader=(\d+) read=(\d+)', out, re.M)
    return out, {k: int(v) for k, v in head.items()}, loads, blocks


def rand_file(path, size, rng):
    with open(path, 'wb') as f:
        f.write(bytes(rng.getrandbits(8) for _ in range(min(size, 4096))) * (size // 4096 + 1))
    with open(path, 'r+b') as f:  # make every 4 KiB distinct so misplaced bytes are caught
        for o in range(0, size, 4096):
            f.seek(o)
            f.write(o.to_bytes(8, 'little'))
        f.truncate(size)


class Manifest:
    def __init__(self, ranks, block):
        self.ranks, self.block, self.lines, self.dests, self.copies = ranks, block, [], [], []

    def group(self, name, prio):
        self.lines.append('G {} {} {}'.format(sum(1 for l in self.lines if l.startswith('G ')), name, prio))

    def dest(self, rank, path, size, mode, sidecar=None):
        did = len(self.dests)
        self.dests.append((rank, path, size, mode, sidecar))
        self.lines.append('D {} {} {} {} {}{}'.format(did, rank, path, size, mode, ' sidecar=' + sidecar if sidecar else ''))
        return did

    def link(self, rank, link, target):
        self.lines.append('S {} {} {}'.format(rank, link, target))

    def copy(self, did, dst_off, src, src_off, length, gid=None):
        self.copies.append((did, dst_off, src, src_off, length))
        self.lines.append('C {} {} {} {} {}{}'.format(did, dst_off, src, src_off, length, '' if gid is None else ' ' + str(gid)))

    def write(self, path):
        with open(path, 'w') as f:
            f.write('PFETCH 1 block={}\n'.format(self.block) + '\n'.join(self.lines) + '\n')

    def expected(self, root):
        """{path: (bytes, mode, sidecar)} for every expanded destination."""
        out = {}
        for did, (rank, path, size, mode, sidecar) in enumerate(self.dests):
            for r in (range(self.ranks) if rank == '*' else [rank]):
                p = path.replace('@L/', root + '/').replace('{rank2}', '%02d' % r).replace('{rank}', str(r))
                s = sidecar.replace('@L/', root + '/').replace('{rank}', str(r)) if sidecar else None
                buf = bytearray(size)
                for d, doff, src, soff, n in self.copies:
                    if d == did:
                        with open(src, 'rb') as f:
                            f.seek(soff)
                            buf[doff:doff + n] = f.read(n)
                out[p] = (bytes(buf), mode, s)
        return out


def check_tree(m, root):
    for path, (data, mode, sidecar) in m.expected(root).items():
        with open(path, 'rb') as f:
            got = f.read()
        assert got == data, 'content differs: ' + path
        assert not os.path.exists(path + '.pfpart'), 'leftover .pfpart: ' + path
        if sidecar:
            with open(sidecar) as f:
                assert f.read() == hashlib.sha256(data).hexdigest() + '\n', 'bad sidecar ' + sidecar


def build_case(tmp, rng, ranks=4, block=1 << 20):
    srcs = []
    for i, size in enumerate([5 * MIB + 12345, 3 * MIB + 7, 9 * MIB, 700 * 1024]):
        p = os.path.join(tmp, 'src', 'shard{}.bin'.format(i))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        rand_file(p, size, rng)
        srcs.append((p, size))
    m = Manifest(ranks, block)
    m.group('head', 0)
    m.group('body', 1)
    # a sparse full-size mirror of shard 2 on rank 1: header + two unaligned ranges crossing block boundaries
    d = m.dest(1, '@L/n{rank}/mirror/shard2.bin', srcs[2][1], 'sparse')
    m.copy(d, 0, srcs[2][0], 0, 4096, 0)
    m.copy(d, 1 * MIB - 1000, srcs[2][0], 1 * MIB - 1000, 2 * MIB + 3001, 1)
    m.copy(d, 6 * MIB + 5, srcs[2][0], 6 * MIB + 5, 777777, 1)
    # atomic tensor files on every rank; the same source bytes go to several ranks and to two files on rank 0
    for r in range(ranks):
        for k in range(3):
            off = rng.randrange(0, srcs[0][1] - 600000)
            n = rng.randrange(1, 600000)
            dd = m.dest(r, '@L/n{rank}/stage/t%d.bin' % k, n, 'atomic', '@L/n{rank}/stage/t%d.sha256' % k)
            m.copy(dd, 0, srcs[0][0], off, n, 1)
    shared = m.dest(0, '@L/n{rank}/stage/dup.bin', 2 * MIB, 'atomic')
    m.copy(shared, 0, srcs[1][0], MIB // 2, 2 * MIB, 1)
    tp = m.dest(0, '@L/n{rank}/tp/dup-shard.bin', MIB, 'atomic')
    m.copy(tp, 0, srcs[1][0], MIB, MIB, 1)
    # one file assembled from two sources; a zero-size file; an all-ranks file
    two = m.dest(3, '@L/n{rank}/stage/two.bin', 300000, 'atomic')
    m.copy(two, 0, srcs[3][0], 100, 100000, 1)
    m.copy(two, 100000, srcs[1][0], 3 * MIB - 200000, 200000, 1)
    m.dest(2, '@L/n{rank}/stage/empty.bin', 0, 'atomic', '@L/n{rank}/stage/empty.sha256')
    allr = m.dest('*', '@L/n{rank}/imatrix.gguf', srcs[3][1], 'atomic')
    m.copy(allr, 0, srcs[3][0], 0, srcs[3][1], 0)
    m.link('*', '@L/n{rank}/mirror/shard0.bin', srcs[0][0])
    path = os.path.join(tmp, 'manifest.txt')
    m.write(path)
    return m, path, srcs


def test_planner(tmp):
    rng = random.Random(1)
    srcs = []
    for i in range(6):
        p = os.path.join(tmp, 'p{}.bin'.format(i))
        open(p, 'wb').close()
        srcs.append(p)
    m = Manifest(12, 1 << 20)
    # 12 ranks, each wants a different slice of every source (like expert % 12), plus an all-ranks file
    for r in range(12):
        for i, s in enumerate(srcs):
            d = m.dest(r, '@L/r{rank}/f%d' % i, 8 * MIB, 'atomic')
            m.copy(d, 0, s, (r * 8 + i % 3) * MIB, 8 * MIB)
    a = m.dest('*', '@L/r{rank}/all', 3 * MIB, 'atomic')
    m.copy(a, 0, srcs[0], 200 * MIB, 3 * MIB)
    path = os.path.join(tmp, 'plan.txt')
    m.write(path)
    for assign in ('local', 'spread'):
        out, head, loads, blocks = plan_dump(path, 12, ['--assign=' + assign, '--streams=6'])
        keys = [(b[0], b[1]) for b in blocks]
        assert len(keys) == len(set(keys)), 'a block is read twice'
        assert head['read_bytes'] == 12 * 6 * 8 * MIB + 3 * MIB, head   # every needed source byte exactly once
        assert head['needed_bytes'] == 12 * 6 * 8 * MIB + 12 * 3 * MIB, head
        reads = [loads[r][0] for r in range(12)]
        assert max(reads) - min(reads) <= MIB + head['read_bytes'] // 100, (assign, reads)
        if assign == 'local':
            assert head['remote_bytes'] <= 11 * 3 * MIB + 2 * MIB * 12, head  # slices stay with their owners
        for r in range(12):
            first = loads[r][1].split(',')
            assert len(set(first)) == min(6, len(first)), (assign, r, first)  # 6 threads start on 6 different files
        starts = {loads[r][1].split(',')[0] for r in range(12)}
        assert len(starts) == 6, starts                                     # ranks start on different files
        assert plan_dump(path, 12, ['--assign=' + assign, '--streams=6'])[0] == out, 'plan not deterministic'
    # rank 0 alone needs extra bytes (the canonical dense copy): other ranks read them
    m2 = Manifest(12, 1 << 20)
    for r in range(12):
        d = m2.dest(r, '@L/x/f', 10 * MIB, 'atomic')
        m2.copy(d, 0, srcs[r % 6], 100 * MIB + r * 10 * MIB, 10 * MIB)
    extra = m2.dest(0, '@L/x/dense', 60 * MIB, 'atomic')
    m2.copy(extra, 0, srcs[5], 500 * MIB, 60 * MIB)
    m2.write(path)
    _, head, loads, _ = plan_dump(path, 12)
    reads = [loads[r][0] for r in range(12)]
    assert max(reads) - min(reads) <= MIB + head['read_bytes'] // 100, reads
    # malformed manifests fail with a message
    for bad in ['PFETCH 2\n', 'PFETCH 1\nD 0 0 /x 10 atomic\nC 0 0 /s 0 5\n', 'PFETCH 1\nD 0 0 /x 10 atomic\nC 0 5 /s 0 6\n',
                'PFETCH 1\nD 0 0 /x 10 sparse\nC 0 0 /s 0 6\nC 0 4 /s 0 6\n', 'PFETCH 1\nD 1 0 /x 10 atomic\n']:
        with open(path, 'w') as f:
            f.write(bad)
        p = run([PUT, '--plan-dump', '--ranks=2', '--manifest=' + path], ok=False)
        assert p.returncode != 0, bad
    print('planner: ok')


def test_copy(tmp):
    rng = random.Random(7)
    m, path, srcs = build_case(tmp, rng)
    root = os.path.join(tmp, 'L')
    out = fetch(path, root, extra=['--coverage-out=@L/n{rank}/coverage', '--status-dir=' + os.path.join(tmp, 'status')]).stdout
    assert 'SENTINEL PFETCH_END state=ok' in out, out
    assert out.count('PFETCH_GROUP_READY') >= 8, out
    check_tree(m, root)
    mirror = os.path.join(root, 'n1/mirror/shard2.bin')
    assert os.path.getsize(mirror) == srcs[2][1]
    for r in range(4):
        assert os.readlink(os.path.join(root, 'n%d/mirror/shard0.bin' % r)) == srcs[0][0]
        assert os.path.exists(os.path.join(tmp, 'status', 'pfetch.rank%02d.status' % r))
        assert os.path.exists(os.path.join(root, '.pfetch', 'head.ready'))
    # coverage of rank 0 lists exactly the bytes written there (merged ranges)
    with open(os.path.join(root, 'n0/coverage')) as f:
        cov_lines = [l.split() for l in f]
    exp = m.expected(root)
    assert sum(int(l[2]) for l in cov_lines) == sum(len(exp[p][0]) for p in exp if '/n0/' in p), cov_lines
    # rerun: atomic files are skipped, the sparse mirror is refetched, results identical
    out = fetch(path, root).stdout
    skipped = sum(int(x) for x in re.findall(r'skipped=(\d+)', out))
    assert skipped >= 4 * 3 + 4, out
    check_tree(m, root)
    # a big sparse destination stays sparse
    big = Manifest(4, 1 << 20)
    d = big.dest(2, '@L/n{rank}/big.bin', 256 * MIB, 'sparse')
    big.copy(d, 200 * MIB, srcs[0][0], 0, MIB)
    big.write(os.path.join(tmp, 'big.txt'))
    fetch(os.path.join(tmp, 'big.txt'), root)
    st = os.stat(os.path.join(root, 'n2/big.bin'))
    assert st.st_size == 256 * MIB and st.st_blocks * 512 < 16 * MIB, st
    print('copy: ok')


def test_faults(tmp):
    rng = random.Random(9)
    m, path, _ = build_case(tmp, rng)
    root = os.path.join(tmp, 'F')
    p = fetch(path, root, extra=['--inject=corrupt:0'], ok=False)
    assert p.returncode != 0 and 'CRC mismatch' in p.stdout, p.stdout[-2000:]
    out = fetch(path, root).stdout
    assert 'state=ok' in out
    check_tree(m, root)
    shutil.rmtree(root)
    t0 = time.time()
    out = fetch(path, root, extra=['--inject=eio:2']).stdout
    assert 'state=ok' in out and time.time() - t0 >= 1, out
    check_tree(m, root)
    shutil.rmtree(root)
    p = fetch(path, root, extra=['--deadline={}'.format(int(time.time()) - 1)], ok=False)
    assert p.returncode == 2 and 'SENTINEL PFETCH_END state=partial' in p.stdout, p.stdout[-2000:]
    out = fetch(path, root).stdout                       # the next run completes what the cut-off one left
    assert 'state=ok' in out
    check_tree(m, root)
    print('faults: ok')


def test_put(tmp):
    rng = random.Random(3)
    src = os.path.join(tmp, 'local')
    dst = os.path.join(tmp, 'shared', 'chunks')
    os.makedirs(src)
    files = []
    for i, size in enumerate([40 * MIB + 3, 5, 0, 17 * MIB]):
        p = os.path.join(src, 'f%d' % i)
        rand_file(p, size, rng)
        files.append(p)
    lst = os.path.join(tmp, 'put.txt')
    with open(lst, 'w') as f:
        f.write('W {} {}/c0.chunk\n'.format(files[0], dst))
        f.write('W {} {}/c0.json after=0\n'.format(files[1], dst))
        f.write('W {} {}/empty\n'.format(files[2], dst))
        f.write('W {} {}/c1.chunk\n'.format(files[3], dst))
    out = run([PUT, '--write=' + lst, '--streams=4', '--readback']).stdout
    assert 'SENTINEL PPUT_END state=ok files=4' in out, out
    for i, name in enumerate(['c0.chunk', 'c0.json', 'empty', 'c1.chunk']):
        with open(files[i], 'rb') as a, open(os.path.join(dst, name), 'rb') as b:
            assert a.read() == b.read(), name
    assert not [x for x in os.listdir(dst) if x.endswith('.pfpart')]
    out = run([PUT, '--write=' + lst, '--streams=2', '--skip-existing']).stdout
    assert 'skipped=4' in out, out
    with open(lst, 'a') as f:
        f.write('W {} {}/x after=9\n'.format(files[1], dst))
    assert run([PUT, '--write=' + lst], ok=False).returncode != 0
    print('put: ok')


def test_measure(tmp):
    d = os.path.join(tmp, 'msrc')
    os.makedirs(d)
    rng = random.Random(5)
    for i in range(3):
        rand_file(os.path.join(d, 's%d' % i), 8 * MIB, rng)
    out = run(MPIRUN + ['-np', '2', FETCH, '--measure=read,write,mpi,local', '--src-dir=' + d, '--min-file-gib=0',
                        '--scratch-shared=' + os.path.join(tmp, 'mshared'), '--local-root=' + os.path.join(tmp, 'mlocal'),
                        '--point-s=1', '--region-mib=2', '--streams-list=1,2']).stdout
    kinds = re.findall(r'PFETCH_MEASURE kind=(\S+) layout=(\S+)', out)
    assert ('read', 'own-file') in kinds and ('read', 'all-one-file') in kinds and ('write', 'one-file') in kinds, out
    assert ('mpi', 'ring') in kinds and ('mpi', 'bcast') in kinds and ('local-read', 'file-per-stream') in kinds, out
    assert re.search(r'kind=sparse .*local_sparse=(yes|no)', out) and 'PFETCH_MEASURE_END state=ok' in out, out
    print('measure: ok')


def main():
    for b in (FETCH, PUT):
        if not os.access(b, os.X_OK):
            sys.exit('build first: make -C ' + HERE)
    with tempfile.TemporaryDirectory(prefix='pfetch-test-') as tmp:
        for t in (test_planner, test_copy, test_faults, test_put, test_measure):
            sub = os.path.join(tmp, t.__name__)
            os.makedirs(sub)
            t(sub)
    print('ALL PASS')


if __name__ == '__main__':
    main()
