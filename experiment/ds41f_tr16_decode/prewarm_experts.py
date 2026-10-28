#!/usr/bin/env python3
"""Read the DS4.1F expert tensor byte ranges into the page cache (outside any decoder scope).
usage: prewarm_experts.py SHARD1_PATH   (needs gguf-py + numpy)"""
import sys, re, os, time
sys.path.insert(0, '/mnt/nvme02/work/llama.cpp/gguf-py')
from gguf import GGUFReader
first = sys.argv[1]; m = re.match(r'(.*-)0*1(-of-0*(\d+)\.gguf)$', first)
n = int(m.group(3)); total = 0; t0 = time.time(); buf = bytearray(64 << 20)
for s in range(1, n + 1):
    path = f'{m.group(1)}{s:05d}{m.group(2)}'
    r = GGUFReader(path)
    ranges = sorted((int(t.data_offset), int(t.n_bytes)) for t in r.tensors if '_exps.' in t.name)
    del r
    with open(path, 'rb', buffering=0) as f:
        for off, nb in ranges:
            f.seek(off); left = nb
            while left:
                k = f.readinto(memoryview(buf)[:min(left, len(buf))]); left -= k; total += k
    print(f'shard {s}: {len(ranges)} expert tensors, cumulative {total/2**30:.1f} GiB, {time.time()-t0:.0f} s', flush=True)
