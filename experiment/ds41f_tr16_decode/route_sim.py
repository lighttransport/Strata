#!/usr/bin/env python3
"""Tier hit / verify-window lane model from DS41 route traces (int32 [layers][top_k] per position).
usage: route_sim.py PROFILE.i32 EVAL.i32 [--window 6] [--affinity-free]"""
import sys, re, collections
import numpy as np
sys.path.insert(0, '/mnt/nvme02/work/llama.cpp/gguf-py')
from gguf import GGUFReader
L, K, NE = 40, 6, 384
base = '/mnt/nvme02/models/ds41f-oct/iq1s/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-0000%d-of-00005.gguf'
eb = np.zeros(L)
for s in range(2, 5):
    for t in GGUFReader(base % s).tensors:
        m = re.match(r'blk\.(\d+)\.ffn_(gate|up|down)_exps', t.name)
        if m: eb[int(m.group(1))] += int(t.n_bytes) / NE
prof = np.fromfile(sys.argv[1], dtype=np.int32).reshape(-1, L, K)
ev = np.fromfile(sys.argv[2], dtype=np.int32).reshape(-1, L, K)
W = int(sys.argv[sys.argv.index('--window') + 1]) if '--window' in sys.argv else 6
print(f'profile positions {len(prof)}, eval positions {len(ev)}, expert MiB per layer min/max {eb.min()/2**20:.2f}/{eb.max()/2**20:.2f}, selected GB/token {sum(eb[l]*K for l in range(L))/1e9:.2f}')
cnt = np.zeros((L, NE))
for l in range(L):
    np.add.at(cnt[l], prof[:, l, :].ravel(), 1)
order = sorted(((cnt[l, e] / eb[l], l, e) for l in range(L) for e in range(NE)), reverse=True)  # hits per byte
ev_bytes = sum(eb[l] for p in ev for l in range(L) for _ in range(K))
def tier_set(gib):
    s, used = set(), 0.0
    for _, l, e in order:
        if used + eb[l] > gib * 2**30: continue
        s.add((l, e)); used += eb[l]
    return s
# lane cost per distinct expert used by n positions in the window (ms, Zen1 1-core synthetic widths 1/2/4)
def unit(n): return 0.414 + 0.0985 * (n - 1)
oracle_cnt = np.zeros((L, NE))
for l in range(L): np.add.at(oracle_cnt[l], ev[:, l, :].ravel(), 1)
g0 = sum(unit(1) * eb[l] for p in ev[:len(ev)//W*W] for l in range(L) for e in p[l])
for gib in (0, 2, 3, 4, 5, 6, 7, 8):
    tier = tier_set(gib)
    hit = sum(eb[l] for p in ev for l in range(L) for e in p[l] if (l, e) in tier) / ev_bytes
    w_cost = g_cost = 0.0
    for s in range(0, len(ev) - W + 1, W):
        win = ev[s:s + W]
        for l in range(L):
            c = collections.Counter(int(e) for p in win for e in p[l] if (l, int(e)) not in tier)
            w_cost += sum(unit(n) * eb[l] for n in c.values())
            g_cost += sum(unit(1) * eb[l] for p in win for e in p[l] if (l, int(e)) not in tier)
    print(f'tier {gib} GiB: {len(tier):5d} experts  byte_hit {hit:.3f}  CPU-lane work vs tier-0 greedy: greedy {g_cost/g0:.3f}  window-{W} {w_cost/g0:.3f}')
