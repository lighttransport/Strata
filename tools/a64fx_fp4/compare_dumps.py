#!/usr/bin/env python3
"""Relative L2 of the runner's residual dumps against the golden reference's (attn-<l>.bin / layer-<l>.bin, F32).

    python3 tools/a64fx_fp4/compare_dumps.py GOLDEN_DUMP_DIR RUNNER_DUMP_DIR [--max 1e-5]

Prints one line per half-layer present in both directories and exits 1 when any exceeds --max. Standard library only
(the compute nodes' python3 has no numpy).
"""
import array
import math
import os
import sys


def load(path):
    a = array.array('f')
    with open(path, 'rb') as f:
        a.frombytes(f.read())
    return a


def main():
    args = sys.argv[1:]
    limit = 1e-5
    if '--max' in args:
        i = args.index('--max')
        limit = float(args[i + 1])
        del args[i:i + 2]
    golden, runner = args
    worst, bad, rows = 0.0, 0, []
    for layer in range(64):
        for kind in ('attn', 'layer'):
            name = '%s-%d.bin' % (kind, layer)
            g, r = os.path.join(golden, name), os.path.join(runner, name)
            if not (os.path.exists(g) and os.path.exists(r)):
                continue
            a, b = load(g), load(r)
            num = sum((x - y) * (x - y) for x, y in zip(a, b))
            den = sum(x * x for x in a) or 1e-30
            rel = math.sqrt(num / den)
            worst = max(worst, rel)
            bad += rel > limit
            rows.append('%s%d %.2e' % (kind[0], layer, rel))
    print(' '.join(rows))
    print('COMPARE half_layers=%d worst=%.2e limit=%.0e %s' % (len(rows), worst, limit, 'FAIL' if bad or not rows else 'PASS'))
    sys.exit(1 if bad or not rows else 0)


if __name__ == '__main__':
    main()
