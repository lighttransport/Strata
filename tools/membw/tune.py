#!/usr/bin/env python3
"""Sequential local-NUMA benchmark sweeps; saves every configuration, never runs competitors concurrently."""
import argparse
import itertools
import json
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('read', 'decode', 'confirm'), default='read')
    parser.add_argument('--mib', type=int, default=2048)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(__file__).with_name(f'{args.suite}-tuning-results.json')
    binary = Path(__file__).resolve().with_name('membw')
    if args.suite != 'read':
        results = []
        if args.suite == 'confirm':
            cases = [('read-int', n, False, 256, 1, 0) for n in (8, 16, 8)]
            cases += [('read', 8, False, 256, 1, 0), ('read-sse', 8, False, 256, 1, 0)]
        else:
            cases = [('decode-stream', n, False, 256, 1, 0) for n in (8, 16)]
            cases += [('decode', n, huge, block, layers, pf)
                      for n in (8, 16)
                      for huge, block, layers, pf in (
                          (False, 256, 1, 0), (True, 256, 1, 0),
                          (False, 4, 1, 0), (True, 4, 1, 0),
                          (False, 256, 64, 0), (True, 256, 64, 0),
                          (False, 256, 64, 2048), (True, 256, 64, 2048))]
        for kernel, n, huge, block, layers, pf in cases:
            cmd = ['numactl', '--localalloc', str(binary), '--spread', '--threads', str(n),
                   '--mib', str(args.mib), '--repeats', str(args.repeats), '--kernel', kernel,
                   '--unroll', '16', '--block-kib', str(block), '--layers', str(layers), '--prefetch', str(pf)]
            if huge: cmd.append('--huge')
            run = subprocess.run(cmd, capture_output=True, text=True, check=True)
            match = re.search(r': median ([\d.]+), best ([\d.]+), min ([\d.]+) GB/s', run.stdout)
            if not match:
                raise RuntimeError(run.stdout + run.stderr)
            entry = dict(kernel=kernel, threads=n, huge=huge, block_kib=block, layers=layers, prefetch=pf,
                         median=float(match[1]), best=float(match[2]), minimum=float(match[3]),
                         command=cmd, stdout=run.stdout)
            results.append(entry)
            args.output.write_text(json.dumps(results, indent=2) + '\n')
            print(f'{kernel} threads={n} huge={huge} block={block} layers={layers} pf={pf}: {entry["median"]:.2f} GB/s', flush=True)
        return
    results = []
    # First compare page size and loop shape, then prefetch the best of those.
    configurations = [(h, u, 0) for h, u in itertools.product((False, True), (4, 8, 16))]
    for phase in range(2):
        for huge, unroll, prefetch in configurations:
            cmd = ['numactl', '--localalloc', str(binary), '--threads', '16',
                   '--mib', str(args.mib), '--repeats', str(args.repeats),
                   '--kernel', 'read', '--unroll', str(unroll), '--prefetch', str(prefetch)]
            if huge:
                cmd.append('--huge')
            run = subprocess.run(cmd, capture_output=True, text=True, check=True)
            match = re.search(r'read: median ([\d.]+), best ([\d.]+), min ([\d.]+)', run.stdout)
            if not match:
                raise RuntimeError(run.stdout + run.stderr)
            entry = dict(huge=huge, unroll=unroll, prefetch=prefetch,
                         median=float(match[1]), best=float(match[2]), minimum=float(match[3]),
                         command=cmd, stdout=run.stdout)
            results.append(entry)
            args.output.write_text(json.dumps(results, indent=2) + '\n')
            print(f'huge={huge} unroll={unroll} prefetch={prefetch}: median {entry["median"]:.2f} GB/s', flush=True)
        best = max(results, key=lambda r: r['median'])
        configurations = [(best['huge'], best['unroll'], pf) for pf in (256, 512, 1024, 2048, 4096, 8192)]
    best = max(results, key=lambda r: r['median'])
    print('Best measured configuration:', json.dumps({k: best[k] for k in ('huge', 'unroll', 'prefetch', 'median', 'best')}))


if __name__ == '__main__':
    main()
