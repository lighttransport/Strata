#!/usr/bin/env python3
"""diverge.py NAME...: first generated index where a run's tokens differ from the CUDA exact reference, and the share of equal positions."""
import sys
ref = open('logs/ref_cuda_tokens.txt').read().split()[1:]
for n in sys.argv[1:]:
    line = next((l for l in open(f'logs/{n}.log') if l.startswith('tokens:')), None)
    if not line:
        print(n, 'no tokens'); continue
    t = line.split()[1:]
    m = min(len(t), len(ref))
    first = next((i for i in range(m) if t[i] != ref[i]), None)
    same = sum(a == b for a, b in zip(t, ref)) / m
    print(f'{n}: generated {len(t)}, first difference at {first}, equal positions {same:.0%}')
