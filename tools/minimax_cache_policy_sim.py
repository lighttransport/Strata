#!/usr/bin/env python3
"""Replay a strata-m3 routing trace (STRATA_M3_ROUTE_TRACE) through VRAM expert-cache policies.

A trace line is "step layer e0 e1 e2 e3".  The cache starts as the engine's prefill leaves it (an LRU that saw every
expert of each layer in order, so it holds the last layers' experts).  Per decode step and layer, experts already
resident are hits; the misses run on the CPU, and up to `admit` of them (in expert-id order, as the engine picks) are
copied into the cache on a background stream, evicting the least recently used slot that this layer did not just use.
Policies: lru (admit every candidate), freq (admit only when the candidate was seen at least as often as the slot's
current owner), freq2 (admit only experts seen before at least once).

    python3 tools/minimax_cache_policy_sim.py trace.txt [--slots 1019] [--layers 3-59] [--admit 1,2,3]
"""
import argparse
import collections
import sys


def load(path):
    steps = collections.defaultdict(dict)
    for line in open(path):
        f = line.split()
        if len(f) < 3:
            continue
        steps[int(f[0])][int(f[1])] = [int(x) for x in f[2:] if int(x) >= 0]   # -1 = an expert dropped by a routing lever
    return [steps[s] for s in sorted(steps)]


def simulate(trace, slots, n_experts, first_layer, last_layer, admit, policy, decay_every=0):
    # prefill leaves the last `slots` accesses of an in-order sweep: whole layers from the end
    order = [(l, e) for l in range(first_layer, last_layer + 1) for e in range(n_experts)]
    resident = collections.OrderedDict((k, True) for k in order[-slots:])   # LRU order: oldest first
    freq = collections.Counter()
    hits = uses = admitted = 0
    for step, layers in enumerate(trace):
        for l in sorted(layers):
            picks = sorted(layers[l])
            used_now = set()
            miss = []
            for e in picks:
                uses += 1
                freq[(l, e)] += 1
                if (l, e) in resident:
                    hits += 1
                    resident.move_to_end((l, e))
                    used_now.add((l, e))
                else:
                    miss.append(e)
            done = 0
            for e in miss:
                if done >= admit:
                    break
                victim = next((k for k in resident if k not in used_now), None)
                if victim is None:
                    break
                if policy == "freq" and freq[(l, e)] < freq[victim]:
                    continue
                if policy == "freq2" and freq[(l, e)] < 2:
                    continue
                del resident[victim]
                resident[(l, e)] = True
                used_now.add((l, e))
                done += 1
                admitted += 1
        if decay_every and (step + 1) % decay_every == 0:
            for k in list(freq):
                freq[k] //= 2
    return hits / max(1, uses), admitted / max(1, len(trace))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace")
    ap.add_argument("--slots", type=int, default=1019)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--layers", default="3-59")
    ap.add_argument("--admit", default="0,1,2,3")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.layers.split("-"))
    trace = load(args.trace)
    print(f"{len(trace)} decode steps, {args.slots} slots ({args.slots / ((hi - lo + 1) * args.experts) * 100:.1f} % of the experts)")
    print(f"{'policy':8s} {'admit/layer':>11s} {'hit rate':>9s} {'admitted/step':>14s}")
    for policy in ("lru", "freq", "freq2"):
        for a in (int(x) for x in args.admit.split(",")):
            if a == 0 and policy != "lru":
                continue
            hit, adm = simulate(trace, args.slots, args.experts, lo, hi, a, policy)
            print(f"{policy:8s} {a:11d} {hit * 100:8.1f}% {adm:14.1f}")


if __name__ == "__main__":
    sys.exit(main())
