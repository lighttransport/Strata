"""Simulate expert residency for GLM decode from routing traces, without loading a model.

Inputs are --routing-trace CSVs (position, layer, eight expert IDs; one decode repetition each) and optionally
STRATA_GLM_ROUTER_SCORE_TRACE files (binary router logits plus a .bias file) from the same runs. Reports:

  tier     share of routes served by a VRAM tier of a given size, static (prior only) and adaptive (decayed
           counts, bounded uploads per token);
  ram      for a host RAM budget below the pack size: routes and bytes per token that miss VRAM and RAM and must
           be read from disk, with CLOCK-style eviction by decayed score;
  affinity share of routes that become resident, and the routing weight that moves, when selection adds a margin
           to resident experts' scores (weights keep the true scores).
"""
import argparse
import collections
import csv
import json
import pathlib

import numpy as np


def read_routes(path):
    """Returns {layer: [(position, ids)]} in trace order."""
    rows = collections.defaultdict(list)
    seen = set()
    with open(path) as file:
        for row in csv.reader(file):
            position, layer, *experts = map(int, row)
            if (position, layer) in seen:
                raise ValueError(f"{path}: repeated position {position} at layer {layer}; trace one repetition")
            seen.add((position, layer))
            rows[layer].append((position, experts))
    return rows


def read_scores(path):
    """Returns ({(position, layer): logits}, {layer: bias})."""
    bias = {}
    data = np.fromfile(str(path) + ".bias", dtype=np.uint8)
    offset = 0
    while offset < len(data):
        layer, experts = np.frombuffer(data, np.int32, 2, offset)
        bias[int(layer)] = np.frombuffer(data, np.float32, experts, offset + 8).copy()
        offset += 8 + 4 * int(experts)
    logits = {}
    data = np.fromfile(path, dtype=np.uint8)
    offset = 0
    while offset < len(data):
        position, layer, experts = np.frombuffer(data, np.int32, 3, offset)
        logits[int(position), int(layer)] = np.frombuffer(data, np.float32, experts, offset + 12)
        offset += 12 + 4 * int(experts)
    return logits, bias


def load_prior(path, experts):
    prior = {}
    for projection in json.load(open(path))["projections"]:
        if projection["name"].endswith(".gu"):
            layer = int(projection["name"].split(".")[1])
            counts = np.array(projection["counts"], dtype=np.float64)
            if len(counts) != experts:
                raise ValueError(f"prior has {len(counts)} experts per layer, expected {experts}")
            prior[layer] = counts / counts.sum()
    return prior


class Residency:
    """Decayed-score residency over (layer, expert) with a VRAM tier and an optional RAM tier."""

    def __init__(self, layers, experts, prior, vram_slots, ram_slots, decay, prior_tokens):
        self.layers, self.experts, self.decay = layers, experts, decay
        self.index = {layer: i for i, layer in enumerate(layers)}
        # The prior enters as prior_tokens pseudo-tokens of routing, as the decoder blends prompt routes.
        self.score = np.stack([prior[layer] * 8 * prior_tokens for layer in layers])
        order = np.argsort(-self.score, axis=None, kind="stable")
        self.vram = np.zeros(self.score.shape, bool)
        self.vram.flat[order[:vram_slots]] = True
        self.ram = np.ones(self.score.shape, bool)
        if ram_slots is not None:
            self.ram[:] = False
            self.ram.flat[order[vram_slots:vram_slots + ram_slots]] = True
        self.ram_slots = ram_slots

    def step(self, routes, adaptive, uploads, admit, margin):
        """routes: [(layer, ids)] of one token. Returns (routes, vram hits, disk misses)."""
        total = hits = misses = 0
        loads = []
        for layer, ids in routes:
            row = self.index[layer]
            for expert in ids:
                total += 1
                if self.vram[row, expert]:
                    hits += 1
                elif not self.ram[row, expert]:
                    misses += 1
                    loads.append((row, expert))
                self.score[row, expert] += 1
        if self.ram_slots is not None:
            for row, expert in loads:
                # Exact mode reads the expert now; the coldest RAM expert that is not in this token's routes leaves.
                self.ram[row, expert] = True
            excess = int(self.ram.sum()) - self.ram_slots
            if excess > 0:
                candidates = np.where(self.ram, self.score, np.inf)
                for row, expert in loads:
                    candidates[row, expert] = np.inf
                victims = np.argpartition(candidates, excess, axis=None)[:excess]
                self.ram.flat[victims] = False
        if adaptive and uploads:
            # Promote the hottest non-resident experts over the coldest resident ones, bounded per token.
            outside = np.where(self.vram | ~self.ram, -np.inf, self.score)
            inside = np.where(self.vram, self.score, np.inf)
            best = np.argpartition(-outside, uploads, axis=None)[:uploads]
            worst = np.argpartition(inside, uploads, axis=None)[:uploads]
            best = best[np.argsort(-outside.flat[best])]
            worst = worst[np.argsort(inside.flat[worst])]
            for candidate, victim in zip(best, worst):
                if outside.flat[candidate] < admit or outside.flat[candidate] < margin * inside.flat[victim]:
                    break
                self.vram.flat[candidate] = True
                self.vram.flat[victim] = False
                if self.ram_slots is not None:
                    # A VRAM expert needs no RAM copy; the victim returns to RAM (its bytes are re-read from disk).
                    self.ram.flat[candidate] = False
                    self.ram.flat[victim] = True
        self.score *= self.decay
        return total, hits, misses


def tokens_of(trace):
    """Yields each position's [(layer, ids)] in position order."""
    by_position = collections.defaultdict(list)
    for layer, rows in trace.items():
        for position, ids in rows:
            by_position[position].append((layer, ids))
    for position in sorted(by_position):
        yield position, sorted(by_position[position])


def simulate(traces, layers, experts, prior, vram_slots, ram_slots, args, adaptive):
    total = hits = misses = tokens = 0
    for trace in traces:
        state = Residency(layers, experts, prior, vram_slots, ram_slots, args.decay, args.prior_tokens)
        for _, routes in tokens_of(trace):
            a, b, c = state.step(routes, adaptive, args.uploads, args.admit, args.margin)
            total += a; hits += b; misses += c; tokens += 1
    return dict(vram_share=hits / total, disk_routes_per_token=misses / tokens, tokens=tokens)


def affinity(traces, score_files, layers, experts, prior, vram_slots, ram_slots, args, margins):
    results = []
    for margin in margins:
        routes = resident = moved = 0
        weight_moved = 0.0
        for trace, score_file in zip(traces, score_files):
            logits, bias = read_scores(score_file)
            state = Residency(layers, experts, prior, vram_slots, ram_slots, args.decay, args.prior_tokens)
            for position, token in tokens_of(trace):
                chosen = []
                for layer, ids in token:
                    row = state.index[layer]
                    p = 1 / (1 + np.exp(-logits[position, layer].astype(np.float64)))
                    score = p + bias[layer]
                    favoured = state.vram[row] if ram_slots is None else (state.vram[row] | state.ram[row])
                    picked = np.argsort(-(score + margin * favoured), kind="stable")[:8]
                    true = set(ids)
                    weights = p[list(true)] / p[list(true)].sum()
                    lost = [e for e in ids if e not in set(picked.tolist())]
                    weight_moved += sum(weights[list(true).index(e)] for e in lost)
                    moved += len(lost)
                    resident += int(favoured[picked].sum())
                    routes += 8
                    chosen.append((layer, picked.tolist()))
                state.step(chosen, True, args.uploads, args.admit, args.margin)
        results.append(dict(margin=margin, resident_share=resident / routes, routes_changed=moved / routes,
                            true_weight_moved=weight_moved / (routes / 8)))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", type=pathlib.Path)
    ap.add_argument("--prior", type=pathlib.Path, required=True, help="calibration coverage.json")
    ap.add_argument("--experts", type=int, default=288)
    ap.add_argument("--expert-mb", type=float, default=8.72, help="mean bytes of one routed expert, in MB")
    ap.add_argument("--vram-gib", type=float, nargs="+", default=[3.5, 5, 6, 6.5, 7, 8, 9])
    ap.add_argument("--ram-gib", type=float, nargs="*", default=[], help="host budgets for experts; omit for all-resident")
    ap.add_argument("--decay", type=float, default=0.98)
    ap.add_argument("--prior-tokens", type=float, default=256)
    ap.add_argument("--uploads", type=int, default=8, help="tier promotions allowed per token")
    ap.add_argument("--admit", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--scores", nargs="*", type=pathlib.Path, default=[], help="router score traces, one per routing trace")
    ap.add_argument("--affinity-margins", type=float, nargs="*", default=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 1e9])
    ap.add_argument("--affinity-vram-gib", type=float, default=6.5)
    ap.add_argument("--output", type=pathlib.Path)
    args = ap.parse_args()

    traces = [read_routes(path) for path in args.traces]
    layers = sorted(traces[0])
    prior = load_prior(args.prior, args.experts)
    slots = lambda gib: int(gib * 2**30 / (args.expert_mb * 1e6))
    report = dict(traces=[str(p) for p in args.traces], tier=[], ram=[], affinity=[])
    for gib in args.vram_gib:
        row = dict(vram_gib=gib, slots=slots(gib),
                   static=simulate(traces, layers, args.experts, prior, slots(gib), None, args, False)["vram_share"],
                   adaptive=simulate(traces, layers, args.experts, prior, slots(gib), None, args, True)["vram_share"])
        report["tier"].append(row)
        print("tier", row, flush=True)
    for gib in args.ram_gib:
        vram = slots(args.affinity_vram_gib)
        result = simulate(traces, layers, args.experts, prior, vram, slots(gib), args, True)
        result.update(ram_gib=gib, vram_gib=args.affinity_vram_gib,
                      disk_gb_per_token=result["disk_routes_per_token"] * args.expert_mb / 1000)
        report["ram"].append(result)
        print("ram", result, flush=True)
    if args.scores:
        if len(args.scores) != len(traces):
            raise SystemExit("one score trace per routing trace")
        for label, ram in [("vram", None)] + [(f"ram{g:g}", slots(g)) for g in args.ram_gib]:
            for row in affinity(traces, args.scores, layers, args.experts, prior, slots(args.affinity_vram_gib), ram,
                                args, args.affinity_margins):
                row["resident"] = label
                report["affinity"].append(row)
                print("affinity", row, flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
