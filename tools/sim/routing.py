"""Routing statistics: how many distinct expert bytes a window of tokens touches, what share a GPU or RAM tier
serves, and how MTP drafts are accepted.

Two sources: the synthetic model (parameters fitted to measurements in docs/GLM_Q2_DECODE_REDESIGN.md and
docs/GLM_BATCH_DECODE.md) and replay of real routing traces through tools/glm_residency_sim.py.
"""
import bisect
import collections
import dataclasses
import functools
import json
import math
import pathlib
import sys

TOOLS = pathlib.Path(__file__).resolve().parents[1]

# Distinct-expert bytes of a window of consecutive tokens relative to one token. Measured in the engine on the
# q23 pack: 1.82x / 2.45x / 3.27x for 2 / 3 / 4 tokens; the routing traces in build-q2-v3 give 1.78 / 2.46 / 3.10
# and, for wider windows, 4.3 (6), 5.3 (8), 7.1 (12), 8.7 (16) -- a power law 1.05 * w^0.763. Independent uniform
# routing would give 1.97x / 2.92x / 7.26x (2 / 3 / 8 tokens); consecutive tokens share more.
UNION_RATIO = {1: 1.0, 2: 1.80, 3: 2.46, 4: 3.15}
UNION_POWER = (1.05, 0.763)

# Greedy acceptance probability of draft position i (i = 1..7) for several measured fixtures
# (accepted / proposed from the MTP sweeps; the per-position split is inferred from tokens per round).
ACCEPTANCE = {
    "prime":  [0.97, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70],    # 87/92 at depth 2, 3.76 tokens/round at depth 3
    "json":   [0.96, 0.94, 0.88, 0.82, 0.78, 0.72, 0.68],    # 137/144 at depth 2
    "csv":    [0.85, 0.74, 0.66, 0.60, 0.55, 0.50, 0.45],    # 300/380 at depth 2
    "coding": [0.70, 0.55, 0.45, 0.40, 0.35, 0.30, 0.25],    # REAP 1K: 131/246 at depth 2
    "mixed":  [0.90, 0.82, 0.75, 0.70, 0.65, 0.60, 0.55],
    # DFlash2 block drafter (block 8): accepted lengths 4.10 MT-Bench, 4.39 HumanEval, 5.46 GSM8K (z-lab model card);
    # a flat per-position probability p gives sum_{i=1..7} p^i = that length.
    "selfspec": [0.86] * 15,         # DraftExpert: 84-87 % draft acceptance
    "dflash_chat": [0.86] * 15,
    "dflash_code": [0.885] * 15,
    "dflash_math": [0.935] * 15,
}


def tokens_per_round(depth, profile="mixed"):
    """Expected tokens delivered by one MTP round of `depth` drafts: the anchor plus the accepted prefix."""
    probs = ACCEPTANCE[profile]
    expected, survive = 1.0, 1.0
    for i in range(depth):
        survive *= probs[min(i, len(probs) - 1)]
        expected += survive
    return expected


def union_ratio(width, experts=288, top_k=8, consecutive=True):
    """Distinct experts touched per layer by `width` tokens, in units of one token's top_k."""
    if width <= 1:
        return 1.0
    independent = experts * (1 - (1 - top_k / experts) ** width) / top_k
    if not consecutive:
        return independent
    if width in UNION_RATIO:
        measured = UNION_RATIO[width]
    else:
        a, b = UNION_POWER
        measured = a * width ** b
    return min(measured, independent)


@functools.lru_cache(maxsize=1)
def lru_table():
    """Warm LRU replay table: coverage -> share of routes, cold-start misses removed (data/lru_curve.json)."""
    data = json.loads((pathlib.Path(__file__).resolve().parent / "data" / "lru_curve.json").read_text())
    pts = data["points"]
    full = pts[-1][1]
    return [p[0] for p in pts], [min(1.0, p[1] / full) for p in pts]


def lru_replay_hit(fraction):
    xs, ys = lru_table()
    if fraction <= xs[0]:
        return ys[0] * fraction / xs[0]
    if fraction >= xs[-1]:
        return 1.0
    i = bisect.bisect_right(xs, fraction)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (fraction - x0) / (x1 - x0)


@dataclasses.dataclass
class TierCurve:
    """Share of routed bytes served by a tier holding a fraction f of all expert slots.

    Static prior fill on the q23 pack measured 18.1 / 20.5 / 25.5 % at 400 / 500 / 750 slots of 12,096
    (docs/GLM_Q2_DECODE_REDESIGN.md); h = a * f^b fits those, saturating at 1.
    """
    a: float = 1.16
    b: float = 0.545
    adaptive_gain: float = 1.08     # adaptive tier reads 3-7 % fewer CPU bytes than the static prior fill
    static_gain: float = 0.5        # static tier ranked by the prompt's routes only: 6.8 % vs 15 % adaptive
    lru_gain: float = 1.22          # ideal per-layer LRU replayed on the build-q2-v3 traces: 26.9 % at 4.7 % of
                                    # slots, 41.2 % at 10 %, 57.7 % at 20 % (adaptive curve 23.7 / 35.7 / 52.1)
    affinity_tau: float = 0.22      # affinity margin x: resident share rises as 1 - exp(-x / tau)
    knee: float = 0.05              # power law below (fitted to the 3-6 % tier measurements), trace replay above

    def hit(self, fraction, policy="adaptive", affinity=0.0, skew=1.0, adaptive=None):
        """policy: static (prompt routes), static_prior (calibration prior), adaptive."""
        if adaptive is not None:
            policy = "adaptive" if adaptive else "static_prior"
        if fraction <= 0:
            return 0.0
        h = min(1.0, self.a * skew * fraction ** self.b)
        if fraction > self.knee:
            # above the calibrated range, follow the shape of the trace replay: scaled to meet the power law at the
            # knee and to reach full residency at full coverage
            scale = self.a * skew * self.knee ** self.b / lru_replay_hit(self.knee)
            s = scale + (1 - scale) * (fraction - self.knee) / (1 - self.knee)
            h = min(1.0, lru_replay_hit(fraction) * s)
        if policy == "adaptive":
            h = min(1.0, h * self.adaptive_gain)
        elif policy == "lru":
            h = min(1.0, h * self.lru_gain)
        elif policy == "static":
            h = h * self.static_gain
        if affinity > 0:
            h = h + (1 - h) * (1 - math.exp(-affinity / self.affinity_tau))
        return min(1.0, h)


def read_routes(path):
    """position -> [(layer, ids)] from a --routing-trace CSV (position, layer, e0..e7)."""
    sys.path.insert(0, str(TOOLS))
    from glm_residency_sim import read_routes as _read  # noqa: E402
    rows = _read(path)
    by_pos = collections.defaultdict(list)
    for layer, entries in rows.items():
        for position, ids in entries:
            by_pos[position].append((layer, ids))
    return {p: sorted(v) for p, v in sorted(by_pos.items())}


def trace_union_ratio(trace, width):
    """Measured distinct-expert ratio over sliding windows of `width` consecutive positions."""
    positions = sorted(trace)
    if width <= 1 or len(positions) < width:
        return 1.0
    one_token = distinct = 0
    for start in range(len(positions) - width + 1):
        per_layer = collections.defaultdict(set)
        for p in positions[start:start + width]:
            for layer, ids in trace[p]:
                per_layer[layer].update(ids)
        distinct += sum(len(s) for s in per_layer.values())
        one_token += sum(len(ids) for _, ids in trace[positions[start]])
    return distinct / one_token


def trace_tier_hit(trace, prior_path, slots, adaptive, expert_mb=9.1, decay=0.98, uploads=16, admit=2.0,
                   margin=1.5, prior_tokens=256):
    """Replay a routing trace through glm_residency_sim.Residency; returns the share of routes served."""
    sys.path.insert(0, str(TOOLS))
    from glm_residency_sim import Residency, load_prior  # noqa: E402
    layers = sorted({layer for routes in trace.values() for layer, _ in routes})
    prior = load_prior(prior_path, 288)
    state = Residency(layers, 288, prior, slots, None, decay, prior_tokens)
    total = hits = 0
    for _, routes in trace.items():
        a, b, _ = state.step(routes, adaptive, uploads, admit, margin)
        total += a
        hits += b
    return hits / total if total else 0.0


def default_trace_paths(root=TOOLS.parent):
    return sorted((root / "build-q2-v3").glob("trace-*.routes"))


def default_prior_path(root=TOOLS.parent):
    return root / "build-q2-redesign" / "calibration" / "coverage.json"
