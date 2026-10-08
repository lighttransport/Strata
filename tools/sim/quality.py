"""Rough quality model: estimated mean KL divergence to the BF16 model for a configuration.

Measured anchors (held-out corpus, q23 pack unless noted; docs/GLM_Q2_DECODE_REDESIGN.md, GLM_Q2_REQUANTIZATION.md):
  q23 pack                                KL 0.137, ppl 4.825, top-1 87.7 %
  affinity 0.05 / 0.10 / 0.15             KL 0.143 / 0.165 / 0.198
  q22 pack (Q2_K down)                    KL 0.164, ppl 4.867
  64 GB hybrid margin 0.10                KL 0.150
  UD-Q2_K_XL original                     KL ~0.38 (unsloth table, different corpus; scaled to ours as ~0.14)
Borrowed or guessed (marked):
  REAP-50                                 ppl 11.1 vs 8.57 on the REAP corpus: treated as +0.25 KL (guess)
  expert skip                             +0.12 KL per 100 % skipped bytes (guess from ACE/MoBiLE small losses)
  verifier tapering, tail affinity        only rejected-or-accepted tokens change; AcceptMoE -0.27 pt -> +0.005
  expert deferral                         KTransformers <= 0.5 % accuracy -> +0.03 x share/0.3 (guess)
  Q4 dense copies                         +0.01 (guess); tier compression 1.4x +0.02 (guess)
  frozen RAM set                          top-1 79.8 % measured -> +0.25
The numbers add; they are for ranking configurations, not for publishing quality claims.
"""
import math

BASE = {"q23": 0.137, "q22": 0.164, "q2_orig": 0.14, "reap50_q23": 0.39, "reap50_q4km": 0.33, "q8": 0.03, "exl3": 0.10}
AFFINITY = [(0.0, 0.0), (0.05, 0.006), (0.10, 0.028), (0.15, 0.061), (0.20, 0.10)]


def _interp(points, x):
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    (x0, y0), (x1, y1) = points[-2], points[-1]
    return y1 + (y1 - y0) * (x - x1) / (x1 - x0)


def estimate_kl(cfg):
    """Returns (kl, breakdown dict)."""
    parts = {"pack": BASE.get(cfg.pack, 0.15)}
    if cfg.affinity > 0:
        parts["affinity"] = _interp(AFFINITY, cfg.affinity)
        if cfg.affinity_rank_lo > 1:
            # the eligible routes are the lowest-weight ones: their share of routes, at half the per-route cost (guess)
            parts["affinity"] *= 0.5 * max(0, 8 - cfg.affinity_rank_lo + 1) / 8
    if cfg.tail_affinity > cfg.affinity:
        parts["tail_affinity"] = 0.5 * (_interp(AFFINITY, cfg.tail_affinity) - _interp(AFFINITY, cfg.affinity))
    if cfg.expert_skip > 0:
        parts["expert_skip"] = 0.12 * cfg.expert_skip / 0.1 * 0.1
    if cfg.spec_tail_topk < 8:
        parts["tail_topk"] = 0.005 * math.log2(8 / cfg.spec_tail_topk)
    if cfg.expert_deferral:
        parts["deferral"] = 0.03 * cfg.deferral_share / 0.3
    if cfg.dense_format == "q4":
        parts["dense_q4"] = 0.01
    if cfg.tier_compress > 1:
        parts["tier_compress"] = 0.05 * (cfg.tier_compress - 1)
    if cfg.placement == "ram_tier" and cfg.ram_mode == "frozen":
        parts["ram_frozen"] = 0.25
    if cfg.placement == "ram_tier" and cfg.ram_mode == "hybrid":
        parts["ram_hybrid"] = _interp(AFFINITY, cfg.ram_margin) * 0.5
    if cfg.cold_share > 0:
        parts["cold_experts"] = 0.08 * cfg.cold_share * (1 - cfg.cold_scale) / 0.4   # guess: IQ1-class cold experts
    if cfg.cold_layers > 0:
        parts["cold_layers"] = 0.08 * (cfg.cold_layers / 42) * (1 - cfg.cold_scale) / 0.4   # guess, same rate as cold experts
    return sum(parts.values()), parts
