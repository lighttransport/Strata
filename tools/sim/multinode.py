"""Multinode split search: which share of the routed expert work each node should take (decoder node + up to 3 workers).

Two split kinds are scored with the decode model:
- row: every CPU-routed expert's FFN rows are split across nodes (what `tools/glm_remote_tp.sh` implements for one
  worker; shares are multiples of 0.25 because the decoder keeps 512-row blocks of the 2048 rows);
- expert: whole experts per node (not implemented; same message shape, uneven per-node load from routing).
A pipeline (layer) split is not scored: stages run in series, so a single stream gains nothing from it.
Predictions carry the remote-TP validation band (about +-30 % for one worker on InfiniBand); more than one worker is
unvalidated and the engine supports one worker today.
"""
import dataclasses
import itertools

import decode
import model
from model import GEOMETRY


def share_grid(workers, mode):
    """Candidate per-worker shares: row split in 0.25 steps (local share >= 0.25), expert split on a 0.05 grid."""
    steps = [0.25, 0.5, 0.75] if mode == "row" else [round(0.05 * i, 2) for i in range(1, 16)]
    cap = 0.75 if mode == "row" else 0.8
    for combo in itertools.product([0.0] + steps, repeat=workers):
        if 0 < sum(combo) <= cap + 1e-9:
            yield combo


def shipping_seconds(hw, pack, share, node):
    """Start-up cost of sending a worker its rows from the decoder's disk over the link (or reusing kept rows)."""
    gib = pack.routed_total_bytes() * share
    rate = min(node.link.gbps, hw.disk.read_gbps) * 1e9 if node.link.gbps > 0 else hw.disk.read_gbps * 1e9
    return gib / max(rate, 1e-6)


def search(hw, workers, cfg, params, mode="row", top=10):
    """Rank share assignments for `cfg` on decoder `hw` with the given WorkerNode list; returns (rows, baseline)."""
    pack = model.pack(cfg.pack)
    multi = hw.copy()
    multi.workers = list(workers)
    base = decode.simulate(hw, dataclasses.replace(cfg, remote_share=0.0, remote_shares=""), params)
    rows = []
    for combo in share_grid(len(workers), mode):
        c = dataclasses.replace(cfg, remote_share=sum(combo), remote_shares=",".join(f"{s:g}" for s in combo), split_mode=mode)
        local_gib = pack.routed_total_bytes() / 2 ** 30 * (1 - sum(combo))
        if local_gib + 7 > hw.memory.gib + hw.memory.page_cache_gib and cfg.placement == "cpu":
            continue
        if any(s > 0 and pack.routed_total_bytes() / 2 ** 30 * s + 2 > w.memory.gib + w.memory.page_cache_gib for s, w in zip(combo, workers)):
            continue
        d = decode.simulate(multi, c, params)
        rows.append(dict(shares=combo, config=c, decode=d.tok_s, gain=d.tok_s / base.tok_s - 1,
                         bottleneck=d.step.bottleneck, remote_ms=d.step.remote_ms, cpu_ms=d.step.cpu_ms,
                         local_gib=local_gib, worker_gib=[pack.routed_total_bytes() / 2 ** 30 * s for s in combo],
                         shipping_s=max(shipping_seconds(hw, pack, s, w) for s, w in zip(combo, workers)),
                         notes=d.notes))
    rows.sort(key=lambda r: -r["decode"])
    return rows[:top], base


def describe(rows, base, hw, workers, mode):
    lines = [f"decoder {hw.name} ({hw.memory.gib:.0f} GiB, {hw.memory.dram_gbps:.0f} GB/s DRAM) + "
             + ", ".join(f"{w.name} ({w.memory.gib:.0f} GiB, {w.memory.dram_gbps:.0f} GB/s, {w.link.name} rtt {w.link.rtt_us:.0f} us)" for w in workers),
             f"single node: {base.tok_s:.2f} tok/s ({base.step.bottleneck}); split: {mode}"]
    if not rows:
        lines.append("no share assignment fits the nodes' RAM")
        return "\n".join(lines)
    lines.append(f"{'shares':<20}{'decode':>8}{'gain':>8}  {'bottleneck':<12}{'remote_ms':>10}{'local_GiB':>10}  worker_GiB        ship_s")
    for r in rows:
        lines.append(f"{'/'.join(f'{s:g}' for s in r['shares']):<20}{r['decode']:>8.2f}{r['gain'] * 100:>7.1f}%  {r['bottleneck']:<12}"
                     f"{r['remote_ms']:>10.1f}{r['local_gib']:>10.1f}  {'/'.join(f'{g:.1f}' for g in r['worker_gib']):<16}{r['shipping_s']:>7.0f}")
    best = rows[0]
    lines.append(f"best: shares {'/'.join(f'{s:g}' for s in best['shares'])} -> {best['decode']:.2f} tok/s ({best['gain'] * 100:+.1f}% vs single node)")
    lines.append("validation: one InfiniBand worker measured within about 30 % of the model (docs/GLM_SIM.md); "
                 "several workers and the expert split are unvalidated, and the engine runs one worker today.")
    return "\n".join(lines)
