#!/usr/bin/env python3
"""MiniMax M3.1 Flash decode projection for the V100 smoke node (tools/sim, `minimax_m31` model).

A MODEL PROJECTION, not a measurement on MiniMax: the byte counts are exact (GGUF census), the hardware constants are
measured on this class of node with GLM (see tools/sim/data/hw_v100_m31_node.json), and three inputs are assumptions
that the tables sweep: the VRAM expert-tier hit rate, the CPU expert flow, and the GPU attention chain cost.

    python3 tools/minimax_m31_sim.py                     # the tables in docs/MINIMAX_M31_SIM.md
    python3 tools/minimax_m31_sim.py --json out.json     # the same numbers, machine readable
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

SIM = pathlib.Path(__file__).resolve().parent / "sim"
sys.path.insert(0, str(SIM))
import config  # noqa: E402
import decode  # noqa: E402
import hw as hwmod  # noqa: E402
import kernels  # noqa: E402
import model  # noqa: E402
import routing  # noqa: E402

HW_PATH = SIM / "data" / "hw_v100_m31_node.json"
TARGET = 25.0

# Measured on the first smoke run (single V100, UD-Q2_K_XL, 656 expert slots, 210-token prompt, 64 new tokens):
# decode expert misses = total misses - prompt copies; accesses = 63 decode steps x 57 layers x 4 experts.
MEASURED = dict(slots=656, decode_steps=63, misses_total=13493, prompt_copies=5083, decode_ms_per_token=1823413.9 / 64)
MEASURED["decode_hit"] = 1 - (MEASURED["misses_total"] - MEASURED["prompt_copies"]) / (MEASURED["decode_steps"] * 57 * 4)


def load_hw(overrides=()):
    h = hwmod.load(str(HW_PATH))
    for item in overrides:
        h.set(*item.split("=", 1))
    return h


def run(model_key, pack, hw, params, hit=None, **kw):
    """Decode tok/s for one configuration. `hit` forces the VRAM tier's byte hit share (None: the GLM-fitted curve)."""
    model.use_model(model_key)
    base = dict(pack=pack, context=8192, prompt=210, generate=64, speculation="none", mtp_depth=0, placement="cpu",
                tier_policy="lru", dense_format="orig", threads=35, split_verify=False, decode_cache_mib=-1,
                prefill_scratch_mib=512, reserve_mib=512, measured_hit_share=-1.0 if hit is None else hit)
    base.update(kw)
    cfg = config.RunConfig(**base)
    return decode.simulate(hw, cfg, params), cfg


def tier_slots_and_curve_hit(model_key, pack, hw, params, **kw):
    d, cfg = run(model_key, pack, hw, params, **kw)
    return d.tier_slots, d.tier_mib, d.hit_bytes_share


def solve(fn, lo, hi, target, iters=40):
    """Smallest x in [lo, hi] with fn(x) >= target (fn nondecreasing), or None."""
    if fn(hi) < target:
        return None
    if fn(lo) >= target:
        return lo
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if fn(mid) >= target else (mid, hi)
    return hi


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=pathlib.Path)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="hardware override, e.g. memory.dram_gbps=100")
    ap.add_argument("--measured", action="store_true",
                    help="print the model fitted on the measured strata-m3 path (tools/minimax_m31_fit.py) next to the GLM-fitted projection and exit")
    args = ap.parse_args(argv)
    if args.measured:
        # the projection below was fitted on GLM; the measured path (FP16 K/V, DSpark, side streams, fused experts, a second GPU) is in
        # tools/sim/data/minimax_m31_measured.json and modelled by tools/sim/minimax_step.py
        import minimax_step as mstep
        p, _ = mstep.fit(mstep.Params())
        hw0 = load_hw(args.set)
        d, _ = run("minimax_m31", "minimax_m31_ud_q2_k_xl", hw0, kernels.Params(), hit=0.565, pcie_prefetch=True)
        print(f"GLM-fitted projection, plain decode, curve-scaled hit, prefetch: {d.tok_s:.1f} tok/s (the engine then measured 16.0 plain)")
        print("measured single-stream decode with DSpark speculation (mean of two prompts) vs the model fitted on it:")
        for gpus, mode, meas, tok, step in mstep.check_anchors(p):
            print(f"  {gpus} GPU  {mode:5s} measured {meas:5.1f}  fitted model {tok:5.1f} tok/s  ({step:5.1f} ms a step)")
        model.use_model("flash")
        return 0
    hw = load_hw(args.set)
    params = kernels.Params()
    key, pack = "minimax_m31", "minimax_m31_ud_q2_k_xl"
    out = {"measured_anchor": MEASURED, "label": "model projection, not measured on MiniMax"}

    # --- the VRAM tier of an engine built for decode (small prompt chunk, FP16 KV), and the hit-rate assumptions
    slots, tier_mib, curve_hit = tier_slots_and_curve_hit(key, pack, hw, params)
    total = 57 * 128
    cover = slots / total
    # the GLM-fitted curve evaluated at the measured engine's 656 slots, to scale it onto the one M3.1 point we have
    model.use_model(key)
    curve = routing.TierCurve()
    at_measured = curve.hit(MEASURED["slots"] / total, policy="lru", skew=1.0)
    scale = MEASURED["decode_hit"] / at_measured
    hits = {
        "uniform routing": cover,
        "GLM-fitted LRU curve": curve_hit,
        "curve scaled to the measured M3.1 point": min(1.0, curve.hit(cover, policy="lru", skew=1.0) * scale),
    }
    out["tier"] = dict(slots=slots, tier_mib=tier_mib, coverage=cover, measured_point_curve_value=at_measured,
                       scale=scale, hits=hits)
    print(f"VRAM expert tier for a decode-oriented engine: {slots} experts ({tier_mib:.0f} MiB, {cover * 100:.1f} % of {total})")
    print(f"measured decode hit at {MEASURED['slots']} slots ({MEASURED['slots'] / total * 100:.1f} %): "
          f"{MEASURED['decode_hit'] * 100:.1f} %  (GLM curve there: {at_measured * 100:.1f} %)")
    print("hit assumptions:", {k: f"{v * 100:.1f} %" for k, v in hits.items()})

    # --- scenarios x hit assumptions
    scenarios = {
        "CPU-lane experts": dict(),
        "+ next-layer PCIe prefetch": dict(pcie_prefetch=True),
    }
    rows = []
    print("\nDecode tok/s, no speculation (CPU flow %.1f GB/s):" % hw.memory.dram_gbps)
    for name, kw in scenarios.items():
        for hname, h in hits.items():
            d, _ = run(key, pack, hw, params, hit=h, **kw)
            rows.append(dict(scenario=name, hit=hname, hit_share=h, tok_s=d.tok_s, cpu_ms=d.step.cpu_ms,
                             gpu_ms=d.step.gpu_ms, gpu_exposed_ms=d.step.gpu_exposed_ms, step_ms=d.round_ms,
                             bottleneck=d.bottleneck, cpu_gb=d.cpu_gb_per_token))
            print(f"  {name:28s} {hname:42s} hit {h * 100:5.1f} %  {d.tok_s:6.2f} tok/s  "
                  f"(CPU {d.step.cpu_ms:5.1f} ms, GPU {d.step.gpu_ms:5.1f} ms, exposed {d.step.gpu_exposed_ms:5.1f}; {d.bottleneck})")
    out["scenarios"] = rows

    central = hits["curve scaled to the measured M3.1 point"]
    low = hits["uniform routing"]

    # --- the GPU dense chain is the bottleneck of the plain-decode rows above; its parameters are assumptions
    profiles = {
        "as fitted on GLM": (params, hw),
        "graph-captured chain (25 kernels/layer, half the exposed launch cost)": (
            dataclasses.replace(params, gpu_kernels_per_layer=25, gpu_launch_scale=0.5, gpu_layer_fixed_us=30.0), hw),
    }
    hw90 = hw.copy(); hw90.set("gpu.gemv_efficiency", 0.90); hw90.set("gpu.tier_gbps", 400.0)
    profiles["+ GEMV 90 % of 576 GB/s and 400 GB/s expert kernel"] = (profiles[
        "graph-captured chain (25 kernels/layer, half the exposed launch cost)"][0], hw90)
    print("\nGPU chain assumptions, plain decode, prefetch on (columns: uniform routing / curve-scaled hit):")
    out["profiles"] = []
    for name, (pp, hh) in profiles.items():
        r = [run(key, pack, hh, pp, hit=h, pcie_prefetch=True)[0] for h in (low, central)]
        out["profiles"].append(dict(profile=name, tok_s=[x.tok_s for x in r], gpu_ms=[x.step.gpu_ms for x in r],
                                    cpu_ms=[x.step.cpu_ms for x in r]))
        print(f"  {name:70s} {r[0].tok_s:6.2f} / {r[1].tok_s:6.2f} tok/s  (GPU {r[1].step.gpu_ms:5.1f} ms, CPU {r[1].step.cpu_ms:5.1f} ms)")

    # --- speculation with the DSpark draft head: acceptance and the draft pass cost are assumptions
    pp, hh = profiles["+ GEMV 90 % of 576 GB/s and 400 GB/s expert kernel"]
    draft_ms = 2.0e9 / (hh.gpu.bandwidth_gbps * hh.gpu.gemv_efficiency * 1e9) * 1e3 + 1.0   # Q8_0 head read once + 1 ms
    print(f"\nDSpark speculation on the last profile (draft head 2.0 GB Q8_0 read per round ~ {draft_ms:.1f} ms, assumed):")
    print("  tok/s  rows: draft tokens n; columns: per-position acceptance p; hit = curve-scaled / uniform")
    spec = []
    for hname, h in (("curve-scaled hit", central), ("uniform routing", low)):
        print(f"  {hname}")
        print("    n \\ p   " + "".join(f"{p:7.2f}" for p in (0.5, 0.7, 0.85)))
        for n in (1, 2, 3, 4):
            vals = []
            for pa in (0.5, 0.7, 0.85):
                d, _ = run(key, pack, hh, pp, hit=h, pcie_prefetch=True, speculation="external", mtp_depth=n,
                           acceptance_probabilities=[pa] * n, external_draft_ms=draft_ms, proposal_probability=1.0)
                vals.append(d.tok_s)
            spec.append(dict(hit=hname, n=n, p=[0.5, 0.7, 0.85], tok_s=vals))
            print(f"    {n}      " + "".join(f"{v:7.1f}" for v in vals))
    out["speculation"] = spec

    # --- what 25 tok/s needs from the last profile
    print(f"\nWhat {TARGET:.0f} tok/s needs on the last profile (prefetch on):")
    need = {}
    for n in (1, 2, 3):
        for hname, h in (("curve-scaled hit", central), ("uniform routing", low)):
            pa = solve(lambda x: run(key, pack, hh, pp, hit=h, pcie_prefetch=True, speculation="external", mtp_depth=n,
                                     acceptance_probabilities=[x] * n, external_draft_ms=draft_ms,
                                     proposal_probability=1.0)[0].tok_s, 0.0, 1.0, TARGET)
            need[f"p_needed_n{n}_{hname}"] = pa
            print(f"  draft n={n}, {hname}: per-position acceptance >= " + ("unreachable" if pa is None else f"{pa:.2f}"))
    out["needs"] = need

    # --- the smaller pack
    print("\nUD-IQ1_M pack (3.74 GB/token of experts), last profile, curve-scaled hit, prefetch, plain decode:")
    k2 = "minimax_m31_ud_iq1_m"
    s2, t2, _ = tier_slots_and_curve_hit(k2, k2, hh, pp)
    d, _ = run(k2, k2, hh, pp, hit=min(1.0, central * (s2 / slots) ** 0.1), pcie_prefetch=True)
    out["iq1_m"] = dict(slots=s2, tok_s=d.tok_s, cpu_ms=d.step.cpu_ms, gpu_ms=d.step.gpu_ms)
    print(f"  {s2} slots: {d.tok_s:.2f} tok/s (CPU {d.step.cpu_ms:.1f} ms, GPU {d.step.gpu_ms:.1f} ms)")
    print("  with DSpark speculation, n=2 draft tokens (per-position acceptance 0.5 / 0.7 / 0.85):")
    iq1_spec = []
    for pa in (0.5, 0.7, 0.85):
        ds, _ = run(k2, k2, hh, pp, hit=min(1.0, central * (s2 / slots) ** 0.1), pcie_prefetch=True, speculation="external",
                    mtp_depth=2, acceptance_probabilities=[pa] * 2, external_draft_ms=draft_ms, proposal_probability=1.0)
        iq1_spec.append(ds.tok_s)
    out["iq1_m"]["speculation_n2"] = dict(p=[0.5, 0.7, 0.85], tok_s=iq1_spec)
    print("    " + "  ".join(f"{v:.1f}" for v in iq1_spec) + " tok/s")
    model.use_model("flash")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
