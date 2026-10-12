#!/usr/bin/env python3
"""Rank GLM-5.3 routed experts by REAP-style saliency and write a pruning plan for strata-glm-prune-experts.

REAP (arXiv 2510.13999) scores expert j by the mean over the tokens routed to it of g_j(x) * ||f_j(x)||, the
router gate weight times the norm of the expert's output. Strata's calibration (STRATA_GLM_CALIBRATION_DIR)
records, per expert, the routing weight sum (gates.json), the visit count (coverage.json) and the per-column
second moments of the down-projection input h (blk.L.down.f32). The output norm is estimated from those moments:

    E||W_down h||^2 ~= sum_k ||W_down[:, k]||^2 * E[h_k^2]      (diagonal approximation)
    saliency_j       = E[g_j] * sqrt(E||W_down h||^2)

so it needs the down weights' column norms, which this tool computes from the model (gguf-py dequantization).
The MTP layer is not covered by calibration: its plan drops the experts with the highest router bias (the
load-balancing bias is largest for experts the router disfavors on affinity alone). That can only change draft
acceptance, not output, because MTP drafts are verified.

Usage:
  glm_reap_rank.py MODEL.gguf CALIBRATION_DIR --keep 224 --out plan.txt [--report report.json] [--jobs 16]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

GGUF_PY = os.environ.get("STRATA_GGUF_PY", "/mnt/nvme02/work/llama.cpp/gguf-py")
sys.path.insert(0, GGUF_PY)
import gguf  # noqa: E402

_reader = None


def _open(path):
    global _reader
    if _reader is None:
        _reader = gguf.GGUFReader(path)
    return _reader


def _tensor(r, name):
    for t in r.tensors:
        if t.name == name:
            return t
    raise KeyError(name)


def column_norms(args):
    """||W_down[:, k]||^2 for every expert of one layer: array [experts, in_features]."""
    path, layer = args
    r = _open(path)
    t = _tensor(r, f"blk.{layer}.ffn_down_exps.weight")
    cols, rows, experts = (int(x) for x in t.shape)  # GGUF order: in (2048), out (4096), experts
    raw = np.asarray(t.data)
    per = raw.reshape(experts, -1)
    out = np.empty((experts, cols), np.float64)
    for e in range(experts):
        w = gguf.quants.dequantize(per[e], t.tensor_type).reshape(rows, cols)
        out[e] = np.einsum("rk,rk->k", w, w, dtype=np.float64)
    return layer, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("calibration")
    ap.add_argument("--keep", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()

    r = gguf.GGUFReader(a.model)
    experts = int(r.fields["glm5next.expert_count"].parts[-1][0])
    if not 0 < a.keep < experts or a.keep % 16:
        sys.exit(f"--keep must be a multiple of 16 below {experts}")
    routed = sorted({int(t.name.split(".")[1]) for t in r.tensors if t.name.endswith("ffn_gate_exps.weight")})
    cov = {p["name"]: p["counts"] for p in json.load(open(os.path.join(a.calibration, "coverage.json")))["projections"]}
    gates = {g["layer"]: g for g in json.load(open(os.path.join(a.calibration, "gates.json")))["layers"]}
    calibrated = [l for l in routed if f"blk.{l}.gu" in cov and l in gates]
    uncal = [l for l in routed if l not in calibrated]
    print(f"experts {experts} -> keep {a.keep}; routed layers {routed[0]}..{routed[-1]}; calibrated {len(calibrated)}, "
          f"by router bias {uncal}", file=sys.stderr)

    with ProcessPoolExecutor(a.jobs) as ex:
        norms = dict(ex.map(column_norms, [(a.model, l) for l in calibrated]))

    plan, report = {}, {"keep": a.keep, "experts": experts, "layers": []}
    for l in calibrated:
        counts = np.array(cov[f"blk.{l}.gu"], np.float64)
        gsum = np.array(gates[l]["sum"], np.float64)
        m2 = np.fromfile(os.path.join(a.calibration, f"blk.{l}.down.f32"), np.float32).reshape(experts, -1).astype(np.float64)
        out2 = (norms[l] * m2).sum(axis=1)
        mean_g = np.divide(gsum, counts, out=np.zeros_like(gsum), where=counts > 0)
        sal = mean_g * np.sqrt(out2)
        sal[counts == 0] = 0.0
        order = np.argsort(-sal, kind="stable")
        keep = np.sort(order[: a.keep])
        drop = np.sort(order[a.keep:])
        plan[l] = keep
        report["layers"].append({
            "layer": l,
            "dropped": drop.tolist(),
            "dropped_traffic_share": float(counts[drop].sum() / max(1.0, counts.sum())),
            "dropped_saliency_share": float(sal[drop].sum() / max(1e-30, sal.sum())),
            "dropped_weighted_share": float((counts * sal)[drop].sum() / max(1e-30, (counts * sal).sum())),
        })
    for l in uncal:
        bias = np.asarray(_tensor(r, f"blk.{l}.exp_probs_b.bias").data, np.float64)
        order = np.argsort(bias, kind="stable")  # ascending bias: keep the experts the router favors most
        plan[l] = np.sort(order[: a.keep])
        report["layers"].append({"layer": l, "dropped": np.sort(order[a.keep:]).tolist(), "criterion": "router_bias"})

    with open(a.out, "w") as f:
        f.write(f"experts {experts} {a.keep}\n")
        for l in sorted(plan):
            f.write(f"{l} " + " ".join(str(int(e)) for e in plan[l]) + "\n")
    cal = [x for x in report["layers"] if "dropped_traffic_share" in x]
    if cal:
        report["mean_dropped_traffic_share"] = float(np.mean([x["dropped_traffic_share"] for x in cal]))
        report["mean_dropped_weighted_share"] = float(np.mean([x["dropped_weighted_share"] for x in cal]))
        print(f"dropped per layer: traffic share mean {100 * report['mean_dropped_traffic_share']:.2f}% "
              f"(max {100 * max(x['dropped_traffic_share'] for x in cal):.2f}%), traffic x saliency share mean "
              f"{100 * report['mean_dropped_weighted_share']:.2f}%", file=sys.stderr)
    if a.report:
        json.dump(report, open(a.report, "w"), indent=1)
    print(f"plan written: {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
