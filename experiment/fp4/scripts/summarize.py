#!/usr/bin/env python3
"""Turn results/<date>/*.csv into the markdown tables used in README.md.
Usage: python3 -I scripts/summarize.py results/<date>"""
import csv
import sys
from collections import defaultdict
from pathlib import Path


def load(p):
    return list(csv.DictReader(open(p))) if p.exists() else []


def f(x, d=1):
    return f"{float(x):.{d}f}"


def best(rows, key):
    """keep the fastest row per key(row)"""
    out = {}
    for r in rows:
        k = key(r)
        if k not in out or float(r["us_med"]) < float(out[k]["us_med"]):
            out[k] = r
    return out


def main(d):
    d = Path(d)
    cpu = load(d / "cpu_dummy.csv")
    cpur = load(d / "cpu_real.csv")
    gpu = load(d / "gpu_dummy.csv")
    gpur = load(d / "gpu_real.csv")
    # a later decode-only rerun (gpu_decode_v4*.csv) replaces the decode rows of the full pass
    for name, rows in (("gpu_decode_v4.csv", gpu), ("gpu_decode_v4_real.csv", gpur)):
        extra = load(d / name)
        if extra:
            rows[:] = [r for r in rows if r["op"] not in ("gu8", "down8")] + extra

    print("### CPU (Threadripper 1950X), dummy weights, 8 experts per call\n")
    thr = sorted({r["threads_or_cfg"] for r in cpu}, key=lambda t: int(t[1:]))
    bw = {r["threads_or_cfg"]: r["gbps"] for r in cpu if r["kernel"] == "read_bw"}
    kern = [k for k in dict.fromkeys(r["kernel"] for r in cpu) if k not in ("read_bw",) and not k.startswith(("moe_", "c4_"))]
    print("GB/s for gate+up of 8 experts (71 MB); first row is the read-bandwidth reference.\n")
    print("| kernel | " + " | ".join(thr) + " | relL2 |")
    print("|---|" + "---:|" * len(thr) + "---:|")
    print("| read bandwidth | " + " | ".join(f(bw[t]) for t in thr) + " | |")
    for k in kern:
        cells, err = [], ""
        for t in thr:
            r = [x for x in cpu if x["kernel"] == k and x["op"] == "gu8" and x["threads_or_cfg"] == t]
            cells.append(f(r[0]["gbps"]) if r else "")
            err = r[0]["rel_l2"] if r else err
        print(f"| {k} | " + " | ".join(cells) + f" | {float(err):.1e} |")
    print("\nMTP/batch verify on CPU (gate+up, N tokens share the 8 experts), time relative to N=1:\n")
    print("| threads | " + " | ".join(f"N={n}" for n in (1, 2, 4, 8)) + " |")
    print("|---|" + "---:|" * 4)
    for t in thr:
        rs = {int(r["n_tok"]): float(r["us_med"]) for r in cpu if r["kernel"].startswith("c4_") and r["threads_or_cfg"] == t}
        if 1 in rs:
            print(f"| {t} | " + " | ".join(f"{rs[n]/1e3:.2f} ms ({rs[n]/rs[1]:.2f}x)" for n in (1, 2, 4, 8) if n in rs) + " |")
    print("\nFull MoE layer (q8(x), 8x gate/up, SwiGLU, q8(h), 8x down, weighted sum):\n")
    print("| kernel | weights | threads | ms per layer | GB/s | expert-only tok/s (47 layers) |")
    print("|---|---|---|---:|---:|---:|")
    for src, rows in (("dummy", cpu), ("real L1", cpur)):
        for r in rows:
            if r["op"] == "moe_layer":
                us = float(r["us_med"])
                print(f"| {r['kernel'][4:]} | {src} | {r['threads_or_cfg']} | {us/1e3:.2f} | {f(r['gbps'])} | {1e6/(us*47):.1f} |")

    peaks = [r for r in gpu if r["kernel"] == "mma_peak"]
    if peaks:
        print("\n### GPU tensor-core peak per instruction (register-only loop)\n")
        print("| instruction | dense TFLOPS / TOPS |\n|---|---:|")
        for r in peaks:
            print(f"| `{r['op']}` | {f(r['tflops'])} |")

    for src, rows in (("dummy", gpu), ("real L1+L47", gpur)):
        dec = [r for r in rows if r["op"] in ("gu8", "down8")]
        if not dec:
            continue
        print(f"\n### GPU decode, {src} weights: 8 experts, best launch config per kernel\n")
        ns = sorted({int(r["n_tok"]) for r in dec})
        b = best(dec, lambda r: (r["kernel"], r["op"], r["n_tok"]))
        kerns = list(dict.fromkeys(r["kernel"] for r in dec))
        print("N=1: time per call, bandwidth, error vs fp64 (error includes the activation format).\n")
        print("| kernel | gate+up us | GB/s | down us | GB/s | relL2 |")
        print("|---|---:|---:|---:|---:|---:|")
        for k in kerns:
            g, dn = b.get((k, "gu8", "1")), b.get((k, "down8", "1"))
            if g and dn:
                print(f"| {k} | {f(g['us_med'])} | {f(g['gbps'])} | {f(dn['us_med'])} | {f(dn['gbps'])} | {float(g['rel_l2']):.1e} |")
        if len(ns) > 1:
            print("\nSmall batch (MTP verify), gate+up of 8 experts, us per call:\n")
            print("| kernel | " + " | ".join(f"N={n}" for n in ns) + " |")
            print("|---|" + "---:|" * len(ns))
            for k in kerns:
                cells = [f(b[(k, "gu8", str(n))]["us_med"]) if (k, "gu8", str(n)) in b else "" for n in ns]
                print(f"| {k} | " + " | ".join(cells) + " |")
        moe = [r for r in rows if r["op"] in ("moe_layer", "moe47_graph")]
        if moe:
            print("\nFused MoE layer (g3, 2 launches):\n")
            print("| measurement | us per layer | GB/s | relL2 |\n|---|---:|---:|---:|")
            for r in moe:
                print(f"| {r['op']} | {f(r['us_med'])} | {f(r['gbps'])} | {float(r['rel_l2']):.1e} |")

    for src, rows in (("dummy", gpu), ("real L1", gpur)):
        pre = [r for r in rows if r["op"].endswith("_prefill")]
        if not pre:
            continue
        print(f"\n### GPU prefill grouped GEMM, {src} weights, all 256 experts (best BN per kind)\n")
        b = best(pre, lambda r: (r["kernel"], r["op"], r["n_tok"]))
        ts = sorted({int(r["n_tok"]) for r in pre})
        print("TFLOPS for gate+up / down; relL2 vs fp64 (includes the activation format).\n")
        print("| kernel | " + " | ".join(f"T={t}" for t in ts) + " | relL2 |")
        print("|---|" + "---:|" * len(ts) + "---:|")
        for k in dict.fromkeys(r["kernel"] for r in pre):
            cells, err = [], 0
            for t in ts:
                g, dn = b.get((k, "gu_prefill", str(t))), b.get((k, "down_prefill", str(t)))
                if g and dn:
                    cells.append(f"{f(g['tflops'])} / {f(dn['tflops'])}")
                    err = float(g["rel_l2"])
                else:
                    cells.append("")
            print(f"| {k} | " + " | ".join(cells) + f" | {err:.1e} |")
        print("\nTime per MoE layer (gate+up + down GEMM only), ms:\n")
        print("| kernel | " + " | ".join(f"T={t}" for t in ts) + " |")
        print("|---|" + "---:|" * len(ts))
        for k in dict.fromkeys(r["kernel"] for r in pre):
            cells = []
            for t in ts:
                g, dn = b.get((k, "gu_prefill", str(t))), b.get((k, "down_prefill", str(t)))
                cells.append(f"{(float(g['us_med']) + float(dn['us_med']))/1e3:.1f}" if g and dn else "")
            print(f"| {k} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else sorted(Path(__file__).resolve().parent.parent.glob("results/2*"))[-1])
