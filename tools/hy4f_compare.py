#!/usr/bin/env python3
"""Compare strata-hy4f --dump output with a llama.cpp dump (tools/ds41f_llama_dump.cpp, same .bin format).

    uv run --no-project --with numpy python -I tools/hy4f_compare.py REF_DIR STRATA_DIR [--split 512,2048] [--names a-0,b-1]

Per block l_out-<il> (the four streams, [n][4][pos]): the relative L2 error per position, as its mean and 99th
percentile, and the mean in each position range of --split (default: below 512, 512-2048, past 2048: the sliding
window and the token selector's limit). result_output: top-1 agreement and KL(ref || strata), mean and 99th
percentile. --names compares further dumped tensors the same way (one row per position).
"""
import argparse
import pathlib

import numpy as np


def load(path):
    with open(path, "rb") as f:
        head = f.read(36)
    t = int(np.frombuffer(head[:4], dtype=np.int32)[0])
    if t != 0:
        raise SystemExit(f"{path}: type {t}, expected f32")
    return np.frombuffer(head[4:36], dtype=np.int64), np.memmap(path, dtype=np.float32, mode="r", offset=36)


def rel_rows(ref, mine, width):
    a = ref.reshape(-1, width).astype(np.float64)
    b = mine.reshape(-1, width).astype(np.float64)
    if a.shape != b.shape:
        raise SystemExit(f"shape mismatch {a.shape} vs {b.shape} (same prompt?)")
    return np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(a, axis=1), 1e-30)


def ranges_text(e, cuts):
    edges = [0] + [c for c in cuts if c < len(e)] + [len(e)]
    parts = []
    for lo, hi in zip(edges, edges[1:]):
        if hi > lo:
            parts.append(f"[{lo},{hi}) {e[lo:hi].mean():.2e}")
    return "  ".join(parts)


def log_softmax(x):
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("mine")
    ap.add_argument("--split", default="512,2048", help="position range boundaries for the per-range means")
    ap.add_argument("--names", default="", help="more tensors to compare, e.g. attn_norm-0,hc_attn_out-0")
    a = ap.parse_args()
    ref, mine = pathlib.Path(a.ref), pathlib.Path(a.mine)
    cuts = [int(x) for x in a.split.split(",") if x]
    worst, il = 0.0, 0
    while (mine / f"l_out-{il}.bin").exists():
        rf = ref / f"l_out-{il}.bin"
        if not rf.exists():
            print(f"l_out-{il}: no reference")
            break
        ne, vm = load(mine / f"l_out-{il}.bin")
        e = rel_rows(load(rf)[1], vm, int(ne[0] * ne[1]))
        worst = max(worst, float(e.mean()))
        print(f"l_out-{il:<2} rel err mean {e.mean():.2e} p99 {np.percentile(e, 99):.2e}   {ranges_text(e, cuts)}")
        il += 1
    for name in [n for n in a.names.split(",") if n]:
        ne, vm = load(mine / f"{name}.bin")
        e = rel_rows(load(ref / f"{name}.bin")[1], vm, int(ne[0]))
        print(f"{name:<14} rel err mean {e.mean():.2e} p99 {np.percentile(e, 99):.2e}")
    if (mine / "result_output.bin").exists() and (ref / "result_output.bin").exists():
        ne, vm = load(mine / "result_output.bin")
        V = int(ne[0])
        x, y = load(ref / "result_output.bin")[1].reshape(-1, V), vm.reshape(-1, V)
        kl, hit = [], []
        for p in range(0, len(y), 256):   # in chunks: a long prompt's logits are GBs in float64
            xa, yb = x[p:p + 256].astype(np.float64), y[p:p + 256].astype(np.float64)
            la, lb = log_softmax(xa), log_softmax(yb)
            kl.append((np.exp(la) * (la - lb)).sum(axis=1))
            hit.append(xa.argmax(axis=1) == yb.argmax(axis=1))
        kl, top = np.concatenate(kl), np.concatenate(hit).mean() * 100
        print(f"result_output: {len(kl)} positions, top-1 {top:.1f} %, KL mean {kl.mean():.3e} p99 {np.percentile(kl, 99):.3e}"
              f"   {ranges_text(kl, cuts)}")
    print(f"worst block mean rel err {worst:.2e}")


if __name__ == "__main__":
    main()
