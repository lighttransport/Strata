#!/usr/bin/env python3
"""Fold run_fp4.sh logs (A64FX_FP4 lines) into tools/sim/data/measured_a64fx_kernels.json.

    python3 tools/a64fx_probe/collect_fp4.py LOGDIR --job 52646912 [--write]

Writes the sections `nvfp4_kernel_v1` (NVFP4 expert kernels: GB/s per node at 1..4 tokens per shape, stream and phase
modes, and `chosen_llvm_best_per_width` that tools/sim/a64fx.py reads), `trunk_kernel_v1` (E4M3 / Q8 / BF16 trunk
kernels at one rank's TP12 shapes, `chosen_llvm_node48_gbps` at one token) and, when A64FX_FP4_GEMM lines are present,
`nvfp4_gemm_v1` (prefill TOPS by M and the factor against the int8 SDOT GEMM the simulator's prefill model uses: 3.18
TOPS at M >= 64). Only sections with data are replaced. Without --write it prints them.
"""
import argparse
import json
import pathlib
import re

DATA = pathlib.Path(__file__).resolve().parents[1] / "sim/data/measured_a64fx_kernels.json"
LINE = re.compile(r"^A64FX_FP4 (.*)$")
GEMM = re.compile(r"^A64FX_FP4_GEMM (.*)$")
INT8_GEMM_TOPS = 3.18      # A64FXNode.int8_tops_m[64]: gemm's int8 SDOT GEMM per node (CALIBRATION.md)
NV = ("w4a8", "w4a16q", "w4a16p")
# the trunk shape each kernel's one-token rate is taken from: one rank's KDA q/k/v slice (MXFP8, Q8, BF16) and the MLA
# o_proj slice (block FP8)
TRUNK_SHAPE = dict(mxfp8="768x4096", q8="768x4096", bf16="768x4096", bfp8="4096x1536")


def parse(logdir, pattern=LINE):
    rows = []
    for path in sorted(pathlib.Path(logdir).glob("*.log")):
        for line in path.read_text().splitlines():
            m = pattern.match(line)
            if m:
                kv = dict(f.split("=", 1) for f in m.group(1).split())
                rows.append(kv)
    return rows


def table(rows, kernels, threads="48"):
    out = {}
    for r in rows:
        if r["kernel"] not in kernels or r["threads"] != threads:
            continue
        shape = f'{r["rows"]}x{r["cols"]}'
        cell = out.setdefault(f'node{threads}_gbps_{r["mode"]}', {}).setdefault(r["kernel"], {}).setdefault(shape, [None] * 4)
        t = int(r["tokens"])
        if 1 <= t <= 4:
            cell[t - 1] = float(r["GBps"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logdir")
    ap.add_argument("--job", required=True)
    ap.add_argument("--node", default="one node of a 1-node interactive allocation", help="description, not a hostname")
    ap.add_argument("--compiler", default="clang 21.1.0 -O3 -mcpu=a64fx -msve-vector-bits=512 -fopenmp")
    ap.add_argument("--date", default="2026-10-22")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    rows = parse(a.logdir)
    nv = table(rows, NV)
    stream = nv.get("node48_gbps_stream", {})
    best16 = {}
    for shape in ("2048x4096", "4096x2048"):
        q, p = stream.get("w4a16q", {}).get(shape), stream.get("w4a16p", {}).get(shape)
        if q and p:
            best16[shape] = [max(x, y) for x, y in zip(q, p)]
    chosen = {}
    if "w4a8" in stream and {"2048x4096", "4096x2048"} <= set(stream["w4a8"]):
        chosen["w4a8"] = dict(gate_up=stream["w4a8"]["2048x4096"], down=stream["w4a8"]["4096x2048"])
    if len(best16) == 2:
        chosen["w4a16"] = dict(gate_up=best16["2048x4096"], down=best16["4096x2048"])
    nvfp4 = dict(
        what=("tools/a64fx_probe/glm_fp4_sve.c NVFP4 kernels on one A64FX node, 48 threads, first-touched copies: w4a8 = rt16q "
              "layout, svtbl E2M1 decode, SDOT by lane per K16 group, int8 activations per 64 (Strata's glm_fp4_quantize_q8 "
              "rounding); w4a16q / w4a16p = exact E2M1 x E4M3 products with F32 activations and F32 accumulation (rt16q byte "
              "extraction / rt16p svtbl_f32). GB/s of packed bytes = 0.5625 B per weight (codes + expanded E4M3 scales)"),
        job=a.job, node=a.node, compiler=a.compiler, date=a.date, bytes_per_weight=0.5625,
        notes="stream: no barrier between matrices (kernel rate); phase: an OpenMP barrier after every matrix. 512-row "
              "slices have 32 tiles for 48 threads.",
        chosen_llvm_best_per_width=chosen, **nv)
    tr = table(rows, ("mxfp8", "bfp8", "q8", "bf16"))
    ts = tr.get("node48_gbps_stream", {})
    pick = {}
    for kernel, shape in TRUNK_SHAPE.items():
        v = ts.get(kernel, {}).get(shape)
        if v and v[0]:
            pick[{"mxfp8": "mxfp8_exact", "bfp8": "block_fp8_exact", "q8": "q8_0", "bf16": "bf16"}[kernel]] = v[0]
    trunk = dict(
        what=("tools/a64fx_probe/glm_fp4_sve.c trunk kernels: mxfp8 / bfp8 = exact E4M3 decode (sign-extending load, shift, "
              "and, F16 widen; no gathers) x F32 activations; q8 = int8 requantization per (row, K32) with F16 scales x int8 "
              "activations by SDOT; bf16 = shift into F32. Shapes are one rank's TP12 slices"),
        job=a.job, node=a.node, compiler=a.compiler, date=a.date,
        chosen_shapes=TRUNK_SHAPE, chosen_llvm_node48_gbps=pick if len(pick) == 4 else {}, **tr)
    tops = {}
    for r in parse(a.logdir, GEMM):
        tops.setdefault(r["kernel"], {})[r["M"]] = float(r["TOPS"])
    sections = {}
    if chosen:
        sections["nvfp4_kernel_v1"] = nvfp4
    if pick:
        sections["trunk_kernel_v1"] = trunk
    if tops:
        at = lambda k: tops.get(k, {}).get("128") or tops.get(k, {}).get("64")
        factor = {name: round(INT8_GEMM_TOPS / at(k), 3) for name, k in
                  (("w4a8", "w4a8"), ("w4a16", "w4a16p"), ("trunk_exact", "mxfp8"), ("trunk_q8", "q8")) if at(k)}
        sections["nvfp4_gemm_v1"] = dict(
            what=("glm_fp4_bench --gemm M: 2048 x 4096, M tokens per call in blocks of 4 over each 16-row tile (decode once per "
                  "block), 48 threads; TOPS = 2 x MACs / s. factor = int8 SDOT GEMM 3.18 TOPS / this kernel at M = 128"),
            job=a.job, node=a.node, compiler=a.compiler, date=a.date, tops_by_m=tops, factor_vs_int8_gemm=factor)
    if not a.write:
        print(json.dumps(sections, indent=1))
        return
    data = json.loads(DATA.read_text())
    data.update(sections)
    DATA.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {', '.join(sections)} to {DATA}")


if __name__ == "__main__":
    main()
