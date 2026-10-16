#!/usr/bin/env python3
"""Muse Spark Open 816B-A42B (UD-IQ1_S, 175.6 GB) on TR16 + B550 over InfiniBand: decode and prefill forecasts.

    python3 tools/muse_sim.py report                    # the tables in docs/MUSE_SPARK.md
    python3 tools/muse_sim.py decode --split row --b-share 0.375 --gpu-layers 32,30
    python3 tools/muse_sim.py decode --set b550.cpu_gmacs.iq256=20     # the B550 while another job holds its CPU
    python3 tools/muse_sim.py prefill --tokens 8192 --chunk 8192

The model is tools/sim/muse.py; every constant there names its measurement or says it is an estimate.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "sim"))
import muse  # noqa: E402


def nodes_from(args):
    a, b = muse.scenarios()[args.scenario]
    nodes = {"tr16": a, "b550": b}
    for item in args.set:
        key, value = item.split("=", 1)
        parts = key.split(".")
        target = nodes[parts[0]]
        if parts[1] == "gpu":                       # tr16.gpu.0.h2d_gbps=7.18
            target = target.gpus[int(parts[2])]
            parts = parts[2:]
        for p in parts[1:-1]:
            target = target[p] if isinstance(target, dict) else getattr(target, p)
        if isinstance(target, dict) and isinstance(target.get(parts[-1]), dict):   # b550.cpu_gmacs.iq256=20
            muse.set_lane_rate(nodes[parts[0]], parts[-1], float(value))
        elif isinstance(target, dict):
            target[parts[-1]] = float(value)
        else:
            setattr(target, parts[-1], type(getattr(target, parts[-1]))(value))
    return [a, b]


def plan_from(args):
    layers = tuple(int(x) for x in args.gpu_layers.split(",")) if args.gpu_layers else ()
    return muse.Plan(split=args.split, b_share=args.b_share, gpu_layers=layers, fixed=args.fixed,
                     cpu_kernel=args.kernel, context=args.context, accept=args.accept, width=args.width,
                     hit_scale=args.hit_scale, top_k=args.top_k)


def describe_placement(place):
    lines = []
    for g in place["gpus"]:
        lines.append(f"  {g['gpu']:<12} node {g['node']}: {len(g['layers']):2d} layers, fixed {g['fixed_gib']:.1f} GiB, "
                     f"KV {g['kv_gib']:.2f} GiB, {g['slots']} expert slots (coverage {g['coverage']:.3f}, hit {g['hit']:.3f})")
    lines.append(f"  RAM for weights: node A {place['node_gib'][0]:.1f} GiB, node B {place['node_gib'][1]:.1f} GiB")
    for n in place["notes"]:
        lines.append(f"  INFEASIBLE: {n}")
    return "\n".join(lines)


def cmd_decode(args):
    nodes, link = nodes_from(args), muse.LINKS[args.link]
    r = muse.decode(plan_from(args), nodes, link)
    print(f"{args.scenario}, {link.name}, {plan_from(args)}")
    print(describe_placement(r["placement"]))
    print(f"decode {r['tok_s']:.2f} tok/s: step {r['step_ms']:.1f} ms = GPU chain {r['gpu_ms']:.1f} + CPU expert lanes "
          f"{r['cpu_ms']:.1f} (of which waiting on the remote lane {r['remote_wait_ms']:.1f}) + mailboxes {r['mailbox_ms']:.1f}"
          f" + hops {r['hops_ms']:.2f} + head/sampler {r['head_tail_ms']:.1f} + host {r['host_ms']:.1f}")


def cmd_prefill(args):
    nodes, link = nodes_from(args), muse.LINKS[args.link]
    r = muse.prefill(plan_from(args), nodes, link, tokens=args.tokens, chunk=args.chunk)
    print(f"{args.scenario}, {link.name}, {plan_from(args)}")
    print(describe_placement(r["placement"]))
    print(f"prefill {args.tokens} tokens: {r['tok_s']:.0f} tok/s ({r['seconds']:.1f} s; attention {r['attention_s']:.1f} s, "
          f"experts {r['experts_s']:.1f} s, hops {r['hops_s']:.2f} s, {r['chunks']} chunk(s) of {r['chunk']})")


def best_rows(nodes, link, base, what, **kw):
    out = []
    for split, (score, plan, r) in muse.search(nodes, link, base, what=what, **kw).items():
        out.append((split, score, plan, r))
    return sorted(out, key=lambda x: -x[1])


def report_rows():
    """(section, rows) for docs/MUSE_SPARK.md."""
    sections = []
    ib = muse.LINKS["ib-qdr"]
    rows = []
    for scen, (a, b) in muse.scenarios().items():
        for kernel in ("ggml", "iq256", "dram"):
            for fixed in ("orig", "q4"):
                best = best_rows([a, b], ib, muse.Plan(cpu_kernel=kernel, fixed=fixed), "decode")
                split, score, plan, r = best[0]
                rows.append([scen, kernel, fixed, f"{score:.2f}", split, f"{plan.b_share:.3f}", ",".join(map(str, plan.gpu_layers)),
                             f"{r['gpu_ms']:.0f}", f"{r['cpu_ms']:.0f}", f"{r['step_ms']:.0f}"])
    sections.append(("decode at 8K context, IB QDR, best plan per row (no speculation)",
                     ["scenario", "CPU kernel", "trunk", "tok/s", "split", "B share", "GPU layers", "GPU ms", "CPU ms", "step ms"], rows))
    rows = []
    for scen, (a, b) in muse.scenarios().items():
        for split in ("row", "expert", "layer"):
            best = dict((s, (sc, p, r)) for s, sc, p, r in best_rows([a, b], ib, muse.Plan(), "decode"))
            if split not in best:
                rows.append([scen, split, "-", "infeasible (RAM)", "", ""])
                continue
            sc, p, r = best[split]
            rows.append([scen, split, f"{sc:.2f}", f"{p.b_share:.3f}", ",".join(map(str, p.gpu_layers)), f"{r['remote_wait_ms']:.0f}"])
    sections.append(("decode by split kind (Strata iq256 kernel, shipped trunk)",
                     ["scenario", "split", "tok/s", "B share", "GPU layers", "remote wait ms"], rows))
    rows = []
    for scen, (a, b) in muse.scenarios().items():
        for tokens in (2048, 8192, 32768):
            best = best_rows([a, b], ib, muse.Plan(context=tokens), "prefill", tokens=tokens, chunks=(1024, 2048, 4096, 8192))
            split, score, plan, r = best[0]
            rows.append([scen, str(tokens), f"{score:.0f}", split, f"{plan.b_share:.3f}", ",".join(map(str, plan.gpu_layers)),
                         str(r["chunk"]), f"{r['attention_s']:.1f}", f"{r['experts_s']:.1f}"])
    sections.append(("prefill, IB QDR, best plan and chunk",
                     ["scenario", "tokens", "tok/s", "split", "B share", "GPU layers", "chunk", "attention s", "experts s"], rows))
    # sensitivities around the measured baseline (split-gpus, iq256, shipped trunk)
    rows = []
    base = muse.Plan()

    def run(label, plan=base, link=ib, patch=None):
        a, b = muse.scenarios()["split-gpus"]
        if patch:
            patch(a, b)
        best = best_rows([a, b], link, plan, "decode")
        split, score, p, r = best[0]
        rows.append([label, f"{score:.2f}", split, f"{p.b_share:.3f}"])

    run("baseline (measured kernels, quiet B550)")
    run("B550 CPU shared with a 26-thread miner (1-token rate 20 GMAC/s, measured 14-25)",
        patch=lambda a, b: muse.set_lane_rate(b, "iq256", 20.0))
    run("1GbE instead of InfiniBand", link=muse.LINKS["gbe"])
    run("routing skew 0.7x the GLM curve", dataclasses.replace(base, hit_scale=0.7))
    run("routing skew 1.3x the GLM curve", dataclasses.replace(base, hit_scale=1.3))
    run("context 32K", dataclasses.replace(base, context=32768))
    run("context 128K", dataclasses.replace(base, context=131072))
    run("top-6 routing (lossy)", dataclasses.replace(base, top_k=6))
    run("n-gram drafts: width 2, 1.4 accepted/step", dataclasses.replace(base, width=2, accept=1.4))
    run("Q4_K trunk + DRAM-bound kernel (both levers)", dataclasses.replace(base, fixed="q4", cpu_kernel="dram"))
    sections.append(("decode sensitivity (split-gpus)", ["case", "tok/s", "split", "B share"], rows))
    return sections


def markdown(sections):
    out = []
    for title, header, rows in sections:
        out.append(f"### {title}\n")
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join("---" for _ in header) + "|")
        for r in rows:
            out.append("| " + " | ".join(r) + " |")
        out.append("")
    return "\n".join(out)


def cmd_report(args):
    sections = report_rows()
    text = markdown(sections)
    if args.output:
        Path(args.output).write_text(text)
    if args.json:
        Path(args.json).write_text(json.dumps([dict(title=t, header=h, rows=r) for t, h, r in sections], indent=1) + "\n")
    print(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("decode", "prefill"):
        p = sub.add_parser(name)
        p.add_argument("--scenario", choices=list(muse.scenarios()), default="split-gpus")
        p.add_argument("--link", choices=list(muse.LINKS), default="ib-qdr")
        p.add_argument("--split", choices=("row", "expert", "layer"), default="row")
        p.add_argument("--b-share", type=float, default=0.375)
        p.add_argument("--gpu-layers", default="", help="layers per GPU, e.g. 32,30 (default: by free VRAM)")
        p.add_argument("--fixed", choices=("orig", "q4"), default="orig")
        p.add_argument("--kernel", choices=("ggml", "iq256", "dram"), default="iq256")
        p.add_argument("--context", type=int, default=8192)
        p.add_argument("--accept", type=float, default=1.0)
        p.add_argument("--width", type=int, default=1)
        p.add_argument("--hit-scale", type=float, default=1.0)
        p.add_argument("--top-k", type=int, default=0)
        p.add_argument("--set", action="append", default=[], metavar="NODE.FIELD=VALUE",
                       help="e.g. b550.cpu_gmacs.iq256=20 (the 1-token rate; widths scale with it), tr16.cpu_gbps=50, tr16.gpu.0.h2d_gbps=7.18")
        if name == "prefill":
            p.add_argument("--tokens", type=int, default=8192)
            p.add_argument("--chunk", type=int, default=8192)
    p = sub.add_parser("report")
    p.add_argument("--output")
    p.add_argument("--json")
    args = ap.parse_args(argv)
    dict(decode=cmd_decode, prefill=cmd_prefill, report=cmd_report)[args.cmd](args)


if __name__ == "__main__":
    main()
