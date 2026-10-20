#!/usr/bin/env python3
"""GLM-5.3-Flash on A64FX nodes: decode and prefill forecasts for 6 and 12 nodes.

    python3 tools/glm_a64fx_sim.py validate                      # predicted vs the measured 12-node runner rows
    python3 tools/glm_a64fx_sim.py estimate --nodes 6 --pack q23_fp8 --profile redesign --mtp 2
    python3 tools/glm_a64fx_sim.py layouts --nodes 6 12          # every layout x profile, decode and prefill
    python3 tools/glm_a64fx_sim.py memory --nodes 6 --pack q23_fp8
    python3 tools/glm_a64fx_sim.py report                        # the markdown tables of docs/GLM_A64FX_PLAN.md
    python3 tools/glm_a64fx_sim.py estimate --nodes 12 --set node.q2k_core_gbps=4 --set profile.mhc_site_us=[40,40]

The model is tools/sim/a64fx.py; every constant there names its measurement or says it is a hypothesis, and
`report` prints the hypothesis list of each profile.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "sim"))
import a64fx  # noqa: E402

REPORT_NODES = (6, 8, 12)   # 8 = 2x2x2 torus: 3 tree steps per allreduce instead of 4 (6 nodes) or 5 (12)
WORKLOADS = [("128-token prompt", 128), ("8K context", 8192), ("32K context", 32196)]
MTP_PROFILES = [("best512", "a64fx_best512"), ("sweep768", "a64fx_sweep768")]


def apply_sets(args, node, tofu, profile):
    for item in args.set or []:
        key, value = item.split("=", 1)
        scope, field = key.split(".", 1)
        target = dict(node=node, tofu=tofu, profile=profile)[scope]
        current = getattr(target, field)
        if isinstance(current, (tuple, list, dict)):
            value = json.loads(value)
            value = tuple(value) if isinstance(current, tuple) else value
        elif isinstance(current, bool):
            value = value.lower() in ("1", "true", "yes")
        else:
            value = type(current)(value)
        setattr(target, field, value)


def parts_from(args):
    return dict(nodes=args.nodes, trunk_tp=args.trunk_tp or args.nodes * args.ranks_per_node,
                expert_layout=args.layout, parts=args.parts, ranks_per_node=args.ranks_per_node,
                trunk_format=args.trunk, pack=args.pack, context=args.context, mtp_depth=args.mtp,
                acceptance=args.acceptance, prefill_chunk=args.chunk, bf16_collectives=args.bf16)


def add_layout_args(ap, nodes_list=False):
    if nodes_list:
        ap.add_argument("--nodes", type=int, nargs="+", default=[6, 8, 12])
    else:
        ap.add_argument("--nodes", type=int, default=12)
    ap.add_argument("--trunk-tp", type=int, default=0, help="ranks sharing the trunk; 0 = all ranks")
    ap.add_argument("--layout", default="ep_parts", choices=["ep", "ep_parts", "tp_parts"])
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--ranks-per-node", type=int, default=1, choices=[1, 4])
    ap.add_argument("--trunk", default="q8", choices=list(a64fx.TRUNK_SCALE))
    ap.add_argument("--pack", default="q23_fp8")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--mtp", type=int, default=0)
    ap.add_argument("--acceptance", default="a64fx_best512")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--prompt", type=int, default=8192)
    ap.add_argument("--bf16", action="store_true", help="bf16 collective payloads")
    ap.add_argument("--profile", default="current", choices=list(a64fx.PROFILES))
    ap.add_argument("--set", action="append", help="node.x=v | tofu.x=v | profile.x=v")


def context_for(args):
    node, tofu, profile = a64fx.A64FXNode(), a64fx.TofuD(), a64fx.PROFILES[args.profile]()
    apply_sets(args, node, tofu, profile)
    return node, tofu, profile


def describe_decode(d):
    return (f"decode {d.tok_s:.1f} tok/s: step {d.step_ms:.1f} ms (width {d.width}, {d.tokens_per_round:.2f} tokens/round, "
            f"round {d.round_ms:.1f} ms)\n  resources: trunk {d.trunk_ms:.1f}  experts {d.experts_ms:.1f} (busiest/mean load "
            f"{d.load_ratio:.2f})  collectives {d.comm_ms:.1f}  mHC {d.mhc_ms:.1f}  phases {d.phases_ms:.1f}  context {d.ctx_ms:.1f}"
            f"  head {d.head_ms:.1f} ms\n  stages: KDA {d.kda_ms:.1f}  sparse {d.sparse_ms:.1f}  MoE {d.moe_ms:.1f}  mHC {d.mhc_ms:.1f} ms")


def describe_fit(fit):
    return (f"memory per rank: experts {fit['expert_gib']:.1f} + trunk {fit['trunk_gib']:.1f} + replicated {fit['replicated_gib']:.1f}"
            f" + KV {fit['kv_gib']:.2f} + MTP {fit['mtp_gib']:.2f} + scratch {fit['scratch_gib']:.1f} = {fit['total_gib']:.1f} GiB"
            f" of {fit['usable_gib']:.1f} usable ({'fits' if fit['fits'] else 'DOES NOT FIT'}, margin {fit['margin_gib']:.1f})")


def cmd_estimate(args):
    node, tofu, profile = context_for(args)
    layout = a64fx.Layout(**parts_from(args))
    d = a64fx.decode(layout, node, tofu, profile)
    p = a64fx.prefill(layout, node, tofu, profile, prompt=args.prompt)
    print(layout.describe(), "| profile", profile.name)
    print(describe_fit(d.fit))
    print(describe_decode(d))
    print(f"prefill {p.tok_s:.0f} tok/s at {p.prompt} tokens, chunk {p.chunk}: {p.ms_per_pos:.2f} ms/pos = compute {p.compute_ms:.2f}"
          f" + collectives {p.comm_ms:.2f} + fixed {p.fixed_ms:.2f} + context {p.ctx_ms:.2f}")
    for n in d.notes:
        print("  NOTE:", n)
    if args.json:
        print(json.dumps(dict(layout=layout.to_dict(), decode=d.to_dict(), prefill=p.to_dict()), indent=1))


def cmd_validate(args):
    node, tofu, _ = context_for(args)
    rows = a64fx.validate(args.records, node, tofu)
    print(f"{'record':<62}{'measured':>10}{'predicted':>10}{'error':>8}  ok")
    bad = 0
    for r in rows:
        print(f"{r['name'][:62]:<62}{r['measured']:>10.2f}{r['predicted']:>10.2f}{r['error'] * 100:>7.1f}%  {'yes' if r['within'] else 'NO'}")
        if r["measured_stages"]:
            print("   stages (predicted/measured ms): " + "  ".join(f"{k} {r['stages'][k]:.2f}/{v}" for k, v in r["measured_stages"].items()))
        bad += not r["within"]
    print(f"{len(rows) - bad} of {len(rows)} rows within tolerance")
    return 1 if bad else 0


def layout_grid(nodes, pack, trunk):
    for rpn in (1, 4):
        for name, layout, parts in (("sliced over 4 ranks (runner today)", "ep_parts", 4), ("sliced over every rank", "tp_parts", 0),
                                    ("whole experts (EP)", "ep", 1), ("sliced over 2 ranks", "ep_parts", 2)):
            yield name + (" x4 CMG ranks" if rpn == 4 else ""), a64fx.Layout(nodes=nodes, trunk_tp=nodes * rpn, expert_layout=layout,
                                                                             parts=parts, ranks_per_node=rpn, pack=pack, trunk_format=trunk)


def cmd_layouts(args):
    node, tofu, _ = context_for(args)
    for nodes in args.nodes:
        for pname in a64fx.PROFILES:
            profile = a64fx.PROFILES[pname]()
            apply_sets(args, node, tofu, profile)
            print(f"\n{nodes} nodes, pack {args.pack}, {args.trunk} trunk, profile {pname}, context {args.context}")
            print(f"{'layout':<44}{'GiB':>6}{'fit':>5}{'plain':>7}{'mtp2 best':>10}{'mtp2 sweep':>11}{'prefill 8K':>11}{'prefill 128':>12}")
            for name, L in layout_grid(nodes, args.pack, args.trunk):
                L.context = args.context
                d = a64fx.decode(L, node, tofu, profile)
                m1 = a64fx.decode(dataclasses.replace(L, mtp_depth=2, acceptance="a64fx_best512"), node, tofu, profile)
                m2 = a64fx.decode(dataclasses.replace(L, mtp_depth=2, acceptance="a64fx_sweep768"), node, tofu, profile)
                p8 = a64fx.prefill(L, node, tofu, profile, 8192)
                p1 = a64fx.prefill(L, node, tofu, profile, 128)
                print(f"{name:<44}{d.fit['total_gib']:>6.1f}{'yes' if d.fit['fits'] else 'NO':>5}{d.tok_s:>7.1f}{m1.tok_s:>10.1f}"
                      f"{m2.tok_s:>11.1f}{p8.tok_s:>11.0f}{p1.tok_s:>12.0f}")


def cmd_memory(args):
    node, _, _ = context_for(args)
    for pack in ([args.pack] if args.pack != "all" else ["q23_fp8", "q23", "q22", "ud_q4kxl", "reap224_q23"]):
        for rpn in (1, 4):
            L = a64fx.Layout(nodes=args.nodes, trunk_tp=args.nodes * rpn, ranks_per_node=rpn, pack=pack, trunk_format=args.trunk,
                             context=args.context)
            print(f"{pack:<12} {args.nodes:2} nodes x{rpn} {args.trunk} trunk: {describe_fit(a64fx.memory_fit(L, node))}")


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def cmd_report(args):
    node, tofu, _ = context_for(args)
    pack, trunk = args.pack, args.trunk
    print(f"Pack {pack} ({a64fx.pack(pack).routed_total_bytes() / 1e9:.1f} GB routed experts), {trunk} trunk. Simulator outputs, not measurements.\n")
    print("### Memory per rank (GiB)\n")
    rows = []
    for nodes in REPORT_NODES:
        for rpn in (1, 4):
            for pk in (pack, "ud_q4kxl"):
                L = a64fx.Layout(nodes=nodes, trunk_tp=nodes * rpn, ranks_per_node=rpn, pack=pk, trunk_format=trunk, context=32768)
                f = a64fx.memory_fit(L, node)
                rows.append([f"{nodes} x{rpn}", pk, f"{f['expert_gib']:.1f}", f"{f['trunk_gib'] + f['replicated_gib']:.1f}", f"{f['kv_gib']:.2f}",
                             f"{f['total_gib']:.1f}", f"{f['usable_gib']:.0f}", "yes" if f["fits"] else "**no**"])
    print(md_table(["nodes x ranks", "pack", "experts", "trunk+replicated", "KV 32K", "total", "usable", "fits"], rows))
    for pname in a64fx.PROFILES:
        profile = a64fx.PROFILES[pname]()
        apply_sets(args, node, tofu, profile)
        print(f"\n### Decode, profile `{pname}` (tok/s)\n")
        rows = []
        for nodes in REPORT_NODES:
            for name, L in layout_grid(nodes, pack, trunk):
                cells = [f"{nodes}", name]
                for wname, ctx in WORKLOADS:
                    Lc = dataclasses.replace(L, context=ctx)
                    d = a64fx.decode(Lc, node, tofu, profile)
                    m = a64fx.decode(dataclasses.replace(Lc, mtp_depth=2, acceptance="a64fx_best512"), node, tofu, profile)
                    s = a64fx.decode(dataclasses.replace(Lc, mtp_depth=2, acceptance="a64fx_sweep768"), node, tofu, profile)
                    cells.append(f"{d.tok_s:.1f} / {m.tok_s:.1f} / {s.tok_s:.1f}")
                d = a64fx.decode(dataclasses.replace(L, context=8192), node, tofu, profile)
                cells.append(f"{d.step_ms:.1f} = trunk {d.trunk_ms:.1f} + experts {d.experts_ms:.1f} + coll {d.comm_ms:.1f} + mHC {d.mhc_ms:.1f}"
                             f" + ctx {d.ctx_ms:.1f} + head {d.head_ms:.1f}" + (f" + phases {d.phases_ms:.1f}" if pname == "redesign" else ""))
                rows.append(cells)
        print(md_table(["nodes", "layout", "128 ctx plain / MTP2 best / MTP2 sweep", "8K", "32K", "8K step ms"], rows))
        print(f"\n### Prefill, profile `{pname}` (tok/s)\n")
        rows = []
        for nodes in REPORT_NODES:
            for name, L in layout_grid(nodes, pack, trunk):
                p8 = a64fx.prefill(L, node, tofu, profile, 8192)
                p1 = a64fx.prefill(L, node, tofu, profile, 128)
                p32 = a64fx.prefill(dataclasses.replace(L, context=32196), node, tofu, profile, 32196)
                rows.append([f"{nodes}", name, f"{p1.tok_s:.0f}", f"{p8.tok_s:.0f}", f"{p32.tok_s:.0f}",
                             f"{p8.ms_per_pos:.2f} = compute {p8.compute_ms:.2f} + coll {p8.comm_ms:.2f} + fixed {p8.fixed_ms:.2f} + ctx {p8.ctx_ms:.2f}"])
        print(md_table(["nodes", "layout", "128 tokens", "8K (chunk 4096)", "32K", "8K ms/position"], rows))
        print(f"\nHypotheses in `{pname}`: " + "; ".join(profile.hypotheses))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("estimate", help="one layout: decode, prefill, memory fit"); add_layout_args(p); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_estimate)
    p = sub.add_parser("validate", help="predicted vs measured 12-node rows"); add_layout_args(p); p.add_argument("--records"); p.set_defaults(func=cmd_validate)
    p = sub.add_parser("layouts", help="every layout x profile for the given node counts"); add_layout_args(p, nodes_list=True); p.set_defaults(func=cmd_layouts)
    p = sub.add_parser("memory", help="memory fit per pack"); add_layout_args(p); p.set_defaults(func=cmd_memory)
    p = sub.add_parser("report", help="markdown tables for docs/GLM_A64FX_PLAN.md"); add_layout_args(p); p.set_defaults(func=cmd_report)
    args = ap.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
