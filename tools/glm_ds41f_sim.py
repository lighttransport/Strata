#!/usr/bin/env python3
"""DeepSeek V4.1 Flash: which quantization on which machine, and the decode/prefill it would give.

    python3 tools/glm_ds41f_sim.py validate                                  # against the gemm A64FX and TR16 runs
    python3 tools/glm_ds41f_sim.py estimate --machine tr16 --pack iq1s --trunk q4 --width 3 --acceptance a64fx_best512
    python3 tools/glm_ds41f_sim.py packs --machine xeon_v100                 # every pack that fits, decode and prefill
    python3 tools/glm_ds41f_sim.py memory --machine tr16                     # RAM / VRAM / SSD per pack
    python3 tools/glm_ds41f_sim.py a64fx --nodes 6 8 12                      # A64FX layouts, both profiles
    python3 tools/glm_ds41f_sim.py report                                    # the markdown tables of docs/DS41F_PLAN.md
    python3 tools/glm_ds41f_sim.py estimate --machine tr16 --pack iq1s --set cpu_rate_override_gbps=60

The model is tools/sim/ds41f.py; every constant there names its measurement or says it is a hypothesis, and every
result prints the hypotheses it rests on.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "sim"))
import ds41f  # noqa: E402

GIB = 2 ** 30
CANDIDATES = {
    "tr16": [("iq1s", "q4", "q4"), ("iq1m", "q4", "q4"), ("iq1m_iq2xxs", "q4", "q4"), ("iq2xxs_q2k_reap256", "q4", "q4"),
             ("iq2xxs_q2k_reap288", "q4", "q4"), ("tetra_reap320", "q4", "q4"), ("iq2xxs_q2k", "q4", "q4")],
    "tr16+b550": [("iq2xxs_q2k", "q4", "q4"), ("tetra", "q4", "q4"), ("iq2xxs", "q4", "q4"), ("iq1s", "q4", "q4"), ("q23", "q4", "q4")],
    "xeon_v100": [("iq2xxs_q2k", "q8", "q8"), ("iq2s_q2k", "q8", "q8"), ("tetra", "q8", "q8"), ("q23", "q8", "q8"), ("iq3xxs", "q8", "q8"), ("iq1s", "q8", "q8")],
}
A64FX_CANDIDATES = {6: ["iq2xxs_q2k", "tetra", "iq1m"], 8: ["iq2xxs_q2k", "q23", "iq3xxs"], 12: ["mxfp4", "iq3xxs", "iq2xxs_q2k"]}


def add_args(ap):
    ap.add_argument("--machine", default="tr16", choices=list(ds41f.MACHINES))
    ap.add_argument("--pack", default="iq1s")
    ap.add_argument("--trunk", default="q4", choices=list(ds41f.TRUNK_SCALE))
    ap.add_argument("--head", default="q4", choices=list(ds41f.HEAD_SCALE))
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--width", type=int, default=1, help="DSpark verify window (1 = plain decode)")
    ap.add_argument("--acceptance", default="", help="routing.ACCEPTANCE profile for the window (a64fx_best512, a64fx_sweep768, ...)")
    ap.add_argument("--prompt", type=int, default=8192)
    ap.add_argument("--set", action="append", default=[], help="plan field=value (tier_gib, engram_cache_hit, worker_share, cpu_rate_override_gbps, ...)")


def plan_from(args, **over):
    fields = dict(machine=args.machine, pack=args.pack, trunk=args.trunk, head=args.head, context=args.context, width=args.width,
                  acceptance=args.acceptance)
    fields.update(over)
    plan = ds41f.Plan(**fields)
    for item in args.set:
        key, value = item.split("=", 1)
        current = getattr(plan, key)
        setattr(plan, key, type(current)(value) if not isinstance(current, bool) else value.lower() in ("1", "true", "yes"))
    return plan


def fit_line(fit):
    return (f"RAM {fit['ram_need_gib']:.1f} of {fit['ram_gib']:.0f} GiB (experts {fit['expert_gib']:.1f}"
            + (f", worker {fit['worker_gib']:.1f}" if fit['worker_gib'] else "")
            + f"); VRAM trunk {fit['trunk_gib']:.1f} + head {fit['head_gib']:.2f} + KV {fit['kv_gib']:.2f} + tier {fit['tier_gib']:.1f} of {fit['vram_gib']:.0f} GiB; "
            f"SSD {fit['ssd_total_gib']:.0f} GiB (Engram {fit['engram_ssd_gib']:.0f}); {'fits' if fit['fits'] else 'DOES NOT FIT'}")


def cmd_estimate(args):
    plan = plan_from(args)
    d = ds41f.decode(plan)
    p = ds41f.prefill(plan, prompt=args.prompt)
    pk = ds41f.pack(plan.pack)
    print(f"{plan.machine}: {plan.pack} ({pk.bpw():.2f} bpw, {pk.total_bytes() / 1e9:.1f} GB, {pk.experts} experts), trunk {plan.trunk}, head {plan.head}, "
          f"context {plan.context}, width {plan.width}" + (f" ({plan.acceptance})" if plan.acceptance else ""))
    print("  " + fit_line(d.fit))
    print(f"  decode {d.tok_s:.1f} tok/s: step {d.step_ms:.1f} ms = CPU experts {d.cpu_ms:.1f} ({d.cpu_bound}-bound, {d.expert_gb_per_token:.2f} GB/token, "
          f"tier serves {d.tier_hit * 100:.0f} %) + exposed GPU {d.gpu_exposed_ms:.1f} + Engram {d.engram_exposed_ms:.2f} (of {d.engram_ms:.2f}) "
          f"+ head {d.head_ms:.1f}" + (f"; remote wait {d.remote_ms:.1f}" if d.remote_ms else "")
          + (f"; round {d.round_ms:.1f} ms for {d.tokens_per_round:.2f} tokens" if plan.width > 1 else ""))
    print(f"  prefill {p.tok_s:.0f} tok/s at {args.prompt} tokens ({p.ms_per_pos:.2f} ms/pos; {p.bottleneck}-bound: upload {p.upload_ms / 1e3:.1f} s, "
          f"GEMM {p.gemm_ms / 1e3:.1f} s, mixers {p.mixer_ms / 1e3:.1f} s, Engram {p.engram_ms / 1e3:.2f} s)")
    for n in d.notes:
        print("  NOTE:", n)
    print("  hypotheses: " + "; ".join(d.hypotheses + p.hypotheses))


def cmd_packs(args):
    m = ds41f.MACHINES[args.machine]()
    print(f"{args.machine}: {m.notes}")
    print(f"{'pack':<22}{'bpw':>5}{'GB':>7}{'fits':>5}{'tier':>6}{'hit':>5}{'plain':>7}{'DSpark best':>12}{'DSpark sweep':>13}{'bound':>8}{'prefill':>8}")
    for pk, trunk, head in CANDIDATES[args.machine]:
        plan = plan_from(args, pack=pk, trunk=trunk, head=head, width=1, acceptance="")
        d = ds41f.decode(plan)
        b = ds41f.decode(dataclasses.replace(plan, width=3, acceptance="a64fx_best512"))
        s = ds41f.decode(dataclasses.replace(plan, width=3, acceptance="a64fx_sweep768"))
        p = ds41f.prefill(plan, prompt=args.prompt)
        P = ds41f.pack(pk)
        print(f"{pk:<22}{P.bpw():>5.2f}{P.total_bytes() / 1e9:>7.1f}{'yes' if d.fit['fits'] else 'NO':>5}{d.fit['tier_gib']:>6.1f}{d.tier_hit:>5.2f}"
              f"{d.tok_s:>7.1f}{b.tok_s:>12.1f}{s.tok_s:>13.1f}{d.cpu_bound:>8}{p.tok_s:>8.0f}")


def cmd_memory(args):
    for pk, trunk, head in CANDIDATES[args.machine]:
        plan = plan_from(args, pack=pk, trunk=trunk, head=head)
        print(f"{pk:<22} " + fit_line(ds41f.memory_fit(plan)))


def cmd_a64fx(args):
    print(f"{'nodes':>5} {'pack':<14}{'profile':<9}{'GiB/node':>9}{'fits':>5}{'plain':>7}{'DSpark best':>12}{'prefill':>8}  step split (ms)")
    for nodes in args.nodes:
        for pk in A64FX_CANDIDATES.get(nodes, ["iq2xxs_q2k"]):
            for prof in ("current", "redesign"):
                ap = ds41f.A64FXPlan(nodes=nodes, pack=pk, profile=prof, context=args.context)
                d = ds41f.a64fx_decode(ap)
                b = ds41f.a64fx_decode(dataclasses.replace(ap, width=3, acceptance="a64fx_best512"))
                p = ds41f.a64fx_prefill(ap, args.prompt)
                print(f"{nodes:>5} {pk:<14}{prof:<9}{d['fit']['total_gib']:>9.1f}{'yes' if d['fit']['fits'] else 'NO':>5}{d['tok_s']:>7.1f}{b['tok_s']:>12.1f}{p['tok_s']:>8.0f}"
                      f"  {d['step_ms']:.1f} = experts {d['experts_ms']:.1f} + trunk {d['trunk_ms']:.1f} + attention {d['attn_ms']:.1f} + collectives {d['comm_ms']:.1f}"
                      f" + Engram {d['engram_ms']:.2f} + other {d['other_ms']:.1f}")


def cmd_validate(args):
    rows = ds41f.validate()
    bad = 0
    print(f"{'record':<74}{'measured':>9}{'predicted':>10}{'error':>8}  ok")
    for r in rows:
        print(f"{r['name'][:74]:<74}{r['measured']:>9.2f}{r['predicted']:>10.2f}{r['error'] * 100:>7.1f}%  {'yes' if r['within'] else 'NO'}"
              + ("" if r["fit"] else "  (cross-check)"))
        bad += not r["within"]
    print(f"{len(rows) - bad} of {len(rows)} rows within tolerance")
    return 1 if bad else 0


def md(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(header))) + "|"]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def cmd_report(args):
    g = ds41f.GEOMETRY
    print("Simulator outputs, not measurements. Every table's hypotheses are listed below it.\n")
    print("### Pack sizes\n")
    rows = []
    for name in ("mxfp4", "iq3xxs", "q23", "iq2xxs_q2k", "iq2s_q2k", "tetra", "iq2xxs", "iq1m", "iq1s", "iq2xxs_q2k_reap288", "iq2xxs_q2k_reap256"):
        P = ds41f.pack(name)
        rows.append([name, f"{P.bpw():.2f}", P.experts, f"{P.total_bytes() / 1e9:.1f}", f"{P.bytes_per_token() / 1e9:.2f}",
                     "yes" if P.lossless else "no", P.description])
    print(md(["pack", "bpw", "experts", "GB", "GB/token (6 active x 40)", "lossless", "formats"], rows))
    print(f"\nEngram: {g.engram_bytes() / 1e9:.1f} GB on SSD, {g.engram_rows_per_token()} rows x {g.engram_row_bytes} B per token. Trunk FP8 {g.trunk_fp8_bytes / 1e9:.2f} GB "
          f"per token (Q4_K {g.trunk_fp8_bytes * ds41f.TRUNK_SCALE['q4'] / 1e9:.2f}, Q8_0 {g.trunk_fp8_bytes * ds41f.TRUNK_SCALE['q8'] / 1e9:.2f}); head BF16 {g.head_bytes / 1e9:.2f} GB; KV {g.kv_bytes_per_token:.0f} B/token.")
    hyps = set()
    for machine in ("tr16", "tr16+b550", "xeon_v100"):
        m = ds41f.MACHINES[machine]()
        print(f"\n### {machine}\n\n{m.notes}\n")
        rows = []
        for pk, trunk, head in CANDIDATES[machine]:
            plan = ds41f.Plan(machine=machine, pack=pk, trunk=trunk, head=head, context=args.context)
            d = ds41f.decode(plan)
            b = ds41f.decode(dataclasses.replace(plan, width=3, acceptance="a64fx_best512"))
            s = ds41f.decode(dataclasses.replace(plan, width=3, acceptance="a64fx_sweep768"))
            p = ds41f.prefill(plan, prompt=args.prompt)
            d32 = ds41f.decode(dataclasses.replace(plan, context=32768))
            P = ds41f.pack(pk)
            hyps.update(d.hypotheses + p.hypotheses)
            rows.append([pk, f"{P.bpw():.2f}", f"{P.total_bytes() / 1e9:.0f}", trunk, "yes" if d.fit["fits"] else "**no**",
                         f"{d.fit['ram_need_gib']:.0f} / {d.fit['tier_gib']:.0f}", f"{d.tier_hit * 100:.0f} %",
                         f"{d.tok_s:.1f} / {b.tok_s:.1f} / {s.tok_s:.1f}", f"{d32.tok_s:.1f}", d.cpu_bound, f"{p.tok_s:.0f}",
                         f"{d.step_ms:.0f} = CPU {d.cpu_ms:.0f} + GPU {d.gpu_exposed_ms:.0f} + Engram {d.engram_exposed_ms:.1f} + head {d.head_ms:.0f}"
                         + (f" (remote wait {d.remote_ms:.0f})" if d.remote_ms else "")])
        print(md(["pack", "bpw", "GB", "trunk", "fits", "RAM GiB / tier GiB", "tier hit", f"decode {args.context // 1024}K plain / DSpark best / sweep", "decode 32K",
                  "CPU bound", f"prefill {args.prompt}", "step ms"], rows))
    print("\n### A64FX nodes\n")
    rows = []
    for nodes in (6, 8, 12):
        for pk in A64FX_CANDIDATES[nodes]:
            for prof in ("current", "redesign"):
                ap = ds41f.A64FXPlan(nodes=nodes, pack=pk, profile=prof, context=args.context)
                d = ds41f.a64fx_decode(ap)
                b = ds41f.a64fx_decode(dataclasses.replace(ap, width=3, acceptance="a64fx_best512"))
                p = ds41f.a64fx_prefill(ap, args.prompt)
                hyps.update(d["hypotheses"])
                rows.append([nodes, pk, prof, f"{d['fit']['total_gib']:.1f}", "yes" if d["fit"]["fits"] else "**no**", f"{d['tok_s']:.1f} / {b['tok_s']:.1f}",
                             f"{p['tok_s']:.0f}", f"{d['step_ms']:.1f} = experts {d['experts_ms']:.1f} + trunk {d['trunk_ms']:.1f} + attention {d['attn_ms']:.1f}"
                             f" + collectives {d['comm_ms']:.1f} + Engram {d['engram_ms']:.2f} + other {d['other_ms']:.1f}"])
    print(md(["nodes", "pack", "profile", "GiB/node", "fits", "decode plain / DSpark best", "prefill", "step ms"], rows))
    print("\nHypotheses: " + "; ".join(sorted(hyps)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name, fn, help_ in (("estimate", cmd_estimate, "one plan"), ("packs", cmd_packs, "candidate packs on one machine"),
                            ("memory", cmd_memory, "memory fit per pack"), ("validate", cmd_validate, "predicted vs measured"),
                            ("report", cmd_report, "markdown tables for docs/DS41F_PLAN.md")):
        p = sub.add_parser(name, help=help_)
        add_args(p)
        p.set_defaults(func=fn)
    p = sub.add_parser("a64fx", help="A64FX node counts, both profiles")
    add_args(p)
    p.add_argument("--nodes", type=int, nargs="+", default=[6, 8, 12])
    p.set_defaults(func=cmd_a64fx)
    args = ap.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
