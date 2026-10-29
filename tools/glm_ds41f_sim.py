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


def cmd_roofline(args):
    """The Strata runner on TR16 + RX 9070 XT (ds41f.RunnerHw): decode and prefill per profile and option."""
    print("decode (exact routing, ms per token)")
    print(f"{'config':<44}{'trunk':>7}{'experts':>9}{'other':>7}{'hit':>6}{'tok/s':>8}")
    rows = [("now (measured 12.5)", ds41f.RunnerPlan()),
            ("now + misses split CPU/PCIe", ds41f.RunnerPlan(set=dict(split_misses=True))),
            ("now + prefetch (+0.09 hit)", ds41f.RunnerPlan(set=dict(prefetch_gain=0.09))),
            ("target: tuned trunk, prefetch, split, syncs", ds41f.RunnerPlan(profile="target")),
            ("target without the split", ds41f.RunnerPlan(profile="target", set=dict(split_misses=False)))]
    for name, rp in rows:
        r = ds41f.runner_decode(rp)
        print(f"{name:<44}{r['trunk_ms']:>7.1f}{r['experts_ms']:>9.1f}{r['other_ms']:>7.1f}{r['hit']:>6.2f}{r['tok_s']:>8.1f}")
    print("\ntwo nodes: a B550 worker holds share s of every expert's FFN rows over IB (decode, ms per token)")
    print(f"{'config':<44}{'experts':>9}{'remote':>8}{'hit':>6}{'tok/s':>8}")
    for name, rp in [("now, alone", ds41f.RunnerPlan()),
                     ("now, s = 2/9 (fits the B550's 32 GB /dev/shm)", ds41f.RunnerPlan(remote_share=2 / 9)),
                     ("now, s = 3/9", ds41f.RunnerPlan(remote_share=3 / 9)),
                     ("now, s = 4/9", ds41f.RunnerPlan(remote_share=4 / 9)),
                     ("target, s = 3/9", ds41f.RunnerPlan(profile="target", remote_share=3 / 9))]:
        r = ds41f.runner_decode(rp)
        print(f"{name:<44}{r['experts_ms']:>9.1f}{r['remote_ms']:>8.1f}{r['hit']:>6.2f}{r['tok_s']:>8.1f}")
    print("\nprefill (tok/s; stream = every touched expert over PCIe once per pass)")
    prompts = (1024, 2048, 4096, 8192)
    print(f"{'config':<52}" + "".join(f"{p:>8}" for p in prompts))
    rows = [("now: CPU experts, chunks of 256", ds41f.RunnerPlan(mode="cpu_experts", chunk=256)),
            ("now: stream, chunks of 1024", ds41f.RunnerPlan(mode="stream", chunk=1024)),
            ("now: stream, one pass (chunk = prompt)", ds41f.RunnerPlan(mode="stream", chunk=1 << 20)),
            ("target: stream, one pass", ds41f.RunnerPlan(profile="target", mode="stream", chunk=1 << 20)),
            ("target: stream, one pass, enc-skip 2560", ds41f.RunnerPlan(profile="target", mode="stream", chunk=1 << 20, enc_skip=2560)),
            ("target, one pass, enc-skip, 15 GB of experts on SSD", ds41f.RunnerPlan(profile="target", mode="stream", chunk=1 << 20, enc_skip=2560, uncached_gb=15))]
    for name, rp in rows:
        print(f"{name:<52}" + "".join(f"{ds41f.runner_prefill(rp, p)['tok_s']:>8.0f}" for p in prompts))
    return 0


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


def cmd_v100(args):
    """The Strata runner on 2x Tesla V100 (ds41f.V100Hw): decode levers, batched sessions, long context."""
    for label, base in (("IQ2_XXSE4, Engram 76 GiB in RAM", None), ("IQ2_XXSE3, Engram whole in RAM", dict(pack="IQ2_XXSE3", engram_gib=79))):
        print(f"decode levers, DSpark, wikitext acceptance - {label}")
        print(f"{'lever':<56}{'tok/s':>7}{'step':>8}{'draft':>7}{'pass':>7}{'Engram':>8}{'hit':>7}")
        for name, plan in ds41f.v100_levers(base):
            d = ds41f.v100_decode(plan)
            print(f"{name:<56}{d['tok_s']:>7.1f}{d['step_ms']:>8.1f}{d['draft_ms']:>7.1f}{d['dense_ms']:>7.1f}{d['engram_ms']:>8.1f}{d['hit']:>7.3f}")
        print()
    print("packs: decode (DSpark; wiki600 2.6 drafted / 0.68, code600 4.2 / 0.65, 106K Q8 KV) and streamed prefill (4K, 8K) tok/s")
    print(f"{'pack':<52}{'wiki600':>8}{'code600':>8}{'106K':>7}{'pf 4K':>7}{'pf 8K':>7}")
    for label, w, c, l, p4, p8 in ds41f.v100_packs_compare():
        print(f"{label:<52}{w:>8.1f}{c:>8.1f}{l:>7.1f}{p4:>7.0f}{p8:>7.0f}")
    for name in ("IQ2_XXSE4", "GSQ-IQ2_XXS"):
        pk = ds41f.V100_PACKS[name]
        print(f"  {name}: experts {pk.expert_gib:.1f} GiB, mean expert {pk.expert_mb:.2f} MB, lane-effective {pk.lane_mb_eff({'IQ1_M': 0.6}):.2f} MB at IQ1_M 0.6")
    print()
    print("batched sessions, IQ2_XXSE4 (aggregate / per session tok/s)")
    for s, spec, d in ds41f.v100_sessions():
        print(f"  S={s:<3}{'DSpark' if spec else 'plain':<8}{d['tok_s']:>7.1f}{d['per_session_tok_s']:>7.1f}   step {d['step_ms']:.1f} ms")
    print("\nlong context, IQ2_XXSE4 DSpark (index scoring per verify pass; KV off the expert tier)")
    for tc in (False, True):
        for ctx, kv, d in ds41f.v100_context(index_tc=tc):
            print(f"  {ctx:>8} {kv:<5}{' FP16 TC' if tc else '':<9}{d['tok_s']:>7.1f} tok/s   index {d['index_ms']:5.1f} ms   KV {d['kv_gib']:5.2f} GiB   hit {d['hit']:.3f}")


def cmd_levers(args):
    """The 2026-10-26 research levers (docs/DS41F_PLAN.md 8k): V100 decode and prefill, TR16."""
    print("V100 node, Engram RAM 56 GiB, DSpark (5 draft positions): decode tok/s by lever (phase model, calibrated to 8j)")
    print(f"{'configuration':<52}{'tok/s':>7}{'tok/step':>9}{'k':>5}{'step':>7}{'KL':>7}")
    for label, lp in ds41f.lever_table():
        r = ds41f.lever_decode(lp)
        print(f"{label:<52}{r['tok_s']:>7.1f}{r['tokens_per_step']:>9.2f}{r['k']:>5.1f}{r['step_ms']:>7.0f}{r['kl']:>7.3f}")
    print("\nV100 node, streamed prefill tok/s (IQ2_XXSE4 / E3 experts; f16 = FP16 tensor-core expert GEMM gain)")
    print(f"{'configuration':<40}{'4K':>7}{'8K':>7}{'32K':>7}{'106K':>7}")
    for label, kw in (("batch 2048", dict(batch=2048)),
                      ("batch 4096 (measured 389 / 412 / - / 377)", dict(batch=4096)), ("batch 8192", dict(batch=8192)),
                      ("batch 4096 + f16 x1.5", dict(batch=4096, f16_gain=1.5)), ("batch 4096 + f16 x2", dict(batch=4096, f16_gain=2.0)),
                      ("batch 8192 + f16 x2", dict(batch=8192, f16_gain=2.0))):
        vals = [ds41f.lever_prefill(n, **kw)["tok_s"] for n in (4096, 8192, 32768, 106000)]
        print(f"{label:<40}" + "".join(f"{v:>7.0f}" for v in vals))
    for prof in ("now", "target"):
        rows, info = ds41f.tr16_levers(prof)
        print(f"\nTR16 (128 GB + RX 9070 XT 16 GB), UD-IQ1_S, Engram on NVMe, runner profile '{prof}' (plain {info['base_tok_s']:.1f} tok/s, hit {info['hit']:.2f})")
        for label, tok_s, step in rows:
            print(f"  {label:<58}{tok_s:>7.1f} tok/s  step {step:6.1f} ms")
        print(f"  VRAM left for the expert tier + drafter: {info['vram_for_experts_and_drafter_gib']:.1f} GiB (DSpark drafter {info['drafter_gib']} GiB)")


def cmd_two_node(args):
    base = ds41f.TwoNodePlan(width=args.width, canon_mib=args.canon_mib, trim=args.trim, grid8=not args.no_grid8)
    print("TR16 + B550, UD-IQ1_S, worker owning 48 GiB, DSpark, calibrated to the ten 2026-10-29 runs "
          "(tools/sim/data/measured_ds41f_two_node.json; generic `estimate` gives 15.0 plain / 33.2 DSpark for this pair)")
    for label, o in ds41f.two_node_levers(base):
        print(f"  {label:<56}{o['tok_s']:6.1f} tok/s  verify {o['verify_ms']:6.1f} ms = experts {o['cpu_ms']:5.1f} + gpu wait {o['g2r_ms']:4.1f} "
              f"+ engram {o['engram_ms']:4.1f} + head {o['head_ms']:3.1f} + other {o['other_ms']:3.1f}; round {o['round_ms']:6.1f}; "
              f"tokens/round {o['tokens_per_round']:.2f}; canon hit {o['canon_hit']:.3f}; VRAM free {o['vram_free_gib']:.2f} GiB")
    print("\n  Roadmap to 2x (cumulative what-ifs, docs: plan 'research-driven 2x decode'; lossless 50 / gated-lossy 60):")
    for label, o in ds41f.two_node_roadmap(base):
        print(f"  {label:<76}{o['tok_s']:6.1f} tok/s  round {o['round_ms']:6.1f} ms (experts {o['cpu_ms']:5.1f}, gpu {o['g2r_ms']:4.1f}, draft {o['draft_ms']:4.1f}); "
              f"prefetch {o['prefetch_moved']:.1f}/{o['prefetch_capacity']:.1f} experts per layer")
    print("  residuals against the measured configuration means: exact +2.7 %, grid8 off +6.5 %, trim 0.125 -4.3 %, trim 0.15 +5.3 % "
          "(individual runs span +/-4.5 %). Unmeasured inputs of the [mine] levers: the root lane's busy ms per layer (root_steal), the stagger efficiency 0.6, the draft GPU contention 0.6 / 0.1. Hypotheses: widths other than 1 and 5, N, the trim curve outside 0.125-0.15 and canon sizes "
          "outside 2302-3458 MiB are extrapolations; what-if rows are not measured.")


def cmd_two_node_prefill(args):
    base = ds41f.TwoNodePrefillPlan(prompt=args.prompt, calibration=args.calibration)
    print(f"TR16 (RX 9070 XT) + B550 (RTX 5060 Ti) whole-expert prefill, {args.prompt} tokens, calibration '{args.calibration}'. UNCALIBRATED for two nodes: "
          "anchored on the single-node 12.7 s / 645 tok/s pass; no two-node prefill has been measured.")
    for label, o in ds41f.two_node_prefill_levers(base):
        print(f"  {label:<72}{o['seconds']:6.2f} s {o['tok_s']:6.0f} tok/s  {'MEETS 1200' if o['meets_1200'] else '          '} bound {o['bound']:<12} "
              f"worker share {o['share']:.2f} (routes {o['route_share']:.2f}); DMA root/worker {o['root_dma_s']:.1f}/{o['worker_dma_s']:.1f} s, "
              f"waits {o['root_wait_s']:.1f}/{o['worker_wait_s']:.1f}; chain {o['chain_s']:.1f} s; IB out/back {o['link_out_s']:.1f}/{o['link_back_s']:.1f} s")
    print("  Hypotheses: copy overlap f = 0.527 from the one 15.22 s profiled pass (0.727 under the 'overlap' calibration); worker GPU as fast as the root's; "
          "route share on the worker = 1.16 p^0.545 of its expert-byte share p (GLM tier curve); worker PCIe 14.38 GB/s measured (root 12.5 GB/s from the profiled pass; 9.6 sustained when re-measured with a miner running); IB 2.44 GB/s; only 8192 tokens "
          "anchored (other prompts scale linearly, attention growth ignored); decode-cache VRAM borrowing, staging-slot limits and cold setup not modeled.")


def cmd_sensitivity(args):
    r = ds41f.two_node_sensitivity(draws=args.draws, perturb=args.perturb)
    print(f"Robustness of the lossless 2x stack (full stack {r['full']:.1f} tok/s; every lever is a what-if on the calibrated TR16+B550 model)")
    print("  lever                alone   stack without it")
    for k in r["alone"]:
        print(f"  {k:<20}{r['alone'][k]:6.1f}   {r['without'][k]:6.1f}  (worth {r['full'] - r['without'][k]:+.1f})")
    print(f"  to still reach 50: prefetch_share >= {r['min_prefetch_share']}, gpu_wait_scale <= {r['max_gpu_wait_scale']} (others at plan values)")
    a, b = r["mc_lossless"], r["mc_lossy"]
    print(f"  Monte Carlo over the lever ranges ({args.draws} draws): lossless p10 {a['p10']:.1f} / median {a['median']:.1f} / p90 {a['p90']:.1f}, "
          f"P(>=50) = {a['p_target']:.0%}; with deferral p10 {b['p10']:.1f} / median {b['median']:.1f} / p90 {b['p90']:.1f}, P(>=60) = {b['p_target']:.0%}")
    print(f"  fitted constants -/+ {args.perturb:.0%} (full stack tok/s):")
    for k, (lo, hi) in r["constants"].items():
        print(f"    {k:<20}{lo:6.1f} .. {hi:6.1f}")


def cmd_no_root_gpu(args):
    chk = ds41f.no_root_gpu_check()
    print("Without the RX 9070 XT: decoder on the B550's RTX 5060 Ti, expert lanes on the B550 and TR16 CPUs (TR16 over IB). Inputs measured 2026-10-29 "
          "(tools/sim/data/measured_ds41f_cpu_gpu_bench.json); the same expert-phase formula on today's layout gives "
          f"{chk:.1f} ms against the measured 88-91 ms. No end-to-end run of this layout exists: the worker needs a CUDA device and the CUDA decoder is untested.")
    for label, o in ds41f.no_root_gpu_levers(ds41f.NoRootGpuPlan(width=args.width)):
        print(f"  {label:<80}{o['tok_s']:6.1f} tok/s  round {o['round_ms']:6.1f} ms; expert phase {o['expert_ms']:5.1f} ms "
              f"(layer {o['layer_ms']:.2f} ms: lanes {o['lane_experts']:.1f} + GPU {o['gpu_experts']:.1f} experts); tier {o['tier_slots']:.0f} slots, hit {o['hit']:.2f}")
    print("  Hypotheses: the layer overhead (1.0 ms) and merge (0.31 ms) are today's; GPU trunk/attention terms are today's 9070 XT figures (the 5060 Ti may differ); "
          "TR16/B550 lane ratio 0.65 (measured 0.58-0.71 with a proxy kernel); the faster, tensor-core and streaming kernels and every lever below the first row are what-ifs.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name, fn, help_ in (("estimate", cmd_estimate, "one plan"), ("packs", cmd_packs, "candidate packs on one machine"),
                            ("memory", cmd_memory, "memory fit per pack"), ("validate", cmd_validate, "predicted vs measured"),
                            ("report", cmd_report, "markdown tables for docs/DS41F_PLAN.md"),
                            ("roofline", cmd_roofline, "the Strata runner on TR16 + RX 9070 XT: decode and prefill per profile")):
        p = sub.add_parser(name, help=help_)
        add_args(p)
        p.set_defaults(func=fn)
    p = sub.add_parser("v100", help="the Strata runner on 2x Tesla V100: decode levers, sessions, long context")
    p.set_defaults(func=cmd_v100)
    p = sub.add_parser("two-node", help="the measured TR16 + B550 DSpark runner: calibrated decode and what-ifs")
    p.add_argument("--width", type=int, default=5)
    p.add_argument("--canon-mib", type=float, default=-1.0)
    p.add_argument("--trim", type=float, default=0.0)
    p.add_argument("--no-grid8", action="store_true")
    p.set_defaults(func=cmd_two_node)
    p = sub.add_parser("two-node-prefill", help="two-node whole-expert prefill estimate and levers (uncalibrated for two nodes)")
    p.add_argument("--prompt", type=int, default=8192)
    p.add_argument("--calibration", choices=("compute", "overlap"), default="compute")
    p.set_defaults(func=cmd_two_node_prefill)
    p = sub.add_parser("no-root-gpu", help="decode without the 9070 XT: decoder on the B550's 5060 Ti, lanes on both CPUs")
    p.add_argument("--width", type=int, default=5)
    p.set_defaults(func=cmd_no_root_gpu)
    p = sub.add_parser("sensitivity", help="robustness of the 2x-decode roadmap: ablations, thresholds, Monte Carlo, constant perturbation")
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--perturb", type=float, default=0.15)
    p.set_defaults(func=cmd_sensitivity)
    p = sub.add_parser("levers", help="research levers: V100 decode / prefill and TR16 estimates")
    p.set_defaults(func=cmd_levers)
    p = sub.add_parser("a64fx", help="A64FX node counts, both profiles")
    add_args(p)
    p.add_argument("--nodes", type=int, nargs="+", default=[6, 8, 12])
    p.set_defaults(func=cmd_a64fx)
    args = ap.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
