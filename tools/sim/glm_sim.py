"""Estimate GLM-5.3-Flash decode and prefill speed for a hardware configuration, without the hardware.

Subcommands:
  estimate   one configuration: tok/s, per-step breakdown, VRAM plan, bottleneck
  sweep      a grid over knobs (--affinity 0 0.05 0.1 --decode-cache-mib 3000 4800 ...)
  validate   predicted vs measured for every record in tools/sim/data/measured_tr16.json
  fit        re-tune the free parameters on the records marked fit:true and write data/params.json
  plan       search pack / tier / MTP depth / affinity / chunk / remote share for a target on given hardware
  hw         print a hardware preset as JSON (edit it and pass the file back with --hw)

Examples:
  python3 tools/sim/glm_sim.py estimate --hw tr16 --pack q23 --decode-cache-mib 4800 --mtp 2 --prompt 1024
  python3 tools/sim/glm_sim.py estimate --hw tr16 --set pcie.h2d_gbps=25 --set memory.dram_gbps=180
  python3 tools/sim/glm_sim.py sweep --hw tr16 --affinity 0 0.05 0.1 --decode-cache-mib 3000 4800 6500
  python3 tools/sim/glm_sim.py plan --hw tr16 --target-decode 30
"""
import argparse
import dataclasses
import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import calibration  # noqa: E402
import decode  # noqa: E402
import hw as hwmod  # noqa: E402
import model  # noqa: E402
import quality  # noqa: E402
import prefill  # noqa: E402
import routing  # noqa: E402
from config import RunConfig  # noqa: E402


def add_hw_args(ap):
    ap.add_argument("--hw", default="tr16", help="preset name or JSON file (presets: %s)" % ", ".join(hwmod.PRESETS))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a hardware field, e.g. memory.dram_gbps=90, pcie.h2d_gbps=25, gpu.vram_mib=24000")
    ap.add_argument("--params", default=None, help="params.json (default: tools/sim/data/params.json if present)")
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                    help="override a calibrated parameter (kernels.Params), e.g. gpu_kernels_per_layer=10, cpu_quant_scale=2")


def add_config_args(ap, sweep=False):
    """One option per RunConfig field; in sweep mode every option takes several values."""
    defaults = RunConfig()

    def opt(flag, field, **kw):
        default = getattr(defaults, field)
        if isinstance(default, bool):
            default, kw["type"] = int(default), int
        elif "type" not in kw and "choices" not in kw:
            kw["type"] = type(default)
        if sweep:
            kw["nargs"] = "+"
            default = [default]
        ap.add_argument(flag, dest=field, default=default, **kw)

    opt("--pack", "pack", choices=list(model.PACKS))
    opt("--threads", "threads")
    opt("--context", "context")
    opt("--prompt", "prompt")
    opt("--generate", "generate")
    opt("--speculation", "speculation", choices=["none", "mtp", "dflash", "selfspec"])
    opt("--mtp", "mtp_depth", help="mtp: draft tokens per round (0 = ordinary decode)")
    opt("--acceptance", "acceptance", choices=list(routing.ACCEPTANCE))
    opt("--draft-block", "draft_block", help="dflash: tokens verified per round (DFlash2: 8)")
    opt("--draft-acceptance", "draft_acceptance", choices=[k for k in routing.ACCEPTANCE if k.startswith("dflash")])
    opt("--draft-model-mib", "draft_model_mib")
    opt("--max-verify-width", "max_verify_width", help="engine cap on tokens per step (cpu::MAXT = 8); raise to explore")
    opt("--expert-skip", "expert_skip", help="share of routed expert bytes skipped by gate thresholds (lossy)")
    opt("--pcie-share", "pcie_share", help="share of non-resident expert bytes the GPU streams over PCIe and computes")
    opt("--pcie-prefetch", "pcie_prefetch", help="1 = predict next-layer routes and prefetch those experts over PCIe (lossless)")
    opt("--expert-deferral", "expert_deferral", help="1 = add part of each layer's routed output one layer late (lossy)")
    opt("--deferral-share", "deferral_share", help="share of routed work that may be deferred (quality cost grows with it)")
    opt("--adaptive-window", "adaptive_window", help="1 = stop drafting at low confidence; shorter verify windows (lossless)")
    opt("--spec-tail-topk", "spec_tail_topk", help="experts per route for draft positions >= 2 (AcceptMoE-style; lossy below 8)")
    opt("--cost-aware-drafts", "cost_aware_drafts", help="1 = EcoSpec-style draft choice that reuses active experts")
    opt("--draft-prefetch", "draft_prefetch", help="1 = fetch RAM-tier misses a round ahead from the drafts' routes")
    opt("--cold-share", "cold_share", help="share of experts (coldest) stored in a smaller format (lossy)")
    opt("--cold-layers", "cold_layers", help="MoE layers stored at --cold-scale (per-layer bit budget)")
    opt("--batch-mtp", "batch_mtp", help="1 = rows' draft windows share one step capped at --max-verify-width")
    opt("--cold-scale", "cold_scale", help="cold expert bytes relative to the pack format")
    opt("--tail-affinity", "tail_affinity", help="route affinity for draft positions >= 2 only (lossy)")
    opt("--tier-compress", "tier_compress", help="GPU tier expert format density: slots x this (lossy above 1)")
    opt("--dense-format", "dense_format", choices=list(model.DENSE_FORMATS))
    opt("--split-verify", "split_verify")
    opt("--decode-cache-mib", "decode_cache_mib", help="GPU expert tier MiB; -1 = fill the budget, 0 = none")
    opt("--tier-policy", "tier_policy", choices=["static", "static_prior", "adaptive", "lru"])
    opt("--affinity", "affinity")
    opt("--gpu-budget-mib", "gpu_budget_mib")
    opt("--reserve-mib", "reserve_mib")
    opt("--prefill-scratch-mib", "prefill_scratch_mib")
    opt("--prefill-chunk", "prefill_chunk")
    opt("--prefetch-groups", "prefetch_groups")
    opt("--prefill-legacy", "prefill_legacy", help="1 = dequantize-per-use GEMMs (before P04)")
    opt("--prefill-mla-f16", "prefill_mla_f16")
    opt("--prefill-kda-parts", "prefill_kda_parts")
    opt("--prefill-stream-depth", "prefill_stream_depth", help="groups in flight beyond the ring; >= 18 = continuous streaming")
    opt("--prefill-gemm-scale", "prefill_gemm_scale", help="relative MoE GEMM speed (what-if)")
    opt("--prefill-experts", "prefill_experts", choices=["gpu", "cpu", "auto"])
    opt("--prefill-cpu-assist", "prefill_cpu_assist", help="1 = CPU computes experts for part of the chunk alongside the GPU")
    opt("--batch", "batch")
    opt("--placement", "placement", choices=["cpu", "gpu_stream", "ram_tier"])
    opt("--ram-mode", "ram_mode", choices=["exact", "frozen", "hybrid"])
    opt("--ram-margin", "ram_margin")
    opt("--remote-share", "remote_share")
    opt("--gpus", "gpus")
    ap.add_argument("--trace", default="", help="routing trace CSV (position,layer,e0..e7) for exact union/hit rates")
    ap.add_argument("--prior", default="", help="coverage.json for trace replay (default: build-q2-redesign/calibration)")


CONFIG_FIELDS = [f.name for f in dataclasses.fields(RunConfig)]


def config_from_args(args, **overrides):
    data = {name: getattr(args, name) for name in CONFIG_FIELDS if hasattr(args, name)}
    data.update(overrides)
    for key in ("split_verify", "prefill_legacy", "pcie_prefetch", "expert_deferral", "adaptive_window",
                "prefill_mla_f16", "prefill_kda_parts", "prefill_cpu_assist", "cost_aware_drafts", "draft_prefetch", "batch_mtp"):
        if key in data:
            data[key] = bool(data[key])
    return RunConfig.from_dict(data)


def hardware_from_args(args):
    hw = hwmod.load(args.hw)
    for item in args.set:
        key, _, value = item.partition("=")
        hw.set(key, value)
    return hw


def params_from_args(args):
    params = calibration.load_params(args.params)
    for item in getattr(args, "param", []):
        key, _, value = item.partition("=")
        if not hasattr(params, key):
            raise SystemExit(f"unknown parameter {key}; see kernels.Params")
        current = getattr(params, key)
        setattr(params, key, type(current)(float(value)) if isinstance(current, int) else float(value))
    return params


def load_trace(args):
    if not args.trace:
        return None, None
    prior = pathlib.Path(args.prior) if args.prior else routing.default_prior_path()
    return routing.read_routes(args.trace), (prior if prior.exists() else None)


def fmt_table(rows, columns):
    widths = [max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns]
    lines = ["  ".join(c.ljust(w) for c, w in zip(columns, widths))]
    lines.append("  ".join("-" * w for w in widths))
    for r in rows:
        lines.append("  ".join(str(r.get(c, "")).ljust(w) for c, w in zip(columns, widths)))
    return "\n".join(lines)


def estimate(args):
    hw = hardware_from_args(args)
    params = params_from_args(args)
    cfg = config_from_args(args)
    trace, prior = load_trace(args)
    d = decode.simulate(hw, cfg, params, trace=trace, prior=prior)
    p = prefill.simulate(hw, cfg, params)
    plan = decode.make_plan(hw, cfg, model.pack(cfg.pack))
    s = d.step
    print(f"hardware {hw.name}: {hw.cpu.name} x{hw.cpu.cores} cores @ {hw.cpu.ghz} GHz ({hw.cpu.peak_gflops():.0f} GFLOPS peak), "
          f"DRAM {hw.memory.dram_gbps} GB/s, {hw.gpu.name} {hw.gpu.vram_mib:.0f} MiB {hw.gpu.bandwidth_gbps} GB/s, "
          f"PCIe gen{hw.pcie.gen} x{hw.pcie.lanes} ({hw.pcie.h2d():.1f} GB/s H2D)")
    print(f"pack {cfg.pack}: {model.pack(cfg.pack).description}")
    spec = (f"mtp depth {cfg.mtp_depth} ({cfg.acceptance})" if cfg.speculation == "mtp" else
            f"dflash block {cfg.draft_block} ({cfg.draft_acceptance}, {cfg.draft_model_mib:.0f} MiB drafter)" if cfg.speculation == "dflash"
            else "no speculation")
    print(f"config: {spec}, split {cfg.split_verify}, tier {cfg.tier_policy} "
          f"{d.tier_mib:.0f} MiB = {d.tier_slots} experts, affinity {cfg.affinity}, placement {cfg.placement}, "
          f"batch {cfg.batch}, remote share {cfg.remote_share}, context {cfg.context}")
    print()
    print(f"DECODE  {d.tok_s:6.2f} tok/s   ({d.tokens_per_round:.2f} tokens per {d.round_ms:.1f} ms round; "
          f"bottleneck: {d.bottleneck}; {'lossless' if d.lossless else 'LOSSY'})")
    print(f"  round: draft {d.draft_ms:.1f} + verify/step {d.verify_ms:.1f} + resync {d.resync_ms:.1f} ms")
    print(f"  step (width {s.width} x {s.rows} rows, {s.groups} group(s)): head {s.head_ms:.1f} | CPU experts {s.cpu_ms:.1f} "
          f"| GPU total {s.gpu_ms:.1f} (exposed {s.gpu_exposed_ms:.1f}) | handoffs {s.handoff_ms:.1f} | tail {s.tail_ms:.1f} "
          f"| tier uploads {s.tier_upload_ms:.1f} | disk {s.disk_ms:.1f} | pcie {s.pcie_ms:.1f} | remote {s.remote_ms:.1f} ms")
    print(f"  bytes/token: CPU {d.cpu_gb_per_token:.2f} GB ({d.cpu_gbps:.1f} GB/s, {s.cpu_bound}-bound), GPU tier {d.gpu_gb_per_token:.2f} GB "
          f"(hit {d.hit_bytes_share * 100:.1f} %), disk {d.disk_gb_per_token:.2f} GB; union ratio {s.union_ratio:.2f}")
    kl, parts = quality.estimate_kl(cfg)
    print(f"  quality (rough): KL ~{kl:.3f} = " + " + ".join(f"{k} {v:.3f}" for k, v in parts.items()))
    print(f"  ceilings: CPU-expert-bound {d.cpu_ceiling_tok_s:.1f} tok/s, GPU-bound {d.gpu_ceiling_tok_s:.1f} tok/s "
          f"(each with the other side fully overlapped)")
    for note in d.notes:
        print(f"  note: {note}")
    print()
    print(f"PREFILL {p.tok_s:6.1f} tok/s   ({cfg.prompt} tokens in {p.total_s:.1f} s, {p.chunks} chunk(s) of {p.chunk_tokens}; bottleneck: {p.bottleneck})")
    print(f"  per chunk: PCIe {p.pcie_gb_per_chunk:.1f} GB in {p.pcie_s_per_chunk:.1f} s, GPU {p.gpu_s_per_chunk:.1f} s "
          f"(mixers {p.mixer_s_per_chunk:.1f}, GEMMs {p.gemm_s_per_chunk:.1f}), CPU experts {p.cpu_s_per_chunk:.1f} s, "
          f"disk {p.disk_s_per_chunk:.1f} s; experts via {p.experts}")
    print()
    req = decode.request_seconds(hw, cfg, params, d, p)
    print(f"REQUEST {cfg.prompt} + {cfg.generate} tokens: {req:.1f} s (first {min(cfg.generate, params.cold_tier_tokens):.0f} "
          f"tokens at {params.cold_tier_speed:.0%} speed while the tier warms)")
    if args.mc:
        import random
        rng = random.Random(1)
        ranges = dict(cpu_bw_scale=0.08, gpu_layer_fixed_us=0.3, tier_upload_ms=0.5, prefetch_accuracy=0.1,
                      truncation_efficiency=0.4, ecospec_union=0.08, tail_topk_acceptance=0.02, tier_expert_us=0.5,
                      prefill_overlap=0.2, prefill_layer_fixed_ms=0.3)
        dec, pre = [], []
        for _ in range(args.mc):
            q = dataclasses.replace(params)
            for k, r in ranges.items():
                setattr(q, k, getattr(q, k) * (1 + rng.uniform(-r, r)))
            q.prefetch_accuracy = min(1.0, q.prefetch_accuracy)
            dec.append(decode.simulate(hw, cfg, q).tok_s)
            pre.append(prefill.simulate(hw, cfg, q).tok_s)
        dec.sort(); pre.sort()
        pick = lambda xs, f: xs[min(len(xs) - 1, int(f * len(xs)))]
        print(f"UNCERTAINTY ({args.mc} samples over parameter ranges): decode p10/p50/p90 {pick(dec, .1):.1f} / "
              f"{pick(dec, .5):.1f} / {pick(dec, .9):.1f} tok/s, prefill {pick(pre, .1):.0f} / {pick(pre, .5):.0f} / {pick(pre, .9):.0f}")
    print("VRAM plan (MiB): " + ", ".join(f"{k} {v:.0f}" for k, v in plan.table()))
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(dict(
            hardware=hw.to_dict(), config=dataclasses.asdict(cfg), decode=d.to_dict(), prefill=p.to_dict(),
            vram=dict(items=plan.items, tier_mib=plan.tier_mib, slots=plan.slots)), indent=2) + "\n")


def sweep(args):
    hw = hardware_from_args(args)
    params = params_from_args(args)
    names = [n for n in CONFIG_FIELDS if hasattr(args, n) and isinstance(getattr(args, n), list)]
    varying = [n for n in names if len(getattr(args, n)) > 1]
    rows = []
    for values in itertools.product(*(getattr(args, n) for n in names)):
        cfg = config_from_args(args, **dict(zip(names, values)))
        d = decode.simulate(hw, cfg, params)
        p = prefill.simulate(hw, cfg, params)
        row = {n: getattr(cfg, n) for n in varying}
        row.update(kl=f"{d.kl_estimate:.3f}", decode_tok_s=f"{d.tok_s:.2f}", tok_round=f"{d.tokens_per_round:.2f}", round_ms=f"{d.round_ms:.0f}", cpu_ceiling=f"{d.cpu_ceiling_tok_s:.1f}", gpu_ceiling=f"{d.gpu_ceiling_tok_s:.1f}", prefill_tok_s=f"{p.tok_s:.1f}", tier_mib=f"{d.tier_mib:.0f}",
                   hit=f"{d.hit_bytes_share:.2f}", cpu_ms=f"{d.step.cpu_ms:.1f}", gpu_ms=f"{d.step.gpu_ms:.1f}",
                   bottleneck=d.bottleneck, lossless=d.lossless)
        rows.append(row)
    print(fmt_table(rows, varying + ["kl", "decode_tok_s", "tok_round", "round_ms", "cpu_ceiling", "gpu_ceiling", "prefill_tok_s", "tier_mib", "hit", "cpu_ms", "gpu_ms", "bottleneck", "lossless"]))
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")


def validate(args):
    params = calibration.load_params(args.params)
    records = calibration.load_records(args.records)
    rows = calibration.evaluate(records, params)
    table = []
    for r in rows:
        table.append(dict(name=r["name"][:60], kind=r["kind"], measured=f"{r['measured']:.2f}", predicted=f"{r['predicted']:.2f}",
                          error=f"{r['error'] * 100:+.1f}%", tol=f"{r['tolerance'] * 100:.0f}%",
                          ok="ok" if r["within"] else "MISS", quiet="q" if r["quiet"] else "n", fit="f" if r["fit"] else "",
                          bottleneck=r["bottleneck"]))
    print(fmt_table(table, ["name", "kind", "measured", "predicted", "error", "tol", "ok", "quiet", "fit", "bottleneck"]))
    fit_rows = [r for r in rows if r["fit"]]
    misses = [r for r in rows if not r["within"]]
    mean_err = sum(abs(r["error"]) for r in fit_rows) / max(1, len(fit_rows))
    print(f"\n{len(rows)} records, {len(misses)} outside tolerance; mean |error| on fit rows {mean_err * 100:.1f}%")
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
    return 1 if any(not r["within"] and r["fit"] for r in rows) else 0


def fit(args):
    params = calibration.load_params(args.params)
    records = calibration.load_records(args.records)
    names = args.names or calibration.FIT_PARAMETERS
    fitted, score = calibration.fit(records, params, names, rounds=args.rounds)
    out = args.output or calibration.PARAMS_FILE
    calibration.save_params(fitted, out, extra=dict(mean_abs_log_error=score, records=str(args.records or calibration.MEASURED_FILE)))
    print(f"wrote {out}")


def plan(args):
    hw = hardware_from_args(args)
    params = params_from_args(args)
    base = config_from_args(args)
    ram_gib = hw.memory.gib + hw.memory.page_cache_gib
    candidates = []
    packs = args.packs or ["q23", "q22", "reap50_q23", "q2_orig"]
    if not args.baseline:
        base = base.copy(tier_policy="lru", pcie_prefetch=True, adaptive_window=True)
    affinities = [0.0] if args.lossless else [0.0, 0.05, 0.1]
    tails = [8] if args.lossless else [8, 4]
    specs = [("none", 0), ("mtp", 1), ("mtp", 2), ("mtp", 3), ("mtp", 5), ("dflash", 8)]
    if args.max_verify_width > 8:
        specs.append(("dflash", args.max_verify_width))
    skips = [0.0] if args.lossless else [0.0, 0.1]
    denses = ["q8"] if args.lossless else ["q8", "q4"]
    chunks = [1024, 2048, 4096, 8192]
    shares = [0.0] + ([0.25, 0.5] if hw.remote.expert_gbps > 0 else [])
    for pack_name, (spec, depth), aff, chunk, share, skip, dense, tail in itertools.product(
            packs, specs, affinities, chunks, shares, skips, denses, tails):
        if tail < 8 and depth < 2:
            continue
        pack = model.pack(pack_name)
        fits = pack.routed_total_bytes() / 2 ** 30 * (1 - share) + 7 <= ram_gib
        if share > 0 and hw.remote.ram_gib and pack.routed_total_bytes() / 2 ** 30 * share + 2 > hw.remote.ram_gib:
            continue
        placements = ["cpu"] if fits else (["ram_tier"] if not args.lossless else ["ram_tier"])
        for placement in placements:
            modes = ["exact"] if placement == "cpu" else (["exact"] if args.lossless else ["exact", "hybrid", "frozen"])
            for mode in modes:
                small_card = hw.gpu.vram_mib < 12000     # 8 GB class: trim scratch and reserve to leave room for a tier
                cfg = base.copy(pack=pack_name, speculation=spec, mtp_depth=depth if spec == "mtp" else 0,
                                draft_block=depth if spec == "dflash" else base.draft_block,
                                affinity=aff, prefill_chunk=chunk, remote_share=share, expert_skip=skip, dense_format=dense,
                                spec_tail_topk=tail,
                                placement=placement, ram_mode=mode, split_verify=depth >= 1,
                                prefill_scratch_mib=256 if small_card else max(1024, 1024 * chunk // 4096),
                                reserve_mib=128 if small_card else base.reserve_mib,
                                context=min(base.context, 2048) if small_card else base.context)
                if chunk > cfg.prompt:
                    continue
                vplan = decode.make_plan(hw, cfg, pack)
                if sum(vplan.items.values()) > vplan.budget_mib:
                    continue        # the fixed residents alone do not fit this card
                d = decode.simulate(hw, cfg, params)
                p = prefill.simulate(hw, cfg, params)
                if args.lossless and not d.lossless:
                    continue
                if args.kl_budget is not None and d.kl_estimate > args.kl_budget:
                    continue
                candidates.append((d.tok_s, p.tok_s, cfg, d, p))
    if args.pareto:
        front = []
        for c in candidates:
            dominated = any(o[0] >= c[0] and o[1] >= c[1] and o[3].kl_estimate <= c[3].kl_estimate
                            and (o[0], o[1], o[3].kl_estimate) != (c[0], c[1], c[3].kl_estimate) for o in candidates)
            if not dominated:
                front.append(c)
        candidates = sorted(front, key=lambda c: c[3].kl_estimate)
    else:
        candidates.sort(key=lambda c: (-(c[0] if c[0] <= args.target_decode else args.target_decode + 1e-3 * c[0]), -c[1]))
    rows = []
    for tok_s, pf, cfg, d, p in candidates[:args.top]:
        spec = f"mtp{cfg.mtp_depth}" if cfg.speculation == "mtp" and cfg.mtp_depth else (
            f"dflash{min(cfg.draft_block, cfg.max_verify_width)}" if cfg.speculation == "dflash" else "none")
        rows.append(dict(pack=cfg.pack, spec=spec, affinity=cfg.affinity, skip=cfg.expert_skip, dense=cfg.dense_format, tail=cfg.spec_tail_topk,
                         placement=cfg.placement if cfg.placement == "cpu" else f"{cfg.placement}/{cfg.ram_mode}",
                         remote=cfg.remote_share, chunk=cfg.prefill_chunk, decode=f"{tok_s:.1f}", prefill=f"{pf:.0f}",
                         tier_mib=f"{d.tier_mib:.0f}", tok_round=f"{d.tokens_per_round:.2f}", bottleneck=d.bottleneck, lossless=d.lossless,
                         kl=f"{d.kl_estimate:.3f}"))
    print(f"target {args.target_decode} tok/s decode on {hw.name} ({ram_gib:.0f} GiB RAM, {hw.gpu.vram_mib:.0f} MiB VRAM, "
          f"{'lossless only' if args.lossless else 'lossy allowed'}): best {len(candidates)} candidates")
    print(fmt_table(rows, ["pack", "spec", "affinity", "skip", "dense", "tail", "placement", "remote", "chunk", "decode", "prefill", "tier_mib", "tok_round", "bottleneck", "kl", "lossless"]))
    if candidates:
        best = candidates[0]
        d = best[3]
        print(f"\nbest: {best[0]:.1f} tok/s; bottleneck {d.bottleneck}; CPU {d.step.cpu_ms:.0f} ms vs GPU {d.step.gpu_ms:.0f} ms per round; "
              f"CPU bytes {d.cpu_gb_per_token:.2f} GB/token at {d.cpu_gbps:.0f} GB/s")
        if best[0] < args.target_decode:
            needed = args.target_decode / best[0]
            print(f"  {args.target_decode} tok/s needs the round {needed:.2f}x faster: e.g. CPU expert bytes/token "
                  f"{d.cpu_gb_per_token / needed:.2f} GB (more tier hits or a smaller pack) or DRAM {hw.memory.dram_gbps * needed:.0f} GB/s")
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")


def show_hw(args):
    hw = hardware_from_args(args)
    print(json.dumps(hw.to_dict(), indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("estimate", help="one configuration")
    add_hw_args(p); add_config_args(p); p.add_argument("--output")
    p.add_argument("--mc", type=int, default=0, help="Monte Carlo samples over uncertain parameters (prints p10/p50/p90)")
    p.set_defaults(func=estimate)

    p = sub.add_parser("sweep", help="grid over knobs; every config option accepts several values")
    add_hw_args(p); add_config_args(p, sweep=True); p.add_argument("--output")
    p.set_defaults(func=sweep)

    p = sub.add_parser("validate", help="predicted vs measured records")
    p.add_argument("--params"); p.add_argument("--records"); p.add_argument("--output")
    p.set_defaults(func=validate)

    p = sub.add_parser("fit", help="re-tune free parameters on fit:true records")
    p.add_argument("--params"); p.add_argument("--records"); p.add_argument("--output")
    p.add_argument("--names", nargs="*", help="parameters to tune (default: calibration.FIT_PARAMETERS)")
    p.add_argument("--rounds", type=int, default=3)
    p.set_defaults(func=fit)

    p = sub.add_parser("plan", help="search configurations for a decode target on given hardware")
    add_hw_args(p); add_config_args(p)
    p.add_argument("--target-decode", type=float, default=30.0)
    p.add_argument("--lossless", action="store_true", help="exclude affinity, REAP packs and non-exact RAM modes")
    p.add_argument("--pareto", action="store_true", help="list the configurations no other beats on decode, prefill and KL at once")
    p.add_argument("--kl-budget", type=float, default=None, help="drop configurations whose rough KL estimate exceeds this (q23 lossless: 0.137)")
    p.add_argument("--baseline", action="store_true", help="today's algorithms only (no LRU tier, route prefetch, adaptive window)")
    p.add_argument("--packs", nargs="*")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--output")
    p.set_defaults(func=plan)

    p = sub.add_parser("hw", help="print a hardware preset as JSON")
    add_hw_args(p)
    p.set_defaults(func=show_hw)

    args = ap.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
