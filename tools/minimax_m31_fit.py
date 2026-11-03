#!/usr/bin/env python3
"""MiniMax M3.1 Flash: the step-time model fitted to the measured strata-m3 path, and projections for batched decode and prefill.

The model (tools/sim/minimax_step.py) takes its inputs from tools/sim/data/minimax_m31_measured.json (dense matvec table, step profile, routing
union law, cache hit rates, the six single-stream decode speeds); only the CPU-lane constants, a host term, the effective stream job time and
three prefill constants are fitted.  Rows that the engine cannot run today (batched decode, the tensor-core dense path, a second GPU in prefill)
are projections.

    python3 tools/minimax_m31_fit.py                    # the tables in docs/MINIMAX_M31_SIM.md
    python3 tools/minimax_m31_fit.py --json out.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "sim"))
import minimax_step as ms  # noqa: E402

MODES = ("exact", "fast", "max")


def best_window(mode, gpus, B, p, ctx, tensor_core):
    """The window length per sequence (rows) that maximizes the aggregate tok/s, among the measured 1 .. 4.1 rows (not extrapolated)."""
    best = None
    for T in (1.0, 2.0, 3.0, 4.1):
        tok, step = ms.decode_tok_s(mode, gpus, B, T, p, ctx=ctx, tensor_core=tensor_core)
        if best is None or tok > best[0]:
            best = (tok, T, step)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args(argv)
    out = {}

    p, err = ms.fit(ms.Params())
    out["fitted"] = {k: getattr(p, k) for k in ("host_pre_us", "cpu_job_us", "cpu_layer_us", "cpu_token_us", "job_us_stream")}
    print("Fitted constants (grid search on six measured speeds + the one-GPU max-mode post-router time):")
    print("  ", out["fitted"], " squared relative error %.4f" % err)

    print("\nSingle-stream decode, measured vs model (verify window 4.1 rows, 3.8 tokens a step; mean of the dijkstra / lru_cache runs):")
    print("  GPUs  mode    measured  model   error   model step")
    rows = []
    for gpus, mode, meas, tok, step in ms.check_anchors(p):
        rows.append(dict(gpus=gpus, mode=mode, measured=meas, model=tok, step_ms=step))
        print(f"  {gpus}     {mode:6s}  {meas:7.1f}  {tok:6.1f}  {(tok / meas - 1) * 100:+5.1f} %  {step:6.1f} ms")
    out["anchors"] = rows
    _, d = ms.step_ms("max", 1, 1, 4.1, p, detail=True)
    print(f"  one-GPU max-mode step by part (model): chain {ms.LAYERS * d['chain_us'] / 1000:.1f} ms, post-router {ms.LAYERS * d['post_us'] / 1000:.1f} ms, "
          f"lm_head {d['head_ms']:.1f} ms, draft {d['draft_ms']:.1f} ms, host {p.host_ms:.1f} ms")

    # --- batched decode: B sequences, each with a speculative window; aggregate and per-sequence tok/s
    ctx = 2048
    print(f"\nBatched decode, PROJECTION (context {ctx} a sequence; the engine runs one sequence of at most 8 rows today). Aggregate tok/s "
          "(per-sequence tok/s) at the best window length (rows/seq):")
    batch = []
    for tensor_core, label in ((False, "dense matvecs as today (8 columns a launch)"), (True, "+ tensor-core dense path above 8 columns")):
        for gpus in (1, 2):
            print(f"\n  {label}, {gpus} GPU{'s' if gpus == 2 else ''}")
            print("    B    " + "".join(f"{m:>26s}" for m in MODES))
            for B in (1, 2, 4, 8, 16, 32):
                cells = []
                for mode in MODES:
                    tok, T, step = best_window(mode, gpus, B, p, ctx, tensor_core)
                    _, d = ms.step_ms(mode, gpus, B, T, p, ctx=ctx, tensor_core=tensor_core, detail=True)
                    cells.append(f"{tok:7.0f} ({tok / B:5.1f}) T={T:3.1f} h={d['hit'] * 100:2.0f}%")
                    batch.append(dict(tensor_core=tensor_core, gpus=gpus, B=B, mode=mode, tok_s=tok, per_seq=tok / B, rows_per_seq=T, step_ms=step,
                                      hit=d["hit"], slots=d["slots"], jobs_cpu=d["jobs_cpu"], jobs_gpu=d["jobs_gpu"], cpu_us=d["cpu_us"],
                                      chain_us=d["chain_us"], post_us=d["post_us"], draft_ms=d["draft_ms"], head_ms=d["head_ms"]))
                print(f"    {B:<3d}  " + "".join(f"{c:>26s}" for c in cells))
    out["batched_decode"] = batch

    # --- what limits a batch: the step parts at B = 8 (max mode, tensor-core path, one and two GPUs)
    print("\nWhat a batch step is made of (max mode, tensor-core dense path, ms): B  rows  chain  post-router  head  draft  | CPU jobs/layer  hit")
    for gpus in (1, 2):
        for B in (1, 4, 8, 16):
            tok, T, step = best_window("max", gpus, B, p, ctx, True)
            _, d = ms.step_ms("max", gpus, B, T, p, ctx=ctx, tensor_core=True, detail=True)
            print(f"  {gpus} GPU B={B:<2d} {B * T:5.1f}  {ms.LAYERS * d['chain_us'] / 1000:5.1f}  {ms.LAYERS * d['post_us'] / 1000:5.1f}  {d['head_ms']:4.1f}  "
                  f"{d['draft_ms']:5.1f}  | {d['jobs_cpu']:4.1f}  {d['hit'] * 100:3.0f} %")

    # --- prefill
    pp, perr = ms.fit_prefill(ms.PrefillParams())
    out["prefill_fit"] = {k: getattr(pp, k) for k in ("warm_s", "touch_tokens", "overlap", "compute_s_per_ktok", "resident_share", "gpu_scaling_2")}
    print("\nPrefill: measured points vs the fitted model (one V100):")
    pre = []
    for n, c, meas, mod in ms.prefill_anchor_errors(pp):
        print(f"  {n:5d} tokens  measured {meas:5.1f} s ({n / meas:5.0f} tok/s)  model {mod:5.1f} s ({n / mod:5.0f} tok/s)")
    print("\nPrefill, PROJECTION (tok/s; prompt tokens / seconds incl. the fixed warm-up):")
    plen = 14336
    print(f"  one long prompt of {plen} tokens, by chunk size (a 14336 chunk needs about twice the activation VRAM of today's 7168):")
    print("                                                       " + "".join(f"{c:>9d}" for c in (2048, 4096, 7168, 14336)))
    for gpus in (1, 2):
        for ov, label in ((pp.overlap, "overlap as fitted"), (0.98, "stream fully hidden (deep prefetch)")):
            row = [ms.prefill_s(plen, c, gpus, pp, overlap=ov) for c in (2048, 4096, 7168, 14336)]
            print(f"    {gpus} GPU, {label:38s} " + "".join(f"{plen / t:9.0f}" for t in row))
            pre.append(dict(prompt=plen, gpus=gpus, overlap=ov, chunks=[2048, 4096, 7168, 14336], tok_s=[plen / t for t in row]))
    print("  many prompts batched into one chunk (the pack streams once for all): total tokens / seconds")
    print("    prompts x tokens        1 GPU        2 GPUs")
    for nprompts, ln in ((1, 210), (8, 210), (32, 210), (64, 210), (8, 1000), (16, 1000)):
        n = nprompts * ln
        chunk = min(n, 7168)
        t1 = ms.prefill_s(n, chunk, 1, pp)
        t2 = ms.prefill_s(n, chunk, 2, pp)
        pre.append(dict(prompts=nprompts, length=ln, tokens=n, tok_s_1gpu=n / t1, tok_s_2gpu=n / t2))
        print(f"    {nprompts:3d} x {ln:5d} = {n:6d}   {n / t1:8.0f}    {n / t2:8.0f}")
    print("  long prompts (several chunks, the pack streams once per chunk):")
    for n in (14000, 28000, 57000):
        c = 7168
        pre.append(dict(prompt=n, chunk=c, tok_s_1gpu=n / ms.prefill_s(n, c, 1, pp), tok_s_2gpu=n / ms.prefill_s(n, c, 2, pp)))
        print(f"    {n:6d} tokens, chunk {c}: {n / ms.prefill_s(n, c, 1, pp):6.0f} (1 GPU)  {n / ms.prefill_s(n, c, 2, pp):6.0f} (2 GPUs)")
    out["prefill"] = pre
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
