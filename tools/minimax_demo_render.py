#!/usr/bin/env python3
"""Render strata-m3 decode runs side by side as an animated GIF, at the speed they were measured (requires Pillow).

Input: the REPL output of each run, recorded with STRATA_M3_STREAM=1. Every `M3_STREAM <ms> <ids...>` line is one decode step: the
tokens it added and the milliseconds since the decode started. A panel shows the text decoded up to each frame's time and a live rate,
tokens shown / seconds since the decode started. After a run ends, its panel shows the final rate from the run's `M3_STATS` line. Each
panel starts at its own decode start, so the prompt processing before it is not shown.

    python3 tools/minimax_demo_render.py --tokenizer MiniMax-M3.1-Flash-UD-Q2_K_XL-00001-of-00004.gguf \\
        --run 'exact routing=exact.txt' --run '--speed max=max.txt' --title '...' out.gif
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from strata_tokenizer import Tokenizer  # noqa: E402


def load_run(path: pathlib.Path):
    text = path.read_text(encoding="utf-8")
    steps = []
    for m in re.finditer(r"^M3_STREAM ([0-9.]+)((?: -?\d+)*)$", text, re.M):
        steps.append((float(m.group(1)) / 1000.0, [int(x) for x in m.group(2).split()]))
    if not steps:
        raise SystemExit(f"{path}: no M3_STREAM lines (record with STRATA_M3_STREAM=1)")
    final = re.search(r"decode (\d+) tokens ([0-9.]+) ms \(([0-9.]+) tok/s\)", text)
    return steps, (int(final.group(1)), float(final.group(2)) / 1000.0, float(final.group(3))) if final else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokenizer", type=pathlib.Path, required=True, help="the model GGUF (first shard) holding the tokenizer")
    ap.add_argument("--run", action="append", required=True, help="LABEL=PATH of a REPL output; two or more side by side")
    ap.add_argument("--title", default="strata-m3 live decode / original speed")
    ap.add_argument("--caption", default="")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--hold", type=float, default=3.0, help="seconds to show the finished state")
    ap.add_argument("--cols", type=int, default=58)
    ap.add_argument("--rows", type=int, default=26)
    ap.add_argument("--font", default="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    ap.add_argument("--size", type=int, default=14)
    ap.add_argument("output", type=pathlib.Path)
    args = ap.parse_args(argv)

    tk = Tokenizer.from_gguf(args.tokenizer)
    runs = []
    for spec in args.run:
        label, _, path = spec.partition("=")
        steps, final = load_run(pathlib.Path(path))
        runs.append(dict(label=label, steps=steps, final=final, cache={}))
    eos_like = re.compile(r"\[e~\[|</s>|<\|im_end\|>")

    def text_at(run, n):
        if n not in run["cache"]:
            ids = [t for _, ts in run["steps"] for t in ts][:n]
            run["cache"][n] = eos_like.sub("", tk.decode(ids))
        return run["cache"][n]

    font = ImageFont.truetype(args.font, args.size)
    bold = font
    cw, lh = font.getlength("M"), int(args.size * 1.45)
    pad, gap, top = 14, 18, 2 * lh + 10
    pw = int(args.cols * cw)
    width = pad * 2 + pw * len(runs) + gap * (len(runs) - 1)
    height = top + (args.rows + 3) * lh + pad + (lh if args.caption else 0)
    end_t = max(r["steps"][-1][0] for r in runs) + args.hold
    frames = []
    for f in range(int(end_t * args.fps) + 1):
        t = f / args.fps
        img = Image.new("RGB", (width, height), "#0d1117")
        d = ImageDraw.Draw(img)
        d.text((pad, 8), args.title, font=bold, fill="#76c7c0")
        for i, run in enumerate(runs):
            x0 = pad + i * (pw + gap)
            n = sum(len(ts) for ts0, ts in run["steps"] if ts0 <= t)
            done = t >= run["steps"][-1][0]
            d.text((x0, top - lh), run["label"], font=bold, fill="#f0c674")
            if done and run["final"]:
                rate = f"FINAL decode {run['final'][2]:5.1f} tok/s   {run['final'][0]} tokens in {run['final'][1]:.1f} s"
            else:
                rate = f"LIVE  decode {n / t if t > 0 else 0.0:5.1f} tok/s   {n} tokens   {t:4.1f} s"
            d.rectangle((x0 - 4, top - 2, x0 + pw + 4, top + lh), fill="#16312a")
            d.text((x0, top), rate, font=font, fill="#b9f0cd")
            lines = []
            for para in text_at(run, n).split("\n"):
                lines += textwrap.wrap(para, args.cols, replace_whitespace=False, drop_whitespace=False) or [""]
            for r, line in enumerate(lines[-args.rows:]):
                d.text((x0, top + (r + 2) * lh), line, font=font, fill="#e1e6ee")
        if args.caption:
            d.text((pad, height - lh - 6), args.caption, font=font, fill="#8b949e")
        frames.append(img)
    palette = frames[-1].quantize(colors=32)
    indexed = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in frames]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    indexed[0].save(args.output, save_all=True, append_images=indexed[1:], duration=round(1000 / args.fps), loop=0, optimize=True, disposal=1)
    print(f"{args.output}: {len(frames)} frames, {len(frames) / args.fps:.1f} s, {args.output.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
