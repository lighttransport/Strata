#!/usr/bin/env python3
"""KL divergence of Strata's logits against a llama.cpp `--kl-divergence-base` file (the Q8_0 reference of a model).

The base file (llama-perplexity --kl-divergence-base OUT, tools/perplexity.cpp): the magic "_logits_", int32 n_ctx,
n_vocab, n_chunk, every chunk's n_ctx token ids (int32, all chunks back to back), then per chunk one row per scored
position i in [n_ctx/2, n_ctx - 1): two floats (scale, min_log_prob) and nv = 2 * ((n_vocab + 1) / 2) uint16 codes,
log_prob = code * scale + min_log_prob (a code of 0 means min_log_prob, which is max - 16 or lower).

    ds41f_kl_base.py info  BASE
    ds41f_kl_base.py tokens BASE OUTDIR           one OUTDIR/chunk_NN.ids per chunk (space-separated ids): Strata prompts
    ds41f_kl_base.py kl BASE OURS [--chunks a,b]  OURS: a directory of chunk_NN.f32 (Strata --save-all-logits of each
                                                  chunk, float32 [n_ctx, n_vocab]) or one f32 file holding the chunks
                                                  back to back; prints mean / p99 KL(base || ours), top-1 agreement and
                                                  both perplexities over the scored positions, per chunk and overall
"""
import argparse
import pathlib
import struct
import sys

import numpy as np

MAGIC = b"_logits_"


class Base:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        with open(self.path, "rb") as f:
            if f.read(8) != MAGIC:
                raise SystemExit(f"{path}: not a kl-divergence-base file")
            self.n_ctx, self.n_vocab, self.n_chunk = struct.unpack("<3i", f.read(12))
        self.nv = 2 * ((self.n_vocab + 1) // 2)
        self.rows = self.n_ctx // 2 - 1                     # positions n_ctx/2 .. n_ctx-2 (each predicts the next token)
        self.row_bytes = (self.nv + 4) * 2
        self.tokens_at = 20
        self.rows_at = 20 + self.n_chunk * self.n_ctx * 4
        self.chunk_bytes = self.rows * self.row_bytes
        expect = self.rows_at + self.chunk_bytes * self.n_chunk
        size = self.path.stat().st_size
        if size != expect:
            raise SystemExit(f"{path}: {size} bytes, expected {expect} for n_ctx {self.n_ctx} n_vocab {self.n_vocab} n_chunk {self.n_chunk}")

    def tokens(self, chunk):
        with open(self.path, "rb") as f:
            f.seek(self.tokens_at + chunk * self.n_ctx * 4)
            return np.frombuffer(f.read(self.n_ctx * 4), np.int32).copy()

    def log_probs(self, chunk):
        """[rows, n_vocab] float32 log-probabilities of the scored positions."""
        with open(self.path, "rb") as f:
            f.seek(self.rows_at + chunk * self.chunk_bytes)
            raw = np.frombuffer(f.read(self.rows * self.row_bytes), np.uint8).reshape(self.rows, self.row_bytes)
        scale = raw[:, 0:4].copy().view(np.float32).reshape(-1, 1)
        mn = raw[:, 4:8].copy().view(np.float32).reshape(-1, 1)
        codes = raw[:, 8:].copy().view(np.uint16)[:, : self.n_vocab].astype(np.float32)
        return codes * scale + mn


def log_softmax(x):
    x = x.astype(np.float64)
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def cmd_info(a):
    b = Base(a.base)
    print(f"{b.path.name}: n_ctx {b.n_ctx} n_vocab {b.n_vocab} chunks {b.n_chunk} scored rows per chunk {b.rows}")
    lp = b.log_probs(0)
    s = np.exp(lp.astype(np.float64)).sum(1)
    print(f"chunk 0: tokens {b.tokens(0)[:6].tolist()} ...; sum of probabilities over rows: min {s.min():.4f} max {s.max():.4f}")
    return 0


def cmd_tokens(a):
    b = Base(a.base)
    out = pathlib.Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for c in range(b.n_chunk):
        (out / f"chunk_{c:02d}.ids").write_text(" ".join(str(int(t)) for t in b.tokens(c)) + "\n")
    print(f"{b.n_chunk} chunks of {b.n_ctx} tokens in {out}")
    return 0


def ours_logits(path, b, chunk):
    p = pathlib.Path(path)
    if p.is_dir():
        f = p / f"chunk_{chunk:02d}.f32"
        if not f.exists():
            return None
        x = np.fromfile(f, np.float32)
        if x.size % b.n_vocab:
            raise SystemExit(f"{f}: {x.size} floats is not a multiple of n_vocab")
        x = x.reshape(-1, b.n_vocab)
        if x.shape[0] < b.n_ctx - 1:
            raise SystemExit(f"{f}: {x.shape[0]} positions, the chunk has {b.n_ctx}")
        return x[: b.n_ctx]
    x = np.memmap(p, np.float32, "r")
    per = b.n_ctx * b.n_vocab
    if x.size < (chunk + 1) * per:
        return None
    return np.asarray(x[chunk * per : (chunk + 1) * per]).reshape(b.n_ctx, b.n_vocab)


def cmd_kl(a):
    b = Base(a.base)
    chunks = [int(c) for c in a.chunks.split(",")] if a.chunks else list(range(b.n_chunk))
    first = b.n_ctx // 2
    all_kl, all_top, nll_b, nll_o, n = [], [], 0.0, 0.0, 0
    for c in chunks:
        ours = ours_logits(a.ours, b, c)
        if ours is None:
            print(f"chunk {c:2d}: no logits")
            continue
        base = b.log_probs(c)                                  # rows: positions first .. first + rows - 1
        toks = b.tokens(c)
        lo = log_softmax(ours[first : first + b.rows])
        # llama.cpp's sum: only the base's tokens above log-probability -16 (the 16-bit codes' tail is not a distribution)
        b64 = base.astype(np.float64)
        kl = np.where(b64 > -16.0, np.exp(b64) * (b64 - lo), 0.0).sum(1)
        nxt = toks[first + 1 : first + 1 + b.rows]
        idx = np.arange(b.rows)
        top = base.argmax(1) == lo.argmax(1)
        ppl_b = np.exp(-base[idx, nxt].mean())
        ppl_o = np.exp(-lo[idx, nxt].mean())
        print(f"chunk {c:2d}: mean KL {kl.mean():.4f}  p99 {np.percentile(kl, 99):.3f}  top1 {100 * top.mean():.1f}%  "
              f"PPL base {ppl_b:.3f} ours {ppl_o:.3f}")
        all_kl.append(kl)
        all_top.append(top)
        nll_b += -base[idx, nxt].sum()
        nll_o += -lo[idx, nxt].sum()
        n += b.rows
    if not all_kl:
        return 1
    kl = np.concatenate(all_kl)
    top = np.concatenate(all_top)
    print(f"all ({len(all_kl)} chunks, {n} positions): mean KL {kl.mean():.4f}  p99 {np.percentile(kl, 99):.3f}  "
          f"top1 {100 * top.mean():.2f}%  PPL base {np.exp(nll_b / n):.3f} ours {np.exp(nll_o / n):.3f}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info"); p.add_argument("base"); p.set_defaults(func=cmd_info)
    p = sub.add_parser("tokens"); p.add_argument("base"); p.add_argument("outdir"); p.set_defaults(func=cmd_tokens)
    p = sub.add_parser("kl"); p.add_argument("base"); p.add_argument("ours"); p.add_argument("--chunks", default="")
    p.set_defaults(func=cmd_kl)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
