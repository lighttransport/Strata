#!/usr/bin/env python3
"""Per-expert Hessians from an expert activation trace (quantization study, docs/GLM_QUANT_LEECH_GSQ_STUDY.md).

The trace (STRATA_GLM_ACTIVATION_TRACE, records: u32 magic 0x31435247, layer, position, tokens, width, top_k; then
int32 ids [tokens*top_k]; float32 MoE inputs [tokens*width]) gives the inputs x of every routed expert. For each
selected expert this writes, split 80/20 by token order:
  hess/e{E}.gu.fit.f32 / .gu.test.f32     H = X^T X / N over x (gate/up input, 4096 x 4096)
  hess/e{E}.down.fit.f32 / .down.test.f32 H over h = silu(min(x Wg^T, L)) * clamp(x Wu^T, -L, L) (2048 x 2048),
computed with the original BF16 gate/up weights (L = swiglu_limit = 10), plus counts.json.

  python3 tools/quant_study/hessians.py --trace trace-layer20.bin --layer 20 --root /mnt/nvme01/models/glm53f/quant-study
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import read_safetensors  # noqa: E402

MAGIC = 0x31435247


def records(path, layer):
    with open(path, 'rb') as f:
        while True:
            head = f.read(24)
            if len(head) < 24:
                return
            magic, lay, pos, nt, n, topk = np.frombuffer(head, dtype=np.uint32)
            if magic != MAGIC:
                raise ValueError(f'bad trace record magic {magic:#x}')
            ids = np.frombuffer(f.read(int(nt * topk) * 4), dtype=np.int32).reshape(nt, topk)
            x = np.frombuffer(f.read(int(nt * n) * 4), dtype=np.float32).reshape(nt, n)
            if lay == layer:
                yield int(pos), ids, x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trace', type=Path, required=True)
    ap.add_argument('--layer', type=int, default=20)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--experts', default='21,29,33,117,120,140,141,146,153,156,209,220,227,244,251,263')
    ap.add_argument('--limit', type=float, default=10.0)
    a = ap.parse_args()
    experts = [int(e) for e in a.experts.split(',')]
    rows = {e: [] for e in experts}
    for _, ids, x in records(a.trace, a.layer):
        for e in experts:
            hit = (ids == e).any(axis=1)
            if hit.any():
                rows[e].append(x[hit])
    bf = read_safetensors(a.root / f'bf16-layer{a.layer}.safetensors')
    out = a.root / 'hess'
    out.mkdir(exist_ok=True)
    counts = {}
    for e in experts:
        x = np.concatenate(rows[e]).astype(np.float64) if rows[e] else np.zeros((0, 4096))
        n = len(x)
        counts[e] = n
        if n < 10:
            print(f'expert {e}: only {n} tokens, skipped')
            continue
        cut = int(n * 0.8)
        p = f'model.language_model.layers.{a.layer}.mlp.experts.{e}.'
        wg = bf[p + 'gate_proj.weight'].astype(np.float64)
        wu = bf[p + 'up_proj.weight'].astype(np.float64)
        g = np.minimum(x @ wg.T, a.limit)
        u = np.clip(x @ wu.T, -a.limit, a.limit)
        h = g / (1 + np.exp(-g)) * u
        for tag, sl in (('fit', slice(0, cut)), ('test', slice(cut, n))):
            for name, v in (('gu', x[sl]), ('down', h[sl])):
                (v.T @ v / max(1, len(v))).astype(np.float32).tofile(out / f'e{e}.{name}.{tag}.f32')
        print(f'expert {e}: {n} tokens ({cut} fit / {n - cut} test)', flush=True)
    (out / 'counts.json').write_text(json.dumps(counts, indent=1) + '\n')


if __name__ == '__main__':
    main()
