#!/usr/bin/env python3
"""GSQ 2-bit on one matrix, layer-wise (quantization study, docs/GLM_QUANT_LEECH_GSQ_STUDY.md).

Uses IST-DASLab/GSQ's own GPTQ prior and Gumbel-Softmax 2-bit quantizer (checkout given by --gsq): GPTQ initialises
a symmetric 2-bit group-wise grid {-2,-1,0,1}*s (group 128), then 2000 Lion steps minimise ||(W - W_q) L||_F^2 with L
the Cholesky factor of the damped Hessian -- the layer-wise loop GSQ's prior runs (src/prior/gptq.py). H is the
identity, a diagonal (per-column E[x^2]) or a full matrix. Hyper-parameters are the repository's 2-bit defaults.

  gsq_layer.py ROWS COLS in.f32 out.f32 --gsq /path/to/GSQ [--hdiag diag.f32 | --hfull h.f32] [--device cuda]
"""
import argparse
import json
import sys
import time
from types import SimpleNamespace

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rows', type=int)
    ap.add_argument('cols', type=int)
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--gsq', required=True)
    ap.add_argument('--hdiag')
    ap.add_argument('--hfull')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--groupsize', type=int, default=128)
    ap.add_argument('--no-gsq', action='store_true', help='GPTQ initialisation only (the GSQ starting point)')
    a = ap.parse_args()
    sys.path.insert(0, a.gsq)
    import torch
    import torch.nn as nn
    from src.prior.gptq import GPTQ
    from src.prior.quant import Quantizer

    dev = torch.device(a.device)
    torch.manual_seed(20261016)
    w = torch.from_numpy(np.fromfile(a.src, np.float32).reshape(a.rows, a.cols)).to(dev)
    if a.hfull:
        h = torch.from_numpy(np.fromfile(a.hfull, np.float32).reshape(a.cols, a.cols)).to(dev)
    elif a.hdiag:
        h = torch.diag(torch.from_numpy(np.fromfile(a.hdiag, np.float32)).to(dev))
    else:
        h = torch.eye(a.cols, device=dev)
    layer = nn.Linear(a.cols, a.rows, bias=False).to(dev)
    layer.weight.data.copy_(w)
    config = SimpleNamespace(
        quantization=SimpleNamespace(gsq_bits=2, std=0.01, strength=6, temperature=[2, 0.05], scale=[100, 500]),
        training=SimpleNamespace(lr1=1e-4, lr2=5e-5, weight_decay=1.0, lion_betas=[0.9, 0.95]))
    # The name selects GSQ's layer-wise Gumbel loop in the prior (it runs it for q_proj/k_proj there).
    g = GPTQ(layer, 'study.gptq_only' if a.no_gsq else 'study.q_proj', config, dev, torch.float32)
    g.H = h.float().clone()
    g.nsamples = 1
    q = Quantizer()
    q.configure(2, perchannel=True, sym=True, mse=False)
    g.quantizer = q
    t0 = time.time()
    wq, scales = g.fasterquant(None, blocksize=128, percdamp=0.1, groupsize=a.groupsize)
    wq.detach().float().cpu().numpy().astype(np.float32).tofile(a.dst)
    bpw = 2.0 + 16.0 / a.groupsize
    print(json.dumps(dict(method='gptq-2bit' if a.no_gsq else 'gsq-2bit', bpw=bpw, seconds=round(time.time() - t0, 1),
                          hessian='full' if a.hfull else 'diag' if a.hdiag else 'identity')))


if __name__ == '__main__':
    main()
