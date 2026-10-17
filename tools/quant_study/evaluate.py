#!/usr/bin/env python3
"""One-layer quantization study driver (docs/GLM_QUANT_LEECH_GSQ_STUDY.md).

For each selected expert projection of one layer it loads the original BF16 weights (ground truth), the EXL3 K2
reconstruction (strata-glm-exl3-dequant) and the current Q23 GGUF tensor, quantizes the BF16 and the EXL3 inputs with
every method, and records per method: bits per weight, relative weight error and activation-weighted error
(per-column E[x^2] of the expert's inputs from the engine's calibration moments) against the BF16 truth and against
the method's own input. Results are appended to results.jsonl (one line per input/expert/projection/method), so the
run can be resumed; `--table` prints the summary.

  python3 tools/quant_study/evaluate.py --root models/glm53f/quant-study --layer 20 --experts ... \
      [--methods q2_K,q3_K,...,llvq-diag-rot] [--inputs bf16,exl3] [--table]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

GGUF_PY = os.environ.get('STRATA_GGUF_PY', 'third_party/llama.cpp/gguf-py')
sys.path.insert(0, GGUF_PY)

PROJ = ('gate_proj', 'up_proj', 'down_proj')
GGML = ['q2_K', 'q3_K', 'q4_K', 'iq2_xs', 'iq2_s', 'iq3_xxs', 'iq3_s']
NEEDS_IMATRIX = {'iq2_xs', 'iq2_s', 'iq3_xxs', 'iq1_s', 'iq1_m', 'iq2_xxs'}
DEFAULT_METHODS = ['q2_K', 'q2_K-im', 'q3_K', 'q3_K-im', 'q4_K-im', 'iq2_xs-im', 'iq2_s-im', 'iq3_xxs-im', 'iq3_s-im',
                   'llvq-id-rot', 'llvq-diag-rot']


def read_safetensors(path):
    with open(path, 'rb') as f:
        n = int.from_bytes(f.read(8), 'little')
        head = json.loads(f.read(n))
        base = 8 + n
    out = {}
    for name, meta in head.items():
        if name == '__metadata__':
            continue
        s, e = meta['data_offsets']
        raw = np.memmap(path, dtype=np.uint8, mode='r', offset=base + s, shape=(e - s,))
        if meta['dtype'] == 'BF16':
            arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif meta['dtype'] == 'F32':
            arr = raw.view(np.float32)
        elif meta['dtype'] == 'F16':
            arr = raw.view(np.float16).astype(np.float32)
        else:
            raise ValueError(meta['dtype'])
        out[name] = np.asarray(arr).reshape(meta['shape'])
    return out


def q23_experts(gguf_path, layer, experts):
    """Current Q23 tensors (Q2_K gate/up, Q3_K down) of the selected experts, dequantized with gguf-py."""
    import gguf
    from gguf import quants
    reader = gguf.GGUFReader(gguf_path)
    names = {'gate_proj': f'blk.{layer}.ffn_gate_exps.weight', 'up_proj': f'blk.{layer}.ffn_up_exps.weight',
             'down_proj': f'blk.{layer}.ffn_down_exps.weight'}
    out = {}
    for proj, tname in names.items():
        t = next(x for x in reader.tensors if x.name == tname)
        n_expert = int(t.shape[-1]) if len(t.shape) == 3 else None
        data = np.asarray(t.data)                      # (experts, rows, row_bytes) for quantized types
        for e in experts:
            out[(e, proj)] = (quants.dequantize(data[e], t.tensor_type).astype(np.float32), t.tensor_type.name)
        assert n_expert == 288 or n_expert is None
    return out


def rel(a, b, weights=None):
    d = a - b
    if weights is None:
        return float((d * d).sum() / (b * b).sum())
    return float(((d * d) * weights[None, :]).sum() / ((b * b) * weights[None, :]).sum())


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f'{cmd[0]} failed: {p.stderr[-500:]}')
    return json.loads(p.stdout.strip().splitlines()[-1])


def oerr(wq, truth, h):
    """Output error on held-out tokens: tr(D H D^T) / tr(W H W^T), D = Wq - W."""
    d = (wq - truth).astype(np.float32)
    return float(((d @ h) * d).sum() / ((truth @ h) * truth).sum())


def quantize(method, w, imp, bins, tmp, hfit=None):
    rows, cols = w.shape
    src, dst, imf = tmp / 'w.f32', tmp / 'q.f32', tmp / 'imp.f32'
    w.astype(np.float32).tofile(src)
    imp.astype(np.float32).tofile(imf)
    if method.startswith('gsq-'):
        cmd = [os.environ.get('STUDY_TORCH_PY', 'python3'), str(Path(__file__).with_name('gsq_layer.py')), str(rows), str(cols),
               str(src), str(dst), '--gsq', os.environ.get('STUDY_GSQ_DIR', str(bins.parent / 'GSQ'))]
        if '-diag' in method:
            cmd += ['--hdiag', str(imf)]
        if '-full' in method:
            cmd += ['--hfull', str(hfit)]
    elif method.startswith('llvq-'):
        cmd = [str(bins / 'llvq_layer'), str(rows), str(cols), str(src), str(dst), '--threads', os.environ.get('STUDY_THREADS', '8')]
        if '-diag' in method:
            cmd += ['--hdiag', str(imf)]
        if '-full' in method:
            cmd += ['--hfull', str(hfit)]
        if method.endswith('-rot'):
            cmd += ['--rot', '20261016']
    else:
        typ = method.split('-')[0]
        cmd = [str(bins / 'ggml_quant'), typ, str(rows), str(cols), str(src), str(dst)]
        if method.endswith('-im') or typ in NEEDS_IMATRIX:
            cmd.append(str(imf))
    info = run(cmd)
    return np.fromfile(dst, dtype=np.float32).reshape(rows, cols), info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--layer', type=int, default=20)
    ap.add_argument('--experts', default='21,29,33,117,120,140,141,146,153,156,209,220,227,244,251,263')
    ap.add_argument('--hot', default='29,117,140,141,220,244,251,263')
    ap.add_argument('--methods', default=','.join(DEFAULT_METHODS))
    ap.add_argument('--inputs', default='bf16,exl3')
    ap.add_argument('--calibration', default='models/glm53f/prune/calibration')
    ap.add_argument('--gguf', default='models/glm53f/assembled/GLM-5.3-Flash-Q23-assembled.gguf')
    ap.add_argument('--table', action='store_true')
    ap.add_argument('--oerr', action='store_true', help='full-Hessian pass: held-out output error (hess/ from hessians.py)')
    a = ap.parse_args()
    results = a.root / (f'results-layer{a.layer}-oerr.jsonl' if a.oerr else f'results-layer{a.layer}.jsonl')
    if a.table:
        return table(results, set(int(x) for x in a.hot.split(',')))
    experts = [int(x) for x in a.experts.split(',')]
    bins = a.root / 'bin'
    bf16 = read_safetensors(a.root / f'bf16-layer{a.layer}.safetensors')
    exl3 = read_safetensors(a.root / f'exl3-layer{a.layer}.safetensors')
    moments = {'gu': np.fromfile(Path(a.calibration) / f'blk.{a.layer}.gu.f32', np.float32).reshape(288, 4096),
               'down': np.fromfile(Path(a.calibration) / f'blk.{a.layer}.down.f32', np.float32).reshape(288, 2048)}
    done = set()
    if results.exists():
        for line in results.open():
            r = json.loads(line)
            done.add((r['input'], r['expert'], r['proj'], r['method']))
    q23 = q23_experts(a.gguf, a.layer, experts)
    with tempfile.TemporaryDirectory() as td, results.open('a') as out:
        tmp = Path(td)
        for e in experts:
            for proj in PROJ:
                name = f'model.language_model.layers.{a.layer}.mlp.experts.{e}.{proj}.weight'
                truth = bf16[name].astype(np.float32)
                imp = moments['down' if proj == 'down_proj' else 'gu'][e]
                imp = np.maximum(imp, imp[imp > 0].min() if (imp > 0).any() else 1e-12)
                kind = 'down' if proj == 'down_proj' else 'gu'
                hfit = a.root / 'hess' / f'e{e}.{kind}.fit.f32'
                htest = None
                if a.oerr:
                    if not hfit.exists():
                        continue
                    n = 2048 if kind == 'down' else 4096
                    htest = np.fromfile(a.root / 'hess' / f'e{e}.{kind}.test.f32', np.float32).reshape(n, n)
                reference = {'q23-stored': q23[(e, proj)], 'exl3-k2-stored': (exl3[name], 'EXL3_K2')}
                for label, (wq, typ) in reference.items():
                    key = ('stored', e, proj, label)
                    if key in done:
                        continue
                    bpw = {'Q2_K': 2.625, 'Q3_K': 3.4375, 'EXL3_K2': 2.0 + 16 * 2 / 4096}.get(typ)
                    row = dict(input='stored', expert=e, proj=proj, method=label, bpw=bpw,
                               err_truth=rel(wq, truth), werr_truth=rel(wq, truth, imp))
                    if htest is not None:
                        row['oerr_truth'] = oerr(wq, truth, htest)
                    out.write(json.dumps(row) + '\n')
                    out.flush()
                inputs = {'bf16': truth, 'exl3': exl3[name].astype(np.float32)}
                for inp in a.inputs.split(','):
                    w = inputs[inp]
                    for method in a.methods.split(','):
                        key = (inp, e, proj, method)
                        if key in done:
                            continue
                        wq, info = quantize(method, w, imp, bins, tmp, hfit)
                        row = dict(input=inp, expert=e, proj=proj, method=method, bpw=info['bpw'],
                                   err_truth=rel(wq, truth), werr_truth=rel(wq, truth, imp),
                                   err_input=rel(wq, w), werr_input=rel(wq, w, imp), seconds=info.get('seconds'))
                        if htest is not None:
                            row['oerr_truth'] = oerr(wq, truth, htest)
                        out.write(json.dumps(row) + '\n')
                        out.flush()
                        print(f'{inp} e{e} {proj} {method}: bpw {row["bpw"]:.3f} werr_truth {row["werr_truth"]:.4f} '
                              f'werr_input {row["werr_input"]:.4f}', flush=True)


def table(path, hot):
    rows = [json.loads(l) for l in path.open()]
    groups = {}
    for r in rows:
        k = (r['input'], r['method'])
        groups.setdefault(k, []).append(r)
    print(f"{'input':<7}{'method':<18}{'bpw':>7}{'err vs BF16':>13}{'weighted':>10}{'hot':>9}{'median':>9}{'vs input':>10}{'output':>9}{'n':>4}")
    for (inp, m), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], np.mean([r['bpw'] or 0 for r in kv[1]]))):
        mean = lambda key, sel=None: np.mean([r[key] for r in rs if r.get(key) is not None and (sel is None or sel(r))])  # noqa: E731
        print(f"{inp:<7}{m:<18}{mean('bpw'):>7.3f}{mean('err_truth'):>13.4f}{mean('werr_truth'):>10.4f}"
              f"{mean('werr_truth', lambda r: r['expert'] in hot):>9.4f}{mean('werr_truth', lambda r: r['expert'] not in hot):>9.4f}"
              f"{(mean('werr_input') if any('werr_input' in r for r in rs) else float('nan')):>10.4f}"
              f"{(mean('oerr_truth') if any('oerr_truth' in r for r in rs) else float('nan')):>9.4f}{len(rs):>4}")


if __name__ == '__main__':
    main()
