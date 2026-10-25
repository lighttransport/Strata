#!/usr/bin/env python3
"""Byte census of the GLM-5.3-Flash FP4-Local checkpoint for tools/sim/a64fx.py, from safetensors headers only.

    python3 tools/glm_fp4_local_census.py MODEL_DIR --out tools/sim/data/glm_fp4_local_trunk.json

Every trunk matrix is recorded once per layer kind (dense 0-2, KDA MoE, MLA MoE) with its stored format, shape, bytes
(weight plus scales) and the split a tensor-parallel runner gives it: `heads` (64 attention heads), `rows` (16-row
tiles of the output), `cols` (16-column tiles of the input, partial sums reduced) or `replicated`. Layers of a kind
must agree tensor by tensor, else the tool refuses. The matrix totals are checked against the census of the V100
FP4 work (tools/sim/data/glm_fp4_v100.json, `main_fixed` and `head`).
"""
import argparse
import hashlib
import json
import pathlib
import re
import struct

SCHEMA = 'strata.glm.fp4_local_trunk.v1'
PREFIX = 'model.language_model.layers.'
DTYPE_BYTES = dict(BF16=2, F16=2, F32=4, F8_E4M3=1, U8=1, U16=2, U32=4)

# sublayer -> (tensor suffix, split). Matrices outside this table are per-token extras (norms, conv, biases).
KDA = dict(q_proj=('self_attn.q_proj', 'heads'), k_proj=('self_attn.k_proj', 'heads'), v_proj=('self_attn.v_proj', 'heads'),
           o_proj=('self_attn.o_proj', 'heads'), f_a_proj=('self_attn.f_a_proj', 'replicated'),
           g_a_proj=('self_attn.g_a_proj', 'replicated'), f_b_proj=('self_attn.f_b_proj', 'heads'),
           g_b_proj=('self_attn.g_b_proj', 'heads'), b_proj=('self_attn.b_proj', 'heads'))
MLA = dict(q_a_proj=('self_attn.q_a_proj', 'replicated'), kv_a_proj=('self_attn.kv_a_proj_with_mqa', 'replicated'),
           q_b_proj=('self_attn.q_b_proj', 'heads'), kv_b_proj=('self_attn.kv_b_proj', 'heads'),
           o_proj=('self_attn.o_proj', 'heads'), indexer_wq_b=('self_attn.indexer.wq_b', 'replicated'),
           indexer_wk=('self_attn.indexer.wk', 'replicated'),
           indexer_weights_proj=('self_attn.indexer.weights_proj', 'replicated'))
MOE = dict(router=('mlp.gate', 'replicated'), shared_gate=('mlp.shared_experts.gate_proj', 'rows'),
           shared_up=('mlp.shared_experts.up_proj', 'rows'), shared_down=('mlp.shared_experts.down_proj', 'cols'))
DENSE_FFN = dict(ffn_gate=('mlp.gate_proj', 'rows'), ffn_up=('mlp.up_proj', 'rows'), ffn_down=('mlp.down_proj', 'cols'))
MHC = dict(hc_attn_fn=('hc_attn_fn', 'replicated'), hc_ffn_fn=('hc_ffn_fn', 'replicated'))
KINDS = dict(dense={**KDA, **DENSE_FFN, **MHC}, kda={**KDA, **MOE, **MHC}, mla={**MLA, **MOE, **MHC})


def headers(model):
    index = model / 'model.safetensors.index.json'
    files = sorted(set(json.loads(index.read_text())['weight_map'].values()))
    out = {}
    for name in files:
        with (model / name).open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(size))
        header.pop('__metadata__', None)
        for key, value in header.items():
            start, end = value['data_offsets']
            out[key] = dict(dtype=value['dtype'], shape=value['shape'], bytes=end - start, file=name)
    return out, hashlib.sha256(index.read_bytes()).hexdigest()


def matrix(hdr, base):
    """Weight (+ scale) of one matrix; format from the stored dtypes."""
    w = hdr.get(base + '.weight') or hdr.get(base)
    if w is None:
        raise ValueError('missing tensor ' + base)
    scale = hdr.get(base + '.weight_scale')
    if w['dtype'] == 'BF16':
        fmt = 'BF16'
    elif w['dtype'] == 'F8_E4M3' and scale and scale['dtype'] == 'U8':
        fmt = 'MXFP8'
    elif w['dtype'] == 'F8_E4M3' and scale and scale['dtype'] == 'F32':
        fmt = 'block_FP8'
    else:
        raise ValueError(f'unsupported storage for {base}: {w["dtype"]}')
    shape = w['shape'] if len(w['shape']) == 2 else [w['shape'][0], int(w['bytes'] // DTYPE_BYTES[w['dtype']] // w['shape'][0])]
    nbytes = w['bytes'] + (scale['bytes'] if scale else 0)
    return dict(format=fmt, rows=shape[0], cols=shape[1], bytes=nbytes)


def census(model):
    hdr, index_sha = headers(model)
    config = json.loads((model / 'config.json').read_text())
    text = config.get('text_config', config)
    layers = text['num_hidden_layers']
    dense = text['first_k_dense_replace']
    mla = [i for i, t in enumerate(text['layer_types']) if t == 'deepseek_sparse_attention']
    kinds = {i: 'dense' if i < dense else 'mla' if i in mla else 'kda' for i in range(layers)}
    sub, extras, used = {}, {}, set()
    for kind in KINDS:
        ids = [i for i in range(layers) if kinds[i] == kind]
        rows = None
        for i in ids:
            row = {}
            for name, (suffix, split) in KINDS[kind].items():
                base = f'{PREFIX}{i}.{suffix}'
                row[name] = dict(matrix(hdr, base), split=split)
                used.update(k for k in (base, base + '.weight', base + '.weight_scale') if k in hdr)
            if rows is not None and row != rows:
                raise ValueError(f'layer {i} differs from the other {kind} layers')
            rows = row
        extra = {}
        for i in ids:
            for key, value in hdr.items():
                if key.startswith(f'{PREFIX}{i}.') and '.experts.' not in key and key not in used:
                    short = key[len(f'{PREFIX}{i}.'):]
                    extra[short] = extra.get(short, 0) + value['bytes']
        sub[kind] = dict(layers=ids, sublayers=rows)
        extras[kind] = {k: v // len(ids) for k, v in sorted(extra.items())}
    head = dict(matrix(hdr, 'lm_head'), split='rows')
    embed = dict(matrix(hdr, 'model.language_model.embed_tokens'), split='replicated')
    expert = {}
    for scope, ids in (('main', range(dense, layers)), ('draft', range(layers, layers + text['num_nextn_predict_layers']))):
        codes = csf = plain = scale2 = 0
        for key, value in hdr.items():
            m = re.match(r'model\.language_model\.layers\.(\d+)\.mlp\.experts\.\d+\.', key)
            if not m or int(m.group(1)) not in ids:
                continue
            if key.endswith('.weight'):
                codes += value['bytes']
            elif '.weight_scale.csf32_' in key:
                csf += value['bytes']
            elif key.endswith('.weight_scale'):
                plain += value['bytes']
            elif key.endswith('.weight_scale_2'):
                scale2 += value['bytes']
        experts = text['n_routed_experts'] * len(ids)
        per = 3 * text['hidden_size'] * text['moe_intermediate_size']
        expert[scope] = dict(experts=experts, code_bytes=codes, csf_bytes=csf, plain_scale_bytes=plain,
                             global_scale_bytes=scale2, checkpoint_bytes=codes + csf + plain + scale2,
                             expanded_bytes=experts * per * 9 // 16,
                             expanded_bytes_per_expert=per * 9 // 16)
    draft_fixed = sum(v['bytes'] for k, v in hdr.items() if k.startswith(f'{PREFIX}{layers}.') and '.experts.' not in k)
    main_fixed = sum(sub[k]['sublayers'][s]['bytes'] * len(sub[k]['layers']) for k in sub for s in sub[k]['sublayers'])
    return dict(schema=SCHEMA, model='nvidia/GLM-5.3-Flash-FP4-Local', index_sha256=index_sha,
                geometry=dict(layers=layers, hidden=text['hidden_size'], heads=text['num_attention_heads'],
                              experts=text['n_routed_experts'], top_k=text['num_experts_per_tok'],
                              expert_ff=text['moe_intermediate_size'], dense_ff=text['intermediate_size'],
                              vocab=text['vocab_size'], mla_layers=mla, dense_layers=dense),
                kinds=sub, extras_per_layer=extras, head=head, embedding=embed, experts=expert,
                draft_fixed_bytes=draft_fixed, main_fixed_matrix_bytes=main_fixed)


def check(result, reference):
    ref = json.loads(reference.read_text())['census']
    pairs = [('main_fixed_matrix_bytes', result['main_fixed_matrix_bytes'], ref['main_fixed']['checkpoint_bytes']),
             ('head', result['head']['bytes'], ref['head']['checkpoint_bytes']),
             ('main experts expanded', result['experts']['main']['expanded_bytes'], ref['main_experts']['gpu_bytes']),
             ('main experts stored', result['experts']['main']['checkpoint_bytes'], ref['main_experts']['checkpoint_bytes'])]
    for name, got, want in pairs:
        if got != want:
            raise SystemExit(f'{name}: {got} != census {want}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model', type=pathlib.Path)
    p.add_argument('--out', type=pathlib.Path)
    p.add_argument('--reference', type=pathlib.Path,
                   default=pathlib.Path(__file__).resolve().parent / 'sim/data/glm_fp4_v100.json')
    args = p.parse_args()
    result = census(args.model)
    check(result, args.reference)
    text = json.dumps(result, indent=1) + '\n'
    if args.out:
        args.out.write_text(text)
    else:
        print(text, end='')
