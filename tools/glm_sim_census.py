"""Export exact model tensor geometry for tools/sim, without reading tensor payloads."""
import argparse
import json
import math
from pathlib import Path
from gguf_reader import GGUFFile


def deployed_fixed_census(path, floating_q8='none'):
    """Streamed main-model bytes after the runner's opt-in floating Q8 repack.

    Embeddings are looked up on the CPU; routed experts and layer 45 belong to
    separate terms. Norms and small coefficients remain in their stored format.
    """
    if floating_q8 not in ('none','mla','all'):
        raise ValueError('unsupported floating Q8 mode')
    result = {str(i):0 for i in range(45)}
    result.update({'shared:'+str(i):0 for i in range(3,45)})
    result['lm_head']=0
    for t in GGUFFile(path).tensors:
        if t.name == 'output.weight':
            result['lm_head']=t.expected_bytes();continue
        if not t.name.startswith('blk.') or '_exps.weight' in t.name:
            continue
        layer=int(t.name.split('.')[1])
        if layer >= 45:continue
        size=t.expected_bytes()
        if size is None:raise ValueError('unknown storage: '+t.name)
        mla=t.name.endswith(('attn_k_b.weight','attn_v_b.weight'))
        hc=t.name.endswith(('hc_attn_fn.weight','hc_ffn_fn.weight','hc_attn_fn','hc_ffn_fn'))
        if t.shape[0] % 32 == 0 and ((mla and floating_q8 != 'none') or (hc and floating_q8 == 'all')):
            size=math.prod(t.shape)//32*34
        result[str(layer)]+=size
        if t.name.endswith(('ffn_gate_shexp.weight','ffn_up_shexp.weight','ffn_down_shexp.weight')):
            result['shared:'+str(layer)]+=size
    if not result['lm_head'] or not all(result[str(i)] for i in range(45)):
        raise ValueError('incomplete main-model fixed census')
    return result


def census(path, name):
    model=GGUFFile(path)
    layers={}
    totals=dict(main_routed=0,main_fixed=0,draft=0)
    for t in model.tensors:
        parts=t.name.split('.')
        layer=int(parts[1]) if parts[0]=='blk' else None
        size=t.expected_bytes()
        if size is None:
            raise ValueError('unknown tensor storage '+t.name)
        routed='_exps.weight' in t.name
        category='draft' if layer is not None and layer>=45 else 'main_routed' if routed else 'main_fixed'
        totals[category]+=size
        if category=='main_routed':
            row=layers.setdefault(str(layer),{})
            row[parts[2].removeprefix('ffn_').removesuffix('_exps')]=dict(format=t.type_name,shape=t.shape,bytes=size)
    experts={t['shape'][-1] for row in layers.values() for t in row.values()}
    if len(experts)!=1 or set(layers)!=set(map(str,range(3,45))):
        raise ValueError('unsupported/incomplete expert geometry')
    return dict(name=name,source=str(path),file_bytes=Path(path).stat().st_size,experts=experts.pop(),layers=layers,bytes=totals)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);p.add_argument('--name',required=True)
    p.add_argument('--fixed',action='store_true');p.add_argument('--floating-q8',choices=['none','mla','all'],default='none')
    args=p.parse_args();print(json.dumps(deployed_fixed_census(args.model,args.floating_q8) if args.fixed else census(args.model,args.name),indent=2))
