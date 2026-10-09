"""Export exact model tensor geometry for tools/sim, without reading tensor payloads."""
import argparse
import json
from pathlib import Path
from gguf_reader import GGUFFile


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
    args=p.parse_args();print(json.dumps(census(args.model,args.name),indent=2))
