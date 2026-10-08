"""Replay recorded unbiased decode routes through exclusive expert caches.

GPU and pinned CPU sets use only the training prefix. Evaluate the remaining
trace in original order; no future frequency or optimal-oracle admission.
Misses fetch exact experts. Transfer time and predictions are modeled separately.
"""
import argparse
import collections
import json
import math
from pathlib import Path

GIB=2**30

def select(ranking,sizes,capacity,exclude=()):
    used=0; chosen=set();exclude=set(exclude)
    for key in ranking:
        if key not in exclude and used+sizes[key]<=capacity:
            chosen.add(key);used+=sizes[key]
    return chosen,used


def replay(train,test,sizes,cpu_bytes,gpu_bytes,policy):
    counts=collections.Counter(k for row in train for k in row[2])
    ranking=sorted(sizes,key=lambda k:(-counts[k]/sizes[k],k))  # unseen experts tie-break by ID; no future data
    gpu,gb=select(ranking,sizes,gpu_bytes)
    pinned,pb=select(ranking,sizes,cpu_bytes*(1 if policy=='static' else .8 if policy=='hybrid' else 0),gpu)
    capacity=cpu_bytes-pb;lru=collections.OrderedDict();used=0
    counters=collections.Counter();per_token=collections.Counter();miss_keys=[]
    def access(key,measuring,pos):
        nonlocal used
        if key in gpu:kind='gpu_hit'
        elif key in pinned:kind='cpu_hit'
        elif key in lru:kind='cpu_hit';lru.move_to_end(key)
        else:
            kind='ssd_miss'
            if measuring:per_token[pos]+=sizes[key];miss_keys.append(key)
            if policy!='static' and sizes[key]<=capacity:
                while used+sizes[key]>capacity:
                    old,_=lru.popitem(last=False);used-=sizes[old]
                    if measuring:counters['cpu_evictions']+=1
                lru[key]=None;used+=sizes[key]
        if measuring:counters[kind]+=1;counters[kind+'_bytes']+=sizes[key]
    for pos,layer,keys in train:
        for k in keys:access(k,False,pos)
    cpu_at_start=pb+used
    for pos,layer,keys in test:
        for k in keys:access(k,True,pos)
    positions=sorted({r[0] for r in test});values=sorted(per_token[p] for p in positions)
    total=len(test)*len(test[0][2]);nt=len(positions)
    return dict(policy=policy,cpu_cache_gib=cpu_bytes/GIB,gpu_cache_gib=gpu_bytes/GIB,
        gpu_resident_gib=gb/GIB,cpu_pinned_gib=pb/GIB,cpu_resident_at_evaluation_start_gib=cpu_at_start/GIB,
        proactive_initial_load_gib=(gb+pb)/GIB,cpu_final_lru_gib=used/GIB,
        gpu_hit_fraction=counters['gpu_hit']/total,cpu_hit_fraction=counters['cpu_hit']/total,
        miss_fraction=counters['ssd_miss']/total,miss_gib_per_token=sum(values)/nt/GIB,
        miss_gib_per_token_p95=values[int((len(values)-1)*.95)]/GIB,
        cold_read_gib=sum(sizes[k] for k in set(miss_keys))/GIB,
        cpu_evictions=counters['cpu_evictions'],evaluated_tokens=nt,
        # Aggregate bandwidth-only ceiling; ignores layer synchronization/latency/compute.
        required_ssd_gib_s_for_10_tok_s=sum(values)/nt/GIB*10)


def load_trace(path,inventory,prompt_tokens,output_ids):
    rows=[];seen=set();stop=next((i for i,t in enumerate(output_ids) if t in inventory['stop_ids']),len(output_ids))
    # Trace at prompt position p consumes generated token 0 to produce token 1.
    end=prompt_tokens+max(0,stop-1)
    for line in path.read_text().splitlines():
        fields=list(map(int,line.split(',')));pos,layer=fields[:2]
        if not prompt_tokens<=pos<end:continue
        if (pos,layer) in seen:raise ValueError('duplicate decode position/layer; use one trial')
        seen.add((pos,layer));keys=[f'{layer}:{e}' for e in fields[2:]]
        if len(set(keys))!=inventory['top_k'] or any(k not in inventory['entries'] for k in keys):raise ValueError('invalid routes')
        rows.append((pos,layer,keys))
    if rows!=sorted(rows,key=lambda r:(r[0],r[1])):raise ValueError('non-sequential trace order')
    positions=sorted({r[0] for r in rows})
    layers=set(range(inventory['first_layer'],inventory['last_layer']))
    for p in positions:
        if {l for pos,l,_ in rows if pos==p}!=layers:raise ValueError('incomplete layer trace')
    if positions!=list(range(prompt_tokens,end)):raise ValueError('missing token positions')
    return rows,dict(first_stop_output_index=stop if stop<len(output_ids) else None,usable_decode_steps=len(positions))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('inventory',type=Path);ap.add_argument('routes',type=Path)
    ap.add_argument('--prompt',type=Path,required=True);ap.add_argument('--output-ids',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--train-tokens',type=int,default=64)
    args=ap.parse_args();inv=json.loads(args.inventory.read_text());sizes={k:v['bytes'] for k,v in inv['entries'].items()}
    count=len(args.prompt.read_text().split(','));ids=list(map(int,args.output_ids.read_text().split()))
    rows,validation=load_trace(args.routes,inv,count,ids)
    split=count+args.train_tokens;train=[r for r in rows if r[0]<split];test=[r for r in rows if r[0]>=split]
    if not test:raise ValueError('too few pre-stop tokens for held-out replay')
    result=dict(validation=validation,train_tokens=args.train_tokens,logical_expert_gib=sum(sizes.values())/GIB,
        assumptions='fixed GPU admission from first 64 generated steps; CPU static/LRU/80%-pinned hybrid, exclusive tiers, exact miss fetch, no speculative routing; no transfer/compute latency modeled',rows=[])
    for gpu in (4,6,8):
        for cpu in (48,52,54,56,58):
            for policy in ('static','lru','hybrid'):
                result['rows'].append(replay(train,test,sizes,cpu*GIB,gpu*GIB,policy))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(validation=validation,rows=len(result['rows']))))


if __name__=='__main__':main()
