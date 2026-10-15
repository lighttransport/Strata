#!/usr/bin/env python3
"""Reproducible measured summaries, held-out checks and scoped TR16 bounds.

The timing holdout is a same-shape, training-prompt forecast. It does not validate
unimplemented kernels or extrapolation to other MTP acceptance/cache policies.
"""
import argparse,hashlib,json,statistics
from pathlib import Path
from sim.heterogeneous import streaming_bounds,validate_predictions


def read(path):
    raw=path.read_bytes();return json.loads(raw),dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())


def summarize(path):
    d,identity=read(path);groups={};excluded=[]
    exclusion_path=path.parent/'measurement-exclusions.json'
    timing_exclusions=json.loads(exclusion_path.read_text()) if exclusion_path.exists() else {}
    for r in d['results']:
        m=r.get('measurement',{})
        tag=f"{r['case']}/{Path(r['prompt']).parent.name}/{Path(r['prompt']).stem}/{r['repetition']}"
        external_reason=timing_exclusions.get(tag)
        if external_reason or r['returncode'] or r.get('instrumented') or not m.get('complete') or not m.get('clean'):
            excluded.append(dict(case=r['case'],prompt=Path(r['prompt']).name,repetition=r['repetition'],
                                 returncode=r['returncode'],reason=external_reason or 'failed, incomplete, instrumented or loop-rejected',
                                 warmup_loop=(r.get('stream_health') or [{}])[0].get('consecutive_cycle'),
                                 timed_loop_trials=[i for i,x in enumerate(r.get('stream_health',[])[1:]) if x.get('consecutive_cycle')],
                                 unqualified_decode_tok_s=m.get('measurements',{}).get('median_decode_tok_s'),
                                 unqualified_prefill_tok_s=m.get('measurements',{}).get('median_prefill_tok_s')))
            continue
        counts=m['measurements'];p=Path(r['prompt']);key=(r['case'],p.parent.name,p.stem)
        groups.setdefault(key,[]).append((r,counts))
    rows=[]
    for (case,size,prompt),items in sorted(groups.items()):
        row=dict(case=case,size=size,prompt=prompt,processes=len(items),quality_qualified=False,
                 repetitions=[r['repetition'] for r,m in items])
        row['timed_decode_trials']=sum(len(m['decode']) for r,m in items)
        for kind in ('decode','prefill'):
            values=[m['median_'+kind+'_tok_s'] for r,m in items]
            row[kind+'_tok_s']=statistics.median(values)
            trial_rates=[x['tokens_per_second'] for r,m in items for x in m[kind]]
            row[kind+'_range']=[min(trial_rates),max(trial_rates)]
        row['elapsed_seconds']=statistics.median(r['elapsed_seconds'] for r,m in items)
        row['model_loading_and_other_seconds']=statistics.median(
            r['elapsed_seconds']-sum(x['milliseconds']/1000 for k in ('prefill','decode','decode_warmup') for x in m.get(k,[]))
            for r,m in items)
        cpu=[]
        for r,m in items:
            if m.get('cpu_expert') and m.get('decode'):
                count=len(m['decode'])
                if len(m['cpu_expert'])>=count:
                    cpu.append(sum(x['bytes'] for x in m['cpu_expert'][-count:])/
                               sum(x['tokens'] for x in m['decode']))
        if cpu:row['cpu_bytes_per_output']=statistics.median(cpu)
        row['last_prompt_logits_hashes']=sorted({r['prefill_logits']['sha256'] for r,m in items if r.get('prefill_logits')})
        rows.append(row)
    executed={(r['case'],Path(r['prompt']).parent.name,Path(r['prompt']).stem) for r in d['results']}
    repo=Path(__file__).resolve().parents[1]
    sources={str(Path(k).relative_to(repo)) if Path(k).is_absolute() and Path(k).is_relative_to(repo) else k:v
             for k,v in d.get('source_sha256',{}).items()}
    inputs=d.get('input_identity',{})
    weight_identity=[{**w,'path':Path(w['path']).name} for w in inputs.get('weight_files',[])]
    return dict(source=identity,executables=d['executables'],source_commit=d.get('source_commit'),
                source_sha256=sources,protocol=d.get('protocol'),timed_trials_per_process=d.get('timed_trials_per_process'),
                weight_identity=weight_identity,weight_identity_scope=inputs.get('note'),expert_prior_sha256=inputs.get('expert_prior_sha256'),
                config_sha256=hashlib.sha256(json.dumps(d['config'],sort_keys=True).encode()).hexdigest(),
                executed_processes=len(d['results']),executed_matrix=len(executed)==12,
                rows=rows,excluded=excluded,complete_matrix=len(rows)==12 and all(r['timed_decode_trials']>=3 for r in rows))


def shape_validation(summary):
    training={(r['case'],r['size']):r for r in summary['rows'] if r['prompt']=='train-code'}
    records=[]
    for row in summary['rows']:
        if row['prompt']!='holdout-chat' or (row['case'],row['size']) not in training:continue
        source=training[row['case'],row['size']]
        for kind in ('decode','prefill'):
            records.append(dict(case=row['case'],size=row['size'],kind=kind,split='holdout',
                                predicted_tok_s=source[kind+'_tok_s'],measured_tok_s=row[kind+'_tok_s']))
    result=validate_predictions(records)
    result['scope']='Same configuration/context training-code throughput predicts held-out chat. No held-out routing or acceptance used in the fit. Does not qualify new scheduling policies.'
    if not summary['complete_matrix']:result['calibrated']=False
    return result


def make_report(root,hardware):
    h,hid=read(hardware);result=dict(hardware=hid,quality_floor=152,quality_total=164,packs={})
    for name in ('reap224','full'):
        p=root/('after-'+name)/'results.json'
        if p.exists():
            s=summarize(p);result['packs'][name]=dict(measurements=s,holdout=shape_validation(s))
    before=root/'before-reap224-trials/results.json'
    if before.exists():
        result['before']=summarize(before)
        prior={(r['case'],r['size'],r['prompt']):r for r in result['before']['rows']}
        comparisons=[]
        for r in result.get('packs',{}).get('reap224',{}).get('measurements',{}).get('rows',[]):
            b=prior.get((r['case'],r['size'],r['prompt']))
            if b:
                comparisons.append(dict(case=r['case'],size=r['size'],prompt=r['prompt'],
                    prefill_change_percent=100*(r['prefill_tok_s']/b['prefill_tok_s']-1),
                    decode_change_percent=100*(r['decode_tok_s']/b['decode_tok_s']-1),
                    prefill_last_logits_equal=b['last_prompt_logits_hashes']==r['last_prompt_logits_hashes'],
                    process_counts=[b['processes'],r['processes']]))
        result['upstream_comparison']=comparisons
    # Retained fixed-route byte count is explicitly a historical diagnostic input,
    # never training data silently blended with the post-merge generation.
    old_bytes=375624433664/512
    result['historical_trace_transport_bounds']={pack:streaming_bounds(old_bytes,size,h)
        for pack,size in [('full',111188901888),('reap224',86480257024)]}
    result['historical_trace_transport_bounds']['full'].pop('all_ram_misses_streamed_to_gpus_tok_s')
    result['historical_trace_transport_bounds']['full'].get('encoding_only_bounds',{}).pop('all_ram_misses_streamed_to_gpus_tok_s',None)
    result['historical_trace_transport_bounds_scope']='CPU bytes from pre-merge redesign-trace-v2, fixed routes/acceptance. Full/reap differ only in cold expert-table read size here; no full-pack decode prediction.'
    result['complete']=len(result['packs'])==2 and all(x['measurements']['complete_matrix'] for x in result['packs'].values())
    result['measurement_execution_complete']=len(result['packs'])==2 and all(x['measurements']['executed_matrix'] for x in result['packs'].values())
    result['follow_up']={}
    for tag in ('maya-old-control','maya-default-control','maya-admission','prefill-10of14',
                'late-upstream-before','late-upstream-after','primary-only-smoke'):
        path=root/tag/'results.json'
        if path.exists():result['follow_up'][tag]=summarize(path)
    if 'primary-only-smoke' in result['follow_up']:
        result['follow_up']['primary-only-smoke']['scope']='32-token numerical/liveness smoke, not performance qualification; parity checks overlapped the single-GPU control'
    wrap=root/'wrap-up.json'
    if wrap.exists():result['wrap_up']=json.loads(wrap.read_text())
    result['limitations']=['Quality cannot be simulated; the existing 152/164 floor remains an empirical gate.',
        'Repeated prompts warm the exact workload cache; these rates do not predict fresh-request throughput.',
        'Cold disk bounds assume reading every expert weight once, without RAM-cache hits.',
        'Three timed trials in one process are a small sample, not a confidence interval; restart controls are retained separately.',
        'Same-shape holdouts do not validate changes of architecture, precision, pruning or speculative acceptance.']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--hardware',type=Path,default=Path('tools/sim/data/hw_tr16_mixed.json'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=make_report(a.runs,a.hardware)
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'complete':r['complete'],'packs':list(r['packs'])}))
if __name__=='__main__':main()
