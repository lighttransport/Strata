"""Audit step 1 evidence and produce its table; never fit or change engine inputs."""
import argparse,datetime,hashlib,json,math,re,statistics,sys
from pathlib import Path

EXPECTED_CASES=[(0,'none'),(.06,'none'),(0,'none'),(.08,'none'),(0,'none'),(0,'mtp'),(0,'none'),(.08,'mtp'),(0,'none')]
PARAMS_SHA='6786b52fe952e5ec3b86af363964c35d80570ba1a3f8e62573e1cb5c563ad9b6'
EXE_SHA='8c2c754264f5c327997826cbe468b953888c3cbb9d8c165833074a1d8e1df98f'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fixtures',type=Path,required=True);ap.add_argument('--partial',action='store_true');a=ap.parse_args()
    root=Path(__file__).resolve().parents[4];sys.path.insert(0,str(root/'tools/sim'))
    from config import RunConfig
    data=json.loads((a.fixtures/'records.json').read_text());rows=data['records']+data['failed_records'];by_name={r['name']:r for r in rows};errors=[]
    def require(ok,message):
        if not ok:errors.append(message)
    require(hashlib.sha256((root/'tools/sim/data/params_b550.json').read_bytes()).hexdigest()==PARAMS_SHA,'Simulator parameters changed')
    merged=json.loads((root/'tools/sim/data/measured_b550_levers.json').read_text())
    merged_rows=merged['records']+merged['failed_records'];merged_by_name={r['name']:r for r in merged_rows}
    prior=json.loads((a.fixtures/'prior-records.json').read_text());prior_rows=prior['records']+prior['failed_records']
    require(len(merged_by_name)==len(merged_rows)==len(prior_rows)+len(rows),'Append count/duplicate names')
    for r in prior_rows:require(merged_by_name.get(r['name'])==r,r['name']+': earlier measurement was modified')
    for r in rows:require(merged_by_name.get(r['name'])==r,r['name']+': primary record differs from step 1 record')
    selected=data.get('selected_budget_mib')
    if not a.partial:
        require(data['status']=='complete' and selected in (15360,14848,14336),'No completed uniform-budget set')
        for i in range(1,10):
            require(any(r['case_number']==i and r['config']['gpu_budget_mib']==selected and r['status']=='complete' and not r['name'].endswith(('-left','-right')) for r in rows),f'Missing successful case {i}')
    for row in rows:
        name=row['name'];d=row['diagnostic'];cfg=row['config'];RunConfig.from_dict(cfg)
        require(name.startswith('step1-') and row['fit'] is False,name+': name/fit')
        for suffix in ('.config.json','.manifest.json','.result.json','.log','.tokens.json','.record.json','.rocm.json'):
            require((a.fixtures/(name+suffix)).exists(),name+': missing '+suffix)
        raw=json.loads((a.fixtures/(name+'.result.json')).read_text());engine=raw['config'];env=engine['env'];cmd=raw['command']
        require(d['binary_sha256']==EXE_SHA and json.loads((a.fixtures/(name+'.manifest.json')).read_text())['executable_sha256']==EXE_SHA,name+': binary')
        require(cfg['gpu_budget_mib']==engine['gpu_budget_mib'] and cfg['reserve_mib']==512 and cfg['decode_cache_mib']==12288,name+': GPU allowance metadata')
        require(engine['threads']==12 and engine['context']==4096 and engine['prefill_batch']==2048,name+': changed setup')
        require(engine['model']=='/mnt/disk01/models/glm53f-reap/GLM-5.3-Flash-REAP50-Q23-assembled.gguf',name+': changed model')
        require(raw['gpu_capacity_mib']==16304 and raw['gpu_requested_used_limit_mib']==15792,name+': wrong physical VRAM guard')
        manifest=json.loads((a.fixtures/(name+'.manifest.json')).read_text())
        reference=json.loads((a.fixtures.parent/'000-base.manifest.json').read_text())
        require(manifest['prompt_tokens']==2048 and manifest['prompt_sha256']==reference['prompt_sha256'],name+': changed prompt')
        require(env['STRATA_GLM_GPU_RESERVE_MIB']=='512' and env['STRATA_GLM_TIER_RUNTIME_RESERVE_MIB']=='512' and env['STRATA_GLM_TIER_ADAPT']=='1' and env['STRATA_GLM_SPLIT_VERIFY']=='1' and env['STRATA_GLM_STEP_TRACE']=='1',name+': changed tier/trace flags')
        expected=EXPECTED_CASES[row['case_number']-1] if not name.endswith(('-left','-right')) else EXPECTED_CASES[0]
        require((cfg['affinity'],cfg['speculation'])==expected,name+': wrong recipe')
        require(cfg['mtp_depth']==(1 if cfg['speculation']=='mtp' else 0),name+': wrong depth')
        require('--bench=3' in cmd and '--decode-bench=3' in cmd and not any(x.startswith('--stop-ids') for x in cmd),name+': trials/stop ids')
        if cfg['speculation']=='mtp':require('--draft-depth=1' in cmd and '--mtp-experts=cpu' in cmd,name+': MTP flags')
        tail=[]
        for sample in reversed(d['quiet_gate_samples']):
            if sample['load'][0]>=3:break
            tail.append(sample)
        interval=(datetime.datetime.fromisoformat(tail[0]['utc'])-datetime.datetime.fromisoformat(tail[-1]['utc'])).total_seconds() if len(tail)>1 else 0
        require(interval>=179.999 and d['load_start'][0]<3 and row['quiet'],name+': quiet gate')
        require(d['clock_start']['memory_clock_mhz']>100 and 'rocm_smi' in d['clock_start'],name+': memory clock preflight')
        if row['status']=='complete':
            require(raw['exit_code']==0 and raw['complete'] and raw['clean'],name+': guard qualification')
            cg=raw['cgroup_after'];require(cg['memory.max']==60*2**30 and cg['memory.swap.max']==0 and cg['memory.swap.current']==0 and cg['memory.events']['oom_kill']==0,name+': cgroup')
            ids=json.loads((a.fixtures/(name+'.tokens.json')).read_text())['trials']
            require(len(ids)==3 and all(len(s)==512 for s in ids),name+': token lengths')
            require(len(d['raw_lines']['STEP_TRACE'])==3 and 'step_trace_totals_trials_2_3' in d,name+': trace totals')
            require(d['rocm_trials_covered']==[0,1,2] and d['rocm_decode_vram_mib'] is not None,name+': ROCm decode telemetry')
            used=[]
            for sample in d['rocm_decode_samples']:
                if sample.get('confirmed_decode') and sample.get('exit_code')==0 and sample.get('vram_used_mib') is not None:
                    value=re.search(r'VRAM Total Used Memory \(B\):\s*(\d+)',sample['stdout'])
                    require(value is not None and math.isclose(int(value[1])/2**20,sample['vram_used_mib']),name+': raw ROCm usage differs')
                    used.append(sample['vram_used_mib'])
            require(used and max(used)<=15792 and math.isclose(max(used),d['rocm_decode_vram_mib']['max']),name+': physical headroom/ROCm peak')
            require(math.isclose(row['measured']['tok_s'],statistics.mean(t['tokens_per_second'] for t in raw['measurements']['decode'][1:3]),rel_tol=1e-12),name+': mean rate')
            neighbours=row.get('neighbour_base_names',[])
            if row['lever']!='BASE15':
                if not a.partial and cfg['gpu_budget_mib']==selected:require(len(neighbours)==2,name+': no brackets')
                if len(neighbours)==2 and all(by_name[n]['status']=='complete' for n in neighbours):
                    bases=[by_name[n] for n in neighbours]
                    require(all(b['lever']=='BASE15' and b['config']['gpu_budget_mib']==cfg['gpu_budget_mib'] for b in bases),name+': unmatched budget')
                    mean=statistics.mean(b['measured']['tok_s'] for b in bases)
                    require(math.isclose(row['neighbour_base_tok_s'],mean,rel_tol=1e-12) and math.isclose(row['ratio_to_base'],row['measured']['tok_s']/mean,rel_tol=1e-12),name+': ratio')
                    for side,base in zip(('left','right'),bases):
                        baseline=json.loads((a.fixtures/(base['name']+'.tokens.json')).read_text())['trials']
                        for i,(s,t) in enumerate(zip(ids,baseline)):
                            first=next((j for j,(x,y) in enumerate(zip(s,t)) if x!=y),None);count=sum(x==y for x,y in zip(s,t))
                            observed=d['ids_against_base15'][side][i]
                            require(observed['first_differing_index']==first and observed['tokens_matching']==count,name+': token comparison')
                    if row['lever']=='mtp1' and row['ids_equal_to_base'] is False:require(d.get('mtp1_identity_bug') is True,name+': missing MTP identity bug')
    # Successful baseline rows compare to the first BASE15 at their own budget.
    first={}
    for row in sorted(rows,key=lambda r:r['name']):
        if row['lever']=='BASE15' and row['status']=='complete':first.setdefault(row['config']['gpu_budget_mib'],row)
    table=['# B550 step 1 measurements','','| Run | Mean tok/s | Neighbour BASE15 tok/s | Ratio | Tier MiB | Decode VRAM used MiB (min–max) | IDs matching BASE15 / 512, trials 2–3 | Start/end load |','|---|---:|---:|---:|---:|---:|---|---|']
    number=lambda v:f'{v:.3f}' if v is not None else '—'
    for row in sorted(rows,key=lambda r:r['name']):
        if selected is not None and row['config']['gpu_budget_mib']!=selected:continue
        d=row['diagnostic'];caches=d.get('actual_decode_cache',[]);cache=caches[-1] if caches else {};used=d.get('rocm_decode_vram_mib');ids='—'
        comparisons=d.get('ids_against_base15')
        if comparisons:ids='; '.join(side[0].upper()+': '+','.join(str(t['tokens_matching']) for t in comparisons[side][1:3]) for side in ('left','right'))
        elif row['lever']=='BASE15' and row['status']=='complete':
            reference=first[row['config']['gpu_budget_mib']];ss=json.loads((a.fixtures/(row['name']+'.tokens.json')).read_text())['trials'];tt=json.loads((a.fixtures/(reference['name']+'.tokens.json')).read_text())['trials']
            ids='Ref: '+','.join(str(sum(x==y for x,y in zip(s,t))) for s,t in zip(ss[1:3],tt[1:3]))
        table.append('| '+' | '.join([row['name'],number(row['measured']['tok_s']),number(row.get('neighbour_base_tok_s')),number(row.get('ratio_to_base')),number(cache.get('MiB')),f"{used['min']:.0f}–{used['max']:.0f}" if used else '—',ids,f"{d['load_start'][0]:.2f}/{d['load_end'][0]:.2f}"])+' |')
    table+=['','L/R compares the lever to its two neighbouring controls, trial by trial. BASE rows compare to the first BASE15 at the same budget. First differing indices are zero-based and retained for all three trials in each record. End load includes the benchmark. VRAM is total ROCm usage, including driver and other clients, sampled during confirmed decode.','',f"Selected allocation budget: {selected or 'pending'} MiB. Parameter SHA256: `{PARAMS_SHA}`."]
    bugs=[r['name'] for r in rows if r['diagnostic'].get('mtp1_identity_bug')]
    if bugs:table+=['','**BUG: MTP1 without affinity failed the exact BASE15 token-identity requirement:** '+', '.join(bugs)+'. BASE-control reproducibility is recorded separately; this does not isolate the root cause.']
    table+=['','## Token differences','','| Lever | First differing index L, trials 1/2/3 | First differing index R, trials 1/2/3 | Matching positions L / 512 | Matching positions R / 512 |','|---|---|---|---|---|']
    for row in sorted(rows,key=lambda r:r['name']):
        if selected is not None and row['config']['gpu_budget_mib']!=selected:continue
        comparisons=row['diagnostic'].get('ids_against_base15')
        if not comparisons:continue
        fields=[row['lever']]
        for key in ('first_differing_index','tokens_matching'):
            for side in ('left','right'):fields.append(','.join(str(t[key]) if t[key] is not None else 'equal' for t in comparisons[side]))
        table.append('| '+' | '.join(fields)+' |')
    (a.fixtures/'SUMMARY.md').write_text('\n'.join(table)+'\n')
    report=dict(partial=a.partial,runs_checked=len(rows),selected_budget_mib=selected,errors=errors,mtp1_identity_bugs=bugs,params_sha256=PARAMS_SHA)
    (a.fixtures/('audit-partial.json' if a.partial else 'audit.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return bool(errors)

if __name__=='__main__':raise SystemExit(main())
