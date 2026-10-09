"""Add diagnostic summaries to completed measurement records; never fit simulator parameters."""
import argparse,hashlib,json,re,statistics
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fixtures',type=Path,required=True);ap.add_argument('--records',type=Path,required=True);a=ap.parse_args()
    data=json.loads(a.records.read_text());rows=data['records']+data.get('failed_records',[])
    by_name={r['name']:r for r in rows}
    for row in rows:
        path=a.fixtures/(row['name']+'.result.json')
        if not path.exists():continue
        raw=json.loads(path.read_text());diag=row['diagnostic'];cfg=raw['config'];env=cfg['env'];row['runtime_headroom_mib']=int(env['STRATA_GLM_TIER_RUNTIME_RESERVE_MIB'])
        if 'STRATA_GLM_TIER_ADAPT' not in env and env.get('STRATA_GLM_EXPERT_PRIOR'):
            row['config']['tier_policy']='static_prior'
            diag['tier_policy_note']='Static residency retains the same expert prior as BASE; simulator static_prior includes that prior, while static means prompt-only ranking.'
        row['config']['prefill_experts']='gpu';row['tolerance']=.10 if row['kind']=='decode' else .15;row['scope']='Measurement only; no fit'
        active=[];all_clocks=[]
        for s in raw.get('samples',[]):
            p=s.get('gpu_performance',{});match=re.search(r'(\d+)Mhz\s*\*',p.get('pp_dpm_mclk') or '')
            if match:
                n=int(match[1]);all_clocks.append(n)
                if str(p.get('gpu_busy_percent','0')).isdigit() and int(p['gpu_busy_percent'])>0:active.append(n)
        diag['guard_mclk_range_mhz']=[min(all_clocks),max(all_clocks)] if all_clocks else None
        diag['loaded_mclk_range_mhz']=[min(active),max(active)] if active else None
        diag['loaded_mclk_max_mhz']=max(active) if active else None
        diag['clock_note']='GPU may return to 96 MHz after exit and during brief idle gaps; a loaded peak above 100 MHz shows it did not remain stuck there. Clock settings were not reset or changed.'
        diag['actual_decode_cache']=raw.get('measurements',{}).get('decode_cache',[])
        diag['minimum_gpu_free_mib']=raw.get('minimum_gpu_free_mib')
        by_trial=diag.get('decode_trial_lines',{})
        parsed_trace={}
        for trial in (0,1,2):
            lines=by_trial.get(str(trial),by_trial.get(trial,{})).get('STEP_TRACE',[])
            if lines:parsed_trace[trial]={k:float(v) for k,v in re.findall(r'(\w+)=([0-9.eE+-]+)(?:\s|$)',lines[-1])}
        if len(parsed_trace)==3:
            totals={}
            for trial in (1,2):
                for k,v in parsed_trace[trial].items():
                    if k in ('pipelined','per_step_ms','tier_swaps','split_windows'):continue
                    totals[k]=totals.get(k,0)+v
            for k in ('tier_swaps','split_windows'):
                if k in parsed_trace[2] and k in parsed_trace[0]:totals[k]=parsed_trace[2][k]-parsed_trace[0][k]
            if totals.get('steps'):totals['weighted_per_step_ms']=sum(totals.get(k,0) for k in ('head_ms','cpu_ms','between_ms','tail_ms'))/totals['steps']
            diag['step_trace_totals_trials_2_3']=totals
            diag['step_trace_totals_note']='Times/steps/layers summed for trials 2-3; cumulative tier_swaps and split_windows use end-trial-3 minus end-trial-1.'
        diag['step_trace_status']='disabled_reference_gate' if row['name']=='000-base' else 'not_emitted_zero_decoder_steps' if row['kind']=='prefill' else 'emitted' if diag.get('raw_lines',{}).get('STEP_TRACE') else 'not_emitted'
        assert not any(arg.startswith('--stop-ids') for arg in raw.get('command',[]))
        if row['status']=='complete':
            assert row['fit'] is False and row['quiet'] is True
            assert raw['memory_limit_gib']==60 and raw['swap_limit_bytes']==0
            assert raw['cgroup_after']['memory.max']==60*2**30 and raw['cgroup_after']['memory.swap.max']==0
            assert len(row['measured']['trials'])==2
            tokens=json.loads((a.fixtures/(row['name']+'.tokens.json')).read_text())
            if row['kind']=='decode':assert len(tokens['trials'])==3 and all(len(t)==512 for t in tokens['trials'])
            else:assert diag['decode_steps']==0 and not raw['measurements']['decode']
        neighbours=row.get('neighbour_base_names',[])
        if len(neighbours)==2 and all(n in by_name for n in neighbours):
            controls=[by_name[n] for n in neighbours]
            if all(r['status']=='complete' for r in controls):
                mean=statistics.mean(r['measured']['tok_s'] for r in controls)
                row['neighbour_base_tok_s']=mean
                row['ratio_to_base']=row['measured']['tok_s']/mean if row['status']=='complete' else None
            else:row['neighbour_base_tok_s']=row['ratio_to_base']=None
        if row['kind']=='decode' and len(neighbours)==2:
            files=[a.fixtures/(n+'.tokens.json') for n in [row['name'],*neighbours]]
            if all(p.exists() for p in files):
                streams=[json.loads(p.read_text()).get('trials',[]) for p in files]
                if all(len(ss)==3 and all(len(s)==512 for s in ss) for ss in streams):
                    row['control_ids_equal']=all(s==streams[1][0] for ss in streams[1:] for s in ss)
                    row['ids_equal_left_base']=all(s==streams[1][0] for s in streams[0])
                    row['ids_equal_right_base']=all(s==streams[2][0] for s in streams[0])
                    diag['comparison_note']='BASE neighbours have different token streams; ratio is not a controlled identical-token speedup.' if not row['control_ids_equal'] else 'BASE neighbours have identical token streams.'
        (a.fixtures/(row['name']+'.record.json')).write_text(json.dumps(row,indent=2)+'\n')
    headers=['Lever','Kind','Measured tok/s','Neighbour BASE tok/s','Ratio','IDs equal to BASE','Actual slots / MiB','Start / end load','Start / end / loaded peak MCLK MHz','Status']
    table=['# B550 lever measurements — 2026-10-09','','| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]
    for row in sorted(rows,key=lambda r:r['name']):
        if row.get('lever')=='BASE' and row['name']!='000-base':continue
        d=row['diagnostic'];b=row.get('neighbour_base_tok_s');ratio=row.get('ratio_to_base');equal=row.get('ids_equal_to_base');caches=d.get('actual_decode_cache',[]);cache=caches[-1] if caches else None
        number=lambda n: f'{n:.4f}' if n is not None else '—'
        vals=[row.get('lever','BASE')+(' (retry)' if row.get('attempt',1)>1 else ''),row['kind'],number(row['measured']['tok_s']),number(b),f'{ratio:.3f}' if ratio is not None else '—','yes' if equal is True else 'no' if equal is False else '—',f"{cache.get('slots')} / {cache.get('MiB'):.2f}" if cache else '—',f"{d['load_start'][0]:.2f} / {d['load_end'][0]:.2f}",f"{d['clock_start']['memory_clock_mhz']} / {d['clock_end']['memory_clock_mhz']} / {d.get('loaded_mclk_max_mhz') or '—'}",row['status']]
        table.append('| '+' | '.join(vals)+' |')
    table+=['','Rates are means of trials 2–3. Each ratio uses the mean of the two neighbouring matched BASE runs. Token equality uses all three 512-token streams and both BASE neighbours. Prefill has no decoder steps; its one emitted token per trial was already computed by prefill. End load includes the benchmark itself. A 96 MHz clock after exit is an idle reading; loaded peaks are reported separately.','',f"Successful runs: {len(data['records'])}; failed attempts: {len(data.get('failed_records',[]))}. Failed attempts are retained separately from importable records, with null tok/s. See each manifest, result, log, token file and record for exact flags and diagnostics."]
    (a.fixtures/'SUMMARY.md').write_text('\n'.join(table)+'\n')
    a.records.write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__':main()
