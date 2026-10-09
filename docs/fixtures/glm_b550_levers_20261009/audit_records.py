"""Verify recorded measurements and protocol, without simulating or refitting."""
import argparse,copy,datetime,hashlib,json,math,statistics,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fixtures',type=Path,required=True);ap.add_argument('--records',type=Path,required=True);ap.add_argument('--partial',action='store_true');a=ap.parse_args()
    root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'tools/sim'))
    from config import RunConfig
    data=json.loads(a.records.read_text());all_rows=data['records']+data.get('failed_records',[]);rows={r['name']:r for r in all_rows};errors=[];checked=[]
    def require(ok,message):
        if not ok:errors.append(message)
    if not a.partial:require(data['status']=='complete','sweep not complete')
    required={'mtp1','mtp2','mtp3','mtp2-unsplit','mtp2-aff05','mtp2-aff10','mtp3-margin2','static-tier','cache3072','largest-admitted-cache','workers8','workers16','pp128','pp1024','pp2048','pp4096','pp4096-prefetch0','pp4096-prefetch8','pp4096-prefetch12'}
    attempted={r.get('lever') for r in all_rows if r.get('attempt',1)==1}
    if not a.partial:require(required<=attempted,'missing levers: '+str(sorted(required-attempted)))
    base=json.loads((a.fixtures/'base.config.json').read_text())
    def expected_config(row):
        cfg=copy.deepcopy(base);env=cfg['env'];lever=row.get('lever');cohort=row.get('cohort')
        if row['name']=='000-base':return cfg
        env['STRATA_GLM_STEP_TRACE']='1'
        if cohort=='mtp512':env['STRATA_GLM_TIER_RUNTIME_RESERVE_MIB']='512'
        elif cohort=='prefill2g8k':
            cfg['context']=8192;env['STRATA_GLM_PREFILL_SCRATCH_CAP_MIB']='2048'
        else:require(cohort=='ordinary256',row['name']+': unknown cohort')
        if lever.startswith('mtp'):
            cfg['speculative']='mtp';cfg['draft_depth']=int(lever[3])
            if lever=='mtp2-unsplit':env['STRATA_GLM_SPLIT_VERIFY']='0'
            elif lever=='mtp2-aff05':env['STRATA_GLM_ROUTE_AFFINITY']='0.05'
            elif lever=='mtp2-aff10':env['STRATA_GLM_ROUTE_AFFINITY']='0.1'
            elif lever=='mtp3-margin2':env['STRATA_GLM_MTP_CONTINUATION_MARGIN']='2'
        elif lever=='static-tier':env.pop('STRATA_GLM_TIER_ADAPT')
        elif lever=='cache3072':cfg['decode_cache_mib']=3072
        elif lever=='largest-admitted-cache':cfg['decode_cache_mib']=12288 if row.get('supported_cli_max_cache') else 14336
        elif lever in ('workers8','workers16'):cfg['threads']=int(lever[7:])
        elif lever.startswith('pp'):
            cfg['prefill_batch']=min(int(lever[2:].split('-')[0]),4096)
            if '-prefetch' in lever:
                groups=int(lever.split('prefetch')[1])
                if groups:env.update(STRATA_GLM_STAGE_PREFETCH='1',STRATA_GLM_PREFETCH_GROUPS=str(groups))
        else:require(lever=='BASE',row['name']+': unknown lever')
        return cfg
    for row in all_rows:
        name=row['name'];diag=row['diagnostic'];require(row['hw']=='b550' and row['fit'] is False,name+': hw/fit')
        try:RunConfig.from_dict(row['config'])
        except Exception as e:errors.append(name+': config schema '+repr(e))
        for suffix in ('.config.json','.manifest.json','.result.json','.log','.tokens.json','.record.json'):
            require((a.fixtures/(name+suffix)).exists(),name+': missing '+suffix)
        raw_path=a.fixtures/(name+'.result.json')
        if not raw_path.exists():continue
        raw=json.loads(raw_path.read_text());cfg=raw['config'];env=cfg['env'];cmd=raw['command'];m=raw['measurements']
        require(cfg==expected_config(row),name+': engine flags differ from the specified lever/control')
        manifest=json.loads((a.fixtures/(name+'.manifest.json')).read_text())
        require(manifest['config']==cfg,name+': manifest/result flags differ')
        require(manifest['executable_sha256']==diag['binary_sha256'],name+': manifest/result binary differs')
        expected_prompt=int(row['lever'][2:].split('-')[0]) if row['kind']=='prefill' and row['lever']!='BASE' else 2048
        require(manifest['prompt_tokens']==expected_prompt and row['config']['prompt']==expected_prompt,name+': incorrect prompt length')
        require(row['config']['generate']==(512 if row['kind']=='decode' else 1),name+': incorrect generation count')
        if row['config']['speculation']=='mtp':require('--mtp-experts=cpu' in cmd,name+': MTP experts not on CPU')
        require(diag['binary_sha256']=='8c2c754264f5c327997826cbe468b953888c3cbb9d8c165833074a1d8e1df98f',name+': wrong binary')
        require(cfg['model']=='/mnt/disk01/models/glm53f-reap/GLM-5.3-Flash-REAP50-Q23-assembled.gguf',name+': wrong model')
        require(cfg['gpu_budget_mib']==14336 and env['STRATA_GLM_GPU_RESERVE_MIB']=='2048',name+': GPU budget/reserve')
        require('--bench=3' in cmd and '--decode-bench=3' in cmd and not any(s.startswith('--stop-ids') for s in cmd),name+': trial/stop flags')
        if name!='000-base':require(env.get('STRATA_GLM_STEP_TRACE')=='1',name+': trace not enabled')
        require(diag['load_start'][0]<3 and row['quiet'] is True,name+': start not quiet')
        tail=[]
        for sample in reversed(diag['quiet_gate_samples']):
            if sample['load'][0]>=3:break
            tail.append(sample)
        elapsed=(datetime.datetime.fromisoformat(tail[0]['utc'])-datetime.datetime.fromisoformat(tail[-1]['utc'])).total_seconds() if len(tail)>1 else 0
        require(elapsed>=179.999,name+': quiet interval shorter than 180 seconds')
        require(diag['clock_start']['memory_clock_mhz']>100 and 'rocm_smi' in diag['clock_start'],name+': clock preflight not verified')
        require('load_end' in diag and 'clock_end' in diag and 'wall_seconds' in diag and 'exit_code' in diag,name+': missing endpoint diagnostics')
        if row['status']=='complete':
            require(raw['exit_code']==0 and raw['clean'] and raw['complete'],name+': underlying run not qualified')
            cg=raw['cgroup_after'];require(cg['memory.max']==60*2**30 and cg['memory.swap.max']==0 and cg['memory.swap.current']==0 and cg['memory.events'].get('oom_kill',0)==0,name+': cgroup/guard')
            trials=m['decode'] if row['kind']=='decode' else m['prefill'];require(len(trials)==3,name+': not three timed trials')
            require(math.isclose(row['measured']['tok_s'],statistics.mean(t['tokens_per_second'] for t in trials[1:3]),rel_tol=1e-12),name+': incorrect trials 2-3 mean')
            tok=json.loads((a.fixtures/(name+'.tokens.json')).read_text())
            if row['kind']=='decode':
                require(len(tok.get('trials',[]))==3 and all(len(t)==512 for t in tok.get('trials',[])),name+': output length')
                if name!='000-base':require(len(diag['raw_lines']['STEP_TRACE'])==3,name+': missing per-trial trace')
            else:
                require(not m['decode'] and len(tok.get('prefill_first_tokens',[]))==3 and diag['decode_steps']==0,name+': prefill performed decode steps')
                require(cfg['context']==8192 and env['STRATA_GLM_PREFILL_SCRATCH_CAP_MIB']=='2048',name+': prefill matched-control cohort')
        else:require(row['measured']['tok_s'] is None,name+': failed attempt has usable-looking rate')
        if row.get('lever')!='BASE':
            neighbours=row.get('neighbour_base_names',[])
            if not a.partial:require(len(neighbours)==2,name+': missing neighbours')
            if len(neighbours)==2 and all(n in rows for n in neighbours):
                left,right=[rows[n] for n in neighbours];require(left.get('lever')==right.get('lever')=='BASE',name+': neighbour not BASE')
                require(left.get('cohort')==right.get('cohort')==row.get('cohort'),name+': mixed control cohorts')
                require(int(left['name'].split('-')[0])<int(name.split('-')[0])<int(right['name'].split('-')[0]),name+': controls not interleaved')
                if all(r['status']=='complete' for r in (left,right)):
                    mean=statistics.mean(r['measured']['tok_s'] for r in (left,right))
                    require(math.isclose(row['neighbour_base_tok_s'],mean,rel_tol=1e-12),name+': missing/incorrect neighbour mean')
                    if row['status']!='complete':require(row['ratio_to_base'] is None,name+': failed lever has a ratio')
                if all(r['status']=='complete' for r in (row,left,right)):
                    mean=statistics.mean(r['measured']['tok_s'] for r in (left,right));require(math.isclose(row['neighbour_base_tok_s'],mean,rel_tol=1e-12) and math.isclose(row['ratio_to_base'],row['measured']['tok_s']/mean,rel_tol=1e-12),name+': neighbour ratio')
                    if row['kind']=='decode':
                        streams=[json.loads((a.fixtures/(r['name']+'.tokens.json')).read_text())['trials'] for r in (row,left,right)];equal=all(t==streams[1][0] for ss in streams for t in ss)
                        require(row['ids_equal_to_base'] is equal,name+': incorrect token equality')
        checked.append(dict(name=name,status=row['status'],quiet_seconds=elapsed))
    if not a.partial:
        require(any(r.get('supported_cli_max_cache') and r['status']=='complete' for r in all_rows),'largest fitting cache not measured at supported CLI cap')
        for r in all_rows:
            if r.get('lever')!='BASE' and r.get('attempt',1)==1 and r['status']!='complete':require(any(x.get('lever')==r['lever'] and x.get('attempt')==2 for x in all_rows),r['name']+': no failure retry')
        gate=rows.get('000-base',{});require(.9<=gate.get('measured',{}).get('tok_s',0)/10.46455<=1.1,'initial decode gate');require(.9<=gate.get('measured',{}).get('prefill_tok_s',0)/230.93<=1.1,'initial prefill gate')
    plan=json.loads((a.fixtures/'measurement-plan.json').read_text());current=hashlib.sha256((root/'tools/sim/data/params_b550.json').read_bytes()).hexdigest();require(current==plan.get('params_b550_expected_sha256',plan['params_b550_before_sha256']),'params_b550 changed outside recorded user-authorized checkpoint')
    report=dict(partial=a.partial,records_checked=len(checked),successful=len(data['records']),failed=len(data.get('failed_records',[])),errors=errors,checks=checked,params_sha256=current)
    (a.fixtures/('audit-partial.json' if a.partial else 'audit.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2));return bool(errors)
if __name__=='__main__':raise SystemExit(main())
