"""Interleaved measurement-only lever sweep; requires successful unchanged-flags BASE gate."""
import copy,hashlib,json,os,re,statistics,subprocess,sys,time
from pathlib import Path
import run_measurements as core
OUT=Path(__file__).resolve().parent
ROOT=core.ROOT
sys.path.insert(0,str(ROOT/'tools'))
import glm_low_memory_bench as guard
CURRENT=ROOT/'tools/sim/data/measured_b550_levers.json'
TAGS=core.TAGS+('DECODE_TRIAL','PREFETCH_STAGED','DECODE_ADAPT','MTP_CONTINUATION')
ROWS=[];COUNTER=0

def enrich(row,cfg,label):
    env=cfg['env'];row['runtime_headroom_mib']=int(env['STRATA_GLM_TIER_RUNTIME_RESERVE_MIB'])
    row['tolerance']=.10 if row['kind']=='decode' else .15;row['scope']='Measurement only; no parameter fit';row['config']['prefill_experts']='gpu';row['diagnostic']['engine_env']=env
    row['diagnostic']['step_trace_note']='Enabled in all interleaved cases and controls after the uninstrumented reference BASE gate.' if env.get('STRATA_GLM_STEP_TRACE') else 'Disabled for unchanged-flags reference BASE gate.'
    text=(OUT/(label+'.log')).read_text() if (OUT/(label+'.log')).exists() else ''
    row['diagnostic']['raw_lines']={tag:[l for l in text.splitlines() if l.startswith(tag+' ')] for tag in TAGS}
    by_trial={};trial=None
    for l in text.splitlines():
        match=re.match(r'DECODE_TRIAL index=(\d+)',l)
        if match:trial=int(match[1]);by_trial[trial]={tag:[] for tag in TAGS}
        if trial is not None:
            for tag in TAGS:
                if l.startswith(tag+' '):by_trial[trial][tag].append(l)
    row['diagnostic']['decode_trial_lines']=by_trial
    total={}
    for trial in (1,2):
        for l in by_trial.get(trial,{}).get('STEP_TRACE',[]):
            for k,v in re.findall(r'(\w+)=([0-9.eE+-]+)(?:\s|$)',l):
                if k in ('pipelined','per_step_ms'):continue
                total[k]=total.get(k,0)+float(v)
    row['diagnostic']['step_trace_totals_trials_2_3']=total
    if total.get('steps'):total['weighted_per_step_ms']=sum(total.get(k,0) for k in ('head_ms','cpu_ms','between_ms','tail_ms'))/total['steps']
    if row['status']=='complete' and row['kind']=='decode':
        for i,t in enumerate(row['measured']['trials'],2):t['trial_number']=i
    core.save(OUT/(label+'.record.json'),row)
    return row

def summarize():
    successful=[r for r in ROWS if r['status']=='complete'];failed=[r for r in ROWS if r['status']!='complete']
    data=dict(records=successful,failed_records=failed,status='running',aggregation='Mean of trials 2-3; trial 1 excluded. Every decode trial emits exactly 512 tokens without stop ids. Prefill runs emit one precomputed first token and execute zero decoder steps.',protocol_authorizations=json.loads((OUT/'measurement-plan.json').read_text()),quiet_definition='1-minute load average <3 continuously for >=180 seconds before each run. End load includes the benchmark itself.',failed_record_note='Failed attempts have null tok_s and are separate from importable records. No engine fix is applied; failed levers are retried once with matched BASE controls at the end.')
    core.save(OUT/'sweep-records.json',data);core.save(CURRENT,data)
    headers=['Lever','Kind','tok/s','Neighbour BASE tok/s','Ratio','IDs equal to BASE','Start/end load','Start/end MCLK MHz','Status']
    lines=['# B550 lever measurements — 2026-10-09','', '| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]
    for r in ROWS:
        if r.get('lever')=='BASE' and r['name']!='000-base':continue
        d=r['diagnostic'];c1=d['clock_start']['memory_clock_mhz'];c2=d['clock_end']['memory_clock_mhz'];b=r.get('neighbour_base_tok_s');ratio=r.get('ratio_to_base');equal=r.get('ids_equal_to_base')
        vals=[r.get('lever','BASE')+(' (retry)' if r.get('attempt',1)>1 else ''),r['kind'],f"{r['measured']['tok_s']:.4f}" if r['measured']['tok_s'] is not None else '—',f'{b:.4f}' if b is not None else '—',f'{ratio:.3f}' if ratio is not None else '—','yes' if equal is True else 'no' if equal is False else '—',f"{d['load_start'][0]:.2f}/{d['load_end'][0]:.2f}",f'{c1}/{c2}',r['status']]
        lines.append('| '+' | '.join(vals)+' |')
    (OUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n')

def capture_prefill(label,cfg,prompt,lever,attempt):
    exe=Path(cfg['exe']);digest=hashlib.sha256(exe.read_bytes()).hexdigest()
    if digest!=core.EXPECTED:raise RuntimeError('pinned binary changed')
    samples,start_clock=core.quiet_gate(OUT,label);start_load=list(os.getloadavg());start=time.monotonic()
    path=OUT/(label+'.config.json');core.save(path,cfg);prefix=OUT/label
    cache=guard.clear_model_cache(cfg)
    worker=[core.PYTHON,str(OUT/'prefill_worker.py'),str(path),str(prompt),'--output',str(prefix)]
    unit='strata-b550-pp-'+label
    cmd=['systemd-run','--user','--wait','--pipe','--unit='+unit,'--property=MemoryMax=60G','--property=MemorySwapMax=0','--property=IOAccounting=yes','--property=OOMPolicy=kill','--property=RuntimeMaxSec=1230','--property=WorkingDirectory='+str(ROOT),*worker]
    core.save(prefix.with_suffix('.manifest.json'),dict(config=cfg,prompt_tokens=len(prompt.read_text().strip().split(',')),prompt_sha256=hashlib.sha256(prompt.read_text().strip().encode()).hexdigest(),executable_sha256=digest,command=cmd,guard_source_sha256=hashlib.sha256(Path(guard.__file__).read_bytes()).hexdigest(),prefill_adapter_sha256=hashlib.sha256((OUT/'prefill_worker.py').read_bytes()).hexdigest(),cache_clear=cache,memory_gib=60,compute_gpu='installed GPU',capacity_simulation=True,fit_only=False,mode='prefill_only'))
    core.save(OUT/'progress.json',dict(state='running',run=label,started_utc=core.now(),command=cmd));print('START '+label,flush=True)
    with prefix.with_suffix('.driver.log').open('w') as log:p=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    end_load=list(os.getloadavg());wall=time.monotonic()-start;end_clock=core.clocks()
    result_path=prefix.with_suffix('.result.json');r=json.loads(result_path.read_text()) if result_path.exists() else {}
    m=r.get('measurements',{});pp=m.get('prefill',[]);good=p.returncode==0 and r.get('complete') and r.get('clean') and len(pp)==3
    n=len(prompt.read_text().strip().split(','));env=cfg['env'];groups=int(env.get('STRATA_GLM_PREFETCH_GROUPS','0'))
    config=dict(pack='reap50_q23',quality_reference='reap50_q23',threads=cfg['threads'],context=cfg['context'],prompt=n,generate=1,prefill_chunk=cfg['prefill_batch'],speculation='none',mtp_depth=1,decode_cache_mib=cfg['decode_cache_mib'],gpu_budget_mib=cfg['gpu_budget_mib'],reserve_mib=float(env['STRATA_GLM_GPU_RESERVE_MIB']),split_verify=True,tier_policy='adaptive',affinity=0,prefetch_groups=groups,tier_owned_reserve=True,prefill_scratch_mib=2048)
    tokens=prefix.with_suffix('.stdout');ids=list(map(int,tokens.read_text().split())) if tokens.exists() else [];core.save(prefix.with_suffix('.tokens.json'),dict(prefill_first_tokens=ids,decode_steps=0))
    row=dict(name=label,kind='prefill',hw='b550',config=config,fit=False,quiet=True,provenance='docs/fixtures/glm_b550_levers_20261009/'+label+'.result.json',status='complete' if good else 'failed',lever=lever,attempt=attempt,measured=dict(tok_s=statistics.mean(x['tokens_per_second'] for x in pp[1:3]) if good else None,trials=pp[1:3],all_trials=pp),diagnostic=dict(parsed=m,load_start=start_load,load_end=end_load,quiet_gate_samples=samples,clock_start=start_clock,clock_end=end_clock,exit_code=p.returncode,engine_exit_code=r.get('exit_code'),wall_seconds=wall,engine_wall_seconds=r.get('wall_seconds'),cgroup=r.get('cgroup_after'),rejected=r.get('rejected'),interference=r.get('interference'),binary_sha256=digest,token_ids_file='docs/fixtures/glm_b550_levers_20261009/'+label+'.tokens.json',decode_steps=0,prefill_mode='One first-token emission per trial, already computed by prefill; no decoder step or timing. Shared 8K context permits the 4K prompt with the engine required steps>=1.'))
    print('DONE '+json.dumps(dict(name=label,status=row['status'],tok_s=row['measured']['tok_s'])),flush=True)
    return row

def execute(case):
    global COUNTER
    label=f'{COUNTER:03d}-'+case['name'];COUNTER+=1
    path=OUT/(label+'.record.json')
    if path.exists():return json.loads(path.read_text())
    if (OUT/'stop_requested').exists():raise RuntimeError('user stop marker: stop before starting another case')
    if case['kind']=='decode':r=core.run_decode(OUT,label,case['config'],case['lever'],case['attempt'])
    else:r=capture_prefill(label,case['config'],case['prompt'],case['lever'],case['attempt'])
    r['cohort']=case['cohort'];return enrich(r,case['config'],label)

def compare(lever,left,right):
    lever['neighbour_base_names']=[left['name'],right['name']]
    if all(x['status']=='complete' for x in (lever,left,right)):
        base=statistics.mean(x['measured']['tok_s'] for x in (left,right));lever['neighbour_base_tok_s']=base;lever['ratio_to_base']=lever['measured']['tok_s']/base
    else:lever['neighbour_base_tok_s']=None;lever['ratio_to_base']=None
    if lever['kind']=='decode' and all(x['status']=='complete' for x in (lever,left,right)):
        streams=[]
        for row in (lever,left,right):streams.append(json.loads((OUT/(row['name']+'.tokens.json')).read_text())['trials'])
        lever['ids_equal_to_base']=all(s==streams[1][0] for ss in streams for s in ss)
        if not lever['ids_equal_to_base']:
            mismatches=[]
            for name,ss in zip([lever['name'],left['name'],right['name']],streams):
                for i,s in enumerate(ss,1):
                    first=next((j for j,(a,b) in enumerate(zip(s,streams[1][0])) if a!=b),None)
                    if first is not None:mismatches.append(dict(run=name,trial=i,first_mismatch=first,actual=s[first],left_base=streams[1][0][first]))
            lever['diagnostic']['token_mismatches']=mismatches
        lever['diagnostic']['lossy']=lever['config']['affinity']>0
    else:lever['ids_equal_to_base']=None
    core.save(OUT/(lever['name']+'.record.json'),lever)

def block(cases,control):
    left=execute(control);ROWS.append(left);summarize();fail=[]
    for case in cases:
        lever=execute(case);ROWS.append(lever);summarize()
        right=execute(control);ROWS.append(right);compare(lever,left,right);summarize()
        if not all(r['status']=='complete' for r in (left,lever,right)):fail.append(case)
        left=right
    return fail

def main():
    global COUNTER
    gate=json.loads((OUT/'records.json').read_text())
    if gate['status']!='baseline_gate_passed':raise RuntimeError('unchanged-flags BASE failed its 10 percent gate; no levers run')
    ROWS.extend(gate['records']);COUNTER=1;summarize();cfg=json.loads((OUT/'base.config.json').read_text());cfg['env']['STRATA_GLM_STEP_TRACE']='1'
    def case(name,config,cohort,kind='decode',prompt=None,attempt=1,lever=None):return dict(name=name,config=copy.deepcopy(config),cohort=cohort,kind=kind,prompt=prompt,attempt=attempt,lever=lever or name)
    mtp_base=copy.deepcopy(cfg);mtp_base['env']['STRATA_GLM_TIER_RUNTIME_RESERVE_MIB']='512';mtp=[]
    for n in (1,2,3):
        c=copy.deepcopy(mtp_base);c.update(speculative='mtp',draft_depth=n);mtp.append(case('mtp'+str(n),c,'mtp512'))
    c=copy.deepcopy(mtp_base);c.update(speculative='mtp',draft_depth=2);c['env']['STRATA_GLM_SPLIT_VERIFY']='0';mtp.append(case('mtp2-unsplit',c,'mtp512'))
    for a in (.05,.10):
        c=copy.deepcopy(mtp_base);c.update(speculative='mtp',draft_depth=2);c['env']['STRATA_GLM_ROUTE_AFFINITY']=str(a);mtp.append(case('mtp2-aff'+str(int(a*100)).zfill(2),c,'mtp512'))
    c=copy.deepcopy(mtp_base);c.update(speculative='mtp',draft_depth=3);c['env']['STRATA_GLM_MTP_CONTINUATION_MARGIN']='2';mtp.append(case('mtp3-margin2',c,'mtp512'))
    ordinary=[];c=copy.deepcopy(cfg);c['env'].pop('STRATA_GLM_TIER_ADAPT');ordinary.append(case('static-tier',c,'ordinary256'))
    for size in (3072,14336):
        c=copy.deepcopy(cfg);c['decode_cache_mib']=size;ordinary.append(case('cache'+str(size),c,'ordinary256',lever='cache3072' if size==3072 else 'largest-admitted-cache'))
    for threads in (8,16):
        c=copy.deepcopy(cfg);c['threads']=threads;ordinary.append(case('workers'+str(threads),c,'ordinary256'))
    pp_base=copy.deepcopy(cfg);pp_base['context']=8192;pp_base['env']['STRATA_GLM_PREFILL_SCRATCH_CAP_MIB']='2048'
    ids=list(map(int,core.PROMPT.read_text().strip().split(',')));prompts={}
    for n in (128,1024,2048,4096):
        p=OUT/('prompt'+str(n)+'.ids');p.write_text(','.join(map(str,(ids*2)[:n]))+'\n');prompts[n]=p
    pp=[]
    for n in (128,1024,2048,4096):
        c=copy.deepcopy(pp_base);c['prefill_batch']=min(n,4096);pp.append(case('pp'+str(n),c,'prefill2g8k','prefill',prompts[n]))
    for groups in (0,8,12):
        c=copy.deepcopy(pp_base);c['prefill_batch']=4096
        if groups:c['env'].update(STRATA_GLM_STAGE_PREFETCH='1',STRATA_GLM_PREFETCH_GROUPS=str(groups))
        pp.append(case('pp4096-prefetch'+str(groups),c,'prefill2g8k','prefill',prompts[4096]))
    blocks=[(mtp,case('base-mtp512',mtp_base,'mtp512',lever='BASE')),(ordinary,case('base-ordinary256',cfg,'ordinary256',lever='BASE')),(pp,case('base-prefill2g8k',pp_base,'prefill2g8k','prefill',prompts[2048],lever='BASE'))]
    core.save(OUT/'case-plan.json',[dict(cohort=control['cohort'],control=control,cases=cases) for cases,control in blocks])
    pending=[]
    for cases,control in blocks:
        for failure in block(cases,control):pending.append((failure,control))
    # Retry only failures from the first sweep, once, with a BASE on each side. Engine/flags unchanged.
    for failure,control in pending:
        retry=copy.deepcopy(failure);retry['name']+='-retry';retry['attempt']=2
        block([retry],control)
    summarize();data=json.loads((OUT/'sweep-records.json').read_text());data['status']='complete';core.save(OUT/'sweep-records.json',data);core.save(CURRENT,data);core.save(OUT/'progress.json',dict(state='complete',records=len(data['records']),failed_attempts=len(data['failed_records'])));print('SWEEP_COMPLETE',flush=True)
if __name__=='__main__':main()
