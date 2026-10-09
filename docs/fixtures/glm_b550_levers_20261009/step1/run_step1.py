"""Nine-run step 1 matrix using existing flags; no engine changes, fitting or commits."""
import copy,hashlib,json,math,os,re,statistics,subprocess,sys,time
from pathlib import Path

OUT=Path(__file__).resolve().parent
PARENT=OUT.parent
sys.path.insert(0,str(PARENT))
import run_measurements as core
sys.path.insert(0,str(core.ROOT/'tools'))
import glm_low_memory_bench as guard

CURRENT=core.ROOT/'tools/sim/data/measured_b550_levers.json'
PARAMS=core.ROOT/'tools/sim/data/params_b550.json'
PARAMS_SHA='6786b52fe952e5ec3b86af363964c35d80570ba1a3f8e62573e1cb5c563ad9b6'
ROWS=[]
TELEMETRY_CMD=['rocm-smi','--showmeminfo','vram','--showclocks','--showuse']
CASES=[('base15',0,False),('aff06',.06,False),('base15',0,False),
       ('aff08',.08,False),('base15',0,False),('mtp1',0,True),
       ('base15',0,False),('aff08-mtp1',.08,True),('base15',0,False)]

def verify_pins():
    exe=Path(json.loads((PARENT/'base.config.json').read_text())['exe'])
    if hashlib.sha256(exe.read_bytes()).hexdigest()!=core.EXPECTED:
        raise RuntimeError('Pinned decoder changed')
    if hashlib.sha256(PARAMS.read_bytes()).hexdigest()!=PARAMS_SHA:
        raise RuntimeError('Simulator parameters changed; step 1 must not refit')

def phase(path):
    trial=None
    if path.exists():
        for line in path.read_text().splitlines():
            m=re.match(r'DECODE_TRIAL index=(\d+)',line)
            if m:trial=int(m[1])
            elif line.startswith(('DECODE ','SPECULATIVE ')):trial=None
    return trial

def probe(prefix,start):
    before=phase(prefix.with_suffix('.log'))
    if before is None:return None
    p=subprocess.run(TELEMETRY_CMD,capture_output=True,text=True,timeout=15)
    after=phase(prefix.with_suffix('.log'))
    used=re.search(r'VRAM Total Used Memory \(B\):\s*(\d+)',p.stdout)
    total=re.search(r'VRAM Total Memory \(B\):\s*(\d+)',p.stdout)
    device=guard.hip_gpu_device(json.loads(prefix.with_suffix('.config.json').read_text()))
    clock=(device/'pp_dpm_mclk').read_text().strip()
    active=re.search(r'(\d+)Mhz\s*\*',clock)
    return dict(utc=core.now(),seconds=time.monotonic()-start,trial_index=before,
        confirmed_decode=before==after,command=TELEMETRY_CMD,exit_code=p.returncode,
        stdout=p.stdout,stderr=p.stderr,vram_used_mib=int(used[1])/2**20 if used else None,
        vram_total_mib=int(total[1])/2**20 if total else None,sysfs_mclk=clock,
        memory_clock_mhz=int(active[1]) if active else None)

def failure_class(raw,text):
    if raw.get('cgroup_after',{}).get('memory.events',{}).get('oom_kill',0):return 'nonbudget'
    if 'owned weights are swapping' in text.lower():return 'nonbudget'
    rejection=str(raw.get('rejected') or '').lower()
    if any(s in rejection for s in ('total gpu memory','actual gpu free memory')):return 'budget'
    errors=text.lower()
    if any(s in errors for s in ('exceeds gpu budget','exceed gpu allocation','gpu allocation budget',
                                  'hiperroroutofmemory','hip error: out of memory','cuda out of memory',
                                  'insufficient gpu memory','gpu budget exceeded')):return 'budget'
    return 'nonbudget'

def capture(name,budget,case,case_number,attempt=1):
    existing=OUT/(name+'.record.json')
    if existing.exists():return json.loads(existing.read_text())
    verify_pins()
    lever,aff,mtp=case;cfg=json.loads((PARENT/'base.config.json').read_text())
    cfg.update(gpu_budget_mib=budget,decode_cache_mib=12288,threads=12,context=4096,
               prefill_batch=2048,speculative='mtp' if mtp else 'none',draft_depth=1)
    cfg['env'].update(STRATA_GLM_GPU_RESERVE_MIB='512',STRATA_GLM_TIER_RUNTIME_RESERVE_MIB='512',
        STRATA_GLM_STEP_TRACE='1',STRATA_GLM_TIER_ADAPT='1',STRATA_GLM_SPLIT_VERIFY='1',
        STRATA_GLM_ROUTE_AFFINITY=str(aff))
    samples,clock_start=core.quiet_gate(OUT,name);load_start=list(os.getloadavg())
    prefix=OUT/name;core.save(prefix.with_suffix('.config.json'),cfg)
    cmd=[core.PYTHON,str(core.ROOT/'tools/glm_low_memory_bench.py'),str(prefix.with_suffix('.config.json')),
         str(core.PROMPT),'--output',str(prefix),'--ram-gib','60','--tokens','512','--trials','3',
         '--timeout','1200','--gpu-capacity-mib','16304','--gpu-used-limit-mib','15792']
    core.save(OUT/'progress.json',dict(state='running',run=name,budget=budget,command=cmd))
    print('START '+name,flush=True);start=time.monotonic();rocm=[]
    with prefix.with_suffix('.driver.log').open('w') as log:
        p=subprocess.Popen(cmd,cwd=core.ROOT,stdout=log,stderr=subprocess.STDOUT)
        while p.poll() is None:
            try:
                sample=probe(prefix,start)
                if sample:rocm.append(sample)
            except (OSError,ValueError,subprocess.SubprocessError) as e:
                rocm.append(dict(utc=core.now(),seconds=time.monotonic()-start,error=repr(e)))
            core.save(prefix.with_suffix('.rocm.json'),dict(samples=rocm,interval_seconds=5))
            time.sleep(5)
        code=p.returncode
    wall=time.monotonic()-start;load_end=list(os.getloadavg());clock_end=core.clocks()
    raw=json.loads(prefix.with_suffix('.result.json').read_text())
    text=prefix.with_suffix('.log').read_text();m=raw['measurements']
    flat=list(map(int,prefix.with_suffix('.stdout').read_text().split()))
    ids=[flat[i:i+512] for i in range(0,len(flat),512)]
    core.save(prefix.with_suffix('.tokens.json'),dict(trials=ids))
    dec=m.get('decode',[]);pp=m.get('prefill',[])
    good=code==0 and raw.get('complete') and raw.get('clean') and len(dec)==3 and len(ids)==3 and all(len(s)==512 for s in ids)
    confirmed=[s for s in rocm if s.get('confirmed_decode') and s.get('exit_code')==0 and s.get('vram_used_mib') is not None]
    used=[s['vram_used_mib'] for s in confirmed]
    covered=sorted(set(s['trial_index'] for s in confirmed))
    good=good and covered==[0,1,2] and sum(l.startswith('STEP_TRACE ') for l in text.splitlines())==3
    config=dict(pack='reap50_q23',quality_reference='reap50_q23',threads=12,context=4096,prompt=2048,
        generate=512,prefill_chunk=2048,speculation=cfg['speculative'],mtp_depth=1 if mtp else 0,
        acceptance='b550_levers512',decode_cache_mib=12288,gpu_budget_mib=budget,reserve_mib=512,
        split_verify=True,tier_policy='adaptive',affinity=aff,prefetch_groups=0,
        tier_owned_reserve=True,prefill_scratch_mib=1024,prefill_experts='gpu')
    rel='docs/fixtures/glm_b550_levers_20261009/step1/'+name
    row=dict(name=name,kind='decode',hw='b550',config=config,fit=False,quiet=True,status='complete' if good else 'failed',
        lever='BASE15' if lever=='base15' else lever,attempt=attempt,case_number=case_number,
        cohort='step1-b'+str(budget),runtime_headroom_mib=512,provenance=rel+'.result.json',
        tolerance=.10,scope='Step 1 measurements only; no parameter refit',
        measured=dict(tok_s=statistics.mean(t['tokens_per_second'] for t in dec[1:3]) if good else None,
            trials=[dict(t,trial_number=i) for i,t in enumerate(dec[1:3],2)],all_trials=dec,
            prefill_tok_s=statistics.mean(t['tokens_per_second'] for t in pp[1:3]) if len(pp)==3 else None,prefill_trials=pp),
        diagnostic=dict(parsed=m,raw_lines={tag:[l for l in text.splitlines() if l.startswith(tag+' ')] for tag in core.TAGS},
            engine_env=cfg['env'],binary_sha256=core.EXPECTED,params_sha256=PARAMS_SHA,
            load_start=load_start,load_end=load_end,quiet_gate_samples=samples,clock_start=clock_start,clock_end=clock_end,
            exit_code=code,engine_exit_code=raw['exit_code'],wall_seconds=wall,engine_wall_seconds=raw['wall_seconds'],
            cgroup=raw['cgroup_after'],rejected=raw.get('rejected'),interference=raw.get('interference'),
            token_ids_file=rel+'.tokens.json',rocm_decode_samples=rocm,
            rocm_decode_vram_mib=dict(min=min(used),max=max(used),median=statistics.median(used)) if used else None,
            rocm_trials_covered=covered,
            failure_class=None if good else failure_class(raw,text),lossy=aff>0))
    by_trial={};trial=None
    for line in text.splitlines():
        match=re.match(r'DECODE_TRIAL index=(\d+)',line)
        if match:trial=int(match[1]);by_trial[trial]={tag:[] for tag in core.TAGS}
        if trial is not None:
            for tag in core.TAGS:
                if line.startswith(tag+' '):by_trial[trial][tag].append(line)
    row['diagnostic']['decode_trial_lines']=by_trial
    manifest=json.loads(prefix.with_suffix('.manifest.json').read_text())
    manifest['step1']=dict(runtime_headroom_mib=512,gpu_physical_reserve_mib=512,
        rocm_decode_command=TELEMETRY_CMD,rocm_sampling_interval_seconds=5,params_sha256=PARAMS_SHA,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    core.save(prefix.with_suffix('.manifest.json'),manifest)
    core.save(existing,row);print('DONE '+json.dumps(dict(name=name,status=row['status'],tok_s=row['measured']['tok_s'],failure=row['diagnostic']['failure_class'])),flush=True)
    return row

def compare(row,left,right):
    row['neighbour_base_names']=[left['name'],right['name']]
    if all(r['status']=='complete' for r in (left,right)):
        mean=statistics.mean(r['measured']['tok_s'] for r in (left,right))
        row['neighbour_base_tok_s']=mean
        row['ratio_to_base']=row['measured']['tok_s']/mean if row['status']=='complete' else None
    else:row['neighbour_base_tok_s']=row['ratio_to_base']=None
    if all(r['status']=='complete' for r in (row,left,right)):
        streams=[json.loads((OUT/(r['name']+'.tokens.json')).read_text())['trials'] for r in (row,left,right)]
        row['ids_equal_to_base']=all(s==streams[1][0] for ss in streams for s in ss)
        row['control_ids_equal']=all(s==streams[1][0] for ss in streams[1:] for s in ss)
        comparisons={}
        for side,base in zip(('left','right'),streams[1:]):
            comparisons[side]=[dict(trial_number=i+1,first_differing_index=next((j for j,(a,b) in enumerate(zip(s,t)) if a!=b),None),
                                   tokens_matching=sum(a==b for a,b in zip(s,t)),total_tokens=512)
                               for i,(s,t) in enumerate(zip(streams[0],base))]
        row['diagnostic']['ids_against_base15']=comparisons
        if row['lever']=='mtp1' and not row['ids_equal_to_base']:
            row['diagnostic']['mtp1_identity_bug']=True
            row['diagnostic']['quality_note']='BUG: MTP1 without affinity does not satisfy exact BASE15 identity. Control stream equality is reported separately; no engine fix attempted.'
    else:row['ids_equal_to_base']=None
    for i,existing in enumerate(ROWS):
        if existing['name']==row['name']:ROWS[i]=row;break
    core.save(OUT/(row['name']+'.record.json'),row)

def publish(state='running',selected=None):
    own=dict(records=[r for r in ROWS if r['status']=='complete'],failed_records=[r for r in ROWS if r['status']!='complete'],
             status=state,selected_budget_mib=selected,params_sha256=PARAMS_SHA)
    core.save(OUT/'records.json',own)
    # Reuse the proven trace aggregation without touching engine inputs or params.
    subprocess.run([sys.executable,str(PARENT/'finalize_records.py'),'--fixtures',str(OUT),'--records',str(OUT/'records.json')],check=True)
    own=json.loads((OUT/'records.json').read_text())
    enriched={r['name']:r for r in own['records']+own['failed_records']}
    for i,r in enumerate(ROWS):ROWS[i]=enriched[r['name']]
    prior=json.loads((OUT/'prior-records.json').read_text())
    prior['records']+=own['records'];prior['failed_records']+=own['failed_records']
    prior['legacy_sweep_status']='terminated by user before step 1';prior['status']='step1_'+state
    prior['step1']=dict(status=state,selected_budget_mib=selected,fixture='docs/fixtures/glm_b550_levers_20261009/step1',params_sha256=PARAMS_SHA)
    core.save(CURRENT,prior)

def epoch(budget):
    cases=[]
    for i,case in enumerate(CASES,1):
        name=f'step1-b{budget}-{i:02d}-'+case[0]
        r=capture(name,budget,case,i);ROWS.append(r);cases.append(r)
        if i in (3,5,7,9):compare(cases[i-2],cases[i-3],r)
        publish()
        if r['diagnostic']['failure_class']=='budget':return False
    pending={i for i,r in enumerate(cases) if r['status']!='complete'}
    for i in (1,3,5,7):
        if any(cases[j]['status']!='complete' for j in (i-1,i,i+1)):pending.add(i)
    for i in sorted(pending):
        middle=None;triple=[]
        for suffix,case in (('left',CASES[0]),('case',CASES[i]),('right',CASES[0])):
            r=capture(f'step1-b{budget}-retry-{i+1:02d}-{suffix}',budget,case,i+1,2)
            ROWS.append(r);triple.append(r)
            publish()
            if r['diagnostic']['failure_class']=='budget':return False
        compare(triple[1],triple[0],triple[2]);publish()
    return True

def main():
    OUT.mkdir(exist_ok=True)
    if not (OUT/'prior-records.json').exists():core.save(OUT/'prior-records.json',json.loads(CURRENT.read_text()))
    core.save(OUT/'plan.json',dict(cases=CASES,budgets=[15360,14848,14336],reserve_mib=512,cache_request_mib=12288,
        runtime_headroom_mib=512,runtime_note='Previously approved matched 512 MiB runtime headroom; MTP rejects 256.',
        params_sha256=PARAMS_SHA,binary_sha256=core.EXPECTED,simulator_reference=dict(base15=12.3,aff06=15.6,aff08=16.9,aff08_mtp1=17.9,tier_GB=8.6)))
    selected=None
    for budget in (15360,14848,14336):
        if epoch(budget):selected=budget;break
    for r in ROWS:r['step1_selected_budget']=r['config']['gpu_budget_mib']==selected
    publish('complete' if selected else 'all_budgets_refused',selected)
    verify_pins();core.save(OUT/'progress.json',dict(state='complete' if selected else 'all_budgets_refused',selected_budget_mib=selected,runs=len(ROWS)))
    print('STEP1_COMPLETE '+str(selected),flush=True)

if __name__=='__main__':main()
