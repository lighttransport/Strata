"""Measurement-only B550 runner. Engine, simulator parameters and Git stay untouched."""
import argparse, copy, datetime, hashlib, json, os, re, statistics, subprocess, sys, time
from pathlib import Path
ROOT=Path('/home/syoyo/work/Strata-b550-calibration-15b787b5')
PYTHON='/home/syoyo/work/Strata/.venv-glm-hip/bin/python'
PROMPT=Path('/home/syoyo/work/Strata/build-hip-glm-v11/single-2048.ids')
EXPECTED='8c2c754264f5c327997826cbe468b953888c3cbb9d8c165833074a1d8e1df98f'
TAGS=('SPECULATIVE','DECODE','PREFILL','STEP_TRACE','STEP_TRACE_SLOTS','CPU_EXPERT','DECODE_CACHE','GPU_LIVE','MTP_TIMING','MTP_POSITION')

def save(path,value):
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2,default=str)+'\n');temporary.replace(path)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def call(argv):
    p=subprocess.run(argv,capture_output=True,text=True,timeout=30)
    return dict(command=argv,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
def clocks():
    # AMD SMI wakes runtime-suspended RDNA4 for read-only telemetry. No clock settings change.
    amd=call(['amd-smi','metric','--clock']);rocm=call(['rocm-smi','--showclocks','--showuse'])
    values=re.findall(r'MEM_0:.*?CLK:\s*([0-9.]+)\s*MHz',amd['stdout'],re.S)
    sysfs={}
    for p in Path('/sys/class/drm').glob('card[0-9]*/device/pp_dpm_mclk'):
        try:sysfs[str(p)]=p.read_text()
        except OSError as e:sysfs[str(p)]=repr(e)
    return dict(utc=now(),amd_smi=amd,rocm_smi=rocm,sysfs=sysfs,memory_clock_mhz=float(values[0]) if values else None)
def quiet_gate(out,label):
    samples=[];since=None;last_report=0
    while True:
        current=time.monotonic();load=list(os.getloadavg());samples.append(dict(utc=now(),load=load))
        if load[0]<3:
            if since is None:since=current
        else:since=None
        elapsed=current-since if since is not None else 0
        save(out/'progress.json',dict(state='quiet_gate',run=label,load=load,quiet_seconds=elapsed))
        if current-last_report>=30:
            print(json.dumps(dict(state='quiet_gate',run=label,load_1=load[0],quiet_seconds=round(elapsed))),flush=True);last_report=current
        if elapsed>=180:break
        time.sleep(10)
    c=clocks()
    for _ in range(3):
        if c['memory_clock_mhz'] is not None and c['memory_clock_mhz']>100:break
        time.sleep(2);c=clocks()
    if c['memory_clock_mhz'] is None or c['memory_clock_mhz']<=100:
        save(out/(label+'.clock-failure.json'),dict(clock=c,quiet_samples=samples))
        raise RuntimeError('active/read-only telemetry cannot verify MCLK above 100 MHz; no clock reset attempted')
    if os.getloadavg()[0]>=3:return quiet_gate(out,label)
    return samples,c

def run_decode(out,label,cfg,lever='BASE',attempt=1):
    exe=Path(cfg['exe']);digest=hashlib.sha256(exe.read_bytes()).hexdigest()
    if digest!=EXPECTED:raise RuntimeError('pinned decoder changed: '+digest)
    samples,start_clock=quiet_gate(out,label);start_load=list(os.getloadavg())
    config=out/(label+'.config.json');save(config,cfg);prefix=out/label
    cmd=[PYTHON,str(ROOT/'tools/glm_low_memory_bench.py'),str(config),str(PROMPT),'--output',str(prefix),'--ram-gib','60','--tokens','512','--trials','3','--timeout','1200','--gpu-capacity-mib','16304','--gpu-used-limit-mib','14256']
    save(out/'progress.json',dict(state='running',run=label,started_utc=now(),command=cmd))
    print('START '+label,flush=True);start=time.monotonic()
    with prefix.with_suffix('.driver.log').open('w') as log:p=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    end_load=list(os.getloadavg());wall=time.monotonic()-start;end_clock=clocks()
    result_path=prefix.with_suffix('.result.json');raw=json.loads(result_path.read_text()) if result_path.exists() else {}
    log_path=prefix.with_suffix('.log');text=log_path.read_text() if log_path.exists() else ''
    ids_path=prefix.with_suffix('.stdout');ids=list(map(int,ids_path.read_text().split())) if ids_path.exists() else []
    streams=[ids[i:i+512] for i in range(0,len(ids),512)];save(prefix.with_suffix('.tokens.json'),dict(trials=streams))
    m=raw.get('measurements',{});dec=m.get('decode',[]);pp=m.get('prefill',[]);selected=dec[1:3]
    good=p.returncode==0 and raw.get('complete') and raw.get('clean') and len(dec)==3 and len(streams)==3 and all(len(s)==512 for s in streams)
    env=cfg['env'];cfg_sim=dict(pack='reap50_q23',quality_reference='reap50_q23',threads=cfg['threads'],context=cfg['context'],prompt=2048,generate=512,prefill_chunk=cfg['prefill_batch'],speculation=cfg['speculative'],mtp_depth=cfg['draft_depth'],acceptance='b550_screen',decode_cache_mib=cfg['decode_cache_mib'],gpu_budget_mib=cfg['gpu_budget_mib'],reserve_mib=float(env['STRATA_GLM_GPU_RESERVE_MIB']),split_verify=env.get('STRATA_GLM_SPLIT_VERIFY')=='1',tier_policy='adaptive' if env.get('STRATA_GLM_TIER_ADAPT')=='1' else 'static',affinity=float(env.get('STRATA_GLM_ROUTE_AFFINITY','0')),prefetch_groups=0,tier_owned_reserve=True,prefill_scratch_mib=1024,adaptive_window=float(env.get('STRATA_GLM_MTP_CONTINUATION_MARGIN','0'))>0)
    row=dict(name=label,kind='decode',hw='b550',config=cfg_sim,fit=False,quiet=True,provenance='docs/fixtures/glm_b550_levers_20261009/'+label+'.result.json',status='complete' if good else 'failed',lever=lever,attempt=attempt,measured=dict(tok_s=statistics.mean(r['tokens_per_second'] for r in selected) if good else None,trials=selected,all_trials=dec,prefill_tok_s=statistics.mean(r['tokens_per_second'] for r in pp[1:3]) if len(pp)==3 else None,prefill_trials=pp),diagnostic=dict(raw_lines={tag:[line for line in text.splitlines() if line.startswith(tag+' ')] for tag in TAGS},step_trace_note='Not enabled in reference BASE flags; no profiling flag silently added.',parsed=m,load_start=start_load,load_end=end_load,quiet_gate_samples=samples,clock_start=start_clock,clock_end=end_clock,exit_code=p.returncode,engine_exit_code=raw.get('exit_code'),wall_seconds=wall,engine_wall_seconds=raw.get('wall_seconds'),cgroup=raw.get('cgroup_after'),rejected=raw.get('rejected'),interference=raw.get('interference'),binary_sha256=digest,token_ids_file='docs/fixtures/glm_b550_levers_20261009/'+label+'.tokens.json',repeated_ids_equal=bool(streams) and all(s==streams[0] for s in streams)))
    if 'STRATA_GLM_TIER_ADAPT' not in env and env.get('STRATA_GLM_EXPERT_PRIOR'):
        row['config']['tier_policy']='static_prior'
    save(prefix.with_suffix('.record.json'),row)
    print('DONE '+json.dumps({k:row[k] for k in ['name','status','measured']}),flush=True)
    return row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output.resolve();out.mkdir(exist_ok=True,parents=True)
    cfg=json.loads((out/'base.config.json').read_text())
    row=run_decode(out,'000-base',cfg)
    records=dict(records=[row],status='baseline_gate_failed',reference=dict(decode_tok_s=10.46455,prefill_tok_s=230.93))
    if row['status']=='complete':
        ratios=dict(decode=row['measured']['tok_s']/10.46455,prefill=row['measured']['prefill_tok_s']/230.93)
        records['baseline_ratios']=ratios
        if all(.9<=v<=1.1 for v in ratios.values()):records['status']='baseline_gate_passed'
    save(out/'records.json',records);save(out/'progress.json',dict(state=records['status'],run='000-base',measured=row['measured']))
    print('BASE_GATE '+records['status'],flush=True)
    return 0 if records['status']=='baseline_gate_passed' else 2
if __name__=='__main__':raise SystemExit(main())
