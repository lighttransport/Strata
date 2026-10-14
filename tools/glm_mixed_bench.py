#!/usr/bin/env python3
"""Measure fresh single/mixed GLM runs with a common numerical profile.

Run under a cgroup with MemorySwapMax=0. Each repetition starts new processes;
The first engine decode repetition is warmup; --trials selects the timed count.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import statistics
import re
import time
from glm_tr16_calibrate import command, summarize, input_identity, wait_idle
from glm_mixed_run import memory_sample


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stream_health(tokens):
    four=[tuple(tokens[i:i+4]) for i in range(max(0,len(tokens)-3))]
    thirty_two=Counter(tuple(tokens[i:i+32]) for i in range(max(0,len(tokens)-31)))
    cycle=None
    for period in range(1,65):
        repeats=max(3,(64+period-1)//period)
        span=period*repeats
        for start in range(len(tokens)-span+1):
            block=tokens[start:start+period]
            if all(tokens[start+j*period:start+(j+1)*period]==block for j in range(1,repeats)):
                cycle=dict(start=start,period=period,repeats=repeats);break
        if cycle: break
    return dict(tokens=len(tokens),distinct_4gram_fraction=len(set(four))/max(1,len(four)),
                max_32gram_occurrences=max(thirty_two.values(),default=0),consecutive_cycle=cycle)


def performance_summary(results, tokens):
    grouped={}
    for row in results:
        measurement=row.get('measurement',{})
        if not measurement.get('complete') or not measurement.get('clean'): continue
        m=measurement['measurements']
        key=(row['prompt'],row['case'],row['fraction'])
        grouped.setdefault(key,[]).append((m['median_prefill_tok_s'],m['median_decode_tok_s']))
    summary=[]
    for (prompt,case,fraction),values in grouped.items():
        prefill=statistics.median(x[0] for x in values);decode=statistics.median(x[1] for x in values)
        count=len([x for x in re.split(r'[,\s]+',Path(prompt).read_text().strip()) if x])
        summary.append(dict(prompt=prompt,case=case,fraction=fraction,processes=len(values),
                            median_prefill_tok_s=prefill,median_decode_tok_s=decode,
                            estimated_response_seconds=count/prefill+tokens/decode,quality_qualified=False))
    for row in summary:
        singles=[x for x in summary if x['prompt']==row['prompt'] and '-' not in x['case']]
        if len(singles)==2:
            row['prefill_speedup']=row['median_prefill_tok_s']/max(x['median_prefill_tok_s'] for x in singles)
            row['decode_speedup']=row['median_decode_tok_s']/max(x['median_decode_tok_s'] for x in singles)
    return summary


def wait_mixed_idle(timeout, allow_desktop=False):
    if not allow_desktop:
        wait_idle(timeout)
        return []
    from glm_q2_run_guard import gpu_compute_processes
    from glm_q2_coding_bench import process_cpu_snapshot
    desktop_names={'chrome','chromium','chromium-browser','gnome-remote-desktop-daemon'}
    deadline=time.monotonic()+timeout
    quiet=None
    previous=process_cpu_snapshot()
    while True:
        clients=gpu_compute_processes(0)
        desktop=[(pid,Path(name.split()[0]).name) for pid,name in clients
                 if name.split() and Path(name.split()[0]).name in desktop_names]
        compute=[(pid,name) for pid,name in clients if pid not in {x[0] for x in desktop}]
        now=process_cpu_snapshot();seconds=now[0]-previous[0]
        busy=[]
        if seconds>=.5:
            for pid,(name,ticks) in now[1].items():
                if pid==os.getpid() or pid not in previous[1]: continue
                if (ticks-previous[1][pid][1])/os.sysconf('SC_CLK_TCK')/seconds*100>=80:
                    busy.append((pid,name))
        previous=now
        if compute or busy: quiet=None
        elif quiet is None: quiet=time.monotonic()
        elif time.monotonic()-quiet>=10: return desktop
        if time.monotonic()>=deadline:
            raise RuntimeError('mixed benchmark idle timeout: compute='+repr(compute)+', CPU='+repr(busy))
        time.sleep(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--combine', choices=('strict','device'), default='strict')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--model', type=Path, help='override model path in config')
    p.add_argument('--prior', type=Path, help='override calibration prior')
    p.add_argument('--cuda-build', type=Path, default=Path('build-glm-release'))
    p.add_argument('--hip-build', type=Path, default=Path('build-hip-rocm101-tr16'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-desktop-gpu', action='store_true', help='allow and record Chrome/desktop GPU contexts during the idle check')
    p.add_argument('--wait-idle-seconds', type=int, default=600)
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--tokens', type=int, default=512)
    p.add_argument('--trials', type=int, default=1, help='timed decode trials per process, after one warmup')
    p.add_argument('--capture-execution-trace', action='store_true',
                   help='record actual routing events and synchronized layer phases; diagnostic timings only')
    p.add_argument('--capture-prefill-logits', action='store_true',
                   help='save the last prompt position for paired numerical diagnostics, not full quality qualification')
    p.add_argument('--prompts', nargs='+', type=Path, default=[Path('docs/fixtures/glm_tr16_20261009/prompts/2k/holdout-code.ids'), Path('docs/fixtures/glm_tr16_20261009/prompts/8k/holdout-code.ids')])
    p.add_argument('--cases', nargs='+', choices=('cuda','hip','hip-cuda','cuda-hip'), default=['cuda','hip','hip-cuda','cuda-hip'])
    p.add_argument('--fractions', nargs='+', default=['0.25','0.5','0.75','auto'])
    a = p.parse_args()
    if a.output.exists() or a.repetitions < 1 or a.tokens < 1 or not 1 <= a.trials <= 9:
        p.error('use a new output directory and positive repetition/token counts')
    a.output.mkdir(parents=True)
    base = json.loads(a.config.read_text())
    if a.model: base['model']=str(a.model.resolve())
    if a.prior: base.setdefault('env',{})['STRATA_GLM_EXPERT_PRIOR']=str(a.prior.resolve())
    env = os.environ.copy()
    # The same profile is used for every placement; placement cannot choose loss knobs.
    env.update(base.get('env', {}))
    env['LD_LIBRARY_PATH']='/opt/rocm/core-10.1/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    env.pop('STRATA_GLM_MIXED_CHECK_ROWS',None)
    env['STRATA_GLM_CANON']='1'
    env['STRATA_GLM_CANON_COMBINE']=a.combine
    env['STRATA_GLM_STEP_PIPELINE']='1'
    env['STRATA_GLM_GPU_RESERVE_MIB']='1024'
    env['STRATA_GLM_BUDGET_RESERVE_MIB']='1024'
    base['prefill_experts']='f16-batched'
    base['env']={k:v for k,v in env.items() if k.startswith('STRATA_') or k=='LD_LIBRARY_PATH'}
    builds={'cuda':a.cuda_build.resolve(),'hip':a.hip_build.resolve()}
    identity=dict(config=base, source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  source_capture_scope='Working tree at benchmark launch; executable hashes identify the tested builds. Sources may include edits made after compilation.',
                  source_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff'])).hexdigest(),
                  executables={f'{b}/{n}':sha(d/n) for b,d in builds.items() for n in ('strata-glm-decode','strata-glm-gpu-worker')},
                  input_identity=input_identity(base),
                  source_sha256={str(x):sha(x) for x in [Path('CMakeLists.txt'),Path('src/program/glm_decode.cpp'),Path('src/program/glm_gpu_worker.cpp'),Path('src/program/glm_gpu_buffers.hpp'),Path('src/program/glm_resident_executor.hpp'),Path('src/host/glm_mixed.c'),Path('include/strata/host/glm_mixed.h'),Path('include/strata/core/glm_mixed.hpp'),Path('tools/glm_mixed_run.py'),Path(__file__)]},
                  prompts={str(x):sha(x) for x in a.prompts},
                  protocol=f'new process per repetition; one {a.tokens}-token engine warmup then {a.trials} timed {a.tokens}-token decode trials',
                  timed_trials_per_process=a.trials,
                  results=[])
    for prompt in a.prompts:
        for case in a.cases:
            primary=case.split('-')[0]
            fractions=a.fractions if '-' in case else [1.0]
            for requested_fraction in fractions:
                if requested_fraction=='auto':
                    controls=performance_summary(identity['results'],a.tokens)
                    speeds={x['case']:x['median_prefill_tok_s'] for x in controls if x['prompt']==str(prompt) and x['case'] in ('cuda','hip')}
                    if len(speeds)!=2: p.error('auto requires fresh CUDA and HIP controls first')
                    fraction=speeds[primary]/sum(speeds.values())
                else: fraction=float(requested_fraction)
                if not 0 < fraction <= 1:
                    p.error('fractions must be in (0,1]')
                for repetition in range(a.repetitions):
                    c=copy.deepcopy(base)
                    c['exe']=str(builds[primary]/'strata-glm-decode')
                    c['gpu_budget_mib']=15360 if primary=='hip' else 14336
                    cmd=command(c,prompt,a.tokens,a.trials)
                    label='auto' if requested_fraction=='auto' else f'{fraction:g}'
                    tag=f'{prompt.parent.name}-{prompt.stem}-{case}-{label}-{repetition}'
                    memory_report=a.output/(tag+'.memory.json')
                    prefill_logits=a.output/(tag+'.prefill-logits.bin')
                    if a.capture_prefill_logits:
                        cmd.append('--dump-prefill-logits='+str(prefill_logits.resolve()))
                    run_env=env.copy()
                    if a.capture_execution_trace:
                        cmd.append('--routing-trace='+str((a.output/(tag+'.routes.jsonl')).resolve()))
                        run_env['STRATA_GLM_ROUTING_EVENTS']='1'
                        run_env['STRATA_GLM_TASK_TRACE_SYNC']='1'
                        run_env['STRATA_GLM_TASK_TRACE']=str((a.output/(tag+'.phases.jsonl')).resolve())
                        run_env['STRATA_GLM_STEP_TRACE']='2'
                    # Keep the requested 1 GiB reserve on the display NVIDIA card.
                    # This affects staging capacity, not numerical precision.
                    if primary=='cuda': run_env['STRATA_GLM_PREFETCH_GROUPS']=str(base.get('cuda_prefetch_groups',8))
                    if '-' in case:
                        worker=case.split('-')[1]
                        # Models have 16-expert groups; launcher reads this explicit partition.
                        run_env['STRATA_GLM_MIXED_PRIMARY_FRACTION']=str(fraction)
                        cmd=['numactl','--interleave=all','python3',str(Path(__file__).with_name('glm_mixed_run.py').resolve()),
                             '--combine',a.combine,'--primary',c['exe'],'--worker',str(builds[worker]/'strata-glm-gpu-worker'),
                             '--model',c['model'],'--worker-cache-mib',str(c.get('worker_cache_mib',12288)),'--worker-gpu-budget-mib','15360' if worker=='hip' else '14336',
                             '--worker-numa-node','0' if worker=='hip' else '1','--report',str(memory_report.resolve()),'--',*cmd[4:]]
                    desktop=wait_mixed_idle(a.wait_idle_seconds,a.allow_desktop_gpu)
                    start=time.monotonic()
                    with (a.output/(tag+'.stdout')).open('w') as stdout, (a.output/(tag+'.log')).open('w') as stderr:
                        proc=subprocess.Popen(cmd,env=run_env,stdout=stdout,stderr=stderr,start_new_session=True)
                        peak_pss=peak_swap=peak_cgroup=0
                        try:
                            while proc.poll() is None:
                                sample=memory_sample([proc],proportional=False)
                                peak_pss=max(peak_pss,sample['combined_pss_bytes']);peak_swap=max(peak_swap,sample['combined_swap_bytes'])
                                try:
                                    path=Path('/sys/fs/cgroup')/Path(f'/proc/{proc.pid}/cgroup').read_text().strip().split('::',1)[1].lstrip('/')
                                    peak_cgroup=max(peak_cgroup,int((path/'memory.current').read_text()))
                                except (FileNotFoundError,ValueError,IndexError,ProcessLookupError): pass
                                if peak_swap: raise RuntimeError('single runner swapped')
                                time.sleep(1)
                        except BaseException:
                            proc.terminate();proc.wait();raise
                    log=(a.output/(tag+'.log')).read_text();out=(a.output/(tag+'.stdout')).read_text()
                    record=dict(instrumented=a.capture_execution_trace,desktop_gpu_contexts=desktop,case=case,fraction=fraction,requested_fraction=requested_fraction,repetition=repetition,prompt=str(prompt),returncode=proc.returncode,
                                elapsed_seconds=time.monotonic()-start,command=cmd,
                                environment={k:v for k,v in run_env.items() if k.startswith('STRATA_') or k=='LD_LIBRARY_PATH'})
                    if memory_report.exists(): record['memory']=json.loads(memory_report.read_text())
                    else: record['memory']=dict(peak_cgroup_memory_bytes=peak_cgroup,peak_combined_swap_bytes=peak_swap)
                    if a.capture_prefill_logits and prefill_logits.exists():
                        record['prefill_logits']=dict(path=str(prefill_logits),bytes=prefill_logits.stat().st_size,
                                                     sha256=sha(prefill_logits),scope='last prompt position only')
                    if proc.returncode==0:
                        record['measurement']=summarize(log,out,{'exit_code':0,'rejected':False},a.tokens,a.trials)
                        ids=[int(x) for x in out.split()]
                        record['stream_health']=[stream_health(ids[i:i+a.tokens]) for i in range(0,len(ids),a.tokens)]
                        record['loop_rejected']=any(x['consecutive_cycle'] for x in record['stream_health'])
                        if record['loop_rejected']: record['measurement']['clean']=False
                    partition=re.search(r'MIXED_PREFILL primary_groups=(\d+) worker_groups=(\d+)',log)
                    if partition: record['prefill_groups']=dict(primary=int(partition[1]),worker=int(partition[2]))
                    identity['results'].append(record)
                    identity['summary']=performance_summary(identity['results'],a.tokens)
                    (a.output/'results.json').write_text(json.dumps(identity,indent=2)+'\n')
                    print(tag,proc.returncode,record.get('measurement',{}),flush=True)
                    if proc.returncode: raise RuntimeError(f'{tag} failed; inspect saved log')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
