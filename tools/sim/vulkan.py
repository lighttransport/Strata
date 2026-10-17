"""RX570 Vulkan calibration and phase-cost forecasts, separate from CUDA fits.

Observed profiles are interpolations of their own measurements. Cache and batch
forecasts are explicitly hypotheses until supported by whole-model holdouts.
"""
import argparse
import collections
import json
import hashlib
import math
from pathlib import Path
import re
import statistics

NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'


def median(values):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('positive finite timing observations required')
    return statistics.median(values)


def calibrate(campaign, microbench):
    summary=json.loads((campaign/'summary.json').read_text())
    measurements=[json.loads(s) for s in microbench.read_text().splitlines()]
    profiles={}
    for row in summary['rows']:
        if not row['qualified']:continue
        name=row['profile'];log=(campaign/(name+'-run.log')).read_text()
        decode=[float(x) for x in re.findall(r'DECODE .*tok_s=('+NUMBER+r')(?=\s|$)',log)]
        prefill=[float(x) for x in re.findall(r'PREFILL .*tok_s=('+NUMBER+r')(?=\s|$)',log)]
        phases=collections.defaultdict(list)
        for phase,trial,fields in re.findall(r'VULKAN_PHASE phase=(\w+) trial=(\d+) (.*)',log):
            phases[phase].append({'trial':int(trial),**{k:float(v) for k,v in re.findall(r'(\w+)=([\d.e+\-]+)',fields)}})
        profiles[name]={'warm_step_ms':1000/median(decode[1:]),
                        'prefill_step_ms':1000/median(prefill),
                        'decode_tok_s':decode,'prefill_tok_s':prefill,'phases':dict(phases),
                        'decode_steps':[int(x) for x in re.findall(r'DECODE steps=(\d+)',log)],
                        'executable_sha256':row['executable_sha256'],'config_sha256':row['config_sha256']}
    transfers={r['direction']:r['payload_bytes']/median(r['milliseconds'])/1e6
               for r in measurements if r['kind']=='transfer'}
    groups=collections.defaultdict(list)
    for r in measurements:
        if r['kind']=='projection':groups[(r['type'],r['in'],r['out'],r['tokens'])].append(r)
    projections=[]
    for key,rows in groups.items():
        typ,cols,out,tokens=key;ms=median([r['milliseconds'] for r in rows]);compute=median([r['compute_ms'] for r in rows])
        projections.append({'type':typ,'columns':cols,'rows':out,'tokens':tokens,
                            'median_wall_ms':ms,'median_compute_ms':compute,
                            'wall_tflops':2*cols*out*tokens/(ms*1e9),
                            'effective_weight_gbps':rows[0]['weight_bytes']/(ms*1e6)})
    cpu=collections.defaultdict(list)
    for r in measurements:
        if r['kind']=='cpu_experts':cpu[(r['layer'],r['workers'])].append(r['bytes']/r['milliseconds']/1e6)
    return {'schema':1,'hardware':{'gpu':'RX570 16 GiB','bandwidth_spec_gbps':224,
                                  'wave_size':64,'matrix_cores':False,'integer_dot':False},
            'profiles':profiles,'transfer_payload_gbps':transfers,'projections':projections,
            'cpu_expert_gbps':[{'layer':l,'workers':w,'gbps':median(r)} for (l,w),r in cpu.items()],
            'training_prompt_sha256':summary['prompt_sha256'],
            'limits':['Phase costs are measured on short prompts; long-context extrapolation is unqualified.',
                      'Projection FLOP rates count useful multiply-adds, including quantization overhead.',
                      'Copy rates include submission and synchronization; device copy counts payload, not read+write traffic.',
                      'Modified-card bandwidth must use observations, not the specification as a fitted rate.']}


def estimate(calibration, profile, prompt, generate, cache_hit=0, gpu_expert_ms=0,
             accepted_tokens=1, verify_scale=1, draft_ms=0):
    if profile not in calibration['profiles']:raise ValueError('profile has no qualified measurements')
    if not (prompt>=1 and generate>=1 and 0<=cache_hit<=1 and gpu_expert_ms>=0 and
            1<=accepted_tokens<=4 and verify_scale>=1 and draft_ms>=0 and
            all(math.isfinite(x) for x in (cache_hit,gpu_expert_ms,accepted_tokens,verify_scale,draft_ms))):raise ValueError('invalid forecast inputs')
    p=calibration['profiles'][profile];step=p['warm_step_ms']
    phases=p['phases'].get('decode',[])
    cpu_ms=0
    if phases:
        steps=p.get('decode_steps',[])
        if len(steps)!=len(phases):raise ValueError('phase/step observations differ')
        cpu_ms=median([r['cpu_expert_ms']/n for r,n in zip(phases[1:],steps[1:])])
    if cache_hit and not phases:raise ValueError('cache forecast requires phase measurements')
    # CPU misses and GPU hits overlap. This is an optimistic candidate model,
    # explicitly separated from the fitted observed profile.
    expert_delta=max(cpu_ms*(1-cache_hit),gpu_expert_ms)-cpu_ms
    target_ms=max(0.001,step+expert_delta)
    round_ms=target_ms*verify_scale+draft_ms
    rate=1000*accepted_tokens/round_ms
    return {'profile':profile,'measured_warm_decode_tok_s':1000/step,
            'forecast_decode_tok_s':rate,'forecast_sequential_prefill_tok_s':1000/p['prefill_step_ms'],
            'forecast_sequential_prefill_seconds':prompt*p['prefill_step_ms']/1000,
            'forecast_decode_seconds':generate/rate,
            'hypothetical':bool(cache_hit or accepted_tokens!=1 or verify_scale!=1 or draft_ms),
            'long_context_qualified':False,
            'warning':'Do not use unmeasured acceptance, overlap or cache-hit assumptions as qualification.'}


def calibrate_prefill(prefix):
    """Import completed measured trials; discard the unnumbered warmup."""
    log = prefix.with_suffix('.log')
    guard = json.loads(prefix.with_suffix('.guard.json').read_text())
    result = json.loads(prefix.with_suffix('.result.json').read_text())
    if not (guard.get('passed') and guard.get('performance_qualified') and
            result.get('complete') and result.get('clean') and result.get('exit_code') == 0):
        raise ValueError('requires completed performance-qualified evidence')
    rows, trials = [], []
    pipeline = None
    is_pipeline = False
    for line in log.read_text().splitlines():
        fields = {k: float(v) for k, v in re.findall(r'(\w+)=(' + NUMBER + r')(?=\s|$)', line)}
        if line.startswith('PREFILL_PIPELINE_EXPERT '):
            if 'shared_submission_ms' not in fields:
                raise ValueError('pipeline calibration requires separately measured shared submissions')
            is_pipeline = True
            rows.append(dict(layer=fields['layer'], prepare_ms=fields['prepare_ms'],
                             cpu_work_ms=fields['cpu_work_ms'], upload_ms=0,
                             queued_work_ms=fields['gpu_branch_ms'] + fields['shared_submission_ms'],
                             shared_submission_ms=fields['shared_submission_ms'],
                             gpu_routes=fields['gpu_routes']))
        elif line.startswith('PREFILL_PIPELINE '):
            pipeline = fields
        elif line.startswith('PREFILL_EXPERT_SPLIT '):
            rows.append({k: fields[k] for k in ('layer', 'prepare_ms', 'cpu_work_ms',
                                               'upload_ms', 'queued_work_ms', 'gpu_routes')})
        elif line.startswith('PREFILL trial='):
            layers = rows[-42:]
            if [r['layer'] for r in layers] != list(range(3, 45)):
                raise ValueError('requires all 42 Flash MoE layers in order')
            tokens = int(fields['tokens'])
            if not 1 <= tokens <= 8192 or tokens != fields['tokens']:
                raise ValueError('invalid prefill token count')
            if tokens != 512 and not is_pipeline:
                raise ValueError('legacy calibration is scoped to 512 tokens')
            trial = {'trial': int(fields['trial']), 'tokens': tokens,
                     'wall_ms_per_token': fields['ms'] / tokens, 'layers': layers}
            if is_pipeline:
                if not pipeline or pipeline['tokens'] != tokens or len(rows[-42:]) != 42:
                    raise ValueError('requires one complete logical pipeline window per trial')
                shared = sum(r['shared_submission_ms'] for r in layers)
                trial.update(serial_gpu_ms_per_token=(pipeline['main_submission_ms'] - shared) / tokens,
                             shared_gpu_ms_per_token=shared / tokens)
            trials.append(trial)
            rows = []
        elif line.startswith('VULKAN_FAST_GPU phase=prefill') and trials and not is_pipeline:
            trials[-1]['serial_gpu_ms_per_token'] = fields['attn_segments_ms'] + fields['head_ms']
            trials[-1]['shared_gpu_ms_per_token'] = fields['shared_segments_ms']
        elif line.startswith('VULKAN_FAST_PREFIX phase=prefill') and trials and not is_pipeline:
            trials[-1]['serial_gpu_ms_per_token'] += fields['ms_per_token']
    measured = result['measurements']['prefill']
    if len(trials) != 3 or len(measured) != 3:
        raise ValueError('requires three measured trials')
    for trial, observed in zip(trials, measured):
        if trial['trial'] != observed['trial'] or trial['wall_ms_per_token'] != observed['milliseconds'] / trial['tokens']:
            raise ValueError('log/result trials disagree')
    clocks = {}
    for key in ('pp_dpm_sclk', 'pp_dpm_mclk'):
        counts = collections.Counter()
        for sample in result.get('samples', []):
            match = re.search(r'(\d+)Mhz \*', sample.get('gpu_performance', {}).get(key, ''))
            if match:
                counts[match[1]] += 1
        clocks[key] = dict(counts)
    return {'schema': 1, 'kind': 'batched_prefill_fp32', 'date': '2026-10-16',
            'executor': 'pipeline' if is_pipeline else 'legacy',
            'hardware': {'gpu': 'RX570 modified 16 GiB', 'cpu': 'Ryzen 9 3950X',
                         'ram': '64 GiB DDR4-2666', 'threads': 12,
                         'stream_processors': 2048, 'nominal_fp32_tflops': 5.1,
                         'nominal_bandwidth_gbps': 224, 'memory_bus_bits': 256,
                         'clock_sample_counts_mhz': clocks,
                         'spec_source': 'https://www.amd.com/en/support/downloads/drivers.html/graphics/radeon-600-500-400/radeon-rx-500-series/radeon-rx-570.html'},
            'evidence': {**{k: guard[k] for k in ('executable_sha256', 'config_sha256', 'prompt_sha256')},
                         'files': {prefix.with_suffix(ext).name: hashlib.sha256(prefix.with_suffix(ext).read_bytes()).hexdigest()
                                   for ext in ('.log', '.guard.json', '.result.json')}},
            'trials': trials,
            'limits': ['One untimed warmup excluded; three trials fit, not independent validation.',
                       'Serial GPU = attention + dense prefix + head; shared FFN overlaps expert processing.',
                       'GPU queued work includes host submission, packing and waits, not pure shader time.',
                       'Residual and preparation costs fixed; routes and chunk size fixed.',
                       'Clock samples include load, warmup and decode; not prefill-only averages.',
                       'Pipeline upload costs remain inside the GPU branch; upload-only sensitivity is unavailable.' if is_pipeline else
                       'No clock changes, GPU reruns, decode extrapolation or new HumanEval qualification.']}


def prefill_forecast(calibration, gpu_speedup=1.0, cpu_speedup=1.0, upload_speedup=1.0):
    """Fixed-route, fixed-window sensitivity model; not a quality prediction.

    CPU jobs overlap streamed GPU experts. Shared FFN is submitted concurrently
    and is already reflected in branch wall times. Preserve each layer's max,
    rather than taking max of totals (which hides layer imbalance).
    """
    for value in (gpu_speedup, cpu_speedup, upload_speedup):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('speedups must be positive and finite')
    if calibration.get('kind') != 'batched_prefill_fp32':
        raise ValueError('requires batched FP32 prefill calibration')
    if calibration.get('executor') == 'pipeline' and upload_speedup != 1:
        raise ValueError('pipeline upload cost is not independently measured')
    predictions = []
    for trial in calibration['trials']:
        tokens = trial['tokens']
        def experts(gs, cs, us):
            return sum(r['prepare_ms'] + max(r['cpu_work_ms'] / cs,
                       r['upload_ms'] / us + r['queued_work_ms'] / gs)
                       for r in trial['layers']) / tokens
        gpu = trial['serial_gpu_ms_per_token']
        residual = trial['wall_ms_per_token'] - gpu - experts(1, 1, 1)
        if residual < 0:
            raise ValueError('overlapping phases exceed wall time; recalibrate accounting')
        duration = residual + gpu / gpu_speedup + experts(gpu_speedup, cpu_speedup, upload_speedup)
        predictions.append(1000 / duration)
    return {'predicted_prefill_tok_s': median(predictions),
            'trial_predictions_tok_s': predictions,
            'gpu_speedup': gpu_speedup, 'cpu_speedup': cpu_speedup,
            'upload_speedup': upload_speedup,
            'hypothetical': (gpu_speedup, cpu_speedup, upload_speedup) != (1, 1, 1),
            'scope': 'Same measured prompt, window size and fixed route assignment only; no quality inference.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('calibrate');c.add_argument('--campaign',type=Path,required=True)
    c.add_argument('--microbench',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    e=sub.add_parser('estimate');e.add_argument('--calibration',type=Path,required=True)
    e.add_argument('--profile',default='joined');e.add_argument('--prompt',type=int,default=8192)
    e.add_argument('--generate',type=int,default=512);e.add_argument('--cache-hit',type=float,default=0)
    e.add_argument('--gpu-expert-ms',type=float,default=0);e.add_argument('--accepted-tokens',type=float,default=1)
    e.add_argument('--verify-scale',type=float,default=1);e.add_argument('--draft-ms',type=float,default=0)
    f=sub.add_parser('prefill');f.add_argument('--calibration',type=Path,required=True)
    f.add_argument('--gpu-speedup',type=float,default=1)
    f.add_argument('--cpu-speedup',type=float,default=1)
    f.add_argument('--upload-speedup',type=float,default=1)
    b=sub.add_parser('calibrate-prefill');b.add_argument('--prefix',type=Path,required=True)
    b.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='calibrate':
        result=calibrate(a.campaign,a.microbench)
        with a.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    elif a.command=='calibrate-prefill':
        with a.output.open('x') as f:
            f.write(json.dumps(calibrate_prefill(a.prefix),indent=2)+'\n')
    elif a.command=='prefill':
        result=prefill_forecast(json.loads(a.calibration.read_text()),a.gpu_speedup,a.cpu_speedup,a.upload_speedup)
        print(json.dumps(result,indent=2))
    else:
        result=estimate(json.loads(a.calibration.read_text()),a.profile,a.prompt,a.generate,
                        a.cache_hit,a.gpu_expert_ms,a.accepted_tokens,a.verify_scale,a.draft_ms)
        print(json.dumps(result,indent=2))

if __name__=='__main__':main()
