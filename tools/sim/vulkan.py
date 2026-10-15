"""RX570 Vulkan calibration and phase-cost forecasts, separate from CUDA fits.

Observed profiles are interpolations of their own measurements. Cache and batch
forecasts are explicitly hypotheses until supported by whole-model holdouts.
"""
import argparse
import collections
import json
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


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('calibrate');c.add_argument('--campaign',type=Path,required=True)
    c.add_argument('--microbench',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    e=sub.add_parser('estimate');e.add_argument('--calibration',type=Path,required=True)
    e.add_argument('--profile',default='joined');e.add_argument('--prompt',type=int,default=8192)
    e.add_argument('--generate',type=int,default=512);e.add_argument('--cache-hit',type=float,default=0)
    e.add_argument('--gpu-expert-ms',type=float,default=0);e.add_argument('--accepted-tokens',type=float,default=1)
    e.add_argument('--verify-scale',type=float,default=1);e.add_argument('--draft-ms',type=float,default=0)
    a=p.parse_args()
    if a.command=='calibrate':
        result=calibrate(a.campaign,a.microbench)
        with a.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    else:
        result=estimate(json.loads(a.calibration.read_text()),a.profile,a.prompt,a.generate,
                        a.cache_hit,a.gpu_expert_ms,a.accepted_tokens,a.verify_scale,a.draft_ms)
        print(json.dumps(result,indent=2))

if __name__=='__main__':main()
