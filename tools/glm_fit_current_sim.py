"""Fit current REAP50 prefix controls using training CPU-byte and timing anchors.

Held-out counters never enter the fit. Per-layer routing margins come from the
frozen input profile. The resident GPU bandwidth is an effective fitted rate,
not a standalone hardware-bandwidth measurement. Independent coding validation
is required before using the fit to rank engine changes.
"""
import argparse
import dataclasses
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'sim'))
import calibration
import hw


def cpu_bytes(record):
    samples=[r for r in record['diagnostic'].get('cpu_expert',[]) if r.get('bytes',0)]
    trials=record['measured']['trials']
    if len(samples)!=len(trials) or any(t['kind']!='single' or not t['tokens'] for t in trials):
        raise ValueError('training CPU-byte counters do not match ordinary decode trials')
    return sum(r['bytes'] for r in samples)/sum(t['tokens'] for t in trials)/1e9


def fit(records,params,hardware):
    train=[r for r in records if r.get('fit') and r['kind']=='decode']
    if not train:raise ValueError('no training decode rows')
    if any(r.get('scope')!='forecast' or r['config'].get('speculation','none')!='none'
           or r['config'].get('measured_hit_share',-1)>=0 for r in train):
        raise ValueError('only clean forecast controls may train this model')
    targets=[cpu_bytes(r) for r in train]
    def predict(r,p,bw):
        copy=dict(r,hw_set={**r.get('hw_set',{}),'gpu.tier_gbps':bw})
        return calibration.predict(copy,p)
    def byte_error(scale):
        trial=dataclasses.replace(params,tier_hit_scale=scale)
        return statistics.mean(abs(math.log(predict(r,trial,hardware.gpu.tier_gbps)[1].cpu_gb_per_token/y))
                               for r,y in zip(train,targets))
    scale=min((.98+i*.0001 for i in range(601)),key=byte_error)
    params=dataclasses.replace(params,tier_hit_scale=scale)
    def timing_error(bw):
        return statistics.mean(abs(math.log(predict(r,params,bw)[0]/r['measured']['tok_s'])) for r in train)
    bandwidth=min((200+i*.5 for i in range(881) if 200+i*.5<=hardware.gpu.bandwidth_gbps),key=timing_error)
    hardware.gpu.tier_gbps=bandwidth
    # Prefill has separate MMQ and WMMA costs. This still uses training rows only.
    modified=[dict(r,hw_set={**r.get('hw_set',{}),'gpu.tier_gbps':bandwidth}) for r in records]
    params,error=calibration.fit(modified,params,names=['prefill_gemm_us','prefill_wmma_gemm_us','prefill_layer_fixed_ms'],rounds=8,log=lambda _:None)
    details=dict(training=[r['name'] for r in records if r.get('fit')],mean_abs_log_error=error,
        cpu_anchor_error=byte_error(scale),tier_hit_scale=scale,effective_resident_gpu_gbps=bandwidth,
        timing_anchor_error=timing_error(bandwidth),
        status='provisional prefix-control calibration; independent coding holdouts blocked by GPU loss',
        fitted_fields=['tier_hit_scale','gpu.tier_gbps','prefill_gemm_us','prefill_wmma_gemm_us','prefill_layer_fixed_ms'])
    return params,hardware,details


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['records','hardware','seed-params','output-params','output-hardware','report']:
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    records=calibration.load_records(a.records)
    params,hardware,details=fit(records,calibration.load_params(a.seed_params),hw.load(str(a.hardware)))
    details.update(records=str(a.records),hardware=str(a.output_hardware),
        note='CPU byte anchors and effective resident GPU rate use training controls only; component costs are not independently identified.')
    a.output_hardware.write_text(json.dumps(hardware.to_dict(),indent=2)+'\n')
    calibration.save_params(params,a.output_params,details)
    a.report.write_text(json.dumps(calibration.evaluate(records,params),indent=2)+'\n')


if __name__=='__main__':main()
