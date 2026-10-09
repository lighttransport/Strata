"""Fit B550's aggregate timing model while anchoring CPU bandwidth to measured counters.

This deliberately does not fit CPU bandwidth against end-to-end throughput: that
can invent bandwidth to compensate for incorrect GPU/routing assumptions.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'sim'))
import calibration
import decode
import hw
import kernels
from config import RunConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,default=Path('tools/sim/data/measured_b550.json'))
    p.add_argument('--output',type=Path,default=Path('tools/sim/data/params_b550.json'))
    p.add_argument('--kernel-summary',type=Path,default=Path('docs/fixtures/glm_b550_sim_20261009/kernel-summary.json'))
    a=p.parse_args();records=calibration.load_records(a.records)
    training=[r for r in records if r['fit'] and r['kind']=='decode']
    if not training:raise ValueError('no training decode records')
    cpu=[x for r in training for x in r['diagnostic']['cpu_expert']]
    params=kernels.Params(cpu_layer_overhead_ms=0,gpu_launch_scale=.5,prefill_stage_gbps=10.53)
    params.cpu_bw_scale=statistics.median(x['bytes']/x['layer_flow_ms']/1e6 for x in cpu)/hw.b550().memory.dram_gbps
    if a.kernel_summary.exists():
        matrix=json.loads(a.kernel_summary.read_text())
        widths={r['width']:r['median_GB_s'] for r in matrix if r['workers']==12}
        if 1 not in widths or len(widths)<2:raise ValueError('kernel anchor needs 12-worker widths 1 and >1')
        params.cpu_width_penalty=max(0.0,statistics.median((1-rate/widths[1])/(width-1) for width,rate in widths.items() if width>1))
    if params.cpu_bw_scale>1.05:
        raise ValueError('logical expert bandwidth exceeds measured DRAM; investigate reuse/accounting before fitting')
    ratios=[]
    for r in training:
        c=RunConfig.from_dict(r['config'])
        if c.speculation!='none':continue
        predicted=decode.simulate(hw.b550(),c,params)
        observed=1-statistics.mean(x['bytes']/d['tokens'] for x,d in zip(r['diagnostic']['cpu_expert'],r['measured']['trials']))/decode.model.pack(c.pack).routed_bytes_per_token()
        ratios.append(observed/predicted.hit_bytes_share)
    params.tier_hit_scale=statistics.mean(ratios)
    params,score=calibration.fit(records,params,names=['gpu_launch_scale','prefill_gemm_us'],rounds=12)
    calibration.save_params(params,a.output,dict(
        status='provisional aggregate calibration; consult record provenance for kernel revision and workload scope',
        training=[r['name'] for r in records if r['fit']],
        holdout=[r['name'] for r in records if not r['fit']],
        records=str(a.records),mean_abs_log_error=score,
        measured_anchors=dict(cpu_counter_bandwidth_GB_s=params.cpu_bw_scale*hw.b550().memory.dram_gbps,
                              prefill_staging_GB_s=params.prefill_stage_gbps,
                              cpu_width_penalty=params.cpu_width_penalty,
                              kernel_matrix=str(a.kernel_summary) if a.kernel_summary.exists() else None),
        fitted_fields=['gpu_launch_scale','prefill_gemm_us'],
        note='The exposed graph-launch fraction and effective GEMM cost are aggregate fit terms, not independently measured kernel rates.'))

if __name__=='__main__':main()
