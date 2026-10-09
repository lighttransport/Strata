"""Calibrate a frozen B550 lever snapshot; raw measurement rows remain fit:false.

Only the unchanged-flags BASE timing trains aggregate GPU launch/prefill terms.
CPU bandwidth and tier share come from its counters. Other timing rows are held
out on the same frozen input, not independent workload/quality qualifications.
"""
import argparse
import copy
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/'sim'))
import calibration
import decode
import model
from config import RunConfig


def acceptance(record):
    m=record['diagnostic']['parsed'];depth=record['config']['mtp_depth']
    rounds=sum(r['rounds'] for r in m['speculation'][1:3])
    positions=m['mtp_positions'][-2*depth:]
    counts={i:sum(p['accepted'] for p in positions if p['index']==i) for i in range(depth)}
    result=[];denominator=rounds
    for i in range(depth):
        result.append(counts[i]/denominator if denominator else 0.0)
        denominator=counts[i]
    return result


def snapshot(source,conditional=False):
    rows=[]
    for raw in source['records']:
        if raw['status']!='complete':continue
        cfg=copy.deepcopy(raw['config'])
        if cfg['speculation']=='mtp':
            cfg['acceptance']='b550_levers512' if cfg['split_verify'] else 'b550_levers512_unsplit'
            if conditional:cfg['acceptance_probabilities']=acceptance(raw)
        diagnostic=dict(source_record=raw['name'],step_trace=raw['diagnostic'].get('step_trace_status'),
                        token_ids_equal_to_controls=raw.get('ids_equal_to_base'),
                        control_ids_equal=raw.get('control_ids_equal'))
        common=dict(hw='b550',config=cfg,fit=raw['name']=='000-base',quiet=raw['quiet'],
                    provenance=raw['provenance'],runtime_headroom_mib=raw['runtime_headroom_mib'],
                    diagnostic=diagnostic,scope='same-fixture conditional acceptance' if conditional else 'same-fixture forecast; acceptance prior calibrated on MTP counters')
        kinds=[raw['kind']]
        if raw['kind']=='decode':kinds.append('prefill')
        for kind in kinds:
            rate=raw['measured']['tok_s'] if kind==raw['kind'] else raw['measured']['prefill_tok_s']
            trials=raw['measured']['trials'] if kind==raw['kind'] else raw['measured']['prefill_trials'][1:3]
            rows.append(dict(copy.deepcopy(common),name=raw['name']+'-'+kind,kind=kind,
                             measured=dict(tok_s=rate,trials=trials),tolerance=.10 if kind=='decode' else .15))
    return dict(records=rows,source='tools/sim/data/measured_b550_levers.json',
                source_runs=[r['name'] for r in source['records']],
                excluded_failed=[r['name'] for r in source['failed_records']],
                note='Frozen snapshot. Instrumented timing checks are retained separately from the uninstrumented training BASE. Acceptance counters are calibration inputs; these are not independent workload holdouts or quality passes.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('tools/sim/data/measured_b550_levers.json'))
    p.add_argument('--seed',type=Path,default=Path('docs/fixtures/glm_b550_levers_20261009/params-before-user-update.json'))
    p.add_argument('--output',type=Path,default=Path('tools/sim/data/params_b550.json'))
    p.add_argument('--fixtures',type=Path,default=Path('docs/fixtures/glm_b550_levers_20261009'))
    a=p.parse_args();source=json.loads(a.source.read_text())
    if any(r.get('fit') for r in source['records']):
        raise ValueError('Raw measurement rows must remain fit:false; training selection belongs in the snapshot')
    forecast=snapshot(source);conditional=snapshot(source,True)
    forecast['source']=conditional['source']=str(a.source)
    forecast_path=a.source.with_name('measured_b550_levers_forecast.json')
    conditional_path=a.source.with_name('measured_b550_levers_conditional.json')
    for path,data in ((forecast_path,forecast),(conditional_path,conditional)):
        path.write_text(json.dumps(data,indent=2)+'\n')
    gate=next(r for r in source['records'] if r['name']=='000-base')
    if gate['diagnostic']['raw_lines']['STEP_TRACE']:
        raise ValueError('Training BASE must use unchanged, uninstrumented timing flags')
    params=calibration.load_params(a.seed)
    cpu=[r for r in gate['diagnostic']['parsed']['cpu_expert'] if r['bytes']>0][-2:]
    bw=statistics.median(r['bytes']/r['layer_flow_ms']/1e6 for r in cpu)
    machine=calibration.hardware_for(gate)
    params.cpu_bw_scale=bw/machine.memory.dram_gbps
    if params.cpu_bw_scale>1.05:raise ValueError('Counter bandwidth exceeds the independent DRAM anchor')
    params.tier_hit_scale=1.0
    cfg=RunConfig.from_dict(gate['config'])
    predicted=decode.simulate(machine,cfg,params)
    observed=1-statistics.mean(c['bytes']/t['tokens'] for c,t in zip(cpu,gate['measured']['trials']))/model.pack(cfg.pack).routed_bytes_per_token()
    params.tier_hit_scale=observed/predicted.hit_bytes_share
    params,score=calibration.fit(forecast['records'],params,names=['gpu_launch_scale','prefill_gemm_us'],rounds=12)
    calibration.save_params(params,a.output,dict(
        status='provisional 2048-input/512-output B550 calibration; current binary 8c2c7542; timing/quality scope differs',
        training=[r['name'] for r in forecast['records'] if r['fit']],
        holdout=[r['name'] for r in forecast['records'] if not r['fit']],
        records=str(forecast_path),conditional_records=str(conditional_path),raw_records=str(a.source),
        mean_abs_log_error=score,
        measured_anchors=dict(cpu_counter_bandwidth_GB_s=bw,prefill_staging_GB_s=params.prefill_stage_gbps,
                             cpu_width_penalty=params.cpu_width_penalty),
        fitted_fields=['gpu_launch_scale','prefill_gemm_us'],
        acceptance_profile='b550_levers512; pooled MTP1/2 counts, historical third-position estimate; unsplit has separate counts',
        note='Only unchanged-flags BASE timing trains aggregate terms. Other rows check timing on the same input. STEP_TRACE rows remain fit:false in the raw dataset; no independent workload or output-identity qualification is claimed. Affinity retries, worker sweep and prefill ladder are still pending.'))
    report={}
    for label,data in (('forecast',forecast),('conditional_acceptance',conditional)):
        rows=calibration.evaluate(data['records'],params)
        report[label]=dict(rows=rows,mean_abs_error=statistics.mean(abs(r['error']) for r in rows),
                           maximum_abs_error=max(abs(r['error']) for r in rows),
                           decode_mean_abs_error=statistics.mean(abs(r['error']) for r in rows if r['kind']=='decode'),
                           decode_lever_mean_abs_error=statistics.mean(abs(r['error']) for r in rows if r['kind']=='decode' and 'base' not in r['name']),
                           misses=[r['name'] for r in rows if not r['within']])
    (a.fixtures/'simulator-update-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{n:v for n,v in x.items() if n!='rows'} for k,x in report.items()},indent=2))

if __name__=='__main__':main()
