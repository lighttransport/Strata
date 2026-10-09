"""Import guarded B550 runs into simulator records with explicit training selection.

Default records predict workload behavior. --observed-routing is a diagnostic
component check using measured hit/acceptance; it is not out-of-sample prediction.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'sim'))
import model


def import_run(path, fit=False, observed=False, allow_routing_bias=False):
    r=json.loads(path.read_text())
    if not r.get('clean') or not r.get('complete'):
        raise ValueError('refuse unclean/incomplete measurement: '+str(path))
    c=r['config'];m=r['measurements'];env=c['env']
    if c.get('decode_checks') or any(k in env for k in ('STRATA_GLM_STEP_TRACE','STRATA_GLM_ACTIVATION_TRACE')) or any(k.startswith('STRATA_GLM_CHECK_') for k in env):
        raise ValueError('profiling/parity runs are not throughput calibration: '+str(path))
    if (float(env.get('STRATA_GLM_ROUTE_AFFINITY', '0')) and not allow_routing_bias) or 'STRATA_GLM_ROUTE_AFFINITY_PROFILE' in env or c.get('speculative') == 'lookup':
        raise ValueError('routing-bias/lookup runs need a separate calibrated model: '+str(path))
    if float(env.get('STRATA_GLM_CACHE_PROBE', '0')) or env.get('STRATA_GLM_TIER_LEARN_UNBIASED') == '1':
        raise ValueError('cache-probe/unbiased-learning runs need separate calibration: '+str(path))
    cfg=dict(pack='reap50_q23',quality_reference='reap50_q23',threads=c['threads'],context=c['context'],
             prompt=m['prefill'][0]['tokens'],generate=m['decode'][0]['tokens']+(m['decode'][0]['kind']=='single'),prefill_chunk=c['prefill_batch'],
             speculation=c['speculative'],mtp_depth=c['draft_depth'],acceptance='b550_screen',decode_cache_mib=c['decode_cache_mib'],
             gpu_budget_mib=c['gpu_budget_mib'],reserve_mib=float(env['STRATA_GLM_GPU_RESERVE_MIB']),
             split_verify=env.get('STRATA_GLM_SPLIT_VERIFY')=='1',tier_policy='adaptive' if 'STRATA_GLM_TIER_ADAPT' in env else 'static_prior')
    if allow_routing_bias:
        cfg['affinity'] = float(env.get('STRATA_GLM_ROUTE_AFFINITY', '0'))
    cpu=[x for x in m.get('cpu_expert',[]) if x.get('bytes',0)>0]
    # All reported positions are prefix acceptances, not independent probabilities.
    specs=m.get('speculation',[])
    probabilities=[]
    if specs:
        rounds=sum(x['rounds'] for x in specs)
        counts={i:sum(x['accepted'] for x in m['mtp_positions'] if x['index']==i) for i in range(c['draft_depth'])}
        denominator=rounds
        for i in range(c['draft_depth']):
            probabilities.append(counts[i]/denominator if denominator else 0)
            denominator=counts[i]
    diagnostic=dict(cpu_expert=cpu,mtp_timing=m.get('mtp_timing',[]),acceptance_probabilities=probabilities,
                    actual_cache=m.get('decode_cache',[]),gpu_live=m['gpu_live'],minimum_gpu_free_mib=r.get('minimum_gpu_free_mib'))
    if observed:
        cfg['acceptance_probabilities']=probabilities
        if not specs and cpu:
            per_token=statistics.mean(x['bytes']/d['tokens'] for x,d in zip(cpu,m['decode']))
            cfg['measured_hit_share']=1-per_token/model.pack(cfg['pack']).routed_bytes_per_token()
        if m.get('decode_cache'):
            cfg['decode_cache_mib']=m['decode_cache'][0]['slots']*model.pack(cfg['pack']).mean_expert_bytes()/2**20
    common=dict(hw='b550',config=cfg,fit=fit,quiet=True,provenance=str(path),diagnostic=diagnostic,
                scope='conditional component check' if observed else 'workload prediction')
    if 'STRATA_GLM_TIER_RUNTIME_RESERVE_MIB' in env:
        common['runtime_headroom_mib'] = int(env['STRATA_GLM_TIER_RUNTIME_RESERVE_MIB'])
    rows=[]
    for kind in ('decode','prefill'):
        rows.append(dict(common,name=path.stem+'-'+kind,kind=kind,tolerance=.10 if kind=='decode' else .15,
                         measured=dict(tok_s=m['median_'+kind+'_tok_s'],trials=m[kind])))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',nargs='+',type=Path)
    p.add_argument('--fit',nargs='*',default=[],help='exact filename stems selected for training')
    p.add_argument('--observed-routing',action='store_true');p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();rows=[]
    for path in a.runs: rows.extend(import_run(path,path.stem in a.fit,a.observed_routing))
    a.output.write_text(json.dumps(dict(records=rows),indent=2)+'\n')
