"""Import current REAP50 measurements without weakening historical B550 gates.

Instrumented rows are component diagnostics only. Forecast rows never contain
observed routing or acceptance; those belong in a separate conditional dataset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def import_run(path, hardware, *, fit=False, components=False, observed=False, routing_profile=None):
    r = json.loads(Path(path).read_text())
    if not r.get('complete') or not r.get('clean') or r.get('rejected') or r.get('exit_code'):
        raise ValueError('unclean or incomplete run')
    c, m = r['config'], r['measurements']
    if c.get('speculative', 'none') != 'none' and not (observed or components):
        raise ValueError('speculative import needs measured conditional/component scope; historical acceptance is not a current forecast')
    env = c.get('env', {})
    instrumented = (c.get('profile_layers') or c.get('decode_checks') or
                    any(k.startswith('STRATA_GLM_CHECK_') for k in env) or
                    any(k in env for k in ['STRATA_GLM_STEP_TRACE', 'STRATA_GLM_ACTIVATION_TRACE',
                                          'STRATA_GLM_TASK_TRACE', 'STRATA_GLM_TASK_TRACE_SYNC',
                                          'STRATA_GLM_STATE_TRACE_DIR']))
    if instrumented and not components:
        raise ValueError('instrumented run requires component-only import')
    if fit and (instrumented or components or observed):
        raise ValueError('only uninstrumented forecast controls may train parameters')
    if not m.get('prefill') or not m.get('decode'):
        raise ValueError('missing timing trials')
    prompt = m['prefill'][0]['tokens']
    backend = c.get('prefill_experts', 'mmq')
    threshold = int(env.get('STRATA_GLM_F16_PREFILL_MIN_TOKENS', '0'))
    if backend == 'bf16-batched' and prompt >= threshold:
        if env.get('STRATA_GLM_BF16_WMMA') != '1':
            raise ValueError('library BF16 is outside this calibration')
        backend = 'bf16-wmma'
    elif backend == 'mmq' or (backend == 'bf16-batched' and prompt < threshold):
        backend = 'mmq'
    else:
        raise ValueError('unsupported prefill backend')
    profile = env.get('STRATA_GLM_ROUTE_AFFINITY_PROFILE', '')
    margins={}
    profile_sha=None
    if profile:
        profile_file=Path(routing_profile or profile)
        if not profile_file.is_file():raise ValueError('frozen routing profile unavailable; provide --routing-profile')
        margins=json.loads(profile_file.read_text())['layers']
        profile_sha=hashlib.sha256(profile_file.read_bytes()).hexdigest()
    cfg = dict(pack='reap50_q22', quality_reference='reap50_q22',
               threads=c['threads'], context=c['context'], prompt=prompt,
               generate=m['decode'][0]['tokens'] + int(m['decode'][0]['kind'] == 'single'),
               prefill_chunk=c['prefill_batch'], prefill_backend=backend,
               speculation=c.get('speculative', 'none'),
               mtp_depth=c.get('lookup_depth', 3) if c.get('speculative') == 'lookup' else c.get('draft_depth', 1),
               gpu_draft_experts=False, skip_anchor=env.get('STRATA_GLM_MTP_SKIP_ANCHOR') == '1',
               decode_cache_mib=c['decode_cache_mib'], gpu_budget_mib=c['gpu_budget_mib'],
               reserve_mib=float(env.get('STRATA_GLM_GPU_RESERVE_MIB', '512')),
               prefill_scratch_mib=float(env.get('STRATA_GLM_PREFILL_SCRATCH_CAP_MIB', '1024')),
               split_verify=env.get('STRATA_GLM_SPLIT_VERIFY') == '1',
               tier_policy='adaptive' if env.get('STRATA_GLM_TIER_ADAPT') == '1' else 'static_prior',
               affinity=float(env.get('STRATA_GLM_ROUTE_AFFINITY', '0')),
               expert_deferral=int(env.get('STRATA_GLM_DEFER_EXPERTS', '0')) > 0,
               deferral_share=int(env.get('STRATA_GLM_DEFER_ROUTES', '0')) / 8,
               routing_profile=Path(profile).name if profile else '',
               routing_affinity_layers=margins,
               pipeline=env.get('STRATA_GLM_STEP_PIPELINE') == '1',
               decode_graphs=bool(c.get('decode_graphs')),
               ordered_kda=env.get('STRATA_GLM_KDA_ORDERED_ROWS') == '1',
               mhc_save_fused=env.get('STRATA_GLM_MHC_SAVE_FUSED') == '1',
               prefill_mla_f16=env.get('STRATA_GLM_MLA_F16') == '1',
               prefill_kda_parts=(prompt >= int(env.get('STRATA_GLM_PREFILL_KDA_MIN_TOKENS', '0')) and
                                  int(env.get('STRATA_GLM_PREFILL_KDA_ROW_PARTS', '1')) > 1),
               tier_owned_reserve=env.get('STRATA_GLM_TIER_OWNED_RESERVE') == '1')
    cpu = [x for x in m.get('cpu_expert', []) if x.get('bytes', 0)]
    if observed:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'sim'))
        import model
        if cfg['speculation'] == 'none' and cpu:
            consumed = statistics.mean(x['bytes'] / t['tokens'] for x, t in zip(cpu, m['decode']))
            share = 1 - consumed / model.pack(cfg['pack']).routed_bytes_per_token()
            if not 0 <= share <= 1:
                raise ValueError('invalid measured routed-byte share')
            cfg['measured_hit_share'] = share
        if cfg['speculation'] in ('mtp','lookup','external'):
            positions=m.get('mtp_positions' if cfg['speculation']=='mtp' else 'lookup_positions',[])
            probabilities=[]
            for index in range(cfg['mtp_depth']):
                entries=[x for x in positions if x['index']==index]
                opportunities=sum(x.get('opportunities',0) for x in entries)
                if not opportunities:
                    raise ValueError('missing measured conditional opportunities at position '+str(index))
                probabilities.append(sum(x['accepted'] for x in entries)/opportunities)
            cfg['acceptance_probabilities']=probabilities
            if cfg['speculation'] in ('lookup','external'):
                timing=m.get('draft_timing',[])
                calls=sum(x.get('proposal_calls',0) for x in timing)
                if not calls:raise ValueError('missing proposal call timing')
                latency=sum(x['draft_ms'] for x in timing)/calls
                cfg['proposal_probability']=sum(x['rounds'] for x in m['speculation'])/calls
                if cfg['speculation']=='lookup':cfg['lookup_draft_ms']=latency
                else:
                    cfg['external_draft_ms']=latency
                    modes=set(m.get('external_verification',[]))
                    if len(modes)!=1 or not modes <= {'batched','sequential'}:
                        raise ValueError('missing external verification strategy')
                    cfg['external_sequential']=modes=={'sequential'}
                    ready=m.get('external_ready',[])
                    if not ready:raise ValueError('missing warmed external VRAM')
                    cfg['external_draft_mib']=max(x['vram_mib'] for x in ready)
    common = dict(hw=str(hardware), config=cfg, fit=fit, quiet=True,
                  provenance=str(path), source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                  routing_profile_sha256=profile_sha,
                  scope='component-only' if components else 'conditional' if observed else 'forecast',
                  diagnostic={k: m.get(k, []) for k in ['step_trace', 'cpu_expert', 'decode_cache', 'gpu_live', 'mtp_timing', 'mtp_positions', 'lookup_positions', 'draft_timing', 'external_prepare', 'external_ready', 'external_verification']})
    rows = []
    for kind in ['decode', 'prefill']:
        row = dict(common, name=Path(path).stem+'-'+kind, kind=kind,
                   tolerance=.10 if kind == 'decode' else .15,
                   measured=dict(tok_s=m['median_'+kind+'_tok_s'], trials=m[kind]))
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--hardware', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--fit', nargs='*', default=[])
    p.add_argument('--components', action='store_true')
    p.add_argument('--observed', action='store_true')
    p.add_argument('--routing-profile',type=Path,help='local frozen profile if the original record path is unavailable')
    a = p.parse_args()
    records = []
    for path in a.runs:
        records.extend(import_run(path, a.hardware, fit=path.stem in a.fit,
                                  components=a.components, observed=a.observed,routing_profile=a.routing_profile))
    a.output.write_text(json.dumps(dict(records=records), indent=2)+'\n')


if __name__ == '__main__':
    main()
