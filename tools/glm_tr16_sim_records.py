"""Import clean TR16 measurements without fitting on held-out routing outcomes.

Training selection is explicit. Conditional records use measured acceptance and
cache size and must not be presented as workload forecasts.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'sim'))
import calibration
import kernels
from config import RunConfig


def acceptance(result):
    m = result['measurements']
    depth = result['config'].get('draft_depth', 0)
    trials = result['manifest']['timed_trials']
    specs = m.get('speculation', [])[-trials:]
    positions = m.get('mtp_positions', [])[-trials * depth:]
    if not specs or not positions:
        return []
    previous = sum(s['rounds'] for s in specs)
    probabilities = []
    for index in range(depth):
        count = sum(p['accepted'] for p in positions if p['index'] == index)
        probabilities.append(count / previous if previous else 0)
        previous = count
    return probabilities


def import_run(path, training=False, observed=False, probabilities=None):
    r = json.loads(path.read_text())
    if not r.get('clean') or not r.get('complete') or not r.get('quiet') or r.get('guard_exit_code'):
        raise ValueError('refuse incomplete, noisy or failed measurement: ' + str(path))
    c, m = r['config'], r['measurements']
    if c.get('mtp_experts', 'cpu') != 'cpu':
        raise ValueError('separate component model required for GPU draft experts: ' + str(path))
    e = c['env']
    unsupported = ('STRATA_GLM_PREFETCH_NEXT_LAYER', 'STRATA_GLM_PREFETCH_STREAM', 'STRATA_GLM_BATCHED_RESIDENT',
                   'STRATA_GLM_MTP_CONTINUATION_MARGIN', 'STRATA_GLM_MTP_FIRST_MARGIN', 'STRATA_GLM_STABLE_ROUTES',
                   'STRATA_GLM_MTP_ACCEPT_MARGIN', 'STRATA_GLM_MTP_ACCEPT_TOP_K',
                   'STRATA_GLM_MTP_ACCEPT_MAX_SOFT', 'STRATA_GLM_MMVQ_IL', 'STRATA_GLM_MTP_REUSE_DRAFT',
                   'STRATA_GLM_HC_FUSED', 'STRATA_GLM_MLA_KERNELS', 'STRATA_GLM_HEADS_BLOCKS',
                   'STRATA_GLM_TIER_DIRECT', 'STRATA_GLM_DEFER_EXPERTS', 'STRATA_GLM_DEFER_ROUTES',
                   'STRATA_GLM_CANON_COMBINE', 'STRATA_GLM_DENSE_Q4', 'STRATA_GLM_DENSE_Q3', 'STRATA_GLM_ROUTE_MIN_SHARE',
                   'STRATA_GLM_GATE_SKIP', 'STRATA_GLM_TIER_LEARN_UNBIASED', 'STRATA_GLM_MTP_GPU_CACHE', 'STRATA_GLM_CACHE_PROBE',
                   'STRATA_GLM_ROUTE_AFFINITY_PROFILE', 'STRATA_GLM_ROUTE_RENORMALIZE', 'STRATA_GLM_MTP_SYNC_STATE_ONLY',
                   'STRATA_GLM_MTP_UNPRUNED', 'STRATA_GLM_MTP_RECURSIVE_BLEND', 'STRATA_GLM_MTP_REPEAT_GUARD',
                   'STRATA_GLM_MOE_MERGE_GROUPS', 'STRATA_GLM_MTP_LOOKUP_DEPTH', 'STRATA_GLM_MTP_LOOKUP_CHAIN')
    if any(key in e for key in unsupported):
        raise ValueError('separate component model required for experimental execution flags: ' + str(path))
    cfg = RunConfig(pack='q23', quality_reference='q23', threads=c['threads'], context=c['context'],
                    prompt=r['manifest']['prompt_tokens'], generate=r['manifest']['decode_tokens'],
                    prefill_chunk=c['prefill_batch'], speculation=c['speculative'], mtp_depth=c['draft_depth'],
                    decode_cache_mib=c['decode_cache_mib'], gpu_budget_mib=c['gpu_budget_mib'],
                    reserve_mib=float(e['STRATA_GLM_GPU_RESERVE_MIB']),
                    tier_owned_reserve=e.get('STRATA_GLM_TIER_OWNED_RESERVE') == '1',
                    split_verify=e.get('STRATA_GLM_SPLIT_VERIFY') == '1',
                    tier_policy='adaptive' if 'STRATA_GLM_TIER_ADAPT' in e else 'static_prior',
                    affinity=float(e.get('STRATA_GLM_ROUTE_AFFINITY', 0)),
                    prefetch_groups=int(e.get('STRATA_GLM_PREFETCH_GROUPS', 0)),
                    prefill_scratch_mib=float(e.get('STRATA_GLM_PREFILL_SCRATCH_CAP_MIB', 1024)),
                    prefill_mla_f16='STRATA_GLM_MLA_F16' in e,
                    prefill_kda_parts=e.get('STRATA_GLM_PREFILL_KDA_ROW_PARTS') == '4',
                    acceptance_probabilities=probabilities or [], acceptance='mixed')
    if observed:
        cfg.acceptance_probabilities = acceptance(r)
        if m.get('decode_cache'):
            import model
            cfg.decode_cache_mib = m['decode_cache'][-1].get('MiB', m['decode_cache'][-1]['slots'] * model.pack('q23').mean_expert_bytes() / 2**20)
    # Measurements of another prefill kernel must have a distinct fitted model.
    prefill_kind = c.get('prefill_experts', 'mmq')
    common = dict(hw='tr16', config=cfg.__dict__, fit=training, quiet=True,
                  hw_set={'gpu.vram_mib': 15907, 'gpu.desktop_mib': 1683, 'pcie.h2d_gbps': 7.2128,
                          'memory.dram_gbps': 72.35, 'memory.per_core_gbps': 72.35 / 15},
                  provenance=str(path), scope='conditional component check' if observed else 'workload forecast',
                  prefill_kernel=prefill_kind, diagnostic=dict(acceptance_probabilities=acceptance(r),
                  gpu_live=m['gpu_live'], cpu_expert=m['cpu_expert'], decode_cache=m['decode_cache']))
    rows = []
    for kind in ('decode', 'prefill'):
        if kind == 'prefill' and prefill_kind != 'mmq':
            continue
        rows.append(dict(common, name=path.parent.name + '/' + path.stem + '-' + kind, kind=kind,
                         tolerance=.10 if kind == 'decode' else .15,
                         measured=dict(tok_s=m['median_' + kind + '_tok_s'], trials=m[kind])))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--fit', nargs='*', default=[], help='exact parent/stem names for training')
    ap.add_argument('--observed-routing', action='store_true')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--params-output', type=Path)
    args = ap.parse_args()
    available = {p.parent.name + '/' + p.stem for p in args.runs}
    if set(args.fit) - available:
        ap.error('training names do not match input records: ' + repr(sorted(set(args.fit) - available)))
    train = [p for p in args.runs if p.parent.name + '/' + p.stem in args.fit]
    probs = [acceptance(json.loads(p.read_text())) for p in train]
    probs = [x for x in probs if x]
    # Freeze the training-only profile before importing any holdout.
    probabilities = [statistics.mean(p[i] for p in probs if i < len(p)) for i in range(max(map(len, probs), default=0))]
    rows = [row for p in args.runs for row in import_run(p, p in train, args.observed_routing, probabilities)]
    args.output.write_text(json.dumps(dict(records=rows, training_acceptance=probabilities), indent=2) + '\n')
    if args.params_output:
        if args.observed_routing:
            ap.error('fit forecasts first; observed-routing is diagnostic only')
        params = kernels.Params(prefill_stage_gbps=7.2128, cpu_width_penalty=72.35 / 71.43 - 1)
        # CPU bandwidth and dot efficiency stay anchored to independent probes.
        # A bandwidth-bound 2K prefill cannot identify GEMM throughput, and a
        # CPU-bound MTP run cannot identify graph launch cost. Do not fit these
        # terms down to zero merely to compensate for another model error.
        names = []
        if any(r['fit'] and r['kind'] == 'decode' and r['config']['speculation'] == 'none' for r in rows):
            names.append('gpu_launch_scale')
        if any(r['fit'] and r['kind'] == 'prefill' and r['config']['prompt'] >= 8192 for r in rows):
            names.append('prefill_gemm_us')
        score = calibration.objective(rows, params)
        if names:
            params, score = calibration.fit(rows, params, names=names, rounds=8)
        calibration.save_params(params, args.params_output, dict(status='provisional; required holdout errors remain visible',
            records=str(args.output), training=args.fit, training_acceptance=probabilities,
            fitted_fields=names,
            mean_abs_log_error=score, measured_anchors=dict(q23_width1_GB_s=72.35, q23_width2_GB_s=71.43,
                dram_read_GB_s=78.2878, h2d_GB_s=7.2128)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
