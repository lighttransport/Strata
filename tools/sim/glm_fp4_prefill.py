"""Current V100 prefill observations, separate from legacy resource accounting.

Failed gates stay failed. Component timings and instrumented activity cannot
replace a full-model timing or supply fitted hardware capacities.
"""
from .glm_fp4 import activity_phase_scenario, positive


def _comparison(name, control, candidate, *, prefill_only, gate_passed,
                quality_scored=False, quality_failed=False, adjacent=True):
    prefill_key = 'prefill_tok_s' if prefill_only else 'prefill'
    baseline = positive(control[prefill_key], 'control prefill rate')
    rate = positive(candidate[prefill_key], 'candidate prefill rate')
    result = dict(
        name=name, prefill_only=prefill_only, adjacent_pair=adjacent,
        timing_qualified=all(row.get('qualified_prefill' if prefill_only else 'qualified', False)
                             for row in (control, candidate)),
        control_prefill_tok_s=baseline, candidate_prefill_tok_s=rate,
        prefill_ratio=rate / baseline, speed_gate_passed=bool(gate_passed),
        quality_scored=bool(quality_scored), quality_failed=bool(quality_failed),
        selected=False,
        raw_sha256=[control['raw_sha256'], candidate['raw_sha256']])
    if not prefill_only:
        decode = positive(control['decode'], 'control decode rate')
        candidate_decode = positive(candidate['decode'], 'candidate decode rate')
        result.update(control_decode_tok_s=decode, candidate_decode_tok_s=candidate_decode,
                      decode_ratio=candidate_decode / decode)
    return result


def _experiments(profile):
    rows = []
    for pair in profile.get('prefill_workspace_comparisons', []):
        control, candidate = pair['results']
        rows.append(_comparison('fused_operand_workspace', control, candidate,
                                prefill_only=False, gate_passed=pair['eligible_for_quality'],
                                quality_scored=pair['quality_scored']))
    kda = profile.get('prefill_kda_comparison')
    if kda:
        rows.append(_comparison('unordered_kda_rows', *kda['results'], prefill_only=True,
                                gate_passed=kda['eligible_for_quality']))
    combined = profile.get('prefill_combined_full_guard')
    if combined:
        rows.append(_comparison('fused_workspace_and_unordered_kda', combined['control'],
                                combined, prefill_only=False, adjacent=False,
                                gate_passed=combined['eligible_for_quality'],
                                quality_scored=bool(combined['quality_panels']),
                                quality_failed=combined['quality_failed']))
    for key in ('prefill_mla_screen', 'prefill_mla_combined_screen'):
        screen = profile.get(key)
        if screen:
            control, *candidates = screen['results']
            for index, candidate in enumerate(candidates):
                rows.append(_comparison('mla_' + candidate['name'] + '_screen', control,
                                        candidate, prefill_only=True,
                                        adjacent=index == 0,
                                        gate_passed=candidate['eligible_for_full_guard']))
    for key in ('prefill_mla_full_comparison', 'prefill_mla_combined_full_comparison'):
        pair = profile.get(key)
        if pair:
            rows.append(_comparison('mla_' + pair['candidate'] + '_full', *pair['results'],
                                    prefill_only=False,
                                    gate_passed=pair['eligible_for_quality'],
                                    quality_scored=pair['reused_quality_scored'],
                                    quality_failed=pair['quality_failed']))
    return rows


def current_prefill_analysis(profile, target_prefill):
    """Summarize measured working rates without borrowing legacy work counters."""
    working = profile['working_decode_profile']
    rate = positive(working['prefill_tok_s'], 'working prefill rate')
    decode = positive(working['decode_tok_s'], 'working decode rate')
    target = positive(target_prefill, 'prefill target')
    tokens = positive(profile['selected']['configuration']['prompt_tokens'], 'prompt tokens')
    observed_ms, budget_ms = tokens / rate * 1000, tokens / target * 1000
    result = dict(
        fitted=False, performance_anchor=False,
        scope='Recorded working observations and experimental comparisons. Legacy decode/prefill '
              'resource blocks retain their qualified workload; no current resource work is inferred.',
        working=dict(profile=working['path'], profile_sha256=working['sha256'],
                     timing_source=working['timing_source'], input_tokens=tokens,
                     decode_tok_s=decode, prefill_tok_s=rate, prefill_ms=observed_ms,
                     decode_optimization_status=working['decode_optimization_status']),
        target=dict(tok_s=target, budget_ms=budget_ms, met=rate >= target,
                    speedup_required=max(1, target / rate),
                    time_reduction_required_ms=max(0, observed_ms - budget_ms)),
        experiments=_experiments(profile),
        fresh_holdout_scored=working['fresh_holdout_scored'])
    component = profile.get('public_prefill_kernels', {}).get('fused_component')
    if component:
        result['operand_component'] = dict(
            original_over_fused=component['selection_geometric_mean_ratio'],
            separate_over_fused=component['direct_geometric_mean_ratio'],
            validation_sha256=component['validation_sha256'],
            scope='Standalone preparation only; excludes matrix DMA and model scheduling. '
                  'This ratio is not applied to model throughput.')
    functional = profile.get('prefill_fused_functional_coding')
    if functional:
        result['fused_functional_followup'] = dict(
            panels=functional['panels'],
            original_speed_gate_passed=functional['original_speed_gate_passed'],
            scope='Separate reused coding panels; does not change the failed model speed gate.')
    diagnostic = profile.get('prefill_activity_diagnostic')
    if diagnostic:
        if not diagnostic['complete'] or not diagnostic['instrumented'] or diagnostic['performance_anchor']:
            raise ValueError('prefill activity must be a complete instrumented diagnostic')
        devices = [d for d in diagnostic['devices'] if d['phase'] == 'prefill']
        activity = activity_phase_scenario(
            devices, diagnostic['prefill_ms'], profile['hardware']['gpus'], tokens, target)
        activity['retained_kernel_margin_ms'] = (
            budget_ms - activity['slowest_gpu_kernel_union_ms'])
        activity['devices'] = [dict(
            device=d['device'], kernel_union_ms=d['kernel_union_ms'],
            copy_union_ms=d['copy_union_ms'], all_activity_union_ms=d['all_activity_union_ms'],
            idle_ms=diagnostic['prefill_ms'] - d['all_activity_union_ms']) for d in devices]
        activity.update(summary_sha256=diagnostic['summary_sha256'],
                        trace_sha256=diagnostic['trace_sha256'],
                        scope='Single instrumented prefill with launcher residency monitoring. '
                              'Unions account for overlap; these deletion scenarios are not '
                              'hardware rooflines, native residency checks or speed forecasts.')
        result['activity'] = activity
    return result
