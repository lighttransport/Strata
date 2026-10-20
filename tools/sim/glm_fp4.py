"""Native FP4 observations and conditional resource roofs, separate from GGUF fits.

Bytes are decimal bytes, bandwidth is GB/s, tensor throughput is TFLOP/s, and
times are seconds. These are optimistic bounds on a recorded workload, not a
fitted predictor. In particular, discarded routing *mass* is not a work fraction.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = Path(__file__).resolve().parent / 'data/glm_fp4_v100.json'


def positive(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def speculative_phase_scenario(request, target_decode):
    """Measured phase deletion bound and an explicitly conditional no-waste case."""
    advances = positive(request['advances'], 'advances')
    elapsed = positive(request['decode_ms'], 'decode_ms')
    verify = positive(request['verify_ms'], 'verify_ms')
    target = positive(target_decode, 'target_decode')
    if verify > elapsed:
        raise ValueError('verification exceeds measured decode interval')
    other = elapsed - verify
    budget = advances * 1000 / target
    remaining = budget - other
    result = dict(name=request['name'], fitted=False, observed_tok_s=advances*1000/elapsed,
                  verification_fraction=verify/elapsed,
                  same_work_ceiling_without_nonverification_tok_s=advances*1000/verify,
                  target=dict(tok_s=target, decode_budget_ms=budget,
                              feasible_with_fixed_other_work=remaining>0,
                              verification_speedup_needed=verify/remaining if remaining>0 else None),
                  scope='Phase deletion for the same measured proposal stream; not a hardware roof.')
    if all(k in request for k in ('verification_rows','rounds','draft_ms','restore_ms','block_size')):
        rows = positive(request['verification_rows'], 'verification_rows')
        rounds = positive(request['rounds'], 'rounds')
        block = positive(request['block_size'], 'block_size')
        if int(block)!=block or not 2<=block<=8:
            raise ValueError('native FP4 block size must be 2..8')
        draft, restore = request['draft_ms'], request['restore_ms']
        if not all(math.isfinite(x) and x>=0 for x in (draft,restore)) or verify+draft+restore>elapsed:
            raise ValueError('invalid measured speculative phases')
        fixed = elapsed-verify-draft-restore
        ideal_rounds = math.ceil(advances/block)
        ideal_ms = verify*advances/rows + draft*ideal_rounds/rounds + fixed
        result['perfect_acceptance_scenario'] = dict(tok_s=advances*1000/ideal_ms,
            assumed_verification_rows=advances, assumed_rounds=ideal_rounds,
            assumptions='Same measured cost per target row and draft call; no restore; remaining overhead fixed. '
                        'Acceptance and routing change the workload, so this is not a throughput forecast or bound.')
    return result


def load_profile(path=DEFAULT_PROFILE):
    profile = json.loads(Path(path).read_text())
    if profile.get('schema') != 'strata.glm.native_fp4.v1':
        raise ValueError('expected native FP4 profile, not a GGUF/RPC kernel table')
    for source in profile['sources']:
        target = (ROOT / source['path']).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():
            raise ValueError('profile source must be a repository file')
        if hashlib.sha256(target.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('source checksum differs: ' + source['path'])
    return profile


def counter_delta(before, after, fields=('uploaded_bytes', 'misses')):
    """Deduplicate repeated shutdown reports; never count them as extra devices."""
    def unique(rows):
        result = {}
        for row in rows:
            device = row['device']
            if device in result and result[device] != row:
                raise ValueError('conflicting duplicate native counter')
            result[device] = row
        return result
    before, after = unique(before), unique(after)
    if before.keys() != after.keys():
        raise ValueError('native counter devices changed')
    out = []
    for device in sorted(before):
        row = {'device': device}
        for key in fields:
            row[key] = after[device][key] - before[device][key]
            if row[key] < 0:
                raise ValueError('native cumulative counter decreased')
        out.append(row)
    return out


def resource_roof(work, capacities, tokens):
    """Independent resources may overlap: lower time bound is max, not sum."""
    positive(tokens, 'tokens')
    seconds = {}
    for name, amount in work.items():
        if not math.isfinite(amount) or amount < 0:
            raise ValueError('work must be finite and nonnegative')
        seconds[name] = amount / positive(capacities[name], name)
    elapsed = max(seconds.values(), default=0)
    positive(elapsed, 'resource time')
    return dict(seconds=elapsed, tok_s=tokens / elapsed,
                limiting_resource=max(seconds, key=seconds.get), resource_seconds=seconds)


def _median_request(profile):
    rows = profile['selected']['requests']
    if len(rows) < 3:
        raise ValueError('at least three qualified timed requests required')
    for row in rows:
        trial = row['trial']
        if trial['major_faults'] or trial['generated_tokens'] != trial['requested_tokens']:
            raise ValueError('faulted or incomplete trial is not a performance anchor')
        if trial['first_eos_position_1based'] is not None:
            raise ValueError('fixed-length performance anchor contains EOS')
        for key in ('decode_ms', 'prefill_ms'):
            positive(trial[key], key)
        positive(row['cpu_expert_groups'], 'CPU expert groups')
    if len({r['trial']['tokens_sha256'] for r in rows}) != 1:
        raise ValueError('timed outputs differ')
    # Select one actual request, retaining covariance of its timing/work fields.
    return sorted(rows, key=lambda r: r['trial']['decode_ms'])[len(rows) // 2]


def activity_phase_scenario(devices, wall_ms, gpus, tokens, target):
    """Retain measured GPU unions; never add overlapping kernel/copy times."""
    positive(wall_ms, 'activity phase time')
    positive(tokens, 'activity tokens')
    positive(target, 'activity target')
    if sorted(d['device'] for d in devices) != list(range(gpus)):
        raise ValueError('activity must cover each GPU exactly once per phase')
    for device in devices:
        kernel, copy, active = (device[k] for k in (
            'kernel_union_ms', 'copy_union_ms', 'all_activity_union_ms'))
        if not all(math.isfinite(v) and v >= 0 for v in (kernel, copy, active)):
            raise ValueError('invalid activity duration')
        if max(kernel, copy) > active + .01 or active > wall_ms + .01:
            raise ValueError('activity unions exceed their enclosing interval')
    kernel_ms = positive(max(d['kernel_union_ms'] for d in devices), 'kernel union')
    active_ms = positive(max(d['all_activity_union_ms'] for d in devices), 'activity union')
    target_ms = tokens / target * 1000
    idle_ms = max(0, wall_ms - active_ms)
    return dict(
        observed_instrumented_ms=wall_ms,
        slowest_gpu_kernel_union_ms=kernel_ms,
        slowest_gpu_activity_union_ms=active_ms,
        conditional_kernel_only_ceiling_tok_s=tokens * 1000 / kernel_ms,
        conditional_ceiling_if_all_gpu_idle_removed_tok_s=tokens * 1000 / active_ms,
        target=dict(tok_s=target, budget_ms=target_ms,
                    retained_kernel_duration_speedup_required=max(1, kernel_ms / target_ms),
                    idle_reduction_fraction_at_retained_activity=(
                        max(0, (wall_ms - target_ms) / idle_ms)
                        if idle_ms and target_ms >= active_ms else None)))


def activity_scenarios(activity, gpus, target_decode, target_prefill):
    """Conditional retained-duration scenarios, separate from qualified anchors.

    A union is already corrected for overlap. Copy and kernel unions must never
    be added, and removing idle time does not remove CPU dependencies in reality.
    """
    capture = activity['capture']
    if not capture['instrumented'] or capture['performance_anchor']:
        raise ValueError('CUDA activity must be explicitly diagnostic')
    result = dict(scope='Single instrumented request; retained GPU durations are conditional scenarios, '
                        'not hardware rooflines or throughput predictions.',
                  limitations=activity['limitations'])
    for phase, tokens, target in [('decode', capture['decode_advances'], target_decode),
                                  ('prefill', capture['input_tokens'], target_prefill)]:
        devices = [d for d in activity['devices'] if d['phase'] == phase]
        result[phase] = activity_phase_scenario(
            devices, activity[phase + '_ms'], gpus, tokens, target)
    return result


def analyze(profile, *, cpu_gbps=None, hbm_gbps=None, tensor_tflops=None,
            h2d_gbps=None, target_decode=50.0, target_prefill=800.0):
    hw, geometry = profile['hardware'], profile['geometry']
    row = _median_request(profile)
    measured_cpu = cpu_gbps is None and 'cpu_stream_gbps' in hw
    if cpu_gbps is None:
        cpu_gbps = (hw['cpu_stream_gbps'] if measured_cpu else
                    hw['sockets'] * hw['cpu_memory_channels_per_socket'] * hw['cpu_max_supported_mt_s'] * 8 / 1000)
    hbm_gbps = hw['hbm_gbps_per_gpu'] if hbm_gbps is None else hbm_gbps
    tensor_tflops = hw['tensor_tflops_per_gpu'] if tensor_tflops is None else tensor_tflops
    links = hw['h2d_gbps_per_gpu'] if h2d_gbps is None else h2d_gbps
    for name, value in [('CPU bandwidth', cpu_gbps), ('HBM bandwidth', hbm_gbps),
                        ('tensor throughput', tensor_tflops), ('decode target', target_decode),
                        ('prefill target', target_prefill)]:
        positive(value, name)
    if len(links) != hw['gpus']:
        raise ValueError('one H2D rate per GPU required')
    for rate in links:
        positive(rate, 'H2D bandwidth')
    if geometry['moe_layers'] * geometry['experts'] * 3 != profile['census']['main_experts']['matrices']:
        raise ValueError('expert census does not match geometry')

    tokens = row['trial']['generated_tokens'] - 1  # prefill supplies the first token
    decode_seconds = row['trial']['decode_ms'] / 1000
    cpu_bytes = row['cpu_expert_groups'] * profile['expert_code_bytes']
    # Only mandatory FFN/head matrices enter this deliberately loose HBM bound.
    # Attention, KDA and mHC traffic are not guessed from checkpoint size.
    draft_tokens = row['speculative']['proposed'] - row['lookup']['proposed']
    primary_bytes = (row['speculative']['rounds'] * profile['mandatory_main_matrix_bytes']
                     + draft_tokens * profile['head_bytes'])
    decode_roof = resource_roof({'cpu_code_stream': cpu_bytes, 'primary_mandatory_matrices': primary_bytes},
                               {'cpu_code_stream': cpu_gbps * 1e9,
                                'primary_mandatory_matrices': hbm_gbps * 1e9}, tokens)
    timing = row['timing']
    verify_seconds = timing['verify_ms'] / 1000
    nonverify_seconds = decode_seconds - verify_seconds
    if nonverify_seconds < 0 or draft_tokens < 0:
        raise ValueError('inconsistent phase timings or proposal counters')
    target_verify = tokens / target_decode - nonverify_seconds
    phases = {k.removesuffix('_ms'): v / 1000 for k, v in timing.items()}
    phases['other'] = decode_seconds - sum(phases.values())
    if phases['other'] < -0.001:
        raise ValueError('phase timers exceed measured decode')
    decode = dict(observed_tok_s=tokens / decode_seconds, observed_seconds=decode_seconds,
                  work=dict(cpu_expert_groups=row['cpu_expert_groups'], cpu_code_bytes=cpu_bytes,
                            cpu_code_GB_per_output=cpu_bytes / tokens / 1e9,
                            main_target_rows=row['main_target_rows'],
                            cpu_retained_route_rows=row['cpu_retained_route_rows'],
                            output_advances=tokens, rounds=row['speculative']['rounds'],
                            advances_per_round=tokens / row['speculative']['rounds']),
                  conditional_streaming_roof=decode_roof,
                  observed_fraction_of_roof=(tokens / decode_seconds) / decode_roof['tok_s'],
                  observed_exceeds_roof_assumptions=decode_seconds < decode_roof['seconds'],
                  phases_seconds=phases, verification_fraction=verify_seconds / decode_seconds,
                  ceiling_if_only_nonverification_is_removed_tok_s=tokens / verify_seconds,
                  target=dict(tok_s=target_decode, required_verification_speedup=(verify_seconds / target_verify
                              if target_verify > 0 else None), feasible_with_fixed_nonverification=target_verify > 0))

    prompt = profile['selected']['configuration']['prompt_tokens']
    prefill_seconds = statistics.median(r['trial']['prefill_ms'] for r in profile['selected']['requests']) / 1000
    expert_elements = 3 * geometry['hidden'] * geometry['expert_ff']
    routed_elements = ((geometry['moe_layers'] + geometry['draft_moe_layers'])
                       * geometry['top_k'] * expert_elements)
    fixed_elements = profile['census']['main_fixed']['elements'] + profile['census']['draft_fixed']['elements']
    # Ideal projection FLOPs only. Fixed work stays on the primary, experts are
    # evenly split for this optimistic compute-only comparison.
    primary_flops = 2 * prompt * (fixed_elements + routed_elements / hw['gpus'])
    total_flops = 2 * prompt * (fixed_elements + routed_elements)
    compute_seconds = max(primary_flops / (tensor_tflops * 1e12),
                          total_flops / (hw['gpus'] * tensor_tflops * 1e12))
    proxy = profile['prefill_traffic_proxy']
    deltas = counter_delta(proxy['native_counters_before'], proxy['native_counters_after'])
    if (len(deltas) != len(proxy['devices']) or len(deltas) != hw['gpus']
            or len(profile['expert_transfer_sizes']) != hw['gpus']):
        raise ValueError('traffic proxy must cover every GPU')
    for upload, delta in zip(proxy['devices'], deltas):
        if (upload['device'] != delta['device'] or upload['expert_uploads'] != delta['misses']
                or upload['expanded_gpu_bytes'] != delta['uploaded_bytes']):
            raise ValueError('traffic proxy disagrees with cumulative counter delta')
    wire = {}
    roofs = {}
    for extent in ('min', 'mean', 'max'):
        amounts = []
        for upload, size in zip(proxy['devices'], profile['expert_transfer_sizes']):
            if upload['device'] != size['device']:
                raise ValueError('upload/codec device order differs')
            if upload['expanded_gpu_bytes'] != upload['expert_uploads'] * profile['expert_gpu_bytes']:
                raise ValueError('expanded allocation bytes disagree with upload count')
            amounts.append(upload['expert_uploads'] * size['source_bytes_' + extent])
        wire[extent] = amounts
        work = {f'gpu{i}_h2d': amount for i, amount in enumerate(amounts)}
        capacities = {f'gpu{i}_h2d': rate * 1e9 for i, rate in enumerate(links)}
        # The concurrent DMA probe also supplies a shared aggregate ceiling.
        work['aggregate_h2d'] = sum(amounts)
        aggregate = hw['h2d_gbps_aggregate'] if h2d_gbps is None else sum(links)
        capacities['aggregate_h2d'] = min(sum(links), aggregate) * 1e9
        work['ideal_projection_compute'] = compute_seconds
        capacities['ideal_projection_compute'] = 1
        roofs[extent] = resource_roof(work, capacities, prompt)
    roof = roofs['mean']
    prefill = dict(observed_tok_s=prompt / prefill_seconds, observed_seconds=prefill_seconds,
                   traffic_scope=proxy['scope'], source_wire_bytes=wire,
                   expanded_allocation_bytes=[d['expanded_gpu_bytes'] for d in proxy['devices']],
                   conditional_proxy_roof=roof,
                   conditional_proxy_roof_tok_s_interval=[roofs['max']['tok_s'], roofs['min']['tok_s']],
                   ideal_projection_only_tok_s=prompt / compute_seconds,
                   observed_fraction_of_proxy_roof=roof['seconds'] / prefill_seconds,
                   observed_exceeds_roof_assumptions=prefill_seconds < roof['seconds'],
                   elapsed_above_proxy_floor_seconds=prefill_seconds - roof['seconds'],
                   target=dict(tok_s=target_prefill, measured_speedup_required=target_prefill / (prompt / prefill_seconds),
                               primary_h2d_gbps_required=wire['mean'][0] * target_prefill / prompt / 1e9,
                               below_conditional_proxy_roof=target_prefill <= roof['tok_s']))
    # Keep measured matrix DMA separate from the frozen proxy evidence.
    # The caller must bracket one request, including its preparation/reset phase.
    measured = profile.get('prefill_transfer')
    if measured is not None:
        exact = counter_delta(measured['before'], measured['after'], ('prefill_bytes',))
        if [d['device'] for d in exact] != [d['device'] for d in proxy['devices']]:
            raise ValueError('measured transfer must cover the same GPU devices')
        amounts = [d['prefill_bytes'] for d in exact]
        work = {f'gpu{i}_h2d': value for i, value in enumerate(amounts)}
        capacity = {f'gpu{i}_h2d': rate * 1e9 for i, rate in enumerate(links)}
        work.update(aggregate_h2d=sum(amounts), ideal_projection_compute=compute_seconds)
        capacity.update(aggregate_h2d=min(sum(links), aggregate) * 1e9,
                        ideal_projection_compute=1)
        exact_roof = resource_roof(work, capacity, prompt)
        prefill['measured_matrix_transfer'] = dict(
            scope=measured['scope'], source_wire_bytes=amounts,
            conditional_roof=exact_roof,
            observed_fraction_of_roof=exact_roof['seconds'] / prefill_seconds,
            target_below_conditional_roof=target_prefill <= exact_roof['tok_s'],
            target_primary_h2d_gbps_required=amounts[0] * target_prefill / prompt / 1e9,
            elapsed_above_floor_seconds=prefill_seconds - exact_roof['seconds'],
            limitations='Matrix DMA only; excludes activation traffic and reconstruction. '
                        'Independent resources are optimistically overlapped.')
    sensitivity = []
    for bandwidth in (80, 120, 180, cpu_gbps):
        candidate = resource_roof({'cpu_code_stream': cpu_bytes, 'primary_mandatory_matrices': primary_bytes},
                                  {'cpu_code_stream': bandwidth * 1e9,
                                   'primary_mandatory_matrices': hbm_gbps * 1e9}, tokens)
        sensitivity.append(dict(cpu_gbps=bandwidth, conditional_roof_tok_s=candidate['tok_s']))
    result = dict(schema='strata.glm.native_fp4.analysis.v1', profile=profile['name'], fitted=False,
                rates=dict(cpu_gbps=cpu_gbps, cpu_bandwidth_is_measured=measured_cpu,
                           hbm_gbps_per_gpu=hbm_gbps, tensor_tflops_per_gpu=tensor_tflops,
                           h2d_gbps_per_gpu=links), observations=profile['observations'],
                decode=decode, prefill=prefill, cpu_bandwidth_scenarios=sensitivity,
                findings=profile['findings'], limitations=profile['limitations'])
    if 'experimental_speculative_requests' in profile:
        result['experimental_decode'] = [speculative_phase_scenario(row,target_decode)
                                         for row in profile['experimental_speculative_requests']]
    if 'diagnostic_activity' in profile:
        result['diagnostic_activity'] = activity_scenarios(
            profile['diagnostic_activity'], hw['gpus'], target_decode, target_prefill)
    if 'working_decode_profile' in profile:
        from .glm_fp4_prefill import current_prefill_analysis
        result['current_prefill'] = current_prefill_analysis(profile, target_prefill)
    return result
