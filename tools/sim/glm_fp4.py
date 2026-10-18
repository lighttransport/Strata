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
    return dict(schema='strata.glm.native_fp4.analysis.v1', profile=profile['name'], fitted=False,
                rates=dict(cpu_gbps=cpu_gbps, cpu_bandwidth_is_measured=measured_cpu,
                           hbm_gbps_per_gpu=hbm_gbps, tensor_tflops_per_gpu=tensor_tflops,
                           h2d_gbps_per_gpu=links), observations=profile['observations'],
                decode=decode, prefill=prefill, cpu_bandwidth_scenarios=sensitivity,
                findings=profile['findings'], limitations=profile['limitations'])
