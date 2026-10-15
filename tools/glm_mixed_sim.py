#!/usr/bin/env python3
"""Measured mixed-GPU latency budgets and conditional redesign simulation.

This is a sensitivity model, not a fitted performance predictor. Measured MTP
phase times are disjoint; CPU layer-flow time is nested inside verification.
Never add worker waits or CPU flow to the phase totals. Future speedups and
critical-path exposure are explicit assumptions, not inferred measurements.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def decode_model(record):
    m = record['measurement']['measurements']
    d, t = m['decode'][-1], m['mtp_timing'][-1]
    phases = dict(draft=t['draft_ms'], verify=t['verify_ms'],
                  resync=t['resync_ms'], prepare=t['prepare_ms'])
    residual = d['milliseconds'] - sum(phases.values())
    if residual < -1:
        raise ValueError('MTP phases exceed wall time; incompatible timer scopes')
    return dict(tokens=d['tokens'], wall_ms=d['milliseconds'], phases_ms=phases,
                residual_ms=max(0, residual),
                cpu_flow_ms=m['cpu_expert'][-1]['layer_flow_ms'])


def decode_seconds(model, verify_speedup=1, draft_speedup=1,
                   cpu_speedup=1, cpu_exposure=0):
    if min(verify_speedup, draft_speedup, cpu_speedup) < 1:
        raise ValueError('speedups must be at least one')
    if not 0 <= cpu_exposure <= 1:
        raise ValueError('exposure must be in [0,1]')
    p = model['phases_ms']
    # Remove only the assumed exposed share of CPU work, then accelerate the
    # remaining verification path. The scopes are deliberately not summed.
    verify = p['verify'] - min(p['verify'], model['cpu_flow_ms']) * cpu_exposure * (1 - 1 / cpu_speedup)
    return (verify / verify_speedup + p['draft'] / draft_speedup +
            p['resync'] + p['prepare'] + model['residual_ms']) / 1000


def prefill_seconds(tokens, nv_rate, amd_rate, nv_fraction, speedup=1,
                    coordination_ms=0, serial_fraction=0):
    """Whole-pipeline work partition with explicit serial work and join cost.

    This assumes proportional work at the measured single-device rates. It
    requires a new scheduler: current expert-only sharding does not satisfy it.
    serial_fraction runs on AMD; the remaining work can run concurrently.
    """
    if not 0 <= nv_fraction <= 1 or not 0 <= serial_fraction <= 1:
        raise ValueError('invalid partition')
    if min(tokens, nv_rate, amd_rate, speedup) <= 0 or coordination_ms < 0:
        raise ValueError('invalid timing inputs')
    serial = tokens * serial_fraction / amd_rate
    parallel = tokens * (1 - serial_fraction) * max(nv_fraction / nv_rate, (1 - nv_fraction) / amd_rate)
    return (serial + parallel) / speedup + coordination_ms / 1000


def read_prefill_timeline(path):
    """Read synchronized TASK_TRACE batches; retain layer and FFN boundaries.

    Times include host submission, synchronization and trace I/O. Millisecond
    timestamps cannot identify tiny kernels or overlap within a phase.
    """
    batches, current, layer = [], None, None
    for line in path.read_text().splitlines():
        e = json.loads(line)
        phase, now = e['phase'], e['unix_ms']
        if phase == 'prefill_batch_begin':
            if current is not None:
                raise ValueError('incomplete or interleaved prefill trace')
            current = dict(tokens=e['tokens'], start_ms=now, layers=[])
        elif current is not None:
            if phase == 'layer_begin':
                layer = dict(layer=e['layer'], start_ms=now)
            elif phase == 'mixer_complete' and layer is not None:
                if layer['layer'] != e['layer']:
                    raise ValueError('mismatched mixer layer')
                layer['mixer_ms'] = now - layer['start_ms']
                layer['mixer_end_ms'] = now
            elif phase == 'ffn_complete' and layer is not None:
                if layer['layer'] != e['layer'] or 'mixer_end_ms' not in layer:
                    raise ValueError('mismatched FFN layer')
                layer['ffn_ms'] = now - layer['mixer_end_ms']
                if min(layer['mixer_ms'], layer['ffn_ms']) < 0:
                    raise ValueError('nonmonotonic trace clock')
                current['layers'].append(layer)
                layer = None
            elif phase == 'prefill_batch_complete':
                current['wall_ms'] = now - current['start_ms']
                accounted = sum(x['mixer_ms'] + x['ffn_ms'] for x in current['layers'])
                current['unassigned_ms'] = current['wall_ms'] - accounted
                if layer is not None or current['unassigned_ms'] < 0:
                    raise ValueError('invalid prefill phase accounting')
                batches.append(current)
                current = None
    if current is not None:
        raise ValueError('trace still contains an unfinished batch')
    return batches


def layer_pipeline_seconds(nv_batch_seconds, amd_batch_seconds, microbatches, nv_layer_fraction):
    """Optimistic two-stage prompt pipeline with measured batch costs.

    Weight streaming repeats in each measured microbatch cost. Whole-layer
    costs scale by layer fraction; state and boundary transfers are free.
    """
    if min(nv_batch_seconds, amd_batch_seconds) <= 0 or microbatches < 1 or int(microbatches) != microbatches:
        raise ValueError('invalid pipeline timing inputs')
    if not 0 <= nv_layer_fraction <= 1:
        raise ValueError('invalid layer fraction')
    a, b = nv_batch_seconds*nv_layer_fraction, amd_batch_seconds*(1-nv_layer_fraction)
    return a+b+(microbatches-1)*max(a,b)


def layer_partition(primary, worker, primary_fraction, communication_ms=0,
                    mixer_speedup=1, expert_speedup=1):
    """Dependency-preserving per-layer join model for expert-only partition.

    The mixer and first three dense FFNs stay primary. Each MoE layer must
    join both devices before the next layer can begin. Full FFN costs scale
    linearly with expert share: optimistic because shared FFN work, staging
    overlap and GEMM efficiency do not actually scale that way.
    """
    from sim.heterogeneous import partition_prefill
    return partition_prefill(primary, worker, primary_fraction, communication_ms,
                             mixer_speedup, expert_speedup)


def simulate(mixed, singles, target_decode, target_prefill):
    records = [r for r in mixed['results'] if r['returncode'] == 0]
    if len(records) != 1:
        raise ValueError('select a source containing exactly one mixed timing record')
    model = decode_model(records[0])
    rates = {}
    for r in singles['results']:
        if '/8k/' in r['prompt'] and r['case'] in ('cuda', 'hip') and r['returncode'] == 0:
            rates[r['case']] = r['measurement']['measurements']['median_prefill_tok_s']
    nv, amd = rates['cuda'], rates['hip']
    fraction = nv / (nv + amd)
    decode_rows = []
    for verify, draft, cpu, exposure in itertools.product(
            [1, 1.25, 1.5, 1.75, 2], [1, 1.5, 2], [1, 2, 4], [0, .5, 1]):
        rate = model['tokens'] / decode_seconds(model, verify, draft, cpu, exposure)
        decode_rows.append(dict(verify_speedup=verify, draft_speedup=draft,
                                cpu_speedup=cpu, cpu_exposure=exposure,
                                tok_s=rate, meets_target=rate >= target_decode))
    prefill_rows = []
    for speed, serial, overhead in itertools.product([1, 1.1, 1.25, 1.5], [0, .1, .25, .5], [0, 250, 500, 1000]):
        seconds = prefill_seconds(8192, nv, amd, fraction, speed, overhead, serial)
        prefill_rows.append(dict(kernel_speedup=speed, serial_fraction=serial,
                                 coordination_ms=overhead, nv_fraction=fraction,
                                 tok_s=8192 / seconds, meets_target=8192 / seconds >= target_prefill))
    budget_ms = model['tokens'] / target_decode * 1000
    fixed_ms = model['wall_ms'] - model['phases_ms']['verify']
    return dict(scope='Conditional sensitivity simulation; no quality or achievable-speed claim',
                targets=dict(decode=target_decode, prefill=target_prefill, humaneval=152),
                decode_calibration=model,
                decode_budget=dict(target_ms=budget_ms, required_saving_ms=model['wall_ms']-budget_ms,
                                   verify_only_required_speedup=(model['phases_ms']['verify'] / (budget_ms-fixed_ms)
                                                                if budget_ms > fixed_ms else None)),
                prefill_calibration=dict(single_rates=rates, ideal_sum=nv+amd,
                                         balanced_nv_fraction=fraction,
                                         ideal_coordination_budget_ms=1000*(8192/target_prefill-8192/(nv+amd))),
                limitations=['Single timing screens; different build/config generations.',
                             'CPU exposure is unknown and swept, not fitted.',
                             'Whole-pipeline prefill partition is hypothetical; dependency, memory, and transport feasibility must be measured.',
                             'No quality prediction; final corpus and HumanEval gates remain mandatory.'],
                decode_scenarios=decode_rows, prefill_scenarios=prefill_rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mixed', type=Path, required=True)
    p.add_argument('--singles', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--target-decode', type=float, default=50)
    p.add_argument('--target-prefill', type=float, default=1000)
    a = p.parse_args()
    if min(a.target_decode, a.target_prefill) <= 0:
        p.error('targets must be positive')
    mixed, mid = read(a.mixed)
    singles, sid = read(a.singles)
    result = simulate(mixed, singles, a.target_decode, a.target_prefill)
    result['sources'] = [mid, sid]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['decode_budget', 'prefill_calibration']}, indent=2))


if __name__ == '__main__':
    main()
