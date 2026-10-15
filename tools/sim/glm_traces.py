"""Readers for the GLM runner's instrumented logs and traces (standard library only).

Every reader returns plain dicts with the numbers exactly as printed by the runner; no
derived rate is computed here. Sources: STEP_TRACE / STEP_TRACE_SLOTS (host step scopes
per mailbox slot), MIXED_CLIENT / MIXED_WORKER_SUMMARY (worker protocol counters),
CPU_EXPERT (CPU expert kernel time and bytes), SPECULATIVE / MTP_TIMING / MTP_POSITION
(verify rounds, draft timing, per-position acceptance), PREFILL (prefill trials), the
routing event trace written with STRATA_GLM_ROUTING_EVENTS=1 and the synchronized
TASK_TRACE prefill timeline.
"""
import hashlib
import json
import re
from pathlib import Path

NUMBER = re.compile(r'(\w+)=([-\d.eE+]+)')


def _values(line):
    out = {}
    for key, value in NUMBER.findall(line):
        try:
            out[key] = float(value)
        except ValueError:
            pass
    return out


def read_log(path):
    """Parse one decoder log into per-trial records; the last trial is the timed one."""
    raw = Path(path).read_bytes()
    text = raw.decode(errors='replace')
    trials = []
    current = None
    shared = dict(prefill=[], mixed_client=None, worker_summary=None, cpu_expert=[], mtp_position=[],
                  worker_samples=[], numa_weights=None)

    def new_trial():
        nonlocal current
        current = dict(step_trace=None, slots={}, speculative=None, mtp_timing=None, mtp_approx=None, cpu_expert=None, mtp_position=[])
        trials.append(current)

    for line in text.splitlines():
        # A DECODE_TRIAL line opens a trial; its SPECULATIVE, MTP_*, CPU_EXPERT and STEP_TRACE lines follow it.
        if line.startswith('DECODE_TRIAL '):
            new_trial()
        elif line.startswith('STEP_TRACE '):
            if current is None: new_trial()
            current['step_trace'] = _values(line)
        elif line.startswith('STEP_TRACE_SLOTS '):
            if current is None: new_trial()
            group = int(re.search(r'group=(\d+)', line)[1])
            for layer, kind, wait, service in re.findall(r'(\d+)([km]):(\d+)/(\d+)', line):
                current['slots'][(int(layer), group)] = dict(mixer='kda' if kind == 'k' else 'mla',
                                                             wait_us=int(wait), service_us=int(service))
        elif line.startswith('SPECULATIVE '):
            if current is None: new_trial()
            current['speculative'] = _values(line)
        elif line.startswith('MTP_TIMING '):
            if current is None: new_trial()
            current['mtp_timing'] = _values(line)
        elif line.startswith('MTP_APPROX '):
            if current is None: new_trial()
            current['mtp_approx'] = _values(line)
        elif line.startswith('MTP_POSITION '):
            shared['mtp_position'].append(_values(line))
            if current is not None: current['mtp_position'].append(_values(line))
        elif line.startswith('PREFILL '):
            shared['prefill'].append(_values(line))
        elif line.startswith('MIXED_CLIENT '):
            shared['mixed_client'] = _values(line)
        elif line.startswith('MIXED_WORKER_SUMMARY '):
            shared['worker_summary'] = _values(line)
        elif line.startswith('MIXED_WORKER '):
            shared['worker_samples'].append(_values(line))
        elif line.startswith('CPU_EXPERT '):
            v = _values(line); shared['cpu_expert'].append(v)
            if current is not None and v.get('bytes', 0) > 0: current['cpu_expert'] = v
        elif line.startswith('Q2_NUMA_WEIGHTS '):
            shared['numa_weights'] = _values(line)
    return dict(source=str(path), sha256=hashlib.sha256(raw).hexdigest(), trials=trials, **shared)


def timed_trial(log):
    """The last complete trial with a STEP_TRACE and SPECULATIVE line, plus its CPU_EXPERT row."""
    trials = [t for t in log['trials'] if t.get('step_trace') and t.get('speculative')]
    if not trials:
        raise ValueError('no timed trial with STEP_TRACE and SPECULATIVE in ' + log['source'])
    return dict(trials[-1])


def round_scopes(trial):
    """Per verify-step milliseconds of each host scope, as the STEP_TRACE partition defines them.

    A step is one verify window (STEP_TRACE steps); SPECULATIVE rounds can exceed steps when a
    round proposes nothing. `other` is the SPECULATIVE wall time not covered by the verify step,
    the draft chain and the resync (sampling, lookup, logits readback, loop control).
    """
    s, spec, mtp = trial['step_trace'], trial['speculative'], trial.get('mtp_timing') or {}
    steps = s['steps']
    per = lambda key: s[key] / steps
    scopes = dict(head=per('head_ms'), cpu_service=per('cpu_ms'), between=per('between_ms'), tail=per('tail_ms'),
                  draft=mtp.get('draft_ms', 0) / steps, resync=mtp.get('resync_ms', 0) / steps)
    scopes['other'] = spec['ms'] / steps - sum(scopes.values())
    scopes['round_ms'] = spec['ms'] / steps
    scopes['accepted_per_round'] = spec['generated'] / steps
    scopes['rounds'] = steps
    scopes['speculative_rounds'] = spec['rounds']
    return scopes


def read_routing_events(path):
    """Routing events (STRATA_GLM_ROUTING_EVENTS=1): one JSON object per published slot."""
    events = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if 'routes' not in e or 'layer' not in e:
            raise ValueError('not a routing event trace: ' + path)
        events.append(e)
    return events


def rounds_from_events(events):
    """Group routing events by (request, round); draft-phase events are kept separately."""
    rounds = {}
    for e in events:
        key = (e['request'], e['round'])
        r = rounds.setdefault(key, dict(request=e['request'], round=e['round'], target=[], draft=[]))
        (r['draft'] if e.get('phase') == 'draft' else r['target']).append(e)
    return [rounds[k] for k in sorted(rounds)]


def read_prefill_timeline(path):
    """Synchronized TASK_TRACE batches: per-layer mixer and FFN milliseconds (host clock)."""
    batches, current, layer = [], None, None
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        phase, now = e['phase'], e['unix_ms']
        if phase == 'prefill_batch_begin':
            current = dict(tokens=e['tokens'], start_ms=now, layers=[])
        elif current is None:
            continue
        elif phase == 'layer_begin':
            layer = dict(layer=e['layer'], start_ms=now)
        elif phase == 'mixer_complete' and layer is not None:
            layer['mixer_ms'] = now - layer['start_ms']; layer['mixer_end_ms'] = now
        elif phase == 'ffn_complete' and layer is not None and 'mixer_end_ms' in layer:
            layer['ffn_ms'] = now - layer['mixer_end_ms']
            current['layers'].append(dict(layer=layer['layer'], mixer_ms=layer['mixer_ms'], ffn_ms=layer['ffn_ms']))
            layer = None
        elif phase == 'prefill_batch_complete':
            current['wall_ms'] = now - current['start_ms']
            current['unassigned_ms'] = current['wall_ms'] - sum(x['mixer_ms'] + x['ffn_ms'] for x in current['layers'])
            batches.append(current); current = None
    return batches


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records_from_bench(results_path, expert_bytes=9109504, experts_per_layer=224, moe_layers=42, slots_per_step=84,
                       unique_per_route=0.93, top_k=8):
    """Simulator records from a tools/glm_mixed_bench.py results.json whose runs were instrumented.

    Decode rows carry the STEP_TRACE scopes per verify step, the accepted tokens per step and the
    measured route shares (CPU from CPU_EXPERT bytes, worker from MIXED_CLIENT rows, primary the rest).
    Prefill rows use the timed (last) PREFILL trial. Rows are `split: fit` candidates; the caller
    decides which stay held out.
    """
    results = json.loads(Path(results_path).read_text())
    base = results.get('config', {})
    cache_mib = base.get('decode_cache_mib', 5632)
    worker_mib = base.get('worker_cache_mib', 12288)
    primary_slots = cache_mib * 1048576 / expert_bytes / moe_layers
    worker_slots = worker_mib * 1048576 / expert_bytes / moe_layers
    out = []
    for r in results.get('results', []):
        if r.get('returncode') != 0:
            continue
        log_path = Path(results_path).parent / (Path(r['prompt']).parent.name + '-' + Path(r['prompt']).stem + f"-{r['case']}-{r['fraction']:g}-{r['repetition']}.log")
        if not log_path.exists():
            continue
        log = read_log(log_path)
        instrumented = any(t.get('step_trace') for t in log['trials'])
        if instrumented:
            trial = timed_trial(log)
            scopes = round_scopes(trial)
        else:
            timed = [t for t in log['trials'] if t.get('speculative')]
            if not timed:
                continue
            trial = timed[-1]; spec = trial['speculative']
            scopes = dict(round_ms=spec['ms'] / spec['rounds'], accepted_per_round=spec['generated'] / spec['rounds'], rounds=spec['rounds'],
                          head=None, cpu_service=None, between=None, tail=None, draft=(trial.get('mtp_timing') or {}).get('draft_ms', 0) / spec['rounds'],
                          other=None, resync=(trial.get('mtp_timing') or {}).get('resync_ms', 0) / spec['rounds'])
        case = r['case']
        primary = case.split('-')[0]
        worker = case.split('-')[1] if '-' in case else ''
        tokens = 8192 if '8k' in log_path.name else 2048
        prompt_name = Path(r['prompt']).stem
        name = f"bench-{log_path.stem}"
        # Route shares per slot: unique experts per slot = unique_per_route * width * top_k (width 2 here).
        cpu_bytes_per_step = trial['cpu_expert']['bytes'] / scopes['rounds'] if trial.get('cpu_expert') else 0.0
        per_slot_experts = unique_per_route * 2 * top_k
        cpu_share = cpu_bytes_per_step / slots_per_step / expert_bytes / per_slot_experts
        worker_share = 0.0
        if worker and log.get('mixed_client'):
            mc = log['mixed_client']
            rows_per_request = mc['decode_rows'] / max(1.0, mc['requests'])
            worker_share = rows_per_request / 1.07 / per_slot_experts
        shares = dict(cpu=round(cpu_share, 4), worker=round(worker_share, 4), primary=round(max(0.0, 1 - cpu_share - worker_share), 4))
        out.append(dict(name=name, kind='decode', split='fit', fit=True, measured_tok_s=scopes['accepted_per_round'] * 1000 / scopes['round_ms'],
                        scopes=dict(head=scopes['head'], cpu_service=scopes['cpu_service'], between=scopes['between'], tail=scopes['tail'], draft=scopes['draft']) if instrumented else {},
                        accepted_per_round=scopes['accepted_per_round'], shares=shares,
                        topology=dict(kind='A', primary=primary, worker=worker, groups=[2, 2], context_tokens=tokens),
                        placement=dict(primary_slots=round(primary_slots, 1), worker_slots=round(worker_slots, 1) if worker else 0.0, experts_per_layer=experts_per_layer),
                        source=dict(log=str(log_path), sha256=log['sha256'], speculative_tok_s=trial['speculative']['tok_s'], other_ms=scopes['other'], resync_ms=scopes['resync']),
                        note='Per verify step from STEP_TRACE=2; one timed trial' if instrumented else 'Per SPECULATIVE round (no STEP_TRACE); last timed trial'))
        if log['prefill']:
            p = log['prefill'][-1]
            out.append(dict(name=name, kind='prefill', split='fit', fit=True, measured_tok_s=p['tok_s'],
                            plan=dict(tokens=int(p['tokens']), primary=primary, worker=worker, primary_fraction=float(r['fraction']) if worker else 1.0,
                                      exchange='f32', return_mode='chunks', stream_overlap='mixer'),
                            source=dict(log=str(log_path), sha256=log['sha256'], trials=[x['tok_s'] for x in log['prefill']])))
    return out
