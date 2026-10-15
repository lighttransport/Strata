"""End-to-end GLM-5.3-Flash decode and prefill model for the TR16 mixed runner.

Built on `heterogeneous.schedule`: every kernel, transfer and host step is an Event with
resource demands; nothing is derived from advertised FLOPS. Per-slot costs come from a
kernel table (`strata.glm.kernels.v1`) whose free constants are fitted to instrumented runs
(`calibrate`) and checked on held-out runs (`validate`). The decode model reproduces the
runner's single-stream enqueue order (group A then group B per MoE layer, each group's
wait-add blocking the stream until its CPU slot is served), so the STEP_TRACE scopes
(head, cpu_service, between, tail) are read off the schedule, not assumed.

Scope: throughput only. No quality (HumanEval, corpus) is predicted here.
"""
import copy
import dataclasses
import json
import math
from pathlib import Path

from sim.heterogeneous import Event, schedule, transfer_events

DATA = Path(__file__).resolve().parent / 'data'
LAYERS = 45
DENSE_LAYERS = (0, 1, 2)
MOE_LAYERS = tuple(range(3, 45))
MLA_LAYERS = frozenset((3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43))
MTP_LAYER = 45
TOP_K = 8
HIDDEN = 4096
# Q2_K gate/up + Q3_K down; layers 11, 12 and 44 keep IQ formats (tools/sim/model.py pack q23/reap224_q23).
EXPERT_BYTES = {l: 9109504 for l in MOE_LAYERS}
EXPERT_BYTES.update({11: 10878976, 12: 9961472, 44: 9961472})
ROW_BYTES = HIDDEN * 4
RESIDUAL_BYTES = 4 * HIDDEN * 4  # mHC: 4 streams of hidden floats per token


def mixer_kind(layer):
    return 'dense' if layer in DENSE_LAYERS else ('mla' if layer in MLA_LAYERS else 'kda')


# ----------------------------------------------------------------------------- tables
def get(table, path, default=None):
    node = table
    for part in path.split('.'):
        if not isinstance(node, dict) or part not in node:
            if default is None:
                raise KeyError(path)
            return default
        node = node[part]
    return node


def patch(table, path, value):
    """Return a deep copy with `path` set (dotted keys; list indexes are not supported)."""
    out = copy.deepcopy(table)
    node = out
    parts = path.split('.')
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
    return out


def load_json(path):
    return json.loads(Path(path).read_text())


def width_value(mapping, width):
    """Per-width tables are keyed by strings; the largest key not above the width applies."""
    keys = sorted(int(k) for k in mapping)
    chosen = keys[0]
    for k in keys:
        if k <= width:
            chosen = k
    return mapping[str(chosen)]


def interpolate(points, x):
    points = sorted(points)
    if x <= points[0][0]:
        return points[0][1] * x / points[0][0] if points[0][0] else points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


# ----------------------------------------------------------------------------- workload
@dataclasses.dataclass
class Slot:
    layer: int
    group: int
    width: int
    cpu_experts: float      # experts served by the CPU lane (fractional averages are allowed)
    worker_experts: float   # experts served by the worker device (topology A)
    primary_experts: float  # experts resident on the device that runs the mixer
    cpu_bytes: float
    worker_rows: float      # token x expert rows returned by the worker


@dataclasses.dataclass
class Round:
    slots: list
    accepted: float
    draft_steps: int
    draft_cpu_experts: float   # per draft step, served by the CPU unless a GPU draft bank exists
    draft_width: int = 1


@dataclasses.dataclass
class Placement:
    """Expert slots per device per layer, the cache policy and where the draft bank lives."""
    primary_slots: float = 15.0
    worker_slots: float = 33.0
    experts_per_layer: int = 224
    policy: str = 'lru'           # lru | lfu (curve policy factor) | recorded
    draft_bank: str = 'cpu'       # cpu | primary | worker
    draft_bank_slots: float = 0.0  # expert slots taken from that device for the MTP layer


@dataclasses.dataclass
class Topology:
    kind: str = 'A'                  # A: expert-parallel worker; B: layer-range pipeline
    primary: str = 'hip'
    worker: str = 'cuda'
    groups: tuple = (2, 2)           # verification widths per group (window = sum)
    combine: str = 'strict'          # strict (CPU adds every row) | device3 (primary adds rows in canonical order)
    dispatch: str = 'socket'         # socket (worker asked when the CPU thread reaches the slot) | direct (at publish)
    ownership: str = 'blocking'      # blocking | async
    cpu_lanes: str = 'single15'      # single15 | node_local8x2 (B only)
    split_layer: int = 24            # B: first layer of the tail stage
    stage_order: str = 'pipeline'    # B: pipeline (group after group) | interleave
    head_hidden_fraction: float = 0.0
    context_tokens: int = 8192       # MLA attention reads the latent cache of this many positions
    residents: str = 'chain'         # chain (resident experts on the mixer stream) | side (own stream, overlapped)


def route_shares(placement, table):
    """Share of routed rows served by primary, worker and CPU from the slot counts and the hit curve."""
    curve = [(p[0], p[1]) for p in get(table, 'model.hit_curve.points')]
    gain = get(table, 'model.hit_curve.policy_gain', {}).get(placement.policy, 1.0)
    total = placement.primary_slots + placement.worker_slots
    coverage = min(1.0, total / placement.experts_per_layer)
    hit = min(1.0, interpolate(curve, coverage) * gain * get(table, 'model.hit_curve.scale', 1.0))
    if total <= 0:
        return dict(primary=0.0, worker=0.0, cpu=1.0)
    return dict(primary=hit * placement.primary_slots / total, worker=hit * placement.worker_slots / total, cpu=1 - hit)


def synthetic_workload(table, placement, topology, rounds=1, accepted_per_round=None, shares=None):
    """Average slots per round. `shares` (primary/worker/cpu) overrides the placement curve."""
    shares = shares or route_shares(placement, table)
    unique = get(table, 'model.unique_experts_per_route')
    depth = sum(topology.groups) - 1
    accepted = accepted_per_round or get(table, 'model.accepted_per_round')
    draft_cpu = 0.0 if placement.draft_bank != 'cpu' else TOP_K * unique
    out = []
    for _ in range(rounds):
        slots = []
        for layer in MOE_LAYERS:
            for g, width in enumerate(topology.groups):
                experts = unique * width * TOP_K
                cpu = experts * shares['cpu']
                worker = experts * shares['worker'] if topology.kind == 'A' else 0.0
                primary = experts - cpu - worker
                slots.append(Slot(layer, g, width, cpu, worker, primary, cpu * EXPERT_BYTES[layer],
                                  worker * shares.get('rows_per_worker_expert', get(table, 'model.rows_per_worker_expert'))))
        out.append(Round(slots, accepted, depth, draft_cpu))
    return out


def workload_from_routes(rounds, placement, table, topology, accepted_per_round):
    """Exact per-slot counts from a routing event trace with recorded primary residency.

    Worker ownership is not recorded: the worker's share of the non-resident experts follows a
    static top-N-per-layer rule over the trace's own route counts (a stand-in for the prior).
    """
    counts = {}
    for r in rounds:
        for e in r['target']:
            for row in e['routes']:
                for x in row:
                    counts[(e['layer'], x)] = counts.get((e['layer'], x), 0) + 1
    worker = {}
    for layer in MOE_LAYERS:
        ranked = sorted(((c, x) for (l, x), c in counts.items() if l == layer), reverse=True)
        worker[layer] = {x for _, x in ranked[:int(round(placement.worker_slots))]} if topology.kind == 'A' else set()
    out = []
    for r in rounds:
        slots = []
        by_slot = {}
        for e in r['target']:
            by_slot.setdefault(e['layer'], []).append(e)
        for layer in MOE_LAYERS:
            for g, e in enumerate(sorted(by_slot.get(layer, []), key=lambda x: x['position'])):
                width = len(e['routes'])
                experts = {x for row in e['routes'] for x in row}
                resident = set(e.get('resident', [])) & experts
                w = (experts - resident) & worker[layer]
                cpu = experts - resident - w
                rows = sum(1 for row in e['routes'] for x in row if x in w)
                slots.append(Slot(layer, g, width, len(cpu), len(w), len(resident), len(cpu) * EXPERT_BYTES[layer], rows))
        draft = [e for e in r['draft']]
        draft_cpu = (sum(len({x for row in e['routes'] for x in row}) for e in draft) / len(draft)) if draft else TOP_K
        if placement.draft_bank != 'cpu':
            draft_cpu = 0.0
        out.append(Round(slots, accepted_per_round, len(draft) if draft else sum(topology.groups) - 1, draft_cpu))
    return out


# ----------------------------------------------------------------------------- decode DAG
class Builder:
    def __init__(self):
        self.events = []
        self.names = set()

    def add(self, name, work=None, after=(), category='compute', minimum_ms=0.0, slots=None):
        if name in self.names:
            raise ValueError('duplicate event ' + name)
        self.names.add(name)
        after = tuple(a for a in (after if isinstance(after, (tuple, list)) else (after,)) if a)
        e = Event(name, dict(work or {}), after, category, dict(slots or {}), minimum_ms)
        self.events.append(e)
        return name

    def transfer(self, name, size, src, dst, hardware, after=()):
        after = tuple(a for a in (after if isinstance(after, (tuple, list)) else (after,)) if a)
        events = transfer_events(name, size, src, dst, hardware, after)
        for e in events:
            if e.name in self.names:
                raise ValueError('duplicate event ' + e.name)
            self.names.add(e.name)
            self.events.append(e)
        return events[-1].name if events else (after[-1] if after else None)


def resources_for(hardware, table, topology):
    r = dict(hardware['resources'])
    lanes = get(table, 'cpu.lanes')
    if topology.kind == 'B' and topology.cpu_lanes == 'node_local8x2':
        for i in (0, 1):
            r[f'cpu.lane{i}'] = 1.0
            r[f'cpu.flow{i}'] = lanes['node8']['gb_s'] * 1e6
    else:
        r['cpu.lane0'] = 1.0
        r['cpu.flow0'] = lanes['single15']['gb_s'] * 1e6
    r['worker.host'] = 1.0
    for dev in ('cuda', 'hip'):
        r[f'{dev}.compute'] = 1.0
        r[f'{dev}.side'] = 1.0
    return r


def _device_costs(table, device, kind, width, context_tokens=8192):
    d = get(table, f'devices.{device}.decode')
    mixer = 0.0
    if kind == 'kda':
        mixer = width_value(d['kda_ms'], width)
    elif kind == 'mla':
        mixer = width_value(d['mla_ms'], width) + d['mla_kv_ms_per_ktoken'] * context_tokens / 1024
    return dict(mixer=mixer,
                publish=d['publish_ms'], resident_fixed=d['resident_fixed_ms'],
                resident_expert=width_value(d['resident_expert_ms'], width), launch=d['launch_ms'],
                combine=d['combine_ms'], draft=width_value(d['draft_step_ms'], 1), head=d['head_ms'], tail=d['tail_ms'],
                vram=get(table, f'devices.{device}.decode.vram_gb_s') * 1e6)


def decode_round_events(round_, table, placement, topology, hardware, prefix='r'):
    """Events of one verify step (+ draft chain) for topology A or B. Returns (builder, markers)."""
    b = Builder()
    cpu_t = get(table, 'cpu')
    w_t = get(table, 'worker')
    markers = dict(cpu_start={}, finish={}, head_end=None, verify_end=None, draft=[], round_end=None)
    if topology.kind == 'A':
        stages = [dict(device=topology.primary, layers=list(MOE_LAYERS), lane=0, worker=topology.worker)]
    else:
        stages = [dict(device=topology.primary, layers=[l for l in MOE_LAYERS if l < topology.split_layer], lane=0, worker=None),
                  dict(device=topology.worker, layers=[l for l in MOE_LAYERS if l >= topology.split_layer],
                       lane=1 if topology.cpu_lanes == 'node_local8x2' else 0, worker=None)]
    groups = list(range(len(topology.groups)))
    slot_of = {(s.layer, s.group): s for s in round_.slots}
    # Head: embedding and the dense layers, per group, on the first stage's device (group A runs ahead).
    dev0 = stages[0]['device']
    c0 = _device_costs(table, dev0, 'dense', 1)
    head = c0['head'] * (1 - topology.head_hidden_fraction)
    last_gpu = {st['device']: None for st in stages}
    lane_tail = {0: None, 1: None}
    group_entry = {g: None for g in groups}               # dependency carried between stages per group
    headed = set()
    finish_prev = {}                                       # (group) -> last finish name within the current stage
    seq_counter = [0]

    def stage_slots(st):
        order = []
        if topology.kind == 'A' or topology.stage_order == 'interleave':
            for l in st['layers']:
                for g in groups:
                    order.append((l, g))
        else:
            for g in groups:
                for l in st['layers']:
                    order.append((l, g))
        return order

    for si, st in enumerate(stages):
        dev, lane = st['device'], st['lane']
        worker = st['worker']
        finish_prev = {}
        entry_dep = dict(group_entry)
        for (l, g) in stage_slots(st):
            s = slot_of[(l, g)]
            c = _device_costs(table, dev, mixer_kind(l), s.width, topology.context_tokens)
            tag = f'{prefix}/s{si}/l{l}/g{g}'
            if si == 0 and g not in headed:
                # Embedding and the dense layers of this group run right before its first publication.
                name = b.add(f'{prefix}/head/g{g}', {f'{dev}.compute': head}, last_gpu[dev], 'head')
                last_gpu[dev] = name; headed.add(g); entry_dep[g] = name
                if markers['head_end'] is None:
                    markers['head_end'] = name
            # GPU stream: wait-add of the previous slot of this group, mixer, publish, residents.
            wait_dep = [last_gpu[dev]]
            if g in finish_prev:
                wait_dep.append(finish_prev[g])
            else:
                wait_dep.append(entry_dep[g])
            waitadd = b.add(tag + '/waitadd', {}, wait_dep, 'wait', minimum_ms=w_t.get('wait_release_ms', 0.0))
            mixer = b.add(tag + '/mixer', {f'{dev}.compute': c['mixer'] + c['launch']}, waitadd, 'mixer')
            publish = b.add(tag + '/publish', {f'{dev}.compute': c['publish']}, mixer, 'publish')
            res_ms = c['resident_fixed'] + s.primary_experts * c['resident_expert'] + \
                s.primary_experts * EXPERT_BYTES[l] * s.width / c['vram'] * 0  # bytes are inside resident_expert_ms
            if topology.residents == 'side':
                residents = b.add(tag + '/residents', {f'{dev}.side': res_ms}, publish, 'resident')
                last_gpu[dev] = publish
            else:
                residents = b.add(tag + '/residents', {f'{dev}.compute': res_ms}, publish, 'resident')
                last_gpu[dev] = residents
            # CPU lane: in publication order; sees the publication after the mailbox latency.
            cpu_dep = [publish, lane_tail[lane]]
            n_cpu = s.cpu_experts
            # The service thread's own work per slot: wake, quantise the activations, run its experts; with a
            # worker it also packs the request (activation copy, ownership memcmp) and unpacks the reply.
            cpu_work = {f'cpu.lane{lane}': cpu_t['slot_fixed_ms'] + s.width * cpu_t['quant_ms_per_token'] +
                        n_cpu * cpu_t['expert_fixed_ms'] + (w_t['client_host_ms'] if worker and s.worker_experts > 0 else 0.0),
                        f'cpu.flow{lane}': s.cpu_bytes}
            cpu = b.add(tag + '/cpu', cpu_work, cpu_dep, 'cpu', minimum_ms=cpu_t['mailbox_latency_ms'])
            markers['cpu_start'][(si, l, g)] = cpu
            rows_back = b.transfer(tag + '/cpu-rows', s.width * TOP_K * ROW_BYTES, 'ram', dev, hardware, cpu)
            ends = [cpu]
            # Worker (topology A): request, execution, rows return.
            worker_end = None
            if worker and s.worker_experts > 0:
                submit_dep = [publish] if topology.dispatch == 'direct' else [publish, lane_tail[lane]]
                submit_ms = w_t['rtt_flag_ms'] if topology.dispatch == 'direct' else w_t['rtt_socket_ms']
                submit = b.add(tag + '/submit', {'host': w_t['submit_host_ms']}, submit_dep, 'worker', minimum_ms=submit_ms / 2)
                wd = get(table, f'devices.{worker}.decode')
                exec_ms = wd['worker_request_fixed_ms'] + s.worker_experts * width_value(wd['resident_expert_ms'], s.width)
                run = b.add(tag + '/worker', {f'{worker}.compute': exec_ms}, submit, 'worker')
                ret = b.transfer(tag + '/worker-rows', s.worker_rows * ROW_BYTES, worker, 'ram', hardware, run)
                copy = b.add(tag + '/worker-copy', {'host': s.worker_rows * ROW_BYTES / (w_t['memcpy_gb_s'] * 1e6)},
                             ret, 'worker', minimum_ms=submit_ms / 2)
                worker_end = copy
            if topology.combine == 'device3':
                dep = [rows_back, residents] + ([b.transfer(tag + '/worker-rows-dev', s.worker_rows * ROW_BYTES, 'ram', dev,
                                                            hardware, worker_end)] if worker_end else [])
                finish = b.add(tag + '/combine', {f'{dev}.compute': c['combine'] * s.width}, dep, 'combine')
                lane_tail[lane] = cpu
                scope_end = cpu  # the CPU thread is done when its rows are posted; the GPU combine is not host service
            else:
                dep = [cpu] + ([worker_end] if worker_end else []) + ([residents] if topology.residents == 'side' else [])
                finish = b.add(tag + '/finish', {f'cpu.lane{lane}': c['combine'] * s.width * cpu_t['strict_combine_factor']},
                               dep, 'finish')
                lane_tail[lane] = finish
                scope_end = finish
            # Ownership refresh (adaptive worker cache): blocks the CPU lane every `interval` requests.
            seq_counter[0] += 1
            if worker and w_t['ownership_interval'] and seq_counter[0] % int(w_t['ownership_interval']) == 0:
                # Ownership refresh: the worker evicts/uploads up to four experts per layer over its own link.
                own_ms = get(table, f'devices.{worker}.decode.ownership_ms', w_t['ownership_ms'])
                own_work = {'host': own_ms, f'{worker}.compute': own_ms}
                if topology.ownership == 'blocking':
                    own_work[f'cpu.lane{lane}'] = own_ms
                own = b.add(tag + '/ownership', own_work, [finish] if topology.ownership == 'blocking' else [publish], 'ownership')
                if topology.ownership == 'blocking':
                    lane_tail[lane] = own
                    finish = own
                    scope_end = own
            markers['finish'][(si, l, g)] = scope_end
            finish_prev[g] = finish
        # Stage boundary (topology B): the residual of each group crosses to the next stage's device.
        if si + 1 < len(stages):
            nxt = stages[si + 1]['device']
            for g in groups:
                width = topology.groups[g]
                last = b.add(f'{prefix}/s{si}/g{g}/stage-exit', {}, [finish_prev[g], last_gpu[dev]], 'wait')
                hand = b.transfer(f'{prefix}/s{si}/g{g}/handoff', width * RESIDUAL_BYTES, dev, nxt, hardware, last)
                group_entry[g] = b.add(f'{prefix}/s{si}/g{g}/handoff-latency', {'host': w_t['bounce_host_ms']}, hand,
                                       'handoff', minimum_ms=w_t['bounce_latency_ms'])
    # Tail: final wait-adds, output norm, logits, readback and sampling on the last stage's device.
    last_dev = stages[-1]['device']
    cl = _device_costs(table, last_dev, 'dense', 1)
    tail_dep = [last_gpu[last_dev]] + [markers['finish'][k] for k in markers['finish'] if k[0] == len(stages) - 1 and k[1] == stages[-1]['layers'][-1]]
    tail = b.add(f'{prefix}/tail', {f'{last_dev}.compute': cl['tail']}, tail_dep, 'tail')
    markers['verify_end'] = tail
    prev = tail
    # Draft chain: depth sequential MTP steps; CPU draft experts go through the last stage's lane.
    lane = stages[-1]['lane']
    for d in range(round_.draft_steps):
        gpu = b.add(f'{prefix}/draft{d}/gpu', {f'{last_dev}.compute': cl['draft']}, prev, 'draft')
        if round_.draft_cpu_experts > 0:
            cpu = b.add(f'{prefix}/draft{d}/cpu', {f'cpu.lane{lane}': cpu_t['slot_fixed_ms'] + cpu_t['quant_ms_per_token'] +
                                                   round_.draft_cpu_experts * cpu_t['expert_fixed_ms'],
                                                   f'cpu.flow{lane}': round_.draft_cpu_experts * EXPERT_BYTES[44]},
                        [gpu, lane_tail[lane]], 'draft', minimum_ms=cpu_t['mailbox_latency_ms'])
            lane_tail[lane] = cpu
            rows = b.transfer(f'{prefix}/draft{d}/rows', TOP_K * ROW_BYTES, 'ram', last_dev, hardware, cpu)
            prev = b.add(f'{prefix}/draft{d}/combine', {f'{last_dev}.compute': cl['combine']}, rows, 'draft')
        else:
            bank_dev = topology.primary if placement.draft_bank == 'primary' else (topology.worker if placement.draft_bank == 'worker' else last_dev)
            bank = b.add(f'{prefix}/draft{d}/bank', {f'{bank_dev}.compute': get(table, f'devices.{bank_dev}.decode.resident_fixed_ms') +
                                                     TOP_K * width_value(get(table, f'devices.{bank_dev}.decode.resident_expert_ms'), 1)},
                         gpu, 'draft')
            if bank_dev != last_dev:
                bank = b.transfer(f'{prefix}/draft{d}/bank-rows', TOP_K * ROW_BYTES, bank_dev, last_dev, hardware, bank)
            prev = b.add(f'{prefix}/draft{d}/combine', {f'{last_dev}.compute': cl['combine']}, bank, 'draft')
        markers['draft'].append(prev)
    resync = b.add(f'{prefix}/resync', {'host': get(table, 'host.resync_ms')}, prev, 'resync')
    markers['round_end'] = b.add(f'{prefix}/other', {'host': get(table, 'host.other_ms')}, resync, 'other')
    return b, markers


def scopes_from_schedule(result, markers):
    """STEP_TRACE-style partition read off the schedule of one round."""
    t = {e['name']: (e['start_ms'], e['end_ms']) for e in result['events']}
    keys = sorted(markers['cpu_start'])
    starts = [t[markers['cpu_start'][k]][0] for k in keys]
    ends = [t[markers['finish'][k]][1] for k in keys]
    head = starts[0]
    service = sum(e - s for s, e in zip(starts, ends))
    between = sum(max(0.0, s - e) for s, e in zip(starts[1:], ends[:-1]))
    verify_end = t[markers['verify_end']][1]
    tail = verify_end - ends[-1]
    draft = (t[markers['draft'][-1]][1] - verify_end) if markers['draft'] else 0.0
    round_ms = t[markers['round_end']][1]
    resync_other = round_ms - verify_end - draft
    return dict(head=head, cpu_service=service, between=between, tail=tail, draft=draft,
                resync_other=resync_other, round_ms=round_ms, verify_ms=verify_end)


def simulate_decode(workload, table, placement, topology, hardware, rounds=None):
    """Sum of independent rounds (round r+1 starts after round r). Returns tok/s and scopes."""
    rounds = rounds or len(workload)
    res = resources_for(hardware, table, topology)
    totals = dict(head=0.0, cpu_service=0.0, between=0.0, tail=0.0, draft=0.0, resync_other=0.0, round_ms=0.0, verify_ms=0.0)
    lower = 0.0
    accepted = 0.0
    counters = dict(cpu_bytes=0.0, worker_rows=0.0, slots=0)
    sample = workload[:rounds]
    for i, r in enumerate(sample):
        b, markers = decode_round_events(r, table, placement, topology, hardware, prefix=f'r{i}')
        result = schedule(b.events, res, hardware.get('slots'))
        s = scopes_from_schedule(result, markers)
        for k in totals:
            totals[k] += s[k]
        lower += result['lower_bound_ms']
        accepted += r.accepted
        counters['cpu_bytes'] += sum(x.cpu_bytes for x in r.slots) + r.draft_steps * r.draft_cpu_experts * EXPERT_BYTES[44]
        counters['worker_rows'] += sum(x.worker_rows for x in r.slots)
        counters['slots'] += len(r.slots)
    n = len(sample)
    per_round = {k: v / n for k, v in totals.items()}
    return dict(tok_s=accepted * 1000 / totals['round_ms'], ms_per_round=per_round['round_ms'],
                bound_tok_s=accepted * 1000 / lower if lower else None, scopes=per_round,
                accepted_per_round=accepted / n, rounds=n,
                cpu_gb_per_output=counters['cpu_bytes'] / accepted / 1e9,
                worker_rows_per_slot=counters['worker_rows'] / max(1, counters['slots']),
                topology=dataclasses.asdict(topology), placement=dataclasses.asdict(placement),
                quality_qualified=False,
                scope='Throughput model from measured per-slot costs; rounds are serial; no quality prediction')


# ----------------------------------------------------------------------------- prefill DAG
@dataclasses.dataclass
class PrefillPlan:
    tokens: int = 8192
    primary: str = 'hip'
    worker: str = 'cuda'           # '' for a single device
    primary_fraction: float = 0.5  # share of the 14 expert groups on the primary
    exchange: str = 'f32'          # f32 | f16 activation exchange
    return_mode: str = 'chunks'    # chunks (512-row chunks with acks) | partial (one partial per layer)
    prefetch_layers: int = 1       # layers of look-ahead for expert streaming when stream_overlap is 'layer'
    resident_groups: float = 0.0   # groups per layer already resident (not streamed)
    mixer_speedup: float = 1.0
    gemm_speedup: float = 1.0
    cpu_groups: float = 0.0        # groups per layer computed by the CPU lane (kappa rule)
    groups_per_layer: int = 14
    stream_overlap: str = 'mixer'  # none (streams after the mixer) | mixer (during this layer's mixer, STAGE_PREFETCH) | layer (one layer ahead)
    worker_stream_overlap: str = 'mixer'  # the worker preloads its next layer's groups after its reply (MIXED_PREFETCH_NEXT)


def prefill_events(table, plan, hardware):
    """Per-layer prefill DAG: serial mixers on the primary, 16-expert groups streamed through a
    staging ring on each device (a group's slot is reused only after its GEMM finished), the
    worker fed through the host exchange, rows or partials returned and accumulated."""
    b = Builder()
    pt = lambda dev: get(table, f'devices.{dev}.prefill')
    p, w = plan.primary, plan.worker
    tokens = plan.tokens
    layers = get(table, f'devices.{p}.prefill.mixer_ms')  # per layer id (string keys) at 8192 tokens
    scale = tokens / 8192
    prev_join = b.add('load', {'host': get(table, f'devices.{p}.prefill.unassigned_ms') * scale}, (), 'other')
    joins = []
    act_bytes = tokens * HIDDEN * (2 if plan.exchange == 'f16' else 4)
    groups = plan.groups_per_layer
    primary_groups = round(groups * plan.primary_fraction) if w else groups
    share_p = primary_groups / groups
    experts_per_layer = get(table, 'model.experts_per_layer')
    rows_per_expert = tokens * TOP_K / experts_per_layer
    tile = get(table, 'worker.prefill_tile_rows')
    padded = math.ceil(rows_per_expert / tile) * tile
    padded_8k = math.ceil(8192 * TOP_K / experts_per_layer / tile) * tile
    worker_gemm_scale = padded / padded_8k
    ring = {d: get(table, f'devices.{d}.prefill.ring_groups') for d in (p, w) if d}
    gemm_history = {d: [] for d in (p, w) if d}  # per device, the GEMM events in stream order (ring reuse)
    for l in range(LAYERS):
        tag = f'l{l}'
        mixer = b.add(tag + '/mixer', {f'{p}.compute': layers[str(l)] * scale / plan.mixer_speedup}, prev_join, 'mixer')
        if l in DENSE_LAYERS:
            ffn = b.add(tag + '/dense', {f'{p}.compute': pt(p)['dense_ffn_ms'] * scale}, mixer, 'dense')
            prev_join = b.add(tag + '/join', {}, ffn, 'join'); joins.append(prev_join)
            continue
        ends = [mixer]

        def stream_dependency(mode):
            if mode == 'none':
                return mixer
            if mode == 'mixer':
                return prev_join
            return joins[-1 - plan.prefetch_layers] if len(joins) > plan.prefetch_layers else None

        group_bytes = 16 * EXPERT_BYTES[l]
        streamed = max(0.0, groups - plan.resident_groups - plan.cpu_groups)
        exchange_in = None
        if w:
            out = b.transfer(tag + '/exchange-out', act_bytes, p, 'ram', hardware, mixer)
            copy = b.add(tag + '/exchange-copy', {'host': act_bytes * get(table, 'worker.exchange_memcpy_passes') /
                                                   (get(table, 'worker.memcpy_gb_s') * 1e6)}, out, 'exchange')
            exchange_in = b.transfer(tag + '/exchange-in', act_bytes, 'ram', w, hardware, copy)
        for dev, n in ((p, streamed * share_p), (w, streamed * (1 - share_p))):
            if not dev or n <= 0:
                continue
            dep0 = stream_dependency(plan.worker_stream_overlap if dev == w else plan.stream_overlap)
            count = int(math.ceil(n - 1e-9))
            last_gemm = None
            resident_n = (plan.resident_groups + plan.cpu_groups) * (share_p if dev == p else 1 - share_p)
            # The worker pads rows per expert to its tile, so its GEMM cost follows the padded rows, not the tokens.
            per_group = (pt(dev)['group_fixed_ms'] + pt(dev)['group_gemm_ms'] * (worker_gemm_scale if dev == w else scale)) / plan.gemm_speedup
            for i in range(count):
                frac = min(1.0, n - i)
                hist = gemm_history[dev]
                reuse = hist[-ring[dev]] if len(hist) >= ring[dev] else None
                up = b.transfer(f'{tag}/{dev}/stream{i}', group_bytes * frac, 'ram', dev, hardware, [dep0, reuse])
                gdep = [up, last_gemm] + ([exchange_in] if dev == w else [mixer])
                gemm = b.add(f'{tag}/{dev}/gemm{i}', {f'{dev}.compute': per_group * frac}, gdep, 'gemm')
                hist.append(gemm); last_gemm = gemm
            if resident_n > 0:
                gemm = b.add(f'{tag}/{dev}/gemm-resident', {f'{dev}.compute': per_group * resident_n},
                             [last_gemm] + ([exchange_in] if dev == w else [mixer]), 'gemm')
                last_gemm = gemm
            if dev == w:
                rows = tokens * get(table, 'worker.return_rows_per_token')
                if plan.return_mode == 'chunks':
                    chunks = max(1, int(math.ceil(rows / 512)))
                    prev = last_gemm
                    for ci in range(chunks):
                        ret = b.transfer(f'{tag}/return{ci}', min(512, rows - 512 * ci) * ROW_BYTES, w, 'ram', hardware, prev)
                        prev = b.add(f'{tag}/ack{ci}', {'host': get(table, 'worker.prefill_ack_ms')}, ret, 'ack')
                    back = b.transfer(tag + '/return-in', rows * ROW_BYTES, 'ram', p, hardware, prev)
                else:
                    ret = b.transfer(tag + '/return', tokens * ROW_BYTES, w, 'ram', hardware, last_gemm)
                    back = b.transfer(tag + '/return-in', tokens * ROW_BYTES, 'ram', p, hardware, ret)
                ends.append(b.add(tag + '/accumulate', {f'{p}.compute': pt(p)['accumulate_ms'] * scale}, back, 'accumulate'))
            else:
                ends.append(last_gemm)
        if plan.cpu_groups > 0:
            cpu = b.add(tag + '/cpu', {'cpu.lane0': 16 * plan.cpu_groups * rows_per_expert * get(table, 'cpu.prefill_row_ms'),
                                       'cpu.flow0': 16 * plan.cpu_groups * EXPERT_BYTES[l]}, mixer, 'cpu')
            ends.append(cpu)
        prev_join = b.add(tag + '/join', {'host': pt(p)['layer_join_ms']}, ends, 'join')
        joins.append(prev_join)
    head = b.add('head', {f'{p}.compute': pt(p)['head_ms'] * scale}, prev_join, 'head')
    return b, head


def simulate_prefill(table, plan, hardware):
    b, head = prefill_events(table, plan, hardware)
    res = dict(hardware['resources'])
    res.update({'cpu.lane0': 1.0, 'cpu.flow0': get(table, 'cpu.lanes.single15.gb_s') * 1e6,
                f'{plan.primary}.compute': 1.0})
    if plan.worker:
        res[f'{plan.worker}.compute'] = 1.0
    result = schedule(b.events, res, hardware.get('slots'))
    t = {e['name']: e for e in result['events']}
    mixer_chain = sum(t[f'l{l}/mixer']['end_ms'] - t[f'l{l}/mixer']['start_ms'] for l in range(LAYERS))
    ms = result['milliseconds']
    return dict(tok_s=plan.tokens * 1000 / ms, milliseconds=ms, lower_bound_ms=result['lower_bound_ms'],
                mixer_chain_ms=mixer_chain, exposed_ffn_ms=ms - mixer_chain - t['load']['end_ms'],
                resource_service_ms=result['resource_service_ms'], plan=dataclasses.asdict(plan),
                quality_qualified=False, scope='Prefill phase model at measured per-layer costs; no quality prediction')


# ----------------------------------------------------------------------------- records, calibration, validation
def topology_from(record, table):
    t = record.get('topology', {})
    return Topology(**{k: (tuple(v) if k == 'groups' else v) for k, v in t.items()})


def placement_from(record):
    return Placement(**record.get('placement', {}))


def predict_record(record, table, hardware):
    if record['kind'] == 'decode':
        topo = topology_from(record, table)
        place = placement_from(record)
        shares = record.get('shares')
        wl = synthetic_workload(table, place, topo, rounds=1, accepted_per_round=record.get('accepted_per_round'), shares=shares)
        return simulate_decode(wl, table, place, topo, hardware)
    plan = PrefillPlan(**record.get('plan', {}))
    return simulate_prefill(table, plan, hardware)


def predicted_scopes(result):
    if 'scopes' in result:
        s = result['scopes']
        return dict(head=s['head'], cpu_service=s['cpu_service'], between=s['between'], tail=s['tail'], draft=s['draft'])
    return {}


def evaluate(records, table, hardware, only=None):
    rows = []
    for r in records:
        if only and r.get('split') != only:
            continue
        result = predict_record(r, table, hardware)
        measured = r['measured_tok_s']
        predicted = result['tok_s']
        tolerance = 0.10 if r['kind'] == 'decode' else 0.15
        row = dict(name=r['name'], kind=r['kind'], split=r.get('split', 'holdout'), measured_tok_s=measured,
                   predicted_tok_s=predicted, error_fraction=predicted / measured - 1, tolerance=tolerance,
                   within=abs(predicted / measured - 1) <= tolerance, scope_errors={})
        row['context_tokens'] = (r.get('plan', {}).get('tokens',8192) if r['kind']=='prefill'
                                 else r.get('topology', {}).get('context_tokens',8192))
        if r.get('scopes'):
            ps = predicted_scopes(result)
            for k, v in r['scopes'].items():
                if k in ps and v:
                    row['scope_errors'][k] = dict(measured=v, predicted=ps[k], error_fraction=ps[k] / v - 1)
        rows.append(row)
    return rows


def objective(records, table, hardware, scope_weight=0.3):
    total = n = 0.0
    for r in records:
        if not r.get('fit'):
            continue
        result = predict_record(r, table, hardware)
        total += abs(math.log(result['tok_s'] / r['measured_tok_s'])); n += 1
        ps = predicted_scopes(result)
        for k, v in (r.get('scopes') or {}).items():
            if k in ps and v > 0 and ps[k] > 0:
                total += scope_weight * abs(math.log(ps[k] / v)); n += scope_weight
    return total / n if n else 0.0


def calibrate(records, table, hardware, free, rounds=3, factors=(0.5, 0.7, 0.85, 0.93, 1.0, 1.08, 1.15, 1.3, 1.6, 2.0), log=print):
    """Coordinate descent over `free` dotted paths, fit rows only. Returns (table, score, history)."""
    if not any(r.get('fit') for r in records):
        raise ValueError('calibration needs at least one record with fit: true')
    table = copy.deepcopy(table)
    best = objective(records, table, hardware)
    history = [dict(step='start', score=best)]
    log(f'start: mean |log error| = {best:.4f}')
    for round_ in range(rounds):
        improved = False
        for path in free:
            base = get(table, path)
            if not isinstance(base, (int, float)) or base == 0:
                continue
            candidates = []
            for f in factors:
                trial = patch(table, path, base * f)
                candidates.append((objective(records, trial, hardware), f))
            score, f = min(candidates)
            if score < best - 1e-6:
                table = patch(table, path, base * f); best = score; improved = True
                history.append(dict(step=f'round {round_ + 1}', path=path, factor=f, value=base * f, score=best))
                log(f'round {round_ + 1}: {path} x{f:g} -> {base * f:.4g} (error {best:.4f})')
        if not improved:
            break
    log(f'done: mean |log error| = {best:.4f}')
    return table, best, history


def validate(records, table, hardware, scope_tolerance=0.15, context_tokens=None):
    rows = evaluate(records, table, hardware)
    gated = {r['name'] + '/' + r['kind'] for r in records if r.get('gate', True)}
    for row in rows:
        row['gate'] = (row['name'] + '/' + row['kind'] in gated and
                       (context_tokens is None or row['context_tokens']==context_tokens))
    holdout = [r for r in rows if r['split'] == 'holdout' and r['gate']]
    fit_rows = [r for r in rows if r['split'] == 'fit' and
                (context_tokens is None or r['context_tokens']==context_tokens)]
    scope_ok = all(abs(e['error_fraction']) <= scope_tolerance for r in holdout + fit_rows for e in r['scope_errors'].values())
    return dict(rows=rows, calibrated=bool(holdout) and all(r['within'] for r in holdout) and scope_ok,
                holdout_count=len(holdout), ungated=[r['name'] for r in rows if not r['gate']], scope_tolerance=scope_tolerance,
                context_tokens=context_tokens,
                scope='Gated holdout rows within 10% decode / 15% prefill and every traced scope within the scope tolerance; ungated rows are reported only')


# ----------------------------------------------------------------------------- levers
def apply_lever(state, lever):
    """A lever is a list of edits on the table, topology, placement or prefill plan."""
    table, topo, place, plan = state['table'], state['topology'], state['placement'], state['plan']
    extra = {}
    for edit in lever['edits']:
        target, path, value = edit['target'], edit['path'], edit['value']
        if target == 'table':
            if edit.get('op') == 'scale':
                value = get(table, path) * value
            table = patch(table, path, value)
        elif target == 'topology':
            topo = dataclasses.replace(topo, **{path: tuple(value) if path == 'groups' else value})
        elif target == 'placement':
            if edit.get('op') == 'add':
                value = getattr(place, path) + value
            place = dataclasses.replace(place, **{path: value})
        elif target == 'plan':
            if edit.get('op') == 'scale':
                value = getattr(plan, path) * value
            plan = dataclasses.replace(plan, **{path: value})
        elif target == 'state':
            extra[path] = value
        else:
            raise ValueError('unknown edit target ' + target)
    out = dict(state)
    out.update(table=table, topology=topo, placement=place, plan=plan, **extra)
    return out


def score_state(state, hardware, want_decode=True, want_prefill=True):
    out = {}
    if want_decode:
        wl = synthetic_workload(state['table'], state['placement'], state['topology'], rounds=1,
                                accepted_per_round=state.get('accepted_per_round'))
        out['decode'] = simulate_decode(wl, state['table'], state['placement'], state['topology'], hardware)
    if want_prefill and state['plan'] is not None:
        out['prefill'] = simulate_prefill(state['table'], state['plan'], hardware)
    return out


def sweep_levers(levers, state, hardware, target_decode=50.0, target_prefill=1000.0, allow_lossy=False, log=print):
    """Greedy forward selection: each step adds the lever with the largest marginal gain on its metric."""
    remaining = [l for l in levers if allow_lossy or l.get('lossless', False)]
    excluded = [l['name'] for l in levers if l not in remaining]
    base = score_state(state, hardware)
    path = []
    current = dict(state)
    dec = base.get('decode', {}).get('tok_s'); pre = base.get('prefill', {}).get('tok_s')
    log(f'base: decode {dec and round(dec, 2)} tok/s, prefill {pre and round(pre, 1)} tok/s')
    while remaining:
        best = None
        for lever in remaining:
            trial = apply_lever(current, lever)
            s = score_state(trial, hardware, want_decode=lever.get('applies_to', 'both') in ('decode', 'both'),
                            want_prefill=lever.get('applies_to', 'both') in ('prefill', 'both'))
            gain = 0.0
            if 'decode' in s and dec:
                gain += (s['decode']['tok_s'] - dec) / target_decode
            if 'prefill' in s and pre:
                gain += (s['prefill']['tok_s'] - pre) / target_prefill
            if best is None or gain > best[0]:
                best = (gain, lever, trial, s)
        gain, lever, trial, s = best
        if gain <= 1e-6:
            break
        current = trial
        dec = s['decode']['tok_s'] if 'decode' in s else dec
        pre = s['prefill']['tok_s'] if 'prefill' in s else pre
        remaining.remove(lever)
        path.append(dict(lever=lever['name'], lossless=lever.get('lossless', False), decode_tok_s=dec, prefill_tok_s=pre,
                         engine_knob=lever.get('engine_knob'), required_measurement=lever.get('required_measurement')))
        log(f'+ {lever["name"]}: decode {dec and round(dec, 2)}, prefill {pre and round(pre, 1)}')
        if (dec or 0) >= target_decode and (pre or 0) >= target_prefill:
            break
    return dict(base=dict(decode_tok_s=base.get('decode', {}).get('tok_s'), prefill_tok_s=base.get('prefill', {}).get('tok_s')),
                path=path, final=dict(decode_tok_s=dec, prefill_tok_s=pre),
                reaches=dict(decode=(dec or 0) >= target_decode, prefill=(pre or 0) >= target_prefill),
                excluded_lossy=excluded, targets=dict(decode=target_decode, prefill=target_prefill),
                scope='Greedy lever path on the calibrated model; gains are conditional on each lever delivering its edited constant')
