"""Replay opt-in GLM routing JSONL without collapsing rejected MTP branches.

An event is one actual expert execution group in one layer. Repeated token positions
remain separate events. Policy results include upload bytes, not just cache hits.
"""
import collections
import json


def read_events(path, experts=144):
    events = []
    seen = set()
    with open(path) as source:
        for line in source:
            e = json.loads(line)
            if e.get('kind') == 'cache':
                continue
            key = e['request'], e['event']
            if key in seen:
                raise ValueError('duplicate routing event')
            seen.add(key)
            for row in e['routes']:
                if len(row) != 8 or len(set(row)) != 8 or any(x < 0 or x >= experts for x in row):
                    raise ValueError('invalid top-8 event')
            events.append(e)
    return events


def replay(events, pack, capacity_mib, policy='lru', initial=()):
    if policy not in ('static', 'lru', 'protected'):
        raise ValueError('unsupported replay policy')
    if len({e['request'] for e in events}) > 1:
        raise ValueError('replay one request at a time with its own initial inventory')
    cache = collections.OrderedDict()
    seen = collections.Counter()
    capacity = int(capacity_mib * 2**20)
    used = 0
    for layer, expert in initial:
        size = pack.expert_bytes(layer)
        if (layer,expert) not in cache and used + size <= capacity:
            cache[layer,expert] = size
            used += size
    total = hits = uploads = 0
    for e in events:
        if e['layer'] not in pack.formats:
            continue
        size = pack.expert_bytes(e['layer'])
        wanted = {(e['layer'], x) for row in e['routes'] for x in row}
        # Count hits before admissions, never use future routes to choose current hits.
        total += len(wanted)*size
        hits += sum(cache.get(k,0) for k in wanted)
        for key in sorted(wanted):
            seen[key] += 1
            if key in cache:
                cache.move_to_end(key)
            elif policy != 'static' and size <= capacity:
                while used + size > capacity:
                    candidates = [k for k in cache if k not in wanted]
                    if not candidates:
                        break
                    victim = candidates[0]
                    if policy == 'protected':
                        victim = min(candidates, key=lambda k: min(seen[k],2))
                    used -= cache.pop(victim)
                if used + size <= capacity:
                    cache[key] = size
                    used += size
                    uploads += size
    return dict(policy=policy, requested_bytes=total, hit_bytes=hits, hit_share=hits/total if total else 0,
                upload_bytes=uploads, resident_bytes=used)


def execution_unions(events, pack):
    rows = []
    for e in events:
        if e['layer'] not in pack.formats:
            continue
        unique = len({x for row in e['routes'] for x in row})
        rows.append(dict(request=e['request'], round=e['round'], phase=e['phase'], event=e['event'],
                         layer=e['layer'], width=len(e['routes']), expert_bytes=unique*pack.expert_bytes(e['layer'])))
    return rows


def read_initial(path, request=None):
    initial = []
    with open(path) as source:
        for line in source:
            e = json.loads(line)
            if e.get('kind') == 'cache' and (request is None or e.get('request') == request):
                initial.extend((e['layer'], x) for x in e['experts'])
    return initial
