"""Dependency/resource simulation for heterogeneous GLM execution.

Units are bytes, milliseconds and bytes/ms. Timings are supplied by measurements,
not inferred from advertised FLOPS. A work item progresses only while all of its
resources are available. Progressive filling redistributes unused bandwidth.
The lower bound belongs to the supplied workload/DAG, never to the model in general.
"""
from dataclasses import dataclass, field, asdict
from collections import OrderedDict
import math


@dataclass(frozen=True)
class Event:
    name: str
    work: dict = field(default_factory=dict)
    after: tuple = ()
    category: str = 'compute'
    slots: dict = field(default_factory=dict)
    minimum_ms: float = 0.0


def positive(value, name, zero=False):
    if not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(f'{name}: expected finite {"nonnegative" if zero else "positive"} value')


def schedule(events, capacities, slots=None):
    """Replay a DAG with max-min fair progress and shared finite resources."""
    slots=slots or {}
    occupied={k:0 for k in slots}
    for k,v in slots.items():
        if not isinstance(v,int) or v<1:raise ValueError('slot count must be a positive integer')
    for k, v in capacities.items(): positive(v, k)
    by_name = {e.name: e for e in events}
    if len(by_name) != len(events): raise ValueError('duplicate event name')
    for e in events:
        positive(e.minimum_ms,e.name+'/minimum duration',zero=True)
        for key,count in e.slots.items():
            if key not in slots or not isinstance(count,int) or not 1<=count<=slots[key]:raise ValueError('invalid slot request')
        if any(d not in by_name or d == e.name for d in e.after):
            raise ValueError('missing/self dependency: ' + e.name)
        for k, v in e.work.items():
            if k not in capacities: raise ValueError('unknown resource: ' + k)
            positive(v, e.name + '/' + k, zero=True)
    pending, active, done, start = dict(by_name), {}, {}, {}
    successors={k:[] for k in by_name}
    waiting={e.name:len(set(e.after)) for e in events}
    for e in events:
        for dependency in set(e.after):successors[dependency].append(e.name)
    ready=[e for e in events if not waiting[e.name]]
    now = 0.0
    def complete(name):
        done[name]=now
        for successor in successors[name]:
            waiting[successor]-=1
            if not waiting[successor]:ready.append(by_name[successor])
    while pending or active:
        # Resolve chains of zero-duration markers before advancing the clock.
        while True:
            if not ready: break
            batch=ready[:];ready.clear()
            admitted=False
            for e in batch:
                if any(occupied[k]+v>slots[k] for k,v in e.slots.items()):
                    ready.append(e);continue
                admitted=True
                del pending[e.name]; start[e.name] = now
                if any(e.work.values()) or e.minimum_ms:
                    active[e.name] = 1.0
                    for k,v in e.slots.items():occupied[k]+=v
                else: complete(e.name)
            if not admitted:break
        if not active:
            if pending: raise ValueError('cyclic event dependencies')
            break
        rates = {k: 0.0 for k in active}
        remaining = dict(capacities)
        unfrozen = set(active)
        while unfrozen:
            demand = {r: sum(by_name[k].work.get(r, 0) for k in unfrozen) for r in capacities}
            limits = {r: remaining[r] / d for r, d in demand.items() if d > 0}
            limits.update({('timer',k):max(0,1/by_name[k].minimum_ms-rates[k])
                           for k in unfrozen if by_name[k].minimum_ms})
            step = max(0.0, min(limits.values()))
            for k in unfrozen: rates[k] += step
            for r, d in demand.items(): remaining[r] = max(0.0, remaining[r] - step*d)
            saturated = {r for r in limits if limits[r] <= step + max(1e-15, step*1e-10)}
            frozen = {k for k in unfrozen if ('timer',k) in saturated or
                      any(by_name[k].work.get(r, 0) > 0 for r in saturated)}
            if not frozen: raise ValueError('resource allocation did not converge')
            unfrozen -= frozen
        dt = min(active[k]/rates[k] for k in active if rates[k] > 0)
        now += dt
        for k in list(active):
            active[k] -= rates[k]*dt
            if active[k] <= 1e-9:
                complete(k); del active[k]
                for r,v in by_name[k].slots.items():occupied[r]-=v
    loads = {r: sum(e.work.get(r, 0) for e in events)/c for r,c in capacities.items()}
    earliest = {}
    for k in sorted(done, key=done.get):
        e = by_name[k]
        earliest[k] = max((earliest[d] for d in e.after), default=0) + max(
            [e.minimum_ms,*(v/capacities[r] for r,v in e.work.items())])
    lower = max([0, *loads.values(), *earliest.values()])
    return dict(milliseconds=now, lower_bound_ms=lower, resource_service_ms=loads,
                events=[dict(name=e.name, category=e.category, start_ms=start[e.name], end_ms=done[e.name]) for e in events])


class ByteCache:
    """Deterministic byte-bounded LRU. Wanted entries cannot be evicted mid-group."""
    def __init__(self, capacity, initial=()):
        positive(capacity, 'cache capacity', zero=True)
        self.capacity, self.used, self.entries = capacity, 0, OrderedDict()
        for key,size in initial:
            if not self.admit(key,size): raise ValueError('initial cache exceeds capacity')

    def contains(self, key):
        if key in self.entries:
            self.entries.move_to_end(key); return True
        return False

    def admit(self, key, size, protected=()):
        positive(size, 'expert size')
        if key in self.entries:
            if self.entries[key] != size: raise ValueError('inconsistent expert size')
            self.entries.move_to_end(key); return True
        if size > self.capacity: return False
        victims = [k for k in self.entries if k not in protected]
        if self.used + size - sum(self.entries[k] for k in victims) > self.capacity: return False
        while self.used + size > self.capacity:
            self.used -= self.entries.pop(victims.pop(0))
        self.entries[key] = size; self.used += size
        return True


def memory_check(hardware, allocations):
    """Count every owned allocation once; callers describe phase peak allocations."""
    result = {}
    pools=set(allocations)|set(hardware.get('cache_bytes',{}))
    for name in sorted(pools):
        parts=dict(allocations.get(name,{}))
        if name in hardware.get('cache_bytes',{}):
            cache=hardware['cache_bytes'][name]
            if 'cache' in parts and parts['cache']!=cache:raise ValueError('cache allocation differs from replay capacity')
            parts['cache']=cache
        if name not in hardware['memory_bytes']: raise ValueError('unknown memory pool: ' + name)
        for key,value in parts.items(): positive(value, name+'/'+key, zero=True)
        used = sum(parts.values()); available = hardware['memory_bytes'][name]
        if used > available: raise ValueError(f'{name} needs {used} bytes, available {available}')
        result[name] = dict(used_bytes=used, free_bytes=available-used, allocations=parts)
    return result


def transfer_events(prefix, size, source, destination, hardware, after=()):
    """Explicit paths include host bounce and NUMA links; never assume peer DMA."""
    positive(size, 'transfer bytes', zero=True)
    path = hardware['paths'].get(source+'->'+destination)
    if path is None: raise ValueError('unmeasured transfer path: '+source+'->'+destination)
    chunk=hardware.get('storage_read_bytes',{}).get(source,0)
    if chunk:
        if destination!='ram' or len(path)!=1:raise ValueError('chunked storage requires a direct RAM path')
        curve=hardware['storage_io'][source].get(str(chunk))
        if curve is None:raise ValueError('unmeasured storage read size')
        if not isinstance(chunk,int) or chunk<4096 or chunk%4096:raise ValueError('unaligned storage read size')
        result=[];stage=path[0]
        for i in range(math.ceil(size/chunk)):
            # Charge full aligned reads, including tail overfetch; cache only the payload.
            result.append(Event(f'{prefix}/chunk{i}',{r:chunk*v for r,v in stage['bytes'].items()},
                                tuple(after),'transfer',stage.get('slots',{}),curve['single_read_ms']))
        result.append(Event(prefix+'/join',{},tuple(e.name for e in result),'transfer'))
        return result
    result = []
    for i,stage in enumerate(path):
        name = f'{prefix}/{i}'
        work = {r:size*multiple for r,multiple in stage['bytes'].items()}
        positive(stage.get('latency_ms',0),'transfer latency',zero=True)
        for value in stage['bytes'].values():positive(value,'traffic multiplier',zero=True)
        if stage.get('latency_ms', 0):
            latency = Event(name+'/submit', {'host':stage['latency_ms']}, tuple(after), 'launch')
            result.append(latency); after=(latency.name,)
        result.append(Event(name,work,tuple(after),'transfer',stage.get('slots',{}),stage.get('minimum_ms',0))); after=(name,)
    return result


def evaluate(document):
    """Versioned measured execution manifest -> timings, bounds and qualification."""
    if document.get('schema') != 'strata.heterogeneous.v1': raise ValueError('unsupported manifest schema')
    hardware = document['hardware']
    memory = memory_check(hardware, document.get('allocations', {}))
    events = [Event(x['name'],x.get('work',{}),tuple(x.get('after',())),x.get('category','compute'),x.get('slots',{}),x.get('minimum_ms',0)) for x in document['events']]
    result = schedule(events,hardware['resources'],hardware.get('slots'))
    tokens = document['output_tokens']; positive(tokens,'output tokens')
    result.update(tok_s=tokens*1000/result['milliseconds'] if result['milliseconds'] else None,
                  optimistic_bound_tok_s=tokens*1000/result['lower_bound_ms'] if result['lower_bound_ms'] else None,
                  memory=memory, quality_qualified=False, provenance=document.get('provenance',{}),
                  scope='Conditional on supplied execution trace, kernel costs, routes and accepted tokens')
    return result


def validate_predictions(records):
    """Holdouts must not be used to fit costs. Failure remains visible."""
    rows=[]
    for r in records:
        positive(r['measured_tok_s'],'measured throughput');positive(r['predicted_tok_s'],'prediction')
        if r['kind'] not in ('decode','prefill'):raise ValueError('invalid timing kind')
        tolerance=.10 if r['kind']=='decode' else .15
        error=r['predicted_tok_s']/r['measured_tok_s']-1
        rows.append(dict(r,error_fraction=error,tolerance=tolerance,within=abs(error)<=tolerance))
    holdouts=[r for r in rows if r.get('split')=='holdout']
    return dict(records=rows,calibrated=bool(holdouts) and all(r['within'] for r in holdouts),
                scope='Only the configurations and workload families represented by the holdouts')


def plan_resident_cache(training_groups, hardware, costs, allocations=None):
    """Greedy disjoint cache candidate, fitted only on explicit training routes.

    Value is saved isolated CPU service per byte. It is not critical-path savings:
    evaluate the returned inventory with held-out routes and full dependencies.
    A caller must supply costs for every observed width, including MTP drafts.
    """
    memory_check(hardware, allocations or {})
    scores={};sizes={}
    for group in training_groups:
        layer=group['layer'];width=len(group['routes'])
        cost=costs[str(layer)][str(width)];size=cost['expert_bytes']
        positive(size,'expert bytes')
        for expert in {e for row in group['routes'] for e in row}:
            if not 0<=expert<cost['experts']:raise ValueError('invalid expert id')
            key=(layer,expert)
            if key in sizes and sizes[key]!=size:raise ValueError('inconsistent expert size')
            sizes[key]=size
            for gpu in ('cuda','hip'):
                cpu=max(cost['cpu_ms'],size/hardware['resources']['ram'])
                service=max(cost[gpu+'_ms'],size/hardware['resources'][gpu+'.vram'])
                positive(cpu,'CPU cost',zero=True);positive(service,'GPU cost',zero=True)
                scores[key,gpu]=scores.get((key,gpu),0)+max(0,cpu-service)
    remaining={g:hardware['cache_bytes'][g] for g in ('cuda','hip')}
    initial={g:[] for g in remaining};assigned=set();saved=0
    ranked=sorted(scores,key=lambda x:(-scores[x]/sizes[x[0]],x[1]!=hardware['primary'],x))
    for key,gpu in ranked:
        if key in assigned or not scores[key,gpu] or sizes[key]>remaining[gpu]:continue
        assigned.add(key);remaining[gpu]-=sizes[key]
        initial[gpu].append([list(key),sizes[key]]);saved+=scores[key,gpu]
    return dict(initial=initial,unallocated_cache_bytes=remaining,
                isolated_service_saving_ms=saved,quality_qualified=False,
                scope='Training-route greedy cache candidate; excludes control/transport and concurrency effects; not a throughput forecast')


def plan_miss_lanes(count, expert_bytes, costs, hardware, width=1,
                   staging_experts=None, fixed_events=(), allocations=None):
    """Price concurrent CPU/GPU miss lanes, rather than each expert in isolation.

    All misses are already in RAM. Caller supplies available transient slots and
    per-expert costs measured at this verification width. Fixed events represent
    resident GPU work competing with the misses. GPU lanes stage their complete
    assigned batch before compute: a conservative schedule, not hidden overlap.
    """
    if not isinstance(count,int) or not 0<=count<=64 or not isinstance(width,int) or width<1:
        raise ValueError('invalid group shape')
    positive(expert_bytes,'expert bytes')
    fixed_events=list(fixed_events)
    staging_experts=staging_experts or {'cuda':0,'hip':0}
    for gpu in ('cuda','hip'):
        if not isinstance(staging_experts[gpu],int) or staging_experts[gpu]<0:
            raise ValueError('invalid GPU staging capacity')
    peak={pool:dict(parts) for pool,parts in (allocations or {}).items()}
    for gpu in ('cuda','hip'):
        staging_bytes=staging_experts[gpu]*expert_bytes
        parts=peak.setdefault(gpu,{})
        if 'miss_staging' in parts and parts['miss_staging']!=staging_bytes:
            raise ValueError('miss staging allocation differs from slot capacity')
        parts['miss_staging']=staging_bytes
    memory=memory_check(hardware,peak)
    for lane in ('cpu','cuda','hip'):positive(costs[lane+'_ms'],lane+' cost',zero=True)
    primary=hardware['primary'];candidates=[]
    for cuda_count in range(min(count,staging_experts['cuda'])+1):
        for hip_count in range(min(count-cuda_count,staging_experts['hip'])+1):
            counts={'cpu':count-cuda_count-hip_count,'cuda':cuda_count,'hip':hip_count}
            events=list(fixed_events);ends=[]
            for lane,n in counts.items():
                if not n:continue
                prefix='miss/'+lane;dependency=()
                destination='ram' if lane=='cpu' else lane
                if destination!=primary:
                    activation=transfer_events(prefix+'/activation',width*4096*4,primary,destination,hardware)
                    events.extend(activation);dependency=(activation[-1].name,) if activation else ()
                size=n*expert_bytes
                if lane!='cpu':
                    upload=transfer_events(prefix+'/weights',size,'ram',lane,hardware)
                    events.extend(upload)
                    if upload:dependency+= (upload[-1].name,)
                work={('cpu' if lane=='cpu' else lane+'.compute'):n*costs[lane+'_ms'],
                      ('ram' if lane=='cpu' else lane+'.vram'):size}
                if lane=='cpu':
                    for resource,fraction in hardware.get('cpu_memory_resources',{}).items():
                        work[resource]=size*fraction
                name=prefix+'/compute';events.append(Event(name,work,dependency,'expert'))
                if destination!=primary:
                    returned=transfer_events(prefix+'/return',width*hardware.get('return_bytes_per_row',{}).get(lane,4096*4),destination,primary,hardware,(name,))
                    events.extend(returned);name=returned[-1].name if returned else name
                ends.append(name)
            # Fixed resident work also belongs to the layer completion barrier.
            events.append(Event('miss/join',{},tuple(ends)+tuple(e.name for e in fixed_events)))
            result=schedule(events,hardware['resources'],hardware.get('slots'))
            candidates.append(dict(counts=counts,milliseconds=result['milliseconds'],
                                   lower_bound_ms=result['lower_bound_ms'],
                                   h2d_bytes=(cuda_count+hip_count)*expert_bytes))
    best=min(candidates,key=lambda r:(r['milliseconds'],r['h2d_bytes']))
    return dict(best=best,candidates=candidates,memory=memory,quality_qualified=False,
                scope='Conditional RAM-miss lane allocation; staging capacity and width-specific costs must be measured')


def trace_replay(groups, hardware, costs, policy='resident-hybrid', initial=None,
                 allocations=None, storage='model', accepted_tokens=None):
    """Replay all actual expert groups, including rejected speculative branches.

    Each group has request, round, phase, layer, routes, and optional mixer_ms.
    Exact expert sizes/costs are indexed by layer and verification width. GPU
    admissions use only current routes. Future-route predictions are not assumed.
    Initial RAM membership is explicit: an omitted RAM inventory is cold storage.
    """
    if policy not in ('resident-hybrid','streamed-gpu','selective-stream'): raise ValueError('unknown policy')
    if accepted_tokens is None: raise ValueError('actual accepted output count is required')
    groups=list(groups)
    requests={g['request'] for g in groups}
    if len(requests)>1:raise ValueError('replay one request; do not share cache state across repetitions')
    initial=initial or {}; allocations=allocations or {}
    memory_check(hardware,allocations)
    caches={n:ByteCache(v,[(tuple(key),size) for key,size in initial.get(n,())])
            for n,v in hardware['cache_bytes'].items()}
    events=[]; prev=(); counters=dict(cpu_bytes=0,gpu_bytes=0,h2d_bytes=0,disk_bytes=0,groups=0)
    seen=set()
    for number,g in enumerate(groups):
        identity=(g['request'],g['event'])
        if identity in seen:raise ValueError('duplicate execution group')
        seen.add(identity)
        rows=g['routes'];width=len(rows)
        if not rows or any(not row or len(set(row)) != len(row) for row in rows):raise ValueError('empty or duplicate routes')
        layer=str(g['layer']); c=costs[layer][str(width)]; size=c['expert_bytes']
        wanted=[(g['layer'],e) for e in sorted({e for row in rows for e in row})]
        if any(e<0 or e>=c['experts'] for _,e in wanted):raise ValueError('invalid expert id')
        tag=f'g{number}';primary=hardware['primary']
        mixer=Event(tag+'/mixer',{primary+'.compute':g.get('mixer_ms',c.get('mixer_ms',0))},prev,'mixer')
        events.append(mixer); base=(mixer.name,); ends=[]
        remote_inputs={}; remote_ends={}; cpu_ends=[]; cpu_input=None
        for key in wanted:
            expert_tag=tag+'/'+str(key[1]);dep=base
            owner=next((n for n in ('cuda','hip') if caches[n].contains(key)),None)
            if owner is None:
                if not caches['ram'].contains(key):
                    transfers=transfer_events(expert_tag+'/read',size,'disk:'+storage,'ram',hardware,dep)
                    events.extend(transfers);dep=(transfers[-1].name,) if transfers else dep
                    counters['disk_bytes']+=sum(e.work.get('disk:'+storage,0) for e in transfers) if 'disk:'+storage in hardware['resources'] else size
                    if not caches['ram'].admit(key,size,wanted):
                        raise ValueError('RAM cache cannot hold active group; describe a staged execution DAG')
                owner='cpu'
                if policy!='resident-hybrid':
                    choices=[]
                    for gpu in ('cuda','hip'):
                        tr=transfer_events('probe',size,'ram',gpu,hardware)
                        tm=schedule(tr,hardware['resources'],hardware.get('slots'))['milliseconds']+c[gpu+'_ms']
                        choices.append((tm,gpu))
                    tm,gpu=min(choices)
                    cpu_ms=max(c['cpu_ms'],size/hardware['resources']['ram'])
                    if policy=='streamed-gpu' or tm<cpu_ms:
                        owner=gpu;tr=transfer_events(expert_tag+'/upload',size,'ram',gpu,hardware,dep)
                        events.extend(tr);dep=(tr[-1].name,) if tr else dep
                        counters['h2d_bytes']+=size
                        if not caches[gpu].admit(key,size,wanted):
                            raise ValueError('GPU cache cannot hold active group; describe a staged execution DAG')
            name=expert_tag+'/compute'
            if owner=='cpu':
                if cpu_input is None:
                    inp=transfer_events(tag+'/activation/cpu',width*4096*4,primary,'ram',hardware,base)
                    events.extend(inp);cpu_input=inp[-1].name if inp else mixer.name
                dep=tuple(dep)+(cpu_input,)
                work={'cpu':c['cpu_ms'],'ram':size};counters['cpu_bytes']+=size
                for resource,fraction in hardware.get('cpu_memory_resources',{}).items():
                    positive(fraction,'CPU memory fraction',zero=True)
                    work[resource]=work.get(resource,0)+size*fraction
            else:
                work={owner+'.compute':c[owner+'_ms'],owner+'.vram':size};counters['gpu_bytes']+=size
                if owner!=primary:
                    if owner not in remote_inputs:
                        inp=transfer_events(tag+'/activation/'+owner,width*4096*4,primary,owner,hardware,base)
                        events.extend(inp);remote_inputs[owner]=inp[-1].name if inp else mixer.name
                    dep=tuple(dep)+(remote_inputs[owner],)
            events.append(Event(name,work,dep,'expert'))
            if owner=='cpu':cpu_ends.append(name)
            elif owner!=primary:remote_ends.setdefault(owner,[]).append(name)
            else:ends.append(name)
        if cpu_ends:
            out=transfer_events(tag+'/cpu-return',width*hardware.get('return_bytes_per_row',{}).get('cpu',4096*4),'ram',primary,hardware,tuple(cpu_ends))
            events.extend(out);ends.extend([out[-1].name] if out else cpu_ends)
        for owner,finished in remote_ends.items():
            reduction=Event(tag+'/reduce/'+owner,{owner+'.compute':c[owner+'_reduce_ms']},tuple(finished),'reduce')
            events.append(reduction)
            out=transfer_events(tag+'/return/'+owner,width*4096*4,owner,primary,hardware,(reduction.name,))
            events.extend(out);ends.append(out[-1].name if out else reduction.name)
        merge=Event(tag+'/join',{'host':c.get('join_ms',0)},tuple(ends),'join')
        events.append(merge);prev=(merge.name,);counters['groups']+=1
    result=schedule(events,hardware['resources'],hardware.get('slots'))
    positive(accepted_tokens,'accepted output count')
    result.update(expert_group_tok_s=accepted_tokens*1000/result['milliseconds'] if result['milliseconds'] else None,
                  expert_group_bound_tok_s=accepted_tokens*1000/result['lower_bound_ms'] if result['lower_bound_ms'] else None,
                  counters=counters,policy=policy,quality_qualified=False,
                  scope='Expert-group replay; add measured draft, dense prefix and sampling events for end-to-end throughput')
    return result


def partition_prefill(primary, worker, fraction, communication_ms=0, mixer_speedup=1, expert_speedup=1):
    """Compatibility adapter for aggregate phase traces, with explicit joins.

    Whole FFN scaling is a sensitivity assumption, not a measured kernel model.
    This adapter deliberately does not add byte traffic to phase times which
    already include staging. Detailed manifests replace these aggregate events.
    """
    if not 0<=fraction<=1:raise ValueError('invalid partition')
    positive(communication_ms,'communication',zero=True)
    positive(mixer_speedup,'mixer speedup');positive(expert_speedup,'expert speedup')
    other={x['layer']:x for x in worker['layers']}
    if primary['tokens']!=worker['tokens'] or set(other)!={x['layer'] for x in primary['layers']}:
        raise ValueError('different batch/layer inventory')
    events=[Event('other',{'primary':primary['unassigned_ms']})];prev=('other',);stages=[]
    for x in primary['layers']:
        tag=str(x['layer']);mix=x['mixer_ms']/mixer_speedup
        events.append(Event(tag+'/mixer',{'primary':mix},prev))
        if x['layer']<3:
            ffn=x['ffn_ms'];events.append(Event(tag+'/join',{'primary':ffn},(tag+'/mixer',)))
        else:
            a=x['ffn_ms']*fraction/expert_speedup;b=other[x['layer']]['ffn_ms']*(1-fraction)/expert_speedup
            join_ms=communication_ms if 0<fraction<1 else 0
            events.extend([Event(tag+'/primary',{'primary':a},(tag+'/mixer',)),
                           Event(tag+'/worker',{'worker':b},(tag+'/mixer',)),
                           Event(tag+'/join',{'host':join_ms},(tag+'/primary',tag+'/worker'))])
            ffn=max(a,b)+join_ms
        stages.append(dict(layer=x['layer'],mixer_ms=mix,ffn_join_ms=ffn));prev=(tag+'/join',)
    result=schedule(events,{'primary':1.,'worker':1.,'host':1.})
    return dict(milliseconds=result['milliseconds'],tok_s=primary['tokens']*1000/result['milliseconds'],stages=stages,
                scope='Aggregate phase sensitivity; staging already included; proportional FFN scaling unvalidated')


def streaming_bounds(cpu_bytes_per_output, expert_bytes, hardware, prompt_tokens=8192):
    """Transport-only ceilings for an explicitly fixed workload; no quality claim.

    The GPU number applies only if every recorded CPU-served weight byte crosses
    PCIe per output, without additional cross-output reuse. It is not a ceiling
    for resident hybrid execution. Cold prefill assumes a full expert-table pass.
    """
    positive(cpu_bytes_per_output,'CPU bytes/output');positive(expert_bytes,'expert table bytes')
    r=hardware['resources'];gpu=r['cuda.h2d']+r['hip.h2d']
    result=dict(scope='Fixed-byte transfer bounds at supplied measured rates; exclude compute, control, protocol overhead and contention',
                all_ram_misses_streamed_to_gpus_tok_s=gpu*1000/cpu_bytes_per_output,
                cold_prefill={name:dict(tok_s=prompt_tokens*rate*1000/expert_bytes,
                                       expert_read_seconds=expert_bytes/(rate*1000),
                                       maximum_disk_MB_per_output_at_50_tok_s=rate/50000)
                              for name,rate in r.items() if name.startswith('disk:')})
    if 'link_encoding_bytes_per_ms' in hardware:
        links=hardware['link_encoding_bytes_per_ms']
        result['encoding_only_bounds']=dict(
            all_ram_misses_streamed_to_gpus_tok_s=(links['cuda']+links['hip'])*1000/cpu_bytes_per_output,
            cold_prefill={name:dict(tok_s=prompt_tokens*rate*1000/expert_bytes,
                                   expert_read_seconds=expert_bytes/(rate*1000))
                          for name,rate in links.items() if name.startswith('disk:')},
            scope='PCIe lane/encoding ceiling only; omits packet overhead, storage limits and all computation')
    return result
