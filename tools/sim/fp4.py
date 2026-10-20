"""Uncalibrated GPU W4A16 / optional CPU W4A8 model using the DAG scheduler.

All rates are explicit assumptions, never measurements or vendor peak claims.
No devices, checkpoint tensors, SSH commands, or benchmark processes are accessed.
"""
import copy
import json
import math
from pathlib import Path

try:
    from .heterogeneous import Event, schedule
except ImportError:
    from heterogeneous import Event, schedule

PROFILE = Path(__file__).with_name('data') / 'fp4_theoretical.json'
GIB = 2**30


def positive(x, name, zero=False):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 or (x == 0 and not zero):
        raise ValueError(name + ': expected finite ' + ('nonnegative' if zero else 'positive') + ' number')


def storage(rows, cols, csf_bytes_per_scale):
    """NVFP4 E2M1, E4M3 scale per 16 K values, one F32 global per matrix."""
    weights = rows * cols
    return dict(weights=weights, scales=weights//16,
                host_bytes=weights/2 + weights/16*csf_bytes_per_scale + 4,
                packed_bytes=weights/2 + weights/16 + 4,
                bf16_bytes=weights*2 + 4)


def projection(node, rows, cols, tokens, mode, miss, overlap):
    """Return explicit GPU stages and roofline components, in milliseconds."""
    w = rows*cols
    tiles = math.ceil(tokens/16)
    bw = node['vram_gbps'] * 1e6 * node['availability']
    compute = node['bf16_tflops'] * 1e9 * node['availability']
    unpack = node['dequant_gweights_s'] * 1e6 * node['availability']
    launch = node['launch_us']/1000
    gemm = 2*w*(16*tiles)/compute  # WMMA still executes padded rows for small M
    packed = w*(.5+1/16)
    activation = 4*tokens*(rows+cols)  # F32 buffers, rounded to BF16 in registers
    decode = w*tiles/unpack
    if mode == 'fused':
        memory = (packed*tiles+activation)/bw
        kernel = max(memory, gemm+(1-overlap)*decode, decode)+launch
        stages = [('fused', kernel)]
    else:
        memory = (2*w*tiles+activation)/bw
        stages = []
        if mode == 'staged' or miss:
            stages.append(('dequant', max((packed+2*w)/bw, w/unpack)+launch))
        stages.append(('gemm', max(memory, gemm)+launch))
    return stages, dict(gemm_roofline_ms=gemm, fused_dequant_roofline_ms=decode,
                        weight_tile_reads=tiles, useful_flops=2*w*tokens, issued_wmma_flops=2*w*16*tiles, kernel_ms=sum(t for _, t in stages))


def validate(profile):
    if profile.get('schema') != 'strata.fp4.theoretical.v1':
        raise ValueError('unsupported FP4 profile schema')
    if len(profile.get('nodes', [])) != 3:
        raise ValueError('this experimental model requires three nodes')
    for i, n in enumerate(profile['nodes']):
        for k in ('vram_gbps','bf16_tflops','dequant_gweights_s','gpu_pcie_gbps','hca_pcie_gbps',
                  'pcie_root_gbps','dram_gbps','csf_gscales_s','host_activation_gitems_s','device_activation_gitems_s','cache_gib','ram_budget_gib','availability'):
            positive(n[k],f'nodes.{i}.{k}')
        for k in ('launch_us','copy_us'):
            positive(n[k],f'nodes.{i}.{k}',True)
        if n['availability'] > 1:raise ValueError('availability must be in (0,1]')
    for key in ('01','02','12'):
        link=profile['links'][key]
        positive(link['gbps'],key+'.gbps');positive(link['latency_us'],key+'.latency_us',True)
    positive(profile['csf_bytes_per_scale'],'csf_bytes_per_scale')
    if not 0 <= profile['dequant_overlap'] <= 1:raise ValueError('dequant_overlap must be in [0,1]')


def simulate(profile, tokens=1, counts=(3,3,2), layers=2, mode='fused', topology='direct',
             cache='cold', activation_path='host', hit_rates=None,
             transfer_overlap=False, csf_preexpanded=False, miss_policy="gpu",
             cpu_rates=None, cpu_gpu_overlap=True, cpu_shared_input=False):
    validate(profile)
    for x,name in ((tokens,'tokens'),(layers,'layers')):
        limit=512 if name=='tokens' else 42
        if not isinstance(x,int) or isinstance(x,bool) or not 1<=x<=limit:raise ValueError(name+f' must be 1..{limit}')
    if len(counts)!=3 or any(not isinstance(x,int) or isinstance(x,bool) or x<0 or x>8 for x in counts) or not 1<=sum(counts)<=8:
        raise ValueError('counts must be three nonnegative integers with sum 1..8')
    if mode not in ('fused','staged','cached-bf16') or topology not in ('direct','relay') or cache not in ('cold','warm','mixed') or activation_path not in ('host','device'):
        raise ValueError('invalid FP4 scenario selector')
    if not isinstance(transfer_overlap, bool) or not isinstance(csf_preexpanded, bool):
        raise ValueError('overlap and preexpanded selectors must be boolean')
    if miss_policy not in ('gpu','cpu') or not isinstance(cpu_gpu_overlap,bool) or not isinstance(cpu_shared_input,bool):
        raise ValueError('invalid CPU miss policy or overlap selector')
    if miss_policy == 'cpu':
        if tokens>8:raise ValueError('CPU FP4 model supports token widths 1..8')
        if not isinstance(cpu_rates,(tuple,list)) or len(cpu_rates)!=3:
            raise ValueError('CPU misses require explicit rates for three nodes')
        for i,r in enumerate(cpu_rates):
            if not isinstance(r,dict):raise ValueError('CPU rate entry must be a mapping')
            for key in ('gweights_s','quant_gitems_s','availability'):
                if key not in r:raise ValueError('missing CPU rate: '+key)
                positive(r[key],f'cpu_rates.{i}.{key}')
            if r['availability']>1:raise ValueError('CPU availability must be in (0,1]')
    elif cpu_rates is not None or not cpu_gpu_overlap or cpu_shared_input:
        raise ValueError('CPU rates and CPU overlap control require CPU miss policy')
    if cache == 'mixed':
        if hit_rates is None or len(hit_rates) != 3:
            raise ValueError('mixed cache requires three hit rates')
        for rate in hit_rates:
            positive(rate, 'hit rate', True)
            if rate > 1:raise ValueError('hit rate must be in [0,1]')
    elif hit_rates is not None:
        raise ValueError('hit rates require mixed cache')
    rates = tuple(hit_rates) if cache == 'mixed' else (1.0 if cache == 'warm' else 0.0,)*3
    # Spread integer hits deterministically across the selected expert accesses.
    # No fractional transfers, launch costs, or dequant kernels.
    hits = [[math.floor((j+1)*rates[i]+0.5) > math.floor(j*rates[i]+0.5)
             for j in range(counts[i]*layers)] for i in range(3)]
    nodes=profile['nodes'];shapes=((2048,4096),(2048,4096),(4096,2048))
    sizes=[storage(n,k,profile['csf_bytes_per_scale']) for n,k in shapes]
    packed=sum(s['packed_bytes'] for s in sizes);host=sum(s['host_bytes'] for s in sizes)
    bf16=sum(s['bf16_bytes'] for s in sizes);scales=sum(s['scales'] for s in sizes)
    capacities={};events=[];work_totals={};resource_slots={}
    residency=[]
    for i,node in enumerate(nodes):
        prefix=f'n{i}'
        for key,rate in [('dram',node['dram_gbps']*1e6),('pcie',node['pcie_root_gbps']*1e6),
                         ('gpu_copy',node['gpu_pcie_gbps']*1e6),('hca',node['hca_pcie_gbps']*1e6),
                         ('csf',node['csf_gscales_s']*1e6),('activation',node['host_activation_gitems_s']*1e6)]:
            capacities[prefix+'.'+key]=rate*node['availability']
        capacities[prefix+'.gpu']=1.0;resource_slots[prefix+'.stream']=1
        if miss_policy=='cpu':
            available=cpu_rates[i]['availability']
            capacities[prefix+'.cpu_kernel']=cpu_rates[i]['gweights_s']*1e6*available
            capacities[prefix+'.cpu_quant']=cpu_rates[i]['quant_gitems_s']*1e6*available
            capacities[prefix+'.csf']=node['csf_gscales_s']*1e6*available
            capacities[prefix+'.activation']=node['host_activation_gitems_s']*1e6*available
            resource_slots[prefix+'.cpu_stream']=1
        resident=bf16 if mode=='cached-bf16' else packed
        # Budget includes all requested layers. Cached BF16 retains no packed GPU copy
        # after preparation, but needs one packed expert transiently on cold preparation.
        hit_count=sum(hits[i]);miss_count=len(hits[i])-hit_count
        gpu_misses=miss_count if miss_policy=='gpu' else 0
        gpu_experts=hit_count+gpu_misses
        scratch=(packed if mode=='cached-bf16' and gpu_misses else bf16 if mode=='staged' and (gpu_experts or miss_policy=='gpu') else 0)
        scratch+=tokens*(4096+2*2048)*4 if gpu_experts or miss_policy=='gpu' else 0
        buffers=min(counts[i], 2 if transfer_overlap else 1) if gpu_misses else 0
        if mode=='cached-bf16' and buffers>1:scratch+=(buffers-1)*packed
        needed=(hit_count+buffers)*resident + (scratch if counts[i] else 0)
        fits=needed<=node['cache_gib']*GIB
        if cache in ('warm','mixed') and not fits:raise ValueError(f'node {i}: warm working set or mixed hit set plus streaming buffers exceeds configured cache budget')
        if counts[i] and buffers*resident+scratch>node['cache_gib']*GIB:
            raise ValueError(f'node {i}: one expert plus scratch cannot fit cache budget')
        host_resident=packed if csf_preexpanded else host
        # Conservative host scratch for simultaneously live CPU expert chains:
        # F32 input/gate/up/hidden/output, Q8 values/scales/sums, expanded CSF.
        cpu_scratch=0
        if miss_policy=='cpu' and miss_count:
            cpu_scratch=counts[i]*(tokens*((2*4096+3*2048)*4+(4096+2048)*(1+4/64+2/16))+(0 if csf_preexpanded else scales))
        if counts[i]*layers*host_resident+cpu_scratch>node['ram_budget_gib']*GIB:
            raise ValueError(f'node {i}: selected expert weights exceed RAM budget')
        residency.append(dict(node=node['name'],selected_host_bytes=counts[i]*layers*host_resident,
                              hits=hit_count,misses=miss_count,realized_hit_rate=hit_count/len(hits[i]) if hits[i] else 0,
                              required_gpu_bytes=needed,streaming_buffers=buffers,
                              selected_gpu_bytes=counts[i]*layers*resident,scratch_bytes=scratch if counts[i] else 0,
                              fits_selected_gpu_set=counts[i]*layers*resident+scratch<=node['cache_gib']*GIB,cache_experts=math.floor(max(0,node['cache_gib']*GIB-scratch)/resident)))
        if miss_policy=='cpu':
            residency[-1].update(cpu_experts=miss_count,cpu_scratch_bytes=cpu_scratch,
                required_host_bytes=counts[i]*layers*host_resident+cpu_scratch,
                selected_gpu_bytes=hit_count*resident,
                fits_selected_gpu_set=hit_count*resident+scratch<=node['cache_gib']*GIB)
    for key,link in profile['links'].items():
        for direction in ('forward','reverse'):capacities['link.'+key+'.'+direction]=link['gbps']*1e6
    def add(name,work,after=(),category='compute',minimum=0,slots=None):
        events.append(Event(name,work,tuple(after),category,slots or {},minimum))
        for key,value in work.items():work_totals[key]=work_totals.get(key,0)+value
        return name
    def gpu(name,ms,after,i,category):
        return add(name,{f'n{i}.gpu':ms},after,category,slots={f'n{i}.stream':1})
    def transfer(name,i,j,size,after):
        key=f'{min(i,j)}{max(i,j)}';direction='forward' if i<j else 'reverse'
        demands={'link.'+key+'.'+direction:size}
        for n in (i,j):
            for r in ('hca','pcie','dram'):demands[f'n{n}.{r}']=size
        return add(name,demands,after,'network',profile['links'][key]['latency_us']/1000)
    def copy_gpu(name,i,size,after):
        return add(name,{f'n{i}.{r}':size for r in ('gpu_copy','pcie','dram')},after,'gpu_transfer',nodes[i]['copy_us']/1000)
    end_layer=None;components=[]
    for layer in range(layers):
        finished=[]
        for i,count in enumerate(counts):
            previous=end_layer
            previous_cpu=previous_gpu=end_layer
            expert_finished=[]
            preload_previous=None
            previous_input_ready=None
            shared_cpu_input=None
            for e in range(count):
                miss=not hits[i][layer*count+e]
                cpu_miss=miss and miss_policy=='cpu'
                if miss_policy=='cpu' and cpu_gpu_overlap:
                    previous=previous_cpu if cpu_miss else previous_gpu
                tag=f'L{layer}/N{i}/E{e}';deps=() if previous is None else (previous,)
                path=[0,i] if i and (topology=='direct' or i==1) else [0,1,2] if i==2 else [0]
                arrival=add(tag+'/ready',{},deps,'marker')
                for hop,(src,dst) in enumerate(zip(path,path[1:])):
                    arrival=transfer(tag+f'/input-hop{hop}',src,dst,tokens*4096*4,(arrival,))
                # Assume each worker's weights are already in its own host RAM.
                # Startup distribution and disk loading are explicitly outside scope.
                ready=arrival
                if cpu_miss:
                    slots={f'n{i}.cpu_stream':1}
                    if not csf_preexpanded:
                        ready=add(tag+'/csf-cpu',{f'n{i}.csf':scales,
                            f'n{i}.dram':scales*(profile['csf_bytes_per_scale']+1)},(ready,),'csf_expand',slots=slots)
                    def quant(name,width,dependency):
                        return add(name,{f'n{i}.cpu_quant':tokens*width,
                            f'n{i}.dram':tokens*width*(4+1+4/64+2/16)},(dependency,),'cpu_quant',slots=slots)
                    if cpu_shared_input and shared_cpu_input is not None:
                        ready=add(tag+'/shared-input-q8',{},(ready,shared_cpu_input),'marker')
                    else:
                        ready=quant(tag+'/input-q8',4096,ready)
                        if cpu_shared_input:
                            shared_cpu_input=ready
                    for p,(rows,cols) in enumerate(shapes):
                        if p==2:
                            ready=add(tag+'/silu-cpu',{f'n{i}.activation':tokens*2048,
                                f'n{i}.dram':tokens*2048*12},(ready,),'activation',slots=slots)
                            ready=quant(tag+'/hidden-q8',2048,ready)
                        work=rows*cols*tokens
                        memory=sizes[p]['packed_bytes']+tokens*(cols*(1+4/64+2/16)+rows*4)
                        compute_ms=work/capacities[f'n{i}.cpu_kernel']
                        memory_ms=memory/capacities[f'n{i}.dram']
                        components.append(dict(layer=layer,node=i,expert=e,projection=p,device='cpu',
                            useful_flops=2*work,issued_wmma_flops=0,weight_tile_reads=1,
                            cpu_kernel_roofline_ms=compute_ms,cpu_memory_roofline_ms=memory_ms,
                            kernel_ms=max(compute_ms,memory_ms)))
                        ready=add(tag+f'/P{p}/cpu-fp4',{f'n{i}.cpu_kernel':work,
                            f'n{i}.dram':memory},(ready,),'cpu_fp4',slots=slots)
                    for hop,(src,dst) in enumerate(zip(path[::-1],path[-2::-1])):
                        ready=transfer(tag+f'/output-hop{hop}',src,dst,tokens*4096*4,(ready,))
                    previous_cpu=previous=ready
                    finished.append(ready);expert_finished.append(ready)
                    continue
                if miss:
                    weight_deps=(arrival,)
                    if transfer_overlap:
                        # Routing is known only at this layer's start. At most
                        # one next expert is prefetched; two streaming buffers.
                        weight_deps=tuple(x for x in (end_layer, preload_previous, previous_input_ready,
                                          expert_finished[e-2] if e>=2 else None) if x is not None)
                    weight_ready=add(tag+'/weights-ready',{},weight_deps,'marker')
                    if not csf_preexpanded:
                        weight_ready=add(tag+'/csf', {f'n{i}.csf':scales,f'n{i}.dram':host+packed},(weight_ready,),'csf_expand')
                    weight_ready=copy_gpu(tag+'/weights-h2d',i,packed,(weight_ready,))
                    preload_previous=weight_ready
                    ready=add(tag+'/input-and-weights',{},(arrival,weight_ready),'marker')
                in_size=tokens*4096*4*(2 if activation_path=='host' else 1)
                ready=copy_gpu(tag+'/input-h2d',i,in_size,(ready,))
                previous_input_ready=ready
                # Separate projection launches; gate and up run before SiLU and down.
                for p,(rows,cols) in enumerate(shapes):
                    if p==2:
                        if activation_path=='host':
                            ready=copy_gpu(tag+'/gate-up-d2h',i,tokens*2048*8,(ready,))
                            ready=add(tag+'/silu-host',{f'n{i}.activation':tokens*2048,f'n{i}.dram':tokens*2048*12},(ready,),'activation',slots={f'n{i}.cpu_stream':1} if miss_policy=='cpu' else None)
                            ready=copy_gpu(tag+'/silu-h2d',i,tokens*2048*4,(ready,))
                        else:
                            ms=max(tokens*2048/(nodes[i]['device_activation_gitems_s']*1e6*nodes[i]['availability']),
                                   tokens*2048*12/(nodes[i]['vram_gbps']*1e6*nodes[i]['availability']))+nodes[i]['launch_us']/1000
                            ready=gpu(tag+'/silu-device',ms,(ready,),i,'activation')
                    stages,detail=projection(nodes[i],rows,cols,tokens,mode,miss,profile['dequant_overlap'])
                    components.append(dict(layer=layer,node=i,expert=e,projection=p,**detail))
                    for stage,ms in stages:
                        ready=gpu(tag+f'/P{p}/{stage}',ms,(ready,),i,'gpu_'+stage)
                ready=copy_gpu(tag+'/result-d2h',i,tokens*4096*4,(ready,))
                for hop,(src,dst) in enumerate(zip(path[::-1],path[-2::-1])):
                    ready=transfer(tag+f'/output-hop{hop}',src,dst,tokens*4096*4,(ready,))
                previous_gpu=previous=ready;finished.append(ready);expert_finished.append(ready)
        end_layer=add(f'L{layer}/join',{},finished,'marker')
    result=schedule(events,capacities,resource_slots)
    output=dict(schema='strata.fp4.estimate.v1',model='GLM-5.3-Flash-FP4-Local routed experts',basis='THEORETICAL / UNCALIBRATED',
                scenario=dict(tokens=tokens,experts_per_node=list(counts),layers=layers,mode=mode,topology=topology,cache=cache,activation_path=activation_path,
                              requested_hit_rates=list(rates),transfer_overlap=transfer_overlap,csf_preexpanded=csf_preexpanded),
                moe_milliseconds=result['milliseconds'],per_layer_ms=result['milliseconds']/layers,
                bytes_per_expert=dict(host_csf=host,gpu_packed=packed,gpu_bf16=bf16),residency=residency,
                schedule=result,work_totals=work_totals,projection_components=components,
                assumptions=copy.deepcopy(profile),
                limitations=['MoE only: no attention, shared expert, router, mHC, embedding, head, MTP or quality prediction',
                             'All tokens reuse the specified distinct experts; no routing trace or popularity model',
                             'Weights initially in worker host RAM; no disk or startup model distribution',
                             'Cold cache streams each selected expert once; warm/mixed pinned hits plus miss buffers must fit',
                             'GPU copy and compute may overlap; GPU VRAM contention between DMA and kernels is not modeled',
                             'Per-node compute serial; optional within-layer double-buffered weight prefetch; nodes overlap',
                             'Mixed hits are deterministic evenly spread accesses, with unique pinned hit weights; no measured routing or eviction policy',
                             'Preexpanded CSF replaces compressed host scales; startup expansion time and transient RAM are excluded',
                             'Example rates are assumptions, not measured hardware rates or mining forecasts',
                             'CSF exception cost represented only by average storage ratio and expansion rate',
                             'No native FP4 assumed; every GPU uses software unpack and BF16 GEMM'])

    if miss_policy=='cpu':
        output['limitations']=[x for x in output['limitations'] if not x.startswith(('Cold cache streams','Per-node compute serial'))]
        output['limitations'].append('GPU hits stay resident; misses use CPU host weights; selected weights and CPU scratch are budgeted, not the full checkpoint')
        output['scenario'].update(miss_policy='cpu',cpu_gpu_overlap=cpu_gpu_overlap,cpu_rates=copy.deepcopy(cpu_rates))
        output['limitations'] += [
            'CPU misses use lossy Q8 activations; no W4A16 quality equivalence is implied',
            'CPU effective weight/token rate and quantization rate are explicit assumptions; batch reuse is not calibrated',
            'CPU and GPU lanes may overlap within a layer; shared DRAM/PCIe/HCA contention is scheduled',
            'CPU projections and activation work serialize on one worker-pool slot per node; dispatch overhead is not separately modeled']
        if cpu_shared_input:
            output['scenario']['cpu_shared_input']=True
            output['limitations'].append('Common CPU input Q8 is reused per node/layer; hidden quantization remains per expert. Per-expert network transfers and conservative scratch budget are unchanged; mixed token routes are not modeled')
    return output


def command(args):
    profile=json.loads(Path(args.profile).read_text())
    for item in args.assume:
        key,raw=item.split('=',1);parts=key.split('.');target=profile
        for part in parts[:-1]:target=target[int(part)] if isinstance(target,list) else target[part]
        last=parts[-1]
        if isinstance(target,list):target[int(last)]=float(raw)
        else:
            if last not in target:raise ValueError('unknown assumption: '+key)
            target[last]=float(raw)
    counts=tuple(int(x) for x in args.experts.split(','))
    cpu_rates=None
    if args.cpu_misses:
        if not args.cpu_gweights_s or not args.cpu_quant_gitems_s:
            raise ValueError('--cpu-misses requires --cpu-gweights-s and --cpu-quant-gitems-s')
        weights=[float(x) for x in args.cpu_gweights_s.split(',')]
        quant=[float(x) for x in args.cpu_quant_gitems_s.split(',')]
        available=[float(x) for x in (args.cpu_availability or '1,1,1').split(',')]
        if any(len(x)!=3 for x in (weights,quant,available)):raise ValueError('CPU assumptions require three values each')
        cpu_rates=[dict(gweights_s=w,quant_gitems_s=q,availability=a) for w,q,a in zip(weights,quant,available)]
    elif args.cpu_gweights_s or args.cpu_quant_gitems_s or args.cpu_availability or args.cpu_no_overlap or args.cpu_shared_input:
        raise ValueError('CPU options require --cpu-misses')
    result=simulate(profile,args.tokens,counts,args.layers,args.mode,args.topology,args.cache,args.activation_path,
                    tuple(float(x) for x in args.hit_rates.split(',')) if args.hit_rates else None,
                    args.transfer_overlap,args.csf_preexpanded,
                    "cpu" if args.cpu_misses else "gpu",cpu_rates,not args.cpu_no_overlap,args.cpu_shared_input)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:Path(args.output).write_text(text)
    print(text)


def add_parser(sub):
    p=sub.add_parser('fp4',help='uncalibrated FP4 dequant + BF16 MoE estimate; no hardware execution')
    p.add_argument('--profile',default=str(PROFILE))
    p.add_argument('--assume',action='append',default=[],metavar='KEY=VALUE',help='explicit numeric assumption, e.g. nodes.2.dequant_gweights_s=20')
    p.add_argument('--tokens',type=int,default=1);p.add_argument('--layers',type=int,default=2)
    p.add_argument('--experts',default='3,3,2',help='distinct active experts on each node, reused by every token')
    p.add_argument('--mode',choices=('fused','staged','cached-bf16'),default='fused')
    p.add_argument('--topology',choices=('direct','relay'),default='direct')
    p.add_argument('--cache',choices=('cold','warm','mixed'),default='cold')
    p.add_argument('--hit-rates',help='three fractions in [0,1], requires --cache mixed')
    p.add_argument('--transfer-overlap',action='store_true',help='prefetch next expert weights within each layer')
    p.add_argument('--csf-preexpanded',action='store_true',help='scales expanded once in host RAM before timing')
    p.add_argument('--activation-path',choices=('host','device'),default='host')
    p.add_argument('--cpu-misses',action='store_true',help='execute GPU-cache misses on CPU with explicit rates; opt-in Q8 activations')
    p.add_argument('--cpu-gweights-s',help='three effective CPU weight/token rates in billions per second')
    p.add_argument('--cpu-quant-gitems-s',help='three CPU activation quantization rates in billions of values per second')
    p.add_argument('--cpu-availability',help='three CPU compute availability fractions; defaults to 1,1,1')
    p.add_argument('--cpu-no-overlap',action='store_true',help='serialize CPU misses and GPU hits for comparison')
    p.add_argument('--cpu-shared-input',action='store_true',help='quantize common CPU input once per node/layer; requires --cpu-misses')
    p.add_argument('--output');p.set_defaults(func=command)
