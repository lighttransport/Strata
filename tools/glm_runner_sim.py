#!/usr/bin/env python3
"""GLM runner simulator with explicit hardware and measured kernel profiles.

Subcommands: calibrate, validate, decode, prefill, sweep, manifest, probe-plan, native-fp4.
Defaults retain the TR16 tables. Other machines use explicit --hardware,
--kernels and --params profiles grounded in their own measurements. No quality claim.

  PYTHONPATH=tools python3 tools/glm_runner_sim.py calibrate --output tools/sim/data/glm_runner_params_tr16.json
  PYTHONPATH=tools python3 tools/glm_runner_sim.py validate --params tools/sim/data/glm_runner_params_tr16.json
  PYTHONPATH=tools python3 tools/glm_runner_sim.py sweep --params ... --target-decode 50 --target-prefill 1000
"""
import argparse
import dataclasses
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sim import glm_runner as g  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent / 'sim' / 'data'


def load_table(args):
    table = g.load_json(args.kernels)
    params = args.params
    # The historical fit belongs only to the default TR16 kernel table. A custom
    # hardware profile must never silently inherit its RTX/Radeon coefficients.
    if params is None and pathlib.Path(args.kernels).resolve() == (DATA / 'glm_kernels_tr16.json').resolve():
        if g.load_json(args.hardware).get('name') != 'tr16-runner':
            raise ValueError('custom hardware requires its own kernel profile or explicit fitted parameters')
        params = DATA / 'glm_runner_params_tr16.json'
    if params and pathlib.Path(params).exists():
        fitted = g.load_json(params)
        table = fitted.get('table', fitted)
    expected = table.get('hardware')
    if expected and g.load_json(args.hardware).get('name') != expected:
        raise ValueError('kernel profile and hardware names differ')
    return table


def common(p):
    p.add_argument('--hardware', default=str(DATA / 'hw_tr16_runner.json'))
    p.add_argument('--kernels', default=str(DATA / 'glm_kernels_tr16.json'))
    p.add_argument('--params', help='fitted table; defaults to the TR16 fit only with the default kernel table')
    p.add_argument('--records', default=str(DATA / 'glm_runner_records_tr16.json'))
    p.add_argument('--output')


def write(args, payload):
    text = json.dumps(payload, indent=1) + '\n'
    if args.output:
        pathlib.Path(args.output).write_text(text)
    return text


def provenance(*paths):
    return [dict(path=str(p), sha256=hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()) for p in paths if p and pathlib.Path(p).exists()]


def cmd_calibrate(args):
    if not (args.output or args.params) and pathlib.Path(args.kernels).resolve() != (DATA / 'glm_kernels_tr16.json').resolve():
        raise ValueError('calibrating custom kernels requires --output or --params')
    hw = g.load_json(args.hardware); table = g.load_json(args.kernels)
    records = g.load_json(args.records)['records']
    if args.context_filter is not None:
        records = [r for r in records if (r.get('plan',{}).get('tokens',8192) if r['kind']=='prefill'
                   else r.get('topology',{}).get('context_tokens',8192)) == args.context_filter]
    free = args.free.split(',') if args.free else table['free']
    fitted, score, history = g.calibrate(records, table, hw, free, rounds=args.rounds)
    out = args.output or args.params or str(DATA / 'glm_runner_params_tr16.json')
    pathlib.Path(out).write_text(json.dumps(dict(schema='strata.glm.kernels.v1', fitted=True, score=score, history=history,
                                                 provenance=provenance(args.kernels, args.records, args.hardware),
                                                 fit_context_tokens=args.context_filter, table=fitted), indent=1) + '\n')
    print(f'wrote {out}: mean |log error| {score:.4f}')


def cmd_validate(args):
    hw = g.load_json(args.hardware); table = load_table(args)
    records = g.load_json(args.records)['records']
    if args.add_record:
        records += g.load_json(args.add_record)['records']
    result = g.validate(records, table, hw, context_tokens=args.context_filter)
    for row in result['rows']:
        flag = 'ok' if row['within'] else 'MISS'
        scopes = ' '.join(f"{k}:{100 * v['error_fraction']:+.0f}%" for k, v in row['scope_errors'].items())
        print(f"{row['name']:<34} {row['kind']:<7} {row['split']:<7} measured {row['measured_tok_s']:8.2f} predicted "
              f"{row['predicted_tok_s']:8.2f} {100 * row['error_fraction']:+6.1f}% {flag} {scopes}")
    print('calibrated:', result['calibrated'])
    write(args, result)
    return 0 if result['calibrated'] else 1


def topology_args(p):
    p.add_argument('--topology', choices=('A', 'B'), default='A')
    p.add_argument('--primary', default='hip', help='device ID in the hardware and kernel tables')
    p.add_argument('--worker', default='cuda', help='worker device ID; empty for one GPU')
    p.add_argument('--groups', default='2,2')
    p.add_argument('--combine', choices=('strict', 'device', 'device3'), default='strict')
    p.add_argument('--dispatch', choices=('socket', 'direct'), default='socket')
    p.add_argument('--ownership', choices=('blocking', 'async'), default='blocking')
    p.add_argument('--cpu-lanes', default='single15', help='CPU lane profile key, or node_local8x2 for two legacy lanes')
    p.add_argument('--split-layer', type=int, default=24)
    p.add_argument('--stage-order', choices=('pipeline', 'interleave'), default='pipeline')
    p.add_argument('--context', type=int, default=8192)
    p.add_argument('--primary-slots', type=float, default=15.0)
    p.add_argument('--worker-slots', type=float, default=33.0)
    p.add_argument('--experts-per-layer', type=int, help='actual model expert count (defaults to the kernel profile)')
    p.add_argument('--policy', choices=('lru', 'lfu', 'recorded'), default='lru')
    p.add_argument('--draft-bank', choices=('cpu', 'primary', 'worker'), default='cpu')
    p.add_argument('--shares', help='cpu=0.22,worker=0.46,primary=0.32 (overrides the hit curve)')
    p.add_argument('--accepted', type=float, help='accepted tokens per verify step (default: table)')


def build_topology(args):
    return g.Topology(kind=args.topology, primary=args.primary, worker=args.worker, groups=tuple(int(x) for x in args.groups.split(',')),
                      combine=args.combine, dispatch=args.dispatch, ownership=args.ownership, cpu_lanes=args.cpu_lanes,
                      split_layer=args.split_layer, stage_order=args.stage_order, context_tokens=args.context)


def build_placement(args, table=None):
    return g.Placement(primary_slots=args.primary_slots, worker_slots=args.worker_slots if args.worker else 0.0,
                       experts_per_layer=(args.experts_per_layer if args.experts_per_layer is not None
                                          else g.get(table or {}, 'model.experts_per_layer', 224)),
                       policy=args.policy, draft_bank=args.draft_bank)


def parse_shares(text):
    if not text:
        return None
    return {k: float(v) for k, v in (x.split('=') for x in text.split(','))}


def cmd_decode(args):
    hw = g.load_json(args.hardware); table = load_table(args)
    topo, place = build_topology(args), build_placement(args, table)
    wl = g.synthetic_workload(table, place, topo, rounds=1, accepted_per_round=args.accepted, shares=parse_shares(args.shares))
    result = g.simulate_decode(wl, table, place, topo, hw)
    result['hardware'] = hw.get('name')
    result['provenance'] = provenance(args.hardware, args.kernels, args.params)
    print(json.dumps({k: result[k] for k in ('tok_s', 'ms_per_round', 'bound_tok_s', 'scopes', 'accepted_per_round', 'cpu_gb_per_output')}, indent=1))
    write(args, result)


def plan_args(p):
    p.add_argument('--primary', default='hip')
    p.add_argument('--worker', default='cuda')
    p.add_argument('--fraction', type=float, default=0.5)
    p.add_argument('--tokens', type=int, default=8192)
    p.add_argument('--exchange', choices=('f32', 'f16'), default='f32')
    p.add_argument('--return-mode', choices=('chunks', 'partial'), default='chunks')
    p.add_argument('--stream-overlap', choices=('none', 'mixer', 'layer'), default='none')
    p.add_argument('--resident-groups', type=float, default=0.0)
    p.add_argument('--mixer-speedup', type=float, default=1.0)
    p.add_argument('--gemm-speedup', type=float, default=1.0)
    p.add_argument('--cpu-groups', type=float, default=0.0)


def build_plan(args):
    return g.PrefillPlan(tokens=args.tokens, primary=args.primary, worker=args.worker, primary_fraction=args.fraction,
                         exchange=args.exchange, return_mode=args.return_mode, stream_overlap=args.stream_overlap,
                         resident_groups=args.resident_groups, mixer_speedup=args.mixer_speedup, gemm_speedup=args.gemm_speedup,
                         cpu_groups=args.cpu_groups)


def cmd_prefill(args):
    hw = g.load_json(args.hardware); table = load_table(args)
    result = g.simulate_prefill(table, build_plan(args), hw)
    print(json.dumps({k: result[k] for k in ('tok_s', 'milliseconds', 'lower_bound_ms', 'mixer_chain_ms', 'exposed_ffn_ms')}, indent=1))
    write(args, result)


def cmd_sweep(args):
    hw = g.load_json(args.hardware); table = load_table(args)
    levers = g.load_json(args.levers)['levers']
    if args.only:
        wanted = set(args.only.split(','))
        levers = [l for l in levers if l['name'] in wanted]
    if args.exclude:
        unwanted = set(args.exclude.split(','))
        levers = [l for l in levers if l['name'] not in unwanted]
    state = dict(table=table, topology=build_topology(args), placement=build_placement(args, table), plan=build_plan(args),
                 accepted_per_round=args.accepted)
    result = g.sweep_levers(levers, state, hw, args.target_decode, args.target_prefill, args.allow_lossy)
    result['base_state'] = dict(topology=dataclasses.asdict(state['topology']), placement=dataclasses.asdict(state['placement']),
                                plan=dataclasses.asdict(state['plan']))
    result['provenance'] = provenance(args.kernels, args.params, args.levers, args.hardware)
    print(json.dumps(dict(base=result['base'], final=result['final'], reaches=result['reaches'], excluded_lossy=result['excluded_lossy']), indent=1))
    write(args, result)


def cmd_manifest(args):
    hw = g.load_json(args.hardware); table = load_table(args)
    topo, place = build_topology(args), build_placement(args, table)
    wl = g.synthetic_workload(table, place, topo, rounds=1, accepted_per_round=args.accepted, shares=parse_shares(args.shares))
    b, markers = g.decode_round_events(wl[0], table, place, topo, hw)
    hardware = dict(hw)
    hardware['resources'] = g.resources_for(hw, table, topo)
    document = dict(schema='strata.heterogeneous.v1', hardware=hardware, allocations={}, output_tokens=wl[0].accepted,
                    provenance=dict(scope='One verify step of the GLM runner model (tools/sim/glm_runner.py); expert-step throughput only',
                                    sources=provenance(args.kernels, args.params)),
                    events=[dict(name=e.name, work=e.work, after=list(e.after), slots=e.slots, minimum_ms=e.minimum_ms, category=e.category)
                            for e in b.events])
    text = write(args, document)
    if not args.output:
        print(text[:2000])
    print(f'{len(b.events)} events; evaluate with: python3 tools/sim/glm_sim.py heterogeneous --manifest {args.output or "<file>"}')


PROBES = [
    ('per-slot GPU decode decomposition (nsys/rocprof over 32 verify steps): kernel time of mixer, router/publish, residents, wait-add gaps per slot',
     'nsys profile --trace=cuda -o build-glm-release/runner-sim/decode-nsys <cuda decode cmd --decode-bench=32> ; rocprofv3 --kernel-trace for the HIP primary',
     'build-glm-release/runner-sim/decode-kernels-cuda.json'),
    ('per-slot decode traces, both orientations and singles, 8K and 2K holdout-code',
     'LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib python3 tools/glm_mixed_bench.py --config <reap224 profile> --cases hip-cuda cuda-hip hip cuda '
     '--fractions 0.5 --tokens 512 --trials 1 --repetitions 1 --capture-execution-trace --allow-desktop-gpu '
     '--prompts docs/fixtures/glm_tr16_20261009/prompts/8k/holdout-code.ids docs/fixtures/glm_tr16_20261009/prompts/2k/holdout-code.ids '
     '--output build-glm-release/runner-sim/traces-v1',
     'build-glm-release/runner-sim/traces-v1/results.json'),
    ('worker round trip (socket vs shared-memory flag) and per-request service histogram',
     'build-glm-release/strata-glm-gpu-worker --loopback-bench (new diagnostic mode) ; STRATA_GLM_MIXED_PROFILE_DECODE=1',
     'build-glm-release/runner-sim/worker-rtt-cuda.json'),
    ('NUMA-local expert lane bandwidth, 8 threads per node, alone and concurrently',
     'numactl --cpunodebind=0 --membind=0 build-glm-release/strata-glm-q2-flow-bench ... flow 8 (and node 1; then both at once)',
     'build-glm-release/runner-sim/cpu-lanes.json'),
    ('prefill per-kernel decomposition at 8K on both cards (mixer vs streaming vs dequant vs GEMM)',
     'rocprofv3 --kernel-trace --stats -- <hip decode cmd --bench=2> ; nsys profile --trace=cuda -- <cuda decode cmd>',
     'build-glm-release/runner-sim/prefill-kernels-hip.json'),
    ('host bounce latency and shared-mapping registration (cudaHostRegister/hipHostRegister on one memfd)',
     '/tmp/tr16-probes/probe-{cuda,hip} --sizes 131072,67108864,134217728 --shared-register FD (tools/glm_transfer_probe.cu extension)',
     'build-glm-release/runner-sim/bounce.json'),
    ('draft step cost with CPU vs GPU draft experts',
     '128-token runs with --mtp-experts=cpu and --mtp-experts=gpu; read MTP_TIMING draft_ms per step and the primary slots lost',
     'build-glm-release/runner-sim/draft-step.json'),
]


def cmd_records_from_bench(args):
    from sim import glm_traces
    rows = glm_traces.records_from_bench(args.results)
    for r in rows:
        if args.holdout:
            r['split'] = 'holdout'; r['fit'] = False
        print(f"{r['name']:<48} {r['kind']:<7} {r['measured_tok_s']:8.2f} {r.get('shares', '')}")
    payload = dict(schema='strata.glm.runner.records.v1', provenance=dict(results=str(args.results)), records=rows)
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(payload, indent=1) + '\n')
    if args.merge_into:
        target = pathlib.Path(args.merge_into)
        existing = g.load_json(target)
        names = {(r['name'], r['kind']) for r in existing['records']}
        existing['records'] += [r for r in rows if (r['name'], r['kind']) not in names]
        target.write_text(json.dumps(existing, indent=1) + '\n')
        print(f'merged into {target}: {len(existing["records"])} records')


def cmd_probe_plan(args):
    for i, (what, command, receipt) in enumerate(PROBES, 1):
        exists = pathlib.Path(receipt).exists()
        print(f"{i}. {what}\n   command: {command}\n   receipt: {receipt} [{'present' if exists else 'missing'}]")


def link_from(text):
    from sim import hw as hwmod
    l = hwmod.parse_link(text)
    return dict(name=l.name, rtt_us=l.rtt_us, gbps=l.gbps, exposed_us=l.exposed_us)


def cmd_multinode(args):
    """TR16 (primary GPU, model, CPU lane) + a second node holding the worker GPU (and optionally a CPU lane) over
    a network link. Searches the worker's expert slots and node-B CPU share for decode, the prefill fraction for
    prefill (1.0 = the worker is not used in prefill), and prints raw and anchored predictions per link."""
    base = g.load_json(args.hardware); table = load_table(args)
    shares = parse_shares(args.shares)

    def decode(hw, worker, slots, cpu_share=0.0):
        topo = g.Topology(primary=args.primary, worker=worker, context_tokens=args.context, worker_cpu_share=cpu_share)
        place = g.Placement(primary_slots=args.primary_slots, worker_slots=slots if worker else 0.0)
        wl = g.synthetic_workload(table, place, topo, rounds=1, accepted_per_round=args.accepted, shares=shares)
        r = g.simulate_decode(wl, table, place, topo, hw)
        return dict(tok_s=r['tok_s'], ms_per_round=r['ms_per_round'], worker_slots=slots if worker else 0.0,
                    worker_cpu_share=cpu_share, scopes={k: round(v, 2) for k, v in r['scopes'].items()})

    def prefill(hw, worker, fraction):
        plan = g.PrefillPlan(tokens=args.tokens, primary=args.primary, worker=worker if fraction < 1 else '',
                             primary_fraction=fraction, exchange='f16', return_mode='partial', stream_overlap='mixer')
        r = g.simulate_prefill(table, plan, hw)
        return dict(tok_s=r['tok_s'], fraction=fraction)

    rows = []
    single_d = decode(base, '', 0.0)
    single_p = prefill(base, '', 1.0)
    rows.append(dict(config=f'{args.primary} alone (node A)', link='-', decode=single_d, prefill=single_p))
    local_d = max((decode(base, args.worker, sl) for sl in args.slots), key=lambda x: x['tok_s'])
    local_p = max((prefill(base, args.worker, f) for f in args.fractions), key=lambda x: x['tok_s'])
    rows.append(dict(config=f'{args.primary} + {args.worker}, one host', link='pcie', decode=local_d, prefill=local_p))
    for text in args.links:
        link = link_from(text)
        hw = g.two_node_hardware(base, link, worker=args.worker)
        gpu = [decode(hw, args.worker, sl) for sl in args.slots]
        both = [decode(hw, args.worker, sl, cs) for sl in args.slots for cs in args.cpu_shares if cs > 0]
        pre = [prefill(hw, args.worker, f) for f in args.fractions + [1.0]]
        forced = max(gpu, key=lambda x: x['tok_s'])
        rows.append(dict(config=f'{args.primary} (A) + {args.worker} on node B', link=link['name'],
                         decode=max([forced, single_d], key=lambda x: x['tok_s']), prefill=max(pre, key=lambda x: x['tok_s']),
                         decode_with_worker=forced, link_constants=link))
        if both:
            rows.append(dict(config=f'... + node-B CPU lane', link=link['name'], decode=max(both, key=lambda x: x['tok_s']),
                             prefill=None, link_constants=link))
    anchor_d = args.anchor_decode / single_d['tok_s'] if args.anchor_decode else None
    anchor_p = args.anchor_prefill / single_p['tok_s'] if args.anchor_prefill else None
    print('decode: best of no worker / worker GPU with the listed slots; slots 0 = the worker is not used.')
    print('prefill: best of the listed fractions; frac 1.00 = the worker is not used. anch. = scaled to the measured')
    print('primary-alone numbers (--anchor-decode/--anchor-prefill).')
    print(f"{'configuration':<38}{'link':<24}{'decode':>8}{'anch.':>8}{'slots':>7}{'cpuB':>6}{'prefill':>9}{'anch.':>8}{'frac':>6}{'forced':>8}")
    for r in rows:
        d, pf = r['decode'], r['prefill']
        print(f"{r['config']:<38}{r['link']:<24}{d['tok_s']:>8.1f}"
              f"{(d['tok_s'] * anchor_d if anchor_d else float('nan')):>8.1f}{d['worker_slots']:>7.0f}{d['worker_cpu_share']:>6.2f}"
              + (f"{pf['tok_s']:>9.0f}{(pf['tok_s'] * anchor_p if anchor_p else float('nan')):>8.0f}{pf['fraction']:>6.2f}" if pf else ' ' * 23)
              + (f"{r['decode_with_worker']['tok_s']:>8.1f}" if 'decode_with_worker' in r else ''))
    write(args, dict(rows=rows, anchor_decode=args.anchor_decode, anchor_prefill=args.anchor_prefill,
                     provenance=provenance(args.kernels, args.params, args.hardware),
                     scope='Throughput model; node-B PCIe and IB QDR constants are estimates; no quality prediction'))


def cmd_split_model(args):
    """A model larger than one node's RAM: the same hardware as one 192 GB node, or as two nodes where the coldest
    experts (the bytes that do not fit node A) live only on node B and are served by its CPU over the link."""
    base = g.load_json(args.hardware); table0 = load_table(args)
    gib = 2 ** 30
    expert_gb = len(g.MOE_LAYERS) * args.experts * g.EXPERT_BYTES[3] * args.expert_scale / 1e9
    model_gb = expert_gb + args.dense_gb
    b_bytes_share = max(0.0, (model_gb - args.node_a_weights_gb) / expert_gb)
    rows = []
    with g.expert_scale(args.expert_scale) as sc:
        table = sc.table(table0)

        def run(hw, worker, primary_slots, worker_slots, b_route_share=0.0):
            topo = g.Topology(primary=args.primary, worker=worker, context_tokens=args.context)
            place = g.Placement(primary_slots=primary_slots, worker_slots=worker_slots if worker else 0.0,
                                experts_per_layer=args.experts)
            cpu_share = g.route_shares(place, table)['cpu']
            if isinstance(b_route_share, dict):
                topo.worker_cpu_share = {l: min(1.0, v / cpu_share) for l, v in b_route_share.items()}
            else:
                topo.worker_cpu_share = min(1.0, b_route_share / cpu_share) if b_route_share else 0.0
            wl = g.synthetic_workload(table, place, topo, rounds=1, accepted_per_round=args.accepted)
            r = g.simulate_decode(wl, table, place, topo, hw)
            return dict(tok_s=r['tok_s'], cpu_share=cpu_share, node_b_cpu_share=topo.worker_cpu_share,
                        cpu_gb_per_token=r['cpu_gb_per_output'])

        ps, ws = args.primary_slots / args.expert_scale, args.worker_slots / args.expert_scale
        per_layer, balanced = {}, {}
        if args.coverage and pathlib.Path(args.coverage).exists():
            # Measured routing counts: the share of each layer's routes that hit its coldest experts (node B's bytes).
            for proj in g.load_json(args.coverage)['projections']:
                if not proj['name'].endswith('.gu'):
                    continue
                layer = int(proj['name'].split('.')[1])
                c = sorted(proj['counts'])
                per_layer[layer] = sum(c[:round(len(c) * b_bytes_share)]) / sum(c)
            # Balanced: node B may hold warmer experts too (same bytes), up to its bandwidth share of the CPU lanes.
            place = g.Placement(primary_slots=ps, worker_slots=ws, experts_per_layer=args.experts)
            lanes = get_lane = table['cpu']['lanes']['single15']['gb_s']
            ideal = g.route_shares(place, table)['cpu'] * args.node_b_flow / (lanes + args.node_b_flow)
            balanced = {l: max(v, ideal) for l, v in per_layer.items()}
        rows.append(('one node, 192 GB RAM, both GPUs', '-', run(base, args.worker, ps, ws)))
        rows.append(('one node, 192 GB RAM, primary GPU only', '-', run(base, '', ps, 0)))
        for text in args.links:
            link = link_from(text)
            for share in args.route_shares:
                hw = g.two_node_hardware(base, link, worker=args.worker, gpu_on_b=False)
                rows.append((f'two nodes: GPUs on A, cold experts on B ({share:.0%} of routes)', link['name'],
                             run(hw, args.worker, ps, ws, share)))
            if per_layer:
                hw = g.two_node_hardware(base, link, worker=args.worker, gpu_on_b=False)
                rows.append((f'two nodes: GPUs on A, coldest {b_bytes_share:.0%} per layer on B', link['name'],
                             run(hw, args.worker, ps, ws, per_layer)))
                rows.append(('two nodes: GPUs on A, B share balanced per layer', link['name'],
                             run(hw, args.worker, ps, ws, balanced)))
            hw = g.two_node_hardware(base, link, worker=args.worker)
            share = args.route_shares[len(args.route_shares) // 2]
            rows.append((f'two nodes: {args.worker} on B, cold experts on B ({share:.0%} of routes)', link['name'],
                         run(hw, args.worker, ps, ws, share)))
    anchor = args.anchor_decode / args.anchor_sim if args.anchor_decode and args.anchor_sim else None
    print(f'model: {model_gb:.0f} GB ({expert_gb:.0f} GB experts at {args.expert_scale}x Q23, {args.experts} per layer); '
          f'node A holds {args.node_a_weights_gb:.0f} GB, so {b_bytes_share:.0%} of expert bytes live only on node B')
    print(f"{'configuration':<62}{'link':<24}{'decode':>8}{'anch.':>8}{'CPU GB/tok':>11}")
    for name, link, r in rows:
        print(f"{name:<62}{link:<24}{r['tok_s']:>8.1f}{(r['tok_s'] * anchor if anchor else float('nan')):>8.1f}{r['cpu_gb_per_token']:>11.2f}")
    write(args, dict(model_gb=model_gb, expert_gb=expert_gb, node_b_bytes_share=b_bytes_share,
                     rows=[dict(config=n, link=l, **r) for n, l, r in rows], anchor=anchor,
                     scope='Throughput model; expert sizes scaled from Q23; no quality prediction'))


def cmd_native_fp4(args):
    from sim import glm_fp4
    result = glm_fp4.analyze(glm_fp4.load_profile(args.profile), cpu_gbps=args.cpu_gbps,
                             hbm_gbps=args.hbm_gbps, tensor_tflops=args.tensor_tflops,
                             h2d_gbps=args.h2d_gbps, target_decode=args.target_decode,
                             target_prefill=args.target_prefill)
    result['provenance'] = []
    for filename in (args.profile, __file__, glm_fp4.__file__):
        path = pathlib.Path(filename).resolve()
        label = str(path.relative_to(glm_fp4.ROOT)) if path.is_relative_to(glm_fp4.ROOT) else '<external-profile>'
        result['provenance'].append(dict(path=label, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    print(json.dumps({key: result[key] for key in ('rates', 'decode', 'prefill')}, indent=2))
    write(args, result)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('native-fp4', help='recorded V100 native FP4 workload and conditional rooflines; no GGUF fit')
    from sim import glm_fp4
    p.add_argument('--profile', default=str(glm_fp4.DEFAULT_PROFILE))
    p.add_argument('--cpu-gbps', type=float, help='assumed streaming bandwidth; default is an optimistic CPU specification peak')
    p.add_argument('--hbm-gbps', type=float, help='per-GPU HBM bandwidth; default is the V100 PCIe specification peak')
    p.add_argument('--tensor-tflops', type=float, help='per-GPU tensor peak; projection-only compute comparison')
    p.add_argument('--h2d-gbps', type=float, nargs=2, help='per-GPU H2D bandwidth scenario')
    p.add_argument('--target-decode', type=float, default=50.0)
    p.add_argument('--target-prefill', type=float, default=800.0)
    p.add_argument('--output')
    p.set_defaults(func=cmd_native_fp4)
    p = sub.add_parser('calibrate'); common(p); p.add_argument('--rounds', type=int, default=3); p.add_argument('--free'); p.add_argument('--context-filter',type=int,choices=(2048,8192)); p.set_defaults(func=cmd_calibrate)
    p = sub.add_parser('validate'); common(p); p.add_argument('--add-record'); p.add_argument('--context-filter',type=int,choices=(2048,8192)); p.set_defaults(func=cmd_validate)
    p = sub.add_parser('decode'); common(p); topology_args(p); p.set_defaults(func=cmd_decode)
    p = sub.add_parser('prefill'); common(p); plan_args(p); p.set_defaults(func=cmd_prefill)
    p = sub.add_parser('sweep'); common(p); topology_args(p)
    for a in plan_args.__code__.co_consts:
        pass
    p.add_argument('--fraction', type=float, default=0.5); p.add_argument('--tokens', type=int, default=8192)
    p.add_argument('--exchange', choices=('f32', 'f16'), default='f32'); p.add_argument('--return-mode', choices=('chunks', 'partial'), default='chunks')
    p.add_argument('--stream-overlap', choices=('none', 'mixer', 'layer'), default='none'); p.add_argument('--resident-groups', type=float, default=0.0)
    p.add_argument('--mixer-speedup', type=float, default=1.0); p.add_argument('--gemm-speedup', type=float, default=1.0)
    p.add_argument('--cpu-groups', type=float, default=0.0)
    p.add_argument('--levers', default=str(DATA / 'glm_levers.json')); p.add_argument('--only'); p.add_argument('--exclude')
    p.add_argument('--target-decode', type=float, default=50.0); p.add_argument('--target-prefill', type=float, default=1000.0)
    p.add_argument('--allow-lossy', action='store_true'); p.set_defaults(func=cmd_sweep)
    p = sub.add_parser('manifest'); common(p); topology_args(p); p.set_defaults(func=cmd_manifest)
    p = sub.add_parser('records-from-bench', help='simulator records from an instrumented glm_mixed_bench results.json')
    common(p); p.add_argument('--results', required=True); p.add_argument('--holdout', action='store_true'); p.add_argument('--merge-into')
    p.set_defaults(func=cmd_records_from_bench)
    p = sub.add_parser('probe-plan'); common(p); p.set_defaults(func=cmd_probe_plan)
    p = sub.add_parser('multinode', help='worker GPU on a second node over a network link (gbe, ib-qdr, ib, 10gbe, rtt_us,gbps,exposed_us)')
    common(p)
    p.add_argument('--links', nargs='+', default=['gbe', 'ib-qdr', 'ib'])
    p.add_argument('--primary', choices=('cuda', 'hip'), default='hip')
    p.add_argument('--worker', choices=('cuda', 'hip'), default='cuda')
    p.add_argument('--context', type=int, default=8192)
    p.add_argument('--tokens', type=int, default=8192, help='prefill prompt tokens')
    p.add_argument('--primary-slots', type=float, default=15.0)
    p.add_argument('--slots', type=float, nargs='+', default=[11.0, 22.0, 33.0, 36.0], help='worker expert slots per layer to try (33 = the measured 12 GB worker cache; 36 = a headless 16 GB card)')
    p.add_argument('--cpu-shares', type=float, nargs='+', default=[0.15, 0.3, 0.45], help="node-B CPU shares of the CPU lane's experts to try")
    p.add_argument('--fractions', type=float, nargs='+', default=[0.5, 0.67, 0.79, 0.9], help='prefill primary fractions to try')
    p.add_argument('--accepted', type=float)
    p.add_argument('--shares')
    p.add_argument('--anchor-decode', type=float, help='measured decode tok/s of the primary alone; scales every row')
    p.add_argument('--anchor-prefill', type=float, help='measured prefill tok/s of the primary alone')
    p.set_defaults(func=cmd_multinode)
    p = sub.add_parser('split-model', help='a model larger than one node: one big node vs cold experts on a second node')
    common(p)
    p.add_argument('--links', nargs='+', default=['ib-qdr', 'gbe'])
    p.add_argument('--primary', choices=('cuda', 'hip'), default='hip')
    p.add_argument('--worker', choices=('cuda', 'hip'), default='cuda')
    p.add_argument('--context', type=int, default=8192)
    p.add_argument('--experts', type=int, default=224, help='routed experts per layer')
    p.add_argument('--expert-scale', type=float, default=1.75, help='expert bytes relative to Q23 (1.75 ~ Q4_K)')
    p.add_argument('--dense-gb', type=float, default=9.8)
    p.add_argument('--node-a-weights-gb', type=float, default=112.0, help="weights node A's RAM can hold")
    p.add_argument('--route-shares', type=float, nargs='+', default=[0.05, 0.10, 0.23],
                   help="share of all routes that hit node B's cold experts (coverage priors: coldest 32%% = 5-23%%, mean 10%%)")
    p.add_argument('--primary-slots', type=float, default=15.0, help='Q23-sized slots; divided by --expert-scale')
    p.add_argument('--worker-slots', type=float, default=33.0)
    p.add_argument('--accepted', type=float)
    p.add_argument('--anchor-decode', type=float, help='measured tok/s of a reference configuration')
    p.add_argument('--anchor-sim', type=float, help='simulated tok/s of that same configuration')
    p.add_argument('--coverage', default='models/glm53f/prune/coverage-keep224.json',
                   help='routing counts per expert (coverage prior); per-layer node-B shares come from it')
    p.add_argument('--node-b-flow', type=float, default=26.0, help="node B's expert-row rate, GB/s")
    p.set_defaults(func=cmd_split_model)
    args = ap.parse_args(argv)
    return args.func(args) or 0


if __name__ == '__main__':
    sys.exit(main())
