#!/usr/bin/env python3
"""GLM runner simulator with explicit hardware and measured kernel profiles.

Subcommands: calibrate, validate, decode, prefill, sweep, manifest, probe-plan.
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
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
    args = ap.parse_args(argv)
    return args.func(args) or 0


if __name__ == '__main__':
    sys.exit(main())
