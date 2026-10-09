"""Run guarded, reproducible B550 calibration cases; never starts the web server.

Run from the isolated B550 source checkout with its HIP build complete.
"""
import argparse
import copy
import hashlib
import json
import pathlib
import re
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=pathlib.Path, default=pathlib.Path('/home/syoyo/work/Strata'))
    ap.add_argument('--decoder', type=pathlib.Path, default=pathlib.Path('build-hip/strata-glm-decode'))
    ap.add_argument('--output', type=pathlib.Path, required=True)
    ap.add_argument('--cases', nargs='+', default=['baseline', 'cache7', 'cache8', 'mtp1', 'mtp2', 'mtp3'])
    ap.add_argument('--tokens', type=int, default=512)
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--prompt', type=pathlib.Path)
    ap.add_argument('--trace', action='store_true')
    ap.add_argument('--fixed-tier', action='store_true')
    ap.add_argument('--require-repeat-equal', action='store_true', help='stop if greedy repetitions differ')
    ap.add_argument('--capture', action='store_true')
    ap.add_argument('--no-graphs', action='store_true')
    ap.add_argument('--checks', nargs='*', choices=['verify','decode_graphs','verify_graphs'], default=[])
    ap.add_argument('--env', action='append', default=[], help='explicit experimental KEY=VALUE override')
    args = ap.parse_args()
    if args.trace and args.trials != 1:
        ap.error('trace one repetition; profiling is separate from throughput')
    root = args.source_root
    cfg = json.loads((root/'configs/glm53f-reap50-q23-b550-60g-9070xt-experimental.json').read_text())
    cfg['exe'] = str(args.decoder.resolve())
    prompt = args.prompt or root/'build-hip-glm-v11/single-2048.ids'
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in args.cases:
        if not re.fullmatch(r'aff(?:[0-3][0-9]|40)', name) and name not in ('baseline', 'cache7', 'cache8', 'cache825', 'mtp1', 'mtp2', 'mtp3', 'mtp1_unsplit', 'mtp2_unsplit', 'threads8', 'threads16', 'lookup3', 'lookup7', 'aff05', 'aff10', 'aff12', 'aff15', 'aff20', 'aff30', 'aff40'):
            ap.error('unknown case ' + name)
        out = (args.output/name).resolve()
        if out.with_suffix('.result.json').exists():
            raise FileExistsError(str(out) + ': use a fresh output directory')
        case = copy.deepcopy(cfg)
        if name != 'baseline':
            case['gpu_budget_mib'] = 15792
            case['env']['STRATA_GLM_GPU_RESERVE_MIB'] = '512'
            case['decode_cache_mib'] = {'cache7':7168, 'cache825':8448}.get(name,8192)
        if name.startswith('aff'):
            case['env']['STRATA_GLM_ROUTE_AFFINITY'] = str(int(name[3:])/100)
            case['env']['STRATA_GLM_TIER_RUNTIME_RESERVE_MIB'] = '256'
        if name.startswith('lookup'):
            case['speculative'] = 'lookup'
            case['lookup_depth'] = int(name[6:])
            case['env']['STRATA_GLM_CACHE_LOOKUP'] = '1'
            case['env']['STRATA_GLM_TIER_RUNTIME_RESERVE_MIB'] = '768'
        if name.startswith('threads'):
            case['threads'] = int(name[7:])
        if name.startswith('mtp'):
            case['speculative'] = 'mtp'
            case['draft_depth'] = int(name[3])
            if name.endswith('_unsplit'):
                case['env'].pop('STRATA_GLM_SPLIT_VERIFY', None)
        if args.fixed_tier:
            case['env'].pop('STRATA_GLM_TIER_ADAPT', None)
        case['decode_checks'] = args.checks
        if args.no_graphs:
            case['decode_graphs'] = False
            case['env'].pop('STRATA_GLM_LAYER_GRAPHS',None)
            case['env'].pop('STRATA_GLM_VERIFY_GRAPHS',None)
        for item in args.env:
            key, sep, value = item.partition('=')
            if not sep or not key.startswith('STRATA_'):
                ap.error('--env requires STRATA_...=value')
            case['env'][key] = value
        if args.capture:
            case['env']['STRATA_GLM_ACTIVATION_TRACE'] = str(out.with_suffix('.capture'))
        if args.trace:
            case['routing_trace'] = str(out.with_suffix('.events.jsonl'))
            case['env']['STRATA_GLM_ROUTING_EVENTS'] = '1'
            case['env']['STRATA_GLM_STEP_TRACE'] = '1'
        config = out.with_suffix('.config.json')
        config.write_text(json.dumps(case, indent=2)+'\n')
        command = [sys.executable, 'tools/glm_low_memory_bench.py', str(config), str(prompt), '--output', str(out),
                   '--ram-gib', '60', '--tokens', str(args.tokens), '--trials', str(args.trials), '--timeout', '1200',
                   '--gpu-capacity-mib', '16304', '--gpu-used-limit-mib', str(14256 if name == 'baseline' else 15792)]
        print('START', name, flush=True)
        with out.with_suffix('.driver.log').open('w') as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        result_path = out.with_suffix('.result.json')
        r = json.loads(result_path.read_text()) if result_path.exists() else {}
        row = dict(case=name, exit_code=run.returncode, clean=r.get('clean', False), rejected=r.get('rejected'),
                   measurement=r.get('measurements',{}), output_sha256=r.get('output_sha256'))
        if r.get('complete'):
            ids = list(map(int,out.with_suffix('.stdout').read_text().split()))
            streams = [ids[i:i+args.tokens] for i in range(0,len(ids),args.tokens)]
            row['repeated_tokens_equal'] = all(x == streams[0] for x in streams)
            row['first_trial_token_sha256'] = hashlib.sha256(json.dumps(streams[0]).encode()).hexdigest()
        summaries.append(row)
        (args.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
        print('DONE', name, json.dumps({k:v for k,v in row.items() if k != 'measurement'}), flush=True)
        status = run.returncode or (0 if row['clean'] else 2)
        if args.require_repeat_equal and not row.get('repeated_tokens_equal', False):
            status = status or 3
        if status:
            print('Case failed a run/quality guard; inspect its logs before scheduling more runs.', flush=True)
            return status
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
