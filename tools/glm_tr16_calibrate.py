"""Guarded CUDA measurements for TR16; no server or serving defaults are changed.

Examples:
  python3 tools/glm_tr16_calibrate.py --prompt runs/prompts/train-code.ids \
      --output runs/tr16-screen --cases baseline mtp1 aff05 --tokens 128 --trials 1
  python3 tools/glm_tr16_calibrate.py --prompt runs/prompts/train-code.ids \
      --output runs/tr16-final --cases baseline mtp1 --interleave

The first decode repetition is a warm-up. Prefill has its own engine warm-up.
Throughput records never include tracing or parity checks. Existing outputs are refused.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

from glm_low_memory_bench import parse_log

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def missing_decoder_flags(configs, binary):
    """Catch silent no-ops in both explicit cases and saved experimental profiles."""
    flags = ('STRATA_GLM_PREFETCH_NEXT_LAYER', 'STRATA_GLM_PREFETCH_STREAM',
             'STRATA_GLM_MTP_ACCEPT_MARGIN', 'STRATA_GLM_MTP_ACCEPT_TOP_K',
             'STRATA_GLM_MTP_ACCEPT_MAX_SOFT', 'STRATA_GLM_MMVQ_IL', 'STRATA_GLM_MTP_FIRST_MARGIN',
             'STRATA_GLM_MTP_REUSE_DRAFT', 'STRATA_GLM_ROUTE_RENORMALIZE', 'STRATA_GLM_MTP_SYNC_STATE_ONLY',
             'STRATA_GLM_MTP_UNPRUNED', 'STRATA_GLM_MTP_RECURSIVE_BLEND', 'STRATA_GLM_MTP_REPEAT_GUARD',
             'STRATA_GLM_DENSE_Q3', 'STRATA_GLM_TIER_LEARN_UNBIASED', 'STRATA_GLM_MTP_GPU_CACHE', 'STRATA_GLM_CACHE_PROBE')
    defaults = {'STRATA_GLM_MTP_RECURSIVE_BLEND': '1'}
    return [flag for flag in flags
            if any(c.get('env', {}).get(flag, defaults.get(flag, '0')) != defaults.get(flag, '0')
                   for c in configs)
            and flag.encode() not in binary]


def input_identity(config):
    """Record weight-file identity without streaming 100+ GB during calibration."""
    model = Path(config['model'])
    match = re.fullmatch(r'(.*)-(\d{5})-of-(\d{5})\.gguf', model.name)
    paths = [model]
    if match:
        paths = [model.with_name(f'{match[1]}-{i:05d}-of-{match[3]}.gguf')
                 for i in range(1, int(match[3]) + 1)]
    if config.get('expert_pack'):
        paths.append(Path(config['expert_pack']))
    weights = []
    for path in paths:
        stat = path.stat()
        weights.append(dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    prior = config.get('env', {}).get('STRATA_GLM_EXPERT_PRIOR')
    return dict(weight_files=weights, expert_prior_sha256=digest(prior) if prior else None,
                note='Weight identity uses size and mtime, not a full content hash.')


def case_config(base, name):
    c = copy.deepcopy(base)
    env = c.setdefault('env', {})
    if name == 'baseline':
        return c
    if name == 'ordinary':
        c['speculative'] = 'none'
    elif re.fullmatch(r'mtp[123]', name):
        c['speculative'], c['draft_depth'] = 'mtp', int(name[-1])
    elif name in ('workers8', 'workers12', 'workers15', 'workers16'):
        c['threads'] = int(name[7:])
    elif name in ('cache3', 'cache48', 'cache55'):
        c['decode_cache_mib'] = {'cache3': 3072, 'cache48': 4800, 'cache55': 5632}[name]
    elif name == 'static':
        env.pop('STRATA_GLM_TIER_ADAPT', None)  # presence-based switch
    elif name == 'unsplit':
        env.pop('STRATA_GLM_SPLIT_VERIFY', None)
    elif name in ('aff02', 'aff05', 'aff08', 'aff10'):
        env['STRATA_GLM_ROUTE_AFFINITY'] = str(int(name[3:]) / 100)
    elif name in ('mmq', 'f16', 'f16-batched'):
        c['prefill_experts'] = name
    elif name == 'f16-buckets':
        c['prefill_experts'] = 'f16-batched'
        env['STRATA_GLM_F16_BUCKETS'] = '1'
        env['STRATA_GLM_F16_BUCKET_STEP'] = '64'
    elif name in ('prefetch8', 'prefetch12', 'prefetch18'):
        env['STRATA_GLM_PREFETCH_GROUPS'] = name[8:]
    elif name == 'numa-after-off':
        env.pop('STRATA_GLM_NUMA_AFTER_PREFILL', None)
    elif name == 'margin2':
        env['STRATA_GLM_MTP_CONTINUATION_MARGIN'] = '2'
    elif name == 'batched-resident':
        env['STRATA_GLM_BATCHED_RESIDENT'] = '1'
    elif name == 'stable-routes':
        env['STRATA_GLM_STABLE_ROUTES'] = '1'
    elif name == 'next-layer-prefetch':
        env['STRATA_GLM_PREFETCH_NEXT_LAYER'] = '1'
    elif name == 'stream-prefetch18':
        env['STRATA_GLM_PREFETCH_STREAM'] = '1'
        env['STRATA_GLM_PREFETCH_GROUPS'] = '18'
    else:
        raise ValueError('unknown case: ' + name)
    return c


def command(c, prompt, tokens, trials):
    cmd = ['numactl', '--interleave=all', str(Path(c['exe']).resolve()), c['model'],
           '@' + str(prompt.resolve()), str(tokens), str(c.get('dense_cache_mib', 4096)),
           str(c['threads']), '--context=' + str(c['context']),
           '--prefill-batch=' + str(c['prefill_batch']), '--gpu-budget-mib=' + str(c['gpu_budget_mib']),
           '--cpu-affinity=' + c.get('cpu_affinity', 'numa'), '--decode-experts=cpu',
           '--bench=' + str(max(2, trials)), '--decode-bench=' + str(trials + 1),
           '--decode-cache-mib=' + str(c['decode_cache_mib']),
           '--prefill-experts=' + c.get('prefill_experts', 'mmq')]
    if c.get('decode_cache_slots'):   # opt-in: pin the tier's slot count (comparisons independent of free VRAM)
        cmd.append('--decode-cache-slots=' + str(int(c['decode_cache_slots'])))
    if c.get('decode_graphs'):
        cmd.append('--decode-graphs')
    if c.get('expert_pack'):
        cmd.append('--expert-pack=' + c['expert_pack'])
    if c.get('speculative') == 'mtp':
        backend = c.get('mtp_experts', 'cpu')
        if backend not in ('cpu', 'gpu'):
            raise ValueError('mtp_experts must be cpu or gpu')
        cmd += ['--speculative=mtp', '--draft-depth=' + str(c['draft_depth']), '--mtp-experts=' + backend]
    if c.get('stop_token_ids'):
        stops = c['stop_token_ids']
        if any(type(token) is not int or token < 0 for token in stops):
            raise ValueError('stop_token_ids must contain nonnegative integer IDs')
        cmd.append('--stop-ids=' + ','.join(map(str, stops)))
    return cmd


def wait_idle(timeout, stable_seconds=10):
    """Wait for a quiet interval without stopping anyone else's process."""
    from glm_q2_run_guard import gpu_compute_processes
    from glm_q2_coding_bench import process_cpu_snapshot
    deadline = time.monotonic() + timeout
    quiet_since = None
    previous = process_cpu_snapshot()
    while True:
        clients = gpu_compute_processes(0)
        now = process_cpu_snapshot()
        seconds = now[0] - previous[0]
        busy = []
        if seconds >= .5:
            for pid, (name, ticks) in now[1].items():
                if pid == os.getpid() or pid not in previous[1]:
                    continue
                usage = (ticks - previous[1][pid][1]) / os.sysconf('SC_CLK_TCK') / seconds * 100
                if usage >= 80:
                    busy.append((pid, name))
        previous = now
        if clients or busy:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since >= stable_seconds:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError('TR16 did not become idle; GPU clients=' + repr(clients) + ', CPU clients=' + repr(busy))
        time.sleep(1)


def summarize(log, stdout, memory, tokens, trials):
    m = parse_log(log)
    ids = [int(x) for x in stdout.split()]
    complete = len(ids) == tokens * (trials + 1) and len(m['decode']) == trials + 1
    complete &= len(m['prefill']) == max(2, trials)
    complete &= all(row['tokens'] == (tokens - 1 if row['kind'] == 'single' else tokens)
                    and row['milliseconds'] > 0 and row['tokens_per_second'] > 0
                    for row in m['decode'])
    streams = [ids[i:i + tokens] for i in range(0, len(ids), tokens)]
    # Exclude the explicit first decode repetition, including its tier warm-up.
    timed = m['decode'][1:]
    m['decode_warmup'] = m['decode'][:1]
    m['decode'] = timed
    m['median_decode_tok_s'] = statistics.median(x['tokens_per_second'] for x in timed) if timed else None
    equal = bool(streams) and all(s == streams[0] for s in streams)
    return dict(complete=bool(complete), clean=bool(complete and memory.get('exit_code') == 0 and not memory.get('rejected')),
                repeated_tokens_equal=equal, measurements=m, output_sha256=hashlib.sha256(stdout.encode()).hexdigest(),
                trial_token_sha256=[hashlib.sha256(json.dumps(s).encode()).hexdigest() for s in streams],
                memory=memory)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT / 'configs/experimental/glm53f-q23-tr16-calibration.json')
    ap.add_argument('--decoder', type=Path)
    ap.add_argument('--prompt', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cases', nargs='+', default=['baseline'])
    ap.add_argument('--tokens', type=int, default=512)
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--prefill-batch', type=int, choices=[2048, 4096, 8192])
    ap.add_argument('--interleave', action='store_true')
    ap.add_argument('--allow-interference', action='store_true', help='diagnostics only; such runs are not quiet calibration')
    ap.add_argument('--require-repeat-equal', action='store_true')
    ap.add_argument('--max-swap-mib', type=int, default=0, help='tolerated process swap for functional runs (timing runs keep 0)')
    ap.add_argument('--timeout', type=int, default=1800)
    ap.add_argument('--wait-idle-seconds', type=int, default=600,
                    help='maximum wait for ten quiet seconds before each case; does not stop other jobs')
    args = ap.parse_args()
    if not 1 <= args.trials <= 9 or not 2 <= args.tokens <= 4096:
        ap.error('trials must be 1..9 and tokens 2..4096 (the first token comes from prefill)')
    base = json.loads(args.config.read_text())
    if args.decoder:
        base['exe'] = str(args.decoder.resolve())
    if args.prefill_batch:
        base['prefill_batch'] = args.prefill_batch
        base['env']['STRATA_GLM_PREFILL_SCRATCH_CAP_MIB'] = '2048' if args.prefill_batch == 8192 else '1024'
    configs = [base, *(case_config(base, name) for name in args.cases)]
    missing = missing_decoder_flags(configs, Path(base['exe']).read_bytes())
    if missing:
        ap.error(', '.join(missing) + ' requires the isolated patched decoder via --decoder')
    for key in base.get('env', {}):
        if key.startswith('STRATA_GLM_CHECK_') or key in ('STRATA_GLM_STEP_TRACE', 'STRATA_GLM_ACTIVATION_TRACE'):
            ap.error('diagnostic instrumentation cannot be used for throughput')
    prompt_tokens = len([x for x in re.split(r'[\s,]+', args.prompt.read_text().strip()) if x])
    if prompt_tokens + args.tokens > base['context']:
        ap.error('prompt and generation exceed context')
    args.output.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    order = ['baseline'] if args.interleave else []
    for name in args.cases:
        order.append(name)
        if args.interleave and name != 'baseline':
            order.append('baseline')
    manifest = dict(workspace_commit=source, source_commit=base.get('source_commit'),
                    source_patch_sha256=digest(ROOT / base['source_patch']) if base.get('source_patch') else None,
                    decoder_sha256=digest(base['exe']), runner_sha256=digest(__file__),
                    prompt_sha256=digest(args.prompt), prompt_tokens=prompt_tokens, cases=order,
                    decode_tokens=args.tokens, timed_trials=args.trials, warmup_decode_trials=1,
                    allow_interference=args.allow_interference, created=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    manifest['inputs'] = input_identity(base)
    manifest['nvidia_smi'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,pcie.link.gen.current,pcie.link.width.current,power.limit,power.default_limit,clocks.current.graphics,clocks.current.memory', '--format=csv'], text=True)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'runner.py').write_bytes(Path(__file__).read_bytes())
    summaries = []
    for index, name in enumerate(order):
        print('WAIT_IDLE', name, flush=True)
        if args.allow_interference:
            print('SKIP_WAIT_IDLE (--allow-interference: functional run, interference is recorded)', flush=True)
        else:
            wait_idle(args.wait_idle_seconds)
        if input_identity(base) != manifest['inputs']:
            raise RuntimeError('model or prior changed during the measurement matrix')
        c = case_config(base, name)
        prefix = (args.output / f'{index:03d}-{name}').resolve()
        prefix.with_suffix('.config.json').write_text(json.dumps(c, indent=2) + '\n')
        env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
        env.update(c.get('env', {}))
        cmd = command(c, args.prompt, args.tokens, args.trials)
        guard = [sys.executable, str(ROOT / 'tools/glm_q2_run_guard.py'), '--output', str(prefix),
                 '--min-gpu-free-mib', '512', '--require-gpu-idle', '--max-swap-mib', str(args.max_swap_mib),
                 '--record-interference' if args.allow_interference else '--require-idle', '--', *cmd]
        prefix.with_suffix('.manifest.json').write_text(json.dumps({**manifest, 'command': cmd, 'config': c}, indent=2) + '\n')
        print('START', prefix.name, flush=True)
        with prefix.with_suffix('.guard.log').open('w') as out:
            run = subprocess.Popen(guard, env=env, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
            try:
                code = run.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                # Guard owns decoder lifetime; let its finally block stop and reap it.
                run.send_signal(2)
                try:
                    run.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    run.kill(); run.wait()
                raise RuntimeError('measurement timed out: ' + str(prefix))
        mempath = prefix.with_suffix('.memory.json')
        memory = json.loads(mempath.read_text()) if mempath.exists() else {'exit_code': code, 'rejected': 'missing guard result'}
        log = prefix.with_suffix('.log').read_text() if prefix.with_suffix('.log').exists() else ''
        stdout = prefix.with_suffix('.stdout').read_text() if prefix.with_suffix('.stdout').exists() else ''
        result = summarize(log, stdout, memory, args.tokens, args.trials)
        result['inputs_unchanged'] = input_identity(base) == manifest['inputs']
        result['clean'] &= result['inputs_unchanged']
        result.update(case=name, config=c, manifest=manifest, quiet=not args.allow_interference, guard_exit_code=code)
        prefix.with_suffix('.result.json').write_text(json.dumps(result, indent=2) + '\n')
        row = dict(case=name, result=str(prefix.with_suffix('.result.json')), clean=result['clean'],
                   repeated_tokens_equal=result['repeated_tokens_equal'],
                   decode_tok_s=result['measurements']['median_decode_tok_s'],
                   prefill_tok_s=result['measurements']['median_prefill_tok_s'])
        summaries.append(row)
        (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
        print('DONE', json.dumps(row), flush=True)
        if code or not result['clean'] or args.require_repeat_equal and not result['repeated_tokens_equal']:
            return code or 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
