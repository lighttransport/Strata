"""Measure a saved mixed-GPU profile using independent prefills and decode replay.

The native CLI provides one untimed prefill warmup and restores the same target
state before each decode. Cache learning can continue: output repeatability is
recorded separately and must not be inferred from state restoration.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

from gguf_reader import GGUFFile
from glm_chat_stops import chat_stop_ids
from glm_low_memory_bench import parse_log, prefill_repeats_passed
from glm_v100_bench import diagnostic_environment, prompt_ids, save_result, stop_process

ROOT = Path(__file__).resolve().parents[1]


def command(profile, model, prompt, decoder, worker, output, tokens=512, trials=3):
    if profile.get('lock_weights'):
        raise ValueError('profile benchmark requires a shared resident owner, not private locking')
    if profile.get('gpu_layout', 'legacy') != 'legacy' or profile.get('gpu_devices') != '0':
        raise ValueError('profile benchmark requires the CUDA expert-worker layout')
    if profile.get('decode_prefill_cache') or profile.get('decode_cache_adapt'):
        raise ValueError('profile benchmark requires fixed decode cache admission')
    if profile.get('speculative', 'none') not in ('none', 'mtp'):
        raise ValueError('profile benchmark supports greedy or MTP')
    env = profile.get('env', {})
    cmd = ['numactl', '--interleave=all', sys.executable,
           str(ROOT / 'tools/glm_mixed_run.py'), '--primary', str(decoder),
           '--worker', str(worker), '--model', str(model),
           '--primary-device', env.get('STRATA_GLM_MIXED_PRIMARY_DEVICE', '0'),
           '--worker-device', env.get('STRATA_GLM_MIXED_WORKER_DEVICE', '1'),
           '--worker-cache-mib', env.get('STRATA_GLM_MIXED_WORKER_CACHE_MIB', '32000'),
           '--worker-gpu-budget-mib', env.get('STRATA_GLM_MIXED_WORKER_GPU_BUDGET_MIB', '32768'),
           '--combine', env.get('STRATA_GLM_CANON_COMBINE', 'native'),
           '--report', str(output.with_suffix('.memory.json'))]
    for role in ('PRIMARY', 'WORKER'):
        mask = env.get('STRATA_GLM_MIXED_' + role + '_CPUS')
        if mask:
            cmd += ['--' + role.lower() + '-cpus', mask]
    stops = chat_stop_ids(GGUFFile(model).metadata)
    cmd += ['--', '@' + str(prompt), str(tokens), str(profile.get('dense_cache_mib', 4096)),
            str(profile['threads']), '--context=' + str(profile['context']),
            '--prefill-batch=' + str(profile['prefill_batch']),
            '--gpu-budget-mib=' + str(profile['gpu_budget_mib']), '--gpu-devices=0',
            '--cpu-affinity=' + profile.get('cpu_affinity', 'numa'), '--decode-experts=cpu',
            '--prefill-experts=' + profile['prefill_experts'],
            '--prefill-expert-cache-mib=' + str(profile['prefill_expert_cache_mib']),
            '--decode-cache-mib=' + str(profile['decode_cache_mib']),
            '--bench=' + str(trials), '--decode-bench=' + str(trials + 1),
            '--check-prefill-repeats', '--warm-weights']
    if stops:
        cmd.append('--stop-ids=' + ','.join(map(str, stops)))
    if profile.get('decode_graphs'):
        cmd.append('--decode-graphs')
    if profile.get('speculative') == 'mtp':
        cmd += ['--speculative=mtp', '--draft-depth=' + str(profile['draft_depth']),
                '--mtp-experts=gpu']
    return cmd


def summarize(log, stdout, *, tokens=512, trials=3, stops=(), speculative='none', draft_depth=3):
    parsed = parse_log(log)
    fields = stdout.split()
    valid_ids = all(re.fullmatch(r'\d+', field) for field in fields)
    ids = [int(x) for x in fields] if valid_ids else []
    decodes = parsed['decode']
    streams = []
    cursor = 0
    for row in decodes:
        count = row['tokens'] + (row['kind'] == 'single')
        stream = ids[cursor:cursor + count]
        cursor += count
        streams.append(dict(token_ids=stream, generated_tokens=len(stream),
                            decode_ms=row['milliseconds'],
                            decode_tok_s=(len(stream) - 1) * 1000 / row['milliseconds']
                            if row['milliseconds'] > 0 else 0))
    expected_kind = 'mtp' if speculative == 'mtp' else 'single'
    depths = [tuple(map(int, row)) for row in re.findall(r'MTP_DEPTH requested=(\d+) effective=(\d+)', log)]
    depth_ok = speculative != 'mtp' or (len(depths) == trials + 1 and
                                        all(requested == effective == draft_depth for requested, effective in depths))
    complete = bool(valid_ids and len(decodes) == trials + 1 and cursor == len(ids)
                    and len(parsed['prefill']) == trials
                    and depth_ok and all(row['kind'] == expected_kind for row in decodes)
                    and all(len(s['token_ids']) == tokens for s in streams)
                    and all(math.isfinite(s['decode_tok_s']) and s['decode_tok_s'] > 0 for s in streams)
                    and all(math.isfinite(p['tokens_per_second']) and p['tokens_per_second'] > 0
                            for p in parsed['prefill']))
    timed = streams[1:]
    repeat = bool(timed) and all(s['token_ids'] == timed[0]['token_ids'] for s in timed)
    warmup_equal = bool(timed) and streams[0]['token_ids'] == timed[0]['token_ids']
    prefill_ok = prefill_repeats_passed(parsed['prefill_repeat'], trials)
    return dict(complete=complete and prefill_ok, measurements=parsed,
                prefill_reproducible=prefill_ok, repetitions_identical=repeat,
                warmup_matches_timed=warmup_equal,
                full_warmup_trials=streams[:1], trials=timed,
                prefill_trials=parsed['prefill'], validation_trials=[],
                median_decode_tok_s=statistics.median(s['decode_tok_s'] for s in timed) if timed else None,
                median_prefill_tok_s=statistics.median(p['tokens_per_second'] for p in parsed['prefill'])
                if parsed['prefill'] else None, mtp_depths=depths,
                trial_token_sha256=[hashlib.sha256(json.dumps(s['token_ids']).encode()).hexdigest()
                                    for s in streams], stop_ids=list(stops))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('profile', 'model', 'prompt', 'decoder', 'worker', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--tokens', type=int, default=512)
    p.add_argument('--trials', type=int, default=3)
    a = p.parse_args()
    if any(path.exists() for path in [a.output, *[a.output.with_suffix(suffix)
                                               for suffix in ('.log', '.tokens', '.memory.json')]]):
        p.error('output evidence already exists; choose a fresh output path')
    if a.tokens < 2 or not 2 <= a.trials <= 9:
        p.error('need at least two tokens and 2..9 timed trials')
    cfg = json.loads(a.profile.read_text())
    ids = prompt_ids(a.prompt)
    if len(ids) != 8192 or len(ids) + a.tokens > cfg['context']:
        p.error('need exactly 8192 prompt tokens within the configured context')
    if a.model.stat().st_size != cfg['model_bytes']:
        p.error('resident model size differs from the saved profile')
    cmd = command(cfg, a.model.absolute(), a.prompt.resolve(), a.decoder.resolve(),
                  a.worker.resolve(), a.output, a.tokens, a.trials)
    stops = chat_stop_ids(GGUFFile(a.model).metadata)
    env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
    env.update(cfg['env'])
    diagnostics = diagnostic_environment(env)
    record = dict(protocol='checkpoint', configuration=dict(prompt_tokens=len(ids),
                  generated_tokens=a.tokens, profile=cfg), command=cmd,
                  prompt_sha256=hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest(),
                  complete=False, instrumented=bool(diagnostics), diagnostic_environment=sorted(set(diagnostics)),
                  timing_scope='prefill excludes reset; decode restores target state and excludes initial token; '
                  'initialization, shared-model staging, cache admission and warmups excluded')
    save_result(a.output, record)
    with a.output.with_suffix('.log').open('x') as log, a.output.with_suffix('.tokens').open('x') as stream:
        process = subprocess.Popen(cmd, env=env, stdout=stream, stderr=log)
        try:
            code = process.wait()
        finally:
            stop_process(process)
    record.update(summarize(a.output.with_suffix('.log').read_text(),
                            a.output.with_suffix('.tokens').read_text(), tokens=a.tokens,
                            trials=a.trials, stops=stops, speculative=cfg.get('speculative', 'none'),
                            draft_depth=cfg.get('draft_depth', 3)))
    record['returncode'] = code
    record['complete'] &= code == 0
    save_result(a.output, record)
    print(json.dumps({k: record[k] for k in ('complete', 'median_decode_tok_s',
                     'median_prefill_tok_s', 'repetitions_identical', 'instrumented')}), flush=True)
    return 0 if record['complete'] else code or 2


if __name__ == '__main__':
    raise SystemExit(main())
