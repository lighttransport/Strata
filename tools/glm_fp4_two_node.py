#!/usr/bin/env python3
"""Dedicated native FP4 launcher: exact weights, one CUDA GPU, remote CPU experts.

Connection settings are read from ignored local JSON. Startup transfers use SSH;
expert computation and prefill fetches use the configured ucomm backend (IB by default).
The launcher never deletes previous DS41 staging or unrelated files automatically.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import queue
import resource
import shlex
import signal
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from glm_fp4_layout import inspect, plan, GIB
from glm_fp4_partition import partition, available_memory
from glm_fp4_layout_rules import rules


def ssh_options(config):
    options = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
               '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4']
    if config.get('ssh_cipher'):
        options += ['-c', config['ssh_cipher']]
    if config.get('ssh_data_address'):
        if not config.get('ssh_host_key_alias'):
            raise ValueError('IB SSH override requires the already trusted SSH host-key alias')
        options += ['-o', 'Hostname=' + config['ssh_data_address'], '-o', 'HostKeyAlias=' + config['ssh_host_key_alias']]
    return options


def ssh(config, command, **kwargs):
    alias = config['ssh_alias']
    if not isinstance(alias, str) or alias.startswith('-'):
        raise ValueError('invalid SSH alias')
    return subprocess.run(['ssh', *ssh_options(config), alias, command], check=True, **kwargs)


def remote_script(config, code, **kwargs):
    return ssh(config, shlex.join([config.get('remote_python', 'python3'), '-c', code]), **kwargs)


def make_command(config, ids, steps, mtp, check, trials=1, mtp_baseline=False):
    if type(trials) is not int or not 1 <= trials <= 10: raise ValueError('decode trials must be in 1..10')
    context = int(config.get('context', 9216))
    if type(mtp) is not bool or type(check) is not bool or type(steps) is not int:
        raise ValueError('generation flags must be boolean and tokens must be an integer')
    if not ids or steps < 1 or len(ids) + steps + 8 > context:
        raise ValueError('prompt, output and verification margin exceed context')
    argv = [config['remote_decoder'], '{model}', ','.join(map(str, ids)), str(steps), '4096',
            str(config.get('threads', 14)), '--fp4-layout={layout}', '--gpu-devices=0',
            '--prefill-batch=' + str(config.get('prefill_batch', 2048)),
            '--gpu-budget-mib=' + str(config.get('gpu_budget_mib', 15000)), '--context=' + str(context),
            '--fp4-kernel=reference', '--profile-prefill', '--profile-decode', '--decode-bench=' + str(trials)]
    if config.get('lock_decoder_weights', False): argv.append('--lock-weights')
    if mtp or mtp_baseline:
        argv += ['--speculative=mtp', '--draft-depth=2']
    if mtp_baseline and not mtp: argv.append('--mtp-greedy-control')
    environment = dict(STRATA_GLM_FP4_RAM_FDS='1', STRATA_GLM_FP4_PARTITION='1',
                       STRATA_GLM_FP4_REMOTE=config['compute_address'], STRATA_GLM_FP4_FETCH=config['fetch_address'],
                       UCOMM_BACKEND=config.get('backend', 'ib'), UCOMM_TIMEOUT_MS='60000', STRATA_FORCE_ISA='avx2',
                       CUDA_VISIBLE_DEVICES=config['gpu_uuid'], STRATA_GLM_FP4_REMOTE_CHECK=str(int(check)),
                       STRATA_GLM_MTP_ACCEPT_MARGIN='0', STRATA_GLM_FP4_PREFILL_RESERVE_MIB='1024',
                       STRATA_GLM_FP4_DECODE_RESERVE_MIB='1536', STRATA_GLM_PHASE_FAULTS='1')
    # A dedicated profile: inherited approximation/async options must not leak in.
    for knob in ('CPU_Q8', 'CPU_FUSED', 'ASYNC_UPLOAD', 'REGISTER_WEIGHTS', 'ROUTE_AFFINITY',
                 'ROUTE_AFFINITY_SECONDARY', 'ROUTE_DROP', 'PREFILL_BANK', 'PREFILL_STREAM',
                 'PREFILL_OPERAND_PIPELINE', 'GLOBAL_CACHE', 'CACHE_ADAPT_ROWS', 'KEEP_RESELECT_PREFETCH'):
        environment['STRATA_GLM_FP4_' + knob] = '0'
    if config.get('cpu_compact_cache', False):
        for knob in ('CPU_SPARSE', 'CPU_BORROW', 'CPU_CSF32_LOCAL', 'CPU_SPIN'):
            environment['STRATA_GLM_FP4_' + knob] = '1'
    return dict(argv=argv, environment=environment)



def verify_runs(generated, specs):
    """Require complete, repeatable trials and plain/MTP agreement for matching prompts."""
    if len(generated) != sum(spec['tokens'] * spec['trials'] for spec in specs):
        raise RuntimeError('decoder returned an incomplete token sequence')
    results, references, offset = [], {}, 0
    for spec in specs:
        width, count = spec['tokens'], spec['trials']
        trials = [generated[offset + i * width:offset + (i + 1) * width] for i in range(count)]
        offset += width * count
        if any(trial != trials[0] for trial in trials):
            raise RuntimeError('restored-state decode trials produced different tokens')
        key = (tuple(spec['ids']), width)
        if key in references and trials[0] != references[key]:
            raise RuntimeError('matching plain/MTP requests produced different tokens')
        references[key] = trials[0]
        results.append(dict(prompt_tokens=len(spec['ids']), output_tokens=width, trials=count,
                            mtp=spec['mtp'], mtp_allocation_baseline=spec.get('mtp_baseline', False),
                            check_remote=spec['check_remote'], repeated_tokens_identical=True))
    return results




def verify_latest(generated, specs):
    """Fail at the first divergent token, before completing expensive trials."""
    if not generated: return
    index, offset, prior = len(generated) - 1, 0, []
    for run, spec in enumerate(specs):
        size = spec['tokens'] * spec['trials']
        if index < offset + size:
            trial, position = divmod(index - offset, spec['tokens'])
            if trial and generated[index] != generated[offset + position]:
                raise RuntimeError(f'decode repeat differs at run={run} trial={trial} token={position}')
            for start, previous in prior:
                if previous['ids'] == spec['ids'] and previous['tokens'] == spec['tokens']:
                    if generated[index] != generated[start + position]:
                        raise RuntimeError(f'plain/MTP output differs at run={run} token={position}')
            return
        prior.append((offset, spec)); offset += size
    raise RuntimeError('decoder returned extra tokens')


def wait_worker_ready(worker, log_path, timeout=600):
    deadline = time.monotonic() + timeout
    while True:
        if worker.poll() is not None:
            raise RuntimeError('CPU worker exited before RAM warmup; see worker.stderr')
        with log_path.open('rb') as log:
            if any(line.startswith(b'FP4_WORKER ready ') and b'no_GPU=1' in line
                   for line in log.read(64 << 10).splitlines()):
                return
        if time.monotonic() >= deadline:
            raise TimeoutError('CPU worker RAM warmup timed out; see worker.stderr')
        time.sleep(.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt', default='Explain how virtual memory works.')
    parser.add_argument('--tokens', type=int, default=16)
    parser.add_argument('--trials', type=int, default=1, help='1..10 restored-state trials; use trial zero as warmup')
    parser.add_argument('--prompt-file', type=Path)
    parser.add_argument('--suite-file', type=Path, help='JSON list of up to 16 generation requests sharing sealed partitions')
    parser.add_argument('--no-mtp', action='store_true')
    parser.add_argument('--check-remote', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(sig, interrupted)
    config = json.loads(args.config.read_text())
    if args.output.exists(): raise FileExistsError('output directory must be new')
    args.output.mkdir(parents=True)
    output = args.output.resolve()
    source = Path(config['model']).resolve()
    root = Path(__file__).resolve().parents[1]
    remote_root = str(Path(config.get('remote_parent', '/var/tmp')) / ('strata-glm-fp4-' + uuid.uuid4().hex))
    remote_code = remote_root + '-code'
    processes, descriptors, worker_descriptors = [], {}, {}
    receiver = worker = None
    deployed = False
    ready = threading.Event()
    reader_errors = queue.Queue()
    readers = []
    try:
        # Only explicitly named binaries are used. Probe support before moving weights.
        result = ssh(config, shlex.join([config['remote_decoder'], '--fp4-distributed-capability']), capture_output=True, text=True)
        if result.stdout.strip() != '1': raise RuntimeError('remote decoder lacks native distributed FP4 support')
        probe = "import json,pathlib,subprocess,resource; m=pathlib.Path('/proc/meminfo').read_text().splitlines(); print(json.dumps({'available':next(int(x.split()[1])*1024 for x in m if x.startswith('MemAvailable:')),'swap_total':next(int(x.split()[1])*1024 for x in m if x.startswith('SwapTotal:')),'memlock':resource.getrlimit(resource.RLIMIT_MEMLOCK)[0],'gpus':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.free','--format=csv,noheader,nounits'],text=True)}))"
        info = json.loads(remote_script(config, probe, capture_output=True, text=True).stdout)
        candidates = list(csv.reader(info['gpus'].splitlines()))
        wanted = config.get('gpu_uuid')
        selected = next((row for row in candidates if wanted and row[0].strip() == wanted), candidates[0] if len(candidates) == 1 else None)
        if not selected or '5060 Ti' not in selected[1] or int(selected[2]) < 14000:
            raise RuntimeError('an idle RTX 5060 Ti with at least 14000 MiB free is required')
        config['gpu_uuid'] = selected[0].strip()
        tokenizer = ids = None
        specs = []
        if not args.plan_only:
            # Include tokenizer residency in the real admission sample.
            from glm_artifact import NativeTokenizer, load_metadata
            from glm_generate import chat_prompt
            tokenizer = NativeTokenizer(source)
            requests = json.loads(args.suite_file.read_text()) if args.suite_file else [dict(
                prompt=args.prompt_file.read_text() if args.prompt_file else args.prompt,
                tokens=args.tokens, mtp=not args.no_mtp, trials=args.trials, check_remote=args.check_remote)]
            if not isinstance(requests, list) or not 1 <= len(requests) <= 16:
                raise ValueError('expected 1..16 suite requests')
            metadata = load_metadata(source)
            for request in requests:
                if set(request) - {'prompt', 'prompt_file', 'tokens', 'mtp', 'trials', 'check_remote'}:
                    raise ValueError('unknown suite request setting')
                prompt = request.get('prompt', args.prompt)
                if request.get('prompt_file'):
                    prompt = (args.suite_file.parent / request['prompt_file']).read_text()
                ids = tokenizer.encode(chat_prompt(metadata, prompt), parse_special=True)
                specs.append(dict(ids=ids, tokens=request.get('tokens', args.tokens),
                                  mtp=request.get('mtp', not args.no_mtp), trials=request.get('trials', args.trials),
                                  check_remote=request.get('check_remote', args.check_remote)))
        text, records, fingerprint = inspect(source)
        # Worker parsing and the launcher's ownership map coexist during startup.
        worker_allowance = max(0, available_memory() - GIB)
        layout = plan(text, records, fingerprint, min(worker_allowance, int(config.get('worker_gib', 118) * GIB)),
                      min(info['available'], int(config.get('decoder_gib', 60) * GIB)), gpu_fixed=config.get('gpu_fixed', True))
        del records
        worker_lock = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
        if worker_lock >= 0 and worker_lock < layout['worker_bytes'] + (64 << 20):
            raise MemoryError('CPU worker memlock limit cannot retain its RAM partition')
        config['lock_decoder_weights'] = info['memlock'] < 0 or info['memlock'] >= layout['decoder_resident_bytes'] + (64 << 20)
        if info['swap_total'] and not config['lock_decoder_weights']:
            raise MemoryError('decoder needs sufficient memlock allowance or a swap-free host for resident weights')
        summary = {key: value for key, value in layout.items() if key != 'tensors'}
        (output / 'layout.json').write_text(json.dumps(layout, sort_keys=True, separators=(',', ':')))
        print(json.dumps(summary, indent=2), flush=True)
        if args.plan_only: return 0
        mtp_requests = {(tuple(spec['ids']), spec['tokens']) for spec in specs if spec['mtp']}
        for spec in specs:
            spec['mtp_baseline'] = not spec['mtp'] and (tuple(spec['ids']), spec['tokens']) in mtp_requests
        commands = [make_command(config, spec['ids'], spec['tokens'], spec['mtp'], spec['check_remote'],
                                 spec['trials'], spec['mtp_baseline']) for spec in specs]
        command = commands[0] if len(commands) == 1 else dict(runs=commands)
        (output / 'run-request.json').write_text(json.dumps(command, indent=2))
        ssh(config, shlex.join(['mkdir', '-m', '700', '--', remote_code]))
        deployed = True
        scripts = ['glm_fp4_ram_receive.py', 'glm_fp4_partition.py', 'glm_fp4_layout.py', 'glm_fp4_layout_rules.py']
        subprocess.run(['scp', '-q', *ssh_options(config), *[str(root / 'tools' / name) for name in scripts], str(rules()._name),
                        config['ssh_alias'] + ':' + remote_code + '/'], check=True)
        receiver_command = shlex.join(['env', 'STRATA_FP4_LAYOUT_RULES=' + remote_code + '/rules.so',
                                      config.get('remote_python', 'python3'), remote_code + '/glm_fp4_ram_receive.py', '--root', remote_root,
                                      '--run-seconds', str(config.get('timeout_seconds', 3600)), '--wait-seconds',
                                      str(config.get('worker_ready_timeout_seconds', 600) + 900)])
        receiver = subprocess.Popen(['ssh', *ssh_options(config), config['ssh_alias'], receiver_command],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(receiver)
        def stderr_reader():
            with (output / 'decoder.stderr').open('wb') as log:
                for line in receiver.stderr:
                    log.write(line); log.flush()
                    if line.strip() == b'FP4_RAM_READY': ready.set()
        def stdout_reader():
            generated, shown = [], ''
            sizes = [spec['tokens'] for spec in specs for _ in range(spec['trials'])]
            trial = 0
            seen_tokens, next_run = [], 0
            boundaries, count = [], 0
            for spec in specs:
                count += spec['tokens'] * spec['trials']; boundaries.append(count)
            with (output / 'decoder.stdout').open('wb') as log:
                for line in receiver.stdout:
                    log.write(line); log.flush()
                    try:
                        token = int(line.strip())
                    except ValueError as error:
                        reader_errors.put(error)
                        continue
                    generated.append(token)
                    seen_tokens.append(token)
                    verify_latest(seen_tokens, specs)
                    if next_run < len(specs) and len(seen_tokens) == boundaries[next_run]:
                        verify_runs(seen_tokens, specs[:next_run + 1])
                        next_run += 1
                    decoded = tokenizer.decode(generated)
                    print(decoded[len(shown):] if decoded.startswith(shown) else '\n' + decoded, end='', flush=True)
                    shown = decoded
                    if trial < len(sizes) and len(generated) == sizes[trial]:
                        print(flush=True)
                        generated, shown = [], ''
                        trial += 1
        def guarded_reader(reader):
            try: reader()
            except Exception as error: reader_errors.put(error)
        readers = [threading.Thread(target=guarded_reader, args=(reader,), daemon=True)
                   for reader in (stderr_reader, stdout_reader)]
        for reader in readers: reader.start()
        decoder_part = output / 'decoder-partition'
        print('FP4_STAGE decoder: copying and verifying local RAM partition', flush=True)
        partition(source, decoder_part, layout, 'decoder', ram_fds=descriptors)
        print('FP4_STAGE decoder: transferring and verifying remote partition', flush=True)
        with tarfile.open(fileobj=receiver.stdin, mode='w|', dereference=True) as archive:
            names = ['partition.json', *sorted(p.name for p in decoder_part.iterdir() if p.name != 'partition.json')]
            for name in names: archive.add(decoder_part / name, arcname=name, recursive=False)
        receiver.stdin.close()
        deadline = time.monotonic() + 120
        while not ready.wait(.2):
            if receiver.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('remote RAM receiver failed; see decoder.stderr')
        # Release all temporary decoder storage here, before allocating the worker.
        for fd in descriptors.values(): os.close(fd)
        descriptors.clear(); shutil.rmtree(decoder_part)
        worker_part = output / 'worker-partition'
        print('FP4_STAGE worker: copying and verifying CPU RAM partition', flush=True)
        partition(source, worker_part, layout, 'worker', ram_fds=worker_descriptors)
        print('FP4_STAGE worker: mapping and locking RAM before decoder startup', flush=True)
        worker_log = (output / 'worker.stderr').open('wb')
        environment = {key: value for key, value in os.environ.items() if not key.startswith(('STRATA_', 'UCOMM_'))}
        environment.update(UCOMM_BACKEND=config.get('backend', 'ib'), UCOMM_TIMEOUT_MS='60000',
                           CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', STRATA_FORCE_ISA='avx2')
        if config.get('cpu_compact_cache', False):
            environment.update(STRATA_GLM_FP4_CPU_SPARSE='1', STRATA_GLM_FP4_CPU_CSF32_LOCAL='1')
        worker_options = ['--spin', '--prewarm'] if config.get('cpu_compact_cache', False) else []
        worker = subprocess.Popen([config['worker_binary'], '--model=' + str(worker_part), '--ram-fds',
                                   '--root=' + config['compute_address'], '--fetch-root=' + config['fetch_address'], '--threads=' + str(config.get('threads', 14)), '--sessions=' + str(len(specs))] + worker_options,
                                  env=environment, stdout=worker_log, stderr=worker_log)
        processes.append(worker)
        wait_worker_ready(worker, output / 'worker.stderr', config.get('worker_ready_timeout_seconds', 600))
        print('FP4_STAGE inference: CPU worker ready; starting CUDA decoder', flush=True)
        code = "import sys,os,pathlib; p=pathlib.Path(" + repr(remote_root) + "); t=p/'run.tmp'; t.write_text(sys.stdin.read()); os.rename(t,p/'run.json')"
        remote_script(config, code, input=json.dumps(command), text=True)
        deadline = time.monotonic() + int(config.get('timeout_seconds', 3600)) * len(specs)
        while receiver.poll() is None:
            if not reader_errors.empty():
                raise RuntimeError('decoder output validation failed') from reader_errors.get()
            if worker.poll() is not None and worker.returncode != 0:
                raise RuntimeError('CPU worker failed; see worker.stderr')
            if time.monotonic() > deadline: raise TimeoutError('two-node generation timed out')
            time.sleep(.2)
        for reader in readers:
            reader.join(timeout=10)
            if reader.is_alive(): raise RuntimeError('decoder output reader did not finish')
        if not reader_errors.empty(): raise RuntimeError('invalid numeric decoder output') from reader_errors.get()
        if receiver.returncode: raise RuntimeError('decoder failed; see decoder.stderr')
        if worker.wait(timeout=30): raise RuntimeError('worker shutdown failed')
        worker_log.close()
        generated = [int(line) for line in (output / 'decoder.stdout').read_text().splitlines()]
        results = verify_runs(generated, specs)
        (output / 'summary.json').write_text(json.dumps(dict(status='passed', layout=summary, runs=results)))
        return 0
    finally:
        if receiver is not None and receiver.poll() is None:
            try:
                remote_script(config, "import pathlib; p=pathlib.Path(" + repr(remote_root) + "); (p/'cancel').touch() if p.exists() else None", timeout=10)
            except Exception: pass
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
        for fd in [*descriptors.values(), *worker_descriptors.values()]: os.close(fd)
        for name in ('decoder-partition', 'worker-partition'):
            if (output / name).exists(): shutil.rmtree(output / name)
        if deployed:
            try: ssh(config, shlex.join(['rm', '-rf', '--', remote_code]), timeout=10)
            except Exception: pass


if __name__ == '__main__': raise SystemExit(main())
