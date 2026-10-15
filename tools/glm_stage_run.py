#!/usr/bin/env python3
"""Experimental Linux layer-split GLM runner: primary decoder (layers [0, K), head, MTP) on one GPU vendor,
stage decoder (layers [K, end)) on the other, sharing one memfd for residual rows. See
include/strata/host/glm_stage.h. The stage process mirrors every state-changing call of the primary."""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm_mixed_run import arm_parent_death, memory_sample  # noqa: E402

# Flags the stage must not see (it never drafts or samples) and flags it gets its own values for.
STAGE_DROP = ('--speculative', '--draft-depth', '--mtp-experts', '--dump-prefill-logits', '--routing-trace')
STAGE_OWN = ('--gpu-budget-mib', '--decode-cache-mib')


def split_args(args, overrides):
    out = []
    skip = False
    for i, value in enumerate(args):
        if skip:
            skip = False
            continue
        key = value.split('=', 1)[0]
        if key in STAGE_DROP or key in STAGE_OWN:
            if '=' not in value and i + 1 < len(args) and not args[i + 1].startswith('--'):
                skip = True
            continue
        out.append(value)
    out += [f'{k}={v}' for k, v in overrides.items() if v is not None]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary', required=True, help='strata-glm-decode for layers [0, K) (HIP or CUDA build)')
    p.add_argument('--stage', required=True, help='strata-glm-decode of the other backend for layers [K, end)')
    p.add_argument('--model', required=True)
    p.add_argument('--boundary', type=int, default=24, help='first stage layer K')
    p.add_argument('--stage-gpu-budget-mib', type=int, default=14336)
    p.add_argument('--stage-decode-cache-mib', type=int)
    p.add_argument('--primary-numa-node', type=int, choices=(0, 1))
    p.add_argument('--stage-numa-node', type=int, choices=(0, 1))
    p.add_argument('--primary-cpus', help='taskset CPU list for the primary (its pool uses only these)')
    p.add_argument('--stage-cpus', help='taskset CPU list for the stage')
    p.add_argument('--pool-spin-us', default='1000', help='pool spin for both processes (they share cores while alternating)')
    p.add_argument('--report', type=Path)
    p.add_argument('--rocm-path', default='/opt/rocm/core-10.1')
    p.add_argument('runner_args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if sys.platform != 'linux':
        p.error('layer split requires Linux')
    args = a.runner_args[1:] if a.runner_args and a.runner_args[0] == '--' else a.runner_args
    if a.report and a.report.exists():
        p.error('report already exists')
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = a.rocm_path + '/lib' + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    backends = [subprocess.check_output([b, '--mixed-backend'], text=True, env=env).strip() for b in (a.primary, a.stage)]
    if set(backends) != {'cuda', 'hip'}:
        p.error('primary and stage must be separately built CUDA and HIP executables')
    sizes = {int(subprocess.check_output([b, '--stage-protocol-size'], text=True, env=env)) for b in (a.primary, a.stage)}
    if len(sizes) != 1:
        p.error('primary and stage stage-protocol sizes differ')
    size = sizes.pop()
    for k in [k for k in env if k.startswith('STRATA_GLM_MIXED_')]:
        del env[k]
    env['STRATA_POOL_SPIN_US'] = a.pool_spin_us
    env.setdefault('STRATA_GLM_GPU_RESERVE_MIB', '1024')
    env.setdefault('STRATA_GLM_BUDGET_RESERVE_MIB', '1024')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    arm_parent_death(os.getppid())
    launcher = os.getpid()

    def child_setup():
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        arm_parent_death(launcher)

    summary = dict(primary=a.primary, stage=a.stage, backends=backends, boundary=a.boundary, shared_mapping_bytes=size,
                   peak_combined_swap_bytes=0)
    processes = []
    started = time.monotonic()
    with tempfile.TemporaryFile(prefix='strata-stage-') as memory:
        memory.truncate(size)
        primary_channel, stage_channel = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        layers = 45
        try:
            stage_env = dict(env, STRATA_GLM_LAYER_RANGE=f'{a.boundary}:{layers}', STRATA_GLM_STAGE_SERVE='1',
                             STRATA_GLM_STAGE_MEMORY_FD=str(memory.fileno()), STRATA_GLM_STAGE_SOCKET_FD=str(stage_channel.fileno()))
            stage_args = split_args(args, {'--gpu-budget-mib': a.stage_gpu_budget_mib, '--decode-cache-mib': a.stage_decode_cache_mib})
            stage_cmd = [a.stage, a.model, *stage_args]
            if a.stage_numa_node is not None:
                stage_cmd = ['numactl', f'--cpunodebind={a.stage_numa_node}', f'--membind={a.stage_numa_node}', *stage_cmd]
            if a.stage_cpus:
                stage_cmd = ['taskset', '-c', a.stage_cpus, *stage_cmd]
            stage = subprocess.Popen(stage_cmd, env=stage_env, pass_fds=(memory.fileno(), stage_channel.fileno()),
                                     start_new_session=True, preexec_fn=child_setup)
            processes.append(stage)
            stage_channel.close()
            primary_env = dict(env, STRATA_GLM_LAYER_RANGE=f'0:{a.boundary}',
                               STRATA_GLM_STAGE_MEMORY_FD=str(memory.fileno()), STRATA_GLM_STAGE_SOCKET_FD=str(primary_channel.fileno()))
            primary_env.pop('STRATA_GLM_STAGE_SERVE', None)
            primary_cmd = [a.primary, a.model, *args]
            if a.primary_numa_node is not None:
                primary_cmd = ['numactl', f'--cpunodebind={a.primary_numa_node}', f'--membind={a.primary_numa_node}', *primary_cmd]
            if a.primary_cpus:
                primary_cmd = ['taskset', '-c', a.primary_cpus, *primary_cmd]
            primary = subprocess.Popen(primary_cmd, env=primary_env, pass_fds=(memory.fileno(), primary_channel.fileno()),
                                       start_new_session=True, preexec_fn=child_setup)
            processes.append(primary)
            primary_channel.close()
            sampled = 0.0
            while primary.poll() is None:
                if time.monotonic() - sampled >= 2:
                    sample = memory_sample(processes, proportional=False)
                    summary['peak_combined_swap_bytes'] = max(summary['peak_combined_swap_bytes'], sample['combined_swap_bytes'])
                    if sample['combined_swap_bytes']:
                        raise RuntimeError('layer-split run swapped; performance measurement rejected')
                    sampled = time.monotonic()
                if stage.poll() is not None and stage.returncode != 0:
                    raise RuntimeError(f'stage failed before the primary finished (code {stage.returncode})')
                time.sleep(.1)
            summary['returncode'] = primary.returncode
            try:
                stage.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
            summary['stage_returncode'] = stage.poll()
            return primary.returncode
        finally:
            summary['elapsed_seconds'] = time.monotonic() - started
            for proc in reversed(processes):
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
            for proc in reversed(processes):
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            if a.report:
                a.report.parent.mkdir(parents=True, exist_ok=True)
                a.report.write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
