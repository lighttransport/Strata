#!/usr/bin/env python3
"""Experimental Linux GLM expert runner with isolated GPU runtimes."""
import argparse
import ctypes
import sys
import json
import mmap
import struct
from pathlib import Path
import os
import signal
import socket
import subprocess
import tempfile
import time


# This launcher is a separate, single-threaded Linux process. Arm each child
# before exec so even a killed launcher cannot leave a GPU process running.
_libc=ctypes.CDLL(None,use_errno=True)
_prctl=_libc.prctl
_prctl.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong]
_prctl.restype=ctypes.c_int


def arm_parent_death(parent):
    if _prctl(1,signal.SIGTERM,0,0,0):
        raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG failed')
    if os.getppid()!=parent:
        raise SystemExit(143)


def memory_sample(processes, proportional=True):
    pss = swap = 0
    for proc in processes:
        try:
            values = {}
            filename = 'smaps_rollup' if proportional else 'status'
            for line in Path(f'/proc/{proc.pid}/{filename}').read_text().splitlines():
                fields = line.split()
                if len(fields) >= 2 and fields[0] in ('Pss:', 'Swap:', 'VmSwap:', 'VmRSS:'):
                    values[fields[0]] = int(fields[1]) * 1024
            pss += values.get('Pss:', 0)
            swap += values.get('Swap:', values.get('VmSwap:', 0))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return dict(combined_pss_bytes=pss, combined_swap_bytes=swap)


def shared_memory():
    # CUDA registration needs anonymous shared pages on the tested Linux hosts.
    if os.environ.get('STRATA_GLM_MIXED_REGISTER') == '1':
        return os.fdopen(os.memfd_create('strata-mixed'), 'w+b', buffering=0)
    return tempfile.TemporaryFile(prefix='strata-mixed-')


def cuda_child_environments(primary, worker, primary_device, worker_device, env):
    """Resolve allocation-visible ordinals before giving each child one UUID.

    Never resolve indices through nvidia-smi: its list is outside CUDA's Slurm
    visibility mask. Both binaries must agree on the visible device identities.
    """
    if primary_device is None or worker_device is None:
        raise ValueError('CUDA+CUDA requires --primary-device and --worker-device')
    if primary_device < 0 or worker_device < 0 or primary_device == worker_device:
        raise ValueError('primary and worker devices must be distinct nonnegative visible CUDA ordinals')
    views = [json.loads(subprocess.check_output([binary, '--device-info'], text=True, env=env))
             for binary in (primary, worker)]
    def select(devices, ordinal):
        matches = [d for d in devices if d.get('ordinal') == ordinal]
        if len(matches) != 1:
            raise ValueError('CUDA device is outside the allocation-visible device list')
        d = matches[0]
        if not isinstance(d.get('uuid'), str) or not d['uuid'].startswith('GPU-') or not d.get('pci_bus_id'):
            raise ValueError('CUDA device report lacks a GPU UUID or PCI identity')
        return d
    selected = [select(views[0], primary_device), select(views[0], worker_device)]
    for ordinal, device in zip((primary_device, worker_device), selected):
        other = select(views[1], ordinal)
        if any(other[k] != device[k] for k in ('uuid', 'pci_bus_id')):
            raise ValueError('primary and worker disagree on allocation-visible CUDA device identities')
    if selected[0]['uuid'] == selected[1]['uuid'] or selected[0]['pci_bus_id'] == selected[1]['pci_bus_id']:
        raise ValueError('primary and worker resolve to the same physical GPU')
    children = [dict(env, CUDA_VISIBLE_DEVICES=d['uuid']) for d in selected]
    return children[0], children[1], selected


def cpu_mask(value, allowed=None):
    """Parse explicit role masks without widening the launcher's allowed CPUs."""
    allowed = set(os.sched_getaffinity(0) if allowed is None else allowed)
    cpus = set()
    for item in value.split(','):
        bounds = item.split('-')
        if len(bounds) not in (1, 2) or any(not x.isdigit() for x in bounds):
            raise ValueError('invalid CPU list')
        first, last = int(bounds[0]), int(bounds[-1])
        if first > last or last - first > len(allowed):
            raise ValueError('invalid CPU range')
        cpus.update(range(first, last + 1))
    if not cpus or not cpus <= allowed:
        raise ValueError('CPU list is outside the allocation affinity')
    return ','.join(str(c) for c in sorted(cpus))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--combine', choices=('strict','device','native'), default='strict', help='canonical strict/device reduction, or native expert arithmetic on CUDA+CUDA')
    p.add_argument('--primary-worker-mode', action='store_true', help='use existing line-protocol engine mode for evaluations')
    p.add_argument('--primary', required=True, help='strata-glm-decode binary')
    p.add_argument('--worker', required=True, help='strata-glm-gpu-worker binary')
    p.add_argument('--model', required=True)
    p.add_argument('--worker-cache-mib', type=int, default=12288)
    p.add_argument('--report', type=Path, help='write process and shared mapping memory summary')
    p.add_argument('--worker-numa-node', type=int)
    p.add_argument('--primary-device', type=int, help='allocation-visible CUDA ordinal; required with CUDA+CUDA')
    p.add_argument('--worker-device', type=int, help='allocation-visible CUDA ordinal; required with CUDA+CUDA')
    p.add_argument('--primary-cpus', help='CPU list for the primary process and its pool')
    p.add_argument('--worker-cpus', help='CPU list for the worker process')
    p.add_argument('--worker-gpu-budget-mib', type=int, default=14336)
    p.add_argument('--rocm-path', default='/opt/rocm/core-10.1')
    p.add_argument('runner_args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    version = os.environ.get('STRATA_GLM_MIXED_PROTOCOL', '4')
    if version not in ('4', '5'):
        p.error('STRATA_GLM_MIXED_PROTOCOL must be 4 or 5')
    if os.environ.get('STRATA_GLM_MIXED_EXCHANGE') == 'f16' and version != '5':
        p.error('STRATA_GLM_MIXED_EXCHANGE=f16 requires STRATA_GLM_MIXED_PROTOCOL=5')
    if a.report and a.report.exists():
        p.error('report already exists')
    if not 0 < a.worker_cache_mib < a.worker_gpu_budget_mib or a.worker_gpu_budget_mib > (1 << 40):
        p.error('worker budgets must be positive, with cache smaller than GPU budget')
    if a.worker_numa_node is not None and a.worker_numa_node < 0:
        p.error('worker NUMA node must be nonnegative')
    try:
        if a.primary_cpus is not None: a.primary_cpus = cpu_mask(a.primary_cpus)
        if a.worker_cpus is not None: a.worker_cpus = cpu_mask(a.worker_cpus)
    except ValueError as exc:
        p.error(str(exc))
    if sys.platform != 'linux': p.error('mixed GPU runner requires Linux')
    args = a.runner_args
    if args and args[0] == '--':
        args = args[1:]
    for i, value in enumerate(args):
        devices=value.split('=',1)[1] if value.startswith('--gpu-devices=') else (args[i+1] if value=='--gpu-devices' and i+1<len(args) else None)
        if devices is not None and devices!='0':
            p.error('mixed runner requires exactly primary GPU device 0')
    # A private file mapping shares bounded activations/results; GGUF pages stay
    # backed by the same read-only model file in both processes.
    def terminate(*_): raise SystemExit(143)
    signal.signal(signal.SIGTERM, terminate)
    arm_parent_death(os.getppid())
    launcher_pid=os.getpid()
    def child_setup():
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        arm_parent_death(launcher_pid)
    library_env = os.environ.copy()
    library_env['LD_LIBRARY_PATH'] = a.rocm_path+'/lib'+(':'+library_env['LD_LIBRARY_PATH'] if library_env.get('LD_LIBRARY_PATH') else '')
    primary_backend=subprocess.check_output([a.primary,'--mixed-backend'],text=True,env=library_env).strip()
    worker_backend=subprocess.check_output([a.worker,'--backend'],text=True,env=library_env).strip()
    cuda_pair = primary_backend == worker_backend == 'cuda'
    if not cuda_pair and {primary_backend,worker_backend}!={'cuda','hip'}:
        p.error('supported pairs are CUDA+HIP, HIP+CUDA and explicitly placed CUDA+CUDA')
    placement = None
    if cuda_pair:
        try:
            _, _, placement = cuda_child_environments(a.primary, a.worker, a.primary_device, a.worker_device, library_env)
        except (ValueError, KeyError, TypeError) as exc:
            p.error(str(exc))
    elif a.primary_device is not None or a.worker_device is not None:
        p.error('explicit device selection currently requires CUDA+CUDA')
    if a.combine=='native' and not cuda_pair:
        p.error('native arithmetic is currently an opt-in CUDA+CUDA experiment')
    layout = None
    if version == '5':
        layouts = [json.loads(subprocess.check_output([binary, '--protocol-layout'], text=True, env=library_env))
                   for binary in (a.primary, a.worker)]
        if layouts[0] != layouts[1] or layouts[0]['version'] != 5:
            p.error('mixed primary and worker protocol layouts differ')
        layout = layouts[0]
    library_env.setdefault('STRATA_GLM_GPU_RESERVE_MIB','1024')
    library_env.setdefault('STRATA_GLM_BUDGET_RESERVE_MIB','1024')
    with shared_memory() as memory:
        size = int(subprocess.check_output([a.worker, '--protocol-size'], text=True, env=library_env))
        if layout is not None and size != layout['bytes']:
            p.error('mixed worker protocol size differs from its layout')
        memory.truncate(size)
        monitor = mmap.mmap(memory.fileno(), 64, access=mmap.ACCESS_READ)
        primary_channel, worker_channel = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        env = library_env
        env['STRATA_GLM_MIXED_MEMORY_FD'] = str(memory.fileno())
        env['STRATA_GLM_MIXED_SOCKET_FD'] = str(primary_channel.fileno())
        env['STRATA_GLM_CANON'] = '0' if a.combine=='native' else '1'
        env['STRATA_GLM_CANON_COMBINE'] = 'device' if a.combine=='native' else a.combine
        env['STRATA_GLM_STEP_PIPELINE'] = '1'
        primary_env = env.copy(); worker_env = env.copy()
        worker_env['STRATA_GLM_MIXED_SOCKET_FD'] = str(worker_channel.fileno())
        if cuda_pair:
            primary_env['CUDA_VISIBLE_DEVICES'] = placement[0]['uuid']
            worker_env['CUDA_VISIBLE_DEVICES'] = placement[1]['uuid']
        processes = []
        summary = dict(shared_mapping_bytes=size, peak_combined_pss_bytes=0, peak_combined_swap_bytes=0,
                       pss_sampling='setup only, before first worker request; not a full-run peak',
                       primary=a.primary, worker=a.worker, primary_backend=primary_backend, worker_backend=worker_backend, worker_cache_mib=a.worker_cache_mib,
                       worker_gpu_budget_mib=a.worker_gpu_budget_mib,
                       device_placement=placement, primary_cpus=a.primary_cpus, worker_cpus=a.worker_cpus,
                       expert_arithmetic='native' if a.combine=='native' else 'canonical',combine=a.combine)
        started = time.monotonic()
        try:
            worker_cmd = [a.worker, a.model, str(memory.fileno()), str(worker_channel.fileno()), str(a.worker_cache_mib), str(a.worker_gpu_budget_mib)]
            if a.worker_numa_node is not None:
                worker_cmd = ['numactl', f'--cpunodebind={a.worker_numa_node}', f'--membind={a.worker_numa_node}', *worker_cmd]
            if a.worker_cpus is not None:
                worker_cmd = ['taskset', '-c', a.worker_cpus, *worker_cmd]
            worker = subprocess.Popen(worker_cmd,
                                      env=worker_env, pass_fds=(memory.fileno(), worker_channel.fileno()), start_new_session=True, preexec_fn=child_setup)
            processes.append(worker)
            worker_channel.close()
            runner_cmd = [a.primary, *(['--serve'] if a.primary_worker_mode else []), a.model, *args]
            if a.primary_cpus is not None:
                runner_cmd = ['taskset', '-c', a.primary_cpus, *runner_cmd]
            runner = subprocess.Popen(runner_cmd, env=primary_env,
                                      pass_fds=(memory.fileno(), primary_channel.fileno()), start_new_session=True, preexec_fn=child_setup)
            processes.append(runner)
            primary_channel.close()
            sampled = 0.0
            pss_sampled = 0.0
            while runner.poll() is None:
                now = time.monotonic()
                if now - sampled >= 1:
                    # PSS walks large model page tables. Sample only during model
                    # setup, before the first GPU request, to keep it out of timing.
                    sequence = struct.unpack_from('Q', monitor, 32)[0]
                    proportional = sequence == 0 and now-pss_sampled >= 10
                    sample = memory_sample(processes, proportional=proportional)
                    if proportional: pss_sampled = now
                    summary['peak_combined_pss_bytes'] = max(summary['peak_combined_pss_bytes'], sample['combined_pss_bytes'])
                    summary['peak_combined_swap_bytes'] = max(summary['peak_combined_swap_bytes'], sample['combined_swap_bytes'])
                    try:
                        path = Path('/sys/fs/cgroup') / Path(f'/proc/{runner.pid}/cgroup').read_text().strip().split('::',1)[1].lstrip('/')
                        current = int((path/'memory.current').read_text())
                        summary['peak_cgroup_memory_bytes'] = max(summary.get('peak_cgroup_memory_bytes',0), current)
                        summary['cgroup_swap_max'] = (path/'memory.swap.max').read_text().strip()
                    except (FileNotFoundError, ValueError, IndexError, ProcessLookupError):
                        pass
                    if sample['combined_swap_bytes']:
                        raise RuntimeError('mixed run swapped; performance measurement rejected')
                    sampled = now
                if worker.poll() is not None:
                    closing = struct.unpack_from('I', monitor, 28)[0] == 1
                    if worker.returncode != 0 or not closing:
                        raise RuntimeError('expert worker exited before primary')
                time.sleep(.1)
            summary['returncode'] = runner.returncode
            return runner.returncode
        finally:
            summary['elapsed_seconds'] = time.monotonic()-started
            monitor.close()
            summary['worker_returncode'] = processes[0].poll() if processes else None
            primary_channel.close()
            worker_channel.close()
            for proc in reversed(processes):
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
            for proc in reversed(processes):
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            if a.report:
                a.report.parent.mkdir(parents=True, exist_ok=True)
                a.report.write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())
