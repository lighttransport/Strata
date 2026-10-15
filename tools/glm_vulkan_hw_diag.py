"""Plan or run model-free Vulkan hardware checks in one owned user service.

Default: print a plan without opening a Vulkan device. --run requires a clean
kernel log, free VRAM with a reserve, unchanged PCI identity, no Strata GPU
workload, and a fresh output directory. Never reset a GPU or change timeouts.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
import fcntl

from glm_guarded_run import FAULT, ROOT, delta_log, gpu_devices, identities, kernel_log, telemetry, write_status

MIB = 1 << 20
PRESETS = {
    'smoke': dict(suite='all', memory_mib=64, rounds=1, hold_ms=0, compute_rounds=2, compute_repeats=1, iterations=64, full=False, timeout=180),
    'vram': dict(suite='memory', memory_mib=14336, rounds=1, hold_ms=100, compute_rounds=1, compute_repeats=1, iterations=64, full=False, timeout=1800),
    'vram-full': dict(suite='memory', memory_mib=0, rounds=1, hold_ms=100, compute_rounds=1, compute_repeats=1, iterations=64, full=True, full_heap=True, timeout=3600),
    'compute': dict(suite='compute', memory_mib=64, rounds=1, hold_ms=0, compute_rounds=1000, compute_repeats=16, iterations=256, full=False, timeout=1800),
    'extended': dict(suite='all', memory_mib=14336, rounds=2, hold_ms=1000, compute_rounds=256, compute_repeats=16, iterations=256, full=True, timeout=3600),
}
SOURCES = ('vulkan/hw_diag.cpp', 'vulkan/fast/shaders/hw_diag.comp',
           'vulkan/fast/device.cpp', 'vulkan/fast/device.hpp', 'vulkan/CMakeLists.txt')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def active_workloads():
    result = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            name = (proc/'exe').resolve(strict=True).name
            if name.startswith(('strata-glm-', 'glm_vulkan_')):
                result.append(dict(pid=int(proc.name), executable=name))
        except OSError:
            pass
    return result


def memory_plan(preset, requested, device, reserve_mib=1024):
    total = int((device/'mem_info_vram_total').read_text()) // MIB
    used = (int((device/'mem_info_vram_used').read_text()) + MIB - 1) // MIB
    if PRESETS[preset].get('full_heap'):
        if not 1 <= total <= 16384:
            raise ValueError('Full heap diagnostic currently supports heaps up to 16 GiB')
        if requested is not None and requested != total:
            raise ValueError('vram-full targets the entire detected heap; omit --memory-mib')
        if used >= total:
            raise RuntimeError('No free VRAM is available for the full heap diagnostic')
        return dict(total_mib=total, used_mib=used, reserve_mib=0,
                    allocation_overhead_budget_mib=0, memory_mib=total,
                    allocation_limit_mib=total-used, full_heap=True,
                    physical_addresses_known=False, physical_range_complete=False)
    # The diagnostic also needs staging, sentinels, compute and copy buffers.
    maximum = min(15360, total - used - reserve_mib - 64)
    if maximum < 1:
        raise RuntimeError('Insufficient free VRAM with the diagnostic reserve')
    memory = min(PRESETS[preset]['memory_mib'], maximum) if requested is None else requested
    if not 1 <= memory <= maximum:
        raise ValueError('Requested VRAM payload exceeds available memory with reserve')
    return dict(total_mib=total, used_mib=used, reserve_mib=reserve_mib,
                allocation_overhead_budget_mib=64, memory_mib=memory,
                physical_addresses_known=False)


def check_manifest(exe, manifest):
    if manifest is None:
        return None
    saved = json.loads(manifest.read_text())
    if saved['executable_sha256'] != sha(exe):
        raise RuntimeError('Frozen diagnostic executable hash mismatch')
    # Sources can change later; a frozen executable is tied to its saved
    # compiled-source manifest, not attributed to the current working tree.
    return dict(manifest_sha256=sha(manifest), compiled_source_sha256=saved['source_sha256'],
                software_validation=saved.get('software_validation'),
                protocol=saved.get('protocol', 1),
                requires_new_boot_from=saved.get('requires_new_boot_from'))


def make_plan(args):
    preset = dict(PRESETS[args.preset])
    device = Path('/sys/bus/pci/devices')/args.pci
    if device.name != args.pci or not device.is_dir():
        raise ValueError('PCI device must be one existing PCI identifier')
    if (device/'vendor').read_text().strip() != '0x1002' or not (device/'class').read_text().strip().startswith('0x03'):
        raise ValueError('This launcher requires an AMD GPU with VRAM sysfs accounting')
    memory = memory_plan(args.preset, args.memory_mib, device)
    preset['memory_mib'] = memory['memory_mib']
    if preset.get('full_heap'):
        preset['allocation_limit_mib'] = memory['allocation_limit_mib']
    if args.compute_rounds is not None:
        if not 1 <= args.compute_rounds <= 10000:
            raise ValueError('compute rounds must be 1..10000')
        preset['compute_rounds'] = args.compute_rounds
    if not 4 <= args.chunk_kib <= 4096 or args.chunk_kib % 4 or not 0 <= args.seed <= 0xffffffff:
        raise ValueError('chunk must be 4..4096 KiB, a multiple of 4; seed must be uint32')
    exe = args.exe.resolve(strict=True)
    command = [str(exe), '--device', args.device, '--suite', preset['suite'],
               '--memory-mib', str(preset['memory_mib']), '--rounds', str(preset['rounds']),
               '--compute-rounds', str(preset['compute_rounds']), '--chunk-kib', str(args.chunk_kib),
               '--compute-repeats', str(preset['compute_repeats']), '--iterations', str(preset['iterations']),
               '--hold-ms', str(preset['hold_ms']),
               '--seed', str(args.seed), '--max-submit-ms', '500']
    if preset['full']:
        command.append('--full-patterns')
    if preset.get('full_heap'):
        command.extend(['--full-heap', '--allocation-limit-mib', str(preset['allocation_limit_mib'])])
    frozen = check_manifest(exe, args.manifest)
    if preset.get('full_heap') and frozen and frozen['protocol'] < 2:
        raise ValueError('Full heap requires a protocol-2 diagnostic executable and manifest')
    return dict(preset=args.preset, options=preset, command=command, memory=memory,
                executable_sha256=sha(exe), frozen_build=frozen,
                working_source_sha256_at_launch={name: sha(ROOT/name) for name in SOURCES},
                launcher_source_sha256=sha(__file__),
                shared_guard_source_sha256=sha(ROOT/'tools/glm_guarded_run.py'),
                robust_buffers=args.robust, pci=args.pci, output=str(args.output.resolve()),
                plan_only=not args.run, passed=False)


def capture_dump(device, output):
    """Best-effort bounded read before devcoredump expires; never clear it."""
    path = device/'devcoredump/data'
    try:
        data = subprocess.run(['head', '-c', str(4 * MIB + 1), str(path)],
                              capture_output=True, check=True, timeout=2).stdout
        truncated = len(data) > 4 * MIB
        data = data[:4 * MIB]
        destination = output/'devcoredump-prefix.bin'
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        return dict(captured=True, bytes=len(data), truncated=truncated, sha256=sha(destination))
    except (OSError, subprocess.SubprocessError) as error:
        return dict(captured=False, error=type(error).__name__, errno=getattr(error, 'errno', None))


def complete_result(result, options):
    if not isinstance(result, dict) or result.get('passed') is not True or result.get('cpu_device') is not False:
        return False
    memory = options['suite'] != 'compute'
    compute = options['suite'] != 'memory'
    expected = dict(suite=options['suite'], memory_mib=options['memory_mib'] if memory else 0,
                    memory_rounds=options['rounds'] if memory else 0,
                    compute_rounds=options['compute_rounds'] if compute else 0,
                    compute_repeats=options['compute_repeats'] if compute else 0,
                    patterns_per_round=73 if options['full'] else 9)
    if options.get('full_heap'):
        target = options['memory_mib'] * MIB
        payload = result.get('memory_payload_bytes')
        unavailable = result.get('unavailable_target_bytes')
        if (result.get('full_heap') is not True or result.get('physical_range_complete') is not False or
                result.get('memory_target_bytes') != target or
                result.get('allocation_limit_bytes') != options['allocation_limit_mib'] * MIB or
                not isinstance(payload, int) or not 0 < payload <= options['allocation_limit_mib'] * MIB or
                not isinstance(unavailable, int) or payload + unavailable != target or
                result.get('cpu_verified_words') != payload // 4 * options['rounds'] * (73 if options['full'] else 9)):
            return False
    return (all(result.get(k) == v for k, v in expected.items()) and
            isinstance(result.get('max_submit_ms'), (int, float)) and
            0 <= result['max_submit_ms'] <= 500 and
            isinstance(result.get('gpu_submissions'), int) and result['gpu_submissions'] > 0)


@contextmanager
def hardware_lock():
    path = Path('/tmp')/('strata-vulkan-hw-diag-'+str(os.getuid())+'.lock')
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another Vulkan hardware diagnostic owns the lock') from error
        yield


def engine_result(path):
    terminal = None
    for line in path.read_text(errors='replace').splitlines():
        if line.startswith('{'):
            row = json.loads(line)
            if row.get('event') == 'result':
                terminal = row
    return terminal


def run(args, plan):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = dict(plan, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  passed=False, termination_reason=None, exit_code=None, faults=[])
    status_path = output/'guard.json'
    write_status(status_path, status)
    try:
        before = kernel_log()
        (output/'kernel-before.log').write_text(before)
        if FAULT.search(before):
            raise RuntimeError('This boot already contains a hardware/GPU fault; no launch')
        if (plan.get('frozen_build') or {}).get('requires_new_boot_from') == status['boot_id']:
            raise RuntimeError('This diagnostic was prepared on the current faulted boot; new boot required, no launch')
        devices = gpu_devices()
        initial = identities(devices)
        busy = active_workloads()
        if busy:
            status['active_workloads'] = busy
            raise RuntimeError('Another Strata GPU workload is active; no launch')
        # A plan is not a reservation: recheck immediately before launch.
        device = Path('/sys/bus/pci/devices')/args.pci
        fresh = memory_plan(args.preset, plan['options']['memory_mib'], device)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        status['termination_reason'] = 'preflight failed: '+str(error)
        write_status(status_path, status)
        return status
    status.update(pci_before=initial, memory_before_launch=fresh)
    execution_options = dict(plan['options'])
    executable_command = list(plan['command'])
    if execution_options.get('full_heap'):
        execution_options['allocation_limit_mib'] = fresh['allocation_limit_mib']
        executable_command[executable_command.index('--allocation-limit-mib')+1] = str(fresh['allocation_limit_mib'])
    status['execution_options'] = execution_options
    unit = 'strata-vulkan-hw-diag-'+uuid.uuid4().hex
    status['own_unit'] = unit
    command = ['systemd-run', '--user', '--wait', '--pipe', '--collect', '--unit='+unit,
               '--property=MemoryMax=2G', '--property=MemorySwapMax=0', '--property=OOMPolicy=kill',
               '--property=RuntimeMaxSec='+str(plan['options']['timeout']),
               '--property=WorkingDirectory='+str(ROOT),
               '--setenv=STRATA_VULKAN_FAST_ROBUST_BUFFERS='+('1' if args.robust else '0'),
               *executable_command]
    status['service_command'] = command
    write_status(status_path, status)
    started = time.monotonic()
    last_heartbeat = started - 5
    process = None
    try:
        with (output/'diagnostic.log').open('x') as log, (output/'telemetry.jsonl').open('x') as trace:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                delta = delta_log(before, kernel_log())
                if FAULT.search(delta):
                    status['termination_reason'] = 'hardware/GPU fault'
                    break
                if identities(devices) != initial:
                    status['termination_reason'] = 'PCI identity changed'
                    break
                if time.monotonic() - started > plan['options']['timeout'] + 10:
                    status['termination_reason'] = 'launcher watchdog timeout'
                    break
                if time.monotonic() - last_heartbeat >= 2:
                    trace.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
                                               seconds=time.monotonic()-started, values=telemetry(devices)))+'\n')
                    trace.flush(); os.fsync(trace.fileno())
                    last_heartbeat = time.monotonic()
                time.sleep(.25)
    except KeyboardInterrupt:
        status['termination_reason'] = 'interrupted'
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        status['termination_reason'] = type(error).__name__+': '+str(error)
    finally:
        if process is not None and process.poll() is None:
            # Only this invocation's literal UUID service and launcher group.
            try:
                subprocess.run(['systemctl', '--user', 'kill', '--signal=SIGKILL', unit+'.service'],
                               capture_output=True, timeout=5)
            except (OSError, subprocess.SubprocessError) as error:
                status['cleanup_error'] = type(error).__name__
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                status['cleanup_error'] = 'launcher did not exit'
        status['exit_code'] = process.poll() if process is not None else None
        try:
            delta = delta_log(before, kernel_log())
            (output/'kernel.log').write_text(delta)
            status['faults'] = [line for line in delta.splitlines() if FAULT.search(line)]
            status['pci_after'] = identities(devices)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            status['termination_reason'] = status['termination_reason'] or 'Final diagnostics unavailable: '+type(error).__name__
        if status['faults']:
            status['devcoredump'] = capture_dump(device, output)
        try:
            result = engine_result(output/'diagnostic.log')
            status['diagnostic_result'] = result
            if execution_options.get('full_heap') and isinstance(result, dict):
                status['coverage'] = dict(target_bytes=result.get('memory_target_bytes'),
                                          tested_payload_bytes=result.get('memory_payload_bytes'),
                                          unavailable_target_bytes=result.get('unavailable_target_bytes'),
                                          physical_range_complete=False,
                                          description='Data checks cover allocated device-local buffers; physical VRAM addresses and complete physical coverage are unknown')
            status['passed'] = (status['exit_code'] == 0 and not status['termination_reason'] and
                                not status['faults'] and status.get('pci_after') == initial and
                                complete_result(result, execution_options))
        except (OSError, ValueError) as error:
            status['termination_reason'] = status['termination_reason'] or 'Diagnostic result unavailable: '+type(error).__name__
        status['wall_seconds'] = time.monotonic()-started
        write_status(status_path, status)
    return status


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', choices=PRESETS, default='smoke')
    p.add_argument('--exe', type=Path, default=ROOT/'build-vulkan-rx570-native/vulkan/glm_vulkan_hw_diag')
    p.add_argument('--manifest', type=Path, help='compiled-source manifest for a frozen executable')
    p.add_argument('--pci', default='0000:07:00.0')
    p.add_argument('--device', default='RX 570', help='unambiguous Vulkan device name substring matching --pci')
    p.add_argument('--output', required=True, type=Path, help='fresh persistent output directory')
    p.add_argument('--memory-mib', type=int)
    p.add_argument('--compute-rounds', type=int)
    p.add_argument('--chunk-kib', type=int, default=1024)
    p.add_argument('--seed', type=int, default=0x53a9c217)
    p.add_argument('--robust', action='store_true')
    p.add_argument('--run', action='store_true', help='execute on a healthy boot; otherwise print a plan only')
    return p


def main():
    args = parser().parse_args()
    try:
        plan = make_plan(args)
        if args.run:
            with hardware_lock():
                result = run(args, plan)
        else:
            result = plan
        print(json.dumps(result, indent=2), flush=True)
        return 0 if not args.run or result['passed'] else 1
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(json.dumps(dict(passed=False, error=type(error).__name__+': '+str(error))), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
