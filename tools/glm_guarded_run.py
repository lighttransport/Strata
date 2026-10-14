"""Run one GLM benchmark in an owned, bounded user service, stopping on GPU faults.

Cleanup uses only the UUID unit created in this invocation, never a result or
manifest supplied by another process. No GPU reset or unrelated cleanup occurs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
GUARD_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
FAULT = re.compile(r'device lost from bus|GPU reset|ring .*timeout|\bVM\b.*\bfault\b|'
                   r'page fault|AER:.*(?:error|fatal)|device inaccessible|'
                   r'Hardware Error|EDAC.*(?:error|\bCE\b|\bUE\b)', re.I)


def task_event(path, task, state, **details):
    """Persist an append-only breadcrumb before risky work, including its boot."""
    now = datetime.now(timezone.utc)
    row = dict(timestamp=now.isoformat(), local_timestamp=now.astimezone().isoformat(),
               monotonic_seconds=time.monotonic(),
               boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
               task=task, state=state, **details)
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    return row


def write_status(path, status):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(status, indent=2)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def telemetry(devices):
    """Read kernel counters only; do not submit work to the GPU."""
    values = {}
    for device in devices:
        for name in ('gpu_busy_percent', 'mem_busy_percent', 'mem_info_vram_used',
                     'mem_info_gtt_used', 'current_link_speed', 'current_link_width'):
            try:
                values[device.name+'/'+name] = (device/name).read_text().strip()
            except OSError:
                pass
        for sensor in (device/'hwmon').glob('hwmon*'):
            for name in ('power1_average', 'power1_cap', 'temp1_input', 'temp2_input',
                         'temp3_input', 'freq1_input', 'freq2_input'):
                try:
                    values[device.name+'/'+name] = (sensor/name).read_text().strip()
                except OSError:
                    pass
    for sensor in Path('/sys/class/hwmon').glob('hwmon*'):
        try:
            if (sensor/'name').read_text().strip() == 'k10temp':
                values['cpu_temperature_millic'] = (sensor/'temp1_input').read_text().strip()
        except OSError:
            pass
    return values


def kernel_log():
    return subprocess.check_output(['dmesg'], text=True, timeout=5)


def gpu_devices():
    return [p for p in sorted(Path('/sys/bus/pci/devices').glob('*'))
            if (p/'vendor').read_text().strip() in ('0x1002', '0x10de')
            and (p/'class').read_text().strip().startswith('0x03')]


def identities(devices):
    values = {}
    for path in devices:
        with (path/'config').open('rb') as stream:
            value = stream.read(4)
        if len(value) != 4 or value[:2] in (b'\xff\xff', b'\0\0'):
            raise RuntimeError('GPU PCI identity lost: '+path.name)
        values[path.name] = value.hex()
    return values


def delta_log(before, after):
    return after[len(before):] if after.startswith(before) else after


def run(config, prompt, output, *, tokens=512, trials=1, timeout=1200, extra=(), correctness_only=False):
    if not 2 <= tokens <= 4096 or not 1 <= trials <= 5 or not 10 <= timeout <= 10800:
        raise ValueError('tokens 2..4096, trials 1..5, timeout 10..10800 required')
    config, prompt, output = Path(config).resolve(), Path(prompt).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    events = output.with_suffix('.tasks.jsonl')
    if events.exists() or output.with_suffix('.guard.json').exists() or output.with_suffix('.result.json').exists():
        raise FileExistsError('Use a fresh output prefix; refusing to overwrite diagnostic evidence')
    task_event(events, output.name, 'preflight', config=str(config))
    devices = gpu_devices()
    if not devices:
        raise RuntimeError('No enumerated GPU; refusing launch')
    before = kernel_log()
    output.with_suffix('.kernel-before.log').write_text(before)
    if FAULT.search(before):
        raise RuntimeError('This boot already contains a hardware/GPU fault; refusing launch')
    initial = identities(devices)
    from glm_low_memory_bench import clear_model_cache
    cfg = json.loads(config.read_text())
    # File pages must be charged to this service, not an earlier cgroup.
    cache_before = clear_model_cache(cfg)
    task_event(events, output.name, 'model_cache_cleared')
    own_unit = 'strata-glm-opt-'+uuid.uuid4().hex
    command = ['systemd-run', '--user', '--wait', '--pipe', '--collect', '--unit='+own_unit,
               '--property=MemoryMax=60G', '--property=MemorySwapMax=0',
               '--property=RuntimeMaxSec='+str(timeout+30), '--property=OOMPolicy=kill',
               '--property=WorkingDirectory='+str(ROOT),
               sys.executable, str(ROOT/'tools/glm_low_memory_bench.py'), str(config), str(prompt),
               '--output', str(output), '--ram-gib', '60', '--tokens', str(tokens),
               '--trials', str(trials), '--timeout', str(timeout),
               '--gpu-capacity-mib', '16304', '--gpu-used-limit-mib', '15792', '--worker', *extra]
    executable = Path(cfg['exe'])
    if not executable.is_absolute():
        executable = ROOT/executable
    status = dict(own_unit=own_unit, command=command, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  pci_before=initial, config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                  prompt_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest(),
                  executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                  loaded_guard_source_sha256=GUARD_SOURCE_SHA256,
                  source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                 for name in ('src/program/glm_decode.cpp', 'tools/glm_guarded_run.py',
                                              'tools/glm_low_memory_bench.py')},
                  model_cache_before=cache_before,
                  termination_reason=None, passed=False)
    status_path = output.with_suffix('.guard.json')
    if cfg.get('backend') == 'vulkan':
        sources = [ROOT/'CMakeLists.txt', *sorted((ROOT/'vulkan').rglob('*'))]
        status['source_sha256'].update({str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in sources if path.is_file()})
    write_status(status_path, status)
    task_event(events, output.name, 'launch', own_unit=own_unit,
               executable_sha256=status['executable_sha256'],
               config_sha256=status['config_sha256'])
    start = time.monotonic()
    last_heartbeat = start
    with output.with_suffix('.launcher.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic()-last_heartbeat >= 5:
                    sizes = {}
                    last_lines = {}
                    raw_files = list(output.parent.glob(output.name+'*.log'))
                    if output.with_suffix('.stdout').exists():
                        raw_files.append(output.with_suffix('.stdout'))
                    for raw in raw_files:
                        with raw.open('rb') as stream:
                            os.fsync(stream.fileno())
                            size = os.fstat(stream.fileno()).st_size
                            sizes[raw.name] = size
                            if raw in (output.with_suffix('.log'), output.with_suffix('.stdout')):
                                stream.seek(max(0, size-8192))
                                lines = stream.read().decode('utf-8', errors='replace').splitlines()
                                last_lines[raw.name] = lines[-3:]
                    task_event(events, output.name, 'running', elapsed_seconds=time.monotonic()-start,
                               log_bytes=sizes, last_engine_lines=last_lines, telemetry=telemetry(devices))
                    last_heartbeat = time.monotonic()
                delta = delta_log(before, kernel_log())
                if FAULT.search(delta):
                    status['termination_reason'] = 'hardware/GPU fault'
                    break
                if identities(devices) != initial:
                    status['termination_reason'] = 'PCI identity changed'
                    break
                if time.monotonic()-start > timeout+20:
                    status['termination_reason'] = 'watchdog timeout'
                    break
                time.sleep(.5)
        except KeyboardInterrupt:
            status['termination_reason'] = 'interrupted'
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            status['termination_reason'] = str(error)
        finally:
            if process.poll() is None:
                # This literal owner was allocated above, in this invocation.
                try:
                    subprocess.run(['systemctl', '--user', 'kill', '--signal=SIGKILL', own_unit+'.service'],
                                   capture_output=True, timeout=5)
                except subprocess.SubprocessError as error:
                    status['cleanup_error'] = str(error)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    status['termination_reason'] = 'test launcher did not exit after termination'
    try:
        after = kernel_log()
    except (OSError, subprocess.SubprocessError) as error:
        status['termination_reason'] = status['termination_reason'] or 'final kernel log unavailable: '+str(error)
        after = before
    delta = delta_log(before, after)
    output.with_suffix('.kernel.log').write_text(delta)
    status.update(exit_code=process.poll(), wall_seconds=time.monotonic()-start,
                  faults=[line for line in delta.splitlines() if FAULT.search(line)])
    try:
        status['pci_after'] = identities(devices)
    except (OSError, RuntimeError) as error:
        status['pci_after'] = str(error)
    result = output.with_suffix('.result.json')
    measurement_clean = False
    correctness_complete = False
    if result.exists():
        try:
            measurement = json.loads(result.read_text())
            measurement_clean = measurement.get('clean') is True
            group = measurement.get('cgroup_after', {})
            correctness_complete = (measurement.get('complete') is True and
                not measurement.get('rejected') and not measurement.get('determinism_failure') and
                group.get('memory.swap.current') == 0 and
                all(not group.get('memory.events', {}).get(key,0) for key in ('oom','oom_kill','oom_group_kill')))
        except (OSError, ValueError) as error:
            status['result_error'] = str(error)
    status['passed'] = (process.poll() == 0 and not status['termination_reason'] and not status['faults']
                        and status['pci_after'] == initial and
                        (correctness_complete if correctness_only else measurement_clean))
    status['test_kind'] = 'correctness' if correctness_only else 'performance'
    status['performance_qualified'] = status['passed'] and measurement_clean and not correctness_only
    write_status(status_path, status)
    task_event(events, output.name, 'passed' if status['passed'] else 'failed',
               termination_reason=status['termination_reason'], exit_code=status['exit_code'],
               faults=status['faults'], pci_unchanged=status['pci_after'] == initial,
               measurement_clean=measurement_clean)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path); p.add_argument('prompt', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=512); p.add_argument('--trials', type=int, default=1)
    p.add_argument('--timeout', type=int, default=1200)
    p.add_argument('--correctness-only', action='store_true',
                   help='retain resource/fault checks; never qualify timings, tolerate recorded CPU competition')
    a = p.parse_args()
    status = run(a.config, a.prompt, a.output, tokens=a.tokens, trials=a.trials, timeout=a.timeout,correctness_only=a.correctness_only)
    print(json.dumps(status, indent=2), flush=True)
    return 0 if status['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
