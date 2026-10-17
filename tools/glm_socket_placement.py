"""First-touch the byte ranges in a frozen socket-row plan; never change model bytes.

The plan is tied to the model's complete SHA-256. It describes offsets and row
halves, so this copier does not define tensor names or assembled-pack rules.
The caller must run under the allocation's intended policy for the other bytes.
"""
import ctypes
import mmap
import os
from pathlib import Path
import re
import threading


def copy_plan(plan, size, expected_sha256):
    if (not isinstance(plan, dict) or plan.get('schema') != 'strata.socket-row-plan.v1'
            or type(plan.get('model_bytes')) is not int or plan['model_bytes'] != size
            or plan.get('model_sha256') != expected_sha256):
        raise ValueError('socket row plan does not match the expected model identity')
    entries = plan.get('row_halves')
    if not isinstance(entries, list) or not 0 < len(entries) <= 4096:
        raise ValueError('socket row plan needs 1..4096 tensor ranges')
    tasks = [[], [], []]
    spans, panels = [], []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('invalid socket row range')
        values = [entry.get(k) for k in ('offset', 'bytes_per_half', 'slices')]
        if any(type(v) is not int for v in values):
            raise ValueError('socket row offsets and sizes must be integers')
        start, half, count = values
        end = start + 2 * half * count
        if start < 0 or half < 1 or count < 1 or end > size:
            raise ValueError('socket row range is outside the model')
        if len(tasks[0]) + count > 500000:
            raise ValueError('socket row plan exceeds the slice limit')
        spans.append((start, end))
        panels.append((entry.get('tensor', str(len(panels))), start, half))
        for index in range(count):
            for side in range(2):
                tasks[side].append((start + (2 * index + side) * half, half))
    at = 0
    for start, end in sorted(spans):
        if start < at:
            raise ValueError('socket row tensor ranges overlap')
        if at < start:
            tasks[2].append((at, start - at))
        at = end
    if at < size:
        tasks[2].append((at, size - at))
    return tasks, panels


def socket_groups():
    groups = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        root = Path(f'/sys/devices/system/cpu/cpu{cpu}')
        package = int((root / 'topology/physical_package_id').read_text())
        nodes = [int(p.name[4:]) for p in root.glob('node[0-9]*')]
        if not nodes:
            raise ValueError('CPU topology has no NUMA node')
        group = groups.setdefault(package, dict(cpus=[], nodes=set()))
        group['cpus'].append(cpu)
        group['nodes'].update(nodes)
    if len(groups) != 2:
        raise ValueError('socket row placement requires CPUs in exactly two physical packages')
    return [groups[k] for k in sorted(groups)]


def set_interleave(nodes):
    library = ctypes.CDLL('libnuma.so.1', use_errno=True)
    policy = library.set_mempolicy
    policy.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong)
    policy.restype = ctypes.c_int
    bits = ctypes.sizeof(ctypes.c_ulong) * 8
    words = (max(nodes) + bits) // bits
    mask = (ctypes.c_ulong * words)()
    for node in nodes:
        mask[node // bits] |= 1 << (node % bits)
    if policy(3, mask, words * bits):  # Linux MPOL_INTERLEAVE
        raise OSError(ctypes.get_errno(), 'cannot set socket interleave policy')


def copy_socket_rows(source, destination_fd, size, expected_sha256, plan):
    tasks, panels = copy_plan(plan, size, expected_sha256)
    groups = socket_groups()
    source_fd = os.open(source, os.O_RDONLY | os.O_CLOEXEC)
    errors, copied = [], [0, 0, 0]
    cancel = threading.Event()

    def copy(side):
        try:
            if side < 2:
                os.sched_setaffinity(0, groups[side]['cpus'])
                set_interleave(groups[side]['nodes'])
            for at, remaining in tasks[side]:
                while remaining:
                    if cancel.is_set():
                        return
                    data = os.pread(source_fd, min(remaining, 16 * 1024 * 1024), at)
                    if not data:
                        raise OSError('short read while staging socket rows')
                    view = memoryview(data)
                    while view:
                        count = os.pwrite(destination_fd, view, at)
                        if count <= 0:
                            raise OSError('short write while staging socket rows')
                        copied[side] += count
                        at += count
                        remaining -= count
                        view = view[count:]
        except BaseException as error:
            errors.append(error)
            cancel.set()

    threads = [threading.Thread(target=copy, args=(side,)) for side in range(3)]
    try:
        if os.fstat(source_fd).st_size != size:
            raise ValueError('model changed size before socket staging')
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        if sum(copied) != size or os.fstat(source_fd).st_size != size:
            raise ValueError('model changed size during socket staging')
    finally:
        cancel.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join()
        os.close(source_fd)
    return groups, panels


def verify_socket_rows(fd, groups, panels):
    """Check whole pages in up to five first-slice panels, on both sockets."""
    if not panels or len(groups) != 2:
        raise ValueError('socket row verification needs panels and two socket groups')
    library = ctypes.CDLL(None, use_errno=True)
    native_mmap = library.mmap
    native_mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                           ctypes.c_int, ctypes.c_longlong)
    native_mmap.restype = ctypes.c_void_p
    munmap = library.munmap
    munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    munmap.restype = ctypes.c_int
    page = mmap.PAGESIZE
    samples = []
    indices = sorted({0, len(panels) // 4, len(panels) // 2,
                      len(panels) * 3 // 4, len(panels) - 1})
    for index in indices:
        name, start, half = panels[index]
        for side in range(2):
            begin = (start + side * half + page - 1) // page * page
            end = (start + (side + 1) * half) // page * page
            length = end - begin
            if length <= 0:
                continue  # A page shared by both halves has no unique owner.
            address = native_mmap(None, length, mmap.PROT_READ, mmap.MAP_SHARED, fd, begin)
            if address == ctypes.c_void_p(-1).value:
                raise OSError(ctypes.get_errno(), 'cannot map socket row sample')
            try:
                data = (ctypes.c_ubyte * length).from_address(address)
                # Populate this temporary VMA so numa_maps can count its pages.
                for offset in range(0, length, page):
                    _ = data[offset]
                line = next(line for line in Path('/proc/self/numa_maps').read_text().splitlines()
                            if line.split()[0] == format(address, 'x'))
                nodes = {int(k): int(v) for k, v in re.findall(r' N(\d+)=(\d+)', line)}
                local = (bool(nodes) and set(nodes) <= groups[side]['nodes']
                         and sum(nodes.values()) == length // page)
                samples.append(dict(tensor=name, side=side, pages=nodes, local=local))
            finally:
                munmap(address, length)
    if {sample['side'] for sample in samples} != {0, 1} or not all(s['local'] for s in samples):
        raise ValueError('socket row placement verification failed')
    return samples
