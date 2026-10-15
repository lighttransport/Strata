"""A sealed, shared RAM GGUF for Linux benchmark subprocesses.

Run the owner under the intended NUMA policy, for example numactl
--interleave=all. Children open ``path`` while this context is alive. Use the
decoder's --warm-weights option: --lock-weights creates private COW copies and
would defeat sharing. No GPU runtime is initialized by this module.
"""
import ctypes
import hashlib
import mmap
import os
from pathlib import Path
import re
import sys
import time

if sys.platform == 'linux':
    import fcntl


class ResidentModel:
    def __init__(self, source, expected_sha256, *, max_bytes):
        if sys.platform != 'linux':
            raise RuntimeError('resident model staging requires Linux memfd seals')
        self.fd = -1
        self.mapping = None
        self.address = None
        self.locked = False
        source = Path(source)
        self.size = source.stat().st_size
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or not 0 < self.size <= max_bytes:
            raise ValueError('model exceeds the explicit shared RAM budget')
        libc = ctypes.CDLL(None, use_errno=True)
        self._mlock = libc.mlock
        self._mlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        self._mlock.restype = ctypes.c_int
        self._munlock = libc.munlock
        self._munlock.argtypes = self._mlock.argtypes
        self._munlock.restype = ctypes.c_int
        native_mmap = libc.mmap
        native_mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong)
        native_mmap.restype = ctypes.c_void_p
        self._munmap = libc.munmap
        self._munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        self._munmap.restype = ctypes.c_int
        memory_view = ctypes.pythonapi.PyMemoryView_FromMemory
        memory_view.argtypes = (ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int)
        memory_view.restype = ctypes.py_object
        try:
            self.fd = os.memfd_create('strata-resident-model', os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
            os.ftruncate(self.fd, self.size)
            start = time.monotonic()
            copied = 0
            with source.open('rb') as reader:
                while chunk := reader.read(16 * 1024 * 1024):
                    if copied + len(chunk) > self.size:
                        raise ValueError('model grew during staging')
                    view = memoryview(chunk)
                    while view:
                        count = os.write(self.fd, view)
                        if not count:
                            raise OSError('short write staging the model')
                        copied += count
                        view = view[count:]
                    view.release()
            if copied != self.size:
                raise ValueError('model changed size during staging')
            seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
            fcntl.fcntl(self.fd, fcntl.F_ADD_SEALS, seals)
            # Older kernels reject shared mappings of a sealed memfd opened
            # for writing, even when the requested mapping is read-only.
            readonly = os.open(f'/proc/self/fd/{self.fd}', os.O_RDONLY | os.O_CLOEXEC)
            os.close(self.fd)
            self.fd = readonly
            address = native_mmap(None, self.size, mmap.PROT_READ, mmap.MAP_SHARED, self.fd, 0)
            if address == ctypes.c_void_p(-1).value:
                raise OSError(ctypes.get_errno(), 'cannot map the resident model')
            self.address = address
            # Locking a writable MAP_PRIVATE view can populate a second COW
            # copy. A read-only shared mapping pins the sealed backing pages.
            if self._mlock(self.address, self.size):
                raise OSError(ctypes.get_errno(), 'cannot lock the resident model')
            self.locked = True
            staged = time.monotonic()
            digest = hashlib.sha256()
            with memory_view(self.address, self.size, 0x100) as whole:  # PyBUF_READ
                for offset in range(0, self.size, 16 * 1024 * 1024):
                    with whole[offset:offset + 16 * 1024 * 1024] as view:
                        digest.update(view)
            actual = digest.hexdigest()
            if actual != expected_sha256:
                raise ValueError('resident model SHA-256 differs from the expected weights')
            self.path = Path(f'/proc/{os.getpid()}/fd/{self.fd}')
            self.identity = dict(source=source.name, bytes=self.size, sha256=actual,
                                 copy_and_lock_seconds=staged-start,
                                 verification_seconds=time.monotonic()-staged,
                                 seals=fcntl.fcntl(self.fd, fcntl.F_GET_SEALS),
                                 numa_pages=self._numa_pages(),
                                 scope='One sealed shared RAM copy, locked by its owner; child mappings share these pages')
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _numa_pages():
        totals = {}
        for line in Path('/proc/self/numa_maps').read_text().splitlines():
            if 'strata-resident-model' not in line:
                continue
            for node, pages in re.findall(r'\bN(\d+)=(\d+)', line):
                totals[node] = totals.get(node, 0) + int(pages)
        return totals

    def close(self):
        if self.locked:
            self._munlock(self.address, self.size)
            self.locked = False
        if self.address is not None:
            self._munmap(self.address, self.size)
            self.address = None
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
