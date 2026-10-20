"""Keep immutable compressed ModelOpt shards in RAM across benchmark processes.

Linux only. The child receives a normal model directory whose indexed shards
refer to sealed memfds owned by this process. No model bytes are written to disk.
"""
import argparse
import concurrent.futures
import ctypes
try:
    import fcntl
except ImportError:
    fcntl = None
import hashlib
import json
import mmap
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time


class RamCache:
    def __init__(self, model, destination, *, workers=4, chunk_bytes=64*1024*1024, reserve_bytes=6*1024**3,
                 direct_read=False):
        if sys.platform != 'linux':
            raise RuntimeError('sealed RAM cache requires Linux')
        self.model = pathlib.Path(model).resolve()
        self.destination = pathlib.Path(destination).absolute()
        self.workers, self.chunk_bytes = workers, chunk_bytes
        self.reserve_bytes = reserve_bytes
        self.direct_read = direct_read
        self.entries = []
        self.created = False
        self.stop = threading.Event()
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
        self.libc.mmap.restype = ctypes.c_void_p
        self.libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

    def _copy(self, relative):
        name = pathlib.PurePosixPath(relative)
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('invalid indexed shard path')
        source = (self.model / relative).resolve()
        source.relative_to(self.model)
        size = source.stat().st_size
        descriptor = os.memfd_create('strata-fp4', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        pointer = None
        direct_descriptor = None
        buffer = view = None
        try:
            os.ftruncate(descriptor, size)
            source_hash, cached_hash = hashlib.sha256(), hashlib.sha256()
            if self.direct_read:
                # mmap provides aligned memory for O_DIRECT. Only a final
                # partial page uses buffered IO, bounding source-cache growth.
                direct_descriptor = os.open(source, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECT)
                buffer = mmap.mmap(-1, self.chunk_bytes)
            else:
                buffer = bytearray(self.chunk_bytes)
            view = memoryview(buffer)
            with source.open('rb', buffering=0) as stream:
                offset = 0
                while offset < size:
                    if self.stop.is_set():
                        raise RuntimeError('RAM cache loading canceled')
                    extent = min(len(view), size-offset)
                    if self.direct_read and extent >= 4096:
                        extent = extent//4096*4096
                        count = os.preadv(direct_descriptor, [view[:extent]], offset)
                        if count % 4096:
                            raise RuntimeError('checkpoint changed or direct read was unaligned')
                    else:
                        count = os.preadv(stream.fileno(), [view[:extent]], offset)
                    if not count:
                        raise RuntimeError('checkpoint changed or truncated during RAM copy')
                    block = view[:count]
                    try:
                        source_hash.update(block)
                        written = 0
                        while written < count:
                            n = os.write(descriptor, block[written:])
                            if n <= 0:
                                raise RuntimeError('short write to RAM cache')
                            written += n
                    finally:
                        block.release()
                    os.posix_fadvise(stream.fileno(), offset, count, os.POSIX_FADV_DONTNEED)
                    offset += count
                stream.seek(size)
                if stream.read(1):
                    raise RuntimeError('checkpoint grew during RAM copy')
            seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
            fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
            # Hash the actual sealed copy too, with bounded storage.
            offset = 0
            while offset < size:
                if self.stop.is_set():
                    raise RuntimeError('RAM cache loading canceled')
                count = os.preadv(descriptor, [view[:min(len(view), size-offset)]], offset)
                if not count:
                    raise RuntimeError('short read from sealed RAM cache')
                cached_hash.update(view[:count])
                offset += count
            if source_hash.digest() != cached_hash.digest():
                raise RuntimeError('RAM copy hash differs from source')
            # Older kernels reject shared mappings of a write-sealed file
            # through its original read/write descriptor, even for PROT_READ.
            readonly = os.open('/proc/self/fd/%d' % descriptor, os.O_RDONLY | os.O_CLOEXEC)
            previous, descriptor = descriptor, readonly
            os.close(previous)
            pointer = self.libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, descriptor, 0)
            if pointer == ctypes.c_void_p(-1).value:
                pointer = None
                raise OSError(ctypes.get_errno(), 'cannot map sealed RAM cache')
            if self.libc.mlock(pointer, size):
                raise OSError(ctypes.get_errno(), 'cannot lock sealed RAM cache')
            target = self.destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to('/proc/%d/fd/%d' % (os.getpid(), descriptor))
            print('RAM_CACHE_SHARD bytes=%d sha256=%s name=%s' % (size, cached_hash.hexdigest(), relative), flush=True)
            return dict(name=relative, bytes=size, sha256=cached_hash.hexdigest(), descriptor=descriptor, pointer=pointer)
        except BaseException:
            if pointer is not None:
                self.libc.munmap(pointer, size)
            os.close(descriptor)
            raise
        finally:
            if direct_descriptor is not None:
                os.close(direct_descriptor)
            if view is not None:
                view.release()
            if isinstance(buffer, mmap.mmap):
                buffer.close()

    def open(self):
        if self.created:
            raise RuntimeError('RAM cache is already open')
        if self.destination.exists() or self.destination.is_symlink():
            raise FileExistsError('cache directory already exists')
        self.stop.clear()
        if sys.platform != 'linux' or not hasattr(os, 'memfd_create'):
            raise RuntimeError('sealed RAM cache requires Linux and Python with memfd_create')
        config = json.loads((self.model/'config.json').read_text())
        if config.get('quantization_config', {}).get('quant_method') != 'modelopt':
            raise ValueError('RAM cache requires a native ModelOpt checkpoint')
        index = json.loads((self.model/'model.safetensors.index.json').read_text())
        names = sorted(set(index['weight_map'].values()))
        if not names or not 1 <= self.workers <= 16 or self.chunk_bytes < 4096 or (self.direct_read and self.chunk_bytes % 4096):
            raise ValueError('invalid RAM cache extent or worker count')
        # Validate every source path before creating or allocating anything.
        for relative in names:
            name = pathlib.PurePosixPath(relative)
            if name.is_absolute() or '..' in name.parts or name.suffix != '.safetensors':
                raise ValueError('invalid indexed shard path')
            (self.model/relative).resolve().relative_to(self.model)
        size = sum((self.model/name).stat().st_size for name in names)
        limit = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGESIZE')
        if os.environ.get('SLURM_MEM_PER_NODE', '').isdigit():
            limit = min(limit, int(os.environ['SLURM_MEM_PER_NODE'])*1024*1024)
        for line in pathlib.Path('/proc/self/cgroup').read_text().splitlines():
            unused, controllers, relative = line.split(':', 2)
            if 'memory' in controllers.split(','):
                root, filename = pathlib.Path('/sys/fs/cgroup/memory'), 'memory.limit_in_bytes'
            elif not controllers:
                root, filename = pathlib.Path('/sys/fs/cgroup'), 'memory.max'
            else:
                continue
            current = root/relative.lstrip('/')
            while current != root.parent and current.is_relative_to(root):
                try:
                    value = (current/filename).read_text().strip()
                    if value.isdigit() and int(value)>0:
                        limit = min(limit, int(value))
                except OSError:
                    pass
                current = current.parent
        if size + self.reserve_bytes > limit:
            raise MemoryError('indexed files plus 6 GiB reserve exceed the memory allowance')
        self.destination.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.created = True
        try:
            metadata = ('config.json', 'model.safetensors.index.json', 'tokenizer.json',
                        'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json',
                        'hf_quant_config.json', 'processor_config.json')
            for name in metadata:
                if (self.model/name).is_file():
                    shutil.copy2(self.model/name, self.destination/name)
            errors = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = [pool.submit(self._copy, name) for name in names]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        self.entries.append(future.result())
                    except BaseException as error:
                        errors.append(error)
                        self.stop.set()
                        for pending in futures:
                            pending.cancel()
            if errors:
                raise errors[0]
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        for entry in self.entries:
            self.libc.munmap(entry['pointer'], entry['bytes'])
            os.close(entry['descriptor'])
        self.entries.clear()
        if self.created:
            shutil.rmtree(self.destination)
            self.created = False

    def __enter__(self):
        return self.open()

    def __exit__(self, *unused):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=pathlib.Path)
    parser.add_argument('--cache-dir', type=pathlib.Path, required=True)
    parser.add_argument('--report', type=pathlib.Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--direct-read', action='store_true', help='use aligned O_DIRECT reads except the final partial page')
    args, command = parser.parse_known_args()
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        parser.error('provide the command to run after --; use {model} for the cached model path')
    if args.report.exists():
        parser.error('report already exists; choose a fresh path')
    started = time.monotonic()
    with RamCache(args.model, args.cache_dir, workers=args.workers, direct_read=args.direct_read) as cache:
        record = dict(backing='sealed memfd; one shared physical copy', startup_seconds=time.monotonic()-started,
                      direct_read=args.direct_read,
                      weights=[{k:v for k,v in entry.items() if k in ('name','bytes','sha256')} for entry in sorted(cache.entries, key=lambda row:row['name'])])
        args.report.write_text(json.dumps(record, indent=2)+'\n')
        environment = dict(os.environ, STRATA_GLM_FP4_RAM_FDS='1', GLM_FP4_MODEL=str(cache.destination))
        environment.pop('STRATA_GLM_FP4_RAM_COPY', None)
        print('RAM_CACHE_READY bytes=%d seconds=%.3f' % (sum(row['bytes'] for row in record['weights']), record['startup_seconds']), flush=True)
        result = subprocess.run([part.replace('{model}', str(cache.destination)) for part in command], env=environment)
        return result.returncode


if __name__ == '__main__':
    sys.exit(main())
