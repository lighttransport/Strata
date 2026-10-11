#!/usr/bin/env python3
"""Read-only Linux sequential direct-I/O benchmark of an existing regular file."""
import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import stat
import statistics
import sys
import time


def benchmark(path, mib=1024, block_mib=8, repeats=3, timeout=120):
    if not 1 <= mib <= 65536 or not 1 <= block_mib <= 64 or not 2 <= repeats <= 15 or timeout <= 0:
        raise ValueError('Read size 1..65536 MiB, block 1..64 MiB, repeats 2..15, positive timeout')
    if not hasattr(os, 'O_DIRECT'):
        raise RuntimeError('Linux O_DIRECT is required; no buffered fallback')
    # Reject devices, pipes and symlinks; never create or truncate the source.
    if not stat.S_ISREG(os.lstat(path).st_mode):
        raise ValueError('Disk input must be an existing regular file, not a symlink or device')
    flags = os.O_RDONLY | os.O_DIRECT | os.O_NOFOLLOW | os.O_NONBLOCK
    fd = os.open(path, flags)
    started = time.monotonic()
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('Disk input is not a regular file')
        count = min(mib*1048576, before.st_size)//4096*4096
        if count < 1048576:
            raise ValueError('Disk input must contain at least 1 MiB of aligned data')
        block = min(block_mib*1048576, count)
        samples = []
        reference = None
        # Anonymous mmap supplies a page-aligned buffer for direct I/O.
        with mmap.mmap(-1, block) as buffer:
            view = memoryview(buffer)
            try:
                for rep in range(repeats):
                    os.lseek(fd, 0, os.SEEK_SET)
                    digest = hashlib.sha256()
                    io_seconds = 0.0
                    sample_start = time.monotonic()
                    offset = 0
                    while offset < count:
                        if time.monotonic()-started > timeout:
                            raise TimeoutError('Disk read deadline exceeded')
                        length = min(block, count-offset)
                        part = view[:length]
                        try:
                            begin = time.perf_counter()
                            got = os.readv(fd, [part])
                            io_seconds += time.perf_counter()-begin
                            if got != length:
                                raise RuntimeError(f'Short direct read: expected {length}, got {got}')
                            # Hashing is outside the read-call timing; wall rate includes it.
                            digest.update(part)
                        finally:
                            part.release()
                        offset += got
                    checksum = digest.hexdigest()
                    if reference is None:
                        reference = checksum
                    elif checksum != reference:
                        raise RuntimeError('Repeated disk reads returned different data')
                    after = os.fstat(fd)
                    if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != \
                            (before.st_size, before.st_mtime_ns, before.st_ctime_ns):
                        raise RuntimeError('Disk input changed during measurement')
                    elapsed = time.monotonic()-sample_start
                    samples.append({'repeat': rep, 'bytes': count, 'read_seconds': io_seconds,
                                    'wall_seconds': elapsed, 'read_GB_s': count/io_seconds/1e9,
                                    'wall_GB_s': count/elapsed/1e9, 'sha256': checksum})
            finally:
                view.release()
        return {'passed': True, 'mode': 'sequential_direct_read', 'path': str(Path(path).absolute()),
                'file_bytes': before.st_size, 'bytes_per_repeat': count, 'block_bytes': block,
                'repeats': repeats, 'offset': 0, 'page_cache_bypassed': True,
                'device_cache_bypassed': False, 'repeated_data_verified': True,
                'median_read_GB_s': statistics.median(s['read_GB_s'] for s in samples),
                'median_wall_GB_s': statistics.median(s['wall_GB_s'] for s in samples),
                'samples': samples}
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path)
    parser.add_argument('--mib', type=int, default=1024)
    parser.add_argument('--block-mib', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    code = 0
    try:
        result = benchmark(args.file, args.mib, args.block_mib, args.repeats, args.timeout)
    except (OSError, ValueError, RuntimeError) as error:
        result = {'passed': False, 'error': str(error)}
        code = 2
    except KeyboardInterrupt:
        result = {'passed': False, 'error': 'Interrupted'}
        code = 130
    payload = json.dumps(result, indent=2)+'\n'
    if args.output:
        args.output.write_text(payload)
    print(payload, end='', flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
