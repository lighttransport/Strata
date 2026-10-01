"""Place existing GLM expert file pages across two NUMA nodes without changing weights."""
import argparse
import collections
import ctypes
import mmap
import pathlib
import re
import time

from gguf_reader import GGUFFile


def memory_available():
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def check_pressure():
    if memory_available() < 32 * 1024**3:
        raise RuntimeError("NUMA migration stopped: less than 32 GiB host memory available")
    for line in pathlib.Path("/proc/pressure/memory").read_text().splitlines():
        if float(line.split()[1].split("=")[1]) > 5:
            raise RuntimeError("NUMA migration stopped due to memory pressure")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--node0-percent", type=int, default=40)
    args = parser.parse_args()
    if not 1 <= args.node0_percent <= 99:
        parser.error("node0 percentage must be 1..99")
    if not all(pathlib.Path(f"/sys/devices/system/node/node{n}").exists() for n in (0, 1)):
        parser.error("requires NUMA nodes 0 and 1")
    lib = ctypes.CDLL("libnuma.so.1", use_errno=True)
    lib.numa_move_pages.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    page_size = mmap.PAGESIZE
    processed = 0
    errors = collections.Counter()
    start = time.monotonic()
    for path in sorted(args.model.parent.glob("*.gguf")):
        gguf = GGUFFile(path)
        regions = []
        for tensor in gguf.tensors:
            match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", tensor.name)
            if match and int(match[1]) < 45:
                size = tensor.expected_bytes()
                if size is None:
                    raise ValueError("unsupported expert tensor")
                begin = (gguf.data_start + tensor.offset) // page_size * page_size
                end = (gguf.data_start + tensor.offset + size + page_size - 1) // page_size * page_size
                if gguf.data_start + tensor.offset + size > path.stat().st_size:
                    raise ValueError("expert tensor extends beyond shard")
                regions.append((begin, end))
        merged = []
        for begin, end in sorted(regions):
            if merged and begin <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((begin, end))
        regions = merged
        total = sum(end - begin for begin, end in regions)
        if not regions:
            continue
        print(f"{path.name}: expert_MiB={total / 1024**2:.1f}", flush=True)
        if not args.apply:
            continue
        with path.open("rb") as file:
            mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY)
            marker = ctypes.c_char.from_buffer(mapping)
            base = ctypes.addressof(marker)
            try:
                for begin, end in regions:
                    for offset in range(begin, end, 16384 * page_size):
                        check_pressure()
                        offsets = range(offset, min(end, offset + 16384 * page_size), page_size)
                        for p in offsets:
                            _ = mapping[p]
                        pages = (ctypes.c_void_p * len(offsets))(*(base + p for p in offsets))
                        # Spread each hundred-page group according to node capacity.
                        nodes = (ctypes.c_int * len(offsets))(*(0 if (p // page_size) % 100 < args.node0_percent else 1 for p in offsets))
                        status = (ctypes.c_int * len(offsets))()
                        result = lib.numa_move_pages(0, len(offsets), pages, nodes, status, 2)
                        if result < 0:
                            raise OSError(ctypes.get_errno(), "NUMA migration failed")
                        errors.update(n for n in status if n < 0)
                        processed += len(offsets) * page_size
            finally:
                del marker
                mapping.close()
        print(f"processed_MiB={processed / 1024**2:.1f} errors={dict(errors)}", flush=True)
    print(f"elapsed_s={time.monotonic() - start:.3f}")


if __name__ == "__main__":
    main()
