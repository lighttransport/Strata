"""Sample resident GLM expert row placement without migrating pages."""
import argparse
import collections
import ctypes
import json
import mmap
import pathlib
import re

from gguf_reader import GGUFFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    lib = ctypes.CDLL("libnuma.so.1", use_errno=True)
    lib.numa_move_pages.argtypes = [ctypes.c_int, ctypes.c_ulong,
                                   ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int),
                                   ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    tensors = collections.defaultdict(dict)
    mappings = []
    try:
        for path in sorted(args.model.parent.glob("*.gguf")):
            gguf = GGUFFile(path)
            regions = [(t, re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", t.name))
                       for t in gguf.tensors]
            regions = [(t, m) for t, m in regions if m and int(m[1]) < 45]
            if not regions:
                continue
            with path.open("rb") as file:
                mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY)
            mappings.append(mapping)
            marker = ctypes.c_char.from_buffer(mapping)
            base = ctypes.addressof(marker)
            del marker
            for tensor, match in regions:
                size = tensor.expected_bytes()
                if size is None or len(tensor.shape) != 3:
                    raise ValueError("unsupported expert tensor")
                offset = gguf.data_start + tensor.offset
                if offset + size > len(mapping):
                    raise ValueError("expert tensor exceeds shard")
                tensors[int(match[1])][match[2]] = (mapping, base, offset, tensor.shape, size)
        if set(tensors) != set(range(3, 45)) or any(set(v) != {"gate", "up", "down"} for v in tensors.values()):
            raise ValueError("incomplete main expert tensors")
        counts = collections.Counter()
        chunks = collections.Counter()
        for layer, parts in sorted(tensors.items()):
            for expert in [(7 + 37 * i) % 288 for i in range(8)]:
                for phase, names in (("gate_up", ("gate", "up")), ("down", ("down",))):
                    rows = parts[names[0]][3][1]
                    for row in range(0, rows, 128):
                        addresses = []
                        for name in names:
                            mapping, base, offset, shape, size = parts[name]
                            if shape[1] != rows or shape[2] != 288:
                                raise ValueError("inconsistent row geometry")
                            row_bytes = size // shape[2] // rows
                            for r in (row, row + min(63, rows - row - 1), min(row + 127, rows - 1)):
                                pos = offset + expert * (size // 288) + r * row_bytes
                                # Read only: establish the mapping's PTE for the placement query.
                                _ = mapping[pos]
                                addresses.append(base + pos // mmap.PAGESIZE * mmap.PAGESIZE)
                        pages = (ctypes.c_void_p * len(addresses))(*addresses)
                        status = (ctypes.c_int * len(addresses))()
                        result = lib.numa_move_pages(0, len(addresses), pages, None, status, 0)
                        if result < 0:
                            raise OSError(ctypes.get_errno(), "read-only page placement query failed")
                        counts.update(status)
                        nodes = set(status)
                        label = "unknown" if any(n < 0 for n in nodes) else "one_node" if len(nodes) == 1 else "mixed"
                        chunks[phase + ":" + label] += 1
        report = dict(model=args.model.name, experts_per_layer=8, row_chunk=128,
                      samples_per_matrix_chunk=3, page_node_counts=dict(counts),
                      chunk_sample_classification=dict(chunks),
                      migration_performed=False,
                      limitations="Sparse first/middle/last row-page samples; not proof of whole-chunk residency, current worker locality, or decode throughput. No page migration or private model copy.")
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        for mapping in mappings:
            mapping.close()


if __name__ == "__main__":
    main()
