#!/usr/bin/env python3
"""Read-only preflight for MiniMax M3.1 Flash split GGUFs; no model payloads are read."""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile


REQUIRED = {
    "block_count": 60,
    "embedding_length": 6144,
    "context_length": 1048576,
    "attention.head_count": 64,
    "attention.head_count_kv": 4,
    "attention.key_length": 128,
    "attention.value_length": 128,
    "expert_count": 128,
    "expert_used_count": 4,
    "leading_dense_block_count": 3,
    "attention.indexer.head_count": 4,
    "attention.indexer.key_length": 128,
    "attention.indexer.top_k": 16,
    "attention.indexer.block_size": 128,
    "attention.indexer.local_blocks": 1,
}
INDEXER = (
    "indexer.q_proj.weight",
    "indexer.k_proj.weight",
    "indexer.q_norm.weight",
    "indexer.k_norm.weight",
)


def inspect(path: pathlib.Path, quant: str = "UD-Q2_K_XL") -> dict:
    root = path if path.is_dir() else path.parent
    if path.is_dir():
        candidates = sorted(root.rglob("*.gguf"))
        # A model folder can contain several quantizations. The requested
        # quant must be identifiable from a directory or shard name; never
        # silently inspect a different variant.
        candidates = [p for p in candidates if quant in p.name or quant in p.parent.name]
    else:
        if quant not in path.name and quant not in path.parent.name:
            raise ValueError(f"selected file does not identify the requested quantization {quant}")
        candidates = [path]
    parsed = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            f = GGUFFile(candidate)
        except (OSError, ValueError):
            continue
        if f.metadata.get("general.architecture") == "minimax-m3":
            parsed.append(f)
    if not parsed:
        raise ValueError(f"no minimax-m3 GGUF shard for {quant} found")
    first = next((f for f in parsed if int(f.metadata.get("split.no", 0)) == 0), parsed[0])
    md = first.metadata
    if md.get("general.architecture") != "minimax-m3":
        raise ValueError("expected general.architecture=minimax-m3")
    prefix = "minimax-m3."
    for key, expected in REQUIRED.items():
        got = md.get(prefix + key)
        if got != expected:
            raise ValueError(f"{prefix + key}: expected {expected}, got {got!r}")
    split_count = int(md.get("split.count", 1))
    files_by_no = {int(f.metadata.get("split.no", 0)): f for f in parsed}
    if len(files_by_no) != len(parsed):
        raise ValueError("duplicate split.no values in matching MiniMax GGUF files")
    if len(files_by_no) != split_count or set(files_by_no) != set(range(split_count)):
        raise ValueError(f"found {len(files_by_no)} of {split_count} MiniMax model shards")
    files = [files_by_no[i] for i in range(split_count)]
    shards = [f.path for f in files]
    for i, f in enumerate(files):
        if int(f.metadata.get("split.count", 1)) != split_count or int(f.metadata.get("split.no", 0)) != i:
            raise ValueError(f"{f.path.name}: split metadata does not match shard order")

    by_name = {t.name: t for f in files for t in f.tensors}
    layers = int(md[prefix + "block_count"])
    hidden = int(md[prefix + "embedding_length"])
    ih = int(md[prefix + "attention.indexer.head_count"])
    ik = int(md[prefix + "attention.indexer.key_length"])
    expected_shapes = ((hidden, ih * ik), (hidden, ik), (ik,), (ik,))
    for layer in range(layers):
        for suffix, shape in zip(INDEXER, expected_shapes):
            name = f"blk.{layer}.{suffix}"
            t = by_name.get(name)
            if t is None:
                raise ValueError(f"missing M3.1 sparse-attention tensor {name}")
            if tuple(t.shape) != shape:
                raise ValueError(f"{name}: expected shape {shape}, got {tuple(t.shape)}")

    types = collections.Counter(t.type_name for t in by_name.values())
    unknown = sorted({t.type_name for t in by_name.values() if t.expected_bytes() is None})
    sizes = [p.stat().st_size for p in shards]
    total = sum(sizes)
    return {
        "architecture": "minimax-m3",
        "shards": [{"name": p.name, "bytes": n} for p, n in zip(shards, sizes)],
        "total_bytes": total,
        "indexer_layers_checked": layers,
        "tensors": len(by_name),
        "tensor_types": dict(sorted(types.items())),
        "types_without_known_size_geometry": unknown,
        "ud_q2_fit_check": {
            "slurm_memory_bytes": 181 * 1024**3,
            "gguf_bytes_below_181G": total < 181 * 1024**3,
            "remaining_bytes_before_runtime": 181 * 1024**3 - total,
            "note": "file size is a residency estimate; runtime, context and OS memory also need headroom",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=pathlib.Path, help="UD-Q2_K_XL directory or first GGUF shard")
    ap.add_argument("--quant", default="UD-Q2_K_XL", help="quantization name to select from a model root")
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args()
    try:
        result = inspect(args.model, args.quant)
    except (OSError, ValueError, KeyError) as e:
        print(f"MiniMax preflight failed: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("MiniMax M3.1 Flash GGUF structure preflight: PASS")
        print(f"  shards: {len(result['shards'])}; total bytes: {result['total_bytes']}")
        print(f"  tensors: {result['tensors']}; indexer tensors validated on {result['indexer_layers_checked']} layers")
        print(f"  tensor types: {', '.join(f'{k}={v}' for k, v in result['tensor_types'].items())}")
        print(f"  unrecognized tensor byte geometries: {result['types_without_known_size_geometry']}")
        print(f"  GGUF bytes below 181 GiB: {result['ud_q2_fit_check']['gguf_bytes_below_181G']}")
        print(f"  {result['ud_q2_fit_check']['note']}")
    if not result["ud_q2_fit_check"]["gguf_bytes_below_181G"]:
        print("MiniMax preflight failed: GGUF shard bytes exceed the 181 GiB allocation", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
