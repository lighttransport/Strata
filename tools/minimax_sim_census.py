#!/usr/bin/env python3
"""Tensor census of a MiniMax M3.1 Flash split GGUF for tools/sim (headers only; no tensor payload is read).

Writes the JSON that tools/sim/model.py loads as the `minimax_m31` model: per-layer routed-expert formats, shapes and
bytes, and the bytes of everything else (attention, indexer, dense FFN, shared expert, router, head) per layer.

    python3 tools/minimax_sim_census.py FIRST_SHARD.gguf --name minimax_m31_ud_q2_k_xl --output census.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402

SHARD = re.compile(r"^(?P<stem>.*)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$")


def shard_paths(first: pathlib.Path) -> list[pathlib.Path]:
    m = SHARD.match(first.name)
    if not m:
        return [first]
    count = int(m["count"])
    paths = [first.with_name(f"{m['stem']}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing shards: {missing}")
    return paths


def census(first: pathlib.Path, name: str) -> dict:
    paths = shard_paths(first)
    layers: dict[str, dict] = {}
    fixed_layer: dict[str, int] = {}
    shared: dict[str, int] = {}
    parts = {"lm_head": 0, "token_embd": 0, "output_norm": 0}
    totals = {"main_routed": 0, "main_fixed": 0}
    file_bytes = 0
    layer_count = 0
    for path in paths:
        file_bytes += path.stat().st_size
        g = GGUFFile(path)
        layer_count = max(layer_count, int(g.metadata.get("minimax-m3.block_count", 0)))
        for t in g.tensors:
            size = t.expected_bytes()
            if size is None:
                raise SystemExit(f"unknown storage for {t.name} ({t.type_name})")
            fields = t.name.split(".")
            if t.name == "output.weight":
                parts["lm_head"] += size
            elif t.name == "token_embd.weight":
                parts["token_embd"] += size
            elif t.name == "output_norm.weight":
                parts["output_norm"] += size
            elif fields[0] == "blk" and "_exps.weight" in t.name:
                row = layers.setdefault(fields[1], {})
                row[fields[2].removeprefix("ffn_").removesuffix("_exps")] = dict(
                    format=t.type_name, shape=t.shape, bytes=size)
                totals["main_routed"] += size
                continue
            elif fields[0] == "blk":
                fixed_layer[fields[1]] = fixed_layer.get(fields[1], 0) + size
                if "_shexp" in t.name:
                    shared[fields[1]] = shared.get(fields[1], 0) + size
            else:
                raise SystemExit(f"unexpected tensor {t.name}")
    totals["main_fixed"] = sum(fixed_layer.values()) + parts["lm_head"] + parts["token_embd"] + parts["output_norm"]
    experts = {t["shape"][-1] for row in layers.values() for t in row.values()}
    if len(experts) != 1 or len(fixed_layer) != layer_count or not parts["lm_head"]:
        raise SystemExit("incomplete census")
    return dict(
        name=name,
        source=f"split GGUF, {len(paths)} shards",
        file_bytes=file_bytes,
        experts=experts.pop(),
        layers=layers,
        bytes=totals,
        fixed=dict(layer={k: fixed_layer[k] for k in sorted(fixed_layer, key=int)},
                   shared={k: shared[k] for k in sorted(shared, key=int)}, **parts),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("first_shard", type=pathlib.Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--output", type=pathlib.Path)
    args = ap.parse_args()
    data = census(args.first_shard, args.name)
    text = json.dumps(data, indent=1)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
