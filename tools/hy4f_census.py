#!/usr/bin/env python3
"""Byte census of a Hy4-Flash (hy_v4_flash) GGUF for tools/sim/hy4f.py.

    uv run --no-project --with numpy --with pyyaml --with tqdm python -I tools/hy4f_census.py \
        --gguf-py LLAMA_CPP/gguf-py MODEL.gguf > tools/sim/data/hy4f_census.json

Per block: routed-expert bytes (gate/up/down, one expert), the rest of the block (read every token on the GPU), and
the metadata the simulator needs (head counts, SWA / indexer / Engram layers). Global tensors listed by name.
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--gguf-py", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.gguf_py)
    import gguf  # noqa: E402

    r = gguf.GGUFReader(a.gguf)
    arch = r.fields["general.architecture"].contents()
    meta = {}
    for k, f in r.fields.items():
        if k.startswith(arch + "."):
            v = f.contents()
            key = k[len(arch) + 1:]
            if key in ("engram.token_map", "ple.layer_multipliers", "ple.head_offsets", "ple.head_vocab_sizes"):
                v = len(v)   # sizes only
            meta[key] = v
    blocks = {}
    glob = {}
    for t in r.tensors:
        p = t.name.split(".")
        if p[0] == "blk":
            b = blocks.setdefault(int(p[1]), dict(expert_bytes=0, experts=0, dense_bytes=0, tensors={}))
            name = ".".join(p[2:])
            b["tensors"][name] = [t.tensor_type.name, int(t.n_bytes)]
            if name.endswith("_exps.weight"):
                b["experts"] = int(t.shape[-1])
                b["expert_bytes"] += int(t.n_bytes) // int(t.shape[-1])
            else:
                b["dense_bytes"] += int(t.n_bytes)
        else:
            glob[t.name] = [t.tensor_type.name, int(t.n_bytes)]
    print(json.dumps(dict(arch=arch, file=os.path.basename(a.gguf), meta=meta, globals=glob,
                          blocks=[blocks[i] for i in sorted(blocks)]), indent=1))


if __name__ == "__main__":
    main()
