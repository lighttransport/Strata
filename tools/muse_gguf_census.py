#!/usr/bin/env python3
"""Census of a Muse Spark Open (`muse-spark`) GGUF from its headers only: no tensor data is read.

The weights do not have to be on disk. Every shard starts with its own tensor table, so the first megabyte of each
shard (an HTTP range request) is enough:

    B=https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-GGUF/resolve/main/UD-IQ1_S
    curl -sL $B/Muse-Spark-Open-816B-A42B-UD-IQ1_S-00001-of-00005.gguf -o s1.gguf        # metadata shard, 13 MB
    for i in 2 3 4 5; do curl -sL -r 0-1048575 $B/Muse-Spark-Open-816B-A42B-UD-IQ1_S-0000$i-of-00005.gguf -o h$i.gguf; done
    python3 tools/muse_gguf_census.py s1.gguf h2.gguf h3.gguf h4.gguf h5.gguf --name ud-iq1s \
        --output tools/sim/data/census-muse-spark-ud-iq1s.json

The output holds the hyperparameters the runner and the simulator need, and per tensor kind its format(s), shape
and total bytes (exact, from the ggml block geometry).
"""
import argparse
import collections
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import BLOCK_GEOMETRY, GGUFFile  # noqa: E402

ARCH = "muse-spark"
KEYS = ("block_count", "context_length", "embedding_length", "feed_forward_length", "attention.head_count",
        "attention.head_count_kv", "attention.key_length", "attention.value_length", "attention.sliding_window",
        "attention.layer_norm_rms_epsilon", "rope.freq_base", "rope.freq_base_swa", "expert_count", "expert_used_count",
        "expert_feed_forward_length", "moe_latent_size", "expert_weights_scale", "expert_weights_norm",
        "expert_gating_func", "logit_scale", "final_logit_softcapping", "post_norm_epsilon")


def tensor_bytes(t):
    block, nbytes = BLOCK_GEOMETRY[t.type_name]
    elements = 1
    for s in t.shape:
        elements *= s
    return elements // block * nbytes, elements


def census(paths, name):
    meta, tensors = {}, []
    for p in paths:
        g = GGUFFile(pathlib.Path(p))
        meta.update(g.metadata)
        tensors += g.tensors
    if meta.get("general.architecture") != ARCH:
        raise SystemExit(f"not a {ARCH} GGUF: general.architecture = {meta.get('general.architecture')!r}")
    expected = meta.get("split.tensors.count")
    if expected is not None and expected != len(tensors):
        raise SystemExit(f"headers list {len(tensors)} tensors, split.tensors.count says {expected}: pass every shard")
    hp = {k: meta[f"{ARCH}.{k}"] for k in KEYS if f"{ARCH}.{k}" in meta}
    pattern = meta[f"{ARCH}.attention.sliding_window_pattern"]
    kinds = collections.OrderedDict()
    per_layer = collections.defaultdict(dict)
    for t in sorted(tensors, key=lambda t: t.name):
        m = re.fullmatch(r"blk\.(\d+)\.(.+)", t.name)
        kind = m.group(2) if m else t.name
        nbytes, elements = tensor_bytes(t)
        k = kinds.setdefault(kind, dict(count=0, bytes=0, elements=0, formats={}, shape=list(t.shape)))
        k["count"] += 1
        k["bytes"] += nbytes
        k["elements"] += elements
        k["formats"][t.type_name] = k["formats"].get(t.type_name, 0) + 1
        if m:
            per_layer[int(m.group(1))][kind] = t.type_name
    expert_formats = {str(l): [row[f"ffn_{p}_exps.weight"] for p in ("gate", "up", "down")]
                      for l, row in sorted(per_layer.items())}
    total = sum(k["bytes"] for k in kinds.values())
    return dict(
        schema="strata.muse.census.v1", name=name, architecture=ARCH, model=meta.get("general.name"),
        file_type=meta.get("general.file_type"), tensors=len(tensors), bytes=total, hyperparameters=hp,
        # sliding_window_pattern: True = sliding-window layer, False = full attention
        full_attention_layers=[i for i, swa in enumerate(pattern) if not swa],
        tokens=dict(vocab=len(meta.get("tokenizer.ggml.tokens", [])), bos=meta.get("tokenizer.ggml.bos_token_id"),
                    eos=meta.get("tokenizer.ggml.eos_token_id"), eot=meta.get("tokenizer.ggml.eot_token_id")),
        kinds=kinds, expert_formats=expert_formats)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="every shard (or its first megabyte)")
    ap.add_argument("--name", default="gguf")
    ap.add_argument("--output")
    args = ap.parse_args(argv)
    data = census(args.paths, args.name)
    text = json.dumps(data, indent=1) + "\n"
    if args.output:
        pathlib.Path(args.output).write_text(text)
    else:
        sys.stdout.write(text)
    print(f"{data['model']}: {data['tensors']} tensors, {data['bytes'] / 1e9:.2f} GB", file=sys.stderr)


if __name__ == "__main__":
    main()
