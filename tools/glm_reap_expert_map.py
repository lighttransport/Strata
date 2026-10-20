#!/usr/bin/env python3
"""Recover which original experts a REAP-pruned GLM-5.3-Flash GGUF kept, for strata-glm-q2-pack --expert-map.

A pruned model keeps the router rows (ffn_gate_inp.weight, F32) and expert biases (exp_probs_b.bias) of the experts it
keeps, unchanged and in order. Each pruned row is matched to the original row that equals it exactly; the result is one
line per layer, "LAYER e0 e1 ...", the original expert index behind each pruned expert.

    tools/glm_reap_expert_map.py PRUNED.gguf ORIGINAL-00001-of-0000N.gguf [more shards...] > map.txt

ORIGINAL is any GGUF of the unpruned model with F32 routers (e.g. Unsloth's UD-Q2_K_XL shards); pass every shard.
Needs gguf-py (pip install gguf, or PYTHONPATH=<llama.cpp>/gguf-py) and numpy.
"""
import re
import sys

import numpy as np
from gguf import GGUFReader


def routers(path):
    out = {}
    for t in GGUFReader(path).tensors:
        if t.name.endswith(("ffn_gate_inp.weight", "exp_probs_b.bias")):
            if t.tensor_type.name != "F32":
                raise SystemExit(f"{path}: {t.name} is {t.tensor_type.name}, expected F32")
            out[t.name] = np.array(t.data, dtype=np.float32).reshape([int(x) for x in reversed(t.shape)])
    return out


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    pruned = routers(sys.argv[1])
    original = {}
    for shard in sys.argv[2:]:
        original.update(routers(shard))
    layers = sorted(int(re.match(r"blk\.(\d+)\.", n)[1]) for n in pruned if n.endswith("ffn_gate_inp.weight"))
    if not layers:
        raise SystemExit("no router tensors in the pruned model")
    for layer in layers:
        name = f"blk.{layer}.ffn_gate_inp.weight"
        if name not in original:
            raise SystemExit(f"the original model has no {name}")
        rows, full = pruned[name], original[name]
        kept = []
        for e, row in enumerate(rows):
            exact = np.flatnonzero((full == row).all(axis=1))
            if len(exact) != 1:
                raise SystemExit(f"layer {layer} expert {e}: {len(exact)} exact matches in the original router")
            kept.append(int(exact[0]))
        if len(set(kept)) != len(kept):
            raise SystemExit(f"layer {layer}: two pruned experts map to one original expert")
        bias = name.replace("ffn_gate_inp.weight", "exp_probs_b.bias")
        if bias in pruned and not np.array_equal(pruned[bias], original[bias][kept]):
            raise SystemExit(f"layer {layer}: expert biases differ from the matched experts'")
        print(layer, *kept)


if __name__ == "__main__":
    main()
