#!/usr/bin/env python3
"""A synthetic DeepSeek V4.1 Flash imatrix (constant importance) for timing IQ formats before a real one exists.

    python3 tools/ds41f_synth_imatrix.py OUT.gguf [--layers 0-11] [--experts 384]

Tensors `blk.L.ffn_{gate,up,down}_exps.weight.in_sum2` F32 [cols, experts] and `.counts` {1, experts} (the shape
llama-imatrix writes); gate/up cols 5120, down 2304. Needs gguf-py (STRATA_GGUF_PY or a llama.cpp checkout next to the repo).
Never use it for a real pack: every column gets the same importance.
"""
import argparse, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_ds41f_pack import gguf_py  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--layers", default="0-11")
    ap.add_argument("--experts", type=int, default=384)
    a = ap.parse_args()
    gg = gguf_py()
    if gg is None:
        sys.exit("gguf-py not found (set STRATA_GGUF_PY)")
    first, _, last = a.layers.partition("-")
    first, last = int(first), int(last or first)
    w = gg.GGUFWriter(a.out, "imatrix")
    w.add_type("imatrix")
    w.add_string("imatrix.datasets", "synthetic-constant (timing only)")
    for layer in range(first, last + 1):
        for proj, cols in (("gate", 5120), ("up", 5120), ("down", 2304)):
            name = f"blk.{layer}.ffn_{proj}_exps.weight"
            w.add_tensor(name + ".in_sum2", np.full((a.experts, cols), 100.0, np.float32))
            w.add_tensor(name + ".counts", np.full((a.experts, 1), 10.0, np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {a.out}: layers {first}-{last}, {a.experts} experts")


if __name__ == "__main__":
    main()
