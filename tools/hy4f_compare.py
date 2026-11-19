#!/usr/bin/env python3
"""Compare strata-hy4f --dump output with a llama.cpp dump (tools/ds41f_llama_dump.cpp, same .bin format).

    python3 tools/hy4f_compare.py REF_DIR STRATA_DIR

Per block l_out-<il> (the four streams [n][4][pos]): relative L2 error and cosine per position; result_output: top-1
agreement and mean KL(ref || strata) over positions. Pure Python (no numpy) so it runs anywhere.
"""
import math
import pathlib
import struct
import sys
from array import array


def load(path):
    b = pathlib.Path(path).read_bytes()
    t, = struct.unpack_from("<i", b, 0)
    ne = struct.unpack_from("<4q", b, 4)
    if t != 0:
        raise SystemExit(f"{path}: type {t}, expected f32")
    v = array("f")
    v.frombytes(b[36:])
    return ne, v


def rows(ne, v, width):
    n = len(v) // width
    return [v[i * width:(i + 1) * width] for i in range(n)]


def compare_vec(a, b):
    num = sum((x - y) ** 2 for x, y in zip(a, b))
    den = sum(x * x for x in a) or 1e-30
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(den), math.sqrt(sum(y * y for y in b)) or 1e-30
    return math.sqrt(num / den), dot / (na * nb)


def softmax_log(v):
    m = max(v)
    s = sum(math.exp(x - m) for x in v)
    ls = m + math.log(s)
    return [x - ls for x in v]


def main():
    ref, mine = map(pathlib.Path, sys.argv[1:3])
    il = 0
    worst = 0.0
    while (mine / f"l_out-{il}.bin").exists():
        rf = ref / f"l_out-{il}.bin"
        if not rf.exists():
            print(f"l_out-{il}: no reference")
            break
        ne_r, vr = load(rf)
        ne_m, vm = load(mine / f"l_out-{il}.bin")
        w = ne_m[0] * ne_m[1]
        rr, mm = rows(ne_r, vr, w), rows(ne_m, vm, w)
        stats = [compare_vec(a, b) for a, b in zip(rr, mm)]
        worst = max(worst, max(s[0] for s in stats))
        print(f"l_out-{il:<2} positions {len(stats)}  rel err " + " ".join(f"{s[0]:.2e}" for s in stats) +
              "  cos min " + f"{min(s[1] for s in stats):.6f}")
        il += 1
    if (mine / "result_output.bin").exists() and (ref / "result_output.bin").exists():
        ne_r, vr = load(ref / "result_output.bin")
        ne_m, vm = load(mine / "result_output.bin")
        V = ne_m[0]
        rr, mm = rows(ne_r, vr, V), rows(ne_m, vm, V)
        agree, kls = 0, []
        for a, b in zip(rr, mm):
            ta = max(range(V), key=a.__getitem__)
            tb = max(range(V), key=b.__getitem__)
            agree += ta == tb
            la, lb = softmax_log(a), softmax_log(b)
            kls.append(sum(math.exp(x) * (x - y) for x, y in zip(la, lb)))
            print(f"  pos top-1 ref {ta} strata {tb}  KL {kls[-1]:.2e}")
        print(f"result_output: top-1 {agree}/{len(kls)}, mean KL {sum(kls) / len(kls):.3e}")
    print(f"worst block rel err {worst:.2e}")


if __name__ == "__main__":
    main()


def named(ref, mine, name):
    """python3 -c 'import hy4f_compare as c; c.named(REF, MINE, "ffn_moe_out-1")' - per-position rel err of one tensor"""
    ne_r, vr = load(pathlib.Path(ref) / f"{name}.bin")
    ne_m, vm = load(pathlib.Path(mine) / f"{name}.bin")
    w = ne_m[0]
    rr, mm = rows(ne_r, vr, w), rows(ne_m, vm, w)
    print(name, " ".join(f"{compare_vec(a, b)[0]:.2e}" for a, b in zip(rr, mm)))
