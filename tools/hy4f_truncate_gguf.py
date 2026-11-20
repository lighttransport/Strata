#!/usr/bin/env python3
"""A Hy4-Flash GGUF cut to its first L blocks (or a chosen set, --blocks), for partial-layer parity against llama.cpp without loading the whole model.

    uv run --no-project --with numpy --with pyyaml --with tqdm python -I tools/hy4f_truncate_gguf.py \
        --gguf-py LLAMA_CPP/gguf-py MODEL.gguf OUT.gguf --layers 2

The cut model has block_count L and no MTP block (nextn_predict_layers 0); every per-layer metadata array is cut to its
first L entries. Engram is dropped (the 62.7 GiB table is not copied): the ple.* / engram.* keys, the table and the
blk.*.engram_* tensors go, so both llama.cpp and strata-hy4f run the blocks without it. The output head is kept, so
the logits of the cut model compare too.
"""
import argparse
import sys


class Rows:
    """Row ranges of a memory-mapped table, written to the file piece by piece (the slices are not loaded at once)."""

    def __init__(self, t, ranges):
        self.t, self.ranges = t, ranges
        self.shape = (sum(hi - lo for lo, hi in ranges),) + tuple(t.shape[1:])
        self.dtype = t.dtype
        self.nbytes = self.shape[0] * (t.nbytes // t.shape[0])

    def tofile(self, f):
        step = max(1, (1 << 30) // (self.t.nbytes // self.t.shape[0]))
        for lo, hi in self.ranges:
            for r in range(lo, hi, step):
                self.t[r:min(hi, r + step)].tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--layers", type=int, help="keep blocks 0..L-1")
    ap.add_argument("--blocks", help="keep these blocks (comma list, ascending), renumbered 0..; their Engram modules' "
                    "table rows are copied and re-based, so later modules can be checked without the blocks in between")
    ap.add_argument("--gguf-py", required=True)
    ap.add_argument("--keep-engram", action="store_true",
                    help="keep the Engram modules of the kept blocks; the table is cut to the rows their hash buckets span")
    a = ap.parse_args()
    sys.path.insert(0, a.gguf_py)
    import gguf  # noqa: E402
    from gguf.constants import GGUFValueType as VT  # noqa: E402

    r = gguf.GGUFReader(a.src)
    arch = r.fields["general.architecture"].contents()
    blocks = int(r.fields[f"{arch}.block_count"].contents())
    n_main = blocks - int(r.fields[f"{arch}.nextn_predict_layers"].contents()) if f"{arch}.nextn_predict_layers" in r.fields else blocks
    if (a.layers is None) == (a.blocks is None):
        raise SystemExit("give one of --layers / --blocks")
    sel = list(range(a.layers)) if a.layers is not None else [int(x) for x in a.blocks.split(",")]
    if not sel or sel != sorted(set(sel)) or sel[-1] >= n_main:
        raise SystemExit(f"blocks must be ascending, distinct, below {n_main}")
    L = len(sel)
    renum = {old: new for new, old in enumerate(sel)}
    w = gguf.GGUFWriter(a.dst, arch, use_temp_file=False)
    skip_prefixes = (f"{arch}.ple.", f"{arch}.engram.")
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture", f"{arch}.embedding_length_per_layer_input"}
    table_rows = 0
    if a.keep_engram:
        F = lambda k: r.fields[f"{arch}.{k}"].contents()  # noqa: E731
        e_layers = [int(x) for x in F("ple.layers")]
        mods = [e for e, x in enumerate(e_layers) if x in renum]
        if not mods:
            raise SystemExit("--keep-engram: no Engram layer among the kept blocks")
        ngram, heads = int(F("ple.ngram_size")), int(F("ple.heads_per_ngram"))
        cols = (ngram - 1) * heads
        O, Z, MU = F("ple.head_offsets"), F("ple.head_vocab_sizes"), F("ple.layer_multipliers")
        ranges, offs, base = [], [], 0   # each module's rows [lo, hi) are contiguous in the table
        for e in mods:
            lo = min(int(O[e * cols + c]) for c in range(cols))
            hi = max(int(O[e * cols + c]) + int(Z[e * cols + c]) for c in range(cols))
            ranges.append((lo, hi))
            offs += [int(O[e * cols + c]) - lo + base for c in range(cols)]
            base += hi - lo
        table_rows = base
        cut = {"ple.layers": [renum[e_layers[e]] for e in mods],
               "ple.layer_multipliers": [MU[e * ngram + j] for e in mods for j in range(ngram)],
               "ple.head_offsets": offs, "ple.head_vocab_sizes": [Z[e * cols + c] for e in mods for c in range(cols)]}
    for key, f in r.fields.items():
        emb_key = f"{arch}.embedding_length_per_layer_input"
        drop = key in skip and not (a.keep_engram and key == emb_key)
        drop = drop or (not a.keep_engram and key.startswith(skip_prefixes))
        if drop:
            continue
        val = f.contents()
        vtype = f.types[0]
        sub = f.types[-1] if vtype == VT.ARRAY else None
        short = key[len(arch) + 1:] if key.startswith(arch + ".") else ""
        if a.keep_engram and short in cut:
            val = cut[short]
        elif key == f"{arch}.block_count":
            val = L
        elif key == f"{arch}.nextn_predict_layers":
            val = 0
        elif vtype == VT.ARRAY and short not in ("engram.token_map",) and key.startswith(arch + ".") and len(val) in (blocks, blocks - 1):
            val = [val[i] for i in sel]
        w.add_key_value(key, val, vtype, sub_type=sub)
    keep = []
    for t in r.tensors:
        n = t.name
        if n == "per_layer_token_embd.weight" and not a.keep_engram:
            continue
        if n.startswith("blk."):
            il = int(n.split(".")[1])
            if il not in renum or (".engram_" in n and not a.keep_engram):
                continue
        keep.append(t)
    data = {}
    names = {}
    for t in keep:
        d = t.data
        if t.name == "per_layer_token_embd.weight":
            d = Rows(d, ranges)
        n = t.name
        if n.startswith("blk."):
            p = n.split(".")
            n = ".".join(["blk", str(renum[int(p[1])])] + p[2:])
        names[t.name] = n
        data[t.name] = d
        w.add_tensor_info(n, d.shape, d.dtype, int(d.nbytes), raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in keep:
        w.write_tensor_data(data[t.name])
    w.close()
    print(f"{a.dst}: {L} blocks, {len(keep)} tensors, {sum(int(d.nbytes) for d in data.values()) / 2**30:.2f} GiB"
          + (f", Engram table {table_rows} rows" if table_rows else ""))


if __name__ == "__main__":
    main()
