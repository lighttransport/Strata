#!/usr/bin/env python3
"""A Hy4-Flash GGUF cut to its first L blocks, for partial-layer parity against llama.cpp without loading the whole model.

    uv run --no-project --with numpy --with pyyaml --with tqdm python -I tools/hy4f_truncate_gguf.py \
        --gguf-py LLAMA_CPP/gguf-py MODEL.gguf OUT.gguf --layers 2

The cut model has block_count L and no MTP block (nextn_predict_layers 0); every per-layer metadata array is cut to its
first L entries. Engram is dropped (the 62.7 GiB table is not copied): the ple.* / engram.* keys, the table and the
blk.*.engram_* tensors go, so both llama.cpp and strata-hy4f run the blocks without it. The output head is kept, so
the logits of the cut model compare too.
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--layers", type=int, required=True)
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
    L = a.layers
    if not 0 < L < blocks:
        raise SystemExit(f"--layers must be in 1..{blocks - 1}")
    w = gguf.GGUFWriter(a.dst, arch, use_temp_file=False)
    skip_prefixes = (f"{arch}.ple.", f"{arch}.engram.")
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture", f"{arch}.embedding_length_per_layer_input"}
    table_rows = 0
    if a.keep_engram:
        F = lambda k: r.fields[f"{arch}.{k}"].contents()  # noqa: E731
        e_layers = [int(x) for x in F("ple.layers")]
        mods = sum(1 for x in e_layers if x < L)
        if not mods:
            raise SystemExit("--keep-engram: no Engram layer below --layers")
        ngram, heads = int(F("ple.ngram_size")), int(F("ple.heads_per_ngram"))
        cols = (ngram - 1) * heads
        offs, sizes = F("ple.head_offsets")[:mods * cols], F("ple.head_vocab_sizes")[:mods * cols]
        table_rows = max(int(o) + int(z) for o, z in zip(offs, sizes))
        cut = {"ple.layers": mods, "ple.layer_multipliers": mods * ngram, "ple.head_offsets": mods * cols,
               "ple.head_vocab_sizes": mods * cols}
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
            val = val[:cut[short]]
        elif key == f"{arch}.block_count":
            val = L
        elif key == f"{arch}.nextn_predict_layers":
            val = 0
        elif vtype == VT.ARRAY and short not in ("engram.token_map",) and key.startswith(arch + ".") and len(val) in (blocks, blocks - 1):
            val = val[:L]
        w.add_key_value(key, val, vtype, sub_type=sub)
    keep = []
    for t in r.tensors:
        n = t.name
        if n == "per_layer_token_embd.weight" and not a.keep_engram:
            continue
        if n.startswith("blk."):
            il = int(n.split(".")[1])
            if il >= L or (".engram_" in n and not a.keep_engram):
                continue
        keep.append(t)
    data = {}
    for t in keep:
        d = t.data
        if t.name == "per_layer_token_embd.weight":
            d = d[:table_rows]
        data[t.name] = d
        w.add_tensor_info(t.name, d.shape, d.dtype, int(d.nbytes), raw_dtype=t.tensor_type)
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
