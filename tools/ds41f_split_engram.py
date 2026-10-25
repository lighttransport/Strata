#!/usr/bin/env python3
"""Split a DeepSeek-V4.1-Flash GGUF pack into a backbone and per-table Engram files, so the experts and the
Engram tables download separately and packs with the same tables share one copy.

    ds41f_split_engram.py split PACK_FIRST_SHARD OUT_ROOT [--name NAME] [--dry-run]

PACK_FIRST_SHARD: `...-00001-of-0000N.gguf` of a split pack whose Engram tables (`blk.L.engram_embd.weight`) each
sit in a shard of their own (the llama.cpp #28696 packs). Writes
    OUT_ROOT/engram/DeepSeek-V4.1-Flash-Engram-L<L>-<TYPE>.gguf   one table each (architecture deepseek41-engram);
                                                                  an existing file of the same name must hold the
                                                                  same bytes (sampled and size-checked) and is reused
    OUT_ROOT/<NAME>/DeepSeek-V4.1-Flash-<NAME>-0000k-of-0000M.gguf the other shards, renumbered, unchanged tensors;
                                                                  the first carries `deepseek41.engram.file` =
                                                                  "../engram/<table file>,..." (Strata's ModelArtifact)
Metadata is copied byte for byte except split.count / split.no / split.tensors.count and the new key; tensor
payloads are copied unchanged (offsets keep the file's alignment). `verify` compares every tensor payload's sha256
against the source.

    ds41f_split_engram.py verify SRC_FIRST_SHARD OUT_FIRST_SHARD   every tensor's payload sha256, source vs split

Converting an already downloaded old shard instead of downloading its split counterpart (the tensor bytes are the
same; only the header and the offsets differ):
    ds41f_split_engram.py header NEW_SHARD > NEW.header             (on the cluster: the split shard's header only)
    ds41f_split_engram.py rewrite OLD_SHARD NEW.header OUT_SHARD     (locally: that header + the old shard's payloads)
Old shard k -> new shard k for k <= 6, old 9 -> new 7; old 7 and 8 are the Engram tables (engram/ files).
rewrite checks that every tensor's name, type and shape agree and writes OUT_SHARD (needs its size free; delete the
old shard afterwards).
"""
import argparse
import hashlib
import os
import pathlib
import re
import struct
import sys

ALIGN_KEY = "general.alignment"
CHUNK = 64 << 20
GGML_BLOCK = {0: (1, 4), 1: (1, 2), 30: (1, 2), 8: (32, 34), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
              10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 16: (256, 66),
              17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
              29: (256, 56), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 39: (32, 17), 42: (64, 18)}
TYPE_NAME = {0: "F32", 1: "F16", 30: "BF16", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
             16: "IQ2_XXS", 17: "IQ2_XS", 19: "IQ1_S", 20: "IQ4_NL", 22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M"}
# GGUF value types: size of the scalar ones
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class Header:
    """A GGUF v3 header kept as raw key/value byte spans, plus the tensor infos."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        with open(self.path, "rb") as f:
            magic, self.version = struct.unpack("<II", f.read(8))
            if magic != 0x46554747 or self.version != 3:
                raise SystemExit(f"{path}: not GGUF v3")
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            self.kv = []   # (key, raw bytes of type+value)
            self.values = {}
            for _ in range(n_kv):
                key = self._str(f)
                start = f.tell()
                vtype = struct.unpack("<I", f.read(4))[0]
                val = self._skip(f, vtype)
                end = f.tell()
                f.seek(start)
                self.kv.append((key, f.read(end - start)))
                self.values[key] = val
            self.tensors = []   # name, dims, type, offset
            for _ in range(n_tensors):
                name = self._str(f)
                nd = struct.unpack("<I", f.read(4))[0]
                dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
                ttype, off = struct.unpack("<IQ", f.read(12))
                self.tensors.append((name, dims, ttype, off))
            self.align = int(self.values.get(ALIGN_KEY, 32))
            pos = f.tell()
            self.data_start = (pos + self.align - 1) // self.align * self.align
        self.size = self.path.stat().st_size

    @staticmethod
    def _str(f):
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8")

    def _skip(self, f, vtype):
        if vtype in SCALAR:
            raw = f.read(SCALAR[vtype])
            fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[vtype]
            return struct.unpack(fmt, raw)[0]
        if vtype == 8:
            return self._str(f)
        if vtype == 9:
            etype, n = struct.unpack("<IQ", f.read(12))
            if etype in SCALAR:
                f.seek(SCALAR[etype] * n, os.SEEK_CUR)
                return None
            for _ in range(n):
                self._skip(f, etype)
            return None
        raise SystemExit(f"{self.path}: unknown GGUF value type {vtype}")

    def nbytes(self, t):
        name, dims, ttype, _ = t
        blk, nb = GGML_BLOCK[ttype]
        n = 1
        for d in dims:
            n *= d
        return n // blk * nb


def kv_raw(vtype, value):
    if vtype == 4:
        return struct.pack("<II", 4, value)
    if vtype == 8:
        b = value.encode("utf-8")
        return struct.pack("<IQ", 8, len(b)) + b
    raise ValueError(vtype)


def write_gguf(path, kvs, tensors, src_of, align, hash_out=None):
    """kvs: [(key, raw type+value)]; tensors: [(name, dims, type, nbytes)]; src_of(i) -> (src path, absolute offset)."""
    head = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(kvs)))
    for key, raw in kvs:
        kb = key.encode("utf-8")
        head += struct.pack("<Q", len(kb)) + kb + raw
    offs = []
    off = 0
    for name, dims, ttype, nb in tensors:
        offs.append(off)
        off = (off + nb + align - 1) // align * align
    for (name, dims, ttype, nb), o in zip(tensors, offs):
        nm = name.encode("utf-8")
        head += struct.pack("<Q", len(nm)) + nm + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        head += struct.pack("<IQ", ttype, o)
    pad = (-len(head)) % align
    head += b"\0" * pad
    tmp = pathlib.Path(str(path) + ".part")
    with open(tmp, "wb") as out:
        out.write(head)
        base = len(head)
        for i, ((name, dims, ttype, nb), o) in enumerate(zip(tensors, offs)):
            assert out.tell() == base + o
            sp, so = src_of(i)
            h = hashlib.sha256() if hash_out is not None else None
            with open(sp, "rb") as src:
                src.seek(so)
                left = nb
                while left:
                    buf = src.read(min(CHUNK, left))
                    if not buf:
                        raise SystemExit(f"{sp}: short read")
                    out.write(buf)
                    if h: h.update(buf)
                    left -= len(buf)
            if h: hash_out[name] = h.hexdigest()
            if i + 1 < len(tensors):
                out.write(b"\0" * (offs[i + 1] - o - nb))
    os.replace(tmp, path)


def shards_of(first):
    first = pathlib.Path(first)
    m = re.match(r"^(.*)-00001-of-(\d{5})\.gguf$", first.name)
    if not m:
        raise SystemExit(f"{first}: not a first shard (...-00001-of-0000N.gguf)")
    n = int(m.group(2))
    return [first.parent / f"{m.group(1)}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)], m.group(1)


def same_bytes(a_path, a_off, b_path, b_off, nb, samples=16, block=1 << 20):
    """Sampled byte comparison (first, last and evenly spaced 1 MiB blocks)."""
    pts = sorted({0, max(0, nb - block)} | {nb * k // samples for k in range(1, samples)})
    with open(a_path, "rb") as fa, open(b_path, "rb") as fb:
        for p in pts:
            n = min(block, nb - p)
            fa.seek(a_off + p)
            fb.seek(b_off + p)
            if fa.read(n) != fb.read(n):
                return False
    return True


def cmd_split(a):
    paths, stem = shards_of(a.first)
    name = a.name or stem.replace("DeepSeek-V4.1-Flash-", "")
    heads = [Header(p) for p in paths]
    arch = heads[0].values.get("general.architecture")
    if arch != "deepseek41":
        raise SystemExit(f"architecture {arch}: only deepseek41")
    root = pathlib.Path(a.out_root)
    eng_dir = root / "engram"
    pack_dir = root / name
    engram_files = []
    backbone = []
    for i, h in enumerate(heads):
        names = [t[0] for t in h.tensors]
        tabs = [t for t in h.tensors if t[0].endswith(".engram_embd.weight")]
        if tabs and len(tabs) != len(names):
            raise SystemExit(f"{h.path.name}: an Engram table shares its shard with other tensors (not supported)")
        if tabs:
            t = tabs[0]
            layer = int(re.match(r"blk\.(\d+)\.", t[0]).group(1))
            fname = f"DeepSeek-V4.1-Flash-Engram-L{layer:02d}-{TYPE_NAME.get(t[2], str(t[2]))}.gguf"
            engram_files.append((fname, h, t))
        else:
            backbone.append(h)
    if not engram_files:
        raise SystemExit("no Engram tables in their own shards")
    total = sum(len(h.tensors) for h in heads)
    nback = sum(len(h.tensors) for h in backbone)
    print(f"{name}: {len(backbone)} backbone shards ({nback} tensors, {sum(h.size for h in backbone)/1e9:.1f} GB), "
          f"Engram {[f for f, _, _ in engram_files]}")
    if a.dry_run:
        return 0
    eng_dir.mkdir(parents=True, exist_ok=True)
    pack_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for fname, h, t in engram_files:
        dst = eng_dir / fname
        nb = h.nbytes(t)
        if dst.exists():
            e = Header(dst)
            et = e.tensors[0] if len(e.tensors) == 1 else None
            if not et or et[0] != t[0] or et[2] != t[2] or tuple(et[1]) != tuple(t[1]) or not \
                    same_bytes(dst, e.data_start + et[3], h.path, h.data_start + t[3], nb):
                raise SystemExit(f"{dst} exists with different contents than {h.path.name}'s {t[0]}")
            print(f"  reuse {fname} (same name, shape, type; sampled bytes equal)")
            continue
        kvs = [("general.architecture", kv_raw(8, "deepseek41-engram")),
               ("general.description", kv_raw(8, "DeepSeek-V4.1-Flash Engram table " + t[0] + " (tools/ds41f_split_engram.py)")),
               (ALIGN_KEY, kv_raw(4, h.align))]
        write_gguf(dst, kvs, [(t[0], t[1], t[2], nb)], lambda i: (h.path, h.data_start + t[3]), h.align, hashes)
        print(f"  wrote {fname} ({nb/1e9:.1f} GB)")
    m = len(backbone)
    ref = ",".join("../engram/" + f for f, _, _ in engram_files)
    for k, h in enumerate(backbone):
        # the split keys keep their original value types (uint16 / int32 in llama.cpp's splits)
        kvs = []
        for key, raw in h.kv:
            if key in ("split.count", "split.no", "split.tensors.count"):
                vt = struct.unpack("<I", raw[:4])[0]
                val = {"split.count": m, "split.no": k, "split.tensors.count": nback}[key]
                fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 10: "<Q", 11: "<q"}[vt]
                raw = struct.pack("<I", vt) + struct.pack(fmt, val)
            kvs.append((key, raw))
        if k == 0:
            kvs.append(("deepseek41.engram.file", kv_raw(8, ref)))
        tens = [(t[0], t[1], t[2], h.nbytes(t)) for t in h.tensors]
        dst = pack_dir / f"DeepSeek-V4.1-Flash-{name}-{k + 1:05d}-of-{m:05d}.gguf"
        write_gguf(dst, kvs, tens, lambda i, h=h: (h.path, h.data_start + h.tensors[i][3]), h.align, hashes)
        print(f"  wrote {dst.name} ({sum(t[3] for t in tens)/1e9:.1f} GB)")
    if total != nback + len(engram_files):
        raise SystemExit("tensor count mismatch")
    return 0


def payload_hashes(first):
    paths, _ = shards_of(first)
    out = {}
    hs = [Header(p) for p in paths]
    ef = hs[0].values.get("deepseek41.engram.file")
    if ef:
        hs += [Header(paths[0].parent / f) for f in ef.split(",") if f]
    for h in hs:
        for t in h.tensors:
            nb = h.nbytes(t)
            dig = hashlib.sha256()
            with open(h.path, "rb") as f:
                f.seek(h.data_start + t[3])
                left = nb
                while left:
                    buf = f.read(min(CHUNK, left))
                    dig.update(buf)
                    left -= len(buf)
            out[t[0]] = (t[2], tuple(t[1]), dig.hexdigest())
    return out


def cmd_verify(a):
    src = payload_hashes(a.src)
    dst = payload_hashes(a.dst)
    bad = [n for n in src if src[n] != dst.get(n)]
    extra = [n for n in dst if n not in src]
    print(f"{len(src)} source tensors, {len(dst)} split tensors, {len(bad)} differ, {len(extra)} extra")
    for n in bad[:10] + extra[:10]:
        print("  ", n)
    return 1 if bad or extra else 0


def cmd_header(a):
    h = Header(a.shard)
    with open(h.path, "rb") as f:
        sys.stdout.buffer.write(f.read(h.data_start))
    return 0


def cmd_rewrite(a):
    old = Header(a.old)
    new = Header(a.header)   # parses the header bytes only
    if [(t[0], tuple(t[1]), t[2]) for t in old.tensors] != [(t[0], tuple(t[1]), t[2]) for t in new.tensors]:
        raise SystemExit("the header's tensors (names, shapes, types, order) differ from the old shard's")
    out = pathlib.Path(a.out)
    tmp = pathlib.Path(str(out) + ".part")
    with open(a.header, "rb") as hf:
        head = hf.read(new.data_start)
    with open(tmp, "wb") as dst, open(old.path, "rb") as src:
        dst.write(head)
        for i, t in enumerate(new.tensors):
            nb = new.nbytes(t)
            dst.seek(new.data_start + t[3])
            src.seek(old.data_start + old.tensors[i][3])
            left = nb
            while left:
                buf = src.read(min(CHUNK, left))
                if not buf:
                    raise SystemExit(f"{old.path}: short read")
                dst.write(buf)
                left -= len(buf)
        dst.truncate(new.data_start + max(t[3] + new.nbytes(t) for t in new.tensors))
    os.replace(tmp, out)
    print(f"wrote {out} ({out.stat().st_size} bytes) from {old.path.name}'s payloads")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("split")
    p.add_argument("first"); p.add_argument("out_root"); p.add_argument("--name"); p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_split)
    p = sub.add_parser("verify"); p.add_argument("src"); p.add_argument("dst"); p.set_defaults(func=cmd_verify)
    p = sub.add_parser("header"); p.add_argument("shard"); p.set_defaults(func=cmd_header)
    p = sub.add_parser("rewrite"); p.add_argument("old"); p.add_argument("header"); p.add_argument("out"); p.set_defaults(func=cmd_rewrite)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
