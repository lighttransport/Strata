#!/usr/bin/env python3
"""strata-ds41f-pack on a synthetic DeepSeek V4.1 Flash checkpoint: no model download, no GPU.

Writes a two-shard safetensors directory with the release layout (MXFP4 experts as I8 nibbles + E8M0 scales, an FP8 dense
projection with E8M0 block scales, BF16 embed/head/norms), then checks:
  1. `dequant` reproduces an independent NumPy MXFP4 decode bit for bit;
  2. `experts --uncalibrated` with K-quants packs, verifies, honours a REAP keep list, and its quality.json is sane;
  3. `experts --imatrix` with IQ1_S (needs an imatrix) packs and verifies when gguf-py is importable (llama.cpp checkout);
  4. `trunk` packs the non-expert tensors and skips the Engram table; verify passes; a corrupted payload fails.

    python3 tools/test_ds41f_pack.py --converter build-ds41f/strata-ds41f-pack
"""
import argparse
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile

import numpy as np

LAYERS, EXPERTS, HIDDEN, FF = 2, 3, 5120, 2304   # the tool's geometry is fixed per layer; we write 2 layers x 3 experts
FP4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)


def write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, (dtype, array) in tensors.items():
        raw = array.tobytes()
        header[name] = dict(dtype=dtype, shape=list(array.shape), data_offsets=[offset, offset + len(raw)])
        blobs.append(raw)
        offset += len(raw)
    text = json.dumps(header).encode()
    text += b" " * ((8 - len(text) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        for b in blobs:
            f.write(b)


def mxfp4_decode(packed, scales, rows, cols):
    codes = np.empty((rows, cols), np.uint8)
    codes[:, 0::2] = packed & 15
    codes[:, 1::2] = packed >> 4
    mag = FP4[codes & 7]
    sign = np.where(codes & 8, -1.0, 1.0).astype(np.float32)
    scale = np.exp2(scales.astype(np.float32) - 127).repeat(32, axis=1)
    return mag * sign * scale


def make_checkpoint(root, rng):
    experts, dense = {}, {}
    expected = {}
    for layer in range(LAYERS):
        for e in range(EXPERTS):
            for w, rows, cols in ((1, FF, HIDDEN), (3, FF, HIDDEN), (2, HIDDEN, FF)):
                packed = rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8)
                scales = rng.integers(120, 128, (rows, cols // 32), dtype=np.uint8)
                experts[f"layers.{layer}.ffn.experts.{e}.w{w}.weight"] = ("I8", packed)
                experts[f"layers.{layer}.ffn.experts.{e}.w{w}.scale"] = ("F8_E8M0", scales)
                expected[(layer, e, w)] = mxfp4_decode(packed, scales, rows, cols)
        # dense FP8 projection with E8M0 blocks of 32 rows x 128 columns, plus BF16 vectors
        rows, cols = 1280, HIDDEN
        dense[f"layers.{layer}.attn.wq_a.weight"] = ("F8_E4M3", rng.integers(0, 256, (rows, cols), dtype=np.uint8) & 0x7E)
        dense[f"layers.{layer}.attn.wq_a.scale"] = ("F8_E8M0", rng.integers(120, 128, (rows // 32, cols // 128), dtype=np.uint8))
        dense[f"layers.{layer}.attn_norm.weight"] = ("BF16", (rng.standard_normal(HIDDEN).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16))
    dense["embed.weight"] = ("BF16", (rng.standard_normal((512, HIDDEN)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16))
    dense["head.weight"] = ("BF16", (rng.standard_normal((512, HIDDEN)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16))
    dense["layers.1.engram.embed.weight"] = ("F8_E4M3", rng.integers(0, 256, (1024, 256), dtype=np.uint8) & 0x7E)
    dense["layers.1.engram.embed.scale"] = ("F8_E8M0", rng.integers(120, 128, (1024, 8), dtype=np.uint8))
    write_safetensors(root / "model-00001-of-00002.safetensors", experts)
    write_safetensors(root / "model-00002-of-00002.safetensors", dense)
    return expected


def run(converter, *args, expect_fail=None):
    result = subprocess.run([converter, *map(str, args)], capture_output=True, text=True, timeout=1800)
    if expect_fail is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0 and expect_fail in result.stderr, (result.returncode, result.stderr)
    return result


def gguf_py():
    for candidate in (os.environ.get("STRATA_GGUF_PY"), "the x4 NVMe work mount/work/llama.cpp/gguf-py"):
        if candidate and pathlib.Path(candidate, "gguf").is_dir():
            sys.path.insert(0, candidate)
            try:
                import gguf  # noqa: F401
                return __import__("gguf")
            except ImportError:
                continue
    return None


def write_imatrix(path, gg, layers, cols_by_proj, experts):
    writer = gg.GGUFWriter(str(path), "imatrix")
    writer.add_type("imatrix")
    for layer in range(layers):
        for proj, cols in cols_by_proj.items():
            name = f"blk.{layer}.ffn_{proj}_exps.weight"
            writer.add_tensor(name + ".in_sum2", np.full((experts, cols), 100.0, np.float32))
            writer.add_tensor(name + ".counts", np.full((experts,), 10.0, np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--converter", default="build-ds41f/strata-ds41f-pack")
    ap.add_argument("--keep-dir", help="keep the synthetic checkpoint and packs here")
    args = ap.parse_args()
    rng = np.random.default_rng(53)
    with tempfile.TemporaryDirectory(prefix="strata-ds41f-pack-") as tmp:
        root = pathlib.Path(args.keep_dir or tmp)
        root.mkdir(parents=True, exist_ok=True)
        ckpt = root / "ckpt"
        ckpt.mkdir(exist_ok=True)
        expected = make_checkpoint(ckpt, rng)
        # 1. exact dequantization
        run(args.converter, "dequant", ckpt, 1, 2, "down", root / "down.f32")
        got = np.fromfile(root / "down.f32", np.float32).reshape(HIDDEN, FF)
        assert np.array_equal(got, expected[(1, 2, 2)]), "MXFP4 dequant differs from the NumPy reference"
        run(args.converter, "dequant", ckpt, 0, 0, "gate", root / "gate.f32")
        assert np.array_equal(np.fromfile(root / "gate.f32", np.float32).reshape(FF, HIDDEN), expected[(0, 0, 1)])
        # 2. K-quant pack with a REAP keep list on layer 1
        (root / "keep.txt").write_text("# layer: kept source expert ids\n1: 0 2\n")
        pack = root / "experts-q2.gguf"
        run(args.converter, "experts", ckpt, pack, "--gate=q2_k", "--up=q2_k", "--down=q3_k", "--uncalibrated", "--layers=0-1", f"--keep={root / 'keep.txt'}", "--threads=3")
        run(args.converter, "verify", ckpt, pack)
        quality = json.loads((pack.with_suffix(".gguf.quality.json")).read_text())
        assert quality["architecture"] == "deepseek4" and quality["calibrated"] is False
        names = {p["name"]: p for p in quality["projections"]}
        assert names["blk.0.ffn_gate_exps.weight"]["experts"] == 3 and names["blk.1.ffn_gate_exps.weight"]["experts"] == 2
        for p in quality["projections"]:
            assert 0 < p["normalized_mse"] < 0.2, p
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from gguf_reader import GGUFFile
        info = GGUFFile(pack)
        tensors = {t.name: t for t in info.tensors}
        assert "blk.1.ffn_exps_keep" in tensors and "blk.0.ffn_exps_keep" not in tensors
        assert tensors["blk.0.ffn_down_exps.weight"].shape == [FF, HIDDEN, 3]
        # IQ1_S without an imatrix must be refused
        run(args.converter, "experts", ckpt, root / "never.gguf", "--gate=iq1_s", "--up=iq1_s", "--down=iq1_s", "--uncalibrated", expect_fail="requires --imatrix")
        # 3. IQ1_S / IQ2_XXS with a synthetic imatrix
        gg = gguf_py()
        if gg is not None:
            write_imatrix(root / "imatrix.gguf", gg, LAYERS, dict(gate=HIDDEN, up=HIDDEN, down=FF), EXPERTS)
            pack1 = root / "experts-iq1.gguf"
            run(args.converter, "experts", ckpt, pack1, "--gate=iq1_s", "--up=iq1_s", "--down=iq2_xxs", f"--imatrix={root / 'imatrix.gguf'}", "--layers=0", "--threads=3")
            run(args.converter, "verify", ckpt, pack1)
            q1 = json.loads(pack1.with_suffix(".gguf.quality.json").read_text())
            assert q1["calibrated"] is True and q1["formats"] == ["iq1_s", "iq1_s", "iq2_xxs"]
            iq1_report = "IQ1_S pack: " + ", ".join(f"{p['name'].split('.')[2]} nmse {p['normalized_mse']:.3f}" for p in q1["projections"])
        else:
            iq1_report = "IQ1_S pack skipped (gguf-py not found; set STRATA_GGUF_PY)"
        # 4. trunk
        trunk = root / "trunk.gguf"
        run(args.converter, "trunk", ckpt, trunk, "--trunk=q4_k", "--head=q8_0")
        run(args.converter, "verify", ckpt, trunk)
        tinfo = GGUFFile(trunk)
        tnames = {t.name: t for t in tinfo.tensors}
        assert "layers.0.attn.wq_a.weight" in tnames and "head.weight" in tnames and "embed.weight" in tnames
        assert not any("engram" in n for n in tnames) and not any("experts" in n for n in tnames)
        assert tnames["layers.0.attn.wq_a.weight"].type_name == "Q4_K" and tnames["head.weight"].type_name == "Q8_0" and tnames["embed.weight"].type_name == "BF16"
        # corruption is caught
        data = bytearray(pack.read_bytes())
        data[-1] ^= 0xFF
        bad = root / "bad.gguf"
        bad.write_bytes(data)
        run(args.converter, "verify", ckpt, bad, expect_fail="payload checksum mismatch")
        print("ds41f pack PASS: exact MXFP4 dequant, Q2_K/Q3_K pack + REAP keep + verify, trunk Q4_K/Q8_0 without Engram, corruption caught;", iq1_report)


if __name__ == "__main__":
    main()
