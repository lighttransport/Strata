#!/usr/bin/env python3
"""strata-ds41f-pack on a synthetic DeepSeek V4.1 Flash checkpoint: no model download, no GPU.

Writes a two-shard safetensors directory with the release layout (MXFP4 experts as I8 nibbles + E8M0 scales, an FP8 dense
projection with E8M0 block scales, BF16 embed/head/norms), then checks:
  1. `dequant` reproduces an independent NumPy MXFP4 decode bit for bit;
  2. `experts --uncalibrated` with K-quants packs, verifies, honours a REAP keep list, and its quality.json is sane;
  3. `experts --imatrix` with IQ1_S (needs an imatrix) packs and verifies when gguf-py is importable (llama.cpp checkout);
  4. `trunk` packs the non-expert tensors and skips the Engram table; verify passes; a corrupted payload fails;
  5. several processes filling one chunk directory (racing on plan.json, scoped stale-partial cleanup), `--plan-only`,
     imatrix counts as [experts] or {1, experts}, a seeded deadline, and a threaded trunk byte-identical to one thread;
  6. with strata-pfetch (tools/pfetch/build, or PFETCH=...) and mpirun: a sparse node-local mirror built from `ranges`
     by tools/ds41f_pfetch_plan.py packs the same chunks and trunk as the shared checkpoint, a missing range fails
     closed (--coverage), and layers already complete are not fetched.

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


def make_checkpoint(root, rng, layers=LAYERS, n_experts=EXPERTS, shard_per_layer=False):
    experts, dense = {}, {}
    expected = {}
    for layer in range(layers):
        for e in range(n_experts):
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
    if shard_per_layer:   # the release layout: one shard holds one layer's experts
        n = layers + 1
        for layer in range(layers):
            write_safetensors(root / f"model-{layer + 1:05d}-of-{n:05d}.safetensors",
                              {k: v for k, v in experts.items() if k.startswith(f"layers.{layer}.")})
        write_safetensors(root / f"model-{n:05d}-of-{n:05d}.safetensors", dense)
        return expected
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
    repo = pathlib.Path(__file__).resolve().parents[1]
    for candidate in (os.environ.get("STRATA_GGUF_PY"), str(repo.parent / "llama.cpp" / "gguf-py")):
        if candidate and pathlib.Path(candidate, "gguf").is_dir():
            sys.path.insert(0, candidate)
            try:
                import gguf  # noqa: F401
                return __import__("gguf")
            except ImportError:
                continue
    return None


def write_imatrix(path, gg, layers, cols_by_proj, experts, counts_shape=None):
    """counts_shape: None -> [experts] (1-D); (experts, 1) -> ggml {1, experts} as llama-imatrix writes it."""
    writer = gg.GGUFWriter(str(path), "imatrix")
    writer.add_type("imatrix")
    for layer in range(layers):
        for proj, cols in cols_by_proj.items():
            name = f"blk.{layer}.ffn_{proj}_exps.weight"
            writer.add_tensor(name + ".in_sum2", np.full((experts, cols), 100.0, np.float32))
            writer.add_tensor(name + ".counts", np.full(counts_shape or (experts,), 10.0, np.float32))
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
        # 3b. streamable chunk pipeline: stop after one layer (exit 3), resume, assemble == single-file output
        cdir = root / "chunks"
        r = subprocess.run([args.converter, "experts", str(ckpt), f"--chunk-dir={cdir}", "--gate=q2_k", "--up=q2_k", "--down=q3_k", "--uncalibrated",
                            "--layers=0-1", f"--keep={root / 'keep.txt'}", "--threads=3", "--stop-after-layers=1"], capture_output=True, text=True)
        assert r.returncode == 3, (r.returncode, r.stderr)
        assert (cdir / "plan.json").exists() and (cdir / "blk.0.ffn_gate_exps.weight.chunk.json").exists() and not (cdir / "blk.1.ffn_gate_exps.weight.chunk.json").exists()
        r = subprocess.run([args.converter, "assemble", str(cdir), "--check"], capture_output=True, text=True)
        assert r.returncode == 3 and "missing" in r.stdout, r.stdout
        stamp = (cdir / "blk.0.ffn_down_exps.weight.chunk").stat().st_mtime_ns
        (cdir / "blk.1.ffn_up_exps.weight.chunk.partial").write_bytes(b"stale")      # an interrupted layer's leftover
        run(args.converter, "experts", ckpt, f"--chunk-dir={cdir}", "--gate=q2_k", "--up=q2_k", "--down=q3_k", "--uncalibrated", "--layers=0-1", f"--keep={root / 'keep.txt'}", "--threads=3")
        assert (cdir / "blk.0.ffn_down_exps.weight.chunk").stat().st_mtime_ns == stamp, "resume rewrote a finished layer"
        assert not (cdir / "blk.1.ffn_up_exps.weight.chunk.partial").exists()
        run(args.converter, "experts", ckpt, f"--chunk-dir={cdir}", "--gate=q2_k", "--up=q2_k", "--down=q2_k", "--uncalibrated", "--layers=0-1", expect_fail="plan mismatch")
        assembled = root / "assembled.gguf"
        # the default range is the whole model: a 2-layer directory needs an explicit --layers
        run(args.converter, "assemble", cdir, assembled, expect_fail="layers missing")
        r = subprocess.run([args.converter, "assemble", str(cdir), "--check"], capture_output=True, text=True)
        assert r.returncode == 3 and "range=0-39" in r.stdout, r.stdout
        run(args.converter, "assemble", cdir, assembled, "--layers=0-1")
        run(args.converter, "verify", assembled)                 # hashes only, no source
        run(args.converter, "verify", ckpt, assembled)           # with the source identity
        a, b = GGUFFile(assembled), GGUFFile(pack)
        assert [(t.name, t.type_name, t.shape) for t in a.tensors] == [(t.name, t.type_name, t.shape) for t in b.tensors]
        assert assembled.read_bytes()[a.data_start:] == pack.read_bytes()[b.data_start:], "assembled payload differs from the single-file pack"
        # the identity ignores mtimes and the directory: a copied checkpoint still verifies
        import shutil
        copy = root / "ckpt-copy"
        shutil.copytree(ckpt, copy)
        for f in copy.iterdir():
            os.utime(f, (1_600_000_000, 1_600_000_000))
        run(args.converter, "verify", copy, assembled)
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
        multi_report = multiprocess_checks(args.converter, root, gg)
        mirror_report = mirror_checks(args.converter, root)
        print("ds41f pack PASS: exact MXFP4 dequant, Q2_K/Q3_K pack + REAP keep + verify, chunk pipeline stop/resume/plan-mismatch/assemble, mtime-independent identity, trunk Q4_K/Q8_0 without Engram, corruption caught;", iq1_report + ";", multi_report + ";", mirror_report)


def multiprocess_checks(converter, root, gg):
    """One chunk directory filled by several processes, --plan-only, imatrix count shapes, deadline seed, trunk threads."""
    import time
    rng = np.random.default_rng(7)
    ck4 = root / "ckpt4"
    ck4.mkdir(exist_ok=True)
    make_checkpoint(ck4, rng, layers=4, n_experts=2)
    kq = ["--gate=q2_k", "--up=q2_k", "--down=q3_k", "--uncalibrated"]
    single = root / "single4.gguf"
    run(converter, "experts", ck4, single, *kq, "--layers=0-3", "--threads=2")
    # 4 processes, one layer each, started together on a fresh directory: they race on plan.json
    for rep_i in range(3):
        d = root / f"shared{rep_i}"
        procs = [subprocess.Popen([converter, "experts", str(ck4), f"--chunk-dir={d}", *kq, f"--layers={l}-{l}", "--threads=2"],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for l in range(4)]
        outs = [pr.communicate(timeout=1800) for pr in procs]
        for pr, (_, err) in zip(procs, outs):
            assert pr.returncode == 0, err
        json.loads((d / "plan.json").read_text())
        assert not list(d.glob("plan.json.*partial")), list(d.glob("plan.json.*"))
        r = subprocess.run([converter, "assemble", str(d), "--check", "--layers=0-3"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout
    out = root / "shared0.gguf"
    run(converter, "assemble", root / "shared0", out, "--layers=0-3")
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from gguf_reader import GGUFFile
    a, b = GGUFFile(out), GGUFFile(single)
    assert out.read_bytes()[a.data_start:] == single.read_bytes()[b.data_start:], "multi-process pack differs from the single-process pack"
    # --plan-only writes the plan and nothing else; other formats are refused
    d = root / "planonly"
    run(converter, "experts", ck4, f"--chunk-dir={d}", *kq, "--layers=0-3", "--plan-only")
    assert (d / "plan.json").exists() and not list(d.glob("*.chunk*"))
    run(converter, "experts", ck4, f"--chunk-dir={d}", "--gate=q2_k", "--up=q2_k", "--down=q2_k", "--uncalibrated", "--plan-only", expect_fail="plan mismatch")
    # stale-partial cleanup is scoped to the invocation's own layers
    (d / "blk.2.ffn_gate_exps.weight.chunk.partial").write_bytes(b"in flight elsewhere")
    (d / "blk.0.ffn_up_exps.weight.chunk.partial").write_bytes(b"stale")
    run(converter, "experts", ck4, f"--chunk-dir={d}", *kq, "--layers=0-0")      # default --threads
    assert (d / "blk.2.ffn_gate_exps.weight.chunk.partial").exists(), "removed another process's in-flight partial"
    assert not (d / "blk.0.ffn_up_exps.weight.chunk.partial").exists()
    # a seeded per-layer estimate makes the deadline apply to the first layer of an invocation
    r = subprocess.run([converter, "experts", str(ck4), f"--chunk-dir={d}", *kq, "--layers=1-1", f"--deadline={int(time.time()) + 10}", "--layer-seconds=1000"],
                       capture_output=True, text=True)
    assert r.returncode == 3 and not (d / "blk.1.ffn_gate_exps.weight.chunk.json").exists(), (r.returncode, r.stderr)
    # trunk: threads change nothing in the file
    t1, t3 = root / "trunk-t1.gguf", root / "trunk-t3.gguf"
    run(converter, "trunk", ck4, t1, "--threads=1")
    run(converter, "trunk", ck4, t3, "--threads=3")
    assert t1.read_bytes() == t3.read_bytes(), "threaded trunk differs from the single-threaded trunk"
    report = "multi-process chunk dir x3 == single pack, --plan-only, scoped cleanup, deadline seed, trunk threads identical"
    if gg is not None:
        # imatrix counts as [experts] and as {1, experts} give the same IQ1_S chunk; other shapes are refused
        cols = dict(gate=HIDDEN, up=HIDDEN, down=FF)
        write_imatrix(root / "im1d.gguf", gg, 4, cols, 2)
        write_imatrix(root / "im2d.gguf", gg, 4, cols, 2, counts_shape=(2, 1))
        write_imatrix(root / "imbad.gguf", gg, 4, cols, 2, counts_shape=(2, 2))
        write_imatrix(root / "im3.gguf", gg, 3, cols, 2)                       # no entries for layer 3
        iq = ["--gate=iq1_s", "--up=iq1_s", "--down=iq1_s"]
        for name in ("im1d", "im2d"):
            run(converter, "experts", ck4, f"--chunk-dir={root / ('iq-' + name)}", *iq, f"--imatrix={root / (name + '.gguf')}", "--layers=0-0", "--threads=2")
        for proj in ("gate", "up", "down"):
            f = f"blk.0.ffn_{proj}_exps.weight.chunk"
            assert (root / "iq-im1d" / f).read_bytes() == (root / "iq-im2d" / f).read_bytes(), f
        run(converter, "experts", ck4, f"--chunk-dir={root / 'iq-bad'}", *iq, f"--imatrix={root / 'imbad.gguf'}", "--layers=0-0", "--threads=2", expect_fail="geometry mismatch")
        run(converter, "experts", ck4, f"--chunk-dir={root / 'iq-l3'}", *iq, f"--imatrix={root / 'im3.gguf'}", "--layers=0-3", "--plan-only", expect_fail="imatrix has no entry")
        report += ", imatrix counts 1-D == 2-D, bad shape and missing layer refused"
    return report


def mirror_checks(converter, root):
    """Pack from a strata-pfetch sparse mirror (2 ranks): same chunks and trunk as from the shared checkpoint."""
    import shlex
    repo = pathlib.Path(__file__).resolve().parents[1]
    pfetch = os.environ.get("PFETCH", str(repo / "tools" / "pfetch" / "build" / "strata-pfetch"))
    mpirun = shlex.split(os.environ.get("MPIRUN", "mpirun --oversubscribe"))
    if not os.access(pfetch, os.X_OK):
        return "mirror checks SKIPPED (no strata-pfetch; make -C tools/pfetch)"
    ck = root / "ckpt-split"
    ck.mkdir(exist_ok=True)
    make_checkpoint(ck, np.random.default_rng(11), layers=4, n_experts=2, shard_per_layer=True)
    local = root / "local"
    plan = [sys.executable, str(repo / "tools" / "ds41f_pfetch_plan.py"), "--bin", converter, "--ckpt", str(ck), "--ranks", "2",
            "--rank-layers", "0:0-1,1:2-3", "--trunk-rank", "1", "--mirror", "@L/n{rank}/ckpt", "--block-mib", "1"]
    man = root / "mirror.manifest"
    man.write_text(subprocess.run(plan, capture_output=True, text=True, check=True).stdout)
    r = subprocess.run(mpirun + ["-np", "2", pfetch, f"--manifest={man}", f"--local-root={local}", "--streams=2",
                                 "--coverage-out=@L/n{rank}/coverage"], capture_output=True, text=True)
    assert r.returncode == 0 and "PFETCH_END state=ok" in r.stdout, r.stdout[-3000:] + r.stderr[-2000:]
    mirror = [local / f"n{k}" / "ckpt" for k in range(2)]
    assert (mirror[0] / "model-00005-of-00005.safetensors").is_symlink()          # rank 0 does not read the trunk shard
    st = os.stat(mirror[1] / "model-00005-of-00005.safetensors")                   # rank 1 mirrors it sparsely (no Engram)
    assert st.st_blocks * 512 < st.st_size, (st.st_blocks, st.st_size)
    kq = ["--gate=q2_k", "--up=q2_k", "--down=q3_k", "--uncalibrated", "--threads=2"]
    for k, layers in ((0, "0-1"), (1, "2-3")):
        run(converter, "experts", ck, f"--chunk-dir={root / 'mirror-ref'}", *kq, f"--layers={layers}")
        run(converter, "experts", mirror[k], f"--chunk-dir={root / 'mirror-pack'}", *kq, f"--layers={layers}",
            f"--coverage={local / f'n{k}' / 'coverage'}")
    names = sorted(p.name for p in (root / "mirror-ref").iterdir())
    assert names == sorted(p.name for p in (root / "mirror-pack").iterdir()) and len(names) == 4 * 6 + 1, names
    for name in names:
        assert (root / "mirror-ref" / name).read_bytes() == (root / "mirror-pack" / name).read_bytes(), name
    run(converter, "trunk", ck, root / "trunk-ref.gguf")
    run(converter, "trunk", mirror[1], root / "trunk-mirror.gguf", f"--coverage={local / 'n1' / 'coverage'}")
    assert (root / "trunk-ref.gguf").read_bytes() == (root / "trunk-mirror.gguf").read_bytes()
    # fail closed: the last 4 KiB of a fetched range missing, or a mirrored shard missing from the coverage
    lines = (local / "n0" / "coverage").read_text().splitlines()
    path, off, n = lines[-1].split()
    cut = root / "coverage-cut"
    cut.write_text("\n".join(lines[:-1] + [f"{path} {off} {int(n) - 4096}"]) + "\n")
    run(converter, "experts", mirror[0], f"--chunk-dir={root / 'mirror-cut'}", *kq, "--layers=0-1", f"--coverage={cut}",
        expect_fail="read outside fetched coverage")
    cut.write_text("\n".join(lines[:-1]) + "\n")
    run(converter, "experts", mirror[0], f"--chunk-dir={root / 'mirror-cut2'}", *kq, "--layers=0-1", f"--coverage={cut}",
        expect_fail="without fetched coverage")
    # complete layers are not fetched again: only symlinks remain for rank 0
    again = subprocess.run(plan + ["--chunk-dirs", str(root / "mirror-pack")], capture_output=True, text=True, check=True)
    assert not [l for l in again.stdout.splitlines() if l.startswith("D ") and l.split()[2] == "0"], again.stdout[:2000]
    assert "0:0L/0.0GB" in again.stderr, again.stderr
    return "pfetch mirror pack == shared pack (chunks, plan, trunk), missing range fails closed, complete layers not fetched"


if __name__ == "__main__":
    main()
