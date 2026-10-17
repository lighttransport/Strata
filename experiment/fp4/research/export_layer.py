#!/usr/bin/env python3
"""Export a bounded unquantized GGUF matrix slice and independently supplied inputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from gguf_reader import GGUFFile
from activation_trace import vector_key


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tensor", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cols", type=int, default=128)
    p.add_argument("--rows", type=int, default=16)
    p.add_argument("--col-start", type=int, default=0)
    p.add_argument("--row-start", type=int, default=0)
    p.add_argument("--inputs", type=Path, required=True, help="directory containing full-width float32 calibration-large, validation, heldout, activation files")
    p.add_argument("--input-source", choices=("captured", "synthetic"), required=True)
    a = p.parse_args()
    g = GGUFFile(a.model)
    matches = [t for t in g.tensors if t.name == a.tensor]
    if len(matches) != 1:
        p.error("tensor must uniquely exist")
    t = matches[0]
    if len(t.shape) != 2 or t.type_name not in ("F32", "F16", "BF16"):
        p.error("only unquantized 2D F32/F16/BF16 tensors are allowed")
    n, m = t.shape
    if (a.cols < 128 or a.cols > 16384 or a.cols % 128 or a.rows < 16 or
            a.rows > 4096 or a.rows % 16 or min(a.col_start, a.row_start) < 0 or
            a.col_start + a.cols > n or a.row_start + a.rows > m or a.rows*a.cols*4 > 512*1024**2):
        p.error("slice exceeds tensor or benchmark geometry")
    size = 4 if t.type_name == "F32" else 2
    data = []
    with a.model.open("rb") as f:
        for r in range(a.row_start, a.row_start + a.rows):
            f.seek(g.data_start + t.offset + (r*n + a.col_start)*size)
            raw = f.read(a.cols*size)
            if len(raw) != a.cols*size:
                raise ValueError("truncated tensor")
            if t.type_name == "BF16":
                data.extend(struct.unpack("<f", struct.pack("<I", v << 16))[0] for v in struct.unpack("<"+"H"*a.cols, raw))
            else:
                data.extend(struct.unpack("<" + ("f" if size == 4 else "e")*a.cols, raw))
    outputs = {"weights.f32": struct.pack("<"+"f"*len(data), *data)}
    source_hashes = {}
    counts = {}
    all_vectors = []
    cropped_vectors = []
    provenance_path = a.inputs / "activation-provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    if provenance:
        if a.input_source != provenance.get("input_source") or provenance.get("input_width") != n:
            raise ValueError("activation provenance source or width mismatch")
        if provenance.get("capture_kind") == "expert_input":
            expected = "blk."+str(provenance["layer"])+".ffn_gate_inp.weight"
            if t.name != expected or provenance.get("expert_filter") is not None:
                raise ValueError("expert-input traces currently support the matching full router tensor only")
    for name in ("calibration-large.f32", "validation.f32", "heldout.f32", "activation.f32"):
        path = a.inputs / name
        if path.stat().st_size > 512*1024**2:
            raise ValueError("input exceeds 512 MiB cap")
        raw = path.read_bytes()
        if not raw or len(raw) % (4*n) or (name == "activation.f32" and len(raw) != 4*n):
            raise ValueError("inputs must have the full tensor input width; activation is one vector")
        counts[name] = len(raw)//(4*n)
        vectors = [raw[i:i+4*n] for i in range(0, len(raw), 4*n)]
        # Validate full vectors, including channels omitted by the crop.
        fingerprints = set(map(vector_key, vectors))
        if name != "activation.f32":
            all_vectors.append(fingerprints)
        outputs[name] = b"".join(v[4*a.col_start:4*(a.col_start+a.cols)] for v in vectors)
        if name != "activation.f32":
            cropped_vectors.append({vector_key(v[4*a.col_start:4*(a.col_start+a.cols)]) for v in vectors})
        source_hashes[name] = hashlib.sha256(raw).hexdigest()
        if provenance and provenance.get("sha256", {}).get(name) != source_hashes[name]:
            raise ValueError("activation provenance hash mismatch")
    if any(all_vectors[i] & all_vectors[j] for i in range(3) for j in range(i)):
        raise ValueError("training/validation/heldout inputs overlap")
    if any(cropped_vectors[i] & cropped_vectors[j] for i in range(3) for j in range(i)):
        raise ValueError("training/validation/heldout inputs overlap after cropping")
    for raw in outputs.values():
        if any(not math.isfinite(v[0]) for v in struct.iter_unpack("<f", raw)):
            raise ValueError("nonfinite source")
    a.output.mkdir(parents=True, exist_ok=True)
    for name, raw in outputs.items():
        (a.output/name).write_bytes(raw)
    (a.output/"fixture.json").write_text(json.dumps({
        "scope": "real unquantized model slice; projection diagnostic, not whole-model quality",
        "tensor": t.name, "source_type": t.type_name, "source_shape": t.shape,
        "input_source": a.input_source, "input": a.cols, "output": a.rows,
        "row_start": a.row_start, "col_start": a.col_start, "counts": counts,
        "source_input_sha256": source_hashes,
        "activation_provenance": provenance,
        "sha256": {k: hashlib.sha256(v).hexdigest() for k, v in outputs.items()},
    }, indent=2)+"\n")
    print("Exported unquantized model slice; activation provenance: " + a.input_source)


if __name__ == "__main__":
    main()
