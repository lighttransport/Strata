"""Validate diagnostic fixture contents before replaying or studying them."""
import hashlib
import json
from pathlib import Path
from activation_trace import vector_key


def verify_fixture(folder):
    folder = Path(folder)
    meta = json.loads((folder / "fixture.json").read_text())
    cols, rows = meta["input"], meta["output"]
    if (type(cols) is not int or type(rows) is not int or cols < 128 or cols > 16384 or
            cols % 128 or rows < 16 or rows > 4096 or rows % 16):
        raise ValueError("invalid fixture geometry")
    keys = []
    for name in ("weights.f32", "calibration-large.f32", "validation.f32", "heldout.f32", "activation.f32"):
        path = folder / name
        size = path.stat().st_size
        if not size or size > 512*1024**2 or size % (4*cols):
            raise ValueError("invalid fixture file size: " + name)
        if name == "weights.f32" and size != 4*cols*rows:
            raise ValueError("weight fixture geometry mismatch")
        if name == "activation.f32" and size != 4*cols:
            raise ValueError("runtime activation must be one vector")
        if name in ("calibration-large.f32", "validation.f32", "heldout.f32") and size < 8*cols:
            raise ValueError("each evaluation split requires two or more vectors")
        digest = hashlib.sha256()
        fingerprints = set()
        with path.open("rb") as file:
            while raw := file.read(4*cols):
                digest.update(raw)
                fingerprints.add(vector_key(raw))
        if meta.get("sha256", {}).get(name) != digest.hexdigest():
            raise ValueError("fixture hash mismatch: " + name)
        if name in ("calibration-large.f32", "validation.f32", "heldout.f32"):
            if any(fingerprints & previous for previous in keys):
                raise ValueError("fixture input splits overlap")
            keys.append(fingerprints)
    return meta
