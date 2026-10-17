#!/usr/bin/env python3
"""CPU HIGGS vector-grid diagnostic using an explicitly supplied upstream source."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

from gyro_study import rotate, nmse
from fixture_validation import verify_fixture


def grids(path):
    # Extract numerical literals only; never import or execute downloaded code.
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_higgs_grid")
    result = {}
    for branch in ast.walk(fn):
        if not isinstance(branch, ast.If) or not isinstance(branch.test, ast.Compare):
            continue
        try:
            key = tuple(ast.literal_eval(branch.test.comparators[0]))
            call = branch.body[0].value
            grid = ast.literal_eval(call.args[0])
        except (ValueError, TypeError, AttributeError, IndexError):
            continue
        if len(grid) != key[1] or any(len(row) != key[0] for row in grid):
            raise ValueError("invalid upstream grid geometry")
        if any(not math.isfinite(x) for row in grid for x in row):
            raise ValueError("nonfinite grid")
        result[key] = grid
    return result


def load(path, cols):
    raw = path.read_bytes()
    if not raw or len(raw) % (cols*4) or len(raw) > 512*1024**2:
        raise ValueError("invalid float32 matrix size")
    vals = list(struct.unpack("<"+"f"*(len(raw)//4), raw))
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("nonfinite matrix")
    return [vals[i:i+cols] for i in range(0, len(vals), cols)]


def encode(row, width, grid):
    transformed = rotate(row, width)
    restored = []
    dimension = len(grid[0])
    for base in range(0, len(row), width):
        scale = math.sqrt(sum(x*x for x in row[base:base+width])/width)
        # Upstream scales are stored in FP16 for its usual FP16 weight path.
        stored_scale = struct.unpack("<e", struct.pack("<e", scale))[0]
        for offset in range(base, base+width, dimension):
            vector = [x/scale if scale else 0 for x in transformed[offset:offset+dimension]]
            chosen = min(grid, key=lambda point: sum((x-y)**2 for x,y in zip(vector, point)))
            restored.extend(struct.unpack("<e", struct.pack("<e", x))[0]*stored_scale if scale else 0 for x in chosen)
    return restored


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True, help="official transformers integrations/higgs.py; literal extraction only")
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hadamard", type=int, default=128)
    a = p.parse_args()
    meta = verify_fixture(a.fixture)
    cols = meta["input"]
    if a.hadamard < 2 or a.hadamard & (a.hadamard-1) or cols % a.hadamard:
        p.error("Hadamard size must be an even power of two dividing input width")
    w, x = load(a.fixture/"weights.f32", cols), load(a.fixture/"heldout.f32", cols)
    wr, xr = [rotate(r, a.hadamard) for r in w], [rotate(r, a.hadamard) for r in x]
    ref = [[sum(v*u for v,u in zip(row, token)) for row in w] for token in x]
    raw = [[sum(v*u for v,u in zip(row, token)) for row in wr] for token in xr]
    invariance = math.sqrt(nmse(raw, ref))
    if invariance > 1e-10:
        raise RuntimeError("Hadamard projection invariance failed")
    table = grids(a.upstream)
    a.output.mkdir(parents=True, exist_ok=True)
    records = []
    for dimension, codes in ((1,16), (2,16), (2,64), (2,256)):
        grid = table[dimension,codes]
        decoded = [encode(row, a.hadamard, grid) for row in w]
        result = [[sum(v*u for v,u in zip(row, token)) for row in decoded] for token in xr]
        bits = math.log2(codes)/dimension
        records.append(dict(vector_dimension=dimension, codes=codes, payload_bpw=bits,
                            weight_nmse=nmse(decoded,wr), heldout_rel_l2=math.sqrt(nmse(result,ref)),
                            rotation_rel_l2=invariance, logical_bytes=len(w)*cols*bits/8 + len(w)*cols/a.hadamard*2 + codes*dimension*2))
    with (a.output/"higgs-reference.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    (a.output/"higgs-reference-scope.json").write_text(json.dumps({
        "upstream_sha256": hashlib.sha256(a.upstream.read_bytes()).hexdigest(),
        "source": "https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/higgs.py",
        "fixture_sha256": meta["sha256"], "hadamard": a.hadamard,
        "scope": "upstream grids and nearest-vector encoding; FP64 arithmetic with FP16 stored grids/scales; no random signs; no FLUTE packing/runtime, GPTQ, or whole-model evaluation",
        "storage": "logical payload, one FP16 scale per Hadamard block and FP16 vector grid; excludes FLUTE duplicated scales/tables/padding/workspace",
    },indent=2)+"\n")
    print("HIGGS scalar/vector reference completed; no timings")


if __name__ == "__main__":
    main()
