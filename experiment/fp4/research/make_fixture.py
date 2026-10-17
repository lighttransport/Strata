#!/usr/bin/env python3
"""Create small synthetic file-input fixtures; never downloads model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cols", type=int, default=128)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--profile", choices=("gaussian", "outliers", "correlated", "rank-deficient", "shift", "fallback"), default="gaussian")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.seed < 0 or args.seed > 2**32-1:
        parser.error("seed must be an unsigned 32-bit integer")
    rng = random.Random(args.seed)
    cols, rows = args.cols, args.rows
    if cols < 128 or cols > 1024 or cols % 128 or rows < 16 or rows > 128 or rows % 16:
        parser.error("fixture cols must be 128..1024 in multiples of 128; rows 16..128 in multiples of 16")
    def vector():
        return [rng.gauss(0, 1) * (8 if c % 97 == 0 else 1) for c in range(cols)]
    # Calibration and held-out projection use disjoint generated vectors.
    values = {"weights.f32": [rng.gauss(0, 1) for _ in range(rows * cols)],
              "zero-weights.f32": [0.0] * (rows * cols),
              "zero-activation.f32": [0.0] * cols,
              "calibration.f32": [v for _ in range(8) for v in vector()],
              "activation.f32": vector(),
              "heldout.f32": [v for _ in range(16) for v in vector()]}
    # Separate seeds preserve the original eight-vector and held-out fixtures.
    def independent_vectors(seed, count):
        extra = random.Random(seed)
        return [extra.gauss(0, 1) * (8 if c % 97 == 0 else 1) for _ in range(count) for c in range(cols)]
    values["calibration-large.f32"] = independent_vectors(args.seed + 792, 96)
    values["validation.f32"] = independent_vectors(args.seed + 793, 32)
    if args.profile != "gaussian":
        def profiled(seed, count, heldout=False):
            source = random.Random(seed)
            data = []
            for _ in range(count):
                latent = [source.gauss(0, 1) for _ in range(8)]
                for c in range(cols):
                    if args.profile == "rank-deficient":
                        v = latent[c % 8]
                    elif args.profile == "correlated":
                        v = latent[c // 16 % 8] + 0.1 * source.gauss(0, 1)
                    else:
                        v = source.gauss(0, 1)
                        if c % 31 == (7 if heldout and args.profile == "shift" else 0):
                            v *= 64
                    data.append(v)
            return data
        for name, seed, count in (("calibration-large.f32", args.seed + 792, 96),
                                  ("validation.f32", args.seed + 793, 32),
                                  ("heldout.f32", args.seed + 794, 16)):
            values[name] = profiled(seed, count, name == "heldout.f32")
        # A zero validation signal cannot establish a gain: exercise exact fallback.
        if args.profile == "fallback":
            values["validation.f32"] = [0.0] * (32 * cols)
    hashes = {}
    for name, data in values.items():
        raw = struct.pack("<" + "f" * len(data), *data)
        (args.output / name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    (args.output / "fixture.json").write_text(json.dumps({
        "scope": "synthetic file-reader smoke fixture, not real model weights",
        "seed": args.seed,
        "profile": args.profile,
        "input": cols, "output": rows, "calibration_tokens": 8,
        "heldout_tokens": 16,
        "large_calibration_tokens": 96, "validation_tokens": 32,
        "encoding": "little-endian float32, row-major", "sha256": hashes,
    }, indent=2) + "\n")
    print("Synthetic file-input fixture written")


if __name__ == "__main__":
    main()
