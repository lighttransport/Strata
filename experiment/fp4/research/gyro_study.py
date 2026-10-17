#!/usr/bin/env python3
"""Dependency-free GyRot-inspired W4A4 grouping study, not a paper reproduction."""
import argparse
import csv
import json
import hashlib
import math
from pathlib import Path
import random
import struct
from fixture_validation import verify_fixture


def rotate(values, width):
    result = values.copy()
    if not width:
        return result
    for base in range(0, len(values), width):
        step = 1
        while step < width:
            for i in range(base, base + width, 2 * step):
                for j in range(step):
                    a, b = result[i + j], result[i + j + step]
                    result[i + j], result[i + j + step] = a + b, a - b
            step *= 2
        for i in range(base, base + width):
            result[i] /= math.sqrt(width)
    return result


def quantum(maximum, limit):
    return 2 ** math.ceil(math.log2(max(maximum / limit, 1e-30)))


def group_params(rows, group, asymmetric):
    params = []
    for row in rows:
        for base in range(0, len(row), group):
            values = row[base:base + group]
            bias = -min(values) if asymmetric else 0
            scale = (max(values) + bias) / 15 if asymmetric else max(map(abs, values)) / 7
            params.append((scale, bias))
    return params


def metadata_quanta(rows, group, asymmetric):
    params = group_params(rows, group, asymmetric)
    return quantum(max(s for s, _ in params), 255), quantum(max(abs(z) for _, z in params), 127)


def quantize(rows, group, asymmetric, metadata, quanta=None):
    params = group_params(rows, group, asymmetric)
    sq, zq = quanta if quanta is not None else metadata_quanta(rows, group, asymmetric)
    result, index = [], 0
    for row in rows:
        restored = []
        for base in range(0, len(row), group):
            scale, bias = params[index]
            index += 1
            if metadata == "int8":
                bias = min(127, max(-128, math.ceil(bias / zq))) * zq if asymmetric else 0
                if asymmetric:
                    scale = (max(row[base:base + group]) + bias) / 15
                scale = max(1, min(255, round(scale / sq))) * sq
            for v in row[base:base + group]:
                code = round((v + bias) / scale) if scale else 0
                code = min(15 if asymmetric else 7, max(0 if asymmetric else -8, code))
                restored.append(code * scale - bias)
        result.append(restored)
    return result


def nmse(actual, reference):
    error = sum((a - b) ** 2 for ar, br in zip(actual, reference) for a, b in zip(ar, br))
    energy = sum(b * b for row in reference for b in row)
    return error / max(energy, 1e-30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true", help="use the larger accuracy-study extent; no timing")
    parser.add_argument("--weights-json", type=Path, help="optional finite [output,input] source matrix")
    parser.add_argument("--fixture", type=Path, help="export_layer.py fixture with weights and disjoint calibration/heldout inputs")
    parser.add_argument("--outlier-score", choices=("energy", "maxabs"), default="energy")
    args = parser.parse_args()
    rng = random.Random(42)
    n, outputs, tokens = (1024, 32, 16) if args.benchmark else (1024, 8, 2)
    weights = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(outputs)]
    if args.weights_json:
        weights = json.loads(args.weights_json.read_text())
        if not isinstance(weights, list) or not weights or not isinstance(weights[0], list):
            raise ValueError("weights must be a nonempty matrix")
        n, outputs = len(weights[0]), len(weights)
        if n < 128 or n & (n - 1) or any(len(row) != n for row in weights):
            raise ValueError("input width must be a power of two >=128; rows must have equal length")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for row in weights for v in row):
            raise ValueError("weights must be finite numbers")
    def activations(count):
        return [[rng.gauss(0, 1) * (20 if c % 97 == 0 else 1) for c in range(n)] for _ in range(count)]
    calibration, heldout = activations(8), activations(tokens)
    fixture_meta = None
    if args.fixture:
        if args.weights_json:
            parser.error("fixture and weights-json are mutually exclusive")
        fixture_meta = verify_fixture(args.fixture)
        n = fixture_meta["input"]
        if n < 128 or n & (n - 1) or n > 16384:
            parser.error("fixture input must be a power of two in 128..16384")
        def read_matrix(name):
            path = args.fixture / name
            if path.stat().st_size > 512*1024**2:
                raise ValueError("matrix exceeds 512 MiB")
            raw = path.read_bytes()
            if not raw or len(raw) % (4*n):
                raise ValueError("invalid fixture geometry")
            values = list(struct.unpack("<" + "f"*(len(raw)//4), raw))
            if not all(math.isfinite(v) for v in values):
                raise ValueError("nonfinite fixture")
            return [values[i:i+n] for i in range(0,len(values),n)]
        weights = read_matrix("weights.f32")
        calibration = read_matrix("calibration-large.f32")
        heldout = read_matrix("heldout.f32")
        if set(map(tuple,calibration)) & set(map(tuple,heldout)):
            raise ValueError("calibration and heldout overlap")
        outputs, tokens = len(weights), len(heldout)
    reference = [[sum(a * b for a, b in zip(w, x)) for w in weights] for x in heldout]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "gyro_accuracy.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["rotation", "group", "permutation", "metadata", "activation_quant", "weight_nmse", "activation_nmse", "projection_rel_l2", "rotation_rel_l2", "weight_bits_with_scale", "weight_bytes_estimate"])
        for width in sorted({0, 32, 128, min(512, n), min(1024, n)}):
            for group in (32, 64, 128):
                if width and width < group:
                    continue
                for hap in (False, True):
                    if hap and (not width or width <= group):
                        continue
                    perm = list(range(n))
                    if hap:
                        # Sylvester rows at multiples of G have constant signs within each G-wide group.
                        ranked = sorted(range(n), key=lambda c: max(abs(x[c]) for x in calibration)
                                        if args.outlier_score == "maxabs" else sum(x[c] ** 2 for x in calibration), reverse=True)
                        slots = [base + i for base in range(0, n, width) for i in range(0, width, group)]
                        rest = [i for i in range(n) if i not in set(slots)]
                        for target, original in zip(slots + rest, ranked):
                            perm[target] = original
                    wr = [rotate([w[c] for c in perm], width) for w in weights]
                    xr = [rotate([x[c] for c in perm], width) for x in heldout]
                    cr = [rotate([x[c] for c in perm], width) for x in calibration]
                    unquantized = [[sum(a * b for a, b in zip(w, x)) for w in wr] for x in xr]
                    for metadata in ("fp32", "int8"):
                        wq = quantize(wr, group, False, metadata)
                        for asym in (False, True):
                            # Shared layer metadata quanta come from calibration only;
                            # activation group ranges themselves remain dynamic per token.
                            xq = quantize(xr, group, asym, metadata, metadata_quanta(cr, group, asym))
                            result = [[sum(a * b for a, b in zip(w, x)) for w in wq] for x in xq]
                            bits = 4 + (8 if metadata == "int8" else 32) / group
                            size = outputs * n * bits / 8 + (4 if metadata == "int8" else 0)
                            metrics = [nmse(wq, wr), nmse(xq, xr), math.sqrt(nmse(result, reference)), math.sqrt(nmse(unquantized, reference))]
                            if not all(math.isfinite(v) for v in metrics):
                                raise RuntimeError("nonfinite study metric")
                            writer.writerow([width, group, "harmonic_slots" if hap else "none", metadata,
                                             "asymmetric_ceiling" if asym else "symmetric", *metrics, bits, size])
    (args.output / "gyro_scope.json").write_text(json.dumps({
        "scope": "GyRot-inspired RTN proxy; no GPTQ, KV quantization, integer accelerator, or model-quality claim",
        "weights": "fixture" if args.fixture else "provided_json" if args.weights_json else "seeded_synthetic",
        "fixture_manifest_sha256": hashlib.sha256((args.fixture / "fixture.json").read_bytes()).hexdigest() if args.fixture else None,
        "activation_source": fixture_meta.get("input_source", "synthetic") if fixture_meta else "synthetic_outliers", "input": n, "output": outputs,
        "calibration_tokens": len(calibration), "heldout_tokens": tokens,
        "metadata": "unsigned 8-bit scale; signed ceiling-rounded 8-bit bias; shared power-of-two quanta",
        "activation_metadata_quanta": "calibration-only; frozen for heldout, with saturation on range overflow",
        "permutation": "calibration " + args.outlier_score + " ranks into block harmonic slots; paper-guided, not official HAP implementation",
        "storage": "weight payload/scales estimate only; excludes permutations, activations and activation metadata",
    }, indent=2) + "\n")
    print("GyRot-inspired accuracy study completed; no timing or model-quality qualification")


if __name__ == "__main__":
    main()
