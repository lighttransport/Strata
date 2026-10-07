"""Per-layer gate thresholds for STRATA_GLM_GATE_SKIP from a STRATA_Q23_GATE_STATS histogram.

For each layer, tau is the largest histogram edge below which at most `fraction` of the routed experts' units have
their |silu(gate)|: that share of up rows is not read on the CPU, and those units contribute zero.
"""
import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stats", type=pathlib.Path)
    ap.add_argument("output", type=pathlib.Path)
    ap.add_argument("--fraction", type=float, required=True, help="share of units to skip per layer, 0..0.9")
    args = ap.parse_args()
    if not 0 <= args.fraction <= 0.9:
        raise SystemExit("fraction must be within 0..0.9")
    stats = json.loads(args.stats.read_text())
    bin0, per_octave = stats["bin0_log2"], stats["bins_per_octave"]
    layers, skipped = {}, {}
    for layer, counts in stats["layers"].items():
        total, running, edge = sum(counts), 0, 0
        for index, count in enumerate(counts):
            if running + count > args.fraction * total:
                break
            running += count
            edge = index + 1
        layers[layer] = 2.0 ** (bin0 + edge / per_octave) if edge else 0.0
        skipped[layer] = running / total
    args.output.write_text(json.dumps(dict(fraction=args.fraction, layers=layers, skipped=skipped), indent=1) + "\n")
    print(f"layers {len(layers)}, mean skipped share {sum(skipped.values()) / len(skipped):.3f}")


if __name__ == "__main__":
    main()
