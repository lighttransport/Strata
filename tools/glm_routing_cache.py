"""Estimate GPU cache coverage from a GLM decode routing trace, without allocating GPU memory."""
import argparse
import collections
import csv
import json
import pathlib
import re

from gguf_reader import GGUFFile


def expert_sizes(model):
    parts = collections.defaultdict(dict)
    for path in sorted(model.parent.glob("*.gguf")):
        for tensor in GGUFFile(path).tensors:
            match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", tensor.name)
            if match and int(match[1]) < 45:
                size = tensor.expected_bytes()
                if size is None or len(tensor.shape) != 3:
                    raise ValueError("unsupported expert geometry")
                parts[int(match[1])][match[2]] = size // tensor.shape[2]
    if set(parts) != set(range(3, 45)) or any(set(p) != {"gate", "up", "down"} for p in parts.values()):
        raise ValueError("expected complete GLM-5.3-Flash main expert tensors")
    return {layer: sum(p.values()) for layer, p in parts.items()}


def read_trace(path, sizes):
    rows = {}
    with path.open() as file:
        for row in csv.reader(file):
            values = list(map(int, row))
            if len(values) != 10:
                raise ValueError("expected position, layer and eight expert IDs")
            position, layer, *experts = values
            if layer not in sizes or position < 0 or len(set(experts)) != 8 or any(e < 0 or e >= 288 for e in experts):
                raise ValueError("invalid routing geometry")
            if (position, layer) in rows:
                raise ValueError("trace contains repeated positions; use one decode repetition")
            rows[position, layer] = experts
    positions = sorted({p for p, _ in rows})
    if not positions or any((p, l) not in rows for p in positions for l in sizes):
        raise ValueError("incomplete routing trace")
    return rows, positions


def counts(rows, positions):
    result = collections.Counter()
    for (p, layer), experts in rows.items():
        if p in positions:
            result.update((layer, e) for e in experts)
    return result


def coverage(training, evaluation, sizes, budget):
    selected = set()
    used = 0
    # Benefit is frequency * bytes; dividing by slot bytes leaves frequency.
    for pair, frequency in sorted(training.items(), key=lambda item: (-item[1], item[0])):
        cost = sizes[pair[0]]
        if cost <= budget - used:
            selected.add(pair)
            used += cost
    total = sum(n * sizes[l] for (l, _), n in evaluation.items())
    hit = sum(n * sizes[l] for (l, e), n in evaluation.items() if (l, e) in selected)
    return dict(slots=len(selected), allocated_bytes=used, byte_hit_fraction=hit / total,
                selection_hit_fraction=sum(n for pair, n in evaluation.items() if pair in selected) / sum(evaluation.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("trace", type=pathlib.Path)
    parser.add_argument("--training-tokens", type=int, default=128)
    parser.add_argument("--cache-mib", type=int, nargs="+", default=[512, 1024, 2048, 3072])
    parser.add_argument("--decode-log", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    sizes = expert_sizes(args.model)
    rows, positions = read_trace(args.trace, sizes)
    if not 0 < args.training_tokens < len(positions) or any(n < 1 for n in args.cache_mib):
        parser.error("need nonempty, separate training/evaluation positions and positive cache budgets")
    training = counts(rows, set(positions[:args.training_tokens]))
    evaluation = counts(rows, set(positions[args.training_tokens:]))
    results = []
    for mib in args.cache_mib:
        results.append(dict(cache_mib=mib, held_out=coverage(training, evaluation, sizes, mib * 1024**2),
                            oracle=coverage(evaluation, evaluation, sizes, mib * 1024**2)))
    result = dict(model=args.model.name, trace_positions=len(positions), training_positions=args.training_tokens,
                  evaluation_positions=len(positions) - args.training_tokens, plans=results,
                  scope="coverage only; oracle uses future routes and is not a deployable predictor")
    if args.decode_log:
        log = args.decode_log.read_text()
        decode = re.search(r"DECODE steps=(\d+) ms=([\d.]+) tok_s=([\d.]+)", log)
        cpu = re.findall(r"CPU_EXPERT gu_ms=([\d.]+) quant_ms=([\d.]+) down_ms=([\d.]+)", log)
        if not decode or not cpu or int(decode[1]) != len(positions):
            raise ValueError("decode timings do not match trace")
        elapsed = float(decode[2])
        expert_ms = sum(map(float, cpu[-1]))
        for plan in results:
            for mode in ("held_out", "oracle"):
                remaining = elapsed - expert_ms * plan[mode]["byte_hit_fraction"]
                plan[mode]["optimistic_tokens_per_second"] = len(positions) * 1000 / remaining
        result["baseline_tokens_per_second"] = float(decode[3])
        result["bound_assumptions"] = "expert time scales with cached bytes; zero GPU-hit cost, fill cost and extra synchronization"
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
