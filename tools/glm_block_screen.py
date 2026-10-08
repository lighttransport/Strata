"""Prepare and run the optimistic GLM expert-block quality gate, never a speed benchmark."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (32, 48, 64, 80, 96, 112, 128)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_corpus(path, targets=512):
    rows = [[int(value) for value in line.split(",")] for line in Path(path).read_text().splitlines() if line]
    if len(rows) != 64 or any(len(row) != targets + 1 or min(row) < 0 for row in rows):
        raise ValueError("the frozen corpus must contain 64 sequences of 512 targets, code first then prose")
    if len({tuple(row) for row in rows}) != len(rows):
        raise ValueError("duplicate corpus sequences")
    return rows


def splits(calibration, evaluation):
    if {tuple(row) for row in calibration} & {tuple(row) for row in evaluation}:
        raise ValueError("calibration/evaluation sequence overlap")
    return {
        "fit": calibration[:24] + calibration[32:56],
        "tune": calibration[24:32] + calibration[56:64],
        "smoke": [calibration[24][:257], calibration[56][:257]],
        "evaluation": evaluation,
    }


def rows_from_stdout(path):
    budgets, current = {}, 128
    for line in Path(path).read_text().splitlines():
        if not line.startswith("{"):
            continue
        value = json.loads(line)
        if "block_budget" in value:
            current = f"{value['block_layer']}:{value['block_budget']}" if "block_layer" in value else value["block_budget"]
        elif "sequence" in value:
            budgets.setdefault(current, []).append(value)
    return budgets


def read_capture(path):
    """Stream input/routing and pre-mask block-energy records; never load a trace whole."""
    def exact(source, size):
        data = source.read(size)
        if len(data) != size:
            raise ValueError("truncated block capture")
        return data
    with Path(path).open("rb") as source:
        while header := source.read(32):
            if len(header) != 32:
                raise ValueError("truncated block capture header")
            magic, kind, sequence, layer, position, first, second, third = struct.unpack("<8I", header)
            if magic != 0x31434247 or kind not in (1, 2) or sequence < 1 or not 3 <= layer <= 44:
                raise ValueError("invalid block capture identity")
            record = {"kind": kind, "sequence": sequence, "layer": layer, "position": position}
            if kind == 1:
                nt, hidden, topk = first, second, third
                if not 1 <= nt <= 8 or hidden != 4096 or not 1 <= topk <= 32:
                    raise ValueError("invalid block capture input shape")
                ids = struct.unpack(f"<{nt * topk}i", exact(source, nt * topk * 4))
                weights = struct.unpack(f"<{nt * topk}f", exact(source, nt * topk * 4))
                values = struct.unpack(f"<{nt * hidden}f", exact(source, nt * hidden * 4))
                if min(ids) < 0 or any(not math.isfinite(v) or v <= 0 for v in weights) or any(not math.isfinite(v) for v in values):
                    raise ValueError("invalid block capture values")
                record.update(tokens=nt, topk=topk, experts=ids, routing=weights, inputs=values)
            else:
                if first >= 512 or second != 128 or third != 0:
                    raise ValueError("invalid block capture energy shape")
                energies = struct.unpack("<128f", exact(source, 128 * 4))
                if any(not math.isfinite(v) or v < 0 for v in energies):
                    raise ValueError("invalid block capture energy")
                record.update(expert=first, energies=energies)
            yield record


def kind_metrics(rows, sequences):
    if len(rows) != sequences or [row["sequence"] for row in rows] != list(range(1, sequences + 1)):
        raise ValueError("incomplete evaluation or repeated sequence records")
    half, last = rows[sequences // 2 - 1], rows[-1]
    prose_tokens = last["tokens"] - half["tokens"]
    if half["tokens"] <= 0 or prose_tokens <= 0:
        raise ValueError("empty code or prose evaluation")
    prose_nll = (last["nll"] * last["tokens"] - half["nll"] * half["tokens"]) / prose_tokens
    return {
        "code": {"tokens": half["tokens"], "perplexity": math.exp(half["nll"])},
        "prose": {"tokens": prose_tokens, "perplexity": math.exp(prose_nll)},
        "all": {"tokens": last["tokens"], "perplexity": last["perplexity"]},
    }


def summary(baseline_rows, control_rows, candidates, sequences, guards):
    base = kind_metrics(baseline_rows, sequences)
    control = kind_metrics(control_rows, sequences)
    if control_rows[-1].get("different_logits") != 0 or control_rows[-1].get("top1_agreement") != 1:
        raise ValueError("the all-block control did not reproduce baseline logits bit for bit")
    if any(abs(control[k]["perplexity"] / base[k]["perplexity"] - 1) > 1e-10 for k in base):
        raise ValueError("the all-block control changed perplexity")
    for guard in guards:
        if guard.get("rejected") or guard.get("exit_code") != 0 or guard.get("peak_swap_kib", 0):
            raise ValueError("screen process failed or exceeded the memory guard")
    result = {"diagnostic_only": True, "qualified": False, "baseline": base, "all_blocks_bitwise_equal": True,
              "quality_limit": 1.01, "candidates": {}}
    for keep, rows in candidates.items():
        metrics = kind_metrics(rows, sequences)
        ratios = {kind: metrics[kind]["perplexity"] / base[kind]["perplexity"] for kind in base}
        result["candidates"][str(keep)] = {"retained_blocks": keep, "retained_fraction": keep / 128,
            "perplexity_ratios": ratios, "quality_screen_pass": all(ratio <= 1.01 for ratio in ratios.values()),
            "metrics": metrics, "kl": rows[-1].get("kl"), "top1_agreement": rows[-1].get("top1_agreement")}
    return result


def environment():
    # A screen is always compared with the same unbiased full-width Q23 target.
    env = {key: value for key, value in os.environ.items() if not key.startswith(("STRATA_GLM_", "STRATA_Q23_"))}
    env.update({"STRATA_GLM_CANON": "1", "STRATA_GLM_CANON_COMBINE": "strict", "STRATA_GLM_STEP_PIPELINE": "1",
        "STRATA_GLM_LAYER_GRAPHS": "1", "STRATA_GLM_VERIFY_GRAPHS": "1", "STRATA_GLM_QUANT_ONCE": "1",
        "STRATA_GLM_Q8_DECODE": "1", "STRATA_GLM_LAYER_FLOW": "1", "STRATA_GLM_MAPPED_OWNED": "1",
        "STRATA_GLM_MAPPED_LAZY": "1", "STRATA_NATIVE_NUMA_LOCAL": "0", "STRATA_NATIVE_TASKS_PER_THREAD": "12"})
    return env


def block_reports(path):
    reports, current = [], {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("BLOCK_SCREEN layer="):
            fields = dict(word.split("=", 1) for word in line.split()[1:])
            current[int(fields["layer"])] = {key: float(value) for key, value in fields.items()}
        elif line.startswith("BLOCK_SCREEN_TOTAL "):
            reports.append(current); current = {}
    return reports


def allocate_layers(baseline, cases, reports, sequences, error_budget=math.log(1.01) * .8):
    """Conservative additive sensitivity heuristic; combined profiles need evaluation."""
    base = kind_metrics(baseline, sequences)
    if len(cases) != len(reports):
        raise ValueError("layer sensitivity metrics and byte reports do not match")
    choices = []
    for (key, rows), report in zip(cases.items(), reports):
        layer, keep = map(int, key.split(":")); metrics = kind_metrics(rows, sequences)
        costs = {kind: max(0, math.log(metrics[kind]["perplexity"] / base[kind]["perplexity"])) for kind in ("code", "prose")}
        saved = report[layer]["full_expert_bytes"] - report[layer]["ideal_union_bytes"]
        if saved > 0:
            choices.append((max(costs.values()) / saved, layer, keep, costs, saved, report[layer]["full_expert_bytes"]))
    layers, used, saved = {}, {"code": 0.0, "prose": 0.0}, 0.0
    for _, layer, keep, costs, benefit, _ in sorted(choices):
        if str(layer) not in layers and all(used[kind] + costs[kind] <= error_budget for kind in used):
            layers[str(layer)] = keep; saved += benefit
            for kind in used:
                used[kind] += costs[kind]
    full = sum(next(iter(reports), {}).get(layer, {}).get("full_expert_bytes", 0) for layer in range(3, 45))
    return {"layers": layers, "estimated_additive_nll_change": used, "ideal_saved_bytes": saved,
            "ideal_saved_fraction": saved / full if full else 0, "requires_combined_validation": True,
            "not_an_error_bound": True}


def stage(work, name, command, env):
    prefix = work / name
    if any(prefix.with_suffix(suffix).exists() for suffix in (".log", ".stdout", ".memory.json")):
        raise ValueError(f"refusing to overwrite stage {name}")
    gpu_guard = ["--min-gpu-free-mib=512"] if any(word.startswith("--gpu-budget-mib=") for word in command) else []
    guarded = [sys.executable, str(ROOT / "tools/glm_q2_run_guard.py"), "--output", str(prefix), *gpu_guard, "--", *command]
    print("START", name, flush=True)
    subprocess.run(guarded, cwd=ROOT, env=env, check=True)
    print("END", name, flush=True)


def prepare(args):
    work = args.work.resolve()
    if work.exists():
        raise ValueError("prepare requires a new work directory")
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    for name in ("calibration", "evaluation"):
        if digest(args.corpus / f"{name}.ids") != manifest[f"{name}_sha256"]:
            raise ValueError("frozen corpus checksum mismatch")
    sets = splits(read_corpus(args.corpus / "calibration.ids"), read_corpus(args.corpus / "evaluation.ids"))
    work.mkdir(parents=True)
    records = {"version": 1, "source_corpus": manifest, "splits": {}}
    for name, rows in sets.items():
        path = work / f"{name}.ids"
        path.write_text("\n".join(",".join(map(str, row)) for row in rows) + "\n")
        records["splits"][name] = {"sha256": digest(path), "targets": sum(len(row) - 1 for row in rows), "sequences": len(rows)}
    stage(work, "norms", [str(args.norms_tool.resolve()), str(args.model.resolve()), str(args.pack.resolve()),
          str(work / "down-norms.bin"), str(args.norm_threads)], environment())
    with (work / "down-norms.bin").open("rb") as source:
        header = source.read(40)
    magic, version, layers, experts, hidden, channels, identity, checksum = struct.unpack("<6I2Q", header)
    if (magic, version, layers, hidden, channels) != (0x314E4247, 1, 45, 4096, 2048):
        raise ValueError("invalid generated norms")
    records["norms"] = {"sha256": digest(work / "down-norms.bin"), "source_identity": f"{identity:016x}", "experts": experts, "payload_hash": checksum}
    records["model"] = str(args.model.resolve()); records["pack"] = str(args.pack.resolve())
    for keep in BUDGETS:
        profile = {"version": 1, "mode": "oracle", "block_channels": 16, "source_identity": f"{identity:016x}",
                   "norms": "down-norms.bin", "retained_blocks": keep, "layers": {}}
        (work / f"oracle-{keep}.json").write_text(json.dumps(profile, indent=2) + "\n")
    (work / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


def run(args):
    work = args.work.resolve(); manifest = json.loads((work / "manifest.json").read_text())
    corpus = work / f"{args.split}.ids"
    if digest(corpus) != manifest["splits"][args.split]["sha256"] or digest(work / "down-norms.bin") != manifest["norms"]["sha256"]:
        raise ValueError("screen input checksum mismatch")
    executable = args.decoder.resolve(); env = environment()
    command = ["numactl", "--interleave=all", str(executable), manifest["model"], "0", "1", "4096", "15",
        "--context=8192", "--prefill-batch=2048", "--gpu-budget-mib=12288", "--cpu-affinity=numa", "--decode-graphs",
        "--expert-pack=" + manifest["pack"], "--eval-corpus=" + str(corpus)]
    reference = work / f"{args.split}-reference.logits"
    if reference.exists():
        raise ValueError("refusing to overwrite screen reference")
    (work / f"{args.split}-run.json").write_text(json.dumps({"decoder_sha256": digest(executable), "command": command,
        "environment": {k: v for k, v in env.items() if k.startswith("STRATA_")}}, indent=2) + "\n")
    profile = "--expert-block-screen=" + str(work / "oracle-128.json")
    stage(work, args.split + "-baseline", [*command, "--eval-save-logits=" + str(reference)], env)
    stage(work, args.split + "-control", [*command, profile, "--eval-reference=" + str(reference)], env)
    stage(work, args.split + "-sweep", [*command, profile, "--eval-reference=" + str(reference),
        "--expert-block-sweep=" + ",".join(map(str, BUDGETS[:-1]))], env)
    guards = [json.loads((work / f"{args.split}-{stage_name}.memory.json").read_text()) for stage_name in ("baseline", "control", "sweep")]
    result = summary(rows_from_stdout(work / f"{args.split}-baseline.stdout")[128], rows_from_stdout(work / f"{args.split}-control.stdout")[128],
        rows_from_stdout(work / f"{args.split}-sweep.stdout"), manifest["splits"][args.split]["sequences"], guards)
    result["split"] = args.split
    result["screen_targets"] = manifest["splits"][args.split]["targets"]
    (work / f"{args.split}-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def layer_screen(args):
    work = args.work.resolve(); manifest = json.loads((work / "manifest.json").read_text())
    corpus = work / f"{args.split}.ids"; executable = args.decoder.resolve()
    if digest(corpus) != manifest["splits"][args.split]["sha256"]:
        raise ValueError("screen corpus checksum mismatch")
    with (work / "down-norms.bin").open("rb") as source:
        source.seek(40); eligible = source.read(45)
    layers = args.layers or [i for i in range(3, 45) if eligible[i]]
    command = ["numactl", "--interleave=all", str(executable), manifest["model"], "0", "1", "4096", "15",
        "--context=8192", "--prefill-batch=2048", "--gpu-budget-mib=12288", "--cpu-affinity=numa", "--decode-graphs",
        "--expert-pack=" + manifest["pack"], "--eval-corpus=" + str(corpus),
        "--expert-block-screen=" + str(work / "oracle-128.json"), "--eval-reference=" + str(work / f"{args.split}-reference.logits"),
        "--expert-block-sweep=" + ",".join(map(str, args.budgets)), "--expert-block-layer-sweep=" + ",".join(map(str, layers))]
    name = args.split + "-layers"
    (work / f"{name}-run.json").write_text(json.dumps({"command": command, "decoder_sha256": digest(executable)}, indent=2) + "\n")
    stage(work, name, command, environment())
    guard = json.loads((work / f"{name}.memory.json").read_text())
    if guard.get("rejected") or guard["exit_code"] or guard["peak_swap_kib"]:
        raise ValueError("invalid layer sensitivity run")
    cases = rows_from_stdout(work / f"{name}.stdout")
    if len(cases) != len(layers) * len(args.budgets):
        raise ValueError("incomplete layer sensitivity sweep")
    baseline = rows_from_stdout(work / f"{args.split}-baseline.stdout")[128]
    allocation = allocate_layers(baseline, cases, block_reports(work / f"{name}.log"), manifest["splits"][args.split]["sequences"])
    result = {"diagnostic_only": True, "split": args.split, "allocation": allocation, "cases": cases}
    (work / f"{name}-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    profile = json.loads((work / "oracle-128.json").read_text()); profile["layers"] = allocation["layers"]
    (work / "adaptive-oracle.json").write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps(allocation, indent=2), flush=True)


def capture(args):
    work = args.work.resolve(); manifest = json.loads((work / "manifest.json").read_text())
    corpus = work / "fit.ids"
    if digest(corpus) != manifest["splits"]["fit"]["sha256"]:
        raise ValueError("fitting corpus checksum mismatch")
    command = ["numactl", "--interleave=all", str(args.decoder.resolve()), manifest["model"], "0", "1", "4096", "15",
        "--context=8192", "--prefill-batch=2048", "--gpu-budget-mib=12288", "--cpu-affinity=numa", "--decode-graphs",
        "--expert-pack=" + manifest["pack"], "--eval-corpus=" + str(corpus), "--expert-block-screen=" + str(work / "oracle-128.json"),
        "--expert-block-capture=" + str(work / "fit-capture")]
    stage(work, "fit-capture", command, environment())
    files = {path.name: digest(path) for path in sorted((work / "fit-capture").glob("*.bin"))}
    (work / "fit-capture/checksums.json").write_text(json.dumps({"files": files, "corpus_sha256": digest(corpus),
        "decoder_sha256": digest(args.decoder.resolve())}, indent=2) + "\n")


def evaluate_profile(args):
    work = args.work.resolve(); manifest = json.loads((work / "manifest.json").read_text())
    corpus = work / f"{args.split}.ids"; profile = args.profile.resolve(); executable = args.decoder.resolve()
    if digest(corpus) != manifest["splits"][args.split]["sha256"]:
        raise ValueError("profile evaluation corpus checksum mismatch")
    reference = work / f"{args.split}-reference.logits"
    if not reference.exists():
        raise ValueError("profile evaluation requires an existing baseline for this split")
    command = ["numactl", "--interleave=all", str(executable), manifest["model"], "0", "1", "4096", "15",
        "--context=8192", "--prefill-batch=2048", "--gpu-budget-mib=12288", "--cpu-affinity=numa", "--decode-graphs",
        "--expert-pack=" + manifest["pack"], "--eval-corpus=" + str(corpus), "--expert-block-screen=" + str(profile),
        "--eval-reference=" + str(reference)]
    if args.capture:
        command.append("--expert-block-capture=" + str(args.capture.resolve()))
    if not args.tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.tag):
        raise ValueError("profile stage tag must contain only letters, digits, underscores or hyphens")
    name = args.split + "-" + args.tag
    (work / f"{name}-run.json").write_text(json.dumps({"command": command, "decoder_sha256": digest(executable),
        "profile_sha256": digest(profile)}, indent=2) + "\n")
    stage(work, name, command, environment())
    guard = json.loads((work / f"{name}.memory.json").read_text())
    if guard.get("rejected") or guard["exit_code"] or guard["peak_swap_kib"]:
        raise ValueError("invalid combined profile evaluation")
    sequences = manifest["splits"][args.split]["sequences"]
    baseline = kind_metrics(rows_from_stdout(work / f"{args.split}-baseline.stdout")[128], sequences)
    rows = rows_from_stdout(work / f"{name}.stdout")[128]
    candidate = kind_metrics(rows, sequences)
    ratios = {kind: candidate[kind]["perplexity"] / baseline[kind]["perplexity"] for kind in baseline}
    total_lines = [line for line in (work / f"{name}.log").read_text().splitlines() if line.startswith("BLOCK_SCREEN_TOTAL ")]
    if len(total_lines) != 1:
        raise ValueError("missing or repeated combined-profile byte report")
    fields = dict(word.split("=", 1) for word in total_lines[0].split()[1:])
    result = {"diagnostic_only": True, "qualified": False, "split": args.split, "profile_sha256": digest(profile),
        "perplexity_ratios": ratios, "quality_screen_pass": all(v <= 1.01 for v in ratios.values()),
        "ideal_union_byte_reduction": float(fields["ideal_byte_reduction"]), "kl": rows[-1]["kl"],
        "top1_agreement": rows[-1]["top1_agreement"], "different_logits": rows[-1]["different_logits"], "memory": guard}
    (work / f"{name}-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("prepare")
    prep.add_argument("model", type=Path); prep.add_argument("pack", type=Path); prep.add_argument("work", type=Path)
    prep.add_argument("--corpus", type=Path, default=ROOT / "build-q2-redesign/corpus")
    prep.add_argument("--norms-tool", type=Path, default=ROOT / "build-q2-v6-cpu/strata-glm-block-norms")
    prep.add_argument("--norm-threads", type=int, choices=range(1, 17), default=4)
    execute = modes.add_parser("run")
    execute.add_argument("work", type=Path); execute.add_argument("--split", choices=("smoke", "tune", "evaluation"), default="smoke")
    execute.add_argument("--decoder", type=Path, default=ROOT / "build-q2-v6/strata-glm-decode")
    sensitivity = modes.add_parser("layers")
    sensitivity.add_argument("work", type=Path); sensitivity.add_argument("--split", choices=("smoke", "tune"), default="smoke")
    sensitivity.add_argument("--decoder", type=Path, default=ROOT / "build-q2-v6/strata-glm-decode")
    sensitivity.add_argument("--budgets", type=int, nargs="+", choices=range(1, 129), default=[64])
    sensitivity.add_argument("--layers", type=int, nargs="+", choices=range(3, 45))
    collect = modes.add_parser("capture")
    collect.add_argument("work", type=Path)
    collect.add_argument("--decoder", type=Path, default=ROOT / "build-q2-v6/strata-glm-decode")
    combined = modes.add_parser("profile")
    combined.add_argument("work", type=Path); combined.add_argument("profile", type=Path)
    combined.add_argument("--split", choices=("smoke", "tune", "evaluation"), default="smoke")
    combined.add_argument("--tag", default="adaptive")
    combined.add_argument("--capture", type=Path)
    combined.add_argument("--decoder", type=Path, default=ROOT / "build-q2-v6/strata-glm-decode")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "layers":
        layer_screen(args)
    elif args.mode == "capture":
        capture(args)
    elif args.mode == "profile":
        evaluate_profile(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
