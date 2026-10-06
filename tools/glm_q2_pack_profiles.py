"""Create deterministic mixed-format profiles from calibration reconstruction errors."""
import argparse
import json
import pathlib
import math
import re

def profiles(report):
    groups = {}
    seen = set()
    for row in report["projections"]:
        name = row["name"]
        match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", name)
        if not match or not 3 <= int(match[1]) <= 44 or name in seen:
            raise ValueError("invalid or duplicate expert projection")
        if not math.isfinite(row["normalized_mse"]) or row["normalized_mse"] < 0:
            raise ValueError("invalid reconstruction error")
        seen.add(name)
        key = name.replace("ffn_up_exps", "ffn_gate_exps")
        groups.setdefault(key, []).append(row)
    if not seen:
        raise ValueError("empty conversion report")
    for name, group in groups.items():
        if "ffn_gate_exps" in name and len(group) != 2:
            raise ValueError("gate/up projections must be paired")
    ranked = sorted(groups.values(), key=lambda group: (-max(row["normalized_mse"] for row in group), min(row["name"] for row in group)))
    result = {"q23": {"retain": []}, "q2": {"retain": sorted(row["name"] for row in report["projections"] if "ffn_down_exps" in row["name"])}}
    result["original"] = {"retain": sorted(seen)}
    count = len(report["projections"])
    for percent in (25, 50, 75):
        selected = []
        for group in ranked:
            if len(selected) * 100 >= percent * count:
                break
            selected.extend(row["name"] for row in group)
        result[f"retain{percent}"] = {"retain": sorted(selected)}
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in profiles(json.loads(args.report.read_text())).items():
        (args.output / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")

if __name__ == "__main__":
    main()
