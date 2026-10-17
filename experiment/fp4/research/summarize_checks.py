#!/usr/bin/env python3
"""Summarize diagnostic CSVs without ranking methods by unqualified timings."""
import argparse
import csv
import math
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    lines = ["# Quantization smoke diagnostics", "",
             "Fixture diagnostics; no model-quality or timing qualification.", "",
             "Held-out metrics use FP64 projections of decoded weights. They exclude q8 activation error.", "",
             "Different bit budgets are shown separately; this table does not establish a best method.", "",
             "| File | Kernel | bits/weight | Held-out tokens | Held-out relative L2 | Worst token |",
             "|---|---|---:|---:|---:|---:|"]
    notes = []
    provenance = args.input / "real-fixture-provenance.json"
    if provenance.exists():
        meta = json.loads(provenance.read_text())
        notes.append(f"Real-weight fixture: {meta['tensor']}, {meta['source_type']}; "
                     f"input provenance: {meta['input_source']}. Cropped projections do not evaluate the full layer.")
    for path in sorted(args.input.glob("*.csv")):
        with path.open() as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            if not row.get("heldout_tokens") or int(row["heldout_tokens"]) == 0:
                continue
            error = float(row["heldout_rel_l2"])
            if not math.isfinite(error):
                raise ValueError("nonfinite held-out report")
            lines.append(f"| {path.name} | {row['kernel']} | {float(row['effective_bpw']):.3f} | "
                         f"{row['heldout_tokens']} | {error:.6f} | {float(row['heldout_worst_rel_l2']):.6f} |")
            baseline = float(row["ldl_baseline_heldout_rel_l2"])
            scale_baseline = float(row.get("scale_cal_baseline_heldout_rel_l2", "nan"))
            if math.isfinite(scale_baseline):
                notes.append(f"Scale calibration {row['device']}: baseline held-out {scale_baseline:.6f}, "
                             f"selected {error:.6f}; candidate {'accepted' if row['scale_cal_selected']=='1' else 'rejected'} "
                             "by paired validation, without held-out selection.")
            if math.isfinite(baseline):
                change = error - baseline
                selection = "validation" if int(row.get("tuning_trials", "0")) else "calibration"
                notes.append(f"LDL {row['device']} ({row['kernel']}): held-out baseline {baseline:.6f}, "
                             f"selected {error:.6f} (change {change:+.6f}); selection used {selection} only.")
                if selection == "validation":
                    notes.append(f"Damping {float(row['selected_hessian_reg']):g}; feedback "
                                 f"{'selected' if row['selected_feedback']=='1' else 'rejected'}; "
                                 f"{row['tuning_trials']} fixed trial evaluations.")
                    if row.get("validation_guard") == "paired":
                        notes.append(f"Paired 2SE heuristic: {row['guard_rejections']} lower-mean candidates rejected; "
                                     f"selected gain {float(row['validation_gain']):.6f}, "
                                     f"standard error {float(row['validation_gain_stderr']):.6f}.")
    for note in notes:
        lines += ["", note]
    seed_records = []
    paths = sorted(args.input.glob("scale-seed-*.csv")) + sorted(args.input.glob("calibrated-scales-*.csv"))
    for path in paths:
        if path.name == "scale-seed-summary.csv":
            continue
        with path.open() as file:
            rows = list(csv.DictReader(file))
        if not rows:
            continue
        row = rows[0]
        baseline = float(row["scale_cal_baseline_heldout_rel_l2"])
        selected = float(row["heldout_rel_l2"])
        fixture = path.name.removesuffix(".csv").removesuffix("-cpu").removesuffix("-gpu")
        seed_records.append({"fixture": fixture, "device": row["device"],
                             "accepted": row["scale_cal_selected"], "baseline_rel_l2": baseline,
                             "selected_rel_l2": selected, "change": selected - baseline})
    if seed_records:
        with (args.input / "scale-seed-summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(seed_records[0]))
            writer.writeheader()
            writer.writerows(seed_records)
        # One backend per fixture; do not count repeated kernels or CPU/CUDA copies as samples.
        unique = [row for row in seed_records if row["device"] == "cpu"]
        improvements = sum(row["change"] < -1e-9 for row in unique)
        regressions = sum(row["change"] > 1e-9 for row in unique)
        fallbacks = sum(row["accepted"] == "0" for row in unique)
        lines += ["", f"Scale-fitting fixtures (each counted once): {len(unique)}; "
                  f"held-out improvements {improvements}, regressions {regressions}, validation fallbacks {fallbacks}.", "",
                  "The fixed seed list includes the reused development fixture and fresh robustness fixtures; "
                  "these are synthetic checks, not model-quality estimates."]
    profiles = []
    for path in sorted(args.input.glob("profile-*-cpu.csv")):
        with path.open() as file:
            row = next(csv.DictReader(file))
        baseline = float(row["scale_cal_baseline_heldout_rel_l2"])
        selected = float(row["heldout_rel_l2"])
        profiles.append((path.name, baseline, selected, row["scale_cal_selected"]))
    if profiles:
        lines += ["", "## Synthetic stress profiles", "",
                  "| Profile | Baseline held-out relative L2 | Selected | Accepted |",
                  "|---|---:|---:|---:|"]
        lines += [f"| {name} | {baseline:.6f} | {selected:.6f} | {accepted} |"
                  for name, baseline, selected, accepted in profiles]
        lines += ["", "Distribution shift can invalidate validation gains; held-out regressions are retained."]
    (args.input / "diagnostic-summary.md").write_text("\n".join(lines) + "\n")
    print("Diagnostic summary written; timings excluded")


if __name__ == "__main__":
    main()
