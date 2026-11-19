#!/usr/bin/env python3
"""Hy4-Flash decode forecast on TR16 (RX 9070 XT). Model: tools/sim/hy4f.py.

    python3 tools/hy4f_sim.py table                          # profile x tier-hit model x miss split x MTP
    python3 tools/hy4f_sim.py estimate --profile now --hit-model fixed --hit 0.5 --split --mtp coding
    python3 tools/hy4f_sim.py estimate --set cpu_lane_gbps=40 --tier-gib 6
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "sim"))
import hy4f  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("table")
    e = sub.add_parser("estimate")
    e.add_argument("--profile", default="now", choices=sorted(hy4f.PROFILES))
    e.add_argument("--hit-model", default="ds41f", choices=("ds41f", "glm", "fixed"))
    e.add_argument("--hit", type=float, default=0.0)
    e.add_argument("--tier-gib", type=float, default=-1.0)
    e.add_argument("--split", action="store_true")
    e.add_argument("--mtp", default="")
    e.add_argument("--context", type=int, default=4096)
    e.add_argument("--set", action="append", default=[], help="Hw field=value")
    a = ap.parse_args()
    if a.cmd == "table":
        print("| profile | tier hit model | misses | MTP | hit | tok/s | ms/round | trunk | experts | draft |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in hy4f.table():
            print(f"| {r['profile']} | {r['hit_model']} | {'CPU+PCIe' if r['split'] else 'CPU'} | {r['mtp']} | {r['hit']:.2f} | "
                  f"{r['tok_s']:.1f} | {r['round_ms']:.1f} | {r['trunk_ms']:.1f} | {r['experts_ms']:.1f} | {r['draft_ms']:.1f} |")
        print("\nHypotheses:\n" + "\n".join("- " + h for h in hy4f.HYPOTHESES))
        return
    p = hy4f.Plan(profile=a.profile, hit_model=a.hit_model, hit=a.hit, tier_gib=a.tier_gib, split_misses=a.split, mtp=a.mtp,
                  context=a.context, set=dict(s.split("=", 1) for s in a.set))
    print(json.dumps(hy4f.decode(p), indent=1))


if __name__ == "__main__":
    main()
