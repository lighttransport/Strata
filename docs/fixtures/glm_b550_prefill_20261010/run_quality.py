"""Run the pinned coding gate using one of this experiment's engine configs.

Launch under systemd-run --user --scope -p MemoryMax=60G -p MemorySwapMax=0.
The guard records memory and interference; generated Python runs inside bwrap.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--task-indices")
    parser.add_argument("--background", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    cfg = json.loads(args.config.read_text())
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
    env.update(cfg["env"])
    command = [sys.executable, str(root / "tools/humaneval_b550.py"), cfg["model"],
               "datasets/HumanEval.jsonl.gz", "--output", str(args.output.resolve()),
               "--decoder", cfg["exe"], "--limit", str(args.limit), "--offset", str(args.offset), "--tokens", "1024",
               "--gpu-budget-mib", str(cfg["gpu_budget_mib"]), "--decode-cache-mib", str(cfg["decode_cache_mib"]),
               "--expert-pack", cfg["expert_pack"], "--context", str(cfg["context"]),
               "--prefill-batch", str(cfg["prefill_batch"]), "--prefill-experts", cfg["prefill_experts"],
               "--threads", str(cfg["threads"])]
    if args.background:
        command += ["--background", str(args.background.resolve())]
    if args.task_indices:
        command += ["--task-indices", args.task_indices]
    guard = [sys.executable, str(root / "tools/glm_q2_run_guard.py"),
             "--output", str(args.output.resolve()) + ".guard", "--record-interference", "--", *command]
    raise SystemExit(subprocess.call(guard, cwd=root, env=env))


if __name__ == "__main__":
    main()
