"""Benchmark warm GLM prefill on a fixed coding-text prefix."""
import argparse
import hashlib
import json
import pathlib
import re
import statistics
import subprocess
import tempfile
import time

from strata_tokenizer import Tokenizer
from glm_generate import metadata_shard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("text", type=pathlib.Path, help="UTF-8 coding prompt with enough tokens")
    parser.add_argument("--decoder", type=pathlib.Path, default=pathlib.Path("build-glm/strata-glm-decode"))
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--batch", choices=("2048", "4096", "auto"), default="4096")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--gpu-budget-mib", type=int, default=12288)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.tokens < 1 or args.context < args.tokens + 2 or args.repetitions < 1:
        parser.error("invalid token count, context or repetitions")
    tokenizer = Tokenizer.from_gguf(metadata_shard(args.model))
    ids = tokenizer.encode(args.text.read_text(), parse_special=True)
    if len(ids) < args.tokens:
        parser.error(f"prompt has {len(ids)} tokens; need {args.tokens}")
    ids = ids[:args.tokens]
    with tempfile.TemporaryDirectory(prefix="strata-glm-bench-") as temporary:
        token_file = pathlib.Path(temporary) / "tokens.ids"
        token_file.write_text(",".join(map(str, ids)))
        command = [str(args.decoder.resolve()), str(args.model.resolve()), f"@{token_file}", "2", "4096",
                   str(args.threads), f"--prefill-batch={args.batch}", f"--context={args.context}",
                   f"--gpu-budget-mib={args.gpu_budget_mib}", f"--bench={args.repetitions}", "--warm-weights"]
        start = time.monotonic()
        completed = subprocess.run(command, text=True, capture_output=True)
        elapsed = time.monotonic() - start
    if completed.returncode:
        raise SystemExit(completed.stderr)
    rows = re.findall(r"PREFILL trial=(\d+) tokens=(\d+) batch=(\d+) ms=([\d.]+) tok_s=([\d.]+)", completed.stderr)
    if len(rows) != args.repetitions:
        raise SystemExit("decoder did not report all measurements\n" + completed.stderr)
    peaks = re.findall(r"GPU peak_allocated_MiB=([\d.]+)", completed.stderr)
    result = {"model": str(args.model.resolve()), "text": str(args.text.resolve()), "tokens": args.tokens,
              "requested_batch": args.batch, "actual_batch": int(rows[0][2]), "context": args.context,
              "gpu_budget_mib": args.gpu_budget_mib, "warm_weights": True,
              "untimed_warmup": args.repetitions > 1,
              "milliseconds": [float(row[3]) for row in rows],
              "tokens_per_second": [float(row[4]) for row in rows],
              "median_tokens_per_second": statistics.median(float(row[4]) for row in rows),
              "peak_allocated_mib": max(map(float, peaks)) if peaks else None,
              "runtime_reserve_mib": 1024, "startup_warmup_and_runs_seconds": elapsed,
              "decode_token_ids": completed.stdout.split(), "prompt_token_sha256": hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest(),
              "page_faults": [{"trial": int(t), "major": int(a), "minor": int(b)}
                              for t, a, b in re.findall(r"FAULTS trial=(\d+) major=(\d+) minor=(\d+)", completed.stderr)],
              "decoder_log": completed.stderr}
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
