"""Compare greedy GLM single-token and prompt-lookup decode on one coding prefix."""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import tempfile

from glm_generate import metadata_shard
from strata_tokenizer import Tokenizer


def measure(command):
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    result = {"token_ids": list(map(int, completed.stdout.split())), "decoder_log": completed.stderr}
    single = re.search(r"DECODE steps=(\d+) ms=([\d.]+) tok_s=([\d.]+)", completed.stderr)
    spec = re.search(r"SPECULATIVE source=lookup generated=(\d+) rounds=(\d+) proposed=(\d+) accepted=(\d+) replayed=(\d+) ms=([\d.]+) tok_s=([\d.]+)", completed.stderr)
    if single:
        result.update(decode_steps=int(single[1]), milliseconds=float(single[2]), tokens_per_second=float(single[3]))
    elif spec:
        result.update(generated=int(spec[1]), rounds=int(spec[2]), proposed=int(spec[3]), accepted=int(spec[4]),
                      replayed=int(spec[5]), milliseconds=float(spec[6]), tokens_per_second=float(spec[7]))
    else:
        raise RuntimeError("missing decode measurements\n" + completed.stderr)
    peaks = re.findall(r"GPU peak_allocated_MiB=([\d.]+)", completed.stderr)
    result["peak_allocated_mib"] = max(map(float, peaks)) if peaks else None
    result["step_milliseconds"] = [float(ms) for ms in re.findall(r"DECODE_STEP index=\d+ ms=([\d.]+)", completed.stderr)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("text", type=pathlib.Path)
    parser.add_argument("--decoder", type=pathlib.Path, default=pathlib.Path("build-glm/strata-glm-decode"))
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--generated-tokens", type=int, default=33)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--prefill-batch", choices=("2048", "4096", "auto"), default="4096")
    parser.add_argument("--decode-experts", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--lookup-depth", type=int, choices=range(1, 8), default=3)
    parser.add_argument("--gpu-budget-mib", type=int, default=12288)
    parser.add_argument("--check-verify", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.prompt_tokens < 1 or args.generated_tokens < 2 or args.context < args.prompt_tokens + max(8, args.generated_tokens):
        parser.error("invalid prompt/output/context lengths")
    ids = Tokenizer.from_gguf(metadata_shard(args.model)).encode(args.text.read_text(), parse_special=True)
    if len(ids) < args.prompt_tokens:
        parser.error("coding text has too few tokens")
    ids = ids[:args.prompt_tokens]
    with tempfile.TemporaryDirectory(prefix="strata-glm-decode-") as temporary:
        token_file = pathlib.Path(temporary) / "tokens.ids"
        encoded = ",".join(map(str, ids))
        token_file.write_text(encoded)
        command = [str(args.decoder.resolve()), str(args.model.resolve()), f"@{token_file}", str(args.generated_tokens),
                   "4096", str(args.threads), f"--prefill-batch={args.prefill_batch}", f"--context={args.context}",
                   f"--gpu-budget-mib={args.gpu_budget_mib}", f"--decode-experts={args.decode_experts}", "--warm-weights"]
        single = measure(command + (["--check-verify"] if args.check_verify else []))
        speculative = measure(command + [f"--lookup-depth={args.lookup_depth}"])
    if single["token_ids"] != speculative["token_ids"]:
        raise RuntimeError("speculative output differs from single-token greedy decode")
    result = {"model": str(args.model.resolve()), "prompt_tokens": len(ids), "generated_tokens": args.generated_tokens,
              "prompt_token_sha256": hashlib.sha256(encoded.encode()).hexdigest(), "context": args.context,
              "decode_experts": args.decode_experts, "threads": args.threads, "lookup_depth": args.lookup_depth,
              "gpu_budget_mib": args.gpu_budget_mib, "runtime_reserve_mib": 1024,
              "timing_scope": "excludes prefill and its first greedy token; includes verify, checkpoint and replay",
              "single": single, "speculative": speculative,
              "speedup": speculative["tokens_per_second"] / single["tokens_per_second"], "greedy_ids_identical": True}
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
