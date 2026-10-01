"""Compare greedy GLM single-token and speculative decode on one coding prefix."""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import tempfile
import statistics
import sys
import os

from glm_generate import metadata_shard
from strata_tokenizer import Tokenizer


def measure(command, environment=None):
    completed = subprocess.run(command, capture_output=True, text=True, env=environment)
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    ids = list(map(int, completed.stdout.split()))
    count = int(command[3])
    trials = []
    for section in re.split(r"DECODE_TRIAL index=\d+\n", completed.stderr)[1:]:
        single = re.search(r"DECODE steps=(\d+) ms=([\d.]+) tok_s=([\d.]+)", section)
        spec = re.search(r"SPECULATIVE source=(lookup|mtp) generated=(\d+) rounds=(\d+) proposed=(\d+) accepted=(\d+) replayed=(\d+) ms=([\d.]+) tok_s=([\d.]+)", section)
        if single:
            trial = dict(decode_steps=int(single[1]), milliseconds=float(single[2]), tokens_per_second=float(single[3]))
        elif spec:
            trial = dict(source=spec[1], generated=int(spec[2]), rounds=int(spec[3]), proposed=int(spec[4]),
                         accepted=int(spec[5]), replayed=int(spec[6]), milliseconds=float(spec[7]), tokens_per_second=float(spec[8]))
        else:
            raise RuntimeError("missing decode measurements")
        trial["step_milliseconds"] = [float(ms) for ms in re.findall(r"DECODE_STEP index=\d+ ms=([\d.]+)", section)]
        mtp_timing = re.search(r"MTP_TIMING draft_ms=([\d.]+) verify_ms=([\d.]+) resync_ms=([\d.]+)", section)
        if mtp_timing:
            trial["mtp_milliseconds"] = dict(zip(("draft", "verify", "resync"), map(float, mtp_timing.groups())))
        expert_timing = re.search(r"CPU_EXPERT gu_ms=([\d.]+) quant_ms=([\d.]+) down_ms=([\d.]+) bytes=(\d+)", section)
        if expert_timing:
            trial["cpu_expert_milliseconds"] = dict(zip(("gate_up", "quantize", "down"), map(float, expert_timing.groups()[:3])))
            trial["cpu_expert_bytes"] = int(expert_timing[4])
        trials.append(trial)
        local_timing = re.search(r"NATIVE_NUMA_TIMING prepare_ms=([\d.]+) queries=(\d+)", section)
        if local_timing:
            trial["native_numa_prepare_milliseconds"] = float(local_timing[1])
            trial["native_numa_queries"] = int(local_timing[2])
    if not trials or len(ids) != count * len(trials):
        raise RuntimeError("unexpected benchmark token count")
    outputs = [ids[i:i + count] for i in range(0, len(ids), count)]
    if any(row != outputs[0] for row in outputs):
        raise RuntimeError("greedy output changed across repetitions")
    peaks = re.findall(r"GPU peak_allocated_MiB=([\d.]+)", completed.stderr)
    cache = re.search(r"DECODE_CACHE prefix_trained=1 slots=(\d+) MiB=([\d.]+)(?: fingerprint=(\d+))?", completed.stderr)
    return dict(token_ids=outputs[0], trials=trials, decoder_log=completed.stderr,
                actual_decode_cache=dict(slots=int(cache[1]), allocated_mib=float(cache[2]), fingerprint=cache[3]) if cache else None,
                tokens_per_second=statistics.median(t["tokens_per_second"] for t in trials),
                milliseconds=statistics.median(t["milliseconds"] for t in trials),
                peak_allocated_mib=max(map(float, peaks)) if peaks else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("text", type=pathlib.Path)
    parser.add_argument("--decoder", type=pathlib.Path, default=pathlib.Path("build-glm/strata-glm-decode"))
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--generated-tokens", type=int, default=256)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--prefill-batch", choices=("2048", "4096", "auto"), default="4096")
    parser.add_argument("--decode-experts", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--lookup-depth", type=int, choices=range(1, 8), default=3)
    parser.add_argument("--gpu-budget-mib", type=int, default=12288)
    parser.add_argument("--check-verify", action="store_true")
    parser.add_argument("--speculative", choices=("lookup", "mtp"), default="mtp")
    parser.add_argument("--mtp-experts", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--draft-depth", type=int, choices=range(1, 8), default=3)
    parser.add_argument("--draft-depths", help="comma-separated MTP depths to compare using one single-decode baseline")
    parser.add_argument("--repetitions", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--cpu-prepack-mib", type=int, default=0)
    parser.add_argument("--cpu-affinity", choices=("none", "auto"), default="none")
    parser.add_argument("--pool-spin-us", type=int, help="expert worker spin duration before sleep; default is inherited")
    parser.add_argument("--single-only", action="store_true", help="measure single decode without a speculative comparison")
    parser.add_argument("--decode-cache-mib", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.single_only and args.draft_depths is not None:
        parser.error("single-only cannot be combined with draft-depths")
    if args.decode_cache_mib not in range(4097) or (args.decode_cache_mib and
            (args.decode_experts != "cpu" or args.cpu_prepack_mib or
             (not args.single_only and (args.speculative != "mtp" or args.mtp_experts != "cpu")))):
        parser.error("decode-cache-mib requires CPU single or CPU-draft MTP, no prepacking, and 0..4096 MiB")
    environment = None
    if args.mtp_experts == "cpu" and (args.single_only or args.speculative != "mtp"):
        parser.error("CPU MTP experts require MTP speculative comparison")
    if args.pool_spin_us is not None:
        if args.pool_spin_us < 0:
            parser.error("pool-spin-us must be nonnegative")
        environment = os.environ.copy()
        environment["STRATA_POOL_SPIN_US"] = str(args.pool_spin_us)
    depths = [args.draft_depth]
    if args.draft_depths is not None:
        try:
            depths = list(dict.fromkeys(int(n) for n in args.draft_depths.split(",")))
        except ValueError:
            parser.error("draft depths must be comma-separated integers")
        if args.speculative != "mtp" or not depths or any(n not in range(1, 8) for n in depths):
            parser.error("draft-depths requires MTP and depths between 1 and 7")
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
                   f"--gpu-budget-mib={args.gpu_budget_mib}", f"--decode-experts={args.decode_experts}",
                   f"--decode-bench={args.repetitions}", f"--cpu-affinity={args.cpu_affinity}", f"--cpu-prepack-mib={args.cpu_prepack_mib}", "--warm-weights"]
        if args.decode_cache_mib:
            command.append(f"--decode-cache-mib={args.decode_cache_mib}")
        single = measure(command + (["--check-verify"] if args.check_verify else []), environment)
        print(f"BENCH_PROGRESS source=single median_tok_s={single['tokens_per_second']}", file=sys.stderr, flush=True)
        sweep = []
        for depth in ([] if args.single_only else depths):
            draft_flags = (["--speculative=mtp", f"--draft-depth={depth}", f"--mtp-experts={args.mtp_experts}"] if args.speculative == "mtp"
                           else [f"--lookup-depth={args.lookup_depth}"])
            if args.decode_cache_mib:
                baseline_cache = single.get("actual_decode_cache")
                if not baseline_cache or not baseline_cache.get("fingerprint") or not baseline_cache["slots"]:
                    raise RuntimeError("missing reproducible resident expert cache")
                draft_flags.append(f"--decode-cache-slots={baseline_cache['slots']}")
            measured = measure(command + draft_flags, environment)
            if args.decode_cache_mib and measured.get("actual_decode_cache") != baseline_cache:
                raise RuntimeError("resident expert cache differs between single and speculative decode")
            if single["token_ids"] != measured["token_ids"]:
                raise RuntimeError(f"speculative output differs from single-token greedy decode at depth {depth}")
            sweep.append({"draft_depth": depth, "measurement": measured})
            print(f"BENCH_PROGRESS source={args.speculative} requested_depth={depth} "
                  f"median_tok_s={measured['tokens_per_second']} greedy_ids_identical=1",
                  file=sys.stderr, flush=True)
        best = max(sweep, key=lambda entry: entry["measurement"]["tokens_per_second"]) if sweep else None
        speculative = best["measurement"] if best else None
    result = {"model": args.model.name, "prompt_tokens": len(ids), "generated_tokens": args.generated_tokens,
              "prompt_token_sha256": hashlib.sha256(encoded.encode()).hexdigest(), "context": args.context,
              "decode_experts": args.decode_experts, "threads": args.threads, "lookup_depth": args.lookup_depth,
              "speculative_source": args.speculative if best else None, "draft_depth": best["draft_depth"] if best else None,
              "decode_cache_mib": args.decode_cache_mib,
              "native_numa_local_requested": os.environ.get("STRATA_NATIVE_NUMA_LOCAL", "0") == "1",
              "mtp_experts": args.mtp_experts if best and args.speculative == "mtp" else None,
              "cpu_prepack_mib": args.cpu_prepack_mib, "repetitions": args.repetitions, "cpu_affinity": args.cpu_affinity, "gpu_budget_mib": args.gpu_budget_mib, "runtime_reserve_mib": 1024,
              "timing_scope": "excludes prefill, MTP priming, expert-cache preparation, state-buffer allocation and first greedy token; includes drafting, verify, checkpoint copies and restore/replay",
              "single": single, "speculative": speculative,
              "speedup": speculative["tokens_per_second"] / single["tokens_per_second"] if best else None,
              "greedy_ids_identical": True,
              "greedy_comparison_scope": "between modes and repetitions" if best else "between repetitions only"}
    if args.draft_depths is not None:
        result["mtp_depth_sweep"] = sweep
    result["cpu_pool_spin_us"] = (args.pool_spin_us if args.pool_spin_us is not None
                                   else os.environ.get("STRATA_POOL_SPIN_US", "default:20000"))
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
