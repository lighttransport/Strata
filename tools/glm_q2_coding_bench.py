"""Q2 coding decode trials, with EOS-aware counts and optional in-process mode sweep."""
import argparse
import hashlib
import json
import os
import pathlib
import re
import statistics
import subprocess
import threading
import time

from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer
from jinja2.sandbox import SandboxedEnvironment

FIXTURES = {
    "prime": "Write a complete C++17 bool is_prime(unsigned n) function. Avoid overflow, handle 0 and 1 correctly, and return only one cpp code block with no explanation. Keep reasoning very brief.",
    "json_escape": "Write a complete C++17 std::string json_escape(std::string_view s) function. Return the escaped contents without surrounding quotes. Escape double quotes, backslashes and every byte below 0x20 using JSON escapes; preserve other bytes. Include needed headers. Return only one cpp code block, no main or explanation. Keep reasoning very brief.",
    "csv": "Write a complete C++17 bool parse_csv(std::string_view s, std::vector<std::string>& out) function for one CSV record. Support empty fields, quoted commas, doubled quotes and trailing empty fields. Reject unclosed quotes, quotes inside unquoted fields and text after a closing quote except comma. Leave out unchanged on failure. Include headers. Return only one cpp code block, no main or explanation. Keep reasoning very brief.",
}


def parse_trials(log, ids, stops, source="none", depth=0):
    cursor = 0
    groups = {}
    for section in re.split(r"(?=DECODE_MODE source=|DECODE_TRIAL index=)", log):
        mode = re.match(r"DECODE_MODE source=(\w+) depth=(\d+)", section)
        if mode:
            source, depth = mode[1], int(mode[2])
        if not section.startswith("DECODE_TRIAL"):
            continue
        single = re.search(r"DECODE steps=(\d+) ms=([\d.]+) tok_s=([\d.]+)", section)
        spec = re.search(r"SPECULATIVE source=(\w+) generated=(\d+) rounds=(\d+) proposed=(\d+) accepted=(\d+) replayed=(\d+) ms=([\d.]+) tok_s=([\d.]+)", section)
        if spec:
            source = spec[1]
            count, ms, rate = int(spec[2]), float(spec[7]), float(spec[8])
            row = dict(rounds=int(spec[3]), proposed=int(spec[4]), accepted=int(spec[5]), replayed=int(spec[6]))
        elif single:
            count, ms, rate = int(single[1]) + 1, float(single[2]), float(single[3])
            row = {}
        else:
            raise ValueError("missing trial timing")
        tokens = ids[cursor:cursor + count]
        if len(tokens) != count:
            raise ValueError("incomplete generated token stream")
        cursor += count
        row.update(generated=count, milliseconds=ms, tokens_per_second=rate,
                   ended_on_stop=tokens[-1] in stops, token_ids=tokens)
        expert = re.search(r"CPU_EXPERT gu_ms=([\d.]+) quant_ms=([\d.]+) down_ms=([\d.]+) bytes=(\d+)", section)
        if expert:
            row["cpu_expert_milliseconds"] = dict(zip(("gate_up", "quantize", "down"), map(float, expert.groups()[:3])))
            row["cpu_expert_bytes"] = int(expert[4])
        timing = re.search(r"MTP_TIMING draft_ms=([\d.]+) verify_ms=([\d.]+) resync_ms=([\d.]+)", section)
        if timing:
            row["mtp_milliseconds"] = dict(zip(("draft", "verify", "resync"), map(float, timing.groups())))
        gpu = re.search(r"GPU peak_allocated_MiB=([\d.]+)", section)
        if gpu:
            row["gpu_peak_allocated_mib"] = float(gpu[1])
        groups.setdefault(f"{source}:{depth}", []).append(row)
    if not groups or cursor != len(ids):
        raise ValueError("unaccounted generated tokens")
    reference = next(iter(groups.values()))[0]["token_ids"]
    for trials in groups.values():
        if any(t["token_ids"] != reference for t in trials):
            raise ValueError("greedy output differs across trials/modes")
    return groups


def process_cpu_snapshot():
    """Linux process CPU counters; never read command arguments."""
    rows = {}
    for stat in pathlib.Path("/proc").glob("[0-9]*/stat"):
        try:
            value = stat.read_text()
            end = value.rindex(")")
            fields = value[end + 2:].split()
            if fields[0] == "Z":
                continue
            rows[int(stat.parent.name)] = (value[value.index("(") + 1:end], int(fields[11]) + int(fields[12]))
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return time.monotonic(), rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=pathlib.Path)
    ap.add_argument("--decoder", default="build-glm/strata-glm-decode")
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--fixtures", nargs="+", choices=FIXTURES, default=list(FIXTURES))
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--reject-compilers", "--reject-competitors", action="store_true", help="stop and discard on concurrent compilers or sustained external CPU work")
    ap.add_argument("--no-warm-weights", action="store_true", help="rely on GPU prefill to visit main expert weights")
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--expert-pack", type=pathlib.Path)
    ap.add_argument("--expert-pack-profile", type=pathlib.Path)
    ap.add_argument("--cpu-expert-backend", choices=("auto", "native", "packed-dot", "packed-lut"), default="auto")
    ap.add_argument("--capture-experts", action="store_true")
    ap.add_argument("--max-swap-mib", type=int, default=0, help="tolerated process swap; 0 rejects any swap")
    ap.add_argument("--decoder-flag", action="append", default=[], help="extra decoder flag, repeatable (e.g. --decoder-flag=--decode-cache-mib=3584)")
    sweeps=ap.add_mutually_exclusive_group()
    sweeps.add_argument("--sweep", action="store_true", help="none, lookup 1..3, MTP 1..3; all retain the MTP allocation")
    sweeps.add_argument("--mtp-sweep", action="store_true", help="none and MTP 1..3; omit lookup modes")
    ap.add_argument("--speculative", choices=("none", "mtp", "lookup"), default="none")
    ap.add_argument("--depth", type=int, choices=(1, 2, 3), default=1)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = args.model.resolve()
    metadata = GGUFFile(model).metadata
    tokenizer = Tokenizer.from_gguf(model)
    stops = {int(metadata[k]) for k in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id") if k in metadata}
    env = SandboxedEnvironment(extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = lambda s: (_ for _ in ()).throw(ValueError(s))
    results = {}
    for fixture in args.fixtures:
        chat = env.from_string(metadata["tokenizer.chat_template"]).render(
            messages=[dict(role="user", content=FIXTURES[fixture])], tools=None,
            add_generation_prompt=True, reasoning_effort="low")
        prompt = tokenizer.encode(chat, parse_special=True)
        prefix = args.output / fixture
        prefix.with_suffix(".prompt.txt").write_text(chat)
        prefix.with_suffix(".ids").write_text(",".join(map(str, prompt)))
        command = [args.decoder, str(model), "@" + str(prefix.with_suffix(".ids").resolve()), str(args.tokens), "4096", "15",
                   "--context=8192", "--prefill-batch=256", "--gpu-budget-mib=12288", "--cpu-affinity=numa",
                   "--decode-graphs", f"--decode-bench={args.repetitions}",
                   "--stop-ids=" + ",".join(map(str, sorted(stops)))]
        if not args.no_warm_weights:
            command.append("--warm-weights")
        if args.expert_pack:
            command += [f"--expert-pack={args.expert_pack.resolve()}", f"--cpu-expert-backend={args.cpu_expert_backend}"]
        if args.expert_pack_profile:
            command.append(f"--expert-pack-profile={args.expert_pack_profile.resolve()}")
        if args.sweep or args.mtp_sweep:
            command.append("--decode-mode-sweep=mtp" if args.mtp_sweep else "--decode-mode-sweep")
        elif args.speculative == "mtp":
            command += ["--speculative=mtp", f"--draft-depth={args.depth}"]
        elif args.speculative == "lookup":
            command.append(f"--lookup-depth={args.depth}")
        command += args.decoder_flag
        memory = dict(peak_rss_kib=0, peak_swap_kib=0)
        with prefix.with_suffix(".tokens").open("w") as out, prefix.with_suffix(".log").open("w") as err:
            child_env = os.environ.copy()
            if args.capture_experts:
                child_env["STRATA_GLM_ACTIVATION_TRACE"] = str(prefix.with_suffix(".capture"))
            process = subprocess.Popen(command, stdout=out, stderr=err, env=child_env)
            def monitor():
                sample = 0
                previous_cpu = process_cpu_snapshot() if args.reject_compilers else None
                busy_samples = {}
                while process.poll() is None:
                    try:
                        status = pathlib.Path(f"/proc/{process.pid}/status").read_text()
                        for field, key in (("VmRSS", "peak_rss_kib"), ("VmSwap", "peak_swap_kib")):
                            match = re.search(rf"^{field}:\s+(\d+)", status, re.M)
                            if match:
                                memory[key] = max(memory[key], int(match[1]))
                    except FileNotFoundError:
                        pass
                    available = next(int(line.split()[1]) for line in pathlib.Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
                    memory["minimum_available_kib"] = min(memory.get("minimum_available_kib", available), available)
                    if memory["peak_rss_kib"] > 118 * 1024**2 or available < 4 * 1024**2:
                        memory["memory_limit"] = True
                    if args.reject_compilers and sample % 10 == 0:
                        sampled_at, current_cpu = process_cpu_snapshot()
                        previous_at, previous_rows = previous_cpu
                        elapsed = sampled_at - previous_at
                        tick_hz = os.sysconf("SC_CLK_TCK")
                        competitors = []
                        for pid, (name, ticks) in current_cpu.items():
                            if pid in {process.pid, os.getpid()}:
                                continue
                            old = previous_rows.get(pid)
                            percent = 100 * (ticks - old[1]) / (elapsed * tick_hz) if old and elapsed else 0
                            busy_samples[pid] = busy_samples.get(pid, 0) + 1 if percent >= 80 else 0
                            if name in {"cc1plus", "cc1", "ptxas", "nvcc", "lto1"} or busy_samples[pid] >= 2:
                                competitors.append(dict(pid=pid, name=name, cpu_percent=round(percent, 1)))
                        previous_cpu = sampled_at, current_cpu
                        if competitors:
                            memory["competing_workloads"] = competitors
                    sample += 1
                    if memory["peak_swap_kib"] > args.max_swap_mib * 1024 or memory.get("competing_workloads") or memory.get("memory_limit"):
                        try:
                            process.terminate()
                        except ProcessLookupError:
                            pass
                        return
                    time.sleep(.1)
            thread = threading.Thread(target=monitor)
            thread.start()
            code = process.wait()
            thread.join()
        prefix.with_suffix(".memory.json").write_text(json.dumps(memory, indent=2) + "\n")
        if memory.get("memory_limit"):
            raise RuntimeError(f"{fixture} exceeded the RAM limit; benchmark rejected")
        if memory.get("competing_workloads"):
            raise RuntimeError(f"{fixture} had concurrent CPU work; stopped and rejected this benchmark: {memory['competing_workloads']}")
        if memory["peak_swap_kib"] > args.max_swap_mib * 1024:
            raise RuntimeError(f"{fixture} swapped {memory['peak_swap_kib']} KiB; stopped and rejected this benchmark")
        if code:
            raise RuntimeError(f"{fixture} exited {code}; see {prefix.with_suffix('.log')}")
        ids = list(map(int, prefix.with_suffix(".tokens").read_text().split()))
        groups = parse_trials(prefix.with_suffix(".log").read_text(), ids, stops, args.speculative, args.depth if args.speculative!="none" else 0)
        output = next(iter(groups.values()))[0]["token_ids"]
        prefix.with_suffix(".output.md").write_text(tokenizer.decode([t for t in output if t not in stops]))
        for trials in groups.values():
            for trial in trials:
                del trial["token_ids"]
        results[fixture] = dict(input_tokens=len(prompt), command=command, memory=memory, trials=groups,
            median_tokens_per_second={k: statistics.median(t["tokens_per_second"] for t in v) for k, v in groups.items()},
            greedy_ids_identical=True, token_sha256=hashlib.sha256(json.dumps(output).encode()).hexdigest())
        (args.output / "measurement.json").write_text(json.dumps(dict(environment={k: v for k, v in os.environ.items() if k.startswith("STRATA_")}, results=results), indent=2) + "\n")
        print(fixture, results[fixture]["median_tokens_per_second"], memory, flush=True)


if __name__ == "__main__":
    main()
