#!/usr/bin/env python3
"""Resume a frozen full-model V100 experiment matrix, preserving failed attempts.

Each case runs the resident benchmark. Speed eligibility is separate from
quality qualification. A matrix contains `cases` with name, prompt, topology
(layer/legacy/worker), optional build (baseline/candidate), env and arguments.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

from gguf_reader import GGUFFile
from glm_mixed_bench import stream_health
from glm_tr16_calibrate import wait_idle
from glm_v100_bench import prompt_ids, save_result

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def remaining_seconds():
    if not os.environ.get("SLURM_JOB_ID"):
        return float("inf")
    output = subprocess.check_output(
        ["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"], text=True
    )
    end = re.search(r"\bEndTime=(\S+)", output)
    if not end:
        raise RuntimeError("cannot determine Slurm allocation end time")
    import datetime

    return datetime.datetime.fromisoformat(end[1]).timestamp() - time.time()


def process_memory(pid):
    """Cheap status reads only; PSS/page-table walks stay outside timings."""
    todo = [pid]
    seen = set()
    swap = rss = 0
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            text = Path(f"/proc/{current}/status").read_text()
            for key, destination in (("VmSwap", "swap"), ("VmRSS", "rss")):
                match = re.search(r"^" + key + r":\s+(\d+)\s+kB$", text, re.M)
                if match:
                    if destination == "swap":
                        swap += int(match[1]) * 1024
                    else:
                        rss += int(match[1]) * 1024
            todo += [
                int(x)
                for x in Path(f"/proc/{current}/task/{current}/children")
                .read_text()
                .split()
            ]
        except (FileNotFoundError, ProcessLookupError):
            continue
    return dict(swap_bytes=swap, summed_rss_bytes=rss, observed_processes=len(seen))


def assess(record, stops, expected_trials=3, tokens=512):
    trials = record.get("trials", [])
    validation = record.get("validation_trials", [])
    warmups = record.get("full_warmup_trials", [])
    health = [
        dict(
            stream_health(row["token_ids"]),
            first_stop=next(
                (i for i, token in enumerate(row["token_ids"]) if token in stops), None
            ),
        )
        for row in trials + validation + warmups
    ]
    eligible = bool(
        record.get("complete")
        and not record.get("instrumented")
        and record.get("repetitions_identical")
        and len(trials) >= expected_trials
        and record["configuration"]["prompt_tokens"] == 8192
        and tokens >= 512
        and len(validation) >= 3
        and len(warmups) >= 2
        and all(
            len(row["token_ids"]) == tokens
            and math.isfinite(row["decode_tok_s"])
            and row["decode_tok_s"] > 0
            for row in trials + validation + warmups
        )
        and all(
            h["consecutive_cycle"] is None and h["first_stop"] in (None, tokens - 1)
            for h in health
        )
    )
    speed = statistics.median(row["decode_tok_s"] for row in trials) if trials else None
    return dict(
        timing_eligible=eligible,
        median_decode_tok_s=speed,
        stream_health=health,
        speed_target_reached=bool(eligible and speed >= 60),
        quality_qualified=False,
        note="Corpus, complete paired HumanEval and long-context qualification are separate.",
    )


def command(case, args, output):
    topology = case.get("topology", "layer")
    if topology not in ("layer", "legacy", "worker"):
        raise ValueError("topology must be layer, legacy or worker")
    binary = args.baseline if case.get("build") == "baseline" else args.decoder
    prompt = Path(case["prompt"]).expanduser().resolve()
    if len(prompt_ids(prompt)) != 8192:
        raise ValueError(
            "each primary timing prompt must contain exactly 8192 frozen IDs"
        )
    cmd = [
        sys.executable,
        str(ROOT / "tools/glm_v100_bench.py"),
        str(args.model),
        str(prompt),
        "--decoder",
        str(binary),
        "--prompt-ids",
        "--prompt-tokens",
        "8192",
        "--generated-tokens",
        "512",
        "--context",
        "32768",
        "--threads",
        str(case.get("threads", 32)),
        "--gpu-budget-mib",
        "32768",
        "--decode-graphs",
        "--full-warmups",
        "1" if args.screen else "2",
        "--repetitions",
        "3",
        "--validation-tokens",
        "0" if args.screen else "512",
        "--validation-repetitions",
        "3",
        "--decode-cache-mib",
        str(case.get("cache_mib", 22528)),
        "--speculative",
        "mtp",
        "--draft-depth",
        str(case.get("depth", 1 if topology == "layer" else 3)),
        "--output",
        str(output),
    ]
    if topology == "layer":
        cmd += [
            "--gpu-layout",
            "layer",
            "--gpu-devices",
            case.get("gpu_devices", "0,1"),
            "--layer-split",
            str(case.get("cut", 23)),
            "--layer-pipeline",
            case.get("pipeline", "mtp"),
            "--layer-memory",
            case.get("memory", "socket"),
            "--prefill-expert-cache-mib",
            "0",
            "--prefill-batch",
            str(case.get("prefill_batch", 4096)),
        ]
    else:
        cmd += [
            "--gpu-devices",
            "0" if topology == "worker" else case.get("gpu_devices", "0"),
            "--prefill-experts",
            case.get("prefill_experts", "f16-batched"),
            "--prefill-expert-cache-mib",
            "0",
            "--prefill-batch",
            str(case.get("prefill_batch", 1024)),
        ]
    if topology == "worker":
        if not args.worker:
            raise ValueError("worker topology needs --worker")
        cmd += [
            "--worker-decoder",
            str(args.worker),
            "--primary-device",
            str(case.get("primary_device", 0)),
            "--worker-device",
            str(case.get("worker_device", 1)),
            "--worker-cache-mib",
            str(case.get("worker_cache_mib", 24576)),
            "--worker-combine",
            case.get("combine", "strict"),
        ]
        for role in ("primary", "worker"):
            if case.get(role + "_cpus"):
                cmd += ["--" + role + "-cpus", case[role + "_cpus"]]
    extra = case.get("arguments", [])
    if not isinstance(extra, list) or any(not isinstance(x, str) for x in extra):
        raise ValueError("case arguments must be a string list")
    controlled = (
        "--output",
        "--decoder",
        "--worker-decoder",
        "--prompt-tokens",
        "--generated-tokens",
        "--repetitions",
        "--full-warmups",
        "--validation-tokens",
        "--validation-repetitions",
    )
    if any(x.split("=", 1)[0] in controlled for x in extra):
        raise ValueError(
            "case arguments cannot replace evidence paths, binaries or the timing protocol"
        )
    if topology != "layer" and case.get("memory", "interleave") == "interleave":
        cmd = ["numactl", "--interleave=all", *cmd]
    return cmd + extra


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--decoder", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--worker", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--minimum-remaining", type=int, default=1200)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument(
        "--screen",
        action="store_true",
        help="one full warmup and three timed requests; cannot qualify the final target",
    )
    a = p.parse_args()
    for key in ("matrix", "model", "decoder", "baseline", "worker", "output"):
        if getattr(a, key) is not None:
            setattr(a, key, getattr(a, key).expanduser().resolve())
    a.baseline = a.baseline or a.decoder
    matrix = json.loads(a.matrix.read_text())
    cases = matrix["cases"]
    names = [case["name"] for case in cases]
    if len(names) != len(set(names)) or any(
        not re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in names
    ):
        p.error("case names must be unique plain filenames")
    a.output.mkdir(parents=True, exist_ok=True)
    identity = dict(
        matrix_sha256=digest(a.matrix),
        decoder_sha256=digest(a.decoder),
        screening=a.screen,
        baseline_sha256=digest(a.baseline),
        worker_sha256=digest(a.worker) if a.worker else None,
        model=str(a.model),
        model_bytes=a.model.stat().st_size,
        model_mtime_ns=a.model.stat().st_mtime_ns,
        harness_sha256={
            name: digest(ROOT / "tools" / name)
            for name in (
                "glm_v100_optimize.py",
                "glm_v100_bench.py",
                "glm_mixed_run.py",
            )
        },
    )
    frozen = a.output / "identity.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        p.error("resume identity differs; use a new output directory")
    save_result(frozen, identity)
    (a.output / "matrix.json").write_bytes(a.matrix.read_bytes())
    metadata = GGUFFile(a.model).metadata
    stops = {
        int(metadata[k])
        for k in (
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.eot_token_id",
            "tokenizer.ggml.eom_token_id",
        )
        if k in metadata
    }
    summary = dict(identity=identity, stop_ids=sorted(stops), cases=[], complete=False)
    failed = False
    for case in cases:
        directory = a.output / case["name"]
        directory.mkdir(exist_ok=True)
        attempts = sorted(directory.glob("attempt-*"))
        previous = attempts[-1] / "record.json" if attempts else None
        if previous and previous.exists():
            row = json.loads(previous.read_text())
            if row.get("returncode") == 0 or not a.retry_failed:
                summary["cases"].append(row)
                failed |= row.get("returncode") != 0
                save_result(a.output / "summary.json", summary)
                continue
        if remaining_seconds() < a.minimum_remaining:
            save_result(a.output / "summary.json", summary)
            print("ALLOCATION_REMAINDER resume required", flush=True)
            return 75
        attempt = directory / f"attempt-{len(attempts) + 1:03d}"
        attempt.mkdir()
        output = attempt / "benchmark.json"
        cmd = command(case, a, output)
        env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
        env.update(case.get("env", {}))
        record = dict(
            name=case["name"],
            job_id=os.environ.get("SLURM_JOB_ID"),
            node=os.uname().nodename,
            screening=a.screen,
            command=cmd,
            environment={k: v for k, v in env.items() if k.startswith("STRATA_")},
            returncode=None,
            benchmark=str(output),
            complete=False,
            peak_swap_bytes=0,
            peak_summed_rss_bytes=0,
        )
        save_result(attempt / "record.json", record)
        print("CASE_START", case["name"], flush=True)
        wait_idle(timeout=600)
        start = time.monotonic()
        with (attempt / "runner.log").open("w") as log:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                while True:
                    sample = process_memory(process.pid)
                    record["peak_swap_bytes"] = max(
                        record["peak_swap_bytes"], sample["swap_bytes"]
                    )
                    record["peak_summed_rss_bytes"] = max(
                        record["peak_summed_rss_bytes"], sample["summed_rss_bytes"]
                    )
                    if sample["swap_bytes"]:
                        raise RuntimeError("experiment swapped; timing rejected")
                    if time.monotonic() - start > 5400:
                        raise TimeoutError("experiment exceeded its time limit")
                    try:
                        record["returncode"] = process.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
            finally:
                record["elapsed_seconds"] = time.monotonic() - start
                if output.exists():
                    record.update(assess(json.loads(output.read_text()), stops))
                if record["peak_swap_bytes"]:
                    record.update(timing_eligible=False, speed_target_reached=False)
                record["complete"] = record["returncode"] == 0
                save_result(attempt / "record.json", record)
        summary["cases"].append(record)
        failed |= record["returncode"] != 0
        save_result(a.output / "summary.json", summary)
        print(
            "CASE_DONE",
            case["name"],
            record["returncode"],
            record.get("median_decode_tok_s"),
            flush=True,
        )
    summary["complete"] = True
    save_result(a.output / "summary.json", summary)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
