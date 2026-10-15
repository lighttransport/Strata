#!/usr/bin/env python3
"""Score a frozen V100 configuration against a full-Q23 reference.

Uses layer target rows or the legacy decoder's real verification path. Draft
acceptance and generated answers require separate HumanEval/long-context gates.
Raw logits stay in the experiment directory and are never repository fixtures.
"""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from glm_regression import parse_eval, check_quality
from glm_v100_bench import save_result
from glm_v100_optimize import process_memory, digest
from glm_tr16_calibrate import wait_idle

TOLERANCE = dict(ppl_rel=0.03, kl_rel=0, kl_abs=0.03, top1_abs=0.01)


def evaluation_command(record, corpus, reference, save=False, prefill=False):
    c = record["configuration"]
    layer = c.get("gpu_layout") == "layer"
    width = 1 if layer else min(int(c.get("draft_depth", 1)) + 1, 8)
    cmd = [
        str(Path(c["decoder"]).resolve()),
        c["model"],
        "0",
        "1",
        "4096",
        str(c["threads"]),
        "--context=" + str(c["context"]),
        "--prefill-batch=" + str(c["prefill_batch"]),
        "--gpu-budget-mib=" + str(c["gpu_budget_mib"]),
        "--gpu-devices=" + c["gpu_devices"],
        "--cpu-affinity=" + c.get("cpu_affinity", "auto"),
        "--lock-weights",
        "--decode-graphs",
        "--decode-cache-mib=" + ("0" if prefill else str(c["decode_cache_mib"])),
        "--prefill-expert-cache-mib=0",
        "--prefill-experts=" + c.get("prefill_experts", "mmq"),
        "--eval-corpus=" + str(corpus),
        ("--eval-save-logits=" if save else "--eval-reference=") + str(reference),
    ]
    if layer:
        cut = c.get("layer_split", "auto")
        if cut == "auto":
            cut = json.loads(Path(c["layer_plan"]).read_text())["layer_split"]
        cmd += [
            "--gpu-layout=layer",
            "--layer-split=" + str(cut),
            "--layer-pipeline=none",
            "--layer-memory=" + str(c.get("layer_memory") or "interleave"),
            "--decode-cache-window=" + str(c.get("decode_cache_window", 256)),
        ]
        for key in ("layer_workers", "layer_cache_mib"):
            if c.get(key):
                cmd += [
                    "--"
                    + key.replace("_", "-")
                    + "="
                    + (
                        "0," * (len(c["gpu_devices"].split(",")) - 1) + "0"
                        if prefill and key == "layer_cache_mib"
                        else c[key]
                    )
                ]
        if c.get("layer_pin_coordinator"):
            cmd.append("--layer-pin-coordinator")
        if c.get("decode_cache_adapt") and not prefill:
            cmd.append("--decode-cache-adapt")
    elif not prefill:
        cmd.append("--eval-verify")
    if prefill:
        cmd.append("--eval-prefill")
    if c.get("worker_decoder"):
        launch = [
            sys.executable,
            str(Path(__file__).with_name("glm_mixed_run.py")),
            "--primary",
            cmd[0],
            "--worker",
            c["worker_decoder"],
            "--model",
            c["model"],
            "--combine",
            c.get("worker_combine", "strict"),
            "--primary-device",
            str(c.get("primary_device", 0)),
            "--worker-device",
            str(c.get("worker_device", 1)),
            "--worker-cache-mib",
            str(c.get("worker_cache_mib", 24576)),
            "--worker-gpu-budget-mib",
            str(c.get("worker_gpu_budget_mib", 32768)),
        ]
        for role in ("primary", "worker"):
            if c.get(role + "_cpus"):
                launch += ["--" + role + "-cpus", c[role + "_cpus"]]
        cmd = launch + ["--"] + cmd[2:]
    if not layer:
        cmd = ["numactl", "--interleave=all", *cmd]
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
    env.update(record.get("environment", {}))
    env["STRATA_GLM_EVAL_WIDTH"] = str(width)
    return cmd, env, width


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prefill", action="store_true")
    refs = p.add_mutually_exclusive_group(required=True)
    refs.add_argument("--save-reference", action="store_true")
    refs.add_argument("--reference", type=Path)
    p.add_argument("--baseline-result", type=Path)
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    record = json.loads(a.benchmark.read_text())
    c = record["configuration"]
    if digest(c["decoder"]) != record["decoder_sha256"] or (
        c.get("worker_decoder")
        and digest(c["worker_decoder"]) != record["worker_sha256"]
    ):
        p.error("benchmark binaries changed")
    corpus = a.output / "corpus.ids"
    corpus.write_bytes(a.corpus.read_bytes())
    lengths = [
        len(prompt_ids_from_line)
        for prompt_ids_from_line in [
            line.replace(",", " ").split()
            for line in corpus.read_text().splitlines()
            if line.strip()
        ]
    ]
    if not lengths or min(lengths) < 2:
        raise ValueError("invalid evaluation corpus")
    reference = (
        a.output / "reference.logits" if a.save_reference else a.reference.resolve()
    )
    reference_stat = reference.stat() if reference.exists() else None
    reference_sha = digest(reference) if reference_stat else None
    cmd, env, width = evaluation_command(
        record, corpus, reference, a.save_reference, a.prefill
    )
    model = Path(c["model"])
    stat = model.stat()
    result = dict(
        benchmark_sha256=digest(a.benchmark),
        decoder_sha256=record["decoder_sha256"],
        worker_sha256=record.get("worker_sha256"),
        command=cmd,
        environment={k: v for k, v in env.items() if k.startswith("STRATA_")},
        model=dict(
            path=str(model.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns
        ),
        corpus_sha256=digest(corpus),
        sequences=len(lengths),
        expected_tokens=sum(n - 1 for n in lengths),
        reference=str(reference),
        reference_sha256=reference_sha,
        prefill=a.prefill,
        evaluation_width=width,
        verification_path=not a.prefill and c.get("gpu_layout") != "layer",
        complete=False,
        corpus_gate_passed=False,
        quality_qualified=False,
        peak_swap_bytes=0,
        scope="Teacher-forced target kernels; generated answers and speculative acceptance are separate gates.",
    )
    save_result(a.output / "result.json", result)
    wait_idle(timeout=600)
    start = time.monotonic()
    with (
        (a.output / "eval.stdout").open("w") as out,
        (a.output / "eval.log").open("w") as err,
    ):
        process = subprocess.Popen(
            cmd, env=env, stdout=out, stderr=err, start_new_session=True
        )
        try:
            while True:
                sample = process_memory(process.pid)
                result["peak_swap_bytes"] = max(
                    result["peak_swap_bytes"], sample["swap_bytes"]
                )
                if sample["swap_bytes"]:
                    raise RuntimeError("evaluation swapped")
                try:
                    result["returncode"] = process.wait(timeout=1)
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
            result["elapsed_seconds"] = time.monotonic() - start
            save_result(a.output / "result.json", result)
    rows = parse_eval((a.output / "eval.stdout").read_text())
    result["rows"] = rows
    result["complete"] = bool(
        result["returncode"] == 0
        and not result["peak_swap_bytes"]
        and [r["sequence"] for r in rows] == list(range(1, len(lengths) + 1))
        and rows[-1]["tokens"] == result["expected_tokens"]
    )
    if rows:
        result["metrics"] = dict(rows[-1])
        result["metrics"].setdefault("kl", 0.0)
        result["metrics"].setdefault("top1_agreement", 1.0)
        result["complete"] &= all(
            math.isfinite(result["metrics"][k])
            for k in ("perplexity", "kl", "top1_agreement")
        )
    if reference_stat:
        result["complete"] &= (
            reference.stat().st_size == reference_stat.st_size
            and reference.stat().st_mtime_ns == reference_stat.st_mtime_ns
            and digest(reference) == reference_sha
        )
    elif reference.exists():
        result["reference_sha256"] = digest(reference)
    result["complete"] &= (
        model.stat().st_size == stat.st_size
        and model.stat().st_mtime_ns == stat.st_mtime_ns
    )
    if a.baseline_result:
        baseline = json.loads(a.baseline_result.read_text())
        failures = []
        if not baseline.get("complete") or not result["complete"]:
            failures.append("incomplete corpus evaluation")
        for key in (
            "model",
            "corpus_sha256",
            "reference_sha256",
            "prefill",
            "expected_tokens",
        ):
            if baseline.get(key) != result.get(key):
                failures.append("incomparable " + key)
        if not failures:
            failures += check_quality(result["metrics"], baseline["metrics"], TOLERANCE)
        result["failures"] = failures
        result["corpus_gate_passed"] = not failures
    save_result(a.output / "result.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "complete",
                    "corpus_gate_passed",
                    "quality_qualified",
                    "metrics",
                )
                if k in result
            }
        )
    )
    return 0 if result["complete"] and not result.get("failures") else 1


if __name__ == "__main__":
    raise SystemExit(main())
