"""Extract compact measured results; retain raw guard records separately."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def performance(prefix):
    result = json.loads((ROOT / (prefix + ".result.json")).read_text())
    values = result["measurements"]
    return {
        "record": prefix + ".result.json",
        "clean": result["clean"],
        "context": result["config"]["context"],
        "prefill_batch": result["config"]["prefill_batch"],
        "prefill": values["prefill"],
        "decode": values["decode"],
        "peak_ram_gib": result["cgroup_after"]["memory.peak"] / 2**30,
        "minimum_gpu_free_mib": result["minimum_gpu_free_mib"],
        "decode_cache": values["decode_cache"],
        "page_faults": values["page_faults"],
    }


def coding(name):
    directory = ROOT / name
    measurement = json.loads((directory / "measurement.json").read_text())
    rows = measurement["results"]
    malformed = []
    for row in rows:
        source = directory / (row["task_id"].replace("/", "_") + ".py")
        try:
            ast.parse(source.read_text())
        except SyntaxError:
            malformed.append(row["task_id"])
    guard = json.loads((ROOT / (name + ".memory.json")).read_text())
    return {
        "record": name + "/measurement.json",
        "task_ids": [row["task_id"] for row in rows],
        "requested_tasks_complete": measurement.get("requested_tasks_complete", False),
        "passed": sum(row["passed"] for row in rows),
        "malformed": malformed,
        "cap_hits": sum(row["generated_tokens"] == 1024 for row in rows),
        "input_tokens": [row["input_tokens"] for row in rows],
        "guard_passed": guard["exit_code"] == 0 and guard["rejected"] is None and guard["peak_swap_kib"] == 0,
    }


def main():
    prefixes = ["base8k-02", "base16k-01", "base32k-02", "best16k-01", "best32k-01",
                "hybrid-buckets-rows4-8k-01", "hybrid-buckets-8k-01", "buckets32k-01",
                "f16-16k-01", "f16-default-kda8k-01"]
    prefixes += sorted(path.name.removesuffix(".result.json") for path in ROOT.glob("final-*.result.json"))
    names = ["he-mmq-short", "he-f16-short", "he-hybrid-short", "he-best-short", "he-gated-best-short",
             "he-mmq-long-targets", "he-gated-best-long-targets", "he-gated-serial-long-targets", "he-mmq-gated-long-targets", "he-mmq-gated-short"]
    summary = {
        "final_comparison": json.loads((ROOT / "final-comparison.json").read_text()),
        "hardware": {"cpu": "Ryzen 9 3950X", "gpu": "RX 9070 XT gfx1201", "rocm": "7.1.4",
                     "ddr_mt_s": 2133, "ram_limit_gib": 60, "swap_limit_bytes": 0,
                     "gpu_physical_mib": 16304, "gpu_allocation_budget_mib": 15360},
        "performance": {name: performance(name) for name in prefixes},
        "coding": {name: coding(name) for name in names if (ROOT / (name + ".memory.json")).exists()},
        "recall": {path.stem: json.loads(path.read_text()) for path in sorted(ROOT.glob("recall-*.quality.json"))},
        "performance_token_parity": json.loads((ROOT / "quality-mmq-performance-parity.json").read_text()),
        "recall_control_parity": json.loads((ROOT / "quality-recall-control.json").read_text()),
        "default_parity": json.loads((ROOT / "default-parity.json").read_text()),
        "mmq_short_parity": json.loads((ROOT / "quality-mmq-short-parity.json").read_text()),
        "short_parity": json.loads((ROOT / "quality-short.json").read_text()),
        "excluded": {"base32k-01": "sustained external hf CPU work", "hybrid-8k-01": "sustained external hf CPU work",
                     "base8k": "invalid token file: trailing newline", "hybrid-b16384-32k-01": "cold cache accounting check failed before launch",
                     "final-mmq-16384-01": "kswapd interference",
                     "final-control-16384-01": "control stopped after the same accounting collapse and low GPU activity",
                     "final-mmq-16384-02": "already disqualified retry stopped after sustained reclaim and accounting collapse",
                     "recall-mmq-gated-32768-01": "service timeout at 15m30s; no completed result; kswapd activity, cause unconfirmed"},
        "selection": "Optional 8K MMQ preset: repeat +22.9% prefill, decode 33.32 vs 33.38; coding/token parity preserved. FP16 rejected; 16K/32K performance repeatability unresolved under shared-system pressure.",
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Updated", ROOT / "summary.json")


if __name__ == "__main__":
    main()
