"""Select a measured GLM layer cut from comparable completed benchmark records."""

import argparse
import hashlib
import json
import math
import pathlib
import statistics

MATCH = (
    "context",
    "threads",
    "gpu_devices",
    "gpu_budget_mib",
    "decode_cache_mib",
    "decode_cache_window",
    "decode_cache_adapt",
    "decode_graphs",
    "prefill_batch",
    "speculative",
    "draft_depth",
    "layer_pipeline",
    "layer_memory",
    "prompt_tokens",
    "generated_tokens",
    "repetitions",
    "validation_tokens",
    "validation_repetitions",
    "full_warmups",
    "cpu_affinity",
    "layer_workers",
    "layer_cache_mib",
    "layer_pin_coordinator",
)
CRITICAL_ENV = (
    "STRATA_GLM_CANON",
    "STRATA_GLM_Q8_DECODE",
    "STRATA_GLM_STEP_PIPELINE",
    "STRATA_GLM_DEVICE_EXPERTS",
    "STRATA_GLM_DEVICE_REDUCTION",
    "STRATA_GLM_LAYER_GRAPHS",
    "STRATA_GLM_KDA_PREPARE",
    "STRATA_GLM_KDA_ROW_PARTS",
    "STRATA_OLD_IQ_MMVQ",
)


def select_plan(records, model):
    if len(records) < 2:
        raise ValueError("provide at least two measured cuts")
    first = records[0]
    cfg = first["configuration"]
    expected = {
        k: cfg.get(k, "interleave" if k == "layer_memory" else None) for k in MATCH
    }
    expected["layer_memory"] = expected["layer_memory"] or "interleave"
    expected["pin_cpu"] = cfg.get("cpu_affinity", "auto") != "none"
    environment = {
        k: first.get("environment", {}).get(k)
        for k in sorted(set(first.get("environment", {})) | set(CRITICAL_ENV))
    }
    candidates = []
    for record in records:
        current = record["configuration"]
        settings = {
            k: current.get(k, "interleave" if k == "layer_memory" else None)
            for k in MATCH
        }
        settings["layer_memory"] = settings["layer_memory"] or "interleave"
        settings["pin_cpu"] = current.get("cpu_affinity", "auto") != "none"
        env = {
            k: record.get("environment", {}).get(k)
            for k in sorted(set(record.get("environment", {})) | set(CRITICAL_ENV))
        }
        if (
            settings != expected
            or env != environment
            or record["decoder_sha256"] != first["decoder_sha256"]
            or record["decoder_fnv64"] != first["decoder_fnv64"]
            or record["prompt_sha256"] != first["prompt_sha256"]
            or current["model"] != cfg["model"]
            or record.get("hardware") != first.get("hardware")
        ):
            raise ValueError(
                "cuts must share model, binary, hardware, prompt, environment and settings"
            )
        if (
            current.get("gpu_layout") != "layer"
            or not record.get("complete")
            or record.get("instrumented")
            or not record.get("repetitions_identical")
            or len(record.get("trials", [])) < 2
        ):
            raise ValueError(
                "only complete, repeated, stable, uninstrumented layer runs qualify for placement selection"
            )
        split = current.get("layer_split")
        devices = [int(x) for x in str(current["gpu_devices"]).split(",")]
        try:
            bounds = [0] + [int(x) for x in str(split).split(",")] + [45]
        except (ValueError, TypeError) as exc:
            raise ValueError("measure explicit GLM-5.3 layer cuts") from exc
        if len(bounds) != len(devices) + 1 or any(
            a >= b for a, b in zip(bounds, bounds[1:])
        ):
            raise ValueError("invalid or empty stage in measured cut")
        if any(
            len(t["token_ids"]) < 2
            or not math.isfinite(t["decode_ms"])
            or t["decode_ms"] <= 0
            or t["token_ids"] != record["trials"][0]["token_ids"]
            for t in record["trials"]
        ):
            raise ValueError("invalid or unstable measured decode trials")
        timings = [
            (len(t["token_ids"]) - 1) * 1000.0 / t["decode_ms"]
            for t in record["trials"]
        ]
        candidates.append(
            {
                "layer_split": str(split),
                "median_decode_tok_s": statistics.median(timings),
                "decode_tok_s": timings,
            }
        )
    if len({r["layer_split"] for r in candidates}) < 2:
        raise ValueError("provide at least two distinct measured cuts")
    best = max(candidates, key=lambda r: (r["median_decode_tok_s"], r["layer_split"]))
    model = pathlib.Path(model)
    if model.resolve() != pathlib.Path(cfg["model"]).resolve():
        raise ValueError("provided model differs from the measured model")
    v2 = bool(
        cfg.get("layer_workers")
        or cfg.get("layer_cache_mib")
        or cfg.get("layer_pin_coordinator")
    )
    if v2:
        expected["layer_pin_coordinator"] = bool(cfg.get("layer_pin_coordinator"))
    return dict(
        schema="strata.glm.layer-plan.v2" if v2 else "strata.glm.layer-plan.v1",
        model_filename=model.name,
        model_bytes=model.stat().st_size,
        configuration=expected,
        environment=environment,
        hardware=first.get("hardware", {}),
        decoder_sha256=first["decoder_sha256"],
        decoder_fnv64=first["decoder_fnv64"],
        prompt_sha256=first["prompt_sha256"],
        layer_split=best["layer_split"],
        candidates=candidates,
        timing_scope="measured single-stream decode; placement screening, not a coding-quality qualification",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("records", type=pathlib.Path, nargs="+")
    ap.add_argument("--model", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("output exists; preserve the previous measured plan")
    try:
        records = [json.loads(p.read_text()) for p in args.records]
        plan = select_plan(records, args.model)
    except (ValueError, KeyError, OSError) as exc:
        ap.error(str(exc))
    plan["records"] = [
        {"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in args.records
    ]
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"selected layer_split={plan['layer_split']}; plan={args.output}")


if __name__ == "__main__":
    main()
