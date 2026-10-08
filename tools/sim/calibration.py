"""Free parameters, the measured-record dataset, validation and fitting.

`fit` is a coordinate grid search: for each selected parameter it tries a few multiples of the current value and
keeps the one that minimises the mean absolute log error over the records marked `fit: true`. It is meant to
re-tune after a model change, not to find physics; the defaults in kernels.Params already fit the quiet rows,
so no params.json is shipped. Check `validate` after a fit: a lower mean error can still push single rows out.
"""
import dataclasses
import json
import math
import pathlib

import decode
import hw as hwmod
import kernels
import prefill
from config import RunConfig

DATA = pathlib.Path(__file__).resolve().parent / "data"
PARAMS_FILE = DATA / "params.json"
MEASURED_FILE = DATA / "measured_tr16.json"

FIT_PARAMETERS = [
    "cpu_bw_scale", "cpu_width_penalty", "cpu_layer_overhead_ms", "cpu_quant_scale", "gpu_width_factor",
    "gpu_layer_fixed_us", "draft_fixed_ms", "tier_upload_ms",
    "prefill_fixed_s", "prefill_ms_per_token", "prefill_sync_ms_per_layer",
]


def load_params(path=None):
    path = pathlib.Path(path) if path else PARAMS_FILE
    params = kernels.Params()
    if path.exists():
        data = json.loads(path.read_text())
        for key, value in data.get("params", data).items():
            if hasattr(params, key):
                setattr(params, key, value)
    return params


def save_params(params, path=None, extra=None):
    path = pathlib.Path(path) if path else PARAMS_FILE
    body = {"params": dataclasses.asdict(params)}
    if extra:
        body.update(extra)
    path.write_text(json.dumps(body, indent=2) + "\n")


def load_records(path=None):
    path = pathlib.Path(path) if path else MEASURED_FILE
    return json.loads(path.read_text())["records"]


def hardware_for(record):
    hw = hwmod.load(record.get("hw", "tr16"))
    for key, value in record.get("hw_set", {}).items():
        hw.set(key, value)
    return hw


def predict(record, params, trace=None, prior=None):
    hw = hardware_for(record)
    cfg = RunConfig.from_dict(record["config"])
    if record["kind"] == "prefill":
        result = prefill.simulate(hw, cfg, params)
        return result.tok_s, result
    result = decode.simulate(hw, cfg, params, trace=trace, prior=prior)
    return result.tok_s, result


def evaluate(records, params, only_fit=False):
    rows = []
    for record in records:
        if only_fit and not record.get("fit", False):
            continue
        predicted, result = predict(record, params)
        measured = record["measured"]["tok_s"]
        error = predicted / measured - 1
        tolerance = record.get("tolerance", 0.15)
        extra = {}
        if record["kind"] == "decode":
            extra = dict(cpu_ms=result.step.cpu_ms, gpu_ms=result.step.gpu_ms, hit=result.hit_bytes_share,
                         cpu_gb_per_token=result.cpu_gb_per_token, bottleneck=result.bottleneck)
        else:
            extra = dict(bottleneck=result.bottleneck, pcie_s=result.pcie_s_per_chunk, gpu_s=result.gpu_s_per_chunk)
        rows.append(dict(name=record["name"], kind=record["kind"], measured=measured, predicted=predicted,
                         error=error, tolerance=tolerance, within=abs(error) <= tolerance,
                         quiet=record.get("quiet", False), fit=record.get("fit", False), **extra))
    return rows


def objective(records, params, other_weight=0.3):
    """Mean |log(predicted / measured)|; fit rows weigh 1, the other records `other_weight` so the search
    does not improve the quiet rows at the expense of the rest."""
    total = weight_sum = 0.0
    for record in records:
        weight = 1.0 if record.get("fit", False) else other_weight
        if weight == 0:
            continue
        predicted, _ = predict(record, params)
        total += weight * abs(math.log(predicted / record["measured"]["tok_s"]))
        weight_sum += weight
    return total / weight_sum if weight_sum else 0.0


def fit(records, params=None, names=None, rounds=3, factors=(0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3), log=print):
    params = dataclasses.replace(params or kernels.Params())
    names = names or FIT_PARAMETERS
    best = objective(records, params)
    log(f"start: mean |log error| = {best:.4f}")
    for round_ in range(rounds):
        improved = False
        for name in names:
            base = getattr(params, name)
            if base == 0:
                continue
            candidates = []
            for factor in factors:
                trial = dataclasses.replace(params, **{name: base * factor})
                candidates.append((objective(records, trial), factor))
            score, factor = min(candidates)
            if score < best - 1e-6:
                setattr(params, name, base * factor)
                best = score
                improved = True
                log(f"round {round_ + 1}: {name} x{factor:g} -> {getattr(params, name):.4g} (error {best:.4f})")
        if not improved:
            break
    log(f"done: mean |log error| = {best:.4f}")
    return params, best
