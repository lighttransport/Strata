import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
runs = []
for p in sorted(root.glob("st-*.err")):
    text = p.read_text()
    d = re.findall(r"^DECODE steps=(\d+) ms=([\d.]+) tok_s=([\d.]+)", text, re.M)
    speculative = not d
    if speculative:
        d = re.findall(r"^SPECULATIVE source=mtp generated=(\d+) .*? ms=([\d.]+) tok_s=([\d.]+)", text, re.M)
    tr = re.findall(r"^STEP_TRACE .*", text, re.M)
    if not d or not tr:
        continue
    v = dict(re.findall(r"(\w+)=([\d.]+)", tr[-1]))
    n = float(v["steps"])
    runs.append({"candidate": p.stem[3:], "tok_s": float(d[-1][2]), "steps": int(d[-1][0]), "speculative": speculative,
                 "per_token_ms": {k: float(v[k]) / n for k in
                      ["cpu_ms", "between_ms", "head_ms", "tail_ms", "tail_sync_ms", "tail_plan_ms", "tail_copy_ms"]},
                 "tier_swaps": int(v["tier_swaps"])})
quality = []
for p in sorted(root.glob("he-*/measurement.json")):
    m = json.loads(p.read_text())
    r = m["results"]
    quality.append({"candidate": p.parent.name[3:], "tasks": len(r), "passed": sum(x["passed"] for x in r),
                    "token_cap_hits": sum(x["generated_tokens"] == 1024 for x in r)})
(root / "measurements.json").write_text(json.dumps({"hardware": "B550 Ryzen 9 3950X / RX 9070 XT / DDR4-2133",
    "protocol": "2048-token prefill, 128 greedy output tokens, 127 timed decode steps", "runs": runs,
    "quality_screens": quality}, indent=2) + "\n")
