import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parent
sets = {"q23mix2_reference": ["q23mix2-reference30"],
        "q22fit5rows4": ["q22fit5rows4-smoke", "q22fit5rows4-rest20"],
        "q22fit4": ["q22fit4-smoke", "q22fit4-rest20"]}
summary = {}
for name, tags in sets.items():
    rows = []
    for tag in tags:
        folder = root / ("he-" + tag)
        path = folder / "measurement.json"
        if not path.exists():
            continue
        for r in json.loads(path.read_text())["results"]:
            r = dict(r)
            source = folder / (r["task_id"].replace("/", "_") + ".py")
            try:
                ast.parse(source.read_text())
                r["syntax_valid"] = True
            except SyntaxError:
                r["syntax_valid"] = False
            rows.append(r)
    ids = [r["task_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate evaluated task")
    summary[name] = {"tasks": len(rows), "complete_first30": set(ids) == {f"HumanEval/{i}" for i in range(30)},
                     "passed": sum(r["passed"] for r in rows), "syntax_valid": sum(r["syntax_valid"] for r in rows),
                     "token_cap_hits": sum(r["generated_tokens"] == 1024 for r in rows),
                     "failed_tasks": [r["task_id"] for r in rows if not r["passed"]]}
(root / "quality-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
