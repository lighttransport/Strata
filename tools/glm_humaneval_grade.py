#!/usr/bin/env python3
"""Grade frozen HumanEval generations locally with the existing isolated runner.

Generation files remain unchanged. The output is a separate complete/partial
qualification record; generation-only results never qualify by themselves.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from glm_q2_humaneval import (
    DATA_SHA256,
    isolated_test,
    result_summary,
    solution,
    atomic_json,
)


def grade(source, dataset, output):
    if output.exists():
        raise ValueError("refusing to overwrite a grading record")
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("HumanEval dataset checksum differs")
    tasks = {
        r["task_id"]: r
        for r in map(json.loads, gzip.decompress(dataset.read_bytes()).splitlines())
    }
    manifest = source / "manifest.json"
    measurement = source / "measurement.json"
    if (
        json.loads(manifest.read_text())["dataset_sha256"] != DATA_SHA256
        or len(tasks) != 164
    ):
        raise ValueError("generation dataset identity differs")
    rows = json.loads(measurement.read_text())["results"]
    ids = {row["task_id"] for row in rows}
    if len(ids) != len(rows) or not ids <= tasks.keys():
        raise ValueError("invalid generation task set")
    if not isolated_test(
        "def candidate(): return 1", "def check(f): assert f()==1", "candidate"
    )["passed"]:
        raise RuntimeError(
            "isolated test runner unavailable; no generated code was executed"
        )
    results = []
    for row in rows:
        task = tasks[row["task_id"]]

        def artifact(key):
            path = (source / row[key]).resolve()
            if path.parent != source.resolve():
                raise ValueError("generation artifact escapes its directory")
            return path

        code = artifact("code_file").read_text()
        text = artifact("text_file").read_text()
        tokens = json.loads(artifact("tokens_file").read_text())
        if (
            hashlib.sha256(code.encode()).hexdigest() != row["code_sha256"]
            or hashlib.sha256(text.encode()).hexdigest() != row["text_sha256"]
            or hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
            != row["token_sha256"]
            or code != solution(text, task["prompt"], task["entry_point"])
        ):
            raise ValueError("generation artifact changed")
        result = dict(row, **isolated_test(code, task["test"], task["entry_point"]))
        results.append(result)
    summary = result_summary(results)
    summary.update(
        generation_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        generation_measurement_sha256=hashlib.sha256(
            measurement.read_bytes()
        ).hexdigest(),
        dataset_sha256=DATA_SHA256,
        isolation="existing bubblewrap runner; no model host access",
    )
    atomic_json(output, summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("dataset", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = grade(a.source, a.dataset, a.output)
    print(
        json.dumps(
            {k: result[k] for k in ("complete", "tasks", "graded_tasks", "pass_at_1")}
        )
    )


if __name__ == "__main__":
    main()
