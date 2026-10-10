"""Pinned HumanEval, one greedy chat sample per task, with isolated Python tests.

Run under glm_q2_run_guard.py to enforce the model's RAM limits. This is a chat
evaluation, not the original raw-completion HumanEval prompting protocol.
"""
import argparse
import gzip
import hashlib
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time
import tempfile

from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer
from jinja2.sandbox import SandboxedEnvironment

REVISION = "6d43fb980f9fee3c892a914eda09951f772ad10d"
DATA_SHA256 = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
DATA_URL = f"https://raw.githubusercontent.com/openai/human-eval/{REVISION}/data/HumanEval.jsonl.gz"


def solution(text, prompt, entry):
    text = text.rsplit("</think>", 1)[-1]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    code = blocks[-1] if blocks else text
    return code if re.search(rf"(?m)^def\s+{re.escape(entry)}\s*\(", code) else prompt + code


def isolated_test(code, tests, entry, timeout=5):
    # No home, repository, model, network, or host-writable filesystem is exposed.
    command = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
               "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
               "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
               "--chdir", "/tmp", "--clearenv", "--setenv", "PATH", "/usr/bin",
               "/usr/bin/python3", "-I", "-"]
    prelude = """import resource
resource.setrlimit(resource.RLIMIT_AS, (512*1024**2, 512*1024**2))
resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
"""
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=output,
                                   stderr=subprocess.DEVNULL, text=True, start_new_session=True)
        try:
            process.communicate(prelude + code + "\n" + tests + f"\ncheck({entry})\nprint('STRATA_TEST_PASSED')\n", timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return {"passed": False, "reason": "timeout"}
        output.seek(0)
        passed = process.returncode == 0 and output.read(1024**2).endswith(b"STRATA_TEST_PASSED\n")
    return {"passed": passed, "reason": "passed" if passed else "test failure",
            "exit_code": process.returncode}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("dataset", type=pathlib.Path, help=f"download {DATA_URL}")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--decoder", default="build-glm/strata-glm-decode")
    parser.add_argument("--expert-pack", type=pathlib.Path)
    parser.add_argument("--expert-pack-profile", type=pathlib.Path)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=164, help="smaller values are smoke tests, not full pass@1")
    parser.add_argument("--offset", type=int, default=0, help="b550 copy: first task index")
    parser.add_argument("--task-indices", type=lambda s: [int(x) for x in s.split(",")],
                        help="explicit zero-based smoke-test subset; overrides offset and limit")
    parser.add_argument("--gpu-budget-mib", type=int, default=12288)
    parser.add_argument("--decode-cache-mib", type=int, default=0, help="GPU expert tier, as the server's decode_cache_mib")
    parser.add_argument("--speculative", choices=("none", "mtp"), default="none")
    parser.add_argument("--draft-depth", type=int, default=1)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--prefill-batch", type=int, choices=(2048, 4096, 8192, 16384), default=2048)
    parser.add_argument("--prefill-experts", choices=("mmq", "f16", "f16-batched", "bf16-batched"), default="mmq")
    parser.add_argument("--threads", type=int, default=15)
    parser.add_argument("--background", type=pathlib.Path, help="unrelated code context prepended to each task")
    parser.add_argument("--route-affinity", type=float, default=0.0, help="lossy: STRATA_GLM_ROUTE_AFFINITY for the engine")
    args = parser.parse_args()
    if not 1 <= args.limit <= 164 or not 1 <= args.tokens <= 4096:
        parser.error("limit must be 1..164 and tokens 1..4096")
    if args.context < args.tokens + 1 or args.threads < 1 or args.offset < 0 or args.offset + args.limit > 164:
        parser.error("invalid context, threads or task range")
    if args.task_indices is not None and (len(set(args.task_indices)) != len(args.task_indices) or
                                         any(i < 0 or i >= 164 for i in args.task_indices)):
        parser.error("task indices must be unique and within 0..163")
    if args.expert_pack_profile and not args.expert_pack:
        parser.error("expert-pack-profile requires expert-pack")
    raw = args.dataset.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("HumanEval dataset checksum differs from the pinned revision")
    tasks = [json.loads(line) for line in gzip.decompress(raw).splitlines()]
    if len(tasks) != 164 or len({task["task_id"] for task in tasks}) != 164:
        raise ValueError("invalid HumanEval task set")
    selected = [tasks[i] for i in args.task_indices] if args.task_indices is not None else tasks[args.offset:args.offset + args.limit]
    if args.output.exists():
        raise ValueError("refusing to overwrite evaluation output")
    args.output.mkdir(parents=True)
    if not isolated_test("def candidate(): return 1", "def check(f): assert f() == 1", "candidate")["passed"]:
        raise RuntimeError("isolated test runner unavailable; no generated code was executed")
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "serve"))
    from server import GlmEngine
    model = args.model.resolve()
    metadata = GGUFFile(model).metadata
    tokenizer = Tokenizer.from_gguf(model)
    stops = {int(metadata[k]) for k in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id") if k in metadata}
    stops.add(154827)  # b550 copy: <|user|> ends the assistant turn in this GGUF but is not in its eos metadata
    environment = SandboxedEnvironment(extensions=["jinja2.ext.loopcontrols"])
    environment.globals["raise_exception"] = lambda s: (_ for _ in ()).throw(ValueError(s))
    template = environment.from_string(metadata["tokenizer.chat_template"])
    engine_args = [str(model), str(args.context), "4096", str(args.threads), "0", str(args.prefill_batch), ",".join(map(str, sorted(stops))),
                   str(args.gpu_budget_mib), "0", "0", args.speculative, str(args.draft_depth), "numa", "0", "0", "0", "0", "0",
                   str(args.decode_cache_mib), "1", "256", "0", args.prefill_experts, "4k"]
    if args.expert_pack:
        engine_args += [str(args.expert_pack.resolve()), str(args.expert_pack_profile.resolve()) if args.expert_pack_profile else "", "native"]
    manifest = {"dataset_revision": REVISION, "dataset_sha256": DATA_SHA256,
                "protocol": "greedy chat, low reasoning, one sample per task, isolated tests",
                "decoder_sha256": hashlib.sha256(pathlib.Path(args.decoder).read_bytes()).hexdigest(),
                "task_ids": [task["task_id"] for task in selected],
                "maximum_generated_tokens": args.tokens, "engine_args": engine_args,
                "environment": {k: v for k, v in os.environ.items() if k.startswith("STRATA_")}}
    background = args.background.read_text() if args.background else ""
    if background:
        manifest["background_sha256"] = hashlib.sha256(background.encode()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    engine_env = os.environ.copy()
    if args.route_affinity:
        engine_env["STRATA_GLM_ROUTE_AFFINITY"] = str(args.route_affinity)
    manifest["route_affinity"] = args.route_affinity
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    engine = GlmEngine(args.decoder, engine_args, log=str(args.output / "engine.log"), env=engine_env)
    results = []
    try:
        for task in selected:
            prompt = "Complete this Python function. Return the complete function in one Python code block with any required imports. Keep reasoning very brief.\n\n" + task["prompt"]
            if background:
                prompt = "The following C++ code is unrelated background, not the function to complete.\n```cpp\n" + background + "\n```\n\n" + prompt
            chat = template.render(messages=[dict(role="user", content=prompt)], tools=None,
                                   add_generation_prompt=True, reasoning_effort="low")
            ids = tokenizer.encode(chat, parse_special=True)
            if len(ids) + args.tokens > args.context:
                raise ValueError("evaluation exceeds context budget")
            start = time.monotonic()
            generated = [t for t in engine.generate(ids, args.tokens, {}, threading.Event()) if t is not None]
            text = tokenizer.decode([t for t in generated if t not in stops])
            code = solution(text, task["prompt"], task["entry_point"])
            result = {"task_id": task["task_id"], "input_tokens": len(ids), "generated_tokens": len(generated),
                      "wall_seconds": time.monotonic() - start, "token_sha256": hashlib.sha256(json.dumps(generated).encode()).hexdigest(),
                      **isolated_test(code, task["test"], task["entry_point"])}
            stem = task["task_id"].replace("/", "_")
            (args.output / (stem + ".txt")).write_text(text)
            (args.output / (stem + ".py")).write_text(code)
            results.append(result)
            summary = {"complete": len(results) == 164, "tasks": len(results),
                       "requested_tasks_complete": len(results) == len(selected),
                       "pass_at_1": sum(r["passed"] for r in results) / len(results), "results": results}
            (args.output / "measurement.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
