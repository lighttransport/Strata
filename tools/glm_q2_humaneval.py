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


def load_engine_arguments(path, model, context, stops):
    """Use a measured engine configuration while keeping model/context/stops bound."""
    values = json.loads(pathlib.Path(path).read_text())
    if not isinstance(values,list) or len(values)<23 or any(not isinstance(v,str) for v in values):
        raise ValueError('engine arguments must be a list of at least 23 strings')
    if pathlib.Path(values[0]).resolve()!=pathlib.Path(model).resolve() or int(values[1])!=context:
        raise ValueError('engine arguments use a different model or context')
    supplied={int(value) for value in values[6].split(',')}
    if not stops or not stops<=supplied:
        raise ValueError('engine arguments must retain every model stop token')
    return values


def result_summary(results):
    graded=[r for r in results if isinstance(r.get('passed'),bool)]
    return dict(complete=len(results)==164 and len(graded)==164,
                generation_complete=len(results)==164, tasks=len(results), graded_tasks=len(graded),
                pass_at_1=sum(r['passed'] for r in graded)/len(graded) if graded else None,results=results)


def atomic_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


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
    parser.add_argument("--config", type=pathlib.Path,
                        help="optional engine config for context, batch, threads, affinity, cache and environment")
    parser.add_argument("--engine-args", type=pathlib.Path, help="JSON engine argument list from the measured configuration")
    parser.add_argument("--generate-only", action="store_true", help="save ungraded answers for isolated grading on another host")
    parser.add_argument("--resume", action="store_true", help="resume completed tasks with the exact same generation manifest")
    parser.add_argument("--bound-binary", type=pathlib.Path, action="append", default=[],
                        help="also bind an underlying decoder/worker when --decoder is a launcher")
    parser.add_argument("--expert-pack", type=pathlib.Path)
    parser.add_argument("--expert-pack-profile", type=pathlib.Path)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=164, help="smaller values are smoke tests, not full pass@1")
    parser.add_argument("--gpu-budget-mib", type=int, default=12288)
    parser.add_argument("--decode-cache-mib", type=int, default=0, help="GPU expert tier, as the server's decode_cache_mib")
    parser.add_argument("--speculative", choices=("none", "mtp"), default="none")
    parser.add_argument("--draft-depth", type=int, default=1)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--prefill-experts", choices=("mmq", "f16", "f16-batched"), default="mmq")
    parser.add_argument("--prefill-batch", type=int, default=256)
    parser.add_argument("--route-affinity", type=float, default=0.0, help="lossy: STRATA_GLM_ROUTE_AFFINITY for the engine")
    args = parser.parse_args()
    if not 1 <= args.limit <= 164 or not 1 <= args.tokens <= 4096:
        parser.error("limit must be 1..164 and tokens 1..4096")
    if args.context < args.tokens or not 1 <= args.prefill_batch <= args.context:
        parser.error("context must hold generation and prefill batch must be 1..context")
    if args.expert_pack_profile and not args.expert_pack:
        parser.error("expert-pack-profile requires expert-pack")
    if args.config and args.engine_args:
        parser.error("choose config or engine-args, not both")
    raw = args.dataset.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("HumanEval dataset checksum differs from the pinned revision")
    tasks = [json.loads(line) for line in gzip.decompress(raw).splitlines()]
    if len(tasks) != 164 or len({task["task_id"] for task in tasks}) != 164:
        raise ValueError("invalid HumanEval task set")
    if args.output.exists() and not args.resume:
        raise ValueError("refusing to overwrite evaluation output")
    args.output.mkdir(parents=True,exist_ok=args.resume)
    if not args.generate_only and not isolated_test("def candidate(): return 1", "def check(f): assert f() == 1", "candidate")["passed"]:
        raise RuntimeError("isolated test runner unavailable; no generated code was executed")
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "serve"))
    from server import GlmEngine
    model = args.model.resolve()
    metadata = GGUFFile(model).metadata
    tokenizer = Tokenizer.from_gguf(model)
    stops = {int(metadata[k]) for k in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id") if k in metadata}
    environment = SandboxedEnvironment(extensions=["jinja2.ext.loopcontrols"])
    environment.globals["raise_exception"] = lambda s: (_ for _ in ()).throw(ValueError(s))
    template = environment.from_string(metadata["tokenizer.chat_template"])
    engine_args = [str(model), str(args.context), "4096", "15", "0", str(args.prefill_batch), ",".join(map(str, sorted(stops))),
                   str(args.gpu_budget_mib), "0", "0", args.speculative, str(args.draft_depth), "numa", "0", "0", "0", "0", "0",
                   str(args.decode_cache_mib), "1", "256", "0", args.prefill_experts, "4k"]
    cfg = json.loads(args.config.read_text()) if args.config else {}
    for index, key in ((1, "context"), (2, "dense_cache_mib"), (3, "threads"), (5, "prefill_batch"),
                       (7, "gpu_budget_mib"), (12, "cpu_affinity"), (18, "decode_cache_mib"),
                       (20, "decode_cache_window"), (22, "prefill_experts")):
        if key in cfg:
            engine_args[index] = str(cfg[key])
    if "decode_graphs" in cfg:
        engine_args[19] = str(int(bool(cfg["decode_graphs"])))
    if args.config:
        # Match the GLM API's REAP end-of-turn handling for configured runs.
        vocabulary = metadata.get("tokenizer.ggml.tokens", [])
        stops.update(vocabulary.index(s) for s in ("<|user|>", "<|observation|>") if s in vocabulary)
        engine_args[6] = ",".join(map(str, sorted(stops)))
    if args.expert_pack:
        engine_args += [str(args.expert_pack.resolve()), str(args.expert_pack_profile.resolve()) if args.expert_pack_profile else "", "native"]
    if args.engine_args:
        if args.expert_pack or args.expert_pack_profile:
            parser.error('engine-args already owns the expert placement configuration')
        engine_args=load_engine_arguments(args.engine_args,model,args.context,stops)
    manifest = {"dataset_revision": REVISION, "dataset_sha256": DATA_SHA256,
                "protocol": "greedy chat, low reasoning, one sample per task, isolated tests",
                "maximum_generated_tokens": args.tokens, "engine_args": engine_args,
                "environment": {k: v for k, v in os.environ.items() if k.startswith("STRATA_")}}
    if args.engine_args:
        manifest['engine_args_sha256']=hashlib.sha256(args.engine_args.read_bytes()).hexdigest()
    engine_env = os.environ.copy()
    engine_env.update(cfg.get("env", {}))
    if args.route_affinity:
        engine_env["STRATA_GLM_ROUTE_AFFINITY"] = str(args.route_affinity)
    manifest["environment"] = {k: v for k, v in engine_env.items() if k.startswith("STRATA_")}
    manifest["decoder_sha256"] = hashlib.sha256(pathlib.Path(args.decoder).read_bytes()).hexdigest()
    manifest["route_affinity"] = args.route_affinity
    if args.config:
        manifest["config"] = cfg
        manifest["environment"] = {k: v for k, v in engine_env.items() if k.startswith("STRATA_")}
    manifest['generation_only']=args.generate_only
    if args.bound_binary:
        manifest['bound_binaries']={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in args.bound_binary}
    manifest_path=args.output/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError('resume generation identity differs; use a new output directory')
    if not manifest_path.exists():atomic_json(manifest_path,manifest)
    measurement=args.output/'measurement.json'
    results=json.loads(measurement.read_text())['results'] if measurement.exists() else []
    done={r['task_id'] for r in results}
    if len(done)!=len(results) or not done<={task['task_id'] for task in tasks}:
        raise ValueError('invalid or duplicate resumed task set')
    if all(task['task_id'] in done for task in tasks[:args.limit]):return
    engine = GlmEngine(args.decoder, engine_args, log=str(args.output / "engine.log"), env=engine_env)
    try:
        for task in tasks[:args.limit]:
            if task['task_id'] in done:continue
            prompt = "Complete this Python function. Return the complete function in one Python code block with any required imports. Keep reasoning very brief.\n\n" + task["prompt"]
            chat = template.render(messages=[dict(role="user", content=prompt)], tools=None,
                                   add_generation_prompt=True, reasoning_effort="low")
            ids = tokenizer.encode(chat, parse_special=True)
            if len(ids) + args.tokens > int(engine_args[1]):
                raise ValueError("evaluation exceeds context budget")
            start = time.monotonic()
            generated = [t for t in engine.generate(ids, args.tokens, {}, threading.Event()) if t is not None]
            text = tokenizer.decode([t for t in generated if t not in stops])
            code = solution(text, task["prompt"], task["entry_point"])
            result = {"task_id": task["task_id"], "input_tokens": len(ids), "generated_tokens": len(generated),
                      "input_token_sha256":hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                      "wall_seconds": time.monotonic() - start, "token_sha256": hashlib.sha256(json.dumps(generated).encode()).hexdigest(),
                      **(dict(passed=None,reason='ungraded generation') if args.generate_only else
                         isolated_test(code, task["test"], task["entry_point"]))}
            stem = task["task_id"].replace("/", "_")
            if any((args.output/(stem+suffix)).exists() for suffix in ('.txt','.py','.tokens.json')):
                attempt=1
                while any((args.output/(stem+f'.retry-{attempt}'+suffix)).exists() for suffix in ('.txt','.py','.tokens.json')):attempt+=1
                stem+=f'.retry-{attempt}'
            (args.output / (stem + ".txt")).write_text(text)
            (args.output / (stem + ".py")).write_text(code)
            atomic_json(args.output/(stem+'.tokens.json'),generated)
            result.update(text_file=stem+'.txt',code_file=stem+'.py',tokens_file=stem+'.tokens.json',
                          code_sha256=hashlib.sha256(code.encode()).hexdigest(),text_sha256=hashlib.sha256(text.encode()).hexdigest())
            results.append(result)
            atomic_json(measurement,result_summary(results))
            print(json.dumps(result), flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
