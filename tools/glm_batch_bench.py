"""Concurrent GLM generation through the API engine adapter, with delivered-token accounting.

Run under glm_q2_run_guard.py for publishable timing. Prompt and output token IDs,
configuration, engine logs and per-request latency are retained with each result.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time

from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer
from jinja2.sandbox import SandboxedEnvironment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.server import GlmEngine, child_env


def engine_args(cfg, stops):
    devices = cfg.get("gpu_devices", "0")
    if isinstance(devices, list):
        devices = ",".join(map(str, devices))
    return [str(cfg["model"]), str(cfg["context"]), str(cfg.get("dense_cache_mib", 4096)),
            str(cfg.get("threads", 15)), str(cfg.get("expert_cache_mib", 0)), str(cfg.get("prefill_batch", 4096)),
            ",".join(map(str, sorted(stops))), str(cfg.get("gpu_budget_mib", 14336)),
            str(cfg.get("lookup_depth", 0)), str(int(cfg.get("decode_experts", "cpu") == "gpu")),
            cfg.get("speculative", "none"), str(cfg.get("draft_depth", 2)), cfg.get("cpu_affinity", "numa"),
            str(cfg.get("cpu_prepack_mib", 0)), str(devices), str(cfg.get("prefill_expert_cache_mib", 0)),
            str(int(bool(cfg.get("lock_weights", False)))), str(int(bool(cfg.get("decode_prefill_cache", False)))),
            str(cfg.get("decode_cache_mib", 5632)), str(int(cfg.get("decode_graphs", True))),
            str(cfg.get("decode_cache_window", 256)), str(int(bool(cfg.get("decode_cache_adapt", False)))),
            "mmq", cfg.get("weight_pages", "4k"), cfg.get("expert_pack", ""),
            cfg.get("expert_pack_profile", ""), cfg.get("cpu_expert_backend", "auto"),
            "--batch", str(cfg.get("parallel", 8))]


def prompts(tokenizer, metadata, length, count):
    template = SandboxedEnvironment(extensions=["jinja2.ext.loopcontrols"])
    template.globals["raise_exception"] = lambda s: (_ for _ in ()).throw(ValueError(s))
    template = template.from_string(metadata["tokenizer.chat_template"])
    topics = ["NUMA tensor ownership and row addressing", "thread affinity and worker dispatch",
              "quantized routed reductions", "expert grouping across tokens", "allocation failure handling",
              "thread pool shutdown and cancellation", "quantization edge cases", "concurrent memory lifetimes"]
    sources = [(ROOT / "docs/fixtures/glm_q2_long_cpp" / name).read_text()
               for name in ("pool.hpp", "numa_weights.hpp")]
    result = []
    for slot in range(count):
        source = sources[slot % len(sources)]
        offset = len(source) * slot // (2 * count)
        source = source[offset:] + source[:offset]
        source = (source + "\n") * (1 + length // 3000)
        task = (f"Write a complete C++17 GoogleTest suite covering {topics[slot % len(topics)]}. "
                "Include helpers, at least 16 substantive TEST cases and explanatory comments. "
                "Return one cpp code block. Keep reasoning brief.\n\nReference code excerpts:\n")
        def encode(n):
            chat = template.render(messages=[dict(role="user", content=task + source[:n])], tools=None,
                                   add_generation_prompt=True, reasoning_effort="low")
            return tokenizer.encode(chat, parse_special=True)
        if len(encode(0)) > length:
            raise ValueError("prompt token target is too short for the benchmark task")
        cycle = source
        while len(encode(len(source))) < length:
            source += cycle
        low, high = 0, len(source)
        while low < high:
            mid = (low + high + 1) // 2
            if len(encode(mid)) <= length:
                low = mid
            else:
                high = mid - 1
        result.append(encode(low))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--prompt-ids", type=Path, help="JSON array of distinct token-ID arrays")
    parser.add_argument("--cancel-slot", type=int, default=-1, help="smoke test: cancel this request after four tokens")
    parser.add_argument("--check-sequential", action="store_true",
                        help="compare each batched output with an isolated request; requires unbiased routing")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    count = cfg.get("parallel", 8)
    if not 1 <= count <= 8 or args.tokens < 1 or args.repetitions < 1:
        parser.error("invalid slots, token limit or repetition count")
    if args.check_sequential and float(cfg.get("env", {}).get("STRATA_GLM_ROUTE_AFFINITY", "0")):
        parser.error("sequential parity requires route affinity zero")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = GGUFFile(Path(cfg["model"])).metadata
    tokenizer = Tokenizer.from_gguf(Path(cfg["model"]))
    stops = {int(metadata[k]) for k in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                                       "tokenizer.ggml.eom_token_id") if k in metadata}
    inputs = json.loads(args.prompt_ids.read_text()) if args.prompt_ids else prompts(tokenizer, metadata, args.prompt_tokens, count)
    if len(inputs) != count or any(not p or len(p) + args.tokens > cfg["context"] for p in inputs):
        parser.error("prompt count or context capacity mismatch")
    (args.output / "prompts.json").write_text(json.dumps(inputs) + "\n")
    environment = child_env(cfg)
    commands = engine_args(cfg, stops)
    manifest = dict(config=cfg, engine_args=commands, output_limit=args.tokens,
                    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    decoder_sha256=hashlib.sha256(Path(cfg["exe"]).read_bytes()).hexdigest(),
                    prompt_lengths=list(map(len, inputs)), repetitions=args.repetitions,
                    environment={k: v for k, v in environment.items() if k.startswith("STRATA_")})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    for trial in range(args.repetitions):
        log = args.output / f"engine-{trial}.log"
        engine = GlmEngine(cfg["exe"], commands, cwd=cfg.get("cwd"), log=str(log), env=environment)
        if engine.batch != count:
            engine.close()
            raise RuntimeError(f"requested {count} slots, engine advertised {engine.batch}")
        barrier = threading.Barrier(count)
        def generate(slot):
            cancel = threading.Event()
            barrier.wait()
            start = time.monotonic()
            output, arrivals = [], []
            for token in engine.generate(inputs[slot], args.tokens, {}, cancel):
                if token is None:
                    continue
                output.append(token); arrivals.append(time.monotonic() - start)
                if slot == args.cancel_slot and len(output) == 4:
                    cancel.set()
            elapsed = time.monotonic() - start
            (args.output / f"trial-{trial}-slot-{slot}.txt").write_text(tokenizer.decode(output))
            return dict(slot=slot, prompt_tokens=len(inputs[slot]), generated=len(output),
                        token_ids=output, token_sha256=hashlib.sha256(json.dumps(output).encode()).hexdigest(),
                        arrivals_seconds=arrivals, wall_seconds=elapsed,
                        ended_on_stop=bool(output and output[-1] in stops), cancelled=cancel.is_set())
        try:
            start = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
                # Queue the simultaneous arrivals before admitting the first request. This avoids
                # measuring a scheduler race between a solo GEN and the initial batch admission.
                engine.ctl.acquire()
                try:
                    futures = [pool.submit(generate, slot) for slot in range(count)]
                    deadline = time.monotonic() + 30
                    while True:
                        with engine.slot_cv:
                            queued = engine.waiting == count
                        if queued:
                            break
                        if time.monotonic() >= deadline:
                            raise RuntimeError("concurrent requests did not reach the admission queue")
                        time.sleep(0.001)
                finally:
                    engine.ctl.release()
                rows = [future.result() for future in futures]
            elapsed = time.monotonic() - start
            if args.check_sequential:
                for row in rows:
                    if row["cancelled"]:
                        continue
                    batch_slots = engine.batch
                    try:
                        engine.batch = 0  # isolated GEN, including its normal MTP verification path
                        reference = [t for t in engine.generate(inputs[row["slot"]], args.tokens, {}, threading.Event())
                                     if t is not None]
                    finally:
                        engine.batch = batch_slots
                    row["sequential_token_ids"] = reference
                    row["sequential_equal"] = reference == row["token_ids"]
                    if not row["sequential_equal"]:
                        (args.output / "parity-failure.json").write_text(json.dumps(row, indent=2) + "\n")
                        raise RuntimeError(f"slot {row['slot']} differs from isolated generation")
        finally:
            engine.close()
        steps = [(int(n), int(g), float(ms)) for n, g, ms in re.findall(
            r"BATCH_DECODE active=(\d+) generated=(\d+) ms=([\d.]+)", log.read_text())]
        full = [s for s in steps if s[0] == count]
        def rate(samples):
            return sum(s[1] for s in samples) * 1000 / sum(s[2] for s in samples) if samples else None
        record = dict(trial=trial, requests=rows, wall_seconds=elapsed,
                      end_to_end_tok_s=sum(r["generated"] for r in rows) / elapsed,
                      decode_tok_s=rate(steps), full_batch_decode_tok_s=rate(full),
                      full_batch_steps=len(full), decode_steps=len(steps),
                      all_reached_limit=all(r["generated"] == args.tokens for r in rows))
        results.append(record)
        (args.output / "measurement.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "requests"}), flush=True)


if __name__ == "__main__":
    main()
