"""Measure repeated requests without reloading GLM weights between trials."""
import argparse
import hashlib
import json
import os
import pathlib
import re
import statistics
import subprocess
import time

from glm_generate import metadata_shard
from strata_tokenizer import Tokenizer


def experiment_environment():
    names = ("STRATA_OLD_IQ_MMVQ", "STRATA_GROUPED_V1", "STRATA_GLM_DIRECT_WEIGHT_UPLOAD", "STRATA_GLM_STAGE_WORKERS",
             "STRATA_GLM_PRIMARY_GROUPS", "STRATA_GLM_DEQUANT_COALESCED", "STRATA_GLM_IQ_COALESCED", "STRATA_GLM_MLA_VECTOR_GATHER", "STRATA_GLM_KDA_ROW_PARTS", "STRATA_GLM_KDA_COLUMNS", "STRATA_GLM_KDA_PREPARE",
             "STRATA_GLM_RESTORE_PREFILL_CACHE", "STRATA_GLM_CHECK_PREFILL_RESTORE",
             "STRATA_GLM_DEVICE_EXPERTS", "STRATA_GLM_DEVICE_REDUCTION", "STRATA_GLM_LAYER_GRAPHS",
             "STRATA_GLM_ACTIVE_POOLS", "STRATA_GLM_F16_BUCKETS", "STRATA_GLM_STABLE_ROUTES",
             "STRATA_GLM_PREFILL_PREALLOC", "STRATA_GLM_DEFER_COPY_WAIT", "STRATA_GLM_DEFER_PREFILL_REPAIR",
             "STRATA_NATIVE_TASKS_PER_THREAD", "STRATA_NATIVE_NUMA_LOCAL")
    prefixes = ("STRATA_GLM_", "STRATA_NATIVE_", "STRATA_POOL_")
    return {name: os.environ[name] for name in sorted(os.environ, key=lambda n: (n not in names, n))
            if name in names or name.startswith(prefixes)}


def save_result(path, result):
    """Atomically preserve completed trials, including an incomplete run."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def request_diagnostics(log):
    """Keep CPU work, GPU residency and cache hits aligned with each request."""
    result = []
    sections = re.split(r"RESPONSE_PREPARE kind=decode_cache ms=", log)
    for index, section in enumerate(sections[1:]):
        row = {"cache_prepare_ms": float(section.splitlines()[0])}
        preparation = sections[index].split("TUNE_RESET_COMPLETE\n")[-1]
        prefill = re.findall(r"PREFILL_WORK device=(\d+) transferred_bytes=(\d+) staging_ms=([\d.]+) "
                             r"actual_rows=(\d+) padded_rows=(\d+) dequant_values=(\d+)", preparation)
        if prefill:
            row["prefill_devices"] = [dict(device=int(d), transferred_bytes=int(b), staging_ms=float(t),
                                           actual_rows=int(a), padded_rows=int(p), dequant_values=int(q))
                                       for d, b, t, a, p, q in prefill]
        reset = re.findall(r"RESPONSE_PREPARE kind=request_reset ms=([\d.]+)", preparation)
        if reset: row["request_reset_ms"] = float(reset[-1])
        restored = re.findall(r"PREFILL_RESTORE groups=(\d+) repaired_slots=(\d+) checked_groups=(\d+) "
                              r"bytes=(\d+) ms=([\d.]+)", preparation)
        if restored:
            groups, repaired, checked, byte_count, elapsed = restored[-1]
            row["prefill_restore"] = dict(groups=int(groups), repaired_slots=int(repaired),
                                          checked_groups=int(checked), bytes=int(byte_count), ms=float(elapsed))
        repairs = re.findall(r"PREFILL_REPAIR device=(\d+) groups=(\d+) checked_groups=(\d+) wait_ms=([\d.]+)", preparation)
        if repairs:
            row["prefill_repairs"] = [dict(device=int(d), groups=int(g), checked_groups=int(c), wait_ms=float(t))
                                       for d, g, c, t in repairs]
        cpu = re.search(r"CPU_EXPERT gu_ms=([\d.]+) quant_ms=([\d.]+) down_ms=([\d.]+) bytes=(\d+)", section)
        if cpu:
            row["cpu_expert_ms"] = dict(zip(("gate_up", "quantize", "down"), map(float, cpu.groups()[:3])))
            row["cpu_expert_bytes"] = int(cpu[4])
        hits = re.search(r"resident expert entries=(\d+)/(\d+)", section)
        if hits:
            row["gpu_expert_hits"], row["expert_entries"] = map(int, hits.groups())
            row["gpu_expert_hit_fraction"] = int(hits[1]) / int(hits[2]) if int(hits[2]) else None
        row["gpu_devices"] = []
        for device, peak, live, tail in re.findall(
                r"GPU_DEVICE device=(\d+) peak_allocated_MiB=([\d.]+) live_MiB=([\d.]+)([^\n]*)", section):
            metrics = {"device": int(device), "peak_owned_mib": float(peak), "live_owned_mib": float(live)}
            for field in ("cuda_used_MiB", "free_MiB"):
                value = re.search(rf"\b{field}=([\d.]+)", tail)
                if value:
                    metrics[field.lower()] = float(value[1])
            row["gpu_devices"].append(metrics)
        result.append(row)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=pathlib.Path)
    ap.add_argument("text", type=pathlib.Path)
    ap.add_argument("--decoder", default="build-glm-v100/strata-glm-decode")
    ap.add_argument("--compute-sanitizer", action="store_true", help="instrument the decoder for memory validation")
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--generated-tokens", type=int, default=128)
    ap.add_argument("--context", type=int, default=131072)
    ap.add_argument("--threads", type=int, default=17)
    ap.add_argument("--gpu-budget-mib", type=int, default=32768)
    ap.add_argument("--gpu-devices", default="0,1")
    ap.add_argument("--prefill-batch", type=int, default=4096)
    ap.add_argument("--prefill-expert-cache-mib", default="auto")
    ap.add_argument("--decode-prefill-cache", action="store_true")
    ap.add_argument("--decode-cache-mib", default="0")
    ap.add_argument("--decode-graphs", action="store_true")
    ap.add_argument("--decode-cache-window", type=int, default=256)
    ap.add_argument("--decode-cache-adapt", action="store_true")
    ap.add_argument("--prefill-experts", choices=("mmq", "f16", "f16-batched"), default="mmq")
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--validation-tokens", type=int, default=0, help="save longer responses for behavioral validation")
    ap.add_argument("--validation-repetitions", type=int, default=3)
    ap.add_argument("--prefill-text", type=pathlib.Path, help="additional long raw-text prefill workload")
    ap.add_argument("--prefill-tokens", type=int, default=8192)
    ap.add_argument("--speculative", choices=("none", "mtp"), default="none")
    ap.add_argument("--draft-depth", type=int, default=1)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()
    ids = Tokenizer.from_gguf(metadata_shard(args.model)).encode(args.text.read_text(), parse_special=True)
    if not 0 < args.prompt_tokens <= len(ids) or args.generated_tokens < 2 or args.repetitions < 1 or args.threads < 1:
        ap.error("invalid prompt, generated token count, or repetitions")
    ids = ids[:args.prompt_tokens]
    if len(ids) + args.generated_tokens > args.context:
        ap.error("request exceeds context")
    encoded = ",".join(map(str, ids))
    extra_ids = None
    if args.prefill_text:
        extra_ids = Tokenizer.from_gguf(metadata_shard(args.model)).encode(args.prefill_text.read_text(), parse_special=True)
        if not 0 < args.prefill_tokens <= len(extra_ids) or args.prefill_tokens + 2 > args.context:
            ap.error("invalid additional prefill length")
        extra_ids = extra_ids[:args.prefill_tokens]
    if (args.validation_tokens < 0 or args.validation_tokens == 1 or args.validation_repetitions < 1 or
            args.validation_tokens + len(ids) > args.context):
        ap.error("invalid validation length")
    command = [str(pathlib.Path(args.decoder).resolve()), "--serve", str(args.model.resolve()),
               str(args.context), "4096", str(args.threads), "0", str(args.prefill_batch), "-1",
               str(args.gpu_budget_mib), "0", "0", args.speculative, str(args.draft_depth), "auto", "0", args.gpu_devices,
               args.prefill_expert_cache_mib, "1"]
    command.extend((str(int(args.decode_prefill_cache)), args.decode_cache_mib, str(int(args.decode_graphs)), str(args.decode_cache_window), str(int(args.decode_cache_adapt))))
    command.append(args.prefill_experts)
    with pathlib.Path(command[0]).open("rb") as binary:
        decoder_sha256 = hashlib.file_digest(binary, "sha256").hexdigest()
    sanitizer_path = args.output.with_suffix(".sanitizer.log") if args.compute_sanitizer else None
    if sanitizer_path:
        if args.output.exists() or sanitizer_path.exists():
            ap.error("instrumented output already exists; choose a fresh --output path")
        # Tool output is not part of the resident decoder's line protocol.
        sanitizer_options = []
        sync_limit = int(os.environ.get("STRATA_GLM_SANITIZER_SYNC_LIMIT", "0"))
        if not 0 <= sync_limit <= 1000000:
            ap.error("sanitizer sync limit must be 0..1000000")
        if sync_limit:
            sanitizer_options = ["--force-synchronization-limit", str(sync_limit)]
        command = ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "97"] + sanitizer_options + [
                   "--log-file", str(sanitizer_path)] + command
    log_path = args.output.with_suffix(".log")
    rows = []
    prefill_rows = []
    validation = None
    result = dict(configuration={k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()},
                  command=command, decoder_sha256=decoder_sha256, startup_seconds=None,
                  prompt_sha256=hashlib.sha256(encoded.encode()).hexdigest(), trials=rows,
                  validation=None, additional_prefill_trials=prefill_rows, complete=False,
                  instrumented=args.compute_sanitizer,
                  sanitizer_log_path=str(sanitizer_path) if sanitizer_path else None,
                  execution=dict(workers=args.threads),
                  environment=experiment_environment(),
                  additional_prefill_sha256=hashlib.sha256(",".join(map(str, extra_ids)).encode()).hexdigest() if extra_ids else None,
                  log_path=str(log_path),
                  timing_scope="warm resident requests; prefill includes CPU preparation; decode advances generated_tokens-1 steps; excludes startup and warmup")

    def save():
        if rows:
            result.update(median_prefill_tok_s=statistics.median(r["prefill_tok_s"] for r in rows),
                          median_decode_tok_s=statistics.median(r["decode_tok_s"] for r in rows),
                          median_response_ms=statistics.median(r["response_ms"] for r in rows),
                          repetitions_identical=all(r["token_ids"] == rows[0]["token_ids"] for r in rows))
        result["validation"] = validation
        result["additional_prefill_median_tok_s"] = statistics.median(r["prefill_tok_s"] for r in prefill_rows) if prefill_rows else None
        result["request_diagnostics"] = request_diagnostics(log_path.read_text()) if log_path.exists() else []
        save_result(args.output, result)

    save()
    started = time.monotonic()
    with log_path.open("w") as log, subprocess.Popen(command, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1) as process:
        def read():
            line = process.stdout.readline().strip()
            if not line:
                raise RuntimeError(f"decoder exited; see {log_path}")
            if line.startswith("ERR "):
                raise RuntimeError(line)
            return line

        def request(count, prompt_ids):
            process.stdin.write(f"GEN {count} {','.join(map(str, prompt_ids))}\n")
            process.stdin.flush()
            generated = []
            while True:
                line = read()
                if line.startswith("T "):
                    generated.append(int(line.split()[1]))
                if line.startswith("DONE "):
                    fields = line.split()
                    if int(fields[1]) != count or int(fields[2]) != len(prompt_ids) or len(generated) != count:
                        raise RuntimeError(f"unexpected response: {line}")
                    prefill, decode = map(float, fields[3:5])
                    return dict(prefill_ms=prefill, decode_ms=decode,
                                prefill_tok_s=len(prompt_ids) * 1000 / prefill,
                                decode_tok_s=(count - 1) * 1000 / decode,
                                response_ms=prefill + decode, token_ids=generated)

        try:
            while True:
                ready = read()
                if ready.startswith("READY "): break
            result["prefill_includes_reset"] = "prefill-includes-reset" in ready.split()
            startup = time.monotonic() - started
            result["startup_seconds"] = startup
            save()
            print(f"READY startup_s={startup:.3f}", flush=True)
            for trial in range(args.repetitions + 1):
                # Populate prefill caches without an extra full decode warmup.
                count = args.generated_tokens if trial else (3 if args.decode_graphs else 2)
                row = request(count, ids)
                if trial:
                    rows.append(row)
                    save()
                print(f"TRIAL {trial} prefill_tok_s={row['prefill_tok_s']:.3f} "
                      f"decode_tok_s={row['decode_tok_s']:.3f}", flush=True)
            if args.validation_tokens:
                result["validation_trials"] = []
                for trial in range(args.validation_repetitions):
                    row = request(args.validation_tokens, ids)
                    result["validation_trials"].append(row)
                    validation = result["validation_trials"][0]
                    result["median_validation_decode_tok_s"] = statistics.median(
                        r["decode_tok_s"] for r in result["validation_trials"])
                    save()
                    if row["token_ids"] != validation["token_ids"]:
                        raise RuntimeError("repeated validation requests produced different token IDs")
                    print(f"VALIDATION trial={trial} decode_tok_s={row['decode_tok_s']:.3f}", flush=True)
            if extra_ids:
                for trial in range(args.repetitions + 1):
                    row = request(2, extra_ids)
                    if trial:
                        prefill_rows.append(row)
                        save()
                    print(f"PREFILL_TRIAL {trial} tokens={len(extra_ids)} tok_s={row['prefill_tok_s']:.3f}", flush=True)
            process.stdin.write("QUIT\n")
            process.stdin.flush()
            process.wait(timeout=30)
            if process.returncode:
                raise RuntimeError(f"decoder failed; see {log_path}")
            if sanitizer_path:
                report = sanitizer_path.read_text() if sanitizer_path.exists() else ""
                summaries = re.findall(r"ERROR SUMMARY: (\d+) errors", report)
                tracking_failed = bool(re.search(r"Internal Sanitizer Error|didn.t track the launch|Errors might go undetected", report, re.I))
                result["sanitizer_clean"] = bool(summaries) and all(int(n) == 0 for n in summaries) and not tracking_failed
                if not result["sanitizer_clean"]:
                    raise RuntimeError(f"missing clean sanitizer summary; see {sanitizer_path}")
            if not result["repetitions_identical"]:
                raise RuntimeError("repeated requests produced different token IDs")
            if validation and validation["token_ids"][:args.generated_tokens] != rows[0]["token_ids"][:args.validation_tokens]:
                raise RuntimeError("validation prefix differs from repeated requests")
        except BaseException as error:
            result["error"] = str(error)
            save()
            raise
        finally:
            stop_process(process)
    result["complete"] = True
    save()


if __name__ == "__main__":
    main()
