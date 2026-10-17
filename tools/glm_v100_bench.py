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
import sys
import threading

from glm_generate import metadata_shard
from strata_tokenizer import Tokenizer


def workload_tokenizer(model):
    if model.is_dir():
        from glm_artifact import NativeTokenizer
        return NativeTokenizer(model)
    return Tokenizer.from_gguf(metadata_shard(model))


def native_identity(model):
    """Hash the small files that select and describe the indexed weight set."""
    if not model.is_dir():
        return None
    index = json.loads((model / "model.safetensors.index.json").read_text())
    files = ("config.json", "model.safetensors.index.json", "tokenizer.json", "chat_template.jinja")
    return dict(files_sha256={name: hashlib.sha256((model / name).read_bytes()).hexdigest()
                              for name in files},
                indexed_file_bytes=sum((model / shard).stat().st_size for shard in set(index["weight_map"].values())),
                indexed_tensors=len(index["weight_map"]),
                scope="metadata hashes and indexed file lengths; weight payloads are not hashed")


def native_process_sample(pid):
    """Cheap status/fault counters; never walk weight page tables during timing."""
    try:
        root = pathlib.Path(f"/proc/{pid}")
        status = dict(line.split(":", 1) for line in (root / "status").read_text().splitlines() if ":" in line)
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        return dict(rss_bytes=int(status["VmRSS"].split()[0])*1024,
                    swap_bytes=int(status["VmSwap"].split()[0])*1024,
                    minor_faults=int(fields[7]), major_faults=int(fields[9]))
    except (OSError, ValueError, KeyError, IndexError):
        return {}


def prompt_ids(path):
    fields = re.split(r"[,\s]+", pathlib.Path(path).read_text().strip())
    if not fields or any(not value.isdigit() for value in fields):
        raise ValueError("prompt IDs must be nonnegative integers separated by commas or whitespace")
    return [int(value) for value in fields]


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


def diagnostic_environment(environment=None):
    # Presence enables STEP_TRACE even for "0"; conservatively keep every
    # explicitly requested diagnostic out of placement/speed qualification.
    names = ("STRATA_GLM_STEP_TRACE", "STRATA_GLM_TASK_TRACE", "STRATA_GLM_TASK_TRACE_SYNC",
             "STRATA_GLM_ROUTING_EVENTS", "STRATA_GLM_STATE_TRACE_DIR", "STRATA_GLM_CHECK_RESIDENT_ROWS",
             "STRATA_GLM_MIXED_PROFILE", "STRATA_GLM_MIXED_DECODE_PROFILE",
             "STRATA_GLM_MIXED_PROFILE_DECODE", "STRATA_GLM_MTP_TIMING",
             "STRATA_GLM_LAYER_VERIFY_CACHE")
    values = os.environ if environment is None else environment
    return [name for name in names if name in values]


def cpu_diagnostics(text):
    cpu = re.search(r"CPU_EXPERT gu_ms=([\d.]+) quant_ms=([\d.]+) down_ms=([\d.]+) "
                    r"(?:layer_flow_ms=[\d.]+ )?bytes=(\d+)", text)
    if not cpu: return {}
    result = dict(cpu_expert_ms=dict(zip(("gate_up", "quantize", "down"), map(float,cpu.groups()[:3]))),
                  cpu_expert_bytes=int(cpu[4]))
    flow = re.search(r"CPU_EXPERT[^\n]*layer_flow_ms=([\d.]+)", text)
    if flow: result['cpu_expert_ms']['layer_flow'] = float(flow[1])
    return result


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
        row.update(cpu_diagnostics(section))
        native = re.findall(r"FP4_ROUTES device=(\d+) hits=(\d+) misses=(\d+) uploaded_bytes=(\d+) "
                            r"cache_bytes=(\d+) mma=(\d+) reference=(\d+) prefill=(\d+)", section)
        if native:
            keys = ("device", "hits", "misses", "uploaded_bytes", "cache_bytes", "mma", "reference", "prefill")
            row["native_devices"] = [dict(zip(keys, map(int, values))) for values in native]
            row["native_counter_scope"] = "cumulative since model load"
        cpu = re.findall(r"FP4_CPU expert_groups=(\d+)", section)
        if cpu: row["native_cpu_expert_groups"] = int(cpu[-1])
        hits = re.search(r"resident expert entries=(\d+)/(\d+)", section)
        if hits:
            row["gpu_expert_hits"], row["expert_entries"] = map(int, hits.groups())
            row["gpu_expert_hit_fraction"] = int(hits[1]) / int(hits[2]) if int(hits[2]) else None
        mtp = re.search(r"LAYER_MTP proposed=(\d+) accepted=(\d+) depth=(\d+)", section)
        if mtp:
            proposed, accepted, depth = map(int, mtp.groups())
            row["layer_mtp"] = dict(proposed=proposed, accepted=accepted, rejected=proposed-accepted,
                                    depth=depth, acceptance_fraction=accepted/proposed if proposed else None)
        layer_work = []
        for stage, device, count, wall, draft, detail in re.findall(
                r"LAYER_WORK stage=(\d+) device=(\d+) decode_rows=(\d+) decode_wall_ms=([\d.]+) "
                r"draft_wall_ms=([\d.]+)[^\n]*\n(.*?)(?=\nLAYER_WORK |\Z)", section, re.S):
            item = dict(stage=int(stage), device=int(device), decode_rows=int(count),
                        decode_wall_ms=float(wall), draft_wall_ms=float(draft))
            hit = re.search(r"resident expert entries=(\d+)/(\d+)", detail)
            if hit:
                item.update(gpu_expert_hits=int(hit[1]), expert_entries=int(hit[2]))
            item.update(cpu_diagnostics(detail))
            traced = re.search(r"STEP_TRACE ([^\n]+)", detail)
            if traced:
                item['step_trace']={k:float(v) for k,v in re.findall(r'(\w+)=([\d.]+)',traced[1])}
            layer_work.append(item)
        if layer_work:
            row["layer_stages"] = layer_work
            measured_cpu=[stage for stage in layer_work if 'cpu_expert_bytes' in stage]
            if measured_cpu:
                row['cpu_expert_bytes']=sum(stage['cpu_expert_bytes'] for stage in measured_cpu)
                row['cpu_expert_ms']={key:sum(stage['cpu_expert_ms'].get(key,0) for stage in measured_cpu)
                                      for key in ('gate_up','quantize','down','layer_flow')}
                row['cpu_expert_scope']='sum of stage work; overlapping stage times are not elapsed decode time'
            measured = [item for item in layer_work if "expert_entries" in item]
            if measured:
                hits = sum(item["gpu_expert_hits"] for item in measured)
                entries = sum(item["expert_entries"] for item in measured)
                row.update(gpu_expert_hits=hits, expert_entries=entries,
                           gpu_expert_hit_fraction=hits / entries if entries else None)
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


def executable_fingerprint(path):
    value = 14695981039346656037
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(131072), b""):
            for byte in chunk:
                value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=pathlib.Path)
    ap.add_argument("text", type=pathlib.Path)
    ap.add_argument("--decoder", default="build-glm-v100/strata-glm-decode")
    ap.add_argument("--fp4-kernel", choices=("reference", "sm70"),
                    help="opt-in native ModelOpt FP4 implementation")
    ap.add_argument("--compute-sanitizer", action="store_true", help="instrument the decoder for memory validation")
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--prompt-ids", action="store_true", help="the prompt file contains frozen token IDs")
    ap.add_argument("--generated-tokens", type=int, default=128)
    ap.add_argument("--context", type=int, default=131072)
    ap.add_argument("--threads", type=int, default=17)
    ap.add_argument("--cpu-affinity", choices=("auto", "numa", "none"), default="auto")
    ap.add_argument("--gpu-budget-mib", type=int, default=32768)
    ap.add_argument("--gpu-devices", default="0,1")
    ap.add_argument("--gpu-layout", choices=("legacy", "layer"), default="legacy")
    ap.add_argument("--layer-split", default="auto")
    ap.add_argument("--layer-pipeline", choices=("none", "mtp"), default="none")
    ap.add_argument("--layer-memory", choices=("interleave", "socket"))
    ap.add_argument("--layer-plan", type=pathlib.Path)
    ap.add_argument("--layer-workers", help="comma-separated worker counts in stage order")
    ap.add_argument("--layer-cache-mib", help="comma-separated cache budgets in stage order")
    ap.add_argument("--layer-pin-coordinator", action="store_true")
    ap.add_argument("--prefill-batch", type=int, default=4096)
    ap.add_argument("--prefill-expert-cache-mib", default="auto")
    ap.add_argument("--decode-prefill-cache", action="store_true")
    ap.add_argument("--decode-cache-mib", default="0")
    ap.add_argument("--decode-graphs", action="store_true")
    ap.add_argument("--decode-cache-window", type=int, default=256)
    ap.add_argument("--decode-cache-adapt", action="store_true")
    ap.add_argument("--prefill-experts", choices=("mmq", "f16", "f16-batched"), default="mmq")
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--full-warmups", type=int, default=0, help="full generated-token requests excluded before timed trials")
    ap.add_argument("--validation-tokens", type=int, default=0, help="save longer responses for behavioral validation")
    ap.add_argument("--validation-repetitions", type=int, default=3)
    ap.add_argument("--prefill-text", type=pathlib.Path, help="additional long raw-text prefill workload")
    ap.add_argument("--prefill-tokens", type=int, default=8192)
    ap.add_argument("--speculative", choices=("none", "mtp"), default="none")
    ap.add_argument("--draft-depth", type=int, default=1)
    ap.add_argument("--worker-decoder", type=pathlib.Path, help="opt-in CUDA+CUDA expert worker binary")
    ap.add_argument("--primary-device", type=int, default=0)
    ap.add_argument("--worker-device", type=int, default=1)
    ap.add_argument("--worker-cache-mib", type=int, default=24576)
    ap.add_argument("--worker-gpu-budget-mib", type=int, default=32768)
    ap.add_argument("--worker-combine", choices=("strict","device","native"), default="strict")
    ap.add_argument("--primary-cpus")
    ap.add_argument("--worker-cpus")
    ap.add_argument("--stop-ids", default="-1", help="real stop IDs; -1 retains fixed-length diagnostic timing")
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()
    modelopt = False
    if args.model.is_dir():
        from glm_artifact import load_metadata
        modelopt = load_metadata(args.model).get("strata.native_codec") == "modelopt"
    if args.fp4_kernel and not modelopt:
        ap.error("--fp4-kernel requires a native ModelOpt checkpoint")
    if modelopt:
        if args.gpu_layout != "legacy" or args.worker_decoder:
            ap.error("native ModelOpt uses its own compressed expert cache across --gpu-devices")
        if (args.decode_prefill_cache or args.decode_cache_mib != "0" or args.decode_cache_adapt or
                args.prefill_expert_cache_mib not in ("0", "auto")):
            ap.error("native ModelOpt requires the native cache without GGUF cache overrides")
        # The native cache is owned by the FP4 executor; the GGUF cache's auto
        # setting has no meaning here. All existing GGUF defaults stay intact.
        args.prefill_expert_cache_mib = "0"
    if args.gpu_layout!="layer" and (args.layer_workers or args.layer_cache_mib or args.layer_pin_coordinator):
        ap.error("per-stage budgets and coordinator affinity require --gpu-layout=layer")
    ids = (prompt_ids(args.text) if args.prompt_ids else
           workload_tokenizer(args.model).encode(args.text.read_text(), parse_special=True))
    if not 0 < args.prompt_tokens <= len(ids) or args.generated_tokens < 2 or args.repetitions < 1 or args.threads < 1:
        ap.error("invalid prompt, generated token count, or repetitions")
    ids = ids[:args.prompt_tokens]
    if len(ids) + args.generated_tokens > args.context:
        ap.error("request exceeds context")
    encoded = ",".join(map(str, ids))
    extra_ids = None
    if args.prefill_text:
        extra_ids = workload_tokenizer(args.model).encode(args.prefill_text.read_text(), parse_special=True)
        if not 0 < args.prefill_tokens <= len(extra_ids) or args.prefill_tokens + 2 > args.context:
            ap.error("invalid additional prefill length")
        extra_ids = extra_ids[:args.prefill_tokens]
    if args.full_warmups < 0: ap.error("full-warmups must be nonnegative")
    if (args.validation_tokens < 0 or args.validation_tokens == 1 or args.validation_repetitions < 1 or
            args.validation_tokens + len(ids) > args.context):
        ap.error("invalid validation length")
    command = [str(pathlib.Path(args.decoder).resolve()), "--serve", str(args.model.resolve()),
               str(args.context), "4096", str(args.threads), "0", str(args.prefill_batch), args.stop_ids,
               str(args.gpu_budget_mib), "0", "0", args.speculative, str(args.draft_depth), args.cpu_affinity, "0", args.gpu_devices,
               args.prefill_expert_cache_mib, "1"]
    command.extend((str(int(args.decode_prefill_cache)), args.decode_cache_mib, str(int(args.decode_graphs)), str(args.decode_cache_window), str(int(args.decode_cache_adapt))))
    command.append(args.prefill_experts)
    if modelopt: command.append(f"--fp4-kernel={args.fp4_kernel or 'reference'}")
    if args.gpu_layout == "layer":
        if args.prefill_expert_cache_mib != "0" or args.decode_prefill_cache or args.decode_cache_mib in ("auto", "extend") or args.prefill_experts != "mmq":
            ap.error("layer benchmark requires fixed decode cache, mmq prefill and no legacy prefill cache")
        # Same positional padding as serve/server.py; argv[29] starts named options.
        command.extend(("4k", "", "", "auto", "--gpu-layout=layer",
                        f"--layer-split={args.layer_split}", f"--layer-pipeline={args.layer_pipeline}"))
        if args.layer_memory: command.append(f"--layer-memory={args.layer_memory}")
        if args.layer_plan: command.append(f"--layer-plan={args.layer_plan}")
        if args.layer_workers: command.append(f"--layer-workers={args.layer_workers}")
        if args.layer_cache_mib: command.append(f"--layer-cache-mib={args.layer_cache_mib}")
        if args.layer_pin_coordinator: command.append("--layer-pin-coordinator")
    with pathlib.Path(command[0]).open("rb") as binary:
        decoder_sha256 = hashlib.file_digest(binary, "sha256").hexdigest()
    worker_sha256 = None
    if args.worker_decoder:
        if args.gpu_layout != "legacy" or args.gpu_devices != "0" or args.compute_sanitizer:
            ap.error("CUDA+CUDA worker requires legacy layout, logical --gpu-devices=0 and separate sanitizer checks")
        with args.worker_decoder.open("rb") as binary:
            worker_sha256 = hashlib.file_digest(binary, "sha256").hexdigest()
        launch = [sys.executable, str(pathlib.Path(__file__).with_name("glm_mixed_run.py")),
                  "--primary-worker-mode", "--primary", command[0], "--worker", str(args.worker_decoder.resolve()),
                  "--model", str(args.model.resolve()), "--primary-device", str(args.primary_device),
                  "--worker-device", str(args.worker_device), "--worker-cache-mib", str(args.worker_cache_mib),
                  "--worker-gpu-budget-mib", str(args.worker_gpu_budget_mib),
                  "--combine", args.worker_combine,
                  "--report", str(args.output.with_suffix(".memory.json"))]
        for role in ("primary", "worker"):
            mask = getattr(args, role + "_cpus")
            if mask: launch += ["--" + role + "-cpus", mask]
        command = launch + ["--"] + command[3:]
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
    hardware = {}
    try:
        hardware["gpus"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL).strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        hardware["gpus"] = []
    cpuinfo = pathlib.Path("/proc/cpuinfo")
    hardware["cpu"] = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                             if line.startswith("model name")), "") if cpuinfo.exists() else ""
    hardware["node"] = os.uname().nodename if hasattr(os, "uname") else ""
    hardware["job_id"] = os.environ.get("SLURM_JOB_ID", "")
    result = dict(hardware=hardware, native_model=native_identity(args.model), configuration={k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()},
                  command=command, decoder_sha256=decoder_sha256, decoder_fnv64=executable_fingerprint(args.decoder), worker_sha256=worker_sha256, startup_seconds=None,
                  prompt_sha256=hashlib.sha256(encoded.encode()).hexdigest(), trials=rows,
                  validation=None, additional_prefill_trials=prefill_rows, complete=False,
                  instrumented=args.compute_sanitizer or bool(diagnostic_environment()),
                  diagnostic_environment=diagnostic_environment(),
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
        monitor_stop = threading.Event()
        memory = dict(peak_rss_bytes=0, peak_swap_bytes=0, samples=0)
        def monitor_memory():
            while not monitor_stop.is_set():
                sample = native_process_sample(process.pid)
                if sample:
                    memory["samples"] += 1
                    memory["peak_rss_bytes"] = max(memory["peak_rss_bytes"], sample["rss_bytes"])
                    memory["peak_swap_bytes"] = max(memory["peak_swap_bytes"], sample["swap_bytes"])
                monitor_stop.wait(0.5)
        monitor = threading.Thread(target=monitor_memory, daemon=True) if modelopt else None
        if monitor:
            result["native_memory"] = memory
            monitor.start()
        def read():
            line = process.stdout.readline().strip()
            if not line:
                raise RuntimeError(f"decoder exited; see {log_path}")
            if line.startswith("ERR "):
                raise RuntimeError(line)
            return line

        def request(count, prompt_ids):
            before = native_process_sample(process.pid) if modelopt else {}
            process.stdin.write(f"GEN {count} {','.join(map(str, prompt_ids))}\n")
            process.stdin.flush()
            generated = []
            while True:
                line = read()
                if line.startswith("T "):
                    generated.append(int(line.split()[1]))
                if line.startswith("DONE "):
                    fields = line.split()
                    actual = int(fields[1])
                    if (int(fields[2]) != len(prompt_ids) or len(generated) != actual or not 1 <= actual <= count or
                            (actual != count and args.stop_ids == "-1")):
                        raise RuntimeError(f"unexpected response: {line}")
                    prefill, decode = map(float, fields[3:5])
                    row = dict(prefill_ms=prefill, decode_ms=decode,
                                prefill_tok_s=len(prompt_ids) * 1000 / prefill,
                                decode_tok_s=(actual - 1) * 1000 / decode if decode > 0 else 0,
                                generated_tokens=actual, requested_tokens=count, stop_reason=fields[5] if len(fields)>5 else None,
                                response_ms=prefill + decode, token_ids=generated)
                    if modelopt:
                        after = native_process_sample(process.pid)
                        row["native_memory_after"] = after
                        row["major_faults"] = after.get("major_faults", 0)-before.get("major_faults", 0)
                        row["minor_faults"] = after.get("minor_faults", 0)-before.get("minor_faults", 0)
                    return row

        try:
            while True:
                ready = read()
                if ready.startswith("READY "): break
            visible_gpus = re.findall(r"LAYER_GPU device=(\d+) name=([^\n]+)", log_path.read_text())
            if visible_gpus:
                hardware["gpus"] = [""] * (max(int(device) for device, _ in visible_gpus) + 1)
                for device, name in visible_gpus: hardware["gpus"][int(device)] = name
            result["prefill_includes_reset"] = "prefill-includes-reset" in ready.split()
            startup = time.monotonic() - started
            result["startup_seconds"] = startup
            save()
            print(f"READY startup_s={startup:.3f}", flush=True)
            result["full_warmup_trials"] = []
            for trial in range(args.full_warmups):
                row = request(args.generated_tokens, ids)
                result["full_warmup_trials"].append(row)
                save()
                print(f"FULL_WARMUP {trial} decode_tok_s={row['decode_tok_s']:.3f}", flush=True)
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
            if modelopt and (memory["peak_swap_bytes"] or any(row.get("major_faults", 0) for row in rows)):
                raise RuntimeError("native FP4 timed requests swapped or faulted weight pages; residency qualification failed")
            if validation and validation["token_ids"][:args.generated_tokens] != rows[0]["token_ids"][:args.validation_tokens]:
                raise RuntimeError("validation prefix differs from repeated requests")
        except BaseException as error:
            result["error"] = str(error)
            save()
            raise
        finally:
            monitor_stop.set()
            if monitor: monitor.join(timeout=2)
            stop_process(process)
    result["complete"] = True
    save()


if __name__ == "__main__":
    main()
