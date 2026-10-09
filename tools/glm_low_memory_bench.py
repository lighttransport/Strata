"""Measure a GLM configuration inside a Linux memory cgroup, including SSD misses.

GPU capacity is enforced by the decoder's allocation budget. This is a capacity
simulation on the installed GPU, not a measurement of another GPU's compute speed.
"""
import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time


def cache_bytes(path):
    """Count cached file pages with mincore without faulting them into memory."""
    lib = ctypes.CDLL(None, use_errno=True)
    lib.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                         ctypes.c_int, ctypes.c_int, ctypes.c_long]
    lib.mmap.restype = ctypes.c_void_p
    lib.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    lib.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    page = os.sysconf("SC_PAGESIZE")
    total = 0
    with path.open("rb") as file:
        size = path.stat().st_size
        for offset in range(0, size, 1 << 30):
            length = min(1 << 30, size - offset)
            address = lib.mmap(None, length, 1, 2, file.fileno(), offset)
            if address == ctypes.c_void_p(-1).value:
                raise OSError(ctypes.get_errno(), "mincore mmap failed")
            try:
                vector = (ctypes.c_ubyte * ((length + page - 1) // page))()
                if lib.mincore(address, length, vector):
                    raise OSError(ctypes.get_errno(), "mincore failed")
                total += sum(x & 1 for x in vector) * page
            finally:
                lib.munmap(address, length)
    return min(total, size)


def model_files(config):
    model = Path(config["model"]).resolve()
    match = re.match(r"(.*)-\d{5}-of-\d{5}\.gguf$", model.name)
    files = sorted(model.parent.glob(match[1] + "-*-of-*.gguf")) if match else [model]
    if config.get("expert_pack"):
        files.append(Path(config["expert_pack"]).resolve())
    return list(dict.fromkeys(files))


def clear_model_cache(config):
    # Do not disturb a live decoder using these files.
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = proc.read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        if argv and Path(os.fsdecode(argv[0])).name.startswith("strata-glm-decode"):
            raise RuntimeError("another GLM decoder is active; cannot clear model cache")
    records = []
    for path in model_files(config):
        with path.open("rb") as file:
            os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        records.append(dict(path=str(path), bytes=path.stat().st_size,
                            cached_bytes_after=cache_bytes(path)))
    cached = sum(row["cached_bytes_after"] for row in records)
    if cached > 256 * 2**20:
        raise RuntimeError(f"model cache still holds {cached / 2**20:.1f} MiB outside the new cgroup")
    return records


def parse_log(log):
    prefill = [dict(trial=int(i), tokens=int(n), batch=int(b), milliseconds=float(ms),
                    tokens_per_second=float(rate)) for i, n, b, ms, rate in re.findall(
        r"PREFILL trial=(\d+) tokens=(\d+) batch=(\d+) ms=([\d.]+) tok_s=([\d.]+)", log)]
    decode = [dict(kind="single", tokens=int(n), milliseconds=float(ms), tokens_per_second=float(rate))
              for n, ms, rate in re.findall(r"DECODE steps=(\d+) ms=([\d.]+) tok_s=([\d.]+)", log)]
    decode += [dict(kind=kind, tokens=int(n), milliseconds=float(ms), tokens_per_second=float(rate))
               for kind, n, ms, rate in re.findall(
        r"SPECULATIVE source=(\w+) generated=(\d+).*? ms=([\d.]+) tok_s=([\d.]+)", log)]
    def records(tag):
        return [{k: int(v) if v.isdigit() else float(v) for k, v in re.findall(r"(\w+)=([0-9.eE+-]+)(?:\s|$)", line)}
                for line in log.splitlines() if line.startswith(tag + " ")]
    return dict(prefill=prefill, decode=decode,
        cpu_expert=records("CPU_EXPERT"), mtp_timing=records("MTP_TIMING"),
        mtp_positions=records("MTP_POSITION"), step_trace=records("STEP_TRACE"),
        decode_cache=records("DECODE_CACHE"), speculation=records("SPECULATIVE"),
        affinity_quality=records("AFFINITY_QUALITY"), lookup_parity=records("LOOKUP_PARITY"),
        median_prefill_tok_s=statistics.median(r["tokens_per_second"] for r in prefill) if prefill else None,
        median_decode_tok_s=statistics.median(r["tokens_per_second"] for r in decode) if decode else None,
        peak_allocated_gpu_mib=max(map(float, re.findall(
            r"GPU(?:_DEVICE device=\d+)? peak_allocated_MiB=([\d.]+)", log)), default=None),
        gpu_live=[dict(device=int(d), tags=dict((k, float(v)) for k, v in re.findall(r"(\w+)_MiB=([\d.]+)", tags)))
                  for d, tags in re.findall(r"GPU_LIVE device=(\d+) (.*)", log)],
        page_faults=[dict(trial=int(i), major=int(a), minor=int(b)) for i, a, b in re.findall(
            r"FAULTS trial=(\d+) major=(\d+) minor=(\d+)", log)])


def group_stats(group):
    values = {}
    for name in ("memory.current", "memory.peak", "memory.swap.current", "memory.max", "memory.swap.max"):
        values[name] = int((group / name).read_text().strip())
    for name in ("memory.events", "memory.stat"):
        values[name] = {key: int(value) for key, value in
                        (row.split() for row in (group / name).read_text().splitlines())}
    io = group / "io.stat"
    values["io.stat"] = {row.split()[0]: dict((k, int(v)) for k, v in
                           (item.split("=") for item in row.split()[1:]))
                         for row in io.read_text().splitlines()} if io.exists() else {}
    return values


def process_memory_maps(pid):
    """Record physical mapping accounting when RSS disagrees with a cgroup."""
    root = Path(f"/proc/{pid}")
    rollup = {key: int(value) * 1024 for key, value in re.findall(
        r"^(\w+):\s+(\d+) kB$", (root / "smaps_rollup").read_text(), re.M)}
    fields = {"Rss", "Pss", "Private_Clean", "Private_Dirty", "Shared_Clean",
              "Shared_Dirty", "Anonymous", "Swap", "Locked"}
    mappings = {}
    current = None
    for line in (root / "smaps").read_text().splitlines():
        if re.match(r"^[0-9a-f]+-[0-9a-f]+ ", line):
            parts = line.split(maxsplit=5)
            current = parts[5] if len(parts) > 5 else "[anonymous]"
            mappings.setdefault(current, {key: 0 for key in fields})
        elif current is not None:
            match = re.match(r"^(\w+):\s+(\d+) kB$", line)
            if match and match[1] in fields:
                mappings[current][match[1]] += int(match[2]) * 1024
    return dict(rollup_bytes=rollup, mappings_bytes=mappings)


def gpu_usage(pid, text):
    return sum(int(row.split(",")[1]) for row in text.splitlines()
               if row.split(",")[0].strip() == str(pid))


def gpu_capacity_passed(capacity_mib, process_mib, total_mib, reserve_mib):
    # Checking Device's allocator alone misses driver/library allocations;
    # checking the process alone misses the desktop and other CUDA clients.
    return max(process_mib, total_mib) + reserve_mib <= capacity_mib


def gpu_used_limit(capacity_mib, physical_mib, reserve_mib, requested_limit=None, reserved_mib=0):
    capacity = min(capacity_mib, physical_mib) - reserved_mib
    limit = capacity - reserve_mib
    if requested_limit is not None:
        limit = min(limit, requested_limit - reserved_mib)
    if limit <= 0:
        raise ValueError("GPU capacity must exceed its physical reserve")
    return capacity, limit


def gpu_snapshot(text):
    total, reserved, free, used = (int(value.strip()) for value in text.strip().split(","))
    if min(total, reserved, free, used) < 0 or reserved >= total:
        raise ValueError("invalid GPU memory snapshot")
    return dict(total_mib=total, reserved_mib=reserved, free_mib=free, used_mib=used,
                usable_mib=total-reserved)


def hip_gpu_device(config):
    if config.get("hip_sysfs_device"):
        return Path(config["hip_sysfs_device"]).resolve()
    devices = [p for p in Path("/sys/class/drm").glob("card[0-9]*/device")
               if re.fullmatch(r"card\d+", p.parent.name) and
               (p / "vendor").read_text().strip() == "0x1002" and
               (p / "mem_info_vram_total").exists() and
               int((p / "mem_info_vram_total").read_text()) > 0]
    if len(devices) != 1:
        raise ValueError("HIP memory guard needs one AMD GPU or an explicit hip_sysfs_device")
    return devices[0].resolve()


def hip_gpu_snapshot(device):
    total = int((device / "mem_info_vram_total").read_text())
    used = int((device / "mem_info_vram_used").read_text())
    if total <= 0 or used < 0 or used > total:
        raise ValueError("invalid AMD GPU memory snapshot")
    mib = 2**20
    result = dict(total_mib=total // mib, reserved_mib=0, free_mib=(total-used) // mib,
                  used_mib=(used+mib-1) // mib, usable_mib=total // mib)
    gtt = device / "mem_info_gtt_used"
    if gtt.exists():
        result["gtt_used_mib"] = (int(gtt.read_text())+mib-1) // mib
    # Optional evidence for interpreting timings; missing clock attributes do
    # not weaken or disable the VRAM guard.
    telemetry = {}
    for name in ("power_dpm_force_performance_level", "pp_dpm_sclk", "pp_dpm_mclk",
                 "gpu_busy_percent", "mem_busy_percent"):
        try:
            telemetry[name] = (device / name).read_text().strip()
        except OSError:
            pass
    if telemetry:
        result["performance"] = telemetry
    return result


def read_gpu(config, pid=None):
    if config.get("backend", "cuda") == "hip":
        # Global VRAM includes the engine, driver and every other GPU client.
        # sysfs cannot attribute it to this PID, so do not invent a process value.
        return None, hip_gpu_snapshot(hip_gpu_device(config))
    if config.get("backend", "cuda") != "cuda":
        raise ValueError("unsupported GPU backend for the memory guard")
    snapshot = gpu_snapshot(subprocess.check_output(["nvidia-smi", "--id=0",
        "--query-gpu=memory.total,memory.reserved,memory.free,memory.used",
        "--format=csv,noheader,nounits"], text=True, timeout=3))
    process = None
    if pid is not None:
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits"], text=True, timeout=3)
        process = gpu_usage(pid, apps)
    return process, snapshot


def eval_counts(path):
    total = 0
    counts = []
    for line in path.read_text().splitlines():
        if line.strip():
            ids = [int(i) for i in line.split(",")]
            if len(ids) < 2:
                raise ValueError("evaluation sequence needs at least two tokens")
            total += len(ids) - 1
            counts.append(total)
    if not counts:
        raise ValueError("empty evaluation corpus")
    return counts


def eval_complete(rows, counts):
    return ([row.get("sequence") for row in rows] == list(range(1, len(counts) + 1)) and
            [row.get("tokens") for row in rows] == counts)


def worker(args):
    from glm_q2_coding_bench import process_cpu_snapshot
    cfg = json.loads(args.config.read_text())
    env = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_GLM_", "STRATA_Q23_", "STRATA_NATIVE_"))}
    env.update(cfg.get("env", {}))
    command = ["numactl", "--interleave=all", str(Path(cfg["exe"]).resolve()), cfg["model"],
        "@" + str(args.prompt.resolve()), str(args.tokens), str(cfg.get("dense_cache_mib", 4096)),
        str(cfg.get("threads", 15)), "--context=" + str(cfg["context"]),
        "--prefill-batch=" + str(cfg["prefill_batch"]), "--gpu-budget-mib=" + str(cfg["gpu_budget_mib"]),
        "--cpu-affinity=" + cfg.get("cpu_affinity", "numa"), "--decode-experts=cpu",
        "--bench=" + str(args.trials), "--decode-bench=" + str(args.trials),
        "--prefill-experts=" + cfg.get("prefill_experts", "mmq"), "--route-affinity=" + env.get("STRATA_GLM_ROUTE_AFFINITY", "0")]
    if args.eval_corpus:
        command += ["--eval-corpus=" + str(args.eval_corpus.resolve())]
        if cfg.get("eval_prefill"):
            command.append("--eval-prefill")
        if args.eval_reference:
            command += ["--eval-reference=" + str(args.eval_reference.resolve())]
        if args.eval_save_logits:
            command += ["--eval-save-logits=" + str(args.eval_save_logits.resolve())]
    else:
        command += ["--dump-logits=" + str(args.output.with_suffix(".logits.bin").resolve())]
    if cfg.get("routing_trace"):
        command.append("--routing-trace=" + str(Path(cfg["routing_trace"]).resolve()))
    if cfg.get("expert_pack"):
        command.append("--expert-pack=" + cfg["expert_pack"])
    if cfg.get("decode_graphs"):
        command.append("--decode-graphs")
    if cfg.get("decode_cache_mib"):
        command.append("--decode-cache-mib=" + str(cfg["decode_cache_mib"]))
    if "decode_cache_window" in cfg:
        command.append("--decode-cache-window=" + str(cfg["decode_cache_window"]))
    for name in cfg.get("decode_checks", []):
        if name not in ("verify", "verify_graphs", "decode_graphs"):
            raise ValueError("unknown decoder validation check: " + str(name))
        command.append("--check-" + name.replace("_", "-"))
    if not args.single and not args.eval_corpus and cfg.get("speculative") == "mtp":
        command += ["--speculative=mtp", "--mtp-experts=cpu", "--draft-depth=" + str(cfg.get("draft_depth", 2))]
    if not args.single and not args.eval_corpus and cfg.get("speculative") == "lookup":
        command += ["--speculative=lookup", "--lookup-depth=" + str(cfg.get("lookup_depth", 3))]
    group = Path("/sys/fs/cgroup" + Path("/proc/self/cgroup").read_text().strip().split(":")[-1])
    before = group_stats(group)
    if before["memory.max"] != args.ram_gib * 2**30 or before["memory.swap.max"] != 0:
        raise RuntimeError("effective cgroup limits do not match the requested benchmark")
    record = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        config=cfg, command=command, memory_limit_gib=args.ram_gib, swap_limit_bytes=0,
        fit_only=args.fit_only,
        cgroup=str(group), cgroup_before=before, interference=[], samples=[])
    start = time.monotonic()
    old = process_cpu_snapshot()
    hot = set()
    rejection = None
    next_gpu = start
    next_progress = start
    record["peak_process_gpu_mib"] = 0
    record["gpu_physical_reserve_mib"] = int(env.get("STRATA_GLM_GPU_RESERVE_MIB", "512"))
    record["gpu_backend"] = cfg.get("backend", "cuda")
    record["gpu_process_accounting"] = "unavailable; global AMD sysfs" if record["gpu_backend"] == "hip" else "NVIDIA process query"
    if record["gpu_backend"] == "hip":
        record["peak_process_gpu_mib"] = None
    _, initial_gpu = read_gpu(cfg)
    record["gpu_initial_snapshot"] = initial_gpu
    record["gpu_physical_capacity_mib"] = initial_gpu["total_mib"]
    record["gpu_capacity_mib"], record["gpu_total_limit_mib"] = gpu_used_limit(
        args.gpu_capacity_mib or cfg["gpu_budget_mib"], initial_gpu["total_mib"],
        record["gpu_physical_reserve_mib"], args.gpu_used_limit_mib, initial_gpu["reserved_mib"])
    record["gpu_allocation_budget_mib"] = cfg["gpu_budget_mib"]
    record["gpu_requested_used_limit_mib"] = args.gpu_used_limit_mib
    record["peak_background_gpu_mib"] = None if record["gpu_backend"] == "hip" else 0
    record["minimum_gpu_free_mib"] = initial_gpu["free_mib"]
    with args.output.with_suffix(".stdout").open("w") as out, args.output.with_suffix(".log").open("w") as err:
        proc = subprocess.Popen(command, stdout=out, stderr=err, env=env, start_new_session=True)
        try:
            while proc.poll() is None:
                stats = group_stats(group)
                try:
                    status = Path(f"/proc/{proc.pid}/status").read_text()
                except (FileNotFoundError, PermissionError):
                    if proc.poll() is not None:
                        break
                    continue
                rss = int(re.search(r"VmRSS:\s+(\d+)", status)[1]) * 1024 if "VmRSS:" in status else 0
                try:
                    proc_io = {k: int(v) for k, v in (row.split(":") for row in
                        Path(f"/proc/{proc.pid}/io").read_text().splitlines())}
                except (FileNotFoundError, PermissionError):
                    if proc.poll() is not None:
                        break
                    continue
                record["samples"].append(dict(seconds=time.monotonic()-start,
                    memory_current=stats["memory.current"], rss_bytes=rss,
                    anon_bytes=stats["memory.stat"]["anon"], file_bytes=stats["memory.stat"]["file"],
                    swap_bytes=stats["memory.swap.current"], io=stats["io.stat"], process_io=proc_io))
                if time.monotonic() >= next_gpu:
                    next_gpu = time.monotonic() + 1
                    try:
                        used, snapshot = read_gpu(cfg, proc.pid)
                        total_used = snapshot["used_mib"]
                        record["minimum_gpu_free_mib"] = min(record["minimum_gpu_free_mib"], snapshot["free_mib"])
                        background = None if used is None else max(0, total_used - used)
                        if used is not None:
                            record["peak_background_gpu_mib"] = max(record["peak_background_gpu_mib"], background)
                            record["peak_process_gpu_mib"] = max(record["peak_process_gpu_mib"], used)
                        record["samples"][-1]["gpu_process_mib"] = used
                        record["samples"][-1]["gpu_total_mib"] = total_used
                        record["samples"][-1]["gpu_background_mib"] = background
                        record["samples"][-1]["gpu_free_mib"] = snapshot["free_mib"]
                        record["samples"][-1]["gpu_reserved_mib"] = snapshot["reserved_mib"]
                        record["samples"][-1]["gpu_used_including_reserved_mib"] = total_used + snapshot["reserved_mib"]
                        if "gtt_used_mib" in snapshot:
                            record["samples"][-1]["gpu_gtt_used_mib"] = snapshot["gtt_used_mib"]
                        if "performance" in snapshot:
                            record["samples"][-1]["gpu_performance"] = snapshot["performance"]
                        _, current_limit = gpu_used_limit(args.gpu_capacity_mib or cfg["gpu_budget_mib"],
                            snapshot["total_mib"], record["gpu_physical_reserve_mib"],
                            args.gpu_used_limit_mib, snapshot["reserved_mib"])
                        record["samples"][-1]["gpu_used_limit_mib"] = current_limit
                        if max(used or 0, total_used) > current_limit:
                            rejection = "total GPU memory (engine and other clients) plus physical reserve exceeds simulated GPU capacity"
                        if snapshot["free_mib"] < record["gpu_physical_reserve_mib"]:
                            rejection = "actual GPU free memory is below the physical reserve"
                    except (ValueError, OSError, subprocess.SubprocessError) as error:
                        rejection = "GPU memory probe failed: " + str(error)
                if rss > args.ram_gib * 2**30:
                    rejection = "RSS exceeds cgroup budget: external file cache may be shared"
                if time.monotonic()-start > args.timeout:
                    rejection = "benchmark timeout"
                if time.monotonic()-old[0] >= 1:
                    now = process_cpu_snapshot()
                    busy = set()
                    for pid, (name, ticks) in now[1].items():
                        if pid in (os.getpid(), proc.pid, os.getppid()) or pid not in old[1]:
                            continue
                        percent = (ticks-old[1][pid][1])/os.sysconf("SC_CLK_TCK")/(now[0]-old[0])*100
                        if percent >= 80:
                            busy.add(pid)
                            if pid in hot:
                                record["interference"].append(dict(seconds=time.monotonic()-start,
                                    pid=pid, name=name, cpu_percent=percent))
                    old, hot = now, busy
                if rejection:
                    try:
                        record["rejection_memory_maps"] = process_memory_maps(proc.pid)
                    except (OSError, ValueError) as error:
                        record["rejection_memory_maps_error"] = str(error)
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    break
                if time.monotonic() >= next_progress:
                    next_progress = time.monotonic() + 10
                    record["cgroup_latest"] = stats
                    temporary = args.output.with_suffix(".progress.tmp")
                    temporary.write_text(json.dumps(record)+"\n")
                    temporary.replace(args.output.with_suffix(".progress.json"))
                time.sleep(.5)
        finally:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
            code = proc.wait()
    record.update(exit_code=code, wall_seconds=time.monotonic()-start,
        cgroup_after=group_stats(group), rejected=rejection,
        measurements=parse_log(args.output.with_suffix(".log").read_text()))
    output = args.output.with_suffix(".stdout").read_text()
    record["output_sha256"] = hashlib.sha256(args.output.with_suffix(".stdout").read_bytes()).hexdigest()
    if args.eval_corpus:
        rows = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
        counts = eval_counts(args.eval_corpus)
        record["evaluation"] = rows
        record["evaluated_targets"] = rows[-1].get("tokens", 0) if rows else 0
        complete = eval_complete(rows, counts)
    else:
        ids = list(map(int, output.split()))
        record["output_token_count"] = len(ids)
        complete = (len(record["measurements"]["prefill"]) == args.trials and
                    len(record["measurements"]["decode"]) == args.trials and
                    len(ids) == args.tokens * args.trials)
    record["complete"] = complete
    record["clean"] = not (args.fit_only or code or rejection or record["interference"] or
        record["cgroup_after"]["memory.events"].get("oom_kill", 0)) and complete
    args.output.with_suffix(".result.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k: record[k] for k in ("exit_code", "wall_seconds", "complete", "clean", "rejected")}
        | {"measurements":record["measurements"]}), flush=True)
    return code or (2 if rejection or not complete else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("prompt", type=Path, help="comma-separated frozen prompt token IDs")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ram-gib", type=int, default=28, help="engine allowance; 28 GiB leaves 4 GiB on a 32 GiB PC")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--gpu-capacity-mib", type=int,
                        help="physical/proxy card capacity; default is the config's GPU allocation budget")
    parser.add_argument("--gpu-used-limit-mib", type=int,
                        help="additional ceiling for total GPU usage; distinct from card capacity")
    parser.add_argument("--eval-corpus", type=Path, help="run ordinary teacher-forced quality evaluation instead of timing")
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--eval-reference", type=Path)
    reference.add_argument("--eval-save-logits", type=Path)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--fit-only", action="store_true", help="skip cache clearing; GPU fit check only, no qualified timing")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.ram_gib, args.tokens, args.trials, args.timeout) < 1:
        parser.error("limits and counts must be positive")
    if any(n is not None and n < 1 for n in (args.gpu_capacity_mib, args.gpu_used_limit_mib)):
        parser.error("GPU capacity and usage limits must be positive")
    if (args.eval_reference or args.eval_save_logits) and not args.eval_corpus:
        parser.error("evaluation logits require --eval-corpus")
    if args.worker:
        return worker(args)
    if args.output.with_suffix(".manifest.json").exists():
        parser.error("output prefix already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text())
    if cfg.get("gpu_devices", "0") not in ("0", 0, [0]):
        parser.error("this single-GPU capacity benchmark currently supports GPU 0 only")
    prompt = args.prompt.read_text().strip()
    count = len(prompt.split(","))
    if count + args.tokens + 8 > cfg["context"]:
        parser.error("context is too small for prompt and output")
    if args.eval_corpus:
        counts = eval_counts(args.eval_corpus)
        lengths = [counts[0] + 1] + [counts[i] - counts[i-1] + 1 for i in range(1, len(counts))]
        if max(lengths) > cfg["context"]:
            parser.error("context is too small for an evaluation sequence")
    if cfg["env"].get("STRATA_GLM_MAPPED_LAZY") != "1" or cfg["env"].get("STRATA_GLM_MAPPED_OWNED") != "1":
        parser.error("low-memory runs require lazy mapped experts")
    unit = "strata-lowmem-" + str(os.getpid())
    cache = [] if args.fit_only else clear_model_cache(cfg)
    command = ["systemd-run", "--user", "--wait", "--pipe", "--unit=" + unit,
        "--property=MemoryMax=" + str(args.ram_gib) + "G", "--property=MemorySwapMax=0",
        "--property=IOAccounting=yes",
        "--property=OOMPolicy=kill", "--property=RuntimeMaxSec=" + str(args.timeout+30),
        "--property=WorkingDirectory=" + str(Path.cwd()), sys.executable, str(Path(__file__).resolve()),
        str(args.config.resolve()), str(args.prompt.resolve()), "--output", str(args.output.resolve()),
        "--ram-gib", str(args.ram_gib), "--tokens", str(args.tokens), "--trials", str(args.trials),
        "--timeout", str(args.timeout), "--worker"] + (["--single"] if args.single else []) + (
        ["--fit-only"] if args.fit_only else [])
    if args.gpu_capacity_mib is not None:
        command += ["--gpu-capacity-mib", str(args.gpu_capacity_mib)]
    if args.gpu_used_limit_mib is not None:
        command += ["--gpu-used-limit-mib", str(args.gpu_used_limit_mib)]
    for flag, path in (("--eval-corpus", args.eval_corpus), ("--eval-reference", args.eval_reference),
                       ("--eval-save-logits", args.eval_save_logits)):
        if path is not None:
            command += [flag, str(path.resolve())]
    manifest = dict(config=cfg, prompt_tokens=count,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), executable_sha256=
            hashlib.sha256(Path(cfg["exe"]).read_bytes()).hexdigest(), command=command,
        guard_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cache_clear=cache, capacity_simulation=not args.fit_only, fit_only=args.fit_only,
        compute_gpu="installed GPU", memory_gib=args.ram_gib)
    if args.eval_corpus:
        manifest.update(evaluation_corpus=str(args.eval_corpus.resolve()), evaluation_targets=counts[-1],
            evaluation_corpus_sha256=hashlib.sha256(args.eval_corpus.read_bytes()).hexdigest(),
            evaluation_reference=str(args.eval_reference.resolve()) if args.eval_reference else None)
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    completed = subprocess.run(command, text=True, capture_output=True)
    args.output.with_suffix(".systemd.log").write_text(completed.stderr)
    status = subprocess.run(["systemctl", "--user", "show", unit + ".service", "--property=Result",
        "--property=ExecMainCode", "--property=ExecMainStatus", "--property=MemoryPeak",
        "--property=MemorySwapPeak"], text=True, capture_output=True)
    args.output.with_suffix(".launcher.json").write_text(json.dumps(dict(exit_code=completed.returncode,
        stdout=completed.stdout, stderr=completed.stderr, service_properties=status.stdout), indent=2)+"\n")
    print(completed.stdout or completed.stderr, end="", flush=True)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
