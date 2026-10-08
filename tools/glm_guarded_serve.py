"""Serve GLM on Linux HIP inside a preconfigured RAM cgroup, with a VRAM guard."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from glm_low_memory_bench import clear_model_cache, group_stats, read_gpu


def warm_expert_cache(config, ram_limit):
    """Prefault routed experts into this service's cgroup, using a bounded buffer."""
    from gguf_reader import GGUFFile
    if config.get("expert_pack") or not Path(config["model"]).is_file():
        raise ValueError("expert-page warmup requires a standalone GGUF")
    model = GGUFFile(Path(config["model"]))
    prefix = model.metadata["general.architecture"] + "."
    first = int(model.metadata[prefix + "leading_dense_block_count"])
    last = int(model.metadata[prefix + "block_count"]) - int(model.metadata[prefix + "nextn_predict_layers"])
    spans = []
    for tensor in model.tensors:
        match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", tensor.name)
        if match and first <= int(match[1]) < last:
            size = tensor.expected_bytes()
            if size is None:
                raise ValueError("unknown expert geometry for warmup")
            spans.append((model.data_start + tensor.offset, size))
    if len(spans) != 3 * (last - first) or not spans:
        raise ValueError("incomplete expert tensors for warmup")
    total = sum(size for _, size in spans)
    if total > ram_limit - 4 * 2**30:
        raise ValueError("expert warmup needs 4 GiB RAM allowance beyond expert bytes")
    start = time.monotonic()
    print(f"WARM_EXPERT_PAGES tensors={len(spans)} bytes={total}", flush=True)
    buffer = bytearray(8 * 2**20)
    with Path(config["model"]).open("rb", buffering=0) as file:
        file_size = os.fstat(file.fileno()).st_size
        for offset, size in sorted(spans):
            if offset + size > file_size:
                raise ValueError("truncated expert tensor during warmup")
            file.seek(offset)
            while size:
                count = file.readinto(memoryview(buffer)[:min(size, len(buffer))])
                if not count:
                    raise ValueError("short read during expert warmup")
                size -= count
    result = dict(tensors=len(spans), bytes=total, seconds=time.monotonic()-start)
    print("WARM_EXPERT_PAGES complete " + json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--ram-gib", type=int, default=60)
    parser.add_argument("--reserve-mib", type=int, default=2048)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--telemetry", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg.get("backend") != "hip" or args.ram_gib < 1 or args.reserve_mib < 1:
        parser.error("requires a HIP config and positive memory limits")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not (os.environ.get("STRATA_API_KEY", "").strip() or str(cfg.get("api_key", "")).strip()):
        parser.error("non-loopback binding requires STRATA_API_KEY or config api_key")
    group = Path("/sys/fs/cgroup" + Path("/proc/self/cgroup").read_text().strip().split(":")[-1])
    initial = group_stats(group)
    limit = args.ram_gib * 2**30
    if initial["memory.max"] != limit or initial["memory.swap.max"] != 0:
        parser.error("run inside a cgroup with the requested MemoryMax and MemorySwapMax=0")
    # Cold file pages must be charged to this service, rather than an old run.
    cache = clear_model_cache(cfg)
    _, gpu = read_gpu(cfg)
    if gpu["free_mib"] < args.reserve_mib:
        parser.error("GPU has less free memory than the requested reserve")
    warmup = warm_expert_cache(cfg, limit) if cfg.get("warm_expert_pages") is True else None
    command = [sys.executable, "-m", "serve.server", "--engine", "glm",
               "--config", str(args.config.resolve()), "--host", args.host, "--port", str(args.port)]
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("STRATA_GLM_", "STRATA_Q23_", "STRATA_NATIVE_"))}
    process = subprocess.Popen(command, start_new_session=True, env=env)
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    record = dict(command=command, cache_clear=cache, expert_warmup=warmup, ram_limit_bytes=limit,
                  gpu_reserve_mib=args.reserve_mib, peak_rss_bytes=0, peak_gpu_used_mib=0)
    reason = None
    try:
        while process.poll() is None and not stop:
            stats = group_stats(group)
            _, gpu = read_gpu(cfg)
            rss = 0
            for pid in (group / "cgroup.procs").read_text().split():
                try:
                    status = Path(f"/proc/{pid}/status").read_text()
                except FileNotFoundError:
                    continue
                match = re.search(r"VmRSS:\s+(\d+)", status)
                rss += int(match[1]) * 1024 if match else 0
            record.update(updated_unix=time.time(), cgroup=stats, gpu=gpu, summed_rss_bytes=rss)
            record["peak_rss_bytes"] = max(record["peak_rss_bytes"], rss)
            record["peak_gpu_used_mib"] = max(record["peak_gpu_used_mib"], gpu["used_mib"])
            if rss > limit:
                reason = "process RSS exceeds RAM limit; check externally charged file pages"
            elif stats["memory.swap.current"]:
                reason = "unexpected service swap usage"
            elif gpu["free_mib"] < args.reserve_mib:
                reason = "global GPU usage consumed the VRAM reserve"
            temporary = args.telemetry.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(args.telemetry)
            if reason:
                print("GLM memory guard: " + reason, file=sys.stderr, flush=True)
                break
            time.sleep(1)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        reason = "memory probe failed: " + str(error)
        print("GLM memory guard: " + reason, file=sys.stderr, flush=True)
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        record.update(exit_code=process.returncode, rejected=reason, stopped=stop,
                      cgroup_after=group_stats(group))
        args.telemetry.write_text(json.dumps(record, indent=2) + "\n")
    return 2 if reason else (0 if stop else process.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
