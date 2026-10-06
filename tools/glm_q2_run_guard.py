"""Run one Q2 experiment with host memory limits and optional CPU interference rejection."""
import argparse
import json
import os
import pathlib
import subprocess
import time
import signal
from glm_q2_coding_bench import process_cpu_snapshot

def descendants(root):
    """Include subprocesses (e.g. a resident engine launched by an evaluator)."""
    parents = {}
    for path in pathlib.Path("/proc").glob("[0-9]*/stat"):
        try:
            value = path.read_text()
            fields = value[value.rindex(")") + 2:].split()
            parents[int(path.parent.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    found = {root}
    while True:
        expanded = found | {pid for pid, parent in parents.items() if parent in found}
        if expanded == found:
            return found
        found = expanded

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--require-idle", action="store_true")
    parser.add_argument("--max-swap-mib", type=int, default=0,
                        help="tolerated process swap; keep 0 for timing runs, a small allowance only for quality runs")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = {"command": command, "peak_rss_kib": 0, "peak_swap_kib": 0, "minimum_available_kib": None,
              "sample_interval_ms": 100, "rejected": None}
    old = process_cpu_snapshot()
    hot = set()
    start = time.monotonic()
    with args.output.with_suffix(".stdout").open("w") as out, args.output.with_suffix(".log").open("w") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        try:
            while process.poll() is None:
                try:
                    owned = descendants(process.pid)
                    rss = swap = 0
                    swapped_pids = []
                    for pid in owned:
                        try:
                            fields = dict(line.split(":", 1) for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines() if ":" in line)
                        except FileNotFoundError:
                            continue
                        rss += int(fields.get("VmRSS", "0 kB").split()[0])
                        swapped = int(fields.get("VmSwap", "0 kB").split()[0])
                        swap += swapped
                        if swapped:
                            swapped_pids.append(pid)
                    mem = dict(line.split(":", 1) for line in pathlib.Path("/proc/meminfo").read_text().splitlines())
                    available = int(mem["MemAvailable"].split()[0])
                    record["peak_rss_kib"] = max(record["peak_rss_kib"], rss)
                    record["peak_swap_kib"] = max(record["peak_swap_kib"], swap)
                    record["minimum_available_kib"] = min(record["minimum_available_kib"] or available, available)
                    if swap > args.max_swap_mib * 1024 or rss > 118 * 1024**2 or available < 4 * 1024**2:
                        record["rejected"] = {"reason": "memory limit", "rss_kib": rss, "swap_kib": swap, "available_kib": available}
                        if swap:
                            swapped = []
                            region = {}
                            lines = []
                            for pid in swapped_pids:
                                try:
                                    lines.extend(pathlib.Path(f"/proc/{pid}/smaps").read_text().splitlines())
                                except FileNotFoundError:
                                    pass
                            for line in lines:
                                if line[:1] and line.split()[0].count("-") == 1:
                                    parts = line.split(maxsplit=5)
                                    region = {"kind": "anonymous" if len(parts) < 6 else "heap" if parts[5] == "[heap]" else "stack" if parts[5].startswith("[stack") else "file"}
                                elif line.startswith("Size:"):
                                    region["size_kib"] = int(line.split()[1])
                                elif line.startswith("Swap:") and int(line.split()[1]):
                                    swapped.append({**region, "swap_kib": int(line.split()[1])})
                            record["swapped_regions"] = swapped
                    if args.require_idle and time.monotonic() - old[0] >= 1:
                        now = process_cpu_snapshot()
                        busy = set()
                        for pid, (name, ticks) in now[1].items():
                            if pid in owned or pid in (os.getpid(), os.getppid()) or pid not in old[1]:
                                continue
                            percent = (ticks - old[1][pid][1]) / os.sysconf("SC_CLK_TCK") / (now[0] - old[0]) * 100
                            if percent >= 80:
                                busy.add(pid)
                                if pid in hot:
                                    record["rejected"] = {"reason": "external CPU work", "pid": pid, "name": name, "cpu_percent": percent}
                        old, hot = now, busy
                    if record["rejected"]:
                        os.killpg(process.pid, signal.SIGTERM)
                        break
                except (FileNotFoundError, ProcessLookupError):
                    pass
                time.sleep(.1)
        finally:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            record["exit_code"] = process.wait()
            record["wall_seconds"] = time.monotonic() - start
            args.output.with_suffix(".memory.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return 2 if record["rejected"] else record["exit_code"]

if __name__ == "__main__":
    raise SystemExit(main())
