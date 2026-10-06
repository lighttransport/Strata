"""Resume the long GLM Q2 conversion/quality/timing qualification from completed calibration.

Each model process runs through the RAM guard. Completed stages are recorded;
partial or rejected stages are not treated as results. No server config changes.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import statistics
import urllib.request
from urllib.parse import urlparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--work", type=pathlib.Path, required=True)
    parser.add_argument("--store", type=pathlib.Path, required=True, help="volume with 130 GiB free for pack and baseline logits")
    parser.add_argument("--calibration-record", type=pathlib.Path, required=True, help="completed run-guard JSON; waits if still running")
    parser.add_argument("--calibration", type=pathlib.Path, required=True)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--decoder", default="build-glm/strata-glm-decode")
    parser.add_argument("--converter", default="build-glm/strata-glm-q2-pack")
    parser.add_argument("--restore-api", help="reload an already-unloaded local service on exit")
    parser.add_argument("--defer-api-unload", action="store_true", help="keep restore-api loaded during CPU conversion, unload before model validation")
    args = parser.parse_args()
    if args.defer_api_unload and not args.restore_api:
        parser.error("defer-api-unload requires restore-api")
    if args.restore_api and urlparse(args.restore_api).hostname not in ("127.0.0.1", "localhost", "::1"):
        parser.error("restore-api must be a loopback service")
    for key in ("model", "work", "store", "calibration_record", "calibration", "corpus", "dataset"):
        setattr(args, key, getattr(args, key).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.store.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(STRATA_GLM_LOCK_RUNTIME="1", STRATA_IQ2_TABLE_SIGNS="1", STRATA_NATIVE_NUMA_LOCAL="0",
               STRATA_GLM_QUANT_ONCE="1", STRATA_GLM_LAYER_GRAPHS="1", STRATA_GLM_VERIFY_GRAPHS="1",
               STRATA_POOL_SPIN_US="20000", STRATA_Q23_AVX2="0", STRATA_GLM_Q2_NUMA_WEIGHTS="0",
               STRATA_GLM_PACK_HUGE="0", STRATA_GLM_PACK_FUSE_QUANT="1")
    env.pop("STRATA_GLM_CALIBRATION_DIR", None)
    env.pop("STRATA_GLM_ACTIVATION_TRACE", None)
    pack = args.store / "experts-q23.gguf"
    reference = args.store / "evaluation.logits"
    profiles = args.work / "profiles"
    status = {"state": "running", "stages": []}
    executables = {name: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
                   for name, path in (("decoder", args.decoder), ("converter", args.converter))}
    def save():
        (args.work / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    def stage(name, command, idle=False):
        prefix = args.work / name
        done = args.work / (name + ".done.json")
        identity = {"command": list(map(str, command)), "executables": executables,
                    "environment": {k:v for k,v in env.items() if k.startswith("STRATA_")}}
        if done.exists():
            if json.loads(done.read_text()) != identity:
                raise RuntimeError(f"{name}: completed stage settings differ; use a new work directory")
            status["stages"].append({"name": name, "reused": True});save();return
        print(json.dumps({"stage": name, "state": "started"}), flush=True)
        status["active_stage"] = name;save()
        guarded = [sys.executable, "tools/glm_q2_run_guard.py", "--output", str(prefix)]
        if idle:
            guarded.append("--require-idle")
        result = subprocess.run(guarded + ["--"] + identity["command"], env=env)
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {prefix}.log and .memory.json")
        done.write_text(json.dumps(identity, indent=2) + "\n")
        status["stages"].append({"name": name, "reused": False});save()
    def api_post(endpoint):
        request = urllib.request.Request(args.restore_api.rstrip("/")+endpoint, data=b"{}", headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read())
    service_unloaded = not args.defer_api_unload
    def decode(extra):
        return ["numactl", "--interleave=all", args.decoder, str(args.model), "0", "1", "4096", "15",
                "--context=8192", "--prefill-batch=256", "--gpu-budget-mib=12288", "--cpu-affinity=numa",
                "--decode-graphs", *extra]
    try:
        status["active_stage"] = "waiting for calibration";save()
        while not args.calibration_record.exists():
            time.sleep(10)
        calibration = json.loads(args.calibration_record.read_text())
        if calibration.get("exit_code") != 0 or calibration.get("rejected"):
            raise RuntimeError("calibration was rejected or failed")
        coverage = json.loads((args.calibration / "coverage.json").read_text())["projections"]
        expected = {f"blk.{layer}.{kind}" for layer in range(3,45) for kind in ("gu", "down")}
        if {row["name"] for row in coverage} != expected or any(len(row["counts"]) != 288 or sum(row["counts"]) != 32768*8 for row in coverage):
            raise RuntimeError("calibration does not cover exactly 32,768 targets on all routed projections")
        manifest = json.loads((args.corpus / "manifest.json").read_text())
        for split in ("calibration", "evaluation"):
            if hashlib.sha256((args.corpus / (split+".ids")).read_bytes()).hexdigest() != manifest[split+"_sha256"]:
                raise RuntimeError("corpus checksum mismatch")
        shutil.copyfile(args.corpus / "manifest.json", args.work / "corpus-manifest.json")
        shutil.copyfile(args.calibration / "coverage.json", args.work / "calibration-coverage.json")
        if not pack.exists() and shutil.disk_usage(args.store).free < 130*1024**3:
            raise RuntimeError("conversion/evaluation volume needs 130 GiB free")
        stage("convert", [args.converter, str(args.model), str(pack), f"--calibration={args.calibration}", "--threads=16"])
        stage("verify-pack", [args.converter, "--verify", str(args.model), str(pack)])
        subprocess.run([sys.executable,"tools/glm_q2_pack_profiles.py",str(pack)+".quality.json",str(profiles)],check=True)
        if args.defer_api_unload:
            service_unloaded = True
            status["service_unload"] = api_post("/unload")
            save()
        # Fail early on graph/rollback correctness and publish an explicitly short
        # performance screen before the multi-hour quality matrix.
        stage("converted-graphs", decode([f"--expert-pack={pack}", "--check-verify-graphs"]))
        env["STRATA_NATIVE_TASKS_PER_THREAD"] = "12"
        stage("early-q23-coding", ["numactl", "--interleave=all", sys.executable,
              "tools/glm_q2_coding_bench.py", str(args.model), "--decoder", args.decoder,
              "--output", str(args.work / "early-q23-coding-answers"), "--fixtures", "prime",
              "--tokens=128", "--mtp-sweep", "--repetitions=1", "--no-warm-weights",
              "--reject-competitors", f"--expert-pack={pack}"], idle=True)
        evaluation = f"--eval-corpus={args.corpus / 'evaluation.ids'}"
        stage("quality-original", decode([evaluation, f"--eval-save-logits={reference}"]))
        candidates = ("q23", "q2", "retain25", "retain50", "retain75")
        for profile in candidates:
            stage("quality-"+profile, decode([evaluation, f"--eval-reference={reference}",
                  f"--expert-pack={pack}", f"--expert-pack-profile={profiles / (profile+'.json')}"]))
        # Alternate baseline/candidate processes rather than comparing only their best trials.
        env["STRATA_NATIVE_TASKS_PER_THREAD"] = "12"
        for profile in candidates:
            for repetition in range(3):
                order = ("original", profile) if repetition%2 == 0 else (profile, "original")
                for current in order:
                    name = f"coding-{profile}-r{repetition}-{current}"
                    command = ["numactl", "--interleave=all", sys.executable, "tools/glm_q2_coding_bench.py", str(args.model),
                               "--decoder", args.decoder, "--output", str(args.work / (name+"-answers")),
                               "--mtp-sweep", "--repetitions=1", "--no-warm-weights", "--reject-competitors"]
                    if current != "original":
                        command += [f"--expert-pack={pack}", f"--expert-pack-profile={profiles / (current+'.json')}"]
                    stage(name, command, idle=True)
                    check_code = """import json, resource, sys
resource.setrlimit(resource.RLIMIT_AS, (1024**3,1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (30,30))
resource.setrlimit(resource.RLIMIT_FSIZE, (16*1024**2,16*1024**2))
sys.path.insert(0,'/tools')
from pathlib import Path
from glm_q2_coding_check import check
results={}
for fixture in ('prime','json_escape','csv'):
    try: results[fixture]={'passed':True,'cases':check(Path('/answers'),fixture)}
    except Exception as error: results[fixture]={'passed':False,'reason':str(error)}
print(json.dumps(results))
"""
                    stage(name+"-correctness", ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
                          "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
                          "--symlink", "usr/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                          "--ro-bind", str(pathlib.Path("tools").resolve()), "/tools",
                          "--ro-bind", str(args.work / (name+"-answers")), "/answers",
                          "--chdir", "/tmp", "--clearenv", "--setenv", "PATH", "/usr/bin",
                          "/usr/bin/python3", "-I", "-c", check_code])
        for profile in ("original", "q23", "q2"):
            command = ["numactl", "--interleave=all", sys.executable, "tools/glm_q2_humaneval.py", str(args.model), str(args.dataset),
                       "--decoder", args.decoder, "--output", str(args.work / ("humaneval-"+profile+"-answers"))]
            if profile != "original":
                command += [f"--expert-pack={pack}", f"--expert-pack-profile={profiles / (profile+'.json')}"]
            stage("humaneval-"+profile, command)
        summary = {"quality": {}, "coding": {}, "humaneval": {}, "target_tokens_per_second": 15}
        for profile in ("original", *candidates):
            rows = [json.loads(line) for line in (args.work / ("quality-"+profile+".stdout")).read_text().splitlines() if line.startswith("{")]
            if not rows or rows[-1]["tokens"] != 32768:
                raise RuntimeError("incomplete held-out evaluation")
            summary["quality"][profile] = rows[-1]
        for profile in candidates:
            summary["coding"][profile] = {}
            for current in ("original", profile):
                runs = [json.loads((args.work / f"coding-{profile}-r{r}-{current}-answers" / "measurement.json").read_text())["results"] for r in range(3)]
                entries = {}
                for fixture in runs[0]:
                    if len({run[fixture]["token_sha256"] for run in runs}) != 1:
                        raise RuntimeError(f"{profile}/{current}/{fixture}: output changed across processes")
                    entries[fixture] = {mode: statistics.median(run[fixture]["median_tokens_per_second"][mode] for run in runs)
                                        for mode in runs[0][fixture]["median_tokens_per_second"]}
                summary["coding"][profile][current] = entries
        for profile in ("original", "q23", "q2"):
            result = json.loads((args.work / ("humaneval-"+profile+"-answers") / "measurement.json").read_text())
            if not result["complete"]:
                raise RuntimeError("incomplete HumanEval evaluation")
            summary["humaneval"][profile] = {key: result[key] for key in ("tasks", "pass_at_1")}
        (args.work / "measurement.json").write_text(json.dumps(summary, indent=2) + "\n")
        status["state"] = "complete"
        status.pop("active_stage", None)
    except BaseException as error:
        status["state"] = "failed";status["error"] = str(error)
        raise
    finally:
        if args.restore_api and service_unloaded:
            try:
                request = urllib.request.Request(args.restore_api.rstrip("/")+"/load", data=b"{}", headers={"Content-Type":"application/json"})
                with urllib.request.urlopen(request, timeout=600) as response:
                    status["service_reload"] = json.loads(response.read())
                with urllib.request.urlopen(args.restore_api.rstrip("/")+"/health", timeout=30) as response:
                    status["service_health"] = json.loads(response.read())
                with urllib.request.urlopen(args.restore_api.rstrip("/")+"/v1/models", timeout=30) as response:
                    model_name = json.loads(response.read())["data"][0]["id"]
                request = urllib.request.Request(args.restore_api.rstrip("/")+"/v1/chat/completions",
                    data=json.dumps({"model":model_name,"messages":[{"role":"user","content":"Reply OK."}],"max_tokens":8,"temperature":0}).encode(),
                    headers={"Content-Type":"application/json"})
                with urllib.request.urlopen(request, timeout=180) as response:
                    status["service_completion_checked"] = bool(json.loads(response.read()).get("choices"))
            except Exception as error:
                status["service_reload_error"] = str(error)
        save()


if __name__ == "__main__":
    main()
