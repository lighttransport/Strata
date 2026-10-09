"""Affinity 0.06 / 0.08 vs base with mmq prefill: 512 tokens, 3 trials, interleaved base."""
import array, json, pathlib, re, subprocess, time
CAL = pathlib.Path("/home/syoyo/work/Strata-b550-calibration-15b787b5")
BASE = CAL / "docs/fixtures/glm_b550_levers_20261009/step1/step1-b15360-01-base15.config.json"
IDS = "/home/syoyo/work/Strata/build-hip-glm-v11/single-2048.ids"
OUT = pathlib.Path("/home/syoyo/work/bisect-20261009/aff_mmq"); OUT.mkdir(exist_ok=True)
PLAN = [("01_base", 0.0), ("02_aff06", 0.06), ("03_base", 0.0), ("04_aff08", 0.08), ("05_base", 0.0)]
def quiet():
    ok = 0
    while ok < 12:
        ok = ok + 1 if float(open("/proc/loadavg").read().split()[0]) < 3 else 0
        time.sleep(10)
rows = []
for tag, aff in PLAN:
    cfg = json.loads(BASE.read_text())
    cfg["prefill_experts"] = "mmq"
    cfg["env"]["STRATA_GLM_ROUTE_AFFINITY"] = str(aff)
    cpath = OUT / f"{tag}.config.json"; cpath.write_text(json.dumps(cfg, indent=1))
    cmd = ["systemd-run", "--user", "--wait", "--pipe", f"--unit=strata-affmmq-{tag.replace('_','-')}",
           "--property=MemoryMax=60G", "--property=MemorySwapMax=0", "--property=OOMPolicy=kill", "--property=RuntimeMaxSec=1800",
           f"--property=WorkingDirectory={CAL}", "/home/syoyo/work/Strata/.venv-glm-hip/bin/python", str(CAL / "tools/glm_low_memory_bench.py"),
           str(cpath), IDS, "--output", str(OUT / tag), "--ram-gib", "60", "--tokens", "512", "--trials", "3", "--timeout", "1700",
           "--worker", "--gpu-capacity-mib", "16304", "--gpu-used-limit-mib", "15792"]
    quiet()
    load0 = open("/proc/loadavg").read().split()[0]
    p = subprocess.run(cmd, capture_output=True, text=True)
    (OUT / f"{tag}.driver.log").write_text(p.stdout[-30000:] + "\n--- stderr\n" + p.stderr[-30000:])
    log = (OUT / f"{tag}.log").read_text() if (OUT / f"{tag}.log").exists() else ""
    speeds = [float(x) for x in re.findall(r"^DECODE steps=\d+ ms=[\d.]+ tok_s=([\d.]+)", log, re.M)]
    stdout = (OUT / f"{tag}.stdout").read_text() if (OUT / f"{tag}.stdout").exists() else ""
    ids = [int(x) for x in stdout.split()]
    trials = [ids[i * 512:(i + 1) * 512] for i in range(len(ids) // 512)]
    rows.append(dict(tag=tag, affinity=aff, exit=p.returncode, tok_s=speeds, load_start=load0, trials=trials))
    print(json.dumps({k: v for k, v in rows[-1].items() if k != "trials"}), flush=True)
base = rows[0]["trials"]
def cmp(a, b):
    d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    return dict(first_diff=d, matching=sum(x == y for x, y in zip(a, b)))
summary = []
for r in rows:
    summary.append(dict(tag=r["tag"], affinity=r["affinity"], exit=r["exit"], tok_s=r["tok_s"],
                        vs_first_base=[cmp(t, base[i]) for i, t in enumerate(r["trials"]) if i < len(base)],
                        trials_equal=len({tuple(t) for t in r["trials"]}) == 1))
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
