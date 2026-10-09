"""Cross-process determinism bisection for the B550 BASE15 config (64 tokens, 2 processes per variant)."""
import copy, json, pathlib, subprocess, sys, time
CAL = pathlib.Path("/home/syoyo/work/Strata-b550-calibration-15b787b5")
BASE = CAL / "docs/fixtures/glm_b550_levers_20261009/step1/step1-b15360-01-base15.config.json"
IDS = "/home/syoyo/work/Strata/build-hip-glm-v11/single-2048.ids"
OUT = pathlib.Path("/home/syoyo/work/bisect-20261009/runs"); OUT.mkdir(parents=True, exist_ok=True)
VARIANTS = {
    "A_base": {},
    "B_no_floating_q8": {"unset": ["STRATA_GLM_FLOATING_Q8"]},
    "C_no_q23_prefetch": {"unset": ["STRATA_Q23_PREFETCH"]},
    "D_static_tier": {"env": {"STRATA_GLM_TIER_ADAPT": "0"}},
    "E_no_tier": {"cfg": {"decode_cache_mib": 0}},
    "F_no_numa_weights": {"unset": ["STRATA_GLM_Q2_NUMA_WEIGHTS"]},
    "G_no_tier_no_floating_q8": {"cfg": {"decode_cache_mib": 0}, "unset": ["STRATA_GLM_FLOATING_Q8"]},
    "H_prefill_mmq": {"cfg": {"prefill_experts": "mmq"}},
    "I_no_layer_flow": {"env": {"STRATA_GLM_LAYER_FLOW": "0"}},
    "J_no_graphs": {"cfg": {"decode_graphs": False}, "env": {"STRATA_GLM_LAYER_GRAPHS": "0", "STRATA_GLM_VERIFY_GRAPHS": "0"}},
    "K_no_tier_no_layer_flow": {"cfg": {"decode_cache_mib": 0}, "env": {"STRATA_GLM_LAYER_FLOW": "0"}},
    "L_no_tier_prefill_mmq": {"cfg": {"decode_cache_mib": 0, "prefill_experts": "mmq"}},
    "A_base_rerun": {},
    "M_mmq_mtp1": {"cfg": {"prefill_experts": "mmq", "speculative": "mtp", "draft_depth": 1}},
}
order = sys.argv[1:] or list(VARIANTS)
COMPARE_ONLY = False
def quiet():
    ok = 0
    while ok < 12:
        ok = ok + 1 if float(open("/proc/loadavg").read().split()[0]) < 3 else 0
        time.sleep(10)
def run(name, rep):
    v = VARIANTS[name]
    cfg = json.loads(BASE.read_text())
    cfg.update(v.get("cfg", {}))
    for k in v.get("unset", []):
        cfg["env"].pop(k, None)
    cfg["env"].update(v.get("env", {}))
    tag = f"{name}-{rep}"
    cpath = OUT / f"{tag}.config.json"; cpath.write_text(json.dumps(cfg, indent=1))
    cmd = ["systemd-run", "--user", "--wait", "--pipe", f"--unit=strata-bisect-{tag.lower().replace('_','-')}",
           "--property=MemoryMax=60G", "--property=MemorySwapMax=0", "--property=OOMPolicy=kill", "--property=RuntimeMaxSec=1200",
           f"--property=WorkingDirectory={CAL}", "/home/syoyo/work/Strata/.venv-glm-hip/bin/python", str(CAL / "tools/glm_low_memory_bench.py"),
           str(cpath), IDS, "--output", str(OUT / tag), "--ram-gib", "60", "--tokens", "64", "--trials", "1", "--timeout", "1100",
           "--worker", "--gpu-capacity-mib", "16304", "--gpu-used-limit-mib", "15792"]
    quiet()
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    (OUT / f"{tag}.driver.log").write_text(p.stdout[-20000:] + "\n--- stderr\n" + p.stderr[-20000:])
    out = OUT / f"{tag}.stdout"
    ids = [int(x) for x in out.read_text().split()] if out.exists() else None
    return ids, p.returncode, time.time() - t
summary = []
for name in order:
    a, rca, ta = run(name, 1)
    b, rcb, tb = run(name, 2)
    if a is None or b is None:
        row = dict(variant=name, error=f"exit {rca}/{rcb}")
    else:
        diff = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        import array
        la, lb = (array.array("f", (OUT / f"{name}-{r}.logits.bin").read_bytes()) for r in (1, 2))
        maxd = max(abs(x - y) for x, y in zip(la, lb))
        row = dict(variant=name, identical=(a == b), first_diff=diff, n=len(a), logits_bitwise=(la == lb),
                   logits_max_abs_diff=maxd, seconds=[round(ta), round(tb)])
    summary.append(row)
    print(json.dumps(row), flush=True)
    (OUT.parent / "summary.json").write_text(json.dumps(summary, indent=1))
