"""GLM-5.3-Flash model regression tests: short-context decode against golden tokens, the decoder's self-checks,
model-based kernel parity, and a held-out quality gate (KL, top-1, perplexity against the BF16 baseline logits).

  python3 tools/glm_regression.py list
  python3 tools/glm_regression.py case decode --decoder build-glm/strata-glm-decode
  python3 tools/glm_regression.py parity glm_quant_parity --bindir build-glm
  python3 tools/glm_regression.py quality --decoder build-glm/strata-glm-decode [--sequences 16] [--update-baseline]

Exit codes: 0 pass, 1 fail, 77 skipped (model files missing). Registered in ctest with the labels glm_model and
glm_quality (CMakeLists.txt); the cases share one RESOURCE_LOCK so the model is never loaded twice at once.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tests" / "glm" / "regression.json"
SKIP = 77
TRAILING = "trailing evaluation baseline records"


def load_config(path=CONFIG):
    cfg = json.loads(pathlib.Path(path).read_text())
    cfg["model"] = os.environ.get("STRATA_GLM_TEST_MODEL", cfg["model"])
    cfg["pack"] = os.environ.get("STRATA_GLM_TEST_PACK", cfg["pack"])
    cfg["quality"]["reference"] = os.environ.get("STRATA_GLM_TEST_EVAL_REFERENCE", cfg["quality"]["reference"])
    return cfg


def missing(*paths):
    return [str(p) for p in paths if not pathlib.Path(p).exists()]


def environment(cfg, extra=None):
    env = dict(os.environ)
    env.update(cfg["env"])
    env.update(extra or {})
    return env


def parse_ids(stdout):
    return [int(t) for t in stdout.split()]


def run_case(cfg, name, decoder, log=print):
    sc = cfg["short_context"]
    case = sc["cases"][name]
    gone = missing(decoder, cfg["model"], cfg["pack"])
    if gone:
        log(f"SKIP {name}: missing {', '.join(gone)}")
        return SKIP
    cmd = [str(decoder), cfg["model"], ",".join(map(str, sc["prompt"])), str(sc["steps"]), *sc["args"],
           f"--expert-pack={cfg['pack']}", *case["flags"]]
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=environment(cfg, case.get("env")))
    seconds = time.time() - start
    if proc.returncode != 0:
        log(f"FAIL {name}: decoder exit {proc.returncode} after {seconds:.0f} s\n" + proc.stderr[-2000:])
        return 1
    ids = parse_ids(proc.stdout)
    if ids != sc["golden"]:
        log(f"FAIL {name}: tokens differ from golden\n  got    {ids}\n  golden {sc['golden']}")
        return 1
    log(f"PASS {name}: {len(ids)} tokens equal golden ({seconds:.0f} s)")
    return 0


def run_parity(cfg, name, bindir, log=print):
    binary = pathlib.Path(bindir) / name
    gone = missing(binary, cfg["model"])
    if gone:
        log(f"SKIP {name}: missing {', '.join(gone)}")
        return SKIP
    proc = subprocess.run([str(binary), cfg["model"], *cfg["parity"][name]], capture_output=True, text=True,
                          env=environment(cfg))
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-1:] or [""]
    log(f"{'PASS' if proc.returncode == 0 else 'FAIL'} {name}: exit {proc.returncode}: {tail[0][:200]}")
    return 0 if proc.returncode == 0 else 1


def parse_eval(stdout):
    rows = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and '"sequence"' in line:
            rows.append(json.loads(line))
    return rows


def check_quality(final, baseline, tol):
    """Returns a list of failure strings (empty = pass)."""
    failures = []
    kl_limit = baseline["kl"] * (1 + tol["kl_rel"]) + tol["kl_abs"]
    if final["kl"] > kl_limit:
        failures.append(f"KL {final['kl']:.5f} > {kl_limit:.5f} (baseline {baseline['kl']:.5f})")
    top1_limit = baseline["top1_agreement"] - tol["top1_abs"]
    if final["top1_agreement"] < top1_limit:
        failures.append(f"top-1 {final['top1_agreement']:.4f} < {top1_limit:.4f} (baseline {baseline['top1_agreement']:.4f})")
    ppl_limit = baseline["perplexity"] * (1 + tol["ppl_rel"])
    if final["perplexity"] > ppl_limit:
        failures.append(f"perplexity {final['perplexity']:.4f} > {ppl_limit:.4f} (baseline {baseline['perplexity']:.4f})")
    if final["tokens"] != baseline["tokens"]:
        failures.append(f"scored {final['tokens']} tokens, baseline {baseline['tokens']}")
    return failures


def run_quality(cfg, decoder, sequences=None, update_baseline=False, config_path=CONFIG, log=print):
    q = cfg["quality"]
    corpus = ROOT / q["corpus"]
    gone = missing(decoder, cfg["model"], cfg["pack"], q["reference"], corpus)
    if gone:
        log(f"SKIP quality: missing {', '.join(gone)}")
        return SKIP
    lines = [l for l in corpus.read_text().splitlines() if l.strip()]
    if sequences:
        lines = lines[:sequences]
    with tempfile.NamedTemporaryFile("w", suffix=".ids", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        subset = f.name
    cmd = [str(decoder), cfg["model"], *q["args"], f"--eval-corpus={subset}", f"--eval-reference={q['reference']}",
           f"--expert-pack={cfg['pack']}"]
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=environment(cfg, q.get("env")))
    finally:
        os.unlink(subset)
    seconds = time.time() - start
    rows = parse_eval(proc.stdout)
    # the reference file holds more sequences than any subset: that exit is expected once every row is scored
    if proc.returncode != 0 and not (TRAILING in proc.stderr and len(rows) == len(lines)):
        log(f"FAIL quality: decoder exit {proc.returncode} after {seconds:.0f} s, {len(rows)}/{len(lines)} sequences\n"
            + proc.stderr[-2000:])
        return 1
    if len(rows) != len(lines):
        log(f"FAIL quality: {len(rows)}/{len(lines)} sequences scored")
        return 1
    final = rows[-1]
    log(f"quality: {len(lines)} sequences, {final['tokens']} tokens, KL {final['kl']:.9f}, top-1 "
        f"{final['top1_agreement']:.6f}, perplexity {final['perplexity']:.9f}, nll {final['nll']:.9f}, "
        f"different_logits {final.get('different_logits')} ({seconds:.0f} s)")
    if update_baseline:
        if sequences and sequences != len([l for l in corpus.read_text().splitlines() if l.strip()]):
            log("refusing to store a baseline from a subset")
            return 1
        raw = json.loads(pathlib.Path(config_path).read_text())
        raw["quality"]["baseline"] = {k: final[k] for k in ("tokens", "kl", "top1_agreement", "perplexity", "nll",
                                                            "different_logits") if k in final}
        raw["quality"]["baseline"]["recorded"] = time.strftime("%Y-%m-%d")
        pathlib.Path(config_path).write_text(json.dumps(raw, indent=1) + "\n")
        log(f"baseline written to {config_path}")
        return 0
    baseline = q.get("baseline")
    if not baseline:
        log("FAIL quality: no baseline recorded; run with --update-baseline on a known-good build")
        return 1
    if sequences:
        log("subset run: metrics printed, thresholds apply to the full corpus only")
        return 0
    failures = check_quality(final, baseline, q["tolerance"])
    for f in failures:
        log("FAIL quality: " + f)
    if not failures:
        log("PASS quality")
    return 1 if failures else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    p = sub.add_parser("case"); p.add_argument("name"); p.add_argument("--decoder", required=True)
    p = sub.add_parser("parity"); p.add_argument("name"); p.add_argument("--bindir", required=True)
    p = sub.add_parser("quality"); p.add_argument("--decoder", required=True)
    p.add_argument("--sequences", type=int, default=None, help="score only the first N sequences (no thresholds)")
    p.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "list":
        print("cases:", " ".join(cfg["short_context"]["cases"]))
        print("parity:", " ".join(cfg["parity"]))
        print("quality: held-out", cfg["quality"]["corpus"])
        return 0
    if args.command == "case":
        return run_case(cfg, args.name, args.decoder)
    if args.command == "parity":
        return run_parity(cfg, args.name, args.bindir)
    return run_quality(cfg, args.decoder, args.sequences, args.update_baseline, args.config)


if __name__ == "__main__":
    sys.exit(main())
