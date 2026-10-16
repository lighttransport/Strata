#!/usr/bin/env python3
"""MXFP4 / Q2_K / Q3_K comparison on Ampere and AVX2; smoke by default."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true",
                        help="collect repeated timings; use only with an idle machine")
    parser.add_argument("--checks-only", action="store_true", help="run structural and small fixture checks only")
    parser.add_argument("--real-fixture", type=Path, help="exported fixture directory with fixture.json and float32 inputs")
    parser.add_argument("--real-only", action="store_true", help="run only supplied real-fixture smoke jobs and Python structural checks")
    args = parser.parse_args()
    if args.checks_only and args.benchmark:
        parser.error("checks-only cannot collect benchmark timings")
    if args.real_only and (not args.real_fixture or args.benchmark):
        parser.error("real-only requires real-fixture and cannot collect timings")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "status.txt").write_text("Pending: preparing and validating diagnostic inputs.\n")
    build = args.build.resolve()
    smoke = not args.benchmark
    jobs = [("cpu", [str(build / "bench_cpu"), "--weights", "dummy",
                     "--sets", "1" if smoke else "4", "--iters", "1" if smoke else "20",
                     "--threads", "2" if smoke else "1,2,4,8,16,32", "--ncols", "1" if smoke else "1,2,4,8"])]
    for mode in ("decode", "kquant", "moe", "prefill"):
        jobs.append(("gpu-" + mode, [str(build / "bench_gpu"), "--weights", "dummy",
                    "--mode", mode, "--experts", "8" if smoke else "64",
                    "--kinds", "0,2", "--ncols", "1" if smoke else "1,2,4,8,16",
                    "--tokens", "16" if smoke else "512,2048,8192",
                    "--reps", "1" if smoke else "20", "--trials", "1" if smoke else "9"]))
    for device in ("cpu", "gpu"):
        for threads in ([2] if smoke or device == "gpu" else [1, 2, 4, 8, 16, 32]):
            name = "exl3-" + device + ("-t" + str(threads) if device == "cpu" else "")
            cmd = [str(build / ("bench_exl3_" + device)), "--threads", str(threads)]
            if not smoke:
                cmd.append("--benchmark")
            jobs.append((name, cmd))
    for device in ("cpu", "gpu"):
        for threads in ([2] if smoke or device == "gpu" else [1, 2, 4, 8, 16, 32]):
            name = "research-" + device + ("-t" + str(threads) if device == "cpu" else "")
            cmd = [str(build / ("bench_research_" + device)), "--threads", str(threads)]
            if device == "cpu":
                cmd += ["--cpu-kernel", "both"]
            else:
                cmd += ["--gpu-kernel", "both"]
            if not smoke:
                cmd.append("--benchmark")
            jobs.append((name, cmd))
    study = [sys.executable, str(Path(__file__).resolve().parent.parent / "research/gyro_study.py"),
             "--output", str(args.output.resolve())]
    if not smoke:
        study.append("--benchmark")
    jobs.append(("gyro-study", study))
    jobs.append(("check-formats", [str(build / "check_research_formats")]))
    jobs.append(("check-python", [sys.executable, str(Path(__file__).resolve().parent.parent / "research/check_python.py")]))
    jobs.append(("check-gpu-decoder", [str(build / "bench_research_gpu"), "--check-gpu-decoder"]))
    for device in ("cpu", "gpu"):
        jobs.append(("check-viterbi-" + device, [str(build / ("bench_research_" + device)),
                     "--encoder", "viterbi", "--rows", "16", "--format", "xor_trellis", "--curvature"]))
    fixture = args.output.resolve() / "input-fixture"
    jobs.insert(0, ("check-make-fixture", [sys.executable,
                    str(Path(__file__).resolve().parent.parent / "research/make_fixture.py"),
                    "--output", str(fixture)]))
    wide_fixture = args.output.resolve() / "wide-input-fixture"
    jobs.insert(1, ("check-make-wide-fixture", [sys.executable,
                    str(Path(__file__).resolve().parent.parent / "research/make_fixture.py"),
                    "--cols", "256", "--rows", "32", "--output", str(wide_fixture)]))
    for device in ("cpu", "gpu"):
        jobs.append(("check-file-" + device, [str(build / ("bench_research_" + device)),
                     "--weights-f32", str(fixture / "weights.f32"), "--cols", "128", "--rows", "16",
                     "--activation-f32", str(fixture / "activation.f32"),
                     "--calibration-f32", str(fixture / "calibration.f32"), "--curvature",
                     "--encoder", "viterbi", "--scale", "both",
                     "--report-csv", str(args.output.resolve() / ("diagnostic-" + device + ".csv")),
                     "--cpu-kernel", "both" if device == "cpu" else "reference"]))
        jobs.append(("check-zero-" + device, [str(build / ("bench_research_" + device)),
                     "--weights-f32", str(fixture / "zero-weights.f32"), "--cols", "128", "--rows", "16",
                     "--activation-f32", str(fixture / "activation.f32"), "--scale", "both",
                     "--cpu-kernel", "both" if device == "cpu" else "reference",
                     "--report-csv", str(args.output.resolve() / ("zero-" + device + ".csv"))]))
        jobs.append(("check-zero-activation-" + device, [str(build / ("bench_research_" + device)),
                     "--weights-f32", str(fixture / "weights.f32"), "--cols", "128", "--rows", "16",
                     "--activation-f32", str(fixture / "zero-activation.f32"), "--scale", "both",
                     "--cpu-kernel", "both" if device == "cpu" else "reference",
                     "--report-csv", str(args.output.resolve() / ("zero-activation-" + device + ".csv"))]))
        jobs.append(("check-wide-" + device, [str(build / ("bench_research_" + device)),
                     "--weights-f32", str(wide_fixture / "weights.f32"), "--cols", "256", "--rows", "32",
                     "--activation-f32", str(wide_fixture / "activation.f32"), "--scale", "both",
                     "--cpu-kernel", "both" if device == "cpu" else "reference",
                     "--report-csv", str(args.output.resolve() / ("wide-" + device + ".csv"))]))
        jobs.append(("check-ldl-" + device, [str(build / ("bench_research_" + device)),
                     "--weights-f32", str(fixture / "weights.f32"), "--cols", "128", "--rows", "16",
                     "--activation-f32", str(fixture / "activation.f32"),
                     "--calibration-f32", str(fixture / "calibration.f32"),
                     "--format", "xor_trellis", "--ldl-feedback",
                     "--cpu-kernel", "both" if device == "cpu" else "reference",
                     "--report-csv", str(args.output.resolve() / ("ldl-" + device + ".csv"))]))
        jobs.append(("check-tuned-ldl-" + device, [str(build / ("bench_research_" + device)),
                     "--weights-f32", str(fixture / "weights.f32"), "--cols", "128", "--rows", "16",
                     "--activation-f32", str(fixture / "activation.f32"),
                     "--calibration-f32", str(fixture / "calibration-large.f32"),
                     "--validation-f32", str(fixture / "validation.f32"),
                     "--heldout-f32", str(fixture / "heldout.f32"),
                     "--format", "xor_trellis", "--tune-ldl",
                     "--cpu-kernel", "both" if device == "cpu" else "reference",
                     "--report-csv", str(args.output.resolve() / ("tuned-ldl-" + device + ".csv"))]))
        paired_cmd = jobs[-1][1].copy()
        paired_cmd[paired_cmd.index("--report-csv") + 1] = str(args.output.resolve() / ("paired-ldl-" + device + ".csv"))
        paired_cmd += ["--validation-guard", "paired"]
        jobs.append(("check-paired-ldl-" + device, paired_cmd))
        for variant, extra in (("base", []), ("boundary", ["--boundary-trials", "4"]),
                               ("refined", ["--boundary-trials", "4", "--xor-scale-search"])):
            jobs.append(("check-ring-" + variant + "-" + device, [str(build / ("bench_research_" + device)),
                         "--weights-f32", str(fixture / "weights.f32"), "--cols", "128", "--rows", "16",
                         "--activation-f32", str(fixture / "activation.f32"),
                         "--heldout-f32", str(fixture / "heldout.f32"), "--format", "xor_trellis", "--encoder", "viterbi",
                         "--cpu-kernel", "both" if device == "cpu" else "reference",
                         "--report-csv", str(args.output.resolve() / ("ring-" + variant + "-" + device + ".csv")), *extra]))
        jobs.append(("check-calibrated-scales-" + device, [str(build / ("bench_research_" + device)),
                     "--weights-f32", str(fixture / "weights.f32"), "--cols", "128", "--rows", "16",
                     "--activation-f32", str(fixture / "activation.f32"),
                     "--calibration-f32", str(fixture / "calibration-large.f32"),
                     "--validation-f32", str(fixture / "validation.f32"),
                     "--heldout-f32", str(fixture / "heldout.f32"),
                     "--format", "xor_trellis", "--calibrate-xor-scales",
                     "--cpu-kernel", "both" if device == "cpu" else "reference",
                     "--report-csv", str(args.output.resolve() / ("calibrated-scales-" + device + ".csv"))]))
    if args.checks_only:
        jobs = [(name, cmd) for name, cmd in jobs if name.startswith("check-")]
    for seed in (5678, 9012, 3456):
        seed_fixture = args.output.resolve() / ("input-seed-" + str(seed))
        jobs.append(("check-make-seed-" + str(seed), [sys.executable,
                     str(Path(__file__).resolve().parent.parent / "research/make_fixture.py"),
                     "--seed", str(seed), "--output", str(seed_fixture)]))
        for device in ("cpu", "gpu"):
            jobs.append(("check-scale-seed-" + str(seed) + "-" + device,
                         [str(build / ("bench_research_" + device)), "--weights-f32", str(seed_fixture / "weights.f32"),
                          "--cols", "128", "--rows", "16", "--activation-f32", str(seed_fixture / "activation.f32"),
                          "--calibration-f32", str(seed_fixture / "calibration-large.f32"),
                          "--validation-f32", str(seed_fixture / "validation.f32"),
                          "--heldout-f32", str(seed_fixture / "heldout.f32"),
                          "--format", "xor_trellis", "--calibrate-xor-scales",
                          "--cpu-kernel", "both" if device == "cpu" else "reference",
                          "--report-csv", str(args.output.resolve() / ("scale-seed-" + str(seed) + "-" + device + ".csv"))]))
    for profile, cols, rows in (("outliers", 128, 16), ("correlated", 256, 32),
                                ("rank-deficient", 128, 16), ("shift", 256, 16), ("fallback", 128, 16)):
        folder = args.output.resolve() / ("input-profile-" + profile)
        jobs.append(("check-make-profile-" + profile, [sys.executable,
                     str(Path(__file__).resolve().parent.parent / "research/make_fixture.py"),
                     "--profile", profile, "--cols", str(cols), "--rows", str(rows), "--output", str(folder)]))
        for device in ("cpu", "gpu"):
            jobs.append(("check-profile-" + profile + "-" + device,
                         [str(build / ("bench_research_" + device)), "--weights-f32", str(folder / "weights.f32"),
                          "--cols", str(cols), "--rows", str(rows), "--activation-f32", str(folder / "activation.f32"),
                          "--calibration-f32", str(folder / "calibration-large.f32"),
                          "--validation-f32", str(folder / "validation.f32"), "--heldout-f32", str(folder / "heldout.f32"),
                          "--format", "xor_trellis", "--calibrate-xor-scales", "--cpu-kernel", "both" if device == "cpu" else "reference",
                          "--report-csv", str(args.output.resolve() / ("profile-" + profile + "-" + device + ".csv"))]))
    for block in (128, 256):
        folder = args.output.resolve() / "input-profile-correlated"
        for device in ("cpu", "gpu"):
            jobs.append(("check-cross-ldl-" + str(block) + "-" + device,
                         [str(build / ("bench_research_" + device)), "--weights-f32", str(folder / "weights.f32"),
                          "--cols", "256", "--rows", "32", "--activation-f32", str(folder / "activation.f32"),
                          "--calibration-f32", str(folder / "calibration-large.f32"),
                          "--validation-f32", str(folder / "validation.f32"), "--heldout-f32", str(folder / "heldout.f32"),
                          "--format", "xor_trellis", "--tune-ldl", "--validation-guard", "paired", "--hessian-block", str(block),
                          "--cpu-kernel", "both" if device == "cpu" else "reference",
                          "--report-csv", str(args.output.resolve() / ("cross-ldl-" + str(block) + "-" + device + ".csv"))]))
    if args.real_fixture:
        import json
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))
        from fixture_validation import verify_fixture
        folder = args.real_fixture.resolve()
        try:
            geometry = verify_fixture(folder)
            if not geometry.get("tensor") or geometry.get("source_type") not in ("F32", "F16", "BF16"):
                raise ValueError("real-fixture requires an export_layer.py manifest")
        except (ValueError, KeyError, OSError) as error:
            (args.output / "status.txt").write_text("FAILED: real-fixture validation: " + str(error) + "\n")
            raise
        (args.output / "real-fixture-provenance.json").write_text(json.dumps(geometry, indent=2) + "\n")
        for device in ("cpu", "gpu"):
            jobs.append(("check-real-" + device, [str(build / ("bench_research_" + device)),
                         "--weights-f32", str(folder / "weights.f32"), "--cols", str(geometry["input"]),
                         "--rows", str(geometry["output"]), "--activation-f32", str(folder / "activation.f32"),
                         "--calibration-f32", str(folder / "calibration-large.f32"),
                         "--heldout-f32", str(folder / "heldout.f32"), "--scale", "both", "--curvature",
                         "--cpu-kernel", "both" if device == "cpu" else "reference",
                         "--report-csv", str(args.output.resolve() / ("real-" + device + ".csv"))]))
            jobs.append(("check-real-scales-" + device, [str(build / ("bench_research_" + device)),
                         "--weights-f32", str(folder / "weights.f32"), "--cols", str(geometry["input"]),
                         "--rows", str(geometry["output"]), "--activation-f32", str(folder / "activation.f32"),
                         "--calibration-f32", str(folder / "calibration-large.f32"), "--validation-f32", str(folder / "validation.f32"),
                         "--heldout-f32", str(folder / "heldout.f32"), "--format", "xor_trellis", "--calibrate-xor-scales",
                         "--cpu-kernel", "both" if device == "cpu" else "reference",
                         "--report-csv", str(args.output.resolve() / ("real-scales-" + device + ".csv"))]))
    if args.real_only:
        jobs = [(name, cmd) for name, cmd in jobs if name.startswith("check-real-") or name == "check-python"]
    (args.output / "status.txt").write_text(
        "SMOKE ONLY: miner may be active; timings are not performance evidence.\n"
        if smoke else "Benchmark requested: verify machine was idle before interpreting CSV.\n")
    for name, cmd in jobs:
        if name in ("check-file-cpu", "check-file-gpu", "check-ldl-cpu", "check-ldl-gpu"):
            cmd += ["--heldout-f32", str(fixture / "heldout.f32")]
        if name in ("check-wide-cpu", "check-wide-gpu"):
            cmd += ["--heldout-f32", str(wide_fixture / "heldout.f32")]
        if name.startswith("check-") and name.endswith("gpu"):
            cmd += ["--gpu-kernel", "both"]
        if not smoke and name != "gyro-study" and not name.startswith("check-"):
            cmd += ["--csv", str((args.output / (name + ".csv")).resolve())]
        print("Running", name, flush=True)
        with (args.output / (name + ".log")).open("w") as log:
            log.write("Command: " + repr(cmd) + "\n")
            log.flush()
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (args.output / "status.txt").open("a") as status:
        status.write(f"All {len(jobs)} processes exited successfully.\n")
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent.parent / "research/summarize_checks.py"),
                    "--input", str(args.output.resolve())], check=True)


if __name__ == "__main__":
    main()
