#!/usr/bin/env python3
"""Compare opt-in V100 paths in one locked-weight process; save every completed case."""
import argparse
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import time

from glm_generate import Tokenizer, metadata_shard
from glm_v100_bench import experiment_environment, request_diagnostics, save_result, stop_process

MODES = {
    "baseline": (0, 0, 0, 0, 3, 0),
    "baseline_tasks2": (0, 0, 0, 0, 2, 0),
    "active_only": (0, 0, 1, 0, 3, 0),
    "device": (1, 0, 0, 0, 2, 0),
    "merge": (1, 1, 0, 0, 2, 0),
    "active": (1, 1, 1, 0, 2, 0),
    "tasks1": (1, 1, 1, 0, 1, 0),
    "tasks4": (1, 1, 1, 0, 4, 0),
    "profile": (1, 1, 1, 0, 2, 1),
}
MODES = {name: values + (0,) for name, values in MODES.items()}
MODES["fused"] = (1, 1, 1, 0, 2, 0, 1)
MODES["fused_tasks1"] = (1, 1, 1, 0, 1, 0, 1)
MODES["profile_stock"] = (0, 0, 0, 0, 3, 1, 0)
MODES["kda32"] = (0, 0, 0, 0, 3, 0, 0, 32)
MODES["kda64"] = (0, 0, 0, 0, 3, 0, 0, 64)
MODES["kda32_active"] = (0, 0, 1, 0, 3, 0, 0, 32)
MODES["kda64_active"] = (0, 0, 1, 0, 3, 0, 0, 64)
MODES["stable"] = (0, 0, 0, 0, 3, 0, 0, 128, 1, 0)
MODES["buckets"] = (0, 0, 0, 0, 3, 0, 0, 128, 1, 1)
MODES["buckets_active"] = (0, 0, 1, 0, 3, 0, 0, 128, 1, 1)
MODES["prealloc"] = (0, 0, 0, 0, 3, 0, 0, 128, 0, 0, 1, 0)
MODES["defer"] = (0, 0, 0, 0, 3, 0, 0, 128, 0, 0, 0, 1)
MODES["prealloc_defer"] = (0, 0, 0, 0, 3, 0, 0, 128, 0, 0, 1, 1)
MODES["scheduled_active"] = (0, 0, 1, 0, 3, 0, 0, 128, 0, 0, 1, 1)
MODES["restore"] = (0, 0, 0, 0, 3, 0, 0, 128, 0, 0, 0, 0, 1)
MODES["restore_active"] = (0, 0, 1, 0, 3, 0, 0, 128, 0, 0, 0, 0, 1)
MODES["row4"] = (0, 0, 0, 0, 3, 0, 0, 128, 0, 0, 0, 0, 0, 4)
MODES["row8"] = (0, 0, 0, 0, 3, 0, 0, 128, 0, 0, 0, 0, 0, 8)
MODES["restore_row4"] = (0, 0, 1, 0, 3, 0, 0, 128, 0, 0, 0, 0, 1, 4)
MODES["restore_row8"] = (0, 0, 1, 0, 3, 0, 0, 128, 0, 0, 0, 0, 1, 8)
MODES["fused_restore_row8"] = (1, 1, 1, 0, 2, 0, 1, 128, 0, 0, 0, 0, 1, 8)
MODES["buckets_restore_row8"] = (0, 0, 1, 0, 3, 0, 0, 128, 1, 1, 0, 0, 1, 8)
MODES["profile_buckets_row8"] = (0, 0, 1, 0, 3, 1, 0, 128, 1, 1, 0, 0, 1, 8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=pathlib.Path)
    ap.add_argument("text", type=pathlib.Path)
    ap.add_argument("--decoder", type=pathlib.Path, default=pathlib.Path("build-glm-v100/strata-glm-decode"))
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--generated-tokens", type=int, default=128)
    ap.add_argument("--repetitions", type=int, default=2)
    ap.add_argument("--threads", type=int, default=17)
    ap.add_argument("--prefill-batch", type=int, default=4096)
    ap.add_argument("--validation-tokens", type=int, default=512)
    ap.add_argument("--validation-repetitions", type=int, default=3)
    ap.add_argument("--validation-cases", choices=("best", "all"), default="best",
                    help="validate the fastest decode case or every measured case")
    ap.add_argument("--boundary-requests", action="store_true", help="repeat short prompts after the timed cases")
    ap.add_argument("--cases", default="baseline,device,merge,active,tasks1,tasks4")
    ap.add_argument("--prefill-experts", choices=("mmq", "f16", "f16-batched"), default="mmq")
    ap.add_argument("--no-decode-graphs", action="store_true",
                    help="launch decode kernels individually so a tracer can itemize them")
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()
    cases = args.cases.split(",")
    if (any(case not in MODES for case in cases) or args.repetitions < 1 or args.generated_tokens < 2 or
            args.threads < 1 or args.validation_repetitions < 1 or args.validation_tokens < 0 or
            args.validation_tokens == 1 or not 1 <= args.prefill_batch <= 16384):
        ap.error("invalid cases or token counts")
    ids = Tokenizer.from_gguf(metadata_shard(args.model)).encode(args.text.read_text(), parse_special=True)
    if not 0 < args.prompt_tokens <= len(ids):
        ap.error("prompt is shorter than requested token count")
    ids = ids[:args.prompt_tokens]
    encoded = ",".join(map(str, ids))
    command = [str(args.decoder.resolve()), "--serve", str(args.model.resolve()), "131072", "4096", str(args.threads),
               "0", str(args.prefill_batch), "-1", "32768", "0", "0", "none", "1", "auto", "0", "0,1", "auto", "1",
               "1", "extend", "0" if args.no_decode_graphs else "1", "256", "1", args.prefill_experts]
    with args.decoder.open("rb") as binary:
        sha = hashlib.file_digest(binary, "sha256").hexdigest()
    log_path = args.output.with_suffix(".log")
    result = dict(command=command, decoder_sha256=sha, complete=False,
                  configuration=dict(prompt_tokens=args.prompt_tokens, prefill_batch=args.prefill_batch,
                                     prefill_experts=args.prefill_experts),
                  execution=dict(workers=args.threads),
                  environment=experiment_environment(),
                  prompt_sha256=hashlib.sha256(encoded.encode()).hexdigest(), cases={},
                  timing_scope="warm resident requests, prefill includes cache preparation; decode counts N-1 advances")

    def save():
        result["request_diagnostics"] = request_diagnostics(log_path.read_text()) if log_path.exists() else []
        save_result(args.output, result)

    save()
    env = dict(os.environ, STRATA_GLM_ALLOW_TUNE="1")
    started = time.monotonic()
    with log_path.open("w") as log, subprocess.Popen(command, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, env=env) as process:
        def read():
            line = process.stdout.readline().strip()
            if not line or line.startswith("ERR "):
                raise RuntimeError(f"{line or 'decoder exited'}; see {log_path}")
            return line

        def tune(case):
            process.stdin.write("TUNE " + " ".join(map(str, MODES[case])) + "\n")
            process.stdin.flush()
            if read() != "TUNED":
                raise RuntimeError("unexpected tuning response")

        def request(count, prompt=None):
            text = encoded if prompt is None else ",".join(map(str, prompt))
            prompt_length = len(ids) if prompt is None else len(prompt)
            process.stdin.write(f"GEN {count} {text}\n")
            process.stdin.flush()
            tokens = []
            while True:
                line = read()
                if line.startswith("T "):
                    tokens.append(int(line.split()[1]))
                elif line.startswith("DONE "):
                    fields = line.split()
                    if len(tokens) != count or int(fields[1]) != count or int(fields[2]) != prompt_length:
                        raise RuntimeError(f"unexpected response: {line}")
                    prefill, decode = map(float, fields[3:5])
                    return dict(prefill_ms=prefill, decode_ms=decode, prefill_tok_s=prompt_length * 1000 / prefill,
                                decode_tok_s=(count - 1) * 1000 / decode, response_ms=prefill + decode,
                                token_ids=tokens)
        try:
            while True:
                ready = read()
                if ready.startswith("READY "): break
            result["prefill_includes_reset"] = "prefill-includes-reset" in ready.split()
            result["startup_seconds"] = time.monotonic() - started
            save()
            print(f"READY startup_s={result['startup_seconds']:.3f}", flush=True)
            for case in cases:
                tune(case)
                request(3)
                rows = []
                instrumented = bool(MODES[case][5])
                entry = dict(mode=MODES[case], trials=rows, complete=False, instrumented=instrumented)
                result["cases"][case] = entry
                for trial in range(args.repetitions):
                    row = request(3 if instrumented else args.generated_tokens)
                    rows.append(row)
                    entry.update(median_prefill_tok_s=statistics.median(r["prefill_tok_s"] for r in rows),
                                 median_decode_tok_s=statistics.median(r["decode_tok_s"] for r in rows),
                                 repetitions_identical=all(r["token_ids"] == rows[0]["token_ids"] for r in rows))
                    save()
                    print(f"CASE {case} trial={trial} prefill={row['prefill_tok_s']:.3f} decode={row['decode_tok_s']:.3f}", flush=True)
                if not entry["repetitions_identical"]:
                    raise RuntimeError("repeated requests produced different token IDs in " + case)
                entry["complete"] = True
                save()
            measured = [case for case in cases if not MODES[case][5]]
            if args.validation_tokens and measured:
                best = max(measured, key=lambda c: result["cases"][c]["median_decode_tok_s"])
                result["validation_case"] = best
                result["validation_by_case"] = {}
                for case in measured if args.validation_cases == "all" else [best]:
                    tune(case)
                    reference = result["cases"][case]["trials"][0]["token_ids"]
                    prefix = min(len(reference), args.validation_tokens)
                    entry = dict(trials=[], complete=False)
                    result["validation_by_case"][case] = entry
                    for trial in range(args.validation_repetitions):
                        row = request(args.validation_tokens)
                        entry["trials"].append(row)
                        entry["median_decode_tok_s"] = statistics.median(r["decode_tok_s"] for r in entry["trials"])
                        if case == best:
                            result["validation_trials"] = entry["trials"]
                            result["validation"] = entry["trials"][0]
                            result["median_validation_decode_tok_s"] = entry["median_decode_tok_s"]
                        save()
                        if row["token_ids"][:prefix] != reference[:prefix]:
                            raise RuntimeError("validation prefix differs from repeated requests in " + case)
                        if row["token_ids"] != entry["trials"][0]["token_ids"]:
                            raise RuntimeError("repeated validation requests produced different token IDs in " + case)
                        print(f"VALIDATION {case} trial={trial} decode={row['decode_tok_s']:.3f}", flush=True)
                    entry["complete"] = True
                    save()
                tune(best)
            if args.boundary_requests and measured:
                result["boundary_requests"] = []
                for length in sorted({min(len(ids), n) for n in (1, 3, 64, 129, 256)}):
                    entry = dict(prompt_tokens=length, trials=[], complete=False)
                    result["boundary_requests"].append(entry)
                    for _ in range(2):
                        entry["trials"].append(request(3, ids[:length]))
                        save()
                    if entry["trials"][0]["token_ids"] != entry["trials"][1]["token_ids"]:
                        raise RuntimeError("short repeated requests produced different token IDs")
                    entry["complete"] = True
                    save()
                    print(f"BOUNDARY prompt={length} repeated_ids=1", flush=True)
            process.stdin.write("QUIT\n")
            process.stdin.flush()
            process.wait(timeout=30)
            if process.returncode:
                raise RuntimeError(f"decoder failed; see {log_path}")
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
