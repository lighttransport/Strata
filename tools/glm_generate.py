"""Text frontend for the experimental native GLM correctness decoder."""
import argparse
import pathlib
import subprocess
import sys
import tempfile

from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer


def metadata_shard(path):
    """The tokenizer and architecture metadata live in shard one."""
    import re
    match = re.fullmatch(r"(.*)-\d{5}-of-(\d{5})\.gguf", path.name)
    return path.with_name(f"{match[1]}-00001-of-{match[2]}.gguf") if match else path


def chat_prompt(metadata, prompt, system=None):
    from jinja2.sandbox import SandboxedEnvironment

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    def raise_exception(message):
        raise ValueError(message)

    env = SandboxedEnvironment(extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = raise_exception
    return env.from_string(metadata["tokenizer.chat_template"]).render(
        messages=messages, tools=None, add_generation_prompt=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path, help="GLM GGUF shard or native EXL3 model directory")
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--prompt-file", type=pathlib.Path, help="UTF-8 prompt, avoiding shell argument limits")
    parser.add_argument("--decoder", type=pathlib.Path, default=pathlib.Path("build/strata-glm-decode"))
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--dense-cache-mib", type=int, default=4096)
    parser.add_argument("--expert-cache-mib", type=int, default=0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--prefill-batch", default="8", help="1-8192 or auto; above 8 uses GPU prefill")
    parser.add_argument("--gpu-budget-mib", type=int, default=12288)
    parser.add_argument("--gpu-devices", help="one or two CUDA device IDs, e.g. 0,1")
    parser.add_argument("--prefill-expert-cache-mib", default="0", help="native prefill cache per GPU: auto or MiB")
    parser.add_argument("--context", type=int, help="total prompt and output capacity")
    parser.add_argument("--lock-weights", action="store_true", help="make byte-identical private mapped weights resident and lock them (Linux)")
    parser.add_argument("--decode-graphs", action=argparse.BooleanOptionalAction, default=False,
                        help="replay single-token recurrent mixers and resident experts with CUDA graphs")
    parser.add_argument("--decode-cache-adapt", action=argparse.BooleanOptionalAction, default=False,
                        help="reuse expert slots with bounded background copies during generation")
    parser.add_argument("--decode-cache-mib", default="0", help="prefix-trained decode cache: extend reuses prefill, auto replaces it, or 0..4096 MiB")
    parser.add_argument("--decode-cache-window", type=int, default=256,
                        help="use the last N prompt tokens for cache ranking; 0 uses the whole prefix")
    parser.add_argument("--decode-experts", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--decode-prefill-cache", action=argparse.BooleanOptionalAction, default=False,
                        help="execute resident prefill experts on their GPU during CPU decode")
    parser.add_argument("--lookup-depth", type=int, default=0, help="0 disables speculation; 1-7 prompt-lookup drafts")
    parser.add_argument("--speculative", choices=("none", "lookup", "mtp"), default="none")
    parser.add_argument("--draft-depth", type=int, choices=range(1, 8), default=3)
    parser.add_argument("--cpu-prepack-mib", type=int, default=0)
    parser.add_argument("--cpu-affinity", choices=("none", "auto", "numa"), default="none")
    parser.add_argument("--weight-pages", choices=("4k", "huge"), default="4k", help="EXL3 routed weight page advice (huge uses Linux THP)")
    parser.add_argument("--system")
    parser.add_argument("--raw", action="store_true", help="tokenize the prompt without a chat template")
    parser.add_argument("--dry-run", action="store_true", help="print prompt token IDs without running the network")
    args = parser.parse_args()
    if (args.prompt is None) == (args.prompt_file is None):
        parser.error("provide either a prompt or --prompt-file")
    text = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    shard = metadata_shard(args.model)
    native = args.model.is_dir()
    if native:
        from glm_artifact import NativeTokenizer, load_metadata
        metadata = load_metadata(args.model)
        tokenizer = NativeTokenizer(args.model)
        if args.speculative != "none" or args.lookup_depth or args.decode_experts != "cpu":
            parser.error("native EXL3 requires CPU decode experts and speculative=none")
        def specified(name): return any(x == name or x.startswith(name + "=") for x in sys.argv[1:])
        if not specified("--threads"): args.threads = 15
        if not specified("--prefill-batch"): args.prefill_batch = "256"
        if not specified("--gpu-budget-mib"): args.gpu_budget_mib = 10240
        if not specified("--context"): args.context = 8192
        if "--cpu-affinity" not in sys.argv and not any(x.startswith("--cpu-affinity=") for x in sys.argv): args.cpu_affinity = "numa"
    else:
        metadata = GGUFFile(shard).metadata
        tokenizer = Tokenizer.from_gguf(shard)
    if metadata.get("general.architecture") != "glm5next":
        parser.error("expected a glm5next artifact")
    prompt = text if args.raw else chat_prompt(metadata, text, args.system)
    ids = tokenizer.encode(prompt, parse_special=True)
    if not ids or args.tokens < 1 or args.dense_cache_mib < 64 or args.expert_cache_mib < 0 or args.threads < 1:
        parser.error("empty prompt or invalid decode settings")
    if args.context is not None and (args.context < len(ids) + args.tokens or
                                    args.context > metadata["glm5next.context_length"]):
        parser.error("prompt plus output exceeds the requested or model context")
    if native and args.context > 65536: parser.error("native EXL3 context above 64K requires deferred host KV offload")
    encoded = ",".join(map(str, ids))
    if args.dry_run:
        print(encoded)
        return
    stops = {int(metadata[k]) for k in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                                       "tokenizer.ggml.eom_token_id") if k in metadata}
    command = [str(args.decoder.resolve()), str(args.model.resolve()), encoded,
               str(args.tokens), str(args.dense_cache_mib), str(args.threads),
               f"--prefill-batch={args.prefill_batch}", f"--gpu-budget-mib={args.gpu_budget_mib}", f"--lookup-depth={args.lookup_depth}", f"--decode-experts={args.decode_experts}", f"--expert-cache-mib={args.expert_cache_mib}",
               f"--stop-ids={','.join(map(str, sorted(stops)))}"]
    if args.speculative != "none": command.append(f"--speculative={args.speculative}")
    command.extend((f"--draft-depth={args.draft_depth}", f"--cpu-affinity={args.cpu_affinity}", f"--cpu-prepack-mib={args.cpu_prepack_mib}"))
    if native: command.append(f"--weight-pages={args.weight_pages}")
    # Read only the token-ID protocol from stdout; decoder failures use stderr.
    if args.context is not None: command.append(f"--context={args.context}")
    if args.lock_weights: command.append("--lock-weights")
    if args.decode_prefill_cache: command.append("--decode-prefill-cache")
    if args.decode_graphs: command.append("--decode-graphs")
    if args.decode_cache_adapt: command.append("--decode-cache-adapt")
    if args.decode_cache_mib != "0": command.append(f"--decode-cache-mib={args.decode_cache_mib}")
    command.append(f"--decode-cache-window={args.decode_cache_window}")
    if args.gpu_devices is not None: command.append(f"--gpu-devices={args.gpu_devices}")
    if args.prefill_expert_cache_mib != "0":
        command.append(f"--prefill-expert-cache-mib={args.prefill_expert_cache_mib}")
    with tempfile.TemporaryDirectory(prefix="strata-glm-") as temporary:
        token_file = pathlib.Path(temporary) / "tokens.ids"
        token_file.write_text(encoded, encoding="ascii")
        command[2] = f"@{token_file}"
        with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as process:
            generated, shown = [], ""
            for line in process.stdout:
                token = int(line.strip())
                if token in stops:
                    continue
                generated.append(token)
                decoded = tokenizer.decode(generated, errors="ignore")
                sys.stdout.write(decoded[len(shown):])
                sys.stdout.flush()
                shown = decoded
            status = process.wait()
    if status:
        raise SystemExit(status)
    print()


if __name__ == "__main__":
    main()
