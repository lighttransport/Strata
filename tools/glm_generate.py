"""Text frontend for the experimental native GLM correctness decoder."""
import argparse
import pathlib
import subprocess
import sys

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
    parser.add_argument("model", type=pathlib.Path, help="any shard of the GLM GGUF")
    parser.add_argument("prompt")
    parser.add_argument("--decoder", type=pathlib.Path, default=pathlib.Path("build/strata-glm-decode"))
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--dense-cache-mib", type=int, default=4096)
    parser.add_argument("--expert-cache-mib", type=int, default=0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--prefill-batch", default="8", help="1-4096 or auto; above 8 uses GPU prefill")
    parser.add_argument("--gpu-budget-mib", type=int, default=12288)
    parser.add_argument("--decode-experts", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--lookup-depth", type=int, default=0, help="0 disables speculation; 1-7 prompt-lookup drafts")
    parser.add_argument("--speculative", choices=("none", "lookup", "mtp"), default="none")
    parser.add_argument("--draft-depth", type=int, choices=range(1, 8), default=3)
    parser.add_argument("--cpu-prepack-mib", type=int, default=0)
    parser.add_argument("--cpu-affinity", choices=("none", "auto"), default="none")
    parser.add_argument("--system")
    parser.add_argument("--raw", action="store_true", help="tokenize the prompt without a chat template")
    parser.add_argument("--dry-run", action="store_true", help="print prompt token IDs without running the network")
    args = parser.parse_args()
    shard = metadata_shard(args.model)
    metadata = GGUFFile(shard).metadata
    if metadata.get("general.architecture") != "glm5next":
        parser.error("expected a glm5next artifact")
    tokenizer = Tokenizer.from_gguf(shard)
    prompt = args.prompt if args.raw else chat_prompt(metadata, args.prompt, args.system)
    ids = tokenizer.encode(prompt, parse_special=True)
    if not ids or args.tokens < 1 or args.dense_cache_mib < 64 or args.expert_cache_mib < 0 or args.threads < 1:
        parser.error("empty prompt or invalid decode settings")
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
    # Read only the token-ID protocol from stdout; decoder failures use stderr.
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
