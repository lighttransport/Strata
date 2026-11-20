#!/usr/bin/env python3
"""Text in, text out for strata-hy4f (the runner itself takes token ids, as Strata's other runners do).

    uv run --no-project --with numpy --with pyyaml --with tqdm python -I tools/hy4f_run.py \
        --gguf-py LLAMA_CPP/gguf-py --tokenize LLAMA_BUILD/bin/llama-tokenize --runner build-hy4f/strata-hy4f \
        MODEL.gguf "The capital of France is" -- --n 32 --threads 8

The prompt goes through llama-tokenize (vocabulary only: it does not load the weights); the ids strata-hy4f prints
are decoded with the GGUF's own vocabulary (GPT-2 byte-level BPE). Raw completion: no chat template is applied.
"""
import argparse
import os
import subprocess
import sys


def byte_decoder():
    # GPT-2's bytes_to_unicode, inverted
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def detokenize(tokens, ids):
    dec = byte_decoder()
    out = bytearray()
    for i in ids:
        t = tokens[i]
        if all(ch in dec for ch in t):
            out += bytes(dec[ch] for ch in t)
        else:   # special / added tokens are stored as plain text
            out += t.encode("utf-8")
    return out.decode("utf-8", errors="replace")


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("prompt")
    ap.add_argument("--gguf-py", required=True)
    ap.add_argument("--tokenize", required=True, help="llama-tokenize binary")
    ap.add_argument("--runner", default="build-hy4f/strata-hy4f")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.gguf_py)
    import gguf  # noqa: E402

    r = subprocess.run([a.tokenize, "-m", a.model, "-p", a.prompt, "--ids", "--log-disable"], capture_output=True, text=True,
                       check=True)
    ids = r.stdout.strip().splitlines()[-1].strip("[] ").replace(" ", "")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/opt/rocm/core-10.1/lib" + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    run = subprocess.run([a.runner, a.model, "--ids", ids] + extra, capture_output=True, text=True, env=env)
    sys.stderr.write(run.stderr)
    if run.returncode != 0:
        raise SystemExit(run.returncode)
    out = [int(x) for x in run.stdout.strip().splitlines()[-1].split(",") if x]
    fields = gguf.GGUFReader(a.model).fields
    tokens = fields["tokenizer.ggml.tokens"].contents()
    if "tokenizer.ggml.eos_token_id" in fields and out and out[-1] == int(fields["tokenizer.ggml.eos_token_id"].contents()):
        out = out[:-1]   # generation stopped at EOS: not part of the text
    print(a.prompt + detokenize(tokens, out))


if __name__ == "__main__":
    main()
