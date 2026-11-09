#!/usr/bin/env python3
"""Ripple tokenizer helper (the `tokenizers` package; no chat-template engine needed for a single user turn).

  ripple_tok.py --model DIR encode "text" OUT_IDS [--chat]   # --chat wraps one user turn in the model's <|im_start|> template
  ripple_tok.py --model DIR decode 1,2,3                     # ids -> text"""
import argparse, sys
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("cmd", choices=["encode", "decode"])
ap.add_argument("arg")
ap.add_argument("out", nargs="?")
ap.add_argument("--chat", action="store_true")
a = ap.parse_args()
tok = Tokenizer.from_file(f"{a.model}/tokenizer.json")
if a.cmd == "encode":
    text = f"<|im_start|>user\n{a.arg}<|im_end|>\n<|im_start|>assistant\n" if a.chat else a.arg
    ids = tok.encode(text).ids
    open(a.out, "w").write(" ".join(map(str, ids)) + "\n")
    print(len(ids), ids[:16])
else:
    print(tok.decode([int(t) for t in a.arg.split(",") if t], skip_special_tokens=False))
