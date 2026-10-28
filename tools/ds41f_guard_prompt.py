#!/usr/bin/env python3
"""Validate DS41 prompt IDs against GGUF metadata without loading tensor data."""
from __future__ import annotations

import struct
import sys
import re
from pathlib import Path


class GGUFHeader:
    scalar_bytes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def __init__(self, path: Path):
        self.file = path.open("rb")
        self.size = path.stat().st_size
        self.vocab = 0
        self.embedding_vocab = 0
        try:
            self._parse()
        finally:
            self.file.close()

    def read(self, size: int) -> bytes:
        if size < 0 or size > self.size - self.file.tell():
            raise ValueError("truncated GGUF header")
        value = self.file.read(size)
        if len(value) != size:
            raise ValueError("truncated GGUF header")
        return value

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self, *, decode: bool = False) -> str | None:
        size = self.u64()
        if size > self.size - self.file.tell() or size > 1_048_576:
            raise ValueError("invalid GGUF string length")
        raw = self.read(size)
        return raw.decode("utf-8") if decode else None

    def skip_string(self) -> None:
        size = self.u64()
        if size > self.size - self.file.tell():
            raise ValueError("invalid GGUF string span")
        self.file.seek(size, 1)

    def skip_value(self, kind: int) -> None:
        if kind in self.scalar_bytes:
            self.read(self.scalar_bytes[kind])
            return
        if kind == 8:
            self.skip_string()
            return
        if kind != 9:
            raise ValueError("unsupported GGUF metadata type")
        elem = self.u32()
        count = self.u64()
        if elem == 8:
            if count > 1_000_000:
                raise ValueError("GGUF string array is too large")
            for _ in range(count):
                self.skip_string()
            return
        if elem not in self.scalar_bytes:
            raise ValueError("unsupported GGUF array element type")
        width = self.scalar_bytes[elem]
        if count > (self.size - self.file.tell()) // width:
            raise ValueError("invalid GGUF array span")
        self.file.seek(count * width, 1)

    def _parse(self) -> None:
        magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", self.read(24))
        if magic != b"GGUF" or version != 3 or n_tensors > 1_000_000 or n_kv > 1_000_000:
            raise ValueError("unsupported GGUF header")

        alignment = 32
        for _ in range(n_kv):
            key = self.string(decode=True)
            kind = self.u32()
            if key == "tokenizer.ggml.tokens":
                if kind != 9 or self.u32() != 8:
                    raise ValueError("tokenizer vocabulary is not a string array")
                self.vocab = self.u64()
                if self.vocab < 1 or self.vocab > 1_000_000:
                    raise ValueError("tokenizer vocabulary size is invalid")
                for _ in range(self.vocab):
                    self.skip_string()
            elif key == "general.alignment" and kind == 4:
                alignment = self.u32()
                if alignment < 1 or alignment > 4096 or alignment & (alignment - 1):
                    raise ValueError("GGUF alignment is invalid")
            else:
                self.skip_value(kind)

        # Tensor descriptors immediately follow metadata; alignment applies to
        # the tensor data section after the descriptors, not to this directory.
        for _ in range(n_tensors):
            name = self.string(decode=True)
            n_dims = self.u32()
            if n_dims < 1 or n_dims > 4:
                raise ValueError("GGUF tensor rank is invalid")
            shape = struct.unpack(f"<{n_dims}Q", self.read(8 * n_dims))
            self.read(12)  # ggml type and tensor data offset
            if name == "token_embd.weight":
                if len(shape) != 2:
                    raise ValueError("input embedding tensor rank is invalid")
                self.embedding_vocab += shape[1]


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: ds41f_guard_prompt.py MODEL_GGUF PROMPT_IDS", file=sys.stderr)
        return 2
    try:
        model = Path(sys.argv[1])
        match = re.fullmatch(r"(.*)-(\d{5})-of-(\d{5})\.gguf", model.name)
        if match:
            prefix, number, total = match.groups()
            count = int(total)
            if int(number) != 1 or count < 1 or count > 1000:
                raise ValueError("invalid first shard or shard count")
            paths = [model.with_name(f"{prefix}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]
        else:
            paths = [model]
        headers = [GGUFHeader(path) for path in paths]
        vocab = headers[0].vocab
        if not vocab or any(h.vocab not in (0, vocab) for h in headers):
            raise ValueError("missing or inconsistent tokenizer vocabulary")
        if sum(h.embedding_vocab for h in headers) != vocab:
            raise ValueError("input embedding size disagrees with vocabulary")
        prompt_path = Path(sys.argv[2])
        if prompt_path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("prompt file is too large")
        count = 0
        with prompt_path.open("r", encoding="ascii") as prompt:
            for line in prompt:
                for raw in line.replace(",", " ").split():
                    if not raw.isascii() or not raw.isdecimal() or len(raw) > 10:
                        raise ValueError("prompt contains a malformed token ID")
                    token = int(raw, 10)
                    if token > 2_147_483_647 or token >= vocab:
                        raise ValueError("prompt token ID is outside the model vocabulary")
                    count += 1
                    if count > 131_072:
                        raise ValueError("prompt contains too many token IDs")
        if count == 0:
            raise ValueError("prompt contains no token IDs")
    except Exception:
        print("refusing decode: GGUF/prompt preflight failed", file=sys.stderr)
        return 1

    print(vocab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
