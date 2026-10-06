"""Freeze disjoint 32K-token calibration/evaluation corpora from committed code and prose."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--tokens", type=int, default=32768)
    args = parser.parse_args()
    if args.tokens < 1024 or args.tokens % 1024:
        parser.error("tokens must be a positive multiple of 1024")
    root = pathlib.Path(__file__).resolve().parents[1]
    def git(*words):
        return subprocess.check_output(["git", *words], cwd=root)
    revision = git("rev-parse", "HEAD").decode().strip()
    tk = Tokenizer.from_gguf(args.model)
    metadata = GGUFFile(args.model).metadata
    bos = metadata.get("tokenizer.ggml.bos_token_id")
    manifest = {"revision": revision, "tokens_per_split": args.tokens, "sequence_targets": 512,
                "source": "committed repository files; calibration/evaluation content hashes disjoint", "files": []}
    buckets = {(split, kind): [] for split in ("calibration", "evaluation") for kind in ("code", "prose")}
    seen = set()
    paths = sorted(git("ls-tree", "-r", "--name-only", revision).decode().splitlines(), key=lambda p: hashlib.sha256(p.encode()).digest())
    for name in paths:
        path = pathlib.PurePosixPath(name)
        if path.suffix not in (".cpp", ".hpp", ".py", ".md") or any(part in ("third_party", "fixtures", "tests", "test") for part in path.parts) or path.name.startswith("test_"):
            continue
        raw = git("show", f"{revision}:{name}")
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen or len(raw) < 1024 or len(raw) > 524288:
            continue
        seen.add(digest)
        split = "calibration" if int(digest[:2], 16) % 2 == 0 else "evaluation"
        kind = "prose" if path.suffix == ".md" else "code"
        bucket = buckets[split, kind]
        if len(bucket) >= args.tokens // 2 + 512:
            continue
        ids = tk.encode(raw.decode("utf-8"), parse_special=False)
        bucket.extend(ids)
        manifest["files"].append({"path": name, "sha256": digest, "split": split, "kind": kind})
        if all(len(v) >= args.tokens // 2 + 512 for v in buckets.values()):
            break
    if any(len(v) < args.tokens // 2 + 1 for v in buckets.values()):
        raise RuntimeError("not enough disjoint corpus content")
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("calibration", "evaluation"):
        output = args.output / (split + ".ids")
        if output.exists():
            raise RuntimeError(f"refusing to overwrite {output}")
        lines = []
        for kind in ("code", "prose"):
            ids = buckets[split, kind]
            for offset in range(0, args.tokens // 2, 512):
                sequence = [bos] + ids[offset:offset + 512] if bos is not None else ids[offset:offset + 513]
                lines.append(",".join(map(str, sequence)))
        output.write_text("\n".join(lines) + "\n")
        manifest[split + "_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

if __name__ == "__main__":
    main()
