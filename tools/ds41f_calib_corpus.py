#!/usr/bin/env python3
"""Calibration prompt files for the DeepSeek V4.1 Flash imatrix job (gemm a64fx/ds41f/pjsub_ds41f_imatrix_12n.sh).

    python3 tools/ds41f_calib_corpus.py --tokenizer TOKENIZER.json --out DIR \
        [--text NAME=FILE ...] [--repo NAME=GIT_DIR ...] [--mix NAME=WEIGHT ...] [--tokens 32768] [--files 20] [--doc-tokens 2048] [--max-pieces 2]
        [--exclude-sha FILE]

Documents: a --text file splits at blank lines (paragraph groups of at least 512 characters); a --repo contributes its tracked
code and docs (.c .cc .cpp .h .hpp .py .md .sh, outside third_party, 1-64 KiB). Each document is BOS (id 0) + its tokens
(plain text, no chat template; as gemm's tokenizer_io.py). Documents are ordered by content sha256 (deterministic), and
every output file draws from the sources in proportion to --mix (default: equal), so any prefix of the file list has the
same mix. A document longer than --doc-tokens is split into pieces (each BOS-prefixed) that are queued in order, so one long
file cannot dominate an output file (at most --max-pieces pieces per document, for diversity); a piece that does not fit
the rest of a file is cut. --exclude-sha drops documents whose sha256 is listed
(keep a held-out evaluation set disjoint). Writes DIR/NNNN.ids (one line of ids, the runner's --prompt-ids format) and
DIR/manifest.json. Needs the Hugging Face `tokenizers` package.
"""
import argparse, hashlib, json, pathlib, subprocess, sys


def text_documents(path):
    docs, cur = [], []
    for para in pathlib.Path(path).read_text(errors="replace").split("\n\n"):
        cur.append(para)
        if sum(len(p) for p in cur) >= 512:
            docs.append("\n\n".join(cur)); cur = []
    if cur:
        docs.append("\n\n".join(cur))
    return docs


def repo_documents(root):
    root = pathlib.Path(root)
    names = subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()
    docs = []
    for n in names:
        p = pathlib.PurePosixPath(n)
        if p.suffix not in (".c", ".cc", ".cpp", ".h", ".hpp", ".py", ".md", ".sh") or "third_party" in p.parts:
            continue
        f = root / n
        try:
            raw = f.read_bytes()
        except OSError:
            continue
        if 1024 <= len(raw) <= 65536:
            docs.append(raw.decode(errors="replace"))
    return docs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True); ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--text", action="append", default=[]); ap.add_argument("--repo", action="append", default=[])
    ap.add_argument("--mix", action="append", default=[]); ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--files", type=int, default=20); ap.add_argument("--doc-tokens", type=int, default=2048)
    ap.add_argument("--max-pieces", type=int, default=2); ap.add_argument("--exclude-sha")
    a = ap.parse_args()
    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit("needs the `tokenizers` package (pip install tokenizers)")
    tok = Tokenizer.from_file(a.tokenizer)
    exclude = set(pathlib.Path(a.exclude_sha).read_text().split()) if a.exclude_sha else set()
    sources = {}
    for spec, reader in [(s, text_documents) for s in a.text] + [(s, repo_documents) for s in a.repo]:
        name, _, path = spec.partition("=")
        docs = [(hashlib.sha256(d.encode()).hexdigest(), d) for d in reader(path)]
        docs = sorted({h: d for h, d in docs if h not in exclude}.items())
        sources[name] = {"path": path, "docs": docs, "next": 0, "used": 0, "pending": []}
    if not sources:
        sys.exit("give at least one --text or --repo")
    weights = {n: 1.0 for n in sources}
    for m in a.mix:
        n, _, w = m.partition("="); weights[n] = float(w)
    total_w = sum(weights[n] for n in sources)
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = {"tokenizer_sha256": hashlib.sha256(pathlib.Path(a.tokenizer).read_bytes()).hexdigest(), "tokens_per_file": a.tokens,
                "sources": {n: {"path": s["path"], "documents": len(s["docs"]), "weight": weights[n]} for n, s in sources.items()}, "files": []}
    for i in range(a.files):
        ids, taken, used = [], {n: 0 for n in sources}, []
        while len(ids) < a.tokens:
            live = [n for n in sources if sources[n]["pending"] or sources[n]["next"] < len(sources[n]["docs"])]
            if not live:
                break
            n = max(live, key=lambda k: weights[k] / total_w * (len(ids) + 1) - taken[k])     # largest deficit
            src = sources[n]
            if not src["pending"]:
                h, d = src["docs"][src["next"]]; src["next"] += 1; used.append(h)
                body = tok.encode(d, add_special_tokens=False).ids; step = a.doc_tokens - 1
                src["pending"] = [[0] + body[k:k + step] for k in range(0, max(len(body), 1), step)][: a.max_pieces]
            t = src["pending"].pop(0)[: a.tokens - len(ids)]
            ids += t; taken[n] += len(t)
        if len(ids) < a.tokens:
            print(f"sources exhausted after {i} full files"); break
        name = f"{i:04d}.ids"
        (a.out / name).write_text(" ".join(map(str, ids)) + "\n")
        manifest["files"].append({"name": name, "tokens": len(ids), "by_source": taken, "documents_sha256": used})
        print(f"{name}: " + ", ".join(f"{n} {taken[n]}" for n in sources))
    for n, s in sources.items():
        manifest["sources"][n]["documents_used"] = s["next"]
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {len(manifest['files'])} files x {a.tokens} tokens to {a.out}")


if __name__ == "__main__":
    main()
