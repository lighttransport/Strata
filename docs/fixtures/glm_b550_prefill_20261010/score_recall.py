"""Score the first assistant turn from the performance tool's raw token output."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--context", type=int, required=True)
    args = parser.parse_args()
    result = json.loads(args.prefix.with_suffix(".result.json").read_text())
    manifest = json.loads(Path(__file__).with_name("recall-manifest.json").read_text())
    prompt = next(p for p in manifest if p["context"] == args.context)
    assert hashlib.sha256((ROOT / prompt["path"]).read_bytes()).hexdigest() == prompt["sha256"]
    if len(result["measurements"]["prefill"]) != 1 or len(result["measurements"]["decode"]) != 1:
        raise ValueError("recall scoring requires exactly one trial")
    metadata = GGUFFile(result["config"]["model"]).metadata
    tokenizer = Tokenizer.from_gguf(result["config"]["model"])
    stops = {int(metadata[k]) for k in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                                      "tokenizer.ggml.eom_token_id") if k in metadata}
    for token in ("<|user|>", "<|observation|>"):
        stops.update(tokenizer.encode(token, parse_special=True))
    ids = list(map(int, args.prefix.with_suffix(".stdout").read_text().split()))
    end = next((i for i, token in enumerate(ids) if token in stops), len(ids))
    text = tokenizer.decode(ids[:end])
    answer = text.rsplit("</think>", 1)[-1].strip()
    if answer.startswith("```json\n") and answer.endswith("```"):
        answer = answer[8:-3].strip()
    try:
        decoded = json.loads(answer)
    except json.JSONDecodeError:
        decoded = None
    count = sum(isinstance(decoded, dict) and decoded.get(k) == v for k, v in prompt["expected"].items())
    scored = dict(context=args.context, input_tokens=prompt["tokens"], expected=prompt["expected"],
                  maximum_generated_tokens=len(ids), first_turn_tokens=end,
                  first_turn_text=text, decoded=decoded, correct_facts=count, total_facts=4,
                  stopped=end < len(ids), valid_run=result["complete"] and result["clean"],
                  passed=result["complete"] and result["clean"] and decoded == prompt["expected"] and end < len(ids))
    args.prefix.with_suffix(".quality.json").write_text(json.dumps(scored, indent=2) + "\n")
    print(json.dumps(scored))


if __name__ == "__main__":
    main()
