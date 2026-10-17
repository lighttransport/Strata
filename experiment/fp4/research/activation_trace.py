#!/usr/bin/env python3
"""Convert independent GLM expert-input traces into full-width research inputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def vector_key(raw):
    """Finite float32 vector fingerprint; canonicalize signed zero."""
    values = [v[0] for v in struct.iter_unpack("<f", raw)]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite activation vector")
    canonical = struct.pack("<"+"f"*len(values), *(0.0 if v == 0 else v for v in values))
    return hashlib.sha256(canonical).digest()


def read_trace(path, layer, experts, expert=None, max_vectors=2048):
    if path.stat().st_size > 512*1024**2:
        raise ValueError("trace exceeds 512 MiB cap")
    digest = hashlib.sha256()
    vectors, provenance, positions = [], [], set()
    with path.open("rb") as file:
        def read(size):
            raw = file.read(size)
            if len(raw) != size:
                raise ValueError("truncated activation trace")
            digest.update(raw)
            return raw
        record = 0
        while True:
            header = file.read(24)
            if not header:
                break
            if len(header) != 24:
                raise ValueError("truncated trace header")
            digest.update(header)
            magic, current, position, tokens, width, topk = struct.unpack("<6I",header)
            if (magic != 0x31435247 or not 3 <= current <= 44 or not 1 <= tokens <= 8192 or
                    width != 4096 or topk != 8 or position + tokens > 2**32):
                raise ValueError("invalid GLM expert-input trace geometry")
            ids = struct.unpack("<"+"i"*(tokens*topk),read(tokens*topk*4))
            for token in range(tokens):
                routing = ids[token*topk:(token+1)*topk]
                if len(set(routing)) != topk or any(e < 0 or e >= experts for e in routing):
                    raise ValueError("invalid trace routing")
                raw = read(width*4)
                if current != layer:
                    continue
                key = position+token
                if key in positions:
                    raise ValueError("repeated layer/token position in one trace")
                positions.add(key)
                if expert is not None and expert not in routing:
                    continue
                vector_key(raw)
                if len(vectors) >= max_vectors:
                    raise ValueError("selected trace exceeds vector cap")
                vectors.append(raw)
                provenance.append({"record":record,"row":token,"position":key})
            record += 1
    if not vectors:
        raise ValueError("trace has no selected layer/expert inputs")
    return vectors, {"sha256":digest.hexdigest(),"records":record,"selected":provenance}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for split in ("train","validation","heldout"):
        p.add_argument("--"+split+"-trace",type=Path,nargs="+",required=True)
    p.add_argument("--layer",type=int,required=True)
    p.add_argument("--experts",type=int,choices=(144,288),required=True)
    p.add_argument("--expert",type=int,help="retain only inputs routed to this expert; omit for router inputs")
    p.add_argument("--max-vectors",type=int,default=2048,help="cap per split; excess is rejected rather than sampled")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if not 3 <= a.layer <= 44 or not 2 <= a.max_vectors <= 2048 or (a.expert is not None and not 0 <= a.expert < a.experts):
        p.error("invalid layer, expert or vector cap")
    vectors, sources, trace_hashes, keys = {}, {}, set(), []
    for split in ("train","validation","heldout"):
        selected, records = [], []
        for path in getattr(a,split+"_trace"):
            rows, info=read_trace(path,a.layer,a.experts,a.expert,a.max_vectors-len(selected))
            if info["sha256"] in trace_hashes:
                raise ValueError("same trace supplied more than once or across splits")
            trace_hashes.add(info["sha256"])
            selected.extend(rows); records.append(info)
        if len(selected) < 2:
            raise ValueError("each split requires at least two selected vectors")
        fingerprints=set(map(vector_key,selected))
        if any(fingerprints & previous for previous in keys):
            raise ValueError("activation vectors overlap across splits")
        keys.append(fingerprints); vectors[split]=selected; sources[split]=records
    outputs={"calibration-large.f32":b"".join(vectors["train"]),
             "validation.f32":b"".join(vectors["validation"]),
             "heldout.f32":b"".join(vectors["heldout"]),
             "activation.f32":vectors["train"][0]}
    a.output.mkdir(parents=True,exist_ok=True)
    for name,raw in outputs.items():
        (a.output/name).write_bytes(raw)
    (a.output/"activation-provenance.json").write_text(json.dumps({
        "scope":"GLM expert-input capture; separate prompt sets must be supplied by caller",
        "input_source":"captured", "capture_kind":"expert_input", "layer":a.layer,
        "input_width":4096,"experts":a.experts,"expert_filter":a.expert,
        "counts":{split:len(rows) for split,rows in vectors.items()},"trace_sources":sources,
        "sha256":{name:hashlib.sha256(raw).hexdigest() for name,raw in outputs.items()},
    },indent=2)+"\n")
    print("Converted disjoint expert-input traces; prompt independence is caller-supplied")


if __name__ == "__main__":
    main()
