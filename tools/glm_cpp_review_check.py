"""Reproducible long-context C++ review request; saves raw API evidence.

Run from the repository root against an already guarded, loopback GLM service.
Scoring is manual against rubric.json; keyword matches are not correctness tests.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--max-tokens', type=int, default=2300)
    args = parser.parse_args()
    fixture = Path('docs/fixtures/glm_cpp_review_20261008')
    prompt = (fixture / 'prompt.txt').read_text()
    body = dict(model='glm53f-reap50-q23-b550-rx9070xt', temperature=0,
                reasoning_effort='low', max_tokens=args.max_tokens,
                messages=[dict(role='user', content=prompt)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.request.json').write_text(json.dumps(body, indent=2))
    start = time.monotonic()
    request = urllib.request.Request(args.url.rstrip('/') + '/v1/chat/completions',
                                     json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    record = dict(wall_seconds=time.monotonic()-start,
                  prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), response=result)
    args.output.with_suffix('.json').write_text(json.dumps(record, indent=2))
    args.output.with_suffix('.md').write_text(result['choices'][0]['message'].get('content') or '')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
