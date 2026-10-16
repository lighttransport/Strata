#!/usr/bin/env python3
"""EffiBench-X adapter for the coding-evaluation suite (docs/GLM53F_EVAL.md).

Uses the official harness (github.com/EffiBench/EffiBench-X, Apache-2.0) for prompts, code extraction, the sandbox
and the test runner; this file only selects the frozen subset, talks to our server and writes results.json.
Run it with the harness's Python environment (it needs the harness requirements plus pyarrow):

  H=/path/to/EffiBench-X; PY=/path/to/venv/bin/python
  $PY tools/glm_eval_effibench.py prepare  --harness $H --out runs/effibench/problems
  $PY tools/glm_eval_effibench.py sandbox  --harness $H --port 8000 &          # podman/docker; Python and C++ only
  $PY tools/glm_eval_effibench.py generate --harness $H --problems runs/effibench/problems \
      --config my-serve-profile.json --run runs/effibench/<profile>
  (stop the model server before evaluating: runtimes are measured on the same CPU)
  $PY tools/glm_eval_effibench.py evaluate --harness $H --problems runs/effibench/problems --run runs/effibench/<profile>
  $PY tools/glm_eval_effibench.py evaluate --harness $H --problems runs/effibench/problems --canonical runs/effibench/canonical
  $PY tools/glm_eval_effibench.py report   --problems runs/effibench/problems --run runs/effibench/<profile> \
      --canonical runs/effibench/canonical --profile <name>
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_eval_common as C  # noqa: E402

SUBSET = C.REPO / 'docs/fixtures/glm_eval/effibench_x_subset.json'
SUITE = 'effibench-x-subset-v1'
LANGS = ('python3', 'cpp')


def use_harness(path):
    path = Path(path or os.environ.get('EFFIBENCH_X_DIR', '')).resolve()
    if not (path / 'effibench' / 'run_tests.py').is_file():
        raise SystemExit(f'not an EffiBench-X checkout: {path} (give --harness or set EFFIBENCH_X_DIR)')
    sys.path.insert(0, str(path))
    sys.path.insert(0, str(path / 'third_party' / 'llm-sandbox'))
    return path


def problem_key(p):
    return f'{p["source"]}_{p["id"]}_{p["title_slug"]}'


def load_problems(directory):
    subset = json.loads(SUBSET.read_text())
    out = []
    for p in subset['problems']:
        f = Path(directory) / f'{problem_key(p)}.json'
        if not f.is_file():
            raise SystemExit(f'missing {f}; run prepare first')
        out.append((problem_key(p), json.loads(f.read_text())))
    return out


def cmd_prepare(a):
    harness = use_harness(a.harness)
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(harness / 'data/hf/data/*.parquet')))
    if not files:
        raise SystemExit(f'no dataset under {harness}/data/hf/data; download EffiBench/effibench-x there first')
    subset = json.loads(SUBSET.read_text())
    want = {(p['source'], p['id']): p for p in subset['problems']}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    found = 0
    for f in files:
        for row in pq.read_table(f).to_pylist():
            p = want.get((row['source'], str(row['id'])))
            if p is None:
                continue
            for k in ('generated_tests', 'solutions', 'starter_code', 'test_runners'):
                if isinstance(row.get(k), str):
                    row[k] = json.loads(row[k])
            (out / f'{problem_key(p)}.json').write_text(json.dumps(row, default=str))
            found += 1
    if found != len(want):
        raise SystemExit(f'found {found} of {len(want)} subset problems; dataset revision differs')
    manifest = dict(dataset_files={Path(f).name: C.sha256_file(f) for f in files}, subset_sha256=C.sha256_file(SUBSET))
    (out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'{found} problems written to {out}')


def cmd_sandbox(a):
    use_harness(a.harness)
    sock = f'/run/user/{os.getuid()}/podman/podman.sock'
    if 'DOCKER_HOST' not in os.environ and Path(sock).exists():
        os.environ['DOCKER_HOST'] = f'unix://{sock}'       # rootless podman's Docker-compatible API
    from effibench.utils import EFFIBENCH_REGISTRY
    for lang in list(EFFIBENCH_REGISTRY):
        if lang not in LANGS:
            del EFFIBENCH_REGISTRY[lang]                   # pull and prepare only the Python and C++ images
    import uvicorn
    from effibench.backends import get_backend
    manager, app = get_backend(backend_type='docker', num_workers=a.workers, setup_logging=True, skip_setup=False)
    print(f'sandbox: {manager.num_workers} workers on 127.0.0.1:{a.port} via {os.environ.get("DOCKER_HOST")}',
          flush=True)
    uvicorn.run(app, host='127.0.0.1', port=a.port, log_level='warning')


def build_prompt(problem, lang):
    from effibench.prompts import prompt_solve_problem_functional, prompt_solve_problem_io
    if problem.get('type', 'functional') == 'functional':
        starter = (problem.get('starter_code') or {}).get(lang)
        if not starter:
            return None
        return prompt_solve_problem_functional(problem=problem['description_md'], target_lang=lang,
                                               starter_code=starter)
    return prompt_solve_problem_io(problem=problem['description_md'], target_lang=lang)


def extract(problem, lang, text):
    from effibench.utils import extract_code_blocks, postprocess_solution
    blocks = extract_code_blocks(text)
    if not blocks:
        return None
    code = blocks[-1]['code']
    if problem.get('type', 'functional') == 'functional':
        code = postprocess_solution(code, lang)
    return code if code.strip() else None


def cmd_generate(a):
    use_harness(a.harness)
    problems = load_problems(a.problems)
    run = Path(a.run)
    if run.exists() and not a.resume:
        raise SystemExit(f'{run} exists; use --resume to continue it')
    (run / 'gen').mkdir(parents=True, exist_ok=True)
    langs = a.langs.split(',')
    todo = [(k, p, lang) for k, p in problems for lang in langs if not (run / 'gen' / f'{k}_{lang}.json').exists()]
    if a.limit:
        todo = todo[:a.limit]
    with C.served(a.config, a.base_url, a.port, run / 'server.log') as (url, info):
        meta = dict(info, max_tokens=a.max_tokens, reasoning_effort=a.reasoning_effort, langs=langs,
                    subset_sha256=C.sha256_file(SUBSET), harness='EffiBench-X')
        (run / 'generation.json').write_text(json.dumps(meta, indent=1) + '\n')
        for i, (k, p, lang) in enumerate(todo, 1):
            prompt = build_prompt(p, lang)
            row = dict(problem=k, lang=lang)
            if prompt is None:
                row.update(code=None, error='no starter code for this language')
            else:
                r = C.chat(url, [{'role': 'user', 'content': prompt}], a.max_tokens, a.reasoning_effort)
                row.update(r, code=extract(p, lang, r['content']), prompt_sha256=C.sha256_text(prompt))
            (run / 'gen' / f'{k}_{lang}.json').write_text(json.dumps(row) + '\n')
            print(f'[{i}/{len(todo)}] {k} {lang}: {"code" if row.get("code") else "NO CODE"} '
                  f'{row.get("completion_tokens")} tok {row.get("seconds")} s finish={row.get("finish")}', flush=True)


def evaluate_one(problem, lang, code):
    from effibench.run_tests import run_tests
    tests = problem['generated_tests']
    return run_tests(lang=lang, solution=code, test_cases=tests, evaluator=problem['evaluator'],
                     test_runner=(problem.get('test_runners') or {}).get(lang), early_stop=False,
                     raise_on_error=False, as_batch=True, polling_interval=5)


def cmd_evaluate(a):
    use_harness(a.harness)
    os.environ.setdefault('BACKEND_BASE_URL', a.sandbox)
    problems = load_problems(a.problems)
    langs = a.langs.split(',')
    if a.canonical:
        out = Path(a.canonical) / 'eval'
        jobs = []
        for k, p in problems:
            for lang in langs:
                sol = (p.get('solutions') or {}).get(lang)
                code = sol.get('code') if isinstance(sol, dict) else None
                jobs.append((k, p, lang, code))
    else:
        run = Path(a.run)
        out = run / 'eval'
        jobs = []
        for k, p in problems:
            for lang in langs:
                g = run / 'gen' / f'{k}_{lang}.json'
                if g.exists():
                    jobs.append((k, p, lang, json.loads(g.read_text()).get('code')))
    out.mkdir(parents=True, exist_ok=True)
    for i, (k, p, lang, code) in enumerate(jobs, 1):
        f = out / f'{k}_{lang}.json'
        if f.exists():
            continue
        t0 = time.monotonic()
        results = evaluate_one(p, lang, code) if code else None
        f.write_text(json.dumps(dict(results=results, seconds=round(time.monotonic() - t0, 1))) + '\n')
        ok = results is not None and all(passed_item(r) for r in results)
        print(f'[{i}/{len(jobs)}] {k} {lang}: {"PASS" if ok else "fail"}', flush=True)


ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def postprocess_text(text):
    """effibench.run_tests.postprocess_text: strip ANSI escapes, normalise line ends, strip."""
    return ANSI.sub('', str(text)).replace('\r\r\n', '\n').replace('\r\n', '\n').strip()


def passed_item(r):
    """EffiBench-X evaluate_solution.is_passed per test (Apache-2.0 harness logic)."""
    if r.get('passed', False):
        return True
    if r.get('status') == 'done' and r.get('text') == '' and r.get('exit_code') == 0:
        return True
    return postprocess_text(r.get('text', '')) == postprocess_text(r.get('input', '') + r.get('output', ''))


def stats(eval_file):
    if not eval_file.exists():
        return None
    results = json.loads(eval_file.read_text())['results']
    if not results:
        return dict(passed=False, tests=0, tests_passed=0)
    results = [r for r in results if r.get('runtime', 0) < 10_000_000_000]
    ok = [passed_item(r) for r in results]
    s = dict(passed=bool(results) and all(ok), tests=len(results), tests_passed=sum(ok))
    if s['passed']:
        s['runtime_sum_s'] = sum(r['runtime'] for r in results) / 1e9
        s['memory_kb'] = max(r['memory'] for r in results) / 1e3
    return s


def cmd_report(a):
    problems = load_problems(a.problems)
    run, canon = Path(a.run), Path(a.canonical) if a.canonical else None
    langs = a.langs.split(',')
    tasks = []
    for k, p in problems:
        for lang in langs:
            s = stats(run / 'eval' / f'{k}_{lang}.json')
            if s is None:
                continue
            g = json.loads((run / 'gen' / f'{k}_{lang}.json').read_text())
            m = dict(s, lang=lang, type=p.get('type'), completion_tokens=g.get('completion_tokens'),
                     finish=g.get('finish'), gen_seconds=g.get('seconds'))
            c = stats(canon / 'eval' / f'{k}_{lang}.json') if canon else None
            if c and c.get('passed') and s['passed']:
                m['runtime_score'] = min(1.0, c['runtime_sum_s'] / max(s['runtime_sum_s'], 1e-9))
                m['memory_score'] = min(1.0, c['memory_kb'] / max(s['memory_kb'], 1e-9))
            elif c and c.get('passed'):
                m['runtime_score'] = m['memory_score'] = 0.0
            tasks.append(dict(id=f'{k}:{lang}', passed=s['passed'], metrics=m))
    meta = dict(generation=json.loads((run / 'generation.json').read_text()), languages=langs,
                canonical=str(canon) if canon else None)
    for lang in langs:
        rows = [t['metrics'] for t in tasks if t['metrics']['lang'] == lang]
        scored = [r['runtime_score'] for r in rows if 'runtime_score' in r]
        meta[f'{lang}_pass'] = f'{sum(r["passed"] for r in rows)}/{len(rows)}'
        if scored:
            meta[f'{lang}_runtime_score'] = round(sum(scored) / len(scored), 4)
    data = C.write_results(run / 'results.json', SUITE, a.profile, tasks, meta)
    print(json.dumps({k: v for k, v in data['meta'].items() if k.endswith(('_pass', '_score'))}, indent=1))
    print(f'{data["passed"]}/{data["total"]} passed -> {run / "results.json"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--harness')
    p.add_argument('--out', required=True)
    p = sub.add_parser('sandbox')
    p.add_argument('--harness')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--workers', type=int, default=8)
    p = sub.add_parser('generate')
    p.add_argument('--harness')
    p.add_argument('--problems', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--config', help='GLM serve profile to start serve.server with')
    p.add_argument('--base-url', help='use a server that is already running')
    p.add_argument('--port', type=int, default=8095)
    p.add_argument('--langs', default=','.join(LANGS))
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--reasoning-effort', default='low')
    p.add_argument('--limit', type=int, help='first N problem/language pairs only (smoke test)')
    p.add_argument('--resume', action='store_true')
    p = sub.add_parser('evaluate')
    p.add_argument('--harness')
    p.add_argument('--problems', required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--run')
    g.add_argument('--canonical', help='evaluate the dataset reference solutions into this directory')
    p.add_argument('--sandbox', default='http://127.0.0.1:8000')
    p.add_argument('--langs', default=','.join(LANGS))
    p = sub.add_parser('report')
    p.add_argument('--problems', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--canonical')
    p.add_argument('--profile', required=True)
    p.add_argument('--langs', default=','.join(LANGS))
    a = ap.parse_args()
    globals()[f'cmd_{a.cmd}'](a)


if __name__ == '__main__':
    main()
