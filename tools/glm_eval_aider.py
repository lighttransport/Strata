#!/usr/bin/env python3
"""Aider polyglot adapter for the coding-evaluation suite (docs/GLM53F_EVAL.md): the 34 Python and 26 C++ Exercism
exercises, edited in place by Aider against our OpenAI-compatible server, two attempts, judged by the exercise tests.

The official harness (aider/benchmark/benchmark.py) runs inside a container because it executes model-written code;
we use rootless podman with a slim image (tools/eval/aider-polyglot.Dockerfile) and host networking so the container
reaches the server on 127.0.0.1.

  python3 tools/glm_eval_aider.py build  --aider /path/to/aider
  python3 tools/glm_eval_aider.py run    --aider /path/to/aider --exercises /path/to/polyglot-benchmark \
      --config my-serve-profile.json --run runs/aider/<profile>
  python3 tools/glm_eval_aider.py report --run runs/aider/<profile> --profile <name>
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_eval_common as C  # noqa: E402

IMAGE = 'strata-aider-polyglot'
SUITE = 'aider-polyglot-cpp-python-v1'
LANGS = ('cpp', 'python')


def podman(*args, **kw):
    return subprocess.run(['podman', *args], check=True, **kw)


def cmd_build(a):
    podman('build', '-t', IMAGE, '-f', str(C.REPO / 'tools/eval/aider-polyglot.Dockerfile'), str(Path(a.aider)))


def git_rev(path):
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=path, capture_output=True, text=True).stdout.strip()


def cmd_run(a):
    run = Path(a.run).resolve()
    if run.exists() and not a.resume:
        raise SystemExit(f'{run} exists; use --resume to continue it')
    bench = run / 'bench'
    src = Path(a.exercises).resolve()
    for lang in LANGS:
        dst = bench / 'polyglot-benchmark' / lang
        if not dst.exists():
            shutil.copytree(src / lang, dst)
    settings = run / 'model-settings.yml'
    settings.write_text(f'- name: openai/{a.model}\n  edit_format: {a.edit_format}\n  use_temperature: true\n'
                        f'  extra_params:\n    max_tokens: {a.max_tokens}\n')
    with C.served(a.config, a.base_url, a.port, run / 'server.log') as (url, info):
        meta = dict(info, edit_format=a.edit_format, tries=a.tries, max_tokens=a.max_tokens,
                    reasoning_effort=a.reasoning_effort, aider_commit=git_rev(a.aider),
                    exercises_commit=git_rev(src), languages=list(LANGS))
        (run / 'generation.json').write_text(json.dumps(meta, indent=1) + '\n')
        cmd = ['run', '--rm', '--network', 'host', '--memory', '12g',
               '-v', f'{bench}:/benchmarks:Z', '-v', f'{settings}:/settings.yml:ro,Z',
               '-e', 'AIDER_DOCKER=1', '-e', 'AIDER_BENCHMARK_DIR=/benchmarks',
               '-e', f'OPENAI_API_BASE={url}/v1', '-e', 'OPENAI_API_KEY=local',
               IMAGE, 'python', 'benchmark/benchmark.py', a.name, '--model', f'openai/{a.model}',
               '--edit-format', a.edit_format, '--read-model-settings', '/settings.yml', '--threads', '1',
               '--tries', str(a.tries), '--languages', ','.join(LANGS), '--exercises-dir', 'polyglot-benchmark']
        if a.reasoning_effort:
            cmd += ['--reasoning-effort', a.reasoning_effort]
        if a.num_tests:
            cmd += ['--num-tests', str(a.num_tests)]
        cmd += ['--cont'] if a.resume and list(bench.glob(f'*--{a.name}')) else ['--new']
        with open(run / 'aider.log', 'ab') as log:
            podman(*cmd, stdout=log, stderr=subprocess.STDOUT)


def cmd_report(a):
    run = Path(a.run)
    dirs = [d for d in (run / 'bench').glob('*--*') if d.is_dir()]
    if len(dirs) != 1:
        raise SystemExit(f'expected one benchmark directory under {run}/bench, found {len(dirs)}')
    tasks = []
    for f in sorted(dirs[0].glob('*/exercises/practice/*/.aider.results.json')):
        r = json.loads(f.read_text())
        lang, name = f.parts[-5], f.parts[-2]
        outcomes = r.get('tests_outcomes') or []
        tasks.append(dict(id=f'{lang}/{name}', passed=bool(outcomes and outcomes[-1]),
                          metrics=dict(lang=lang, first_try=bool(outcomes and outcomes[0]), tries=len(outcomes),
                                       seconds=r.get('duration'), malformed=r.get('num_malformed_responses', 0),
                                       timeouts=r.get('test_timeouts', 0),
                                       completion_tokens=r.get('completion_tokens'))))
    meta = dict(generation=json.loads((run / 'generation.json').read_text()))
    for lang in LANGS:
        rows = [t for t in tasks if t['metrics']['lang'] == lang]
        meta[f'{lang}_pass'] = f'{sum(t["passed"] for t in rows)}/{len(rows)}'
        meta[f'{lang}_pass_first_try'] = f'{sum(t["metrics"]["first_try"] for t in rows)}/{len(rows)}'
    data = C.write_results(run / 'results.json', SUITE, a.profile, tasks, meta)
    print(json.dumps({k: v for k, v in meta.items() if k.endswith(('_pass', '_try'))}, indent=1))
    print(f'{data["passed"]}/{data["total"]} passed -> {run / "results.json"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('build')
    p.add_argument('--aider', required=True, help='Aider checkout (build context)')
    p = sub.add_parser('run')
    p.add_argument('--aider', required=True)
    p.add_argument('--exercises', required=True, help='polyglot-benchmark checkout')
    p.add_argument('--run', required=True)
    p.add_argument('--name', default='strata')
    p.add_argument('--config', help='GLM serve profile to start serve.server with')
    p.add_argument('--base-url', help='use a server that is already running')
    p.add_argument('--port', type=int, default=8095)
    p.add_argument('--model', default='glm')
    p.add_argument('--edit-format', default='diff')
    p.add_argument('--tries', type=int, default=2)
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--reasoning-effort', default='low')
    p.add_argument('--num-tests', type=int, help='first N exercises only (smoke test)')
    p.add_argument('--resume', action='store_true')
    p = sub.add_parser('report')
    p.add_argument('--run', required=True)
    p.add_argument('--profile', required=True)
    a = ap.parse_args()
    globals()[f'cmd_{a.cmd}'](a)


if __name__ == '__main__':
    main()
