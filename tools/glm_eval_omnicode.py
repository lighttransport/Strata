#!/usr/bin/env python3
"""OmniCode adapter for the coding-evaluation suite, tier 2 (docs/GLM53F_EVAL.md).

Repository-level tasks from github.com/seal-research/OmniCode: bug fixing and test generation on real pull requests in
Python (Django, SymPy, Scrapy, ...) and C++ (nlohmann/json, fmt, Catch2, simdjson). An agent (SWE-agent, from the
OmniCode checkout's submodules) works in the instance's container against our OpenAI-compatible server and produces a
patch; OmniCode's evaluator applies it and runs the tests. Containers run on rootless podman through its Docker API
socket; images are removed after each repository to stay inside the free disk space.

  python3 tools/glm_eval_omnicode.py sample --omnicode /path/to/OmniCode          # once: the frozen sample (committed)
  $PY tools/glm_eval_omnicode.py agent    --omnicode $O --task bugfixing --lang cpp --config <serve profile> --run R
  $PY tools/glm_eval_omnicode.py evaluate --omnicode $O --task bugfixing --lang cpp --run R
  python3 tools/glm_eval_omnicode.py report --run R --profile <name>
$PY is a Python 3.13 environment with OmniCode's requirements and its SWE-agent, SWE-ReX, SWE-bench and
multi-swe-bench submodules installed (see docs/GLM53F_EVAL.md).
"""
import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_eval_common as C  # noqa: E402

SAMPLE = C.REPO / 'docs/fixtures/glm_eval/omnicode_sample.json'
SUITE = 'omnicode-sample-v1'
# (language, task) -> instances drawn, proportional to each repository's share
SIZES = {('python', 'bugfixing'): 40, ('cpp', 'bugfixing'): 40, ('python', 'testgeneration'): 20,
         ('cpp', 'testgeneration'): 20}
MODES = {('python', 'bugfixing'): 'bugfixing', ('cpp', 'bugfixing'): 'bugfixing-cpp',
         ('python', 'testgeneration'): 'testgen', ('cpp', 'testgeneration'): 'testgen-cpp'}


def repo_of(instance_id):
    org, rest = instance_id.split('__', 1)
    return f'{org}/{rest.rsplit("-", 1)[0].rsplit("_", 1)[0]}'


def draw(ids, n, salt):
    by_repo = collections.defaultdict(list)
    for i in ids:
        by_repo[repo_of(i)].append(i)
    key = lambda i: hashlib.sha256(f'{salt}:{i}'.encode()).hexdigest()  # noqa: E731
    # largest-remainder apportionment, at least one per repository when n allows
    quotas = {r: n * len(v) / len(ids) for r, v in by_repo.items()}
    take = {r: max(1, int(q)) if n >= len(by_repo) else int(q) for r, q in quotas.items()}
    for r in sorted(quotas, key=lambda r: quotas[r] - int(quotas[r]), reverse=True):
        if sum(take.values()) >= n:
            break
        take[r] += 1
    while sum(take.values()) > n:
        r = max(take, key=lambda r: (take[r], r))
        take[r] -= 1
    out = []
    for r, v in sorted(by_repo.items()):
        out += sorted(v, key=key)[:min(take[r], len(v))]
    return sorted(out)


def cmd_sample(a):
    root = Path(a.omnicode)
    sample = dict(benchmark='OmniCode', source='data/summary/<lang>_instance_list.json in the OmniCode checkout',
                  omnicode_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True,
                                                 text=True).stdout.strip(),
                  selection='per (language, task): largest-remainder apportionment by repository, then '
                            'sha256("omnicode-sample-v1:<lang>:<task>:<id>") order within a repository',
                  sets={})
    for (lang, task), n in SIZES.items():
        ids = json.loads((root / f'data/summary/{lang}_instance_list.json').read_text())[task]
        pick = draw(ids, n, f'omnicode-sample-v1:{lang}:{task}')
        sample['sets'][f'{lang}/{task}'] = dict(population=len(ids), count=len(pick),
                                                repos=dict(collections.Counter(map(repo_of, pick))),
                                                instances=pick)
        print(f'{lang}/{task}: {len(pick)} of {len(ids)} {dict(collections.Counter(map(repo_of, pick)))}')
    SAMPLE.write_text(json.dumps(sample, indent=1) + '\n')


def sample_ids(lang, task):
    return json.loads(SAMPLE.read_text())['sets'][f'{lang}/{task}']['instances']


def dataset(root, lang):
    f = Path(root) / f'data/omnicode_instances_{lang}.json'
    if not f.is_file():
        raise SystemExit(f'{f} missing: hf download seal-research/OmniCode --repo-type dataset --local-dir {f.parent}')
    return {d['instance_id']: d for d in json.loads(f.read_text())}


def podman_env():
    sock = f'/run/user/{os.getuid()}/podman/podman.sock'
    env = dict(os.environ)
    if 'DOCKER_HOST' not in env and Path(sock).exists():
        env['DOCKER_HOST'] = f'unix://{sock}'
    return env


def image_for(inst, lang, mode):
    if lang == 'cpp':
        return f'omnicodeorg/omnicode:{inst["repo"].replace("/", "_")}_base'
    return f'omnicodeorg/omnicode:{inst["instance_id"]}'


def cmd_agent(a):
    """SWE-agent per instance against our server; the arguments follow OmniCode's baselines/sweagent/sweagent_regular.py
    with the model pointed at a local OpenAI-compatible endpoint and the dollar cost limits disabled."""
    root = Path(a.omnicode).resolve()
    run = Path(a.run).resolve()
    out = run / f'{a.lang}-{a.task}'
    out.mkdir(parents=True, exist_ok=True)
    data = dataset(root, a.lang)
    ids = sample_ids(a.lang, a.task)[:a.limit or None]
    mode = MODES[(a.lang, a.task)]
    config = root / 'baselines/sweagent/configs' / Path(
        {'bugfixing': 'bugfixing.yaml', 'bugfixing-cpp': 'bugfixing_cpp.yaml', 'testgen': 'testgen.yaml',
         'testgen-cpp': 'testgen_cpp.yaml'}[mode])
    env = podman_env()
    with C.served(a.config, a.base_url, a.port, run / 'server.log') as (url, info):
        (run / 'generation.json').write_text(json.dumps(dict(info, agent='SWE-agent (OmniCode fork)',
                                                             step_limit=a.step_limit, max_tokens=a.max_tokens),
                                                        indent=1) + '\n')
        for n, iid in enumerate(ids, 1):
            pred = out / iid / f'{iid}.pred'
            if pred.exists():
                continue
            inst = data[iid]
            ps = out / f'{iid}.problem.txt'
            ps.write_text(inst['problem_statement'])
            cmd = [sys.executable, '-m', 'sweagent', 'run', '--config', str(config),
                   f'--agent.model.name=openai/{a.model}', f'--agent.model.api_base={url}/v1',
                   '--agent.model.api_key=local', '--agent.model.per_instance_cost_limit=0',
                   '--agent.model.total_cost_limit=0', f'--agent.model.per_instance_call_limit={a.step_limit}',
                   '--agent.model.temperature=0', f'--agent.model.max_output_tokens={a.max_tokens}',
                   f'--env.repo.github_url=https://github.com/{inst["repo"]}',
                   f'--env.repo.base_commit={inst["base_commit"]}',
                   f'--env.deployment.image={image_for(inst, a.lang, mode)}',
                   '--env.deployment.docker_args=["-w","/"]',
                   f'--problem_statement.path={ps}', f'--problem_statement.id={iid}', f'--output_dir={out}']
            with open(out / f'{iid}.agent.log', 'ab') as log:
                rc = subprocess.run(cmd, cwd=root / 'baselines/sweagent', env=env, stdout=log,
                                    stderr=subprocess.STDOUT).returncode
            print(f'[{n}/{len(ids)}] {iid}: rc={rc} {"pred" if pred.exists() else "NO PRED"}', flush=True)
    preds = []
    for iid in ids:
        pred = out / iid / f'{iid}.pred'
        patch = json.loads(pred.read_text()).get('model_patch', '') if pred.exists() else ''
        preds.append(dict(instance_id=iid, model_name_or_path=a.model, model_patch=patch or '', full_output=''))
    (out / 'predictions.json').write_text(json.dumps(preds, indent=1) + '\n')


def cmd_evaluate(a):
    """OmniCode's evaluator, one repository at a time; the repository's images are removed afterwards."""
    root = Path(a.omnicode).resolve()
    run = Path(a.run).resolve()
    out = run / f'{a.lang}-{a.task}'
    preds = out / 'predictions.json' if not a.gold else None
    ids = sample_ids(a.lang, a.task)[:a.limit or None]
    flag = {('python', 'bugfixing'): '--BugFixing', ('python', 'testgeneration'): '--TestGeneration',
            ('cpp', 'bugfixing'): '--MSWEBugFixing', ('cpp', 'testgeneration'): '--MSWETestGeneration'}[
        (a.lang, a.task)]
    env = podman_env()
    by_repo = collections.defaultdict(list)
    for iid in ids:
        by_repo[repo_of(iid)].append(iid)
    for repo, group in sorted(by_repo.items()):
        cmd = [sys.executable, 'omnicode.py', flag, '--dataset_name', f'data/omnicode_instances_{a.lang}.json',
               '--predictions_path', 'gold' if a.gold else str(preds), '--run_id', f'strata-{a.lang}-{a.task}',
               '--max_workers', '1', '--use_apptainer', 'False', '--instance_ids', *group]
        if a.lang == 'cpp':
            cmd += ['--mswe_phase', 'all']
        if a.lang == 'python' and a.task == 'testgeneration':
            cmd += ['--language', 'python']
        with open(out / 'evaluate.log', 'ab') as log:
            subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        if not a.keep_images:
            subprocess.run(['podman', 'image', 'prune', '-af'], stdout=subprocess.DEVNULL)
        print(f'{repo}: {len(group)} instances evaluated', flush=True)


def cmd_report(a):
    """Collect resolved / not resolved per instance from OmniCode's report files (JSON with resolved_ids)."""
    run = Path(a.run)
    tasks = []
    for d in sorted(run.glob('*-*')):
        if not d.is_dir():
            continue
        lang, task = d.name.split('-', 1)
        resolved = set()
        for f in list(d.rglob('*.json')):
            try:
                rep = json.loads(f.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(rep, dict) and isinstance(rep.get('resolved_ids'), list):
                resolved.update(rep['resolved_ids'])
        for iid in sample_ids(lang, task):
            tasks.append(dict(id=f'{lang}/{task}/{iid}', passed=iid in resolved,
                              metrics=dict(lang=lang, task=task, repo=repo_of(iid))))
    meta = dict(generation=json.loads((run / 'generation.json').read_text()))
    for key in sorted({(t['metrics']['lang'], t['metrics']['task']) for t in tasks}):
        rows = [t for t in tasks if (t['metrics']['lang'], t['metrics']['task']) == key]
        meta[f'{key[0]}_{key[1]}_pass'] = f'{sum(t["passed"] for t in rows)}/{len(rows)}'
    data = C.write_results(run / 'results.json', SUITE, a.profile, tasks, meta)
    print(json.dumps({k: v for k, v in meta.items() if k.endswith('_pass')}, indent=1))
    print(f'{data["passed"]}/{data["total"]} passed -> {run / "results.json"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('sample')
    p.add_argument('--omnicode', required=True)
    for name in ('agent', 'evaluate'):
        p = sub.add_parser(name)
        p.add_argument('--omnicode', required=True)
        p.add_argument('--lang', choices=('python', 'cpp'), required=True)
        p.add_argument('--task', choices=('bugfixing', 'testgeneration'), required=True)
        p.add_argument('--run', required=True)
        p.add_argument('--limit', type=int, help='first N sampled instances only (smoke test)')
        if name == 'agent':
            p.add_argument('--config', help='GLM serve profile to start serve.server with')
            p.add_argument('--base-url', help='use a server that is already running')
            p.add_argument('--port', type=int, default=8095)
            p.add_argument('--model', default='glm')
            p.add_argument('--step-limit', type=int, default=75)
            p.add_argument('--max-tokens', type=int, default=8192)
        else:
            p.add_argument('--gold', action='store_true', help="evaluate the reference patches (harness check)")
            p.add_argument('--keep-images', action='store_true')
    p = sub.add_parser('report')
    p.add_argument('--run', required=True)
    p.add_argument('--profile', required=True)
    a = ap.parse_args()
    globals()[f'cmd_{a.cmd}'](a)


if __name__ == '__main__':
    main()
