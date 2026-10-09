"""Compare document Q&A and C++ review on guarded loopback GLM servers.

One fresh server per affinity/variant. Requests run sequentially in fixed order;
model pages and buffers stay warm, while decoder state and the expert tier reset
for each request. No serving default changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.error
from glm_long_chat_check import build_fixture

ROOT = Path(__file__).resolve().parents[1]
CODE = r'''#include <cstdint>
#include <string_view>
#include <vector>
#include <mutex>
#include <string>
bool parse_u64(std::string_view s, std::uint64_t& out) noexcept {
    out = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
        out = out * 10 + (c - '0');
    }
    return true;
}
class Registry {
    std::mutex mu;
    std::vector<std::string> names;
public:
    void add(std::string s) { std::lock_guard<std::mutex> lock(mu); names.push_back(std::move(s)); }
    std::string_view get(std::size_t i) {
        std::lock_guard<std::mutex> lock(mu);
        return names[i];
    }
};
'''
REVIEW = '''Review the C++17 code below for correctness. Contract: parse_u64 accepts only nonempty ASCII digits, allows leading zeros, rejects values above UINT64_MAX, and leaves out unchanged on EVERY failure. Registry is used concurrently; readers may retain returned values after get returns, while writers keep calling add. Identify concrete contract violations and memory/thread-safety defects, give a minimal triggering example for each, and describe safe fixes. Distinguish actual bugs from style issues. Do not invent code that is not shown. Give at most six concise numbered findings and a short corrected parse_u64 function in one cpp block. Keep the whole answer within 650 words and do not repeat yourself.\n\n'''


def run(*args, check=True):
    return subprocess.run(args, cwd=ROOT, check=check, capture_output=True, text=True)


def request(endpoint, body):
    req = urllib.request.Request(endpoint+'/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1200) as response:
        return json.load(response)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--decoder', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--affinities', type=float, nargs='+', required=True)
    ap.add_argument('--port', type=int, default=8081)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--variants', type=Path, help='named configuration overrides; affinity remains explicit')
    ap.add_argument('--tasks', nargs='+', choices=['document', 'cpp-review'], default=['document', 'cpp-review'])
    ap.add_argument('--trials', type=int, default=1)
    ap.add_argument('--continue-on-error', action='store_true', help='record failed tuning variants and continue')
    args = ap.parse_args()
    if args.trials < 1: ap.error('trials must be positive')
    if any(not 0 <= a <= .2 for a in args.affinities): ap.error('this quality sweep accepts 0..0.2')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text())
    cfg.update(exe=str(args.decoder.resolve()), context=8192, gpu_budget_mib=15792, decode_cache_mib=8192)
    cfg['env'].update(STRATA_GLM_GPU_RESERVE_MIB='512', STRATA_GLM_TIER_RUNTIME_RESERVE_MIB='256')
    doc, fixture = build_fixture(cfg['model'], 8192, out/'document')
    # Remove the redundant executive summary, leaving space for every answer and actions.
    doc['messages'][0]['content'] = doc['messages'][0]['content'].replace(
        'Please give a useful executive summary of the objective, decisions, finances, progress, risks, and next actions. Then answer all eight questions with evidence IDs:',
        'Omit an executive summary. Answer these eight questions concisely with evidence IDs:').replace(
        'Finish with a practical next-actions list grounded in the document.',
        'Finish with exactly three practical next actions. Distinguish crate completion from checksum agreement; these measure different quantities.')
    doc['max_tokens'] = 768
    code = dict(model=cfg['model_name'], temperature=0, reasoning_effort='low', max_tokens=1280,
                messages=[dict(role='user', content=REVIEW+CODE)])
    from gguf_reader import GGUFFile
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate
    tokenizer = Tokenizer.from_gguf(Path(cfg['model']))
    template = ChatTemplate(source=GGUFFile(Path(cfg['model'])).metadata['tokenizer.chat_template'])
    # Frozen integration notes make this a ~4K-input code review, not a tiny function prompt.
    notes = ''
    for i in range(200):
        candidate = notes + (f'Record {i}: telemetry counter_{i} stores unsigned decimal digits. '
            f'The archived value {i*123456789} is data, not an implementation. Readers may retain registry results while new names are added.\n')
        content = REVIEW + 'Background records (context only):\n' + candidate + '\nCode to review:\n' + CODE
        rendered = template.render([dict(role='user', content=content)], tools=None, reasoning_effort='low', add_generation_prompt=True)
        if len(tokenizer.encode(rendered, parse_special=True)) > 4096:
            break
        notes = candidate
    code['messages'][0]['content'] = REVIEW + 'Background records (context only):\n' + notes + '\nCode to review:\n' + CODE
    for name, body in [('document', doc), ('cpp-review', code)]:
        text = template.render(body['messages'], tools=None, reasoning_effort='low', add_generation_prompt=True)
        count = len(tokenizer.encode(text, parse_special=True))
        if count + body['max_tokens'] + 8 > cfg['context']:
            raise ValueError(name+' exceeds the context budget')
        (out/(name+'.fixture.json')).write_text(json.dumps(dict(prompt_tokens=count, output_cap=body['max_tokens'], prompt_sha256=hashlib.sha256(text.encode()).hexdigest()), indent=2)+'\n')
        (out/(name+'.request.json')).write_text(json.dumps(body, indent=2)+'\n')
    provenance = dict(decoder_sha256=hashlib.sha256(args.decoder.read_bytes()).hexdigest(),
                      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      model=cfg['model'], base_cache_mib=8192, reserve_mib=512, context=8192,
                      order=args.tasks, trials=args.trials, variants=json.loads(args.variants.read_text()) if args.variants else None, affinities=args.affinities,
                      notes='Sequential requests in task/trial order. Variant overrides follow the base configuration; each case config.json is authoritative. Real EOS honored. Model pages/buffers stay warm, but decoder state and expert tier reset per request; no prefix reuse or untimed request.')
    (out/'run.json').write_text(json.dumps(provenance, indent=2)+'\n')
    if args.prepare_only: return
    unit = 'strata-glm-sweet-quality'; endpoint = f'http://127.0.0.1:{args.port}'
    run('systemctl', '--user', 'stop', 'strata-glm-b550.service', check=False)
    summary = []
    import copy
    variants = json.loads(args.variants.read_text()) if args.variants else [dict(name='default')]
    base_cfg = copy.deepcopy(cfg)
    for affinity, variant in [(a, v) for a in args.affinities for v in variants]:
        name = variant['name']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('invalid variant name')
        case = out/(('aff'+format(affinity, '.2f')) if not args.variants else name+'-aff'+format(affinity, '.2f')); case.mkdir()
        cfg = copy.deepcopy(base_cfg)
        cfg.update(variant.get('config', {}))
        cfg['env'].update(variant.get('env', {}))
        for key in variant.get('unset_env', []): cfg['env'].pop(key, None)
        cfg['env']['STRATA_GLM_ROUTE_AFFINITY'] = str(affinity)
        cfg['log'] = str(case/'engine.log')
        config = case/'config.json'; config.write_text(json.dumps(cfg, indent=2)+'\n')
        try:
            run('systemctl', '--user', 'reset-failed', unit, check=False)
            run('systemd-run', '--user', '--unit='+unit, '--property=WorkingDirectory='+str(ROOT),
                '--property=MemoryMax=60G', '--property=MemorySwapMax=0', '--property=OOMPolicy=kill',
                '--property=KillMode=control-group', sys.executable, str(ROOT/'tools/glm_guarded_serve.py'),
                str(config), '--ram-gib', '60', '--reserve-mib', '512', '--host', '127.0.0.1',
                '--port', str(args.port), '--telemetry', str(case/'memory.json'))
            deadline = time.monotonic()+300
            while True:
                try:
                    with urllib.request.urlopen(endpoint+'/health', timeout=3) as r: health = json.load(r)
                    if health.get('loaded') and health.get('max_context') == 8192: break
                except (OSError, ValueError): pass
                if run('systemctl', '--user', 'is-active', unit, check=False).stdout.strip() in ('failed', 'inactive'):
                    raise RuntimeError('quality service failed')
                if time.monotonic() > deadline: raise TimeoutError('readiness')
                time.sleep(2)
            for task, trial in [(task, trial) for task in args.tasks for trial in range(args.trials)]:
                name = task if args.trials == 1 else task+'-trial'+str(trial)
                body = doc if task == 'document' else code
                response = request(endpoint, body)
                expected = json.loads((out/(task+'.fixture.json')).read_text())['prompt_tokens']
                if response['usage']['prompt_tokens'] != expected:
                    raise RuntimeError(name+' prompt count differs from prepared fixture')
                (case/(name+'.response.json')).write_text(json.dumps(response, indent=2)+'\n')
                choice = response['choices'][0]
                (case/(name+'.answer.md')).write_text(choice['message'].get('content') or '')
                row = dict(affinity=affinity, variant=variant['name'], task=task, trial=trial, timings=response.get('timings'),
                           usage=response.get('usage'), finish_reason=choice.get('finish_reason'))
                summary.append(row); print(json.dumps(row), flush=True)
                (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        except Exception as error:
            failure = dict(affinity=affinity, variant=variant['name'], status='failed', error=repr(error))
            if isinstance(error, urllib.error.HTTPError): failure['http_body'] = error.read().decode(errors='replace')
            (case/'failure.json').write_text(json.dumps(failure, indent=2)+'\n')
            summary.append(failure)
            (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            print(json.dumps(failure), flush=True)
            if not args.continue_on_error: raise
        finally:
            run('systemctl', '--user', 'stop', unit, check=False)
            (case/'journal.txt').write_text(run('journalctl', '--user', '-u', unit, '--no-pager', '-n', '30', check=False).stdout)
    (out/'restore.txt').write_text('Serving demo deliberately left stopped.\n')


if __name__ == '__main__':
    main()
