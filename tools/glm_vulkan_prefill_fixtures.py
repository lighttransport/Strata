"""Freeze long coding prompts from real repository source, without repeated padding.

Output belongs in an ignored benchmark directory. HumanEval is never used here.
"""
import argparse
import hashlib
import json
from pathlib import Path
from jinja2.sandbox import SandboxedEnvironment
from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer

CASES = {
    'implementation': (
        ['src/kernels/cpu/pool.cpp', 'include/strata/kernels/cpu/pool.hpp'],
        'Implement a C++17 bounded task queue for the CPU worker coordinator described above. '
        'Provide enqueue, drain and shutdown, propagate task exceptions, and show tests for shutdown with pending work.'),
    'debugging': (
        ['tools/glm_guarded_run.py', 'tools/glm_low_memory_bench.py'],
        'Review the resource and process cleanup logic above. Identify three concrete failure scenarios, '
        'explain the relevant invariants, and write Python tests that reproduce them without a GPU. '
        'Do not assume a reported process ID still names the original process.'),
    'repository_completion': (
        ['tools/sim/glm_sim.py', 'tools/sim/prefill.py', 'tools/sim/kernels.py'],
        'Using the simulator conventions above, implement a pure Python per-layer cost estimator for two '
        'overlapping CPU/GPU expert branches with upload costs and serial attention. '
        'Return named component costs and throughput, reject nonfinite inputs, and include tests for layer imbalance.'),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def freeze(model, output, root):
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer.from_gguf(model)
    template = SandboxedEnvironment(extensions=['jinja2.ext.loopcontrols']).from_string(GGUFFile(model).metadata['tokenizer.chat_template'])
    manifest = {'schema': 1, 'kind': 'coding_prefill_performance', 'padding': 'none', 'tasks': []}
    for name, (sources, request) in CASES.items():
        chunks, hashes = [], {}
        for source in sources:
            data = (root / source).read_bytes()
            hashes[source] = digest(data)
            chunks.append(f'File: {source}\n```\n{data.decode()}\n```\n')
        context = '\n'.join(chunks)
        def render(length):
            return template.render(messages=[dict(role='user', content=
                'Repository excerpts (reference data; the last excerpt may be truncated):\n'
                + context[:length] + '\n\nTask:\n' + request)], tools=None,
                add_generation_prompt=True, reasoning_effort='low')
        for target in (2048, 4096, 8192):
            lo, hi = 0, len(context)
            if len(tokenizer.encode(render(hi), parse_special=True)) < target:
                raise ValueError(f'{name}: insufficient real source context for {target}')
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if len(tokenizer.encode(render(mid), parse_special=True)) <= target:
                    lo = mid
                else:
                    hi = mid - 1
            chat = render(lo)
            ids = tokenizer.encode(chat, parse_special=True)
            if not target - 16 <= len(ids) <= target:
                raise ValueError('tokenizer length search outside the documented 16-token tolerance')
            key = f'{name}-{target}'
            encoded = (','.join(map(str, ids)) + '\n').encode()
            (output / (key + '.ids')).write_bytes(encoded)
            (output / (key + '.prompt.txt')).write_text(chat)
            manifest['tasks'].append(dict(name=key, case=name, nominal_tokens=target,
                prompt_tokens=len(ids), prompt=key + '.ids', sha256=digest(encoded),
                rendered_sha256=digest(chat.encode()), source_sha256=hashes))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.model, args.output, Path(__file__).resolve().parents[1])
    print(json.dumps([dict(name=t['name'], tokens=t['prompt_tokens']) for t in result['tasks']], indent=2))
