"""Prepare a local GLM source candidate with per-file checksums; never publish.

Includes tracked files plus named GLM integration paths. Excludes raw logit dumps,
historical benchmark files with authentication values, and unrelated untracked work.
The same exclusions apply to the accompanying patch. Other curated tracked data
stays included. Existing output is refused. Files are hashed from exactly the
bytes placed in the archive.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

# Preserve the frozen benchmark in the checkout without redistributing its authentication literal.
EXCLUDED_SOURCE = {
    'bench/results/2026-10-03-community-z840-rtx3090ti/strata-numa-ab.sh':
        'historical benchmark script contains a hard-coded authentication value',
}


# Older community environment snapshots include a possible bearer credential.
EXCLUDED_SOURCE.update({name: 'historical environment snapshot contains a possible bearer credential'
                        for name in (
    'bench/results/2026-10-04-community-rtx4090-iq3xxs-200k-code/config.json',
    'bench/results/2026-10-04-community-rtx4090-iq3xxs-200k-code/env.json',
    'bench/results/2026-10-04-community-rtx4090-iq3xxs-200k-code/runs.json',
    'bench/results/2026-10-04-community-rtx4090-iq3xxs-200k-ru/config.json',
    'bench/results/2026-10-04-community-rtx4090-iq3xxs-200k-ru/env.json',
    'bench/results/2026-10-04-community-rtx4090-iq3xxs-200k-ru/runs.json',
    'bench/results/2026-10-08-community-rtx4090-iq3xxs-200k-code/env.json',
    'bench/results/2026-10-08-community-rtx4090-iq3xxs-200k-code/runs-0.1.40-baseline.json',
    'bench/results/2026-10-08-community-rtx4090-iq3xxs-200k-code/runs-0.1.40.2-previous.json',
    'bench/results/2026-10-08-community-rtx4090-iq3xxs-200k-code/runs.json',
    'bench/results/2026-10-08-community-rtx4090-iq3xxs-200k-ru/env.json',
    'bench/results/2026-10-08-community-rtx4090-iq3xxs-200k-ru/runs-0.1.40-baseline.json',
    'bench/results/2026-10-08-community-rtx4090-iq3xxs-200k-ru/runs-0.1.40.2-previous.json',
    'bench/results/2026-10-08-community-rtx4090-iq3xxs-200k-ru/runs.json',
 )})

PATCH_ONLY_EXCLUDED_SOURCE = {
    'configs/experimental/glm53f-q23-reap224-tr16-tuned.json':
        'archive includes the normalized profile; omit old machine-specific paths from patch',
}


def git(*args):
    return subprocess.check_output(['git', *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tracked = set(filter(None, git('ls-files', '-z').decode().split('\0')))
    untracked = set(filter(None, git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')))
    prefixes = ('configs/glm53f-', 'docs/GLM_', 'docs/glm53_flash_',
                'docs/benchmarks/glm_b550_', 'docs/fixtures/glm_cpp_review_20261008/', 'docs/fixtures/glm_long_chat_20261008/',
                'docs/fixtures/glm_tr16_decode35_20261013/', 'docs/fixtures/glm_tr16_arch_20261013/',
                'docs/fixtures/glm_tr16_incremental_20261013/', 'docs/fixtures/glm_release_20261013/',
                'hf-model-card/', 'tools/glm_', 'tools/test_glm_')
    extras = {'include/strata/artifact/expert_block_screen.hpp',
              'include/strata/hip_compat/cuda_profiler_api.h',
              'include/strata/kernels/cpu/expert_block_mask.hpp',
              'src/kernels/cpu/expert_block_screen_test.cpp', 'src/program/glm_batch_serve.inc',
              'include/strata/host/glm_expert_replay.h', 'tests/host/glm_expert_replay_test.c',
              'src/prefill/glm_f16_precision_test.cu', 'tests/host/glm_bounded_accept_test.cpp'}
    selected = tracked | {p for p in untracked if p.startswith(prefixes) or p in extras}
    # Frozen benchmark dumps remain in the checkout; the source candidate records their omission.
    excluded_tracked = sorted(p for p in tracked if p.endswith('.logits.bin') or p in EXCLUDED_SOURCE)
    selected = {p for p in selected if not p.endswith('.logits.bin') and p not in EXCLUDED_SOURCE}
    records = {}
    archive = args.output / 'strata-glm-experimental-source.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name in sorted(selected):
            path = Path(name)
            if path.is_symlink():
                raise ValueError('Inspect symlink before release: ' + name)
            if not path.is_file():
                continue
            data = path.read_bytes()
            records[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            info = tarfile.TarInfo('Strata/' + name)
            info.size = len(data)
            info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
            tar.addfile(info, io.BytesIO(data))
    manifest = dict(base_commit=git('rev-parse', 'HEAD').decode().strip(),
                    kind='experimental working-tree source snapshot', files=records,
                    excluded_tracked=excluded_tracked,
                    exclusion_reasons={p: EXCLUDED_SOURCE.get(p, 'raw benchmark logit dump')
                                       for p in excluded_tracked},
                    patch_only_exclusions=PATCH_ONLY_EXCLUDED_SOURCE,
                    excluded_untracked=sorted(untracked-selected))
    (args.output/'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    patch_paths = ['.', ':(glob,exclude)**/*.logits.bin']
    patch_paths.extend(':(literal,exclude)' + name for name in EXCLUDED_SOURCE)
    patch_paths.extend(':(literal,exclude)' + name for name in PATCH_ONLY_EXCLUDED_SOURCE)
    (args.output/'working-tree.patch').write_bytes(git('diff', '--binary', 'HEAD', '--', *patch_paths))
    (args.output/'status.txt').write_bytes(git('status', '--short'))
    checksums = []
    for p in sorted(args.output.iterdir()):
        checksums.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name)
    (args.output/'SHA256SUMS').write_text('\n'.join(checksums)+'\n')
    print(json.dumps(dict(output=str(args.output), files=len(records), archive_bytes=archive.stat().st_size)))


if __name__ == '__main__':
    main()
