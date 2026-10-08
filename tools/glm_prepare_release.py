"""Prepare a local GLM source candidate with per-file checksums; never publish.

Includes tracked source plus named GLM integration paths. Excludes unrelated
untracked work, model weights, build outputs and binary logits. Existing output
is refused. Source files are hashed from exactly the bytes placed in the archive.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


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
                'hf-model-card/', 'tools/glm_', 'tools/test_glm_')
    extras = {'include/strata/artifact/expert_block_screen.hpp',
              'include/strata/hip_compat/cuda_profiler_api.h',
              'include/strata/kernels/cpu/expert_block_mask.hpp',
              'src/kernels/cpu/expert_block_screen_test.cpp', 'src/program/glm_batch_serve.inc'}
    selected = tracked | {p for p in untracked if p.startswith(prefixes) or p in extras}
    records = {}
    archive = args.output / 'strata-glm-experimental-source.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name in sorted(selected):
            path = Path(name)
            if not path.is_file():
                continue
            if path.is_symlink():
                raise ValueError('Inspect symlink before release: ' + name)
            data = path.read_bytes()
            records[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            info = tarfile.TarInfo('Strata/' + name)
            info.size = len(data)
            info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
            tar.addfile(info, io.BytesIO(data))
    manifest = dict(base_commit=git('rev-parse', 'HEAD').decode().strip(),
                    kind='experimental working-tree source snapshot', files=records,
                    excluded_untracked=sorted(untracked-selected))
    (args.output/'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (args.output/'working-tree.patch').write_bytes(git('diff', '--binary', 'HEAD'))
    (args.output/'status.txt').write_bytes(git('status', '--short'))
    checksums = []
    for p in sorted(args.output.iterdir()):
        checksums.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name)
    (args.output/'SHA256SUMS').write_text('\n'.join(checksums)+'\n')
    print(json.dumps(dict(output=str(args.output), files=len(records), archive_bytes=archive.stat().st_size)))


if __name__ == '__main__':
    main()
