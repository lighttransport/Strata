"""Release archives must include integration dependencies and exclude local evidence dumps."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import glm_prepare_release


class PrepareReleaseTest(unittest.TestCase):
    def test_archive_dependencies_checksums_and_exclusions(self):
        blocked = next(iter(glm_prepare_release.EXCLUDED_SOURCE))
        bearer = next(p for p, reason in glm_prepare_release.EXCLUDED_SOURCE.items() if 'bearer' in reason)
        profile = next(iter(glm_prepare_release.PATCH_ONLY_EXCLUDED_SOURCE))
        tracked = ['CMakeLists.txt', 'bench/frozen.logits.bin', blocked, bearer, profile]
        new = ['include/strata/host/glm_expert_replay.h', 'tests/host/glm_expert_replay_test.c',
               'src/prefill/glm_f16_precision_test.cu', 'tests/host/glm_bounded_accept_test.cpp',
               'docs/fixtures/glm_release_20261013/summary.json', 'resume-tr16.md',
               'release/private.txt', 'build-local/model.gguf']
        def git(*args):
            if args[:4] == ('diff', '--binary', 'HEAD', '--'):
                self.assertIn(':(glob,exclude)**/*.logits.bin', args)
                for name in glm_prepare_release.EXCLUDED_SOURCE:
                    self.assertIn(':(literal,exclude)' + name, args)
                self.assertIn(':(literal,exclude)' + profile, args)
                return b'patch\n'
            return {('ls-files', '-z'): '\0'.join(tracked).encode(),
                    ('ls-files', '--others', '--exclude-standard', '-z'): '\0'.join(new).encode(),
                    ('rev-parse', 'HEAD'): b'0123456789\n',
                    ('status', '--short'): b' M CMakeLists.txt\n'}[args]
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                for name in tracked + new:
                    p = Path(name); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(name + '\n')
                with patch.object(glm_prepare_release, 'git', git), patch('sys.argv', ['release', '--output', 'candidate']):
                    glm_prepare_release.main()
                    with self.assertRaises(FileExistsError):
                        glm_prepare_release.main()
                manifest = json.loads(Path('candidate/source-manifest.json').read_text())
                self.assertEqual(set(manifest['files']), {tracked[0], profile, *new[:5]})
                self.assertEqual(manifest['excluded_tracked'], sorted(tracked[1:4]))
                self.assertIn(profile, manifest['patch_only_exclusions'])
                self.assertIn('authentication', manifest['exclusion_reasons'][blocked])
                self.assertIn('bearer', manifest['exclusion_reasons'][bearer])
                with tarfile.open('candidate/strata-glm-experimental-source.tar.gz') as archive:
                    self.assertEqual(set(archive.getnames()), {'Strata/' + n for n in manifest['files']})
                    for name, record in manifest['files'].items():
                        data = archive.extractfile('Strata/' + name).read()
                        self.assertEqual(record, dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
                for line in Path('candidate/SHA256SUMS').read_text().splitlines():
                    expected, name = line.split('  ', 1)
                    self.assertEqual(hashlib.sha256((Path('candidate') / name).read_bytes()).hexdigest(), expected)
            finally:
                os.chdir(previous)

    def test_broken_symlink_is_rejected(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                Path('tracked-link').symlink_to('missing')
                def git(*args):
                    return b'tracked-link\0' if args == ('ls-files', '-z') else b''
                with patch.object(glm_prepare_release, 'git', git), patch('sys.argv', ['release', '--output', 'candidate']):
                    with self.assertRaisesRegex(ValueError, 'symlink'):
                        glm_prepare_release.main()
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
