import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from glm_resident_model import ResidentModel


@unittest.skipUnless(sys.platform == 'linux', 'Linux memfd and mlock')
class ResidentModelTests(unittest.TestCase):
    def test_subprocess_reads_sealed_bytes_and_owner_releases_them(self):
        data = bytes(range(256)) * 256
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'fixture.gguf'
            source.write_bytes(data)
            with ResidentModel(source, hashlib.sha256(data).hexdigest(), max_bytes=len(data)) as model:
                path = model.path
                actual = subprocess.check_output([sys.executable, '-c',
                    'import pathlib,sys;sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())', str(path)])
                self.assertEqual(actual, data)
                maps = Path('/proc/self/maps').read_text().splitlines()
                entry = next(line for line in maps if 'strata-resident-model' in line)
                self.assertEqual(entry.split()[1], 'r--s', 'mlock must not create a private writable copy')
                # Reopen for writing so failures prove the seals, rather than
                # merely the owner's read-only descriptor mode.
                writable = os.open(path, os.O_RDWR)
                try:
                    with self.assertRaises(OSError):
                        os.pwrite(writable, b'changed', 0)
                    with self.assertRaises(OSError):
                        os.ftruncate(writable, 1)
                finally:
                    os.close(writable)
            self.assertFalse(path.exists())
            self.assertEqual(source.read_bytes(), data)

    def test_budget_and_hash_fail_before_a_model_can_be_used(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'fixture.gguf'
            source.write_bytes(b'x'*4096)
            with self.assertRaisesRegex(ValueError, 'budget'):
                ResidentModel(source, '0'*64, max_bytes=1)
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                ResidentModel(source, '0'*64, max_bytes=4096)
