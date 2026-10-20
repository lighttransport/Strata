"""Sealed cache lifetime and child-process access, without a model or GPU."""
import hashlib
import errno
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from glm_ram_cache import RamCache, fcntl


@unittest.skipUnless(sys.platform == 'linux' and hasattr(os, 'memfd_create'), 'Linux memfd required')
class CacheTest(unittest.TestCase):
    def fixture(self, root):
        source = root/'model'
        (source/'tensors').mkdir(parents=True)
        (source/'config.json').write_text(json.dumps({'quantization_config':{'quant_method':'modelopt'}}))
        (source/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'tensors/a.safetensors'}}))
        payload = bytes(range(256))*97
        (source/'tensors/a.safetensors').write_bytes(payload)
        return source, payload

    def test_hash_seals_readonly_child_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source, payload = self.fixture(root)
            destination = root/'cache'
            with RamCache(source, destination, workers=2, chunk_bytes=4096, reserve_bytes=0) as cache:
                entry, = cache.entries
                self.assertEqual(entry['sha256'], hashlib.sha256(payload).hexdigest())
                seals = fcntl.fcntl(entry['descriptor'], fcntl.F_GET_SEALS)
                self.assertTrue(seals & fcntl.F_SEAL_WRITE)
                with self.assertRaises(OSError):
                    os.pwrite(entry['descriptor'], b'X', 0)
                # The child opens its own descriptor, rather than inheriting one.
                output = subprocess.check_output([sys.executable, '-c',
                    'import pathlib,sys;sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())',
                    str(destination/'tensors/a.safetensors')], close_fds=True)
                self.assertEqual(output, payload)
                (source/'tensors/a.safetensors').write_bytes(b'changed')
                self.assertEqual((destination/'tensors/a.safetensors').read_bytes(), payload)
            self.assertFalse(destination.exists())
            with self.assertRaises(OSError):
                os.fstat(entry['descriptor'])

    def test_existing_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source, unused = self.fixture(root)
            destination = root/'cache'
            destination.mkdir()
            (destination/'keep').write_text('owned by caller')
            with self.assertRaises(FileExistsError):
                RamCache(source, destination).open()
            self.assertEqual((destination/'keep').read_text(), 'owned by caller')

    @unittest.skipUnless(hasattr(os, 'O_DIRECT'), 'O_DIRECT required')
    def test_direct_read_aligned_blocks_and_partial_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source, payload = self.fixture(root)
            try:
                with RamCache(source, root/'cache', workers=1, chunk_bytes=4096,
                              reserve_bytes=0, direct_read=True) as cache:
                    self.assertEqual(cache.entries[0]['sha256'], hashlib.sha256(payload).hexdigest())
                    self.assertEqual((root/'cache/tensors/a.safetensors').read_bytes(), payload)
            except OSError as error:
                if error.errno in (errno.EINVAL, errno.EOPNOTSUPP):
                    self.skipTest('filesystem does not support O_DIRECT')
                raise

    def test_escape_and_budget_rejected_before_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source, unused = self.fixture(root)
            with mock.patch.dict(os.environ, SLURM_MEM_PER_NODE='1'):
                with self.assertRaises(MemoryError):
                    RamCache(source, root/'cache').open()
            self.assertFalse((root/'cache').exists())
            (source/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'../escape.safetensors'}}))
            with self.assertRaises(ValueError):
                RamCache(source, root/'cache').open()
            self.assertFalse((root/'cache').exists())


if __name__ == '__main__':
    unittest.main()
