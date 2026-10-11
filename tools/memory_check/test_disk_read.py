import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import disk_read


class DiskReadTests(unittest.TestCase):
    def test_non_regular_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'regular file'):
                disk_read.benchmark(Path(directory))

    def test_read_only_checksum_and_bounded_short_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input'
            data = bytes(range(256))*4096
            path.write_bytes(data+b'tail')
            original = os.open
            # Portable unit test for accounting/checksums; live direct I/O is tested separately.
            def open_file(name, flags):
                self.assertTrue(flags & os.O_DIRECT)
                self.assertFalse(flags & (os.O_CREAT | os.O_TRUNC | os.O_WRONLY | os.O_RDWR))
                return original(name, flags & ~os.O_DIRECT)
            with patch.object(disk_read.os, 'open', side_effect=open_file):
                result = disk_read.benchmark(path, mib=4, repeats=2)
            self.assertTrue(result['passed'])
            self.assertEqual(result['bytes_per_repeat'], len(data))
            self.assertEqual(result['samples'][0]['sha256'], hashlib.sha256(data).hexdigest())
            self.assertEqual(path.read_bytes(), data+b'tail')

    def test_short_read_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input'; path.write_bytes(b'x'*1048576)
            with patch.object(disk_read.os, 'readv', return_value=4096):
                with self.assertRaisesRegex(RuntimeError, 'Short direct read'):
                    disk_read.benchmark(path, mib=1, repeats=2)

    def test_changed_read_data_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input'; path.write_bytes(b'x'*1048576)
            reads = 0
            def read(fd, buffers):
                nonlocal reads
                reads += 1
                buffers[0][:] = bytes([reads])*len(buffers[0])
                return len(buffers[0])
            with patch.object(disk_read.os, 'readv', side_effect=read):
                with self.assertRaisesRegex(RuntimeError, 'different data'):
                    disk_read.benchmark(path, mib=1, repeats=2)


if __name__ == '__main__':
    unittest.main()
