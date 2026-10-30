"""No-network tests of exact RAM reception and rejection before publication."""
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from glm_fp4_partition import partition
from glm_fp4_ram_receive import receive
import test_glm_fp4_partition as partition_test


def archive(root, corrupt=False):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as output:
        names = ['partition.json', *sorted(p.name for p in root.iterdir() if p.name != 'partition.json')]
        for name in names:
            payload = (root / name).read_bytes()
            if corrupt and name.endswith('.safetensors'):
                payload = b'x' + payload[1:]
            info = tarfile.TarInfo(name); info.size = len(payload)
            output.addfile(info, io.BytesIO(payload))
    stream.seek(0)
    return stream


class RamReceiveTest(unittest.TestCase):
    def test_sealed_payload_and_readback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, _, layout = partition_test.PartitionTest().fixture(root)
            report = partition(source, root / 'partition', layout, 'decoder')
            descriptors = receive(archive(root / 'partition'), root / 'received')
            try:
                self.assertEqual(set(descriptors), set(report['shards']))
                for name in descriptors:
                    self.assertEqual((root / 'received' / name).read_bytes(), (root / 'partition' / name).read_bytes())
                    with self.assertRaises(OSError): os.pwrite(descriptors[name], b'x', 0)
            finally:
                for fd in descriptors.values(): os.close(fd)

    def test_gpu_fixed_files_use_disk_and_native_residency(self):
        import struct
        import subprocess
        from glm_fp4_layout import inspect, plan
        scratch = Path(__file__).resolve().parents[1] / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            root = Path(tmp); source, _, _ = partition_test.PartitionTest().fixture(root)
            file = source / 'weights.safetensors'
            raw = file.read_bytes(); size = struct.unpack('<Q', raw[:8])[0]
            header = json.loads(raw[8:8 + size]); header['lm_head.weight']['shape'] = [2, 2]
            updated = json.dumps(header).encode()
            file.write_bytes(struct.pack('<Q', len(updated)) + updated + raw[8 + size:])
            text, records, fingerprint = inspect(source)
            layout = plan(text, records, fingerprint, 192, 1000, reserve=0, gpu_fixed=True)
            report = partition(source, root / 'partition', layout, 'decoder')
            descriptors = receive(archive(root / 'partition'), root / 'received')
            try:
                self.assertTrue(report['gpu_fixed_shards'])
                for name in report['gpu_fixed_shards']:
                    self.assertFalse((root / 'received' / name).is_symlink())
                    self.assertEqual((root / 'received' / name).read_bytes(), (root / 'partition' / name).read_bytes())
                if os.environ.get('FP4_CATALOG_TEST'):
                    subprocess.run([os.environ['FP4_CATALOG_TEST'], str(root / 'received'), 'ram'], check=True, capture_output=True)
            finally:
                for fd in descriptors.values(): os.close(fd)

    def test_sequential_commands_and_failure_cleanup(self):
        import subprocess
        import sys
        import threading
        import time
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); source, _, layout = partition_test.PartitionTest().fixture(root)
                partition(source, root / 'partition', layout, 'decoder')
                received = root / 'received'
                requests = dict(runs=[dict(argv=[sys.executable, '-c',
                    'import sys; sys.exit(3)' if fail else 'print(7)']),
                    dict(argv=[sys.executable, '-c', 'print(8)'])])
                errors = []
                def write_request():
                    try:
                        deadline = time.monotonic() + 5
                        while not received.exists():
                            if time.monotonic() > deadline: raise TimeoutError('receiver did not start')
                            time.sleep(.01)
                        pending = received / 'run.tmp'
                        pending.write_text(json.dumps(requests)); pending.rename(received / 'run.json')
                    except Exception as error: errors.append(error)
                writer = threading.Thread(target=write_request); writer.start()
                result = subprocess.run([sys.executable, str(Path(__file__).with_name('glm_fp4_ram_receive.py')),
                    '--root', str(received)], input=archive(root / 'partition').read(), capture_output=True, timeout=10)
                writer.join(timeout=6)
                self.assertFalse(errors)
                self.assertEqual(result.returncode, 3 if fail else 0, result.stderr)
                self.assertEqual(result.stdout, b'' if fail else b'7\n8\n')
                self.assertFalse(received.exists())

    def test_corrupt_weight_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, _, layout = partition_test.PartitionTest().fixture(root)
            partition(source, root / 'partition', layout, 'decoder')
            with self.assertRaises(ValueError): receive(archive(root / 'partition', True), root / 'received')
            self.assertFalse((root / 'received').exists())

    def test_traversal_rejected_without_touching_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); sentinel = root / 'sentinel'; sentinel.write_bytes(b'keep')
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode='w') as output:
                info = tarfile.TarInfo('../sentinel'); info.size = 1
                output.addfile(info, io.BytesIO(b'x'))
            stream.seek(0)
            with self.assertRaises(ValueError): receive(stream, root / 'received')
            self.assertEqual(sentinel.read_bytes(), b'keep')
            self.assertFalse((root / 'received').exists())

    def test_existing_destination_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'sentinel').write_text('keep')
            with self.assertRaises(FileExistsError): receive(io.BytesIO(), root)
            self.assertEqual((root / 'sentinel').read_text(), 'keep')


if __name__ == '__main__': unittest.main()
