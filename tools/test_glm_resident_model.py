import hashlib
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from glm_resident_model import ResidentModel
from glm_socket_placement import copy_plan


def socket_plan(size, digest):
    return dict(schema='strata.socket-row-plan.v1', model_bytes=size, model_sha256=digest,
                row_halves=[dict(tensor='tensor.a', offset=4096, bytes_per_half=4096, slices=2),
                            dict(tensor='tensor.b', offset=32768, bytes_per_half=4096, slices=1)])


class SocketPlanTests(unittest.TestCase):
    def test_every_byte_is_assigned_once_and_opposite_halves_have_opposite_owners(self):
        tasks, _ = copy_plan(socket_plan(65536, 'a'*64), 65536, 'a'*64)
        owners = [-1]*65536
        for side, ranges in enumerate(tasks):
            for start, length in ranges:
                for offset in range(start, start+length):
                    self.assertEqual(owners[offset], -1)
                    owners[offset] = side
        self.assertNotIn(-1, owners)
        for start in (4096, 12288, 32768):
            self.assertEqual(owners[start:start+4096], [0]*4096)
            self.assertEqual(owners[start+4096:start+8192], [1]*4096)

    def test_wrong_identity_overlap_and_invalid_ranges_are_rejected(self):
        plan = socket_plan(65536, 'a'*64)
        for field, value in [('model_bytes', 1), ('model_sha256', 'b'*64), ('schema', 'unknown')]:
            bad = copy.deepcopy(plan)
            bad[field] = value
            with self.assertRaises(ValueError):
                copy_plan(bad, 65536, 'a'*64)
        for field, value in [('offset', -1), ('offset', 65536), ('bytes_per_half', 0),
                             ('bytes_per_half', True), ('slices', 0), ('slices', 0.5)]:
            bad = copy.deepcopy(plan)
            bad['row_halves'][0][field] = value
            with self.assertRaises(ValueError):
                copy_plan(bad, 65536, 'a'*64)
        bad = copy.deepcopy(plan)
        bad['row_halves'][1]['offset'] = 8192
        with self.assertRaisesRegex(ValueError, 'overlap'):
            copy_plan(bad, 65536, 'a'*64)


@unittest.skipUnless(sys.platform == 'linux', 'Linux memfd and mlock')
class ResidentModelTests(unittest.TestCase):
    def test_socket_copy_preserves_sealed_bytes_with_partial_reads_and_writes(self):
        # Fake topology tests copying and lifecycle. Actual NUMA placement needs
        # the separate two-socket hardware check; it is not inferred from this mock.
        data = bytes((i * 17 + i // 4096) % 256 for i in range(65536))
        digest = hashlib.sha256(data).hexdigest()
        affinity = os.sched_getaffinity(0)
        groups = [dict(cpus=[min(affinity)], nodes={0}), dict(cpus=[max(affinity)], nodes={1})]
        pread, pwrite = os.pread, os.pwrite
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'fixture.gguf'
            source.write_bytes(data)
            with (mock.patch('glm_socket_placement.socket_groups', return_value=groups),
                  mock.patch('glm_socket_placement.set_interleave') as policy,
                  mock.patch('glm_socket_placement.verify_socket_rows', return_value=[dict(local=True)]),
                  mock.patch('glm_socket_placement.os.pread', side_effect=lambda fd,n,at: pread(fd,min(n,307),at)),
                  mock.patch('glm_socket_placement.os.pwrite', side_effect=lambda fd,v,at: pwrite(fd,v[:113],at))):
                with ResidentModel(source, digest, max_bytes=len(data),
                                   socket_row_plan=socket_plan(len(data), digest)) as model:
                    path = model.path
                    self.assertEqual(path.read_bytes(), data)
                    self.assertEqual(model.identity['seals'], 15)
                    self.assertEqual(model.identity['socket_nodes'], [[0], [1]])
                self.assertCountEqual([frozenset(c.args[0]) for c in policy.call_args_list],
                                      [frozenset({0}), frozenset({1})])
            self.assertFalse(path.exists())
            self.assertEqual(source.read_bytes(), data)
        self.assertEqual(os.sched_getaffinity(0), affinity)

    def test_socket_policy_failure_joins_workers_and_preserves_source(self):
        import threading
        data = bytes(range(256))*256
        digest = hashlib.sha256(data).hexdigest()
        affinity = os.sched_getaffinity(0)
        groups = [dict(cpus=[min(affinity)], nodes={0}), dict(cpus=[max(affinity)], nodes={1})]
        threads = set(threading.enumerate())
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'fixture.gguf'
            source.write_bytes(data)
            with (mock.patch('glm_socket_placement.socket_groups', return_value=groups),
                  mock.patch('glm_socket_placement.set_interleave', side_effect=OSError('denied'))):
                with self.assertRaisesRegex(OSError, 'denied'):
                    ResidentModel(source, digest, max_bytes=len(data),
                                  socket_row_plan=socket_plan(len(data), digest))
            self.assertEqual(source.read_bytes(), data)
        self.assertEqual(set(threading.enumerate()), threads)
        self.assertEqual(os.sched_getaffinity(0), affinity)

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
