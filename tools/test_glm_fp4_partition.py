"""Compact partition readback, failure atomicity and source isolation."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from glm_fp4_layout import inspect, plan
from glm_fp4_partition import partition, F_GET_SEALS, F_SEAL_WRITE


class PartitionTest(unittest.TestCase):
    def fixture(self, root):
        source = root / 'source'; source.mkdir()
        config = dict(text_config=dict(first_k_dense_replace=3, num_hidden_layers=4, n_routed_experts=2))
        (source / 'config.json').write_text(json.dumps(config))
        tensors = {'model.language_model.embed_tokens.weight': b'fixed', 'model.visual.weight': b'vision', 'lm_head.weight': b'head'}
        for layer in (3, 4):
            for expert in (0, 1):
                for projection in ('gate_proj', 'up_proj', 'down_proj'):
                    tensors[f'model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.weight'] = bytes([layer, expert]) * 32
        header, body, index = {}, bytearray(), {}
        for name, payload in tensors.items():
            header[name] = dict(dtype='U8', shape=[len(payload)], data_offsets=[len(body), len(body) + len(payload)])
            body.extend(payload); index[name] = 'weights.safetensors'
        raw = json.dumps(header).encode()
        (source / 'weights.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw + body)
        (source / 'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=index)))
        text, records, fingerprint = inspect(source)
        layout = plan(text, records, fingerprint, 192, 1000, reserve=0)
        return source, tensors, layout

    def test_memfd_fallback_is_sealable(self):
        import fcntl
        import os
        from unittest.mock import patch
        from glm_fp4_partition import create_memfd, F_ADD_SEALS
        # Exercise standalone Python builds that omit the os wrapper.
        import glm_fp4_partition as module
        class WithoutMemfd:
            def __getattr__(self, name):
                if name == 'memfd_create': raise AttributeError(name)
                return getattr(os, name)
        with patch.object(module, 'os', WithoutMemfd()):
            fd = create_memfd('fallback-test')
        try:
            os.write(fd, b'exact')
            fcntl.fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE)
            with self.assertRaises(OSError): os.pwrite(fd, b'x', 0)
            self.assertEqual(os.pread(fd, 5, 0), b'exact')
        finally:
            os.close(fd)

    def test_payloads_are_exact_and_sources_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, tensors, layout = self.fixture(root)
            before = (source / 'weights.safetensors').read_bytes()
            for owner in ('worker', 'decoder'):
                output = root / owner
                report = partition(source, output, layout, owner, chunk_bytes=7)
                _, rows, _ = inspect(output)
                expected = {name for name in tensors if layout['tensors'][name]['owner'] == owner}
                self.assertEqual(set(rows), expected)
                for name, row in rows.items():
                    with (output / row['file']).open('rb') as stream:
                        stream.seek(row['offset']); self.assertEqual(stream.read(row['bytes']), tensors[name])
                for filename, digest in report['shards'].items():
                    self.assertEqual(hashlib.sha256((output / filename).read_bytes()).hexdigest(), digest)
            self.assertEqual((source / 'weights.safetensors').read_bytes(), before)

    def test_sealed_ram_payload_and_write_rejection(self):
        import fcntl
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, tensors, layout = self.fixture(root)
            descriptors = {}
            try:
                report = partition(source, root / 'ram', layout, 'worker', ram_fds=descriptors)
                self.assertTrue(descriptors)
                for filename, descriptor in descriptors.items():
                    self.assertTrue((root / 'ram' / filename).is_symlink())
                    self.assertTrue(fcntl.fcntl(descriptor, F_GET_SEALS) & F_SEAL_WRITE)
                    with self.assertRaises(OSError): os.pwrite(descriptor, b'x', 0)
                    self.assertEqual(hashlib.sha256((root / 'ram' / filename).read_bytes()).hexdigest(), report['shards'][filename])
            finally:
                for descriptor in descriptors.values(): os.close(descriptor)

    def test_native_catalog_and_memfd_reader(self):
        import os
        import subprocess
        checker = os.environ.get('FP4_CATALOG_TEST')
        if not checker:
            self.skipTest('set FP4_CATALOG_TEST to the native catalog checker')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, _, layout = self.fixture(root)
            for ram in (False, True):
                descriptors = {} if ram else None
                output = root / ('ram' if ram else 'disk')
                try:
                    partition(source, output, layout, 'decoder', ram_fds=descriptors)
                    subprocess.run([checker, str(output), *(['ram'] if ram else [])], check=True, capture_output=True)
                finally:
                    for descriptor in (descriptors or {}).values(): os.close(descriptor)

    def test_refuses_source_destination_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, _, layout = self.fixture(root)
            for output in (source, source / 'nested', root):
                with self.assertRaises(ValueError): partition(source, output, layout, 'worker')

    def test_rejects_tampered_layout_without_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, _, layout = self.fixture(root)
            layout['remote_experts_per_layer'] += 1
            with self.assertRaises(ValueError): partition(source, root / 'output', layout, 'worker')
            self.assertFalse((root / 'output').exists())

    def test_rejects_metadata_mismatch_without_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, _, layout = self.fixture(root)
            next(iter(layout['tensors'].values()))['bytes'] += 1
            with self.assertRaises(ValueError): partition(source, root / 'output', layout, 'worker')
            self.assertFalse((root / 'output').exists())


if __name__ == '__main__': unittest.main()
