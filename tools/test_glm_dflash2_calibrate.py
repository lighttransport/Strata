"""Independent causal-index and integrity checks for offline captures."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
try:
    import numpy as np
    import torch
    from glm_dflash2_adapter import file_sha
    from glm_dflash2_calibrate import CalibrationData
except ImportError:
    torch = None


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


@unittest.skipIf(torch is None, 'PyTorch/numpy not installed')
class CaptureIndexTest(unittest.TestCase):
    def dataset(self, root, *, bad_anchor=False, bad_accept=False, offset=0):
        captures = root/'captures'
        request = captures/'request-00000'
        request.mkdir(parents=True)
        hidden, width = 4, 20
        features = np.arange(80, dtype='<f4').reshape(4, width)
        embeddings = np.arange(64, dtype='<f4').reshape(2, 8, hidden)
        (request/'features.f32').write_bytes(features.tobytes())
        (request/'embeddings.f32').write_bytes(embeddings.tobytes())
        frames = []
        for i, (op, start, end) in enumerate([('prepare', 0, 2), ('draft', 2, 2), ('draft', 2, 4)]):
            raw = features[start:end].tobytes()
            frame = dict(op=op, request_id=i+1, start=start+offset, end=end+offset,
                         feature_byte_offset=start*width*4, feature_bytes=len(raw),
                         feature_sha256=hashlib.sha256(raw).hexdigest())
            if op == 'draft':
                anchor = 20 if i == 1 else 22
                frame.update(embedding_byte_offset=(i-1)*8*hidden*4,
                             embedding_sha256=hashlib.sha256(embeddings[i-1].tobytes()).hexdigest(),
                             anchor=anchor+int(bad_anchor and i == 1),
                             selected=list(range(anchor+1, anchor+8)))
            frames.append(frame)
        manifest = dict(complete=True, feature_width=width, embedding_width=hidden, embedding_rows=8,
                        logical_first=offset, logical_end=offset+4, frames=frames, files={})
        for name in ('features.f32', 'embeddings.f32'):
            manifest['files'][name] = dict(bytes=(request/name).stat().st_size, sha256=file_sha(request/name))
        write_json(request/'manifest.json', manifest)
        (root/'prompt.ids').write_text(','.join(map(str, [9]*offset+[10, 11])))
        suffix = list(range(20, 31))
        if bad_accept:
            suffix[1] = 99
        (root/'generated.ids').write_text(','.join(map(str, suffix)))
        write_json(root/'prompts.json', dict(frozen_before_generation=True, fresh_holdout_scored=False,
                   prompts=[dict(index=0, ids_file='prompt.ids', prompt_sha256=file_sha(root/'prompt.ids'))]))
        write_json(captures/'capture.json', dict(complete=True, child_returncode=0, requests=1))
        write_json(root/'result.json', dict(complete=True, decoder_returncode=0, performance_anchor=False,
                   fresh_holdout_scored=False, prompt_manifest_sha256=file_sha(root/'prompts.json'),
                   capture_manifest_sha256=file_sha(captures/'capture.json'),
                   tasks=[dict(index=0, prompt_tokens=offset+2, generated_tokens=len(suffix),
                               generated_file='generated.ids', tokens_sha256=file_sha(root/'generated.ids'),
                               capture_directory=request.name, capture_manifest_sha256=file_sha(request/'manifest.json'))]))
        return features, embeddings

    def test_context_excludes_anchor_and_labels_start_after_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features, embeddings = self.dataset(root)
            data = CalibrationData(root, hidden=4, window=2)
            self.assertEqual(len(data.samples), 2)
            a, b, labels = data.tensors(data.samples[0], 'cpu')
            np.testing.assert_array_equal(a.numpy(), features[:2])
            np.testing.assert_array_equal(b.numpy(), embeddings[0])
            self.assertEqual(labels.tolist(), list(range(21, 28)))
            a, b, labels = data.tensors(data.samples[1], 'cpu')
            np.testing.assert_array_equal(a.numpy(), features[2:4])
            np.testing.assert_array_equal(b.numpy(), embeddings[1])
            self.assertEqual(labels.tolist(), list(range(23, 30)))

    def test_hash_valid_but_shifted_anchor_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.dataset(root, bad_anchor=True)
            with self.assertRaisesRegex(ValueError, 'next teacher token'):
                CalibrationData(root, hidden=4, window=2)

    def test_nonzero_absolute_feature_origin_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features, _ = self.dataset(root, offset=100)
            data = CalibrationData(root, hidden=4, window=2)
            self.assertEqual(data.samples[0]['first'], 100)
            self.assertEqual(data.samples[0]['end'], 102)
            context, _, labels = data.tensors(data.samples[0], 'cpu')
            np.testing.assert_array_equal(context.numpy(), features[:2])
            self.assertEqual(labels.tolist(), list(range(21, 28)))

    def test_hash_valid_but_wrong_accepted_token_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.dataset(root, bad_accept=True)
            with self.assertRaisesRegex(ValueError, 'strict accepted proposals'):
                CalibrationData(root, hidden=4, window=2)

    def test_corrupt_captured_tensor_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.dataset(root)
            path = root/'captures/request-00000/embeddings.f32'
            raw = path.read_bytes()
            path.write_bytes(bytes([raw[0] ^ 1])+raw[1:])
            with self.assertRaisesRegex(ValueError, 'tensor checksum'):
                CalibrationData(root, hidden=4, window=2)


if __name__ == '__main__':
    unittest.main()
