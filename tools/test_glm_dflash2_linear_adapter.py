"""CPU contracts for optional FC/MLP residual calibration."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
try:
    import torch
    from torch.nn import functional as F
    from safetensors.torch import save_file
    from glm_dflash2_adapter import apply_fc_adapter, FcAdapter, file_sha
    from glm_dflash2_calibrate import restore_weights
    from glm_dflash2_linear_adapter import LinearAdapters, SCHEMA, apply_linear_adapters, load_linear_adapters, target_shapes
    from glm_dflash2_training import LinearTrainingDraft
    import test_glm_dflash2_training as fixture
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch not installed')
class LinearAdapterTest(unittest.TestCase):
    def models(self):
        serving, old, _ = fixture.CalibrationTest().models()
        weights = {n: w.clone() for n, w in restore_weights(old).items()}
        adapter = LinearAdapters(target_shapes(old.c, weights), 2, 2)
        training = LinearTrainingDraft(old.c, weights, 'cpu', adapter)
        return serving, training, adapter

    def test_zero_and_nonzero_training_match_serving_and_all_targets_learn(self):
        serving, model, adapter = self.models()
        frozen = {n: w.clone() for n, w in model.w.items()}
        features, embedding = torch.randn(8, 80)*100000, torch.randn(8, 16)
        serving.append(features, 0, 8)
        expected = serving.forward(embedding).clone()
        model.append(features, 0, 8)
        actual = model.forward(embedding)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        head = torch.randn(32, 16)
        F.cross_entropy(F.linear(actual, head), torch.arange(7)).backward()
        for module in adapter.up:
            self.assertIsNotNone(module.weight.grad)
            self.assertTrue(torch.isfinite(module.weight.grad).all())
            self.assertGreater(module.weight.grad.abs().sum().item(), 0)
        torch.optim.SGD(adapter.parameters(), lr=.01).step()
        adapter.zero_grad(set_to_none=True)
        apply_linear_adapters(serving, adapter)
        serving.cache = [];model.cache = []
        serving.append(features, 0, 8);model.append(features, 0, 8)
        actual = model.forward(embedding)
        torch.testing.assert_close(actual, serving.forward(embedding), rtol=0, atol=0)
        F.cross_entropy(F.linear(actual, head), torch.arange(7)).backward()
        for module in adapter.down:
            self.assertTrue(torch.isfinite(module.weight.grad).all())
            self.assertGreater(module.weight.grad.abs().sum().item(), 0)
        for name, weight in model.w.items():
            self.assertFalse(weight.requires_grad)
            self.assertIsNone(weight.grad)
            torch.testing.assert_close(weight, frozen[name], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'already'):
            apply_linear_adapters(serving, adapter)
        with self.assertRaisesRegex(ValueError, 'already'):
            apply_fc_adapter(serving, FcAdapter(16, 80, 2, 2))

    def test_target_manifest_is_checked_before_allocation(self):
        serving, _, adapter = self.models()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);checkpoint = root/'base';checkpoint.write_bytes(b'base checkpoint')
            weights = root/'linear.safetensors';save_file(adapter.state_dict(), str(weights))
            meta = dict(schema=SCHEMA, complete=True, targets=adapter.shapes, rank=2, alpha=2,
                        base_checkpoint_sha256=file_sha(checkpoint), weights_sha256=file_sha(weights))
            (root/'adapter.json').write_text(json.dumps(meta))
            load_linear_adapters(root, serving, checkpoint)
            self.assertFalse(next(serving.linear_adapters.parameters()).requires_grad)
            meta['targets'][0]['inputs'] = 65536
            (root/'adapter.json').write_text(json.dumps(meta))
            with mock.patch('glm_dflash2_linear_adapter.LinearAdapters', side_effect=AssertionError('unexpected allocation')):
                with self.assertRaisesRegex(ValueError, 'targets'):
                    load_linear_adapters(root, self.models()[0], checkpoint)

    def test_corrupt_and_nonfinite_weights_are_rejected(self):
        _, _, adapter = self.models()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);checkpoint = root/'base';checkpoint.write_bytes(b'base checkpoint')
            path = root/'linear.safetensors';save_file(adapter.state_dict(), str(path))
            meta = dict(schema=SCHEMA, complete=True, targets=adapter.shapes, rank=2, alpha=2,
                        base_checkpoint_sha256=file_sha(checkpoint), weights_sha256='0'*64)
            manifest = root/'adapter.json';manifest.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_linear_adapters(root, self.models()[0], checkpoint)
            state = adapter.state_dict();state['up.0.weight'][0,0] = float('nan')
            save_file(state, str(path));meta['weights_sha256'] = file_sha(path);manifest.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'dtype or values'):
                load_linear_adapters(root, self.models()[0], checkpoint)

    def test_duplicate_targets_and_excessive_parameters_fail_before_allocation(self):
        target = dict(name='fc',inputs=65536,outputs=65536)
        with mock.patch('torch.nn.Linear', side_effect=AssertionError('unexpected allocation')):
            with self.assertRaisesRegex(ValueError, 'geometry'):
                LinearAdapters([target, target],2,2)
            with self.assertRaisesRegex(ValueError, 'budget'):
                LinearAdapters([dict(target,name=str(i)) for i in range(3)],64,64)


if __name__ == '__main__':
    unittest.main()
