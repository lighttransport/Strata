"""CPU checks for calibration gradients, serving rounding and adapter identity."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
try:
    import torch
    from torch.nn import functional as F
    import test_glm_dflash2_worker as fixture
    from glm_dflash2_worker import Draft
    from glm_dflash2_adapter import FcAdapter, SCHEMA, apply_fc_adapter, file_sha, load_fc_adapter
    from glm_dflash2_training import TrainingDraft, Linear, HalfBmm
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch not installed')
class CalibrationTest(unittest.TestCase):
    def models(self):
        original = fixture.DFlash2Test().tiny()
        weights = {n: w.half().clone() for n, w in original.w.items()}
        for i, qkv in enumerate(original.qkv):
            for name, weight in zip(('q_proj', 'k_proj', 'v_proj'), qkv.split((16, 8, 8))):
                weights[f'layers.{i}.self_attn.{name}.weight'] = weight.half().clone()
        class Serving(Draft):
            def project(self, x, weight, packed=False):
                scale = (x.abs().amax(-1, keepdim=True) / 16384).clamp_min(1)
                return F.linear((x / scale).half().float(), weight.float()) * scale
        control = Serving(original.c, {n: w.clone() for n, w in weights.items()}, 'cpu', torch.float32)
        adapter = FcAdapter(16, 80, 2, 2)
        model = TrainingDraft(original.c, weights, 'cpu', adapter)
        return control, model, adapter

    def test_zero_adapter_matches_serving_and_context_gradients_reach_bridge(self):
        control, model, adapter = self.models()
        features, embedding = torch.randn(8, 80)*100000, torch.randn(8, 16)
        control.append(features, 0, 8)
        expected = control.forward(embedding).clone()
        model.append(features, 0, 8)
        actual = model.forward(embedding)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        head = torch.randn(32, 16)
        loss = F.cross_entropy(F.linear(actual, head), torch.arange(7))
        loss.backward()
        self.assertIsNotNone(adapter.up.weight.grad)
        self.assertTrue(torch.isfinite(adapter.up.weight.grad).all())
        self.assertGreater(adapter.up.weight.grad.abs().sum().item(), 0)
        # Anchor/mask embeddings do not require gradients. Only committed
        # context reaches the FC adapter, so this also detects detached KV.
        self.assertFalse(features.requires_grad)
        for weight in list(model.w.values()) + model.qkv:
            self.assertFalse(weight.requires_grad)
            self.assertIsNone(weight.grad)

    def test_linear_retains_wide_outputs_and_fp32_gradients(self):
        x = torch.full((8, 64), 200., requires_grad=True)
        w = torch.full((16, 64), 20., dtype=torch.float16)
        y = Linear.apply(x, w, w.float(), False)
        self.assertEqual(y.dtype, torch.float32)
        torch.testing.assert_close(y, torch.full_like(y, 256000.), rtol=0, atol=0)
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 320.), rtol=0, atol=0)

    def test_half_attention_forward_keeps_small_fp32_gradients(self):
        a = torch.ones((2, 3, 4), requires_grad=True)
        b = torch.ones((2, 4, 5), requires_grad=True)
        result = HalfBmm.apply(a, b)
        torch.testing.assert_close(result, torch.bmm(a.half(), b.half()).float(), rtol=0, atol=0)
        (result * 1e-10).sum().backward()
        torch.testing.assert_close(a.grad, torch.full_like(a, 5e-10))
        torch.testing.assert_close(b.grad, torch.full_like(b, 3e-10))

    def test_adapter_attaches_once_and_only_changes_fc(self):
        control, _, adapter = self.models()
        x = torch.randn(8, 80)
        expected = control.linear(x, 'fc')
        other_input = torch.randn(8, 16)
        other_name = 'layers.0.attention_conv.kernel_projection'
        other_expected = control.linear(other_input, other_name)
        apply_fc_adapter(control, adapter)
        torch.testing.assert_close(control.linear(x, 'fc'), expected, rtol=0, atol=0)
        with torch.no_grad():
            adapter.up.weight.fill_(.01)
        torch.testing.assert_close(control.linear(x, 'fc'), expected + adapter(x))
        torch.testing.assert_close(control.linear(other_input, other_name), other_expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'already attached'):
            apply_fc_adapter(control, adapter)

    def test_checksums_and_nonfinite_adapter_weights_are_rejected(self):
        from safetensors.torch import save_file
        control, _, adapter = self.models()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root/'base'
            checkpoint.write_bytes(b'checkpoint identity')
            weights = root/'fc.safetensors'
            save_file(adapter.state_dict(), str(weights))
            metadata = dict(schema=SCHEMA, complete=True, hidden=16, features=80, rank=2, alpha=2,
                            base_checkpoint_sha256=file_sha(checkpoint), weights_sha256=file_sha(weights))
            manifest = root/'adapter.json'
            manifest.write_text(json.dumps(metadata))
            load_fc_adapter(root, control, checkpoint)
            self.assertFalse(control.fc_adapter.up.weight.requires_grad)
            metadata.update(hidden=32, features=160)
            manifest.write_text(json.dumps(metadata))
            with mock.patch('glm_dflash2_adapter.FcAdapter', side_effect=AssertionError('unexpected allocation')):
                with self.assertRaisesRegex(ValueError, 'geometry'):
                    load_fc_adapter(root, self.models()[0], checkpoint)
            metadata.update(hidden=16, features=80)
            metadata['weights_sha256'] = '0'*64
            manifest.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_fc_adapter(root, self.models()[0], checkpoint)
            state = adapter.state_dict()
            state['up.weight'][0, 0] = float('nan')
            save_file(state, str(weights))
            metadata['weights_sha256'] = file_sha(weights)
            manifest.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'shapes, dtype or values'):
                load_fc_adapter(root, self.models()[0], checkpoint)


if __name__ == '__main__':
    unittest.main()
