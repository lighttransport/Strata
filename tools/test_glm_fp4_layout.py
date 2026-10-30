"""CPU-only ownership and admission tests; no downloads or GPU."""
import unittest
from glm_fp4_layout import GIB, plan


class LayoutTest(unittest.TestCase):
    def fixture(self):
        text = dict(first_k_dense_replace=1, num_hidden_layers=3, n_routed_experts=4)
        records = {}
        for layer in (1, 2):
            for expert in range(4):
                for projection in ('gate_proj', 'up_proj', 'down_proj'):
                    name = f'model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.weight'
                    records[name] = dict(bytes=GIB, file='weights.safetensors', offset=0, dtype='U8', shape=[1, 1])
        records['model.language_model.embed_tokens.weight'] = dict(bytes=GIB)
        records['model.language_model.layers.3.mlp.experts.0.gate_proj.weight'] = dict(bytes=GIB)
        records['model.visual.weight'] = dict(bytes=GIB)
        return text, records

    def test_complete_disjoint_ownership_and_draft_locality(self):
        text, records = self.fixture()
        result = plan(text, records, 'fixture', 18 * GIB, 20 * GIB)
        self.assertEqual(result['remote_experts_per_layer'], 2)
        self.assertEqual(result['worker_bytes'], 12 * GIB)
        self.assertEqual(result['decoder_bytes'], 14 * GIB)
        self.assertEqual(result['excluded_bytes'], GIB)
        self.assertEqual(set(records), set(result['tensors']))
        self.assertEqual(result['tensors']['model.language_model.layers.3.mlp.experts.0.gate_proj.weight']['owner'], 'decoder')
        self.assertEqual(sum(r['bytes'] for r in records.values()), sum(result[k] for k in ('worker_bytes', 'decoder_bytes', 'excluded_bytes')))

    def test_refuses_insufficient_combined_capacity(self):
        text, records = self.fixture()
        with self.assertRaises(ValueError):
            plan(text, records, 'fixture', 10 * GIB, 10 * GIB)

    def test_payload_cannot_wrap_c11_admission_arguments(self):
        text, records = self.fixture()
        records[next(iter(records))]['bytes'] = 1 << 64
        with self.assertRaises(ValueError): plan(text, records, 'fixture', 30 * GIB, 30 * GIB)

    def test_missing_expert_rejected(self):
        text, records = self.fixture()
        records = {name: row for name, row in records.items() if '.layers.1.mlp.experts.0.' not in name}
        with self.assertRaises(ValueError):
            plan(text, records, 'fixture', 30 * GIB, 30 * GIB)

    def test_gpu_fixed_sources_reduce_ram_charge_without_changing_payload(self):
        text, records = self.fixture()
        name = 'model.language_model.layers.0.self_attn.q_proj.weight'
        records[name] = dict(bytes=GIB, file='weights.safetensors', offset=0, dtype='U8', shape=[32768, 32768])
        result = plan(text, records, 'fixture', 18 * GIB, 20 * GIB, gpu_fixed=True)
        self.assertEqual(result['gpu_fixed_bytes'], GIB)
        self.assertEqual(result['decoder_resident_bytes'], result['decoder_bytes'] - GIB)
        self.assertEqual(result['tensors'][name]['residency'], 'gpu')
        self.assertEqual(result['tensors']['model.language_model.embed_tokens.weight']['residency'], 'ram')

    def test_root_vocabulary_head_is_never_excluded(self):
        text, records = self.fixture()
        records['lm_head.weight'] = dict(bytes=GIB, dtype='F8_E4M3', shape=[32768, 32768])
        records['lm_head.weight_scale'] = dict(bytes=1024, dtype='U8', shape=[1024])
        result = plan(text, records, 'fixture', 18 * GIB, 20 * GIB, gpu_fixed=True)
        for name in ('lm_head.weight', 'lm_head.weight_scale'):
            self.assertEqual(result['tensors'][name]['owner'], 'decoder')
            self.assertEqual(result['tensors'][name]['residency'], 'gpu')
        self.assertEqual(result['gpu_fixed_bytes'], GIB + 1024)

    def test_deterministic_manifest(self):
        text, records = self.fixture()
        a = plan(text, records, 'fixture', 18 * GIB, 20 * GIB)
        b = plan(text, dict(reversed(list(records.items()))), 'fixture', 18 * GIB, 20 * GIB)
        self.assertEqual(a['layout_sha256'], b['layout_sha256'])


if __name__ == '__main__':
    unittest.main()
