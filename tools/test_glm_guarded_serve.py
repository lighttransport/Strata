"""Bounded expert-cache warmup checks; no GPU or real model needed."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from glm_guarded_serve import warm_expert_cache


class ExpertWarmupTest(unittest.TestCase):
    def test_complete_spans_and_capacity_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.gguf'
            path.write_bytes(bytes(range(64)))
            tensors = [SimpleNamespace(name=f'blk.3.ffn_{part}_exps.weight', offset=i*16,
                                       expected_bytes=lambda:16) for i,part in enumerate(('gate','up','down'))]
            model = SimpleNamespace(data_start=8, tensors=tensors, metadata={
                'general.architecture':'glm5-next', 'glm5-next.leading_dense_block_count':3,
                'glm5-next.block_count':5, 'glm5-next.nextn_predict_layers':1})
            with patch('gguf_reader.GGUFFile', return_value=model):
                result=warm_expert_cache({'model':str(path)}, 60*2**30)
                self.assertEqual((result['tensors'],result['bytes']),(3,48))
                with self.assertRaisesRegex(ValueError,'4 GiB'):
                    warm_expert_cache({'model':str(path)}, 4*2**30)
                path.write_bytes(b'bad')
                with self.assertRaisesRegex(ValueError,'truncated'):
                    warm_expert_cache({'model':str(path)}, 60*2**30)
                model.tensors=model.tensors[:2]
                with self.assertRaisesRegex(ValueError,'incomplete'):
                    warm_expert_cache({'model':str(path)}, 60*2**30)


if __name__ == '__main__':
    unittest.main()
