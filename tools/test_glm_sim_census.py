from types import SimpleNamespace
from unittest.mock import patch
import unittest
from glm_sim_census import deployed_fixed_census


def tensor(name, size=1024, shape=(32,8)):
    return SimpleNamespace(name=name,shape=shape,type_name='F32',expected_bytes=lambda:size)


class FixedCensusTest(unittest.TestCase):
    def test_alias_repack_and_exclusions(self):
        tensors=[tensor(f'blk.{i}.attn_q.weight') for i in range(45)]
        tensors += [tensor('blk.0.hc_attn_fn'),tensor('blk.3.ffn_gate_shexp.weight'),
                    tensor('blk.3.ffn_gate_exps.weight',999999),tensor('blk.45.attn_q.weight',999999),
                    tensor('token_embd.weight',999999),tensor('output.weight',2048)]
        with patch('glm_sim_census.GGUFFile',return_value=SimpleNamespace(tensors=tensors)):
            raw=deployed_fixed_census('unused','none');q8=deployed_fixed_census('unused','all')
        self.assertEqual(raw['0'],2048)
        self.assertEqual(q8['0'],1024+272)
        self.assertEqual(q8['3'],2048)
        self.assertEqual(q8['shared:3'],1024)
        self.assertEqual(q8['lm_head'],2048)
        self.assertNotIn('45',q8)

    def test_incomplete_model_is_rejected(self):
        with patch('glm_sim_census.GGUFFile',return_value=SimpleNamespace(tensors=[])):
            with self.assertRaises(ValueError):deployed_fixed_census('unused')


if __name__=='__main__':unittest.main()
