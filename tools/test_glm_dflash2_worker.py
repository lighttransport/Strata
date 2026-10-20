"""CPU checks for DFlash2 state, convolution and selector semantics."""
import io
import json
from pathlib import Path
import unittest
try:
    import torch
    from glm_dflash2_worker import Draft, convolve, read_exact, rms, select_path, tensor_frame, validate_config
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch not installed')
class DFlash2Test(unittest.TestCase):
    def test_incremental_warmup_preserves_serving_state_and_visits_all_widths(self):
        from glm_dflash2_worker import warm_incremental
        model = self.tiny()
        features, embedding = torch.randn(8, 80), torch.randn(8, 16)
        model.append(features, 0, 8)
        expected = model.forward(embedding).clone()
        cache, hidden = model.cache, model.last_hidden
        cached_values = [(k.clone(), v.clone()) for k, v in cache]
        weights = {name: value.clone() for name, value in model.w.items()}
        original_append, widths = model.append, []
        def append(features, start, end):
            widths.append(end-start)
            return original_append(features, start, end)
        model.append = append
        warm_incremental(model)
        self.assertEqual(widths, list(range(1, 8)))
        self.assertIs(model.cache, cache)
        self.assertIs(model.last_hidden, hidden)
        self.assertEqual((model.start, model.end), (0, 8))
        for (k, v), (a, b) in zip(model.cache, cached_values):
            torch.testing.assert_close(k, a, rtol=0, atol=0)
            torch.testing.assert_close(v, b, rtol=0, atol=0)
        torch.testing.assert_close(model.forward(embedding), expected, rtol=0, atol=0)
        for name, value in model.w.items():
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
        model.cache = []
        with self.assertRaisesRegex(ValueError, 'prepared context'):
            warm_incremental(model)

    def test_protocol_warmup_preserves_hidden_and_parameters(self):
        from glm_dflash2_worker import warm_protocol
        model = self.tiny()
        model.w['candidate_selector.hidden_projection.weight'] = torch.randn(8, 16)*.1
        model.w['candidate_selector.predecessor_codebook'] = torch.randn(64, 8)*.1
        model.w['candidate_selector.successor_codebook'] = torch.randn(64, 8)*.1
        features, embedding = torch.randn(8, 80), torch.randn(8, 16)
        model.append(features, 0, 8)
        expected = model.forward(embedding).clone()
        weights = {name: weight.clone() for name, weight in model.w.items()}
        warm_protocol(model)
        torch.testing.assert_close(model.last_hidden, expected, rtol=0, atol=0)
        torch.testing.assert_close(model.forward(embedding), expected, rtol=0, atol=0)
        for name, weight in model.w.items():
            torch.testing.assert_close(weight, weights[name], rtol=0, atol=0)

    def tiny(self):
        torch.manual_seed(4)
        c = dict(hidden_size=16, head_dim=4, num_attention_heads=4, num_key_value_heads=2,
                 num_hidden_layers=2, sliding_window=8, rms_norm_eps=1e-5,
                 dflash_config=dict(conv_group_size=4))
        w = {}
        def weight(name, *shape):w[name] = torch.randn(*shape) * .08
        weight('fc.weight',16,80)
        for n in ['hidden_norm','norm']:w[n+'.weight'] = torch.ones(16)
        for i in range(2):
            p = f'layers.{i}'
            for n in ['input_layernorm','post_attention_layernorm']:w[p+'.'+n+'.weight'] = torch.ones(16)
            for n in ['q_norm','k_norm']:w[p+'.self_attn.'+n+'.weight'] = torch.ones(4)
            for n, rows in [('q_proj',16),('k_proj',8),('v_proj',8),('o_proj',16)]:weight(p+'.self_attn.'+n+'.weight',rows,16)
            for n in ['attention_conv','mlp_conv']:
                weight(p+'.'+n+'.kernel_projection.weight',16,16)
                w[p+'.'+n+'.base_kernel'] = torch.zeros(2,2,16)
                w[p+'.'+n+'.base_kernel'][:,0] = 1
            for n, rows, cols in [('gate_proj',32,16),('up_proj',32,16),('down_proj',16,32)]:weight(p+'.mlp.'+n+'.weight',rows,cols)
        return Draft(c,w,'cpu',torch.float32)

    def test_dynamic_conv_channel_groups_and_block_boundary(self):
        x = torch.arange(48,dtype=torch.float32).reshape(4,12)
        delta = torch.arange(24,dtype=torch.float32).reshape(4,2,3)/10
        base = torch.arange(24,dtype=torch.float32).reshape(2,12)/20
        actual = convolve(x,delta,base,4)
        expected = torch.zeros_like(x)
        for row in range(4):
            for col in range(12):
                for tap in range(2):
                    if row >= tap:expected[row,col] += (base[tap,col]+delta[row,tap,col//4])*x[row-tap,col]
        torch.testing.assert_close(actual,expected)
        self.assertEqual(convolve(x[:1],delta[:1],base,4).shape,(1,12))

    def test_rms_float_reduction_avoids_half_overflow(self):
        x = torch.full((8,16),40000.,dtype=torch.float16)
        self.assertTrue(torch.isfinite(rms(x,torch.ones(16,dtype=torch.float16),1e-5)).all())

    def test_incremental_committed_kv_equals_recomputed_window(self):
        m = self.tiny();other = self.tiny()
        features = torch.randn(12,80);embedding = torch.randn(8,16)
        m.append(features[:8],0,8);m.forward(embedding)
        before = [(k.clone(),v.clone()) for k,v in m.cache]
        # Rejected proposals never alter the context KV cache.
        m.forward(embedding*2)
        for (a,b),(c,d) in zip(before,m.cache):
            torch.testing.assert_close(a,c);torch.testing.assert_close(b,d)
        m.append(features[8:],8,12)
        other.append(features[4:],4,12)
        torch.testing.assert_close(m.forward(embedding),other.forward(embedding))
        self.assertEqual(m.start,4)
        self.assertEqual(m.end,12)

    def test_request_reset_and_empty_delta(self):
        m = self.tiny();x = torch.randn(8,80);emb = torch.randn(8,16)
        m.append(x,0,8);expected = m.forward(emb).clone()
        m.append(x[:0],8,8);torch.testing.assert_close(m.forward(emb),expected)
        m.append(x,0,8);torch.testing.assert_close(m.forward(emb),expected)
        with self.assertRaisesRegex(ValueError,'missing committed'):m.append(x[:2],1,3)

    def test_selector_uses_selected_predecessor(self):
        hidden = torch.tensor([[1.,0.],[1.,0.]])
        candidates = torch.tensor([[1,2],[3,4]])
        unary = torch.tensor([[0.,.1],[.1,0.]])
        pred = torch.tensor([[0.,0.],[5.,0.],[-5.,0.],[0.,0.],[0.,0.]])
        succ = torch.tensor([[0.,0.],[1.,0.],[0.,0.],[-1.,0.],[1.,0.]])
        result = select_path(hidden,candidates,unary,0,torch.eye(2),pred,succ)
        self.assertEqual(result.tolist(),[2,3])

    def test_unary_ablation_bypasses_selector_and_keeps_row_order(self):
        m=self.tiny();m.selector_scale=0
        candidates=torch.tensor([[3,7,1],[2,4,6]])
        unary=torch.tensor([[1.,3.,2.],[4.,2.,1.]])
        self.assertEqual(m.select(candidates,unary,0).tolist(),[7,2])

    def test_software_bf16_convolution_matches_dtype_operations(self):
        m=self.tiny();m.round_bf16=True
        x=torch.randn(8,16).to(torch.bfloat16)
        delta=torch.randn(8,2,4).to(torch.bfloat16)
        base=torch.randn(2,16).to(torch.bfloat16)
        values=x.reshape(8,4,4);expected=torch.zeros_like(values)
        for tap in range(2):
            shifted=values if not tap else torch.nn.functional.pad(values[:-1],(0,0,0,0,1,0))
            expected=expected+base[tap].reshape(1,4,4)*shifted
            expected=torch.addcmul(expected,delta[:,tap,:,None],shifted)
        torch.testing.assert_close(m.conv(x.float(),delta.float(),base.float()),expected.float().reshape(8,16),rtol=0,atol=0)

    def test_bf16_rounding_preserves_wide_projection_range(self):
        m=self.tiny();m.round_bf16=True
        actual=m.project(torch.full((8,64),200.),torch.full((16,64),20.))
        self.assertEqual(actual.dtype,torch.float32)
        self.assertTrue(torch.equal(actual,torch.full_like(actual,256000.)))

    def test_frames_reject_truncation_and_handle_empty_delta(self):
        self.assertEqual(read_exact(io.BytesIO(b'abc'),3),b'abc')
        with self.assertRaises(EOFError):read_exact(io.BytesIO(b'ab'),3)
        self.assertEqual(tuple(tensor_frame(io.BytesIO(),0,80,'cpu').shape),(0,80))

    def test_wrong_target_configuration_rejected(self):
        with self.assertRaises(ValueError):validate_config({'architectures':['DFlashDraftModel']})


if __name__ == '__main__':unittest.main()
