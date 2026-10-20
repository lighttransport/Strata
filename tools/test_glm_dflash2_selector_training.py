"""Selector-aware loss matches the serving decision and preserves default CE."""
from types import SimpleNamespace
import unittest
try:
    import torch
    from torch.nn import functional as F
    from glm_dflash2_calibrate import restore_weights
    from glm_dflash2_linear_adapter import LinearAdapters,apply_linear_adapters,target_shapes
    from glm_dflash2_selector_training import calibration_loss,correct_prefix,selection_key,selector_scores
    from glm_dflash2_training import LinearTrainingDraft
    import test_glm_dflash2_training as fixture
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'PyTorch not installed')
class SelectorTrainingTest(unittest.TestCase):
    def models(self):
        serving,old,_=fixture.CalibrationTest().models()
        serving.gemm_f32=True
        serving.w.update({'candidate_selector.hidden_projection.weight':torch.randn(8,16).half()*.1,
                          'candidate_selector.predecessor_codebook':torch.randn(64,8).half()*.1,
                          'candidate_selector.successor_codebook':torch.randn(64,8).half()*.1})
        weights={name:weight.clone() for name,weight in restore_weights(old).items()}
        weights.update({name:weight.clone() for name,weight in serving.w.items() if name.startswith('candidate_selector.')})
        adapter=LinearAdapters(target_shapes(old.c,weights,include_selector=True),2,2)
        model=LinearTrainingDraft(old.c,weights,'cpu',adapter)
        return serving,model,adapter

    def test_default_loss_and_gradients_are_exact_cross_entropy(self):
        logits=torch.randn(7,32,requires_grad=True);labels=torch.arange(7)
        actual,details=calibration_loss(logits,labels)
        expected=F.cross_entropy(logits,labels)
        self.assertTrue(torch.equal(actual,expected));self.assertEqual(details,{})
        a=torch.autograd.grad(actual,logits,retain_graph=True)[0]
        b=torch.autograd.grad(expected,logits)[0]
        self.assertTrue(torch.equal(a,b))

    def test_every_adapter_learns_and_selector_scores_match_serving_path(self):
        serving,model,adapter=self.models()
        frozen={name:weight.clone() for name,weight in model.w.items()}
        features,embedding=torch.randn(8,80)*1000,torch.randn(8,16)
        model.append(features,0,8);hidden=model.forward(embedding)
        head=torch.randn(32,16)*.1;logits=F.linear(hidden,head);labels=logits.detach().argmax(-1)
        loss,details=calibration_loss(logits,labels,model=model,anchor=3,selector_weight=1,prefix_decay=.8)
        self.assertEqual(details['selector_target_coverage'],1.)
        loss.backward()
        for module in adapter.up:
            self.assertIsNotNone(module.weight.grad)
            self.assertTrue(torch.isfinite(module.weight.grad).all())
            self.assertGreater(module.weight.grad.abs().sum().item(),0)
        torch.optim.SGD(adapter.parameters(),lr=.01).step()
        adapter.zero_grad(set_to_none=True)
        apply_linear_adapters(serving,adapter)
        serving.append(features,0,8);model.cache=[];model.append(features,0,8)
        actual=model.forward(embedding);expected=serving.forward(embedding)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        logits=F.linear(actual,head);ids=logits.detach().argsort(dim=-1,descending=True,stable=True)[:,:16]
        unary=logits.gather(1,ids);chosen=serving.select(ids,unary,3)
        self.assertTrue(torch.equal(model.select(ids,unary,3),chosen))
        previous=torch.cat((chosen.new_tensor([3]),chosen[:-1]))
        scores=selector_scores(model,actual,ids,unary,previous)
        self.assertTrue(torch.equal(ids.gather(1,scores.argmax(-1)[:,None])[:,0],chosen))
        loss,_=calibration_loss(logits,logits.detach().argmax(-1),model=model,anchor=3,selector_weight=1,prefix_decay=.8)
        loss.backward()
        for module in adapter.down:
            self.assertTrue(torch.isfinite(module.weight.grad).all())
            self.assertGreater(module.weight.grad.abs().sum().item(),0)
        for name,weight in model.w.items():
            self.assertIsNone(weight.grad);self.assertFalse(weight.requires_grad)
            torch.testing.assert_close(weight,frozen[name],rtol=0,atol=0)

    def test_inference_selector_evaluation_before_training(self):
        _,model,adapter=self.models()
        features,embedding=torch.randn(8,80),torch.randn(8,16)
        head=torch.randn(32,16)*.1
        with torch.no_grad():
            model.append(features,0,8)
            logits=F.linear(model.forward(embedding),head)
            ids=logits.argsort(dim=-1,descending=True,stable=True)[:,:16]
            model.select(ids,logits.gather(1,ids),3)
        self.assertTrue(model.backward_weights)
        self.assertFalse(any(weight.is_inference() for weight in model.backward_weights.values()))
        model.cache=[]
        model.append(features,0,8)
        logits=F.linear(model.forward(embedding),head)
        loss,_=calibration_loss(logits,logits.detach().argmax(-1),model=model,anchor=3,
                                selector_weight=1,prefix_decay=.8)
        loss.backward()
        for module in adapter.up:
            self.assertIsNotNone(module.weight.grad)
            self.assertTrue(torch.isfinite(module.weight.grad).all())
            self.assertGreater(module.weight.grad.abs().sum().item(),0)

    def test_missing_top16_teacher_token_keeps_finite_unary_loss(self):
        logits=torch.zeros(7,32,requires_grad=True)
        with torch.no_grad():logits[:,20]=-100;logits[:,31]=float('-inf')
        hidden=torch.randn(7,16,requires_grad=True);projection=torch.randn(8,16)
        model=SimpleNamespace(last_hidden=hidden,selector_scale=1.,rounded=lambda x:x,
                              linear=lambda x,name:F.linear(x,projection),
                              w={'candidate_selector.predecessor_codebook':torch.randn(32,8),
                                 'candidate_selector.successor_codebook':torch.randn(32,8)})
        labels=torch.full((7,),20,dtype=torch.long)
        loss,details=calibration_loss(logits,labels,model=model,anchor=3,selector_weight=1)
        self.assertEqual(details['selector_target_coverage'],0.)
        torch.testing.assert_close(loss,F.cross_entropy(logits,labels),rtol=0,atol=0)
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())

    def test_prefix_weighting_emphasizes_early_errors(self):
        early=torch.zeros(7,32);late=early.clone();labels=torch.zeros(7,dtype=torch.long)
        early[:,0]=5;late[:,0]=5;early[0,0]=-5;late[6,0]=-5
        a,_=calibration_loss(early,labels,prefix_decay=.8)
        b,_=calibration_loss(late,labels,prefix_decay=.8)
        self.assertGreater(float(a),float(b))

    def test_prefix_selection_differs_from_default_agreement(self):
        a=dict(evaluation_agreement=4/7,evaluation_loss=2.,selector_agreement=4/7,selector_mean_prefix=4)
        b=dict(evaluation_agreement=6/7,evaluation_loss=1.,selector_agreement=6/7,selector_mean_prefix=1)
        self.assertGreater(selection_key(a,'selector-prefix'),selection_key(b,'selector-prefix'))
        self.assertLess(selection_key(a,'token-agreement'),selection_key(b,'token-agreement'))
        self.assertEqual(correct_prefix([True,True,False,True]),2)


if __name__=='__main__':unittest.main()
