"""Experiment gates and prefill selection; no model or GPU required."""
import copy
import unittest
from glm_tr16_quality import compare, evaluation_prefill, TOLERANCE

class QualityTest(unittest.TestCase):
    def result(self):
        return dict(config={'env':{}}, complete=True, corpus_sha256='corpus', reference='ref',
                    reference_size=1, reference_mtime_ns=1, sequences=16, inputs={},
                    metrics=dict(perplexity=5.,kl=.1,top1_agreement=.9,tokens=8192))

    def test_explicit_resumed_gate(self):
        baseline=self.result();candidate=copy.deepcopy(baseline)
        candidate['metrics'].update(perplexity=5.25,kl=.17,top1_agreement=.875)
        self.assertTrue(compare(candidate,baseline))
        candidate['quality_gate']='tr16-35'
        candidate['inputs']={'weight_files':['reap224']}
        self.assertEqual(compare(candidate,baseline),[])
        for key,value in [('perplexity',5.31),('kl',.181),('top1_agreement',.869)]:
            bad=copy.deepcopy(candidate);bad['metrics'][key]=value
            self.assertTrue(compare(bad,baseline))
        self.assertEqual(TOLERANCE,dict(ppl_rel=.03,kl_rel=0,kl_abs=.03,top1_abs=.01))

    def test_prefill_coverage_and_comparability(self):
        self.assertEqual(evaluation_prefill({'context':12288}),256)
        self.assertEqual(evaluation_prefill({'context':12288,'prefill_batch':8192}),8192)
        self.assertEqual(evaluation_prefill({'context':12288,'prefill_batch':8192},1024),1024)
        for invalid in (0,8193,'8192',True):
            with self.assertRaises(ValueError): evaluation_prefill({'context':8192,'prefill_batch':invalid})
        with self.assertRaises(ValueError): evaluation_prefill({'context':2048},8192)
        baseline=self.result();candidate=self.result();candidate['eval_prefill']=True
        self.assertTrue(compare(candidate,baseline))
        baseline.update(eval_prefill=True,prefill_batch=8192);candidate['prefill_batch']=256
        self.assertTrue(compare(candidate,baseline))
        candidate['prefill_batch']=8192
        self.assertEqual(compare(candidate,baseline),[])
