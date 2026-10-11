import copy
import unittest
from pathlib import Path
from glm_coding_qualification import gate, evaluate
from glm_coding_quality import HOLDOUTS, TRAINING
from glm_coding_perf_matrix import aggregate, length_summary, validate_configs, valid_quality_gate


class QualificationTest(unittest.TestCase):
    def test_invalid_task_mode_is_rejected_before_any_gpu_work(self):
        for scope,mode in [('tuning','decode'),('holdout','prefill')]:
            with self.assertRaisesRegex(ValueError,'holdout scope and decode mode'):
                evaluate(Path('unused-config'),Path('unused-inputs'),Path('unused-output'),
                         scope,tasks=True,mode=mode)

    def reference(self):
        return dict(scope='holdout',targets=32768,corpus_sha256='same',perplexity=3.,passed=True,
                    tasks={name:{'passed':True} for name in HOLDOUTS},prefill_passed=True,
                    prefill_perplexity=3.,prefill_targets=32768,prefill_corpus_sha256='same')

    def test_strict_budget_and_complete_holdout(self):
        a=self.reference();b=copy.deepcopy(a);b['perplexity']=3.02
        self.assertTrue(gate(a,b,True)['passed'])
        b['perplexity']=3.04
        self.assertFalse(gate(a,b,True)['passed'])
        b=copy.deepcopy(a);b['tasks'].pop(next(iter(b['tasks'])))
        self.assertFalse(gate(a,b,True)['passed'])
        b=copy.deepcopy(a);b['corpus_sha256']='other'
        self.assertFalse(gate(a,b,True)['passed'])
        self.assertFalse(gate(a,a,False)['passed'])

    def test_no_new_failures_and_baseline_failures_remain_visible(self):
        a=self.reference();b=copy.deepcopy(a);name=next(iter(HOLDOUTS))
        b['tasks'][name]['passed']=False
        self.assertEqual(gate(a,b,True)['heldout_new_failures'],[name])
        a['tasks'][name]['passed']=False
        self.assertTrue(gate(a,b,True)['passed'])

    def test_nonfinite_or_zero_ppl_cannot_pass(self):
        a=self.reference()
        for value in [0,float('nan'),float('inf')]:
            b=copy.deepcopy(a);b['perplexity']=value
            self.assertFalse(gate(a,b,True)['passed'])

    def test_tuning_is_disjoint_from_twenty_holdouts(self):
        self.assertEqual(len(HOLDOUTS),20)
        self.assertFalse(set(TRAINING)&set(HOLDOUTS))
        self.assertFalse(gate(dict(self.reference(),scope='tuning'),self.reference(),True)['passed'])

    def test_lossy_candidates_still_require_their_own_speculative_control(self):
        a=dict(model='m',expert_pack='q23',threads=12,decode_cache_slots=800,env={})
        b=dict(a,expert_pack='q22')
        configs=[('reference',a),('candidate',b),('spec',dict(b,speculative='external'))]
        self.assertEqual(validate_configs(configs,True)['spec'],'candidate')
        with self.assertRaises(ValueError):validate_configs([configs[0],configs[2]],True)

    def test_quality_only_cannot_satisfy_warm_timing_targets(self):
        row=dict(config='x',fixture='prime',length=8192,trial=0,clean=True,phase='quality',
                 prefill_tok_s=2000,decode_tok_s=100,response_seconds=1,output_sha256='same',quality={'status':'passed'})
        self.assertEqual(aggregate([row]),[])

    def test_speed_alone_is_not_release_qualification(self):
        row=dict(config='x',fixture='prime',length=8192,trial=0,clean=True,prefill_tok_s=1100,
                 decode_tok_s=42,response_seconds=1,output_sha256='same',quality={'status':'passed'})
        summary=length_summary(aggregate([row]),'x',{8192:['prime']},1,prefill_target=1000)[0]
        self.assertTrue(summary['greedy_target_met'])
        self.assertFalse(summary['release_qualified'])
        self.assertFalse(valid_quality_gate({'passed':True}))


if __name__=='__main__':unittest.main()
