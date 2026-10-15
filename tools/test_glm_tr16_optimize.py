import json
from pathlib import Path
import tempfile
import unittest
from glm_tr16_optimize import profile, results_summary, no_swap_scope


class OptimizeTest(unittest.TestCase):
    def test_scope_requires_zero_swap_and_checks_ancestors(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);child=root/'task'/'child';child.mkdir(parents=True)
            (root/'memory.swap.max').write_text('max')
            self.assertIsNone(no_swap_scope('0::/task/child',root))
            (root/'task'/'memory.swap.max').write_text('0')
            self.assertEqual(no_swap_scope('0::/task/child',root),str(root/'task'))
            self.assertIsNone(no_swap_scope('0::/../../outside',root))

    def base(self):
        return dict(decode_cache_mib=12288,env={'STRATA_GLM_BATCHED_RESIDENT':'0',
                    'STRATA_GLM_MIXED_PROTOCOL':'5','STRATA_GLM_MIXED_EXCHANGE':'f16',
                    'STRATA_GLM_ROUTE_AFFINITY':'0.14'})

    def test_parent_and_transport_are_explicit(self):
        base=self.base()
        c=profile(base,'base','parent','cuda-hip')
        self.assertNotIn('STRATA_GLM_BATCHED_RESIDENT',c['env'])
        self.assertNotIn('STRATA_GLM_MIXED_PROTOCOL',c['env'])
        self.assertEqual(c['decode_cache_mib'],10240)
        self.assertEqual(c['env']['STRATA_GLM_ROUTE_AFFINITY'],'0.14')
        self.assertEqual(base['env']['STRATA_GLM_MIXED_PROTOCOL'],'5')
        c=profile(base,'batched-mla','exchange','hip-cuda')
        self.assertEqual(c['env']['STRATA_GLM_BATCHED_RESIDENT'],'1')
        self.assertEqual(c['env']['STRATA_GLM_MLA_GRAPHS'],'1')
        self.assertEqual(c['env']['STRATA_GLM_MIXED_PROTOCOL'],'5')
        self.assertEqual(c['env']['STRATA_GLM_MIXED_REGISTER'],'1')
        self.assertEqual(profile(base,'batched','gpu-draft','hip-cuda')['mtp_experts'],'gpu')

    def test_short_or_incomplete_runs_cannot_pass_speed_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);prompt=folder/'prompt.ids'
            prompt.write_text(','.join(['1']*8192))
            data=dict(results=[dict(returncode=0,measurement=dict(complete=True,clean=True))],
                      summary=[dict(prompt=str(prompt),fraction=.5,median_decode_tok_s=51,median_prefill_tok_s=901)])
            path=folder/'results.json';path.write_text(json.dumps(data))
            self.assertTrue(results_summary(path,3)['rows'][0]['speed_gate'])
            self.assertFalse(results_summary(path,3,128)['rows'][0]['speed_gate'])
            self.assertFalse(results_summary(path,1)['rows'][0]['speed_gate'])
            self.assertFalse(results_summary(path,3)['quality_qualified'])
            prompt.write_text('1,2')
            self.assertFalse(results_summary(path,3)['rows'][0]['speed_gate'])
            data['results'][0]['measurement']['clean']=False;path.write_text(json.dumps(data))
            self.assertFalse(results_summary(path,3)['eligible'])
