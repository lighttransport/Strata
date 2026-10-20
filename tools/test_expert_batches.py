import unittest
from tools.sim.expert_batches import analyze


class ExpertBatchCoverage(unittest.TestCase):
    def test_request_width_is_not_expert_width(self):
        r=analyze([dict(routes=[[0,1],[0,2],[0,3],[0,4]],resident=[0])])['phases']['unspecified']
        self.assertEqual(r['all_routes']['job_width_histogram'],{1:4,4:1})
        self.assertEqual(r['all_routes']['exact_width_route_fraction'],.5)
        self.assertEqual(r['nonresident_proxy']['exact_width_jobs'],0)

    def test_no_merging_across_slots_or_phases(self):
        events=[dict(routes=[[1]],phase='target') for _ in range(4)]
        events.append(dict(routes=[[1]]*4,phase='draft'))
        r=analyze(events)['phases']
        self.assertEqual(r['target']['all_routes']['exact_width_jobs'],0)
        self.assertEqual(r['draft']['all_routes']['exact_width_jobs'],1)
        self.assertIsNone(r['target']['nonresident_proxy'])

    def test_exact_width_not_at_least(self):
        r=analyze([dict(routes=[[1]]*8)])['phases']['unspecified']
        self.assertEqual(r['all_routes']['exact_width_jobs'],0)
        self.assertEqual(r['event_width_histogram'],{8:1})

    def test_empty_cpu_route_sets_and_incomplete_metadata(self):
        r=analyze([dict(routes=[[],[]],resident=[]),dict(routes=[[1]])])['phases']['unspecified']
        self.assertIsNone(r['nonresident_proxy'])
        self.assertEqual(r['all_routes']['routed_token_expert_rows'],1)
        self.assertEqual(analyze([])['phases'],{})

    def test_bad_ids_and_width(self):
        for routes in [[],[[1,1]],[[-1]],[[True]],[[1.0]],['bad']]:
            with self.subTest(routes=routes),self.assertRaises(ValueError):analyze([dict(routes=routes)])
        for width in [0,-1,True,1.5]:
            with self.assertRaises(ValueError):analyze([],width)


if __name__=='__main__':unittest.main()
