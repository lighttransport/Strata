import unittest
from glm_tier_window_replay import compare


class WindowReplayTest(unittest.TestCase):
    def test_repetition_within_group_does_not_count_as_another_read(self):
        def row(routes): return dict(layer=3, routes=routes, resident=[])
        train = [row([[0], [0], [0], [0]]), row([[1]]), row([[1]])]
        test = [row([[1], [1]]), row([[1]])]
        r = compare(train, test, {3: 100}, {3}, 100)
        self.assertEqual(r['routes']['cpu_bytes'], 200)
        self.assertEqual(r['window']['cpu_bytes'], 0)
        self.assertEqual(r['window']['initial_upload_bytes'], 100)
        self.assertTrue(r['implementation_gate'])

    def test_ineligible_layers_cannot_become_gpu_hits(self):
        rows = [dict(layer=11, routes=[[1], [1]], resident=[])]
        r = compare(rows, rows, {11: 100}, set(), 1000)
        self.assertEqual(r['window']['cpu_bytes'], 100)
        self.assertEqual(r['window']['slots'], 0)
        self.assertFalse(r['implementation_gate'])


if __name__ == '__main__': unittest.main()
