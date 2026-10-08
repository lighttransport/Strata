import tempfile
import unittest
from pathlib import Path
from glm_cache_replay import replay,load_trace

class CacheReplayTest(unittest.TestCase):
    def test_static_exclusive_training_only(self):
        sizes={'a':1,'b':1,'c':1}
        train=[(0,0,['a','a','a','b','b','c'])]
        test=[(1,0,['c','a','b','c'])]
        r=replay(train,test,sizes,1,1,'static')
        self.assertEqual(r['gpu_hit_fraction'],.25)
        self.assertEqual(r['cpu_hit_fraction'],.25)
        self.assertEqual(r['miss_fraction'],.5)
        # CPU is B, not a duplicate of GPU's A, and future C frequency cannot
        # change either resident set.
        self.assertEqual(r['cpu_pinned_gib']*2**30,1)
        self.assertEqual(r['gpu_resident_gib']*2**30,1)

    def test_unseen_static_slots_have_no_future_information(self):
        r=replay([(0,0,['a','a','b'])],[(1,0,['d','d','d','c'])],
                 {'a':1,'b':1,'c':1,'d':1},2,1,'static')
        self.assertEqual(r['cpu_pinned_gib']*2**30,2)
        self.assertEqual(r['cpu_hit_fraction'],.25)  # C wins the zero-count tie, not future-hot D.
        self.assertEqual(r['miss_fraction'],.75)

    def test_lru_capacity_and_warm_state(self):
        r=replay([(0,0,['a','a','b','c'])],[(1,0,['c','a','b','c'])],
                 {'a':1,'b':1,'c':1},1,1,'lru')
        self.assertEqual(r['cpu_hit_fraction'],.25)
        self.assertEqual(r['miss_fraction'],.5)
        self.assertEqual(r['cpu_evictions'],2)
        self.assertEqual(r['cpu_final_lru_gib']*2**30,1)

    def test_trace_rejects_missing_layers_and_discards_post_stop(self):
        inv=dict(stop_ids=[99],top_k=1,first_layer=0,last_layer=2,
                 entries={'0:0':{},'1:0':{}})
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace.csv'
            p.write_text('10,0,0\n10,1,0\n11,0,0\n11,1,0\n12,0,0\n12,1,0\n')
            rows,valid=load_trace(p,inv,10,[1,2,99,3])
            self.assertEqual(valid['usable_decode_steps'],1)
            self.assertEqual(len(rows),2)
            p.write_text('10,0,0\n')
            with self.assertRaises(ValueError):load_trace(p,inv,10,[1,2,99])

if __name__=='__main__':unittest.main()
