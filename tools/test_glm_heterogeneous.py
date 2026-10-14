"""No GPU/model required: resource conservation and trace semantics."""
import unittest
from tools.sim.heterogeneous import Event,schedule,ByteCache,memory_check,transfer_events,evaluate,validate_predictions,partition_prefill

class ResourceTests(unittest.TestCase):
    def test_independent_overlap(self):
        r=schedule([Event('a',{'a':10}),Event('b',{'b':7})],{'a':1,'b':1})
        self.assertEqual(r['milliseconds'],10);self.assertEqual(r['lower_bound_ms'],10)
    def test_shared_bandwidth(self):
        r=schedule([Event('cpu',{'ram':60}),Event('dma',{'ram':40})],{'ram':10})
        self.assertAlmostEqual(r['milliseconds'],10)
    def test_redistribution(self):
        r=schedule([Event('cpu',{'ram':1,'cpu':10}),Event('dma',{'ram':90})],{'ram':10,'cpu':1})
        self.assertAlmostEqual(r['milliseconds'],10)
    def test_join(self):
        r=schedule([Event('a',{'a':5}),Event('b',{'b':8}),Event('c',{'a':2},('a','b'))],{'a':1,'b':1})
        self.assertEqual(r['milliseconds'],10)
    def test_bad_graphs(self):
        for events in [[Event('a',{},('b',))],[Event('a',{},('b',)),Event('b',{},('a',))],[Event('a'),Event('a')]]:
            with self.assertRaises(ValueError):schedule(events,{'a':1})
        with self.assertRaises(ValueError):schedule([Event('a',{'a':float('nan')})],{'a':1})
    def test_cache(self):
        c=ByteCache(20,[('a',10),('b',10)])
        self.assertFalse(c.admit('c',10,('a','b')));self.assertEqual(c.used,20)
        c.contains('a');self.assertTrue(c.admit('c',10));self.assertNotIn('b',c.entries)
        self.assertFalse(c.admit('large',21));self.assertEqual(c.used,20)
    def test_memory(self):
        h={'memory_bytes':{'cuda':10}}
        self.assertEqual(memory_check(h,{'cuda':{'desktop':3,'cache':7}})['cuda']['free_bytes'],0)
        with self.assertRaises(ValueError):memory_check(h,{'cuda':{'desktop':3,'cache':8}})
    def test_transfer_path(self):
        h={'paths':{'cuda->hip':[{'bytes':{'cuda.d2h':1,'ram':1}},{'bytes':{'hip.h2d':1,'ram':1}}]}}
        e=transfer_events('bounce',10,'cuda','hip',h)
        self.assertEqual(e[1].after,('bounce/0',))
        self.assertEqual(schedule(e,{'cuda.d2h':1,'hip.h2d':1,'ram':100})['milliseconds'],20)
    def test_gate(self):
        self.assertFalse(validate_predictions([])['calibrated'])
        r=[dict(kind='decode',split='holdout',measured_tok_s=30,predicted_tok_s=50)]
        self.assertFalse(validate_predictions(r)['calibrated'])
    def test_aggregate_adapter(self):
        a=dict(tokens=100,unassigned_ms=1,layers=[dict(layer=3,mixer_ms=4,ffn_ms=20)])
        b=dict(tokens=100,unassigned_ms=1,layers=[dict(layer=3,mixer_ms=3,ffn_ms=10)])
        self.assertEqual(partition_prefill(a,b,.5,2)['milliseconds'],17)
        self.assertEqual(partition_prefill(a,b,1,2)['milliseconds'],25)
        self.assertEqual(partition_prefill(a,b,0,2)['milliseconds'],15)
    def test_zero_markers(self):
        self.assertEqual(schedule([Event('a'),Event('b',{},('a',))],{})['milliseconds'],0)
    def test_long_trace(self):
        events=[Event(str(i),{'cpu':1},(str(i-1),) if i else ()) for i in range(10000)]
        self.assertEqual(schedule(events,{'cpu':1})['milliseconds'],10000)
    def test_resource_bound_random_dags(self):
        import random
        randomizer=random.Random(734)
        for _ in range(100):
            events=[]
            for i in range(20):
                dependencies=tuple(str(j) for j in range(i) if randomizer.random()<.1)
                work={r:randomizer.randrange(10) for r in ('cpu','ram','gpu')}
                events.append(Event(str(i),work,dependencies))
            result=schedule(events,{'cpu':1,'ram':3,'gpu':2})
            self.assertGreaterEqual(result['milliseconds']+1e-8,result['lower_bound_ms'])


class TraceTests(unittest.TestCase):
    def hardware(self):
        return dict(primary='hip',resources={'host':1,'cpu':1,'ram':100,'cuda.compute':1,'hip.compute':1,'cuda.vram':100,'hip.vram':100,'disk':10,'h2d':10},
            memory_bytes={'ram':1000,'cuda':1000,'hip':1000},cache_bytes={'ram':1000,'cuda':0,'hip':0},
            paths={'disk:model->ram':[{'bytes':{'disk':1,'ram':1}}],
                   'ram->cuda':[{'bytes':{'h2d':1,'ram':1}}],'ram->hip':[],'hip->ram':[],
                   'hip->cuda':[{'bytes':{'h2d':1}}],'cuda->hip':[{'bytes':{'h2d':1}}]})
    def test_rejected_round_replayed_and_disk_cache_warms(self):
        from tools.sim.heterogeneous import trace_replay
        groups=[dict(request=0,event=i,round=i,phase='verify',layer=3,routes=[[1]]) for i in (0,1)]
        c={'3':{'1':dict(expert_bytes=100,experts=4,cpu_ms=3,cuda_ms=1,hip_ms=1)}}
        r=trace_replay(groups,self.hardware(),c,accepted_tokens=1)
        self.assertEqual(r['counters']['groups'],2)
        self.assertEqual(r['counters']['disk_bytes'],100)
        self.assertEqual(r['counters']['cpu_bytes'],200)
        self.assertEqual(r['milliseconds'],16)
    def test_cache_budget_included(self):
        h=self.hardware();h['memory_bytes']['ram']=1001
        with self.assertRaises(ValueError):memory_check(h,{'ram':{'scratch':2}})
    def test_no_acceptance_guess(self):
        from tools.sim.heterogeneous import trace_replay
        with self.assertRaises(ValueError):trace_replay([],self.hardware(),{})
    def test_remote_group_single_activation_and_return(self):
        from tools.sim.heterogeneous import trace_replay
        h=self.hardware();h['cache_bytes']['cuda']=200
        groups=[dict(request=0,event=0,layer=3,routes=[[1,2]])]
        costs={'3':{'1':dict(expert_bytes=100,experts=4,cpu_ms=3,cuda_ms=1,hip_ms=1,cuda_reduce_ms=.5)}}
        # JSON-decoded [layer, expert] keys must work without caller conversion.
        initial={'cuda':[[[3,1],100],[[3,2],100]]}
        result=trace_replay(groups,h,costs,initial=initial,accepted_tokens=1)
        transfers=[e for e in result['events'] if e['category']=='transfer']
        self.assertEqual(len(transfers),2)
        self.assertEqual(result['counters']['gpu_bytes'],200)
        self.assertEqual(result['counters']['disk_bytes'],0)
    def test_no_unaccounted_gpu_staging(self):
        from tools.sim.heterogeneous import trace_replay
        groups=[dict(request=0,event=0,layer=3,routes=[[1]])]
        costs={'3':{'1':dict(expert_bytes=100,experts=4,cpu_ms=3,cuda_ms=1,hip_ms=1)}}
        with self.assertRaisesRegex(ValueError,'GPU cache cannot hold'):
            trace_replay(groups,self.hardware(),costs,policy='streamed-gpu',accepted_tokens=1)
    def test_resident_plan_is_disjoint_and_bounded(self):
        from tools.sim.heterogeneous import plan_resident_cache
        h=self.hardware();h['cache_bytes'].update(cuda=100,hip=100)
        groups=[dict(layer=3,routes=[[1,2,3]])]*3+[dict(layer=3,routes=[[1]])]
        costs={'3':{'1':dict(expert_bytes=100,experts=4,cpu_ms=3,cuda_ms=1,hip_ms=1)}}
        result=plan_resident_cache(groups,h,costs)
        inventory=[tuple(key) for entries in result['initial'].values() for key,size in entries]
        self.assertEqual(len(inventory),2);self.assertEqual(len(set(inventory)),2)
        self.assertIn((3,1),inventory)
        self.assertEqual(result['unallocated_cache_bytes'],{'cuda':0,'hip':0})

class QueueTests(unittest.TestCase):
    def test_queue_limit(self):
        # Four independent 1 ms operations, but only two may be in flight.
        ev=[Event(str(i),{'r'+str(i):1},slots={'io':1}) for i in range(4)]
        r=schedule(ev,{'r'+str(i):1 for i in range(4)},{'io':2})
        self.assertEqual(r['milliseconds'],2)
    def test_invalid_queue(self):
        with self.assertRaises(ValueError):schedule([Event('x',{'r':1},slots={'io':2})],{'r':1},{'io':1})
    def test_independent_io_latency_and_shared_bandwidth(self):
        e=[Event(str(i),{'disk':10},slots={'io':1},minimum_ms=2) for i in range(4)]
        self.assertEqual(schedule(e,{'disk':100},{'io':4})['milliseconds'],2)
        self.assertEqual(schedule(e,{'disk':10},{'io':4})['milliseconds'],4)
        self.assertEqual(schedule(e,{'disk':100},{'io':1})['milliseconds'],8)
        self.assertEqual(schedule([Event('timer',minimum_ms=3)],{})['milliseconds'],3)
    def test_storage_chunks_charge_overfetch(self):
        h={'resources':{'disk:model':100000,'ram':100000},'slots':{'io':2},
           'paths':{'disk:model->ram':[{'bytes':{'disk:model':1,'ram':1},'slots':{'io':1}}]},
           'storage_read_bytes':{'disk:model':4096},
           'storage_io':{'disk:model':{'4096':{'single_read_ms':1}}}}
        events=transfer_events('read',9000,'disk:model','ram',h)
        self.assertEqual(sum(e.work.get('disk:model',0) for e in events),3*4096)
        self.assertEqual(schedule(events,h['resources'],h['slots'])['milliseconds'],2)

class MissLaneTests(unittest.TestCase):
    hardware=TraceTests.hardware
    def test_slower_individual_gpu_can_help_concurrent_group(self):
        from tools.sim.heterogeneous import plan_miss_lanes
        h=self.hardware();h['resources']['ram']=1e9
        h['return_bytes_per_row']={'cpu':0}  # isolate the concurrent expert lanes
        h['paths']['ram->hip']=[{'bytes':{'h2d':1,'ram':1}}]
        result=plan_miss_lanes(8,100,{'cpu_ms':3,'cuda_ms':1,'hip_ms':1},h,
                               staging_experts={'cuda':0,'hip':3})
        self.assertEqual(result['best']['counts'],{'cpu':7,'cuda':0,'hip':1})
        self.assertAlmostEqual(result['best']['milliseconds'],21)
    def test_shared_ram_can_remove_lane_gain(self):
        from tools.sim.heterogeneous import plan_miss_lanes
        h=self.hardware();h['resources']['ram']=10
        h['return_bytes_per_row']={'cpu':0}
        h['paths']['ram->hip']=[{'bytes':{'h2d':1,'ram':1}}]
        result=plan_miss_lanes(8,100,{'cpu_ms':3,'cuda_ms':1,'hip_ms':1},h,
                               staging_experts={'cuda':0,'hip':3})
        self.assertEqual(result['best']['counts']['cpu'],8)
    def test_staging_memory_is_charged(self):
        from tools.sim.heterogeneous import plan_miss_lanes
        h=self.hardware();h['memory_bytes']['hip']=200
        with self.assertRaisesRegex(ValueError,'hip needs'):
            plan_miss_lanes(8,100,{'cpu_ms':3,'cuda_ms':1,'hip_ms':1},h,
                           staging_experts={'cuda':0,'hip':3})

if __name__=='__main__':unittest.main()
