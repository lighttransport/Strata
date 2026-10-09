"""B550 calibration correctness and prediction isolation tests."""
import copy
import contextlib
import io
from types import SimpleNamespace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent/'sim'))
import calibration
import glm_sim
import decode
import events
import hw
import kernels
import model
import routing
from config import RunConfig
sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm_low_memory_bench import parse_log
from glm_b550_sim_records import import_run


class B550Test(unittest.TestCase):
    def test_lossy_calibration_cannot_change_ordinary_prediction(self):
        record = dict(kind='decode', hw='b550', config=dict(pack='reap50_q23', speculation='none'))
        original = kernels.Params()
        changed = copy.deepcopy(original)
        changed.routing_affinity_tau = .168
        changed.biased_resident_gpu_scale = .7
        self.assertEqual(calibration.predict(record, original)[0], calibration.predict(record, changed)[0])
        record['config']['affinity'] = .15
        self.assertNotEqual(calibration.predict(record, original)[0], calibration.predict(record, changed)[0])

    def test_holdout_never_changes_fit(self):
        train = dict(name='train', fit=True, kind='decode', hw='b550', config=dict(pack='reap50_q23', speculation='none'), measured=dict(tok_s=10))
        held = copy.deepcopy(train); held.update(name='holdout',fit=False); held['measured']['tok_s']=1000
        params = kernels.Params()
        self.assertEqual(calibration.objective([train],params),calibration.objective([train,held],params))
        with self.assertRaises(ValueError): calibration.fit([held],log=lambda _:None)

    def test_record_trace_is_loaded(self):
        r = dict(kind='decode', config=dict(trace='fixture.csv', prior='prior.json'), measured=dict(tok_s=10))
        with patch.object(routing,'read_routes',return_value={0:[]}) as read, patch.object(decode,'simulate') as simulate:
            simulate.return_value.tok_s=10
            calibration.predict(r,kernels.Params())
            read.assert_called_once_with('fixture.csv')
            self.assertEqual(simulate.call_args.kwargs['trace'],{0:[]})
            self.assertEqual(simulate.call_args.kwargs['prior'],'prior.json')

    def test_reap_trace_not_silently_ignored(self):
        cfg=RunConfig(pack='reap50_q23',speculation='none')
        with patch.object(routing,'trace_tier_hit',return_value=.42) as replay:
            r=decode.simulate(hw.b550(),cfg,trace={0:[]},prior='prior.json')
            self.assertEqual(r.hit_bytes_share,.42)
            self.assertEqual(replay.call_args.kwargs['experts'],144)

    def test_runtime_profile_cli_override(self):
        machine=hw.b550().set('runtime_vram.reap50_q23.runtime_headroom','256')
        self.assertEqual(machine.runtime_vram['reap50_q23']['runtime_headroom'],256)
        with self.assertRaises(KeyError):machine.set('runtime_vram.reap50_q23.unknown','1')

    def test_measured_phase_memory(self):
        cfg=RunConfig(pack='reap50_q23',speculation='none',gpu_budget_mib=14336,reserve_mib=2048,decode_cache_mib=6144)
        plan=decode.make_plan(hw.b550(),cfg,model.pack(cfg.pack))
        self.assertAlmostEqual(plan.tier_mib,6142.06,delta=.01)
        self.assertEqual(plan.items['prefill_scratch'],32)
        self.assertEqual(plan.items['dense'],5029.21)
        self.assertLessEqual(sum(plan.items.values())+plan.tier_mib,plan.budget_mib)

    def test_dma_does_not_create_host_bandwidth(self):
        machine=hw.b550()
        cfg=RunConfig(pack='reap50_q23',speculation='none',pcie_share=.5)
        r=decode.simulate(machine,cfg)
        self.assertGreaterEqual(r.step.cpu_ms+1e-6,(r.step.cpu_bytes+r.step.pcie_bytes)/(machine.memory.dram_gbps*1e6))

    def test_named_metrics_preserve_large_integer_fingerprint(self):
        r=parse_log('CPU_EXPERT gu_ms=1 quant_ms=2 down_ms=3 layer_flow_ms=4 bytes=123\nDECODE_CACHE prefix_trained=1 slots=707 fingerprint=1234567890123456789\n')
        self.assertEqual(r['cpu_expert'][0]['bytes'],123)
        self.assertEqual(r['decode_cache'][0]['fingerprint'],1234567890123456789)

    def test_rejected_positions_stay_separate(self):
        row=dict(request=1,round=0,phase='verify',event=0,position=10,layer=3,routes=[list(range(8)),list(range(4,12))])
        other=dict(row,event=1,round=1,position=10)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'events.jsonl';p.write_text(json.dumps(row)+'\n'+json.dumps(other)+'\n')
            e=events.read_events(p)
        stats=events.execution_unions(e,model.pack('reap50_q23'))
        self.assertEqual(len(stats),2)
        self.assertEqual(stats[0]['expert_bytes'],12*model.pack('reap50_q23').expert_bytes(3))
        r=events.replay(e,model.pack('reap50_q23'),128)
        self.assertEqual(r['hit_share'],.5)
        self.assertEqual(r['upload_bytes'],stats[0]['expert_bytes'])

    def test_validation_fails_held_out_miss(self):
        row = dict(name='heldout',kind='decode',measured=10,predicted=20,error=1,tolerance=.1,
                   within=False,quiet=True,fit=False,bottleneck='cpu')
        with patch.object(calibration,'load_records',return_value=[]), patch.object(calibration,'evaluate',return_value=[row]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(glm_sim.validate(SimpleNamespace(params=None,records=None,output=None)),1)

    def test_initial_inventory_is_not_an_execution(self):
        cache = dict(kind='cache',layer=3,experts=[1,2])
        event = dict(request=1,round=-1,phase='target',event=0,position=2048,layer=3,routes=[list(range(8))])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'events.jsonl';p.write_text(json.dumps(cache)+'\n'+json.dumps(event)+'\n')
            self.assertEqual(len(events.read_events(p)),1)
            self.assertEqual(events.read_initial(p),[(3,1),(3,2)])

    def test_profiled_run_cannot_be_fit_as_throughput(self):
        r=dict(clean=True,complete=True,config=dict(env={'STRATA_GLM_STEP_TRACE':'1'}),measurements={})
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'profile.result.json';p.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'profiling/parity'):
                import_run(p,fit=True)

    def test_uncalibrated_routing_and_lookup_cannot_be_fit(self):
        for env, spec in [({'STRATA_GLM_ROUTE_AFFINITY': '.12'}, 'none'),
                          ({'STRATA_GLM_ROUTE_AFFINITY_PROFILE': 'layers.json'}, 'none'),
                          ({}, 'lookup')]:
            r = dict(clean=True, complete=True, config=dict(env=env, speculative=spec), measurements={})
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp)/'lossy.result.json'; p.write_text(json.dumps(r))
                with self.assertRaisesRegex(ValueError, 'separate calibrated model'):
                    import_run(p, fit=True)

    def test_cache_learning_experiments_cannot_enter_old_routing_fit(self):
        for env in [{'STRATA_GLM_CACHE_PROBE': '24'}, {'STRATA_GLM_TIER_LEARN_UNBIASED': '1'}]:
            r = dict(clean=True, complete=True, config=dict(env=env, speculative='none'), measurements={})
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp)/'probe.result.json'; p.write_text(json.dumps(r))
                with self.assertRaisesRegex(ValueError, 'separate calibration'):
                    import_run(p, fit=True, allow_routing_bias=True)

    def test_replay_does_not_mix_request_histories(self):
        a=dict(request=1,layer=3,routes=[list(range(8))])
        with self.assertRaises(ValueError):
            events.replay([a,dict(a,request=2)],model.pack('reap50_q23'),128)

    def test_implemented_plan_rejects_hypothetical_dma(self):
        glm_sim.check_implemented_config(RunConfig())
        with self.assertRaises(ValueError):
            glm_sim.check_implemented_config(RunConfig(pcie_share=.5))
        with self.assertRaises(ValueError):
            glm_sim.check_implemented_config(RunConfig(tier_policy='lru'))

    def test_quality_reference_is_explicit(self):
        cfg=RunConfig(pack='reap50_q23',speculation='none')
        self.assertFalse(decode.simulate(hw.b550(),cfg).lossless)
        self.assertTrue(decode.simulate(hw.b550(),cfg.copy(quality_reference=cfg.pack)).lossless)
        self.assertFalse(decode.simulate(hw.b550(),cfg.copy(quality_reference=cfg.pack,affinity=.1)).lossless)

if __name__=='__main__': unittest.main()
