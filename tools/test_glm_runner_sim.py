"""No GPU/model required: GLM runner model structure, calibration mechanics and lever sweep."""
import copy
import json
import pathlib
import re
import tempfile
import unittest
from types import SimpleNamespace

from tools.sim import glm_runner as g
from tools.sim import glm_traces
from tools.sim.heterogeneous import evaluate as hetero_evaluate

DATA = pathlib.Path(__file__).resolve().parent / 'sim' / 'data'


def hardware():
    return json.loads((DATA / 'hw_tr16_runner.json').read_text())


def table():
    return json.loads((DATA / 'glm_kernels_tr16.json').read_text())


SHARES = dict(cpu=0.22, worker=0.46, primary=0.32)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.hw = hardware(); self.table = table()

    def decode(self, topo, place=None, **kw):
        place = place or g.Placement(primary_slots=15, worker_slots=33 if topo.worker and topo.kind == 'A' else 0)
        wl = g.synthetic_workload(self.table, place, topo, rounds=1, shares=kw.pop('shares', SHARES))
        return g.simulate_decode(wl, self.table, place, topo, self.hw)

    def test_scope_conservation(self):
        r = self.decode(g.Topology(kind='A', primary='cuda', worker='hip'))
        s = r['scopes']
        self.assertAlmostEqual(s['head'] + s['cpu_service'] + s['between'] + s['tail'], s['verify_ms'], places=6)
        self.assertAlmostEqual(s['verify_ms'] + s['draft'] + s['resync_other'], s['round_ms'], places=6)
        self.assertGreater(r['tok_s'], 20); self.assertLess(r['tok_s'], 60)

    def test_slots_and_worker_requests(self):
        topo = g.Topology(kind='A', primary='hip', worker='cuda')
        wl = g.synthetic_workload(self.table, g.Placement(), topo, rounds=1, shares=SHARES)
        self.assertEqual(len(wl[0].slots), 84)
        b, markers = g.decode_round_events(wl[0], self.table, g.Placement(), topo, self.hw)
        self.assertEqual(sum(1 for e in b.events if e.name.endswith('/submit')), 84)
        self.assertEqual(len(markers['cpu_start']), 84)
        self.assertEqual(sum(1 for e in b.events if e.name.endswith('/handoff-latency')), 0)

    def test_trace_round_can_have_one_verification_group(self):
        topo = g.Topology(primary='hip', worker='cuda')
        place = g.Placement()
        full = g.synthetic_workload(self.table, place, topo, shares=SHARES)[0]
        short = g.Round([slot for slot in full.slots if slot.group==0], 1.0, 0, 0.0)
        result = g.simulate_decode([short], self.table, place, topo, self.hw)
        self.assertGreater(result['tok_s'], 0)
        b, markers = g.decode_round_events(short, self.table, place, topo, self.hw)
        self.assertEqual(len(markers['cpu_start']), 42)

    def test_release_joint_dense_head_is_charged_once(self):
        topo = g.Topology(primary='hip', worker='cuda')
        kernels = g.patch(self.table, 'model.joint_dense', True)
        place = g.Placement()
        wl = g.synthetic_workload(kernels, place, topo, shares=SHARES)
        b, _ = g.decode_round_events(wl[0], kernels, place, topo, self.hw)
        heads = [e for e in b.events if e.category=='head']
        self.assertEqual(len(heads), 2)
        self.assertGreater(heads[0].work['hip.compute'], 0)
        self.assertEqual(heads[1].work['hip.compute'], 0)

    def test_single_gpu_has_no_worker(self):
        topo = g.Topology(kind='A', primary='hip', worker='')
        wl = g.synthetic_workload(self.table, g.Placement(worker_slots=0), topo, rounds=1, shares=dict(cpu=0.4, worker=0.0, primary=0.6))
        b, _ = g.decode_round_events(wl[0], self.table, g.Placement(worker_slots=0), topo, self.hw)
        self.assertEqual(sum(1 for e in b.events if '/worker' in e.name), 0)

    def test_layer_pipeline_handoffs(self):
        topo = g.Topology(kind='B', primary='hip', worker='cuda', cpu_lanes='node_local8x2', split_layer=24)
        wl = g.synthetic_workload(self.table, g.Placement(), topo, rounds=1, shares=SHARES)
        b, markers = g.decode_round_events(wl[0], self.table, g.Placement(), topo, self.hw)
        self.assertEqual(sum(1 for e in b.events if e.name.endswith('/handoff-latency')), 2)  # one per group
        self.assertEqual(sum(1 for e in b.events if e.name.endswith('/submit')), 0)
        lanes = {k for e in b.events for k in e.work if k.startswith('cpu.lane')}
        self.assertEqual(lanes, {'cpu.lane0', 'cpu.lane1'})
        r = g.simulate_decode(wl, self.table, g.Placement(), topo, self.hw)
        self.assertGreater(r['tok_s'], 10)

    def test_device3_combine_decouples_cpu_from_worker(self):
        base = g.Topology(kind='A', primary='hip', worker='cuda')
        slow = g.patch(self.table, 'devices.cuda.decode.worker_request_fixed_ms', 3.0)
        strict = g.simulate_decode(g.synthetic_workload(slow, g.Placement(), base, shares=SHARES), slow, g.Placement(), base, self.hw)
        dev3 = g.simulate_decode(g.synthetic_workload(slow, g.Placement(), base, shares=SHARES), slow, g.Placement(),
                                 g.Topology(kind='A', primary='hip', worker='cuda', combine='device3'), self.hw)
        self.assertLess(dev3['scopes']['cpu_service'], strict['scopes']['cpu_service'] * 0.7)

    def test_release_device_combine_keeps_the_cpu_worker_join(self):
        kernels = g.patch(self.table, 'devices.cuda.decode.worker_request_fixed_ms', 3.0)
        place = g.Placement()
        topo = g.Topology(primary='hip', worker='cuda', combine='device')
        wl = g.synthetic_workload(kernels, place, topo, shares=SHARES)
        actual = g.simulate_decode(wl, kernels, place, topo, self.hw)
        future = g.simulate_decode(wl, kernels, place, g.Topology(primary='hip', worker='cuda', combine='device3'), self.hw)
        self.assertGreater(actual['scopes']['cpu_service'], future['scopes']['cpu_service']*1.5)
        b, _ = g.decode_round_events(wl[0], kernels, place, topo, self.hw)
        self.assertEqual(sum(e.name.endswith('/host-join') for e in b.events), 84)

    def test_measured_cpu_service_can_add_host_work_to_kernel_time(self):
        kernels = copy.deepcopy(self.table)
        kernels['cpu'].update(additive_service=True, slot_fixed_ms=1.0, quant_ms_per_token=0.0)
        kernels['cpu']['lanes']['single15']['gb_s']=100.0
        topo=g.Topology(primary='hip', worker='', combine='device')
        place=g.Placement(worker_slots=0, draft_bank='primary')
        wl=g.synthetic_workload(kernels, place, topo, shares=dict(cpu=1.0, primary=0.0, worker=0.0))
        result=g.simulate_decode(wl, kernels, place, topo, self.hw)
        expected=sum(slot.cpu_bytes for slot in wl[0].slots)/100e6+len(wl[0].slots)
        self.assertAlmostEqual(result['scopes']['cpu_service'], expected, places=6)

    def test_rpc_probe_does_not_charge_return_copies_twice(self):
        kernels = g.patch(self.table, 'worker.service_includes_return', True)
        topo = g.Topology(primary='hip', worker='cuda', combine='device')
        place = g.Placement()
        wl = g.synthetic_workload(kernels, place, topo, shares=SHARES)
        b, _ = g.decode_round_events(wl[0], kernels, place, topo, self.hw)
        self.assertFalse(any('/worker-rows' in e.name for e in b.events))
        copies = [e for e in b.events if e.name.endswith('/worker-copy')]
        self.assertEqual(len(copies), 84)
        self.assertTrue(all(not e.work and e.minimum_ms > 0 for e in copies))
        self.assertTrue(any('/cpu-rows' in e.name for e in b.events))

    def test_direct_dispatch_and_async_ownership_never_slower(self):
        base = g.Topology(kind='A', primary='hip', worker='cuda')
        r0 = self.decode(base)
        r1 = self.decode(g.Topology(kind='A', primary='hip', worker='cuda', dispatch='direct'))
        r2 = self.decode(g.Topology(kind='A', primary='hip', worker='cuda', ownership='async'))
        self.assertLessEqual(r1['ms_per_round'], r0['ms_per_round'] * 1.01)  # the worker path is off the critical path here
        self.assertLess(r2['ms_per_round'], r0['ms_per_round'])

    def test_gpu_draft_bank_removes_cpu_draft_bytes(self):
        topo = g.Topology(kind='A', primary='hip', worker='cuda')
        cpu = g.synthetic_workload(self.table, g.Placement(draft_bank='cpu'), topo, shares=SHARES)[0]
        gpu = g.synthetic_workload(self.table, g.Placement(draft_bank='primary'), topo, shares=SHARES)[0]
        self.assertGreater(cpu.draft_cpu_experts, 0); self.assertEqual(gpu.draft_cpu_experts, 0)

    def test_release_draft_uses_four_experts(self):
        topo = g.Topology(primary='hip', worker='cuda')
        kernels = g.patch(self.table, 'model.draft_top_k', 4)
        cpu = g.synthetic_workload(kernels, g.Placement(), topo, shares=SHARES)[0]
        self.assertAlmostEqual(cpu.draft_cpu_experts, 4*kernels['model']['unique_experts_per_route'])
        place = g.Placement(draft_bank='primary')
        a = self.decode(topo, place)
        b = g.simulate_decode(g.synthetic_workload(kernels, place, topo, shares=SHARES), kernels, place, topo, self.hw)
        self.assertLess(b['scopes']['draft'], a['scopes']['draft'])

    def test_context_raises_mla_cost(self):
        short = self.decode(g.Topology(kind='A', primary='hip', worker='cuda', context_tokens=2048))
        long = self.decode(g.Topology(kind='A', primary='hip', worker='cuda', context_tokens=8192))
        self.assertLess(short['ms_per_round'], long['ms_per_round'])

    def test_manifest_round_trip(self):
        topo = g.Topology(kind='A', primary='cuda', worker='hip')
        wl = g.synthetic_workload(self.table, g.Placement(), topo, shares=SHARES)
        b, _ = g.decode_round_events(wl[0], self.table, g.Placement(), topo, self.hw)
        hardware = dict(self.hw); hardware['resources'] = g.resources_for(self.hw, self.table, topo)
        document = dict(schema='strata.heterogeneous.v1', hardware=hardware, allocations={}, output_tokens=wl[0].accepted,
                        provenance={}, events=[dict(name=e.name, work=e.work, after=list(e.after), slots=e.slots, minimum_ms=e.minimum_ms) for e in b.events])
        direct = g.simulate_decode(wl, self.table, g.Placement(), topo, self.hw)
        via = hetero_evaluate(document)
        self.assertAlmostEqual(via['milliseconds'], direct['ms_per_round'], places=6)

    def test_hit_curve(self):
        few = g.route_shares(g.Placement(primary_slots=5, worker_slots=5), self.table)
        many = g.route_shares(g.Placement(primary_slots=40, worker_slots=60), self.table)
        self.assertGreater(few['cpu'], many['cpu'])
        self.assertAlmostEqual(sum(many.values()), 1.0)
        lfu = g.route_shares(g.Placement(primary_slots=15, worker_slots=33, policy='lfu'), self.table)
        lru = g.route_shares(g.Placement(primary_slots=15, worker_slots=33, policy='lru'), self.table)
        self.assertLess(lfu['cpu'], lru['cpu'])

    def test_two_cuda_devices_with_a_measured_cpu_profile(self):
        # Device IDs describe physical cards, independently of their backend.
        rename = lambda s: re.sub(r'\b(cuda|hip)\b', lambda m: {'cuda':'gpu0', 'hip':'gpu1'}[m[0]], s)
        hw = json.loads(rename(json.dumps(self.hw)))
        kernels = json.loads(rename(json.dumps(self.table)))
        kernels['cpu']['lanes']['all34'] = kernels['cpu']['lanes'].pop('single15')
        topo = g.Topology(primary='gpu0', worker='gpu1', cpu_lanes='all34')
        place = g.Placement(experts_per_layer=288)
        wl = g.synthetic_workload(kernels, place, topo, shares=SHARES)
        result = g.simulate_decode(wl, kernels, place, topo, hw)
        self.assertGreater(result['tok_s'], 0)
        resources = g.resources_for(hw, kernels, topo)
        self.assertIn('gpu0.compute', resources)
        self.assertIn('gpu1.compute', resources)
        self.assertNotIn('cuda.compute', resources)
        self.assertNotIn('hip.compute', resources)

    def test_acceptance_is_finite_positive_and_can_include_lookup(self):
        topo = g.Topology(groups=(1, 2))  # MTP depth 2, floor(window/2) first group.
        for accepted in (0, -1, float('inf'), float('nan')):
            with self.subTest(accepted=accepted), self.assertRaises(ValueError):
                g.synthetic_workload(self.table, g.Placement(), topo, accepted_per_round=accepted)
        r = g.synthetic_workload(self.table, g.Placement(), topo, accepted_per_round=2.6)
        self.assertEqual(r[0].draft_steps, 2)
        self.assertEqual(r[0].accepted, 2.6)
        lookup = g.synthetic_workload(self.table, g.Placement(), topo, accepted_per_round=4.1)
        self.assertEqual(lookup[0].accepted, 4.1)

    def test_invalid_route_shares_are_rejected(self):
        for shares in ({'cpu':-0.1, 'worker':0.5, 'primary':0.6}, {'cpu':0.9, 'worker':0.9}):
            with self.assertRaises(ValueError):
                g.synthetic_workload(self.table, g.Placement(), g.Topology(), shares=shares)

    def test_routes_workload(self):
        events = []
        for rnd in range(2):
            for layer in g.MOE_LAYERS:
                for pos in (0, 2):
                    events.append(dict(request=1, round=rnd, phase='target', event=len(events), position=pos, layer=layer,
                                       routes=[[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 8, 9, 10, 11]], resident=[0, 1]))
            events.append(dict(request=1, round=rnd, phase='draft', event=len(events), position=4, layer=45, routes=[[0, 1, 2, 3, 4, 5, 6, 7]], resident=[]))
        rounds = glm_traces.rounds_from_events(events)
        wl = g.workload_from_routes(rounds, g.Placement(primary_slots=2, worker_slots=2), self.table, g.Topology(kind='A', primary='hip', worker='cuda'), 3.0)
        self.assertEqual(len(wl), 2); self.assertEqual(len(wl[0].slots), 84)
        s = wl[0].slots[0]
        self.assertEqual(s.primary_experts, 2); self.assertEqual(s.worker_experts, 2); self.assertEqual(s.cpu_experts, 8); self.assertEqual(s.width, 2)
        self.assertEqual(wl[0].draft_steps, 1)
        # An observed worker bitmap must override the trace-derived prior.
        for rnd in rounds:
            for event in rnd['target']:
                event['worker_owned'] = [10, 11]
        exact = g.workload_from_routes(rounds, g.Placement(primary_slots=2, worker_slots=2), self.table,
                                      g.Topology(kind='A', primary='hip', worker='cuda'), 3.0)
        self.assertEqual(exact[0].slots[0].worker_experts, 2)
        self.assertEqual(exact[0].slots[0].worker_rows, 2)


class ProfileSelectionTests(unittest.TestCase):
    def test_profile_keeps_mtp2_and_mtp3_costs_separate(self):
        kernels = g.patch(table(), 'supported_verification_groups', [[2, 2]])
        topo = g.Topology(groups=(1, 2))
        place = g.Placement()
        wl = g.synthetic_workload(kernels, place, topo, shares=SHARES)
        with self.assertRaisesRegex(ValueError, 'verification grouping'):
            g.simulate_decode(wl, kernels, place, topo, hardware())

    def test_profile_rejects_an_unmeasured_model_and_group_width(self):
        kernels = table()
        kernels.update(supported_expert_counts=[224], supported_group_widths=[1, 2])
        topo = g.Topology()
        wrong_model = g.Placement(experts_per_layer=288)
        wl = g.synthetic_workload(kernels, wrong_model, topo, shares=SHARES)
        with self.assertRaisesRegex(ValueError, 'model expert count'):
            g.simulate_decode(wl, kernels, wrong_model, topo, hardware())
        wide = g.Topology(groups=(4,))
        place = g.Placement()
        wl = g.synthetic_workload(kernels, place, wide, shares=SHARES)
        with self.assertRaisesRegex(ValueError, 'group width'):
            g.simulate_decode(wl, kernels, place, wide, hardware())

    def test_profile_without_a_hit_curve_requires_recorded_shares(self):
        kernels = g.patch(table(), 'model.require_recorded_shares', True)
        with self.assertRaisesRegex(ValueError, 'recorded route shares'):
            g.synthetic_workload(kernels, g.Placement(), g.Topology())
        self.assertTrue(g.synthetic_workload(kernels, g.Placement(), g.Topology(), shares=SHARES))

    def test_decode_profile_does_not_predict_unmeasured_phases(self):
        kernels=table();kernels['supported_phases']=['decode'];kernels['supported_topologies']=['A']
        with self.assertRaisesRegex(ValueError, 'prefill'):
            g.simulate_prefill(kernels, g.PrefillPlan(), hardware())
        topo=g.Topology(kind='B')
        wl=g.synthetic_workload(kernels, g.Placement(), topo, shares=SHARES)
        with self.assertRaisesRegex(ValueError, 'topology B'):
            g.simulate_decode(wl, kernels, g.Placement(), topo, hardware())

    def test_custom_kernels_do_not_inherit_tr16_fit(self):
        from tools import glm_runner_sim as cli
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder)/'kernels.json'
            path.write_text(json.dumps({'sentinel':'custom measurement'}))
            args = SimpleNamespace(kernels=str(path), params=None, hardware='unused')
            self.assertEqual(cli.load_table(args), {'sentinel':'custom measurement'})

    def test_profile_rejects_a_different_machine(self):
        from tools import glm_runner_sim as cli
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder)/'kernels.json'
            hw = pathlib.Path(folder)/'hardware.json'
            path.write_text(json.dumps({'hardware':'v100s node'}))
            hw.write_text(json.dumps({'name':'tr16-runner'}))
            with self.assertRaisesRegex(ValueError, 'hardware names'):
                cli.load_table(SimpleNamespace(kernels=str(path), params=None, hardware=str(hw)))

    def test_model_expert_count_comes_from_profile(self):
        from tools import glm_runner_sim as cli
        args = SimpleNamespace(primary_slots=15, worker_slots=33, worker='gpu1',
                               experts_per_layer=None, policy='lru', draft_bank='primary')
        self.assertEqual(cli.build_placement(args, {'model':{'experts_per_layer':288}}).experts_per_layer, 288)
        args.experts_per_layer=224
        self.assertEqual(cli.build_placement(args, {'model':{'experts_per_layer':288}}).experts_per_layer, 224)


class PrefillTests(unittest.TestCase):
    def setUp(self):
        self.hw = hardware(); self.table = table()

    def test_f16_halves_exchange_and_partial_removes_acks(self):
        f32 = g.prefill_events(self.table, g.PrefillPlan(primary='hip', worker='cuda', exchange='f32'), self.hw)[0]
        f16 = g.prefill_events(self.table, g.PrefillPlan(primary='hip', worker='cuda', exchange='f16', return_mode='partial'), self.hw)[0]
        out32 = sum(e.work.get('hip.d2h', 0) for e in f32.events if 'exchange-out' in e.name)
        out16 = sum(e.work.get('hip.d2h', 0) for e in f16.events if 'exchange-out' in e.name)
        self.assertAlmostEqual(out16, out32 / 2)
        self.assertGreater(sum(1 for e in f32.events if '/ack' in e.name), 0)
        self.assertEqual(sum(1 for e in f16.events if '/ack' in e.name), 0)

    def test_single_device_and_split_run(self):
        alone = g.simulate_prefill(self.table, g.PrefillPlan(primary='hip', worker=''), self.hw)
        split = g.simulate_prefill(self.table, g.PrefillPlan(primary='hip', worker='cuda', primary_fraction=0.67), self.hw)
        self.assertGreater(alone['tok_s'], 200); self.assertGreater(split['tok_s'], 200)
        self.assertAlmostEqual(alone['mixer_chain_ms'], sum(self.table['devices']['hip']['prefill']['mixer_ms'].values()))

    def test_streaming_hidden_when_mixer_long(self):
        slow_mixer = copy.deepcopy(self.table)
        for k in slow_mixer['devices']['hip']['prefill']['mixer_ms']:
            slow_mixer['devices']['hip']['prefill']['mixer_ms'][k] = 2000.0
        hidden = g.simulate_prefill(slow_mixer, g.PrefillPlan(primary='hip', worker='', stream_overlap='mixer'), self.hw)
        exposed = g.simulate_prefill(slow_mixer, g.PrefillPlan(primary='hip', worker='', stream_overlap='none'), self.hw)
        self.assertLess(hidden['milliseconds'], exposed['milliseconds'])

    def test_tile_padding_makes_2k_worker_cost_sticky(self):
        plan2k = g.PrefillPlan(primary='hip', worker='cuda', tokens=2048)
        plan8k = g.PrefillPlan(primary='hip', worker='cuda', tokens=8192)
        e2 = g.prefill_events(self.table, plan2k, self.hw)[0]; e8 = g.prefill_events(self.table, plan8k, self.hw)[0]
        w2 = sum(e.work.get('cuda.compute', 0) for e in e2.events if '/cuda/gemm' in e.name)
        w8 = sum(e.work.get('cuda.compute', 0) for e in e8.events if '/cuda/gemm' in e.name)
        self.assertAlmostEqual(w2, w8)  # 73 and 293 rows per expert both pad to one 512-row tile

    def test_cpu_groups_use_lane(self):
        b = g.prefill_events(self.table, g.PrefillPlan(primary='hip', worker='', cpu_groups=1.0), self.hw)[0]
        self.assertGreater(sum(e.work.get('cpu.flow0', 0) for e in b.events), 0)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.hw = hardware(); self.table = table()

    def synthetic_records(self, truth):
        records = []
        for name, prim, work, ctx, split in (('a', 'cuda', 'hip', 8192, 'fit'), ('b', 'hip', 'cuda', 8192, 'fit'), ('c', 'hip', '', 2048, 'holdout'),
                                             ('d', 'cuda', '', 8192, 'holdout')):
            rec = dict(name=name, kind='decode', split=split, fit=split == 'fit', measured_tok_s=0.0, shares=SHARES if work else dict(cpu=0.4, worker=0.0, primary=0.6),
                       topology=dict(kind='A', primary=prim, worker=work, groups=[2, 2], context_tokens=ctx),
                       placement=dict(primary_slots=15, worker_slots=33 if work else 0))
            rec['measured_tok_s'] = g.predict_record(rec, truth, self.hw)['tok_s']
            records.append(rec)
        return records

    def test_calibrate_recovers_planted_constants(self):
        truth = g.patch(g.patch(self.table, 'worker.rtt_socket_ms', 0.3), 'cpu.slot_fixed_ms', 0.06)
        records = self.synthetic_records(truth)
        start = g.patch(g.patch(self.table, 'worker.rtt_socket_ms', 0.1), 'cpu.slot_fixed_ms', 0.03)
        fitted, score, history = g.calibrate(records, start, self.hw, ['worker.rtt_socket_ms', 'cpu.slot_fixed_ms'], rounds=4, log=lambda *_: None)
        self.assertLess(score, 0.01)
        for rec in records:
            self.assertLess(abs(g.predict_record(rec, fitted, self.hw)['tok_s'] / rec['measured_tok_s'] - 1), 0.02)

    def test_fit_ignores_holdouts(self):
        records = self.synthetic_records(self.table)
        records[2]['measured_tok_s'] *= 3  # a corrupted holdout must not move the fit
        fitted, score, _ = g.calibrate(records, self.table, self.hw, ['cpu.slot_fixed_ms'], rounds=1, log=lambda *_: None)
        self.assertEqual(g.get(fitted, 'cpu.slot_fixed_ms'), g.get(self.table, 'cpu.slot_fixed_ms'))
        self.assertFalse(g.validate(records, fitted, self.hw)['calibrated'])

    def test_validate_passes_exact_rows_and_honours_gate(self):
        records = self.synthetic_records(self.table)
        self.assertTrue(g.validate(records, self.table, self.hw)['calibrated'])
        records[2]['measured_tok_s'] *= 2; records[2]['gate'] = False
        result = g.validate(records, self.table, self.hw)
        self.assertTrue(result['calibrated']); self.assertEqual(result['ungated'], ['c'])

    def test_context_gate_keeps_excluded_failures_visible(self):
        records=self.synthetic_records(self.table)
        records[2]['measured_tok_s']*=2
        self.assertFalse(g.validate(records,self.table,self.hw)['calibrated'])
        result=g.validate(records,self.table,self.hw,context_tokens=8192)
        self.assertTrue(result['calibrated']);self.assertEqual(result['ungated'],['c'])
        self.assertFalse(result['rows'][2]['within'])
        self.assertEqual(result['holdout_count'],1)
        records[3]['measured_tok_s']*=2
        self.assertFalse(g.validate(records,self.table,self.hw,context_tokens=8192)['calibrated'])

    def test_calibrate_requires_fit_rows(self):
        with self.assertRaises(ValueError):
            g.calibrate([dict(name='x', kind='decode', split='holdout', measured_tok_s=30)], self.table, self.hw, ['cpu.slot_fixed_ms'])


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.hw = hardware(); self.table = table()
        self.levers = json.loads((DATA / 'glm_levers.json').read_text())['levers']
        self.state = dict(table=self.table, topology=g.Topology(kind='A', primary='hip', worker='cuda'),
                          placement=g.Placement(), plan=g.PrefillPlan(primary='hip', worker='cuda'), accepted_per_round=None)

    def test_sweep_monotone_and_lossless_only(self):
        result = g.sweep_levers(self.levers, self.state, self.hw, target_decode=1000, target_prefill=10000, log=lambda *_: None)
        self.assertIn('lfu_partition', result['excluded_lossy']); self.assertIn('three_groups', result['excluded_lossy'])
        decode = [p['decode_tok_s'] for p in result['path']]
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(decode, decode[1:])))
        self.assertTrue(all(p['lossless'] for p in result['path']))
        self.assertFalse(result['reaches']['decode'])

    def test_sweep_stops_at_target(self):
        result = g.sweep_levers(self.levers, self.state, self.hw, target_decode=1.0, target_prefill=1.0, log=lambda *_: None)
        self.assertTrue(result['reaches']['decode'] and result['reaches']['prefill'])
        self.assertLessEqual(len(result['path']), 1)

    def test_lossy_allowed(self):
        result = g.sweep_levers([l for l in self.levers if l['name'] in ('lfu_partition', 'three_groups')], self.state, self.hw,
                                allow_lossy=True, log=lambda *_: None)
        self.assertEqual(result['excluded_lossy'], [])
        # three verify groups add 42 slots for 25% more accepted tokens: the model must not pick it as a gain
        self.assertNotIn('three_groups', [p['lever'] for p in result['path']])

    def test_side_stream_residents_shorten_gpu_chain(self):
        chain = self.state['topology']
        side = g.Topology(kind='A', primary='hip', worker='cuda', residents='side')
        a = g.simulate_decode(g.synthetic_workload(self.table, g.Placement(), chain, shares=SHARES), self.table, g.Placement(), chain, self.hw)
        b = g.simulate_decode(g.synthetic_workload(self.table, g.Placement(), side, shares=SHARES), self.table, g.Placement(), side, self.hw)
        self.assertLess(b['ms_per_round'], a['ms_per_round'])

    def test_apply_lever_targets(self):
        lever = dict(name='x', edits=[dict(target='table', path='cpu.slot_fixed_ms', op='scale', value=2.0), dict(target='topology', path='combine', value='device3'),
                                      dict(target='placement', path='primary_slots', op='add', value=-5.0), dict(target='plan', path='exchange', value='f16'),
                                      dict(target='state', path='accepted_per_round', value=4.3)])
        out = g.apply_lever(self.state, lever)
        self.assertEqual(g.get(out['table'], 'cpu.slot_fixed_ms'), 2 * g.get(self.table, 'cpu.slot_fixed_ms'))
        self.assertEqual(out['topology'].combine, 'device3'); self.assertEqual(out['placement'].primary_slots, 10.0)
        self.assertEqual(out['plan'].exchange, 'f16'); self.assertEqual(out['accepted_per_round'], 4.3)
        with self.assertRaises(ValueError):
            g.apply_lever(self.state, dict(name='bad', edits=[dict(target='nowhere', path='x', value=1)]))


class TraceReaderTests(unittest.TestCase):
    def test_repeated_decode_trials_do_not_merge_rounds(self):
        events=[dict(request=1, round=rnd, phase='target', layer=3, routes=[[1]])
                for rnd in (0, 1, 2, 0, 1, 2)]
        rounds=glm_traces.rounds_from_events(events)
        self.assertEqual(len(rounds), 6)
        self.assertTrue(all(len(r['target'])==1 for r in rounds))

    def test_cache_snapshots_are_not_executed_slots(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d)/'routes.jsonl'
            event = dict(request=1, round=0, phase='target', layer=3, routes=[[0, 1]], resident=[0])
            path.write_text(json.dumps(dict(kind='cache', request=1, layer=3, experts=[0]))+'\n'+json.dumps(event)+'\n')
            self.assertEqual(glm_traces.read_routing_events(path), [event])
            path.write_text(json.dumps(dict(kind='cache', layer=3))+'\n')
            with self.assertRaises(ValueError):
                glm_traces.read_routing_events(path)

    def test_log_and_timeline_readers(self):
        with tempfile.TemporaryDirectory() as d:
            log = pathlib.Path(d) / 'run.log'
            log.write_text('\n'.join([
                'CPU_EXPERT gu_ms=0 quant_ms=0 down_ms=0 layer_flow_ms=0 bytes=0',
                'DECODE_TRIAL index=0',
                'SPECULATIVE source=mtp generated=34 rounds=20 proposed=30 accepted=24 replayed=0 ms=3000 tok_s=11',
                'MTP_TIMING prepare_ms=1 draft_ms=900 verify_ms=2000 resync_ms=50',
                'CPU_EXPERT gu_ms=1 quant_ms=1 down_ms=1 layer_flow_ms=900 bytes=50000000000',
                'STEP_TRACE steps=20 moe_layers=1680 pipelined=1 head_ms=70 cpu_ms=1500 between_ms=400 tail_ms=90 per_step_ms=103',
                'DECODE_TRIAL index=1',
                'STEP_TRACE steps=10 moe_layers=840 pipelined=1 head_ms=35 cpu_ms=550 between_ms=210 tail_ms=45 per_step_ms=84',
                'STEP_TRACE_SLOTS group=0 wait_cpu_us_per_step=3m:3577/1333 4k:112/857',
                'STEP_TRACE_SLOTS group=1 wait_cpu_us_per_step=3m:2105/1006 4k:71/839',
                'SPECULATIVE source=mtp generated=34 rounds=11 proposed=30 accepted=24 replayed=0 ms=1100 tok_s=30.9',
                'MTP_TIMING prepare_ms=1 draft_ms=100 verify_ms=900 resync_ms=20',
                'CPU_EXPERT gu_ms=1 quant_ms=1 down_ms=1 layer_flow_ms=500 bytes=25000000000',
                'MIXED_CLIENT requests=840 decode_rows=6000 prefill_rows=0 blocked_wait_ms=1 decode_wait_ms=100 ownership_wait_ms=50',
            ]) + '\n')
            parsed = glm_traces.read_log(log)
            trial = glm_traces.timed_trial(parsed)
            scopes = glm_traces.round_scopes(trial)
            self.assertAlmostEqual(scopes['cpu_service'], 55.0); self.assertAlmostEqual(scopes['draft'], 10.0)
            self.assertAlmostEqual(scopes['round_ms'], 110.0); self.assertAlmostEqual(scopes['other'], 110 - 84 - 10 - 2)
            self.assertAlmostEqual(scopes['accepted_per_round'], 3.4)
            self.assertEqual(trial['slots'][(3, 1)]['wait_us'], 2105); self.assertEqual(trial['cpu_expert']['bytes'], 25000000000)
            self.assertEqual(parsed['mixed_client']['decode_rows'], 6000)
            tl = pathlib.Path(d) / 'timeline.jsonl'
            rows = [dict(unix_ms=0, phase='prefill_batch_begin', position=0, tokens=8, layer=-1),
                    dict(unix_ms=1, phase='layer_begin', position=0, tokens=8, layer=0),
                    dict(unix_ms=4, phase='mixer_complete', position=0, tokens=8, layer=0),
                    dict(unix_ms=9, phase='ffn_complete', position=0, tokens=8, layer=0),
                    dict(unix_ms=12, phase='prefill_batch_complete', position=0, tokens=8, layer=-1)]
            tl.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
            batches = glm_traces.read_prefill_timeline(tl)
            self.assertEqual(batches[0]['layers'][0], dict(layer=0, mixer_ms=3, ffn_ms=5)); self.assertEqual(batches[0]['unassigned_ms'], 4)


class DataFileTests(unittest.TestCase):
    def test_shipped_files_are_consistent(self):
        t = table(); hw = hardware()
        self.assertEqual(t['schema'], 'strata.glm.kernels.v1'); self.assertEqual(hw['name'], 'tr16-runner')
        for path in t['free']:
            self.assertIsInstance(g.get(t, path), (int, float), path)
        records = json.loads((DATA / 'glm_runner_records_tr16.json').read_text())['records']
        self.assertTrue(any(r['fit'] for r in records))
        for r in records:
            self.assertIn(r['kind'], ('decode', 'prefill'))
            g.predict_record(r, t, hw)
        levers = json.loads((DATA / 'glm_levers.json').read_text())['levers']
        self.assertEqual(len({l['name'] for l in levers}), len(levers))


if __name__ == '__main__':
    unittest.main()
