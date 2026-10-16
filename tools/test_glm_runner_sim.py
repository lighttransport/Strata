"""No GPU/model required: GLM runner model structure, calibration mechanics and lever sweep."""
import copy
import json
import pathlib
import tempfile
import unittest

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


class MtpTests(unittest.TestCase):
    """Acceptance curve, verify-width union, draft cost terms and priming (llama.cpp #29928 study)."""
    def setUp(self):
        self.hw = hardware(); self.table = table()

    def run_decode(self, topo, place=None, accepted=None):
        place = place or g.Placement()
        wl = g.synthetic_workload(self.table, place, topo, rounds=1, accepted_per_round=accepted, shares=SHARES)
        return g.simulate_decode(wl, self.table, place, topo, self.hw)

    def test_default_curve_reproduces_the_reference_constant(self):
        topo = g.Topology(kind='A', primary='hip', worker='cuda')
        self.assertAlmostEqual(g.accepted_per_round_for(self.table, topo, g.Placement()),
                               g.get(self.table, 'model.accepted_per_round'), delta=0.02)

    def test_acceptance_rises_with_depth_and_saturates(self):
        values = [g.accepted_per_round_for(self.table, g.Topology(groups=w), g.Placement())
                  for w in ((1,), (2,), (2, 1), (2, 2), (3, 2), (3, 3))]
        self.assertEqual(values[0], 1.0)
        self.assertTrue(all(b > a for a, b in zip(values, values[1:])))
        gains = [b - a for a, b in zip(values, values[1:])]
        self.assertTrue(all(b < a for a, b in zip(gains, gains[1:])))
        self.assertEqual(g.acceptance_curve(self.table, '0.9,0.5'), [0.9, 0.5])
        self.assertEqual(g.acceptance_curve(self.table, draft_quant='q4_0'),
                         g.get(self.table, 'model.acceptance_profiles.llamacpp_q4_full288.conditional'))
        with self.assertRaises(KeyError):
            g.acceptance_curve(self.table, 'nope')

    def test_explicit_accepted_overrides_the_curve(self):
        topo = g.Topology(kind='A', primary='hip', worker='cuda', groups=(3, 2))
        self.assertAlmostEqual(self.run_decode(topo, accepted=3.2)['accepted_per_round'], 3.2)

    def test_union_keeps_width_two_and_discounts_wider_groups(self):
        unique = g.get(self.table, 'model.unique_experts_per_route')
        self.assertAlmostEqual(g.experts_per_slot(self.table, 2, 224), unique * 2 * g.TOP_K)
        self.assertEqual(g.experts_per_slot(self.table, 1, 224), g.TOP_K)
        self.assertLess(g.experts_per_slot(self.table, 3, 224), 1.5 * g.experts_per_slot(self.table, 2, 224))
        self.assertLess(g.experts_per_slot(self.table, 4, 224), 4 * g.TOP_K)

    def test_other_widths_scale_the_fitted_width_two_entry(self):
        d = dict(kda_ms={'1': 0.36, '2': 0.17, '4': 0.45}, width_ratio=dict(kda_ms={'1': 0.9, '2': 1.0, '4': 1.2}))
        self.assertAlmostEqual(g.per_width(d, 'kda_ms', 2), 0.17)
        self.assertAlmostEqual(g.per_width(d, 'kda_ms', 1), 0.17 * 0.9)
        self.assertAlmostEqual(g.per_width(d, 'kda_ms', 3), 0.17 * 1.1)
        self.assertAlmostEqual(g.per_width(dict(kda_ms=d['kda_ms']), 'kda_ms', 3), (0.17 + 0.45) / 2)
        self.assertAlmostEqual(g.width_value({'1': 1.0, '4': 4.0}, 8), 4.0)

    def test_index_share_saves_more_at_long_context_and_needs_depth(self):
        def draft(ctx, share, groups=(2, 2)):
            return self.run_decode(g.Topology(kind='A', primary='hip', worker='cuda', groups=groups, context_tokens=ctx,
                                              draft_index_share=share))['scopes']['draft']
        short, long_ = draft(1024, False) - draft(1024, True), draft(32768, False) - draft(32768, True)
        self.assertGreater(short, 0); self.assertGreater(long_, 3 * short)
        self.assertAlmostEqual(draft(8192, False, (2,)), draft(8192, True, (2,)))  # one draft step: nothing to share
        self.assertGreater(draft(32768, False), draft(1024, False))

    def test_draft_format_scales_cpu_bytes_and_bank_slots(self):
        topo = g.Topology(kind='A', primary='hip', worker='cuda')
        base = self.run_decode(topo, g.Placement())
        q4 = self.run_decode(topo, g.Placement(draft_quant='q4_0'))
        self.assertGreater(q4['cpu_gb_per_output'] * q4['accepted_per_round'], base['cpu_gb_per_output'] * base['accepted_per_round'])
        bank = g.Placement(draft_bank='primary', draft_bank_slots=5.8, draft_quant='q4_0')
        self.assertAlmostEqual(g.bank_placement(bank).primary_slots, 15 - 5.8 * (g.DRAFT_FORMATS['q4_0'][0] * 288 / 224 - 1))
        self.assertEqual(g.bank_placement(g.Placement(draft_quant='q4_0')).primary_slots, 15)

    def test_priming_modes_only_change_time_to_first_token(self):
        rows = {m: g.simulate_prefill(self.table, g.PrefillPlan(primary='hip', worker='cuda', mtp_prime=m), self.hw)
                for m in ('serial', 'batched', 'crop', 'none')}
        self.assertEqual(len({round(r['tok_s'], 9) for r in rows.values()}), 1)
        self.assertGreater(rows['serial']['mtp_prime_ms'], rows['batched']['mtp_prime_ms'])
        self.assertGreater(rows['batched']['mtp_prime_ms'], rows['crop']['mtp_prime_ms'])
        self.assertEqual(rows['none']['mtp_prime_ms'], 0.0); self.assertAlmostEqual(rows['none']['ttft_tok_s'], rows['none']['tok_s'])

    def test_mtp_sweep_and_levers(self):
        topo = g.Topology(kind='A', primary='hip', worker='cuda')
        result = g.mtp_sweep(self.table, topo, g.Placement(), self.hw, contexts=(8192,), windows=((2, 2), (2, 1)),
                             quants=('base', 'q4_0'), banks=('cpu',), shares=(False,))
        self.assertEqual(len(result['rows']), 4)
        for key, best in result['best'].items():
            self.assertEqual(best['tok_s'], max(r['tok_s'] for r in result['rows'] if f"{r['context']}/{r['acceptance']}" == key))
        levers = {l['name']: l for l in json.loads((DATA / 'glm_levers.json').read_text())['levers']}
        state = dict(table=self.table, topology=topo, placement=g.Placement(), plan=g.PrefillPlan(primary='hip', worker='cuda'),
                     accepted_per_round=None)
        crop = g.sweep_levers([levers['mtp_prime_crop']], state, self.hw, log=lambda *_: None, prefill_metric='ttft_tok_s')
        self.assertEqual([p['lever'] for p in crop['path']], ['mtp_prime_crop'])
        self.assertEqual(g.sweep_levers([levers['mtp_prime_crop']], state, self.hw, log=lambda *_: None)['path'], [])
        share = g.apply_lever(state, levers['mtp_index_share'])
        self.assertTrue(share['topology'].draft_index_share)

    def test_trace_acceptance_by_position(self):
        trial = dict(mtp_position=[dict(index=1, proposed=80, opportunities=80, accepted=60),
                                   dict(index=0, proposed=100, opportunities=100, accepted=90),
                                   dict(index=2, proposed=0, opportunities=0, accepted=0)])
        self.assertEqual(glm_traces.acceptance_by_position(trial), [0.9, 0.75])


class TraceReaderTests(unittest.TestCase):
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


class TwoNodeTests(unittest.TestCase):
    """The worker GPU on a second host (two_node_hardware); the one-host model is untouched."""
    QDR = dict(name='ib-qdr', rtt_us=15.0, gbps=3.2, exposed_us=20.0)
    GBE = dict(name='1gbe', rtt_us=401.0, gbps=0.117, exposed_us=80.0)

    def setUp(self):
        self.hw = hardware(); self.table = table()

    def decode(self, hw, share=0.0):
        topo = g.Topology(kind='A', primary='hip', worker='cuda', worker_cpu_share=share)
        place = g.Placement(primary_slots=15, worker_slots=33)
        wl = g.synthetic_workload(self.table, place, topo, rounds=1, shares=SHARES)
        return g.simulate_decode(wl, self.table, place, topo, hw), wl, topo, place

    def test_base_manifest_is_not_modified(self):
        before = copy.deepcopy(self.hw)
        g.two_node_hardware(self.hw, self.QDR)
        self.assertEqual(before, self.hw)

    def test_free_link_matches_one_host(self):
        local = self.decode(self.hw)[0]['tok_s']
        free = self.decode(g.two_node_hardware(self.hw, dict(name='free', rtt_us=0.0, gbps=1000.0, exposed_us=0.0),
                                               node=dict(h2d_gb_s=7.2, d2h_gb_s=7.1, ram_gb_s=60.0, memcpy_gb_s=5.0)))[0]['tok_s']
        self.assertAlmostEqual(free, local, delta=0.05 * local)

    def test_gbe_slower_than_qdr(self):
        qdr = self.decode(g.two_node_hardware(self.hw, self.QDR))[0]['tok_s']
        gbe = self.decode(g.two_node_hardware(self.hw, self.GBE))[0]['tok_s']
        self.assertLess(gbe, qdr)
        self.assertLess(gbe, 0.85 * qdr)

    def test_one_request_crosses_the_network_per_worker_slot(self):
        hw = g.two_node_hardware(self.hw, self.QDR)
        _, wl, topo, place = self.decode(hw)
        b, _ = g.decode_round_events(wl[0], self.table, place, topo, hw)
        requests = [e for e in b.events if '/request/' in e.name and e.work.get('net.ab')]
        self.assertEqual(len(requests), sum(1 for s in wl[0].slots if s.worker_experts > 0))
        self.assertTrue(all(e.work['net.ab'] == 2 * g.HIDDEN * 4 + 4096 for e in requests))

    def test_node_b_cpu_share_moves_expert_bytes(self):
        hw = g.two_node_hardware(self.hw, self.QDR)
        _, wl, topo, place = self.decode(hw, share=0.5)
        b, _ = g.decode_round_events(wl[0], self.table, place, topo, hw)
        local = sum(e.work.get('cpu.flow0', 0) for e in b.events if '/draft' not in e.name)
        remote = sum(e.work.get('b.cpu.flow', 0) for e in b.events)
        slots = sum(s.cpu_bytes for s in wl[0].slots)
        self.assertAlmostEqual(local + remote, slots, delta=1e-6 * slots)
        self.assertGreater(remote, 0.4 * slots)

    def test_prefill_streams_locally_and_exchanges_over_the_link(self):
        hw = g.two_node_hardware(self.hw, self.QDR)
        plan = g.PrefillPlan(primary='hip', worker='cuda', primary_fraction=0.5, exchange='f16', return_mode='partial')
        b, _ = g.prefill_events(self.table, plan, hw)
        streams = [e for e in b.events if '/cuda/stream' in e.name]
        self.assertTrue(streams)
        self.assertFalse(any(e.work.get('net.ab') or e.work.get('net.ba') for e in streams))
        sent = sum(e.work.get('net.ab', 0) for e in b.events if '/exchange-in/' in e.name)
        self.assertEqual(sent, len(g.MOE_LAYERS) * 8192 * g.HIDDEN * 2)
        returned = sum(e.work.get('net.ba', 0) for e in b.events if '/return/' in e.name)
        self.assertGreater(returned, 0)

    def test_cpu_node_keeps_gpu_paths_local(self):
        hw = g.two_node_hardware(self.hw, self.QDR, gpu_on_b=False)
        self.assertEqual(hw['paths']['ram->cuda'], self.hw['paths']['ram->cuda'])
        local = self.decode(self.hw)[0]['tok_s']
        self.assertAlmostEqual(self.decode(hw)[0]['tok_s'], local, delta=1e-6 * local)   # no remote share: unchanged
        r, wl, topo, place = self.decode(hw, share={l: 0.3 for l in g.MOE_LAYERS})
        b, _ = g.decode_round_events(wl[0], self.table, place, topo, hw)
        self.assertTrue(any(e.work.get('b.cpu.flow') for e in b.events))
        self.assertFalse(any('/worker-input/' in e.name for e in b.events))

    def test_expert_scale_restores_sizes(self):
        before = dict(g.EXPERT_BYTES)
        with g.expert_scale(1.75) as sc:
            self.assertEqual(g.EXPERT_BYTES[3], int(before[3] * 1.75))
            t = sc.table(self.table)
            self.assertAlmostEqual(t['devices']['cuda']['decode']['resident_expert_ms']['2'],
                                   1.75 * self.table['devices']['cuda']['decode']['resident_expert_ms']['2'])
        self.assertEqual(g.EXPERT_BYTES, before)


if __name__ == '__main__':
    unittest.main()
