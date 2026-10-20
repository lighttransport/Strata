"""CPU-only mathematical and regression checks for the uncalibrated FP4 model."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from tools.sim import fp4

class Fp4Model(unittest.TestCase):
    def setUp(self):self.p=json.loads(fp4.PROFILE.read_text())
    def run_model(self,**kw):return fp4.simulate(self.p,**kw)
    def test_actual_packed_expert_geometry(self):
        r=self.run_model(counts=(1,0,0),layers=1)
        self.assertEqual(r['bytes_per_expert']['gpu_packed'],14155788)
        self.assertEqual(r['bytes_per_expert']['gpu_bf16'],50331660)
        self.assertNotIn('tok_s',r)
    def test_one_projection_roofline_units(self):
        n=dict(self.p['nodes'][0],vram_gbps=1,bf16_tflops=1,dequant_gweights_s=1,launch_us=0)
        stages,part=fp4.projection(n,16,16,1,'fused',False,0)
        expected_memory=(256*(.5+1/16)+2*32+4*16)/1e6
        self.assertAlmostEqual(part['gemm_roofline_ms'],8192/1e9)
        self.assertAlmostEqual(stages[0][1],max(expected_memory,8192/1e9+256/1e6))
    def test_tile_redecode_at_seventeen_tokens(self):
        _,a=fp4.projection(self.p['nodes'][0],2048,4096,16,'fused',False,0)
        _,b=fp4.projection(self.p['nodes'][0],2048,4096,17,'fused',False,0)
        self.assertEqual(b['fused_dequant_roofline_ms'],a['fused_dequant_roofline_ms']*2)
    def test_warm_avoids_csf_and_weight_uploads(self):
        warm=self.run_model(cache='warm');cold=self.run_model(cache='cold')
        self.assertLess(warm['moe_milliseconds'],cold['moe_milliseconds'])
        self.assertFalse(any('weights-h2d' in x['name'] or x['category']=='csf_expand' for x in warm['schedule']['events']))
    def test_dequant_rate_changes_fused_kernel(self):
        a=self.run_model(cache='warm',counts=(1,0,0))
        self.p['nodes'][0]['dequant_gweights_s']/=10
        self.assertGreater(self.run_model(cache='warm',counts=(1,0,0))['moe_milliseconds'],a['moe_milliseconds'])
    def test_cached_bf16_warm_has_no_dequant_stage(self):
        r=self.run_model(cache='warm',mode='cached-bf16')
        self.assertFalse(any(e['category'] in ('gpu_dequant','gpu_fused','csf_expand') for e in r['schedule']['events']))
    def test_staged_materializes_dequant_even_when_warm(self):
        r=self.run_model(cache='warm',mode='staged',counts=(1,0,0),layers=1)
        self.assertEqual(sum(e['category']=='gpu_dequant' for e in r['schedule']['events']),3)
    def test_direct_has_one_hop_to_leaf(self):
        direct=self.run_model(counts=(0,0,1),layers=1,cache='warm')
        relay=self.run_model(counts=(0,0,1),layers=1,cache='warm',topology='relay')
        self.assertEqual(sum(e['category']=='network' for e in direct['schedule']['events']),2)
        self.assertEqual(sum(e['category']=='network' for e in relay['schedule']['events']),4)
        self.assertGreater(relay['moe_milliseconds'],direct['moe_milliseconds'])
    def test_hca_x4_bottleneck_is_configurable(self):
        a=self.run_model(counts=(0,0,1),tokens=16,cache='warm')
        self.p['nodes'][2]['hca_pcie_gbps']=.01
        self.assertGreater(self.run_model(counts=(0,0,1),tokens=16,cache='warm')['moe_milliseconds'],a['moe_milliseconds'])
    def test_host_intermediate_transfers(self):
        a=self.run_model(counts=(1,0,0),layers=1,cache='warm',tokens=16)
        b=self.run_model(counts=(1,0,0),layers=1,cache='warm',tokens=16,activation_path='device')
        self.assertEqual(a['work_totals']['n0.gpu_copy'],4*16*(3*4096+3*2048))
        self.assertEqual(b['work_totals']['n0.gpu_copy'],4*16*2*4096)
    def test_more_layers_charge_more_work(self):
        a=self.run_model(counts=(1,0,0),layers=1)
        b=self.run_model(counts=(1,0,0),layers=2)
        self.assertAlmostEqual(b['moe_milliseconds'],a['moe_milliseconds']*2)
    def test_csf_ratio_affects_host_not_gpu_storage(self):
        a=self.run_model();self.p['csf_bytes_per_scale']=1
        b=self.run_model()
        self.assertGreater(b['bytes_per_expert']['host_csf'],a['bytes_per_expert']['host_csf'])
        self.assertEqual(b['bytes_per_expert']['gpu_packed'],a['bytes_per_expert']['gpu_packed'])
    def test_warm_working_set_must_fit(self):
        self.p['nodes'][0]['cache_gib']=.02
        with self.assertRaisesRegex(ValueError,'warm working set'):self.run_model(cache='warm')
    def test_mixed_endpoints_match_cold_warm(self):
        for rate,cache in ((0,'cold'),(1,'warm')):
            a=self.run_model(cache=cache)
            b=self.run_model(cache='mixed',hit_rates=(rate,)*3)
            self.assertAlmostEqual(a['moe_milliseconds'],b['moe_milliseconds'])
            self.assertEqual(a['work_totals'],b['work_totals'])
    def test_mixed_counts_real_events_and_bf16_dequant(self):
        r=self.run_model(counts=(2,0,0),layers=2,cache='mixed',hit_rates=(.5,0,0),mode='cached-bf16')
        self.assertEqual(r['residency'][0]['hits'],2)
        self.assertEqual(r['residency'][0]['misses'],2)
        self.assertEqual(sum(e['category']=='gpu_dequant' for e in r['schedule']['events']),6)
        self.assertEqual(sum('weights-h2d' in e['name'] for e in r['schedule']['events']),2)
    def test_preexpanded_omits_csf_and_charges_host_storage(self):
        a=self.run_model();b=self.run_model(csf_preexpanded=True)
        self.assertNotIn('n0.csf',b['work_totals'])
        self.assertEqual(a['work_totals']['n0.gpu_copy'],b['work_totals']['n0.gpu_copy'])
        self.assertGreater(b['residency'][0]['selected_host_bytes'],a['residency'][0]['selected_host_bytes'])
        self.assertLess(b['moe_milliseconds'],a['moe_milliseconds'])
    def test_prefetch_overlaps_but_respects_layers_and_buffers(self):
        kw=dict(counts=(3,0,0),layers=2,csf_preexpanded=True)
        a=self.run_model(**kw);b=self.run_model(**kw,transfer_overlap=True)
        self.assertEqual(a['work_totals'],b['work_totals'])
        self.assertLess(b['moe_milliseconds'],a['moe_milliseconds'])
        e={x['name']:x for x in b['schedule']['events']}
        self.assertLess(e['L0/N0/E1/weights-h2d']['start_ms'],e['L0/N0/E0/result-d2h']['end_ms'])
        self.assertGreaterEqual(e['L0/N0/E2/weights-h2d']['start_ms'],e['L0/N0/E0/result-d2h']['end_ms'])
        self.assertGreaterEqual(e['L1/N0/E0/weights-h2d']['start_ms'],e['L0/join']['end_ms'])
        self.assertEqual(b['residency'][0]['streaming_buffers'],2)
    def test_prefetch_buffer_and_hit_set_capacity(self):
        self.p['nodes'][0]['cache_gib']=.02
        self.run_model(counts=(2,0,0),layers=1)
        with self.assertRaises(ValueError):self.run_model(counts=(2,0,0),layers=1,transfer_overlap=True)
        with self.assertRaises(ValueError):self.run_model(counts=(2,0,0),layers=2,cache='mixed',hit_rates=(.5,0,0))
    def test_bad_hit_rates(self):
        for rates in (None,(.5,),(-.1,0,0),(1.1,0,0),(float('nan'),0,0),(True,0,0)):
            with self.assertRaises(ValueError):self.run_model(cache='mixed',hit_rates=rates)
        with self.assertRaises(ValueError):self.run_model(hit_rates=(0,0,0))

    def cpu_rates(self,weight=50,quant=1,availability=1):
        return [dict(gweights_s=weight,quant_gitems_s=quant,availability=availability) for _ in range(3)]

    def test_cpu_only_exact_compute_and_quantization_cost(self):
        self.p['nodes'][0]['dram_gbps']=1e9
        self.p['nodes'][0]['host_activation_gitems_s']=1
        r=self.run_model(counts=(1,0,0),layers=1,miss_policy='cpu',cpu_rates=self.cpu_rates(weight=1),csf_preexpanded=True)
        expected=(3*4096*2048+4096+2048+2048)/1e6
        self.assertAlmostEqual(r['moe_milliseconds'],expected)
        self.assertEqual(r['work_totals']['n0.cpu_kernel'],3*4096*2048)
        self.assertEqual(r['work_totals']['n0.cpu_quant'],4096+2048)
        self.assertNotIn('n0.gpu_copy',r['work_totals'])
        self.assertNotIn('n0.gpu',r['work_totals'])
        self.assertEqual(r['residency'][0]['required_gpu_bytes'],0)
        self.assertEqual(r['residency'][0]['streaming_buffers'],0)
        self.assertEqual(len(r['projection_components']),3)
        self.assertTrue(all(x['device']=='cpu' for x in r['projection_components']))

    def test_cpu_shared_input_is_per_node_and_layer(self):
        kw=dict(counts=(2,1,0),layers=2,tokens=2,miss_policy='cpu',
                cpu_rates=self.cpu_rates(),csf_preexpanded=True)
        baseline=self.run_model(**kw)
        shared=self.run_model(**kw,cpu_shared_input=True)
        self.assertEqual(shared['work_totals']['n0.cpu_quant'],2*2*(4096+2*2048))
        self.assertEqual(shared['work_totals']['n1.cpu_quant'],2*2*(4096+2048))
        self.assertEqual(baseline['work_totals']['n0.cpu_quant']-
                         shared['work_totals']['n0.cpu_quant'],2*2*4096)
        self.assertEqual(baseline['projection_components'],shared['projection_components'])
        self.assertEqual(baseline['residency'],shared['residency'])
        self.assertLessEqual(shared['moe_milliseconds'],baseline['moe_milliseconds'])
        events={e['name']:e for e in shared['schedule']['events']}
        self.assertNotIn('L0/N0/E1/input-q8',events)
        self.assertIn('L1/N0/E0/input-q8',events)
        self.assertGreaterEqual(events['L0/N0/E1/P0/cpu-fp4']['start_ms'],
                                events['L0/N0/E0/input-q8']['end_ms'])
        self.assertEqual(baseline,self.run_model(**kw,cpu_shared_input=False))

    def test_cpu_shared_input_requires_cpu_and_handles_gpu_hits(self):
        with self.assertRaises(ValueError):
            self.run_model(cpu_shared_input=True)
        with self.assertRaises(ValueError):
            self.run_model(miss_policy='cpu',cpu_rates=self.cpu_rates(),cpu_shared_input=1)
        kw=dict(counts=(3,0,0),layers=1,cache='mixed',hit_rates=(1/3,0,0),
                miss_policy='cpu',cpu_rates=self.cpu_rates(),cpu_shared_input=True)
        r=self.run_model(**kw)
        self.assertEqual(r['work_totals']['n0.cpu_quant'],4096+2*2048)
        warm=self.run_model(counts=(2,0,0),layers=1,cache='warm',
                            miss_policy='cpu',cpu_rates=self.cpu_rates(),cpu_shared_input=True)
        self.assertNotIn('n0.cpu_quant',warm['work_totals'])

    def test_cpu_csf_expansion_and_preexpanded_ram(self):
        kw=dict(counts=(1,0,0),layers=1,miss_policy='cpu',cpu_rates=self.cpu_rates())
        a=self.run_model(**kw);b=self.run_model(**kw,csf_preexpanded=True)
        self.assertIn('n0.csf',a['work_totals']);self.assertNotIn('n0.csf',b['work_totals'])
        self.assertGreater(a['moe_milliseconds'],b['moe_milliseconds'])
        self.assertGreater(b['residency'][0]['selected_host_bytes'],a['residency'][0]['selected_host_bytes'])
        self.assertGreater(a['residency'][0]['cpu_scratch_bytes'],b['residency'][0]['cpu_scratch_bytes'])

    def test_cpu_gpu_overlap_and_layer_barriers(self):
        kw=dict(counts=(2,0,0),layers=2,cache='mixed',hit_rates=(.5,0,0),
                miss_policy='cpu',cpu_rates=self.cpu_rates(),csf_preexpanded=True,activation_path='device')
        serial=self.run_model(**kw,cpu_gpu_overlap=False);parallel=self.run_model(**kw)
        self.assertEqual(serial['work_totals'],parallel['work_totals'])
        self.assertLess(parallel['moe_milliseconds'],serial['moe_milliseconds'])
        events={e['name']:e for e in parallel['schedule']['events']}
        self.assertLess(events['L0/N0/E1/P0/cpu-fp4']['start_ms'],events['L0/N0/E0/result-d2h']['end_ms'])
        self.assertGreaterEqual(events['L1/N0/E1/ready']['start_ms'],events['L0/join']['end_ms'])
        self.assertFalse(any('weights-h2d' in name for name in events))
        self.assertEqual(parallel['residency'][0]['hits'],2)
        self.assertEqual(parallel['residency'][0]['cpu_experts'],2)

    def test_cpu_warm_remains_gpu_work(self):
        kw=dict(counts=(1,1,1),layers=1,cache='warm')
        a=self.run_model(**kw);b=self.run_model(**kw,miss_policy='cpu',cpu_rates=self.cpu_rates())
        self.assertEqual(a['work_totals'],b['work_totals'])
        self.assertEqual(a['moe_milliseconds'],b['moe_milliseconds'])
        self.assertTrue(all(n['cpu_experts']==0 for n in b['residency']))

    def test_cpu_pool_slot_serializes_host_gpu_activation(self):
        r=self.run_model(counts=(2,0,0),layers=1,cache='mixed',hit_rates=(.5,0,0),
                         miss_policy='cpu',cpu_rates=self.cpu_rates(),csf_preexpanded=True)
        events=[e for e in r['schedule']['events'] if '/cpu-fp4' in e['name'] or '/silu-host' in e['name'] or e['category']=='cpu_quant']
        for i,a in enumerate(events):
            for b in events[i+1:]:
                self.assertTrue(a['end_ms']<=b['start_ms']+1e-9 or b['end_ms']<=a['start_ms']+1e-9)

    def test_cpu_ram_includes_scratch(self):
        r=self.run_model(counts=(1,0,0),layers=1,miss_policy='cpu',cpu_rates=self.cpu_rates(),csf_preexpanded=True)
        self.p['nodes'][0]['ram_budget_gib']=(r['residency'][0]['selected_host_bytes']+1)/fp4.GIB
        with self.assertRaisesRegex(ValueError,'RAM budget'):
            self.run_model(counts=(1,0,0),layers=1,miss_policy='cpu',cpu_rates=self.cpu_rates(),csf_preexpanded=True)

    def test_cpu_availability_and_shared_dram(self):
        kw=dict(counts=(1,0,0),layers=1,miss_policy='cpu',csf_preexpanded=True)
        a=self.run_model(**kw,cpu_rates=self.cpu_rates())
        b=self.run_model(**kw,cpu_rates=self.cpu_rates(availability=.25))
        self.assertGreater(b['moe_milliseconds'],a['moe_milliseconds'])
        self.p['nodes'][0]['dram_gbps']=.01
        c=self.run_model(**kw,cpu_rates=self.cpu_rates())
        self.assertGreater(c['moe_milliseconds'],b['moe_milliseconds'])

    def test_cpu_remote_direct_and_relay(self):
        kw=dict(counts=(0,0,1),layers=1,miss_policy='cpu',cpu_rates=self.cpu_rates(),csf_preexpanded=True)
        a=self.run_model(**kw);b=self.run_model(**kw,topology='relay')
        self.assertEqual(sum(e['category']=='network' for e in a['schedule']['events']),2)
        self.assertEqual(sum(e['category']=='network' for e in b['schedule']['events']),4)
        self.assertGreater(b['moe_milliseconds'],a['moe_milliseconds'])

    def test_cpu_assumptions_are_required_and_copied(self):
        for rates in (None,[],[{}]*3,self.cpu_rates(weight=0),self.cpu_rates(quant=float('nan')),self.cpu_rates(availability=2)):
            with self.assertRaises(ValueError):self.run_model(miss_policy='cpu',cpu_rates=rates)
        with self.assertRaises(ValueError):self.run_model(cpu_rates=self.cpu_rates())
        with self.assertRaises(ValueError):self.run_model(miss_policy='cpu',cpu_rates=self.cpu_rates(),tokens=9)
        rates=self.cpu_rates();r=self.run_model(miss_policy='cpu',cpu_rates=rates)
        r['scenario']['cpu_rates'][0]['gweights_s']=999
        self.assertEqual(rates[0]['gweights_s'],50)
        self.assertTrue(any('lossy Q8' in x for x in r['limitations']))

    def test_cpu_cli(self):
        cmd=[sys.executable,str(Path(__file__).parent/'sim/glm_sim.py'),'fp4','--experts','1,0,0','--layers','1','--cpu-misses','--cpu-gweights-s','50,40,40','--cpu-quant-gitems-s','1,1,1','--cpu-no-overlap','--csf-preexpanded']
        result=json.loads(subprocess.run(cmd,check=True,capture_output=True,text=True).stdout)
        self.assertEqual(result['scenario']['miss_policy'],'cpu')
        self.assertFalse(result['scenario']['cpu_gpu_overlap'])
        self.assertEqual(result['scenario']['cpu_rates'][1]['gweights_s'],40)
        bad=subprocess.run(cmd[:cmd.index('--cpu-gweights-s')],capture_output=True,text=True)
        self.assertNotEqual(bad.returncode,0)

    def test_invalid_inputs(self):
        for kw in ({'tokens':0},{'layers':-1},{'counts':(0,0,0)},{'counts':(-1,2,1)}, {'counts':(1.5,1,1)},{'counts':(3,3,3)},{'layers':43},{'mode':'native-fp4'}):
            with self.assertRaises(ValueError):self.run_model(**kw)
        for value in (0,-1,float('nan'),float('inf')):
            self.p['nodes'][0]['dequant_gweights_s']=value
            with self.assertRaises(ValueError):self.run_model()
    def test_result_carries_assumptions_and_no_profile_mutation(self):
        original=copy.deepcopy(self.p);r=self.run_model()
        self.assertEqual(original,self.p);self.assertIn('UNCALIBRATED',r['basis'])
        r['assumptions']['nodes'][0]['vram_gbps']=1;self.assertEqual(original,self.p)
    def test_cli(self):
        cmd=[sys.executable,str(Path(__file__).parent/'sim/glm_sim.py'),'fp4','--experts','1,1,1','--cache','warm','--assume','nodes.2.dequant_gweights_s=20','--csf-preexpanded','--transfer-overlap']
        r=subprocess.run(cmd,check=True,capture_output=True,text=True)
        result=json.loads(r.stdout)
        self.assertEqual(result['assumptions']['nodes'][2]['dequant_gweights_s'],20)
        self.assertTrue(result['scenario']['transfer_overlap'])
        self.assertTrue(result['scenario']['csf_preexpanded'])

    def test_wrong_model_cli_rejected(self):
        cmd=[sys.executable,str(Path(__file__).parent/'sim/glm_sim.py'),'--model','glm53','fp4']
        r=subprocess.run(cmd,capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertIn('GLM-5.3-Flash only',r.stderr)

if __name__=='__main__':unittest.main()
