"""CPU tests for RX570 qualification accounting and simulator failure boundaries."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import glm_guarded_run as guard
from glm_vulkan_qualify import compare_ids, profile_config
from sim.vulkan import calibrate, estimate, median
import glm_vulkan_quality as quality

class VulkanOptimizationTest(unittest.TestCase):
    def test_calibration_preserves_scientific_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'summary.json').write_text(json.dumps({'rows':[{'qualified':True,'profile':'strict',
                'executable_sha256':'exe','config_sha256':'cfg'}],'prompt_sha256':'prompt'}))
            (root/'strict-run.log').write_text('PREFILL tok_s=2e-3\nDECODE tok_s=4e-3\nDECODE tok_s=5e-3\n')
            microbench=root/'microbench.jsonl';microbench.write_text('')
            result=calibrate(root,microbench)['profiles']['strict']
            self.assertEqual(result['prefill_step_ms'],500000)
            self.assertEqual(result['warm_step_ms'],200000)

    def test_recheck_only_repairs_completion_parser_with_frozen_safe_evidence(self):
        for unsafe in (None,'oom','fault','swap','tamper'):
            with self.subTest(unsafe=unsafe),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);prefix=root/'run';exe=root/'exe';exe.write_bytes(b'frozen')
                config=root/'cfg.json';config.write_text(json.dumps({'exe':str(exe),'backend':'vulkan','stop_ids':[9]}))
                prompt=root/'prompt';prompt.write_text('1,2')
                prefix.with_suffix('.stdout').write_text('3\n9\n')
                prefix.with_suffix('.vulkan.json').write_text(json.dumps({'output_ids':[[3,9]]}))
                prefix.with_suffix('.kernel.log').write_text('')
                prefix.with_suffix('.log').write_text('PREFILL trial=0 tokens=2 batch=1 ms=1.5e+06 tok_s=1e-3\nDECODE steps=1 ms=20 tok_s=50\n')
                guard={'passed':False,'exit_code':2,'test_kind':'correctness','termination_reason':None,
                       'faults':['VM fault'] if unsafe=='fault' else [],'pci_before':{'GPU':'abc'},'pci_after':{'GPU':'abc'},
                       'config_sha256':quality.digest(config),'prompt_sha256':quality.digest(prompt),
                       'executable_sha256':quality.digest(exe),'command':['--tokens','3','--trials','1']}
                result={'exit_code':0,'complete':False,'rejected':None,'measurements':{'prefill':[]},
                        'output_sha256':quality.digest(prefix.with_suffix('.stdout')),
                        'cgroup_after':{'memory.swap.current':1 if unsafe=='swap' else 0,
                                        'memory.events':{'oom_kill':1 if unsafe=='oom' else 0}}}
                prefix.with_suffix('.guard.json').write_text(json.dumps(guard))
                prefix.with_suffix('.result.json').write_text(json.dumps(result))
                if unsafe=='tamper':prefix.with_suffix('.stdout').write_text('4\n9\n')
                if unsafe:
                    with self.assertRaises(ValueError):quality.recheck_scientific_timing(prefix,config,prompt)
                    self.assertFalse(prefix.with_suffix('.rechecked.guard.json').exists())
                else:
                    recovered=quality.recheck_scientific_timing(prefix,config,prompt)
                    self.assertFalse(recovered['performance_qualified'])
                    self.assertFalse(json.loads(prefix.with_suffix('.guard.json').read_text())['passed'])
                    self.assertEqual(quality.load_completed(prefix)[0],[3,9])
                    prefix.with_suffix('.stdout').write_text('4\n9\n')
                    with self.assertRaisesRegex(ValueError,'evidence changed'):quality.load_completed(prefix)

    def test_strict_control_resets_candidate_features(self):
        cfg={'prefill_batch':32,'prefill_experts':'gpu','decode_cache_mib':4096,
             'speculative':'mtp','draft_depth':3,'verify_batch':True,
             'env':{'STRATA_GLM_MAPPED_LAZY':'1','STRATA_VULKAN_DEVICE_STATE':'1',
                    'GGML_VK_DISABLE_INTEGER_DOT_PRODUCT':'1',
                    'GGML_VK_ALLOW_GRAPHICS_QUEUE':'1'}}
        strict=profile_config(cfg,'strict')
        self.assertEqual(strict['prefill_batch'],1)
        self.assertEqual(strict['prefill_experts'],'cpu')
        self.assertEqual(strict['decode_cache_mib'],0)
        self.assertEqual(strict['speculative'],'none')
        self.assertNotIn('verify_batch',strict)
        self.assertEqual(strict['env'],{'STRATA_GLM_MAPPED_LAZY':'1','GGML_VK_DISABLE_INTEGER_DOT_PRODUCT':'1'})
        self.assertEqual(profile_config(cfg,'configured'),cfg)

    def test_frozen_inputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/'manifest.json';manifest.write_text('frozen')
            with self.assertRaises(FileExistsError):quality.freeze(root/'missing-config',root,root/'dataset')
            self.assertEqual(manifest.read_text(),'frozen')

    def test_partial_quality_campaign_is_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);inputs=root/'inputs';inputs.mkdir();output=root/'output'
            exe=root/'exe';exe.write_bytes(b'frozen executable')
            config=root/'config.json';config.write_text(json.dumps({'exe':str(exe),'model':'model','context':64}))
            tasks=[]
            for name in ('first','second'):
                prompt=inputs/(name+'.ids');prompt.write_text('1,2')
                tasks.append({'name':name,'split':'training','prompt':prompt.name,
                              'sha256':quality.digest(prompt),'prompt_tokens':2,'generate':8})
            (inputs/'manifest.json').write_text(json.dumps({'tasks':tasks,'stop_ids':[9]}))
            def run(config,prompt,prefix,**kwargs):
                if prefix.name=='second-control':raise RuntimeError('interrupted')
                guard={'passed':True,'test_kind':'correctness','executable_sha256':quality.digest(exe)}
                prefix.with_suffix('.guard.json').write_text(json.dumps(guard))
                prefix.with_suffix('.vulkan.json').write_text(json.dumps({'output_ids':[[3,9]]}))
                return guard
            with patch.object(quality,'run',side_effect=run),patch('builtins.print'):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    quality.evaluate(config,config,inputs,output,['training'],None,10)
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['completed_cases'],1)
            self.assertEqual(summary['requested_cases'],2)
            self.assertFalse(summary['all_requested_passed'])
            self.assertFalse(summary['timings_qualified'])
            with patch.object(quality,'run',side_effect=run) as launched,patch('builtins.print'):
                quality.evaluate(config,config,inputs,root/'reuse',['training'],['first'],10,output)
                self.assertEqual(launched.call_count,1)
            exe.write_bytes(b'changed executable')
            with self.assertRaisesRegex(ValueError,'unknown or unselected'):
                quality.evaluate(config,config,inputs,root/'unknown',['training'],['first','missing'],10)
            with patch.object(quality,'run') as launched:
                with self.assertRaisesRegex(ValueError,'executable changed'):
                    quality.evaluate(config,config,inputs,root/'changed',['training'],['first'],10,output)
                launched.assert_not_called()

    def test_output_gate_requires_complete_nonempty_reference(self):
        self.assertTrue(compare_ids([1,2],[[1,2],[1,2]]))
        for candidate in ([],[[1]],[[1,3]],[[1,2],[1,3]]):
            self.assertFalse(compare_ids([1,2],candidate))
        self.assertFalse(compare_ids([], [[]]))

    def test_correctness_mode_does_not_qualify_competing_timings(self):
        for correctness_only,swap,passed in ((False,0,False),(True,0,True),(True,4096,False)):
            with self.subTest(correctness_only=correctness_only,swap=swap),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);exe=root/'exe';exe.write_bytes(b'fake')
                cfg=root/'cfg.json';cfg.write_text(json.dumps({'exe':str(exe)}))
                prompt=root/'prompt';prompt.write_text('1,2');output=root/'run'
                process=SimpleNamespace(pid=123,poll=lambda:0)
                def launch(*args,**kwargs):
                    output.with_suffix('.result.json').write_text(json.dumps({
                        'clean':False,'complete':True,'rejected':None,
                        'cgroup_after':{'memory.swap.current':swap,'memory.events':{'oom_kill':0}},
                        'interference':[{'name':'cc1plus'}]}))
                    return process
                with patch.object(guard,'gpu_devices',return_value=[Path('GPU')]), \
                     patch.object(guard,'identities',return_value={'GPU':'0210df67'}), \
                     patch.object(guard,'kernel_log',side_effect=['','']), \
                     patch('glm_low_memory_bench.clear_model_cache',return_value=[]), \
                     patch.object(guard.subprocess,'Popen',side_effect=launch):
                    result=guard.run(cfg,prompt,output,tokens=2,timeout=10,correctness_only=correctness_only)
                self.assertEqual(result['passed'],passed)
                self.assertFalse(result['performance_qualified'])

    def test_simulator_rejects_unsupported_or_invalid_forecasts(self):
        calibration={'profiles':{'strict':{'warm_step_ms':250,'prefill_step_ms':260,'phases':{}}}}
        result=estimate(calibration,'strict',8192,512)
        self.assertEqual(result['measured_warm_decode_tok_s'],4)
        self.assertFalse(result['hypothetical'])
        self.assertFalse(result['long_context_qualified'])
        with self.assertRaises(ValueError):estimate(calibration,'missing',32,64)
        with self.assertRaises(ValueError):estimate(calibration,'strict',32,64,cache_hit=.5)
        with self.assertRaises(ValueError):estimate(calibration,'strict',32,64,accepted_tokens=0)
        with self.assertRaises(ValueError):estimate(calibration,'strict',32,64,accepted_tokens=5)
        for values in ([],[0],[float('nan')],[float('inf')]):
            with self.assertRaises(ValueError):median(values)

if __name__=='__main__':unittest.main()
