import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from glm_low_memory_bench import vulkan_command, read_gpu, benchmark_environment

class VulkanBenchTest(unittest.TestCase):
    def test_vulkan_control_does_not_inherit_host_experiments(self):
        with patch.dict('os.environ',{'STRATA_VULKAN_DEVICE_STATE':'1',
                                      'GGML_VK_ALLOW_GRAPHICS_QUEUE':'1','PATH':'keep'},clear=True):
            env=benchmark_environment({'backend':'vulkan','env':{'STRATA_VULKAN_FUSED_KDA':'1'}})
        self.assertEqual(env,{'PATH':'keep','STRATA_VULKAN_FUSED_KDA':'1'})

    def setUp(self):
        self.cfg = dict(exe='build-vulkan/vulkan/strata-glm-vulkan', model='model.gguf',
                        context=4096, gpu_budget_mib=15360, prefill_batch=1, backend='vulkan')
        self.args = SimpleNamespace(single=False, eval_corpus=None, prompt=Path('prompt.ids'),
                                    output=Path('result'), tokens=32, trials=3)

    def test_command_records_fixed_geometry(self):
        cmd = vulkan_command(self.cfg, self.args)
        self.assertIn('--gpu-budget-mib', cmd)
        self.assertEqual(cmd[cmd.index('--trials')+1], '3')
        self.assertNotIn('--decode-cache-mib', cmd)

    def test_unsupported_features_fail_before_launch(self):
        for key, value in [('speculative','external'), ('decode_cache_slots',1),
                           ('decode_graphs',True), ('prefill_batch',2048)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                vulkan_command(self.cfg | {key:value}, self.args)
        with self.assertRaises(ValueError):
            vulkan_command(self.cfg | {'env':{'STRATA_GLM_STEP_PIPELINE':'1'}}, self.args)

    def test_batch_and_cache_are_explicit(self):
        cmd=vulkan_command(self.cfg | {'prefill_batch':128,'decode_cache_mib':4096},self.args)
        self.assertEqual(cmd[cmd.index('--prefill-batch')+1],'128')
        self.assertEqual(cmd[cmd.index('--decode-cache-mib')+1],'4096')
        for key,value in [('prefill_batch',0),('prefill_batch',True),('prefill_batch',1025),
                          ('decode_cache_mib',-1),('decode_cache_mib','auto')]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                vulkan_command(self.cfg | {key:value},self.args)

    def test_speculation_is_explicit(self):
        cmd=vulkan_command(self.cfg | {'speculative':'mtp','draft_depth':2,'verify_batch':True},self.args)
        self.assertEqual(cmd[cmd.index('--speculative')+1],'mtp')
        self.assertEqual(cmd[cmd.index('--draft-depth')+1],'2')
        self.assertIn('--verify-batch',cmd)
        with self.assertRaises(ValueError):
            vulkan_command(self.cfg | {'speculative':'lookup','draft_depth':4},self.args)

    def test_stop_ids_reach_runner(self):
        cmd = vulkan_command(self.cfg | {'stop_ids':[2,3]}, self.args)
        self.assertEqual(cmd[cmd.index('--stop-ids')+1], '2,3')

    def test_vulkan_uses_amd_global_accounting(self):
        with patch('glm_low_memory_bench.hip_gpu_device', return_value=Path('/fake')), \
             patch('glm_low_memory_bench.hip_gpu_snapshot', return_value={'used_mib':123}):
            used, snapshot = read_gpu(self.cfg, 42)
            self.assertIsNone(used)
            self.assertEqual(snapshot['used_mib'],123)

if __name__=='__main__':unittest.main()
