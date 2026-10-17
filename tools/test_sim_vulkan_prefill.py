"""CPU-only checks for measured Vulkan prefill overlap and forecast limits."""
import copy
import json
import math
from pathlib import Path
import unittest
import tempfile
from sim.vulkan import calibrate_prefill, prefill_forecast


class PrefillForecastTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = json.loads((Path(__file__).parent / 'sim/data/rx570_vulkan_prefill_fp32_20261016.json').read_text())

    def test_reproduces_all_three_measured_trials(self):
        result = prefill_forecast(self.calibration)
        for rate, trial in zip(result['trial_predictions_tok_s'], self.calibration['trials']):
            self.assertAlmostEqual(rate, 1000 / trial['wall_ms_per_token'])
        self.assertAlmostEqual(result['predicted_prefill_tok_s'], 43.6587, places=3)
        self.assertFalse(result['hypothetical'])

    def test_layer_imbalance_is_not_hidden_by_aggregate_max(self):
        # CPU then GPU dominates: 9 + 9, not max(10, 10).
        c = {'kind': 'batched_prefill_fp32', 'trials': [dict(tokens=1,
             wall_ms_per_token=20, serial_gpu_ms_per_token=2,
             layers=[dict(prepare_ms=0, cpu_work_ms=9, upload_ms=0, queued_work_ms=1),
                     dict(prepare_ms=0, cpu_work_ms=1, upload_ms=0, queued_work_ms=9)])]}
        self.assertAlmostEqual(prefill_forecast(c, cpu_speedup=2)['predicted_prefill_tok_s'], 1000/15.5)

    def test_gpu_speedup_cannot_remove_cpu_bound(self):
        self.assertLess(prefill_forecast(self.calibration, gpu_speedup=1e9)['predicted_prefill_tok_s'], 80)
        self.assertLess(prefill_forecast(self.calibration, 2, 2)['predicted_prefill_tok_s'], 75)
        self.assertGreater(prefill_forecast(self.calibration, 3, 3, 3)['predicted_prefill_tok_s'], 100)

    def test_import_excludes_warmup_and_rejects_unqualified_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'run'
            guard = dict(passed=True, performance_qualified=True,
                         executable_sha256='exe', config_sha256='config', prompt_sha256='prompt')
            result = dict(complete=True, clean=True, exit_code=0,
                          measurements={'prefill': []})
            lines = []
            for i in range(-1, 3):
                for layer in range(3, 45):
                    lines.append(f'PREFILL_EXPERT_SPLIT layer={layer} prepare_ms=1 cpu_work_ms={999 if i < 0 else 2} upload_ms=1 queued_work_ms=1 gpu_routes=3')
                if i >= 0:
                    lines.extend([f'PREFILL trial={i} tokens=512 ms=5120',
                                  'VULKAN_FAST_GPU phase=prefill attn_segments_ms=1 head_ms=0.1 shared_segments_ms=0.2',
                                  'VULKAN_FAST_PREFIX phase=prefill ms_per_token=0.5'])
                    result['measurements']['prefill'].append(dict(trial=i, milliseconds=5120))
            prefix.with_suffix('.log').write_text('\n'.join(lines))
            prefix.with_suffix('.guard.json').write_text(json.dumps(guard))
            prefix.with_suffix('.result.json').write_text(json.dumps(result))
            c = calibrate_prefill(prefix)
            self.assertEqual(len(c['trials']), 3)
            self.assertEqual(c['trials'][0]['layers'][0]['cpu_work_ms'], 2)
            self.assertAlmostEqual(c['trials'][0]['serial_gpu_ms_per_token'], 1.6)
            self.assertAlmostEqual(prefill_forecast(c)['predicted_prefill_tok_s'], 100)
            guard['performance_qualified'] = False
            prefix.with_suffix('.guard.json').write_text(json.dumps(guard))
            with self.assertRaises(ValueError):
                calibrate_prefill(prefix)

    def test_invalid_scaling_and_double_counted_phases_rejected(self):
        for bad in (0, -1, math.inf, math.nan):
            for key in ('gpu_speedup', 'cpu_speedup', 'upload_speedup'):
                with self.assertRaises(ValueError):
                    prefill_forecast(self.calibration, **{key: bad})
        c = copy.deepcopy(self.calibration)
        c['trials'][0]['serial_gpu_ms_per_token'] = 100
        with self.assertRaises(ValueError):
            prefill_forecast(c)
        with self.assertRaises(ValueError):
            prefill_forecast({'kind': 'sequential'})

    def test_pipeline_shared_work_is_counted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'pipeline'
            prefix.with_suffix('.guard.json').write_text(json.dumps(dict(
                passed=True, performance_qualified=True, executable_sha256='exe',
                config_sha256='config', prompt_sha256='prompt')))
            measured, lines = [], []
            for trial in range(-1, 3):
                for layer in range(3, 45):
                    lines.append(f'PREFILL_PIPELINE_EXPERT layer={layer} prepare_ms=1 '
                                 'cpu_work_ms=4 gpu_branch_ms=3 shared_submission_ms=2 gpu_routes=100')
                # 84 ms shared + 20 ms serial. Expert max(4, 2+3) is 5 ms.
                lines.append('PREFILL_PIPELINE tokens=4096 main_submission_ms=104')
                if trial >= 0:
                    lines.append(f'PREFILL trial={trial} tokens=4096 ms=300')
                    measured.append(dict(trial=trial, milliseconds=300))
            prefix.with_suffix('.log').write_text('\n'.join(lines))
            prefix.with_suffix('.result.json').write_text(json.dumps(dict(
                complete=True, clean=True, exit_code=0, measurements=dict(prefill=measured))))
            c = calibrate_prefill(prefix)
            self.assertEqual(c['executor'], 'pipeline')
            self.assertAlmostEqual(c['trials'][0]['serial_gpu_ms_per_token'], 20/4096)
            self.assertAlmostEqual(prefill_forecast(c)['predicted_prefill_tok_s'], 4096/0.3)
            self.assertAlmostEqual(prefill_forecast(c, gpu_speedup=2)['predicted_prefill_tok_s'],
                                   4096/0.248)
            with self.assertRaises(ValueError):
                prefill_forecast(c, upload_speedup=2)
            prefix.with_suffix('.log').write_text('\n'.join(lines).replace(' shared_submission_ms=2', ''))
            with self.assertRaises(ValueError):
                calibrate_prefill(prefix)


if __name__ == '__main__':
    unittest.main()
