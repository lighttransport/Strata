"""Native FP4 resource accounting; no model, GPU or fitting required."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.sim import glm_fp4 as sim


class NativeFp4Tests(unittest.TestCase):
    def setUp(self):
        self.profile = sim.load_profile()

    def test_units_and_independent_resource_overlap(self):
        result = sim.resource_roof({'a': 8e9, 'b': 6e9}, {'a': 2e9, 'b': 3e9}, 80)
        self.assertEqual(result['seconds'], 4)  # overlap, rather than 4 + 2
        self.assertEqual(result['tok_s'], 20)
        self.assertEqual(result['limiting_resource'], 'a')

    def test_duplicate_reports_are_not_extra_gpu_traffic(self):
        before = [{'device': 0, 'uploaded_bytes': 100, 'misses': 2}]
        after = [{'device': 0, 'uploaded_bytes': 300, 'misses': 5}] * 2
        self.assertEqual(sim.counter_delta(before, after),
                         [{'device': 0, 'uploaded_bytes': 200, 'misses': 3}])
        with self.assertRaises(ValueError):
            sim.counter_delta(after, before)
        with self.assertRaises(ValueError):
            sim.counter_delta(before, after + [{'device': 0, 'uploaded_bytes': 400, 'misses': 5}])

    def test_code_traffic_uses_groups_and_511_advances(self):
        result = sim.analyze(self.profile, cpu_gbps=100)
        work = result['decode']['work']
        self.assertEqual(work['cpu_code_bytes'], 27590 * 3 * 4096 * 2048 // 2)
        self.assertEqual(work['output_advances'], 511)
        expected = 511 / (work['cpu_code_bytes'] / 100e9)
        self.assertAlmostEqual(result['decode']['conditional_streaming_roof']['tok_s'], expected)
        altered = copy.deepcopy(self.profile)
        altered['selected']['route_drop']['mean_share'] = .01
        self.assertEqual(sim.analyze(altered)['decode']['work'], sim.analyze(self.profile)['decode']['work'])

    def test_amdahl_bound_includes_remainder_outside_mtp_timers(self):
        result = sim.analyze(self.profile, target_decode=50)
        decode = result['decode']
        self.assertAlmostEqual(decode['ceiling_if_only_nonverification_is_removed_tok_s'], 511 / 14.3281)
        self.assertGreater(decode['phases_seconds']['other'], .3)
        self.assertAlmostEqual(sum(decode['phases_seconds'].values()), decode['observed_seconds'])
        faster = sim.analyze(self.profile, target_decode=1000)['decode']['target']
        self.assertFalse(faster['feasible_with_fixed_nonverification'])
        self.assertIsNone(faster['required_verification_speedup'])

    def test_prefill_uses_compressed_wire_extents_and_slowest_gpu(self):
        result = sim.analyze(self.profile)
        prefill = result['prefill']
        self.assertTrue(all(a < b for a, b in zip(prefill['source_wire_bytes']['mean'],
                                                prefill['expanded_allocation_bytes'])))
        self.assertEqual(prefill['conditional_proxy_roof']['limiting_resource'], 'gpu0_h2d')
        self.assertLess(prefill['conditional_proxy_roof_tok_s_interval'][0],
                        prefill['conditional_proxy_roof_tok_s_interval'][1])
        # Halving both PCIe links halves the proxy roof; it does not refit observations.
        links = [x / 2 for x in self.profile['hardware']['h2d_gbps_per_gpu']]
        slower = sim.analyze(self.profile, h2d_gbps=links)
        self.assertAlmostEqual(slower['prefill']['conditional_proxy_roof']['tok_s'],
                               prefill['conditional_proxy_roof']['tok_s'] / 2)
        self.assertEqual(slower['prefill']['observed_tok_s'], prefill['observed_tok_s'])
        self.assertEqual(slower['decode']['observed_tok_s'], result['decode']['observed_tok_s'])

    def test_measured_phase_bytes_are_deltas_not_allocations(self):
        profile = copy.deepcopy(self.profile)
        profile['prefill_transfer'] = dict(scope='one request preparation and prefill',
            before=[dict(device=i, prefill_bytes=10) for i in range(2)],
            after=[dict(device=i, prefill_bytes=10 + 80_000_000_000) for i in range(2)])
        exact = sim.analyze(profile)['prefill']['measured_matrix_transfer']
        self.assertEqual(exact['source_wire_bytes'], [80_000_000_000] * 2)
        self.assertGreater(exact['conditional_roof']['seconds'], 6)
        profile['prefill_transfer']['after'][0]['prefill_bytes'] = 9
        with self.assertRaises(ValueError):
            sim.analyze(profile)

    def test_measured_cpu_capacity_and_explicit_override(self):
        profile = copy.deepcopy(self.profile)
        profile['hardware']['cpu_stream_gbps'] = 117.816
        measured = sim.analyze(profile)
        self.assertTrue(measured['rates']['cpu_bandwidth_is_measured'])
        self.assertEqual(measured['rates']['cpu_gbps'], 117.816)
        override = sim.analyze(profile, cpu_gbps=180)
        self.assertFalse(override['rates']['cpu_bandwidth_is_measured'])
        self.assertEqual(override['rates']['cpu_gbps'], 180)
        self.assertEqual(measured['decode']['observed_tok_s'], override['decode']['observed_tok_s'])
        profile['hardware']['cpu_stream_gbps'] = 0
        with self.assertRaises(ValueError):
            sim.analyze(profile)

    def test_invalid_rates_faulted_runs_and_census_rejected(self):
        for rate in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                sim.analyze(self.profile, cpu_gbps=rate)
            with self.assertRaises(ValueError):
                sim.analyze(self.profile, h2d_gbps=[rate, 12])
        for key, value in [('major_faults', 1), ('generated_tokens', 511),
                           ('first_eos_position_1based', 13)]:
            bad = copy.deepcopy(self.profile)
            bad['selected']['requests'][0]['trial'][key] = value
            with self.assertRaises(ValueError):
                sim.analyze(bad)
        bad = copy.deepcopy(self.profile)
        bad['geometry']['experts'] = 224  # reject accidental GGUF/REAP geometry
        with self.assertRaises(ValueError):
            sim.analyze(bad)

    def test_source_provenance_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profile.json'
            bad = copy.deepcopy(self.profile)
            bad['sources'][0]['sha256'] = '0' * 64
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                sim.load_profile(path)

    def test_activity_unions_do_not_replace_qualified_timing(self):
        root = Path(__file__).resolve().parents[1]
        base = sim.analyze(sim.load_profile(root/'tools/sim/data/glm_fp4_v100_optimization.json'), target_prefill=1000)
        profile = sim.load_profile(root/'tools/sim/data/glm_fp4_v100_activity.json')
        result = sim.analyze(profile, target_prefill=1000)
        self.assertEqual(result['decode'], base['decode'])
        self.assertEqual(result['prefill'], base['prefill'])
        diagnostic = result['diagnostic_activity']
        # The observed union already accounts for overlapping kernels/copies.
        self.assertGreater(diagnostic['prefill']['conditional_kernel_only_ceiling_tok_s'],
                           diagnostic['prefill']['conditional_ceiling_if_all_gpu_idle_removed_tok_s'])
        self.assertGreater(diagnostic['prefill']['target']['retained_kernel_duration_speedup_required'], 1.4)
        self.assertGreater(diagnostic['decode']['target']['idle_reduction_fraction_at_retained_activity'], .9)
        profile['diagnostic_activity']['devices'][0]['all_activity_union_ms'] = 0
        with self.assertRaisesRegex(ValueError, 'unions'):
            sim.analyze(profile)

    def test_cli_uses_native_profile_without_changing_gguf_defaults(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.json'
            run = subprocess.run([sys.executable, str(root / 'tools/glm_runner_sim.py'),
                                  'native-fp4', '--cpu-gbps', '120', '--output', str(output)],
                                 cwd=directory, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(output.read_text())
            self.assertFalse(result['fitted'])
            self.assertEqual(result['rates']['cpu_gbps'], 120)
            self.assertAlmostEqual(result['decode']['observed_tok_s'], 31.01951619267308)


if __name__ == '__main__':
    unittest.main()
