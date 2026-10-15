"""Launcher safety tests; optional integration uses only the explicit LVP ICD.

STRATA_HW_DIAG_TEST_EXE=build-.../vulkan/glm_vulkan_hw_diag PYTHONPATH=tools
python -m unittest tools/test_glm_vulkan_hw_diag.py
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import glm_vulkan_hw_diag as diag


class GuardTests(unittest.TestCase):
    def test_memory_reserve_and_large_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p/'mem_info_vram_total').write_text(str(16384*diag.MIB))
            (p/'mem_info_vram_used').write_text(str(6*diag.MIB))
            self.assertEqual(diag.memory_plan('vram', None, p)['memory_mib'], 14336)
            with self.assertRaises(ValueError):
                diag.memory_plan('vram', 15360, p)
            (p/'mem_info_vram_used').write_text(str(8000*diag.MIB))
            self.assertEqual(diag.memory_plan('vram', None, p)['memory_mib'], 7296)
            (p/'mem_info_vram_used').write_text(str(16300*diag.MIB))
            with self.assertRaises(RuntimeError):
                diag.memory_plan('smoke', None, p)

    def test_full_heap_targets_16_gib_without_the_old_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p/'mem_info_vram_total').write_text(str(16384*diag.MIB))
            (p/'mem_info_vram_used').write_text(str(6*diag.MIB))
            result = diag.memory_plan('vram-full', None, p)
            self.assertEqual(result['memory_mib'], 16384)
            self.assertEqual(result['allocation_limit_mib'], 16378)
            self.assertEqual(result['reserve_mib'], 0)
            self.assertFalse(result['physical_range_complete'])
            with self.assertRaises(ValueError):
                diag.memory_plan('vram-full', 14336, p)
            (p/'mem_info_vram_used').write_text(str(16384*diag.MIB))
            with self.assertRaises(RuntimeError):
                diag.memory_plan('vram-full', None, p)

    def test_faulted_boot_never_launches_or_kills(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'new'
            args = diag.parser().parse_args(['--output', str(output), '--run'])
            with patch.object(diag, 'kernel_log', return_value='ring comp_1.2.0 timeout'), \
                 patch.object(diag.subprocess, 'Popen') as launch, \
                 patch.object(diag.subprocess, 'run') as kill:
                result = diag.run(args, dict(options=dict(diag.PRESETS['smoke'])))
            launch.assert_not_called(); kill.assert_not_called()
            self.assertFalse(result['passed'])
            self.assertIn('no launch', result['termination_reason'])
            self.assertIsNone(result['exit_code'])
            self.assertFalse((output/'diagnostic.log').exists())
            self.assertEqual(json.loads((output/'guard.json').read_text()), result)

    def test_unreadable_kernel_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            args = diag.parser().parse_args(['--output', str(Path(directory)/'new'), '--run'])
            with patch.object(diag, 'kernel_log', side_effect=PermissionError('dmesg denied')), \
                 patch.object(diag.subprocess, 'Popen') as launch:
                result = diag.run(args, dict(options=dict(diag.PRESETS['smoke'])))
            launch.assert_not_called()
            self.assertFalse(result['passed'])
            self.assertIn('preflight failed', result['termination_reason'])

    def test_frozen_faulted_boot_marker_survives_kernel_log_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = diag.parser().parse_args(['--output', str(Path(directory)/'new'), '--run'])
            boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            plan = dict(options=dict(diag.PRESETS['smoke']), frozen_build=dict(requires_new_boot_from=boot))
            with patch.object(diag, 'kernel_log', return_value='older fault lines no longer present'), \
                 patch.object(diag.subprocess, 'Popen') as launch:
                result = diag.run(args, plan)
            launch.assert_not_called()
            self.assertFalse(result['passed'])
            self.assertIn('new boot required', result['termination_reason'])

    def test_active_workload_never_launches(self):
        with tempfile.TemporaryDirectory() as directory:
            args = diag.parser().parse_args(['--output', str(Path(directory)/'new'), '--run'])
            with patch.object(diag, 'kernel_log', return_value='clean'), \
                 patch.object(diag, 'gpu_devices', return_value=[]), \
                 patch.object(diag, 'identities', return_value={}), \
                 patch.object(diag, 'active_workloads', return_value=[dict(pid=123, executable='strata-glm-vulkan')]), \
                 patch.object(diag.subprocess, 'Popen') as launch:
                result = diag.run(args, dict(options=dict(diag.PRESETS['smoke'])))
            launch.assert_not_called()
            self.assertFalse(result['passed'])
            self.assertIn('workload', result['termination_reason'])

    def test_result_requires_full_preset_and_physical_device(self):
        options = dict(diag.PRESETS['smoke'])
        result = dict(passed=True, cpu_device=False, suite='all', memory_mib=64, memory_rounds=1,
                      compute_rounds=2, compute_repeats=1, patterns_per_round=9, max_submit_ms=5, gpu_submissions=100)
        self.assertTrue(diag.complete_result(result, options))
        for key, value in [('cpu_device', True), ('memory_mib', 1), ('memory_rounds', 0),
                           ('compute_rounds', 0), ('compute_repeats', 0), ('patterns_per_round', 1), ('max_submit_ms', 501),
                           ('max_submit_ms', float('nan')), ('gpu_submissions', 0), ('passed', False)]:
            with self.subTest(key=key, value=value):
                self.assertFalse(diag.complete_result(dict(result, **{key: value}), options))
        self.assertFalse(diag.complete_result(None, options))

    def test_existing_evidence_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            args = diag.parser().parse_args(['--output', directory, '--run'])
            p = Path(directory)/'guard.json'; p.write_text('preserve')
            with self.assertRaises(FileExistsError):
                diag.run(args, {})
            self.assertEqual(p.read_text(), 'preserve')

    def test_full_heap_requires_honest_byte_accounting(self):
        options = dict(diag.PRESETS['vram-full'], memory_mib=16384, allocation_limit_mib=16378)
        payload = 16376*diag.MIB
        result = dict(passed=True, cpu_device=False, suite='memory', memory_mib=16384, memory_rounds=1,
                      compute_rounds=0, compute_repeats=0, patterns_per_round=73, max_submit_ms=5, gpu_submissions=100,
                      full_heap=True, physical_range_complete=False, memory_target_bytes=16384*diag.MIB,
                      memory_payload_bytes=payload, unavailable_target_bytes=8*diag.MIB,
                      allocation_limit_bytes=16378*diag.MIB, cpu_verified_words=payload//4*73)
        self.assertTrue(diag.complete_result(result, options))
        for key, value in [('physical_range_complete', True), ('unavailable_target_bytes', 0),
                           ('memory_target_bytes', 14336*diag.MIB), ('memory_payload_bytes', 0),
                           ('allocation_limit_bytes', 14336*diag.MIB), ('cpu_verified_words', 0)]:
            with self.subTest(key=key):
                self.assertFalse(diag.complete_result(dict(result, **{key: value}), options))

    def test_full_heap_rechecks_available_vram_in_the_executed_command(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'new'
            args = diag.parser().parse_args(['--preset', 'vram-full', '--output', str(output), '--run'])
            options = dict(diag.PRESETS['vram-full'], memory_mib=16384, allocation_limit_mib=16378)
            plan = dict(options=options, command=['/not/executed', '--allocation-limit-mib', '16378'])
            process = Mock(pid=987654); process.poll.return_value = 0
            with patch.object(diag, 'kernel_log', return_value='clean'), \
                 patch.object(diag, 'gpu_devices', return_value=[]), \
                 patch.object(diag, 'identities', return_value={'pci': '0210df67'}), \
                 patch.object(diag, 'active_workloads', return_value=[]), \
                 patch.object(diag, 'memory_plan', return_value={'allocation_limit_mib': 16284}), \
                 patch.object(diag.subprocess, 'Popen', return_value=process) as launch:
                result = diag.run(args, plan)
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index('--allocation-limit-mib')+1], '16284')
            self.assertEqual(result['execution_options']['allocation_limit_mib'], 16284)
            self.assertEqual(options['allocation_limit_mib'], 16378)
            self.assertFalse(result['passed'])  # No terminal result was produced.

    def test_timeout_kills_only_owned_unit_and_never_accepts_partial_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'new'
            args = diag.parser().parse_args(['--output', str(output), '--run'])
            plan = dict(options=dict(diag.PRESETS['smoke']), command=['/not/executed'])
            process = Mock(pid=987654)
            process.poll.side_effect = [None, None, -9]
            with patch.object(diag, 'kernel_log', side_effect=['clean', 'ring comp_1.2.0 timeout', 'ring comp_1.2.0 timeout']), \
                 patch.object(diag, 'gpu_devices', return_value=[]), \
                 patch.object(diag, 'identities', return_value={'pci': '0210df67'}), \
                 patch.object(diag, 'active_workloads', return_value=[]), \
                 patch.object(diag, 'memory_plan', return_value={}), \
                 patch.object(diag.subprocess, 'Popen', return_value=process), \
                 patch.object(diag.subprocess, 'run') as kill, \
                 patch.object(diag.os, 'killpg') as kill_group, \
                 patch.object(diag, 'capture_dump', return_value={'captured': False}):
                result = diag.run(args, plan)
            self.assertFalse(result['passed'])
            self.assertEqual(result['exit_code'], -9)
            self.assertEqual(result['termination_reason'], 'hardware/GPU fault')
            self.assertEqual(kill.call_args.args[0], ['systemctl', '--user', 'kill', '--signal=SIGKILL', result['own_unit']+'.service'])
            self.assertTrue(result['own_unit'].startswith('strata-vulkan-hw-diag-'))
            kill_group.assert_called_once_with(987654, diag.signal.SIGKILL)
            self.assertEqual(len(result['faults']), 1)

    def test_exit_zero_requires_complete_diagnostic_result(self):
        for passed in (True, False):
            with self.subTest(passed=passed), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)/'new'
                args = diag.parser().parse_args(['--output', str(output), '--run'])
                plan = dict(options=dict(diag.PRESETS['smoke']), command=['/not/executed'])
                row = dict(event='result', passed=passed, cpu_device=False, suite='all', memory_mib=64,
                           memory_rounds=1, compute_rounds=2, compute_repeats=1, patterns_per_round=9, max_submit_ms=1, gpu_submissions=100)
                process = Mock(pid=987654); process.poll.return_value = 0
                def launch(command, **kwargs):
                    kwargs['stdout'].write(json.dumps(row)+'\n'); kwargs['stdout'].flush()
                    return process
                with patch.object(diag, 'kernel_log', return_value='clean'), \
                     patch.object(diag, 'gpu_devices', return_value=[]), \
                     patch.object(diag, 'identities', return_value={'pci': '0210df67'}), \
                     patch.object(diag, 'active_workloads', return_value=[]), \
                     patch.object(diag, 'memory_plan', return_value={}), \
                     patch.object(diag.subprocess, 'Popen', side_effect=launch), \
                     patch.object(diag.subprocess, 'run') as kill:
                    result = diag.run(args, plan)
                self.assertEqual(result['passed'], passed)
                kill.assert_not_called()

    def test_coredump_prefix_bounded_private_and_missing_dump_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory); device = p/'device'; device.mkdir()
            out = p/'evidence'; out.mkdir()
            self.assertFalse(diag.capture_dump(device, out)['captured'])
            (device/'devcoredump').mkdir()
            (device/'devcoredump/data').write_bytes(b'x'*(4*diag.MIB+100))
            result = diag.capture_dump(device, out)
            self.assertTrue(result['captured']); self.assertTrue(result['truncated'])
            saved = out/'devcoredump-prefix.bin'
            self.assertEqual(saved.stat().st_size, 4*diag.MIB)
            self.assertEqual(saved.stat().st_mode & 0o777, 0o600)

    def test_frozen_manifest_checks_executable_and_retains_compiled_source(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory); exe = p/'exe'; exe.write_bytes(b'frozen')
            manifest = p/'manifest.json'
            manifest.write_text(json.dumps(dict(executable_sha256=diag.sha(exe), source_sha256={'old.cpp': 'compiled'})))
            result = diag.check_manifest(exe, manifest)
            self.assertEqual(result['compiled_source_sha256'], {'old.cpp': 'compiled'})
            exe.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                diag.check_manifest(exe, manifest)


class SoftwareVulkanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        exe = os.environ.get('STRATA_HW_DIAG_TEST_EXE')
        if not exe:
            raise unittest.SkipTest('Set STRATA_HW_DIAG_TEST_EXE for explicit software Vulkan checks')
        cls.exe = str(Path(exe).resolve(strict=True))
        icd = Path('/usr/share/vulkan/icd.d/lvp_icd.json')
        if not icd.is_file():
            raise unittest.SkipTest('LVP ICD unavailable; never fall back to hardware')
        cls.env = dict(os.environ, VK_DRIVER_FILES=str(icd), VK_ICD_FILENAMES=str(icd), LP_NUM_THREADS='4')

    def invoke(self, *extra, robust=False):
        env = dict(self.env, STRATA_VULKAN_FAST_ROBUST_BUFFERS='1' if robust else '0')
        run = subprocess.run([self.exe, '--cpu', '--memory-mib', '1', '--chunk-kib', '64',
                              '--max-submit-ms', '1000', *extra],
                             capture_output=True, text=True, env=env, timeout=60)
        rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')]
        self.assertTrue(rows, run.stderr)
        return run, rows

    def test_clean_all_with_minimum_chunk_and_maximum_seed(self):
        run, rows = self.invoke('--chunk-kib', '4', '--seed', '4294967295', '--iterations', '256')
        self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
        self.assertTrue(rows[-1]['passed']); self.assertTrue(rows[-1]['cpu_device'])
        self.assertEqual(rows[-1]['cpu_verified_words'], 9*262144+2*(65536+128*128))

    def test_all_walking_bits_and_changed_seed_second_round(self):
        run, rows = self.invoke('--suite', 'memory', '--full-patterns', '--rounds', '2', '--chunk-kib', '256')
        self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
        self.assertEqual(len([r for r in rows if r['event'] == 'memory_pattern']), 146)
        self.assertEqual(rows[-1]['cpu_verified_words'], 2*73*262144)

    def test_multiple_allocations_and_large_logical_address(self):
        run, rows = self.invoke('--suite', 'memory', '--memory-mib', '65', '--chunk-kib', '4096')
        self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
        allocation = next(r for r in rows if r['event'] == 'memory_allocated')
        self.assertEqual(allocation['blocks'], 2)
        self.assertEqual(rows[-1]['cpu_verified_words'], 65*9*262144)

    def test_non_power_of_two_window_permutation_and_robust_mode(self):
        run, rows = self.invoke('--suite', 'all', '--memory-mib', '3', '--chunk-kib', '4096', robust=True)
        self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
        self.assertTrue(rows[-1]['passed'])

    def test_every_compute_repetition_is_read_back(self):
        run, rows = self.invoke('--suite', 'compute', '--compute-rounds', '2', '--compute-repeats', '3')
        self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
        self.assertEqual(rows[-1]['compute_repeats'], 3)
        self.assertEqual(rows[-1]['cpu_verified_words'], 2*3*(65536+128*128))

    def test_full_heap_tests_available_tail_and_reports_unavailable_bytes(self):
        run, rows = self.invoke('--suite', 'memory', '--memory-mib', '80', '--chunk-kib', '1024',
                                '--full-heap', '--allocation-limit-mib', '70', '--cpu-allocation-max-mib', '16')
        self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
        allocation = next(r for r in rows if r['event'] == 'memory_allocated')
        self.assertGreaterEqual(allocation['blocks'], 5)
        self.assertGreater(allocation['allocation_retries'], 0)
        cursor = 0
        for block in [r for r in rows if r['event'] == 'memory_block']:
            self.assertEqual(block['logical_begin_byte'], cursor)
            self.assertEqual(block['logical_end_byte_exclusive']-cursor, block['payload_bytes'])
            cursor = block['logical_end_byte_exclusive']
        self.assertEqual(cursor, allocation['payload_bytes'])
        self.assertGreater(allocation['payload_bytes'], 67*diag.MIB)
        self.assertLess(allocation['payload_bytes'], 70*diag.MIB)
        self.assertLessEqual(allocation['payload_allocation_bytes']+allocation['local_aux_bytes'], 70*diag.MIB)
        self.assertLess(70*diag.MIB-allocation['payload_allocation_bytes']-allocation['local_aux_bytes'], 65536)
        result = rows[-1]
        self.assertTrue(result['full_heap']); self.assertFalse(result['physical_range_complete'])
        self.assertEqual(result['memory_payload_bytes']+result['unavailable_target_bytes'], 80*diag.MIB)
        self.assertEqual(result['cpu_verified_words'], result['memory_payload_bytes']//4*9)

    def test_last_mib_at_exact_16_gib_logical_boundary(self):
        run, rows = self.invoke('--suite', 'memory', '--logical-base-mib', '16383', '--seed', '4294967295')
        self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
        self.assertEqual(rows[-1]['logical_begin_byte'], 16383*diag.MIB)
        self.assertEqual(rows[-1]['logical_end_byte_exclusive'], 16384*diag.MIB)
        self.assertEqual(rows[-1]['cpu_verified_words'], 9*262144)

    def test_error_offset_near_16_gib_does_not_wrap(self):
        run, rows = self.invoke('--suite', 'memory', '--logical-base-mib', '16383', '--inject', 'memory')
        self.assertEqual(run.returncode, 1)
        mismatch = next(r for r in rows if r['event'] == 'mismatch')
        self.assertEqual(mismatch['logical_word'], 16383*262144+17)
        self.assertEqual(mismatch['xor'], 1)

    def test_one_bit_corruption_is_detected_in_each_checker(self):
        for target, stage in [('memory', 'memory_verify_invert'), ('integer', 'uint32_alu'), ('float', 'fp32_gemm')]:
            with self.subTest(target=target):
                run, rows = self.invoke('--suite', 'memory' if target == 'memory' else 'compute', '--inject', target)
                self.assertEqual(run.returncode, 1)
                mismatch = next(r for r in rows if r['event'] == 'mismatch')
                self.assertEqual(mismatch['stage'], stage)
                self.assertEqual(mismatch['logical_word'], 17)
                self.assertEqual(mismatch['xor'], 1)
                self.assertFalse(rows[-1]['passed'])

    def test_bad_arguments_rejected_before_vulkan_device_creation(self):
        for extra in [('--memory-mib', '16385'), ('--iterations', '257'), ('--suite', 'other'), ('--chunk-kib', '5'),
                      ('--logical-base-mib', '16384'), ('--allocation-limit-mib', '1'), ('--full-heap',)]:
            run, rows = self.invoke(*extra)
            self.assertEqual(run.returncode, 1)
            self.assertFalse(any(r['event'] == 'device' for r in rows))


if __name__ == '__main__':
    unittest.main()
