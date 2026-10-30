"""Launch profile contracts without SSH, a checkpoint, or GPU allocation."""
import unittest
from glm_fp4_two_node import make_command, verify_runs, verify_latest, wait_worker_ready


class CommandTest(unittest.TestCase):
    def config(self):
        return dict(remote_decoder='/opt/strata/bin/strata-glm-decode', gpu_uuid='GPU-test',
                    compute_address='192.0.2.1:29581', fetch_address='192.0.2.1:29582')

    def test_exact_mtp_and_single_gpu(self):
        request = make_command(self.config(), [1, 2], 16, True, True)
        self.assertIn('--speculative=mtp', request['argv'])
        self.assertIn('--draft-depth=2', request['argv'])
        self.assertIn('--gpu-devices=0', request['argv'])
        self.assertEqual(request['environment']['CUDA_VISIBLE_DEVICES'], 'GPU-test')
        self.assertEqual(request['environment']['STRATA_GLM_MTP_ACCEPT_MARGIN'], '0')
        for name in ('CPU_Q8', 'CPU_FUSED', 'ASYNC_UPLOAD', 'REGISTER_WEIGHTS', 'ROUTE_DROP', 'ROUTE_AFFINITY'):
            self.assertEqual(request['environment']['STRATA_GLM_FP4_' + name], '0')
        self.assertEqual(request['environment']['STRATA_GLM_FP4_REMOTE_CHECK'], '1')

    def test_compact_cache_is_opt_in_and_exact(self):
        base = make_command(self.config(), [1], 16, False, False)['environment']
        tuned = make_command(dict(self.config(), cpu_compact_cache=True), [1], 16, False, False)['environment']
        for knob in ('CPU_SPARSE', 'CPU_BORROW', 'CPU_CSF32_LOCAL', 'CPU_SPIN'):
            self.assertNotIn('STRATA_GLM_FP4_' + knob, base)
            self.assertEqual(tuned['STRATA_GLM_FP4_' + knob], '1')
        self.assertEqual(tuned['STRATA_GLM_FP4_CPU_Q8'], '0')

    def test_repeated_trials_are_explicit_and_bounded(self):
        request = make_command(self.config(), [1], 16, True, False, 4)
        self.assertIn('--decode-bench=4', request['argv'])
        for trials in (0, 11):
            with self.assertRaises(ValueError): make_command(self.config(), [1], 16, True, False, trials)

    def test_batch_repeats_and_mtp_agreement(self):
        plain = dict(ids=[1], tokens=2, trials=2, mtp=False, check_remote=False)
        mtp = dict(plain, mtp=True)
        self.assertEqual(len(verify_runs([3, 4] * 4, [plain, mtp])), 2)
        for tokens in ([3, 4], [3, 4, 3, 5, 3, 4, 3, 4], [3, 4, 3, 4, 3, 5, 3, 5]):
            with self.assertRaises(RuntimeError): verify_runs(tokens, [plain, mtp])

    def test_paired_baseline_and_immediate_mismatch(self):
        request = make_command(self.config(), [1], 16, False, False, mtp_baseline=True)
        self.assertIn('--speculative=mtp', request['argv'])
        self.assertIn('--mtp-greedy-control', request['argv'])
        specs = [dict(ids=[1], tokens=3, trials=1, mtp=False, check_remote=False),
                 dict(ids=[1], tokens=3, trials=2, mtp=True, check_remote=False)]
        for count in range(1, 10): verify_latest(([3, 4, 5] * 3)[:count], specs)
        with self.assertRaisesRegex(RuntimeError, 'run=1 token=1'):
            verify_latest([3, 4, 5, 3, 9], specs)
        with self.assertRaisesRegex(RuntimeError, 'trial=1 token=1'):
            verify_latest([3, 4, 5, 3, 4, 5, 3, 9], specs)

    def test_worker_warmup_gate(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'worker.stderr'; log.write_bytes(b'FP4_WORKER ready payload_bytes=123 no_GPU=1\n')
            wait_worker_ready(Mock(poll=Mock(return_value=None)), log, timeout=0)
            log.write_bytes(b'')
            with self.assertRaises(TimeoutError): wait_worker_ready(Mock(poll=Mock(return_value=None)), log, timeout=0)
            with self.assertRaises(RuntimeError): wait_worker_ready(Mock(poll=Mock(return_value=1)), log)

    def test_plain_mode_and_context_rejection(self):
        request = make_command(self.config(), [1], 16, False, False)
        self.assertFalse(any(arg.startswith('--speculative=') for arg in request['argv']))
        for ids, steps in (([], 1), ([1], 0), ([1] * 9216, 1)):
            with self.assertRaises(ValueError): make_command(self.config(), ids, steps, True, False)


if __name__ == '__main__': unittest.main()
