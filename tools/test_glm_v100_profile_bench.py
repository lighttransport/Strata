"""Check native timing normalization and saved-profile execution without a GPU."""
import json
import contextlib
import fcntl
import hashlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import unittest
from unittest import mock

import glm_v100_optimize as optimize
import glm_v100_profile_bench as bench


class ProfileTiming(unittest.TestCase):
    def test_idle_query_timeout_requires_a_fresh_quiet_interval(self):
        with mock.patch.object(optimize.time, 'monotonic', side_effect=[10, 10, 13]), \
             mock.patch.object(optimize, 'wait_idle', side_effect=[
                 optimize.subprocess.TimeoutExpired('nvidia-smi', 3), None]) as idle:
            optimize.wait_for_idle(timeout=20)
        self.assertEqual(idle.call_args_list, [mock.call(timeout=20), mock.call(timeout=17)])

    def test_idle_query_retries_keep_the_original_deadline(self):
        with mock.patch.object(optimize.time, 'monotonic', side_effect=[10, 10, 31]), \
             mock.patch.object(optimize, 'wait_idle', side_effect=
                 optimize.subprocess.TimeoutExpired('nvidia-smi', 3)) as idle:
            with self.assertRaises(TimeoutError):
                optimize.wait_for_idle(timeout=20)
        self.assertEqual(idle.call_count, 1)
        with mock.patch.object(optimize, 'wait_idle', side_effect=RuntimeError('busy')):
            with self.assertRaisesRegex(RuntimeError, 'busy'):
                optimize.wait_for_idle()

    def fixture(self, speculative='none'):
        log = ''.join('PREFILL trial=%d tokens=8192 batch=8192 ms=10000 tok_s=819.2\n'
                      'PREFILL_REPEAT trial=%d logits_identical=1 routes_identical=1\n' % (i, i)
                      for i in range(3))
        tag = 'SPECULATIVE source=mtp generated=512' if speculative == 'mtp' else 'DECODE steps=511'
        depth = 'MTP_DEPTH requested=3 effective=3\n' if speculative == 'mtp' else ''
        log += (depth + tag + ' ms=10000 tok_s=51.2\n') * 4
        stdout = '\n'.join(map(str, list(range(512)) * 4))
        return log, stdout

    def test_both_modes_exclude_the_initial_token(self):
        for mode in ('none', 'mtp'):
            log, stdout = self.fixture(mode)
            record = bench.summarize(log, stdout, speculative=mode)
            self.assertTrue(record['complete'])
            self.assertEqual(record['median_decode_tok_s'], 51.1)
            record.update(protocol='checkpoint', instrumented=False,
                          configuration=dict(prompt_tokens=8192))
            assessment = optimize.assess(record, set(), decode_target=50, prefill_target=700)
            self.assertTrue(assessment['speed_target_reached'])
            self.assertTrue(assessment['prefill_target_reached'])
            self.assertFalse(assessment['quality_qualified'])

    def test_cache_drift_is_recorded_without_claiming_repeatability(self):
        log, stdout = self.fixture()
        stdout = stdout.rsplit('\n', 1)[0] + '\n999'
        record = bench.summarize(log, stdout)
        record.update(protocol='checkpoint', instrumented=False, configuration=dict(prompt_tokens=8192))
        assessment = optimize.assess(record, set())
        self.assertTrue(assessment['performance_measured'])
        self.assertFalse(assessment['timing_eligible'])
        self.assertTrue(assessment['prefill_timing_eligible'])

    def test_untimed_warmup_does_not_require_identical_output(self):
        log, stdout = self.fixture()
        stdout = '999\n' + stdout.split('\n', 1)[1]
        record = bench.summarize(log, stdout)
        self.assertTrue(record['complete'])
        self.assertTrue(record['repetitions_identical'])
        self.assertFalse(record['warmup_matches_timed'])

    def test_invalid_runs_cannot_qualify(self):
        log, stdout = self.fixture()
        for modified_log, modified_stdout in (
            (log, stdout + '\ninvalid'), (log, stdout.rsplit('\n', 1)[0]),
            (log.replace('routes_identical=1', 'routes_identical=0'), stdout),
            (log.replace('ms=10000 tok_s=51.2', 'ms=0 tok_s=0'), stdout),
        ):
            self.assertFalse(bench.summarize(modified_log, modified_stdout)['complete'])
        self.assertFalse(bench.summarize(log, stdout, speculative='mtp')['complete'])
        mtp_log, mtp_stdout = self.fixture('mtp')
        self.assertFalse(bench.summarize(mtp_log.replace('effective=3', 'effective=2'),
                                        mtp_stdout, speculative='mtp')['complete'])
        record = bench.summarize(log, stdout)
        record.update(protocol='checkpoint', instrumented=False, configuration=dict(prompt_tokens=8192))
        self.assertFalse(optimize.assess(record, {100})['performance_measured'])
        record['instrumented'] = True
        self.assertFalse(optimize.assess(record, set())['performance_measured'])

    def test_saved_profiles_keep_modes_precision_and_worker_limits(self):
        with mock.patch.object(bench, 'GGUFFile', return_value=SimpleNamespace(metadata={})):
            for filename, mode in [('greedy', 'none'), ('aff008-mtp3', 'mtp')]:
                profile = json.loads((bench.ROOT / 'configs/experimental' /
                    ('glm53f-ud-q3-k-xl-v100s-' + filename + '.json')).read_text())
                cmd = bench.command(profile, Path('model'), Path('prompt'), Path('decoder'),
                                    Path('worker'), Path('result.json'))
                self.assertIn('--context=12288', cmd)
                self.assertIn('--prefill-experts=f16-batched', cmd)
                self.assertIn('--decode-cache-mib=30720', cmd)
                self.assertEqual(cmd[cmd.index('--worker-cache-mib') + 1], '32000')
                self.assertEqual('--speculative=mtp' in cmd, mode == 'mtp')
                self.assertNotIn('--lock-weights', cmd)
                self.assertNotIn('STRATA_GLM_MLA_F16', profile['env'])

    def test_matrix_overrides_do_not_mutate_the_source_profile(self):
        profile = bench.ROOT / 'configs/experimental/glm53f-ud-q3-k-xl-v100s-greedy.json'
        before = profile.read_bytes()
        case = dict(profile=str(profile), prompt='prompt.ids', build='baseline',
                    settings=dict(threads=18), env={'STRATA_GLM_PREFETCH_GROUPS': '8'})
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(model='model', baseline='control', decoder='candidate',
                                   worker='worker', baseline_worker='control-worker')
            cmd = optimize.command(case, args, Path(directory) / 'benchmark.json')
            cfg = json.loads((Path(directory) / 'benchmark.profile.json').read_text())
            self.assertEqual(cfg['threads'], 18)
            self.assertEqual(cfg['env']['STRATA_GLM_ROUTE_AFFINITY'], '0')
            self.assertEqual(cfg['env']['STRATA_GLM_PREFETCH_GROUPS'], '8')
            self.assertEqual(cmd[cmd.index('--decoder') + 1], 'control')
            self.assertEqual(cmd[cmd.index('--worker') + 1], 'control-worker')
        self.assertEqual(profile.read_bytes(), before)

    @unittest.skipUnless(hasattr(os, 'memfd_create'), 'Linux sealed resident model')
    def test_matrix_resumes_with_a_new_resident_descriptor(self):
        """A proc descriptor must survive path handling, and renewal changes it."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'decoder'
            binary.write_bytes(b'fake binary')
            profile = root / 'profile.json'
            profile.write_text(json.dumps(dict(model_sha256=hashlib.sha256(b'model').hexdigest(),
                                               model_bytes=5, env={})))
            matrix = root / 'matrix.json'
            matrix.write_text(json.dumps(dict(cases=[dict(name='control', profile=str(profile),
                                                        prompt=str(root / 'prompt.ids'))])))
            identity = root / 'resident.json'
            identity.write_text(json.dumps(dict(sha256=hashlib.sha256(b'model').hexdigest(),
                                                bytes=5, seals=15, socket_row_samples=[dict(local=True)],
                                                socket_row_plan_sha256='plan')))
            log, stdout = self.fixture()
            record = bench.summarize(log, stdout)
            record.update(protocol='checkpoint', instrumented=False, configuration=dict(prompt_tokens=8192))
            expected = root / 'fixture.json'
            expected.write_text(json.dumps(record))
            fake = root / 'fake.py'
            fake.write_text('import pathlib,sys\n'
                            'model=pathlib.Path(sys.argv[sys.argv.index("--model")+1])\n'
                            'assert model.read_bytes()==b"model"\n'
                            'out=pathlib.Path(sys.argv[sys.argv.index("--output")+1])\n'
                            'out.write_bytes(pathlib.Path(' + repr(str(expected)) + ').read_bytes())\n')
            original = optimize.command

            def command(case, args, output):
                cmd = original(case, args, output)
                cmd[1] = str(fake)
                return cmd

            fds = []
            try:
                with mock.patch.object(optimize, 'GGUFFile', return_value=SimpleNamespace(metadata={})), \
                     mock.patch.object(optimize, 'wait_idle'), \
                     mock.patch.object(optimize, 'command', side_effect=command) as runner:
                    for _ in range(2):
                        fd = os.memfd_create('strata-test', os.MFD_ALLOW_SEALING)
                        fds.append(fd)
                        os.write(fd, b'model')
                        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, 15)
                        argv = ['glm_v100_optimize', '--matrix', str(matrix), '--decoder', str(binary),
                                '--worker', str(binary), '--model', '/proc/%d/fd/%d' % (os.getpid(), fd),
                                '--resident-identity', str(identity), '--output', str(root / 'results')]
                        with mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                            self.assertEqual(optimize.main(), 0)
                    self.assertEqual(runner.call_count, 1)
            finally:
                for fd in fds:
                    os.close(fd)


if __name__ == '__main__':
    unittest.main()
