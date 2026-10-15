"""Measurement validity and case isolation for the TR16 runner."""
import unittest
from unittest.mock import patch
from glm_tr16_calibrate import case_config, summarize, missing_decoder_flags


class CalibrationTest(unittest.TestCase):
    def test_idle_check_does_not_import_coding_fixtures(self):
        import sys
        from glm_tr16_calibrate import wait_idle

        with patch.dict(sys.modules, {"glm_q2_coding_bench": None}), \
             patch("glm_q2_run_guard.gpu_compute_processes", return_value=[]), \
             patch("glm_q2_run_guard.process_cpu_snapshot", side_effect=[(0, {}), (1, {}), (2, {})]), \
             patch("glm_tr16_calibrate.time.monotonic", return_value=0), \
             patch("glm_tr16_calibrate.time.sleep"):
            wait_idle(timeout=10, stable_seconds=0)

    def test_quality_cpu_recording_keeps_gpu_and_memory_guards(self):
        from pathlib import Path
        from glm_tr16_quality import quality_guard
        quiet = quality_guard(Path('/tmp/test'), ['decoder', '--eval-corpus=test'])
        recorded = quality_guard(Path('/tmp/test'), ['decoder', '--eval-corpus=test'], True)
        self.assertIn('--require-idle', quiet)
        self.assertEqual(recorded, ['--record-interference' if x == '--require-idle' else x for x in quiet])
        for args in (quiet, recorded):
            self.assertIn('--require-gpu-idle', args)
            self.assertEqual(args[args.index('--min-gpu-free-mib') + 1], '512')
            self.assertEqual(args[-3:], ['--', 'decoder', '--eval-corpus=test'])

    def test_quality_desktop_allowance_keeps_guards(self):
        from pathlib import Path
        from glm_tr16_quality import quality_guard
        args=quality_guard(Path('/tmp/test'),['decoder'],True,True,True)
        self.assertIn('--allow-desktop-gpu',args)
        self.assertIn('--require-gpu-idle',args)
        self.assertIn('--memory-metric=pss',args)
        self.assertIn('--record-interference',args)

    def test_draft_backend_is_explicit_and_gpu_is_not_fit_as_cpu(self):
        import json
        import tempfile
        from pathlib import Path
        from glm_tr16_calibrate import command
        from glm_tr16_sim_records import import_run
        c = dict(exe='/tmp/decoder', model='/tmp/model', threads=15, context=4096,
                 prefill_batch=2048, gpu_budget_mib=14336, decode_cache_mib=4096,
                 speculative='mtp', draft_depth=1)
        default = command(c, Path('/tmp/prompt'), 128, 1)
        self.assertIn('--mtp-experts=cpu', default)
        self.assertEqual(default, command({**c, 'mtp_experts': 'cpu'}, Path('/tmp/prompt'), 128, 1))
        gpu = command({**c, 'mtp_experts': 'gpu'}, Path('/tmp/prompt'), 128, 1)
        self.assertEqual(gpu, [x.replace('--mtp-experts=cpu', '--mtp-experts=gpu') for x in default])
        with self.assertRaises(ValueError):
            command({**c, 'mtp_experts': 'auto'}, Path('/tmp/prompt'), 128, 1)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'result.json'
            p.write_text(json.dumps(dict(clean=True, complete=True, quiet=True, guard_exit_code=0,
                                         config={**c, 'mtp_experts': 'gpu', 'env': {}}, measurements={})))
            with self.assertRaisesRegex(ValueError, 'GPU draft'):
                import_run(p)

    def test_approximate_acceptance_is_recorded_but_not_fit_to_exact_simulator(self):
        import json
        import tempfile
        from pathlib import Path
        from glm_low_memory_bench import parse_log
        from glm_tr16_sim_records import import_run
        rows = parse_log('MTP_APPROX margin=1 top_k=4 max_soft_per_window=3 soft_accepted=21 '
                         'sum_logit_regret=9.3257 max_logit_regret=0.868864\n')['mtp_approx']
        self.assertEqual(rows[0]['soft_accepted'], 21)
        self.assertAlmostEqual(rows[0]['sum_logit_regret'], 9.3257)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'result.json'
            for key, value in [('STRATA_GLM_MTP_ACCEPT_MARGIN', '1'), ('STRATA_GLM_DEFER_EXPERTS', '1'),
                               ('STRATA_GLM_DENSE_Q4', 'outputs'), ('STRATA_GLM_DENSE_Q3', 'all'), ('STRATA_GLM_TIER_LEARN_UNBIASED', '1'), ('STRATA_GLM_MTP_GPU_CACHE', '1'),
                               ('STRATA_GLM_CACHE_PROBE', '32'), ('STRATA_GLM_MMVQ_IL', '1')]:
                p.write_text(json.dumps(dict(clean=True, complete=True, quiet=True, guard_exit_code=0,
                                            config={'env': {key: value}}, measurements={})))
                with self.subTest(flag=key), self.assertRaisesRegex(ValueError, 'component model'):
                    import_run(p)

    def test_deferral_quality_scores_single_token_path(self):
        from glm_tr16_quality import evaluation_width, compare
        self.assertEqual(evaluation_width({}), 4)
        config = {'env': {'STRATA_GLM_DEFER_EXPERTS': '1'}}
        with self.assertRaises(ValueError):
            evaluation_width(config)
        config['env']['STRATA_GLM_EVAL_WIDTH'] = '1'
        self.assertEqual(evaluation_width(config), 1)
        base = dict(corpus_sha256='a', reference='/frozen', reference_size=100,
                    reference_mtime_ns=1, sequences=16, complete=True, inputs={},
                    metrics=dict(perplexity=5, kl=.13, top1_agreement=.87, tokens=8192))
        self.assertIn('incomparable evaluation width', compare({**base, 'config': config}, base))
        baseline = {**base, 'config': {'env': {'STRATA_GLM_EVAL_WIDTH': '1'}}}
        self.assertEqual(compare({**base, 'config': config}, baseline), [])

    def test_saved_experimental_profile_cannot_silently_use_old_decoder(self):
        flag = 'STRATA_GLM_PREFETCH_STREAM'
        profile = {'env': {flag: '1'}}
        self.assertEqual(missing_decoder_flags([case_config(profile, 'baseline')], b'old decoder'), [flag])
        self.assertEqual(missing_decoder_flags([profile], flag.encode()), [])
        self.assertEqual(missing_decoder_flags([{'env': {flag: '0'}}], b'old decoder'), [])
        margin = 'STRATA_GLM_MTP_ACCEPT_MARGIN'
        self.assertEqual(missing_decoder_flags([{'env': {margin: '0.25'}}], b'old decoder'), [margin])
        self.assertEqual(missing_decoder_flags([{'env': {margin: '0'}}], b'old decoder'), [])
        blend = 'STRATA_GLM_MTP_RECURSIVE_BLEND'
        for value in ('0', '0.5'):
            self.assertEqual(missing_decoder_flags([{'env': {blend: value}}], b'old decoder'), [blend])
            self.assertEqual(missing_decoder_flags([{'env': {blend: value}}], blend.encode()), [])
        self.assertEqual(missing_decoder_flags([{'env': {blend: '1'}}], b'old decoder'), [])
        self.assertEqual(missing_decoder_flags([{'env': {}}], b'old decoder'), [])
        for flag, value in [('STRATA_GLM_TIER_LEARN_UNBIASED', '1'), ('STRATA_GLM_MTP_GPU_CACHE', '1'), ('STRATA_GLM_CACHE_PROBE', '32')]:
            self.assertEqual(missing_decoder_flags([{'env': {flag: value}}], b'old decoder'), [flag])
            self.assertEqual(missing_decoder_flags([{'env': {flag: '0'}}], b'old decoder'), [])
        q3 = 'STRATA_GLM_DENSE_Q3'
        self.assertEqual(missing_decoder_flags([{'env': {q3: 'all'}}], b'old decoder'), [q3])
        self.assertEqual(missing_decoder_flags([{'env': {q3: 'all'}}], q3.encode()), [])
        self.assertEqual(missing_decoder_flags([{'env': {q3: '0'}}], b'old decoder'), [])
        repeat = 'STRATA_GLM_MTP_REPEAT_GUARD'
        self.assertEqual(missing_decoder_flags([{'env': {repeat: '4'}}], b'old decoder'), [repeat])
        self.assertEqual(missing_decoder_flags([{'env': {repeat: '0'}}], b'old decoder'), [])
        from glm_low_memory_bench import parse_log
        self.assertEqual(parse_log('MTP_REPEAT_GUARD ngram=4 history=128 blocked=9\n')['mtp_repeat_guard'],
                         [dict(ngram=4, history=128, blocked=9)])

    def test_humaneval_requires_all_tasks_and_allows_only_one_lost_pass(self):
        from glm_tr16_quality import compare_humaneval
        def result(passes):
            return dict(complete=True, tasks=164,
                        results=[dict(task_id=f'HumanEval/{i}', passed=i < passes) for i in range(164)])
        self.assertEqual(compare_humaneval(result(159), result(160)), [])
        self.assertTrue(compare_humaneval(result(158), result(160)))
        partial = result(160)
        partial['results'].pop()
        self.assertTrue(compare_humaneval(partial, result(160)))
        duplicate = result(160)
        duplicate['results'][0]['task_id'] = 'HumanEval/1'
        self.assertTrue(compare_humaneval(duplicate, result(160)))

    def test_quality_gate_is_paired_and_rejects_invalid_metrics(self):
        import copy
        from glm_tr16_quality import compare
        base = dict(corpus_sha256='a', reference='/frozen', reference_size=100,
                    reference_mtime_ns=1, sequences=16, complete=True, inputs={'model': 'same'},
                    metrics=dict(perplexity=5, kl=.13, top1_agreement=.87, tokens=8192))
        candidate = copy.deepcopy(base)
        candidate['metrics'].update(perplexity=5.1, kl=.15, top1_agreement=.865)
        self.assertEqual(compare(candidate, base), [])
        candidate['metrics']['perplexity'] = 5.2
        self.assertTrue(compare(candidate, base))
        candidate['metrics']['perplexity'] = float('nan')
        self.assertTrue(compare(candidate, base))
        candidate = copy.deepcopy(base)
        candidate['corpus_sha256'] = 'different'
        self.assertTrue(compare(candidate, base))

    def test_gpu_clients_do_not_hide_a_competing_decoder(self):
        from glm_q2_run_guard import gpu_compute_processes
        sample = '4094, /usr/libexec/gnome-remote-desktop-daemon\n12345, /tmp/strata-glm-decode\n'
        with patch('glm_q2_run_guard.subprocess.check_output', return_value=sample):
            self.assertEqual(gpu_compute_processes(0), [(12345, '/tmp/strata-glm-decode')])

    def test_acceptance_is_conditional_and_discards_warmup(self):
        from glm_tr16_sim_records import acceptance
        r = {'config': {'draft_depth': 2}, 'manifest': {'timed_trials': 1}, 'measurements': {
            'speculation': [{'rounds': 100}, {'rounds': 10}],
            'mtp_positions': [{'index': 0, 'accepted': 100}, {'index': 1, 'accepted': 100},
                              {'index': 0, 'accepted': 8}, {'index': 1, 'accepted': 4}]}}
        self.assertEqual(acceptance(r), [.8, .5])

    def test_presence_flags_are_removed_and_base_is_unchanged(self):
        base = {'env': {'STRATA_GLM_TIER_ADAPT': '1', 'STRATA_GLM_SPLIT_VERIFY': '1'}}
        self.assertNotIn('STRATA_GLM_TIER_ADAPT', case_config(base, 'static')['env'])
        self.assertNotIn('STRATA_GLM_SPLIT_VERIFY', case_config(base, 'unsplit')['env'])
        self.assertEqual(len(base['env']), 2)
        with self.assertRaises(ValueError):
            case_config(base, 'aff50')

    def test_warmup_and_failed_guard(self):
        log = '\n'.join(['PREFILL trial=0 tokens=2048 batch=2048 ms=20000 tok_s=102.4',
                         'PREFILL trial=1 tokens=2048 batch=2048 ms=20000 tok_s=102.4',
                         'DECODE steps=2 ms=2000 tok_s=1', 'DECODE steps=2 ms=100 tok_s=20'])
        good = dict(exit_code=0, rejected=None)
        r = summarize(log, '1 2 3 1 2 3', good, 3, 1)
        self.assertTrue(r['clean'])
        self.assertTrue(r['repeated_tokens_equal'])
        self.assertEqual(r['measurements']['median_decode_tok_s'], 20)
        self.assertFalse(summarize(log, '1 2 3', good, 3, 1)['clean'])
        self.assertFalse(summarize(log.replace('steps=2', 'steps=1'), '1 2 3 1 2 3', good, 3, 1)['clean'])
        self.assertFalse(summarize(log, '1 2 3 1 2 3', dict(exit_code=0, rejected='swap'), 3, 1)['clean'])
        self.assertFalse(summarize(log, '1 2 3 1 2 4', good, 3, 1)['repeated_tokens_equal'])


if __name__ == '__main__':
    unittest.main()
