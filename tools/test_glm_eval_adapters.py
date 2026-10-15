import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_eval_aider as aider  # noqa: E402
import glm_eval_effibench as eb  # noqa: E402
import glm_eval_omnicode as oc  # noqa: E402


class OmniCodeSampleTest(unittest.TestCase):
    ids = [f'django__django-{i}' for i in range(60)] + [f'sympy__sympy-{i}' for i in range(30)] + \
          [f'psf__requests-{i}' for i in range(2)] + [f'fmtlib__fmt_{i}' for i in range(8)]

    def test_size_and_determinism(self):
        a = oc.draw(self.ids, 20, 'salt')
        self.assertEqual(len(a), 20)
        self.assertEqual(a, oc.draw(self.ids, 20, 'salt'))
        self.assertNotEqual(a, oc.draw(self.ids, 20, 'other'))

    def test_every_repository_represented(self):
        a = oc.draw(self.ids, 20, 'salt')
        self.assertEqual({oc.repo_of(i) for i in a}, {oc.repo_of(i) for i in self.ids})

    def test_repo_names(self):
        self.assertEqual(oc.repo_of('django__django-11099'), 'django/django')
        self.assertEqual(oc.repo_of('nlohmann__json_4537'), 'nlohmann/json')
        self.assertEqual(oc.repo_of('ytdl-org__youtube-dl-123'), 'ytdl-org/youtube-dl')

    def test_committed_sample_matches_sizes(self):
        s = json.loads(oc.SAMPLE.read_text())
        for (lang, task), n in oc.SIZES.items():
            got = s['sets'][f'{lang}/{task}']['instances']
            self.assertEqual(len(got), n)
            self.assertEqual(len(set(got)), n)


class EffiBenchTest(unittest.TestCase):
    def test_passed_item_rules(self):
        self.assertTrue(eb.passed_item({'passed': True}))
        self.assertTrue(eb.passed_item({'status': 'done', 'text': '', 'exit_code': 0}))
        self.assertTrue(eb.passed_item({'text': '1 2\r\n3\n', 'input': '1 2', 'output': '\n3'}))
        self.assertFalse(eb.passed_item({'status': 'done', 'text': 'Wrong', 'exit_code': 0, 'input': 'a',
                                         'output': 'b'}))

    def test_committed_subset(self):
        s = json.loads(eb.SUBSET.read_text())
        self.assertEqual(s['count'], 100)
        self.assertEqual(len({(p['source'], p['id']) for p in s['problems']}), 100)

    def test_stats_runtime_and_failure(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'x.json'
            rows = [dict(passed=True, runtime=2_000_000, memory=3000), dict(passed=True, runtime=1_000_000, memory=5000)]
            f.write_text(json.dumps(dict(results=rows)))
            s = eb.stats(f)
            self.assertTrue(s['passed'])
            self.assertAlmostEqual(s['runtime_sum_s'], 0.003)
            self.assertEqual(s['memory_kb'], 5.0)
            f.write_text(json.dumps(dict(results=None)))
            self.assertFalse(eb.stats(f)['passed'])


class AiderReportTest(unittest.TestCase):
    def test_report_reads_last_try(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / 'generation.json').write_text('{}')
            for lang, name, outcomes in (('cpp', 'bank', [False, True]), ('python', 'zipper', [False, False]),
                                         ('python', 'grep', [True])):
                ex = run / 'bench' / '2026-10-16--strata' / lang / 'exercises' / 'practice' / name
                ex.mkdir(parents=True)
                (ex / '.aider.results.json').write_text(json.dumps(dict(tests_outcomes=outcomes, duration=1.0)))
            aider.cmd_report(types.SimpleNamespace(run=str(run), profile='p'))
            data = json.loads((run / 'results.json').read_text())
            got = {t['id']: (t['passed'], t['metrics']['first_try']) for t in data['tasks']}
            self.assertEqual(got, {'cpp/bank': (True, False), 'python/zipper': (False, False),
                                   'python/grep': (True, True)})
            self.assertEqual(data['meta']['cpp_pass'], '1/1')


if __name__ == '__main__':
    unittest.main()
