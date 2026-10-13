import copy
import unittest
from glm_tr16_decode35 import checked_speed, summary


class Decode35Test(unittest.TestCase):
    def record(self):
        return dict(complete=True, clean=True, quiet=True, inputs_unchanged=True, guard_exit_code=0,
                    manifest=dict(allow_interference=False, timed_trials=1, warmup_decode_trials=1, decode_tokens=512),
                    memory=dict(exit_code=0, rejected=None, peak_swap_kib=0),
                    measurements=dict(decode=[dict(tokens=512, tokens_per_second=36)]))

    def test_rejects_interrupted_swapped_or_nonquiet_measurements(self):
        self.assertEqual(checked_speed(self.record(), 512), 36)
        for section, key, value in [('', 'quiet', False), ('', 'complete', False), ('', 'inputs_unchanged', False),
                                    ('memory', 'peak_swap_kib', 1), ('manifest', 'allow_interference', True),
                                    ('manifest', 'timed_trials', 3)]:
            r = copy.deepcopy(self.record())
            (r[section] if section else r)[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                checked_speed(r, 512)

    def test_both_contexts_and_complete_repetitions_are_required(self):
        rows = [dict(name='candidate', width=width, tok_s=36) for width in (2048, 8192) for _ in range(3)]
        self.assertTrue(summary(rows, ['candidate'], 3)['candidate']['speed_gate'])
        self.assertFalse(summary(rows[:-1], ['candidate'], 3)['candidate']['speed_gate'])
        for row in rows:
            if row['width'] == 2048: row['tok_s'] = 34
        self.assertFalse(summary(rows, ['candidate'], 3)['candidate']['speed_gate'])
        self.assertFalse(summary(rows, ['candidate'], 3)['candidate']['quality_qualified'])

    def test_single_decode_excludes_prefill_token(self):
        r = self.record()
        r['measurements']['decode'][0].update(kind='single', tokens=511)
        self.assertEqual(checked_speed(r, 512), 36)
        r['measurements']['decode'][0]['tokens'] = 510
        with self.assertRaisesRegex(ValueError, 'incomplete continuation'):
            checked_speed(r, 512)

    def test_speculative_decode_requires_every_token(self):
        r = self.record()
        r['measurements']['decode'][0].update(kind='mtp', tokens=511)
        with self.assertRaisesRegex(ValueError, 'incomplete continuation'):
            checked_speed(r, 512)

    def test_single_screen_cannot_pass_repeated_speed_gate(self):
        rows = [dict(name='candidate', width=width, tok_s=36) for width in (2048, 8192)]
        result = summary(rows, ['candidate'], 1)['candidate']
        self.assertTrue(result['screen_above_target'])
        self.assertFalse(result['speed_gate'])


if __name__ == '__main__':
    unittest.main()
