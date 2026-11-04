"""CPU-only checks for benchmark command construction and timing extraction."""
import unittest
from bench_decode import counters, flags, token_ids


class DecodeBenchTests(unittest.TestCase):
    def test_token_reference_accepts_engine_delimiter_and_rejects_errors(self):
        self.assertEqual(token_ids(b'1,2,3,\n'), (1, 2, 3))
        for value in (b'', b'error: model failed', b'1,,2', b'1,-2'):
            with self.assertRaises(ValueError):
                token_ids(value)

    def test_replaces_all_occurrences_and_preserves_unrelated_flags(self):
        self.assertEqual(flags(['--pack', 'p', '--n', '3', '--serve', '--n', '4'],
                               {'--serve': None, '--n': 128, '--no-expert-cache': True}),
                         ['--pack', 'p', '--n', '128', '--no-expert-cache'])

    def test_counters_ignore_incomplete_or_failed_run(self):
        self.assertEqual(counters('remote service failed before decode'), {})

    def test_extracts_measured_decode_and_exposed_wait(self):
        value = counters('decode: 128 tokens in 18490 ms (6.87 tok/s); CPU lane: 24415\n'
                         'decode per token: layers 4.21 ms (of it: routing waits 0.00, '
                         'CPU lane 116.87, the rest host launches), head wait 141.06, Engram reads 0.12')
        self.assertEqual(value, dict(tokens=128, decode_ms=18490.0, tok_s=6.87,
                                    cpu_ms_per_token=116.87, head_wait_ms_per_token=141.06,
                                    engram_wait_ms_per_token=0.12))


if __name__ == '__main__':
    unittest.main()
