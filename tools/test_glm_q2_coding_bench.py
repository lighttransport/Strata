"""Check EOS-aware accounting and parity rejection for coding benchmarks."""
import unittest
from glm_q2_coding_bench import parse_trials


class TrialAccounting(unittest.TestCase):
    def test_eos_and_speculation_use_actual_counts(self):
        log = """DECODE_MODE source=none depth=0 shared_mtp_allocation=1
DECODE_TRIAL index=0
DECODE steps=2 ms=200 tok_s=10
DECODE_MODE source=mtp depth=1 shared_mtp_allocation=1
DECODE_TRIAL index=0
SPECULATIVE source=mtp generated=3 rounds=1 proposed=1 accepted=1 replayed=0 ms=100 tok_s=20
"""
        result = parse_trials(log, [11, 12, 99, 11, 12, 99], {99})
        self.assertTrue(result["none:0"][0]["ended_on_stop"])
        self.assertEqual(result["mtp:1"][0]["generated"], 3)
        self.assertEqual(result["mtp:1"][0]["accepted"], 1)

    def test_reject_output_divergence(self):
        log = "DECODE_TRIAL index=0\nDECODE steps=1 ms=100 tok_s=10\n" * 2
        with self.assertRaisesRegex(ValueError, "differs"):
            parse_trials(log, [1, 99, 2, 99], {99})

    def test_reject_unaccounted_tokens(self):
        with self.assertRaisesRegex(ValueError, "unaccounted"):
            parse_trials("DECODE_TRIAL index=0\nDECODE steps=1 ms=100 tok_s=10\n", [1, 99, 3], {99})


if __name__ == "__main__":
    unittest.main()
