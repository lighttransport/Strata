import unittest
from glm_v100_optimize import assess


class EligibilityTest(unittest.TestCase):
    def record(self):
        record = dict(
            complete=True,
            instrumented=False,
            repetitions_identical=True,
            configuration=dict(prompt_tokens=8192),
            trials=[
                dict(token_ids=list(range(512)), decode_tok_s=61) for _ in range(3)
            ],
        )
        record["full_warmup_trials"] = record["trials"][:2]
        record["validation_trials"] = record["trials"][:]
        return record

    def test_speed_does_not_qualify_quality(self):
        result = assess(self.record(), {9000})
        self.assertTrue(result["speed_target_reached"])
        self.assertFalse(result["quality_qualified"])

    def test_post_eos_or_cycle_cannot_qualify(self):
        self.assertFalse(assess(self.record(), {200})["timing_eligible"])
        row = self.record()
        for trial in row["trials"]:
            trial["token_ids"] = list(range(32)) * 16
        self.assertFalse(assess(row, set())["timing_eligible"])

    def test_incomplete_instrumented_short_and_wrong_context_are_rejected(self):
        for key, value in [
            ("complete", False),
            ("instrumented", True),
            ("repetitions_identical", False),
        ]:
            row = self.record()
            row[key] = value
            self.assertFalse(assess(row, set())["timing_eligible"])
        row = self.record()
        row["trials"] = row["trials"][:1]
        self.assertFalse(assess(row, set())["timing_eligible"])
        row = self.record()
        row["configuration"]["prompt_tokens"] = 4096
        self.assertFalse(assess(row, set())["timing_eligible"])
        self.assertFalse(assess(self.record(), set(), tokens=128)["timing_eligible"])

    def test_validation_cycles_also_reject_timing(self):
        row = self.record()
        row["validation_trials"] = [
            dict(token_ids=[7] * 512, decode_tok_s=61) for _ in range(3)
        ]
        self.assertFalse(assess(row, set())["timing_eligible"])


if __name__ == "__main__":
    unittest.main()
