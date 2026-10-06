import unittest
from glm_q2_pack_profiles import profiles
from glm_q2_humaneval import solution


class Profiles(unittest.TestCase):
    def setUp(self):
        self.report = {"projections": [
            {"name": f"blk.{layer}.ffn_{part}_exps.weight", "normalized_mse": error}
            for layer in (3, 4) for part, error in (("gate", .1), ("up", .2), ("down", .15))]}

    def test_retention_keeps_pairs_and_q2_down(self):
        result = profiles(self.report)
        self.assertEqual(result["q23"]["retain"], [])
        self.assertEqual(len(result["original"]["retain"]), 6)
        self.assertEqual(len(result["q2"]["retain"]), 2)
        self.assertTrue(all("down" in name for name in result["q2"]["retain"]))
        self.assertEqual(result["retain25"]["retain"], ["blk.3.ffn_gate_exps.weight", "blk.3.ffn_up_exps.weight"])
        self.assertEqual(result, profiles({"projections": list(reversed(self.report["projections"]))}))

    def test_invalid_report(self):
        for rows in ([], self.report["projections"][:-2], self.report["projections"] * 2,
                     [{"name": "blk.3.ffn_down_exps.weight", "normalized_mse": float("nan")} ]):
            with self.assertRaises(ValueError):
                profiles({"projections": rows})

    def test_extract_chat_solution(self):
        self.assertEqual(solution("reason</think>\n```python\ndef f(): return 1\n```", "def f():\n", "f"), "def f(): return 1\n")
        self.assertEqual(solution("    return 1", "def f():\n", "f"), "def f():\n    return 1")


if __name__ == "__main__":
    unittest.main()
