import json
from pathlib import Path
import tempfile
import unittest
from glm_q2_humaneval import load_engine_arguments, result_summary


class ArgumentIdentityTest(unittest.TestCase):
    def test_ungreaded_and_partial_runs_cannot_qualify(self):
        ungraded = [dict(task_id=f"HumanEval/{i}", passed=None) for i in range(164)]
        self.assertTrue(result_summary(ungraded)["generation_complete"])
        self.assertFalse(result_summary(ungraded)["complete"])
        self.assertIsNone(result_summary(ungraded)["pass_at_1"])
        graded = [dict(task_id=f"HumanEval/{i}", passed=True) for i in range(164)]
        self.assertTrue(result_summary(graded)["complete"])
        self.assertFalse(result_summary(graded[:163])["complete"])

    def test_profile_binds_model_context_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "full-q23.gguf"
            path = Path(directory) / "arguments.json"
            values = [
                str(model),
                "32768",
                "4096",
                "32",
                "0",
                "4096",
                "154820,154827",
            ] + ["0"] * 16
            path.write_text(json.dumps(values))
            self.assertEqual(
                load_engine_arguments(path, model, 32768, {154820, 154827}), values
            )
            for index, value in [
                (0, str(model.with_name("other.gguf"))),
                (1, "8192"),
                (6, "154820"),
            ]:
                broken = values.copy()
                broken[index] = value
                path.write_text(json.dumps(broken))
                with self.assertRaises(ValueError):
                    load_engine_arguments(path, model, 32768, {154820, 154827})


if __name__ == "__main__":
    unittest.main()
