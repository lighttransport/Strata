import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import glm_humaneval_grade as grader


class FrozenGenerationTest(unittest.TestCase):
    def fixture(self, root):
        tasks = [
            dict(
                task_id=f"HumanEval/{i}",
                prompt="def candidate():\n",
                entry_point="candidate",
                test="def check(f): assert f()==1",
            )
            for i in range(164)
        ]
        raw = gzip.compress("\n".join(map(json.dumps, tasks)).encode())
        dataset = root / "tasks.gz"
        dataset.write_bytes(raw)
        source = root / "generations"
        source.mkdir()
        checksum = hashlib.sha256(raw).hexdigest()
        (source / "manifest.json").write_text(json.dumps(dict(dataset_sha256=checksum)))
        text = "```python\ndef candidate(): return 1\n```"
        code = "def candidate(): return 1\n"
        tokens = [1, 2]
        (source / "answer.txt").write_text(text)
        (source / "answer.py").write_text(code)
        (source / "answer.json").write_text(json.dumps(tokens))
        row = dict(
            task_id="HumanEval/0",
            text_file="answer.txt",
            code_file="answer.py",
            tokens_file="answer.json",
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            code_sha256=hashlib.sha256(code.encode()).hexdigest(),
            token_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        )
        (source / "measurement.json").write_text(json.dumps(dict(results=[row])))
        return source, dataset, checksum

    def test_partial_grading_never_qualifies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, dataset, checksum = self.fixture(root)
            with (
                patch.object(grader, "DATA_SHA256", checksum),
                patch.object(grader, "isolated_test", return_value=dict(passed=True)),
            ):
                result = grader.grade(source, dataset, root / "grade.json")
            self.assertFalse(result["complete"])
            self.assertEqual(result["graded_tasks"], 1)

    def test_changed_artifact_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, dataset, checksum = self.fixture(root)
            (source / "answer.py").write_text("raise RuntimeError()")
            with (
                patch.object(grader, "DATA_SHA256", checksum),
                patch.object(
                    grader, "isolated_test", return_value=dict(passed=True)
                ) as execute,
            ):
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    grader.grade(source, dataset, root / "grade.json")
            self.assertEqual(execute.call_count, 1)  # only the trusted runner self-test


if __name__ == "__main__":
    unittest.main()
