"""Validate repeated decode measurements without loading model weights."""
import subprocess
import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from glm_decode_bench import main, measure


class DecodeMeasurementTest(unittest.TestCase):
    command = ["decoder", "model.gguf", "1,2", "3"]

    def run_measurement(self, stdout, stderr):
        result = subprocess.CompletedProcess(self.command, 0, stdout, stderr)
        with patch("glm_decode_bench.subprocess.run", return_value=result):
            return measure(self.command)

    def test_speculative_median_and_repeated_ids(self):
        log = "GPU peak_allocated_MiB=9000\n"
        for i, rate in enumerate((4, 12, 6)):
            log += (f"DECODE_TRIAL index={i}\n"
                    f"SPECULATIVE source=mtp generated=3 rounds=1 proposed=2 "
                    f"accepted=1 replayed=0 ms=100 tok_s={rate}\n")
        result = self.run_measurement("1 2 3 " * 3, log)
        self.assertEqual(result["tokens_per_second"], 6)
        self.assertEqual(result["token_ids"], [1, 2, 3])
        self.assertEqual(result["peak_allocated_mib"], 9000)
        self.assertEqual(len(result["trials"]), 3)
        self.assertEqual(result["trials"][0]["replayed"], 0)

    def test_changed_greedy_ids_are_rejected(self):
        log = "".join(f"DECODE_TRIAL index={i}\nDECODE steps=2 ms=100 tok_s=20\n" for i in range(2))
        with self.assertRaisesRegex(RuntimeError, "greedy output changed"):
            self.run_measurement("1 2 3 1 2 4", log)

    def test_partial_run_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "missing decode measurements"):
            self.run_measurement("1 2 3", "DECODE_TRIAL index=0\n")
        with self.assertRaisesRegex(RuntimeError, "unexpected benchmark token count"):
            self.run_measurement("1 2", "DECODE_TRIAL index=0\nDECODE steps=2 ms=100 tok_s=20\n")

    def run_sweep(self, measurements, mode_flags=None):
        with tempfile.TemporaryDirectory() as directory:
            text = pathlib.Path(directory) / "prompt.txt"
            text.write_text("coding prompt")
            args = ["bench", "model.gguf", str(text), "--prompt-tokens=2",
                    "--generated-tokens=3"] + (mode_flags if mode_flags is not None else ["--draft-depths=1,3"])
            output = io.StringIO()
            with patch("sys.argv", args), patch("glm_decode_bench.metadata_shard"), \
                    patch("glm_decode_bench.Tokenizer.from_gguf") as tokenizer, \
                    patch("glm_decode_bench.measure", side_effect=measurements), \
                    contextlib.redirect_stdout(output):
                tokenizer.return_value.encode.return_value = [1, 2]
                main()
            return json.loads(output.getvalue())

    def test_sweep_selects_measured_winner(self):
        runs = [dict(token_ids=[1, 2, 3], tokens_per_second=rate) for rate in (6, 8, 7)]
        result = self.run_sweep(runs)
        self.assertEqual(result["draft_depth"], 1)
        self.assertEqual(result["speculative"]["tokens_per_second"], 8)
        self.assertEqual(len(result["mtp_depth_sweep"]), 2)

    def test_sweep_rejects_incorrect_losing_variant(self):
        runs = [dict(token_ids=[1, 2, 3], tokens_per_second=rate) for rate in (6, 8, 7)]
        runs[-1]["token_ids"] = [1, 2, 4]
        with self.assertRaisesRegex(RuntimeError, "at depth 3"):
            self.run_sweep(runs)

    def test_cached_single_run_has_no_speculative_claim(self):
        result = self.run_sweep([dict(token_ids=[1, 2, 3], tokens_per_second=9)],
                                ["--single-only", "--decode-cache-mib=256"])
        self.assertIsNone(result["speculative"])
        self.assertIsNone(result["speedup"])
        self.assertEqual(result["decode_cache_mib"], 256)
        self.assertEqual(result["greedy_comparison_scope"], "between repetitions only")

    def test_cached_cpu_mtp_requires_identical_resident_set(self):
        cache = dict(slots=32, allocated_mib=251.375, fingerprint="12345")
        runs = [dict(token_ids=[1, 2, 3], tokens_per_second=rate,
                     actual_decode_cache=cache.copy()) for rate in (6, 8)]
        flags = ["--decode-cache-mib=256", "--mtp-experts=cpu", "--draft-depth=1"]
        result = self.run_sweep(runs, flags)
        self.assertEqual(result["mtp_experts"], "cpu")
        self.assertEqual(result["speculative"]["actual_decode_cache"], cache)
        runs[1]["actual_decode_cache"]["fingerprint"] = "54321"
        with self.assertRaisesRegex(RuntimeError, "resident expert cache differs"):
            self.run_sweep(runs, flags)


if __name__ == "__main__":
    unittest.main()
