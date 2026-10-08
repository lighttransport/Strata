import math
import json
import unittest
from glm_block_screen import splits, summary, allocate_layers, rows_from_stdout, block_reports, read_capture
from glm_q2_run_guard import gpu_free_mib
from unittest.mock import patch
from pathlib import Path
import tempfile
import struct


def metrics(ppl, different=None):
    rows = [{"sequence": 1, "tokens": 256, "nll": math.log(ppl), "perplexity": ppl},
            {"sequence": 2, "tokens": 512, "nll": math.log(ppl), "perplexity": ppl}]
    if different is not None:
        for row in rows:
            row.update(different_logits=different, top1_agreement=1, kl=0)
    return rows


class BlockScreen(unittest.TestCase):
    def test_capture_stream_and_truncation(self):
        raw = struct.pack("<8I", 0x31434247, 1, 1, 3, 0, 1, 4096, 1)
        raw += struct.pack("<if", 7, .5) + struct.pack("<4096f", *([1.] * 4096))
        raw += struct.pack("<8I", 0x31434247, 2, 1, 3, 0, 7, 128, 0) + struct.pack("<128f", *([2.] * 128))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace"
            path.write_bytes(raw)
            records = list(read_capture(path))
            self.assertEqual(records[0]["experts"], (7,))
            self.assertEqual(records[1]["energies"], (2.,) * 128)
            path.write_bytes(raw[:-1])
            with self.assertRaises(ValueError):
                list(read_capture(path))
    def test_gpu_probe(self):
        with patch("glm_q2_run_guard.subprocess.check_output", return_value="1024\n"):
            self.assertEqual(gpu_free_mib(0), 1024)
        for reading in ("bad", "-1"):
            with patch("glm_q2_run_guard.subprocess.check_output", return_value=reading), self.assertRaises(ValueError):
                gpu_free_mib(0)

    def test_layer_budget_and_reports(self):
        reports = [{3: {"full_expert_bytes": 100, "ideal_union_bytes": 50},
                    4: {"full_expert_bytes": 100, "ideal_union_bytes": 100}},
                   {3: {"full_expert_bytes": 100, "ideal_union_bytes": 100},
                    4: {"full_expert_bytes": 100, "ideal_union_bytes": 50}}]
        allocation = allocate_layers(metrics(8), {"3:64": metrics(8.001), "4:64": metrics(8.5)}, reports, 2)
        self.assertEqual(allocation["layers"], {"3": 64})
        self.assertEqual(allocation["ideal_saved_fraction"], .25)
        self.assertTrue(allocation["requires_combined_validation"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records"
            path.write_text('{"block_budget":64,"block_layer":3}\n' + '\n'.join(json.dumps(row) for row in metrics(8)))
            self.assertEqual(list(rows_from_stdout(path)), ["3:64"])
            path.write_text("BLOCK_SCREEN layer=3 rows=8 full_expert_bytes=100 ideal_union_bytes=50\nBLOCK_SCREEN_TOTAL rows=8\n")
            self.assertEqual(block_reports(path)[0][3]["ideal_union_bytes"], 50)

    def test_split_sizes_and_overlap(self):
        calibration = [[i] * 513 for i in range(64)]
        evaluation = [[i] * 513 for i in range(64, 128)]
        result = splits(calibration, evaluation)
        self.assertEqual([len(result[name]) for name in ("fit", "tune", "smoke", "evaluation")], [48, 16, 2, 64])
        self.assertEqual(sum(len(row) - 1 for row in result["fit"]), 24576)
        self.assertEqual(sum(len(row) - 1 for row in result["tune"]), 8192)
        self.assertFalse({tuple(row) for row in result["fit"]} & {tuple(row) for row in result["tune"]})
        with self.assertRaises(ValueError):
            splits(calibration, calibration)

    def test_quality_gate_and_control(self):
        valid_guard = {"exit_code": 0, "peak_swap_kib": 0}
        result = summary(metrics(8), metrics(8, 0), {64: metrics(8.04), 32: metrics(9)}, 2, [valid_guard])
        self.assertTrue(result["candidates"]["64"]["quality_screen_pass"])
        self.assertFalse(result["candidates"]["32"]["quality_screen_pass"])
        self.assertFalse(result["qualified"])
        self.assertTrue(result["diagnostic_only"])
        for control, guard in ((metrics(8, 1), valid_guard), (metrics(8, 0), {"exit_code": 0, "peak_swap_kib": 1}),
                               (metrics(8, 0), {"exit_code": 1})):
            with self.assertRaises(ValueError):
                summary(metrics(8), control, {}, 2, [guard])

    def test_prose_regression_cannot_hide_in_average(self):
        candidate = metrics(8)
        candidate[-1].update(nll=(math.log(7.9) + math.log(8.2)) / 2,
                             perplexity=math.sqrt(7.9 * 8.2))
        candidate[0].update(nll=math.log(7.9), perplexity=7.9)
        result = summary(metrics(8), metrics(8, 0), {64: candidate}, 2, [])
        self.assertLess(result["candidates"]["64"]["perplexity_ratios"]["all"], 1.01)
        self.assertFalse(result["candidates"]["64"]["quality_screen_pass"])
        with self.assertRaises(ValueError):
            summary(metrics(8), metrics(8, 0), {64: candidate[:1]}, 2, [])


if __name__ == "__main__":
    unittest.main()
