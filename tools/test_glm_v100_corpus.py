import unittest
from pathlib import Path
from glm_v100_corpus import evaluation_command


class EvaluationCommandTest(unittest.TestCase):
    def record(self):
        return dict(
            configuration=dict(
                decoder="/tmp/decode",
                model="/tmp/full.gguf",
                threads=32,
                context=32768,
                prefill_batch=4096,
                gpu_budget_mib=32768,
                gpu_devices="0,1",
                decode_cache_mib=22528,
                gpu_layout="layer",
                layer_split="23",
                layer_workers="16,15",
                layer_cache_mib="24576,22528",
                layer_pin_coordinator=True,
            ),
            environment={"STRATA_GLM_Q8_DECODE": "1"},
        )

    def test_layer_target_rows_keep_placement_and_clear_prefill_cache(self):
        command, env, width = evaluation_command(
            self.record(), Path("/tmp/corpus"), Path("/tmp/reference")
        )
        self.assertEqual(width, 1)
        self.assertIn("--layer-workers=16,15", command)
        self.assertIn("--layer-pin-coordinator", command)
        self.assertIn("--layer-pipeline=none", command)
        self.assertNotIn("--eval-verify", command)
        self.assertEqual(env["STRATA_GLM_EVAL_WIDTH"], "1")
        command, _, _ = evaluation_command(
            self.record(), Path("/tmp/corpus"), Path("/tmp/reference"), prefill=True
        )
        self.assertIn("--layer-cache-mib=0,0", command)
        self.assertIn("--decode-cache-mib=0", command)
        self.assertIn("--eval-prefill", command)

    def test_worker_scores_real_verification_arithmetic(self):
        record = self.record()
        record["configuration"].update(
            gpu_layout="legacy",
            gpu_devices="0",
            draft_depth=3,
            worker_decoder="/tmp/worker",
            worker_combine="native",
            primary_cpus="0-16",
            worker_cpus="18",
        )
        command, env, width = evaluation_command(
            record, Path("/tmp/corpus"), Path("/tmp/reference")
        )
        self.assertEqual(command[:2], ["numactl", "--interleave=all"])
        self.assertEqual(width, 4)
        self.assertIn("--eval-verify", command)
        self.assertIn("native", command)
        self.assertIn("0-16", command)
        self.assertIn("--primary-device", command)
        self.assertEqual(env["STRATA_GLM_Q8_DECODE"], "1")


if __name__ == "__main__":
    unittest.main()
