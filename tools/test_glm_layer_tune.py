"""Placement selection rejects incomplete and incomparable timing evidence."""

import copy
import pathlib
import tempfile
import unittest
from glm_layer_tune import select_plan, MATCH


class LayerPlan(unittest.TestCase):
    def test_v2_binds_stage_budgets_and_coordinator_affinity(self):
        with tempfile.TemporaryDirectory() as directory:
            model = pathlib.Path(directory) / "m.gguf"
            model.write_bytes(b"fixture")
            rows = [self.fixture(model, "23", 100), self.fixture(model, "24", 110)]
            for row in rows:
                row["configuration"].update(
                    layer_workers="16,15",
                    layer_cache_mib="24576,22528",
                    layer_pin_coordinator=True,
                )
            plan = select_plan(rows, model)
            self.assertEqual(plan["schema"], "strata.glm.layer-plan.v2")
            self.assertEqual(plan["configuration"]["layer_workers"], "16,15")
            rows[1]["configuration"]["layer_cache_mib"] = "22528,24576"
            with self.assertRaises(ValueError):
                select_plan(rows, model)

    def fixture(self, model, split, ms):
        cfg = dict(
            context=32768,
            threads=32,
            gpu_devices="0,1",
            gpu_budget_mib=32768,
            decode_cache_mib="22528",
            decode_cache_window=256,
            decode_cache_adapt=False,
            decode_graphs=True,
            prefill_batch=256,
            speculative="mtp",
            draft_depth=1,
            layer_pipeline="mtp",
            layer_memory="socket",
            gpu_layout="layer",
            layer_split=split,
            model=str(model),
        )
        return dict(
            configuration=cfg,
            complete=True,
            instrumented=False,
            repetitions_identical=True,
            decoder_sha256="abc",
            decoder_fnv64=123,
            prompt_sha256="def",
            environment={},
            hardware={
                "cpu": "Xeon",
                "gpus": ["V100", "V100"],
                "node": "node",
                "job_id": "1",
            },
            trials=[
                dict(token_ids=[1] * 64, decode_ms=ms),
                dict(token_ids=[1] * 64, decode_ms=ms + 1),
            ],
        )

    def test_measured_choice_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            model = pathlib.Path(directory) / "m.gguf"
            model.write_bytes(b"fixture")
            records = [self.fixture(model, "24", 100), self.fixture(model, "26", 120)]
            plan = select_plan(records, model)
            self.assertEqual(plan["layer_split"], "24")
            self.assertEqual(plan["model_bytes"], 7)
            self.assertEqual(plan["decoder_fnv64"], 123)
            for field in MATCH:
                broken = copy.deepcopy(records)
                broken[1]["configuration"][field] = "changed"
                with self.assertRaises(ValueError):
                    select_plan(broken, model)
            for field, value in (
                ("complete", False),
                ("instrumented", True),
                ("repetitions_identical", False),
                ("decoder_sha256", "different"),
                ("decoder_fnv64", 2),
                ("environment", {"STRATA_GLM_CANON": "1"}),
                ("hardware", {}),
                ("prompt_sha256", "different"),
            ):
                broken = copy.deepcopy(records)
                broken[1][field] = value
                with self.assertRaises(ValueError):
                    select_plan(broken, model)
            with self.assertRaises(ValueError):
                select_plan(records[:1], model)
            with self.assertRaises(ValueError):
                select_plan([records[0], records[0]], model)
            broken = copy.deepcopy(records)
            broken[1]["configuration"]["layer_split"] = "0"
            with self.assertRaises(ValueError):
                select_plan(broken, model)


if __name__ == "__main__":
    unittest.main()
