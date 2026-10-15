"""Allocation visibility and role affinity checks; no GPU runtime needed."""

import json
import os
import unittest
from unittest.mock import patch

from glm_mixed_run import cpu_mask, cuda_child_environments, shared_memory


class PlacementTest(unittest.TestCase):
    def test_registered_mapping_uses_anonymous_shared_pages(self):
        with (
            patch.dict(os.environ, {"STRATA_GLM_MIXED_REGISTER": "1"}),
            shared_memory() as memory,
        ):
            self.assertIn(
                "memfd:strata-mixed", os.readlink(f"/proc/self/fd/{memory.fileno()}")
            )
            memory.truncate(8193)
            self.assertEqual(os.fstat(memory.fileno()).st_size, 8193)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("glm_mixed_run.tempfile.TemporaryFile") as temporary,
        ):
            shared_memory()
            temporary.assert_called_once()

    def devices(self):
        return [
            dict(ordinal=0, uuid="GPU-first", pci_bus_id="0000:17:00.0"),
            dict(ordinal=1, uuid="GPU-second", pci_bus_id="0000:65:00.0"),
        ]

    def test_children_use_resolved_uuids_and_preserve_parent_mask(self):
        env = dict(CUDA_VISIBLE_DEVICES="4,7", SLURM_JOB_ID="123")
        with patch(
            "glm_mixed_run.subprocess.check_output",
            return_value=json.dumps(self.devices()),
        ) as query:
            primary, worker, selected = cuda_child_environments(
                "decode", "worker", 1, 0, env
            )
        self.assertEqual(primary["CUDA_VISIBLE_DEVICES"], "GPU-second")
        self.assertEqual(worker["CUDA_VISIBLE_DEVICES"], "GPU-first")
        self.assertEqual(primary["SLURM_JOB_ID"], "123")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "4,7")
        self.assertEqual(selected[0]["ordinal"], 1)
        self.assertEqual(query.call_count, 2)
        self.assertTrue(all(call.kwargs["env"] == env for call in query.call_args_list))

    def test_requires_two_explicit_distinct_visible_devices(self):
        for pair in [(None, 1), (0, None), (0, 0), (-1, 1)]:
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                cuda_child_environments("decode", "worker", *pair, {})
        with patch(
            "glm_mixed_run.subprocess.check_output",
            return_value=json.dumps(self.devices()),
        ):
            with self.assertRaises(ValueError):
                cuda_child_environments("decode", "worker", 0, 2, {})

    def test_rejects_aliases_and_disagreeing_binary_views(self):
        aliased = self.devices()
        aliased[1]["uuid"] = aliased[0]["uuid"]
        with patch(
            "glm_mixed_run.subprocess.check_output", return_value=json.dumps(aliased)
        ):
            with self.assertRaises(ValueError):
                cuda_child_environments("decode", "worker", 0, 1, {})
        other = self.devices()
        other.reverse()
        other[0]["ordinal"], other[1]["ordinal"] = 0, 1
        with patch(
            "glm_mixed_run.subprocess.check_output",
            side_effect=[json.dumps(self.devices()), json.dumps(other)],
        ):
            with self.assertRaises(ValueError):
                cuda_child_environments("decode", "worker", 0, 1, {})

    def test_cpu_masks_cannot_expand_allocation_affinity(self):
        self.assertEqual(cpu_mask("3-5,1,3", {1, 3, 4, 5}), "1,3,4,5")
        for value in ("", "-1", "5-3", "1,,3", "1-999999999", "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cpu_mask(value, {1, 3, 4, 5})


if __name__ == "__main__":
    unittest.main()
