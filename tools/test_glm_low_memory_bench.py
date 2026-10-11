from pathlib import Path
import tempfile
import unittest

from glm_low_memory_bench import (
    cache_bytes, eval_counts, eval_complete, gpu_capacity_passed, gpu_used_limit,
    gpu_snapshot, gpu_usage, group_stats, hip_gpu_snapshot, model_files, parse_log,
    prefill_repeats_passed, validate_hip_link, decode_complete, decode_trials_identical,
)


class LowMemoryBenchTest(unittest.TestCase):
    def test_determinism_uses_eos_trial_boundaries(self):
        rows = [dict(kind='single', tokens=2)] * 2
        self.assertTrue(decode_trials_identical([1, 2, 9, 1, 2, 9], rows))
        self.assertFalse(decode_trials_identical([1, 2, 9, 1, 3, 9], rows))
        self.assertFalse(decode_trials_identical([1, 2, 9, 1, 2], rows))
        self.assertFalse(decode_trials_identical([1, 2, 9, 1, 2, 9, 9], rows))
        self.assertFalse(decode_trials_identical([], []))
        self.assertFalse(decode_trials_identical([1, 2, 9, 1, 9],
                         [dict(kind='single', tokens=2), dict(kind='single', tokens=1)]))

    def test_prefill_repeat_check_requires_all_trials_and_both_invariants(self):
        rows = parse_log("PREFILL_REPEAT trial=0 logits_identical=1 routes_identical=1\n"
                         "PREFILL_REPEAT trial=1 logits_identical=1 routes_identical=1\n")["prefill_repeat"]
        self.assertTrue(prefill_repeats_passed(rows, 2))
        self.assertFalse(prefill_repeats_passed([], 2))
        self.assertFalse(prefill_repeats_passed(rows[:1], 2))
        self.assertFalse(prefill_repeats_passed(rows[::-1], 2))
        self.assertFalse(prefill_repeats_passed([rows[0], rows[0]], 2))
        self.assertFalse(prefill_repeats_passed(rows[:1], 1))
        for key in ("logits_identical", "routes_identical"):
            self.assertFalse(prefill_repeats_passed([rows[0], dict(rows[1], **{key: 0})], 2))
            missing = dict(rows[1])
            missing.pop(key)
            self.assertFalse(prefill_repeats_passed([rows[0], missing], 2))

    def test_pci_preflight_rejects_cached_memory_after_bus_loss(self):
        with tempfile.TemporaryDirectory() as folder:
            device=Path(folder)
            (device/'config').write_bytes(bytes.fromhex('02105075'))
            validate_hip_link(device)
            (device/'mem_info_vram_total').write_text(str(16*2**30))
            (device/'mem_info_vram_used').write_text(str(2**30))
            self.assertNotIn('performance',hip_gpu_snapshot(device,False,True))
            (device/'config').write_bytes(b'\xff'*4)
            with self.assertRaises(ValueError):hip_gpu_snapshot(device,False,True)
            (device/'config').write_bytes(b'\x02')
            with self.assertRaises(ValueError):validate_hip_link(device)

    def test_eos_completion_and_partial_output(self):
        rows=[dict(kind='single',tokens=2),dict(kind='external',tokens=4)]
        self.assertTrue(decode_complete([1,2,9,1,2,3,4],rows,4,2,[9]))
        self.assertFalse(decode_complete([1,2,3,1,2,3,4],rows,4,2,[9]))
        self.assertFalse(decode_complete([1,9,9,1,2,3,4],rows,4,2,[9]))
        self.assertFalse(decode_complete([1,2,9,1,2,3],rows,4,2,[9]))
        self.assertFalse(decode_complete([1,2,9,1,2,3,4,5],rows,4,2,[9]))

    def test_timing_scopes_and_missing_measurements(self):
        log = """PREFILL trial=0 tokens=128 batch=64 ms=80000.5 tok_s=1.6
PREFILL trial=1 tokens=128 batch=64 ms=64000 tok_s=2
FAULTS trial=0 major=500 minor=1000
DECODE_FAULTS trial=0 major=0 minor=3
DECODE steps=7 ms=7000 tok_s=1
GPU peak_allocated_MiB=7429.27
GPU_LIVE device=0 dense_MiB=6153.68 dense_mtp_MiB=220.84 other_MiB=90
"""
        result = parse_log(log)
        self.assertEqual(result["median_prefill_tok_s"], 1.8)
        self.assertEqual(result["decode"][0]["tokens"], 7)
        self.assertEqual(result["gpu_live"][0]["tags"]["dense_mtp"], 220.84)
        self.assertEqual(result["page_faults"][0]["major"], 500)
        self.assertEqual(len(result["page_faults"]), 1)
        self.assertEqual(result["decode_page_faults"][0]["major"], 0)
        failed = parse_log("GLM: prefill cannot fit GPU budget")
        self.assertIsNone(failed["median_prefill_tok_s"])
        self.assertIsNone(failed["median_decode_tok_s"])
        legacy = parse_log("GPU_DEVICE device=0 peak_allocated_MiB=4292.01 live_MiB=4269.96")
        self.assertEqual(legacy["peak_allocated_gpu_mib"], 4292.01)

    def test_speculative_delivered_rate(self):
        result = parse_log("SPECULATIVE source=mtp generated=64 rounds=40 proposed=80 "
                           "accepted=24 replayed=0 ms=50000 tok_s=1.28")
        self.assertEqual(result["decode"][0], dict(kind="mtp", tokens=64,
                          milliseconds=50000, tokens_per_second=1.28))

    def test_gpu_guard_counts_only_owned_process(self):
        self.assertEqual(gpu_usage(12, "1, 130\n12, 7976\n19, [N/A]\n"), 7976)
        self.assertEqual(gpu_usage(12, "1, 130\n"), 0)
        with self.assertRaises(ValueError):
            gpu_usage(12, "12, [N/A]\n")

    def test_capacity_includes_driver_desktop_and_free_reserve(self):
        self.assertFalse(gpu_capacity_passed(8192, 8014, 9275, 512))
        self.assertFalse(gpu_capacity_passed(8192, 8104, 8104, 512))
        self.assertTrue(gpu_capacity_passed(8192, 4696, 5971, 512))
        self.assertFalse(gpu_capacity_passed(8192, 8000, 1256, 512))

    def test_gpu_margin_is_not_subtracted_twice(self):
        self.assertEqual(gpu_used_limit(16384, 16311, 2048, 14336), (16311, 14263))
        self.assertEqual(gpu_used_limit(8192, 16311, 512), (8192, 7680))
        self.assertEqual(gpu_used_limit(16384, 16311, 2048, 13000), (16311, 13000))
        with self.assertRaises(ValueError):
            gpu_used_limit(1024, 16311, 2048)

    def test_reserved_gpu_memory_is_not_available(self):
        snapshot = gpu_snapshot("16311, 404, 14653, 1255\n")
        self.assertEqual(snapshot["usable_mib"], 15907)
        self.assertEqual(gpu_used_limit(16384, snapshot["total_mib"], 2048, 14336, 404), (15907, 13859))
        self.assertEqual(gpu_used_limit(8192, 16311, 512, reserved_mib=404), (7788, 7276))
        self.assertEqual(gpu_used_limit(16384, 16311, 2048, 13000, 404), (15907, 12596))
        with self.assertRaises(ValueError):
            gpu_snapshot("16311, 16311, 0, 0")

    def test_amd_global_vram_guard_and_rounding(self):
        with tempfile.TemporaryDirectory() as folder:
            device = Path(folder)
            (device / "mem_info_vram_total").write_text(str(16 * 2**30))
            (device / "mem_info_vram_used").write_text(str(14 * 2**30 + 1))
            (device / "mem_info_gtt_used").write_text(str(2**20 + 1))
            (device / "power_dpm_force_performance_level").write_text("manual\n")
            (device / "pp_dpm_mclk").write_text("0: 96Mhz *\n1: 1258Mhz\n")
            snapshot = hip_gpu_snapshot(device)
            self.assertEqual(snapshot["performance"]["power_dpm_force_performance_level"], "manual")
            self.assertIn("96Mhz *", snapshot["performance"]["pp_dpm_mclk"])
            self.assertEqual(snapshot["used_mib"], 14337)
            self.assertEqual(snapshot["free_mib"], 2047)
            self.assertEqual(snapshot["gtt_used_mib"], 2)
            _, limit = gpu_used_limit(16384, snapshot["total_mib"], 2048)
            self.assertGreater(snapshot["used_mib"], limit)
            (device / "mem_info_vram_used").write_text(str(16 * 2**30 + 1))
            with self.assertRaises(ValueError):
                hip_gpu_snapshot(device)

    def test_quality_run_completeness(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "eval.ids"
            path.write_text("1,2,3\n\n4,5,6,7\n")
            counts = eval_counts(path)
            self.assertEqual(counts, [2, 5])
            rows = [dict(sequence=1, tokens=2), dict(sequence=2, tokens=5)]
            self.assertTrue(eval_complete(rows, counts))
            self.assertFalse(eval_complete(rows[:1], counts))
            self.assertFalse(eval_complete(rows[::-1], counts))
            self.assertFalse(eval_complete([dict(sequence=1, tokens=2), dict(sequence=2, tokens=4)], counts))

    def test_decode_only_quality_completeness(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "eval.ids"
            path.write_text("1,2,3,4,5\n6,7,8,9\n")
            counts = eval_counts(path, skip_prefix=2)
            self.assertEqual(counts, [2, 3])
            rows = [dict(sequence=1, tokens=2, decode_only=True, prefix_tokens=2),
                    dict(sequence=2, tokens=3, decode_only=True, prefix_tokens=2)]
            self.assertTrue(eval_complete(rows, counts, decode_prefix=2))
            self.assertFalse(eval_complete(rows[:1], counts, decode_prefix=2))
            self.assertFalse(eval_complete(rows, counts, decode_prefix=1))
            self.assertFalse(eval_complete([dict(sequence=1, tokens=2), rows[1]], counts, decode_prefix=2))
            with self.assertRaises(ValueError):
                eval_counts(path, skip_prefix=3)

    def test_cache_inventory_excludes_other_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            names = ["GLM-00001-of-00002.gguf", "GLM-00002-of-00002.gguf", "OTHER-00001-of-00002.gguf"]
            for name in names:
                (root / name).touch()
            (root / "pack.gguf").touch()
            (root / "pack-link.gguf").symlink_to(root / "pack.gguf")
            files = model_files(dict(model=str(root / names[0]), expert_pack=str(root / "pack-link.gguf")))
            self.assertEqual(files, [root / names[0], root / names[1], root / "pack.gguf"])

    def test_mincore_handles_empty_and_partial_pages(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "file"
            path.touch()
            self.assertEqual(cache_bytes(path), 0)
            path.write_bytes(b"x" * 4101)
            self.assertEqual(cache_bytes(path), 4101)

    def test_cgroup_limits_and_optional_io(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, value in {"memory.current": 100, "memory.peak": 200, "memory.swap.current": 0,
                                "memory.max": 28 * 2**30, "memory.swap.max": 0}.items():
                (root / name).write_text(str(value))
            (root / "memory.events").write_text("max 15\noom_kill 0\n")
            (root / "memory.stat").write_text("anon 40\nfile 60\n")
            self.assertEqual(group_stats(root)["memory.max"], 28 * 2**30)
            self.assertEqual(group_stats(root)["io.stat"], {})
            (root / "io.stat").write_text("259:0 rbytes=100 wbytes=0 rios=20\n")
            self.assertEqual(group_stats(root)["io.stat"]["259:0"]["rbytes"], 100)


if __name__ == "__main__":
    unittest.main()
