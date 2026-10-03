"""Resident request diagnostics must distinguish cards and request boundaries."""
import unittest
import contextlib
import io
import json
import hashlib
import os
import pathlib
import tempfile
from unittest import mock

from glm_v100_bench import request_diagnostics
import glm_v100_bench
import glm_v100_sweep


class RequestDiagnostics(unittest.TestCase):
    def test_upstream_kernel_choices_are_recorded(self):
        with mock.patch.dict(os.environ, STRATA_OLD_IQ_MMVQ="1", STRATA_GROUPED_V1="1"):
            recorded = glm_v100_bench.experiment_environment()
        self.assertEqual(recorded["STRATA_OLD_IQ_MMVQ"], "1")
        self.assertEqual(recorded["STRATA_GROUPED_V1"], "1")

    def test_two_cards_and_repeated_requests(self):
        log = """GPU_DEVICE device=0 peak_allocated_MiB=123 live_MiB=123
RESPONSE_PREPARE kind=decode_cache ms=1200.5
CPU_EXPERT gu_ms=10 quant_ms=1 down_ms=5 bytes=1000
GPU_DEVICE device=0 peak_allocated_MiB=31000 live_MiB=30900 cache_hits=0 cuda_used_MiB=32000 free_MiB=768
GPU_DEVICE device=1 peak_allocated_MiB=31300 live_MiB=31200 cache_hits=0 cuda_used_MiB=32100 free_MiB=668
resident expert entries=750/1000
RESPONSE_PREPARE kind=decode_cache ms=1100
CPU_EXPERT gu_ms=9 quant_ms=2 down_ms=4 bytes=900
GPU_DEVICE device=0 peak_allocated_MiB=31000 live_MiB=30800 cache_hits=0
GPU_DEVICE device=1 peak_allocated_MiB=31300 live_MiB=31100 cache_hits=0
resident expert entries=800/1000
"""
        first, second = request_diagnostics(log)
        self.assertEqual(first["cache_prepare_ms"], 1200.5)
        self.assertEqual(first["cpu_expert_bytes"], 1000)
        self.assertEqual(first["gpu_expert_hit_fraction"], .75)
        self.assertEqual([g["device"] for g in first["gpu_devices"]], [0, 1])
        self.assertEqual(first["gpu_devices"][0]["free_mib"], 768)
        self.assertEqual(first["gpu_devices"][1]["cuda_used_mib"], 32100)
        self.assertEqual(second["cpu_expert_ms"]["quantize"], 2)
        self.assertEqual(second["gpu_expert_hit_fraction"], .8)
        self.assertNotIn("free_mib", second["gpu_devices"][0])

    def test_empty_generation_has_no_hit_fraction(self):
        row, = request_diagnostics("RESPONSE_PREPARE kind=decode_cache ms=0\nresident expert entries=0/0\n")
        self.assertIsNone(row["gpu_expert_hit_fraction"])
        self.assertEqual(row["gpu_devices"], [])

    def test_prefill_work_precedes_its_request_boundary(self):
        rows = request_diagnostics("PREFILL_WORK device=0 transferred_bytes=123 staging_ms=1.5 actual_rows=10 padded_rows=20 dequant_values=1000\n"
                                   "RESPONSE_PREPARE kind=decode_cache ms=1\nresident expert entries=1/2\n"
                                   "PREFILL_WORK device=0 transferred_bytes=456 staging_ms=2.5 actual_rows=10 padded_rows=12 dequant_values=500\n"
                                   "RESPONSE_PREPARE kind=decode_cache ms=2\nresident expert entries=2/2\n")
        self.assertEqual(rows[0]["prefill_devices"][0]["transferred_bytes"], 123)
        self.assertEqual(rows[1]["prefill_devices"][0]["padded_rows"], 12)

    def test_restoration_belongs_to_next_request(self):
        rows = request_diagnostics("RESPONSE_PREPARE kind=decode_cache ms=1\n"
                                   "PREFILL_RESTORE groups=20 repaired_slots=5 checked_groups=20 bytes=123 ms=4.5\n"
                                   "RESPONSE_PREPARE kind=request_reset ms=6\n"
                                   "RESPONSE_PREPARE kind=decode_cache ms=2\n")
        self.assertNotIn("prefill_restore", rows[0])
        self.assertEqual(rows[1]["prefill_restore"]["bytes"], 123)
        self.assertEqual(rows[1]["request_reset_ms"], 6)

    def test_deferred_repairs_complete_at_the_next_request(self):
        log = ("RESPONSE_PREPARE kind=decode_cache ms=1\n"
               "PREFILL_RESTORE groups=20 repaired_slots=5 checked_groups=0 bytes=123 ms=0.5 deferred=1\n"
               "RESPONSE_PREPARE kind=request_reset ms=2\n"
               "PREFILL_REPAIR device=0 groups=11 checked_groups=11 wait_ms=0.3\n"
               "PREFILL_REPAIR device=1 groups=9 checked_groups=9 wait_ms=0.2\n"
               "RESPONSE_PREPARE kind=decode_cache ms=3\n")
        first, second = request_diagnostics(log)
        self.assertNotIn("prefill_repairs", first)
        self.assertEqual(second["prefill_restore"]["checked_groups"], 0)
        self.assertEqual(sum(r["checked_groups"] for r in second["prefill_repairs"]), 20)
        self.assertEqual([r["device"] for r in second["prefill_repairs"]], [0, 1])

    def test_tuning_reset_is_outside_following_request(self):
        row, = request_diagnostics("PREFILL_RESTORE groups=20 repaired_slots=5 checked_groups=20 bytes=123 ms=4.5\n"
                                   "TUNE_RESET_COMPLETE\nRESPONSE_PREPARE kind=request_reset ms=1\n"
                                   "RESPONSE_PREPARE kind=decode_cache ms=2\n")
        self.assertNotIn("prefill_restore", row)
        self.assertEqual(row["request_reset_ms"], 1)


class ResidentProgress(unittest.TestCase):
    def run_fake(self, module, *, fail=False, cases=None, boundary=False, sanitizer=None, validate_all=False, prefill_batch=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = pathlib.Path(temporary.name)
        decoder = root / "decoder"
        decoder.write_text('''#!/usr/bin/env python3
import json, pathlib, sys
pathlib.Path(sys.argv[0]).with_suffix('.argv.json').write_text(json.dumps(sys.argv))
print('READY fake', flush=True)
requests = 0
for line in sys.stdin:
    if line.startswith('TUNE '):
        print('TUNED', flush=True)
    elif line.startswith('GEN '):
        requests += 1
        if FAIL and requests == 3:
            print('ERR injected request failure', flush=True)
            sys.exit(7)
        _, count, ids = line.split()
        for token in range(1, int(count) + 1):
            print('T', token, flush=True)
        print('DONE', count, len(ids.split(',')), 10, 20, 'length 0 0 0', flush=True)
    elif line.strip() == 'QUIT':
        break
'''.replace('FAIL', str(fail)))
        decoder.chmod(0o700)
        text = root / "prompt.txt"
        text.write_text("fixture")
        output = root / "result.json"
        argv = ["bench", str(root / "model.gguf"), str(text), "--decoder", str(decoder),
                "--prompt-tokens", "4", "--generated-tokens", "2", "--repetitions", "2",
                "--validation-tokens", "3", "--threads", "5", "--output", str(output)]
        if prefill_batch is not None:
            argv.extend(("--prefill-batch", str(prefill_batch)))
        if cases:
            argv.extend(("--cases", cases))
        if boundary:
            argv.append("--boundary-requests")
        if validate_all:
            argv.extend(("--validation-cases", "all"))
        if sanitizer is not None:
            argv.append("--compute-sanitizer")
            wrapper = root / "compute-sanitizer"
            wrapper.write_text('''#!/usr/bin/env python3
import pathlib, subprocess, sys
log_index = sys.argv.index('--log-file') + 1
rc = subprocess.run(sys.argv[log_index + 1:]).returncode
if MODE != 'missing':
    pathlib.Path(sys.argv[log_index]).write_text('ERROR SUMMARY: %d errors\\n' % (1 if MODE == 'failure' else 0))
if MODE == 'internal':
    with pathlib.Path(sys.argv[log_index]).open('a') as log:
        log.write("Internal Sanitizer Error: Unable to allocate enough memory; errors might go undetected\\n")
sys.exit(97 if MODE == 'failure' else rc)
'''.replace('MODE', repr(sanitizer)))
            wrapper.chmod(0o700)
        tokenizer = mock.Mock()
        tokenizer.encode.return_value = [10, 11, 12, 13]
        with mock.patch("sys.argv", argv), \
             mock.patch.dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"]), \
             mock.patch.object(module, "metadata_shard", side_effect=lambda p: p), \
             mock.patch.object(module.Tokenizer, "from_gguf", return_value=tokenizer), \
             contextlib.redirect_stdout(io.StringIO()):
            if sanitizer in ("failure", "missing", "internal"):
                message = "decoder failed" if sanitizer == "failure" else "sanitizer"
                with self.assertRaisesRegex(RuntimeError, message):
                    module.main()
            elif fail:
                with self.assertRaisesRegex(RuntimeError, "injected request failure"):
                    module.main()
            else:
                module.main()
        self.assertFalse(output.with_suffix(".json.tmp").exists())
        self.assertEqual(json.loads(decoder.with_suffix(".argv.json").read_text())[5], "5")
        if prefill_batch is not None:
            self.assertEqual(json.loads(decoder.with_suffix(".argv.json").read_text())[7], str(prefill_batch))
        result = json.loads(output.read_text())
        self.assertEqual(result["decoder_sha256"], hashlib.sha256(decoder.read_bytes()).hexdigest())
        return result

    def test_sanitizer_wraps_decoder_and_retains_actual_binary_identity(self):
        result = self.run_fake(glm_v100_bench, sanitizer="success")
        self.assertTrue(result["complete"])
        self.assertTrue(result["instrumented"])
        self.assertTrue(result["sanitizer_clean"])
        self.assertEqual(result["command"][:5], ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "97"])

    def test_sanitizer_launch_limit_is_recorded(self):
        with mock.patch.dict(os.environ, STRATA_GLM_SANITIZER_SYNC_LIMIT="1024"):
            result = self.run_fake(glm_v100_bench, sanitizer="success")
        self.assertTrue(result["complete"])
        at = result["command"].index("--force-synchronization-limit")
        self.assertEqual(result["command"][at + 1], "1024")

    def test_sanitizer_failure_prevents_completion_claim(self):
        result = self.run_fake(glm_v100_bench, sanitizer="failure")
        self.assertFalse(result["complete"])
        self.assertTrue(result["instrumented"])
        self.assertEqual(len(result["trials"]), 2)

    def test_missing_sanitizer_summary_prevents_completion_claim(self):
        result = self.run_fake(glm_v100_bench, sanitizer="missing")
        self.assertFalse(result["complete"])
        self.assertTrue(result["instrumented"])

    def test_internal_sanitizer_error_rejects_zero_error_summary(self):
        result = self.run_fake(glm_v100_bench, sanitizer="internal")
        self.assertFalse(result["complete"])
        self.assertFalse(result["sanitizer_clean"])
        self.assertEqual(len(result["trials"]), 2)

    def test_benchmark_preserves_trial_before_failure(self):
        result = self.run_fake(glm_v100_bench, fail=True)
        self.assertFalse(result["complete"])
        self.assertEqual(len(result["trials"]), 1)
        self.assertIn("injected request failure", result["error"])

    def test_sweep_preserves_trial_before_failure(self):
        result = self.run_fake(glm_v100_sweep, fail=True, cases="baseline")
        self.assertFalse(result["complete"])
        self.assertFalse(result["cases"]["baseline"]["complete"])
        self.assertEqual(len(result["cases"]["baseline"]["trials"]), 1)

    def test_completed_benchmark(self):
        result = self.run_fake(glm_v100_bench)
        self.assertTrue(result["complete"])
        self.assertTrue(result["repetitions_identical"])
        self.assertEqual(len(result["trials"]), 2)
        self.assertEqual(len(result["validation_trials"]), 3)
        self.assertGreater(result["median_validation_decode_tok_s"], 0)

    def test_sweep_executes_and_records_selected_prefill_batch(self):
        result = self.run_fake(glm_v100_sweep, cases="baseline", prefill_batch=8192)
        self.assertEqual(result["command"][7], "8192")
        self.assertEqual(result["configuration"]["prefill_batch"], 8192)
        self.assertEqual(result["configuration"]["prompt_tokens"], 4)

    def test_active_pool_only_sweep(self):
        result = self.run_fake(glm_v100_sweep, cases="baseline,active_only")
        self.assertTrue(result["complete"])
        self.assertTrue(all(row["complete"] for row in result["cases"].values()))
        self.assertEqual(result["cases"]["active_only"]["mode"][:3], [0, 0, 1])
        self.assertEqual(result["cases"]["baseline"]["mode"][4], 3)
        self.assertEqual(len(result["validation_trials"]), 3)

    def test_bucket_sweep_enables_stable_routes_and_keeps_stock_kda(self):
        result = self.run_fake(glm_v100_sweep, cases="stable,buckets,buckets_active")
        self.assertTrue(result["complete"])
        self.assertEqual(result["cases"]["stable"]["mode"][-3:], [128, 1, 0])
        self.assertEqual(result["cases"]["buckets"]["mode"][-3:], [128, 1, 1])
        self.assertEqual(result["cases"]["buckets_active"]["mode"][2], 1)

    def test_schedule_sweep_preserves_arithmetic_options(self):
        result = self.run_fake(glm_v100_sweep, cases="baseline,prealloc,defer,prealloc_defer,scheduled_active", boundary=True)
        self.assertTrue(result["complete"])
        self.assertEqual(result["cases"]["prealloc"]["mode"][-2:], [1, 0])
        self.assertEqual(result["cases"]["defer"]["mode"][-2:], [0, 1])
        self.assertEqual(result["cases"]["prealloc_defer"]["mode"][7:], [128, 0, 0, 1, 1])
        self.assertEqual([r["prompt_tokens"] for r in result["boundary_requests"]], [1, 3, 4])
        self.assertTrue(all(r["complete"] for r in result["boundary_requests"]))

    def test_every_candidate_gets_repeated_long_validation(self):
        result = self.run_fake(glm_v100_sweep, cases="baseline,row4,row8,restore_row4", validate_all=True)
        self.assertTrue(result["complete"])
        self.assertEqual(set(result["validation_by_case"]), set(result["cases"]))
        for row in result["validation_by_case"].values():
            self.assertTrue(row["complete"])
            self.assertEqual(len(row["trials"]), 3)
        self.assertEqual(result["cases"]["row4"]["mode"][-1], 4)
        self.assertEqual(result["cases"]["row8"]["mode"][-1], 8)
        self.assertEqual(result["cases"]["restore_row4"]["mode"][-2:], [1, 4])
        self.assertEqual(result["validation_trials"], result["validation_by_case"][result["validation_case"]]["trials"])

    def test_opt_in_upload_environment_is_saved(self):
        with mock.patch.dict(os.environ, STRATA_GLM_DIRECT_WEIGHT_UPLOAD="1", STRATA_GLM_STAGE_WORKERS="8"):
            for module in (glm_v100_bench, glm_v100_sweep):
                result = self.run_fake(module, cases="baseline" if module is glm_v100_sweep else None)
                self.assertEqual(result["environment"]["STRATA_GLM_DIRECT_WEIGHT_UPLOAD"], "1")
                self.assertEqual(result["environment"]["STRATA_GLM_STAGE_WORKERS"], "8")


if __name__ == "__main__":
    unittest.main()
