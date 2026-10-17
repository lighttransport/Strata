"""GLM API boundary regressions; no weights or GPU required."""
import threading
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock
import unittest

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, GlmEngine, Service


class GlmBoundary(unittest.TestCase):
    def test_reap_alias_server_startup(self):
        import serve.server as server
        import gguf_reader
        import strata_tokenizer

        class FakeGlmEngine(GlmEngine):
            def __init__(self, *args, **kwargs):
                self.max_context = 128
                self.launch_args = args
                self.info, self.last = {}, {}
                self.ended, self.unloaded = False, False

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "glm.json"
            model = Path(directory) / "reap.gguf"
            model.touch()                     # the server checks that the model exists before loading
            config.write_text(json.dumps({"exe": "unused", "model": str(model), "prefill_experts": "f16-batched"}))
            metadata = {"general.architecture": "glm5-next", "tokenizer.ggml.eos_token_id": 100,
                        "tokenizer.ggml.tokens": ["ordinary", "<|user|>", "<|observation|>", "<|assistant|>"],
                        "tokenizer.chat_template": "{% for m in messages %}{{m.content}}{% endfor %}"}
            with mock.patch.object(sys, "argv", ["server", "--engine", "glm", "--config", str(config), "--port", "0"]), \
                 mock.patch.object(gguf_reader, "GGUFFile", return_value=mock.Mock(metadata=metadata)), \
                 mock.patch.object(strata_tokenizer.Tokenizer, "from_gguf", return_value=ByteTokenizer()), \
                 mock.patch.object(server, "GlmEngine", FakeGlmEngine), \
                 mock.patch.object(server, "Server"), \
                 mock.patch.object(server, "serve") as http, \
                 mock.patch.object(server.time, "sleep", side_effect=KeyboardInterrupt):
                self.assertEqual(server.main(), 0)
            service = http.call_args.args[0]
            self.assertFalse(service.effort_end)
            self.assertEqual(service.stop_ids, {1, 2, 100})
            self.assertIn("1,2,100", service.engine.launch_args[1])
            self.assertEqual(service.engine.launch_args[1][22], "f16-batched")

    def test_sampling_rejected_before_engine_write(self):
        engine = GlmEngine.__new__(GlmEngine)
        # There is deliberately no process: invalid settings must fail before IO.
        with self.assertRaisesRegex(ValueError, "greedy"):
            list(engine.generate([1], 2, {"temperature": 0.7}, threading.Event()))
        with self.assertRaisesRegex(ValueError, "text"):
            list(engine.generate([1], 2, {}, threading.Event(), embeddings="image"))
        self.assertEqual(GlmEngine.sampling_keys({"temperature": 0, "top_p": 1}), "")

    def test_parallel_glm_admits_directly_without_solo_reprefill(self):
        import serve.server as server
        from serve.test_parallel import FAKE_BATCH
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "fake.py"
            script.write_text(FAKE_BATCH)
            log = Path(directory) / "requests.log"
            real = server.subprocess.Popen
            with mock.patch.object(server.subprocess, "Popen",
                                   lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
                engine = GlmEngine("glm", ["--batch", "2", "--log", str(log)])
            try:
                output = [t for t in engine.generate([10, 11], 4, {}, threading.Event()) if t is not None]
                self.assertEqual(output, list(b"ok, "))
                self.assertTrue(log.read_text().startswith("BGEN "))
                self.assertNotIn("\nGEN ", log.read_text())
            finally:
                engine.close()

    def test_chat_request_metadata_is_not_sampling(self):
        request = {"model": "glm-5.3-flash-q2", "messages": [{"role": "user", "content": "hello"}],
                   "stream": True, "max_tokens": 512, "reasoning_effort": "low", "temperature": 0}
        self.assertEqual(GlmEngine.sampling_keys(request), "")
        for field, value in (("top_p", 0.9), ("seed", 42), ("frequency_penalty", 0.5)):
            with self.assertRaisesRegex(ValueError, "greedy"):
                GlmEngine.sampling_keys({**request, field: value})

    def test_gguf_template_and_stop_ids(self):
        engine = GlmEngine.__new__(GlmEngine)
        engine.max_context = 128
        tok = ByteTokenizer()
        template = ChatTemplate(source="{% for m in messages %}{{ m.content }}{% endfor %}")
        svc = Service(engine, tok, template, stop_ids={154820, 154827, 154829})
        ids, _, _ = svc.prepare([{"role": "user", "content": "ordinary text"}], None, {}, 2)
        self.assertEqual(tok.decode(ids), "ordinary text")
        self.assertEqual(svc.stop_ids, {154820, 154827, 154829})
        with self.assertRaisesRegex(ValueError, "tool-call"):
            svc.prepare([{"role": "user", "content": "hello"}], [{"name": "test"}], {}, 2)


class GlmConfigChecks(unittest.TestCase):
    """Combinations the engine refuses only after loading the weights are refused at startup."""

    def start(self, cfg, extra_files=(), native=False):
        import serve.server as server
        import gguf_reader
        import strata_tokenizer
        import glm_artifact
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "m.gguf"
            model.mkdir() if native else model.touch()
            for name in extra_files:
                (Path(directory) / name).touch()
            cfg = {k: (str(Path(directory) / v) if isinstance(v, str) and v.endswith(".gguf") else v)
                   for k, v in cfg.items()}
            config = Path(directory) / "glm.json"
            config.write_text(json.dumps({"exe": "unused", "model": str(model), **cfg}))
            metadata = {"general.architecture": "glm5next", "tokenizer.ggml.eos_token_id": 1,
                        "tokenizer.ggml.tokens": ["a", "b"], "tokenizer.chat_template": "{{ messages }}"}
            if native: metadata["strata.native_codec"] = "modelopt"
            def engine_start(*args, **kwargs):
                self.launch_args = args
                raise RuntimeError("engine would start")
            with mock.patch.object(sys, "argv", ["server", "--engine", "glm", "--config", str(config), "--port", "0"]), \
                 mock.patch.object(gguf_reader, "GGUFFile", return_value=mock.Mock(metadata=metadata)), \
                 mock.patch.object(strata_tokenizer.Tokenizer, "from_gguf", return_value=ByteTokenizer()), \
                 mock.patch.object(glm_artifact, "load_metadata", return_value=metadata), \
                 mock.patch.object(glm_artifact, "NativeTokenizer", return_value=ByteTokenizer()), \
                 mock.patch.object(server, "Server"), \
                 mock.patch.object(server, "GlmEngine", side_effect=engine_start), \
                 mock.patch("sys.stderr"):
                with self.assertRaises((SystemExit, RuntimeError)) as caught:
                    server.main()
            return caught.exception

    def test_modelopt_kernel_and_exact_mtp(self):
        cfg = {"fp4_kernel": "sm70", "speculative": "mtp", "draft_depth": 3,
               "context": 12288, "prefill_batch": 8192, "gpu_devices": [0, 1]}
        self.assertIsInstance(self.start(cfg, native=True), RuntimeError)
        self.assertIn("--fp4-kernel=sm70", self.launch_args[1])
        self.assertIsInstance(self.start({**cfg, "fp4_kernel": "unknown"}, native=True), SystemExit)
        self.assertIsInstance(self.start({**cfg, "decode_cache_mib": 1024}, native=True), SystemExit)
        self.assertRefused({"fp4_kernel": "sm70"})

    def assertRefused(self, cfg, extra_files=()):
        self.assertIsInstance(self.start(cfg, extra_files), SystemExit, cfg)

    def assertAccepted(self, cfg, extra_files=()):
        e = self.start(cfg, extra_files)
        self.assertIsInstance(e, RuntimeError, cfg)  # reached the engine constructor

    def test_refusals(self):
        self.assertRefused({"expert_pack": "missing-pack.gguf"})
        self.assertRefused({"speculative": "none", "lookup_depth": 3})
        self.assertRefused({"speculative": "lookup", "lookup_depth": 0})
        self.assertRefused({"decode_cache_mib": True})
        self.assertRefused({"decode_cache_mib": 99999})
        self.assertRefused({"decode_cache_mib": 1024, "decode_experts": "gpu"})
        self.assertRefused({"decode_cache_mib": "auto", "prefill_batch": 8})
        self.assertRefused({"speculative": "mtp", "expert_cache_mib": 512})
        self.assertRefused({"parallel": 9})
        self.assertRefused({"parallel": 2, "speculative": "lookup"})
        self.assertRefused({"reasoning_loop_recovery": "recover"})

    def test_preallocated_cache_uses_the_explicit_gpu_budget(self):
        cfg = {"gpu_budget_mib": 32768, "decode_cache_mib": 32000,
               "env": {"STRATA_GLM_VRAM_PREALLOC": "1"}}
        self.assertAccepted(cfg)
        self.assertEqual(self.launch_args[1][18], "32000")
        self.assertRefused({**cfg, "decode_cache_mib": 32769})
        self.assertRefused({**cfg, "gpu_budget_mib": 16384})
        for value in ("0", "2"):
            self.assertRefused({**cfg, "env": {"STRATA_GLM_VRAM_PREALLOC": value}})
        with mock.patch.dict("os.environ", {"STRATA_GLM_VRAM_PREALLOC": "1"}):
            self.assertAccepted({"gpu_budget_mib": 32768, "decode_cache_mib": 32000})
            self.assertRefused({**cfg, "env": {"STRATA_GLM_VRAM_PREALLOC": "0"}})

    def test_layer_layout(self):
        self.assertAccepted({"gpu_layout": "layer", "gpu_devices": [0, 1],
                             "layer_split": [24], "layer_pipeline": "mtp", "speculative": "mtp"})
        args = self.launch_args[1]
        self.assertEqual(args[11], "1")
        self.assertEqual(args[12], "numa")
        self.assertEqual(args[14], "0,1")
        self.assertEqual(args[22:27], ["mmq", "4k", "", "", "auto"])
        self.assertEqual(args[27:], ["--gpu-layout=layer", "--layer-split=24", "--layer-pipeline=mtp"])
        self.assertAccepted({"gpu_layout": "layer", "gpu_devices": [0, 1], "gpu_budget_mib": 32768,
                             "decode_cache_mib": 22528})
        self.assertRefused({"decode_cache_mib": 22528})
        self.assertAccepted({"gpu_layout": "layer", "gpu_devices": [0, 1], "decode_cache_mib": 4096,
                             "decode_cache_adapt": True, "decode_cache_window": 64})
        for cfg in ({"gpu_layout": "wrong"},
                    {"gpu_layout": "layer", "gpu_devices": [0]},
                    {"gpu_layout": "layer", "gpu_devices": [0, 1], "layer_pipeline": "mtp"},
                    {"gpu_layout": "layer", "gpu_devices": [0, 1], "speculative": "mtp", "draft_depth": 3},
                    {"gpu_layout": "layer", "gpu_devices": [0, 1], "decode_cache_adapt": True}):
            self.assertRefused(cfg)

    def test_legacy_ignores_generic_layer_split(self):
        self.assertAccepted({"layer_split": "auto"})
        self.assertEqual(len(self.launch_args[1]), 23)
        self.assertNotIn("--gpu-layout=layer", self.launch_args[1])

    def test_layer_stage_budgets(self):
        cfg=dict(gpu_layout='layer',gpu_devices=[0,1],threads=32,gpu_budget_mib=32768,
                 layer_workers=[16,15],layer_cache_mib=[24576,22528],layer_pin_coordinator=True)
        self.assertAccepted(cfg)
        self.assertIn('--layer-workers=16,15',self.launch_args[1])
        self.assertIn('--layer-cache-mib=24576,22528',self.launch_args[1])
        self.assertIn('--layer-pin-coordinator',self.launch_args[1])
        self.assertAccepted(dict(cfg,decode_cache_adapt=True))
        for changes in (dict(layer_workers=[17,17]),dict(layer_workers=[16,0]),dict(layer_workers=[True,16]),
                        dict(layer_cache_mib=[32769,0]),dict(layer_cache_mib=[1]),dict(layer_cache_mib=10),
                        dict(layer_pin_coordinator='yes'),dict(cpu_affinity='none')):
            self.assertRefused(dict(cfg,**changes))

    def test_accepted(self):
        self.assertAccepted({})
        self.assertAccepted({"speculative": "lookup"})
        self.assertAccepted({"speculative": "mtp", "draft_depth": 3, "decode_cache_mib": "auto", "prefill_batch": 2048})
        self.assertAccepted({"expert_pack": "pack.gguf"}, extra_files=("pack.gguf",))
        self.assertAccepted({"parallel": 4})


class GlmEngineBehaviour(unittest.TestCase):
    def test_greedy_noop_settings_accepted(self):
        llama_defaults = {"temperature": 0, "repeat_penalty": 1, "penalty_last_n": 64, "typical_p": 1,
                          "mirostat": 0, "top_k": -1, "seed": -1, "dyn_temp_range": 0}
        self.assertEqual(GlmEngine.sampling_keys(llama_defaults), "")
        with self.assertRaisesRegex(ValueError, "repeat_penalty"):
            GlmEngine.sampling_keys({"repeat_penalty": 1.1})
        with self.assertRaisesRegex(ValueError, "typical_p"):
            GlmEngine.sampling_keys({"typical_p": 0.9})

    def test_no_yield_and_no_sessions(self):
        from serve.server import SessionRefused
        self.assertEqual(GlmEngine.YIELDS_MAX, 0)
        engine = GlmEngine.__new__(GlmEngine)
        with self.assertRaises(SessionRefused):
            engine.session_file("save", "x.bin")


if __name__ == "__main__":
    unittest.main()
