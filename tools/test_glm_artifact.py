"""Native tokenizer adapter checks; optionally uses the local pack."""
import os
import pathlib
import unittest

from glm_artifact import NativeTokenizer, load_metadata

MODEL = pathlib.Path(os.environ.get("STRATA_EXL3_MODEL", "/mnt/nvme01/models/glm53f/glm53f-exl"))


@unittest.skipUnless((MODEL / "tokenizer.json").exists(), "native tokenizer fixture not present")
class NativeTokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = NativeTokenizer(MODEL)

    def test_round_trip_and_streaming_bytes(self):
        for text in ("Hello 世界", "日本語 😀 café e\u0301", "a\n\t  b", "\x00code()", "[gMASK]<|user|>"):
            ids = self.tokenizer.encode(text, parse_special=True)
            self.assertEqual(self.tokenizer.decode(ids), text)
            self.assertEqual(b"".join(self.tokenizer.token_bytes(i) for i in ids).decode("utf-8"), text)

    def test_special_opt_in(self):
        token, identifier = next(iter(self.tokenizer.special_ids.items()))
        self.assertEqual(self.tokenizer.encode(token, parse_special=True), [identifier])
        self.assertNotEqual(self.tokenizer.encode(token, parse_special=False), [identifier])
        self.assertEqual(self.tokenizer.decode([identifier]), token)

    def test_metadata(self):
        metadata = load_metadata(MODEL)
        self.assertEqual(metadata["glm5next.context_length"], 1048576)
        self.assertEqual(metadata["tokenizer.ggml.eos_token_id"], 154820)
        self.assertIn("messages", metadata["tokenizer.chat_template"])


if __name__ == "__main__":
    unittest.main()
