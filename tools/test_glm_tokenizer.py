"""GLM4 pre-tokenization boundaries and byte-level round trips, without weights."""
import unittest

from strata_tokenizer import Tokenizer, bytes_to_unicode


class GlmTokenizerTest(unittest.TestCase):
    def tokenizer(self, pre="glm4"):
        tokens = list(bytes_to_unicode().values()) + ["12", "123", "45", "456", "123456", "<sop>"]
        return Tokenizer(tokens, ["1 2", "12 3", "4 5", "45 6", "123 456"],
                         [1] * (len(tokens) - 1) + [3], pre=pre)

    def test_three_digit_boundaries(self):
        t = self.tokenizer()
        self.assertEqual(t.encode("1234567"), [t.ids["123"], t.ids["456"], t.ids["7"]])
        self.assertEqual(t.decode(t.encode("1234567")), "1234567")
        self.assertEqual(self.tokenizer("qwen35").encode("1234567"),
                         [t.ids[str(i)] for i in range(1, 8)])

    def test_alias_and_round_trips(self):
        for text in ["Hello, world!", "I'm 1234567", "e\u0301", "日本語 🐈\n\n", "  a\t\r\nb"]:
            a, b = self.tokenizer(), self.tokenizer("chatglm-bpe")
            self.assertEqual(a.encode(text), b.encode(text))
            self.assertEqual(a.decode(a.encode(text)), text)

    def test_special_token(self):
        t = self.tokenizer()
        self.assertEqual(t.encode("<sop>", parse_special=True), [t.ids["<sop>"]])
        self.assertNotEqual(t.encode("<sop>"), [t.ids["<sop>"]])


if __name__ == "__main__":
    unittest.main()
