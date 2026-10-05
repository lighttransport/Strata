"""Metadata and tokenizer adapter for the native GLM safetensors pack."""
import json
from pathlib import Path


def load_metadata(directory):
    root = Path(directory)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    text = config["text_config"]
    quant = config.get("quantization_config", {})
    if config.get("model_type") != "glm5_next" or quant.get("quant_method") != "exl3":
        raise ValueError("expected a GLM-5.3-Flash EXL3 directory")
    result = {"general.architecture": "glm5next", "glm5next.context_length": text["max_position_embeddings"],
              "tokenizer.chat_template": (root / "chat_template.jinja").read_text(encoding="utf-8")}
    for key, value in zip(("eos", "eot", "eom"), text["eos_token_id"]):
        result[f"tokenizer.ggml.{key}_token_id"] = value
    return result


class NativeTokenizer:
    def __init__(self, directory):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("Native GLM requires tokenizers: pip install -r requirements-exl3.txt") from exc
        root = Path(directory)
        source = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
        self._special = Tokenizer.from_str(json.dumps(source, ensure_ascii=False))
        plain = dict(source)
        plain["added_tokens"] = [token for token in source.get("added_tokens", []) if not token.get("special")]
        self._plain = Tokenizer.from_str(json.dumps(plain, ensure_ascii=False))
        vocabulary = self._special.get_vocab()
        self.tokens = [self._special.id_to_token(i) for i in range(max(vocabulary.values()) + 1)]
        self.ids = vocabulary
        self.special_ids = {t["content"]: t["id"] for t in source.get("added_tokens", []) if t.get("special")}

    def encode(self, text, parse_special=False):
        return (self._special if parse_special else self._plain).encode(text, add_special_tokens=False).ids

    def decode(self, ids, errors="replace"):
        return self._special.decode(list(ids), skip_special_tokens=False)

    def token_bytes(self, token):
        # ByteLevel token text contains the reversible byte alphabet. Decode a
        # single incomplete UTF-8 token only after the server joins its bytes.
        from strata_tokenizer import UNICODE_TO_BYTE
        text = self._special.id_to_token(token)
        if text is None:
            raise ValueError(f"unknown token {token}")
        if text in self.special_ids:
            return text.encode("utf-8")
        try:
            return bytes(UNICODE_TO_BYTE[c] for c in text)
        except KeyError:
            return text.encode("utf-8")
