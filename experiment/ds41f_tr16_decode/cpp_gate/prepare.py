"""Prepare the parser coding task with the existing DS4 tokenizer assets."""
import hashlib
import json
import pathlib
import sys
from jinja2.sandbox import SandboxedEnvironment
from tokenizers import AddedToken
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'tok'))
from ds4tok import load, D
root = pathlib.Path(__file__).resolve().parent
tok = load()
types = json.loads(pathlib.Path(D, 'token_type.json').read_text())
special = [s for s, i in tok.get_vocab().items() if types[i] in (3, 4)]
tok.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in special])
env = SandboxedEnvironment()
def fail(message):
    raise ValueError(message)
env.globals['raise_exception'] = fail
chat = env.from_string(pathlib.Path(D, 'chat_template.jinja').read_text()).render(
    messages=[{'role': 'user', 'content': (root / 'prompt.txt').read_text()}],
    bos_token='<｜begin▁of▁sentence｜>', add_generation_prompt=True, thinking=False)
ids = tok.encode(chat).ids
assert ids[0] == 0 and len(ids) < 800, (len(ids), ids[:8])
(root / 'chat.txt').write_text(chat)
(root / 'prompt_ids.txt').write_text(' '.join(map(str, ids)) + '\n')
manifest = json.loads((root / 'manifest.json').read_text())
manifest.update(input_tokens=len(ids), chat_sha256=hashlib.sha256(chat.encode()).hexdigest())
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('prepared', len(ids), 'tokens')
