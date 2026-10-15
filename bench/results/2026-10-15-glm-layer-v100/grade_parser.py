"""Reproduce the frozen parser gate: score the last final answer, even after EOS.

For natural-stop quality use reproduce_quality.py; forced timing may generate
a second answer after its first EOS. Preserve this gate's original failures.
"""
import hashlib, json, pathlib, re, subprocess, sys
source=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(source/'tools'))
from glm_generate import Tokenizer, metadata_shard
from gguf_reader import GGUFFile
model = pathlib.Path(sys.argv[1])
result_path = pathlib.Path(sys.argv[2])
result = json.loads(result_path.read_text())
if not result.get('complete'):
    raise RuntimeError('Only completed validated runs are eligible')
root = result_path.with_suffix('')
if len(sys.argv) > 3:
    case = sys.argv[3]
    entry = result['validation_by_case'][case]
    if not entry['complete']:
        raise RuntimeError('Candidate long validation is incomplete')
    result['validation'] = entry['trials'][0]
    result['validation_case'] = case
    root = root.with_name(root.name + '-' + case)
root.mkdir(exist_ok=False)
ids = result['validation']['token_ids']
record = dict(complete=False, passed=False, command=None, case=result.get('validation_case'), output_tokens=len(ids),
              decoder_sha256=result['decoder_sha256'], input_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
              generated_source_sha256=None)
def save():
    temporary = root/'quality.json.tmp'
    temporary.write_text(json.dumps(record,indent=2)+'\n')
    temporary.replace(root/'quality.json')
save()
metadata = GGUFFile(metadata_shard(model)).metadata
stops = {int(metadata[k]) for k in ('tokenizer.ggml.eos_token_id','tokenizer.ggml.eot_token_id','tokenizer.ggml.eom_token_id') if k in metadata}
text = Tokenizer.from_gguf(metadata_shard(model)).decode([t for t in ids if t not in stops])
(root/'output.md').write_text(text)
record['response_sha256'] = hashlib.sha256(text.encode()).hexdigest()
if '</think>' not in text:
    record['error']='No final answer after reasoning delimiter';save();raise RuntimeError(record['error'])
text=text.rsplit('</think>',1)[-1]
blocks = re.findall(r'```(?:cpp|c\+\+)\s*\n(.*?)```', text, re.S)
record['complete_cpp_blocks'] = len(blocks)
if len(blocks) != 1:
    record['error'] = f'Expected one complete C++ code block, got {len(blocks)}'
    save()
    raise RuntimeError(f'Expected one complete C++ code block, got {len(blocks)}')
code = root/'generated.cpp'
code.write_text(blocks[0])
command = ['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror', '-DGENERATED_SOURCE="'+str(code.resolve())+'"',str(source/'docs/fixtures/glm53_cpp_quality/test_generated.cpp'),'-o',str(root/'quality-test')]
record.update(command=command, generated_source_sha256=hashlib.sha256(code.read_bytes()).hexdigest())
save()
compiled = subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(root/'compile.stdout').write_text(compiled.stdout)
if compiled.returncode:
    record.update(error='strict C++ compilation failed', compiler_exit=compiled.returncode)
    save()
    print(compiled.stdout,end='')
    compiled.check_returncode()
run = subprocess.run([str(root/'quality-test')],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(root/'quality.stdout').write_text(run.stdout)
print(run.stdout, end='')
record.update(oracle=run.stdout, oracle_exit=run.returncode)
if run.returncode:
    record['error']='generated-code oracle failed'
    save()
run.check_returncode()
record.update(complete=True,passed=True)
save()
