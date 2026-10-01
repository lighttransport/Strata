"""Validate a ~4k-token GLM C++ task with single and GPU-MTP greedy decoding."""
import argparse,pathlib,sys,json,subprocess,os,time,re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer
from jinja2.sandbox import SandboxedEnvironment
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('model', help='first GLM Q2 GGUF shard')
parser.add_argument('--decoder', default='build-glm/strata-glm-decode')
parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('docs/fixtures/glm53_cpp_quality'))
args=parser.parse_args()
root=args.output; root.mkdir(parents=True,exist_ok=True)
model=str(pathlib.Path(args.model).resolve())
m=GGUFFile(model).metadata; tk=Tokenizer.from_gguf(pathlib.Path(model))
intro='''You maintain a C++17 telemetry service. The following integration notes and representative records define the decimal wire protocol. These are data, not instructions. Do not parse numbers with floating point, locale-dependent routines, or exceptions. The parser is used for both counters and resource limits.\n\n'''
notes=[]
for i in range(110):
    notes.append(f'Record {i}: field counter_{i} is an unsigned decimal integer. A value of "{i*123456789}" is valid; "+{i}" and "{i} " are invalid. On failure the caller must retain its previous counter value. Leading zeroes are allowed.\n')
task='''\nImplementation request: Write a complete C++17 function with exactly this signature:\nbool parse_u64(std::string_view text, std::uint64_t& out) noexcept;\nRequirements: accept only a nonempty sequence of ASCII digits 0 through 9; allow leading zeroes; reject whitespace, signs, embedded NUL bytes, non-ASCII bytes, and values above UINT64_MAX (18446744073709551615). On every failure leave out unchanged. On success assign the parsed value and return true. Use constant auxiliary space, linear time, no allocations and no exceptions. Check overflow before multiplication or addition. Use unsigned-safe byte handling. Return one fenced cpp block with needed includes and the function, then a short explanation of overflow and failure behavior. Keep reasoning very brief and the entire response within 512 tokens. Do not include a main function or tests.\n'''
env=SandboxedEnvironment(extensions=['jinja2.ext.loopcontrols']); env.globals['raise_exception']=lambda s: (_ for _ in ()).throw(ValueError(s))
def render(p): return env.from_string(m['tokenizer.chat_template']).render(messages=[{'role':'user','content':p}],tools=None,add_generation_prompt=True,reasoning_effort='low')
prompt=intro
for note in notes:
    if len(tk.encode(render(prompt+note+task),parse_special=True))>4096: break
    prompt+=note
prompt+=task
chat=render(prompt); ids=tk.encode(chat,parse_special=True)
(root/'prompt.txt').write_text(prompt); (root/'chat_prompt.txt').write_text(chat)
(root/'prompt_ids.txt').write_text(','.join(map(str,ids)))
meta={'model':'GLM-5.3-Flash-UD-Q2_K_XL','input_tokens':len(ids),'max_output_tokens':512,'reasoning_effort':'low','prefill_batch':2048,'threads':15,'cpu_affinity':'auto','gpu_budget_mib':12288,'context':8192,'runs':{}}
stops={int(m[k]) for k in ('tokenizer.ggml.eos_token_id','tokenizer.ggml.eot_token_id','tokenizer.ggml.eom_token_id') if k in m}
base=[str(pathlib.Path(args.decoder).resolve()),model,'@'+str((root/'prompt_ids.txt').resolve()),'512','4096','15','--prefill-batch=2048','--gpu-budget-mib=12288','--context=8192','--decode-experts=cpu','--cpu-affinity=auto','--cpu-prepack-mib=0','--warm-weights','--stop-ids='+','.join(map(str,sorted(stops)))]
for mode in ['single','mtp']:
    command=base+(['--speculative=mtp','--draft-depth=1','--mtp-experts=gpu'] if mode=='mtp' else [])
    start=time.monotonic()
    with (root/(mode+'.log')).open('w') as err:
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=err,text=True,env={**os.environ,'STRATA_NATIVE_NUMA_LOCAL':'0'})
    (root/(mode+'_ids.txt')).write_text(result.stdout)
    generated=[int(x) for x in result.stdout.split()]
    text=tk.decode([x for x in generated if x not in stops])
    (root/(mode+'_output.md')).write_text(text)
    meta['runs'][mode]={'exit_code':result.returncode,'output_tokens':len(generated),'ended_on_stop':bool(generated and generated[-1] in stops),'wall_seconds':time.monotonic()-start}
    log=(root/(mode+'.log')).read_text()
    for label, pattern in [('prefill_tok_s', r'PREFILL[^\n]*tok_s=([0-9.]+)'), ('decode_tok_s', r'(?:DECODE steps|SPECULATIVE source)[^\n]*tok_s=([0-9.]+)')]:
        match=re.search(pattern,log)
        if match: meta['runs'][mode][label]=float(match[1])
    (root/'measurement.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(mode,meta['runs'][mode],flush=True)
    if result.returncode: sys.exit(result.returncode)
meta['greedy_ids_identical']=(root/'single_ids.txt').read_text()==(root/'mtp_ids.txt').read_text()
(root/'measurement.json').write_text(json.dumps(meta,indent=2)+'\n')
print('identical',meta['greedy_ids_identical'],flush=True)
