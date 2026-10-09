"""Freeze chat-template prompts for B550 calibration and held-out validation."""
import argparse
import hashlib
import json
from pathlib import Path
from jinja2.sandbox import SandboxedEnvironment
from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--tokens',type=int,default=2048);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    tokenizer=Tokenizer.from_gguf(args.model)
    template=SandboxedEnvironment(extensions=['jinja2.ext.loopcontrols']).from_string(GGUFFile(args.model).metadata['tokenizer.chat_template'])
    fixtures=[('train-code','tools/glm_low_memory_bench.py','Review this Python benchmark for correctness, measurement bias and failure handling. Explain concrete bugs and propose patches.'),
              ('train-doc','docs/AMD_HIP.md','Summarize this document in detail. Explain supported devices, setup steps, limitations and the evidence behind performance claims.'),
              ('train-chat','docs/DETAILS.md','Explain how this inference engine works to a curious programmer. Use examples and discuss memory bandwidth, caching and latency.'),
              ('holdout-code','include/strata/kernels/cpu/pool.hpp','Review this C++ code for concurrency and memory safety. Give a detailed review with concrete invariants, edge cases, and test cases.'),
              ('holdout-doc','docs/MULTI_GPU.md','Summarize this document, then answer: how are GPUs selected, what constrains speed, and which details would you verify before deployment?'),
              ('holdout-chat','docs/AI_SETUP.md','Help a new user understand installation and troubleshooting. Explain the process clearly, with a worked example and questions to diagnose a slow machine.')]
    records=[]
    for name,path,question in fixtures:
        source=(root/path).read_text()
        body='Source document (excerpts may be omitted):\n'+(source+'\n')*max(1,args.tokens//500)+'\n\n'+question+' Give a thorough answer of at least 800 words.'
        text=template.render(messages=[dict(role='user',content=body)],tools=None,add_generation_prompt=True,reasoning_effort='low')
        ids=tokenizer.encode(text,parse_special=True)
        # Preserve the task and assistant header; freeze identical IDs for all arms.
        if len(ids)<args.tokens: raise ValueError('source too short')
        ids=ids[:args.tokens-192]+ids[-192:]
        output=args.output/(name+'.ids');output.write_text(','.join(map(str,ids)))
        records.append(dict(name=name,split=name.split('-')[0],source=path,tokens=len(ids),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),question=question))
    (args.output/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
