import json,sys
from pathlib import Path
sys.path[:0]=['tools','.']
from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate
from glm_long_chat_check import FACTS
out=Path('build-hip-glm-v11/q3-cache-study');model=Path('/mnt/disk01/models/glm53f-reap/GLM-5.3-Flash-REAP50-Q3_K_M.gguf')
g=GGUFFile(model);m=g.metadata;prefix=m['general.architecture']+'.';first=m[prefix+'leading_dense_block_count'];last=m[prefix+'block_count']-m[prefix+'nextn_predict_layers'];entries={}
for t in g.tensors:
 import re
 match=re.fullmatch(r'blk\.(\d+)\.ffn_(gate|up|down)_exps.weight',t.name)
 if match and first<=int(match[1])<last:
  size=t.expected_bytes()//t.shape[-1]
  for e in range(t.shape[-1]):
   key=f'{match[1]}:{e}';entry=entries.setdefault(key,dict(bytes=0,spans=[]));entry['bytes']+=size;entry['spans'].append([g.data_start+t.offset+e*size,size])
stops={int(m[k]) for k in ('tokenizer.ggml.eos_token_id','tokenizer.ggml.eot_token_id','tokenizer.ggml.eom_token_id') if k in m}
stops.update(i for i,t in enumerate(m['tokenizer.ggml.tokens']) if t in ('<|user|>','<|observation|>'))
(out/'inventory.json').write_text(json.dumps(dict(model=str(model),model_bytes=model.stat().st_size,first_layer=first,last_layer=last,top_k=m[prefix+'expert_used_count'],experts=m[prefix+'expert_count'],stop_ids=sorted(stops),entries=entries),indent=2))
tok=Tokenizer.from_gguf(model);template=ChatTemplate(source=m['tokenizer.chat_template'])
prompts={'code':'Review this C++17 code in depth. Explain every correctness bug, provide corrected code and a detailed test plan.\n'+Path('docs/fixtures/glm_cpp_review_20261008/review.cpp').read_text(), 'document':'Write a detailed project handover memo based only on this fictional dossier. Explain decisions, calculations, risks and next actions with evidence IDs. Then give a question-and-answer guide for the incoming project lead.\n'+''.join(f'[{k}] {v}\n' for k,v in FACTS)}
for name,text in prompts.items():
 ids=tok.encode(template.render([dict(role='user',content=text)],tools=None,reasoning_effort='low',add_generation_prompt=True),parse_special=True)
 (out/(name+'.ids')).write_text(','.join(map(str,ids)));(out/(name+'.prompt.txt')).write_text(text)
 print(name,len(ids))
print('experts GiB',sum(e['bytes'] for e in entries.values())/2**30)
