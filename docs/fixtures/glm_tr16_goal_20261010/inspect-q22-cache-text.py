"""Short continuation diagnostics; manual review and long generation remain necessary."""
import hashlib,json,sys
from pathlib import Path
r=Path(__file__).resolve().parent;sys.path.insert(0,str(r.parent/'tools'))
from strata_tokenizer import Tokenizer
out=r/'q22-cache-screen';rows=json.loads((out/'summary.json').read_text())
tokenizer=None;summary=[]
for row in rows:
 if not row['clean'] or not row['name'].startswith('depth3'):continue
 c=json.loads(Path(row['config']).read_text())
 if tokenizer is None:tokenizer=Tokenizer.from_gguf(Path(c['model']))
 folder=out/row['name'];path=folder/'000-baseline.stdout';ids=list(map(int,path.read_text().split()));assert len(ids)==256
 trials=[]
 for i in range(2):
  seq=ids[i*128:(i+1)*128];grams=[tuple(seq[j:j+4]) for j in range(125)]
  trials.append(dict(trial=i,warmup=i==0,text=tokenizer.decode(seq),distinct_fourgram_fraction=len(set(grams))/len(grams)))
 result=dict(scope='128-token diagnostic only; absence of short repetition does not qualify task quality or long continuation',tokens_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),quality_qualified=False,manual_review_pending=True,trials=trials)
 (folder/'text-samples.json').write_text(json.dumps(result,indent=2)+'\n')
 summary.append(dict(name=row['name'],decode_tok_s=row['decode_tok_s'],distinct_fourgram_fractions=[v['distinct_fourgram_fraction'] for v in trials],samples=str(folder/'text-samples.json'),quality_qualified=False))
(out/'text-diagnostics.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
