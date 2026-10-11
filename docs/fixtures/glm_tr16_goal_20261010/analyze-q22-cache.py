"""Summarize completed cache screens without treating requested capacity as allocated."""
import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent;o=r/'q22-cache-screen'
rows=json.loads((o/'summary.json').read_text())
a={v['name']:v for v in rows}
result={'quality_qualified':False,'complete_cases':len(rows),'pairs':[]}
left,right='aff08-cache7168','aff08-cache7424'
if left in a and right in a:
 x,y=a[left],a[right]
 paths=[o/n/'000-baseline.stdout' for n in (left,right)]
 tokens=[p.read_text().split() for p in paths]
 assert all(len(t)==256 for t in tokens)
 result['pairs'].append({'left':left,'right':right,'same_allocated_capacity':all(x[k]==y[k] for k in ('actual_cache_slots','actual_cache_mib','cache_fingerprint')),'all_256_tokens_identical':tokens[0]==tokens[1],'stdout_sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'decode_tok_s':[x['decode_tok_s'],y['decode_tok_s']],'nominal_speed_delta_percent':100*(y['decode_tok_s']/x['decode_tok_s']-1),'interpretation':'If capacity and tokens match, this one-trial timing difference does not establish benefit from the larger requested cache.'})
result['capacity_note']='Cache admission enforces both the allocator budget and physical free memory after the runtime reserve. Requested MiB alone do not establish allocated capacity.'
(o/'capacity-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
