"""Keep repetition diagnostics separate from task-quality qualification."""
import json,statistics
from pathlib import Path
r=Path(__file__).resolve().parent;path=r/'repeat-guard-summary.json'
rows=json.loads(path.read_text());by_name={v['name']:v for v in rows};result=[]
for row in rows:
 if not row.get('clean'):continue
 folder=r/'repeat-guard'/row['name'];samples=json.loads((folder/'text-samples.json').read_text())['trials']
 assert len(samples)==4 and sum(v['warmup'] for v in samples)==1
 grams=[v['distinct_fourgram_fraction'] for v in samples if not v['warmup']]
 control='03-aggressive-control' if row['name'].startswith(('03-','04-')) else '00-repeat-control'
 item={'name':row['name'],'median_decode_tok_s':row['decode'],'control':control,'speed_change_percent':100*(row['decode']/by_name[control]['decode']-1),'timed_distinct_fourgram_fractions':grams,'median_distinct_fourgram_fraction':statistics.median(grams),'repeat_guard_blocked':[v['blocked'] for v in row.get('repeat_guard',[])],'soft_accepted':[v['soft_accepted'] for v in row.get('approx',[])],'quality_qualified':False,'text_samples':str(folder/'text-samples.json')}
 if row.get('repeat_guard'):assert len(row['repeat_guard'])==4
 result.append(item)
(r/'repeat-guard-analysis.json').write_text(json.dumps({'scope':'Training prompt,512-token warmup andthree512-token timed continuations. Distinct n-grams describe repetition; they do not measure correctness or establish output quality. Inspect every saved continuation and run task/corpus tests before qualification.','rows':result},indent=2)+'\n')
print(json.dumps(result,indent=2))
