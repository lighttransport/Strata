"""Summarize completed proposal controls without turning short screens into forecasts."""
import json
from pathlib import Path
r=Path(__file__).resolve().parent;p=r/'proposal-model-controls'
completed=json.loads((p/'summary.json').read_text());rows=[]
control=next(v for v in completed if v['name']=='03-proposal-default')
for row in completed:
 if int(row['name'].split('-')[0])<3:continue
 x=json.loads((p/row['name']/'000-baseline.result.json').read_text())
 assert x['clean'] and row['tokens_identical']==256
 m=x['measurements'];spec=m['speculation'][-1];positions=m['mtp_positions'][-3:]
 rows.append({'name':row['name'],'decode_tok_s':row['decode_tok_s'],'change_percent':100*(row['decode_tok_s']/control['decode_tok_s']-1),'rounds':spec['rounds'],'accepted':spec['accepted'],'accepted_per_round':spec['accepted']/spec['rounds'],'positions':positions,'tokens_identical_to_control':256,'quality_qualified':False})
(r/'proposal-control-analysis.json').write_text(json.dumps({'scope':'One128-token timed trial plus128-token warmup, training prompt; all256 target tokens matched control. Grouped runs, not interleaved. Different round boundaries mean position rates are not paired predictions on the same prefixes.','rows':rows},indent=2)+'\n')
print(json.dumps(rows,indent=2))
