"""Measured round costs and fixed-cost counterfactuals, not throughput forecasts."""
import json
from pathlib import Path
r=Path(__file__).resolve().parent
rows=[]
for s in ('eighth-summary.json','split-q4-summary.json','gpu-draft-summary.json'):
 if not (r/s).exists():continue
 for row in json.loads((r/s).read_text()):
  if not row.get('clean'):continue
  c=json.loads(Path(row['config']).read_text())
  if c['env'].get('STRATA_GLM_DENSE_Q4')!='all':continue
  folder=s.removesuffix('-summary.json')
  x=json.loads((r/folder/row['name']/'000-baseline.result.json').read_text())
  m=x['measurements'];sp=m['speculation'][-1];t=m['mtp_timing'][-1]
  elapsed=sp['ms'];rounds=sp['rounds'];count=sp['generated']-1
  rows.append(dict(name=folder+'/'+row['name'],measured_tok_s=sp['tok_s'],rounds=rounds,
   accepted_per_round=sp['accepted']/rounds,proposed_per_round=sp['proposed']/rounds,
   round_ms=elapsed/rounds,draft_ms_per_round=t['draft_ms']/rounds,
   verify_ms_per_round=t['verify_ms']/rounds,resync_ms_per_round=t['resync_ms']/rounds,
   zero_draft_cost_same_output_tok_s=count/(elapsed-t['draft_ms'])*1000,
   all_accepted_same_round_cost_tok_s=(c['draft_depth']+1)*rounds/elapsed*1000,
   tokens_per_round_required_for_30=.03*elapsed/rounds,
   tokens_per_round_required_for_50=.05*elapsed/rounds,
   quality_qualified=False))
result=dict(note='Training-prompt128-token screening only. Counterfactuals hold measured round cost fixed: neither hardware rooflines nor forecasts. GPU draft also consumes target-cache space, changes proposal numerics and may change routing. Full acceptance changes workload and thus actual cost. Timed token numerator excludes the seed token, matching runner telemetry.',rows=rows)
(r/'q4-round-cost-screen.json').write_text(json.dumps(result,indent=2)+'\n')
for x in rows:print(x['name'], 'measured',round(x['measured_tok_s'],2),'zero-draft',round(x['zero_draft_cost_same_output_tok_s'],2),'all-accepted',round(x['all_accepted_same_round_cost_tok_s'],2))
