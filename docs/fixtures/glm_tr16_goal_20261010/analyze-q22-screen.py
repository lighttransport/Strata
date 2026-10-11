"""Completed Q22 screen traffic and cache evidence; no quality promotion."""
import json,re
from pathlib import Path
r=Path(__file__).resolve().parent;out=r/'q22-screen'
rows=[]
for row in json.loads((out/'summary.json').read_text()):
 if not row['clean']:continue
 folder=out/row['name'];x=json.loads((folder/'000-baseline.result.json').read_text());log=(folder/'000-baseline.log').read_text()
 m=x['measurements'];sp=m['speculation'][-1];t=m['mtp_timing'][-1]
 cache=re.findall(r'DECODE_CACHE prefix_trained=1 slots=(\d+) MiB=([\d.]+) fingerprint=(\d+)',log)
 flow=re.findall(r'CPU_EXPERT gu_ms=[\d.]+ quant_ms=[\d.]+ down_ms=[\d.]+ layer_flow_ms=([\d.]+) bytes=(\d+)',log)
 assert cache and flow and sp['generated']==128
 ms,count=flow[-1];ms=float(ms);count=int(count)
 rows.append(dict(name=row['name'],decode_tok_s=row['decode_tok_s'],prefill_tok_s=row['prefill_tok_s'],target_cache_slots=int(cache[-1][0]),target_cache_mib=float(cache[-1][1]),cache_fingerprint=cache[-1][2],rounds=sp['rounds'],accepted=sp['accepted'],draft_ms=t['draft_ms'],verify_ms=t['verify_ms'],reported_cpu_expert_bytes=count,reported_layer_flow_ms=ms,reported_bytes_per_generated_token=count/(sp['generated']-1),reported_flow_gb_s=count/ms/1e6,quality_qualified=False))
by_name={row['name']:row for row in rows}
comparisons=[]
if '02-q22-aff08' in by_name:
 base=by_name['00-q23-q4-control'];candidate=by_name['02-q22-aff08']
 tokens=[]
 for row in (base,candidate):
  ids=(out/row['name']/'000-baseline.stdout').read_text().split()
  assert len(ids)==256
  tokens.append(ids)
 comparisons.append(dict(
  baseline=base['name'],candidate=candidate['name'],
  scope='Same scalar affinity and requested cache; changed expert precision, actual cache contents, routing and generated workload. Not a pure kernel comparison.',
  decode_change_percent=100*(candidate['decode_tok_s']/base['decode_tok_s']-1),
  prefill_change_percent=100*(candidate['prefill_tok_s']/base['prefill_tok_s']-1),
  all_256_tokens_identical=tokens[0]==tokens[1],
  first_difference_by_trial=[next((i for i,(a,b) in enumerate(zip(tokens[0][start:start+128],tokens[1][start:start+128])) if a!=b),None) for start in (0,128)],
  draft_ms_per_round={row['name']:row['draft_ms']/row['rounds'] for row in (base,candidate)},
  verify_ms_per_round={row['name']:row['verify_ms']/row['rounds'] for row in (base,candidate)},
  quality_qualified=False))
(out/'traffic-analysis.json').write_text(json.dumps(dict(scope='Timed training-prompt128-token screens; CPU counters include reported expert work, not unique physical DRAM transactions. Routing, weights and generated outputs may differ. No speed extrapolation or quality qualification.',rows=rows,comparisons=comparisons),indent=2)+'\n')
print(json.dumps(rows,indent=2))
print(json.dumps(comparisons,indent=2))
