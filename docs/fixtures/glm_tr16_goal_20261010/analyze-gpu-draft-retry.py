"""Summarize completed matched draft screens; never extrapolate a speed target."""
import json
from pathlib import Path
r=Path(__file__).resolve().parent
rows=[];pairs=[];kernel_pairs=[]
for group in ('gpu-draft-retry','q3-gpu-combined','gpu-draft-v2','q3-gpu-v2'):
 path=r/(group+'-summary.json')
 if not path.exists():continue
 local={}
 for record in json.loads(path.read_text()):
  if not record.get('clean'):continue
  folder=r/group/record['name']
  result=json.loads((folder/'000-baseline.result.json').read_text())
  config=json.loads(Path(record['config']).read_text())
  m=result['measurements'];sp=m['speculation'][-1];t=m['mtp_timing'][-1]
  rounds=sp['rounds'];assert rounds>0
  row=dict(group=group,name=record['name'],decode_tok_s=sp['tok_s'],draft_backend=config.get('mtp_experts','cpu'),draft_depth=config['draft_depth'],rounds=rounds,accepted_per_round=sp['accepted']/rounds,proposed_per_round=sp['proposed']/rounds,round_ms=sp['ms']/rounds,draft_ms_per_round=t['draft_ms']/rounds,verify_ms_per_round=t['verify_ms']/rounds,resync_ms_per_round=t['resync_ms']/rounds,quality_qualified=False)
  rows.append(row);local[row['name']]=row
 for name,gpu in local.items():
  if not name.endswith('-gpu'):continue
  cpu=local.get(name.removesuffix('-gpu')+'-cpu')
  if not cpu:continue
  left=(r/group/cpu['name']/'000-baseline.stdout').read_text().split()
  right=(r/group/name/'000-baseline.stdout').read_text().split()
  trials=[{'trial':i,'tokens_identical':left[i*128:(i+1)*128]==right[i*128:(i+1)*128],'first_different_token':next((j for j,(a,b) in enumerate(zip(left[i*128:(i+1)*128],right[i*128:(i+1)*128])) if a!=b),None)} for i in range(2)]
  pairs.append(dict(trial_comparisons=trials,group=group,gpu=name,cpu=cpu['name'],all_tokens_identical=len(left)==len(right)==256 and left==right,nominal_rate_change_percent=100*(gpu['decode_tok_s']/cpu['decode_tok_s']-1),draft_ms_per_round_change=gpu['draft_ms_per_round']-cpu['draft_ms_per_round'],verify_ms_per_round_change=gpu['verify_ms_per_round']-cpu['verify_ms_per_round'],quality_qualified=False))
 if 'q3-depth3-soft2-native-cpu' in local and 'q3-depth3-soft2-cpu' in local:
  native=local['q3-depth3-soft2-native-cpu'];il=local['q3-depth3-soft2-cpu']
  left=(r/group/native['name']/'000-baseline.stdout').read_text().split()
  right=(r/group/il['name']/'000-baseline.stdout').read_text().split()
  kernel_pairs.append(dict(group=group,native=native['name'],interleaved=il['name'],all_tokens_identical=len(left)==len(right)==256 and left==right,native_tok_s=native['decode_tok_s'],interleaved_tok_s=il['decode_tok_s'],nominal_rate_change_percent=100*(il['decode_tok_s']/native['decode_tok_s']-1),quality_qualified=False))
(r/'gpu-draft-retry-analysis.json').write_text(json.dumps(dict(scope='Completed 128-token training-prompt screens. Different outputs prevent isolated overhead attribution. No quality qualification or throughput forecast.',rows=rows,pairs=pairs,kernel_pairs=kernel_pairs),indent=2)+'\n')
print(json.dumps(pairs,indent=2))
