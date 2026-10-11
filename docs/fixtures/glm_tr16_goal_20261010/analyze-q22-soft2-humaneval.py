"""Compare full task outcomes without hiding regressions behind net pass counts."""
import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent
out=r/'q22-soft2-humaneval-full'
base=r.parent/'build-tr16-20261009/humaneval-baseline'
candidate=out/'q22-soft2'
paths=[p/'measurement.json' for p in (base,candidate)]
measure=[json.loads(p.read_text()) for p in paths]
assert all(v['complete'] and v['tasks']==164 and len(v['results'])==164 for v in measure)
records=[{v['task_id']:v for v in x['results']} for x in measure]
assert len(records[0])==164 and records[0].keys()==records[1].keys()
assert all(isinstance(v['passed'],bool) for m in records for v in m.values())
manifest=[json.loads((p/'manifest.json').read_text()) for p in (base,candidate)]
for key in ('dataset_revision','dataset_sha256','protocol','maximum_generated_tokens'):assert manifest[0][key]==manifest[1][key]
assert manifest[1]['environment']['STRATA_GLM_MTP_ACCEPT_MARGIN']=='2'
assert manifest[1]['engine_args'][11]=='3'
config=json.loads((out/'q22-soft2.config.json').read_text())
for key,value in config['env'].items():assert manifest[1]['environment'].get(key)==value,key
assert manifest[1]['decoder_sha256']==hashlib.sha256(Path(config['exe']).read_bytes()).hexdigest()
assert manifest[1]['engine_args'][24]==str(Path(config['expert_pack']).resolve())
rows=[dict(task_id=k,baseline_passed=b['passed'],candidate_passed=records[1][k]['passed'],baseline_generated_tokens=b['generated_tokens'],candidate_generated_tokens=records[1][k]['generated_tokens']) for k,b in records[0].items()]
lost=[v['task_id'] for v in rows if v['baseline_passed'] and not v['candidate_passed']]
gained=[v['task_id'] for v in rows if not v['baseline_passed'] and v['candidate_passed']]
x=dict(full164_complete=True,quality_qualified=False,baseline_passes=sum(v['baseline_passed'] for v in rows),candidate_passes=sum(v['candidate_passed'] for v in rows),lost_pass_task_ids=lost,gained_pass_task_ids=gained,net_pass_delta=len(gained)-len(lost),candidate_at_generation_cap=[v['task_id'] for v in rows if v['candidate_generated_tokens']>=manifest[1]['maximum_generated_tokens']],scope='Full pinned-task comparison; task regressions and net score are separate. Corpus limits and speed targets still apply. No automatic quality promotion.',measurement_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},tasks=rows)
(out/'paired-comparison.json').write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps({k:v for k,v in x.items() if k!='tasks'},indent=2))
