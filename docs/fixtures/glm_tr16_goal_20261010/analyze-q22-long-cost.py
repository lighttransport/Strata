"""Measured component budgets for the50tok/s target; not a fitted simulator."""
import hashlib,json,re
from pathlib import Path
r=Path(__file__).resolve().parent
out=r/'long-finalists/q22-soft4-diagnostic';rows=[]
for context in ('2k','8k'):
 path=out/context/'000-baseline.log';s=path.read_text()
 specs=re.findall(r'SPECULATIVE source=mtp generated=(\d+).*?ms=([\d.]+) tok_s=([\d.]+)',s)
 flows=[(float(ms),int(n)) for ms,n in re.findall(r'CPU_EXPERT .*?layer_flow_ms=([\d.]+) bytes=(\d+)',s) if int(n)]
 assert len(specs)==len(flows)==4
 for trial,((generated,total,rate),(flow,n)) in enumerate(zip(specs,flows)):
  steps=int(generated)-1;assert steps==511
  rows.append(dict(context=context,trial=trial,warmup=trial==0,decode_tok_s=float(rate),decode_ms_per_step=float(total)/steps,expert_flow_ms_per_step=flow/steps,reported_expert_GB_per_step=n/steps/1e9,reported_bytes_over_flow_GB_s=n/flow/1e6,target50_ms_per_step=20,required_total_time_reduction_percent=100*(1-float(rate)/50),source_log_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
x=dict(quality_qualified=False,scope='Completed512-token measured traces. Flow can include synchronization and overlap other work. Reported bytes are logical expert traffic, not measured DRAM transactions. These figures are not additive timing fractions, hardware ceilings, or performance predictions.',rows=rows)
(out/'target50-cost-analysis.json').write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps([v for v in rows if not v['warmup']],indent=2))
