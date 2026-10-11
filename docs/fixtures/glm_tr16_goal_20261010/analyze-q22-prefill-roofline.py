"""Bandwidth-only prefill component estimates anchored to independent TR16 H2D measurements."""
import hashlib,json,re
from pathlib import Path
r=Path(__file__).resolve().parent
hardware=r.parent/'build-tr16-20261009/hardware.json'
h=json.loads(hardware.read_text());bandwidth=h['measured_anchors']['pinned_h2d_GB_s']
rows=[]
for name in ('00-q23-q4-control','02-q22-aff08'):
 folder=r/'q22-screen'/name;log=folder/'000-baseline.log';result=folder/'000-baseline.result.json'
 x=json.loads(result.read_text());assert x['clean'] and x['complete']
 reported=[int(v) for v in re.findall(r'^GPU .*native_expert_bytes=(\d+)',log.read_text(),re.M) if int(v)>0]
 assert len(reported)==2 and len(set(reported))==1, 'Require equal traffic in both completed prefill passes'
 assert x['manifest']['prompt_tokens']==2048 and x['config']['prefill_batch']==2048
 count=reported[0];seconds=count/(bandwidth*1e9);estimate=2048/seconds
 rows.append(dict(name=name,reported_upload_bytes_per_pass=count,pinned_h2d_GB_s=bandwidth,bandwidth_only_seconds=seconds,bandwidth_only_2k_tok_s=estimate,observed_prefill_tok_s=x['measurements']['median_prefill_tok_s'],observed_fraction_of_bandwidth_estimate=x['measurements']['median_prefill_tok_s']/estimate,bandwidth_only_8k_tok_s_if_single_pass=8192/seconds,provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (log,result)}))
out=dict(scope='Independent measured-bandwidth component estimate, not a theoretical device limit or end-to-end forecast. Uses reported native upload traffic including layout overhead. Omits dense/attention compute and synchronization. The 8K estimate assumes a single all-expert pass and sufficient memory; 8K Q22 performance has not been measured here.',hardware_provenance={str(hardware):hashlib.sha256(hardware.read_bytes()).hexdigest()},rows=rows)
(r/'q22-prefill-roofline.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(rows,indent=2))
