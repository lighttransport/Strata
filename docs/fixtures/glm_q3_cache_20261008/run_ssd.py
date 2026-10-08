import json,subprocess,time
from pathlib import Path
root=Path('build-hip-glm-v11/q3-cache-study');inv=json.loads((root/'inventory.json').read_text());rows=[]
for trial in range(2):
 for workers in ([1,4,8,16] if trial==0 else [16,8,4,1]):
  cmd=[str(root/'expert-read-bench'),inv['model'],str(root/'ssd-spans.txt'),str(workers)]
  p=subprocess.run(cmd,check=True,capture_output=True,text=True);d=json.loads(p.stdout);d['trial']=trial
  if d['process_read_bytes']<d['aligned_bytes']*.99:raise RuntimeError('physical read accounting does not confirm direct reads')
  rows.append(d);(root/'ssd-results.json').write_text(json.dumps(dict(mode='O_RDONLY|O_DIRECT; no file modifications; aligned real expert projections; isolated from inference',rows=rows),indent=2)+'\n');print(json.dumps(d),flush=True)
