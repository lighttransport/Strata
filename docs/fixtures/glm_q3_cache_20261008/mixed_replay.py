import json,sys
from pathlib import Path
sys.path.insert(0,'tools')
from glm_cache_replay import load_trace,replay,GIB
p=Path(sys.argv[1]);inv=json.loads((p/'inventory.json').read_text());sizes={k:v['bytes'] for k,v in inv['entries'].items()};combined=[];position=0;boundaries=[]
for name in ('code','document'):
 rows,v=load_trace(p/(name+'.routes.csv'),inv,len((p/(name+'.ids')).read_text().split(',')),list(map(int,(p/(name+'-run.stdout')).read_text().split())))
 first=rows[0][0];combined += [(pos-first+position,l,keys) for pos,l,keys in rows];boundaries.append(dict(workload=name,start=position,steps=v['usable_decode_steps']));position+=v['usable_decode_steps']
train=[r for r in combined if r[0]<64];test=[r for r in combined if r[0]>=64]
rows=[replay(train,test,sizes,cpu*GIB,gpu*GIB,policy) for gpu in (4,6,8) for cpu in (48,52,54,56,58) for policy in ('static','lru','hybrid')]
(p/'mixed-replay.json').write_text(json.dumps(dict(kind='code then document; carried cache state; first 64 code steps train admission; traces independently generated',boundaries=boundaries,train_tokens=64,rows=rows),indent=2)+'\n')
