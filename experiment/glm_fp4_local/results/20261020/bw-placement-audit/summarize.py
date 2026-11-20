from pathlib import Path
import json,re,shutil,hashlib,statistics
p=Path('build-fp4-local/bw-audit');o=Path('experiment/glm_fp4_local/results/20261020/bw-placement-audit');o.mkdir(exist_ok=True)
rows=[]
for f in sorted(p.glob('*.log')):
 text=f.read_text();rates=re.findall(r'^(read|copy|decode): median ([0-9.]+)',text,re.M)
 if not rates:continue
 assert text.count('validation OK')==len(rates)
 samples=[dict(x.split('=',1) for x in l.split()[1:]) for l in text.splitlines() if l.startswith('SAMPLE ')]
 rows.append({'log':f.name,'median_GBps':dict((k,float(v)) for k,v in rates),'placement':[l for l in text.splitlines() if l.startswith('PLACEMENT ')],'median_active_cores':statistics.median([float(r['active_cores']) for r in samples]) if samples else None})
for name in ['runs.json','locality-runs.json']:assert all(r['returncode']==0 for r in json.loads((p/name).read_text()))
summary={'measurements':rows,'finding':'Default first touch placed all buffers on node 1 in low-bandwidth runs. Explicit per-worker local binding split pages evenly and restored the high-bandwidth regime. Subsequent default allocations also split evenly after local memory was reclaimed.','limits':['Earlier 34/83 GB/s runs did not record placement, so the exact historical cause is inferred, not directly observed.','Page counts aggregate buffer-containing mappings; not per-thread memory-controller traffic.','Measured read bandwidth is not theoretical DRAM bandwidth or quantized-kernel utilization.','Strict binding can induce reclaim; allocation and reclaim are outside the scan timer.'],'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path('tools/membw/membw.cpp'),p/'baseline.cpp',p/'probe.cpp',p/'locality.cpp']}}
for f in p.iterdir():
 if f.suffix in ['.cpp','.json','.log','.py']:shutil.copyfile(f,o/f.name)
(o/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('PASS',len(rows),'validated bandwidth runs archived')
