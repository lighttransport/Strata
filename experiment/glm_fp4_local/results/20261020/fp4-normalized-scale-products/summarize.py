from pathlib import Path
import json,statistics,hashlib,shutil,sys
name=sys.argv[1] if len(sys.argv)>1 else 'fp4-normalized-scale-products'
p=Path('build-fp4-local')/name;d=Path('experiment/glm_fp4_local/results/20261020')/name;d.mkdir(parents=True,exist_ok=True)
s={'scope':'CPU-only synthetic paired GU screen; production unchanged','cells':[],'timed_checks':0,'timed_calls':0}
for run in json.loads((p/'runs.json').read_text()):
 assert run['returncode']==0
 rows=[dict(x.split('=',1) for x in l.split()) for l in (p/(run['name']+'.log')).read_text().splitlines()]
 assert len(rows)==24 and all(r['exact']=='1' for r in rows)
 pairs={}
 for r in rows:pairs.setdefault(r['round'],{})[r['candidate']]=r
 ratios=[float(v['1']['wall_ms'])/float(v['0']['wall_ms']) for v in pairs.values()]
 s['cells'].append(dict(run,scale_pattern=rows[0]['scale_pattern'],median_wall_ratio=statistics.median(ratios),wins=sum(r<1 for r in ratios),pairs=len(ratios)))
 s['timed_checks']+=len(rows);s['timed_calls']+=sum(int(r['calls']) for r in rows)
for f in p.iterdir():
 if f.suffix in ['.cpp','.py','.json','.log']:shutil.copyfile(f,d/f.name)
shutil.copyfile(p/'probe.cpp',d/'probe.cpp')
s['sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p/'probe',p/'candidate.cpp',Path('src/kernels/cpu/nvfp4_avx2.cpp'),Path('build-hip-rocm101-tr16/libstrata_kernels_cpu.a')]}
(d/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
for c in s['cells']:print(c['name'],round(c['median_wall_ratio'],4),c['wins'])
print('checks',s['timed_checks'],'calls',s['timed_calls'])
