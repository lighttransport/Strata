from pathlib import Path
import json,statistics,hashlib,shutil,re
p=Path('build-fp4-local/q3-direct-scale-loads-matched');out=Path('experiment/glm_fp4_local/results/20261020/q3-direct-scale-loads-matched');out.mkdir(exist_ok=True)
summary={'scope':'Synthetic Q3 NT1 direct scalar scale loads versus same-source baseline with identical flags','cells':[],'timed_arm_checks':0,'timed_calls':0,'limitations':['Single-thread screening with concurrent host load','No complete expert service, full-model throughput, GPU execution or theoretical peak qualification','Q3 safe fixtures replace -128 with -127 before the independent FP64 reference; fallback fixtures inject -128','Q2 and NT2 are unchanged controls; NTA disabled in both arms']}
for run in json.loads((p/'runs.json').read_text()):
 assert run['returncode']==0
 rows=[dict(v.split('=',1) for v in line.split()) for line in (p/(run['name']+'.log')).read_text().splitlines()];assert len(rows)==24
 pairs={}
 for row in rows:pairs.setdefault(row['round'],{})[row['candidate']]=float(row['wall_ms'])
 ratios=[x['1']/x['0'] for x in pairs.values()]
 cell=dict(run,median_ratio=statistics.median(ratios),candidate_wins=sum(x<1 for x in ratios),pairs=len(ratios));summary['cells'].append(cell)
 summary['timed_arm_checks']+=len(rows);summary['timed_calls']+=sum(int(x['calls']) for x in rows)
 print(run['name'],round(cell['median_ratio'],4),cell['candidate_wins'])
summary['decision']='Not selected: small mixed NT1 changes do not establish a service gain'
summary['sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p/'candidate.cpp',p/'candidate.o',p/'baseline.cpp',p/'baseline.o',p/'probe',Path('src/kernels/cpu/q23_avx2.cpp'),Path('build-hip-rocm101-tr16/libstrata_kernels_cpu.a')]}
for f in p.iterdir():
 if f.suffix in ['.py','.cpp','.json','.log']:
  assert not re.search(r'/(?:home|mnt)/[A-Za-z0-9_]|\b(?:\d{1,3}\.){3}\d{1,3}\b',f.read_text()),f
  shutil.copyfile(f,out/f.name)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(summary['timed_arm_checks'],summary['timed_calls'])
