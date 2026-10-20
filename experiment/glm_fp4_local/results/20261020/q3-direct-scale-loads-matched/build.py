from pathlib import Path
import subprocess,json
old=Path('build-fp4-local/q3-direct-scale-loads');p=Path('build-fp4-local/q3-direct-scale-loads-matched')
(p/'candidate.cpp').write_bytes((old/'candidate.cpp').read_bytes());(p/'baseline.cpp').write_bytes(Path('src/kernels/cpu/q23_avx2.cpp').read_bytes())
s=(old/'probe.cpp').read_text().replace('void q23_candidate_rows(', 'void baseline_q23_rows(int,const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int);\nvoid q23_candidate_rows(').replace('candidate?q23_candidate_rows:q23_rows','candidate?q23_candidate_rows:baseline_q23_rows')
(p/'probe.cpp').write_text(s)
prev=json.loads((old/'build.json').read_text());candidate=[x.replace(str(old),str(p)) for x in prev[0]];candidate.insert(1,'-fPIC')
baseline=[x.replace('q23_candidate_rows','baseline_q23_rows').replace('q23_candidate_gu_rows','baseline_q23_gu_rows').replace('candidate_q3k','baseline_q3k').replace('candidate.cpp','baseline.cpp').replace('candidate.o','baseline.o') for x in candidate]
link=[x.replace(str(old),str(p)) for x in prev[-1]];link.insert(link.index(str(p/'candidate.o')),str(p/'baseline.o'));link.insert(1,'-fPIC')
cmds=[baseline,candidate,link]
with (p/'build.log').open('w') as f:
 for cmd in cmds:subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT)
(p/'build.json').write_text(json.dumps(cmds,indent=2)+'\n')
for f in ['run.py','analyze.py']:(p/f).write_text((old/f).read_text().replace(old.name,p.name))
