from pathlib import Path
import json,subprocess
old=Path('build-fp4-local/fp4-compact-scale-products');p=Path('build-fp4-local/fp4-normalized-scale-products')
s=(old/'candidate.cpp').read_text()
assert s.count('x == 0 || x == 0x7F')==1
s=s.replace('x == 0 || x == 0x7F','x == 0')
s=s.replace('return code == 127 ? 128 : (code & 127);','return code;')
s=s.replace('[129]','[128]').replace('scale < 129','scale < 128')
s=s.replace('scale == 128 ? 127 : (scale | 128)','scale')
(p/'candidate.cpp').write_text(s)
s=(old/'probe.cpp').read_text()
needle=' std::vector<block_q8_0> a(nt*K/32);'
s=s.replace(needle,''' // Prepared format: collapse duplicate codes, preserving 0x7f zero and 0xff finite.
 auto normalized = w;
 for (auto& b:normalized) for(auto& scale:b.d) scale=scale==127?0:(scale&127);
'''+needle)
s=s.replace('w.data())+size_t(bank)','(candidate?normalized.data():w.data()))+size_t(bank)')
s=s.replace('" bytes="','" prepared_weight_copy_bytes="<<normalized.size()*sizeof(block_nvfp4)<<" bytes="')
(p/'probe.cpp').write_text(s)
cmds=[[x.replace(str(old),str(p)) for x in c] for c in json.loads((old/'build.json').read_text())]
with (p/'build.log').open('w') as log:
 for cmd in cmds:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
(p/'build.json').write_text(json.dumps(cmds,indent=2)+'\n')
for f in ['run.py','summarize.py']:(p/f).write_text((old/f).read_text().replace(old.name,p.name))
