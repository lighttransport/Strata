from pathlib import Path
import subprocess,re,json
p=Path('build-fp4-local/q3-direct-scale-loads');meta={}
objects={'baseline':Path('build-hip-rocm101-tr16/CMakeFiles/strata_kernels_cpu.dir/src/kernels/cpu/q23_avx2.cpp.o'),'candidate':p/'candidate.o'}
symbol='_ZN6strata7kernels3cpu12_GLOBAL__N_1L3dotILi11ELi1ELb0ELb0EEEvPKhiPKPKvPfPKDv4_xi'
for name,obj in objects.items():
 text=subprocess.check_output(['objdump','-d','--no-show-raw-insn',str(obj)],text=True)
 lines=text.split(f'<{symbol}>:',1)[1].split('\n\n',1)[0].splitlines();ins=[]
 for line in lines:
  m=re.match(r'\s*([0-9a-f]+):\s+(.*)',line)
  if m:ins.append((int(m[1],16),m[2]))
 loops=[]
 for address,instruction in ins:
  m=re.match(r'jne\s+([0-9a-f]+) ',instruction)
  if m and int(m[1],16)<address:
   start=int(m[1],16);body=[(a,i) for a,i in ins if start<=a<=address]
   if len(body)>50:loops.append(body)
 assert len(loops)==2,(name,len(loops))
 meta[name]=[]
 for n,body in enumerate(loops):
  code=['.text','loop:']
  for a,i in body:
   i=i.split('#')[0].strip()
   if i.startswith('jne '):i='jne loop'
   code.append(i)
  asm=p/f'{name}-loop{n}.s';asm.write_text('\n'.join(code)+'\n')
  cmd=['/usr/lib/llvm-21/bin/llvm-mca','-mcpu=znver1','-iterations=100','-bottleneck-analysis',str(asm)]
  r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,r.stderr
  (p/f'{name}-loop{n}-mca.txt').write_text(r.stdout+r.stderr)
  fields={}
  for key in ['Instructions','Total Cycles','Total uOps','Block RThroughput']:
   fields[key]=float(re.search(r'^'+key+r':\s*([0-9.]+)',r.stdout,re.M)[1])
  fields['has_prefetch']=any('prefetch' in i for _,i in body);meta[name].append(fields)
meta['limitations']=['Static znver1 scheduling model, not hardware measurement or peak utilization','Loop addresses normalized to local branch label; actual loop instructions retained','Baseline object and candidate use different build configurations; paired runtime compares these same objects','Does not model cache misses or concurrent host work']
(p/'model.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
