from pathlib import Path
import subprocess,json,os
p=Path('build-fp4-local/fp4-normalized-scale-products-confirm');settings=dict(STRATA_NVFP4_PREFETCH='2048',STRATA_NVFP4_128='0',STRATA_NVFP4_OFFSET_DOT='1',STRATA_NVFP4_CLANG_NT4='0');runs=[]
cases=[(1,scale,act,banks,core) for scale in ['normal'] for act in ['safe','min128'] for banks in [1,64] for core in [3,11]]+[(2,'normal','safe',banks,core) for banks in [1,64] for core in [3,11]]
for nt,scale,act,banks,core in reversed(cases):
 name=f'nt{nt}-{scale}-{act}-banks{banks}-cpu{core}';cmd=['taskset','-c',str(core),str(p/'probe'),str(nt),str(banks),scale,act]
 with (p/(name+'.log')).open('w') as f:r=subprocess.run(cmd,env=dict(os.environ,**settings),stdout=f,stderr=subprocess.STDOUT)
 runs.append(dict(name=name,tokens=nt,scale=scale,activation=act,banks=banks,core=core,command=cmd,returncode=r.returncode));(p/'runs.json').write_text(json.dumps(runs,indent=2)+'\n');print(name,r.returncode,flush=True);assert r.returncode==0
(p/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
