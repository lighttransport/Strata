from pathlib import Path
import subprocess,os,json
p=Path('build-fp4-local/q3-direct-scale-loads');runs=[]
settings=dict(STRATA_Q23_PREFETCH='1024',STRATA_Q23_NTA='0',STRATA_Q2K_LUT='0',STRATA_Q3_CLANG_NT2='0',STRATA_Q3_CLANG_NT4='0')
for ty,nt,raw in [(11,1,False),(11,1,True),(11,2,False),(10,1,False)]:
 for copies in [1,64]:
  for core in [3,11]:
   name=f'ty{ty}-nt{nt}-min128{int(raw)}-banks{copies}-cpu{core}'
   env=dict(os.environ,**settings);env.pop('STRATA_PROBE_MIN128',None)
   if raw:env['STRATA_PROBE_MIN128']='1'
   cmd=['taskset','-c',str(core),str(p/'probe'),str(nt),str(copies),str(ty)]
   with (p/(name+'.log')).open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
   runs.append(dict(name=name,type=ty,tokens=nt,min128=raw,copies=copies,core=core,command=cmd,returncode=r.returncode));(p/'runs.json').write_text(json.dumps(runs,indent=2)+'\n');print(name,r.returncode,flush=True);assert r.returncode==0
(p/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
