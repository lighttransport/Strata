from pathlib import Path
import subprocess,os,json,hashlib
p=Path('build-fp4-local/bw-audit');env=dict(os.environ,OMP_WAIT_POLICY='PASSIVE',OMP_DYNAMIC='FALSE');runs=[]
for i,(name,size) in enumerate([('installed',1024),('baseline',1024),('probe',1024),('probe',4096),('baseline',1024),('installed',1024),('probe',1024)]):
 binary='tools/membw/membw' if name=='installed' else str(p/name)
 args=[binary,'--threads','16','--mib',str(size),'--repeats','30','--kernel','read']
 print('START',i,name,size,flush=True)
 with (p/f'{i}-{name}-{size}.log').open('w') as f:r=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT)
 runs.append({'i':i,'kind':name,'mib':size,'command':args,'returncode':r.returncode});assert r.returncode==0
 print('DONE',i,flush=True)
(p/'runs.json').write_text(json.dumps(runs,indent=2)+'\n')
(p/'settings.json').write_text(json.dumps({'OMP_WAIT_POLICY':'PASSIVE','OMP_DYNAMIC':'FALSE','numa_balancing':Path('/proc/sys/kernel/numa_balancing').read_text().strip(),'binary_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p/'baseline',p/'probe',Path('tools/membw/membw')]}},indent=2)+'\n')
