from pathlib import Path
import subprocess,os,json
p=Path('build-fp4-local/bw-audit');records=[]
for i,mode in enumerate(['default','strict','strict','default']):
 args=[str(p/(mode+'-locality')),'--threads','16','--mib','1024','--repeats','30','--kernel','read']
 with (p/f'locality-{i}-{mode}.log').open('w') as f:r=subprocess.run(args,env=dict(os.environ,OMP_WAIT_POLICY='PASSIVE',OMP_DYNAMIC='FALSE'),stdout=f,stderr=subprocess.STDOUT)
 records.append({'i':i,'mode':mode,'command':args,'returncode':r.returncode});(p/'locality-runs.json').write_text(json.dumps(records,indent=2)+'\n');assert r.returncode==0
 print('DONE',i,mode,flush=True)
