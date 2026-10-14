#!/usr/bin/env python3
"""Read-only I/O/device probes. Build commands are in docs/GLM_TR16_MIXED_REDESIGN.md."""
import subprocess,os,json,pathlib,concurrent.futures,time
import argparse,sys,hashlib
def main():
    parser=argparse.ArgumentParser(description='Bounded TR16 component probes; requires idle GPUs/CPU and prebuilt probe binaries')
    parser.add_argument('--binaries',type=pathlib.Path,required=True)
    parser.add_argument('--model-file',type=pathlib.Path,required=True)
    parser.add_argument('--work-file',type=pathlib.Path,required=True)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    parser.add_argument('--rocm',default='/opt/rocm/core-10.1')
    args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env['LD_LIBRARY_PATH']=args.rocm+'/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    env['OMP_PROC_BIND']='spread';env['OMP_PLACES']='cores'
    base=str(args.binaries.resolve())+'/'
    topology=[]
    for device in pathlib.Path('/sys/bus/pci/devices').iterdir():
        cls=(device/'class').read_text().strip()
        if cls.startswith(('0x03','0x0108')):
            links=[]
            for node in [device.resolve(),*device.resolve().parents]:
                if (node/'current_link_speed').exists():
                    links.append(dict(pci=node.name,speed=(node/'current_link_speed').read_text().strip(),
                                      width=int((node/'current_link_width').read_text()),numa=int((node/'numa_node').read_text())))
            topology.append(dict(pci=device.name,pci_class=cls,links=links))
    (out/'topology.json').write_text(json.dumps(dict(pci=topology,block_devices=json.loads(subprocess.check_output(
        ['lsblk','-J','-o','NAME,UUID,MOUNTPOINTS,MODEL,SIZE'],text=True))),indent=2)+'\n')
    def run(tag,cmd):
     t=time.time();p=subprocess.run(cmd,env=env,capture_output=True,text=True)
     (out/(tag+'.log')).write_text(p.stdout);(out/(tag+'.stderr')).write_text(p.stderr)
     rows=[json.loads(s) for s in p.stdout.splitlines() if s.startswith('{')]
     d=dict(tag=tag,command=cmd,returncode=p.returncode,started_unix=t,seconds=time.time()-t,rows=rows)
     (out/(tag+'.json')).write_text(json.dumps(d,indent=2)+'\n');print(tag,p.returncode,flush=True)
     if p.returncode:raise RuntimeError(tag+': '+p.stderr[-500:])
     return d
    for gpu,node in [('cuda',1),('hip',0)]:
     for mem in (node,1-node):
      for mode in ('h2d','d2h'):
       run(f'{gpu}-{mode}-mem{mem}',['numactl','--cpunodebind='+str(node),'--membind='+str(mem),base+'probe-'+gpu,mode,'.35'])
     run(gpu+'-kernel',['numactl','--cpunodebind='+str(node),'--membind='+str(node),base+'probe-'+gpu,'kernel','.35'])
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
     fs=[pool.submit(run,gpu+'-h2d-concurrent',['numactl','--cpunodebind='+str(node),'--membind='+str(node),base+'probe-'+gpu,'h2d','1']) for gpu,node in [('cuda',1),('hip',0)]]
     for f in fs:f.result()
    for node in (0,1):
     for mem in (0,1):run(f'ram-cpu{node}-mem{mem}',['numactl','--cpunodebind='+str(node),'--membind='+str(mem),base+'probe-ram','8','4k'])
    run('ram-interleave',['numactl','--interleave=all',base+'probe-ram','15','4k'])
    files={'model':str(args.model_file),'work':str(args.work_file)}
    for disk,path in files.items():
     for depth in (1,4):
      for access in ('sequential','random'):
       cmd=[sys.executable,str(pathlib.Path(__file__).with_name('glm_nvme_read_probe.py')),path,'--seconds','2','--depth',str(depth)]
       if access=='random':cmd+=['--random']
       run(f'nvme-{disk}-qd{depth}-{access}',cmd)

if __name__=="__main__":main()
