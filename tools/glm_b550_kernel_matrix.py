"""Measure canonical CPU expert widths on one captured B550 routing window.

Run only while the decoder/GPU benchmarks are idle. Results are expert-only;
route reduction uses uniform weights and excludes the rest of the model.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path);p.add_argument('capture',type=Path)
    p.add_argument('--decoder',type=Path,default=Path('build-hip/strata-glm-q2-kernel-bench'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,nargs='+',default=[4,8,12,16])
    p.add_argument('--widths',type=int,nargs='+',default=[1,2,4])
    p.add_argument('--rounds',type=int,default=5);p.add_argument('--skip',type=int,default=16)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    env={k:v for k,v in os.environ.items() if not k.startswith(('STRATA_','GLIBC_TUNABLES'))}
    env.update(STRATA_GLM_CANON='1',STRATA_GLM_LAYER_FLOW='1',STRATA_NATIVE_NUMA_LOCAL='0',
               STRATA_NATIVE_TASKS_PER_THREAD='6',STRATA_POOL_SPIN_US='20000',STRATA_Q23_PREFETCH='512',
               GLIBC_TUNABLES='glibc.cpu.x86_non_temporal_threshold=1048576')
    rows=[];hashes={}
    for width in a.widths:
        for workers in a.workers:
            command=['numactl','--interleave=all',str(a.decoder.resolve()),str(a.model.resolve()),
                     str(a.capture.resolve()),'mmap',str(width),str(workers),str(a.rounds),'native','3','44','row',str(a.skip)]
            out=a.output/f'w{width}-t{workers}.log'
            with out.open('w') as f:
                run=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
            if run.returncode:raise RuntimeError(f'kernel case failed: {out}')
            text=out.read_text();samples=[]
            for line in text.splitlines():
                if line.startswith('round='):
                    samples.append(dict(re.findall(r'(\w+)=([^ ]+)',line)))
            if len(samples)!=a.rounds:raise RuntimeError('missing kernel samples')
            current={x['output_hash'] for x in samples}
            if len(current)!=1 or width in hashes and hashes[width]!=current:
                raise RuntimeError('canonical outputs differ across workers')
            hashes[width]=current
            row=dict(width=width,workers=workers,rounds=a.rounds,skip_positions=a.skip,
                     median_ms=statistics.median(float(x['ms']) for x in samples),
                     median_GB_s=statistics.median(float(x['packed_GB_s']) for x in samples),
                     output_hash=next(iter(current)),command=command,samples=samples)
            rows.append(row);(a.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k not in ('command','samples')}),flush=True)


if __name__=='__main__':main()
