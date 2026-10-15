#!/usr/bin/env python3
"""Bounded read-only O_DIRECT probe of an existing file; never drops page cache."""
import argparse,concurrent.futures,json,mmap,os,random,threading,time
from pathlib import Path

def probe(path,block,depth,seconds,random_reads):
    size=path.stat().st_size
    if size<block*depth or block%4096:raise ValueError('file too small or unaligned block')
    barrier=threading.Barrier(depth)
    def worker(index):
        fd=os.open(path,os.O_RDONLY|os.O_DIRECT);buf=mmap.mmap(-1,block)
        rng=random.Random(734+index);count=0;latencies=[];pos=index
        try:
            barrier.wait(timeout=10);start=time.monotonic()
            while time.monotonic()-start<seconds:
                offset=(rng.randrange(size//block) if random_reads else pos%(size//block))*block
                t=time.monotonic();n=os.preadv(fd,[buf],offset)
                if n!=block:raise OSError('short direct read')
                latencies.append((time.monotonic()-t)*1000);count+=n;pos+=depth
            return count,time.monotonic()-start,latencies
        finally:buf.close();os.close(fd)
    with concurrent.futures.ThreadPoolExecutor(depth) as pool:r=list(pool.map(worker,range(depth)))
    latency=sorted(x for item in r for x in item[2]);elapsed=max(x[1] for x in r)
    return dict(block_bytes=block,queue_depth=depth,access='random' if random_reads else 'sequential',
                bytes=sum(x[0] for x in r),seconds=elapsed,GB_s=sum(x[0] for x in r)/elapsed/1e9,
                median_read_ms=latency[len(latency)//2],p95_read_ms=latency[int(.95*(len(latency)-1))],direct=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('file',type=Path);p.add_argument('--seconds',type=float,default=2)
    blocks=p.add_mutually_exclusive_group()
    blocks.add_argument('--block-mib',type=int)
    blocks.add_argument('--block-kib',type=int,help='small random-I/O probe, aligned to 4 KiB')
    p.add_argument('--depth',type=int,default=1);p.add_argument('--random',action='store_true')
    a=p.parse_args()
    block=a.block_kib*1024 if a.block_kib is not None else (a.block_mib if a.block_mib is not None else 8)*2**20
    if not 0<a.seconds<=30 or not 1<=a.depth<=16 or not 4096<=block<=64*2**20 or block%4096:
        p.error('invalid bounded probe parameters')
    print(json.dumps(probe(a.file,block,a.depth,a.seconds,a.random)))
if __name__=='__main__':main()
