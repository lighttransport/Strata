#!/usr/bin/env python3
"""CPU-only paired, in-place K32 packing benchmark; does not open model files."""
import argparse
import concurrent.futures
import ctypes as C
import hashlib
import json
import mmap
import os
from pathlib import Path
import statistics
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',required=True,type=Path)
    ap.add_argument('--candidate',required=True,type=Path)
    ap.add_argument('--baseline-source',required=True,type=Path)
    ap.add_argument('--candidate-source',type=Path,default=Path('src/host/glm_fp4.c'))
    ap.add_argument('--mib',type=int,default=128)
    ap.add_argument('--repeats',type=int,default=7)
    args=ap.parse_args()
    if args.mib<16 or args.mib%16 or args.repeats<3:ap.error('MiB must be a positive multiple of 16, at least 16; repeats >=3')
    libs=[C.CDLL(str(p.resolve())) for p in (args.baseline,args.candidate)]
    functions=[]
    for lib in libs:
        fn=lib.strata_glm_fp4_repack_k32
        fn.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int];fn.restype=C.c_int
        functions.append(fn)
    libc=C.CDLL(None);libc.memcmp.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];libc.memcmp.restype=C.c_int
    cores={}
    for cpu in sorted(os.sched_getaffinity(0)):
        root=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key=((root/'physical_package_id').read_text(),(root/'core_id').read_text())
        cores.setdefault(key,cpu)
    cpus=list(cores.values());size=args.mib*1024*1024;tile=65536
    adjacent=bytes(range(256))
    planar=bytearray()
    for offset in range(0,256,16):
        q=[(adjacent[offset+j//2]>>((j%2)*4))&15 for j in range(32)]
        planar.extend(q[j]|q[j+16]<<4 for j in range(16))
    patterns=[C.create_string_buffer(p*(tile//256)) for p in (adjacent,bytes(planar))]
    results=[]
    for workers in (1,8,16):
        if workers>len(cpus):continue
        chunk=size//workers
        with mmap.mmap(-1,size) as memory, concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            buffer=(C.c_uint8*size).from_buffer(memory);base=C.addressof(buffer)
            def fill(worker):
                os.sched_setaffinity(0,{cpus[worker]})
                for offset in range(0,chunk,tile):C.memmove(base+worker*chunk+offset,patterns[0],tile)
            def convert(worker,mode,direction):
                os.sched_setaffinity(0,{cpus[worker]})
                p=base+worker*chunk
                if functions[mode](p,p,chunk,direction):raise RuntimeError('packing failed')
            def verify(worker,which):
                for offset in range(0,chunk,tile):
                    if libc.memcmp(base+worker*chunk+offset,patterns[which],tile):raise RuntimeError('packing differs from independent pattern oracle')
            list(pool.map(fill,range(workers)))
            samples={(mode,direction):[] for mode in range(2) for direction in range(2)}
            for repeat in range(-1,args.repeats):
                for mode in ((0,1) if repeat%2==0 else (1,0)):
                    for direction in (0,1):
                        start=time.perf_counter()
                        list(pool.map(lambda w:convert(w,mode,direction),range(workers)))
                        seconds=time.perf_counter()-start
                        list(pool.map(lambda w:verify(w,1-direction),range(workers)))
                        if repeat>=0:samples[mode,direction].append(seconds)
            for (mode,direction),values in samples.items():
                results.append(dict(workers=workers,cpus=cpus[:workers],mode=('baseline','candidate')[mode],direction=('pack','unpack')[direction],seconds=values,median_ms=statistics.median(values)*1000,payload_GB_s=size/statistics.median(values)/1e9))
            del buffer
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    print(json.dumps(dict(payload_bytes=size,repeats=args.repeats,protocol='In-place, worker first-touch; physical-core affinity; alternating implementations; one warmup each. Thread dispatch and affinity are timed. Every full buffer checked against independent repeated pattern outside timing. Throughput counts input bytes once, not read+write DRAM traffic.',limitations=['Background CPU/DRAM contention not controlled','Synthetic pattern, no model file or GPU activity','Does not measure decode throughput'],baseline_library_sha256=sha(args.baseline),candidate_library_sha256=sha(args.candidate),baseline_source_sha256=sha(args.baseline_source),candidate_source_sha256=sha(args.candidate_source),results=results),indent=2))


if __name__=='__main__':main()
