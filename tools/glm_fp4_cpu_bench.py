#!/usr/bin/env python3
"""CPU-only CSF and host-copy benchmark. No GPU API, network, or weight writes.

Build library: cc -O2 -std=c11 -shared -fPIC -Iinclude src/host/glm_fp4.c -lm -o build-fp4-local/libcsf-bench.so
Run: python3 tools/glm_fp4_cpu_bench.py --model MODEL_DIR --library build-fp4-local/libcsf-bench.so
"""
import argparse
import concurrent.futures
import ctypes as C
import hashlib
import json
import mmap
import os
from pathlib import Path
import statistics
import struct
import time


def summary(seconds, amount):
    return dict(seconds=seconds, median_seconds=statistics.median(seconds),
                median_units_per_second=amount/statistics.median(seconds),
                min_units_per_second=amount/max(seconds), max_units_per_second=amount/min(seconds))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True,type=Path)
    ap.add_argument('--library',required=True,type=Path)
    ap.add_argument('--compare-library',type=Path)
    ap.add_argument('--compare-source',type=Path)
    ap.add_argument('--decoder-source',type=Path,default=Path('src/host/glm_fp4.c'))
    ap.add_argument('--repeats',type=int,default=7)
    args=ap.parse_args()
    if args.repeats<3:ap.error('at least three repeats required')
    lib=C.CDLL(str(args.library.resolve()))
    decode=lib.strata_glm_csf32_decode
    decode.argtypes=[C.c_size_t,C.c_size_t]+[v for _ in range(6) for v in (C.c_void_p,C.c_size_t)]
    decode.restype=C.c_int
    compare=None
    if args.compare_library:
        if not args.compare_source:ap.error('--compare-library requires --compare-source')
        compare_lib=C.CDLL(str(args.compare_library.resolve()))
        compare=compare_lib.strata_glm_csf32_decode
        compare.argtypes=decode.argtypes;compare.restype=decode.restype
    index=json.loads((args.model/'model.safetensors.index.json').read_text())['weight_map']
    headers={}
    def tensor(name):
        shard=index[name]
        if shard not in headers:
            with (args.model/shard).open('rb') as f:
                length=struct.unpack('<Q',f.read(8))[0]
                headers[shard]=(8+length,json.loads(f.read(length)))
        start,header=headers[shard]; t=header[name];lo,hi=t['data_offsets']
        with (args.model/shard).open('rb') as f:f.seek(start+lo);data=f.read(hi-lo)
        if len(data)!=hi-lo:raise ValueError('short tensor read')
        return t,data
    samples=[];digest=hashlib.sha256()
    for layer in (3,4):
        for expert in (0,1,2):
            for proj in ('gate_proj','up_proj','down_proj'):
                prefix=f'model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}'
                # Read only the small scale payloads; matrix dimensions from header.
                shard=index[prefix+'.weight']
                if shard not in headers:
                    with (args.model/shard).open('rb') as f:
                        length=struct.unpack('<Q',f.read(8))[0];headers[shard]=(8+length,json.loads(f.read(length)))
                shape=headers[shard][1][prefix+'.weight']['shape']
                rows,cols=shape[0],shape[1]*2//16
                streams=[]
                for part in ('bases','codes','page_bases','offsets','exceptions'):
                    _,data=tensor(prefix+'.weight_scale.csf32_'+part)
                    digest.update(data);streams.extend((C.create_string_buffer(data),len(data)))
                output=C.create_string_buffer(rows*cols)
                if decode(rows,cols,*streams,output,len(output)):raise ValueError('invalid CSF')
                samples.append((rows,cols,streams,bytes(output)))
    # One logical CPU per physical core, selected from this process's allowed set.
    cores={}
    for cpu in sorted(os.sched_getaffinity(0)):
        root=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key=((root/'physical_package_id').read_text().strip(),(root/'core_id').read_text().strip())
        cores.setdefault(key,cpu)
    cpus=list(cores.values())
    results=[]
    for workers in (1,8,16):
        if workers>len(cpus):continue
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            outputs=[C.create_string_buffer(max(r*c for r,c,_,_ in samples)) for _ in range(workers)]
            def csf(worker, fn=decode):
                os.sched_setaffinity(0,{cpus[worker]})
                out=outputs[worker]
                for _ in range(4):
                    for r,c,streams,_ in samples:
                        if fn(r,c,*streams,out,r*c):raise ValueError('decode failed')
            def run(fn):
                start=time.perf_counter()
                list(pool.map(fn,range(workers)))
                return time.perf_counter()-start
            run(csf)
            times=[];compare_times=[]
            if compare:run(lambda w:csf(w,compare))
            for repeat in range(args.repeats):
                if compare and repeat%2==0:compare_times.append(run(lambda w:csf(w,compare)))
                times.append(run(csf))
                if compare and repeat%2:compare_times.append(run(lambda w:csf(w,compare)))
            r,c,_,reference=samples[-1]
            if any(bytes(out)[:r*c]!=reference for out in outputs):raise ValueError('parallel output mismatch')
            amount=workers*4*sum(r*c for r,c,_,_ in samples)
            csf_result=summary(times,amount)
            comparison=summary(compare_times,amount) if compare else None
            # Separate 512 MiB source + 512 MiB destination, worker first-touch.
            size=512*1024*1024;chunk=size//workers
            with mmap.mmap(-1,size) as src, mmap.mmap(-1,size) as dst:
                srcbuf=(C.c_char*size).from_buffer(src);dstbuf=(C.c_char*size).from_buffer(dst)
                sp=C.addressof(srcbuf);dp=C.addressof(dstbuf)
                def touch(worker):
                    os.sched_setaffinity(0,{cpus[worker]})
                    C.memset(sp+worker*chunk,worker+1,chunk)
                    C.memset(dp+worker*chunk,0,chunk)
                run(touch)
                def copy(worker):
                    os.sched_setaffinity(0,{cpus[worker]})
                    for _ in range(8):C.memmove(dp+worker*chunk,sp+worker*chunk,chunk)
                run(copy)
                copy_times=[run(copy) for _ in range(args.repeats)]
                for worker in range(workers):
                    if dst[worker*chunk]!=worker+1 or dst[(worker+1)*chunk-1]!=worker+1:raise ValueError('copy mismatch')
                del srcbuf,dstbuf
            results.append(dict(workers=workers,cpus=cpus[:workers],csf_scales=csf_result,
                                comparison_csf_scales=comparison,
                                copy_payload_bytes=summary(copy_times,size*8)))
    report=dict(cpu=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
                samples=18,scale_payload_sha256=digest.hexdigest(),
                decoder_source_sha256=hashlib.sha256(args.decoder_source.read_bytes()).hexdigest(),
                library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                protocol='7 repeats by default; one warmup; thread-pool dispatch and affinity calls included; CSF warm-cache samples; memcpy 512 MiB source + destination worker first-touch, 8 passes; payload GB/s is not total DRAM traffic',
                limitations=['No GPU timing; miners remain running','CSF outputs checked against serial decoder, not an independent format oracle','No full-model cold-memory CSF timing or pinned-DMA measurement','Memory clock and background DRAM contention not measured'],results=results)
    if compare:
        report['comparison']=dict(order='alternating baseline/candidate each repeat',library_sha256=hashlib.sha256(args.compare_library.read_bytes()).hexdigest(),decoder_source_sha256=hashlib.sha256(args.compare_source.read_bytes()).hexdigest())
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
