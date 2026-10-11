"""Persistent local coding drafter; stdin/stdout carry protocol-1 target token IDs.

Uses a pinned llama-server executable on loopback and the exact target GGUF
 tokenizer. Weights, KV and scratch are constrained by measured startup VRAM.
The engine owns verification; this worker never commits its own draft tokens.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import threading
import urllib.request
from strata_tokenizer import Tokenizer
from glm_low_memory_bench import validate_hip_link


def bridge(tokenizer, context, proposal, limit):
    # Strict UTF-8 decoding rejects byte fragments rather than inventing replacement characters.
    text = tokenizer.decode(context, errors='strict')
    canonical = tokenizer.encode(text, parse_special=True)
    combined = tokenizer.encode(text + proposal, parse_special=True)
    if combined[:len(canonical)] != canonical:
        return []
    # Generated target IDs need not be the canonical tokenization of their
    # text. Return only the new suffix; the engine retains the original IDs.
    return combined[len(canonical):len(canonical)+limit]


def request(base, key, path, body=None, timeout=1):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base+path, data=data,
        headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server',required=True,type=Path)
    p.add_argument('--draft-model',required=True,type=Path)
    p.add_argument('--target-model',required=True,type=Path)
    p.add_argument('--context',type=int,default=4096)
    p.add_argument('--vram-mib',type=int,default=768)
    p.add_argument('--pci-device',default='0000:0a:00.0')
    a=p.parse_args()
    if a.context<512 or a.context>4096 or a.vram_mib<1 or a.vram_mib>768:
        p.error('context must be 512..4096; VRAM cap 1..768 MiB')
    if not all(x.is_file() for x in [a.server,a.draft_model,a.target_model]):p.error('missing executable/model')
    gpu=Path('/sys/bus/pci/devices')/a.pci_device
    validate_hip_link(gpu)
    before=int((gpu/'mem_info_vram_used').read_text())
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    key=secrets.token_hex(24);base=f'http://127.0.0.1:{port}'
    command=[str(a.server.resolve()),'-m',str(a.draft_model.resolve()),'-ngl','99','-c',str(a.context),
             '--split-mode','none','--main-gpu','0',
             '--parallel','1','--host','127.0.0.1','--port',str(port),'--api-key',key,
             '--no-webui','--no-context-shift','--cache-reuse','256','-b','128','-ub','128']
    # Parent death terminates the server, including when the engine disables this worker.
    worker_pid=os.getpid()
    def child_setup():
        import ctypes,signal
        if ctypes.CDLL(None).prctl(1,signal.SIGKILL,0,0,0):os._exit(127)
        if os.getppid()!=worker_pid:os._exit(127)
    server=subprocess.Popen(command,stdout=sys.stderr,stderr=sys.stderr,preexec_fn=child_setup)
    try:
        start=time.monotonic()
        while True:
            if server.poll() is not None:raise RuntimeError('draft server exited')
            if time.monotonic()-start>25:raise RuntimeError('draft server startup timeout')
            try:
                if request(base,key,'/health',timeout=.5).get('status')=='ok':break
            except (OSError,ValueError):pass
            time.sleep(.1)
        # Before the target loads, exercise a full draft context and sample allocation
        # peaks. A startup-only read misses lazy graph/scratch allocations.
        peaks=[int((gpu/'mem_info_vram_used').read_text())-before]
        done=threading.Event()
        def sample():
            while not done.wait(.005):
                try:peaks.append(int((gpu/'mem_info_vram_used').read_text())-before)
                except OSError:pass
        sampler=threading.Thread(target=sample);sampler.start()
        try:request(base,key,'/completion',{'prompt':[0]*(a.context-64),'n_predict':1,'temperature':0},timeout=10)
        finally:done.set();sampler.join()
        peaks.append(int((gpu/'mem_info_vram_used').read_text())-before)
        used=max(peaks)
        if used>a.vram_mib*2**20:raise RuntimeError(f'drafter VRAM exceeds {a.vram_mib} MiB')
        tokenizer=Tokenizer.from_gguf(a.target_model)
        print(json.dumps({'protocol':1,'ready':True,'vram_mib':used/2**20}),flush=True)
        last_id=0
        for line in sys.stdin:
            if len(line)>1024*1024:raise ValueError('oversized worker request')
            r=json.loads(line);identity=r['request_id'];context=r['context'];limit=r['max_tokens']
            if r['protocol']!=1 or type(identity)!=int or identity<=last_id or type(limit)!=int or not 0<=limit<=7:
                raise ValueError('invalid worker request')
            last_id=identity
            if not isinstance(context,list) or len(context)>131072 or any(type(t)!=int or t<0 or t>=len(tokenizer.tokens) for t in context):
                raise ValueError('invalid context')
            prime=r.get('prime',False)
            if type(prime)!=bool:raise ValueError('invalid prime flag')
            try:
                text=tokenizer.decode(context,errors='strict')
                if prime:tokenizer.encode(text,parse_special=True) # warm target BPE cache once
                # Use a stable tail in draft-token space; target verification uses the full context.
                draft_ids=request(base,key,'/tokenize',{'content':text,'add_special':False})['tokens']
                tail=draft_ids[-(a.context-64):]
                answer=request(base,key,'/completion',{'prompt':tail,'n_predict':1 if prime else max(1,limit*2),
                    'temperature':0,'cache_prompt':True,'stream':False},timeout=10 if prime else .1)
                tokens=[] if prime else bridge(tokenizer,context,answer.get('content',''),limit)
            except (OSError,ValueError,UnicodeError):
                if prime:raise
                tokens=[]
            print(json.dumps({'protocol':1,'request_id':identity,'tokens':tokens,'primed':prime}),flush=True)
    finally:
        server.terminate()
        try:server.wait(timeout=3)
        except subprocess.TimeoutExpired:server.kill();server.wait(timeout=3)


if __name__=='__main__':main()
