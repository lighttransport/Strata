"""Small ASCII terminal chat for the native GLM decoder, with live decode timing."""
import argparse
import curses
import json
import os
import pathlib
import sys
import tempfile
import textwrap
import threading
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'tools')]
from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer
from glm_generate import metadata_shard
from serve.frontend import ChatTemplate
from serve.server import GlmEngine


def visible(text):
    # The embedded template starts an open reasoning block.
    if '</think>' in text:
        return text.split('</think>', 1)[1].strip()
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=pathlib.Path)
    parser.add_argument('--decoder', type=pathlib.Path, default=REPO/'build-glm/strata-glm-decode')
    parser.add_argument('--prompt', default='Write a complete C++17 bool is_prime(unsigned n) function. Avoid overflow, handle 0 and 1 correctly, and return only one cpp code block with no explanation. Keep reasoning very brief.')
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--threads', type=int, default=15)
    parser.add_argument('--context', type=int, default=8192)
    parser.add_argument('--prefill-batch', choices=('8','2048','4096','8192','16384','auto'), default='2048')
    parser.add_argument('--speculative', choices=('none','mtp'), default='none')
    parser.add_argument('--prompt-file', type=pathlib.Path, help='UTF-8 prompt; overrides --prompt')
    parser.add_argument('--raw-prompt', action='store_true', help='use an already templated prompt')
    parser.add_argument('--gpu-devices', default='0')
    parser.add_argument('--gpu-budget-mib', type=int, default=12288)
    parser.add_argument('--prefill-experts', choices=('mmq','f16','f16-batched'), default='mmq')
    parser.add_argument('--prefill-expert-cache-mib', default='0')
    parser.add_argument('--lock-weights', action='store_true')
    parser.add_argument('--decode-prefill-cache', action='store_true')
    parser.add_argument('--decode-cache-mib', default='0')
    parser.add_argument('--decode-graphs', action='store_true')
    parser.add_argument('--decode-cache-adapt', action='store_true')
    parser.add_argument('--ignore-stop', action='store_true', help='fixed token cap, as in the decode benchmark')
    parser.add_argument('--warmup-repetitions', type=int, default=1)
    parser.add_argument('--warmup-tokens', type=int, default=0, help='untimed request before showing the TUI')
    parser.add_argument('--demo', action='store_true', help='exit 4 seconds after the first answer')
    parser.add_argument('--save', type=pathlib.Path, help='save the answer and timing JSON under this prefix')
    args = parser.parse_args()
    if args.tokens < 1 or args.threads < 1 or args.context < args.tokens+1:
        parser.error('invalid token, thread or context counts')
    if args.warmup_tokens < 0 or args.warmup_tokens + 1 > args.context or args.warmup_repetitions < 1:
        parser.error('invalid warmup token count')
    if args.prompt_file:
        args.prompt = args.prompt_file.read_text(encoding='utf-8')
    metadata = GGUFFile(metadata_shard(args.model)).metadata
    if metadata.get('general.architecture') != 'glm5next':
        parser.error('expected a glm5next GGUF')
    tok = Tokenizer.from_gguf(metadata_shard(args.model))
    template = ChatTemplate(source=metadata['tokenizer.chat_template'])
    stops = {int(metadata[k]) for k in ('tokenizer.ggml.eos_token_id','tokenizer.ggml.eot_token_id',
              'tokenizer.ggml.eom_token_id') if k in metadata}
    runtime = pathlib.Path(tempfile.mkdtemp(prefix='strata-chat-'))
    engine = None
    state = dict(text='', phase='ready', count=0, started=0, first=None, done=None,
                 result={}, error=None, prompt=args.prompt, prompt_tokens=0)
    lock = threading.Lock()
    cancel = threading.Event()
    worker = None
    messages = []

    def run_request(prompt):
        nonlocal messages
        try:
            pending = messages + [{'role':'user','content':prompt}]
            ids = tok.encode(prompt if args.raw_prompt else template.render(pending, reasoning_effort='low'), parse_special=True)
            if len(ids)+args.tokens > args.context:
                raise ValueError('Context full; restart chat or increase --context.')
            with lock:
                state['prompt_tokens'] = len(ids)
            generated = []
            for token in engine.generate(ids,args.tokens,{},cancel):
                if token is None:
                    continue
                if token in stops:
                    continue
                generated.append(token)
                now = time.monotonic()
                decoded = tok.decode(generated, errors='ignore')
                with lock:
                    state.update(text=decoded,phase='decode',count=len(generated))
                    if state['first'] is None:
                        state['first'] = now
            result = dict(engine.last)
            with lock:
                state.update(phase='done',done=time.monotonic(),result=result)
            if not cancel.is_set():
                messages = pending + [{'role':'assistant','content':visible(decoded) if generated else ''}]
        except Exception as exc:
            with lock:
                state.update(phase='error',error=str(exc),done=time.monotonic())

    def launch(prompt):
        nonlocal worker
        cancel.clear()
        with lock:
            state.update(text='',phase='prefill',count=0,started=time.monotonic(),first=None,
                         done=None,result={},error=None,prompt=prompt,prompt_tokens=0)
        worker = threading.Thread(target=run_request,args=(prompt,),daemon=True)
        worker.start()

    def screen(ui):
        curses.curs_set(0)
        ui.timeout(100)
        launch(args.prompt)
        entry = ''
        while True:
            with lock:
                snapshot = dict(state)
            height,width = ui.getmaxyx()
            ui.erase()
            def line(row,text):
                if 0 <= row < height:
                    try:
                        ui.addnstr(row,0,text,max(0,width-1))
                    except curses.error:
                        pass
            line(0,'GLM5.3Flash Q2 | Strata native chat | '+args.speculative+' decode')
            line(1,f'{args.threads} workers | GPU {args.gpu_devices} | budget {args.gpu_budget_mib / 1024:g} GiB/card | q: quit')
            line(2,'-'*max(0,width-1))
            line(3,'You: '+snapshot['prompt'].replace('\n',' ')[:max(0,width-6)])
            lines=[]
            output = snapshot['error'] or visible(snapshot['text']) or '[Waiting for first token...]'
            for text in output.splitlines():
                lines.extend(textwrap.wrap(text,width=max(1,width-2),replace_whitespace=False) or [''])
            capacity = max(1,height-10)
            for row,text in enumerate(lines[-capacity:],5):
                line(row,text)
            now=time.monotonic()
            elapsed=(snapshot['done'] or now)-snapshot['started']
            line(height-4,'-'*max(0,width-1))
            result=snapshot['result']
            if result:
                count=result['generated']; seconds=result['decode_ms']/1000
                rate=max(0,count-1)/seconds if seconds else 0
                line(height-3,f"FINAL decode {rate:.2f} tok/s | {count} tokens incl stop | {seconds:.2f}s | finish={result['finish']}")
                prefill=result['prompt_ms']/1000
                line(height-2,f"Prompt {snapshot['prompt_tokens']} tokens | prefill/prime {prefill:.2f}s | total {elapsed:.2f}s")
            elif snapshot['first'] is not None:
                seconds=now-snapshot['first']; rate=max(0,snapshot['count']-1)/seconds if seconds else 0
                line(height-3,f"LIVE decode {rate:.2f} tok/s | {snapshot['count']} visible token IDs | total {elapsed:.1f}s")
                line(height-2,'Live rate excludes first-token wait. Final rate uses the backend timer.')
            else:
                line(height-3,f"{snapshot['phase'].upper()} | input {snapshot['prompt_tokens']} tokens | waiting {elapsed:.1f}s")
                line(height-2,'Loading/prefill is excluded from decode tok/s. Memory guards remain enabled.')
            line(height-1,'> '+entry)
            ui.refresh()
            if args.demo and snapshot['done'] and now-snapshot['done']>4:
                break
            key=ui.getch()
            busy=worker is not None and worker.is_alive()
            if key==3 or (key==ord('q') and not entry and not busy):
                break
            if not busy and not args.demo:
                if key in (10,13) and entry.strip():
                    launch(entry.strip()); entry=''
                elif key in (curses.KEY_BACKSPACE,127,8):
                    entry=entry[:-1]
                elif 32 <= key <=126:
                    entry+=chr(key)
        cancel.set()
        if worker is not None:
            worker.join()

    try:
        print('Loading GLM with display-memory guards enabled...',flush=True)
        engine_args=[str(args.model.resolve()),str(args.context),'4096',str(args.threads),'0',
                     args.prefill_batch,'-1' if args.ignore_stop else ','.join(map(str,sorted(stops))),str(args.gpu_budget_mib),'0','0',
                     args.speculative,'1','auto','0',args.gpu_devices,args.prefill_expert_cache_mib,str(int(args.lock_weights)),
                     str(int(args.decode_prefill_cache)),args.decode_cache_mib,str(int(args.decode_graphs)),'256',
                     str(int(args.decode_cache_adapt)),args.prefill_experts]
        env=dict(os.environ,STRATA_NATIVE_NUMA_LOCAL='0')
        engine=GlmEngine(str(args.decoder.resolve()),engine_args,log=str(runtime/'engine.log'),env=env)
        if args.warmup_tokens:
            prompt = args.prompt if args.raw_prompt else template.render([{'role':'user','content':args.prompt}], reasoning_effort='low')
            warm_ids = tok.encode(prompt, parse_special=True)
            if len(warm_ids) + args.warmup_tokens > args.context:
                raise ValueError('Warmup exceeds context')
            for _ in range(args.warmup_repetitions):
                for _ in engine.generate(warm_ids,args.warmup_tokens,{},cancel):
                    pass
        curses.wrapper(screen)
    finally:
        cancel.set()
        if worker is not None and worker.is_alive():
            worker.join()
        if engine is not None:
            engine.close()
    if args.save:
        args.save.parent.mkdir(parents=True,exist_ok=True)
        args.save.with_suffix('.txt').write_text(visible(state['text'])+'\n')
        measurement=dict(state['result'],prompt_tokens=state['prompt_tokens'],
                         speculative=args.speculative,worker_threads=args.threads,
                         gpu_budget_mib=args.gpu_budget_mib,gpu_devices=args.gpu_devices,
                         warmup_tokens=args.warmup_tokens,warmup_repetitions=args.warmup_repetitions,
                         ignore_stop=args.ignore_stop,error=state['error'])
        args.save.with_suffix('.json').write_text(json.dumps(measurement,indent=2)+'\n')
    if state['error']:
        raise SystemExit(state['error'])
    result=state['result']
    if result:
        print(f"Decode: {max(0,result['generated']-1)*1000/max(result['decode_ms'],1e-9):.2f} tok/s; "
              f"{result['generated']} tokens (including stop); prefill/prime {result['prompt_ms']/1000:.2f}s")


if __name__=='__main__':
    main()
