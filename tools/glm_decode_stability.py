"""Repeat GLM short/4k coding decode and sample competing CPU activity."""
import argparse,pathlib,sys,subprocess,json,time,os,re,statistics,hashlib,tempfile
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer
from glm_generate import chat_prompt
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('model',type=pathlib.Path)
parser.add_argument('--decoder',type=pathlib.Path,default=pathlib.Path('build-glm/strata-glm-decode'))
parser.add_argument('--output',type=pathlib.Path,default=pathlib.Path('docs/glm53_flash_q2_initial_decode_measurement.json'))
args=parser.parse_args()
model=str(args.model.resolve())
tk=Tokenizer.from_gguf(pathlib.Path(model));m=GGUFFile(model).metadata
stops={int(m[k]) for k in ('tokenizer.ggml.eos_token_id','tokenizer.ggml.eot_token_id','tokenizer.ggml.eom_token_id') if k in m}
out=args.output;out.parent.mkdir(parents=True,exist_ok=True)
work=pathlib.Path(tempfile.mkdtemp(prefix='glm-stability-remeasure-'))
print('Temporary answers/logs:',work,flush=True)
short_prompt='Write a complete C++17 bool is_prime(unsigned n) function. Avoid overflow, handle 0 and 1 correctly, and return only one cpp code block with no explanation. Keep reasoning very brief.'
from jinja2.sandbox import SandboxedEnvironment
env=SandboxedEnvironment(extensions=['jinja2.ext.loopcontrols'])
env.globals['raise_exception']=lambda s: (_ for _ in ()).throw(ValueError(s))
short=env.from_string(m['tokenizer.chat_template']).render(messages=[{'role':'user','content':short_prompt}],tools=None,add_generation_prompt=True,reasoning_effort='low')
long=pathlib.Path('docs/fixtures/glm53_cpp_quality/chat_prompt.txt').read_text()
result={'model':pathlib.Path(model).name,'repetitions':3,'threads':15,'cpu_affinity':'auto','context':8192,'prefill_batch':2048,'gpu_budget_mib':12288,'cpu_prepack_mib':0,'decode_cache_mib':0,'result_stage':'initial Q2 measurements','runs':{}}
hz=os.sysconf('SC_CLK_TCK')
ancestors={os.getpid()};p=os.getpid()
while p>1:
 try:
  p=int(pathlib.Path(f'/proc/{p}/stat').read_text().split(') ',1)[1].split()[1]);ancestors.add(p)
 except OSError:break

def procs():
 values={}
 for path in pathlib.Path('/proc').iterdir():
  if not path.name.isdigit():continue
  try:
   s=(path/'stat').read_text();comm=s.split('(',1)[1].rsplit(')',1)[0];f=s.rsplit(') ',1)[1].split()
   values[int(path.name)]=(int(f[11])+int(f[12]),int(f[19]),comm)
  except (OSError,ValueError,IndexError):pass
 return values

def save():out.write_text(json.dumps(result,indent=2)+'\n')
for case,chat,mode,cap in [('short_chat_single',short,'none',256),('cpp4096_single',long,'none',512),('cpp4096_mtp',long,'mtp',512)]:
 ids=tk.encode(chat,parse_special=True); tokenfile=work/(case+'.ids');tokenfile.write_text(','.join(map(str,ids)))
 command=[str(args.decoder.resolve()),model,'@'+str(tokenfile),str(cap),'4096','15','--prefill-batch=2048','--gpu-budget-mib=12288','--context=8192','--decode-experts=cpu','--cpu-affinity=auto','--cpu-prepack-mib=0','--warm-weights','--decode-bench=3','--stop-ids='+','.join(map(str,sorted(stops)))]
 if mode=='mtp':command+=['--speculative=mtp','--draft-depth=1','--mtp-experts=gpu']
 print('START',case,'input_tokens',len(ids),flush=True)
 foreign=[];top={};phases=[];previous=procs();last_time=time.monotonic();start=last_time
 with (work/(case+'.log')).open('w') as err,(work/(case+'.tokens')).open('w') as stdout:
  proc=subprocess.Popen(command,stdout=stdout,stderr=err,env={**os.environ,'STRATA_NATIVE_NUMA_LOCAL':'0','STRATA_POOL_SPIN_US':'20000'})
  while proc.poll() is None:
   time.sleep(1);now=time.monotonic();current=procs();total=0
   for pid,(ticks,birth,name) in current.items():
    if pid in ancestors or pid==proc.pid or pid not in previous or previous[pid][1]!=birth:continue
    pct=max(0,(ticks-previous[pid][0])/hz/(now-last_time)*100);total+=pct
    top[name]=max(top.get(name,0),pct)
   log=(work/(case+'.log')).read_text(); trials=re.findall(r'DECODE_TRIAL index=(\d+)',log)
   phase='decode' if trials else 'prefill/setup'
   foreign.append(total);phases.append((phase,total))
   previous=current;last_time=now
 logs=(work/(case+'.log')).read_text(); generated=list(map(int,(work/(case+'.tokens')).read_text().split()))
 matches=re.findall(r'DECODE steps=(\d+) ms=([\d.]+) tok_s=([\d.]+)',logs) if mode=='none' else re.findall(r'SPECULATIVE source=mtp generated=(\d+)[^\n]* ms=([\d.]+) tok_s=([\d.]+)',logs)
 counts=[int(n)+(1 if mode=='none' else 0) for n,_,_ in matches]
 segments=[];i=0
 for n in counts:segments.append(generated[i:i+n]);i+=n
 decodeforeign=[v for phase,v in phases if phase=='decode']
 measured={'input_tokens':len(ids),'output_cap':cap,'speculative':mode,'exit_code':proc.returncode,'wall_seconds':time.monotonic()-start,'trials':[{'output_tokens_including_stop':n,'decode_ms':float(ms),'decode_tok_s':float(rate)} for n,(_,ms,rate) in zip(counts,matches)],'median_decode_tok_s':statistics.median(float(rate) for _,_,rate in matches) if matches else None,'repeat_ids_identical':len(segments)==3 and all(s==segments[0] for s in segments),'ended_on_stop':bool(segments) and all(s and s[-1] in stops for s in segments),'prompt_token_sha256':hashlib.sha256(','.join(map(str,ids)).encode()).hexdigest(),'foreign_cpu_percent_of_one_core':{'decode_mean':statistics.mean(decodeforeign) if decodeforeign else None,'decode_peak':max(decodeforeign) if decodeforeign else None,'all_mean':statistics.mean(foreign) if foreign else None,'top_process_peak':sorted(top.items(),key=lambda x:-x[1])[:8]},'major_page_faults':[int(x) for x in re.findall(r'FAULTS[^\n]*major=(\d+)',logs)],'prefill':re.findall(r'PREFILL[^\n]*',logs),'peak_gpu_allocated_mib':max(map(float,re.findall(r'peak_allocated_MiB=([\d.]+)',logs)),default=0)}
 if segments:
  measured['output_ids']=segments[0]; measured['output_token_sha256']=hashlib.sha256(','.join(map(str,segments[0])).encode()).hexdigest()
  (work/(case+'.md')).write_text(tk.decode([x for x in segments[0] if x not in stops]))
 result['runs'][case]=measured;save();print('DONE',case,'median',measured['median_decode_tok_s'],'trials',[t['decode_tok_s'] for t in measured['trials']],'foreign CPU',measured['foreign_cpu_percent_of_one_core'],flush=True)
 if proc.returncode or len(segments)!=3 or not measured['repeat_ids_identical']:raise SystemExit('Measurement failed; inspect temporary logs')
result['cpp_single_mtp_ids_identical'] = result['runs']['cpp4096_single']['output_ids'] == result['runs']['cpp4096_mtp']['output_ids']
result['stability_spread_percent'] = {name: (max(t['decode_tok_s'] for t in run['trials']) - min(t['decode_tok_s'] for t in run['trials'])) / run['median_decode_tok_s'] * 100 for name, run in result['runs'].items()}
result['timing_scope'] = 'Single: timed next-token transitions excluding initial greedy token; MTP: generated progress including stop. Both exclude setup, prefill and priming.'
save()
print('ALL COMPLETE; single/MTP equality:',result['cpp_single_mtp_ids_identical'],flush=True)
