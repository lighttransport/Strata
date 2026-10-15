"""Resident paired C++/recall gates; diagnostic timing only, preserve every completed task."""
import argparse,hashlib,json,os,pathlib,re,subprocess,sys,time
source=pathlib.Path(__file__).resolve().parents[3]
fixtures=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(source/'tools'))
from glm_generate import chat_prompt,metadata_shard
from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer
from glm_coding_quality import HOLDOUTS,TRAINING,check_output
from glm_v100_bench import experiment_environment,stop_process
from reproduce import profile_environment
ap=argparse.ArgumentParser();ap.add_argument('--decoder',required=True);ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--model',required=True);ap.add_argument('--tokens',type=int,default=1024);ap.add_argument('--pipeline',choices=('none','mtp'),required=True);ap.add_argument('--cut',required=True);ap.add_argument('--profile',choices=('q2','q23-native','q23-canonical'),required=True);args=ap.parse_args()
args.decoder=str(pathlib.Path(args.decoder).resolve());args.model=str(pathlib.Path(args.model).resolve());args.output=args.output.resolve()
for name in list(os.environ):
 if name.startswith('STRATA_'):del os.environ[name]
os.environ.update({k:v for k,v in profile_environment(args.profile).items() if k.startswith('STRATA_')})
args.output.mkdir(exist_ok=False)
tokenizer=Tokenizer.from_gguf(metadata_shard(pathlib.Path(args.model)))
metadata=GGUFFile(metadata_shard(pathlib.Path(args.model))).metadata
stops={int(metadata[k]) for k in ('tokenizer.ggml.eos_token_id','tokenizer.ggml.eot_token_id','tokenizer.ggml.eom_token_id') if k in metadata}
for spelling in ('<|user|>','<|observation|>'):stops.update(tokenizer.encode(spelling,parse_special=True))
command=[args.decoder,'--serve',args.model,'32768','4096','32','0','4096',','.join(map(str,sorted(stops))),'32768','0','0','mtp','1','auto','0','0,1','0','1','0','22528','1','256','0','mmq','4k','','','auto','--gpu-layout=layer','--layer-split='+args.cut,'--layer-pipeline='+args.pipeline,'--layer-memory=socket']
record={'complete':False,'performance_comparable':False,'command':command,'environment':experiment_environment(),'decoder_sha256':hashlib.sha256(pathlib.Path(args.decoder).read_bytes()).hexdigest(),'job':os.environ.get('SLURM_JOB_ID'),'tasks':[]}
def final_answer(text):
 if '</think>' in text:return text.rsplit('</think>',1)[-1]
 rendered=chat_prompt(metadata,'test')
 if rendered.rfind('<think>')>rendered.rfind('</think>'):raise ValueError('no final answer after reasoning delimiter')
 return text
def save():
 p=args.output/'results.json.tmp';p.write_text(json.dumps(record,indent=2)+'\n');p.replace(args.output/'results.json')
save()
with (args.output/'decoder.log').open('w') as error, (args.output/'protocol.log').open('w') as protocol, subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=error,text=True,bufsize=1,cwd=source) as process:
 def read():
  line=process.stdout.readline()
  if not line:raise RuntimeError('decoder ended; inspect decoder.log')
  protocol.write(line);protocol.flush();line=line.strip()
  if line.startswith('ERR '):raise RuntimeError(line)
  return line
 def request(ids,count):
  process.stdin.write(f"GEN {count} {','.join(map(str,ids))}\n");process.stdin.flush();answer=[]
  while True:
   line=read()
   if line.startswith('T '):answer.append(int(line.split()[1]))
   if line.startswith('DONE '):
    fields=line.split();assert int(fields[1])==len(answer) and int(fields[2])==len(ids)
    return answer,fields
 try:
  while not read().startswith('READY '):pass
  for name,prompt in dict(TRAINING,**HOLDOUTS).items():
   rendered=chat_prompt(metadata,prompt);ids=tokenizer.encode(rendered,parse_special=True)
   task={'name':name,'kind':'training_cpp' if name in TRAINING else 'heldout_cpp','prompt_tokens':len(ids),'prompt_sha256':hashlib.sha256(','.join(map(str,ids)).encode()).hexdigest(),'complete':False};record['tasks'].append(task);save()
   answer,done=request(ids,args.tokens);text=tokenizer.decode([i for i in answer if i not in stops]);(args.output/(name+'.output.md')).write_text(text)
   task.update(token_ids=answer,generated_tokens=len(answer),done=done,cap_hit=done[5]=='length',complete=True,token_sha256=hashlib.sha256(','.join(map(str,answer)).encode()).hexdigest())
   try:task['oracle_cases']=check_output(final_answer(text),name);task['passed']=True
   except (ValueError,subprocess.SubprocessError) as exc:task.update(passed=False,error=str(exc))
   save();print('QUALITY_TASK',name,task['passed'],len(answer),flush=True)
  # Preserve the 512-token timing failure; separately grade a longer completion.
  ids=tokenizer.encode((source/'docs/fixtures/glm53_cpp_quality/chat_prompt.txt').read_text(),parse_special=True)[:4096]
  answer,done=request(ids,args.tokens);text=tokenizer.decode([i for i in answer if i not in stops]);(args.output/'fixed-parser-4k.output.md').write_text(text)
  task=dict(name='fixed-parser-4k',kind='fixed_parser_4k',prompt_tokens=len(ids),prompt_sha256=hashlib.sha256(','.join(map(str,ids)).encode()).hexdigest(),token_ids=answer,done=done,cap_hit=done[5]=='length',complete=True,token_sha256=hashlib.sha256(','.join(map(str,answer)).encode()).hexdigest());record['tasks'].append(task)
  oracle_input=args.output/'fixed-parser-input.json'
  oracle_input.write_text(json.dumps(dict(complete=True,performance_comparable=False,decoder_sha256=record['decoder_sha256'],validation=dict(token_ids=answer)),indent=2)+'\n')
  with (args.output/'fixed-parser-oracle.log').open('w') as oracle_log:
   checked=subprocess.run([sys.executable,str(pathlib.Path(__file__).resolve().parent/'grade_parser.py'),args.model,str(oracle_input)],stdout=oracle_log,stderr=subprocess.STDOUT)
  oracle=json.loads((args.output/'fixed-parser-input/quality.json').read_text());task.update(passed=bool(oracle.get('passed')),oracle_exit=checked.returncode,oracle=oracle.get('oracle'),error=oracle.get('error'));save();print('FIXED_PARSER_4K',task['passed'],len(answer),flush=True)
  # Three separate held-out contracts after the same unrelated 8K coding prefix.
  background=tokenizer.encode((fixtures/'code-fixed-8192.txt').read_text(),parse_special=True)
  for name in ('fnv1a32','parse_i64','adler32'):
   ids=background+tokenizer.encode(chat_prompt(metadata,HOLDOUTS[name]),parse_special=True)
   answer,done=request(ids,args.tokens);text=tokenizer.decode([i for i in answer if i not in stops]);(args.output/('long-'+name+'.output.md')).write_text(text)
   task=dict(name='long-'+name,kind='long_cpp',prompt_tokens=len(ids),token_ids=answer,done=done,cap_hit=done[5]=='length',complete=True,token_sha256=hashlib.sha256(','.join(map(str,answer)).encode()).hexdigest());record['tasks'].append(task)
   try:task['oracle_cases']=check_output(final_answer(text),name);task['passed']=True
   except (ValueError,subprocess.SubprocessError) as exc:task.update(passed=False,error=str(exc))
   save();print('LONG_CPP',name,task['passed'],len(answer),flush=True)
  manifest=json.loads((source/'docs/fixtures/glm_b550_prefill_20261010/recall-manifest.json').read_text())
  for context in (8192,16384):
   fixture=next(x for x in manifest if x['context']==context);path=source/fixture['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==fixture['sha256']
   ids=[int(x) for x in path.read_text().split(',')];answer,done=request(ids,512);text=tokenizer.decode([i for i in answer if i not in stops]);(args.output/f'recall-{context}.output.md').write_text(text)
   value=text.rsplit('</think>',1)[-1].strip()
   if value.startswith('```json\n') and value.endswith('```'):value=value[8:-3].strip()
   try:decoded=json.loads(value)
   except json.JSONDecodeError:decoded=None
   task=dict(name=f'recall-{context}',kind='recall',prompt_tokens=len(ids),token_ids=answer,done=done,complete=True,decoded=decoded,expected=fixture['expected'],passed=decoded==fixture['expected'] and done[5]=='stop',token_sha256=hashlib.sha256(','.join(map(str,answer)).encode()).hexdigest());record['tasks'].append(task);save();print('RECALL',context,task['passed'],flush=True)
  process.stdin.write('QUIT\n');process.stdin.flush();process.wait(timeout=30)
  if process.returncode:raise RuntimeError('decoder exit '+str(process.returncode))
  record.update(complete=True,heldout_passes=sum(t['passed'] for t in record['tasks'] if t['kind']=='heldout_cpp'),heldout_tasks=20);save()
 finally:stop_process(process)
