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
ap=argparse.ArgumentParser();ap.add_argument('--decoder',required=True);ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--model',required=True);ap.add_argument('--tokens',type=int,default=1024);args=ap.parse_args()
args.output.mkdir(exist_ok=False)
tokenizer=Tokenizer.from_gguf(metadata_shard(pathlib.Path(args.model)))
metadata=GGUFFile(metadata_shard(pathlib.Path(args.model))).metadata
stops={int(metadata[k]) for k in ('tokenizer.ggml.eos_token_id','tokenizer.ggml.eot_token_id','tokenizer.ggml.eom_token_id') if k in metadata}
for spelling in ('<|user|>','<|observation|>'):stops.update(tokenizer.encode(spelling,parse_special=True))
command=[args.decoder,'--serve',args.model,'131072','4096',os.environ.get('GLM_WORKERS','35'),'0','8192',','.join(map(str,sorted(stops))),'32768','0','0','none','1','auto','0','0,1','auto','1','1','extend','1','256','1',os.environ.get('GLM_QUALITY_PREFILL','f16-batched')]
record={'complete':False,'performance_comparable':False,'command':command,'environment':experiment_environment(),'decoder_sha256':hashlib.sha256(pathlib.Path(args.decoder).read_bytes()).hexdigest(),'job':os.environ.get('SLURM_JOB_ID'),'tasks':[]}
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
  for name,prompt in HOLDOUTS.items():
   rendered=chat_prompt(metadata,prompt);ids=tokenizer.encode(rendered,parse_special=True)
   task={'name':name,'kind':'heldout_cpp','prompt_tokens':len(ids),'prompt_sha256':hashlib.sha256(','.join(map(str,ids)).encode()).hexdigest(),'complete':False};record['tasks'].append(task);save()
   answer,done=request(ids,args.tokens);text=tokenizer.decode([i for i in answer if i not in stops]);(args.output/(name+'.output.md')).write_text(text)
   task.update(token_ids=answer,generated_tokens=len(answer),done=done,cap_hit=done[5]=='length',complete=True,token_sha256=hashlib.sha256(','.join(map(str,answer)).encode()).hexdigest())
   try:task['oracle_cases']=check_output(text,name);task['passed']=True
   except (ValueError,subprocess.SubprocessError) as exc:task.update(passed=False,error=str(exc))
   save();print('QUALITY_TASK',name,task['passed'],len(answer),flush=True)
  # Same held-out task after an unrelated 8K coding prefix: inspect the complete response with its oracle.
  background=tokenizer.encode((fixtures/'code-fixed-8192.txt').read_text(),parse_special=True)
  name='fnv1a32';ids=background+tokenizer.encode(chat_prompt(metadata,HOLDOUTS[name]),parse_special=True)
  answer,done=request(ids,args.tokens);text=tokenizer.decode([i for i in answer if i not in stops]);(args.output/'long-fnv1a32.output.md').write_text(text)
  task=dict(name='long-fnv1a32',kind='long_cpp',prompt_tokens=len(ids),token_ids=answer,done=done,cap_hit=done[5]=='length',complete=True,token_sha256=hashlib.sha256(','.join(map(str,answer)).encode()).hexdigest());record['tasks'].append(task)
  try:task['oracle_cases']=check_output(text,name);task['passed']=True
  except (ValueError,subprocess.SubprocessError) as exc:task.update(passed=False,error=str(exc))
  save()
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
