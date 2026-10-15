"""Grade final answers without changing frozen generation records or raw text."""
import argparse,hashlib,json,pathlib,subprocess,sys
source=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(source/'tools'))
from glm_coding_quality import check_output,HOLDOUTS
from glm_generate import chat_prompt,metadata_shard
from gguf_reader import GGUFFile
ap=argparse.ArgumentParser();ap.add_argument('directory',type=pathlib.Path);a=ap.parse_args();raw=a.directory/'results.json';d=json.loads(raw.read_text());assert d['complete'];dest=a.directory/'results.final-answer-v3.json';assert not dest.exists()
metadata=GGUFFile(metadata_shard(pathlib.Path(d['command'][2]))).metadata
records=[]
for task in d['tasks']:
 r={k:task[k] for k in ('name','kind','complete','generated_tokens','cap_hit','token_sha256','prompt_tokens') if k in task}
 r['generated_tokens']=len(task['token_ids']);r['cap_hit']=task['done'][5]=='length'
 name='fnv1a32' if task['kind'] in ('long_cpp','recall') else task['name'];text=(a.directory/(task['name']+'.output.md')).read_text();rendered=chat_prompt(metadata,HOLDOUTS[name]);thinking=rendered.rfind('<think>')>rendered.rfind('</think>')
 r.update(raw_text_sha256=hashlib.sha256(text.encode()).hexdigest(),raw_grade_passed=task['passed'],naturally_stopped=task['done'][5]=='stop',thinking_prefix=thinking)
 if '</think>' in text:answer=text.rsplit('</think>',1)[1].strip()
 elif thinking:r.update(passed=False,error='No final answer after reasoning delimiter',final_answer_missing=True);records.append(r);continue
 else:answer=text
 r['final_answer_sha256']=hashlib.sha256(answer.encode()).hexdigest()
 if task['kind']=='recall':
  if answer.startswith('```json\n') and answer.endswith('```'):answer=answer[8:-3].strip()
  try:decoded=json.loads(answer)
  except json.JSONDecodeError:decoded=None
  r.update(decoded=decoded,expected=task['expected'],passed=decoded==task['expected'] and task['done'][5]=='stop');records.append(r);continue
 try:r.update(oracle_cases=check_output(answer,name),passed=True)
 except (ValueError,subprocess.SubprocessError) as exc:r.update(passed=False,error=str(exc))
 records.append(r);print('FINAL_GRADE',r['name'],r['passed'],r.get('cap_hit',False),flush=True)
result={'complete':True,'source_results_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'policy':'Grade only the final section after </think> when the prompt opens reasoning; missing final answers fail; preserve raw generations and caps','tasks':records,'heldout_passes':sum(t['passed'] for t in records if t['kind']=='heldout_cpp'),'heldout_tasks':20,'caps':sum(t.get('cap_hit',False) for t in records)}
p=dest.with_suffix('.tmp');p.write_text(json.dumps(result,indent=2)+'\n');p.replace(dest)
