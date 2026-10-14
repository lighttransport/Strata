"""Freeze and resume strict-Vulkan output comparisons; timings are unqualified.

Training and holdout prompts stay separate. A gate needs complete stop-terminated
outputs, identical token IDs and passing resource/fault checks, not just a prefix.
"""
import argparse
import hashlib
import gzip
import json
from pathlib import Path
from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer
from glm_coding_perf_matrix import prepare, FIXTURES
from glm_coding_quality import TRAINING, check_output
from glm_guarded_run import run
from glm_low_memory_bench import parse_log, decode_complete
from glm_chat_stops import chat_stop_ids
from glm_q2_humaneval import DATA_SHA256, REVISION, solution, isolated_test
from jinja2.sandbox import SandboxedEnvironment


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(config, output, dataset):
    output.mkdir(parents=True,exist_ok=True)
    manifest=output/'manifest.json'
    if manifest.exists():raise FileExistsError('quality fixtures are already frozen')
    cfg=json.loads(config.read_text());model=Path(cfg['model']);tk=Tokenizer.from_gguf(model)
    metadata=GGUFFile(model).metadata;template=metadata['tokenizer.chat_template'];rows=[]
    tasks=TRAINING
    specs=[(name,0,'training',1536) for name in tasks]
    specs += [(name,length,'long_holdout',4096) for name,length in
              [('prime',8192),('json_escape',8192),('csv',16384),('parse_u64',16384)]]
    for name,length,split,maximum in specs:
        key=name+('-'+str(length) if length else '')
        chat,ids=prepare(tk,template,name,length,tasks=tasks if not length else FIXTURES)
        path=output/(key+'.ids');path.write_text(','.join(map(str,ids))+'\n')
        (output/(key+'.prompt.txt')).write_text(chat)
        rows.append({'name':key,'task':name,'split':split,'prompt':path.name,'sha256':digest(path),
                     'prompt_tokens':len(ids),'generate':maximum})
    raw=dataset.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=DATA_SHA256:raise ValueError('HumanEval dataset checksum changed')
    human=[json.loads(line) for line in gzip.decompress(raw).splitlines()]
    env=SandboxedEnvironment(extensions=['jinja2.ext.loopcontrols'])
    env.globals['raise_exception']=lambda message:(_ for _ in ()).throw(ValueError(message))
    chat_template=env.from_string(template)
    for task in human[:30]:
        key=task['task_id'].replace('/','_')
        prompt='Complete this Python function. Return the complete function in one Python code block with any required imports. Keep reasoning very brief.\n\n'+task['prompt']
        chat=chat_template.render(messages=[dict(role='user',content=prompt)],tools=None,
                                   add_generation_prompt=True,reasoning_effort='low')
        ids=tk.encode(chat,parse_special=True);path=output/(key+'.ids')
        path.write_text(','.join(map(str,ids))+'\n');(output/(key+'.prompt.txt')).write_text(chat)
        oracle=output/(key+'.case.json');oracle.write_text(json.dumps(task)+'\n')
        rows.append({'name':key,'task':task['task_id'],'split':'holdout','prompt':path.name,
                     'sha256':digest(path),'prompt_tokens':len(ids),'generate':1024,
                     'oracle':oracle.name,'oracle_sha256':digest(oracle)})
    manifest.write_text(json.dumps({'schema':2,'dataset_revision':REVISION,'dataset_sha256':DATA_SHA256,
                       'model_header_reference':cfg['model'],
                       'stop_ids':chat_stop_ids(metadata),'tasks':rows},indent=2)+'\n')


def recheck_scientific_timing(prefix, config, prompt):
    """Recover only an offline completion-parser failure; preserve raw evidence."""
    guard=json.loads(prefix.with_suffix('.guard.json').read_text())
    result=json.loads(prefix.with_suffix('.result.json').read_text())
    cfg=json.loads(config.read_text())
    if (cfg.get('backend')!='vulkan' or guard.get('test_kind')!='correctness' or guard.get('passed') is not False or
        guard.get('exit_code')!=2 or guard.get('termination_reason') or guard.get('faults') or
        not guard.get('pci_before') or guard.get('pci_before')!=guard.get('pci_after') or
        result.get('exit_code')!=0 or result.get('rejected') or result.get('determinism_failure') or
        result.get('complete') is not False or result.get('measurements',{}).get('prefill')):
        raise ValueError('failure is not an isolated timing-parser completion error')
    group=result.get('cgroup_after',{})
    if group.get('memory.swap.current')!=0 or any(group.get('memory.events',{}).get(k,0)
                                              for k in ('oom','oom_kill','oom_group_kill')):
        raise ValueError('resource guard failed; refusing offline revalidation')
    if (digest(config)!=guard['config_sha256'] or digest(prompt)!=guard['prompt_sha256'] or
        digest(Path(cfg['exe']))!=guard['executable_sha256'] or
        digest(prefix.with_suffix('.stdout'))!=result['output_sha256']):
        raise ValueError('frozen revalidation inputs changed')
    measurements=parse_log(prefix.with_suffix('.log').read_text())
    command=guard['command'];maximum=int(command[command.index('--tokens')+1]);trials=int(command[command.index('--trials')+1])
    ids=list(map(int,prefix.with_suffix('.stdout').read_text().split()))
    engine=json.loads(prefix.with_suffix('.vulkan.json').read_text())['output_ids']
    count=len(prompt.read_text().replace(',',' ').split())
    if (trials!=1 or len(measurements['prefill'])!=1 or measurements['prefill'][0]['tokens']!=count or
        measurements['prefill'][0]['trial']!=0 or engine!=[ids] or not ids or ids[-1] not in cfg['stop_ids'] or
        not decode_complete(ids,measurements['decode'],maximum,trials,cfg['stop_ids'])):
        raise ValueError('saved run is not complete and stop-terminated')
    suffixes=('.guard.json','.result.json','.stdout','.log','.vulkan.json','.kernel.log')
    evidence={'schema':1,'method':'scientific-timing-parser','test_kind':'correctness',
              'passed':True,'performance_qualified':False,'original_guard_passed':False,
              'original_guard_exit_code':guard['exit_code'],'executable_sha256':guard['executable_sha256'],
              'config_sha256':guard['config_sha256'],'prompt_sha256':guard['prompt_sha256'],
              'raw_sha256':{suffix:digest(prefix.with_suffix(suffix)) for suffix in suffixes},
              'measurements':measurements}
    with prefix.with_suffix('.rechecked.guard.json').open('x') as stream:
        stream.write(json.dumps(evidence,indent=2)+'\n')
    return evidence


def load_completed(prefix):
    guard=json.loads(prefix.with_suffix('.guard.json').read_text())
    recovered=prefix.with_suffix('.rechecked.guard.json')
    if recovered.exists():
        revised=json.loads(recovered.read_text())
        if (revised.get('method')!='scientific-timing-parser' or revised.get('performance_qualified') is not False or
            set(revised['raw_sha256'])!={'.guard.json','.result.json','.stdout','.log','.vulkan.json','.kernel.log'} or
            any(digest(prefix.with_suffix(suffix))!=sha for suffix,sha in revised['raw_sha256'].items()) or
            any(revised[k]!=guard[k] for k in ('executable_sha256','config_sha256','prompt_sha256'))):
            raise ValueError('rechecked evidence changed')
        guard=revised
    result=json.loads(prefix.with_suffix('.vulkan.json').read_text())
    if not guard['passed'] or guard.get('test_kind')!='correctness':raise RuntimeError('quality run did not pass the correctness guard')
    ids=result['output_ids']
    if len(ids)!=1 or not ids[0]:raise ValueError('quality requires exactly one complete trial')
    return ids[0],guard


def evaluate(reference, candidate, inputs, output, splits, names, timeout, control_results=None):
    output.mkdir(parents=True,exist_ok=True)
    if control_results and not control_results.is_dir():raise ValueError('control evidence directory is missing')
    fixture=json.loads((inputs/'manifest.json').read_text());rows=[]
    configs={kind:json.loads(path.read_text()) for kind,path in [('control',reference),('candidate',candidate)]}
    if configs['control'].get('speculative','none')!='none':raise ValueError('strict control cannot speculate')
    if configs['control']['model']!=configs['candidate']['model']:raise ValueError('model must stay identical')
    available={task['name'] for task in fixture['tasks'] if task['split'] in splits}
    if names and set(names)-available:raise ValueError('unknown or unselected quality tasks')
    selected=[task for split in dict.fromkeys(splits) for task in fixture['tasks']
              if task['split']==split and (not names or task['name'] in names)]
    if not selected:raise ValueError('no requested fixtures')
    for task in selected:
        prompt=inputs/task['prompt']
        if digest(prompt)!=task['sha256']:raise ValueError('frozen prompt changed')
        records={}
        for kind,cfg in configs.items():
            reuse=kind=='control' and control_results and (control_results/(task['name']+'-control.guard.json')).exists()
            directory=control_results if reuse else output
            work=directory/(task['name']+'-'+kind)
            path=directory/(task['name']+'-'+kind+'.config.json')
            frozen=cfg|{'context':max(cfg['context'],task['prompt_tokens']+task['generate']),
                        'stop_ids':fixture['stop_ids']}
            if frozen['context']>32768:raise ValueError('quality fixture exceeds Vulkan context bound')
            text=json.dumps(frozen,indent=2)+'\n'
            if reuse and not path.exists():
                raise ValueError('requested frozen control evidence is missing')
            if path.exists() and path.read_text()!=text:raise ValueError('resumed config changed')
            if not path.exists():path.write_text(text)
            if not work.with_suffix('.guard.json').exists():
                if reuse:raise ValueError('requested frozen control guard is missing')
                result=run(path,prompt,work,tokens=task['generate'],trials=1,timeout=timeout,correctness_only=True)
                if not result['passed']:raise RuntimeError('quality guard failed; inspect retained evidence')
            records[kind]=load_completed(work)
            if records[kind][1]['executable_sha256']!=digest(Path(cfg['exe'])):
                raise ValueError('resumed executable changed; use a fresh output directory')
        expected,control_guard=records['control'];actual,candidate_guard=records['candidate']
        complete=expected[-1] in fixture['stop_ids'] and actual[-1] in fixture['stop_ids']
        row={'name':task['name'],'split':task['split'],'prompt_sha256':task['sha256'],
             'output_ids_match':actual==expected,'stop_terminated':complete,'passed':complete and actual==expected,
             'generated':len(actual),'control_executable_sha256':control_guard['executable_sha256'],
             'candidate_executable_sha256':candidate_guard['executable_sha256']}
        if task.get('oracle'):
            oracle_path=inputs/task['oracle']
            if digest(oracle_path)!=task['oracle_sha256']:raise ValueError('frozen oracle changed')
            oracle=json.loads(oracle_path.read_text());tokenizer=Tokenizer.from_gguf(Path(configs['control']['model']))
            text=tokenizer.decode([token for token in expected if token not in fixture['stop_ids']])
            code=solution(text,oracle['prompt'],oracle['entry_point'])
            row['strict_control_oracle']=isolated_test(code,oracle['test'],oracle['entry_point'])
            row['oracle_regression']=False if actual==expected else None
        elif task.get('task'):
            tokenizer=Tokenizer.from_gguf(Path(configs['control']['model']))
            text=tokenizer.decode([token for token in expected if token not in fixture['stop_ids']])
            try:
                row['strict_control_oracle']={'passed':True,'checks':check_output(text,task['task'])}
            except (ValueError, RuntimeError) as error:
                row['strict_control_oracle']={'passed':False,'reason':str(error)}
            row['oracle_regression']=False if actual==expected else None
        rows.append(row)
        (output/'summary.json').write_text(json.dumps({'test_kind':'correctness','timings_qualified':False,
                         'requested_cases':len(selected),'completed_cases':len(rows),
                         'rows':rows,'all_requested_passed':len(rows)==len(selected) and all(x['passed'] for x in rows),
                         'all_requested_oracles_passed':len(rows)==len(selected) and
                            all(x.get('strict_control_oracle',{}).get('passed',False) for x in rows)},indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if not row['passed']:raise RuntimeError('output gate failed; do not tune using holdout failures')
    if not rows:raise ValueError('no requested fixtures')


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    f=s.add_parser('freeze');f.add_argument('config',type=Path);f.add_argument('--output',type=Path,required=True)
    f.add_argument('--dataset',type=Path,default=Path('build-glm-3070-cuda12/HumanEval.jsonl.gz'))
    r=s.add_parser('run');r.add_argument('--control',type=Path,required=True);r.add_argument('--candidate',type=Path,required=True)
    r.add_argument('--inputs',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    r.add_argument('--splits',nargs='+',choices=['training','holdout','long_holdout'],default=['training'])
    r.add_argument('--tasks',nargs='+');r.add_argument('--timeout',type=int,default=10800)
    r.add_argument('--control-results',type=Path,help='reuse completed strict controls with identical config and executable')
    c=s.add_parser('recheck-timing');c.add_argument('prefix',type=Path)
    c.add_argument('--config',type=Path,required=True);c.add_argument('--prompt',type=Path,required=True)
    a=p.parse_args()
    if a.command=='freeze':freeze(a.config,a.output,a.dataset)
    elif a.command=='recheck-timing':recheck_scientific_timing(a.prefix,a.config,a.prompt)
    else:evaluate(a.control,a.candidate,a.inputs,a.output,a.splits,a.tasks,a.timeout,a.control_results)

if __name__=='__main__':main()
