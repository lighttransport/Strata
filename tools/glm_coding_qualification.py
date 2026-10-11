"""Freeze and evaluate disjoint C++ quality inputs; emit a strict loss-budget gate.

Holdout tasks and corpus are evaluated only after candidate selection. No fitting
or configuration search is performed using their results.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from glm_coding_quality import ROOT, HOLDOUTS, TRAINING, check_output
from glm_guarded_run import run
from glm_chat_stops import chat_stop_ids
from glm_coding_perf_matrix import prepare
from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corpus(tokenizer, files, targets, output, sequence_targets=512):
    needed=targets+math.ceil(targets/sequence_targets)
    ids=[];inputs=[]
    for path in files:
        part=tokenizer.encode(path.read_text(),parse_special=False)
        take=min(len(part),needed-len(ids))
        if take:
            ids.extend(part[:take]);inputs.append({'path':str(path.relative_to(ROOT)),'sha256':digest(path)})
        if len(ids)==needed:break
    if len(ids)!=needed:raise ValueError('not enough disjoint code tokens for corpus')
    lines=[];remaining=targets;offset=0
    while remaining:
        count=min(sequence_targets,remaining)
        lines.append(','.join(map(str,ids[offset:offset+count+1])));offset+=count+1;remaining-=count
    output.write_text('\n'.join(lines)+'\n')
    return {'path':output.name,'targets':targets,'sequence_targets':sequence_targets,
            'sha256':digest(output),'source_files':inputs}


def freeze(config, output):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists(): raise ValueError('quality inputs already frozen; choose a new output directory')
    cfg=json.loads(config.read_text());model=Path(cfg['model']);tk=Tokenizer.from_gguf(model)
    meta=GGUFFile(model).metadata;template=meta['tokenizer.chat_template']
    stops=chat_stop_ids(meta)
    tasks=[]
    for name in HOLDOUTS:
        chat,ids=prepare(tk,template,name,1024,tasks=HOLDOUTS)
        p=output/(name+'.ids');p.write_text(','.join(map(str,ids)))
        (output/(name+'.prompt.txt')).write_text(chat)
        tasks.append({'name':name,'ids':p.name,'sha256':digest(p),'tokens':len(ids)})
    files=sorted([* (ROOT/'src/core').rglob('*.cpp'),*(ROOT/'include/strata/core').rglob('*.hpp')])
    holdout=[p for p in files if int(hashlib.sha256(str(p.relative_to(ROOT)).encode()).hexdigest()[:8],16)%4==0]
    tuning=[p for p in files if p not in holdout]
    corpora={'tuning':corpus(tk,tuning,4096,output/'tuning.ids'),
             'holdout':corpus(tk,holdout,32768,output/'holdout.ids'),
             'tuning_prefill':corpus(tk,tuning,4096,output/'tuning-prefill.ids',4096),
             'holdout_prefill':corpus(tk,holdout,32768,output/'holdout-prefill.ids',8192)}
    manifest={'schema_version':2,'reference_config_sha256':digest(config),'quality_module_sha256':digest(ROOT/'tools/glm_coding_quality.py'),
              'stop_policy_sha256':digest(ROOT/'tools/glm_chat_stops.py'),
              'stops':stops,'tasks':tasks,'corpora':corpora,'code_ppl_relative_budget':0.01,
              'heldout_task_count':20,'rule':'No additional heldout task failures; mandatory coding suite must pass; finite positive PPL with <=1% relative increase.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def evaluate(config, inputs, output, scope, tasks=False, mode="decode"):
    if tasks and (scope != 'holdout' or mode != 'decode'):
        raise ValueError('heldout tasks require holdout scope and decode mode')
    prefix=output/('code-'+scope+'-'+mode)
    if prefix.with_suffix('.tasks.jsonl').exists() or prefix.with_suffix('.guard.json').exists():
        raise FileExistsError('evaluation evidence already exists; choose a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((inputs/'manifest.json').read_text())
    if manifest.get('stop_policy_sha256')!=digest(ROOT/'tools/glm_chat_stops.py'):
        raise ValueError('quality stop policy changed or was not frozen; prepare fresh inputs')
    if manifest['quality_module_sha256']!=digest(ROOT/'tools/glm_coding_quality.py'):
        raise ValueError('oracle changed since freezing inputs')
    cfg=json.loads(config.read_text());cfg['speculative']='none';cfg.pop('stop_ids',None)
    cfg['decode_checks']=[];cfg['env']['STRATA_GLM_EVAL_WIDTH']='1'
    if mode=='prefill':
        cfg['eval_prefill']=True;cfg['decode_cache_mib']=0;cfg['decode_cache_slots']=0
    frozen=output/('evaluation-'+mode+'.config.json');frozen.write_text(json.dumps(cfg,indent=2)+'\n')
    corpus_key=scope if mode=='decode' else scope+'_prefill'
    cp=inputs/manifest['corpora'][corpus_key]['path']
    if digest(cp)!=manifest['corpora'][corpus_key]['sha256']: raise ValueError('frozen corpus changed')
    prefix=output/('code-'+scope+'-'+mode)
    print('EVAL',config.stem,scope,flush=True)
    guard=run(frozen,inputs/manifest['tasks'][0]['ids'],prefix,tokens=2,trials=1,timeout=3600,extra=['--eval-corpus',str(cp.resolve())])
    result={'scope':scope,'corpus_sha256':digest(cp),'config_sha256':digest(config),'passed':False,'tasks':{},'guard_passed':guard['passed']}
    if guard['passed']:
        record=json.loads(prefix.with_suffix('.result.json').read_text());evaluation=record['evaluation']
        last=evaluation[-1];result.update(perplexity=last['perplexity'],nll=last['nll'],targets=last['tokens'],passed=True)
    result['decode_passed']=result['passed']
    if mode=='prefill':
        previous=json.loads((output/'qualification.json').read_text()) if (output/'qualification.json').exists() else {}
        if previous and (previous.get('config_sha256')!=digest(config) or previous.get('scope')!=scope):
            raise ValueError('cannot combine different candidate evaluations')
        previous.update(prefill_passed=guard['passed'],prefill_perplexity=result.get('perplexity'),
                        prefill_targets=result.get('targets'),prefill_corpus_sha256=result['corpus_sha256'])
        previous['passed']=bool(previous.get('decode_passed') and guard['passed'])
        (output/'qualification.json').write_text(json.dumps(previous,indent=2)+'\n')
        return previous
    (output/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    if not guard['passed']: return result
    if tasks:
        if scope!='holdout': raise ValueError('heldout tasks may only be evaluated with holdout scope')
        tk=Tokenizer.from_gguf(Path(cfg['model']));cfg=json.loads(config.read_text());cfg['speculative']='none'
        cfg['stop_ids']=manifest['stops'];cfg['decode_checks']=[]
        cfg['context']=max(cfg['context'],8192)
        for task in manifest['tasks']:
            prompt=inputs/task['ids']
            if digest(prompt)!=task['sha256']:raise ValueError('frozen task changed')
            prefix=output/task['name'];path=prefix.with_suffix('.config.json');path.write_text(json.dumps(cfg,indent=2)+'\n')
            print('TASK',config.stem,task['name'],flush=True)
            guard=run(path,prompt,prefix,tokens=1536,trials=1,timeout=600)
            if not guard['passed']:raise RuntimeError('heldout guard failed; stop campaign')
            ids=list(map(int,prefix.with_suffix('.stdout').read_text().split()))
            text=tk.decode([t for t in ids if t not in manifest['stops']]);prefix.with_suffix('.answer.md').write_text(text)
            checked={'passed':False,'output_tokens':len(ids),'ended_on_stop':bool(ids and ids[-1] in manifest['stops'])}
            try:
                if not checked['ended_on_stop']:raise ValueError('incomplete answer')
                checked['cases']=check_output(text,task['name']);checked['passed']=True
            except (ValueError,OSError,subprocess.SubprocessError) as error:checked['error']=str(error)
            result['tasks'][task['name']]=checked
            (output/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def gate(reference, candidate, mandatory_passed=False):
    errors=[]
    if reference.get('scope')!='holdout' or candidate.get('scope')!='holdout':errors.append('holdout evaluation required')
    if reference.get('corpus_sha256')!=candidate.get('corpus_sha256'):errors.append('different corpora')
    if reference.get('targets')!=32768 or candidate.get('targets')!=32768:errors.append('complete 32768-target evaluation required')
    a,b=reference.get('perplexity',0),candidate.get('perplexity',0)
    increase=b/a-1 if all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in [a,b]) else None
    if increase is None or increase>.01+1e-12:errors.append('code perplexity exceeds 1% budget or is invalid')
    if not reference.get('passed') or not candidate.get('passed'):errors.append('evaluation not clean/complete')
    pa,pb=reference.get('prefill_perplexity',0),candidate.get('prefill_perplexity',0)
    prefill_increase=pb/pa-1 if all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in [pa,pb]) else None
    prefill_ok=(reference.get('prefill_passed') and candidate.get('prefill_passed')
                and reference.get('prefill_targets')==32768 and candidate.get('prefill_targets')==32768
                and reference.get('prefill_corpus_sha256')==candidate.get('prefill_corpus_sha256')
                and prefill_increase is not None and prefill_increase<=.01+1e-12)
    if not prefill_ok:errors.append('complete matched prefill quality within 1% required')
    if len(reference.get('tasks',{}))!=20 or set(reference.get('tasks',{}))!=set(candidate.get('tasks',{})):errors.append('complete matched 20-task holdout required')
    new=[name for name,r in reference.get('tasks',{}).items() if r['passed'] and not candidate.get('tasks',{}).get(name,{}).get('passed')]
    if new:errors.append('new heldout failures')
    if not mandatory_passed:errors.append('mandatory coding suite not passed')
    return {'passed':not errors,'errors':errors,'code_perplexity_increase':increase,'reference_perplexity':a,
            'candidate_perplexity':b,'heldout_new_failures':new,'heldout_count':len(candidate.get('tasks',{})),
            'mandatory_checks_passed':mandatory_passed,'prefill_quality_passed':bool(prefill_ok),
            'prefill_perplexity_increase':prefill_increase,'candidate_config_sha256':candidate.get('config_sha256')}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('prepare');q.add_argument('reference',type=Path);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('evaluate');q.add_argument('config',type=Path);q.add_argument('--inputs',type=Path,required=True)
    q.add_argument('--output',type=Path,required=True);q.add_argument('--scope',choices=['tuning','holdout'],required=True)
    q.add_argument('--tasks',action='store_true');q.add_argument('--mode',choices=['decode','prefill'],default='decode')
    q=sub.add_parser('gate');q.add_argument('reference',type=Path);q.add_argument('candidate',type=Path)
    q.add_argument('--mandatory-results',type=Path,required=True);q.add_argument('--name',required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.action=='prepare':print(json.dumps(freeze(a.reference,a.output),indent=2))
    elif a.action=='evaluate':return 0 if evaluate(a.config,a.inputs,a.output,a.scope,a.tasks,a.mode)['passed'] else 1
    else:
        mandatory=json.loads(a.mandatory_results.read_text());rows=[r for r in mandatory['rows'] if r['config']==a.name and r.get('phase')=='quality'];required={'prime','json_escape','csv',*TRAINING,'long_cpp'}
        passed=required.issubset({r['fixture'] for r in rows}) and all(r['quality']['status']=='passed' for r in rows)
        r=gate(json.loads(a.reference.read_text()),json.loads(a.candidate.read_text()),passed)
        a.output.write_text(json.dumps({a.name:r},indent=2)+'\n');print(json.dumps(r,indent=2));return 0 if r['passed'] else 1
    return 0


if __name__=='__main__':raise SystemExit(main())
