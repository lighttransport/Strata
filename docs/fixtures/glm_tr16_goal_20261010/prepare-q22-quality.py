from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'screen-q22.py').read_text();s=s[:s.index('cases=[]')].replace('out.mkdir(exist_ok=False)\n','')
s=s.replace("out=r/'q22-screen'","out=r/'q22-quality'")
s+='''from glm_tr16_quality import check_quality,TOLERANCE,evaluation_width
assert (r/'q22-screen/complete.json').exists(), 'Finish Q22 speed screens first'
rows=json.loads((r/'q22-screen/summary.json').read_text())
candidates=[v for v in rows if v['clean'] and v['name']!='00-q23-q4-control']
# Evaluate the fastest unpruned precision control and the fastest overall case.
unpruned=[v for v in candidates if v['name'] in ('01-q22-aff0','02-q22-aff08')]
selected={v['name']:v for v in [max(unpruned,key=lambda v:v['decode_tok_s']),max(candidates,key=lambda v:v['decode_tok_s'])]}
baseline=r/'renorm-quality-pilot/00-renorm-default/result.json';b=json.loads(baseline.read_text())
assert b['complete'] and b['expected_tokens']==2048
out.mkdir(exist_ok=False)
results=[]
for name,row in selected.items():
 c=json.loads(Path(row['config']).read_text());c['env'].update(STRATA_GLM_EVAL_WIDTH='1',STRATA_GLM_MTP_ACCEPT_MARGIN='0')
 if c.get('route_affinity_profile_sha256'):assert digest(c['env']['STRATA_GLM_ROUTE_AFFINITY_PROFILE'])==c['route_affinity_profile_sha256']
 cfg=out/(name+'.json');cfg.write_text(json.dumps(c,indent=2)+'\\n');dest=out/name
 print('START_Q22_QUALITY',name,flush=True)
 # No same-input --baseline comparison: the expert pack intentionally differs.
 code=run([python,'tools/glm_tr16_quality.py','--config',str(cfg),'--output',str(dest),'--regression-config',str(r/'renorm-quality-pilot/regression.json'),'--wait-idle-seconds','900','--record-cpu-interference'],out/(name+'-driver.log'),1800)
 x=json.loads((dest/'result.json').read_text())
 if c.get('route_affinity_profile_sha256'):assert digest(c['env']['STRATA_GLM_ROUTE_AFFINITY_PROFILE'])==c['route_affinity_profile_sha256']
 assert x['complete'] and x['expected_tokens']==2048 and x['metrics']['tokens']==2048, 'Incomplete Q22 evaluation'
 assert evaluation_width(x['config'])==evaluation_width(b['config'])
 for key in ('corpus_sha256','reference','reference_size','reference_mtime_ns','sequences'):assert x[key]==b[key],key
 candidate_pack=str(Path(c['expert_pack']).resolve())
 baseline_pack=str(Path(b['config']['expert_pack']).resolve())
 assert Path(candidate_pack).name=='experts-q22.gguf' and Path(baseline_pack).name=='experts-q23.gguf'
 left=copy.deepcopy(x['inputs']);right=copy.deepcopy(b['inputs'])
 changed=[]
 for identity,path in ((left,candidate_pack),(right,baseline_pack)):
  pack=[v for v in identity['weight_files'] if v['path']==path];assert len(pack)==1
  changed.append(pack[0]);identity['weight_files']=[v for v in identity['weight_files'] if v['path']!=path]
 assert left==right,'Unexpected additional input identity change'
 import math
 assert all(math.isfinite(x['metrics'][key]) for key in ('perplexity','kl','top1_agreement')), 'Nonfinite quality metrics'
 failures=check_quality(x['metrics'],b['metrics'],TOLERANCE)
 result=dict(name=name,exit_code=code,complete=True,metrics=x['metrics'],baseline=str(baseline),baseline_sha256=digest(baseline),intentional_weight_change={'candidate':changed[0],'baseline':changed[1]},all_other_inputs_equal=True,failures=failures,ppl_increase_percent=100*(x['metrics']['perplexity']/b['metrics']['perplexity']-1),top1_drop_percentage_points=100*(b['metrics']['top1_agreement']-x['metrics']['top1_agreement']),quality_qualified=False,scope='2048-target precision/routing pilot only; relaxed acceptance disabled; full corpus and generated task quality remain required')
 results.append(result);(out/'summary.json').write_text(json.dumps(results,indent=2)+'\\n')
 print('DONE_Q22_QUALITY',json.dumps(result),flush=True)
'''
(r/'quality-q22.py').write_text(s);compile(s,'quality-q22.py','exec')
