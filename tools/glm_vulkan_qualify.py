"""Compare opt-in Vulkan profiles against a frozen strict greedy control.

Run each owned benchmark sequentially; preserve failures and reject changed IDs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
from glm_guarded_run import run
from glm_low_memory_bench import parse_log

FEATURES = {
    'strict': {},
    'async': {'STRATA_VULKAN_ASYNC_STAGING':'1'},
    'joined': {'STRATA_VULKAN_ASYNC_STAGING':'1','STRATA_VULKAN_JOINED_PROJECTIONS':'1'},
    'mhc': {'STRATA_VULKAN_CPU_MHC':'1'},
    'fused': {'STRATA_VULKAN_FUSED_KDA':'1'},
    'overlap': {'STRATA_VULKAN_OVERLAP_EXPERTS':'1','STRATA_VULKAN_CPU_PLACEMENT':'split'},
    'graphics': {'GGML_VK_ALLOW_GRAPHICS_QUEUE':'1'},
    'activations': {'STRATA_VULKAN_AVX2_ACTIVATIONS':'1'},
}
FEATURES['combined'] = {k:v for name, feature in FEATURES.items() if name not in ('graphics','activations') for k,v in feature.items()}

FEATURES['configured']={}

def compare_ids(reference, candidate):
    return bool(reference) and bool(candidate) and all(ids == reference for ids in candidate)


def profile_config(cfg,name):
    experiment_keys={k for feature in FEATURES.values() for k in feature}
    base={k:v for k,v in cfg.get('env',{}).items() if k not in experiment_keys}
    trial=cfg | {'env':cfg.get('env',{}) if name=='configured' else base | FEATURES[name]}
    if name=='strict':
        trial.update(prefill_batch=1,prefill_experts='cpu',decode_cache_mib=0,speculative='none')
        trial['env']={k:v for k,v in cfg.get('env',{}).items()
                      if not k.startswith('STRATA_VULKAN_') and k not in experiment_keys}
        trial.pop('verify_batch',None);trial.pop('draft_depth',None)
    return trial


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path);p.add_argument('prompt',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profiles',nargs='+',choices=list(FEATURES),default=['strict','async','joined','mhc','fused','overlap','graphics','combined'])
    p.add_argument('--reference-exe',type=Path)
    p.add_argument('--reference-result',type=Path)
    p.add_argument('--tokens',type=int,default=64);p.add_argument('--trials',type=int,default=3)
    p.add_argument('--timeout',type=int,default=1200)
    p.add_argument('--correctness-only',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifest=a.output/'summary.json'
    if manifest.exists():raise FileExistsError('refusing to overwrite qualification evidence')
    cfg=json.loads(a.config.read_text())
    if cfg.get('backend') != 'vulkan':raise ValueError('requires Vulkan config')
    # Feature experiments cannot inherit undocumented Vulkan/GGML options.
    reference=None
    if a.reference_result:
        value=json.loads(a.reference_result.read_text())['output_ids']
        if not value or not compare_ids(value[0],value):raise ValueError('reference is not deterministic')
        reference=value[0]
    profiles=list(a.profiles)
    if reference is None and (not profiles or profiles[0]!='strict'):profiles.insert(0,'strict')
    rows=[]
    for name in profiles:
        trial=profile_config(cfg,name)
        if name=='strict' and a.reference_exe:trial['exe']=str(a.reference_exe.resolve())
        config=a.output/(name+'.json');config.write_text(json.dumps(trial,indent=2)+'\n')
        prefix=a.output/(name+'-run')
        guard=run(config,a.prompt,prefix,tokens=a.tokens,trials=a.trials,timeout=a.timeout,correctness_only=a.correctness_only)
        row={'profile':name,'guard_passed':guard['passed'],'executable_sha256':guard['executable_sha256'],
             'config_sha256':guard['config_sha256'],'qualified':False}
        result=prefix.with_suffix('.vulkan.json')
        if result.exists():
            value=json.loads(result.read_text());ids=value['output_ids']
            if reference is None:
                if not ids or not compare_ids(ids[0],ids):raise RuntimeError('strict control is not deterministic')
                reference=ids[0]
            log=prefix.with_suffix('.log').read_text()
            measurements=parse_log(log)
            row.update(output_ids_match=compare_ids(reference,ids),generated=[len(x) for x in ids],
                       prefill_tok_s=[r['tokens_per_second'] for r in measurements['prefill']],
                       decode_tok_s=[r['tokens_per_second'] for r in measurements['decode']])
            row['correctness_passed']=guard['passed'] and row['output_ids_match']
            row['timings_qualified']=guard.get('performance_qualified',guard['passed'])
            row['qualified']=row['correctness_passed'] and row['timings_qualified']
            row['median_prefill_tok_s']=statistics.median(row['prefill_tok_s'])
            row['median_decode_tok_s']=statistics.median(row['decode_tok_s'])
        rows.append(row)
        manifest.write_text(json.dumps({'prompt_sha256':hashlib.sha256(a.prompt.read_bytes()).hexdigest(),
                       'tokens':a.tokens,'trials':a.trials,'rows':rows},indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if not guard['passed']:raise RuntimeError('guard failed; inspect evidence before continuing')
    return 0

if __name__=='__main__':raise SystemExit(main())
