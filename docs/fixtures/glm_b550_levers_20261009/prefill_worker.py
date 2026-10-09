"""Use the existing memory/GPU guard for prefill with one precomputed token, zero decode steps."""
import argparse, json, sys
from pathlib import Path
ROOT=Path('/home/syoyo/work/Strata-b550-calibration-15b787b5')
sys.path.insert(0,str(ROOT/'tools'))
import glm_low_memory_bench as guard

def main():
    ap=argparse.ArgumentParser();ap.add_argument('config',type=Path);ap.add_argument('prompt',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    args=argparse.Namespace(config=a.config,prompt=a.prompt,output=a.output,ram_gib=60,tokens=1,trials=3,timeout=1200,gpu_capacity_mib=16304,gpu_used_limit_mib=14256,eval_corpus=None,eval_reference=None,eval_save_logits=None,single=True,fit_only=False,worker=True)
    code=guard.worker(args)
    path=a.output.with_suffix('.result.json');r=json.loads(path.read_text());a.output.with_suffix('.guard-result.json').write_text(json.dumps(r,indent=2)+'\n')
    pp=r['measurements']['prefill'];dec=r['measurements']['decode'];ids=list(map(int,a.output.with_suffix('.stdout').read_text().split()))
    complete=len(pp)==3 and len(dec)==0 and len(ids)==3
    good=complete and r['exit_code']==0 and not r['rejected'] and not r['interference'] and not r['cgroup_after']['memory.events'].get('oom_kill',0) and r['cgroup_after']['memory.swap.current']==0
    r.update(measurement_mode='prefill_only',helper_complete=r['complete'],helper_clean=r['clean'],helper_returncode=code,complete=complete,clean=good,prefill_only_note='Engine requires steps>=1. Emit only the first token already computed by prefill, once per trial; execute zero decoder steps. Existing helper expects three DECODE lines, so its original completeness/clean fields are preserved in guard-result.json and reclassified here for prefill.')
    path.write_text(json.dumps(r,indent=2)+'\n');return 0 if good else 2
if __name__=='__main__':raise SystemExit(main())
