"""Recompute decisions under the v2 loop-proxy rule (d4 >= 0.85 on timed trials, no repeated 32-gram); keep the v1 decision."""
import json,sys
from pathlib import Path
for d in sys.argv[1:]:
    p=Path(d)/'summary.json';rows=json.loads(p.read_text())
    for r in rows:
        if not r.get('clean') or r.get('is_control') or r.get('decision','').startswith('FAILED'):continue
        d4=r['distinct_4gram_fraction'][1:];gain=r.get('speed_change_percent') or 0
        loop_ok=all(v is not None and v>=0.85 for v in d4) and not any(r['repeated_32gram'][1:])
        if not loop_ok:dec='DROP:loop-proxy'
        elif r['exact_expected'] and not r['tokens_identical_to_control']:dec=('NOTE' if gain>=1 else 'DROP')+':reclassified-lossy'
        elif gain>=3:dec='KEEP'
        elif gain>=1:dec='NOTE'
        else:dec='DROP:no-gain'
        if dec!=r['decision']:
            r.setdefault('decision_v1',r['decision']);r['decision']=dec;print(d,r['name'],r['decision_v1'],'->',dec)
    p.write_text(json.dumps(rows,indent=2)+'\n')
