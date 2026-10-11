"""Print a table across screen directories (summary.json rows) and write all-screens.json."""
import json, sys
from pathlib import Path
R = Path('/mnt/nvme02/work/Strata/build-tr16-35-20261010')
dirs = [Path(a) for a in sys.argv[1:]] or sorted(p.parent for p in R.glob('s*/summary.json'))
rows = []
for d in dirs:
    for r in json.loads((d / 'summary.json').read_text()):
        r['screen'] = d.name; rows.append(r)
cols = [('screen', 10), ('name', 26), ('decode_tok_s', 8), ('speed_change_percent', 7), ('tokens_per_round', 6), ('accepted_per_round', 6),
        ('expert_gb_per_token', 6), ('flow_ms_per_token', 6), ('draft_ms_per_round', 6), ('verify_ms_per_round', 6), ('actual_cache_mib', 8),
        ('soft_accepted', 5), ('logit_regret_per_token', 6), ('tokens_identical_to_control', 5), ('decision', 22)]
hdr = {'decode_tok_s': 'tok/s', 'speed_change_percent': 'd%', 'tokens_per_round': 'tpr', 'accepted_per_round': 'apr', 'expert_gb_per_token': 'GB/t',
       'flow_ms_per_token': 'flow', 'draft_ms_per_round': 'draft', 'verify_ms_per_round': 'verif', 'actual_cache_mib': 'cache', 'soft_accepted': 'soft',
       'logit_regret_per_token': 'regr', 'tokens_identical_to_control': 'same'}
print(' '.join(hdr.get(c, c)[:w].ljust(w) for c, w in cols))
for r in rows:
    out = []
    for c, w in cols:
        v = r.get(c)
        if isinstance(v, float): v = f'{v:.3f}' if abs(v) < 100 else f'{v:.1f}'
        out.append(str(v)[:w].ljust(w))
    d4 = r.get('distinct_4gram_fraction') or []
    print(' '.join(out), 'd4=' + ','.join(f'{v:.3f}' for v in d4 if v is not None))
(R / 'all-screens.json').write_text(json.dumps(rows, indent=1) + '\n')
