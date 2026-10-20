"""Conditional full decode budget; no inference or hardware benchmarking.
Run from the repository root: python3 experiment/glm_fp4_local/simulator/full_decode_report.py
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.sim import fp4

profile = json.loads(fp4.PROFILE.read_text())
rows = []
for availability in (1.0, 0.5, 0.25):
    assumed = copy.deepcopy(profile)
    for node in assumed['nodes']:
        node['availability'] = availability
    result = fp4.simulate(assumed, tokens=1, counts=(3, 3, 2), layers=42,
                          mode='fused', topology='direct', cache='cold',
                          activation_path='host')
    moe_ms = result['moe_milliseconds']
    # Explicit sensitivity budgets, NOT measured/calibrated trunk predictions.
    trunk_ms = [50 / availability, 150 / availability]
    rows.append(dict(assumed_available_fraction=availability, moe_ms=moe_ms,
                     other_decoder_budget_ms=trunk_ms,
                     conditional_full_decode_tok_s=[1000/(moe_ms+trunk_ms[1]),
                                                    1000/(moe_ms+trunk_ms[0])]))
cache_sweep = []
for hit in (0, .5, .8):
    for overlap in (False, True):
        result = fp4.simulate(profile, tokens=1, counts=(3, 3, 2), layers=42,
                              cache='mixed', hit_rates=(hit,)*3,
                              transfer_overlap=overlap, csf_preexpanded=True)
        ms = result['moe_milliseconds']
        cache_sweep.append(dict(requested_hit_rate=hit, transfer_overlap=overlap,
                               csf_preexpanded=True, moe_ms=ms,
                               realized_hit_rates=[r['realized_hit_rate'] for r in result['residency']],
                               conditional_full_decode_tok_s=[1000/(ms+150),1000/(ms+50)]))
print(json.dumps(dict(basis='THEORETICAL / UNCALIBRATED / CONDITIONAL',
                     batch=1, mtp=False, rows=rows, cache_sweep=cache_sweep), indent=2))
