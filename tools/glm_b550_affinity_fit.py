"""Fit a separate lossy-routing response; preserve the ordinary B550 calibration.

Only scalar-affinity, clean ordinary throughput runs are accepted. Two fitted
aggregate terms do not identify physical GPU bandwidth or combine costs.
"""
import argparse
import dataclasses
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent/'sim'))
import calibration
from glm_b550_sim_records import import_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--fit', nargs='+', required=True, help='exact result filename stems')
    parser.add_argument('--base-params', type=Path, default=Path('tools/sim/data/params_b550.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in args.runs:
        records.extend(r for r in import_run(path, fit=path.stem in args.fit, allow_routing_bias=True)
                       if r['kind'] == 'decode')
    if len([r for r in records if r['fit']]) < 2:
        parser.error('need at least two distinct training measurements')
    params = calibration.load_params(args.base_params)
    best = None
    for tau in range(100, 241, 2):
        for scale in range(40, 101, 2):
            candidate = dataclasses.replace(params, routing_affinity_tau=tau/1000,
                                            biased_resident_gpu_scale=scale/100)
            rows = calibration.evaluate(records, candidate, only_fit=True)
            error = sum(abs(math.log(r['predicted']/r['measured'])) for r in rows)/len(rows)
            if best is None or error < best[0]:
                best = error, candidate
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'routing-sim-records.json').write_text(json.dumps(dict(records=records), indent=2)+'\n')
    calibration.save_params(best[1], args.output/'routing-sim-params.json', dict(
        training=[r['name'] for r in records if r['fit']],
        holdout=[r['name'] for r in records if not r['fit']],
        fitted_fields=['routing_affinity_tau', 'biased_resident_gpu_scale'],
        note='Only positive-affinity aggregate costs fitted. CPU bandwidth and ordinary parameters fixed. '
             'Not a physical GPU/combine measurement or semantic quality certificate.'))
    rows = calibration.evaluate(records, best[1])
    (args.output/'routing-sim-validation.json').write_text(json.dumps(rows, indent=2)+'\n')
    for r in rows:
        print(r['name'], 'predicted', round(r['predicted'], 3), 'error', round(100*r['error'], 2), '%')


if __name__ == '__main__':
    main()
