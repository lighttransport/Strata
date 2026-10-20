#!/usr/bin/env python3
"""Compare frozen partial CPU service measurements with explicit simulator assumptions.

Run from repository root. Outputs diagnostics, never full-model tok/s or a fitted
hardware profile. Measured service excludes input quantization; the model includes it.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.sim import fp4


def main():
    evidence = ROOT / 'experiment/glm_fp4_local/results/20261020/fp4-single-unfused/summary.json'
    measured = json.loads(evidence.read_text())['results']
    profile = json.loads(fp4.PROFILE.read_text())

    def rates(values):
        return [dict(gweights_s=x, quant_gitems_s=1, availability=1) for x in values]

    comparison = []
    for tokens in (1, 4):
        for experts in (1, 3):
            # Explicitly enabled selection; owned flow with fusion requested.
            samples = [float(x['median_ms']) for x in measured
                       if x['log'] == f'nv-single-unfused-1-{experts}.log'
                       and int(x['tokens']) == tokens and x['owned'] == '1'
                       and x['flow'] == '1' and x['fuse_h_quant'] == '1']
            if len(samples) != 4:
                raise ValueError('expected two layers and two runs for each geometry')
            model = fp4.simulate(profile, tokens=tokens, counts=(experts, 0, 0),
                                 layers=1, cache='cold', csf_preexpanded=True,
                                 miss_policy='cpu', cpu_rates=rates((50, 40, 40)))
            comparison.append(dict(tokens=tokens, experts=experts,
                                   measured_medians_ms=samples,
                                   modeled_service_ms=model['moe_milliseconds']))
    sensitivity = []
    for values in ((50,40,40), (75,40,40), (100,40,40), (125,40,40),
                   (50,80,40), (50,40,80), (100,80,80)):
        model = fp4.simulate(profile, layers=42, cache='mixed', hit_rates=(.5,.5,.5),
                             activation_path='device', transfer_overlap=True,
                             csf_preexpanded=True, miss_policy='cpu', cpu_rates=rates(values))
        sensitivity.append(dict(cpu_gweights_s=list(values), moe_ms=model['moe_milliseconds']))
    print(json.dumps(dict(
        basis='Diagnostic comparison and uncalibrated sensitivity; not achieved throughput',
        evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        simulator_sha256=hashlib.sha256(Path(fp4.__file__).read_bytes()).hexdigest(),
        profile_sha256=hashlib.sha256(fp4.PROFILE.read_bytes()).hexdigest(),
        comparison=comparison, sensitivity=sensitivity,
        limitations=[
            'Partial real experts, fixed inputs/routing and warm repeated weights; not full-checkpoint streaming',
            'Measured CPU service excludes input quantization; simulated service includes it',
            'Scalar weight/token rates cannot capture expert-count scheduling or token reuse',
            'Only primary CPU has this measurement; worker CPU and GPU/network rates remain assumptions',
            'Nonmonotonic sensitivity is possible with shared resources and fixed scheduling; not a hardware slowdown claim',
            'Attention, shared experts and other non-MoE decoder work are outside this model',
            'Do not infer a full-model tok/s or replace default rates from this comparison']), indent=2))


if __name__ == '__main__':
    main()
