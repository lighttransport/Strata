"""Reproduce placement screening and same-policy ABBA; preserves failed quality gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO/'tools'))
from glm_layer_tune import select_plan


def profile_environment(profile):
    env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
    env.update({k: '1' for k in ('STRATA_GLM_STEP_PIPELINE', 'STRATA_GLM_ACTIVE_POOLS',
        'STRATA_GLM_DEVICE_EXPERTS', 'STRATA_GLM_DEVICE_REDUCTION', 'STRATA_GLM_LAYER_GRAPHS',
        'STRATA_GLM_KDA_PREPARE', 'STRATA_GLM_DEQUANT_COALESCED', 'STRATA_GLM_DEQUANT_ONCE',
        'STRATA_OLD_IQ_MMVQ')})
    env.update(STRATA_NATIVE_NUMA_LOCAL='0', STRATA_GLM_KDA_ROW_PARTS='8',
        STRATA_GLM_IQ_COALESCED='8', STRATA_GLM_PACK_FUSE_QUANT='0',
        STRATA_GLM_Q2_NUMA_WEIGHTS='0', STRATA_GLM_STABLE_ROUTES='1')
    if profile == 'q23-canonical':
        env['STRATA_GLM_CANON'] = '1'
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--decoder', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--profile', choices=('q2', 'q23-native', 'q23-canonical'), required=True)
    ap.add_argument('--cuts', nargs='+', default=['24', '26'])
    ap.add_argument('--memory', choices=('socket', 'interleave'), default='socket')
    ap.add_argument('--print-commands', action='store_true')
    args = ap.parse_args()
    if len(set(args.cuts)) < 2 or any(not c.isdigit() or not 0 < int(c) < 45 for c in args.cuts):
        ap.error('provide at least two distinct cuts in 1..44')
    decoder, model, output = (p.resolve() for p in (args.decoder, args.model, args.output_dir))
    env = profile_environment(args.profile)
    if not args.print_commands:
        output.mkdir(parents=True, exist_ok=False)
    failures = []

    def bench(label, cut, pipeline, plan=None):
        result = output/(label+'.json')
        command = [sys.executable, str(REPO/'tools/glm_v100_bench.py'), str(model),
            str(REPO/'docs/fixtures/glm53_cpp_quality/chat_prompt.txt'), '--decoder', str(decoder),
            '--prompt-tokens', '4096', '--generated-tokens', '512', '--context', '32768',
            '--threads', '32', '--cpu-affinity', 'auto', '--gpu-devices', '0,1',
            '--gpu-budget-mib', '32768', '--gpu-layout', 'layer', '--layer-split', cut,
            '--layer-pipeline', pipeline, '--layer-memory', args.memory, '--prefill-batch', '4096',
            '--prefill-expert-cache-mib', '0', '--decode-cache-mib', '22528', '--decode-graphs',
            '--decode-cache-window', '256', '--speculative', 'mtp', '--draft-depth', '1',
            '--full-warmups', '2', '--repetitions', '3', '--validation-tokens', '512',
            '--validation-repetitions', '3', '--output', str(result)]
        if plan:
            command.extend(('--layer-plan', str(plan)))
        if args.print_commands:
            print(json.dumps(dict(command=command, environment={k:v for k,v in env.items() if k.startswith('STRATA_')})))
            return None
        subprocess.run(command, env=env, cwd=REPO, check=True)
        with result.with_suffix('.oracle.log').open('w') as log:
            grade = subprocess.run([sys.executable, str(Path(__file__).with_name('grade_parser.py')),
                str(model), str(result)], env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        if grade.returncode:
            failures.append(label)
        return json.loads(result.read_text())

    records = [bench('cut'+cut, cut, 'mtp') for cut in args.cuts]
    if args.print_commands:
        print('After screening: select the measured plan, then serial/pipeline/pipeline/serial.')
        return
    plan = select_plan(records, model)
    plan['records'] = [dict(name='cut'+cut+'.json', sha256=hashlib.sha256(
        (output/('cut'+cut+'.json')).read_bytes()).hexdigest()) for cut in args.cuts]
    plan_path = output/'layer-plan.json'
    plan_path.write_text(json.dumps(plan, indent=2)+'\n')
    cut = plan['layer_split']
    rows = [bench(label, cut if pipeline == 'none' else 'auto', pipeline,
                  plan_path if pipeline == 'mtp' else None)
            for label, pipeline in (('A1','none'), ('B1','mtp'), ('B2','mtp'), ('A2','none'))]
    sequences = [t['token_ids'] for row in rows for key in ('trials','validation_trials') for t in row[key]]
    if not all(ids == sequences[0] for ids in sequences):
        raise RuntimeError('same-policy ABBA token streams differ')
    summary = dict(timing_complete=True, same_policy_token_ids_equal=True,
                   parser_failures=failures, forced_length_parser_gate_passed=not failures, selected_cut=cut)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
