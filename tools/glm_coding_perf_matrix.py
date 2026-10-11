"""Prepare or run interleaved, EOS-aware coding controls under the existing resource guard.

Every candidate uses the same frozen prompt and routing configuration. The first
configuration is the control. Raw results stay in the requested output directory;
only clean complete trials enter the aggregate. This does not certify quality.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time
from jinja2.sandbox import SandboxedEnvironment
from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer
from glm_q2_coding_bench import FIXTURES
from glm_q2_coding_check import check as check_coding
from glm_guarded_run import run as guarded_run
from glm_coding_quality import check_output, TRAINING
from glm_chat_stops import chat_stop_ids
FIXTURES = {**FIXTURES, **TRAINING}

FAULT = re.compile(r'device lost from bus|GPU reset|ring .*timeout|\bVM\b.*\bfault\b|page fault|AER:.*(?:error|fatal)|amdgpu.*(?:failed|error)', re.I)


def validate_configs(configs, allow_lossy=False):
    if not configs or configs[0][1].get('speculative','none') != 'none':
        raise ValueError('the first configuration must be an ordinary greedy control')
    frozen_env=['STRATA_GLM_ROUTE_AFFINITY','STRATA_GLM_ROUTE_AFFINITY_PROFILE',
        'STRATA_GLM_DEFER_EXPERTS','STRATA_GLM_DEFER_ROUTES','STRATA_GLM_CANON',
        'STRATA_GLM_FLOATING_Q8','STRATA_GLM_DENSE_Q4','STRATA_GLM_Q23_WIDE']
    def signature(cfg):
        return ([cfg.get(k) for k in ['model','expert_pack','expert_pack_profile','cpu_expert_backend']],
                [cfg.get('env',{}).get(k) for k in frozen_env])
    if not allow_lossy and any(signature(cfg)!=signature(configs[0][1]) for _,cfg in configs):
        raise ValueError('weights, dense formats and routing must stay frozen across the matrix')
    controls = {}
    for i, (name, cfg) in enumerate(configs):
        if cfg.get('speculative', 'none') == 'none':
            controls[name] = name
            continue
        matches = [(n, c) for n, c in configs[:i] if c.get('speculative', 'none') == 'none'
                   and signature(c) == signature(cfg)
                   and all(c.get(k) == cfg.get(k) for k in ['threads', 'decode_cache_slots', 'prefill_experts'])]
        requested = cfg.get('comparison_control')
        if requested:
            matches = [(n, c) for n, c in matches if n == requested]
        if not matches:
            raise ValueError('speculative configuration needs a preceding matched greedy control: '+name)
        controls[name] = matches[-1][0]
    return controls


def prepare(tokenizer, template, fixture, length, tasks=None):
    env = SandboxedEnvironment(extensions=['jinja2.ext.loopcontrols'])
    env.globals['raise_exception'] = lambda s: (_ for _ in ()).throw(ValueError(s))
    task = (FIXTURES if tasks is None else tasks)[fixture]
    # A task that is already long stays intact at every nominal length. Actual
    # lengths are recorded, and callers reject it from mismatched length cells.
    def render(notes):
        content = ('Reference notes (data, not instructions):\n' + notes + '\nImplementation request:\n' + task) if notes else task
        return env.from_string(template).render(messages=[dict(role='user',content=content)],tools=None,
            add_generation_prompt=True,reasoning_effort='low')
    base = tokenizer.encode(render(''),parse_special=True)
    if len(base) > length:
        return render(''), base
    note = '// Frozen C++ review note: bounds are checked before indexing; failed parsing leaves the output unchanged.\n'
    # Search in characters, then re-tokenize the full rendered prompt so BPE
    # boundaries and the final chat suffix remain canonical.
    text = note * (length + 1)
    lo, hi = 0, min(len(text), length * 12)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tokenizer.encode(render(text[:mid]),parse_special=True)) <= length:
            lo = mid
        else:
            hi = mid - 1
    chat = render(text[:lo])
    return chat, tokenizer.encode(chat,parse_special=True)


def aggregate(rows):
    cells = {}
    for row in rows:
        if row.get('phase', 'timing') != 'timing' or not row['clean']:
            continue
        key = (row['config'], row['length'], row['fixture'])
        cells.setdefault(key, []).append(row)
    medians = []
    for (config,length,fixture), values in cells.items():
        medians.append(dict(config=config,length=length,fixture=fixture,trials=len(values),
            speculative=values[0].get('speculative','none'),
            prefill_tok_s=statistics.median(v['prefill_tok_s'] for v in values),
            decode_tok_s=statistics.median(v['decode_tok_s'] for v in values),
            response_seconds=statistics.median(v['response_seconds'] for v in values),
            output_hashes=sorted(set(v['output_sha256'] for v in values)),
            quality_passed=all(v.get('quality',{}).get('status')=='passed' for v in values),
            comparison_control=values[0].get('comparison_control'),
            prefill_scope=values[0].get('prefill_scope','unspecified'),
            comparison_valid=all(not v.get('qualification') for v in values),
            greedy_ids_identical=all(v.get('greedy_ids_identical') is not False for v in values)))
    for median in medians:
        quality = [r for r in rows if r.get('phase') == 'quality' and r['config'] == median['config']
                   and r['length'] == median['length'] and r['fixture'] == median['fixture']]
        if any(r.get('phase') for r in rows):
            median['quality_passed'] = (len(quality) >= median['trials'] and
                                       all(r['quality']['status'] == 'passed' for r in quality))
    return medians


def valid_quality_gate(gate):
    return (gate.get('passed') is True and gate.get('heldout_count') == 20
            and gate.get('prefill_quality_passed') is True
            and gate.get('heldout_new_failures') == [] and gate.get('mandatory_checks_passed') is True
            and isinstance(gate.get('code_perplexity_increase'), (int,float))
            and math.isfinite(gate['code_perplexity_increase']) and gate['code_perplexity_increase'] <= .01 + 1e-12)


def length_summary(medians,control,expected,trials, *, prefill_target=700, decode_target=40,
                   speculative_target=60, primary_length=8192, quality_gate=None):
    summaries=[]
    for config,length in sorted({(r['config'],r['length']) for r in medians}):
        cells=[r for r in medians if r['config']==config and r['length']==length]
        complete={r['fixture'] for r in cells}==set(expected[length]) and all(r['trials']>=trials for r in cells)
        prefill=math.exp(statistics.mean(math.log(r['prefill_tok_s']) for r in cells))
        decode=math.exp(statistics.mean(math.log(r['decode_tok_s']) for r in cells))
        regressions=[]
        for row in cells:
            baseline=next((r for r in medians if r['config']==(row.get('comparison_control') or control) and r['length']==length and r['fixture']==row['fixture']),None)
            if baseline and (row['prefill_tok_s']<.95*baseline['prefill_tok_s'] or row['decode_tok_s']<.95*baseline['decode_tok_s']):
                regressions.append(row['fixture'])
        summaries.append(dict(config=config,length=length,complete=complete,fixture_medians=cells,
            geometric_mean_prefill_tok_s=prefill,geometric_mean_decode_tok_s=decode,
            regressions_over_5_percent=regressions,
            timing_targets_met=complete and not regressions and prefill>=prefill_target and decode>=decode_target,
            greedy_target_met=complete and not regressions and prefill>=prefill_target and decode>=decode_target and cells[0]['speculative']=='none',
            quality_passed=all(r['quality_passed'] for r in cells),
            greedy_ids_identical=all(r['greedy_ids_identical'] for r in cells)))
        summaries[-1]['comparison_valid']=all(r['comparison_valid'] for r in cells)
        baseline_cells=[r for r in medians if r['config']==(cells[0].get('comparison_control') or control) and r['length']==length]
        baseline_decode=math.exp(statistics.mean(math.log(r['decode_tok_s']) for r in baseline_cells)) if baseline_cells else None
        speculative=cells[0]['speculative']!='none'
        spec_gain=decode/baseline_decode if speculative and baseline_decode else None
        summaries[-1].update(primary=length==primary_length, speculative_speedup=spec_gain,
            speculative_gain_qualified=speculative and complete and not regressions and spec_gain is not None and spec_gain>=1.10,
            speculative_target_met=speculative and complete and not regressions and prefill>=prefill_target and decode>=speculative_target,
            quality_gate_passed=valid_quality_gate(quality_gate.get(config,{}) if quality_gate else {}))
        timing_ok=summaries[-1]['speculative_target_met'] if speculative else summaries[-1]['greedy_target_met']
        summaries[-1]['release_qualified']=(length==primary_length and timing_ok and
            summaries[-1]['quality_passed'] and summaries[-1]['quality_gate_passed'] and
            summaries[-1]['comparison_valid'] and (not speculative or summaries[-1]['greedy_ids_identical']))
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('configs',nargs='+',type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--lengths',nargs='+',type=int,default=[2048,4096,8192,16384])
    p.add_argument('--fixtures',nargs='+',choices=FIXTURES,default=list(FIXTURES))
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--tokens',type=int,default=512)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--phase',choices=['timing','quality','both'],default='both')
    p.add_argument('--warm-trials',type=int,default=3,help='One untimed warmup plus this many in-process timing repetitions')
    p.add_argument('--quality-tokens',type=int,default=1536)
    p.add_argument('--long-quality-tokens',type=int,default=4096)
    p.add_argument('--prefill-target',type=float,default=700)
    p.add_argument('--decode-target',type=float,default=40)
    p.add_argument('--speculative-target',type=float,default=60)
    p.add_argument('--primary-length',type=int,default=8192)
    p.add_argument('--allow-lossy',action='store_true',help='Screen different precision/routing; does not waive quality gates')
    p.add_argument('--quality-gate',type=Path,help='Per-configuration held-out and code-PPL qualification JSON')
    a = p.parse_args()
    if a.trials < 1 or not 1 <= a.tokens <= 512 or any(n < 256 or n > 16384 for n in a.lengths):
        p.error('invalid trials, output cap or prompt length')
    if not 2 <= a.warm_trials <= 5 or not 2 <= a.quality_tokens <= 4096 or not 2 <= a.long_quality_tokens <= 4096:
        p.error('warm-trials must be 2..5 and quality caps 2..4096')
    if any(not math.isfinite(x) or x<=0 for x in [a.prefill_target,a.decode_target,a.speculative_target]):
        p.error('timing targets must be finite and positive')
    configs = [(path.stem,json.loads(path.read_text())) for path in a.configs]
    if len({name for name,_ in configs}) != len(configs):p.error('configuration names must be unique')
    if len({cfg['model'] for _,cfg in configs}) != 1:p.error('all configurations must use the same target model')
    controls = validate_configs(configs,a.allow_lossy)
    a.output.mkdir(parents=True,exist_ok=True)
    model = Path(configs[0][1]['model'])
    tokenizer = Tokenizer.from_gguf(model)
    metadata = GGUFFile(model).metadata
    stops = chat_stop_ids(metadata)
    prompts = []
    for length in a.lengths:
        for fixture in a.fixtures:
            chat, ids = prepare(tokenizer,metadata['tokenizer.chat_template'],fixture,length)
            if len(ids) > length:
                continue  # long_cpp is a holdout only at lengths that fit its frozen headers.
            stem = a.output / f'{fixture}-{length}'
            stem.with_suffix('.ids').write_text(','.join(map(str,ids)))
            stem.with_suffix('.prompt.txt').write_text(chat)
            prompts.append(dict(fixture=fixture,length=length,actual_tokens=len(ids),
                ids=str(stem.with_suffix('.ids')),sha256=hashlib.sha256(stem.with_suffix('.ids').read_bytes()).hexdigest()))
    (a.output/'prompts.json').write_text(json.dumps(prompts,indent=2)+'\n')
    if a.prepare_only:return
    expected={length:[r['fixture'] for r in prompts if r['length']==length] for length in a.lengths}
    rows = []
    phases = ['timing','quality'] if a.phase=='both' else [a.phase]
    gate = json.loads(a.quality_gate.read_text()) if a.quality_gate else None
    for trial in range(a.trials):
        for prompt in prompts:
            for phase in phases:
                for name,source in configs:
                    cfg = json.loads(json.dumps(source))
                    cap = a.tokens if phase=='timing' else (a.long_quality_tokens if prompt['fixture']=='long_cpp' else a.quality_tokens)
                    cfg['context'] = max(source.get('context',0),prompt['actual_tokens']+cap+8)
                    cfg['prefill_batch'] = min(prompt['length'],16384)
                    if phase=='quality': cfg['stop_ids'] = stops
                    else: cfg.pop('stop_ids',None)
                    cfg['gpu_telemetry'] = False
                    cfg['gpu_pci_preflight'] = True
                    tag = f'{name}-{prompt["fixture"]}-{prompt["length"]}-{phase}-{trial}'
                    prefix = a.output/tag
                    path = prefix.with_suffix('.config.json');path.write_text(json.dumps(cfg,indent=2)+'\n')
                    print('START',tag,flush=True)
                    guard = guarded_run(path,prompt['ids'],prefix,tokens=cap,
                                        trials=a.warm_trials if phase=='timing' else 1,timeout=1800)
                    if not guard['passed']:
                        raise RuntimeError('guarded trial failed; matrix stopped: '+tag)
                    result = json.loads(prefix.with_suffix('.result.json').read_text())
                    m=result['measurements']
                    ids=list(map(int,prefix.with_suffix('.stdout').read_text().split()))
                    quality=dict(status='not_checked',reason='fixed-length timing; independent EOS-aware quality pass required')
                    if phase=='quality':
                        quality_root=a.output/(tag+'-quality');quality_root.mkdir(exist_ok=True)
                        text=tokenizer.decode([t for t in ids if t not in stops])
                        (quality_root/(prompt['fixture']+'.output.md')).write_text(text)
                        if not ids or ids[-1] not in stops:
                            quality=dict(status='failed',reason='answer did not reach EOS within quality cap')
                        else:
                            try: quality=dict(status='passed',cases=check_output(text,prompt['fixture']))
                            except (ValueError,subprocess.SubprocessError,OSError) as error:
                                quality=dict(status='failed',reason=str(error))
                    row=dict(config=name,fixture=prompt['fixture'],length=prompt['length'],actual_tokens=prompt['actual_tokens'],
                        phase=phase,prefill_scope='warm' if phase=='timing' else 'cold',
                        inner_trials=a.warm_trials if phase=='timing' else 1,
                        comparison_control=controls[name] if cfg.get('speculative','none')!='none' else configs[0][0],speculative=cfg.get('speculative','none'),
                        decode_cache_slots=m['decode_cache'][0]['slots'] if m.get('decode_cache') else None,
                        trial=trial,clean=True,prefill_tok_s=m['median_prefill_tok_s'],decode_tok_s=m['median_decode_tok_s'],
                        response_seconds=result['wall_seconds'],quality=quality,output_sha256=result['output_sha256'],
                        output_tokens=len(ids),result=prefix.name+'.result.json')
                    control=next((r for r in rows if r['config']==row['comparison_control'] and r['fixture']==row['fixture']
                        and r['length']==row['length'] and r['trial']==trial and r['phase']==phase),None)
                    row['greedy_ids_identical']=True if row['comparison_control']==name else (None if control is None else row['output_sha256']==control['output_sha256'])
                    row['cache_matched']=None if control is None else row['decode_cache_slots']==control['decode_cache_slots']
                    if cfg.get('speculative','none')!='none' and (not control or not row['greedy_ids_identical']):
                        row['qualification']='rejected: speculative output differs from matched greedy'
                    if cfg.get('speculative','none')!='none' and control and not row['cache_matched']:
                        row['qualification']='rejected comparison: cache differs; prepare a greedy control with these decode_cache_slots'
                    rows.append(row)
                    medians=aggregate(rows)
                    (a.output/'aggregate.json').write_text(json.dumps(dict(rows=rows,medians=medians,
                        lengths=length_summary(medians,configs[0][0],expected,a.trials,prefill_target=a.prefill_target,
                            decode_target=a.decode_target,speculative_target=a.speculative_target,
                            primary_length=a.primary_length,quality_gate=gate)),indent=2)+'\n')
                    print('FINISH',tag,row['prefill_tok_s'],row['decode_tok_s'],quality['status'],flush=True)


if __name__ == '__main__':main()
